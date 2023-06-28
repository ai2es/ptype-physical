import numpy as np
import xarray as xr

import sys
sys.path.append('../') 

import sounding_utils
from xhistogram.xarray import histogram

from joblib import dump

from dask.distributed import Client
from dask_jobqueue import PBSCluster
import dask
from os.path import join
import time
import subprocess


def load_dask(model):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(f"/glade/campaign/cisl/aiml/ptype/ptype_case_studies/*/{model}/*/*/*.nc",
                               parallel=True, engine='netcdf4', 
                               decode_cf=False, concat_dim='valid_time', combine='nested', 
                               chunks={'time':1, 'heightAboveGround': 21, 'isobaricInhPa': 37})
        ds.attrs['nwp'] = model
        return ds


def agg_delayed(ds, save_dir='/glade/scratch/dkimpara/composite_calcs'):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        try:
            print(ds.attrs['nwp'])
        except: 
            raise ValueError('dataset must have nwp attr set')

        ds = sounding_utils.filter_latlon(ds)

        precip_mask = (
            (ds["crain"] == 1)
            | (ds["csnow"] == 1)
            | (ds["cicep"] == 1)
            | (ds["cfrzr"] == 1)
            )

        ds = ds.where(precip_mask)
        print('filtered')
        if 'wb_h' not in list(ds.keys()):
            ds = sounding_utils.wb_stull(ds)
        print('wb computed')
        ptypes = ['rain', 'snow', 'icep', 'frzr']
        prof_vars = ['t_h', 'dpt_h', 'wb_h']

        persist_vars = (prof_vars + 
                        [f'ML_c{var}' for var in ptypes] +
                        [f'c{var}' for var in ptypes])

        #ds[persist_vars].persist() 
        total_obs = ds.t_h.isel(heightAboveGround=0).count(dim=('x','y','time','valid_time'))
        metadata = {'total_obs': total_obs}
        print('total obs')
        
        res_dict = {'mean': [],
                    'hist': [],
                    }


        lazy_results = []
        ####################################
        remote_ds = client.scatter(ds)
        for ptype in ptypes:
            for model in ['ML_c', 'c']:
                predtype = model + ptype
                
                lazy_result = dask.delayed(agg_parallel)(predtype, remote_ds) #this fn returns a dict of datasets
                lazy_results.append(lazy_result)
        
        for i in range(len(ptypes) * 2):
            res, meta = lazy_results[i].compute()
            metadata = metadata | meta #merge metadata dictionary
            for k in res.keys():
                res_dict[k].append(res[k])
        print('extracted')
        ds_concat = [xr.concat(res_ds_list, dim='predtype') for res_ds_list in res_dict.values()]
        result = xr.merge(ds_concat)
        #save
        result.to_netcdf(path=join(save_dir, ds.attrs['nwp']))
        
        print('computing metadata')
        dask.persist(metadata) #dump method does not trigger a compute
        dump(metadata, join(save_dir, f"{ds.attrs['nwp']}_metadata"))

        return result, metadata

def agg_parallel(predtype, ds):
    metadata = {}
    prof_vars = ['t_h', 'dpt_h', 'wb_h']
    bins = np.arange(-40, 40, 0.1)
    
    subset = ds[prof_vars].where(ds[predtype] == 1)

    ### num_obs per hr
    counts = subset.t_h.count(dim=('x','y'))
    obs_per_hr = counts.isel(heightAboveGround=0).mean(dim=('time', 'valid_time'))
    metadata[f'{predtype}_obs_per_hr'] = obs_per_hr

    # num_obs of predtype==1
    num_obs = subset.t_h.isel(heightAboveGround=0).count(dim=('x','y','time','valid_time'))
    metadata[f'{predtype}_num_obs'] = num_obs

    # num_obs w frac abv zero
    for var in prof_vars:
        metadata[f"{predtype}_{var}_frac_abv_zero"] = (
            frac_abv_zero(subset, var, num_obs)
        )

    # means and quantiles
    mean = subset.mean(dim=('valid_time', 'time', 'x', 'y')) #returns dataset objects
    mean = mean.rename({var: f'{var}_mean' for var in prof_vars})

    #### densities ####
    densities = ({f'{var}_hist': (
            histogram(subset[var], bins=bins, dim=['valid_time', 'time', 'x', 'y'], density=True)
            .rename({f'{var}_bin': 'bin'})
            ) for var in prof_vars})
    densities = xr.Dataset(densities) #arrays already named histograms
    
    res_datasets = {'mean': mean,
                'hist': densities}
    
    res_datasets = {k: v.expand_dims({'predtype': [predtype]}) for k,v in res_datasets.items()}
    
    return res_datasets, metadata

def frac_abv_zero(ds, x_col, total):
    num_over_zero = (ds[x_col] > 0).any(dim="heightAboveGround").sum()
    return num_over_zero / total

if __name__ == '__main__':
    cluster = PBSCluster(account='NAML0001',
                     queue='casper',
                     walltime='08:00:00',
                     memory="4000 GB",
                     n_workers=20,
                     resource_spec='select=1:ncpus=16:mem=200GB', # Specify resources
                     interface='ib0',
                     local_directory='/glade/work/dkimpara/dask/',
                     log_directory="/glade/work/dkimpara/dask_logs/")

    # Change your url to the dask dashboard so you can see it
    #dask.config.set({'distributed.dashboard.link':'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'})
    print(f"Use this link to monitor the workload: {cluster.dashboard_link}")
    client = Client(cluster)

    for model in ['rap', 'gfs', 'hrrr']:
        tic = time.time()
        ds = load_dask(model)
        sounding_utils.timer(tic)
        
        tic = time.time()
        _ = agg_delayed(ds)
        sounding_utils.timer(tic)
        del ds
    
    client.shutdown()
    subprocess.run("qdel $PBS_JOBID", shell=True, capture_output=True, encoding='utf-8')