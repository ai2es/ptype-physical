import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from functools import partial

import os
from os.path import join
import subprocess

import sys
sys.path.append('../') # lets us import ptype package from the subdir
#import ptype.

import sounding_utils
from importlib import reload

from joblib import Parallel, delayed
from xhistogram.xarray import histogram

import time

def get_metadata(path):
    path = os.path.normpath(path)
    split_path = path.split(os.sep)    
    metadata_dict = {'case_study': [split_path[-4]],
                     'day': [split_path[-2]],
                    }
    return metadata_dict
    
def xr_map_reduce(base_path, func, n_jobs=-1): #works only on campg
    dirpaths = []
    for (dirpath, dirnames, filenames) in os.walk(base_path):
        #if there are subdirs in the dir skip this loop
        if dirnames or not filenames: continue 
    
        dirpaths.append(dirpath)
    if n_jobs == -1:
        num_cpus = (subprocess.run("qstat -f $PBS_JOBID | grep Resource_List.ncpus", 
                                  shell=True, capture_output=True, encoding='utf-8').stdout.split()[-1]
                    if 'glade' in os.getcwd() else
                    os.cpu_count()
        ) 
        n_jobs = min(len(dirpaths), int(num_cpus))
        
    ########################## map and reduce ##############################
    results = Parallel(n_jobs=n_jobs)(delayed(xr_map)(path, func) for path in dirpaths)
    return xr.concat(results, dim=('time')) #each result ds will be for a different time
        
def xr_map(dirpath, func):
    ds = xr.open_mfdataset(join(dirpath, "*.nc"), 
                            concat_dim='valid_time', 
                            combine='nested',
                            engine='netcdf4', 
                            decode_cf=False)
    
    ds = sounding_utils.filter_latlon(ds)
    ds = ds.where((
      (ds['crain'] == 1) | 
      (ds['csnow'] == 1) | 
      (ds['cicep'] == 1) | 
      (ds['cfrzr'] == 1)
    ))
    
    if 'wb_h' not in list(ds.keys()):
        ds = sounding_utils.wb_stull(ds)

    #adds metadata corresponding to the folder
    metadata_dict = get_metadata(dirpath)
    ds = ds.expand_dims(metadata_dict)
    return func(ds)

def compute_func(ds):
    proftypes = ['t_h','dpt_h','wb_h']
    
    res_dict = {'num_obs': [],
                'frac_abv': [],
                'means': [],
                'hists': []}
    
    for ptype in ['icep', 'frzr', 'snow', 'rain']:
        for model in ['ML_c', 'c']:
            predtype = model + ptype
            subset = ds[proftypes].where(ds[predtype] == 1)
            
            num_obs = subset['t_h'].isel(heightAboveGround=0).count(dim=('x','y'))
            obs = num_obs
            num_obs = xr.Dataset({'num_obs': num_obs}).drop_vars('heightAboveGround')
            
            frac_abv = sounding_utils.frac_abv_split_time(subset, proftypes)
            frac_abv = frac_abv.rename({var: f'{var}_fabv' for var in proftypes})
            ####### compute means ##############
            mean = subset[proftypes].mean(dim=('x','y'))
            mean = mean.rename({var: f'{var}_mean' for var in proftypes})

            ####### compute histograms ############
            bins = np.arange(-60,40,0.1)
            densities = ({f'{var}_hist': (
                    histogram(subset[var], bins=bins, dim=['x', 'y'], density=True)
                    .rename({f'{var}_bin': 'bin'})
                    ) for var in proftypes})
            densities = xr.Dataset(densities)
            
            results = {'num_obs': num_obs,
                       'frac_abv': frac_abv,
                        'means': mean,
                        'hists': densities}
            results = {k: v.expand_dims({'predtype':[predtype]}) for k,v in results.items()}
            
            for k,v in res_dict.items():
                res_dict[k].append(results[k])
    
    ds_concat = [xr.concat(res_ds_list, dim='predtype') for res_ds_list in res_dict.values()]
    result = xr.merge(ds_concat)
    
    return result
        
if __name__ == '__main__':
    model = sys.argv[1]
    filename = sys.argv[2]
    
    tic = time.time()
    
    res = xr_map_reduce(f"/glade/campaign/cisl/aiml/ptype/ptype_case_studies/*/{model}/*/*", 
                        compute_func, -1)
    res.to_netcdf(f'/glade/work/dkimpara/ptype-aggs/trial_{filename}.nc')
    
    sounding_utils.timer(tic)