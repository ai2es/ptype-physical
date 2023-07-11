import numpy as np
import xarray as xr
import pandas as pd

import os
from os.path import join
import sys

sys.path.append("../")  # lets us import sounding utils package
# import ptype.

import soundings.utils as sounding_utils

from joblib import Parallel, delayed, dump
from xhistogram.xarray import histogram

import subprocess


def get_metadata(path):
    path = os.path.normpath(path)
    split_path = path.split(os.sep)
    metadata_dict = {"case_study_day": [f"{split_path[-4]}-{split_path[-2]}"]}
    return metadata_dict


def time_to_inithr(ds):
    datetime = np.datetime64(int(ds["time"].values[0]), "s")
    hour = pd.Timestamp(datetime).hour
    ds = ds.rename({"time": "init_hr"}).assign_coords({"init_hr": [hour]})
    ds["valid_time"] = ds["valid_time"].expand_dims({"init_hr": ds.init_hr})
    return ds.compute()

def get_num_cpus():
    if "glade" in os.getcwd():
        num_cpus = subprocess.run(
            f"qstat -f $PBS_JOBID | grep Resource_List.ncpus",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout.split()[-1]
    else:
        num_cpus = os.cpu_count()
    return num_cpus

def get_dirpaths(model, base_path):
    lower_dir_path = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        # if there are subdirs in the dir skip this loop
        if dirnames or not filenames:
            continue
        if model in dirpath:
            lower_dir_path.append(dirpath)
    return lower_dir_path

def xr_map_reduce(base_path, model, func, save_file, intermediate_file='', n_jobs=-1):
    print(f"opening {base_path}\n")
    print(f"saving to {save_file}\n")

    dirpaths = get_dirpaths(model, base_path)
    if n_jobs == -1:  # setting n_jobs
        num_cpus = get_num_cpus()
        print(len(dirpaths), num_cpus)
        n_jobs = min(len(dirpaths), max(int(num_cpus) - 4, 4))

    ########################## map  ##############################
    results = Parallel(n_jobs=n_jobs, timeout=99999)(
        delayed(xr_map)(path, func) for path in dirpaths #input a partial func here for various computations
    )
    #convert time to init_hr
    results = Parallel(n_jobs=n_jobs, timeout=99999)(
        delayed(time_to_inithr)(res) for res in results
    )
    # optional: could dump the intermediate file incase merging is taking a really long time
    if intermediate_file:
        dump(results, intermediate_file)
        print("dumped")
    ########################## reduce ##############################
    print('merging')
    merged_res = xr.merge(results)
    merged_res.to_netcdf(save_file)
    print(f"write to {save_file} successful")
    return merged_res  # datasets will all have some overlapping coords so need to merge


def xr_map(dirpath, func):  
    # function to call with Parallel, need to preprocess each dataset before computing
    # preprocessing shared between computations, so they are written here
    ds = xr.open_mfdataset(
        join(dirpath, "*.nc"),
        concat_dim="step",
        combine="nested",
        engine="netcdf4",
        decode_cf=False,
    )

    valid_time = ds["valid_time"]
    ds = sounding_utils.filter_latlon(ds)
    ds = ds.where(
        (
            (ds["crain"] == 1)
            | (ds["csnow"] == 1)
            | (ds["cicep"] == 1)
            | (ds["cfrzr"] == 1)
        )
    )
    if "wb_h" not in list(ds.keys()):
        ds = sounding_utils.wb_stull(ds)
    # adds metadata as a dim (case_study_day)
    metadata_dict = get_metadata(dirpath)
    ds = ds.expand_dims(metadata_dict)

    res = func(ds)
    res["valid_time"] = valid_time.expand_dims(metadata_dict)

    return res.compute()  # dont care about keeping original ds

def compute_func(ds):
    # computing mean, and densities
    proftypes = ["t_h", "dpt_h", "wb_h"]

    res_dict = {"num_obs": [], "frac_abv": [], "means": [], "hists": []}

    for ptype in ["icep", "frzr", "snow", "rain"]:
        for model in ["ML_c", "c"]:
            predtype = model + ptype
            subset = ds[proftypes].where(ds[predtype] == 1)

            num_obs = subset["t_h"].isel(heightAboveGround=0).count(dim=("x", "y"))
            num_obs = xr.Dataset({"num_obs": num_obs}).drop_vars("heightAboveGround")

            frac_abv = sounding_utils.frac_abv_split_time(subset, proftypes)
            frac_abv = frac_abv.rename({var: f"{var}_fabv" for var in proftypes})
            ####### compute means ##############
            mean = subset[proftypes].mean(dim=("x", "y"))
            mean = mean.rename({var: f"{var}_mean" for var in proftypes})

            ####### compute histograms ############
            bins = np.arange(-60, 40, 0.1)
            densities = {
                f"{var}_hist": (
                    histogram(
                        subset[var], bins=bins, dim=["x", "y"], density=True
                    ).rename({f"{var}_bin": "bin"})
                )
                for var in proftypes
            }
            densities = xr.Dataset(densities)

            results = {
                "num_obs": num_obs,
                "frac_abv": frac_abv,
                "means": mean,
                "hists": densities,
            }
            results = {
                k: v.expand_dims({"predtype": [predtype]}) for k, v in results.items()
            }

            for k in res_dict.keys():
                res_dict[k].append(results[k])

    ds_concat = [
        xr.concat(res_ds_list, dim="predtype") for res_ds_list in res_dict.values()
    ]
    result = xr.merge(ds_concat)

    return result

def compute_stats(subset, label, proftypes, predtype):
    num_obs = subset["t_h"].isel(heightAboveGround=0).count(dim=("x", "y"))
    num_obs = xr.Dataset({f"num_obs_{label}": num_obs}).drop_vars("heightAboveGround")

    frac_abv = sounding_utils.frac_abv_split_time(subset, proftypes)
    frac_abv = frac_abv.rename({var: f"{var}_fabv_{label}" for var in proftypes})
    ####### compute means ##############
    mean = subset[proftypes].mean(dim=("x", "y"))
    mean = mean.rename({var: f"{var}_mean_{label}" for var in proftypes})

    ####### compute histograms ############
    bins = np.arange(-60, 40, 0.1)
    densities = {
        f"{var}_hist_{label}": (
            histogram(
                subset[var], bins=bins, dim=["x", "y"], density=True
            ).rename({f"{var}_bin": "bin"})
        )
        for var in proftypes
    }
    densities = xr.Dataset(densities)
    ######## combine and return #############
    results = {
        "num_obs": num_obs,
        "frac_abv": frac_abv,
        "means": mean,
        "hists": densities,
    }
    results = {
                k: v.expand_dims({"predtype": [predtype]}) for k, v in results.items()
              }
    return results

def compute_by_disagree(ds):
    ptypes = ["icep", "frzr", "snow", "rain"]
    proftypes = ["t_h", "dpt_h", "wb_h"]
    other_pred = ({f'ML_c{ptype}': f'c{ptype}' for ptype in ptypes} |
                  {f'c{ptype}': f'ML_c{ptype}' for ptype in ptypes}) 
    
    res_dict = {"num_obs": [], "frac_abv": [], "means": [], "hists": []}

    for ptype in ptypes:
        for model in ["ML_c", "c"]:
            predtype = model + ptype
            
            # compute disagreement
            mask = (ds[predtype] == 1) & (ds[other_pred[predtype]] == 0)
            masked_ds = ds[proftypes].where(mask)
            results = compute_stats(masked_ds, 'disagree', proftypes, predtype)
            for k in res_dict.keys():
                res_dict[k].append(results[k])
    ds_concat = [
        xr.concat(res_ds_list, dim="predtype") for res_ds_list in res_dict.values()
    ]
    result = xr.merge(ds_concat)
    return result

def compute_by_conf(conf_levels, ds): #conf levels sublist of [0.3,0.5,0.7,0.9]
    if not isinstance(conf_levels, list):
        conf_levels = list(conf_levels)

    ptypes = ["icep", "frzr", "snow", "rain"]
    proftypes = ["t_h", "dpt_h", "wb_h"]
    
    res_dict = {"num_obs": [], "frac_abv": [], "means": [], "hists": []}

    for ptype in ptypes:
        for model in ["ML_c", "c"]:
            predtype = model + ptype
            # compute confident preds
            for confidence in conf_levels:
                if 'ML' in predtype:
                    mask = (ds['ML_' + ptype] >= confidence) & (ds['ML_c' + ptype] == 1) #confident and predicted the ptype
                else:
                    mask = (ds['ML_' + ptype] >= confidence) & (ds[predtype] == 0) #ML confident and nwp did not predict the ptype
                masked_ds = ds[proftypes].where(mask)
                results = compute_stats(masked_ds, f'{confidence}', proftypes, predtype)
                for k in res_dict.keys():
                    res_dict[k].append(results[k])
    ds_concat = [
        xr.concat(res_ds_list, dim="predtype") for res_ds_list in res_dict.values()
    ]
    result = xr.merge(ds_concat)
    return result