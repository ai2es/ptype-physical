import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from joblib import load

from metpy.plots import SkewT
from metpy.calc import (
    relative_humidity_from_dewpoint,
    mixing_ratio_from_relative_humidity,
)
from metpy.units import units

from wrf import wetbulb, enable_xarray
import os

import time

# temp r, dpt bl, wetbulb  c (cyan)

COLOR_DICT = {"t_h": "r", "dpt_h": "b", "wb_h": "c"}
BBOX = {'lon_min': 225.90453, 
        'lon_max': 299.0828, 
        'lat_min': 21.138123, 
        'lat_max': 52.615654
        }

def wb_stull(ds):
    rh_h = relative_humidity_from_dewpoint(ds.t_h * units.degC, ds.dpt_h * units.degC)
    rh_h = rh_h.metpy.magnitude * 100 
    T = ds.t_h
    ds['wb_h'] = (T * np.arctan(0.151977 * np.sqrt(rh_h + 8.313659)) + 
                  np.arctan(T + rh_h) -
                  np.arctan(rh_h - 1.676331) + 
                  0.00391838 * np.power(rh_h, 3/2) * np.arctan(0.023101 * rh_h) -
                  4.686035
                 )
    return ds

def wet_bulb_from_rel_humid(ds):
    enable_xarray()
    rh_h = relative_humidity_from_dewpoint(ds.t_h * units.degC, ds.dpt_h * units.degC)
    mr_h = mixing_ratio_from_relative_humidity(
        ds.isobaricInhPa_h * units.hPa, ds.t_h * units.degC, rh_h
    )
    ds["wb_h"] = wetbulb(
        ds.isobaricInhPa_h * 100, ds.t_h + 273.15, mr_h, units="degC"
    )  # output wbulb in degC
    return ds


def filter_latlon(ds):
    mask = (
            (ds.latitude <= BBOX['lat_max']) &
            (ds.latitude >= BBOX['lat_min']) &
            (ds.longitude <= BBOX['lon_max']) &
            (ds.longitude >= BBOX['lon_min'])
           )
    return ds.where(mask.compute(), drop=True)


def open_ds_dkimpara(hour, model, cluster='casper', concat_dim="time", parallel=False):
    if "glade" in os.getcwd():
        if cluster != 'casper':
            filepath = (f"""/glade/scratch/dkimpara/ptype_case_studies/"""
                        f"""kentucky/{model}/20220223/{hour}/*.nc""")
        else:
            filepath = (f"""/glade/campaign/cisl/aiml/ptype/ptype_case_studies/"""
                        f"""kentucky/{model}/20220223/{hour}/*.nc""")
            
        ds = xr.open_mfdataset(filepath,
            concat_dim=concat_dim,
            combine="nested",
            parallel=parallel
        )
    else:  # running locally
        ds = xr.open_mfdataset(
            f"""/Users/dkimpara/gh/ptype-data/{model}/{hour}/*.nc""",
            concat_dim=concat_dim,
            combine="nested",
        )
    ds.attrs["nwp"] = model

    # todo: filter latlon

    #if "wb_h" not in list(ds.keys()):
    #    ds = wet_bulb_from_rel_humid(ds)
    return ds


### refactoring to use linespecs in the future


# set up the fig and ax
def skewCompositeFigAx(figsize=(5, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot((0, 0, 1, 1), projection="skewx", rotation=30)
    ax.grid(which="both")

    major_ticks = np.arange(-100, 100, 5)
    ax.set_xticks(major_ticks)
    ax.grid(which="major", alpha=0.5)

    # minor_ticks = np.arange(xlowlim - 60, xhighlim, 1)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.grid(which='minor', alpha=0.2)

    ax.axvline(x=0, ymin=0, ymax=1, c="0")

    ax.set_ylabel("Height above ground (m)")
    ax.set_ylim(-100, 5100)
    ax.set_xlim(-20, 20)

    return fig, ax


def frac_abv_zero(ds, x_col, total):
    num_over_zero = (ds[x_col] > 0).any(dim="heightAboveGround").sum()
    return num_over_zero / total

def frac_abv_split_time(ds, x_cols):
    np.seterr(divide='ignore', invalid='ignore')
    ds = ds[x_cols]
    num_over_zero = (ds > 0).any(dim="heightAboveGround").sum(dim=('x','y'))
    obs = ds[x_cols[0]].isel(heightAboveGround=0).count(dim=('x','y')).drop_vars('heightAboveGround')
    
    #if ((obs == 0).sum().values > 0) or (obs.isnull().sum().values > 0):
    #    print(f'Warning: obs has a 0 val ({ds["time"]}, {ds.case_study_day})')
    
    return num_over_zero / obs

def count_nulls(ds):
    nulls = ds.isnull().sum()

    for var in list(nulls.keys()):
        print(nulls[var].values)

def timer(tic, message=''):
    toc = time.time()
    duration = toc - tic
    minutes = int(duration/60)
    print(f"{message}. Elapsed time: {str(minutes) + ' minutes, ' if minutes else ''}{int(duration % 60)} seconds")


def fn_timer(fn):
    tic = time.time()
    
    out = fn()
    
    toc = time.time()    
    duration = toc - tic
    minutes = int(duration/60)
    print(f"Elapsed time: {str(minutes) + ' minutes, ' if minutes else ''}{int(duration % 60)} seconds")
    
    return out