import os
from os.path import join
from glob import glob

import pandas as pd
import xarray as xr
import numpy as np

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def get_tle_files(base_path, valid_time, n_members=1):
    '''
    Get ptype files at a given valid time.

    Arguments:
        base_path (str): path to directory
        valid_time (str): YYYY-mm-dd HHMM
        n_members (int): for gathering an ensemble of times, default is no ensemble
    '''
    date = pd.to_datetime(valid_time)
    file_names = []
    if n_members == 1:
        dt = date - pd.Timedelta(f"1h")
        date_str = f"MILES_ptype_hrrr_{dt.strftime('%Y-%m-%d_%H%M')}"
        return [join(base_path, dt.strftime("%Y%m%d"), dt.strftime("%H%M"), f"{date_str}_f01.nc")]
    for time_delta in range(n_members, 0, -1):
        dt = date - pd.Timedelta(f"{time_delta}h")
        date_str = f"MILES_ptype_hrrr_{dt.strftime('%Y-%m-%d_%H%M')}"
        path = join(base_path, dt.strftime("%Y%m%d"), dt.strftime("%H%M"), f"{date_str}_f{str(time_delta).zfill(2)}.nc")
        file_names.append(path)
    return file_names


def load_data(files, variables):
    '''
    Load ptype files with xarray and pre process before use
    '''
    ptypes = ['snow', 'rain', 'frzr', 'icep']
    def process(ds):
        '''
        Used for loading single ptype files. Ptype data is masked where HRRR has precip, ie. where the sum across 4 ptype
        classifications is greater than or equal to 1. 

        The only new variable added is {ptype} - masked evidential classification of each ptype scaled by masked evidential probability
        '''
        ds = ds[variables]
        precip_sum = ds[['crain', 'csnow', 'cicep', 'cfrzr']].to_array().sum(dim='variable')
        for ptype in ptypes:
            ds[f'ML_c{ptype}'] = xr.where(precip_sum >= 1, x=ds[f"ML_c{ptype}"], y=np.nan)
            ds[f'ML_{ptype}'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}"], y=np.nan)
            ds[f'ML_{ptype}_epi'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_epi"], y=np.nan)
            ds[f'ML_{ptype}_ale'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_ale"], y=np.nan)
            ds[ptype] = ds[f'ML_c{ptype}'].where(ds[f'ML_c{ptype}'] >= 1) * ds[f'ML_{ptype}']  # evidential classification scaled by probability
            ds[f'c{ptype}'] = xr.where(ds[f'c{ptype}'] == 1, 1, np.nan)  # convert 0s to nans for plotting purposes
        return ds
    def process_ensemble(ds):
        '''
        Used for loading ensemble of ptype files with a single valid time and multiple different init times. 
        Ptype data is masked where HRRR has precip. Masked values are set to zero before taking the average across
        ensemble members and set to nan after for plotting purposes. 

        New classifications are made for evidential predictions (f'ML_c{ptype}') and HRRR predictions (f'c{ptype}')
        with a hierarchy of importance: freezing rain > sleet > snow > rain, such that classifications don't overlap.
        Evidential categorization prediction is computed by taking the maximum mean probability across 4 ptypes and setting
        a value of 1 to the highest ptype and 0 to the rest. HRRR categorization prediction is computed in a similar 
        way, however, the classification is based on max proportion of HRRR predictions across ptypes.
        '''
        ds = ds[variables]
        precip_sum = ds[['crain', 'csnow', 'cicep', 'cfrzr']].to_array().sum(dim='variable')
        for ptype in ptypes:
            ds[f'ML_{ptype}'] = xr.where(precip_sum >= 1, x=ds[f'ML_{ptype}'], y=0) 
            ds[f'ML_{ptype}_epi'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_epi"], y=0)
            ds[f'ML_{ptype}_ale'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_ale"], y=0)

        ptype_hier = ['frzr', 'icep', 'snow', 'rain']
        concat = xr.concat([ds[f'ML_{ptype}'] for ptype in ptype_hier], dim='ptype')
        concat_tle = concat.mean("time")
        max_idx = concat_tle.argmax(dim='ptype')

        concat_hrrr = xr.concat([ds[f'c{ptype}'] for ptype in ptype_hier], dim='ptype')
        concat_hrrr_tle = concat_hrrr.mean("time")
        max_idx_tle = concat_hrrr_tle.argmax(dim='ptype')

        for i, ptype in enumerate(ptype_hier):
            ds[f'ML_c{ptype}'] = xr.where(max_idx == i, 1, np.nan) # set categorical values
            ds[ptype] = ds[f'ML_c{ptype}'] * ds[f'ML_{ptype}'] # set categorical scaled by probability
            ds[f'c{ptype}'] = xr.where((max_idx_tle == i) & (concat_hrrr_tle[i] != 0), concat_hrrr_tle[i], np.nan)
        ds = ds.mean("time").apply(lambda x: x.where(x != 0, np.nan)) # average and set 0 to nan in one line
        return ds
        
    if len(files) == 1:
        ds = xr.open_dataset(files[0]).squeeze()
        return process(ds)
    else:
        ds = xr.open_mfdataset(files, parallel=True)
        ds = process_ensemble(ds)
        return ds


def plot_hrrr_ptype(ds, cmap=['Greens', 'Blues', 'Reds', 'Greys'], ptypes=['snow', 'rain', 'frzr', 'icep'], extent=[-108, -91, 37, 47.5]):
    '''plot hrrr precip categorizations'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    lat = ds['latitude']
    lon = ds['longitude']

    for i, ptype in enumerate(ptypes):
        h = ax.pcolormesh(lon, lat, ds[f'c{ptype}'], transform=ccrs.PlateCarree(), cmap=cmap[i], vmin=0, vmax=1)
    
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    return ax
    

def plot_ptype(ds, ptype=None, cmap=['Greens', 'Blues', 'Reds', 'Greys'], ptypes=['snow', 'rain', 'frzr', 'icep'], extent=[-108, -91, 37, 47.5]):
    '''plots probabilities of each precipitation type based on the type with maximum probability'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    lat = ds['latitude']
    lon = ds['longitude']
   
    for i, ptype in enumerate(ptypes):
        h = ax.pcolormesh(lon, lat, ds[f'{ptype}'], transform=ccrs.PlateCarree(), cmap=cmap[i], vmin=0, vmax=1)
    
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    return ax


def plot_probability(ds, ptype, cmap='GnBu', extent=[-108, -91, 37, 47.5]):
    '''plots probabilities where a certain precipitation type occurs'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES)
    
    lat = ds['latitude']
    lon = ds['longitude']
    prob_data = ds[f'ML_{ptype}']
    pcm = ax.pcolormesh(lon, lat, prob_data, transform=ccrs.PlateCarree(), cmap=cmap)
    
    cbar = plt.colorbar(pcm, ax=ax, pad=0.025, fraction=0.042)
    cbar.set_label(f'Probability of {ptype}')
    return ax


def plot_uncertainty(ds, ptype, cmap='viridis', extent=[-108, -91, 37, 47.5]):
    '''plot epistemic and aleatoric uncertainty where a certain precipitation type occurs'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': projection})
    
    for ax in axs:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.4)
    
    lat = ds['latitude']
    lon = ds['longitude']
    
    ale_data = ds[f'ML_{ptype}_ale']
    epi_data = ds[f'ML_{ptype}_epi']
    
    pcm_ale = axs[0].pcolormesh(lon, lat, ale_data, transform=ccrs.PlateCarree(), vmin=0, cmap=cmap)
    plt.colorbar(pcm_ale, ax=axs[0], fraction=0.042, pad=0.025,)
    
    pcm_epi = axs[1].pcolormesh(lon, lat, epi_data, transform=ccrs.PlateCarree(), vmin=0, cmap=cmap)
    plt.colorbar(pcm_epi, ax=axs[1], fraction=0.042, pad=0.025)
    
    plt.tight_layout()
    return axs


def plot_winds(ds, base_plot, ptype=None, **kwargs):
    '''plots u and v wind quivers on top of another plot'''
    ax = base_plot(ds, ptype, **kwargs)
    u = ds['u10'].values
    v = ds['v10'].values
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    if isinstance(ax, np.ndarray): # case of multiple subplots (possibly add loop later)
        ax[0].barbs(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], length=4, linewidth=0.7, transform=ccrs.PlateCarree())
        ax[1].barbs(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], length=4, linewidth=0.7, transform=ccrs.PlateCarree())
    else:
        ax.barbs(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], length=4, linewidth=0.7, transform=ccrs.PlateCarree())
    return ax
    

def plot_temp(ds, base_plot, ptype=None, **kwargs):
    '''plots temp contours on top of another plot'''
    ax = base_plot(ds, ptype, **kwargs)
    lat = ds['latitude']
    lon = ds['longitude']
    temp_data = ds['t2m'] - 273.15  # Convert from Kelvin to Celsius
    if isinstance(ax, np.ndarray):
        for a in ax:
            contour(a, lon, lat, temp_data)
    else:
        contour(ax, lon, lat, temp_data)
    return ax
    

def plot_dpt(ds, base_plot, ptype=None, **kwargs):
    '''plots dewpoint contours on top of another plot'''
    ax = base_plot(ds, ptype, **kwargs)
    lat = ds['latitude']
    lon = ds['longitude']
    dpt_data = ds['d2m'] - 273.15  # Convert from Kelvin to Celsius
    
    if isinstance(ax, np.ndarray):
        for a in ax:
            contour(a, lon, lat, dpt_data)
    else:
        contour(ax, lon, lat, dpt_data)
    return ax


def contour(ax, lon, lat, data):
    contours = ax.contour(lon, lat, data, transform=ccrs.PlateCarree(), levels=10)
    #zero_cel = ax.contour(lon, lat, dpt_data, transform=ccrs.PlateCarree(), levels=[0], colors='black', linewidths=2)
    ax.clabel(contours, fontsize=10, colors='black')
    #cbar = plt.colorbar(contours, orientation="horizontal", fraction=0.15, pad=0.025)
    #cbar.set_label('Dew Pt Temperature (°C)')


def plot_sp(ds, base_plot, ptype=None, **kwargs):
    '''plot surface pressure contours on top of base plot'''
    ax = base_plot(ds, ptype, **kwargs)
    lat = ds['latitude']
    lon = ds['longitude']
    sp_data = ds['sp']

    if isinstance(ax, np.ndarray):
        for a in ax:
            contour(a, lon, lat, sp_data)
    else:
        contour(ax, lon, lat, sp_data)
    return ax


def save(title, outname, ax=None):
    print(f'saving {outname}')
    if isinstance(ax, np.ndarray):
        ax[0].set_title(title[0])
        ax[1].set_title(title[1])
    else:
        plt.title(title)
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close()

