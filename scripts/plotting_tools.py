import os
from os.path import join
from glob import glob

import pandas as pd
import xarray as xr
import numpy as np

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import logging

ptypes = ['snow', 'rain', 'frzr', 'icep']

def get_tle_files(base_path, valid_time, n_members=1, min_lead_time=1, init_step=1):
    '''
    Get ptype files at a given valid time.

    Arguments:
        base_path (str): path to directory
        valid_time (str): YYYY-mm-dd HHMM
        n_members (int): for gathering an ensemble of times, default is no ensemble
        min_lead_time (int): ?
        init_step (int): not implemented 
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
    def process(ds):
        ds = ds[variables]
        precip_sum = ds[['crain', 'csnow', 'cicep', 'cfrzr']].to_array().sum(dim='variable')
        for ptype in ptypes:
            ds[f'ML_c{ptype}'] = xr.where(precip_sum >= 1, x=ds[f"ML_c{ptype}"], y=np.nan)
            ds[f'ML_{ptype}'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}"], y=np.nan)
            ds[f'ML_{ptype}_epi'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_epi"], y=np.nan)
            ds[f'ML_{ptype}_ale'] = xr.where(precip_sum >= 1, x=ds[f"ML_{ptype}_ale"], y=np.nan)
            ds[ptype] = ds[f'ML_c{ptype}'].where(ds[f'ML_c{ptype}'] >= 1) * ds[f'ML_{ptype}'] # mask where precip sum >= 1 and where ML_cptype >= 1
            ds[f'c{ptype}'] = xr.where(ds[f'c{ptype}'] == 1, 1, np.nan)
        return ds
    def process_ensemble(ds):
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
            #ds[f'ML_max_c{ptype}'] = xr.where(max_idx == i, concat_tle[i], np.nan)
            ds[ptype] = ds[f'ML_c{ptype}'] * ds[f'ML_{ptype}'] # set categorical multiplied by prob
            ds[f'max_c{ptype}'] = xr.where((max_idx_tle == i) & (concat_hrrr_tle[i] != 0), concat_hrrr_tle[i], np.nan)
        ds = ds.mean("time").apply(lambda x: x.where(x != 0, np.nan)) # sum and set 0 to nan in one line
        return ds
        
    if len(files) == 1:
        ds = xr.open_dataset(files[0]).squeeze()
        return process(ds)
    else:
        ds = xr.open_mfdataset(files, parallel=True)
        ds = process_ensemble(ds)
        return ds


def plot_hrrr_ptype(ds):
    '''plot hrrr precip categorizations'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent([-108, -91, 37, 47.5], crs=ccrs.PlateCarree())
    cmaps = ["Greens", "Blues", "Greys", "Reds"]
    ptyped = ['rain', 'snow', 'icep', 'frzr'] # ordered in terms of importance, most important plotted last
    lat = ds['latitude']
    lon = ds['longitude']

    for i, ptype in enumerate(ptyped):
        h = ax.pcolormesh(lon, lat, ds[f'max_c{ptype}'] * 0.9, transform=ccrs.PlateCarree(), cmap=cmaps[i], vmin=0, vmax=1)
    
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    return ax
    

def plot_ptype(ds, ptype=None):
    '''plots probabilities of each precipitation type based on the type with maximum probability'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    #projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent([-108, -91, 37, 47.5], crs=ccrs.PlateCarree())
    cmaps = ["Blues", "Greens", "Greys", "Reds"]
    ptyped = ['snow', 'rain', 'icep', 'frzr']
    lat = ds['latitude']
    lon = ds['longitude']
   
    for i, ptype in enumerate(ptyped):
        h = ax.pcolormesh(lon, lat, ds[f'{ptype}'], transform=ccrs.PlateCarree(), cmap=cmaps[i], vmin=0, vmax=1)
    
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    return ax


def plot_probability(ds, ptype):
    '''plots probabilities where a certain precipitation type occurs'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    extent = [-108, -91, 37, 47.5]    
    #projection=ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': projection})
    ax.set_extent([-108, -91, 37, 47.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES)
    
    lat = ds['latitude']
    lon = ds['longitude']
    prob_data = ds[f'ML_{ptype}']
    pcm = ax.pcolormesh(lon, lat, prob_data, transform=ccrs.PlateCarree(), cmap='GnBu')
    
    cbar = plt.colorbar(pcm, ax=ax, pad=0.025, fraction=0.042)
    cbar.set_label(f'Probability of {ptype}')
    return ax


def plot_uncertainty(ds, ptype):
    '''plot epistemic and aleatoric uncertainty where a certain precipitation type occurs'''
    projection = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5, 38.5))
    extent = [-108, -91, 37, 47.5]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': projection})
    
    for ax in axs:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.4)
    
    lat = ds['latitude']
    lon = ds['longitude']
    
    ale_data = ds[f'ML_{ptype}_ale']
    epi_data = ds[f'ML_{ptype}_epi']
    
    pcm_ale = axs[0].pcolormesh(lon, lat, ale_data, transform=ccrs.PlateCarree(), vmin=0)
    plt.colorbar(pcm_ale, ax=axs[0], fraction=0.042, pad=0.025,)
    
    pcm_epi = axs[1].pcolormesh(lon, lat, epi_data, transform=ccrs.PlateCarree(), vmin=0)
    plt.colorbar(pcm_epi, ax=axs[1], fraction=0.042, pad=0.025)
    
    plt.tight_layout()
    return axs


def plot_winds(ds, base_plot, ptype=None):
    '''plots u and v wind quivers on top of another plot'''
    ax = base_plot(ds, ptype)
    u = ds['u10'].values
    v = ds['v10'].values
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    if isinstance(ax, np.ndarray): # case of multiple subplots (possibly add loop later)
        ax[0].quiver(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], width=0.0015, transform=ccrs.PlateCarree())
        ax[1].quiver(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], width=0.0015, transform=ccrs.PlateCarree())
    else:
        ax.quiver(lon[::20, ::20], lat[::20, ::20], u[::20, ::20], v[::20, ::20], width=0.0015, transform=ccrs.PlateCarree())
    return ax
    

def plot_temp(ds, base_plot, ptype=None):
    '''plots temp contours on top of another plot'''
    ax = base_plot(ds, ptype)
    lat = ds['latitude']
    lon = ds['longitude']
    temp_data = ds['t2m'] - 273.15  # Convert from Kelvin to Celsius
    if isinstance(ax, np.ndarray):
        for a in ax:
            contour(a, lon, lat, temp_data)
    else:
        contour(ax, lon, lat, temp_data)
    return ax
    

def plot_dpt(ds, base_plot, ptype=None):
    '''plots dewpoint contours on top of another plot'''
    ax = base_plot(ds, ptype)
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


def plot_sp(ds, base_plot, ptype=None):
    '''plot surface pressure contours on top of base plot'''
    ax = base_plot(ds, ptype)
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


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    base_path = '/glade/derecho/scratch/cbecker/ptype_real_time/winter_2023_2024/hrrr'
    valid_time = '2023-12-26 0100'
    time = valid_time.replace(' ', '_')
    n_members = 18
    output_dir = f'./plots/{time}/'

    # check if path exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    variables = [
        'u10', 'v10', 'ML_rain', 'ML_crain', 'ML_snow', 'ML_csnow', 'ML_frzr', 'ML_cfrzr', 'ML_icep', 'ML_cicep',
        'crain', 'csnow', 'cfrzr', 'cicep', 'ML_rain_ale', 'ML_rain_epi', 'ML_snow_ale', 'ML_snow_epi', 'ML_frzr_ale',
        'ML_frzr_epi', 'ML_icep_ale', 'ML_icep_epi', 'd2m', 't2m', 'sp'
    ]
    title = f''
    outname = f''
    
    if n_members > 1:
        title += f'Time Lagged Ensemble'
        outname += f'time_lagged_'
    
    files = get_tle_files(base_path, valid_time, n_members)
    ds = load_data(files, variables)
    logger.info(f'{len(files)} files loaded')
    
    title_ptype = f'{title} {valid_time} Precip\nSnow = Blue, Rain = Green, Sleet = Purple, Freezing Rain = Red'
    out_ptype = f'{output_dir}{outname}ptype_{time}'
    
    plot_hrrr_ptype(ds)
    save(title_ptype, f'{out_ptype}_hrrr.png')

    plot_ptype(ds)
    save(title_ptype, f'{out_ptype}.png')

    plot_winds(ds, plot_ptype)
    save(title_ptype, f'{out_ptype}_quivers.png')

    plot_temp(ds, plot_ptype)
    save(title_ptype, f'{out_ptype}_temp.png')

    plot_dpt(ds, plot_ptype)
    save(title_ptype, f'{out_ptype}_dpt.png')

    plot_sp(ds, plot_ptype)
    save(f'{title_ptype}', f'{out_ptype}_sp.png')

    for ptype in ptypes:
        logger.info(f'plotting prob {ptype}')
        title_prob = f'{title} probability {ptype} {valid_time}'
        out_prob = f'{output_dir}{outname}prob_{time}_{ptype}'
        plot_probability(ds, ptype)
        save(title_prob, f'{out_prob}.png')
        
        plot_winds(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_quivers.png')

        plot_temp(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_temp.png')

        plot_dpt(ds, plot_probability, ptype=ptype)
        save(f'{title_prob}', f'{out_prob}_dpt.png')
        
        logger.info(f'plotting uncertainty {ptype}')
        title_uncert = [f'{title} Aleatoric Uncertainty {ptype} {valid_time}', f'{title} Epistemic Uncertainty {ptype} {valid_time}']
        out_uncert = f'{output_dir}{outname}uncert_{time}_{ptype}'
        ax = plot_uncertainty(ds, ptype)
        save(title_uncert, f'{out_uncert}.png', ax)

        ax = plot_winds(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_quivers.png', ax)
        
        ax = plot_temp(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_temp.png', ax)

        ax = plot_dpt(ds, plot_uncertainty, ptype=ptype)
        save(title_uncert, f'{out_uncert}_dpt.png', ax)

    