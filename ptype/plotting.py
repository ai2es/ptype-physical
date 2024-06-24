# plotting utilities
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix

from cartopy import crs as ccrs
from cartopy import feature as cfeature

import os
from os.path import join
from glob import glob


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


# plotting functions
def ptype_hist(df, col, dataset, model_name, bins=None, save_location=None, transparent_fig=False):
    """
    Function to plot a histogram of a specified variable when
    the percent of each ptype is greater than 0.

    requires 4 columns to be within the dataframe: ra_percent, sn_percent,
    pl_percent, rzra_percent.

    transparent_fig only matters if saving out the figure. True or False are the options
    """
    if not type(df) is pd.core.frame.DataFrame:
        raise TypeError("df needs to be a dataframe")

    ra = df[col][df["ra_percent"] > 0]
    sn = df[col][df["sn_percent"] > 0]
    pl = df[col][df["pl_percent"] > 0]
    fzra = df[col][df["fzra_percent"] > 0]
    classes = ["ra", "sn", "pl", "fzra"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    if bins is None:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, density=True)
            ax.ravel()[p].set_title(f"{dataset} {col} {classes[p]}")
    else:
        for p, ptype in enumerate([ra, sn, pl, fzra]):
            ax.ravel()[p].hist(ptype, bins=bins, density=True)
            ax.ravel()[p].set_title(f"{dataset} {col} {classes[p]}")

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight", transparent=transparent_fig)

    plt.show()


def plot_2d_hist(
    x,
    y,
    model_name,
    bins=None,
    title=None,
    xlabel=None,
    ylabel=None,
    save_location=None,
    transparent_fig=False):

    """
    Parameters:
        x (array-like): Values of the first variable.
        y (array-like): Values of the second variable.
        bins (int or array-like, optional): The number of bins to use for the histogram. If not provided, a default binning will be used.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        save_location (str, optional): The file path where the plot will be saved.
        transparent_fig: if you want it the saved out figure to be transparent 
    """

    fig, ax = plt.subplots(dpi=150)
    cmap = cm.binary
    if bins:
        ax.hist2d(x, y, bins, cmap=cmap)
    else:
        ax.hist2d(x, y, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.colorbar(cm.ScalarMappable(cmap=cmap))
    ax.grid(True, alpha=0.25)

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight", transparent=transparent_fig)

    plt.show()


def plot_scatter(
    x, y, model_name, title=None, xlabel=None, ylabel=None, save_location=None, transparent_fig=False):
    """
    Plot a scatter plot of data points with optional title, labels, and saving options.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        title (str, optional): The title of the plot. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        save_location (str, optional): The file path to save the plot image. Defaults to None.
        transparent_fig: if you want it the saved out figure to be transparent 
    """

    fig, ax = plt.subplots(dpi=150)
    ax.scatter(x, y, s=2, c="k")
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    x1 = np.linspace(-40, 36, 1000) # any reason why a user would want to change this?
    y1 = x1
    ax.plot(x1, y1, "-b")
    ax.grid(True, alpha=0.25)

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight", transparent=transparent_fig)

    plt.show()


def plot_confusion_matrix(
    data, classes, font_size=10, normalize=False, axis=1, cmap=plt.cm.Blues, save_location=None
):
    """
    Function to plot a confusion matrix using seaborn heatmap

    data: dictionary, generally has test, validate, and training data
    classes: different p-types to be tested, list
    normalize: if you want the confusion matrix to be normalized or not. Needs to be True or False.
    """
    if not isinstance(data, dict):
        raise TypeError("Data needs to be a dictionary")

    fig, axs = plt.subplots(
        nrows=1, ncols=len(data), figsize=(10, 3.5), sharex="col", sharey="row"
    )

    for i, (key, ds) in enumerate(data.items()):
        ax = axs[i]
        if normalize:
            norm = 'true'
        else:
            norm = None
        cm = confusion_matrix(ds["true_label"], ds["pred_label"], normalize=norm)

        if normalize:    
            sns.heatmap(cm,
                annot=True,
                xticklabels=classes,
                yticklabels=classes,
                cmap=cmap,
                vmin=0, 
                vmax=1,
                fmt='.2f',         
                ax=ax)
        else:
            sns.heatmap(cm,
                annot=True,
                xticklabels=classes,
                yticklabels=classes,
                cmap=cmap,
                fmt='.0f',         
                ax=ax)

        ax.set_title(key.title(), fontsize=font_size)
        ax.tick_params(axis='y', rotation=0)
        
        if i == 0:
            ax.set_ylabel("True label", fontsize=font_size)
        ax.set_xlabel("Predicted label", fontsize=font_size)
    
    plt.tight_layout()
    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")



def compute_cov(df, col="pred_conf", quan="uncertainty", ascending=False):
    df = df.copy()
    df = df.sort_values(col, ascending=ascending)
    df["dummy"] = 1
    df[f"cu_{quan}"] = df[quan].cumsum() / df["dummy"].cumsum()
    df[f"cu_{col}"] = df[col].cumsum() / df["dummy"].cumsum()
    df[f"{col}_cov"] = df["dummy"].cumsum() / len(df)
    return df


def coverage_figures(
    test_data, output_cols, colors=None, title=None, save_location=None
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey="col")

    test_data["accuracy"] = (
        test_data["pred_label"] == test_data["true_label"]
    ).values.astype(int)

    _test_data_sorted = compute_cov(test_data, col="pred_conf", quan="accuracy")
    ax1.plot(_test_data_sorted["pred_conf_cov"], _test_data_sorted["cu_accuracy"])

    num_classes = test_data["true_label"].nunique()
    for label in range(num_classes):
        cond = test_data["true_label"] == label
        _test_data_sorted = compute_cov(
            test_data[cond], col="pred_conf", quan="accuracy"
        )
        ax2.plot(
            _test_data_sorted["pred_conf_cov"],
            _test_data_sorted["cu_accuracy"],
            c=colors[label],
        )

    if "evidential" in test_data:
        _test_data_sorted = compute_cov(
            test_data, col="evidential", quan="accuracy", ascending=True
        )
        ax1.plot(
            _test_data_sorted["evidential_cov"],
            _test_data_sorted["cu_accuracy"],
            ls="--",
        )
        for label in range(num_classes):
            c = test_data["true_label"] == label
            _test_data_sorted = compute_cov(
                test_data[c], col="evidential", quan="accuracy", ascending=True
            )
            ax2.plot(
                _test_data_sorted["evidential_cov"],
                _test_data_sorted["cu_accuracy"],
                c=colors[label],
                ls="--",
            )

    if title is not None:
        ax1.set_title(title)

    ax1.set_ylabel("Cumulative accuracy")
    ax1.set_xlabel("Coverage (sorted by confidence/uncertainty)")
    ax2.set_xlabel("Coverage (sorted by confidence/uncertainty)")
    ax1.legend(["Confidence", "Uncertainty"], loc="best")
    ax2.legend(output_cols, loc="best")
    plt.tight_layout()

    if save_location:
        plt.savefig(save_location, dpi=300, bbox_inches="tight")


def conus_plot(
    df, dataset="mping", column="pred_label", title="Predicted", save_path=False
):
    '''
    Contiential US plot. Depends on cartopy.
    '''

    latN = 54.0
    latS = 20.0
    lonW = -63.0
    lonE = -125.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    colors = {0: "lime", 1: "dodgerblue", 2: "red", 3: "black"}

    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    _ = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1, 2, 4, 3]
    if dataset == "ASOS":
        for i in range(4):
            ax.scatter(
                df["rand_lon"][df[column] == i] - 360,
                df["rand_lat"][df[column] == i],
                c=df["true_label"][df[column] == i].map(colors),
                s=3,
                transform=ccrs.PlateCarree(),
                zorder=zorder[i],
                alpha=0.2,
            )
    else:
        for i in range(4):
            ax.scatter(
                df["lon"][df[column] == i] - 360,
                df["lat"][df[column] == i],
                c=df[column][df[column] == i].map(colors),
                s=60,
                transform=ccrs.PlateCarree(),
                zorder=zorder[i],
                alpha=0.2,
            )

    first_day = str(min(df["datetime"])).split(" ")[0]
    last_day = str(max(df["datetime"])).split(" ")[0]
    plt.legend(
        colors.values(),
        labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
        fontsize=24,
        markerscale=3,
        loc="lower right",
    )
    plt.title(f"{dataset} {first_day} to {last_day} {title} Labels", fontsize=30)
    if save_path is not False:
        fn = os.path.join(
            save_path, f"{dataset}_{column}_{first_day}_{last_day}_truelabels.png"
        )
        plt.savefig(fn, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def save(title, outname, ax=None):
    '''save figure to specified location'''
    print(f'saving {outname}')
    if isinstance(ax, np.ndarray):
        ax[0].set_title(title[0])
        ax[1].set_title(title[1])
    else:
        plt.title(title)
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close()


def labels_video(
    test_data,
    case,
    dataset="mping",
    column="pred_label",
    title="Predicted",
    save_path=False,
):

    latN = 50.0
    latS = 23.0
    lonW = -74.0
    lonE = -120.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    colors = {0: "lime", 1: "dodgerblue", 2: "red", 3: "black"}
    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    zorder = [1, 2, 4, 3]

    def update(k):
        # fig = plt.figure(figsize=(12, 8))
        ax.cla()
        ax.set_extent([lonW, lonE, latS, latN])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))

        case_ids = case[k : (k + 1)]
        CCC = test_data["day"].isin(case_ids)
        df = test_data[CCC].copy()
        if dataset == "ASOS":
            for i in range(4):
                ax.scatter(
                    df["lon"][df[column] == i] - 360,
                    df["lat"][df[column] == i],
                    c=df["true_label"][df[column] == i].map(colors),
                    s=10,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder[i],
                    alpha=0.25,
                )
        else:
            for i in range(4):
                ax.scatter(
                    df["lon"][df[column] == i] - 360,
                    df["lat"][df[column] == i],
                    c=df[column][df[column] == i].map(colors),
                    s=10,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder[i],
                    alpha=0.25,
                )

        first_day = str(min(df["datetime"])).split(" ")[0]
        ax.legend(
            colors.values(),
            labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"],
            fontsize=10,
            markerscale=1,
            loc="lower right",
        )
        ax.set_title(f"{first_day} {title}", fontsize=12)
        plt.tight_layout()
        return ax

    ani = FuncAnimation(fig, update, frames=np.arange(len(case)))
    plt.show()
    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif, dpi=300)


def video(
    test_data,
    case,
    col="pred_conf",
    label="probability",
    title="NENoreaster",
    save_path=False,
):

    latN = 50.0
    latS = 23.0
    lonW = -74.0
    lonE = -120.0
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)
    res = "50m"  # Coarsest and quickest to display; other options are '10m' (slowest) and '50m'.
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonW, lonE, latS, latN])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))

    def update(k):
        ax.cla()
        ax.set_extent([lonW, lonE, latS, latN])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))
        case_ids = case[k : (k + 1)]
        CCC = test_data["day"].isin(case_ids)
        df = test_data[CCC].copy()
        for i in range(4):
            sc = ax.scatter(
                df["lon"][df["pred_label"] == i] - 360,
                df["lat"][df["pred_label"] == i],
                c=df[col][df["pred_label"] == i],
                s=10,
                transform=ccrs.PlateCarree(),
                cmap="cool",
                vmin=0,
                vmax=df[col].max(),
            )
        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.025, shrink=0.9325)
        cbar.set_label(f"{label}", size=12)
        ax.set_title(case_ids[0])
        plt.tight_layout()
        return (ax,)

    ani = FuncAnimation(fig, update, frames=np.arange(len(case)))
    # plt.show()
    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif, dpi=300)

# end of file