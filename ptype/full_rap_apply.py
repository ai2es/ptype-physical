import glob
import os
import pandas as pd
import tqdm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors
import keras
from mlguess.keras.losses import evidential_cat_loss

from cartopy import crs as ccrs
from cartopy import feature as cfeature
import imageio

from pathlib import Path
from datetime import datetime

import xarray as xr
from scipy.ndimage.filters import gaussian_filter


# Code based on https://unidata.github.io/MetPy/latest/examples/Four_Panel_Map.html 
def plot_background(ax, bbox):
    res = "50m"
    ax.set_extent([bbox[2], bbox[3], bbox[1], bbox[0]])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))
    return ax

def full_rap_apply(date, time, model_loc, means, stdevs, preds_save_loc, rap_loc="/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_height"):
    '''
    
    '''
    
    # Load RAP data
    rap_data = xr.open_dataset(os.path.join(rap_loc, date, f"rap_130_{date}_{time}_000.nc"))
    temp = np.squeeze(rap_data["TMP"].data).T
    ugrd = np.squeeze(rap_data["UGRD"].data).T
    vgrd = np.squeeze(rap_data["VGRD"].data).T
    dew = np.squeeze(rap_data["T_DEWPOINT"].data).T
    input = np.concatenate((temp, dew, ugrd, vgrd), axis=2)

    # Apply scaling parameters
    means = np.load(means)
    stdevs = np.load(stdevs)
    input = ((input.reshape(-1, 268) - means) / stdevs).reshape(451, 337, 268)

    model = keras.models.load_model(model_loc, custom_objects=dict(loss=evidential_cat_loss))

    # Row-Wise Traversal
    preds = np.expand_dims(model.predict(input[0, :, :]), axis=0)
    for i in range(1, input.shape[0]):
        preds = np.concatenate((preds, np.expand_dims(model.predict(input[i, :, :]), axis=0)), axis=0)

    in_out = rap_data.assign(mlp_preds = (["x", "y", "ptype"], preds))
    in_out.to_netcdf(os.path.join(preds_save_loc, f"/{date}_{time}_preds.nc"))

    print("Predictions done.")
    return


def full_rap_map(date, time, preds_loc, map_save_loc, wind, lvl, ptypewise=True, rap_loc="/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_height", img_format="png", res="50m"):
    """
    Generates various maps and visualizations based on mPING forecasts, RAP weather
    data, and predictions for a specific date and time, providing a detailed view
    of precipitation-type probabilities and related meteorological data.

    Args:
        date (str): Date in the format 'YYYYMMDD' for which the maps are generated.
        time (str): Time in the format 'HHMM' (UTC) for which the maps are generated.
        preds_loc (str): Location or path of the prediction dataset file in NetCDF format.
        map_save_loc (str): Directory path to save the generated maps.
        wind (bool): If True, plots wind barbs using RAP UGRD and VGRD data.
        lvl (int): Index of the vertical level for retrieving weather-related data,
                   such as temperature and wind, from RAP.
        ptypewise (bool, optional): Indicates whether to generate individual maps for
                                    each precipitation type (True) or only a combined
                                    majority map (False). Defaults to True.
        rap_loc (str, optional): Directory path that contains RAP NCEI height datasets.
                                 Defaults to "/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_height".
        img_format (str, optional): Format to save maps (e.g., 'png', 'jpg'). Defaults to "png".
        res (str, optional): Resolution for cartographic features (e.g., "50m", "110m").
                             Defaults to "50m".

    Raises:
        TypeError: If any input arguments are provided with invalid data types.
        FileNotFoundError: If the specified path(s) to predictions or RAP data are invalid.
        ValueError: If the provided date/time does not match the available data.

    Returns:
        None: The function generates and saves maps to the specified location but
              does not return any value.
    """
    preds_data = xr.open_dataset(preds_loc)
    preds = preds_data["mlp_preds"].values
    
    rap_data = xr.open_dataset(os.path.join(rap_loc, date, f"rap_130_{date}_{time}_000.nc"))

    # Get mPING observations
    ptypes = ["ra_percent", "sn_percent", "pl_percent", "fzra_percent"]
    arr = sorted(glob.glob("/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/mPING_interpolated/*"))
    arr = [file for file in arr if date in file]   
    df = pd.concat([pd.read_parquet(x) for x in tqdm.tqdm(arr)])
    df_time = df[df["datetime"] == datetime.strptime(date + " " + time, "%Y%m%d %H%M")]
    df_time["ptype_max"] = np.argmax(df_time[ptypes].values, axis=1)

    colors = {0:'lime', 1:'dodgerblue', 2:'red', 3:'black'}

    rain = mlines.Line2D([], [], color=colors[0], marker='o', linestyle="None", label="Rain", markersize=6)
    snow = mlines.Line2D([], [], color=colors[1], marker='o', linestyle="None", label="Snow", markersize=6)
    ice = mlines.Line2D([], [], color=colors[2], marker='o', linestyle="None", label="Ice Pellets", markersize=6)
    fzra = mlines.Line2D([], [], color=colors[3], marker='o', linestyle="None", label="Freezing Rain", markersize=6)

    print("Retrieved mPING observations.")

    p_levels = [i*0.1 for i in range(11)]
    bbox = [50.0, 20.0, -63.0, -125.0]
    proj = ccrs.LambertConformal(central_longitude=(bbox[2] + bbox[3])/2, central_latitude=(bbox[0] + bbox[1])/2)
    lats, lons = rap_data["latitude"].data, rap_data["longitude"].data-360  

    if not os.path.isdir(map_save_loc):
        os.mkdir(map_save_loc)
    
    if ptypewise:

        # Map predictions by p-type
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), constrained_layout=True,
                                subplot_kw={'projection': proj})
        fig.suptitle(f"mPING Simple MLP on RAP {date} {time} UTC", fontsize=18)
        axlist = axarr.flatten()
        for ax in axlist:
            plot_background(ax, bbox)
            
        ra = axlist[0].contourf(lons.T, lats.T, preds[:, :, 0], levels=p_levels, transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Greens"))
        clb = fig.colorbar(ra, fraction=0.025, pad=0.04, ax=axlist[0])
        axlist[0].set_title("Rain Probabilities", fontsize=16)
        axlist[0].scatter(df_time["lon"][df_time["ptype_max"] == 0]-360, df_time["lat"][df_time["ptype_max"] == 0],
                            c=df_time["ptype_max"][df_time["ptype_max"] == 0].map(colors),
                            s=50, transform=ccrs.PlateCarree())

        sn = axlist[1].contourf(lons.T, lats.T, preds[:, :, 1], levels=p_levels, transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Blues"))
        clb = fig.colorbar(sn, fraction=0.025, pad=0.04, ax=axlist[1])
        axlist[1].set_title("Snow Probabilities", fontsize=16)
        axlist[1].scatter(df_time["lon"][df_time["ptype_max"] == 1]-360, df_time["lat"][df_time["ptype_max"] == 1],
                            c=df_time["ptype_max"][df_time["ptype_max"] == 1].map(colors),
                            s=50, transform=ccrs.PlateCarree())

        pl = axlist[2].contourf(lons.T, lats.T, preds[:, :, 2], levels=p_levels, transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Oranges"))
        clb = fig.colorbar(pl, fraction=0.025, pad=0.04, ax=axlist[2])
        axlist[2].set_title("Ice Pellet Probabilities", fontsize=16)
        axlist[2].scatter(df_time["lon"][df_time["ptype_max"] == 2]-360, df_time["lat"][df_time["ptype_max"] == 2],
                            c=df_time["ptype_max"][df_time["ptype_max"] == 2].map(colors),
                            s=50, transform=ccrs.PlateCarree())

        fzra = axlist[3].contourf(lons.T, lats.T, preds[:, :, 3], levels=p_levels, transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Purples"))
        clb = fig.colorbar(fzra, fraction=0.025, pad=0.04, ax=axlist[3])
        axlist[3].set_title("Freezing Rain Probabilities", fontsize=16)
        axlist[3].scatter(df_time["lon"][df_time["ptype_max"] == 3]-360, df_time["lat"][df_time["ptype_max"] == 3],
                            c=df_time["ptype_max"][df_time["ptype_max"] == 3].map(colors),
                            s=50, transform=ccrs.PlateCarree())

        plt.savefig(map_save_loc + f"/full_rap_pred_{date}_{time}_ptypes.{img_format}", dpi=300, bbox_inches="tight")
        plt.close()

    # Map majority p-type predictions
    colors = {0:'green', 1:'blue', 2:'red', 3:'black'}

    rain = mlines.Line2D([], [], color=colors[0], marker='o', linestyle="None", label="Rain", markersize=6)
    snow = mlines.Line2D([], [], color=colors[1], marker='o', linestyle="None", label="Snow", markersize=6)
    ice = mlines.Line2D([], [], color=colors[2], marker='o', linestyle="None", label="Ice Pellets", markersize=6)
    fzra = mlines.Line2D([], [], color=colors[3], marker='o', linestyle="None", label="Freezing Rain", markersize=6)

    cmap = matplotlib.colors.ListedColormap(["green", "blue", "red", "black"])
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,4), cmap.N)


    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([bbox[2], bbox[3], bbox[1], bbox[0]])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res), alpha=0.25)
    ax.legend(handles = [rain, snow, ice, fzra], labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=16, markerscale=3, loc="lower left")
    
    # Plotting P-Type Prediction
    all_cont = ax.contourf(lons.T, lats.T, np.argmax(preds, axis=2), transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.5)
    # snow_map_only = ax.contourf(lons.T, lats.T, preds[:, :, 1], levels=p_levels, transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Blues"))

    # Plotting mPING Observations
    ax.add_feature(cfeature.STATES.with_scale(res))
    ax.legend(handles = [rain, snow, ice, fzra], labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=16, markerscale=3, loc="lower left")
    for i in range(4):
        ax.scatter(df_time["lon"][df_time["ptype_max"] == i]-360,
                df_time["lat"][df_time["ptype_max"] == i],
                c=df_time["ptype_max"][df_time["ptype_max"] == i].map(colors),
                s=50, transform=ccrs.PlateCarree())

    # Plotting RAP Weather Data
    temp = rap_data["TMP"].values[0, lvl, :, :]
    print(temp.shape)
    # pres = np.squeeze(rap_data["MEAN_SEA_LEVEL"].values)
    ugrd = np.squeeze(rap_data["UGRD"].values)[lvl]
    vgrd = np.squeeze(rap_data["VGRD"].values)[lvl]

    if wind:
        # Make Wind data sparse before plotting
        width = lats.shape[1]
        height = lats.shape[0]
        lat_wind = lats[::int(width/10), ::int(height/10)]
        lon_wind = lons[::int(width/10), ::int(height/10)]
        u_sparse = ugrd[::int(width/10), ::int(height/10)]
        v_sparse = vgrd[::int(width/10), ::int(height/10)]
        ax.barbs(lon_wind, lat_wind, u_sparse, v_sparse, length=6, sizes=dict(emptybarb=0.25, height=0.5), transform=ccrs.PlateCarree())

    # Plot Temperature
    sigma = 0.9
    temp = gaussian_filter(temp, sigma)

    plt.rcParams['contour.negative_linestyle'] = 'solid'

    temp_levels = [-30, -20, -10, 0, 10, 20, 30]
    c = ax.contour(lons, lats, temp, levels=temp_levels, transform=ccrs.PlateCarree(), colors="k", alpha=1.0, zorder=3, extend="both")
    ax.clabel(c, c.levels, fontsize=15, colors="k")

    # Labeling
    ax.legend(handles = [rain, snow, ice, fzra], labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=16, markerscale=3, loc="lower left")
    ax.set_title(f"Majority P-Type | mPING Simple MLP on RAP {date} {time} UTC", fontsize=18)

    plt.savefig(map_save_loc + f"/maj_rap_pred_{date}_{time}.{img_format}", dpi=300, bbox_inches="tight")
    plt.close()

def full_rap_gif(starttime, endtime, model_loc, means, stdevs, preds_save_loc, map_save_loc, gifname, duration=0.7, wind=False, lvl=0, ptypewise=True):
    '''
    
    '''

    startdate = datetime.strptime(starttime, "%Y%m%d %H%M")
    enddate = datetime.strptime(endtime, "%Y%m%d %H%M")    
    time_range = pd.date_range(startdate, enddate, freq="h").strftime("%Y%m%d %H%M")

    for t in time_range:
        date = t.split(" ")[0]
        time = t.split(" ")[1]  
        print(t)
        full_rap_apply(date, time, model_loc, means, stdevs, preds_save_loc)
        full_rap_map(date, time, f"{preds_save_loc}/{date}_{time}_preds.nc", map_save_loc, wind, lvl, ptypewise)

    image_path = Path(map_save_loc)
    images = sorted(list(image_path.glob('maj_rap_pred_*.png')))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))
    
    imageio.mimwrite(f'{map_save_loc}/{gifname}.gif', image_list, duration=duration)

