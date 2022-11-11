import yaml, glob
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
import utils
import time
import random

import xarray as xr

from cartopy import crs as ccrs
from cartopy import feature as cfeature
import imageio
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

import matplotlib.colors

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

def full_rap_apply(date, time, model_loc, means, stdevs, preds_save_loc, map_save_loc):
    '''
    
    '''
    
    # Load RAP data
    rap_data = xr.open_dataset(f"/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_height/{date}/rap_130_{date}_{time}_000.nc")
    temp = np.squeeze(rap_data["TMP"].data).T
    ugrd = np.squeeze(rap_data["UGRD"].data).T
    vgrd = np.squeeze(rap_data["VGRD"].data).T
    dew = np.squeeze(rap_data["T_DEWPOINT"].data).T
    input = np.concatenate((temp, dew, ugrd, vgrd), axis=2)

    # Apply scaling parameters
    means = np.load(means)
    stdevs = np.load(stdevs)
    input = ((input.reshape(-1, 268) - means) / stdevs).reshape(451, 337, 268)

    model = tf.keras.models.load_model(model_loc)

    # Row-Wise Traversal
    preds = np.expand_dims(model.predict(input[0, :, :]), axis=0)
    for i in range(1, input.shape[0]):
        preds = np.concatenate((preds, np.expand_dims(model.predict(input[i, :, :]), axis=0)), axis=0)

    in_out = rap_data.assign(mlp_preds = (["x", "y", "ptype"], preds))
    in_out.to_netcdf(preds_save_loc + f"/{date}_{time}_preds.nc")

    print("Predictions done.")

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

    # Map predictions by p-type
    bbox = [50.0, 20.0, -63.0, -125.0]
    proj = ccrs.LambertConformal(central_longitude=(bbox[2] + bbox[3])/2, central_latitude=(bbox[0] + bbox[1])/2)
    lats, lons = rap_data["latitude"].data, rap_data["longitude"].data-360

    p_levels = [i*0.1 for i in range(11)]

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

    plt.savefig(map_save_loc + f"/full_rap_pred_{date}_{time}_ptypes.jpg", dpi=300)
    plt.close()

    # Map majority p-type predictions
    colors = {0:'green', 1:'blue', 2:'red', 3:'black'}

    rain = mlines.Line2D([], [], color=colors[0], marker='o', linestyle="None", label="Rain", markersize=6)
    snow = mlines.Line2D([], [], color=colors[1], marker='o', linestyle="None", label="Snow", markersize=6)
    ice = mlines.Line2D([], [], color=colors[2], marker='o', linestyle="None", label="Ice Pellets", markersize=6)
    fzra = mlines.Line2D([], [], color=colors[3], marker='o', linestyle="None", label="Freezing Rain", markersize=6)

    cmap = matplotlib.colors.ListedColormap(["green", "blue", "red", "black"])
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,4), cmap.N) 

    res = "50m"

    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([bbox[2], bbox[3], bbox[1], bbox[0]])
    ax.add_feature(cfeature.LAND.with_scale(res))
    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.COASTLINE.with_scale(res))
    ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
    ax.add_feature(cfeature.STATES.with_scale(res))
    ax.legend(handles = [rain, snow, ice, fzra], labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=16, markerscale=3, loc="lower left")
    all = ax.contourf(lons.T, lats.T, np.argmax(preds, axis=2), transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.5)
    for i in range(4):
        ax.scatter(df_time["lon"][df_time["ptype_max"] == i]-360,
                df_time["lat"][df_time["ptype_max"] == i],
                c=df_time["ptype_max"][df_time["ptype_max"] == i].map(colors),
                s=50, transform=ccrs.PlateCarree())
    ax.legend(handles = [rain, snow, ice, fzra], labels=["Rain", "Snow", "Ice Pellets", "Freezing Rain"], fontsize=16, markerscale=3, loc="lower left")
    ax.set_title(f"Majority P-Type | mPING Simple MLP on RAP {date} {time} UTC", fontsize=18)
    plt.savefig(map_save_loc + f"/maj_rap_pred_{date}_{time}.jpg", dpi=300)
    plt.close()

def full_rap_gif(starttime, endtime, model_loc, means, stdevs, preds_save_loc, map_save_loc, gifname, duration):

    startdate = datetime.strptime(starttime, "%Y%m%d %H%M")
    enddate = datetime.strptime(endtime, "%Y%m%d %H%M")    
    time_range = pd.date_range(startdate, enddate, freq="h").strftime("%Y%m%d %H%M")

    for t in time_range:
        date = t.split(" ")[0]
        time = t.split(" ")[1]  
        print(t)
        full_rap_apply(date, time, model_loc, means, stdevs, preds_save_loc, map_save_loc)

    image_path = Path(map_save_loc)
    images = sorted(list(image_path.glob(f'maj_rap_pred_*.jpg')))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite(f'{map_save_loc}/{gifname}.gif', image_list, duration=duration)







