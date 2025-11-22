#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Eliot Kim
Purpose: Visualization utilities for Winter P-Type Project.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import tqdm
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import imageio
from pathlib import Path
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

# Bounding Box Coordinates: [latN, latS, lonE, lonW]
coord_dict = {"na":[50.0, 20.0, -63.0, -125.0],
 "west":[50.0, 24.0, -92.0, -125.0],
 "east":[50.0, 20.0, -63.0, -92.0],
 "southwest":[38.0, 24.0, -92.0, -125.0],
 "southeast":[41.0, 20.0, -73.0, -95.0],
 "midwest":[50.0, 36.0, -81.0, -105.0],
 "northeast":[50.0, 37.0, -63.0, -92.0],
 "northwest":[50.0, 35.0, -92.0, -125.0],
 "okla":[39.0, 31.0, -90.0, -106.0]}
 
# colors = {0:'lime', 1:'darkturquoise', 2:'red', 3:'black'}
colors = {0: 'lime', 1: 'dodgerblue', 2: 'red', 3: 'black'}
datapath = "/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/"


def ptype_map(datatype, starttime, endtime, gifname,
              imgsavepath="gif_images", gifsavepath="gifs", coords="na",
              duration=0.5):
    """
    Create and save GIF of P-Type data over specific CONUS region and time range.

    :param datatype:
    :param starttime:
    :param endtime:
    :param gifname:
    :param imgsavepath:
    :param gifsavepath:
    :param coords:
    :param duration:
    """

    startdate = datetime.strptime(starttime, "%Y%m%d %H:%M:%S")
    enddate = datetime.strptime(endtime, "%Y%m%d %H:%M:%S")    
    time_range = pd.date_range(startdate, enddate, freq="h").strftime("%Y%m%d %H:%M:%S")
    coords = coord_dict[coords]

    # Account for differences between mPING and ASOS.
    if datatype == "mping":
        if enddate >= datetime.strptime("20180101", "%Y%m%d"):
            print("Cannot map mPING after 2017.")
            return -1
        arr = sorted(glob.glob(datapath + "mPING_mixture/*"))
        lon_adj = 360
        filetype = "mPING"
    elif datatype == "asos":
        arr = sorted(glob.glob(datapath + "ASOS_mixture/*"))
        lon_adj = 0
        filetype = "ASOS"

    # Combine desired data files into pandas Dataframe
    day_range = pd.date_range(startdate, enddate, freq="d").strftime("%Y%m%d")
    files = [filename for filename in arr if any(date in filename for date in day_range)]
    df = pd.concat([pd.read_parquet(file) for file in tqdm.tqdm(files)])

    # Index Dataframe by desired mapping area
    df_area = df[(df["lat"] <= coords[0]) & (df["lat"] >= coords[1]) &
                 (df["lon"] <= coords[2] + lon_adj) & (df["lon"] >= coords[3] + lon_adj)]

    # Set ptype as maximum percentage ptype for each observation
    ptypes = ["ra_percent", "sn_percent", "pl_percent", "fzra_percent"]
    df_area["ptype_max"] = np.argmax(df_area[ptypes].values, axis=1)

    print(df_area.shape)
    
    # Set projection and map resolution
    proj = ccrs.LambertConformal(central_longitude=(coords[2] + coords[3])/2, central_latitude=(coords[0] + coords[1])/2)
    res = '50m'
    
    # Initialize legend handles
    rain = mlines.Line2D([], [], color=colors[0], marker='o', linestyle="None", label="Rain", markersize=6)
    snow = mlines.Line2D([], [], color=colors[1], marker='o', linestyle="None", label="Snow", markersize=6)
    ice = mlines.Line2D([], [], color=colors[2], marker='o', linestyle="None", label="Ice Pellets", markersize=6)
    fzra = mlines.Line2D([], [], color=colors[3], marker='o', linestyle="None", label="Freezing Rain", markersize=6)
          
    # Make map for each hour in specified time range
    for d in time_range:
        df_area_time = df_area[df_area["datetime"] == d]
        print(d, df_area_time.shape)
        plt.figure(figsize=(18, 12))
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.set_extent([coords[2], coords[3], coords[1], coords[0]])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))

        # Plot each of the four p-types
        for i in range(4):
            ax.scatter(df_area_time["lon"][df_area_time["ptype_max"] == i]-360,
                       df_area_time["lat"][df_area_time["ptype_max"] == i],
                       c=df_area_time["ptype_max"][df_area_time["ptype_max"] == i].map(colors),
                       s=50, transform=ccrs.PlateCarree())

        plt.legend(handles = [rain, snow, ice, fzra], fontsize=16, markerscale=3, loc="lower left")
        plt.title(f"{filetype} Precipitation Type, {pd.to_datetime(d).strftime('%m-%d-%Y, %H:%M:%S')} UTC", fontsize=24)
        plt.savefig(f"{imgsavepath}/{d}_{datatype}.png")
        plt.close()

    # Create GIF, remove each .png file, and save GIF
    image_path = Path(imgsavepath)

    images = sorted(list(image_path.glob(f'*_{datatype}.png')))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))
        os.remove(file_name)

    imageio.mimwrite(f'{gifsavepath}/{gifname}.gif', image_list, duration=duration)


def ptype_weather_map(datatype, starttime, endtime, gifname, imgsavepath="", gifsavepath="weather_gifs", coords="na", duration=0.5):
    """
    Create and save GIF of P-Type, Pressure, Wind, and Temperature data over specific CONUS region and time range.
    
    :param datatype: 
    :param starttime:
    :param endtime:
    :param gifname:
    :param imgsavepath:
    :param gifsavepath:
    :param coords:
    :param duration:
    """

    startdate = datetime.strptime(starttime, "%Y%m%d %H:%M:%S")
    enddate = datetime.strptime(endtime, "%Y%m%d %H:%M:%S")    
    time_range = pd.date_range(startdate, enddate, freq="h").strftime("%Y%m%d %H:%M:%S")
    day_range = pd.date_range(startdate, enddate, freq="d").strftime("%Y%m%d")
    coords = coord_dict[coords]
    
    # Account for differences between mPING and ASOS.
    if datatype == "mping":
        if enddate >= datetime.strptime("20180101", "%Y%m%d"):
            print("Cannot map mPING after 2017.")
            return -1
        arr = sorted(glob.glob(datapath + "mPING_mixture/*"))
        lon_adj = 360
        filetype = "mPING"
    elif datatype == "asos":
        arr = sorted(glob.glob(datapath + "ASOS_mixture/*"))
        lon_adj = 0
        filetype = "ASOS"

    # TODO: get RAP data files
    rap_path = "/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_nc/"
    arr_weather = sorted(glob.glob(rap_path + "/*"))
    days_files = [filename for filename in arr_weather if any(date in filename for date in day_range)]
    hours_files = []
    for file in days_files:
        arr_weather_day = sorted(glob.glob(file + "/*.nc"))
        hour_range = [time[9:11] + time[12:14] for time in time_range if time[:8] == day_range[0]]
        hours_files = hours_files + [filename for filename in arr_weather_day if any(hour in filename for hour in hour_range)]


    # Combine desired data files into pandas Dataframe
    files = [filename for filename in arr if any(date in filename for date in day_range)]

    df = pd.concat([pd.read_parquet(file) for file in tqdm.tqdm(files)])

    # Index Dataframe by desired mapping area
    df_area = df[(df["lat"] <= coords[0]) & (df["lat"] >= coords[1]) &
                 (df["lon"] <= coords[2] + lon_adj) & (df["lon"] >= coords[3] + lon_adj)]

    # Set ptype as maximum percentage ptype for each observation
    ptypes = ["ra_percent", "sn_percent", "pl_percent", "fzra_percent"]
    df_area["ptype_max"] = np.argmax(df_area[ptypes].values, axis=1)

    print(df_area.shape)
    
    # Set projection and map resolution
    proj = ccrs.LambertConformal(central_longitude=(coords[2] + coords[3])/2, central_latitude=(coords[0] + coords[1])/2)
    res = '50m'
    
    # Initialize legend handles
    rain = mlines.Line2D([], [], color=colors[0], marker='o', linestyle="None", label="Rain", markersize=6)
    snow = mlines.Line2D([], [], color=colors[1], marker='o', linestyle="None", label="Snow", markersize=6)
    ice = mlines.Line2D([], [], color=colors[2], marker='o', linestyle="None", label="Ice Pellets", markersize=6)
    fzra = mlines.Line2D([], [], color=colors[3], marker='o', linestyle="None", label="Freezing Rain", markersize=6)

    if imgsavepath != "":        
        if not os.path.isdir("gif_images/" + imgsavepath):
            os.mkdir("gif_images/" + imgsavepath)
        imgsavepath = "gif_images/" + imgsavepath
    else:
        imgsavepath = "gif_images"
          
    # Make map for each hour in specified time range
    for d, rap_file in zip(time_range, hours_files):

        # Get RAP Variables
        rap_data = xr.open_dataset(rap_file)
        lats = np.squeeze(rap_data["latitude"].values)
        lons = np.squeeze(rap_data["longitude"].values)
        data = np.squeeze(rap_data["TEMPERATURE_2M"].values)
        pres = np.squeeze(rap_data["MEAN_SEA_LEVEL"].values)
        u = np.squeeze(rap_data["UGRD"].values)[0]
        v = np.squeeze(rap_data["VGRD"].values)[0]

        # Smooth Pressure data
        sigma = 0.9
        pres = gaussian_filter(pres, sigma)

        # Get data in area 
        idx = np.where((rap_data["longitude"].values <= coords[2]+360) & (rap_data["longitude"].values >= coords[3]+360) 
                        & (rap_data["latitude"].values <= coords[0]) & (rap_data["latitude"].values >= coords[1]))

        width = idx[0].max() - idx[0].min()
        height = idx[1].max() - idx[1].min()    

        lats = lats[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]
        lons = lons[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]
        data = data[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]
        pres = pres[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]
        u = u[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]
        v = v[idx[0].min()-2:idx[0].max()+2, idx[1].min()-2:idx[1].max()+2]

        # Make Wind data sparse
        lat_wind = lats[::int(width/10), ::int(height/10)]
        lon_wind = lons[::int(width/10), ::int(height/10)]
        u_sparse = u[::int(width/10), ::int(height/10)]
        v_sparse = v[::int(width/10), ::int(height/10)]

        print(lat_wind.shape)

        df_area_time = df_area[df_area["datetime"] == d]
        print(d, df_area_time.shape)
        fig = plt.figure(figsize=(18, 12))
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.set_extent([coords[2], coords[3], coords[1], coords[0]])
        ax.add_feature(cfeature.LAND.with_scale(res))
        ax.add_feature(cfeature.OCEAN.with_scale(res))
        ax.add_feature(cfeature.COASTLINE.with_scale(res))
        ax.add_feature(cfeature.LAKES.with_scale(res), alpha=0.5)
        ax.add_feature(cfeature.STATES.with_scale(res))

        # Temperature
        temp = ax.contourf(lons-360, lats, data-273, levels=[-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30], extend="both", transform=ccrs.PlateCarree(), cmap=plt.get_cmap("Purples"))
        clb = fig.colorbar(temp, fraction=0.025, pad=0.04)
        clb.set_label('Temperature (°C)', size=20)

        # Wind
        ax.barbs(lon_wind, lat_wind, u_sparse, v_sparse, length=6, sizes=dict(emptybarb=0.25, height=0.5), transform=ccrs.PlateCarree())

        # Pressure
        c = ax.contour(lons-360, lats, pres / 100, levels=[1000, 1005, 1010, 1015, 1020, 1025], transform=ccrs.PlateCarree(), colors="k", alpha=0.5)
        ax.clabel(c, c.levels, fontsize=15, colors="k")

        # Plot each of the four p-types
        for i in range(4):
            ax.scatter(df_area_time["lon"][df_area_time["ptype_max"] == i]-360,
                       df_area_time["lat"][df_area_time["ptype_max"] == i],
                       c=df_area_time["ptype_max"][df_area_time["ptype_max"] == i].map(colors),
                       s=50, transform=ccrs.PlateCarree())

        plt.legend(handles = [rain, snow, ice, fzra], fontsize=16, markerscale=3, loc="lower left")
        plt.title(f"{filetype} Precipitation Type, {pd.to_datetime(d).strftime('%m-%d-%Y, %H:%M:%S')} UTC", fontsize=24)
        plt.savefig(f"{imgsavepath}/{d}_{datatype}.png")
        plt.close()

    # Create GIF, remove each .png file, and save GIF
    image_path = Path(imgsavepath)

    images = sorted(list(image_path.glob(f'*_{datatype}.png')))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite(f'{gifsavepath}/{gifname}.gif', image_list, duration=duration)