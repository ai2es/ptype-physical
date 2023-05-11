'''
Author: Justin Willson
Purpose: This is a collection of functions that perform QC on the mPING datasets.
'''
import os
import tqdm
import numpy as np
import xarray as xr
import metpy.calc
from metpy.units import units
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import cartopy.io.shapereader as shpreader

def add_latlon(df, date="20190202", hour="14", path_rap="/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_height"):
    '''
    Add latitude and longitude columns to the mPING data.
    '''
    x = df['x_m'].values      #get x and y indices from data
    y = df['y_m'].values
    total_pts = df.shape[0]
    
    rap_data = xr.open_dataset(os.path.join(path_rap, date, f"rap_130_{date}_{hour}00_000.nc")) #open example RAP dataset
    grid_lons = np.subtract(rap_data.longitude.values, 360)
    grid_lats = rap_data.latitude.values
    
    lats = []
    lons = []
    for i in range(total_pts): #calculate lat and lon points based on the indices
        lats.append(grid_lats[y[i], x[i]])
        lons.append(grid_lons[y[i], x[i]])
    
    df['lon'], df['lat'] = np.array(lons), np.array(lats)
    
    return df

def usa_mask(df):
    '''
    Function that flags points that are in the continental United States.
    '''
    df_lons, df_lats = df['lon'].values, df['lat'].values
    total_pts = df.shape[0]
    points = np.array([Point(lon, lat) for lon, lat in zip(df_lons, df_lats)]) #create list of shapely points
    
    shapefile = shpreader.natural_earth(resolution='110m',                
                                       category='cultural',
                                       name='admin_0_countries')   #open natural earth shapefile
    geo_df = gpd.read_file(shapefile)
    cont_usa = geo_df[geo_df['ADMIN'] == 'United States of America']['geometry'].values[0] #extract continental US polygon
    cont_usa = list(cont_usa.geoms)[0]
    
    mask = np.zeros(total_pts)
    
    for i in tqdm.tqdm(range(total_pts)):  #flip 0's to 1's if the point is within the US
        if cont_usa.contains(points[i]):
            mask[i] = 1
            
    df['usa'] = mask
        
    return df

def filter_precip(df):
    '''
    Function that flags points where the RAP model
    indicates precipitation is occurring. Precipitation can
    be any ptype.
    '''
    total_pts = df.shape[0]
    mask = np.zeros(total_pts)
    
    crain = df.CRAIN.values  #get categorical precip values
    csnow = df.CSNOW.values
    cfrzr = df.CFRZR.values
    cicep = df.CICEP.values
    
    for i in tqdm.tqdm(range(total_pts)):
        if any([crain[i] == 1.0, csnow[i] == 1.0, cicep[i] == 1.0, cfrzr[i] == 1.0]):
                mask[i] = 1
    
    df['cprecip'] = mask
                
    return df

def wetbulb_filter(df, threshold=5.0):
    '''
    Function that flags points where frozen ptypes
    are occurring and the wet bulb temperature is greater
    than a certain threshold (default is 5C).
    '''
    total_pts = df.shape[0]    #calculate total points and initialize mask
    mask = np.zeros(total_pts)
    
    ra = df.ra_percent.values  #load all relevant quantities
    sn = df.sn_percent.values
    pl = df.pl_percent.values
    fzra = df.fzra_percent.values
    
    try:                       #load wetbulb temp if present
        wetbulb = df.wetbulb_temp_0m_C.values
    except:                    #calculate wetbulb temp manually and add to dataset
        wetbulb = np.zeros(total_pts)
        temp0mC = df.TEMP_C_0_m.values
        dewtemp0mC = df.T_DEWPOINT_C_0_m.values
        pres0mhPa = df.PRES_Pa_0_m.values/100.0
        
        for i in range(total_pts):
            wetbulb_val = metpy.calc.wet_bulb_temperature(pres0mhPa[i] * units.hPa, 
                                                          temp0mC[i] * units.degC, 
                                                          dewtemp0mC[i] * units.degC)
            wetbulb_val = np.float64(wetbulb_val)
            wetbulb[i] = wetbulb_val
            
        df['wetbulb_temp_0m_C'] = wetbulb  #add new wetbulb temp column
        
    for i in range(total_pts):      #loop through all values and flag points that meet the criteria
        if any([sn[i] > 0.0, fzra[i] > 0.0, pl[i] > 0.0]) and (wetbulb[i] > threshold):
            mask[i] = 1.0
        if (ra[i] > 0.0) and (wetbulb[i] < -threshold):
            mask[i] = 1.0
                
    df[f"wetbulb{threshold}_filter"] = mask     #assign mask to new dataframe column
    
    return df