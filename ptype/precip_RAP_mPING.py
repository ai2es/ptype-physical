import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
from pyproj import CRS, Transformer, Proj
from scipy.spatial.distance import cdist
from metpy.calc import dewpoint_from_relative_humidity
from metpy.units import units


idx_s, idx_e = [int(a) for a in sys.argv[1:]]
print(idx_s, idx_e)

path_precip = "/glade/p/cisl/aiml/ai2es/winter_ptypes/"
path_rap = "/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_nc/"
path_save = "/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/"

precip_files = [f for f in os.listdir(path_precip) if f.endswith('.csv')]
precip_files.sort()


def find_coord_indices(lon_array, lat_array, lon_points, lat_points, dist_proj='lcc_RAP'):
    """
    Find indices of nearest lon/lat pair on a grid. Supports rectilinear and curilinear grids.
    lon_points / lat_points must be received as a list.
    Args:
        lon_array (np.array): Longitude values of coarse grid you are matching against
        lat_array (np.array): Latitude values of coarse grid you are matching against
        lon_points (list): List of Longitude points from orginal grid/object
        lat_points (list): List of Latitude points from original grid/object
        dist_proj (str): Name of projection for pyproj to calculate distances
    Returns (list):
        List of i, j (Lon/Lat) indices for coarse grid.
    """
    if dist_proj == 'lcc_WRF':
        proj = Proj(proj='lcc', R=6371229, lat_0=38, lon_0=-97.5, lat_1=32, lat_2=46)  ## from WRF HWT data
    if dist_proj == 'lcc_RAP':
        proj = Proj(proj='lcc', R=6371229, lat_0=25, lon_0=265, lat_1=25, lat_2=25)

    proj_lon, proj_lat = np.array(proj(lon_array, lat_array))  # transform to distances using specified projection
    lonlat = np.column_stack(
        (proj_lon.ravel(), proj_lat.ravel()))  # Stack all coarse x, y distances for array shape (n, 2)
    ll = np.array(proj(lon_points, lat_points)).T  # transform lists of fine grid x, y to match shape (n, 2)
    idx = cdist(lonlat, ll).argmin(0)  # Calculate all distances and get index of minimum

    return np.column_stack((np.unravel_index(idx, lon_array.shape))).tolist()


precip_types = ['ra', 'sn', 'pl', 'fzra']
df_mPING = pd.DataFrame()
for precip in precip_files:
    if precip.startswith('ASOS'):    
        continue
    else:
        df_temp = pd.read_csv(os.path.join(path_precip, precip))
        df_temp['precip'] = list(set(precip.split('.')).intersection(set(precip_types)))[0]

        if df_temp.isna().sum().sum() > 0:
            print(f"Dropping {df_temp.isna().sum().sum()} rows from {precip} because NaNs are present.")
            df_temp.dropna(inplace=True)

        try:
            datetime.strptime(df_temp.index[0], '%M/%d/%Y')
            df_temp = df_temp.reset_index().rename(columns={'index':'obdate'})
        except:
            pass

        df_temp['datetime'] = pd.to_datetime(df_temp['obdate'] + ' ' + df_temp['obtime'], format="%m/%d/%Y %H:%M:%S")
        df_temp['datetime'] = df_temp['datetime'].dt.floor(freq='H')
        df_mPING = df_mPING.append(df_temp, ignore_index=True)
del df_temp

with open("./missing_mPING.pkl", "rb") as f:
    missing_mPING = pickle.load(f)

print(df_mPING.shape)
missing_mPING = [datetime.strptime(x, '%Y%m%d').strftime('%m/%d/%Y') for x in missing_mPING][idx_s:idx_e]
df_mPING = df_mPING[df_mPING.obdate.isin(missing_mPING)]
print(df_mPING.shape)

duplicate_counts = df_mPING.groupby(['obdate', 'lat', 'lon', 'precip', 'datetime']).count()
df_mPING = df_mPING.drop_duplicates(subset=['obdate', 'lat', 'lon', 'precip', 'datetime'], keep='first', ignore_index=True)
df_mPING['precip_count_byhr'] = list(duplicate_counts['obtime'])


varsSave = ['SNOW_WATER_EQ',
            'HGT_ON_SFC',
            'SNOW_DEPTH',
            'EL_HGT',
            'TROP_PRES',
            'CRAIN',
            'CFRZR',
            'CICEP',
            'CSNOW',
            'TMP_ON_SURFACE',
            'MEAN_SEA_LEVEL',
            'PRES_ON_SURFACE',
            'POT_TEMP_2M',
            'DEWPOINT_2M',
            'DEWPOINT_DEPRES_2M',
            'UGRD_10M',
            'VGRD_10M',
            'PRES_ON_0CISOTHM',
            'HGT_ON_0CISOTHM']
varsPressure = ['HGT', 'TMP', 'RH', 'UGRD', 'VGRD', 'VVEL']
varsSurface = list(set(varsSave) - set(varsPressure))


def df_flatten(ds, x, y, varsP, varsS):
    
    df = ds.isel(x=x,y=y).to_dataframe()[varsP]
    idx0 = df.index.levels[0].astype(int).astype(str)
    idx1 = df.index.levels[1]
    df.index = df.index.set_levels([idx0, idx1])
    df = df.unstack(level='press').sort_index()
    df.columns = df.columns.map('_'.join)
    
    varsAvailable = list(set(varsS).intersection(set(ds.variables)))
    dfS = ds[varsAvailable].isel(x=x,y=y).to_dataframe()[varsAvailable]
    
    df = df.join(dfS).reset_index(drop=True)
    
    return df

def calc_dewpoint(df): # Create T_DEWPOINT columns from RH and TMP
    if df.isnull().any().any():
        print(f"DEWPOINT CONVERSION: {df['obdate'][0]}")
        print(df[df.isnull().any(axis=1)][['datetime'] + list(df.columns[df.isna().any()])])
        df = df[~df.isnull().any(axis=1)]
    for p in list(range(100, 1025, 25)):
        df_RH = units.Quantity(np.array(df[f'RH_{p}'])/100., "dimensionless")
        df_TMP =  units.Quantity(np.array(df[f'TMP_{p}']), "K")
        df[f'T_DEWPOINT_{p}'] = dewpoint_from_relative_humidity(df_TMP, df_RH) 
    return df

def convert_KtoC(df, varsUnits_dict):
    # change variables from Kelvin to Celsius
    for var, units in varsUnits_dict.items():
        if units == 'K':
            try:
                df[var] = df[var] - 273.15            
                varsUnits_dict[var] = 'C'
            except:
                continue
    return df

def add_units(df, varsUnits_dict):
    # Rename columns to include units
    for column in list(df.columns):
        if column in list(varsUnits_dict.keys()):
            df.rename(columns={column: column + '_' + varsUnits_dict[column]}, inplace=True)
    return df


columns = list(df_mPING.columns) + [v+'_'+str(i) for v in varsPressure for i in list(range(100, 1025, 25))] + varsSurface
date_group = df_mPING.groupby('obdate')
for name, date_chunk in date_group:
    with open(os.path.join(path_save, "varsUnits_dict.pkl"), 'rb') as f:
        varsUnits_dict = pickle.load(f)
    df_save = pd.DataFrame(columns=columns)
    date = datetime.strptime(name, '%m/%d/%Y').strftime('%Y%m%d')    
    datetime_group = date_chunk.groupby('datetime')
    for name, datetime_chunk in datetime_group:
        hour = name.strftime('%H')
        # try to open a dataset if one is available and not corrupted
        try:
            ds = xr.open_dataset(os.path.join(path_rap, date, f"rap_130_{date}_{hour}00_000.nc"))
        except FileNotFoundError:
            try:
                ds= xr.open_dataset(os.path.join(path_rap, date, f"ruc2anl_130_{date}_{hour}00_000.nc"))
            except Exception as e:
                print(date, hour, e)
                continue

        # calculate projected indices
        datetime_chunk['idx'] = find_coord_indices(ds['longitude'].values, ds['latitude'].values,
                                                   list(datetime_chunk['lon']), list(datetime_chunk['lat']))

        # create new merged dataframe
        for index, row in datetime_chunk.iterrows():
            try:
                ds_temp = df_flatten(ds, row['idx'][1], row['idx'][0], varsPressure, varsSurface) 
            except Exception as e:
                print("\t- flattening not possible: ", date, hour, e)
                continue
            df_temp = pd.DataFrame(row).T.join(ds_temp.rename(index={0:row.name}))
            df_save = df_save.append(df_temp, ignore_index = True)

    # add dewpoint, convert K to C, rename columns to add units, sort by datetime, and save
    df_save = calc_dewpoint(df_save)
    df_save = convert_KtoC(df_save, varsUnits_dict)
    df_save = add_units(df_save, varsUnits_dict)
    df_save = df_save.sort_values(by="datetime")
    print(f"For {date}, was able to load {df_save.shape[0]} rows out of {date_chunk.shape[0]}")
    if 0 in df_save.shape:
        print(f"Nothing to save for {date}")
    else:
        df_save.to_parquet(os.path.join(path_save, f"mPING_raw/mPING_rap_{date}.parquet"))
