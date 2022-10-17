import os
import pandas as pd
import numpy as np
import xarray as xr
from pyproj import Proj
from scipy.spatial.distance import cdist

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

source = "mPING_converted"
path_save = "/glade/p/cisl/aiml/ai2es/winter_ptypes/precip_rap/"
path_rap = "/glade/p/cisl/aiml/conv_risk_intel/rap_ncei_nc/"

ds = xr.open_dataset(os.path.join(path_rap, "20130603", "rap_130_20130603_0100_000.nc"))
files = os.listdir(os.path.join(path_save, source))
files.sort()

df = pd.read_parquet(os.path.join(path_save, source, files[0]))
precip_types = ['ra', 'sn', 'pl', 'fzra']
cols_subset = ['obdate', 'obtime', 'lat', 'lon', 'precip', 'datetime', 'precip_count_byhr']
columns_rap = sorted(list(set(df.columns) - set(cols_subset)))
columns_new = ["datetime", "lat", "lon", "report_count", "ra_percent", "sn_percent", "pl_percent", "fzra_percent"] + columns_rap

for i, file in enumerate(files):
    df = pd.read_parquet(os.path.join(path_save, source, file))
    indices = find_coord_indices(ds['longitude'].values, ds['latitude'].values,
                                 list(df['lon']), list(df['lat']))
    df['lat'] = [ds['latitude'].values[x[0], x[1]] for x in indices]
    df['lon'] = [ds['longitude'].values[x[0], x[1]] for x in indices]
    df_new = pd.DataFrame(columns=columns_new)
    
    i=0
    group = df.groupby(["datetime", "lat", "lon"])
    for name, chunk in group:
        df_new.loc[i, 'datetime'], df_new.loc[i, 'lat'] , df_new.loc[i, 'lon']  = name[0], name[1], name[2]
        df_new.loc[i, columns_rap] = chunk.loc[chunk.index[0], columns_rap]
        df_new.loc[i, 'report_count'] = chunk["precip_count_byhr"].sum()
        for precip_type in precip_types:
            if chunk[chunk['precip'] == precip_type].shape[0]:
                precip_count = float(chunk.loc[chunk['precip'] == precip_type, 'precip_count_byhr'].sum())
                df_new.loc[i, f"{precip_type}_percent"] = precip_count / df_new.loc[i, 'report_count']
            else:
                df_new.loc[i, f"{precip_type}_percent"] = 0.0
        i += 1
    df_new.to_parquet(os.path.join(path_save, "mPING_mixture", file))