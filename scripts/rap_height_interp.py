import sys
import datetime
from os import listdir, walk, makedirs
from os.path import join
import numpy as np
import xarray as xr
from numba import jit
from metpy import calc
from metpy.units import units


varsPressure = ['HGT', 'TMP', 'RH', 'UGRD', 'VGRD']
height_levels = np.arange(0, 16750, 250)
pressure_levels = np.arange(1000, 75, -25)
vars_to_load = ['HGT', 'RH', 'TMP', 'UGRD', 'VGRD',
                'TEMPERATURE_2M', 'UGRD_10M', 'VGRD_10M',
                'PRES_ON_SURFACE', 'DEWPOINT_2M']

def getDatesWithData(path_data):
    directoryList = []

    directoryPotential = [x[0] for x in walk(path_data)][1:]
    directoryPotential.sort()

    for directory in directoryPotential:
        if len([f for f in listdir(directory) if f.endswith('.nc')]) > 0:
            directoryList.append(directory)
    return directoryList

def calc_dewpoint(ds):
    ds['RH'] = (ds['RH'].where(ds['RH'] > 0.0, 1.0))/100

    RH = units.Quantity(np.array(ds['RH']), "dimensionless")
    TMP = units.Quantity(np.array(ds['TMP']), "K")

    dims = ['time', 'press', 'y', 'x']
    dewpoints = calc.dewpoint_from_relative_humidity(TMP, RH).magnitude
    ds['T_DEWPOINT'] = (dims, dewpoints)
    ds['T_DEWPOINT'].attrs = {'short_name': 'T_DEWPOINT',
                              'long_name': 'Dewpoint Temperature',
                              'level': '1000 mb',
                              'units': 'C'}
    ds['TMP'] = ds['TMP'] - 273.15
    return ds

@jit(nopython=True)
def interp_height_fast(x_arr, x_heights, interp_heights):
    output_arr = np.zeros((interp_heights.shape[0], x_arr.shape[1], x_arr.shape[2]), dtype=x_arr.dtype)
    for y in range(x_arr.shape[1]):
        for x in range(x_arr.shape[2]):
            output_arr[:, y, x] = np.interp(interp_heights,
                                            x_heights[:, y, x],
                                            x_arr[:, y, x])
    return output_arr

@jit(nopython=True)
def interp_height_pres(pres_arr, x_heights, interp_heights):
    output_arr = np.zeros((interp_heights.shape[0], x_heights.shape[1], x_heights.shape[2]), dtype=pres_arr.dtype)
    for y in range(x_heights.shape[1]):
        for x in range(x_heights.shape[2]):
            output_arr[:, y, x] = np.interp(interp_heights,
                                            x_heights[:, y, x],
                                            pres_arr)
    return output_arr

def interpolate_file_nc(ds, height_levels):
    empty_array = np.zeros((1, len(height_levels), ds.y.values.shape[0], ds.x.values.shape[0]), dtype=np.float32)
    ds_h = xr.Dataset(
        data_vars=dict(
            TMP=(["time", "height", "y", "x"], empty_array.copy()),
            UGRD=(["time", "height", "y", "x"], empty_array.copy()),
            VGRD=(["time", "height", "y", "x"], empty_array.copy()),
            PRES=(["time", "height", "y", "x"], empty_array.copy()),
            T_DEWPOINT=(["time", "height", "y", "x"], empty_array.copy()),
        ),
        coords=dict(
            y=ds.y.values.astype(np.float32),
            x=ds.x.values.astype(np.float32),
            latitude=(["y", "x"], ds.latitude.values.astype(np.float32)),
            longitude=(["y", "x"], ds.longitude.values.astype(np.float32)),
            time=np.atleast_1d(ds.time.values[0]),
            height=height_levels,))
    for var in ['TMP', 'UGRD', 'VGRD', 'T_DEWPOINT']:
        ds_h[var][0] = interp_height_fast(ds[var][0].values, ds["HGT_AGL"][0].values, height_levels)
    ds_h["PRES"][0] = interp_height_pres(ds["press"].values, ds["HGT_AGL"][0].values, height_levels)
    return ds_h

def set_surface_values(ds_h, ds):
    ds_h['TMP'][0, 0] = ds['TEMPERATURE_2M'][0] - 273.15
    ds_h['UGRD'][0, 0] = ds['UGRD_10M'][0]
    ds_h['VGRD'][0, 0] = ds['VGRD_10M'][0]
    ds_h['PRES'][0, 0] = ds['PRES_ON_SURFACE'][0]
    TMP = np.clip(ds['DEWPOINT_2M'].values, a_max=ds['TEMPERATURE_2M'].values, a_min=False)
    ds_h['T_DEWPOINT'][0, 0] = TMP[0] - 273.15
    return ds_h

def set_attributes(ds_h):
    ds_h['TMP'].attrs = {'short_name': 'TMP',
                         'long_name': 'Temperature',
                         'units': 'C'}
    ds_h['UGRD'].attrs = {'short_name': 'UGRD',
                          'long_name': 'U-Component of Wind',
                          'units': 'm/s'}
    ds_h['VGRD'].attrs = {'short_name': 'VGRD',
                          'long_name': 'V-Component of Wind',
                          'units': 'm/s'}
    ds_h['PRES'].attrs = {'short_name': 'PRES',
                          'long_name': 'Pressure',
                          'units': 'Pa'}
    ds_h['T_DEWPOINT'].attrs = {'short_name': 'T_DEWPOINT',
                                'long_name': 'Dewpoint Temperature',
                                'units': 'C'}
    return ds_h

def main():
    
    path_data = "/glade/scratch/ggantos/conv_risk_intel/rap_ncei_nc/"
    path_save = "/glade/scratch/ggantos/conv_risk_intel/rap_ncei_height/"
    start_date = sys.argv[1] #'20160509'
    end_date = sys.argv[2] #'20160510'
    print(start_date, end_date)
    
    directories = getDatesWithData(path_data)
    dates = [x[-8:] for x in directories]
    dates.sort()

    start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    dates_dt = [datetime.datetime.strptime(dt, "%Y%m%d") for dt in dates]
    dates_to_process = [dt for dt in dates_dt if start_date <= dt and dt < end_date]
    dates_to_process = [dt.strftime("%Y%m%d") for dt in dates_to_process]
    
    height_vars = ["TMP", "UGRD", "VGRD", "PRES", "T_DEWPOINT"]
    comp = dict(zlib=True, complevel=4, least_significant_digit=3)
    encoding = {var: comp for var in height_vars}

    for date in dates_to_process:
        makedirs(join(path_save, date), exist_ok=True)
        files = [f for f in listdir(join(path_data, date)) if f.endswith('.nc')]

        for file in files:
            try:
                ds = xr.open_dataset(join(path_data, date, file))
                ds['HGT_AGL'] = (ds['HGT'] - ds['HGT_ON_SFC'])
                ds[vars_to_load].load()
            except Exception as e:
                print(e)
                continue

            # interpolate
            ds = calc_dewpoint(ds)
            ds_h = interpolate_file_nc(ds, height_levels)
            ds_h = set_surface_values(ds_h, ds)
            ds_h = set_attributes(ds_h)

            # save each datetime
            ds_h.to_netcdf(join(path_save, date, file), encoding=encoding)
            ds.close()
            ds_h.close()


if __name__ == "__main__":
    main()
