import pandas as pd
from herbie import Herbie
from metpy.units import units
from metpy.calc import dewpoint_from_relative_humidity, dewpoint_from_specific_humidity
import numpy as np
import xarray as xr
import cfgrib
import os
from bridgescaler import load_scaler
from evml.keras.models import CategoricalDNN
import yaml
import numba
from numba import jit
import glob
import zarr
import pygrib
from pyproj import CRS, Transformer

def df_flatten(ds, varsP, vertical_level_name='isobaricInhPa'):
    """  Split pressure level variables by pressure level, reassign and return as flattened Dataframe.
    Args:
        ds (xr.dataset): Dataset of pressure level variables
        varsP (list): List of pressure level variables to flatten
        vertical_level_name (str): Name of the pressure level dimension
    Returns:
         Pandas Dataframe of flattened variables split out by variable name in format: <varName>_<pressure_level>
    """
    pressure_lev_data, cols = [], []
    for v in varsP:
        for p in ds[vertical_level_name].values:
            cols.append(f"{v}_{int(p)}")
            pressure_lev_data.append(ds[v].sel(isobaricInhPa=p).values.reshape(-1, 1))
    flat_data = np.concatenate(pressure_lev_data, axis=1)

    return pd.DataFrame(flat_data, columns=cols)


def kelvin_to_celsius(temp):
    """ Convert Kelvin Temperatures to Celcius."""
    return temp - 273.15


def convert_longitude(lon):
    """ Convert longitude from -180-180 to 0-360"""
    return lon % 360

def download_data(date, model, product, save_dir, forecast_hour):
    """ Download data use Herbie for specified dates, model and forecast range.
    Args:
        date (List of pandas date times): List of Model initialization times
        model (str): Model to download data from. Supports "hrrr", "rap", or "nam"
        product (str): Product string for NWP grib data Herbie Accessor
        save_dir (str): Directory to save model data to
        forecast_hour (int): Forecast hour
    Returns:
        None
    """

    h = Herbie(date=date,
               model=model,
               product=product,
               save_dir=save_dir,
               fxx=forecast_hour)
    h.download()

    return h.get_localFilePath()



def load_data(var_dict, file, model, extent, drop):
    """
    Load variables from grib file, flatten pressure variables and convert to DataFrame. Supports "gfs", "rap", "hrrr"
    and "nam" models.
    Args:
        var_dict: Dictionary of variables to process. Requires "isobaricInPa", "surface", and "heightAboveGround"
        file: Path to grib file.
        model: Model name. Supports "gfs", "rap", "hrrr" and "nam".
        drop: Whether to drop pressure level variables for final written output (not dropped from flattened df).

    Returns:
        xarray dataset, flatten pandas dataframe, surface data (flattened df)
    """
    grib_data = []
    for key, value in var_dict.items():
        if key == "product":
            continue
        for var in value:
            grib = cfgrib.open_dataset(file, backend_kwargs={
                "filter_by_keys": {'typeOfLevel': key, 'cfVarName': var, 'stepType': 'instant'}})
            if len(grib) == 0:
                grib = cfgrib.open_dataset(file, backend_kwargs={
                    "filter_by_keys": {'typeOfLevel': key, 'shortName': var, 'stepType': 'instant'}})
            grib_data.append(grib)

    for idx in glob.glob(str(file) + '*.idx'):
        os.remove(idx)  # delete index files that are created when opening grib
    nwp_dataset = xr.merge(grib_data, compat='override').load()
    nwp_dataset['t'].values = kelvin_to_celsius(nwp_dataset['t'].values)
    if model == "rap":
        nwp_dataset['dpt'] = dewpoint_from_relative_humidity(nwp_dataset['t'] * units.degC,
                                                             nwp_dataset['r'].values / 100)
    elif model == "gfs":
        z = np.zeros(shape=(nwp_dataset['isobaricInhPa'].size,
                            nwp_dataset['latitude'].size,
                            nwp_dataset['longitude'].size))
        for i in range(z.shape[1]):
            for j in range(z.shape[2]):
                z[:, i, j] = nwp_dataset['isobaricInhPa'].values
        dpt = dewpoint_from_specific_humidity(z * units.hPa,
                                              nwp_dataset['t'].values * units.degC,
                                              nwp_dataset['q'].values * units('kg/kg'))
        nwp_dataset['dpt'] = (["isobaricInhPa", "latitude", "longitude"], dpt)
        nwp_dataset = nwp_dataset.rename_dims({'latitude': 'y', 'longitude': 'x'})
    else:
        nwp_dataset['dpt'].values = kelvin_to_celsius(nwp_dataset['dpt'].values)
    nwp_dataset['hgt_above_sfc'] = nwp_dataset['gh'] - nwp_dataset['orog']
    nwp_dataset = add_coord_data(file, nwp_dataset, extent)
    flattened_df = df_flatten(nwp_dataset, ['t', 'dpt', 'u', 'v', 'hgt_above_sfc'])

    surface_vars = {x: nwp_dataset[x].values.flatten() for x in var_dict["heightAboveGround"] + var_dict["surface"]}
    surface_vars['t2m'] = kelvin_to_celsius(surface_vars['t2m'])
    surface_vars['d2m'] = kelvin_to_celsius(surface_vars['d2m'])

    os.remove(str(file))  # delete grib file

    if drop:
        dropped = var_dict["isobaricInhPa"] + ['hgt_above_sfc'] + ['dpt']
        return nwp_dataset.drop_vars(dropped), flattened_df, surface_vars
    else:
        return nwp_dataset, flattened_df, surface_vars


def subset_extent(nwp_data, extent, transformer=None):
        """
        Subset data by given extent in projection coordinates
        Args:
            nwp_data: Xr.dataset of NWP data
            extent: List of coordinates for subsetting (lon_min, lon_max, lat_min, lat_max)
            transformer: Pyproj Projection transformer object

        Returns:
            Subsetted Xr.Dataset
        """
        lon_min, lon_max, lat_min, lat_max = extent
        if transformer is not None:
            x_coords, y_coords = transformer.transform([lon_min, lon_max], [lat_min, lat_max])
            subset = nwp_data.swap_dims({'y': 'y_projection_coordinate', 'x': 'x_projection_coordinate'}).sel(
                y_projection_coordinate=slice(y_coords[0], y_coords[1]),
                x_projection_coordinate=slice(x_coords[0], x_coords[1])).swap_dims(
                    {'y_projection_coordinate': 'y', 'x_projection_coordinate': 'x'})
        else:
            subset = nwp_data.sel(longitude=slice(convert_longitude(lon_min), convert_longitude(lon_max)),
                                  latitude=slice(lat_max, lat_min))  # GFS orders latitude in descending values :/
            print(subset)
        return subset


def add_coord_data(file_path, grib_data, extent):
    """
    Add cf-style projection information and projection coordinates.
    Args:
        file_path (str): Path to grib2 file
        grib_data (xr.Dataset): Dataset to add coordinate information to
        extent (list): List of coordinates to subset (lon_min, lon_max, lat_min, lat_max)

    Returns:

    """
    with pygrib.open(str(file_path)) as grb:
        msg = grb.message(1)
        print(msg.projparams)
        cf_params = CRS(msg.projparams).to_cf()

    grib_data.attrs['projection'] = str(cf_params)

    if msg.projparams['proj'] == 'lcc':

        transformer = Transformer.from_crs('epsg:4326', CRS(msg.projparams))
        x_proj_coord, y_proj_coord = transformer.transform(grib_data['longitude'], grib_data['latitude'])
        grib_data["y_projection_coordinate"] = ('y', y_proj_coord[:, 0])
        grib_data["x_projection_coordinate"] = ('x', x_proj_coord[0, :])
        grib_data["y_projection_coordinate"].attrs["Description"] = "Lambert Conformal Conic y-projection coordinates"
        grib_data["x_projection_coordinate"].attrs["Description"] = "Lambert Conformal Conic x-projection coordinates"

        grib_data = grib_data.set_coords(["y_projection_coordinate", "x_projection_coordinate"])
        if extent != "full" and extent is not None:

            return subset_extent(grib_data, extent, transformer)
        else:
            return grib_data

    else:
        if extent != "full" and extent is not None:
            return subset_extent(grib_data, extent, None)
        else:
            return grib_data


def convert_and_interpolate(data, surface_data, pressure_levels, height_levels):
    """
    Convert Pressure level data to height above surface and interpolate data across specified height levels.
    Args:
        data: Pandas DataFrame of flattened pressure level data.
        surface_data: Pandas DataFrame of flattened surface data.
        pressure_levels: List of pressure levels from model.
        height_levels: Dictionary of height levels (low, high, interval)

    Returns:
        Pandas Dataframe of interpolated data at height above the surface.
    """
    cols = {}
    height_levels = np.arange(start=height_levels["low"],
                              stop=height_levels["high"] + height_levels["interval"],
                              step=height_levels["interval"])
    for var in ['t', 'dpt', 'u', 'v', 'hgt_above_sfc']:
        cols[var] = [f"{var}_{int(x)}" for x in pressure_levels]

    var_arrays = []
    variables = ['t', 'dpt', 'u', 'v']
    surface_variables = ['t2m', 'd2m', 'u10', 'v10']
    height_data = data[cols['hgt_above_sfc']].values

    for v, sv in zip(variables, surface_variables):
        pressure_level_data = data[cols[v]].values
        height_interp_data = interpolate(height_data, pressure_level_data, height_levels)
        height_interp_data[:, 0] = surface_data[sv]
        var_arrays.append(height_interp_data)

    pl_array = np.tile(pressure_levels, len(height_data)).reshape(len(height_data), len(pressure_levels))
    interpolated_pl = interpolate(height_data, pl_array, height_levels)

    all_data = np.concatenate(var_arrays, axis=1)

    return all_data, interpolated_pl


@jit(nopython=True, parallel=True, cache=True)
def interpolate(x, y, height_levels):

    arr = np.zeros(shape=(x.shape[0], len(height_levels)))

    for i in numba.prange(arr.shape[0]):
        arr[i] = np.interp(x=height_levels,
                           xp=x[i],
                           fp=y[i])
    return arr

def transform_data(input_data, transformer):
    """
    Transform data for input into ML model.
    Args:
        input_data: Pandas Dataframe of input data
        transformer: Bridgescaler object used to fit data.

    Returns:
        Pandas dataframe of transformed input.
    """
    transformed_data = transformer.transform(input_data)

    return transformed_data




def load_model(model_path, model_file, input_scaler_file, output_scaler_file):
    """
    Load ML model and bridgescaler object.
    Args:
        model_path: Path to ML model.

    Returns:
        Loaded Tensorflow model, bridgescaler object
    """
    config = os.path.join(model_path, "model.yml")
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        conf['batch_size'] = 1000

    x_transformer = load_scaler(os.path.join(model_path, "scalers", input_scaler_file))

    with open(os.path.join(model_path, "model.yml")) as f:
        conf = yaml.safe_load(f)

    model = CategoricalDNN().load_model(conf)

    return model, x_transformer

def grid_predictions(data, preds, interp_df=None, interpolated_pl=None, height_levels=None, add_interp_data=False):
    """
    Populate gridded xarray dataset with ML probabilities and categorical predictions as separate variables.
    Args:
        data: Xarray dataset of input data.
        preds: Pandas Dataframe of ML predictions.

    Returns:
        Xarray dataset of ML predictions and surface variables on model grid.
    """
    ptype = preds.argmax(axis=1).reshape(-1, 1)
    preds = np.hstack([preds, ptype])
    reshaped_preds = preds.reshape(data['y'].size, data['x'].size, preds.shape[-1])
    for i, (long_v, v) in enumerate(zip(
            ['rain', 'snow', 'ice pellets', 'freezing rain'], ['rain', 'snow', 'icep', 'frzr'])):
        data[f"ML_{v}"] = (['y', 'x'], reshaped_preds[:, :, i].astype('float32'))  # ML probability
        data[f"ML_{v}"].attrs = {"Description": f"Machine Learned Probability of {long_v}"}
        data[f"ML_c{v}"] = (['y', 'x'], np.where(reshaped_preds[:, :, -1] == i, 1, 0).astype('uint8'))  # ML categorical
        data[f"ML_c{v}"].attrs = {"Description": f"Machine Learned Categorical {long_v}"}

    for var in ["crain", "csnow", "cicep", "cfrzr"]:
        if var in list(data.data_vars):
            data[var] = data[var].astype('uint8')

    for v in data.coords:
        if data[v].dtype == 'float64':
            data[v] = data[v].astype('float32')

    drop_vars = []
    for v in ["heightAboveGround", "surface"]:
        if v in data.coords:
            drop_vars.append(v)
    if add_interp_data:

        interpolated_gridded = add_interp_gridded(data, interp_df, height_levels)
        interpolated_gridded['isobaricInhPa_h'] = (['heightAboveGround', 'y', 'x'],
                                                   np.moveaxis(interpolated_pl.reshape(data['y'].size,
                                                                                       data['x'].size,
                                                                                       interpolated_pl.shape[-1]),
                                                               [0, 1, 2],
                                                               [1, 2, 0]))
        interpolated_gridded['isobaricInhPa_h'].attrs = {"Description": "Pressure level at height interpolated levels",
                                                         "Units": "hPa"}
        return xr.merge([interpolated_gridded, data.drop(drop_vars)])
    else:
        return data.drop(drop_vars)


def save_data(dataset, out_path, date, model, forecast_hour, save_format):
    """
    Save ML predictions and surface data as netCDf file.
    Args:
        dataset: Xarray dataset with ML predictions and surface data.
        out_path: Path to save data.
        date: Datetime object for predictions.
        model: NWP model name.
        forecast_hour: Forecast hour of ML predictions.

    Returns:
        None
    """
    date_str = date.strftime("%Y-%m-%d")
    dir_str = date.strftime("%Y%m%d")
    model_run_str = date.strftime("%H%M")
    os.makedirs(os.path.join(out_path, model, dir_str, model_run_str), exist_ok=True)
    file_str = f"MILES_ptype_{model}_{date_str}_{model_run_str}_f{forecast_hour:02}"
    full_path = os.path.join(out_path, model, dir_str, model_run_str, file_str)

    dataset = dataset.expand_dims('time')

    encoding_vars = [v for v in list(dataset.data_vars)]
    if save_format == "netcdf":
        encoding = {var: {"zlib": True, "complevel": 4, "least_significant_digit": 4} for var in encoding_vars}
        dataset.to_netcdf(full_path + ".nc", encoding=encoding)
    elif save_format == "zarr":
        compressor = zarr.Blosc(cname="zlib", clevel=4, shuffle=1)
        encoding = {var: {'compressor': compressor, 'chunks': {100, 100}} for var in encoding_vars}
        dataset.to_zarr(full_path + ".zarr", mode='w', encoding=encoding, consolidated=True)
    print(f"Successfully wrote: {full_path}")

    return


def add_interp_gridded(nwp_data, interp_data, height_levels):
    """
    Convert height interpolated data to xarray format and merge with main dataset.
    Args:
        nwp_data (xr.Dataset): Main Numerical Weather Prediciton dataset with ML preds
        interp_data (np.array): Numpy array of height interpolated values for ML input
        height_levels: Height levels used for ML input

    Returns:
        Merged xr.Dataset of NWP/ML data and height interpolated data
    """
    x = nwp_data.stack(d=['y', 'x'])['x'].values.astype('int16')
    y = nwp_data.stack(d=['y', 'x'])['y'].values.astype('int16')

    variables = ['t', 'dpt', 'u', 'v']
    height_levels = np.arange(height_levels['high'], height_levels['low'] - height_levels['interval'],
                              -height_levels['interval'])
    columns = [f"{var}_{level}" for var in variables for level in height_levels]

    new_df = pd.DataFrame(interp_data, columns=columns)
    new_df['x'] = x
    new_df['y'] = y
    new_df = new_df.set_index(['y', 'x'])
    gridded = new_df.to_xarray()

    datasets = []
    for var, long_name in zip(['t', 'dpt', 'u', 'v'],
                              ['temperature', 'dewpoint', 'u-component of wind', 'v-component of wind']):
        dataset = xr.concat([gridded[f"{var}_{i}"] for i in height_levels],
                            dim='heightAboveGround').to_dataset().rename({f"{var}_{int(height_levels.max())}": f"{var}_h"})
        dataset[f"{var}_h"].attrs['Description'] = f"Height interpolated {long_name}"
        dataset[f"{var}_h"] = dataset[f"{var}_h"].astype('float32')
        datasets.append(dataset)

    return xr.merge(datasets).drop(['x', 'y']).assign_coords({'heightAboveGround': np.flip(height_levels)})

