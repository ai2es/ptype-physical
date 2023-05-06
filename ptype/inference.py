import pandas as pd
from herbie import FastHerbie
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
import json


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


def download_data(date, model, product, save_dir, forecast_range):
    """ Download data use Herbie for specified dates, model and forecast range.
    Args:
        date (List of pandas date times): List of Model initialization times
        model (str): Model to download data from. Supports "hrrr", "rap", or "nam"
        product (str): Product string for NWP grib data Herbie Accessor
        save_dir (str): Directory to save model data to
        forecast_range (tuple): Tuple of (start forecast hour, end_forecast_hour)
    Returns:
        None
    """
    forecast_hours = range(forecast_range["start"],
                           forecast_range["end"] + forecast_range["interval"],
                           forecast_range["interval"])
    h = FastHerbie(DATES=[date],
                   model=model,
                   product=product,
                   save_dir=save_dir,
                   fxx=forecast_hours)
    h.download()
    return


def load_data(var_dict, file, model, drop):
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

    for idx in glob.glob(file + '*.idx'):
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
    flattened_df = df_flatten(nwp_dataset, ['t', 'dpt', 'u', 'v', 'hgt_above_sfc'])

    surface_vars = {x: nwp_dataset[x].values.flatten() for x in var_dict["heightAboveGround"] + var_dict["surface"]}
    surface_vars['t2m'] = kelvin_to_celsius(surface_vars['t2m'])
    surface_vars['d2m'] = kelvin_to_celsius(surface_vars['d2m'])
    os.remove(file)     # delete grib file

    if drop:
        dropped = var_dict["isobaricInhPa"] + ['hgt_above_sfc'] + ['dpt']
        return nwp_dataset.drop_vars(dropped), flattened_df, surface_vars
    else:
        return nwp_dataset, flattened_df, surface_vars


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

    all_data = np.concatenate(var_arrays, axis=1)

    return all_data


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
    transformed_data = transformer.transform(pd.DataFrame(input_data, columns=transformer.x_columns_))

    return transformed_data.values


def load_model(model_path):
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

    x_transformer = load_scaler(os.path.join(model_path, "scalers", "input_11.json"))
    with open(os.path.join(model_path, "scalers", "output_label_11.json")) as f:
        output_scaler = json.load(f)
    model = CategoricalDNN(**conf["model"])
    model.build_neural_network(len(x_transformer.x_columns_), len(output_scaler['classes_']))
    model.model.load_weights(os.path.join(model_path, "models", "model_11.h5"))

    return model, x_transformer


def grid_preditions(data, preds):
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
    for i, v in enumerate(['Rain', 'Snow', 'Sleet', 'FreezingRain']):

        data[v] = (['y', 'x'], reshaped_preds[:, :, i])                               # probability
        data[f"C{v}"] = (['y', 'x'], np.where(reshaped_preds[:, :, -1] == i, 1, 0))   # categorical

    return data


def save_data(dataset, out_path, date, model, forecast_hour):
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
    dir_str = date.strftime("%Y%m%d")
    date_str = date.strftime(f"%Y%m%d_{forecast_hour:02}00")
    file_str = f"ptype_predictions_{model}_{date_str}.nc"
    full_path = os.path.join(out_path, model, dir_str, file_str)
    encoding_vars = [v for v in list(dataset.data_vars)]
    encoding = {var: {"zlib": True, "complevel": 4, "least_significant_digit": 4.0} for var in encoding_vars}
    dataset.to_netcdf(full_path, encoding=encoding)
    print(f"Successfully wrote: {full_path}")

    return


def get_file_paths(base_path, model, date):
    """
    Retrieve file paths of grib data for specified date.
    Args:
        base_path: Base path to NWP data storage.
        model: NWP model name.
        date: Datetime object for grib data.

    Returns:
        List of file names of grib data to process.
    """
    files = sorted([os.path.join(base_path, model, date.strftime("%Y%m%d"), f) for f in
                    os.listdir(os.path.join(base_path, model, date.strftime("%Y%m%d")))])

    return files


