import numpy as np
# from numba import jit
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

# @jit
def precip_type_partial_thickness(pressure_profile_hPa, geopotential_height_profile_m,
                                  temperature_profile_C, surface_pressure_hPa):
    """
    Estimate precipitation type based on Bhupal's implemenation of the partial thickness method.
    The method outputs an array of the probability of four precipitation types in the following order:
    [rain, snow, ice pellets/sleet, freezing rain].

    Args:
        pressure_profile_hPa: numpy array containing vertical profile of pressure in units of hPa
        geopotential_height_profile_m: geopotential height (height relative to sea level) profile in units of m
        temperature_profile_C: temperature profile in units of Celsius
        surface_pressure_hPa: pressure at the land surface elevation in units of hPa
    Returns:
        precip_type: array of dimensions (1, 4) with values ranging from 0 to 1.
    """
    thickness_850_700_m = thickness_profile(850.0, 700.0, pressure_profile_hPa, geopotential_height_profile_m)
    thickness_1000_850_m = thickness_profile(1000.0, 850.0, pressure_profile_hPa, geopotential_height_profile_m)
    surface_index = np.argmin(np.abs(pressure_profile_hPa - surface_pressure_hPa))
    # filter out temperatures occurring below ground.
    above_surface = pressure_profile_hPa <= surface_pressure_hPa
    temperature_max = temperature_profile_C[above_surface].max()
    surface_50_index = np.argmin(np.abs(pressure_profile_hPa - surface_pressure_hPa + 50))
    temperature_surface = temperature_profile_C[surface_index]
    if surface_50_index < surface_index:
        near_slice = slice(surface_50_index, surface_index + 1)
    else:
        near_slice = slice(surface_index, surface_50_index + 1)
    temp_surf_50_max = np.max(temperature_profile_C[near_slice])
    temp_surf_50_min = np.min(temperature_profile_C[near_slice])
    precip_type = np.zeros(4, dtype=np.float32)
    if thickness_850_700_m < 1540.0:
        if thickness_1000_850_m < 1300:
            precip_type[1] = 1
        elif 1300 <= thickness_1000_850_m <= 1320:
            precip_type[1] = 0.75
            precip_type[2] = 0.25
        else:
            precip_type[0] = 1
    elif 1540 <= thickness_850_700_m < 1570:
        if thickness_1000_850_m < 1300:
            if thickness_850_700_m <= 1545:
                precip_type[1] = 1
            else:
                precip_type[1] = 0.75
                precip_type[2] = 0.25
        elif 1300 <= thickness_1000_850_m <= 1320:
            precip_type[0] = 0.25
            precip_type[3] = 0.75
        else:
            precip_type[0] = 1
    elif 1570 <= thickness_850_700_m < 1595:
        precip_type[0] = 0.25
        precip_type[3] = 0.75
    elif 1595 <= thickness_850_700_m < 1605:
        precip_type[2] = 0.25
        precip_type[3] = 0.75
    else:
        if thickness_1000_850_m < 1310:
            precip_type[2] = 1
        else:
            precip_type[3] = 1
    if precip_type[1] > 0.5 and temperature_surface > 0:
        precip_type[:] = 0
        precip_type[1] = 0.25
        precip_type[0] = 0.75
    if thickness_1000_850_m > 1335 and temperature_surface > -1:
        precip_type[:] = 0
        precip_type[0] = 1
    if temperature_max <= -3:
        precip_type[:] = 0
        precip_type[1] = 1
    if temperature_surface > 7:
        precip_type[:] = 0
        precip_type[0] = 1
    if temp_surf_50_min > 0 and temp_surf_50_max > 2:
        precip_type[:] = 0
        precip_type[0] = 1
    if np.abs(surface_pressure_hPa - 1000.0) < 10 and precip_type[3] >= 0.5 and temperature_surface < -3.0 and thickness_1000_850_m < 1320.0:
        precip_type[:] = 0
        precip_type[2] = 1
    if surface_pressure_hPa > 1000:
        if precip_type[1] == 0.5 and precip_type[2] == 0.5 and temperature_surface < -1:
            precip_type[2] = 0.25
            precip_type[3] = 0.75
        if precip_type[0] == 1 and temperature_surface < -1:
            precip_type[0] = 0.25
            precip_type[3] = 0.75
        if thickness_850_700_m > 1600 and thickness_1000_850_m > 1325:
            precip_type[:] = 0
            precip_type[0] = 1
    return precip_type.reshape(1, -1)
#
# @jit
def thickness_profile(bottom_pressure, top_pressure, pressure_profile, geopotential_height_profile):
    """
    Calculate thickness of layer between two pressure levels.

    Args:
        bottom_pressure: pressure at bottom of thickness layer (closer to sea level)
        top_pressure: pressure at top of thickness layer (farther from sea level)

    """
    assert top_pressure < bottom_pressure, "Top pressure is greater than bottom pressure"
    bottom_index = np.argmin(np.abs(pressure_profile - bottom_pressure))
    top_index = np.argmin(np.abs(pressure_profile - top_pressure))
    return geopotential_height_profile[top_index] - geopotential_height_profile[bottom_index]


def partial_thickness_full_grid_single_time(rap_ds):
    pressure_hPa = rap_ds["press"].values
    geopotential_height_m = rap_ds["HGT"][0].values
    temperature_C = rap_ds["TMP"][0].values - 273.15
    surface_pressure_hPa = rap_ds["PRES_ON_SURFACE"][0].values / 100.0
    precip_type_probs = _partial_thickness_grid_loop(pressure_hPa, geopotential_height_m, temperature_C, surface_pressure_hPa)
    return precip_type_probs

@jit
def _partial_thickness_grid_loop(pressure_hPa, geopotential_height_m,
                                  temperature_C, surface_pressure_hPa):
    precip_type_probs = np.zeros((surface_pressure_hPa.shape[0], surface_pressure_hPa.shape[1], 4))
    for i in range(precip_type_probs.shape[0]):
        for j in range(precip_type_probs.shape[1]):
            precip_type_probs[i, j] = precip_type_partial_thickness(pressure_hPa,
                                                                    geopotential_height_m[:, i, j],
                                                                    temperature_C[:, i, j],
                                                                    surface_pressure_hPa[i, j])
    return precip_type_probs

class PartialThicknessClassifier(BaseEstimator):
    """
    Scikit-learn formatted PartialThickness classifier

    """
    def __init__(self,
                 height_col_name="HGT_{0:d}_m",
                 temperature_col_name="TMP_{0:d}_C",
                 surface_pressure_col_name="PRES_ON_SURFACE_Pa",
                 pressure_levels=np.arange(100, 1025, 25),
                 n_jobs=1,
                 verbose=1):
        self.height_col_name = height_col_name
        self.temperature_col_name = temperature_col_name
        self.surface_pressure_col_name = surface_pressure_col_name
        self.pressure_levels = pressure_levels
        self.height_cols = [height_col_name.format(p) for p in pressure_levels]
        self.temperature_cols = [temperature_col_name.format(p) for p in pressure_levels]
        self.p_type_labels = ["rain", "snow", "ice pellets", "freezing rain"]
        self.n_jobs = n_jobs
        self.verbose = verbose
        return

    def fit(self, x, y):
        return

    def predict(self, x):
        p_type_probs = self.predict_proba(x)
        p_type_index = p_type_probs.argmax(axis=1)
        return p_type_index

    def predict_proba(self, x):
        height_data = x[self.height_cols].values
        temperature_data = x[self.temperature_cols].values
        surface_pressure_data = x[self.surface_pressure_col_name].values / 100.0
        p_type_probs = np.zeros((x.shape[0], len(self.p_type_labels)))
        if self.n_jobs == 1:
            for i in range(p_type_probs.shape[0]):
                p_type_probs[i] = precip_type_partial_thickness(self.pressure_levels,
                                                                height_data[i],
                                                                temperature_data[i],
                                                                surface_pressure_data[i])
        else:
            p_type_probs[:] = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                [delayed(precip_type_partial_thickness)(self.pressure_levels,
                                                        height_data[i],
                                                        temperature_data[i],
                                                        surface_pressure_data)
                 for i in range(p_type_probs.shape[0])])
        return p_type_probs
