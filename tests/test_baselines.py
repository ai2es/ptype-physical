from ptype.baselines import precip_type_partial_thickness
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units


def test_precip_type_partial_thickness():
    pressure_profile = np.arange(1000, 100, -25)
    height = pressure_to_height_std(pressure_profile * units("hPa")).m
    temperatures = np.linspace(0, -50, pressure_profile.shape[0])
    surface_pressure = 1000
    p_type_out = precip_type_partial_thickness(pressure_profile, height, temperatures, surface_pressure)
    assert p_type_out.shape == (1, 4), "Incorrect output shape"
    assert p_type_out.sum() == 1, "Sum not equal to 1"
    assert p_type_out.min() >= 0, "Min p_type probability not 0"

