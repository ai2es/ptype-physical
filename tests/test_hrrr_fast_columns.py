"""Tests for ``ptype.hrrr_fast.columns`` against numpy, MetPy and pyproj."""

import warnings

import numpy as np
import pytest
from metpy.calc import dewpoint_from_specific_humidity as mp_dewpoint_from_q
from metpy.calc import wet_bulb_temperature as mp_wet_bulb
from metpy.units import units
from ptype.hrrr_fast import columns as C


def synthetic_native(ny=7, nx=9, nlev=30, seed=0):
    """Hybrid-level-like columns: heights vary per column, first level 5-20 m AGL, top > 5 km."""
    rng = np.random.default_rng(seed)
    dz = rng.uniform(20, 400, (nlev, ny, nx))
    dz[0] = rng.uniform(5, 20, (ny, nx))
    z = np.cumsum(dz, axis=0).astype(np.float32)
    psfc = rng.uniform(70000, 104000, (ny, nx))
    p = (psfc * np.exp(-z / 8000.0)).astype(np.float32)
    t_k = (rng.uniform(255, 295, (ny, nx)) - 0.0065 * z + rng.normal(0, 1, z.shape)).astype(np.float32)
    q = (rng.uniform(1e-4, 1.5e-2, (ny, nx)) * np.exp(-z / 2500.0)).astype(np.float32)
    u = rng.normal(5, 10, z.shape).astype(np.float32)
    v = rng.normal(0, 10, z.shape).astype(np.float32)
    sfc = {
        "t2m_k": (t_k[0] + rng.normal(0, 1, (ny, nx))).astype(np.float32),
        "td2m_k": (t_k[0] - rng.uniform(0, 5, (ny, nx))).astype(np.float32),
        "u10": rng.normal(3, 5, (ny, nx)).astype(np.float32),
        "v10": rng.normal(0, 5, (ny, nx)).astype(np.float32),
        "psfc_pa": psfc.astype(np.float32),
    }
    return z, t_k, q, p, u, v, sfc


# ------------------------------------------------------------------------------------------------
# Dewpoint
# ------------------------------------------------------------------------------------------------
def test_dewpoint_matches_metpy():
    rng = np.random.default_rng(1)
    q = 10 ** rng.uniform(-6, np.log10(0.03), 200000)
    p = rng.uniform(30000, 105000, q.size)
    ours = C.dewpoint_from_specific_humidity(q.astype(np.float32), p.astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = mp_dewpoint_from_q(p.astype(np.float32).astype(np.float64) * units.Pa, q.astype(np.float32).astype(np.float64) * units("kg/kg")).m_as("degC")
    diff = np.abs(ours.astype(np.float64) - ref)
    # float32 output: rounding of values up to ~40 degC is <= ~2e-6.
    assert diff.max() < 1e-5, diff.max()


def test_dewpoint_handles_zero_q():
    td = C.dewpoint_from_specific_humidity(np.array([0.0, 1e-3], np.float32), np.array([50000.0, 50000.0], np.float32))
    assert np.isfinite(td).all() and td[0] < -90


# ------------------------------------------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------------------------------------------
def test_interp_equals_np_interp():
    z, t_k, *_ = synthetic_native(ny=40, nx=50)
    # Targets reach below the lowest level and above the top to test clamping, and hit native
    # heights exactly.
    targets = np.concatenate([[0.0, 1.0], C.NETWORK_HEIGHTS[1:], [z.max() + 100.0]])
    targets[5] = float(z[3, 0, 0])
    targets = np.unique(targets)
    out = C.interp_to_heights(z, t_k, targets)
    ref = np.empty_like(out)
    for j in range(z.shape[1]):
        for i in range(z.shape[2]):
            ref[:, j, i] = np.interp(targets, z[:, j, i].astype(np.float64), t_k[:, j, i].astype(np.float64))
    np.testing.assert_array_equal(out, ref)


def test_interp_log_pressure_equals_numpy():
    z, _, _, p, *_ = synthetic_native(ny=20, nx=30)
    out = C.interp_log_pressure_to_heights(z, p, C.NETWORK_HEIGHTS)
    ref = np.empty_like(out)
    for j in range(z.shape[1]):
        for i in range(z.shape[2]):
            ref[:, j, i] = np.exp(np.interp(C.NETWORK_HEIGHTS, z[:, j, i].astype(np.float64), np.log(p[:, j, i].astype(np.float64))))
    np.testing.assert_allclose(out, ref, rtol=2e-7)


def test_interp_rejects_non_monotonic():
    z, t_k, *_ = synthetic_native(ny=3, nx=3)
    z = z.copy()
    z[4, 1, 1] = z[3, 1, 1]
    with pytest.raises(ValueError, match="1 columns"):
        C.interp_to_heights(z, t_k, C.NETWORK_HEIGHTS)


# ------------------------------------------------------------------------------------------------
# Sounding assembly
# ------------------------------------------------------------------------------------------------
def old_style(z, t_k, q, p, u, v, sfc):
    """The old pipeline's arithmetic (np.interp per column, then 2 m / 10 m overrides at 0 m)."""
    ny, nx = z.shape[1:]
    nt = C.NETWORK_HEIGHTS.size
    ref = np.empty((ny * nx, 5, nt), np.float32)
    td = C.dewpoint_from_specific_humidity(q, p)
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            zz = z[:, j, i].astype(np.float64)
            for g, f in enumerate([t_k[:, j, i].astype(np.float64) - 273.15, td[:, j, i].astype(np.float64), u[:, j, i], v[:, j, i]]):
                ref[c, g] = np.interp(C.NETWORK_HEIGHTS, zz, np.asarray(f, np.float64))
            ref[c, 4] = np.interp(C.NETWORK_HEIGHTS, zz, p[:, j, i].astype(np.float64)) / 100.0
    return ref


def test_build_soundings_policies():
    z, t_k, q, p, u, v, sfc = synthetic_native()
    ref = old_style(z, t_k, q, p, u, v, sfc)
    out, n_short = C.build_soundings(z, t_k, q, p, u, v, surface_temperature="interp", surface_wind="interp", surface_pressure="interp")
    assert n_short == (z[-1] < C.NETWORK_HEIGHTS[-1]).sum()
    assert out.flags.c_contiguous and out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=2e-5)
    # Defaults: 2 m T/Td, 10 m wind, surface pressure at 0 m; levels above 0 m unchanged.
    out2, _ = C.build_soundings(z, t_k, q, p, u, v, **sfc)
    np.testing.assert_array_equal(out2[:, :, 1:], out[:, :, 1:])
    np.testing.assert_allclose(out2[:, 0, 0], sfc["t2m_k"].ravel().astype(np.float64) - 273.15, atol=1e-5)
    np.testing.assert_allclose(out2[:, 1, 0], sfc["td2m_k"].ravel().astype(np.float64) - 273.15, atol=1e-5)
    np.testing.assert_array_equal(out2[:, 2, 0], sfc["u10"].ravel())
    np.testing.assert_array_equal(out2[:, 3, 0], sfc["v10"].ravel())
    np.testing.assert_allclose(out2[:, 4, 0], sfc["psfc_pa"].ravel() / 100.0, rtol=1e-7)
    # Dewpoint clip at 0 m.
    sfc_bad = dict(sfc, td2m_k=sfc["t2m_k"] + 2.0)
    out3, _ = C.build_soundings(z, t_k, q, p, u, v, clip_surface_dewpoint=True, **sfc_bad)
    np.testing.assert_array_equal(out3[:, 1, 0], out3[:, 0, 0])
    # Log-pressure option: differs from linear by at most ~p (dz/H)^2 / 8 (synthetic dz <= 400 m).
    out4, _ = C.build_soundings(z, t_k, q, p, u, v, pressure_interp="log", **sfc)
    assert np.abs(out4[:, 4, 1:] - out2[:, 4, 1:]).max() < 0.4
    assert (out4[:, 4, 1:] <= out2[:, 4, 1:] + 1e-3).all()  # exp is convex, so log-p interp <= linear


def test_build_soundings_matches_interp_to_heights():
    z, t_k, q, p, u, v, _ = synthetic_native(ny=11, nx=13)
    out, _ = C.build_soundings(z, t_k, q, p, u, v, surface_wind="interp", surface_temperature="interp", surface_pressure="interp")
    ui = C.interp_to_heights(z, u, C.NETWORK_HEIGHTS)
    np.testing.assert_array_equal(out[:, 2, :], ui.reshape(21, -1).T)


def test_network_and_sounding_matrices():
    z, t_k, q, p, u, v, sfc = synthetic_native()
    out, _ = C.build_soundings(z, t_k, q, p, u, v, **sfc)
    x = C.network_inputs(out)
    assert x.shape == (out.shape[0], 84) and x.flags.c_contiguous and x.dtype == np.float32
    s = C.sounding_matrix(out)
    assert s.shape == (out.shape[0], 105)
    np.testing.assert_array_equal(x, s[:, :84])
    # Group order: T, Td, U, V, P; each surface -> top.
    np.testing.assert_array_equal(x[:, 0:21], out[:, 0, :])
    np.testing.assert_array_equal(x[:, 63:84], out[:, 3, :])
    np.testing.assert_array_equal(s[:, 84:105], out[:, 4, :])


def test_build_soundings_rotation_equals_rotating_after():
    z, t_k, q, p, u, v, sfc = synthetic_native()
    lon = np.linspace(-125, -70, z.shape[2])[np.newaxis, :].repeat(z.shape[1], 0)
    out, _ = C.build_soundings(z, t_k, q, p, u, v, rotation=C.grid_rotation_cos_sin(lon), **sfc)
    plain, _ = C.build_soundings(z, t_k, q, p, u, v, **sfc)
    ny, nx = lon.shape
    ue, ve = C.rotate_grid_to_earth(plain[:, 2, :].T.reshape(21, ny, nx), plain[:, 3, :].T.reshape(21, ny, nx), lon)
    np.testing.assert_allclose(out[:, 2, :], ue.reshape(21, -1).T, atol=1e-5)
    np.testing.assert_allclose(out[:, 3, :], ve.reshape(21, -1).T, atol=1e-5)


# ------------------------------------------------------------------------------------------------
# Rotation
# ------------------------------------------------------------------------------------------------
def test_rotation_round_trip_and_central_meridian():
    rng = np.random.default_rng(3)
    lon = rng.uniform(-135, -60, (20, 30))
    u = rng.normal(0, 15, (4, 20, 30)).astype(np.float32)
    v = rng.normal(0, 15, (4, 20, 30)).astype(np.float32)
    ue, ve = C.rotate_grid_to_earth(u, v, lon)
    ug, vg = C.rotate_earth_to_grid(ue, ve, lon)
    np.testing.assert_allclose(ug, u, atol=1e-5)
    np.testing.assert_allclose(vg, v, atol=1e-5)
    np.testing.assert_allclose(np.hypot(ue, ve), np.hypot(u, v), rtol=1e-6)  # speed preserved
    lon0 = np.full((20, 30), C.HRRR_LON_0)
    ue0, ve0 = C.rotate_grid_to_earth(u, v, lon0)
    np.testing.assert_array_equal(ue0, u)
    np.testing.assert_array_equal(ve0, v)
    ue360, _ = C.rotate_grid_to_earth(u, v, lon + 360.0)  # 0..360 longitudes give the same answer
    np.testing.assert_allclose(ue360, ue, atol=1e-5)


def test_rotation_sign_east_of_central_meridian():
    """A grid-"north" wind east of the central meridian has positive earth-relative u.

    Geometry: the Lambert cone's apex lies above the pole on the central meridian, so east of it
    the meridians lean towards the north-west on the grid; grid +y therefore points east of true
    north and a wind blowing toward grid +y has an eastward component.
    """
    lon = np.array([[-80.0, -115.0]])
    ue, ve = C.rotate_grid_to_earth(np.zeros((1, 2), np.float32), np.full((1, 2), 10.0, np.float32), lon)
    assert ue[0, 0] > 0 and ve[0, 0] > 0
    assert ue[0, 1] < 0  # west of the central meridian the sign flips


def test_rotation_matches_pyproj():
    pyproj = pytest.importorskip("pyproj")
    proj = pyproj.Proj(proj="lcc", lat_0=38.5, lon_0=-97.5, lat_1=38.5, lat_2=38.5, R=6371229)
    rng = np.random.default_rng(5)
    lons = rng.uniform(-134, -61, 200)
    lats = rng.uniform(21, 53, 200)
    d = 1e-4
    x0, y0 = proj(lons, lats)
    xe, ye = proj(lons + d, lats)
    xn, yn = proj(lons, lats + d)
    east = np.stack([xe - x0, ye - y0]) / np.hypot(xe - x0, ye - y0)  # unit east vector in grid coords
    north = np.stack([xn - x0, yn - y0]) / np.hypot(xn - x0, yn - y0)
    ug = rng.normal(0, 10, 200)
    vg = rng.normal(0, 10, 200)
    # Earth components are the projections of the grid vector on the local east / north unit vectors.
    ue_ref = ug * east[0] + vg * east[1]
    ve_ref = ug * north[0] + vg * north[1]
    ue, ve = C.rotate_grid_to_earth(ug[np.newaxis].astype(np.float32), vg[np.newaxis].astype(np.float32), lons[np.newaxis])
    np.testing.assert_allclose(ue[0], ue_ref, atol=2e-3)
    np.testing.assert_allclose(ve[0], ve_ref, atol=2e-3)


# ------------------------------------------------------------------------------------------------
# Wet bulb
# ------------------------------------------------------------------------------------------------
def test_wet_bulb_matches_metpy():
    rng = np.random.default_rng(11)
    n = 3000
    t = rng.uniform(-40, 30, n)
    td = t - rng.exponential(6, n)
    td[: n // 10] = t[: n // 10]  # saturated
    p = rng.uniform(450, 1050, n)
    ours = C.wet_bulb_temperature(t, td, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = mp_wet_bulb(p * units.hPa, t * units.degC, td * units.degC).m_as("degC")
    diff = np.abs(ours - ref)
    assert diff.max() < 1e-3, diff.max()
    assert (ours <= t + 1e-9).all() and (ours >= td - 1e-9).all()
    np.testing.assert_allclose(ours[: n // 10], t[: n // 10], atol=1e-6)


def test_wet_bulb_float32_and_nan():
    t = np.array([[5.0, np.nan]], np.float32)
    out = C.wet_bulb_temperature(t, t - 3, np.array([[900.0, 900.0]], np.float32))
    assert out.dtype == np.float32 and out.shape == (1, 2)
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1])
