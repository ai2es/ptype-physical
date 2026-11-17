"""Per-column HRRR sounding preparation for the precipitation-type network, in numba.

The network expects, for every grid column, temperature (degC), dewpoint (degC) and the
u and v wind components (m/s) on 21 heights above ground level (AGL): 0, 250, ..., 5000 m.
The original implementation (``ptype/inference.py``: ``load_data``,
``convert_and_interpolate`` and ``interpolate``) flattens every pressure level into a pandas
DataFrame, interpolates each column with ``np.interp`` and then overwrites the 0 m level with
the 2 m (and, for winds, 10 m) values.  That is correct but slow and memory hungry.

This module does the same physics directly on the gridded ``(level, y, x)`` arrays that come out
of the GRIB decoder, without pandas and without reshaping copies:

* :func:`dewpoint_from_specific_humidity` - Bolton (1980) dewpoint from q and p, identical to
  ``metpy.calc.dewpoint_from_specific_humidity``.
* :func:`interp_to_heights` / :func:`interp_log_pressure_to_heights` - linear interpolation in
  height that reproduces ``np.interp`` bit for bit (same arithmetic, same end clamping).
* :func:`build_soundings` - one fused pass that, per column, finds the bracketing native levels
  once, computes dewpoint only on the levels it needs, interpolates T, Td, u, v and p to the
  target heights, applies the surface (0 m) policy and optionally rotates the winds to earth
  relative.  It writes a ``(n_columns, 5, n_heights)`` float32 array, i.e. the
  ``(n_columns, 105)`` sounding matrix, from which :func:`network_inputs` copies the
  ``(n_columns, 84)`` network matrix.
* :func:`rotate_grid_to_earth` / :func:`rotate_earth_to_grid` - Lambert conformal wind rotation.
* :func:`wet_bulb_temperature` - Normand's rule (LCL, then moist adiabat back down), the same
  method as ``metpy.calc.wet_bulb_temperature`` which the training QC used (``ptype/qc.py:107``).

Why numba: each column is an independent, short (tens of levels), branchy computation.  That is
a poor fit for whole-array numpy (it needs ``searchsorted`` per column, many temporaries the size
of the full 3-D grid, and Python loops for the iterative parts) but an ideal fit for a compiled
loop that keeps one column in registers/L1 cache.  ``parallel=True`` with ``prange`` over grid
rows gives near-linear thread scaling because rows share nothing; looping over ``x`` innermost
keeps the reads of each ``(level, y, x)`` field contiguous in memory.  ``cache=True`` stores the
compiled machine code on disk so that batch jobs do not pay the compile time on every run.

All arithmetic is done in float64 (the inputs are usually float32) and only the results are stored
as float32; that is what ``np.interp`` does and it is why the interpolation results match the old
code exactly.

Training-data provenance of the 0 m level (from the repository, see the module report):
``scripts/precip_RAP_interpolate.py:79-83`` (2 m T and Td, 10 m u and v, surface pressure) and
``scripts/rap_height_interp.py:88-94`` (the same, plus the 2 m dewpoint clipped to be no larger
than the 2 m temperature).  The defaults of :func:`build_soundings` follow that evidence.
"""

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------------------------
# Physical constants.  These are MetPy 1.7.1's values (``metpy.constants.nounit``), copied so the
# numba kernels can use them as compile-time constants and so that the results agree with MetPy,
# which is what the training pipeline used for dewpoint (from RH) and wet-bulb temperature.
# ---------------------------------------------------------------------------------------------
EPSILON = 0.6219569100577033  # Rd / Rv, molecular weight ratio of water vapour to dry air.
RD = 287.04749097718457  # Gas constant of dry air, J kg-1 K-1.
RV = 461.52311572606084  # Gas constant of water vapour, J kg-1 K-1.
CP_D = 1004.6662184201462  # Specific heat of dry air at constant pressure, J kg-1 K-1.
CP_V = 1860.078011865639  # Specific heat of water vapour at constant pressure, J kg-1 K-1.
CP_L = 4219.400000000001  # Specific heat of liquid water, J kg-1 K-1.
CP_I = 2090.0  # Specific heat of ice, J kg-1 K-1.
LV = 2500840.0  # Latent heat of vaporisation at T0, J kg-1.
LS = 2834540.0  # Latent heat of sublimation at T0, J kg-1.
T0 = 273.16  # Triple point of water, K (reference temperature of MetPy's vapour-pressure formula).
ZERO_DEGC = 273.15  # 0 degC in K (used only for unit conversion).
SAT_PRESSURE_0C = 611.2  # Saturation vapour pressure at T0, Pa.

# Bolton (1980) magnus coefficients used by ``metpy.calc.dewpoint``.
BOLTON_A = 17.67
BOLTON_B = 243.5  # degC

# Specific humidity below this floor (kg/kg) is raised to it before taking a logarithm.  HRRR
# GRIB packing can round very small upper-level q to exactly 0, which would give log(0) = -inf and
# a NaN dewpoint.  1e-8 kg/kg corresponds to a dewpoint near -100 degC at 500 hPa, far below any
# value that matters for precipitation type.
MIN_SPECIFIC_HUMIDITY = 1.0e-8

# HRRR Lambert conformal projection (GRIB2 template 3.30): LoV = 262.5 E (-97.5), Latin1 =
# Latin2 = 38.5 N, i.e. a tangent cone, so the cone constant is n = sin(38.5 deg).
HRRR_LON_0 = -97.5
HRRR_LAT_STD = 38.5

# Target heights of the network: 0, 250, ..., 5000 m AGL (21 levels), as in
# ``config/inference.yml`` (``height_levels``) and ``config/ptype.yml`` (``TEMP_C_0_m`` ...
# ``TEMP_C_5000_m``).
NETWORK_HEIGHTS = np.arange(0.0, 5001.0, 250.0)

# Order of the variable groups in the sounding array.  The first four groups, in this order, are
# the network input (``config/ptype.yml``: input_features TEMP_C, T_DEWPOINT_C, UGRD_m/s, VGRD_m/s).
SOUNDING_VARIABLES = ("temperature_c", "dewpoint_c", "u_ms", "v_ms", "pressure_hpa")

# Surface (0 m) policies.  Integers so that numba can branch on them cheaply.
_TEMP_POLICIES = {"2m": 0, "interp": 1}
_WIND_POLICIES = {"10m": 0, "interp": 1}
_PRES_POLICIES = {"surface": 0, "interp": 1}
_PRES_INTERP = {"linear": 0, "log": 1}

_LEFT = -1  # Target below the lowest native level: clamp to the lowest level (np.interp ``left``).
_RIGHT = -2  # Target at/above the highest native level: clamp to the highest level (``right``).


# ---------------------------------------------------------------------------------------------
# Scalar helpers (inlined into the kernels).
# ---------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _dewpoint_c(q, p_pa):
    """Dewpoint (degC) from specific humidity (kg/kg) and pressure (Pa); see the public wrapper."""
    qq = max(np.float64(q), MIN_SPECIFIC_HUMIDITY)
    w = qq / (1.0 - qq)  # mixing ratio, kg/kg
    e = np.float64(p_pa) * w / (EPSILON + w)  # vapour pressure, Pa
    val = math.log(e / SAT_PRESSURE_0C)
    return BOLTON_B * val / (BOLTON_A - val)


@njit(cache=True, inline="always")
def _np_interp_segment(x, xa, xb, fa, fb):
    """One interior step of ``np.interp`` with exactly NumPy's arithmetic.

    NumPy (``numpy/_core/src/multiarray/compiled_base.c``, ``arr_interp``) returns ``fp[j]`` when
    ``x == xp[j]`` and otherwise ``slope * (x - xp[j]) + fp[j]`` with
    ``slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])``.  Using the same expression (rather than,
    say, a weight ``w`` and ``(1 - w) * fa + w * fb``) is what makes the result identical to the
    old ``np.interp``-based code rather than merely close.
    """
    if x == xa:
        return fa
    slope = (fb - fa) / (xb - xa)
    return slope * (x - xa) + fa


@njit(cache=True, inline="always")
def _es_liquid(t_k):
    """Saturation vapour pressure over liquid water (Pa), MetPy 1.7 formula.

    This is the integrated Clausius-Clapeyron equation with a temperature-dependent latent heat
    (``metpy.calc.thermo._saturation_vapor_pressure_liquid``); MetPy >= 1.6 uses it in place of
    Bolton's fit, and the wet-bulb temperature and the LCL are computed with it.
    """
    latent_heat = LV - (CP_L - CP_V) * (t_k - T0)
    heat_power = (CP_L - CP_V) / RV
    exp_term = (LV / T0 - latent_heat / t_k) / RV
    return SAT_PRESSURE_0C * (T0 / t_k) ** heat_power * math.exp(exp_term)


@njit(cache=True, inline="always")
def _es_ice(t_k):
    """Saturation vapour pressure over ice (Pa), MetPy 1.7 formula (``phase='solid'``)."""
    latent_heat = LS - (CP_I - CP_V) * (t_k - T0)
    heat_power = (CP_I - CP_V) / RV
    exp_term = (LS / T0 - latent_heat / t_k) / RV
    return SAT_PRESSURE_0C * (T0 / t_k) ** heat_power * math.exp(exp_term)


@njit(cache=True, inline="always")
def _lambertw_m1(z):
    """Lower branch W_{-1}(z) of the Lambert W function for -1/e <= z < 0, by Halley iteration.

    ``scipy.special.lambertw`` is not available inside numba.  W_{-1} solves ``w * exp(w) = z``
    with ``w <= -1``.  The starting guess is the branch-point series near ``z = -1/e`` and the
    asymptotic expansion ``log(-z) - log(-log(-z))`` near ``z = 0``; Halley's method then converges
    cubically, so a handful of iterations reach machine precision.
    """
    if z >= 0.0 or z < -1.0 / math.e:
        return np.nan
    if z < -0.25:
        p = -math.sqrt(2.0 * (math.e * z + 1.0))
        w = -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    else:
        l1 = math.log(-z)
        l2 = math.log(-l1)
        w = l1 - l2 + l2 / l1
    for _ in range(50):
        ew = math.exp(w)
        f = w * ew - z
        wp1 = w + 1.0
        if wp1 == 0.0:
            break
        denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1)
        dw = f / denom
        w -= dw
        if abs(dw) <= 1.0e-15 * (1.0 + abs(w)):
            break
    return w


@njit(cache=True, inline="always")
def _moist_lapse_dtdlnp(p_pa, t_k):
    """dT/dln(p) along a saturated (pseudo-)adiabat, MetPy's ``moist_lapse`` right-hand side.

    MetPy integrates ``dT/dp = frac / p`` with
    ``frac = (Rd*T + Lv*rs) / (Cp_d + Lv**2 * rs * eps / (Rd * T**2))``.  Integrating in ``ln p``
    instead of ``p`` gives the same ODE (``dT/dln p = frac``) with a much smoother right-hand side,
    so a fixed-step RK4 is accurate with few steps.
    """
    es = _es_liquid(t_k)
    rs = EPSILON * es / (p_pa - es)  # saturation mixing ratio, kg/kg
    return (RD * t_k + LV * rs) / (CP_D + LV * LV * rs * EPSILON / (RD * t_k * t_k))


# Step size in ln(p) for the RK4 moist-adiabat integration (0.1 is ~100 hPa near 1000 hPa; the
# step is shortened so that a whole number of steps spans LCL -> p).  The right-hand side varies
# slowly in ln(p), so RK4's O(h**4) error is tiny: against a 0.005 step the result changes by at
# most 1.9e-5 degC (2 M random points, -40..30 degC, 450..1050 hPa), and the agreement with MetPy
# (max 8.2e-5 degC on 13,744 points) is the same as with steps of 0.02 or 0.05, which cost 2.2x and
# 1.3x more time.  See benchmarks/results/columns_local.md.
_WB_DLNP = 0.1


@njit(cache=True)
def _wet_bulb_scalar(t_c, td_c, p_pa):
    """Wet-bulb temperature (degC) by Normand's rule, matching ``metpy.calc.wet_bulb_temperature``.

    1. Lifting condensation level with Romps' (2017) exact solution, as in ``metpy.calc.lcl``:
       ``T_lcl = c / W_{-1}(RH**(1/a) * c * exp(c)) * T`` and
       ``p_lcl = p * (T_lcl / T) ** (cp_m / R_m)``, with RH over liquid water.
    2. Follow the saturated adiabat from the LCL back down to the original pressure; the
       temperature reached there is the (pseudo-)wet-bulb temperature.

    MetPy integrates step 2 with LSODA; here a fixed-step RK4 in ln(p) is used because it is
    deterministic, branch-free and fast inside numba.  The dewpoint is clipped to the temperature
    (supersaturation is not physical in these inputs and would make RH > 1).
    """
    if not (np.isfinite(t_c) and np.isfinite(td_c) and np.isfinite(p_pa)):
        return np.nan
    t_k = np.float64(t_c) + ZERO_DEGC
    td_k = min(np.float64(td_c), np.float64(t_c)) + ZERO_DEGC
    if td_k >= t_k:
        return np.float64(t_c)
    p = np.float64(p_pa)
    # --- LCL (metpy.calc.lcl) ---
    e_d = _es_liquid(td_k)
    w_s = EPSILON * e_d / (p - e_d)
    q = w_s / (1.0 + w_s)
    moist_heat_ratio = (CP_D + q * (CP_V - CP_D)) / (RD + q * (RV - RD))
    spec_heat_diff = CP_L - CP_V
    a = moist_heat_ratio + spec_heat_diff / RV
    b = -(LV + spec_heat_diff * T0) / (RV * t_k)
    c = b / a
    rh = e_d / _es_liquid(t_k)
    w_m1 = _lambertw_m1(rh ** (1.0 / a) * c * math.exp(c))
    t_lcl = c / w_m1 * t_k
    p_lcl = p * (t_lcl / t_k) ** moist_heat_ratio
    # --- moist adiabat from (p_lcl, t_lcl) down to p, RK4 in ln(p) ---
    s0 = math.log(p_lcl)
    s1 = math.log(p)
    n = math.ceil(abs(s1 - s0) / _WB_DLNP)
    if n < 1:
        return t_lcl - ZERO_DEGC
    h = (s1 - s0) / n
    t = t_lcl
    s = s0
    for _ in range(n):
        k1 = _moist_lapse_dtdlnp(math.exp(s), t)
        k2 = _moist_lapse_dtdlnp(math.exp(s + 0.5 * h), t + 0.5 * h * k1)
        k3 = _moist_lapse_dtdlnp(math.exp(s + 0.5 * h), t + 0.5 * h * k2)
        k4 = _moist_lapse_dtdlnp(math.exp(s + h), t + h * k3)
        t += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        s += h
    return t - ZERO_DEGC


# ---------------------------------------------------------------------------------------------
# Dewpoint
# ---------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _dewpoint_kernel(q, p, out):
    for idx in prange(q.shape[0]):
        out[idx] = _dewpoint_c(q[idx], p[idx])


def dewpoint_from_specific_humidity(q, p_pa):
    """Dewpoint temperature (degC) from specific humidity and pressure.

    The formula is MetPy's (``metpy.calc.dewpoint_from_specific_humidity`` ->
    ``metpy.calc.dewpoint``):

    * mixing ratio ``w = q / (1 - q)``,
    * vapour pressure ``e = p * w / (eps + w)`` with ``eps = Rd/Rv = 0.6219569100577033``,
    * Bolton (1980): ``Td = 243.5 * ln(e / 611.2 Pa) / (17.67 - ln(e / 611.2 Pa))`` degC.

    This is the inverse of the Magnus/Bolton saturation curve over liquid water, i.e. the dewpoint
    is defined with respect to water at all temperatures - the same convention as HRRR's
    2 m dewpoint and as the training data, whose upper-air dewpoint came from
    ``metpy.calc.dewpoint_from_relative_humidity`` (also Bolton over water).
    ``q`` is floored at :data:`MIN_SPECIFIC_HUMIDITY` to avoid ``log(0)``.

    Args:
        q: Specific humidity, kg/kg, any shape.
        p_pa: Pressure, Pa, same shape as ``q``.

    Returns:
        Dewpoint in degC as float32, same shape as ``q``.
    """
    q = np.asarray(q)
    p_pa = np.asarray(p_pa)
    if q.shape != p_pa.shape:
        raise ValueError(f"q shape {q.shape} != p shape {p_pa.shape}")
    out = np.empty(q.shape, dtype=np.float32)
    _dewpoint_kernel(q.reshape(-1), p_pa.reshape(-1), out.reshape(-1))
    return out


# ---------------------------------------------------------------------------------------------
# Interpolation in height
# ---------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _column_is_increasing(z, j, i):
    for k in range(z.shape[0] - 1):
        if not (z[k + 1, j, i] > z[k, j, i]):  # also catches NaN
            return False
    return True


@njit(cache=True, inline="always")
def _bracket_column(z, j, i, targets, jidx):
    """Fill ``jidx`` with the bracketing native level for each target height of column (j, i).

    ``jidx[t] = k`` means ``z[k] <= targets[t] < z[k + 1]``; ``_LEFT`` / ``_RIGHT`` mean the target
    is below the lowest / at or above the highest native level, where ``np.interp`` returns the end
    value.  Because both the native heights and the targets increase, a single forward-moving
    pointer finds all brackets in O(n_levels + n_targets) instead of a binary search per target.
    Returns the highest native level index that will be read.
    """
    nlev = z.shape[0]
    z0 = np.float64(z[0, j, i])
    ztop = np.float64(z[nlev - 1, j, i])
    k = 0
    kmax = 0
    for t in range(targets.shape[0]):
        x = targets[t]
        if x < z0:
            jidx[t] = _LEFT
        elif x >= ztop:
            jidx[t] = _RIGHT
            kmax = nlev - 1
        else:
            while np.float64(z[k + 1, j, i]) <= x:
                k += 1
            jidx[t] = k
            kmax = max(kmax, k + 1)
    return kmax


@njit(parallel=True, cache=True)
def _interp_kernel(z, f, targets, out, use_log):
    nlev, ny, nx = z.shape
    nt = targets.shape[0]
    n_bad = 0
    for j in prange(ny):
        jidx = np.empty(nt, dtype=np.int64)
        for i in range(nx):
            if not _column_is_increasing(z, j, i):
                n_bad += 1
                for t in range(nt):
                    out[t, j, i] = np.nan
                continue
            _bracket_column(z, j, i, targets, jidx)
            for t in range(nt):
                k = jidx[t]
                if k == _LEFT:
                    val = np.float64(f[0, j, i])
                    val = math.log(val) if use_log else val
                elif k == _RIGHT:
                    val = np.float64(f[nlev - 1, j, i])
                    val = math.log(val) if use_log else val
                else:
                    fa = np.float64(f[k, j, i])
                    fb = np.float64(f[k + 1, j, i])
                    if use_log:
                        fa = math.log(fa)
                        fb = math.log(fb)
                    val = _np_interp_segment(targets[t], np.float64(z[k, j, i]), np.float64(z[k + 1, j, i]), fa, fb)
                out[t, j, i] = math.exp(val) if use_log else val
    return n_bad


def _check_targets(target_heights):
    targets = np.ascontiguousarray(target_heights, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("target_heights must be a non-empty 1-D array")
    if np.any(np.diff(targets) <= 0):
        raise ValueError("target_heights must be strictly increasing")
    return targets


def _interp(height_agl, field, target_heights, out, use_log):
    if height_agl.ndim != 3 or field.shape != height_agl.shape:
        raise ValueError(f"height_agl {height_agl.shape} and field {field.shape} must have the same (level, y, x) shape")
    targets = _check_targets(target_heights)
    if out is None:
        out = np.empty((targets.size,) + height_agl.shape[1:], dtype=np.float32)
    elif out.shape != (targets.size,) + height_agl.shape[1:]:
        raise ValueError(f"out has shape {out.shape}, expected {(targets.size,) + height_agl.shape[1:]}")
    n_bad = _interp_kernel(height_agl, field, targets, out, use_log)
    if n_bad:
        raise ValueError(f"{n_bad} columns have heights that are not strictly increasing (or NaN); their output was set to NaN")
    return out


def interp_to_heights(height_agl, field, target_heights, out=None):
    """Linearly interpolate a ``(level, y, x)`` field to fixed heights above ground, per column.

    This reproduces ``np.interp(target_heights, height_agl[:, j, i], field[:, j, i])`` exactly,
    including its clamping: targets below the lowest level take the lowest-level value and targets
    above the highest level take the highest-level value (no extrapolation).  Linear-in-height is
    what both the old inference code (``ptype/inference.py:281``) and the training preprocessing
    (``scripts/precip_RAP_interpolate.py:64-74``, ``scripts/rap_height_interp.py:51``) use.

    Levels must be ordered from the surface upward with strictly increasing heights in every column
    (true for HRRR hybrid levels).  This is checked per column at the cost of one pass over the
    heights; violating columns are set to NaN and a ``ValueError`` is raised afterwards.

    Args:
        height_agl: Heights above ground level (m), shape ``(level, y, x)``; heights may differ
            between columns.
        field: Values to interpolate, same shape as ``height_agl``.
        target_heights: 1-D strictly increasing target heights (m AGL).
        out: Optional float32 output array of shape ``(len(target_heights), y, x)``.

    Returns:
        float32 array of shape ``(len(target_heights), y, x)``.
    """
    return _interp(height_agl, field, target_heights, out, False)


def interp_log_pressure_to_heights(height_agl, pressure, target_heights, out=None):
    """Interpolate pressure to fixed heights linearly in ``log(p)`` (i.e. exponentially in height).

    In a hydrostatic atmosphere ``ln p`` is close to linear in height (exactly so in an isothermal
    layer), so this is more accurate than linear-in-p.  Note that the training preprocessing
    interpolated pressure *linearly* (``scripts/precip_RAP_interpolate.py:76``,
    ``scripts/rap_height_interp.py:61``), as does the old inference code (``ptype/inference.py:267-268``);
    between two native levels ``dz`` apart they differ by at most about ``p * (dz / H)**2 / 8``
    (scale height ``H`` ~ 8 km), i.e. ~0.1 hPa for a 250 m layer at 1000 hPa.  Equal to
    ``np.exp(np.interp(target_heights, z, np.log(p)))`` per column.

    Args:
        height_agl: Heights above ground level (m), shape ``(level, y, x)``.
        pressure: Pressure (any positive unit), same shape.
        target_heights: 1-D strictly increasing target heights (m AGL).
        out: Optional float32 output array of shape ``(len(target_heights), y, x)``.

    Returns:
        float32 pressure (same unit as the input) of shape ``(len(target_heights), y, x)``.
    """
    return _interp(height_agl, pressure, target_heights, out, True)


# ---------------------------------------------------------------------------------------------
# Wind rotation
# ---------------------------------------------------------------------------------------------
def grid_rotation_cos_sin(lon, lon_0=HRRR_LON_0, lat_std=HRRR_LAT_STD):
    """Cosine and sine of the grid-to-earth rotation angle of a tangent Lambert conformal grid.

    For a Lambert conformal projection the angle between grid "north" (+y) and true north at a
    point is ``alpha = n * (lon - lon_0)``, with cone constant ``n = sin(lat_std)`` for a tangent
    cone (one standard parallel, or Latin1 == Latin2 as for HRRR).  Meridians converge towards the
    apex of the cone, which lies on the central meridian above the pole, so east of the central
    meridian (``alpha > 0``) true north points up and to the *left* of grid +y; equivalently, grid
    +y points clockwise of (to the east of) true north by ``alpha``.

    ``lon - lon_0`` is wrapped into [-180, 180) so that longitudes in 0..360 (cfgrib) and -180..180
    both work.

    Args:
        lon: Longitude of each grid point, degrees, any shape.
        lon_0: Central meridian (GRIB LoV), degrees.
        lat_std: Standard (tangent) latitude, degrees.

    Returns:
        ``(cos(alpha), sin(alpha))`` as float64 arrays shaped like ``lon``.
    """
    n = math.sin(math.radians(lat_std))
    dlon = (np.asarray(lon, dtype=np.float64) - lon_0 + 180.0) % 360.0 - 180.0
    alpha = n * np.radians(dlon)
    return np.cos(alpha), np.sin(alpha)


@njit(parallel=True, cache=True)
def _rotate_kernel(u, v, cosa, sina, sign, u_out, v_out):
    nlev, ny, nx = u.shape
    for j in prange(ny):
        for lev in range(nlev):
            for i in range(nx):
                c = np.float64(cosa[j, i])
                s = sign * np.float64(sina[j, i])
                uu = np.float64(u[lev, j, i])
                vv = np.float64(v[lev, j, i])
                u_out[lev, j, i] = c * uu + s * vv
                v_out[lev, j, i] = -s * uu + c * vv


def _rotate(u, v, lon, lon_0, lat_std, sign):
    u = np.asarray(u)
    v = np.asarray(v)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    squeeze = u.ndim == 2
    if squeeze:
        u = u[np.newaxis]
        v = v[np.newaxis]
    if u.ndim != 3 or u.shape[1:] != np.shape(lon):
        raise ValueError(f"u/v must be (y, x) or (level, y, x) with (y, x) == lon.shape; got {u.shape} and {np.shape(lon)}")
    cosa, sina = grid_rotation_cos_sin(lon, lon_0, lat_std)
    u_out = np.empty(u.shape, dtype=np.float32)
    v_out = np.empty(v.shape, dtype=np.float32)
    _rotate_kernel(u, v, cosa, sina, sign, u_out, v_out)
    if squeeze:
        return u_out[0], v_out[0]
    return u_out, v_out


def rotate_grid_to_earth(u, v, lon, lon_0=HRRR_LON_0, lat_std=HRRR_LAT_STD):
    """Rotate grid-relative wind components to earth-relative (east, north) components.

    HRRR and RAP GRIB files store u and v relative to the Lambert conformal grid axes (GRIB2
    resolution-and-component flag bit 5 set).  With ``alpha = sin(lat_std) * (lon - lon_0)``
    (see :func:`grid_rotation_cos_sin`)::

        u_earth =  cos(alpha) * u_grid + sin(alpha) * v_grid
        v_earth = -sin(alpha) * u_grid + cos(alpha) * v_grid

    This is the NCEP / wgrib2 convention.  Sign check: east of the central meridian
    (``alpha > 0``) grid +y points to the east of true north, so a pure grid-"north" wind
    (``u_grid = 0, v_grid > 0``) has a positive (westerly-from, i.e. eastward) earth-relative u.

    Args:
        u: Grid-relative u (m/s), shape ``(y, x)`` or ``(level, y, x)``.
        v: Grid-relative v (m/s), same shape as ``u``.
        lon: Longitudes (degrees), shape ``(y, x)``.
        lon_0: Central meridian (degrees); HRRR -97.5.
        lat_std: Tangent latitude (degrees); HRRR 38.5.

    Returns:
        ``(u_earth, v_earth)`` as float32 arrays shaped like ``u``.
    """
    return _rotate(u, v, lon, lon_0, lat_std, 1.0)


def rotate_earth_to_grid(u, v, lon, lon_0=HRRR_LON_0, lat_std=HRRR_LAT_STD):
    """Inverse of :func:`rotate_grid_to_earth` (rotation by ``-alpha``).

    Args:
        u: Earth-relative u (m/s), shape ``(y, x)`` or ``(level, y, x)``.
        v: Earth-relative v (m/s), same shape as ``u``.
        lon: Longitudes (degrees), shape ``(y, x)``.
        lon_0: Central meridian (degrees).
        lat_std: Tangent latitude (degrees).

    Returns:
        ``(u_grid, v_grid)`` as float32 arrays shaped like ``u``.
    """
    return _rotate(u, v, lon, lon_0, lat_std, -1.0)


# ---------------------------------------------------------------------------------------------
# Fused sounding builder
# ---------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _build_soundings_kernel(z, t_k, q, p_pa, u, v, t2m_k, td2m_k, u10, v10, psfc_pa, targets, t_policy, wind_policy, p_policy, p_log, clip_td, do_rot, rcos, rsin, out):
    nlev, ny, nx = z.shape
    nt = targets.shape[0]
    n_bad = 0
    n_short = 0
    for j in prange(ny):
        jidx = np.empty(nt, dtype=np.int64)
        td_buf = np.empty(nlev, dtype=np.float64)
        for i in range(nx):
            c = j * nx + i
            if not _column_is_increasing(z, j, i):
                n_bad += 1
                for g in range(5):
                    for t in range(nt):
                        out[c, g, t] = np.nan
                continue
            if np.float64(z[nlev - 1, j, i]) < targets[nt - 1]:
                n_short += 1
            kmax = _bracket_column(z, j, i, targets, jidx)
            # Dewpoint only on the native levels the interpolation will read.
            for k in range(kmax + 1):
                td_buf[k] = _dewpoint_c(q[k, j, i], p_pa[k, j, i])
            for t in range(nt):
                k = jidx[t]
                if k == _LEFT or k == _RIGHT:
                    kk = 0 if k == _LEFT else nlev - 1
                    tc = np.float64(t_k[kk, j, i]) - ZERO_DEGC
                    td = td_buf[kk]
                    uu = np.float64(u[kk, j, i])
                    vv = np.float64(v[kk, j, i])
                    pp = np.float64(p_pa[kk, j, i])
                    if p_log:
                        pp = math.log(pp)
                else:
                    x = targets[t]
                    za = np.float64(z[k, j, i])
                    zb = np.float64(z[k + 1, j, i])
                    tc = _np_interp_segment(x, za, zb, np.float64(t_k[k, j, i]) - ZERO_DEGC, np.float64(t_k[k + 1, j, i]) - ZERO_DEGC)
                    td = _np_interp_segment(x, za, zb, td_buf[k], td_buf[k + 1])
                    uu = _np_interp_segment(x, za, zb, np.float64(u[k, j, i]), np.float64(u[k + 1, j, i]))
                    vv = _np_interp_segment(x, za, zb, np.float64(v[k, j, i]), np.float64(v[k + 1, j, i]))
                    pa = np.float64(p_pa[k, j, i])
                    pb = np.float64(p_pa[k + 1, j, i])
                    if p_log:
                        pa = math.log(pa)
                        pb = math.log(pb)
                    pp = _np_interp_segment(x, za, zb, pa, pb)
                if p_log:
                    pp = math.exp(pp)
                if t == 0:
                    if t_policy == 0:
                        tc = np.float64(t2m_k[j, i]) - ZERO_DEGC
                        td = np.float64(td2m_k[j, i]) - ZERO_DEGC
                    if clip_td and td > tc:
                        td = tc
                    if wind_policy == 0:
                        uu = np.float64(u10[j, i])
                        vv = np.float64(v10[j, i])
                    if p_policy == 0:
                        pp = np.float64(psfc_pa[j, i])
                if do_rot:
                    cs = np.float64(rcos[j, i])
                    sn = np.float64(rsin[j, i])
                    ue = cs * uu + sn * vv
                    vv = -sn * uu + cs * vv
                    uu = ue
                out[c, 0, t] = tc
                out[c, 1, t] = td
                out[c, 2, t] = uu
                out[c, 3, t] = vv
                out[c, 4, t] = pp / 100.0
    return n_bad, n_short


_DUMMY_2D = np.zeros((1, 1), dtype=np.float32)


def _as_f32(a, name, shape):
    if a is None:
        raise ValueError(f"{name} is required by the selected surface policy")
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.shape != shape:
        raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
    return a


def build_soundings(
    height_agl,
    temperature_k,
    specific_humidity,
    pressure_pa,
    u,
    v,
    t2m_k=None,
    td2m_k=None,
    u10=None,
    v10=None,
    psfc_pa=None,
    target_heights=NETWORK_HEIGHTS,
    surface_temperature="2m",
    surface_wind="10m",
    surface_pressure="surface",
    pressure_interp="linear",
    clip_surface_dewpoint=False,
    rotation=None,
    out=None,
):
    """Build per-column soundings on fixed heights AGL from native-level fields, in one pass.

    For every column this finds, once, the native levels that bracket each target height, computes
    the dewpoint from specific humidity on just those levels, and interpolates temperature,
    dewpoint, u, v (linear in height, exactly like ``np.interp``) and pressure (linear or log-p).
    Then the 0 m level is set according to the surface policies:

    * ``surface_temperature="2m"`` (default): T and Td at 0 m are the 2 m values.  Evidence: the old
      inference code (``ptype/inference.py:250,264``) and both training preprocessing scripts
      (``scripts/precip_RAP_interpolate.py:79-80``, ``scripts/rap_height_interp.py:89,93-94``).
      ``"interp"`` instead clamps to the lowest native level (np.interp behaviour).
    * ``surface_wind="10m"`` (default): u and v at 0 m are the 10 m winds.  Evidence:
      ``scripts/precip_RAP_interpolate.py:81-82``, ``scripts/rap_height_interp.py:90-91`` and
      ``ptype/inference.py:250`` (``surface_match`` maps u -> u10, v -> v10).  ``"interp"`` clamps
      to the lowest native level.
    * ``surface_pressure="surface"`` (default): p at 0 m is the surface pressure.  Evidence:
      ``scripts/precip_RAP_interpolate.py:83`` and ``scripts/rap_height_interp.py:92``
      (PRES_ON_SURFACE).  ``"interp"`` reproduces the old inference code, which did not override
      the pressure (``ptype/inference.py:267-268``) and therefore clamped to the lowest level.
    * ``clip_surface_dewpoint``: if True, the 0 m dewpoint is limited to the 0 m temperature, as
      ``scripts/rap_height_interp.py:93`` did; ``scripts/precip_RAP_interpolate.py`` and the old
      inference code did not, hence the default False.

    Above 0 m nothing is extrapolated: a target above the top native level takes the top value (the
    number of such columns is returned so the caller can pass enough levels).

    Wind rotation: the old inference code used the GRIB (grid-relative) winds unrotated, and the
    repository does not show whether the RAP training winds were rotated, so the default is no
    rotation.  Pass ``rotation=grid_rotation_cos_sin(lon)`` to rotate all levels (and the 10 m
    wind) to earth-relative.  Rotating after interpolation is exact because the rotation angle is
    constant in a column and interpolation is linear.

    Args:
        height_agl: Height above ground (m) of each native level, ``(level, y, x)``, surface upward.
            Only the lowest levels that reach the top target height are needed; pass a slice.
        temperature_k: Temperature (K), ``(level, y, x)``.
        specific_humidity: Specific humidity (kg/kg), ``(level, y, x)``.
        pressure_pa: Pressure (Pa), ``(level, y, x)``.
        u: u wind (m/s), ``(level, y, x)``.
        v: v wind (m/s), ``(level, y, x)``.
        t2m_k: 2 m temperature (K), ``(y, x)``; required for ``surface_temperature="2m"``.
        td2m_k: 2 m dewpoint (K), ``(y, x)``; required for ``surface_temperature="2m"``.
        u10: 10 m u wind (m/s), ``(y, x)``; required for ``surface_wind="10m"``.
        v10: 10 m v wind (m/s), ``(y, x)``; required for ``surface_wind="10m"``.
        psfc_pa: Surface pressure (Pa), ``(y, x)``; required for ``surface_pressure="surface"``.
        target_heights: Strictly increasing target heights (m AGL); default 0..5000 by 250.
        surface_temperature: ``"2m"`` or ``"interp"``.
        surface_wind: ``"10m"`` or ``"interp"``.
        surface_pressure: ``"surface"`` or ``"interp"``.
        pressure_interp: ``"linear"`` (training and old code) or ``"log"``.
        clip_surface_dewpoint: Limit the 0 m dewpoint to the 0 m temperature.
        rotation: Optional ``(cos_alpha, sin_alpha)`` arrays of shape ``(y, x)`` from
            :func:`grid_rotation_cos_sin`; rotates winds from grid- to earth-relative.
        out: Optional float32 output of shape ``(y * x, 5, n_targets)``.

    Returns:
        ``(soundings, n_short)`` where ``soundings`` is a C-contiguous float32 array of shape
        ``(y * x, 5, n_targets)`` (columns in row-major (y, x) order; groups T degC, Td degC,
        u m/s, v m/s, p hPa; heights surface to top).  ``soundings.reshape(n, -1)`` is the
        ``(n_columns, 5 * n_targets)`` sounding matrix.  ``n_short`` counts columns whose top
        native level is below the top target height.

    Raises:
        ValueError: On bad shapes/policies, or if any column's heights are not strictly increasing.
    """
    try:
        t_policy = _TEMP_POLICIES[surface_temperature]
        wind_policy = _WIND_POLICIES[surface_wind]
        p_policy = _PRES_POLICIES[surface_pressure]
        p_log = _PRES_INTERP[pressure_interp]
    except KeyError as err:
        raise ValueError(f"unknown policy {err}") from None
    z = np.asarray(height_agl)
    if z.ndim != 3:
        raise ValueError("height_agl must be (level, y, x)")
    shape3 = z.shape
    shape2 = shape3[1:]
    fields = []
    for name, a in (("temperature_k", temperature_k), ("specific_humidity", specific_humidity), ("pressure_pa", pressure_pa), ("u", u), ("v", v)):
        a = np.asarray(a)
        if a.shape != shape3:
            raise ValueError(f"{name} has shape {a.shape}, expected {shape3}")
        fields.append(a)
    t2m = _as_f32(t2m_k, "t2m_k", shape2) if t_policy == 0 else _DUMMY_2D
    td2m = _as_f32(td2m_k, "td2m_k", shape2) if t_policy == 0 else _DUMMY_2D
    u10a = _as_f32(u10, "u10", shape2) if wind_policy == 0 else _DUMMY_2D
    v10a = _as_f32(v10, "v10", shape2) if wind_policy == 0 else _DUMMY_2D
    psfc = _as_f32(psfc_pa, "psfc_pa", shape2) if p_policy == 0 else _DUMMY_2D
    if rotation is not None:
        rcos = np.asarray(rotation[0], dtype=np.float64)
        rsin = np.asarray(rotation[1], dtype=np.float64)
        if rcos.shape != shape2 or rsin.shape != shape2:
            raise ValueError("rotation arrays must have shape (y, x)")
    else:
        rcos = rsin = np.zeros((1, 1), dtype=np.float64)
    targets = _check_targets(target_heights)
    n_columns = shape2[0] * shape2[1]
    if out is None:
        out = np.empty((n_columns, 5, targets.size), dtype=np.float32)
    elif out.shape != (n_columns, 5, targets.size) or out.dtype != np.float32:
        raise ValueError(f"out must be float32 with shape {(n_columns, 5, targets.size)}")
    n_bad, n_short = _build_soundings_kernel(
        z, *fields, t2m, td2m, u10a, v10a, psfc, targets, t_policy, wind_policy, p_policy, bool(p_log), bool(clip_surface_dewpoint), rotation is not None, rcos, rsin, out
    )
    if n_bad:
        raise ValueError(f"{n_bad} columns have heights that are not strictly increasing (or NaN); their output was set to NaN")
    return out, int(n_short)


@njit(parallel=True, cache=True)
def _copy_network_kernel(sounding, out):
    n, _, nt = sounding.shape
    for c in prange(n):
        for g in range(4):
            for t in range(nt):
                out[c, g * nt + t] = sounding[c, g, t]


def network_inputs(sounding, out=None):
    """The ``(n_columns, 4 * n_heights)`` network input matrix from a sounding array.

    Column order is T(0..5000 m), Td(0..5000 m), U(0..5000 m), V(0..5000 m), matching the
    feature order of ``config/ptype.yml`` (input_features / the per-variable height lists) and of
    the old ``convert_and_interpolate`` (``np.concatenate(var_arrays, axis=1)``).  The copy is a
    parallel numba loop because ``sounding[:, :4, :]`` is a strided view and Keras/torch want a
    C-contiguous float32 matrix.

    Args:
        sounding: float32 ``(n_columns, 5, n_heights)`` from :func:`build_soundings`.
        out: Optional C-contiguous float32 output ``(n_columns, 4 * n_heights)``.

    Returns:
        C-contiguous float32 ``(n_columns, 4 * n_heights)`` array (84 columns for 21 heights).
    """
    n, g, nt = sounding.shape
    if g != 5:
        raise ValueError("sounding must be (n_columns, 5, n_heights)")
    if out is None:
        out = np.empty((n, 4 * nt), dtype=np.float32)
    _copy_network_kernel(sounding, out)
    return out


def sounding_matrix(sounding):
    """The ``(n_columns, 5 * n_heights)`` sounding matrix (T, Td, U, V, p hPa) as a zero-copy view.

    Args:
        sounding: float32 ``(n_columns, 5, n_heights)`` from :func:`build_soundings`.

    Returns:
        A ``(n_columns, 105)`` view for 21 heights.
    """
    return sounding.reshape(sounding.shape[0], -1)


# ---------------------------------------------------------------------------------------------
# Wet-bulb temperature
# ---------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _wet_bulb_kernel(t, td, p_pa, out):
    for idx in prange(t.shape[0]):
        out[idx] = _wet_bulb_scalar(t[idx], td[idx], p_pa[idx])


def wet_bulb_temperature(t_c, td_c, p_hpa):
    """Wet-bulb temperature (degC) by Normand's rule, vectorised and parallel.

    Same method and constants as ``metpy.calc.wet_bulb_temperature`` (which the training QC used,
    ``ptype/qc.py:107``): lift to the LCL (Romps 2017 exact LCL, solved with an iterative
    Lambert-W), then descend along the saturated adiabat (fixed-step RK4 in ln p) to the original
    pressure.  Dewpoints above the temperature are clipped to the temperature.  NaN in -> NaN out.

    Args:
        t_c: Temperature (degC), any shape.
        td_c: Dewpoint (degC), same shape.
        p_hpa: Pressure (hPa), same shape.

    Returns:
        Wet-bulb temperature (degC), float32 if ``t_c`` is float32 else float64, shaped like ``t_c``.
    """
    t_c = np.asarray(t_c)
    td_c = np.asarray(td_c)
    p_pa = np.asarray(p_hpa, dtype=np.float64) * 100.0
    if not (t_c.shape == td_c.shape == p_pa.shape):
        raise ValueError("t_c, td_c and p_hpa must have the same shape")
    out = np.empty(t_c.shape, dtype=np.float32 if t_c.dtype == np.float32 else np.float64)
    _wet_bulb_kernel(t_c.reshape(-1), td_c.reshape(-1), p_pa.reshape(-1), out.reshape(-1))
    return out
