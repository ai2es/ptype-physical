"""Validate ``ptype.hrrr_fast.bourgouin``: the revised (paper) method and the legacy repo method.

Two independent pure-Python references live here:

* ``paper_*``: the revised Bourgouin method of Birk et al. (2021, Wea. Forecasting 36, 425-438),
  written with plain loops and MetPy's saturation vapour pressures, independently of the numba code
  path.  The readings chosen where the paper is not explicit are documented in
  ``ptype/hrrr_fast/bourgouin.py`` and ``benchmarks/results/bourgouin_questions.md``.
* ``ref_*``: a copy of the modified-Bourgouin code in ``ptype/algorithms.py`` with only the changes
  needed to make its evident intent run.  ``ptype/algorithms.py`` itself is not imported (it is left
  untouched and, as written, raises on every profile with a 0 degC crossing).

Legacy reference corrections relative to ``ptype/algorithms.py`` (each marked ``# CORRECTION`` below):

1. ``calc_sounding_stats`` (lines 67-75) uses the *values* ``cold_area`` (a list, unhashable ->
   ``TypeError``), ``warm_area``, ``cold_thickness`` ... as dict keys.  They are replaced by the
   string keys used in the no-crossing branch (lines 42-50).  The insertion order, which
   ``metrics.values()`` unpacking in ``calc_modified_bourgouin_ptype`` relies on, is unchanged.
2. ``calc_prob_ice`` line 180 ``min_layer_temp <+ -15`` parses as ``< -15`` (unary plus).  It is read
   as a typo for ``<=`` (``+`` and ``=`` share a key); with the literal ``<``, ``Tmin == -15`` falls
   through to ``prob_ice = 0`` instead of 100.  Only the exact value -15.0 is affected.
3. ``calc_sounding_stats`` line 40 bare ``except:`` -> ``except ValueError:`` (the only exception the
   ``try`` block can raise for finite input is ``ValueError`` from ``.max()`` of an empty array).
   This keeps ruff happy and does not change behaviour.

Not reproduced: ``calc_bourgouin_ptype`` / ``bourgouin_ip_frzr`` (the classic, non-modified method),
which are not part of the modified method; see ``benchmarks/results/bourgouin_questions.md``.
"""

import itertools
import warnings

import numpy as np
import pytest
from metpy.calc import relative_humidity_from_dewpoint, saturation_vapor_pressure
from metpy.units import units
from ptype.hrrr_fast.bourgouin import (
    _revised_bourgouin_from_sounding_full,
    legacy_repo_bourgouin,
    legacy_repo_bourgouin_from_sounding,
    revised_bourgouin,
    revised_bourgouin_from_sounding,
    revised_probabilities,
)
from ptype.hrrr_fast.columns import NETWORK_HEIGHTS, wet_bulb_temperature


# ------------------------------------------------------------------------------------------------
# Reference: ptype/algorithms.py with the minimal corrections listed in the module docstring.
# ------------------------------------------------------------------------------------------------
def ref_add_zero_crossings(profile, heights):
    pre_crossing_level = np.argwhere((np.diff(np.sign(profile)) != 0) * 1).flatten()
    post_crossing_level = pre_crossing_level + 1
    indices = []
    crossings_m = []
    for pre, post in zip(pre_crossing_level, post_crossing_level):
        xp = profile[pre : post + 1]
        fp = heights[pre : post + 1]
        if xp[0] > xp[1]:
            xp = xp[::-1]
            fp = fp[::-1]
        crossings_m.append(np.interp(0, xp, fp))
        indices.append(post)
    profile = np.insert(profile, indices, 0)
    heights = np.insert(heights, indices, crossings_m)
    return profile, heights


def ref_calc_sounding_stats(profile, heights):
    FREEZING_K = 273.15
    GRAVITY = 9.81
    cold_area, warm_area = [], []
    low_i, cold_thickness, warm_thickness = 0, 0, 0
    profile, heights = ref_add_zero_crossings(profile, heights)
    surface = profile[0]
    metrics = {}
    try:
        upper_bound_index = np.argwhere(profile == 0).max() + 1
        lower_bound_index = np.argwhere(profile == 0).min()
        highest_crossing = heights[upper_bound_index - 1]
        lowest_crossing = heights[lower_bound_index]
    except ValueError:  # CORRECTION 3: was a bare ``except:``
        min_cold, max_warm, t_span, highest_crossing, lowest_crossing = np.nan, np.nan, np.nan, np.nan, np.nan  # noqa: F841
        metrics = dict(  # noqa: C408
            cold_area=np.sum(cold_area),
            warm_area=np.sum(warm_area),
            cold_thickness=cold_thickness,
            warm_thickness=warm_thickness,
            min_cold=min_cold,
            max_warm=max_warm,
            highest_crossing=highest_crossing,
            lowest_crossing=lowest_crossing,
            surface=surface,
        )
        return metrics
    min_cold = profile[:upper_bound_index].min()
    max_warm = profile[:upper_bound_index].max()
    for i in range(upper_bound_index):
        if profile[i] == 0:
            energy = np.trapezoid(GRAVITY * (profile[low_i : i + 1] / FREEZING_K), heights[low_i : i + 1])
            if energy <= 0:
                cold_area.append(energy)
                cold_thickness += heights[i] - heights[low_i]
            else:
                warm_area.append(energy)
                warm_thickness += heights[i] - heights[low_i]
            low_i = i
        # CORRECTION 1: string keys instead of the list/float values used as keys.
        metrics["cold_area"] = np.sum(cold_area)
        metrics["warm_area"] = np.sum(warm_area)
        metrics["cold_thickness"] = cold_thickness
        metrics["warm_thickness"] = warm_thickness
        metrics["min_cold"] = min_cold
        metrics["max_warm"] = max_warm
        metrics["highest_crossing"] = highest_crossing
        metrics["lowest_crossing"] = lowest_crossing
        metrics["surface"] = surface
    return metrics


def ref_get_layer_indices(arr, min_length=5):
    start = None
    for i in range(len(arr)):
        if arr[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                return (start, i - 1)
            start = None
    if start is not None and len(arr) - start >= min_length:
        return (start, len(arr) - 1)
    return (None, None)


def ref_calc_prob_ice(t_profile, td_profile, height_resolution=250):
    SAT_LAYER_THRESH = 1000
    UNSAT_LAYER_THRESH = 1500
    rh_i = relative_humidity_from_dewpoint(t_profile * units.degC, td_profile * units.degC, phase="solid").magnitude[::-1]
    saturated = np.where(rh_i > 0.75, 1, 0)
    unsaturated = np.where(saturated == 1, 0, 1)
    layer_top_i, layer_bottom_i = ref_get_layer_indices(arr=saturated, min_length=np.ceil(SAT_LAYER_THRESH / height_resolution) + 1)
    if layer_top_i is not None:
        sfc_layer_top_i, sfc_layer_bottom_i = ref_get_layer_indices(  # noqa: RUF059
            arr=unsaturated[layer_bottom_i:], min_length=np.ceil(UNSAT_LAYER_THRESH / height_resolution) + 1
        )
        if sfc_layer_top_i is not None:
            return 0
        min_layer_temp = min(t_profile[::-1][layer_top_i : layer_bottom_i + 1])
        if min_layer_temp <= -15:  # CORRECTION 2: was ``<+ -15``
            return 100
        elif (min_layer_temp > -15) & (min_layer_temp < -7):
            return (-0.065 * min_layer_temp**4) - (3.1544 * min_layer_temp**3) - (56.414 * min_layer_temp**2) - (449.6 * min_layer_temp) - (1308)
        else:
            return 0
    else:
        return 0


def ref_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ref_calc_modified_bourgouin_ptype(t_profile, td_profile, twb_profile, h_profile):
    prob_ice = ref_calc_prob_ice(t_profile, td_profile)
    wb_profile, height_profile = ref_add_zero_crossings(twb_profile, h_profile)
    metrics = ref_calc_sounding_stats(wb_profile, height_profile)
    NA, PA, CT, WT, min_C, max_C, max_cross_m, min_cross_m, sfc = metrics.values()  # noqa: RUF059
    prob_snow_i = 1540 * np.exp(-(0.29 * PA))
    prob_snow = (prob_ice / 100) * prob_snow_i
    if PA > 0:
        prob_icep_i = ((2.3 * NA) - (42 * np.log(PA + 1)) + 3) * prob_ice
    else:
        prob_icep_i = 0
    prob_icep = (prob_ice / 100) * prob_icep_i
    prob_frzr_i = (-2.1 * NA) + (0.2 * PA) + 458
    if PA < 5:
        prob_frzr_i = prob_frzr_i * 0.2 * PA
    prob_frzr = (100 - prob_ice) + ((prob_ice / 100) * prob_frzr_i)
    if sfc > 0:
        prob_rain = prob_frzr
        prob_frzr = 0
    else:
        prob_rain = 0
    ptypes = ref_softmax(np.array([prob_rain, prob_snow, prob_icep, prob_frzr]) / 100)
    cat_pred = np.argwhere(ptypes == np.max(ptypes)).flatten()[-1]
    return ptypes, cat_pred


# ------------------------------------------------------------------------------------------------
# Synthetic but realistic winter soundings.
# ------------------------------------------------------------------------------------------------
def make_profiles(n, seed=0, heights=NETWORK_HEIGHTS, tsfc_range=(-14.0, 10.0)):
    """Random winter-like T, Td (degC) and p (hPa) profiles on the 21 network heights.

    Mixes: surface temperature -14..+10 degC (3 % warm-season columns at 25..32 degC) with 2-8.5 K/km lapse rates; elevated warm noses
    (60 %, up to +12 K at 0.5-3 km) giving freezing rain / sleet structures; shallow surface cold
    pools (30 %); small oscillations (20 %) that create 3+ wet-bulb crossings; dew-point
    depressions from saturated to very dry, with dry slots in 50 % of columns; surface pressure
    650-1045 hPa (high terrain to cold highs).  ``heights`` and ``tsfc_range`` (degC) allow other
    grids (e.g. 0-8000 m) and warmer mixes.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(heights, dtype=np.float64)[np.newaxis, :]
    tsfc = rng.uniform(tsfc_range[0], tsfc_range[1], (n, 1))
    lapse = rng.uniform(2.0, 8.5, (n, 1)) / 1000.0
    warm = rng.random((n, 1)) < 0.03  # a few warm-season columns with no 0 degC crossing below 5 km
    tsfc = np.where(warm, rng.uniform(25, 32, (n, 1)), tsfc)
    lapse = np.where(warm, rng.uniform(2.0, 4.0, (n, 1)) / 1000.0, lapse)
    t = tsfc - lapse * z
    nose = (rng.random((n, 1)) < 0.6) * rng.uniform(0, 12, (n, 1))
    t += nose * np.exp(-(((z - rng.uniform(500, 3000, (n, 1))) / rng.uniform(250, 900, (n, 1))) ** 2))
    pool = (rng.random((n, 1)) < 0.3) * rng.uniform(0, 6, (n, 1))
    t -= pool * np.exp(-z / rng.uniform(200, 800, (n, 1)))
    wiggle = (rng.random((n, 1)) < 0.2) * rng.uniform(0, 3, (n, 1))
    t += wiggle * np.sin(2 * np.pi * z / rng.uniform(800, 2000, (n, 1)) + rng.uniform(0, 2 * np.pi, (n, 1)))
    dd = rng.uniform(0, 2, (n, 1)) * rng.random((n, z.shape[1])) + rng.exponential(1.0, (n, 1))
    dry = rng.random((n, 1)) < 0.5
    z0 = rng.uniform(0, 4000, (n, 1))
    dd += dry * ((z >= z0) & (z <= z0 + rng.uniform(500, 3000, (n, 1)))) * rng.uniform(3, 25, (n, 1))
    td = t - dd
    psfc = rng.uniform(650, 1045, (n, 1))
    tmean_k = tsfc + 273.15 - lapse * z / 2.0
    p = psfc * np.exp(-9.81 * z / (287.0 * tmean_k))
    return t.astype(np.float32), td.astype(np.float32), p.astype(np.float32)


def count_crossings(tw):
    s = np.sign(tw)
    return (np.diff(s, axis=1) != 0).sum(axis=1)


@pytest.fixture(scope="module")
def profiles():
    n = 20000
    t, td, p = make_profiles(n, seed=42)
    tw = wet_bulb_temperature(t, td, p)
    # 5 % of the columns get Tw (and T, Td) rounded to 0.5 degC, which puts exact zeros into the
    # profiles and exercises the touch-zero / duplicate-crossing paths of add_zero_crossings.
    rnd = np.random.default_rng(7).random(n) < 0.05
    tw[rnd] = np.round(tw[rnd] * 2) / 2
    t[rnd] = np.round(t[rnd] * 2) / 2
    td[rnd] = np.minimum(np.round(td[rnd] * 2) / 2, t[rnd])
    # A few columns with Tmin exactly -15 / -7 inside a saturated layer (prob_ice branch edges).
    t[:20, 8:16] = -15.0
    td[:20, 8:16] = -15.0
    t[20:40, 8:16] = -7.0
    td[20:40, 8:16] = -7.0
    return t, td, tw, p


@pytest.fixture(scope="module")
def reference(profiles):
    t, td, tw, _ = profiles
    h = NETWORK_HEIGHTS.astype(np.float64)
    probs = np.empty((t.shape[0], 4))
    cls = np.empty(t.shape[0], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(t.shape[0]):
            probs[i], cls[i] = ref_calc_modified_bourgouin_ptype(t[i].astype(np.float64), td[i].astype(np.float64), tw[i].astype(np.float64), h)
    return probs, cls


def test_profile_coverage(profiles):
    """The random profiles cover the regimes the method distinguishes."""
    _, _, tw, _ = profiles
    nc = count_crossings(tw)
    sfc_warm = tw[:, 0] > 0
    assert (nc == 0).sum() > 500
    assert (nc == 1).sum() > 500
    assert (nc == 2).sum() > 500
    assert (nc >= 3).sum() > 200
    # Odd crossing counts force the surface to the opposite side of 0 degC from the top (always
    # below freezing at 5 km here), so only these combinations can occur.
    assert ((nc == 0) & sfc_warm).sum() > 50 and ((nc == 0) & ~sfc_warm).sum() > 50
    assert ((nc == 1) & sfc_warm).sum() > 500
    assert ((nc == 2) & ~sfc_warm).sum() > 500
    assert ((nc >= 3) & sfc_warm).sum() > 200 and ((nc >= 3) & ~sfc_warm).sum() > 20
    assert (tw == 0).any(axis=1).sum() > 100  # exact zeros present


def test_numba_matches_reference(profiles, reference):
    t, td, tw, _ = profiles
    ref_probs, ref_cls = reference
    probs, cls = legacy_repo_bourgouin(t, td, tw)
    assert probs.dtype == np.float32 and cls.dtype == np.uint8
    diff = np.abs(probs.astype(np.float64) - ref_probs)
    assert diff.max() < 1e-6, diff.max()
    assert (cls == ref_cls).mean() == 1.0
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_class_occurrence(reference):
    """Rain, snow and freezing rain occur; ice pellets never win with the literal formulas.

    This documents a property of the original equations (see bourgouin_questions.md): with the
    cold area NA <= 0 and prob_ice applied twice, the ice-pellet score is at most
    3 * prob_ice <= 300 while the snow score is 1540 * exp(-0.29 PA), and for PA large enough to
    shrink the snow score the -42 ln(PA + 1) term already makes the ice-pellet score negative.
    """
    _, ref_cls = reference
    assert set(np.unique(ref_cls)) == {0, 1, 3}


def test_from_sounding_matches(profiles):
    t, td, _, p = profiles
    n = 2000
    sounding = np.zeros((n, 5, 21), dtype=np.float32)
    sounding[:, 0], sounding[:, 1], sounding[:, 4] = t[:n], td[:n], p[:n]
    tw = wet_bulb_temperature(t[:n], td[:n], p[:n])
    probs_a, cls_a = legacy_repo_bourgouin(t[:n], td[:n], tw)
    probs_b, cls_b, tw_b = legacy_repo_bourgouin_from_sounding(sounding, return_wet_bulb=True)
    np.testing.assert_array_equal(tw_b, tw)
    np.testing.assert_array_equal(probs_a, probs_b)
    np.testing.assert_array_equal(cls_a, cls_b)


def test_strided_view_input(profiles):
    t, td, tw, _ = profiles
    n = 500
    sounding = np.zeros((n, 5, 21), dtype=np.float32)
    sounding[:, 0], sounding[:, 1], sounding[:, 2] = t[:n], td[:n], tw[:n]
    probs_a, cls_a = legacy_repo_bourgouin(t[:n], td[:n], tw[:n])
    probs_b, cls_b = legacy_repo_bourgouin(sounding[:, 0], sounding[:, 1], sounding[:, 2])
    np.testing.assert_array_equal(probs_a, probs_b)
    np.testing.assert_array_equal(cls_a, cls_b)


def test_nan_column_flagged(profiles):
    t, td, tw, _ = profiles
    t = t[:3].copy()
    t[1, 5] = np.nan
    probs, cls = legacy_repo_bourgouin(t, td[:3], tw[:3])
    assert cls[1] == 255 and np.isnan(probs[1]).all()
    assert cls[0] != 255 and cls[2] != 255


def test_known_cases():
    """Hand-checkable profiles."""
    h = NETWORK_HEIGHTS
    # Warm, moist, no crossing: prob_ice = 0, so frzr score = 100 is relabelled rain.
    t = np.full((1, 21), 5.0, np.float32)
    probs, cls = legacy_repo_bourgouin(t, t - 1, t - 0.5)
    ref, ref_cls = ref_calc_modified_bourgouin_ptype(t[0].astype(float), t[0].astype(float) - 1, t[0].astype(float) - 0.5, h)
    assert cls[0] == ref_cls == 0
    np.testing.assert_allclose(probs[0], ref, atol=1e-7)
    # Deep cold saturated column: prob_ice = 100, snow score 1540 dominates.
    t = (-5.0 - 0.006 * h[np.newaxis]).astype(np.float32)
    probs, cls = legacy_repo_bourgouin(t, t, t)
    assert cls[0] == 1


# ================================================================================================
# Revised (paper) method: independent pure-Python reference (Birk et al. 2021).
# ================================================================================================
G = 9.81
T0 = 273.15


def paper_rh_ice(t, td):
    """RH w.r.t. ice = e_w(Td) / e_i(T) (dewpoint is over water), MetPy saturation pressures."""
    e = saturation_vapor_pressure(np.asarray(td, float) * units.degC, phase="liquid")
    e_si = saturation_vapor_pressure(np.asarray(t, float) * units.degC, phase="solid")
    return (e / e_si).m_as("dimensionless")


def paper_runs(mask):
    """List of (bottom_index, top_index) of consecutive True values."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def paper_prob_ice(t, td, z):
    """Eqs. 2a-c with the generation-layer / dry-layer rules (sect. 3e); returns (ProbIce, layer top)."""
    rh = paper_rh_ice(t, td)
    gen = [r for r in paper_runs(rh > 0.75) if z[r[1]] - z[r[0]] > 1000.0]
    dry = [r for r in paper_runs(rh < 0.75) if z[r[1]] - z[r[0]] > 1500.0]
    valid = [r for r in gen if not any(d[1] < r[0] for d in dry)]  # eliminated if a deep dry layer is below
    if not valid:
        return 0.0, np.nan
    bottom, top = max(valid, key=lambda r: r[1])
    tmin = min(t[bottom : top + 1])
    if tmin <= -15:
        p = 100.0
    elif tmin < -7:
        p = -0.065 * tmin**4 - 3.1544 * tmin**3 - 56.414 * tmin**2 - 449.6 * tmin - 1308
    else:
        p = 0.0
    return p, z[top]


def paper_layers(tw, z):
    """Melting/freezing layers of the Tw profile as a list of [is_warm, energy >= 0] (Eq. 1)."""
    pts = [(z[0], tw[0])]
    for i in range(1, len(tw)):
        a, b = tw[i - 1], tw[i]
        if (a > 0) != (b > 0):
            pts.append((z[i - 1] + (0.0 - a) / (b - a) * (z[i] - z[i - 1]), 0.0))
        pts.append((z[i], b))
    layers = []
    for (za, va), (zb, vb) in itertools.pairwise(pts):
        if zb - za <= 0:
            continue
        warm = va > 0 or vb > 0
        energy = abs(G * (va + vb) / 2.0 / T0 * (zb - za))
        if layers and layers[-1][0] == warm:
            layers[-1][1] += energy
        else:
            layers.append([warm, energy])
    return layers


def paper_energies(layers):
    me_all = sum(e for w, e in layers if w)
    aloft = [i for i, (w, _) in enumerate(layers) if w and i >= 1]
    re_fz = me_pl = re_pl = 0.0
    if aloft:
        cold_below = [i for i in range(aloft[-1]) if not layers[i][0]]
        re_fz = sum(layers[i][1] for i in cold_below)
        if cold_below:
            c = cold_below[0]
            re_pl = layers[c][1]
            me_pl = sum(e for w, e in layers[c + 1 :] if w)
    return me_all, re_fz, me_pl, re_pl


def paper_probabilities(prob_ice, me_all, re_fz, me_pl, re_pl, sfc_warm):
    """Appendix a-c, percent: [rain, snow, icep, frzr]."""
    sn = np.clip(prob_ice / 100 * np.clip(1540 * np.exp(-0.29 * me_all), 0, 100), 0, 100)
    pl = 0.0
    if me_pl > 0 and re_pl > 0:
        pl = np.clip(prob_ice / 100 * np.clip(2.3 * re_pl - 42 * np.log(me_pl + 1) + 3, 0, 100), 0, 100)
    fz_i = np.clip(-2.1 * re_fz + 0.2 * me_all + 458, 0, 100)
    if me_all < 5:
        fz_i = fz_i * 0.2 * me_all
    fz = np.clip((100 - prob_ice) + prob_ice / 100 * fz_i, 0, 100)
    if sfc_warm:
        return [fz, sn, pl, 0.0]
    return [0.0, sn, pl, fz]


def paper_class(p):
    if max(p) <= 0:
        return 254
    return max(i for i in range(4) if p[i] == max(p))  # ties -> later type


def paper_revised_bourgouin(t, td, tw, z):
    prob_ice, top = paper_prob_ice(t, td, z)
    me_all, re_fz, me_pl, re_pl = paper_energies(paper_layers(tw, z))
    p = paper_probabilities(prob_ice, me_all, re_fz, me_pl, re_pl, tw[0] > 0)
    return np.array(p) / 100.0, paper_class(p), np.array([prob_ice, me_all, re_fz, me_pl, re_pl, top])


def scenario(tw, z):
    """Fig. 1 scenario: a all freezing, b surface melting only, c elevated melting + surface freezing,
    d elevated melting + elevated freezing + surface melting."""
    layers = paper_layers(tw, z)
    aloft = any(w for w, _ in layers[1:])
    sfc_warm = layers[0][0]
    if not any(w for w, _ in layers):
        return "a"
    if not aloft:
        return "b"
    return "d" if sfc_warm else "c"


@pytest.fixture(scope="module")
def paper_reference(profiles):
    t, td, tw, _ = profiles
    z = NETWORK_HEIGHTS.astype(float)
    n = t.shape[0]
    probs, cls, diag, scen = np.empty((n, 4)), np.empty(n, int), np.empty((n, 6)), []
    for i in range(n):
        ti, tdi, twi = t[i].astype(float), td[i].astype(float), tw[i].astype(float)
        probs[i], cls[i], diag[i] = paper_revised_bourgouin(ti, tdi, twi, z)
        scen.append(scenario(twi, z))
    return probs, cls, diag, np.array(scen)


def test_paper_coverage(paper_reference):
    """The profiles span the Fig. 1 scenarios and ProbIce 0 / partial / 100."""
    _, cls, diag, scen = paper_reference
    for s in "abcd":
        assert (scen == s).sum() > 200, s
    pi = diag[:, 0]
    assert (pi == 0).sum() > 500 and (pi == 100).sum() > 500 and ((pi > 0) & (pi < 100)).sum() > 200
    assert ((diag[:, 3] > 0) & (diag[:, 4] > 0)).sum() > 1000  # PL computed
    assert set(np.unique(cls)) >= {0, 1, 2, 3}


def test_revised_numba_matches_paper_reference(profiles, paper_reference):
    t, td, tw, _ = profiles
    ref_probs, ref_cls, ref_diag, _ = paper_reference
    probs, cls, diag = revised_bourgouin(t, td, tw, return_diagnostics=True)
    assert probs.dtype == np.float32 and cls.dtype == np.uint8
    assert np.abs(probs - ref_probs).max() < 1e-6
    assert (cls == ref_cls).all()
    np.testing.assert_allclose(diag[:, :5], ref_diag[:, :5], rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(np.isnan(diag[:, 5]), np.isnan(ref_diag[:, 5]))
    assert ((probs >= 0) & (probs <= 1)).all()


def test_revised_from_sounding_matches(profiles):
    t, td, _, p = profiles
    n = 2000
    sounding = np.zeros((n, 5, 21), dtype=np.float32)
    sounding[:, 0], sounding[:, 1], sounding[:, 4] = t[:n], td[:n], p[:n]
    tw = wet_bulb_temperature(t[:n], td[:n], p[:n])
    pa, ca, da = revised_bourgouin(t[:n], td[:n], tw, return_diagnostics=True)
    pb, cb, db, twb = revised_bourgouin_from_sounding(sounding, return_diagnostics=True, return_wet_bulb=True)
    np.testing.assert_array_equal(twb, tw)
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(da, db)


def test_revised_nan_column_flagged(profiles):
    t, td, tw, _ = profiles
    t = t[:3].copy()
    t[1, 5] = np.nan
    probs, cls = revised_bourgouin(t, td[:3], tw[:3])
    assert cls[1] == 255 and np.isnan(probs[1]).all() and cls[0] != 255


# ----- hand-computed cases ----------------------------------------------------------------------
Z = NETWORK_HEIGHTS


def run1(t, td, tw):
    p, c, d = revised_bourgouin(*(np.asarray(a, np.float64)[np.newaxis] for a in (t, td, tw)), return_diagnostics=True)
    return p[0].astype(float), int(c[0]), d[0].astype(float)


def probs_pct(prob_ice, me_all, re_fz, me_pl, re_pl, sfc_warm):
    out = np.empty(4)
    revised_probabilities(prob_ice, me_all, re_fz, me_pl, re_pl, sfc_warm, out)
    return out


def test_all_freezing_saturated_is_snow():
    """ME = 0 and ProbIce = 100 -> SN = 1.0, FZRA = 0 (Fig. 1a)."""
    t = -5.0 - 0.006 * Z  # Tmin = -35 degC
    p, c, d = run1(t, t, t)
    assert d[0] == 100 and d[1] == 0 and d[5] == 5000
    np.testing.assert_allclose(p, [0, 1, 0, 0], atol=0)
    assert c == 1


def test_no_ice_no_melting_is_freezing_drizzle():
    """ME = 0 and ProbIce = 0 (dry column, no generation layer) -> FZRA = 1.0."""
    t = -5.0 - 0.006 * Z
    p, c, d = run1(t, t - 30, t - 2)
    assert d[0] == 0 and np.isnan(d[5]) and d[1] == 0
    np.testing.assert_allclose(p, [0, 0, 0, 1], atol=0)
    assert c == 3


def test_no_ice_surface_above_freezing_is_rain():
    """ProbIce = 0 with surface Tw > 0 -> RA = 1.0, FZRA = 0."""
    t = -5.0 - 0.006 * Z
    tw = t - 2
    tw[0] = 0.5
    p, c, _ = run1(t, t - 30, tw)
    np.testing.assert_allclose(p, [1, 0, 0, 0], atol=0)
    assert c == 0
    np.testing.assert_allclose(probs_pct(0, 0, 0, 0, 0, True), [100, 0, 0, 0])


def test_analytic_energies_and_equations():
    """Constant-Tw layers: Tw = -2 (0-750 m), +3 (1000-2250 m), -4 (2500-5000 m).

    Linear between levels, so the crossings are at 850 m (2/5 of 750->1000) and 2250 + 3/7 * 250 m.
    RE (surface cold layer) = g/T0 (2 * 750 + 2/2 * 100); ME = g/T0 (3/2 * 150 + 3 * 1250 + 3/2 * 750/7);
    the cold air above the warm nose is not beneath a warm layer and is not counted.
    """
    tw = np.where(Z <= 750, -2.0, np.where(Z <= 2250, 3.0, -4.0))
    re = G / T0 * (2 * 750 + 1.0 * 100)
    me = G / T0 * (1.5 * 150 + 3 * 1250 + 1.5 * 750 / 7)
    for t_top, prob_ice in ((-20.0, 100.0), (-10.0, 51.0)):
        t = tw.copy()
        t[Z >= 2500] = t_top  # saturated column (Td = T); Tmin decides ProbIce
        p, _, d = run1(t, t, tw)
        np.testing.assert_allclose(d[:5], [prob_ice, me, re, me, re], rtol=1e-6, atol=0.01)
        pi = d[0]
        sn = pi / 100 * min(1540 * np.exp(-0.29 * me), 100)
        pl = pi / 100 * np.clip(2.3 * re - 42 * np.log(me + 1) + 3, 0, 100)
        fz = min((100 - pi) + pi / 100 * np.clip(-2.1 * re + 0.2 * me + 458, 0, 100), 100)
        np.testing.assert_allclose(p * 100, [0, sn, pl, fz], rtol=1e-5)
    # -10 degC: quartic = -650 + 3154.4 - 5641.4 + 4496 - 1308 = 51.0 (checked above via d[0]).


def test_elevated_freezing_with_surface_melting():
    """Fig. 1d: warm surface, cold layer, warm nose: PL uses the elevated cold layer and the ME above it,
    FZRA becomes RA because the surface Tw > 0, ME_all includes the surface melting layer."""
    tw = np.where(Z <= 250, 1.0, np.where(Z <= 750, -3.0, np.where(Z <= 1750, 2.0, -5.0)))
    t = np.where(Z >= 2000, -20.0, tw)
    p, _, d = run1(t, t, tw)
    z1 = 250 + 1 / 4 * 250  # 1 -> -3 crossing
    z2 = 750 + 3 / 5 * 250  # -3 -> 2
    z3 = 1750 + 2 / 7 * 250  # 2 -> -5
    me_sfc = G / T0 * (1.0 * 250 + 0.5 * (z1 - 250))
    re_el = G / T0 * (1.5 * (500 - z1) + 3 * 250 + 1.5 * (z2 - 750))
    me_aloft = G / T0 * (1.0 * (1000 - z2) + 2 * 750 + 1.0 * (z3 - 1750))
    np.testing.assert_allclose(d[1:5], [me_sfc + me_aloft, re_el, me_aloft, re_el], rtol=1e-6)
    assert p[3] == 0 and p[0] > 0


def test_clipping_and_ramp():
    # ProbSN_i = 1540 clipped to 100, then scaled by ProbIce.
    np.testing.assert_allclose(probs_pct(50, 0, 0, 0, 0, False)[1], 50)
    # ProbPL_i negative -> 0; above 100 -> 100 then x ProbIce/100 (ProbIce applied once).
    assert probs_pct(100, 100, 1, 100, 1, False)[2] == 0
    np.testing.assert_allclose(probs_pct(60, 10, 300, 10, 300, False)[2], 60)
    # ProbPL only when ME (aloft) and RE are nonzero.
    assert probs_pct(100, 10, 300, 0, 300, False)[2] == 0 and probs_pct(100, 10, 0, 10, 0, False)[2] == 0
    # ProbFZRA_i clipped to 100 / 0, then (100 - ProbIce) + ProbIce/100 x ProbFZRA_i.
    np.testing.assert_allclose(probs_pct(100, 50, 0, 0, 0, False)[3], 100)
    np.testing.assert_allclose(probs_pct(100, 10, 300, 10, 300, False)[3], 0)
    np.testing.assert_allclose(probs_pct(40, 10, 300, 10, 300, False)[3], 60)
    # ME < 5 ramp: x 0.2 ME after clipping.
    np.testing.assert_allclose(probs_pct(100, 2.5, 0, 0, 0, False)[3], 100 * 0.5)
    np.testing.assert_allclose(probs_pct(100, 2.5, 200, 0, 0, False)[3], (458 - 420 + 0.5) * 0.5)
    np.testing.assert_allclose(probs_pct(100, 5.0, 0, 0, 0, False)[3], 100)  # no ramp at exactly 5
    # Surface Tw > 0: rain takes the FZRA value.
    np.testing.assert_allclose(probs_pct(100, 50, 0, 0, 0, True), [100, 1540 * np.exp(-0.29 * 50), 0, 0], rtol=1e-12)
    # Ties go to the later type (frzr > icep > snow > rain): ProbIce 100, ME 7 J/kg, no RE gives
    # SN = 100 (1540 e^-2.03 = 202 -> 100) and FZRA = 100 (459.4 -> 100).
    p = probs_pct(100, 7.0, 0, 0, 0, False)
    assert p[1] == 100 and p[3] == 100
    tw = np.where(Z <= 500, -1.0, np.where(Z <= 750, 2.0, -6.0))  # warm nose aloft, ME = 8.2 J/kg
    t = np.where(Z >= 1000, -20.0, tw)
    pp, c, d = run1(t, t, tw)
    assert pp[1] == pp[3] == 1.0 and c == 3, (pp, d)


@pytest.mark.parametrize("me", [5.0, 10.0, 30.0, 100.0, 250.0, 500.0])
def test_eq6_zones(me):
    """Eq. 6 (p. 430): FZRA dominant below 41 + 17.9 ln(ME+1); PL dominant above 170 + 0.08 ME;
    in between both near 100 % (p. 430: "we define the probabilities of FZRA and PL to be both 100%")."""
    lo = 41 + 17.9 * np.log(me + 1)
    hi = 170 + 0.08 * me
    for re in np.linspace(0, lo, 20)[:-1]:
        p = probs_pct(100, me, re, me, re, False)
        assert p[3] > p[2], (re, p)
    for re in np.linspace(lo, hi, 20):
        p = probs_pct(100, me, re, me, re, False)
        assert p[2] >= 90 and p[3] >= 90, (re, p)
    for re in np.linspace(hi, 600, 20)[1:]:
        p = probs_pct(100, me, re, me, re, False)
        assert p[2] >= p[3], (re, p)


# ----- equivalence of the wet-bulb skip ----------------------------------------------------------
DEEP_HEIGHTS = np.arange(0.0, 8001.0, 250.0)  # 33 levels


@pytest.mark.parametrize("heights", [NETWORK_HEIGHTS, DEEP_HEIGHTS], ids=["21lev", "33lev"])
@pytest.mark.parametrize("tsfc_range", [(-14.0, 10.0), (5.0, 30.0)], ids=["cold", "warm"])
def test_wet_bulb_skip_is_exact(heights, tsfc_range):
    """The optimised kernel (wet bulb only where it can matter) equals the full one bit for bit."""
    n = 200_000
    t, td, p = make_profiles(n, seed=123, heights=heights, tsfc_range=tsfc_range)
    snd = np.zeros((n, 5, heights.size), np.float32)
    snd[:, 0], snd[:, 1], snd[:, 4] = t, td, p
    snd[:500, 0] = np.round(snd[:500, 0])  # exact 0 degC levels and T exactly at the skip edge region
    snd[500:600, 0] = -0.5
    pa, ca, da = _revised_bourgouin_from_sounding_full(snd, heights, return_diagnostics=True)
    pb, cb, db = revised_bourgouin_from_sounding(snd, heights, return_diagnostics=True)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(da, db)
    # the skip must still flag non-finite input
    snd[0, 4, -1] = np.nan
    assert revised_bourgouin_from_sounding(snd[:1], heights)[1][0] == 255
