"""Fast per-column Bourgouin precipitation type, in numba: the paper method and the repo method.

Two methods are provided, side by side, so that they can be compared on the same columns:

* :func:`revised_bourgouin` / :func:`revised_bourgouin_from_sounding` - the **revised ("Modified")
  Bourgouin method as published** in Birk, Lenning, Donofrio and Friedlein (2021), "A revised
  Bourgouin precipitation-type algorithm", *Weather and Forecasting* 36, 425-438 (sections 3d-3e,
  section 4 and the Appendix).  Four independent probabilities of weather type, each 0-1, no
  normalisation.  Every reading of the paper that is not explicit is listed in
  ``benchmarks/results/bourgouin_questions.md`` and the differences from the repo code in
  ``benchmarks/results/bourgouin_paper_vs_repo.md``.
* :func:`legacy_repo_bourgouin` / :func:`legacy_repo_bourgouin_from_sounding` - a bit-faithful port of
  ``calc_modified_bourgouin_ptype`` in ``ptype/algorithms.py`` (with ``calc_prob_ice``,
  ``get_layer_indices``, ``add_zero_crossings``, ``calc_sounding_stats`` and ``softmax``) after the
  minimum corrections needed for it to run.  Kept only for comparison; it differs from the paper
  (sign of the refreezing energy, ProbIce applied twice to ice pellets, no clipping, softmax, ...).

Both have pure-Python references in ``tests/test_hrrr_fast_bourgouin.py``.

The rest of this docstring describes the legacy (repo) method; the revised method is documented in
the "Revised (paper) method" section further down.

Outline of the method, per column (profiles ordered surface -> top on a fixed height grid):

1. ``prob_ice`` (0-100): scanning from the top of the profile down, find the first layer at least
   ``1000 m`` deep (``ceil(1000/250) + 1 = 5`` consecutive levels) where the "relative humidity with
   respect to ice" exceeds 0.75.  If, below that layer, there is a run of at least
   ``ceil(1500/250) + 1 = 7`` consecutive levels that are not saturated, ``prob_ice = 0``.
   Otherwise ``prob_ice`` depends on the minimum temperature in the saturated layer:
   100 for ``Tmin <= -15 degC``, a quartic in ``Tmin`` for ``-15 < Tmin < -7`` and 0 otherwise.
2. Insert the heights where the wet-bulb profile crosses 0 degC (the original inserts them twice,
   see below), and integrate ``g * Tw / 273.15`` over height between consecutive crossings below
   the highest one.  Negative layer integrals are summed into ``NA`` (cold area, J/kg, <= 0) and
   positive ones into ``PA`` (warm area, J/kg, >= 0).
3. Empirical scores for snow, ice pellets and freezing rain from ``PA``, ``NA`` and ``prob_ice``;
   if the surface wet-bulb is above 0 degC the freezing-rain score is relabelled as rain.
4. ``softmax(scores / 100)`` gives the four "probabilities"; the class is the last index of the max.

Why numba: each column needs a sequence of short, data-dependent loops (layer searches, crossing
insertion, piecewise integration) that cannot be vectorised across columns in numpy without huge
temporaries.  A compiled per-column loop with small per-thread scratch buffers processes ~2 M
columns in a fraction of a second per thread, and ``prange`` over blocks of columns scales with
cores.  The wet-bulb temperature, if not supplied, is computed inside the same loop
(``*_from_sounding``) so that no extra full-size array is written.

Exactness notes:

* Everything is float64 internally (the reference is float64 numpy); outputs are float32 / uint8.
* ``calc_sounding_stats`` calls ``add_zero_crossings`` on a profile that already had its crossings
  inserted by the caller.  The second pass turns each inserted 0 into three zeros at the same
  height.  The extra zero-width segments have zero energy, so the sums are unchanged, but the
  second pass is reproduced literally anyway so that exact-zero inputs behave identically.
* ``cache=True`` caveat: numba's on-disk cache is keyed on *this* file, so after changing a jitted
  helper imported from ``columns.py`` (e.g. the wet-bulb step size) delete the ``__pycache__``
  ``*.nbi``/``*.nbc`` files (or ``NUMBA_CACHE_DIR``) or this module keeps the old machine code.
* The ice "relative humidity" is MetPy's ``relative_humidity_from_dewpoint(T, Td, phase='solid')``,
  i.e. ``e_ice(Td) / e_ice(T)`` with MetPy's saturation formula, as in the original.
"""

import math

import numpy as np
from numba import njit, prange

from ptype.hrrr_fast.columns import NETWORK_HEIGHTS, ZERO_DEGC, _es_ice, _es_liquid, _wet_bulb_scalar

# Constants of the original code (ptype/algorithms.py).
FREEZING_K = 273.15  # algorithms.py:27, divisor that turns Tw (degC) into a fractional anomaly.
GRAVITY = 9.81  # algorithms.py:28, m s-2; g * (Tw / 273.15) * dz has units of J/kg.
SAT_LAYER_THRESH = 1000.0  # algorithms.py:162, minimum depth (m) of the saturated (cloud) layer.
UNSAT_LAYER_THRESH = 1500.0  # algorithms.py:163, depth (m) of a dry sub-cloud layer that kills ice.
RH_ICE_THRESHOLD = 0.75  # algorithms.py:167, "saturated" if RH w.r.t. ice > 0.75 (fraction).

CLASS_NAMES = ("rain", "snow", "icep", "frzr")
INVALID_CLASS = 255  # class written for columns with non-finite input


@njit(cache=True, inline="always")
def _sign(x):
    # np.sign semantics for finite x: -1, 0 or +1.
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@njit(cache=True)
def _add_zero_crossings(prof, hgt, n, prof_out, hgt_out):
    """``add_zero_crossings`` (algorithms.py:5-22) on ``prof[:n]``; returns the new length.

    The original finds every ``k`` with ``np.diff(np.sign(profile))[k] != 0`` (note that 0 has its
    own sign, so touching 0 counts), computes the crossing height with
    ``np.interp(0, xp, fp)`` on the two points (re-ordered so ``xp`` increases) and inserts
    ``(0, crossing)`` before index ``k + 1`` with ``np.insert``.  The ``np.interp`` arithmetic is
    reproduced exactly, including its end-point cases.
    """
    m = 0
    for k in range(n):
        if k >= 1 and _sign(prof[k - 1]) != _sign(prof[k]):
            a = prof[k - 1]
            b = prof[k]
            ha = hgt[k - 1]
            hb = hgt[k]
            if a > b:  # reverse so that xp increases, as in the original
                xp0, xp1, fp0, fp1 = b, a, hb, ha
            else:
                xp0, xp1, fp0, fp1 = a, b, ha, hb
            if 0.0 > xp1:
                cross = fp1
            elif 0.0 < xp0:
                cross = fp0
            elif 0.0 == xp1:
                cross = fp1
            elif 0.0 == xp0:
                cross = fp0
            else:
                slope = (fp1 - fp0) / (xp1 - xp0)
                cross = slope * (0.0 - xp0) + fp0
            prof_out[m] = 0.0
            hgt_out[m] = cross
            m += 1
        prof_out[m] = prof[k]
        hgt_out[m] = hgt[k]
        m += 1
    return m


@njit(cache=True)
def _sounding_areas(prof, hgt, n):
    """Cold (``NA`` <= 0) and warm (``PA`` >= 0) areas, J/kg: ``calc_sounding_stats`` (lines 25-77).

    Integrates ``GRAVITY * (Tw / FREEZING_K)`` with the trapezoid rule from the surface to each
    zero in turn, stopping at the highest zero; each segment's integral goes into the cold sum if it
    is <= 0 and into the warm sum otherwise.  Without any zero both are 0.
    """
    last_zero = -1
    for k in range(n):
        if prof[k] == 0.0:
            last_zero = k
    cold = 0.0
    warm = 0.0
    if last_zero < 0:
        return cold, warm
    low = 0
    for i in range(last_zero + 1):
        if prof[i] == 0.0:
            energy = 0.0
            for m in range(low, i):
                energy += (hgt[m + 1] - hgt[m]) * (GRAVITY * (prof[m + 1] / FREEZING_K) + GRAVITY * (prof[m] / FREEZING_K)) / 2.0
            if energy <= 0.0:
                cold += energy
            else:
                warm += energy
            low = i
    return cold, warm


@njit(cache=True, inline="always")
def _first_run(flags, start, stop, min_length):
    """``get_layer_indices`` (lines 195-211) on ``flags[start:stop]``; returns (top, bottom) or (-1, -1).

    Returns the first run (in array order) of at least ``min_length`` consecutive ones, as indices
    relative to ``start``.
    """
    run_start = -1
    for i in range(stop - start):
        if flags[start + i]:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0 and i - run_start >= min_length:
                return run_start, i - 1
            run_start = -1
    if run_start >= 0 and (stop - start) - run_start >= min_length:
        return run_start, stop - start - 1
    return -1, -1


@njit(cache=True)
def _prob_ice(t, td, n, sat_rev, sat_len, unsat_len):
    """``calc_prob_ice`` (lines 160-193); ``sat_rev`` is a scratch boolean buffer of length >= n.

    The original reverses the profile (``[::-1]``) so that index 0 is the top, finds the first
    (highest) saturated run, then looks for a deep unsaturated run from the bottom of that layer
    down to the surface.  Index ``r`` of the reversed profile is level ``n - 1 - r``.
    """
    for r in range(n):
        lev = n - 1 - r
        tk = t[lev] + ZERO_DEGC
        tdk = td[lev] + ZERO_DEGC
        sat_rev[r] = (_es_ice(tdk) / _es_ice(tk)) > RH_ICE_THRESHOLD
    top, bottom = _first_run(sat_rev, 0, n, sat_len)
    if top < 0:
        return 0.0
    # Unsaturated flags are the negation of saturated ones, searched over sat_rev[bottom:].
    run_start = -1
    found = False
    for i in range(n - bottom):
        if not sat_rev[bottom + i]:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0 and i - run_start >= unsat_len:
                found = True
                break
            run_start = -1
    if not found and run_start >= 0 and (n - bottom) - run_start >= unsat_len:
        found = True
    if found:
        return 0.0
    tmin = np.inf
    for r in range(top, bottom + 1):
        tmin = min(tmin, t[n - 1 - r])
    if tmin <= -15.0:  # original: ``<+ -15``, read as a typo for ``<=`` (see bourgouin_questions.md)
        return 100.0
    elif tmin > -15.0 and tmin < -7.0:
        return (-0.065 * math.pow(tmin, 4.0)) - (3.1544 * math.pow(tmin, 3.0)) - (56.414 * math.pow(tmin, 2.0)) - (449.6 * tmin) - 1308.0
    return 0.0


@njit(cache=True)
def _modified_bourgouin_column(t, td, tw, hgt, n, sat_len, unsat_len, sat_rev, p1, h1, p2, h2, probs):
    """One column of ``calc_modified_bourgouin_ptype`` (lines 130-157); returns the class index."""
    prob_ice = _prob_ice(t, td, n, sat_rev, sat_len, unsat_len)
    # The caller adds the crossings, and calc_sounding_stats adds them again (literal reproduction).
    n1 = _add_zero_crossings(tw, hgt, n, p1, h1)
    n2 = _add_zero_crossings(p1, h1, n1, p2, h2)
    na, pa = _sounding_areas(p2, h2, n2)
    sfc = p2[0]
    prob_snow_i = 1540.0 * math.exp(-(0.29 * pa))
    prob_snow = (prob_ice / 100.0) * prob_snow_i
    if pa > 0.0:
        prob_icep_i = ((2.3 * na) - (42.0 * math.log(pa + 1.0)) + 3.0) * prob_ice
    else:
        prob_icep_i = 0.0
    prob_icep = (prob_ice / 100.0) * prob_icep_i
    prob_frzr_i = (-2.1 * na) + (0.2 * pa) + 458.0
    if pa < 5.0:
        prob_frzr_i = prob_frzr_i * 0.2 * pa
    prob_frzr = (100.0 - prob_ice) + ((prob_ice / 100.0) * prob_frzr_i)
    if sfc > 0.0:
        prob_rain = prob_frzr
        prob_frzr = 0.0
    else:
        prob_rain = 0.0
    # softmax(x / 100): exp(x - max(x)) / sum
    x0 = prob_rain / 100.0
    x1 = prob_snow / 100.0
    x2 = prob_icep / 100.0
    x3 = prob_frzr / 100.0
    xmax = max(max(x0, x1), max(x2, x3))
    e0 = math.exp(x0 - xmax)
    e1 = math.exp(x1 - xmax)
    e2 = math.exp(x2 - xmax)
    e3 = math.exp(x3 - xmax)
    total = e0 + e1 + e2 + e3
    probs[0] = e0 / total
    probs[1] = e1 / total
    probs[2] = e2 / total
    probs[3] = e3 / total
    best = 0
    for k in range(1, 4):  # last index of the maximum, as np.argwhere(p == max)[-1]
        if probs[k] >= probs[best]:
            best = k
    return best


_BLOCK = 2048  # columns per parallel work item; scratch buffers are allocated once per block


@njit(parallel=True, cache=True)
def _bourgouin_kernel(t, td, tw, hgt, sat_len, unsat_len, probs_out, cls_out):
    ncol, n = t.shape
    nblocks = (ncol + _BLOCK - 1) // _BLOCK
    for b in prange(nblocks):
        sat_rev = np.empty(n, dtype=np.bool_)
        p1 = np.empty(2 * n, dtype=np.float64)
        h1 = np.empty(2 * n, dtype=np.float64)
        p2 = np.empty(4 * n, dtype=np.float64)
        h2 = np.empty(4 * n, dtype=np.float64)
        tt = np.empty(n, dtype=np.float64)
        dd = np.empty(n, dtype=np.float64)
        ww = np.empty(n, dtype=np.float64)
        probs = np.empty(4, dtype=np.float64)
        for c in range(b * _BLOCK, min(ncol, (b + 1) * _BLOCK)):
            ok = True
            for k in range(n):
                tt[k] = t[c, k]
                dd[k] = td[c, k]
                ww[k] = tw[c, k]
                if not (np.isfinite(tt[k]) and np.isfinite(dd[k]) and np.isfinite(ww[k])):
                    ok = False
            if not ok:
                for m in range(4):
                    probs_out[c, m] = np.nan
                cls_out[c] = INVALID_CLASS
                continue
            cls_out[c] = _modified_bourgouin_column(tt, dd, ww, hgt, n, sat_len, unsat_len, sat_rev, p1, h1, p2, h2, probs)
            for m in range(4):
                probs_out[c, m] = probs[m]


@njit(parallel=True, cache=True)
def _bourgouin_sounding_kernel(sounding, hgt, sat_len, unsat_len, probs_out, cls_out, tw_out, store_tw):
    ncol = sounding.shape[0]
    n = sounding.shape[2]
    nblocks = (ncol + _BLOCK - 1) // _BLOCK
    for b in prange(nblocks):
        sat_rev = np.empty(n, dtype=np.bool_)
        p1 = np.empty(2 * n, dtype=np.float64)
        h1 = np.empty(2 * n, dtype=np.float64)
        p2 = np.empty(4 * n, dtype=np.float64)
        h2 = np.empty(4 * n, dtype=np.float64)
        tt = np.empty(n, dtype=np.float64)
        dd = np.empty(n, dtype=np.float64)
        ww = np.empty(n, dtype=np.float64)
        probs = np.empty(4, dtype=np.float64)
        for c in range(b * _BLOCK, min(ncol, (b + 1) * _BLOCK)):
            ok = True
            for k in range(n):
                tt[k] = sounding[c, 0, k]
                dd[k] = sounding[c, 1, k]
                # Round Tw to float32 so the result is identical to computing a float32 Tw array
                # first and passing it to legacy_repo_bourgouin().
                ww[k] = np.float32(_wet_bulb_scalar(tt[k], dd[k], np.float64(sounding[c, 4, k]) * 100.0))
                if store_tw:
                    tw_out[c, k] = ww[k]
                if not (np.isfinite(tt[k]) and np.isfinite(dd[k]) and np.isfinite(ww[k])):
                    ok = False
            if not ok:
                for m in range(4):
                    probs_out[c, m] = np.nan
                cls_out[c] = INVALID_CLASS
                continue
            cls_out[c] = _modified_bourgouin_column(tt, dd, ww, hgt, n, sat_len, unsat_len, sat_rev, p1, h1, p2, h2, probs)
            for m in range(4):
                probs_out[c, m] = probs[m]


def _layer_lengths(heights):
    """Minimum run lengths (in levels) as in calc_prob_ice: ``ceil(thresh / resolution) + 1``.

    The original passes ``height_resolution=250`` (its default) rather than deriving it from the
    profile; here it is derived from the (assumed uniform) height grid, which gives the same
    numbers (5 and 7) for the 0..5000 m by 250 m network grid.
    """
    dz = np.diff(heights)
    if not np.allclose(dz, dz[0]):
        raise ValueError("heights must be uniformly spaced (calc_prob_ice assumes a fixed resolution)")
    res = float(dz[0])
    return float(np.ceil(SAT_LAYER_THRESH / res) + 1), float(np.ceil(UNSAT_LAYER_THRESH / res) + 1)


def legacy_repo_bourgouin(t_c, td_c, tw_c, heights=NETWORK_HEIGHTS):
    """Repo (legacy) modified-Bourgouin probabilities for many columns (softmax output, see module doc).

    Args:
        t_c: Temperature (degC), shape ``(n_columns, n_levels)``, levels surface -> top.  Strided
            views (e.g. ``sounding[:, 0, :]``) are fine.
        td_c: Dewpoint (degC), same shape.
        tw_c: Wet-bulb temperature (degC), same shape.
        heights: Heights AGL (m) of the levels, uniformly spaced; default 0..5000 by 250.

    Returns:
        ``(probs, cls)``: float32 ``(n_columns, 4)`` probabilities in the order rain, snow, icep,
        frzr, and uint8 ``(n_columns,)`` class index (255 for columns with non-finite input).
    """
    t_c = np.asarray(t_c)
    td_c = np.asarray(td_c)
    tw_c = np.asarray(tw_c)
    hgt = np.ascontiguousarray(heights, dtype=np.float64)
    if t_c.ndim != 2 or t_c.shape != td_c.shape or t_c.shape != tw_c.shape or t_c.shape[1] != hgt.size:
        raise ValueError("t_c, td_c and tw_c must be (n_columns, n_levels) with n_levels == len(heights)")
    sat_len, unsat_len = _layer_lengths(hgt)
    probs = np.empty((t_c.shape[0], 4), dtype=np.float32)
    cls = np.empty(t_c.shape[0], dtype=np.uint8)
    _bourgouin_kernel(t_c, td_c, tw_c, hgt, sat_len, unsat_len, probs, cls)
    return probs, cls


def legacy_repo_bourgouin_from_sounding(sounding, heights=NETWORK_HEIGHTS, return_wet_bulb=False):
    """Repo (legacy) modified Bourgouin directly from a ``(n_columns, 5, n_levels)`` sounding array.

    The wet-bulb temperature is computed per level inside the same parallel loop
    (:func:`ptype.hrrr_fast.columns.wet_bulb_temperature`'s scalar kernel, from T, Td and p), so no
    full-size wet-bulb array has to be written unless ``return_wet_bulb`` is True.

    Args:
        sounding: float32 ``(n_columns, 5, n_levels)`` from
            :func:`ptype.hrrr_fast.columns.build_soundings` (T degC, Td degC, u, v, p hPa).
        heights: Heights AGL (m) of the levels; default 0..5000 by 250.
        return_wet_bulb: Also return the float32 ``(n_columns, n_levels)`` wet-bulb array.

    Returns:
        ``(probs, cls)`` or ``(probs, cls, tw)``; see :func:`legacy_repo_bourgouin`.
    """
    hgt = np.ascontiguousarray(heights, dtype=np.float64)
    if sounding.ndim != 3 or sounding.shape[1] != 5 or sounding.shape[2] != hgt.size:
        raise ValueError("sounding must be (n_columns, 5, n_levels) with n_levels == len(heights)")
    sat_len, unsat_len = _layer_lengths(hgt)
    ncol = sounding.shape[0]
    probs = np.empty((ncol, 4), dtype=np.float32)
    cls = np.empty(ncol, dtype=np.uint8)
    tw = np.empty((ncol, hgt.size) if return_wet_bulb else (1, 1), dtype=np.float32)
    _bourgouin_sounding_kernel(sounding, hgt, sat_len, unsat_len, probs, cls, tw, return_wet_bulb)
    if return_wet_bulb:
        return probs, cls, tw
    return probs, cls


# =============================================================================================
# Revised (paper) method: Birk et al. (2021), Wea. Forecasting 36, 425-438.
# =============================================================================================
#
# Specification (paper page numbers):
#
# * Energy (Eq. 1, p. 427, applied to the wet-bulb profile per section 3d, p. 428):
#   ``area = integral of g (Tw_env - T0) / T0 dz`` with ``T0 = 273.15 K``; melting energy (ME) and
#   refreezing energy (RE) are both expressed as POSITIVE numbers (p. 427).  ``Tw_env - T0`` is the
#   wet-bulb temperature in degC.  Tw is linear in height between levels, so each layer is split
#   exactly at the 0 degC crossing and integrated with the trapezoid rule (exact for linear pieces).
#   g = 9.81 m s-2 as in the repo (the paper does not give a value; 9.80665 would change areas by
#   0.03 %).
# * ProbIce (Eqs. 2a-c, p. 428): a precipitation-generation layer is saturated, i.e. RH w.r.t. ice
#   > 75 %, over more than 1 km; ProbIce from the MIN temperature in it: <= -15 degC -> 100,
#   -15 < T < -7 -> quartic, >= -7 -> 0.  "If a precipitation generation layer exists above a dry
#   layer with RH_ice < 75 % for more than 1500 m, sublimation is assumed, and this upper layer is
#   eliminated from consideration."
# * Appendix (pp. 436-437): ProbSN, ProbPL (only when ME is nonzero), ProbFZRA / ProbRA, each
#   intermediate and final value constrained to 0-100.  Section 3b (p. 428) and section 5 (p. 434):
#   the four probabilities are independent (a 100 %/100 % mix is allowed); no normalisation.
#
# Readings chosen where the paper is not explicit (all listed in bourgouin_questions.md):
#
# * RH_ice = e_w(Td) / e_i(T) (Td is a dew point over water; MetPy 1.7 saturation formulas).
# * Layer depth = height of the top saturated (dry) level minus height of the bottom one, compared
#   STRICTLY (> 1000 m, > 1500 m: "exceed 1 km", "more than 1500 m").  On the 250 m grid that is
#   >= 6 and >= 8 consecutive levels (the repo used >= 5 and >= 7).
# * Saturated: RH_ice > 0.75; dry: RH_ice < 0.75 (both strict, as worded).
# * Elimination: a generation layer is eliminated when a deep dry layer lies anywhere below it (down
#   to the surface); the HIGHEST remaining generation layer gives ProbIce (top-down, as the
#   traditional method).  No remaining layer -> ProbIce = 0 (and the diagnostic layer top is NaN).
# * Tmin is the minimum of the (dry-bulb) temperature over the levels of that layer.
# * Layers of the Tw profile: maximal height intervals where Tw > 0 (melting) or Tw <= 0 (freezing),
#   zero-length pieces (a level exactly at 0 degC) are dropped.  Layer 0 is surface based; a warm
#   layer with index >= 1 is a "warm layer aloft".
# * ME for snow (App. a): all melting energy in the 0-5 km column, surface-based layer included.
# * RE for FZRA (App. c): refreezing energy "beneath any warm layers aloft" = sum of all freezing
#   layers below the highest warm layer aloft (0 if there is no warm layer aloft).
# * ME for FZRA/RA (App. c, unqualified "the wet-bulb melting energy"): the same all-column ME as for
#   snow (so a surface-based melting layer alone gives RA; with ME aloft only, Fig. 1b soundings
#   would get ProbRA = 100 - ProbIce).  Also used for the ME < 5 J/kg ramp.
# * PL (App. b): the "near-surface cold layer" is the LOWEST freezing layer that has a warm layer
#   aloft above it (surface based in Fig. 1c, elevated in Fig. 1d); RE_PL is that layer's energy and
#   ME_PL is all melting energy above it.  ProbPL is computed only when ME_PL > 0 and RE_PL > 0,
#   otherwise ProbPL = 0.
# * Surface Tw > 0 degC (strict) -> ProbRA = ProbFZRA and ProbFZRA = 0.
# * Dominant class: argmax with ties going to the LAST of (rain, snow, icep, frzr), as the repo's
#   ``argwhere(p == max)[-1]``; if all four are 0 the class is ``NO_TYPE_CLASS`` (254).

T0_K = 273.15  # Eq. 1 reference temperature, K
RH_ICE_SATURATED = 0.75  # Sect. 3e: RH_ice > 75 % (fraction)
GENERATION_LAYER_DEPTH = 1000.0  # Sect. 3e: "exceed 1 km in depth", m
DRY_LAYER_DEPTH = 1500.0  # Sect. 3e: dry layer "for more than 1500 m", m
FZRA_RAMP_ME = 5.0  # App. c step 2: ramp below 5 J/kg
NO_TYPE_CLASS = 254  # all four probabilities are zero
N_DIAGNOSTICS = 6
DIAGNOSTIC_NAMES = ("prob_ice_percent", "me_all_jkg", "re_fzra_jkg", "me_pl_jkg", "re_pl_jkg", "generation_layer_top_m")


@njit(cache=True, inline="always")
def _clip100(x):
    return min(max(x, 0.0), 100.0)


@njit(cache=True)
def _revised_prob_ice(t, td, hgt, n, gen_depth, dry_depth, sat, dry, diag):
    """ProbIce (percent) per Eqs. 2a-c with the generation/dry layer rules; writes diag[5] = layer top."""
    for k in range(n):
        rh_ice = _es_liquid(td[k] + ZERO_DEGC) / _es_ice(t[k] + ZERO_DEGC)
        sat[k] = rh_ice > RH_ICE_SATURATED
        dry[k] = rh_ice < RH_ICE_SATURATED
    # Lowest deep dry layer (bottom index); generation layers above it are eliminated.
    low_dry_bottom = n
    k = 0
    while k < n:
        if dry[k]:
            s = k
            while k + 1 < n and dry[k + 1]:
                k += 1
            if hgt[k] - hgt[s] > dry_depth:
                low_dry_bottom = s
                break
        k += 1
    # Highest generation layer lying entirely below that dry layer.
    best_b = -1
    best_t = -1
    k = 0
    while k < n:
        if sat[k]:
            s = k
            while k + 1 < n and sat[k + 1]:
                k += 1
            if hgt[k] - hgt[s] > gen_depth and k < low_dry_bottom:
                best_b = s
                best_t = k
        k += 1
    if best_b < 0:
        diag[5] = np.nan
        return 0.0
    diag[5] = hgt[best_t]
    tmin = np.inf
    for k in range(best_b, best_t + 1):
        tmin = min(tmin, t[k])
    if tmin <= -15.0:
        return 100.0
    if tmin < -7.0:
        return -0.065 * tmin**4 - 3.1544 * tmin**3 - 56.414 * tmin**2 - 449.6 * tmin - 1308.0
    return 0.0


@njit(cache=True)
def _tw_layers(tw, hgt, n, warm, energy):
    """Split the Tw profile into alternating melting/freezing layers; returns the number of layers.

    ``warm[i]`` is True for a melting layer (Tw > 0) and ``energy[i]`` is the layer's |Eq. 1| area in
    J/kg.  Each piece between two levels is linear in height; a piece that crosses 0 degC is split at
    the crossing ``zc = z0 - a (z1 - z0) / (b - a)`` and each part is a triangle.
    """
    nl = 0
    for i in range(n - 1):
        a = tw[i]
        b = tw[i + 1]
        z0 = hgt[i]
        z1 = hgt[i + 1]
        wa = a > 0.0
        wb = b > 0.0
        for part in range(2):
            if wa == wb:
                if part == 1:
                    break
                kind = wa
                length = z1 - z0
                e = GRAVITY * 0.5 * (a + b) / T0_K * length
            else:
                zc = z0 - a * (z1 - z0) / (b - a)
                if part == 0:
                    kind = wa
                    length = zc - z0
                    e = GRAVITY * 0.5 * a / T0_K * length
                else:
                    kind = wb
                    length = z1 - zc
                    e = GRAVITY * 0.5 * b / T0_K * length
            if length <= 0.0:
                continue
            if nl > 0 and warm[nl - 1] == kind:
                energy[nl - 1] += abs(e)
            else:
                warm[nl] = kind
                energy[nl] = abs(e)
                nl += 1
    return nl


@njit(cache=True)
def _layer_energies(warm, energy, nl, diag):
    """ME_all, RE_FZRA, ME_PL, RE_PL (all >= 0, J/kg) from the layer list, into diag[1:5]."""
    me_all = 0.0
    highest_warm_aloft = -1
    for i in range(nl):
        if warm[i]:
            me_all += energy[i]
            if i >= 1:
                highest_warm_aloft = i
    re_fz = 0.0
    me_pl = 0.0
    re_pl = 0.0
    if highest_warm_aloft >= 1:
        near_sfc_cold = -1
        for i in range(highest_warm_aloft):
            if not warm[i]:
                re_fz += energy[i]
                if near_sfc_cold < 0:
                    near_sfc_cold = i
        if near_sfc_cold >= 0:
            re_pl = energy[near_sfc_cold]
            for i in range(near_sfc_cold + 1, nl):
                if warm[i]:
                    me_pl += energy[i]
    diag[1] = me_all
    diag[2] = re_fz
    diag[3] = me_pl
    diag[4] = re_pl


@njit(cache=True)
def revised_probabilities(prob_ice, me_all, re_fz, me_pl, re_pl, sfc_warm, out):
    """Appendix a-c of Birk et al. (2021): percent probabilities (rain, snow, icep, frzr) into ``out``.

    Args:
        prob_ice: ProbIce, percent (Eq. 2).
        me_all: All wet-bulb melting energy in the column, J/kg (App. a; also App. c).
        re_fz: Refreezing energy beneath any warm layers aloft, J/kg (App. c).
        me_pl: Melting energy above the near-surface cold layer, J/kg (App. b).
        re_pl: Refreezing energy of that near-surface cold layer, J/kg (App. b).
        sfc_warm: True when the surface wet-bulb temperature is above 0 degC.
        out: float64 array of length 4.
    """
    # a. ProbSN
    sn_i = _clip100(1540.0 * math.exp(-0.29 * me_all))
    sn = _clip100(prob_ice / 100.0 * sn_i)
    # b. ProbPL, only when ME (above a near-surface cold layer with nonzero RE) is nonzero
    if me_pl > 0.0 and re_pl > 0.0:
        pl_i = _clip100(2.3 * re_pl - 42.0 * math.log(me_pl + 1.0) + 3.0)
        pl = _clip100(prob_ice / 100.0 * pl_i)
    else:
        pl = 0.0
    # c. ProbFZRA / ProbRA
    fz_i = _clip100(-2.1 * re_fz + 0.2 * me_all + 458.0)
    if me_all < FZRA_RAMP_ME:
        fz_i = fz_i * 0.2 * me_all
    fz = _clip100((100.0 - prob_ice) + prob_ice / 100.0 * fz_i)
    if sfc_warm:
        ra = fz
        fz = 0.0
    else:
        ra = 0.0
    out[0] = ra
    out[1] = sn
    out[2] = pl
    out[3] = fz


@njit(cache=True)
def _revised_column(t, td, tw, hgt, n, gen_depth, dry_depth, sat, dry, warm, energy, probs, diag):
    prob_ice = _revised_prob_ice(t, td, hgt, n, gen_depth, dry_depth, sat, dry, diag)
    diag[0] = prob_ice
    nl = _tw_layers(tw, hgt, n, warm, energy)
    _layer_energies(warm, energy, nl, diag)
    revised_probabilities(prob_ice, diag[1], diag[2], diag[3], diag[4], tw[0] > 0.0, probs)
    best = 0
    for k in range(1, 4):
        if probs[k] >= probs[best]:
            best = k
    if probs[best] <= 0.0:
        return NO_TYPE_CLASS
    return best


@njit(parallel=True, cache=True)
def _revised_kernel(t, td, tw, sounding, use_sounding, hgt, gen_depth, dry_depth, probs_out, cls_out, diag_out, store_diag, tw_out, store_tw):
    if use_sounding:
        ncol = sounding.shape[0]
    else:
        ncol = t.shape[0]
    n = hgt.shape[0]
    nblocks = (ncol + _BLOCK - 1) // _BLOCK
    for b in prange(nblocks):
        sat = np.empty(n, dtype=np.bool_)
        dry = np.empty(n, dtype=np.bool_)
        warm = np.empty(2 * n, dtype=np.bool_)
        energy = np.empty(2 * n, dtype=np.float64)
        tt = np.empty(n, dtype=np.float64)
        dd = np.empty(n, dtype=np.float64)
        ww = np.empty(n, dtype=np.float64)
        probs = np.empty(4, dtype=np.float64)
        diag = np.empty(N_DIAGNOSTICS, dtype=np.float64)
        for c in range(b * _BLOCK, min(ncol, (b + 1) * _BLOCK)):
            ok = True
            for k in range(n):
                if use_sounding:
                    tt[k] = sounding[c, 0, k]
                    dd[k] = sounding[c, 1, k]
                    # float32 rounding: identical to passing a float32 Tw array to revised_bourgouin
                    ww[k] = np.float32(_wet_bulb_scalar(tt[k], dd[k], np.float64(sounding[c, 4, k]) * 100.0))
                    if store_tw:
                        tw_out[c, k] = ww[k]
                else:
                    tt[k] = t[c, k]
                    dd[k] = td[c, k]
                    ww[k] = tw[c, k]
                if not (np.isfinite(tt[k]) and np.isfinite(dd[k]) and np.isfinite(ww[k])):
                    ok = False
            if not ok:
                for m in range(4):
                    probs_out[c, m] = np.nan
                if store_diag:
                    for m in range(N_DIAGNOSTICS):
                        diag_out[c, m] = np.nan
                cls_out[c] = INVALID_CLASS
                continue
            cls_out[c] = _revised_column(tt, dd, ww, hgt, n, gen_depth, dry_depth, sat, dry, warm, energy, probs, diag)
            for m in range(4):
                probs_out[c, m] = probs[m] / 100.0
            if store_diag:
                for m in range(N_DIAGNOSTICS):
                    diag_out[c, m] = diag[m]


def _revised_run(t, td, tw, sounding, hgt, return_diagnostics, return_wet_bulb):
    use_sounding = sounding is not None
    hgt = np.ascontiguousarray(hgt, dtype=np.float64)
    if hgt.ndim != 1 or hgt.size < 2 or np.any(np.diff(hgt) <= 0):
        raise ValueError("heights must be a strictly increasing 1-D array")
    dummy2 = np.zeros((1, 1), np.float32)
    dummy3 = np.zeros((1, 1, 1), np.float32)
    if use_sounding:
        if sounding.ndim != 3 or sounding.shape[1] != 5 or sounding.shape[2] != hgt.size:
            raise ValueError("sounding must be (n_columns, 5, n_levels) with n_levels == len(heights)")
        ncol = sounding.shape[0]
        t = td = tw = dummy2
    else:
        t, td, tw = np.asarray(t), np.asarray(td), np.asarray(tw)
        if t.ndim != 2 or t.shape != td.shape or t.shape != tw.shape or t.shape[1] != hgt.size:
            raise ValueError("t_c, td_c and tw_c must be (n_columns, n_levels) with n_levels == len(heights)")
        if not (t.dtype == td.dtype == tw.dtype):
            raise ValueError("t_c, td_c and tw_c must have the same dtype")
        ncol = t.shape[0]
        sounding = dummy3
    probs = np.empty((ncol, 4), dtype=np.float32)
    cls = np.empty(ncol, dtype=np.uint8)
    diag = np.empty((ncol, N_DIAGNOSTICS) if return_diagnostics else (1, 1), dtype=np.float32)
    tw_out = np.empty((ncol, hgt.size) if return_wet_bulb else (1, 1), dtype=np.float32)
    _revised_kernel(t, td, tw, sounding, use_sounding, hgt, GENERATION_LAYER_DEPTH, DRY_LAYER_DEPTH, probs, cls, diag, return_diagnostics, tw_out, return_wet_bulb)
    out = [probs, cls]
    if return_diagnostics:
        out.append(diag)
    if return_wet_bulb:
        out.append(tw_out)
    return tuple(out)


def revised_bourgouin(t_c, td_c, tw_c, heights=NETWORK_HEIGHTS, return_diagnostics=False):
    """Revised Bourgouin precipitation-type probabilities (Birk et al. 2021) for many columns.

    Args:
        t_c: Temperature (degC), ``(n_columns, n_levels)``, levels surface -> top (strided views OK).
        td_c: Dewpoint over water (degC), same shape and dtype.
        tw_c: Wet-bulb temperature (degC), same shape and dtype.
        heights: Heights AGL (m) of the levels, strictly increasing; default 0..5000 by 250.
        return_diagnostics: Also return the float32 ``(n_columns, 6)`` diagnostics
            :data:`DIAGNOSTIC_NAMES`: ProbIce (percent), ME_all, RE_FZRA, ME_PL, RE_PL (J/kg, >= 0)
            and the height of the top of the generation layer used for ProbIce (NaN if none).

    Returns:
        ``(probs, cls)`` or ``(probs, cls, diagnostics)``.  ``probs`` is float32 ``(n_columns, 4)``
        with the paper's percent probabilities / 100 in the order rain, snow, icep, frzr; they are
        independent (not normalised; a mix can have several near 1).  ``cls`` is uint8: the index of
        the largest probability, ties to the later type (frzr > icep > snow > rain), 254 when all four
        are 0 and 255 for columns with non-finite input.
    """
    return _revised_run(t_c, td_c, tw_c, None, heights, return_diagnostics, False)


def revised_bourgouin_from_sounding(sounding, heights=NETWORK_HEIGHTS, return_diagnostics=False, return_wet_bulb=False):
    """Revised Bourgouin directly from a ``(n_columns, 5, n_levels)`` sounding (T, Td, u, v, p hPa).

    The wet-bulb temperature is computed per level inside the same parallel loop (rounded to float32,
    so the result equals :func:`revised_bourgouin` given ``columns.wet_bulb_temperature`` output).

    Args:
        sounding: float32 ``(n_columns, 5, n_levels)`` from :func:`ptype.hrrr_fast.columns.build_soundings`.
        heights: Heights AGL (m) of the levels; default 0..5000 by 250.
        return_diagnostics: Also return the ``(n_columns, 6)`` diagnostics (see :func:`revised_bourgouin`).
        return_wet_bulb: Also return the float32 ``(n_columns, n_levels)`` wet-bulb array (last).

    Returns:
        ``(probs, cls[, diagnostics][, tw])``; see :func:`revised_bourgouin`.
    """
    return _revised_run(None, None, None, sounding, heights, return_diagnostics, return_wet_bulb)
