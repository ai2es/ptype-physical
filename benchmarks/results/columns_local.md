# HRRR column kernels (`ptype/hrrr_fast/columns.py`, `ptype/hrrr_fast/bourgouin.py`): local validation and timings

Machine: Apple M4 Pro, 14 cores, 48 GB. Python 3.12.12, numpy 2.5.5, numba 0.68.1 (default threading
layer), MetPy 1.7.1, pyproj 3.8.0, in a private uv venv. **Other jobs were running on the machine**
(load average 14.5-19.5 before and after the benchmark), so the timings are indicative, not clean.
Scripts (not in the repo): `scratchpad/scripts/validate.py` (numbers in section 2) and
`scratchpad/scripts/bench.py` (section 3); tests: `tests/test_hrrr_fast_columns.py`,
`tests/test_hrrr_fast_bourgouin.py` (21 passed).

## 1. How the training data built the 0 m level (evidence from the repository)

| Item | What the code shows |
|---|---|
| Heights AGL | Geopotential height of each isobaric level minus model surface height: `scripts/rap_height_interp.py:144` (`HGT - HGT_ON_SFC`), `scripts/precip_RAP_interpolate.py:58` (`HGT_{p}_m - HGT_ON_SFC_m`). Old inference: `ptype/inference.py:131` (`gh - orog`). Levels are RAP isobaric levels 100-1000 hPa every 25 hPa (`scripts/precip_RAP_interpolate.py:28`, `scripts/rap_height_interp.py:14`), not native levels. Below-ground isobaric levels keep their negative heights. |
| Vertical interpolation | Linear in height. `scripts/precip_RAP_interpolate.py:64-77`: `scipy.interpolate.interp1d(..., fill_value="extrapolate")` (linear **extrapolation** outside the data). `scripts/rap_height_interp.py:51,61`: `np.interp` (**clamping**). Old inference `ptype/inference.py:281`: `np.interp` (clamping). On native HRRR levels (lowest ~10 m AGL, top > 5 km) the 250-5000 m targets are always interior, so clamp vs extrapolate only matters at 0 m, which is overwritten anyway. |
| T at 0 m | 2 m temperature: `scripts/precip_RAP_interpolate.py:80`, `scripts/rap_height_interp.py:89`; old inference `ptype/inference.py:250,264` (`t -> t2m`). |
| Td at 0 m | 2 m dewpoint: `scripts/precip_RAP_interpolate.py:79`; `scripts/rap_height_interp.py:93-94` additionally clips it to `<= T2m` (`np.clip(DEWPOINT_2M, a_max=TEMPERATURE_2M, a_min=False)`); old inference `ptype/inference.py:250,264` (`dpt -> d2m`, no clip). Upper-air Td came from RH via `metpy.calc.dewpoint_from_relative_humidity` (`ptype/precip_RAP_mPING.py:140`, `scripts/rap_height_interp.py:37`), i.e. over liquid water. |
| U, V at 0 m | 10 m wind: `scripts/precip_RAP_interpolate.py:81-82`, `scripts/rap_height_interp.py:90-91`; old inference `ptype/inference.py:250` (`u -> u10`, `v -> v10`) - but the shipped inference configs pass only `ml_atm_varaibles: ['t', 'dpt']` (`config/inference.yml:24`), so there u/v were not interpolated by `convert_and_interpolate` at all. |
| p at 0 m | Surface pressure: `scripts/precip_RAP_interpolate.py:83` (`PRES_ON_SURFACE_Pa`, **in Pa while the other levels of that column group are hPa**, `:28,76`), `scripts/rap_height_interp.py:92`. Old inference: no override (`ptype/inference.py:267-268`, clamped `np.interp` of the level pressures). Pressure is linear in height everywhere. |
| Wind rotation | No rotation code anywhere in the repo (grep for "rotat"). The old inference used the cfgrib HRRR winds as stored (grid-relative). |

Cannot be determined from the repository: which script produced the training file actually used
(`config/ptype.yml:6`, `mPING_interpolated_QC2.parquet`; the name matches the output directory of
`scripts/precip_RAP_interpolate.py`, but the QC step that wrote it is not in the repo); whether the
RAP NCEI netCDF winds (`rap_ncei_nc`) were grid- or earth-relative; whether `PRES_Pa_0_m` was later
converted to hPa; the units of `ds["press"]` in `scripts/rap_height_interp.py` (attribute says Pa).

Defaults chosen in `build_soundings`: T/Td 0 m = 2 m values, U/V 0 m = 10 m wind, p 0 m = surface
pressure (all supported by both training scripts), no Td clip (matches `precip_RAP_interpolate.py`
and the old inference; `clip_surface_dewpoint=True` reproduces `rap_height_interp.py`), pressure
linear in height (training and old code; `pressure_interp="log"` available), no wind rotation
(old inference; `rotation=grid_rotation_cos_sin(lon)` available). Each is a parameter.

## 2. Validation (`validate.py`, random inputs)

| Check | Result |
|---|---|
| Dewpoint from q vs `metpy.calc.dewpoint_from_specific_humidity` (1,000,000 points, q 1e-6..0.03, p 300-1050 hPa, Td -84.8..32.7 °C) | max abs diff 3.8e-06 °C for the float32 output; 2.8e-14 °C for the float64 kernel |
| `interp_to_heights` vs `np.interp` (50,000 columns, 30 random levels, 23 targets incl. below/above range and exact level hits) | 100 % bit-identical (max diff 0) |
| `interp_log_pressure_to_heights` vs `exp(np.interp(z, log p))` | 100 % bit-identical |
| Rotation vs pyproj (LCC lat_0 = lat_1 = lat_2 = 38.5, lon_0 = -97.5; earth components from finite-difference east/north unit vectors in projected coordinates; 20,000 points) | max abs diff 2.5e-06 m/s (u), 1.9e-06 m/s (v); round trip 3.5e-06 m/s |
| Rotation sign | Grid-"north" 10 m/s wind at 80° W, 40° N → earth u = +1.890 m/s (pyproj: +1.890). East of the central meridian grid +y points east of true north, so the earth-relative u is positive. Zero rotation on the central meridian is bit-exact (test). |
| Wet bulb vs `metpy.calc.wet_bulb_temperature` (13,744 points: grid T -40..30 °C × p 1050..450 hPa × T-Td 0..30 K, plus 10,000 random) | max 5.1e-04 °C, mean 3.2e-06 °C, 99th percentile 1.3e-05 °C. The max is at an almost saturated point (T = Td = -37.63 °C to 2 decimals), where MetPy's `moist_lapse` returns the LCL temperature without integrating when p_lcl is `np.isclose` to p. |
| Wet-bulb RK4 step (`_WB_DLNP` in ln p) | 0.02 / 0.05 / 0.1 all give max 8.2e-05 °C vs MetPy on the same 13,744-point set (different random draw); 0.2 gives 1.9e-04. 0.1 vs 0.005: max 1.9e-05 °C on 2,000,000 random points. Chosen 0.1 (1 thread: 727 / 428 / 335 ns per point for 0.02 / 0.05 / 0.1). |
| Modified Bourgouin numba vs corrected reference (50,000 profiles; also 20,000 in the pytest) | max abs probability diff 2.98e-08; class agreement 100.0000 % (0 differ). Reference classes rain/snow/icep/frzr = 17746/24640/0/7614. Coverage (Tw zero crossings, surface Tw > 0 / ≤ 0): 0: 1481/29401; 1: 15379/25; 2: 249/1544; 3+: 1825/96; 384 profiles with exact zeros. |

## 3. Timings on a synthetic full-CONUS problem (`bench.py`)

1059 × 1799 = 1,905,141 columns, 30 native levels (lowest 5-20 m AGL, top ≥ 6136 m), float32
inputs, outputs float32. Best of 3 runs, after JIT warm-up. Threads set with `numba.set_num_threads`.

| Kernel | 1 thread (s) | 14 threads (s) | Speed-up |
|---|---:|---:|---:|
| `dewpoint_from_specific_humidity` (30 levels) | 0.242 | 0.040 | 6.1x |
| `interp_to_heights` (1 field, 21 targets) | 0.360 | 0.049 | 7.4x |
| `interp_log_pressure_to_heights` (21 targets) | 0.494 | 0.068 | 7.2x |
| `rotate_grid_to_earth` (30 levels) | 0.085 | 0.032 | 2.6x |
| `build_soundings` (Td + T, Td, u, v, p interp + 0 m policy) | 0.636 | 0.070 | 9.1x |
| `build_soundings` + wind rotation | 0.604 | 0.080 | 7.6x |
| `network_inputs` ((n, 84) float32 copy) | 0.064 | 0.012 | 5.2x |
| `wet_bulb_temperature` (21 levels, 40 M points) | 10.315 | 1.091 | 9.5x |
| `legacy_repo_bourgouin` (Tw given) | 0.963 | 0.125 | 7.7x |
| `legacy_repo_bourgouin_from_sounding` (incl. wet bulb) | 10.690 | 1.158 | 9.2x |

So sounding preparation for the network (build + 84-column matrix) is ~0.7 s on one thread and
~0.08 s on 14; the Bourgouin diagnostic is dominated by the wet-bulb temperature (~96 % of its time).
A previous run with the RK4 step at 0.02 took 29.1 s (1 thread) / 2.86 s (14 threads) for the wet
bulb. Synthetic class counts (not meaningful climatologically): 914000/847321/0/143820.

Not measured: real HRRR GRIB decode / I/O, memory peak, behaviour on Linux/x86 or other numba
threading layers, and comparison against the old pandas pipeline's wall time.

## 4. Revised Bourgouin: wet-bulb skip (exact optimisation)

`revised_bourgouin_from_sounding` now solves the wet bulb only from the surface up to one level above
the highest level with T > -0.5 °C (or any level where the skip's safety test fails). The other
levels get T (≤ -0.5 °C) as a placeholder. The argument that this cannot change any output is in the
`_tw_levels_needed` docstring:

* Tw ≤ T. Measured on 40 M random float32 inputs: max(Tw − T) = 0, no NaN
  (`scratchpad/scripts/tw_bound.py`).
* Levels above the highest warm level plus one only extend the top freezing layer, whose energy is
  never used.
* A column with every level at T ≤ -0.5 °C needs no wet bulb.

When `return_wet_bulb=True` every level is still solved. The unoptimised path is kept as
`_revised_bourgouin_from_sounding_full` for the tests.

**Equivalence.** `test_wet_bulb_skip_is_exact` runs 200,000 profiles for each case: 21 and 33 levels
(0-8000 m), crossed with a cold mix (surface -14..10 °C) and a warm mix (5..30 °C). That includes
T exactly 0 °C and exactly -0.5 °C. In every case the classes, probabilities and diagnostics are
**bit-identical** (`assert_array_equal`). The benchmark below also compared outputs on all 1,905,141
columns per case: identical in every row.

**Timing** (1,905,141 columns, best of 2, Apple M4 Pro, other jobs running, load 7.4-16):

| Levels | Mix | Threads | Before (s) | After (s) | Speed-up | Tw solved (fraction of levels) |
|---:|---|---:|---:|---:|---:|---:|
| 21 | cold | 1 | 14.679 | 3.412 | 4.3x | 0.183 |
| 21 | cold | 14 | 1.512 | 0.353 | 4.3x | 0.183 |
| 21 | warm | 1 | 14.634 | 10.933 | 1.3x | 0.734 |
| 21 | warm | 14 | 1.443 | 1.118 | 1.3x | 0.734 |
| 33 | cold | 1 | 21.780 | 4.060 | 5.4x | 0.127 |
| 33 | cold | 14 | 2.127 | 0.408 | 5.2x | 0.127 |
| 33 | warm | 1 | 21.859 | 12.694 | 1.7x | 0.535 |
| 33 | warm | 14 | 2.168 | 1.236 | 1.8x | 0.535 |

Profiles come from `make_profiles` in `tests/test_hrrr_fast_bourgouin.py` (synthetic, seed 7). They are not the
`build_soundings` data of section 3, which is why the "before" time here (14.7 s for 21 levels)
differs from the 10.3 s there; I did not investigate the cause. Script:
`scratchpad/scripts/bench_skip.py`.

**Considered and not applied:**

* Re-using `exp(s + h)` and `exp(s + h/2)` between RK4 stages. Bit-identical, but no gain: 354 vs
  356 ns per point, because LLVM already removes the duplicate.
* Fusing `(T0/T)**k * exp(x)` into a single `exp(k ln(T0/T) + x)` in the saturation vapour pressure.
  About 10 % faster (356 → 319 ns per point), but not bit-identical (max |ΔTw| 1.7e-13 °C), and it
  needs a change to `columns.py` that would also alter `wet_bulb_temperature`. Not done.
