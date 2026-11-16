# Revised Bourgouin: paper (Birk et al. 2021) vs repo (`ptype/algorithms.py`)

Paper: Birk, Lenning, Donofrio and Friedlein (2021), *Wea. Forecasting* 36, 425-438; equation and page
numbers below refer to it. Repo: `calc_modified_bourgouin_ptype` and helpers in `ptype/algorithms.py`
(unchanged). Implementations: `revised_bourgouin*` (paper) and `legacy_repo_bourgouin*` (repo, after the
minimal run-fixes A1-A3 of `bourgouin_questions.md`) in `ptype/hrrr_fast/bourgouin.py`. Where the paper
leaves room for interpretation, the reading used is listed in `bourgouin_questions.md`.

## 1. Differences

| # | Topic | Paper | Repo | Implemented in `revised_bourgouin` |
|---|---|---|---|---|
| D1 | Sign of refreezing energy | ME and RE are both positive (p. 427, after Eq. 1) | Cold-layer integrals are negative and summed into `NA <= 0` (lines 58-60) and used with that sign: `2.3 * NA` (line 139) and `-2.1 * NA` (line 143), so more refreezing lowers PL and raises FZRA | RE >= 0 |
| D2 | ProbIce on PL | Applied once: `ProbPL = (ProbIce/100) ProbPL_i` (App. b step 2; p. 432) | Applied twice: `(... ) * prob_ice` (line 139) and then `(prob_ice / 100) *` (line 142) | Once |
| D3 | Clipping | Every intermediate and final value is constrained to 0-100 (App. a-c; text after Eqs. 7, 8, 9) | No clipping (lines 136-146): snow score up to 1540, PL down to -1.5e5 on synthetic profiles | Clipped at each step, as in the Appendix |
| D4 | Output | Four independent probabilities of weather type; mixes allowed, "a 50%/50% mix is equivalent to a 100%/100% mix" (sect. 3b p. 428; p. 434) | `softmax(scores / 100)` (line 154): normalised, every type always > 0 | No normalisation, no softmax; fractions = percent / 100 |
| D5 | Dry layer under a generation layer | "sublimation is assumed, and this upper layer is eliminated from consideration" (sect. 3e, p. 428), i.e. other generation layers can still give ProbIce | Returns `prob_ice = 0` as soon as the highest saturated layer has a deep unsaturated run below it (lines 174-176) | Upper layer eliminated; highest remaining layer used |
| D6 | Which generation layer | "layers ... when they exceed 1 km in depth" (p. 428), not specified further | Only the highest saturated run is examined (lines 165-178) | Highest non-eliminated layer (see Q3) |
| D7 | RH w.r.t. ice | RH_ice > 75 % (p. 428); with a dewpoint over water the physical value is `e_w(Td)/e_i(T)` | MetPy `relative_humidity_from_dewpoint(..., phase='solid')` = `e_i(Td)/e_i(T)` (line 165), which is lower than true RH_ice below 0 °C (1.000 vs 1.216 at saturation, -20 °C) | `e_w(Td)/e_i(T)`, MetPy 1.7 saturation formulas |
| D8 | Layer depth thresholds | "exceed 1 km", dry "for more than 1500 m" (p. 428): strict | `ceil(thresh / 250) + 1` levels with `>=` (lines 170, 174): 5 levels (= 1000 m) and 7 levels (= 1500 m) | Strict on the level-height span: >= 6 and >= 8 levels on the 250 m grid (Q2) |
| D9 | Saturated vs dry | Saturated: RH_ice > 75 %; dry: RH_ice < 75 % (p. 428) | Unsaturated = not saturated (`<= 0.75`, line 168) | `> 0.75` and `< 0.75` |
| D10 | ProbIce at -15 °C | `T <= -15` -> 100 (Eq. 2a) | `min_layer_temp <+ -15` = `< -15` (line 180); -15.0 exactly gives 0 | `<= -15` |
| D11 | ME for snow | "all wet-bulb melting energy in the atmospheric column" (App. a) | `PA`: sum of warm segments below the highest 0 °C crossing (lines 53-66); a warm segment at the top of the profile is excluded | All melting layers in 0-5 km |
| D12 | ME and RE for PL | ME above a near-surface cold layer with nonzero RE; PL only when that ME is nonzero (App. b) | Same `PA` and `NA` as for snow / FZRA; PL computed whenever `PA > 0` (line 138), including a surface-based melting layer with no refreezing layer (Fig. 1b) | Lowest freezing layer under a warm layer aloft; RE of that layer, all ME above it; PL = 0 otherwise (Q5) |
| D13 | RE for FZRA | "refreezing energy beneath any warm layers aloft" (App. c) | `NA`: all cold segments below the highest crossing (equivalent except when the top segment of the profile is warm) | Freezing layers below the highest warm layer aloft |
| D14 | ME for FZRA ramp / Eq. 7 | "the wet-bulb melting energy" (App. c), ramp when < 5 J/kg | `PA` | All-column ME, as for snow (Q4) |
| D15 | Zero crossings | Not discussed | Inserted twice (lines 133 and 32); harmless | Each layer split exactly once at the linear crossing; zero-length pieces dropped |
| D16 | Dominant class | Not defined (probabilities of weather type) | Last index of the softmax maximum (line 155) | Argmax, ties to the later type (frzr > icep > snow > rain), 254 if all four are 0 |
| D17 | Height resolution | Not applicable | `height_resolution=250` hard coded (line 160) | Depths from the heights passed in |
| D18 | Unchanged | Eq. 1 with T0 = 273.15 K on Tw (sects. 3a, 3d); Eq. 2b quartic; Eqs. 7-9 constants; surface Tw > 0 °C -> ProbRA = ProbFZRA (App. c) | Same (g = 9.81, line 28) | Same |

## 2. Measured impact (50,000 synthetic profiles)

Same profiles for both methods (`make_profiles(50000, seed=99)` from the test file, Tw from
`columns.wet_bulb_temperature`, 5 % of columns rounded to 0.5 °C). Script:
`scratchpad/scripts/compare_paper.py`. The profiles are synthetic and very moist (top level RH_ice >
0.75 in 89.6 %), so the frequencies show the size of the change, not a climatology.

| Dominant class | Legacy repo | Revised paper |
|---|---:|---:|
| rain | 17,746 | 17,001 |
| snow | 24,640 | 28,243 |
| ice pellets | 0 | 490 |
| freezing rain | 7,614 | 4,266 |
| none (all four 0) | 0 | 0 |

**Dominant class changed for 10.31 % of profiles.** Transitions (rows legacy, columns revised: rain,
snow, icep, frzr):

| Legacy \ Revised | rain | snow | icep | frzr |
|---|---:|---:|---:|---:|
| rain | 16,933 | 642 | 171 | 0 |
| snow | 68 | 24,025 | 168 | 379 |
| icep | 0 | 0 | 0 | 0 |
| frzr | 0 | 3,576 | 151 | 3,887 |

The biggest movement is frzr -> snow (3,576 profiles). In 3,561 of them the repo's `prob_ice` is 0, and
in 3,567 the paper ProbIce is higher (3,233 have ProbIce = 100). So this change comes from the ProbIce
rules (D5-D9: RH formula, elimination instead of zeroing, depth thresholds). With ProbIce = 0 both
methods give FZRA = 100. Over all 50,000 profiles, ProbIce is higher in the paper version for 8,489,
lower for 2,930 and equal for 38,581. Ice pellets appear only once D1 and D2 are fixed. The other
transitions were not attributed to individual differences; they interact, and they were not toggled
one at a time.

Other numbers from the same run: Fig. 1 scenarios a/b/c/d = 29,443 / 17,122 / 1,615 / 1,820;
ProbIce 0 / partial / 100 = 9,100 / 6,577 / 34,323 (5,322 columns have no generation layer); ties for
the maximum probability in 929 profiles (1.9 %); 2,073 profiles have two or more probabilities >= 0.5;
the sum of the four probabilities ranges 0.00-3.00 (the lowest: 2.6e-08, a column whose only
near-surface cold layer is 0.1 J/kg under a 234 J/kg cold layer, see Q5).

## 3. Agreement and timing of the implementation

* `revised_bourgouin` (numba) vs the independent pure-Python paper reference in the test file, same
  50,000 profiles: max |probability difference| 2.98e-08, class agreement 100.0000 %, max |ΔME| 2.4e-4
  J/kg, max |ΔRE| 2.7e-5 J/kg, max |ΔProbIce| 3.8e-6 %. The pytest repeats this on 20,000 other profiles.
* 1,905,141 columns, wet bulb computed inside the kernel (`revised_bourgouin_from_sounding`), Apple M4
  Pro, other jobs running (load 6.8-10): **10.270 s on 1 thread, 0.980 s on 14 threads** (with
  diagnostics: 10.288 / 1.009 s; with Tw given: 0.898 / 0.117 s; legacy repo method incl. wet bulb:
  10.309 / 1.013 s). Script: `scratchpad/scripts/bench_revised.py`.
