# Revised Bourgouin: remaining questions

The scientist chose to re-derive the method from the paper: Birk, Lenning, Donofrio and Friedlein (2021),
"A revised Bourgouin precipitation-type algorithm", *Wea. Forecasting* 36, 425-438. It is implemented as
`revised_bourgouin*` in `ptype/hrrr_fast/bourgouin.py`. The repo method is kept, for comparison only, as
`legacy_repo_bourgouin*`. The differences between the two, with measured impact, are listed in
`bourgouin_paper_vs_repo.md`.

Section 1 lists the places where the **paper** is not explicit, and the reading implemented for each.
Section 2 lists the earlier questions about the repo code that the paper answers. Section 3 records the
run-fixes made to the legacy repo port.

Numbers marked "(synthetic)" come from 50,000 synthetic profiles (`make_profiles(50000, seed=99)` in
`tests/test_hrrr_fast_bourgouin.py`; `scratchpad/scripts/compare_paper.py`). They show the size of an
effect, not a climatology.

## 1. Open: ambiguities in the paper (reading implemented → alternative)

| # | Paper text | Implemented reading | Alternative(s) and notes |
|---|---|---|---|
| Q1 | "relative humidity with respect to ice (RH_Ice) greater than 75%" (sect. 3e, p. 428); no formula given | `RH_ice = e_w(Td) / e_i(T)`, because Td is a dewpoint over water; MetPy 1.7 saturation formulas; used at all temperatures | (a) The repo's `e_i(Td)/e_i(T)` (MetPy `phase='solid'`), which is lower than true RH_ice below 0 °C: 1.000 vs 1.216 at water saturation at -20 °C. (b) RH over water where T > 0 °C. |
| Q2 | Generation layer "exceed 1 km in depth"; dry layer "for more than 1500 m" (p. 428) | Depth = height of the top level of the run minus height of its bottom level, compared strictly. On the 250 m grid that needs ≥ 6 saturated levels and ≥ 8 dry levels. | (a) `>=` (the repo: 5 and 7 levels). (b) Count each level as a 250 m slab (4 and 6 levels). With 250 m sampling, the true layer can be up to 250 m deeper than the sampled span. |
| Q3 | "If a precipitation generation layer exists above a dry layer ... this upper layer is eliminated from consideration" (p. 428); ProbIce from "the minimum temperature in such a layer" | A generation layer is eliminated if a deep dry layer lies **anywhere** below it, down to the surface. Every layer above the lowest deep dry layer is therefore eliminated. ProbIce comes from the **highest** remaining layer (top-down, like the traditional method). | (a) Eliminate only when the dry layer is directly beneath the generation layer. (b) Use the coldest remaining layer, or the maximum ProbIce over remaining layers (seeder-feeder). |
| Q4 | App. c: "ME_Tw is the wet-bulb melting energy" (unqualified). Eq. 7 and the 5 J/kg ramp were fitted against "melting energy aloft" (Figs. 3, 5). | All-column ME, the same as App. a. A surface-based melting layer alone (Fig. 1b) then gives ProbFZRA_i = 100, so ProbRA = 100 when ProbIce = 100. | ME aloft only. That gives ME = 0 for Fig. 1b soundings, so the ramp makes ProbRA = 100 − ProbIce (0 when ProbIce = 100), which looks unintended. |
| Q5 | App. b: "ME_Tw is the wet-bulb melting energy above a near-surface cold layer with nonzero RE_Tw" (single-layer cases, Figs. 1c-d) | Layers are maximal Tw > 0 / Tw ≤ 0 intervals; layer 0 is surface based, and a warm layer with index ≥ 1 is "aloft". The near-surface cold layer is the **lowest** freezing layer that has a warm layer aloft above it. `RE_PL` is that layer's energy only; `ME_PL` is all melting energy above it. PL is computed only if `ME_PL > 0` and `RE_PL > 0`. FZRA uses `RE` = all freezing layers below the highest warm layer aloft (App. c "beneath any warm layers aloft"). | (a) `RE_PL = RE_FZRA`. (b) `ME_PL` = only the warm layer directly above the cold layer. Matters only with several warm/cold layers: `RE_PL ≠ RE_FZRA` in 138 of 50,000 profiles (synthetic). Example: a 0.1 J/kg near-surface cold layer under a 234 J/kg cold layer, ME 69.6, ProbIce 100. All four probabilities are ≈ 0 (sum 2.6e-08); with reading (a), PL = 100. |
| Q6 | Eq. 2: "T is the temperature in degrees Celsius" (minimum in the generation layer) | Dry-bulb T at the levels of the layer | Tw |
| Q7 | Not discussed: a column with **no** generation layer (dry, or every layer eliminated) | ProbIce = 0 (no ice nucleation). That gives FZRA = 1 (RA = 1 if surface Tw > 0), i.e. freezing drizzle. 5,322 of 50,000 profiles (synthetic). The diagnostic layer top is NaN. | Treat as "no precipitation" / NaN, or leave ProbIce undefined and mask downstream. |
| Q8 | Profiles are limited to 0-5000 m (network grid) | Generation layers above 5 km are invisible; a layer that reaches 5 km is cut off there (its Tmin may be too warm) | Read more levels (e.g. native HRRR levels up to ~10 km) **for ProbIce only**. Synthetic: the top level has RH_ice > 0.75 in 89.6 % of profiles, and the chosen layer reaches 5000 m in 66.5 % (74.4 % of profiles that have one). These synthetic profiles are very moist; check on real HRRR. ME/RE also stop at 5 km. |
| Q9 | A melting layer needs at least 2 J/kg (pp. 427, 432, describing the **original** Bourgouin method) | Not applied: the Appendix gives no threshold | Ignore melting layers < 2 J/kg in ME. |
| Q10 | Tw exactly 0 °C | Tw ≤ 0 counts as freezing. Surface "Tw > 0 °C" for rain is strict (App. c). Zero-length pieces are dropped. | Tw ≥ 0 counts as melting. |
| Q11 | Eq. 1 uses g without a value | g = 9.81 m s⁻² (as in the repo); T0 = 273.15 K (given) | 9.80665 changes energies by 0.03 %. |
| Q12 | No "dominant class": the output is four probabilities (sect. 3b) | `cls` = argmax; ties go to the later type (frzr > icep > snow > rain, as the repo); 254 if all four are 0. Ties: 929 of 50,000 (1.9 %); all-zero: 0. | First-index argmax, or report the full probability vector only. |
| Q13 | Eqs. 2a-c are piecewise: the quartic is 98.33 at -15 °C and 0.81 at -7 °C | Literal (jumps 100 → 98.3 and 0.81 → 0 at the ends) | Rescale so the ends are 100 and 0. |
| Q14 | Wet bulb: the paper uses Tw profiles but does not specify the method | Normand's rule with MetPy constants (`columns.wet_bulb_temperature`, ≤ 5.1e-4 °C from MetPy) | A psychrometric (isobaric) wet bulb. |
| Q15 | App. b "only when ME_Tw is nonzero" | ME_PL > 0 **and** RE_PL > 0 (the cold layer must have "nonzero RE") | ME_all > 0. |

## 2. Resolved by the paper (former questions about `ptype/algorithms.py`)

| Former # | Repo issue | Resolution from the paper |
|---|---|---|
| B1 | Sign of the cold area `NA` (≤ 0) in the formulas (lines 58-60, 139, 143) | ME and RE are both positive (p. 427). Implemented. |
| B2 | ProbIce applied twice to PL (lines 139, 142) | Applied once (App. b step 2; p. 432). |
| B3 | Ice pellets can never win (consequence of B1+B2) | Fixed by B1/B2: 490 ice-pellet dominant classes in 50,000 (synthetic), versus 0 in the repo. |
| B4 | Raw scores outside 0-100 | Every step is constrained to 0-100 (App. a-c). |
| B5 | `softmax(scores/100)` | No softmax and no normalisation: four independent probabilities of weather type (sect. 3b; p. 434). |
| B6 | FZRA relabelled RA when surface Tw > 0 | Confirmed (App. c, "Where surface Tw > 0 °C: ProbRA = ProbFZRA"). |
| B7 | PA = 0 and ProbIce = 0 gives FZRA | Intended: freezing drizzle "is the most likely outcome if ice nucleation does not occur" (sect. 3e). Remaining question: the no-generation-layer case (Q7). |
| B8 | Source of the quartic | Eq. 2b (from Baumgardt et al. 2017, Fig. 2). End jumps remain (Q13). |
| B9 | Reversed indexing / highest layer only | Paper: layers "exceed 1 km" and are eliminated above deep dry layers. Which remaining layer to use is Q3. |
| B10 | Deep dry sub-cloud layer sets ProbIce to 0 | Paper: the upper layer is "eliminated from consideration", not ProbIce = 0. Implemented as Q3. |
| B11 | RH w.r.t. ice formula and the 0.75 threshold | Threshold confirmed (> 75 %). Formula not given → Q1. |
| B12 | Area units and Tw vs T | Eq. 1 with T0 = 273.15 K on the Tw profile (sects. 3a, 3d); the thresholds (Eqs. 3-9) were re-derived from Tw data. |
| B13 | Summing all warm / cold layers | Appendix defines ME (all column for SN), ME above the near-surface cold layer (PL), RE beneath warm layers aloft (FZRA). Multi-layer details → Q4, Q5. |
| B14 | Crossings inserted twice | Not part of the method; each layer is now split once. |
| B15 | Tie rule | Not defined by the paper → Q12. |
| B16 | Hard-coded 250 m resolution | Depths computed from the heights. |
| B17 | Wet-bulb method | Not specified → Q14. |
| B18 | Surface Tw from 2 m T/Td | Unchanged (inference policy, not part of the paper). |

## 3. Legacy repo port (`legacy_repo_bourgouin*`): run-fixes

The minimum changes needed to run `calc_modified_bourgouin_ptype`, used in the legacy reference and port:

* **A1**: string dict keys in `calc_sounding_stats` (lines 67-75).
* **A2**: `<+ -15` read as `<=` (line 180).
* **A3**: bare `except:` → `except ValueError:` (line 40).

The legacy port matches that corrected reference: max probability difference 2.98e-08, 100 % class agreement on 50,000 profiles. The classic `calc_bourgouin_ptype` (lines 81-127) is not ported. It has these bugs:

* `PA[-1]` indexes a float (line 120).
* It returns `None` for 1 crossing with surface ≤ 0, and for 2 crossings with surface > 0.
