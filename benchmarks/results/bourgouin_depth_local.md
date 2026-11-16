# How deep must the Bourgouin height grid be? (local, one real hour)

The revised Bourgouin method (Birk et al. 2021; `ptype/hrrr_fast/bourgouin.py`) gets ProbIce from the
**precipitation-generation layer**: the highest layer deeper than 1 km with RH over ice > 75 % that has no
> 1.5 km dry layer below it, and ProbIce follows from the minimum temperature in that layer. On the network's
0–5000 m grid, a cloud layer that continues above 5 km is cut at the grid top, so its minimum temperature
(and possibly the choice of layer) is wrong. This measures how much that matters, and what a deeper grid
costs.

## Method

Command (all numbers below are from this run; JSON in the scratch directory, `depth_f12.json`):

```
PYTHONPATH=. python benchmarks/pipeline_benchmark.py bourgouin-depth \
    --cache <scratch>/reader_data/f12_n50c.pkl --key hrrr.20231226/conus/hrrr.t00z.wrfnatf12.grib2 \
    --threads 8 --fetch-s3 --out <scratch>/pipeline_data/depth/depth_f12
```

* Input: hrrr.20231226 t00z **f12** (the cached hour, all 50 hybrid levels already in the cache).
  1,905,141 columns; 441,326 (23.2 %) inside `precip_mask` (any HRRR CRAIN/CSNOW/CICEP/CFRZR = 1).
* For each top (5, 8, 10, 12 km): read the smallest number of hybrid levels whose top is ≥ the grid top
  at **every** column (minimum height above ground over the grid), interpolate T, Td and p to
  0, 250, …, top m with the same kernel and 0 m policies as the network sounding
  (`pipeline.run_bourgouin`), and run `revised_bourgouin_from_sounding(..., return_diagnostics=True)`.
* "Touches the top": the diagnostic `generation_layer_top_m` equals the grid top, i.e. the chosen
  generation layer reaches the top of the grid and may continue above it.
* Changes are relative to the deepest grid (12 km). "ProbIce changed": |ΔProbIce| > 0.001 percentage points.
* Extra bytes: exact message sizes from the idx of TMP/SPFH/PRES/HGT on the levels above the 27 the network
  needs. Extra fetch seconds: one real `obstore.get_ranges` of exactly those messages from the NOAA bucket
  (this laptop's link; two runs shown, the network is noisy). Bourgouin seconds: best of 2, 8 numba threads,
  wet-bulb included.
* Machine: Apple M4 Pro, 14 cores, 48 GB, shared with other jobs (load average 6.4 → 7.5 during the run), so
  seconds are indicative only. Python 3.12.12, numpy 2.5.5, numba 0.68.1, obstore 0.12.1, eccodes 2.50.1.

## Result

| grid top | hybrid levels needed (min AGL of that level) | columns whose generation layer touches the top: all / in mask | ProbIce changed vs 12 km: all / in mask | mean abs ΔProbIce in mask (pct points) | ΔProbIce ≥ 50 pts, in mask | dominant class changed: all / in mask | any of the 4 probabilities changed by > 0.01 / > 0.1, in mask (max change) | extra bytes fetched | extra fetch s (run 1, run 2) | extra decode s | Bourgouin s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 km | 25 (5,155 m) | 15.6 % / 42.2 % | 5.11 % / 11.07 % | 6.47 | 5.60 % | 0.043 % / 0.127 % | 0.322 % / 0.231 % (1.00) | 0 | – | – | 3.33 |
| **8 km** | 33 (8,322 m) | 10.9 % / 31.6 % | 0.35 % / 0.88 % | 0.43 | 0.33 % | **0 / 0** | **0 / 0** (0.008) | 36.3 MB (+14.5 % of the 250 MB network read) | 1.08, 1.40 | 0.06 | 4.99 |
| 10 km | 36 (10,024 m) | 9.5 % / 26.9 % | 0.015 % / 0.029 % | 0.012 | 0.007 % | 0 / 0 | 0 / 0 (2e-16) | 56.6 MB (+22.7 %) | 1.76, 1.82 | 0.07 | 6.06 |
| 12 km | 39 (12,286 m) | 1.0 % / 2.4 % | (reference) | – | – | – | – | 77.1 MB (+30.9 %) | 1.90, 6.32 | 0.09 | 7.35 |

(The 250 MB network read is the 27-level, 173-message plan of this file without APCP, 249.6 MB.)

What the numbers say:

* **5 km is not deep enough.** 42 % of the precipitating columns have a generation layer cut off by the grid
  top. For 11 % of them ProbIce differs from the deep answer (by 6.5 points on average, by ≥ 50 points for
  5.6 %), and that reaches the output: 0.13 % of precipitating columns change dominant class and 0.23 % have a
  probability that moves by more than 0.1 (up to 1.0).
* **8 km removes every visible effect on this hour.** A third of the precipitating columns still touch the
  8 km top, but the minimum temperature reached by then is almost always ≤ −15 °C (ProbIce = 100 already), so
  ProbIce still changes in only 0.9 % of them, no dominant class changes, and no probability changes by more
  than 0.008 — about 2 quantisation steps of the stored uint8 probabilities (1/254). Cost: +36 MB per hour
  (+15 %), ~1–1.4 s more fetch on a 15–25 MiB/s link, and +1.7 s of Bourgouin compute on 8 threads.
* 10 km and 12 km change the stored output by nothing measurable (≤ 2e-16) for +57/77 MB and +2.7/4.0 s.
* Touching the top drops sharply only at 12 km, where the grid reaches the dry stratosphere; the earlier
  "touches" at 8–10 km are deep, cold cloud layers whose cut-off no longer changes ProbIce.

## Recommendation

**`bourgouin_top_m: 8000`** (the default in `config/hrrr_fast.yml` and `PipelineConfig`). It is the
shallowest top for which, on this hour, the Bourgouin output (dominant class and all four probabilities,
to within 0.008) equals that of the 12 km grid, whereas 5 km changes 0.13 % of precipitating columns' class
and moves probabilities by up to 1.0. 10 km would cost another 20 MB and ~1 s per hour for no measured change.

Caveats: this is ONE winter hour. Deep warm-season convection has colder, higher generation layers, but those
columns are rain at the surface regardless, which is why ProbIce rarely matters there; a cold-season case
with high cirrus decks over a shallow precipitating layer is the case to re-check. The pipeline reads 34 levels
by default for 8 km (the measured 33 plus one level of margin) and **checks every hour**: if any column's top
level is below 8 km AGL it fetches more levels automatically and logs it (`rereads` in the manifest).
