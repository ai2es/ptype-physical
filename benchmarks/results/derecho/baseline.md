# Baseline: original HRRR p-type inference on Derecho (CPU node)

Case: HRRR 2023-12-26 00 UTC init, `prs` (wrfprs) product, model `ptype_model_20240909.keras` + scaler
`ptype_scaler_20240909.json`, full CONUS grid. Original `scripts/run_inference.py` and `ptype/inference.py` are
**unmodified**; everything is run through the wrapper `benchmarks/baseline_run.py`.

## Does the original run as-is?  No -- three environment/config issues, none needing a source change

| Problem | Where | Resolution |
|---|---|---|
| `TypeError: Could not locate class 'CategoricalDNN'` (mlguess not imported) and then `Could not locate function 'loss'` (compile config of the saved model) | `ptype.inference.load_saved_model` -> `keras.models.load_model(path)` | wrapper imports `mlguess.keras.models` / `callbacks` and replaces the name `load_model` inside `ptype.inference` by `keras.models.load_model(path, custom_objects={"loss": mlguess.keras.losses.evidential_cat_loss})` (same call as `applications/export_ptype_weights.py`). |
| `ModuleNotFoundError: dask.distributed` (script imports `dask.distributed` and `dask_jobqueue` at top, even with `use_dask: False`) | `scripts/run_inference.py` | `uv pip install distributed dask-jobqueue` into the venv (add-only: dask-jobqueue 0.9.0, distributed 2026.8.0, msgpack, sortedcontainers, tblib, zict; nothing existing changed). |
| `ValueError: Invalid frequency: 1H` (pandas 3) | `pd.date_range(... freq=config["dates"]["frequency"])` | config uses `"1h"`. |

What the wrapper adds, and nothing else: (1) the loading fix above; (2) wall-clock timers around `load_saved_model`,
`download_data`, `load_data`, `convert_and_interpolate`, `transform_data`, `model.predict` (via a proxy object),
`grid_predictions`, `save_data`, appended to `stages.jsonl`; (3) optional dump of the model input matrix;
(4) writes a per-run copy of the config (out_path, forecast range, n_processors) and runs the original script with
`runpy.run_path(..., run_name="__main__")` so its own `multiprocessing.Pool` code path is used. Arguments/results pass
through untouched. The original `main` is called by the script itself in the original order.

## Config: `config/inference_baseline_20240909.yml` (derived from `inference_rt_central_winter_2023_2024_00Z.yml`)

Deviations from the stock config (that file already had `evidential: True`; `config/inference.yml` does not, so the rt file was the base):

| Key | Stock | Baseline | Why |
|---|---|---|---|
| `ML_model_path`, `input_scaler_file` | cbecker evi_model_ptype_050124 / evi_scaler | `/glade/work/dgagne/ptype_fast/model/ptype_model_20240909.keras`, `ptype_scaler_20240909.json` | 20240909 model |
| `out_path` | cbecker scratch | `.../ptype_fast/baseline/out/` (wrapper overrides per run to `<run>/out/`) | outputs to scratch `baseline/` |
| `n_processors` | 60 | 1 (wrapper overrides per run) | |
| `extent` | `[-108,-89,36,48]` | `[-122.8,-60.9,21.0,47.9]` | `"full"` is NOT supported: `get_subset` unpacks `sw_lon, ne_lon, sw_lat, ne_lat = extent` (a 4-char string "full" would fail at `'f' < 0`). Corner-based nearest-point slicing with exclusive end gives the largest possible grid: **y=1058, x=1798** (native HRRR is 1059 x 1799; last row/column are dropped by `slice(sw, ne)`). 1,902,284 columns. |
| `dates` | 2024-03-01..04-30 | start = end = `2023-12-26 00:00` | benchmark case |
| `dates.frequency` | `"1H"` | `"1h"` | pandas 3 rejects `H` |
| `forecast_range` | 1..18 | 12..12 (single run); 0..18 for scaling (wrapper) | |
| `ml_atm_varaibles` (sic) | `['t','dpt']` | `['t','dpt','u','v']` | 84-input model: T, Td, U, V |

How the input matrix is built (`convert_and_interpolate`): `height_levels` low/high/interval 0/5000/250 gives 21 levels (np.arange(0, 5250, 250)). For each variable in `ml_atm_varaibles`
the 40 isobaric levels are interpolated to those heights (`hgt_above_sfc = gh - orog`), then column 0 (0 m) is overwritten with the surface
variable via `surface_match = {t:t2m, dpt:d2m, u:u10, v:v10}`; blocks are concatenated in the order of `ml_atm_varaibles`. u/v are
supported with **no code change** (u10/v10 are in the config's heightAboveGround list). Columns are named positionally
from the scaler's `groups_` (T, Td, U, V; each 0..5000 m ascending), so order of `ml_atm_varaibles` must be T,Td,U,V. Verified from the saved matrix: first/last columns of each group are `TEMP_C_0_m..TEMP_C_5000_m, T_DEWPOINT_C_0_m..5000, UGRD_m/s_0_m..5000, VGRD_m/s_0_m..5000`; no NaN in the first 100k rows.
Note the data source is Herbie HTTPS download of the whole `wrfprsf??` file (AWS open-data mirror, "GRIB2 @ aws" in the log; no AWS CLI used).

## Environment

Derecho CPU node, queue `main`: 2 x AMD EPYC 7543 32-core (128 cores, SMT visible as 128 CPUs), 235 GB requested (qstat); smoke job on `develop` (8 CPUs, 60 GB).
Python 3.12.11, `KERAS_BACKEND=torch`. Repo commit at run time `b344d81` (single/smoke) / `079942f` (scaling).

```
bridgescaler==0.9.0   cfgrib==0.9.15.1   dask==2026.8.0   dask-jobqueue==0.9.0   distributed==2026.8.0
eccodes==2.50.1   herbie-data==2026.10.0   keras==3.15.1   miles-guess==2025.1.0
netcdf4==1.7.5.1.post1   numba==0.68.1   numpy==2.5.5   pandas==3.1.1   torch==2.15.0   xarray==2026.9.0
```
(`uv pip freeze | grep ...`; the venv has no `pip`.)

## Jobs

| Job | Queue / resources | Purpose | Outcome |
|---|---|---|---|
| 8207318 | develop 8 cpu | smoke | failed fast: `dask.distributed` missing |
| 8207337 | develop 8 cpu | smoke | failed fast: pandas `1H` |
| 8207341 | develop 8 cpu | resubmitted by mistake with the old config (qdel too late) | failed fast (same `1H`) |
| 8207343 | develop 8 cpu, 60 GB | smoke f12 (also saved input matrix) | OK, 140.3 s wall |
| 8207402 | main 128 cpu | single-process runs A (default threads), B (`taskset -c 0`), A2 (repeat of A) | OK |
| 8207403 | main 128 cpu | scaling n=1 default threads | **deleted (qdel) after ~12 min** once run A showed 343 s/file at default threads (19 files would exceed the 1 h limit); 2 files had finished |
| 8207404 | main 128 cpu | scaling n=4,8,16,32 default threads | **killed at walltime (3634 s)**: n=4 did not finish; n=8/16/32 never started |
| 8207605 | main 128 cpu | scaling n=1 pinned to one core (`taskset -c 0`) | still queued when written |

8 jobs is the budget; all used.

## Commands

```
# on Derecho, from /glade/work/dgagne/ptype_fast/ptype-physical (env as in benchmarks/pbs/baseline_*.pbs)
export KERAS_BACKEND=torch PYTHONPATH=$PWD; source /glade/work/dgagne/ptype_fast/venv/bin/activate
/usr/bin/time -v python benchmarks/baseline_run.py -c config/inference_baseline_20240909.yml \
    --out-dir $BL/runs/single_f12_A_default --fxx-start 12 --fxx-end 12 --n-procs 1
/usr/bin/time -v taskset -c 0 python benchmarks/baseline_run.py ... --out-dir $BL/runs/single_f12_B_taskset1 ...
/usr/bin/time -v python benchmarks/baseline_run.py -c config/inference_baseline_20240909.yml \
    --out-dir $BL/runs/scaling_n4 --fxx-start 0 --fxx-end 18 --n-procs 4          # etc.
```
`BL=/glade/derecho/scratch/dgagne/ptype_fast/baseline`. PBS scripts: `benchmarks/pbs/baseline_{smoke,single,scaling}.pbs`.

## (a) Single forecast hour (f12), 1 process: stage timing (seconds, measured by the wrapper)

| Stage | A: default threads (job 8207402) | A2: repeat (same job, fresh download) | B: `taskset -c 0` (1 core) | smoke, 8 cores (job 8207343) |
|---|---|---|---|---|
| load model + scaler | 0.4 | 0.1 | 0.1 | 1.1 |
| download (whole wrfprsf12 grib) | 87.0 | 10.2 | 29.3 | 38.1 |
| load_data (cfgrib + flatten + subset) | 19.3 | 16.1 | 16.0 | 19.5 |
| convert_and_interpolate (numba) | 11.8 (incl. first-call JIT compile) | 0.9 (numba cache warm) | 4.4 | 11.9 |
| transform_data (scaler) | 1.5 | 1.5 | 1.4 | 1.5 |
| model.predict (batch 2048) | **162.6** | **164.0** | **9.7** | 5.2 |
| grid_predictions | 3.4 | 3.5 | 3.3 | 3.3 |
| save_data (netCDF zlib-4) | 54.5 | 54.2 | 54.3 | 54.9 |
| **wrapper wall total** | **343.6** | **250.7** | **118.8** | 140.3 |
| `/usr/bin/time` elapsed (incl. imports) | 6:56.45 | 4:22.11 | 2:09.93 | 3:49.00 (incl. input dump) |
| peak RSS (`/usr/bin/time`) | 11.25 GB | 11.18 GB | 11.10 GB | 11.18 GB |

Findings: peak memory per process is about 11 GB for the full CONUS grid. Download time varies run-to-run by 8x (10-87 s) and is network-bound. `save_data` (54 s) and
`predict` dominate once threads are sane. **At default threading on a 128-core node `model.predict` is 17x slower (about 163 s vs 9.7 s on one pinned core)**: torch (and numba) start 128 threads
per process for batch-2048 matmuls; this oversubscription effect is the dominant cost of the stock script on a full node. The 1-core pinned run is the true single-processor reference (118.8 s).

Input/outputs of run (a) (the smoke run job 8207343 additionally saved the inputs; same code and data):
- `.../baseline/f12_inputs/input_matrix.npy` float64 (1902284, 84), 1.28 GB; `input_columns.json` (the 84 scaler column names, order = columns); `lat.npy`, `lon.npy` float64 (1058, 1798) (lon in 0-360); `grid_shape.json` (`y=1058,x=1798,n_rows=1902284`). Row order = C-order flatten of (y, x).
- Output netCDF kept: `.../baseline/runs/single_f12_A_default/out/hrrr/20231226/0000/MILES_ptype_hrrr_2023-12-26_0000_f12.nc` (783,232,401 B = 0.78 GB; B/A2/smoke copies kept in their run dirs).

### Output variables (`MILES_ptype_hrrr_2023-12-26_0000_f12.nc`; dims time=1, heightAboveGround=21, y=1058, x=1798, isobaricInhPa=40)

float32 (1,21,y,x): `t_h, dpt_h, u_h, v_h`; float64 (1,21,y,x): `isobaricInhPa_h`;
float32 (1,40,y,x): `t, u, v, dpt, gh, hgt_above_sfc`; float32 (1,y,x): `refc, t2m, d2m, u10, v10, orog, sp, tp, ML_u,
ML_{rain,snow,icep,frzr}, ML_{rain,snow,icep,frzr}_ale, ML_{rain,snow,icep,frzr}_epi`;
uint8 (1,y,x): `crain, csnow, cicep, cfrzr, ML_crain, ML_csnow, ML_cicep, ML_cfrzr`. (Mean ML_rain 0.657, ML_snow 0.297 for the smoke run, f12.)
Size per file about 0.53-0.78 GB (n=4 run mean 0.76 GB over 17 files; f01 was 0.53 GB).

## (b) Scaling reference, f00-f18 (19 files), script's own multiprocessing.Pool

Stock threading (nothing limits torch/numba threads, so each worker starts up to 128 threads):

| n_processors | Result | Evidence |
|---|---|---|
| 1 (default threads) | not completed (job deleted by me after 12 min); 2 files done in that time with `predict` 164.3 s each | job 8207403, `runs/scaling_n1/stages.jsonl` |
| 4 (default threads) | **did not finish in the 1 h limit**: 16 of 19 files written (17 had reached `grid_predictions`) when killed at 3634 s; per-file stage means: download 156 s (max 490), load_data 17.4, convert 1.6, transform 1.3, **predict 529.6 s (max 667)**, grid 3.6, save 59.9 (4 workers x 128 threads oversubscribed). Node memory (free -m used) peaked at 52.6 GB; per-process RSS peak about 11.1 GB. No OOM. | job 8207404, `runs/scaling_n4/` |
| 8, 16, 32 (default threads) | **not run** (job 8207404 hit walltime during n=4; the 8-job budget was used up) | |
| 1 (pinned `taskset -c 0`) | see below | job 8207605 |

**Job 8207605 (n=1 pinned to one core, 19 files, expected ~40 min at the 118.8 s/file measured in run B) was still queued (state Q) after about 75 minutes when this document was written; no result yet.** When it finishes, results are in `/glade/derecho/scratch/dgagne/ptype_fast/baseline/runs/scaling_n1_taskset/` (`stages.jsonl`, `summary.json`, `node_mem_used_mb.txt`, `out/`) and the log `bl_scaling.o8207605` in the repo checkout; summarize with the per-stage aggregation of `stages.jsonl`.

Memory per worker is about 11 GB, so by this evidence 16 workers need about 180 GB and 32 workers about 350 GB: 32 would not fit in the 235 GB node (extrapolation from per-process RSS, not run).

## Not completed / caveats
- Scaling at n = 4 (full), 8, 16, 32 with sane thread settings (e.g. `OMP_NUM_THREADS=NUMBA_NUM_THREADS=128/n`) was **not measured**; with default threads the original script is dominated by thread oversubscription in `predict`.
- Input matrix dumped in the smoke job (8207343), not in the f12 timing jobs; the two were generated by the same code path from the same file (the output grids match: 1058 x 1798).
- Output netCDFs of the 19-hour run are incomplete (n=4: 16 files); a complete 19-file set exists only if job 8207605 finishes.

## (c) Scaling reference, completed: thread-split runs (8 threads per process or fewer), f00-f18, 2023-12-26 00 UTC

The sections above measured the stock script with default threading (every process starts 128 torch/numba threads).
These runs give each of the N processes `128 / N` threads (`OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS`, see
`benchmarks/pbs/baseline_scaling.pbs` with `SPLIT_THREADS=1`), which is the most favourable setting for the unmodified
code. Source: `$BL/runs/scaling_n{4,8,16}_split/{summary.json,stages.jsonl,node_mem_used_mb.txt}` and
`scaling_n1_taskset/stages.jsonl`, aggregated by me (all numbers below were recomputed from those files).

| processes x threads | wall s (`summary.json`) | netCDF files written | netCDF size on disk | peak RSS per process (`children_maxrss_mb`) | node memory used, peak (`free -m`, 5 s samples) |
|---|---|---|---|---|---|
| 4 x 32 | 1073.8 | 19 | 14 GB | 10.9 GB | 55.5 GB |
| 8 x 16 | 527.9 | 19 | 14 GB | 10.9 GB | 76.0 GB |
| 16 x 8 | 267.6 | 19 | 14 GB | 10.9 GB | 152.5 GB |

(`error: null` in all three summaries. 14 GB is `du -sh` of each run's `out/` directory.) Wall time scales almost
exactly with 1/N, i.e. the run is not bound by anything shared at these process counts.

Per-forecast-hour stage medians over the 19 hours of each run, seconds (stage timers of `benchmarks/baseline_run.py`):

| stage | 4 procs | 8 procs | 16 procs |
|---|---|---|---|
| download (whole wrfprs GRIB) | 81.3 | 121.0 | 150.6 |
| load_data (cfgrib + flatten + subset) | 14.0 | 15.7 | 17.2 |
| convert_and_interpolate | 0.8 | 0.8 | 1.2 |
| transform_data (scaler) | 1.2 | 1.4 | 1.5 |
| model.predict | 6.5 | 6.0 | 6.7 |
| grid_predictions | 3.3 | 3.4 | 3.6 |
| save_data (netCDF zlib-4) | 54.5 | 54.4 | 54.6 |

Download is the only stage that changes with the process count (81 -> 151 s per hour); `save_data` is constant at about
54.5 s. Everything is of course per process, so at 16 processes the node writes 19 files in 267.6 s.

**Pinned single-core job 8207605** (`scaling_n1_taskset`, 1 process, `taskset -c 0`, 19 hours requested): killed by PBS at the
1 h walltime (`job killed: walltime 3628 exceeded limit 3600` in `bl_scaling.o8207605`). 10 hours had completed (10
`save_data` records, 10 netCDF files, 7.3 GB). Download mean 252.1 s per hour (median 278.7, max 382.6), predict 9.7 s,
save 54.0 s. Ten of 19 hours in 3600 s, so this configuration cannot do 19 hours in 1 h; a 19-hour total was not measured.

This supersedes the "not run" / "still queued" entries in section (b) and the first caveat under "Not completed" for
n = 4, 8, 16 with sane thread settings; n = 32 (thread-split) and n = 1 on 19 hours were still not measured.

## Comparison with the new pipeline (same case, f00-f18, one Derecho CPU node, torch on CPU, no upload)

New pipeline numbers are from `benchmarks/results/derecho/smoke_fetch_8210303/README.md` (job 8210303, 15 workers x 8
compute threads, 19 hours, wrfnat byte ranges, commit `b48317b`, five cases back to back in one job). The original rows are
the thread-split runs above (16 x 8).

| pipeline | input / source | processes x threads | wall s, 19 hours | per-hour medians (s) |
|---|---|---|---|---|
| original, `run_inference.py` | whole wrfprs GRIB via Herbie (AWS) | 16 x 8 | 267.6 | download 150.6, load_data 17.2, convert 1.2, transform 1.5, predict 6.7, grid 3.6, save 54.6 |
| original | same | 8 x 16 | 527.9 | download 121.0, load 15.7, predict 6.0, save 54.4 |
| new `hrrr_fast` | wrfnat ranges, AWS S3, 8 fetch threads, first case (pays numba compile) | 15 x 8 | 201 | fetch 46.8, inference 7.2, Bourgouin 12.1, write 5.5 |
| new | AWS S3, 8 fetch threads, repeat 01:30 | 15 x 8 | 138 | fetch 44.4, inference 3.3, Bourgouin 3.4, write 5.0 |
| new | Google mirror (gcs), 8 fetch threads | 15 x 8 | 67 | fetch 8.2, inference 6.5, Bourgouin 3.3, write 5.3 |
| new | AWS S3, 1 fetch thread | 15 x 8 | 440 | fetch 192.4 |
| new | gcs, 1 fetch thread | 15 x 8 | 111 | fetch 33.4 |

Like for like on AWS (the original's only source) the new pipeline takes 138-201 s against 267.6 s, 1.3-1.9x faster, and
201 s includes the one-time compilation; with the Google mirror it is 67 s, 4.0x. Caveats: (1) download dominates in both
and varied strongly between runs and times of day (the original download median was 81-151 s depending on the run, the new S3
fetch 44-192 s), so the wall ratios mix network state with code; the runs were made on different days and not back to
back. (2) The two pipelines do different work: the original writes full netCDF files with 40-level pressure data and
interpolated profiles (14 GB), the new one writes the two Zarr stores (stored sizes in `compression_local.md`) and also runs
Bourgouin. (3) Process counts differ by one (16 vs 15). (4) The per-hour stages the new README lists (inference + Bourgouin + write) sum to 11.7 s on the warm S3 repeat and
15.1 s on gcs (24.8 s in the first case, with compilation), against 84.8 s for the original's non-download stages
(load_data + convert + transform + predict + grid + save, 16-process run); the new README does not list the decode and
column-building stages, so this is not a full per-hour compute comparison.
