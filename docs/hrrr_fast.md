# Fast HRRR precipitation-type pipeline

From HRRR model output to the two Zarr stores the website reads: per forecast hour, the evidential neural
network's four precipitation-type probabilities and uncertainty, the revised Bourgouin method (Birk et al.
2021), the HRRR categorical flags, the precipitation rate, and a 0–5000 m sounding at every grid column.

Code: `ptype/hrrr_fast/` (`reader`, `columns`, `engine`, `bourgouin`, `writers`, `index`, `runlist`, and the
driver `pipeline` + `config`), `applications/hrrr_ptype_run.py` (one node), `applications/hrrr_ptype_submit.py`
(PBS launcher), `config/hrrr_fast.yml` (settings), `benchmarks/pipeline_benchmark.py`.

Measured results this document relies on:
[reader](../benchmarks/results/reader_local.md),
[columns and Bourgouin kernels](../benchmarks/results/columns_local.md),
[engines](../benchmarks/results/engine_local.md),
[stores and compression](../benchmarks/results/compression_local.md),
[Bourgouin grid depth](../benchmarks/results/bourgouin_depth_local.md),
[end-to-end pipeline](../benchmarks/results/pipeline_local.md),
[upload and bucket](aws_upload.md). All "local" numbers are from a shared laptop, not Derecho.

## 1. What happens to one forecast hour

| stage | what | why this way |
|---|---|---|
| fetch | `.idx` sidecar, then ONE `obstore.get_ranges` call for the needed GRIB2 messages | see 2.1 |
| decode | ecCodes in a thread pool, straight into preallocated cubes | see 2.2 |
| columns | heights AGL = `HGT − HGT_sfc`; `columns.build_soundings` to 0, 250, …, 5000 m with grid-relative winds; `columns.network_inputs` | see 2.3, 2.4 |
| inference | `engine.make_engine(...)`; scaler and evidential head are inside the engine | see 2.5 |
| bourgouin | T, Td, p on 0…`bourgouin_top_m` (8 km), `revised_bourgouin_from_sounding` | see 2.6 |
| rotate | stored sounding winds → earth-relative, in place | see 2.4 |
| write | one hour slice of the gridded store and of the sounding store, then the hour's marker | see 2.7 |

Per run, after its last hour, the parent process finalizes the gridded store (`writers.finalize_ptype_store`),
uploads both stores, and only then updates `ptype/inits.json`.

Predictions are made at **every** grid point. `precip_mask` (any HRRR CRAIN/CSNOW/CICEP/CFRZR = 1) is stored
next to them and never used to blank anything; the website decides what to shade, and an analysis can ask
what the model says where HRRR has no precipitation.

Stored per hour (gridded, pyramid of 3/6/12/24 km): `ML_rain`, `ML_snow`, `ML_icep`, `ML_frzr`, `ML_u`,
`BOURG_rain`, `BOURG_snow`, `BOURG_icep`, `BOURG_frzr` (the paper's independent, un-normalised probabilities
as fractions 0–1), `BOURG_class` (0–3, 254 = no type, 255 = missing), `precip_mask`, `CRAIN`, `CSNOW`, `CICEP`,
`CFRZR`, `prate`. Soundings: T, Td, u, v (earth-relative), p at 0–5000 m every 250 m.

## 2. Design choices and the reasoning behind them

### 2.1 Why wrfnat and byte ranges

The network was trained on height-interpolated profiles; HRRR's native hybrid levels (wrfnat) are dense near
the ground (level 1 ≈ 10 m AGL, ~20 levels below 5 km), so interpolating to 250 m spacing loses nothing, and the
file has the surface fields we need. A wrfnat file is ~790 MB, but the idx lists every message's byte offset,
so only the needed messages are downloaded: 27 levels × 6 variables + 11 surface fields ≈ 250 MB (32 % of the
file), plus TMP/SPFH/PRES/HGT on levels 28–34 for the 8 km Bourgouin grid (+36–43 MB). obstore merges adjacent
ranges and issues them concurrently: 201 messages become 74 HTTP requests with no surplus bytes at
`coalesce = 1 MiB`; larger values pull 60–70 % more bytes for no gain on a bandwidth-bound link (reader
benchmark). The same code reads a local mirror (`LocalStore`), which is how tests and benchmarks avoid the
network.

The top hybrid level read must be ≥ the top of every height grid at **every** column, because interpolation
clamps above the top level and would silently corrupt the profile there. The pipeline checks this on the decoded
heights each hour; if any column is short it fetches `level_step` more levels (only the new messages), logs a
warning, records `rereads` in the manifest, and remembers the higher count for the worker's next hours.
Defaults: 27 levels for 5 km (level 25 is the first whose minimum AGL is ≥ 5 km over the Rockies in the measured
files), and the first guess for the Bourgouin top from the measured minimum-AGL table plus one level.

### 2.2 Why ecCodes in threads

ecCodes releases the GIL: 8 threads decode 192 messages 6.3x faster than one; pygrib and gribberish do not scale
with threads, and process pools lose to the cost of pickling ~9 MB per message (reader benchmark). Each message is
decoded directly into its slot of a preallocated `(levels, ny, nx)` cube, so no cube is ever stacked/copied.

### 2.3 The 0 m level policies

The training data built the 0 m level from the 2 m temperature and dewpoint, the 10 m wind and the surface
pressure (`scripts/precip_RAP_interpolate.py:79-83`, `scripts/rap_height_interp.py:88-94`; details in
`columns_local.md`). The pipeline uses exactly that (`surface_temperature: 2m`, `surface_wind: 10m`,
`surface_pressure: surface`, pressure linear in height, no dewpoint clip). Every policy is a config key so a
different choice can be tested without code changes.

### 2.4 Grid- vs earth-relative winds

HRRR GRIB winds are relative to the Lambert grid axes. The old inference code fed them to the network unrotated,
and nothing in the repository rotated the training winds, so the **network gets grid-relative winds**.
The **stored soundings are earth-relative** (east/north), because that is what a person reading a hodograph
expects; the store's root attributes give the rotation back. Rotating after vertical interpolation is exact (the
angle is constant in a column), so it is done once, in place, on the 21-level sounding (0.01 s per hour).

### 2.5 Engine choice

All engines take raw physical inputs; the scaler is folded into the first layer and the evidential head is in
the graph, so no pandas and no extra pass over the 1.9 M × 84 matrix. The default is `engine: auto`, which
means numba on CPU nodes and torch on GPUs. On the laptop torch was fastest (3.1 M rows/s on 14 threads; ONNX
2.4 M, numpy 2.3 M, numba 1.1 M; `engine_local.md`), but on a Derecho CPU node (128 cores, job 8207280) numba was
the only engine that kept scaling with threads: 0.21 s per full-CONUS hour at 128 threads, against 1.3-2.1 s for
torch, ONNX Runtime and NumPy, which stop scaling at 16-32 threads. At the driver's 8 threads per worker the four
are within about 25% (torch 2.07 s, ONNX 1.98 s, numpy 2.26 s, numba 2.51 s), so the choice is not critical there;
numba also has no dependency beyond numpy. On a GPU node each worker runs torch
on its own GPU (2.8).

### 2.6 Bourgouin height grid

The revised Bourgouin ProbIce needs the precipitation-generation (cloud) layer, which often extends above 5 km.
Bourgouin therefore runs on its own grid, 0…`bourgouin_top_m` every 250 m, with T, Td and p (Tw is computed inside
the kernel; winds are not used), while the network inputs and the stored soundings stay 0–5000 m. The deep grid
uses the same interpolation kernel and 0 m policies, so its 0–5000 m part is bit-identical to the network
sounding (unit-tested), and it is built in blocks of 128 grid rows to keep the temporary array ~200 MB.

Measured on a real winter hour (`bourgouin_depth_local.md`): with a 5 km top, 42 % of precipitating columns have a
generation layer cut off at the grid top, ProbIce changes for 11 % of them, and 0.13 % change dominant class; with
8 km nothing in the stored output changes by more than 0.008 relative to a 12 km grid, for +36 MB and ~+1.7 s
compute per hour. **Default: 8000 m.**

Bourgouin is the most expensive stage (5 s of ~10 s per hour at 8 threads), almost all of it the wet-bulb
temperature at 33 levels × 1.9 M columns. A possible speed-up in `bourgouin.py` (not done here, it is another
module): Tw ≤ T, so above the highest level with T > 0 °C every level is freezing, and the energy of that topmost
freezing layer is not used by the method; Tw is only needed up to the first level above the highest T > 0 °C
level (and its sign elsewhere is known), which in a winter hour is a small fraction of the levels.

### 2.7 Stores: chunking, sharding, compression

(Detailed numbers in `compression_local.md`.) Gridded store: one multiscale group per pyramid level, arrays
`(forecast_hour, y, x)` with a whole level per chunk, so the website reads one object per variable per map frame.
During a run, workers write per-hour objects (the "staging" layout: every hour maps to disjoint objects, so many
processes write one store without locks). After the last hour the parent calls `writers.finalize_ptype_store`
once, which rewrites each array into ONE shard per level spanning all hours (inner chunk = one hour), so each
object is large enough for S3 Standard-IA's 128 KB minimum billable size (2.9) and the site still reads one inner
chunk per variable. Probabilities and `ML_u` are uint8 steps of 1/254 behind a float32 array (`scale_offset` +
`cast_value`, decoded by the site's zarrita); flags uint8; `prate` float32 + zstd. Soundings: one shard per hour,
inner chunks of 1024 locations, scaled integers (default 0.1 units, int16; `sounding_encoding` in the config),
transposed and blosc-zstd bit-shuffled.

### 2.8 Parallelism inside one node

Forecast hours are independent, so the unit of scheduling is the hour: a pool of `processes` worker processes,
each handling whole hours (fetch → write), each limited to `threads_per_process` threads for every multi-threaded
library. The thread limits are put in the environment before the workers start (`NUMBA_NUM_THREADS`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`; BLAS sizes its pool at
import) and set again in each worker (`numba.set_num_threads`, torch `threads`), so processes × threads = cores
and nothing is oversubscribed. Workers are **spawned**, not forked: forking a parent that has started OpenMP/numba
/torch thread pools can deadlock the child. Each worker loads the model once and keeps it.

Why processes × small thread counts instead of one process using all cores: the stages of one hour scale
sub-linearly (Bourgouin 9.8 → 4.5 s from 4 to 12 threads, writes not at all), but independent hours overlap
perfectly; on 12 laptop cores 3 × 4 gave 8.8 s of wall time per hour vs 10.9 s for 1 × 12
(`pipeline_local.md`). Memory is bounded by the process count: one hour per worker, measured peak 9.7 GB per
worker with the current writers (4.3 GB before the writers' representability check; see the report for the fix),
hence `memory_per_process_gb: 12`. `processes: auto` = min(cores ÷ threads, 0.8 × available memory ÷ 12 GB)
→ 15 × 8 on a 235 GB, 128-core CPU node.

Runs are pipelined: hours of up to `max_runs_in_flight` runs are queued together and a finished run is finalized
and uploaded in a background thread of the parent, so the workers never wait for an upload.

GPU nodes: each worker process is pinned to one GPU by `CUDA_VISIBLE_DEVICES` (round-robin, e.g. 8 workers × 8
threads on 64 cores = 2 workers per A100) and runs torch there. This was chosen over a single long-lived GPU
inference server fed through shared memory because inference is < 10 % of an hour's work, the server would add a
640 MB inter-process copy per hour and a single point of failure, and per-worker GPUs need no IPC at all. A100
engine throughput is in `engine_local.md` once measured on Derecho; the GPU path was not run locally (no NVIDIA GPU).

### 2.9 Upload, storage class and the index

Order per run: finalize → upload both stores → update the index → (optionally) delete the local copy. The website
learns about a run only from `ptype/inits.json`, so the index must never point at data that is not completely
uploaded; any upload error leaves the index untouched and the local copy in place. A run with failed hours is not
finalized or uploaded (a re-run retries them); missing HRRR files are expected (archive gaps) and the index lists
only the hours actually written.

Backends: `local` (copy into a directory laid out like the bucket; tests) and `s3`: `aws s3 sync` with
`$AWS_PROFILE` (the credentials path documented in `aws_upload.md`), `--region us-east-1`, never `--delete`, two
passes per store with explicit content types (`zarr.json` → `application/json`, chunk/shard objects →
`application/octet-stream`; no Content-Encoding since Zarr already compresses), and an `aws sts
get-caller-identity` check before any compute so expired keys fail in seconds rather than after an hour.

**Storage class.** The Zarr objects are uploaded as **S3 Standard-Infrequent Access** (`storage_class:
STANDARD_IA`, passed as `--storage-class`): archived cases are read rarely, IA costs about half of S3 Standard
per GB-month, and reads are still immediate (no restore step). Caveats, *from AWS documentation, not tested*:
a per-GB retrieval fee on reads, a 30-day minimum storage charge per object, and a 128 KB minimum billable object
size (smaller objects, e.g. `zarr.json` documents and coarse-pyramid-level objects, are billed as 128 KB — the
reason for the finalize step). The index `ptype/inits.json` stays in **STANDARD**: it is rewritten after every
run, and every rewrite of an IA object would be billed for 30 days. It never goes through the sync (the sync
commands only target `.zarr` prefixes and exclude `*inits.json`); it is written by `index.update_index` with an
ETag-conditional put and no storage class.

### 2.10 Robustness and resume

A run directory `<work_dir>/<YYYYMMDDHH>/` holds the two stores, `status/fFF.json` per-hour markers,
`status/stores_created`, `status/finalized.json`, `grid.json` and `manifest.json`.

* An hour is done only when its marker exists; the marker is written after both stores were written, so a crash
  mid-hour just means the hour is redone (writes are whole-chunk overwrites).
* Re-running the same command skips runs already in the index and hours whose marker exists; a run interrupted
  before finalize resumes the missing hours, then finalizes, uploads and indexes. A run that was finalized but
  whose upload failed goes straight to the upload. Hours cannot be added to a finalized store (they are logged
  and left out); delete the run directory to rebuild such a run.
* Missing files (404 on idx/GRIB) are not retried and are recorded as `missing`; transient errors are retried
  with full-jitter exponential backoff (`fetch_retries`, `fetch_backoff_s`).
* The manifest records per-hour status, stage timings, bytes, levels and sanity counts, the finalize result and
  duration, the upload commands and index entry, the code commit (and whether the checkout was dirty), the
  package versions, the host and PBS job id, and the full config of every invocation.
* Logs are JSON lines (`log_json: true`) with `run`, `fxx`, `worker` and other fields as keys.

## 3. How to run

### One case (one node, interactive or batch)

```bash
source /glade/work/dgagne/ptype_fast/ptype-physical/benchmarks/pbs/common.sh   # venv, repo, NUMBA cache
export AWS_PROFILE=ai2es-testbed-user
python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600 --dry-run   # bytes, missing files
python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600              # 0-48, upload, index
python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600 --forecast-hours 0-6 --no-upload
```

`--processes/--threads/--engine/--device/--out-dir/--source/--bourgouin-top-m/--upload/--upload-root/--keep-local`
override the YAML. `--no-upload` leaves the (finalized) stores in `--out-dir`. Exit status 1 if any run did not end
uploaded+indexed (or complete with `--no-upload`).

### The archive

```bash
python -m ptype.hrrr_fast.runlist config/hrrr_fast_events.yml --out runs.txt
python applications/hrrr_ptype_submit.py --runs-file runs.txt --jobs-dir $SCRATCH/ptype_fast/jobs --dry-run
python applications/hrrr_ptype_submit.py --runs-file runs.txt --jobs-dir $SCRATCH/ptype_fast/jobs
```

The launcher drops runs already in `s3://wwe-ptype/ptype/inits.json` (or `--done-list` if the index is not
readable) and runs of its own jobs that `qstat` still lists, assigns node types (free CPU nodes first, then free
GPU nodes, then the type whose queue is shorter; `--node-type` overrides), packs runs into jobs of ≤ 2 h from
`--seconds-per-forecast-hour-cpu/-gpu` (replace the defaults 5/8 s with the scaling-benchmark result) or
`--runs-per-job`, writes one PBS script per job (`-A NAML0001 -q main`, `select=1:ncpus=128:mpiprocs=1` or
`select=1:ncpus=64:mpiprocs=1:ngpus=4`, sourcing `benchmarks/pbs/common.sh`) and submits at most `--max-jobs`
(10). Run it again to submit the next batch.

### Resume

Run the same command again. Nothing is processed twice: indexed runs and completed hours are skipped (see 2.10).
To force a run to be redone, remove its entry from the index (or use `--ignore-index`) and delete its run
directory.

## 4. Benchmarks on Derecho

`benchmarks/pipeline_benchmark.py` runs unchanged on Derecho:

* `mirror`: download the needed byte ranges of some hours once into a sparse local mirror (so the scaling study
  measures compute, not repeated downloads);
* `run --layouts PxT ...`: end-to-end on those hours for each layout, JSON + markdown with wall time per hour,
  per-stage medians, peak RSS per worker and finalize time;
* `bourgouin-depth`: the grid-depth study (needs a cache with ≥ 39 levels from `reader_benchmark.py fetch`).

Commands are in the final report of this work and in `pipeline_local.md`.

## 5. Known limitations

* Per-worker memory is currently dominated by the writers' representability check (2.8).
* The GPU path, the `aws s3 sync` backend against the real bucket and conditional index writes on real S3 were
  not exercised locally.
* Hours cannot be added to a run after it has been finalized (2.10).
