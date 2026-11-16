# End-to-end pipeline: local (laptop) stage timings, memory and verification

**These are laptop numbers, not Derecho numbers.** Machine: Apple M4 Pro, 14 cores (10 performance + 4
efficiency), 48 GB, macOS 26.7.2, shared with other agents' jobs during every run (1-minute load average
9–13 at the start of the timing runs). Python 3.12.12 in a private uv venv: numpy 2.5.5, numba 0.68.1,
zarr 3.4.2, cast-value-rs 0.5.1, eccodes 2.50.1 (eccodeslib 2.50.0.34), obstore 0.12.1, pyproj 3.8.0,
torch 2.15.0, onnxruntime 1.32.0. Engine: torch on CPU. Bourgouin grid top 8 km (34 thermo levels read,
27 wind levels). Data: hrrr.20231226 t00z f11–f14.

## 1. Stage times per forecast hour (`benchmarks/pipeline_benchmark.py run`)

Source = a sparse local mirror of the needed byte ranges (`pipeline_benchmark.py mirror`; f11/f13 fetched
from the bucket, f12 written from the cached reader bytes), so the network is out of the loop and the same
3 hours (f11–f13) are processed for each `processes x threads` layout. Writers as of this run: 0.1-unit int16
soundings with the representability check, and the gridded store finalized (sharded) once per run by the
parent.

```
PYTHONPATH=. python benchmarks/pipeline_benchmark.py run --config config/hrrr_fast.yml --source <mirror> \
    --weights <scratch>/weights/ptype_weights.npz --engine torch --init 2023122600 --hours 11-13 \
    --layouts 1x4 1x8 1x12 3x4 --work-dir <scratch>/bench_work2 --out <scratch>/bench/mirror_layouts_v2
```

| layout PxT | cores | hours | wall s | wall s / hour | fetch | decode | columns | inference | bourgouin | rotate | write | hour total | peak RSS GB / worker | MiB fetched / hour | finalize s (per run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x4 | 4 | 3/3 | 50.7 | 16.89 | 0.07 | 0.63 | 0.15 | 0.99 | 9.77 | 0.01 | 3.03 | 14.78 | 9.41 | 279 | 2.86 |
| 1x8 | 8 | 3/3 | 34.9 | 11.62 | 0.01 | 0.36 | 0.09 | 0.89 | 5.20 | 0.01 | 2.99 | 9.65 | 9.54 | 279 | 2.76 |
| 1x12 | 12 | 3/3 | 32.7 | 10.90 | 0.01 | 0.36 | 0.10 | 0.94 | 4.47 | 0.01 | 3.06 | 9.10 | 9.72 | 279 | 2.86 |
| 3x4 | 12 | 3/3 | 26.5 | 8.83 | 0.03 | 0.93 | 0.50 | 1.74 | 11.44 | 0.02 | 5.97 | 20.94 | 6.96 | 279 | 2.83 |

Stage columns are medians over the 3 hours of the per-hour stage seconds inside a worker. "wall s / hour" is
wall / hours and includes worker start-up and model load (~7 s: torch import in each spawned process) and the
parent's finalize (~2.8 s), which a 3-hour run cannot amortise; on a 19- or 49-hour run they matter far less.

Reading the table:

* **Bourgouin is the dominant stage** (≈ 5 s of ≈ 9.7 s per hour at 8 threads). Almost all of it is the
  wet-bulb temperature (Normand's rule with an RK4 moist adiabat) at 33 levels × 1.9 M columns; the 5 km grid
  costs 3.3 s and 8 km 5.0 s on 8 threads (`bourgouin_depth_local.md`). See the possible speed-up noted in
  `docs/hrrr_fast.md` (wet bulb is only needed below the highest level with T > 0 °C).
* Decode (ecCodes, 201 messages) 0.36 s, columns 0.09 s, inference 0.9 s, rotation 0.01 s: small.
* Write ≈ 3 s per hour; most of it is the sounding store (1.9 M × 105 values).
* Throughput: on 12 cores, 3 processes × 4 threads beat 1 × 12 (8.8 vs 10.9 s per hour of wall time) even
  though each hour takes 2.3x longer inside a worker, because whole hours overlap perfectly while the stages
  of one hour do not scale (Bourgouin 9.8 → 4.5 s for 4 → 12 threads, write not at all). This is the reason
  for the process-per-hour design; Derecho's scaling study (section 5) should confirm it at 128 cores.

### Peak memory per worker

`peak_rss_bytes` is the worker's `ru_maxrss` after the hour (lifetime peak, so model load included).

| writers version | peak RSS GB / worker (1x4, 1x8, 1x12, 3x4) | write s (1x8) |
|---|---|---|
| int32 soundings, no representability check, no finalize (first run, `mirror_layouts`) | 4.18, 4.30, 4.71, 3.96 | 1.71 |
| int16 + `check_representable` + finalize (current, table above) | 9.41, 9.54, 9.72, 6.96 | 2.99 |

The ~5 GB increase comes from `writers.check_representable` called by `write_sounding_hour` on the whole
(1,905,141 × 105) hour: it makes float64 copies of the data (`np.asarray(data, dtype=np.float64)`, the finite
subset, and `np.rint(...)`), ≈ 1.6 GB each. The fix proposed to the writers' owner (check only the finite
min and max, which is sufficient because rounding is monotonic) would bring the peak back to ≈ 4.5 GB. Until then
`memory_per_process_gb` is **12** (measured 9.7 GB + margin): on a 235 GB Derecho CPU node with
`memory_fraction 0.8` that allows 15 workers (vs 16 by cores at 8 threads); after the fix 6 GB would do.

## 2. One hour straight from the NOAA bucket (`--source s3`, layout 1x8, f14)

Earlier writers version (int32 soundings, no finalize); only fetch differs materially from section 1.

| fetch | decode | columns | inference | bourgouin | rotate | write | hour total | peak RSS GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27.82 | 0.52 | 0.38 | 0.90 | 5.21 | 0.01 | 2.12 | 37.24 | 4.01 |

293,342,932 bytes (280 MiB) in 201 messages and 74 HTTP requests (obstore coalescing, 1 MiB) took 27.5 s =
10.2 MiB/s: **on this laptop the network dominates** (the reader benchmark measured 14–17 MiB/s on a better
moment). The two mirror downloads of f11/f13 ran at 19.1 and 27.5 MiB/s. On Derecho the link and the number
of concurrent workers fetching decide; the scaling benchmark measures it.

## 3. End-to-end verification (3 real hours, local upload backend, local index)

Commands (paths shortened; `run` = `applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600
--forecast-hours 11-13 --source <mirror> --weights <w> --out-dir <work> --upload local --upload-root <bucket>
--keep-local --processes 3 --threads 4 --engine torch`):

1. f13's idx truncated on purpose → **run 1**: f11, f12 done, f13 `failed`
   (messages absent from the idx), run status `incomplete`, exit 1, nothing finalized, uploaded or indexed.
2. idx restored and **f12's marker deleted** → **run 2**: log "hours already complete, skipping [11]", then only
   f13 and f12 were processed; "gridded store finalized" (3.08 s), "run uploaded and indexed"; exit 0.
3. **run 3**: "run already in the index; skipping", status `skipped_indexed`, nothing read.

Index written (`<bucket>/ptype/inits.json`):
`{"version":1,"inits":[{"init":"2023-12-26T00:00Z","forecast_hours":[11,12,13],"store":"2023/12/26/hrrr_ptype_20231226_0000.zarr","soundings":"soundings/2023/12/26/hrrr_soundings_20231226_0000.zarr"}]}`.
Uploaded: gridded store 12 MB in 161 objects (finalize report: 144 objects still < 128 KiB with only 3 of 49
hours filled), soundings 383 MB. The manifest records both invocations, per-hour status/timings/bytes/levels,
the finalize result and seconds, the upload commands, the commit and package versions.

**Independent recomputation of f12** (scratch script `verify_hour.py`: decode the cached bytes with
`reader.decode`, `columns.build_soundings` on the whole grid, torch engine, the 0–8000 m Bourgouin sounding in ONE
pass (no row chunks), `columns.rotate_grid_to_earth`) against the uploaded, finalized stores read with zarr:

| quantity | max abs difference | tolerance from storage |
|---|---|---|
| ML_rain/snow/icep/frzr, ML_u | 0.0019685 | 1/508 = 0.0019685 (uint8 step 1/254, round to nearest) |
| BOURG_rain/snow/icep/frzr | 0.0019685 | same |
| BOURG_class, CRAIN/CSNOW/CICEP/CFRZR, precip_mask | 0 mismatching cells | exact |
| prate | 0.0 | exact (float32 + zstd) |
| sounding (T, Td, u, v earth-relative, p) | 0.0500 (p: 0.05005) | 0.05 (int16 step 0.1) + float32 rounding of ~1000 hPa |
| level-1 ML_rain vs 2x2 mean of stored level 0 | 0.0030 | quantisation of both levels |
| ML argmax class agreement | 99.987 % | ties broken differently after quantisation |

Columns short of the grid top: 0 (network) and 0 (Bourgouin). An earlier run of the same check against the
pre-finalize/int32 layout gave the same probability errors and a sounding error of 0.0051.

**Automatic level top-up on real data**: f12 run in-process with `n_levels=20` and a forced first guess of 28
thermo levels logged two top-ups (wind 20→23→26, thermo 28→31→34; the minimum top-level AGL was 3,414 m and
then 4,443 m for the wind levels) and finished with 26/34 levels, minimum top heights 5,516 m / 8,844 m, 0 short
columns.

## 4. Not measured here

GPU workers (no NVIDIA GPU on this machine), Linux/x86 numba and BLAS behaviour, more than 3 concurrent
workers, the `aws s3 sync` backend against the real bucket (only its command lines were unit-tested), S3
conditional writes of the index (only `MemoryStore` and the local backend were exercised), runs longer than 3
hours.

## 5. On Derecho

The same benchmark runs unchanged; see `docs/hrrr_fast.md` ("Benchmarks on Derecho") for the scaling command
(1, 2, 4, …, 128 total cores).
