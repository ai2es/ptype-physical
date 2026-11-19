# Casper measurements for realtime inference

Jobs 6735820 (AMD EPYC 9554 "Genoa") and 6735821 (Intel Xeon Gold 6240 "Cascade Lake"), 2026-11-18,
`benchmarks/pbs/casper_measure.pbs`, 20 cores and 60 GB each, queue `casper` (routed to `htc`), commit `f167a44`.
Inputs: HRRR 2026-11-17 12Z (single hours) and 13Z (the 18-hour run). Nothing was uploaded.

## Environment

The Derecho environment does not load on Casper: it was built on `/usr/bin/python3.12`, which Casper does not
have (the first two jobs, 6735701 and 6735702, stopped at `No module named 'numpy'`). `venv_casper` is built on
a uv-managed Python 3.12.13 with the same package versions (constraints taken from the Derecho environment,
CPU-only torch 2.15.0) and imports the pipeline on both CPU types.

## Network from a compute node

| Test | Genoa | Cascade Lake |
|---|---|---|
| First 200 MB of one wrfnat file from Google, one stream | 78 MB/s | 23 MB/s |
| Same from AWS | 63 MB/s | 58 MB/s |
| Anonymous GET of `ptype/inits.json` from the output bucket | 200 in 0.45 s | 200 in 0.43 s |

Both HRRR buckets and the output bucket's endpoint are reachable. One stream each is one sample; inside the
pipeline the fetch of a forecast hour took about a second from either source (below), so the source does not
matter for speed here. (On Derecho the same fetch was the bottleneck and Google was 4 to 6 times faster.)

## Speed

Seconds per forecast hour are medians of the per-hour `total` the pipeline records (fetch, decode, columns,
network, Bourgouin, write).

| Test | Genoa | Cascade Lake |
|---|---|---|
| One hour, 1 worker x 8 threads, Google | 12.4 s (fetch 1.1) | 13.6 s (fetch 0.7) |
| One hour, 1 worker x 8 threads, AWS | 12.0 s (fetch 1.3) | 13.9 s (fetch 1.1) |
| 18 h run (19 hours), 2 workers x 8 threads: wall time | 122 s | 145 s |
| same: per hour | 11.0 s | 13.2 s |
| 18 h run, 4 workers x 4 threads: wall time | 94 s | 114 s |
| same: per hour (slowest hour) | 16.1 s (17.6) | 19.9 s (20.6) |
| Peak memory of one worker | 4.3 GB | 4.4 GB |
| Consolidating the gridded store of the run | 7.7 s | 10.2 s |

The first hour of a job also pays about 25 to 35 s of start-up (imports and compiling the numba kernels):
48 s and 45 s for the warm-up hour against 21 s and 23 s wall for the next single-hour runs.

## Queue wait

Four jobs submitted at 15:38 and 15:51 UTC each started 3 seconds after submission. That is one time of day;
the realtime jobs are submitted ahead of the data so that a longer wait costs nothing (see below).

## What follows for the realtime jobs

- HRRR files arrive a minute or more apart (`arrival.md`); one worker of four threads does an hour in 16 to
  20 s. So a realtime job is **6 cores, 24 GB, one worker of four threads**: it keeps up with a factor of three
  to spare, and a catch-up job with two whole runs (up to 98 hours) needs about half an hour.
- Jobs are submitted 45 minutes after initialization, 7 to 9 minutes before the first file.
- Estimated cost, to be replaced by a measured day: an 18 h cycle's job lives about 50 minutes and a 48 h
  cycle's about 75, so 20 x 5 + 4 x 7.5 = about 130 core-hours a day. Most of that is waiting for files.
