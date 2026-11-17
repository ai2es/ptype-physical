# Derecho download throughput for HRRR range reads (2026-11-16)

Everything below was measured by `benchmarks/network_benchmark.py` (raw records: `network_*.json` in this directory;
the JSON also holds per-fetch times). Each fetch is the pipeline's message selection for one forecast hour of
HRRR 2023-12-26 00Z (`select_messages` with the production level counts, 72-75 ranges, about 0.30 GB), downloaded with
one `obstore.get_ranges` call (default `coalesce` 1 MiB, obstore 0.12.1) and discarded. Throughput is decimal MB/s of
bytes received (`agg`) over the time from a common start to the last finish of any process. N processes each fetch a
different forecast hour; a process starts another file only if less than 30 s have elapsed (so large N is mostly one
file per process). Times are Mountain time.

Jobs: CPU sweep 8208955 (killed at the 30 min walltime after `threads8`), sources/knobs 8209244, 2-node 8209245,
GPU 8209246. Two earlier submissions of the sources and 2-node jobs were deleted within a minute of starting
because they would have overlapped the CPU sweep (see "Caveats"); they produced no data.

## Main finding: throughput changed by 1-2 orders of magnitude with the time of day, independent of our load

The same default configuration (S3Store, 8 processes) on the same node type gave:

| time | job / node | 8 processes, default settings |
|---|---|---|
| 15:06 | CPU 8208955 | 16.5 MB/s |
| 15:14 | CPU 8208955 | 32.9 MB/s |
| 15:43 | CPU 8209244 | **407 MB/s** (1 process alone: 106 MB/s, fetch 2.0-4.1 s) |
| 15:54 | CPU 8209244 (coalesce 0) | 29.5 MB/s |
| 17:26 | 2-node job, node A alone | 11.5 MB/s |
| 17:51 | GPU 8209246 | 11.9 MB/s |
| 17:59 | GPU 8209246 | 10.6 MB/s |

There were two regimes. In the "slow" regime (most of the day) a default process gets 1-3 MB/s no matter what else is running,
so aggregate throughput is simply proportional to the number of processes. In the "fast" regime (about 15:43-15:44) a
single process got 106 MB/s and 8 processes 407 MB/s, matching the 108 MiB/s seen earlier today.

## 1. Aggregate throughput vs processes, one node (AWS S3Store, defaults)

CPU node dec2040 (job 8208955, 15:00-15:12, slow regime):

| procs | files | wall s | agg MB/s | per-process MB/s (min/med/max) | fetch s (min/med/max) |
|---|---|---|---|---|---|
| 1 | 1 | 91.0 | 3.1 | 3.1 | 91.0 |
| 2 | 2 | 132.4 | 4.3 | 2.1/2.2/2.3 | 125.5/129.0/132.4 |
| 4 | 4 | 132.7 | 8.6 | 2.1/2.3/2.6 | 109.3/125.2/132.7 |
| 8 | 8 | 139.5 | 16.5 | 2.0/2.4/2.5 | 113.3/120.8/139.5 |
| 16 | 16 | 145.6 | 31.9 | 2.0/2.6/5.5 | 52.7/112.1/145.6 |
| 32 | 32 | 167.1 | 56.0 | 1.8/2.6/3.9 | 75.2/112.7/167.1 |

GPU node deg0026 (job 8209246, 17:39-17:57, slow regime, 64 cores):

| procs | files | wall s | agg MB/s | per-process MB/s (min/med/max) | fetch s (min/med/max) |
|---|---|---|---|---|---|
| 1 | 1 | 210.0 | 1.3 | 1.3 | 210.0 |
| 2 | 2 | 249.6 | 2.3 | 1.1/1.1/1.1 | 248.8/249.2/249.6 |
| 4 | 4 | 241.2 | 4.7 | 1.2/1.3/1.4 | 201.6/220.9/241.2 |
| 8 | 8 | 193.2 | 11.9 | 1.5/1.6/1.7 | 163.5/176.1/193.2 |
| 16 | 18 | 106.1 | 49.3 | 2.7/4.3/8.5 | 5.6/66.6/106.1 |
| 32 | 32 | 128.7 | 72.7 | 2.3/3.2/6.8 | 42.3/91.6/128.7 |

Both node types scale about linearly with the number of processes (1.1-3 MB/s per process) up to 8-16 processes. The
16-process GPU row contains the first fast fetches (min 5.6 s), i.e. the regime was starting to change. CPU and GPU nodes
behaved the same in the slow regime (per process 1.3-1.6 MB/s at 1-8 processes on both). 1, 2, 4, 8 processes were
measured only in the slow regime; 1 and 8 processes in the fast regime on the CPU node (above: 106 and 407 MB/s;
`src_aws` rows in `network_cpu_sources_8209244.json`).

## Within-process concurrency (obstore knobs)

obstore 0.12.1: `get_ranges` merges ranges closer than `coalesce` (default 1 MiB) and issues at most 10 requests at once
inside one call (hard-coded, stated in its docstring; not configurable). Other knobs are in `client_options`:
`allow_http`, `connect_timeout` (default 5 s), `timeout` (per request, default 30 s), `read_timeout`, `http1_only`
(HTTP/1 is the default), `http2_only`, `http2_keep_alive_*`, `pool_idle_timeout`, `pool_max_idle_per_host`,
`proxy_url`, `randomize_addresses`, `user_agent`; plus `retry_config`. Only the ones below were measured.

Threads each calling `get_ranges` on an equal share of the ranges (8 processes, CPU node, job 8208955, 15:14-15:18; the
default 8-process runs at 15:06 and 15:14 gave 16.5 and 32.9 MB/s):

| threads/proc | files | wall s | agg MB/s | per-process MB/s (min/med/max) | fetch s (min/med/max) |
|---|---|---|---|---|---|
| 2 | 20 | 80.1 | 72.6 | 10.1/11.1/23.1 | 2.0/16.2/60.5 |
| 4 | 14 | 63.3 | 64.1 | 8.8/10.4/14.1 | 15.3/29.7/36.7 |
| 8 | 16 | 46.7 | 99.4 | 12.2/13.8/16.9 | 6.0/21.3/35.2 |

Same on the GPU node (job 8209246): 8 procs x 8 threads = **97.7 MB/s** (fetch 17.5/21.3/26.5 s) at 17:59, whereas the
default 8-process run 50 s later (17:59:49) gave 10.6 MB/s (fetch 144-216 s) and the one 10 min earlier gave 11.9 MB/s.
A 9x gain, measured back to back, in the slow regime. In the fast regime on the CPU node, 1 process with 8 threads got
91.1 MB/s (job 8209244, 15:44) versus 106.3 MB/s with 1 thread 30 s earlier: threads did not help (or hurt slightly) when
the link was not the limit; only one such pair was measured.

Other knobs (8 processes, CPU node, job 8209244, slow regime): `coalesce=0` gave 29.5 MB/s (75 requests per file, versus
72-74 with the 1 MiB default; the gaps between wanted messages are mostly larger than 1 MiB, so coalescing hardly changes
the request count) - within the range of the default runs of that period (16.5-32.9), so no measurable effect.
`client_options={"http2_only": True}` **fails with S3Store** (`GoAway FRAME_SIZE_ERROR` on the first `.idx` GET); it
was never used successfully. `coalesce` of 16 MiB, `pool_max_idle_per_host`, `timeout`, `get_ranges_async` and a
single process with fewer threads were not measured.

## 3. Is bandwidth shared between nodes? (2-node job 8209245, 17:11-17:39, slow regime)

Node A = dec1864 (forecast hours f00-f15), node B = dec1867 (f16-f31), common start time, 16 processes each unless stated.

| step | node A agg MB/s | node B agg MB/s | total MB/s |
|---|---|---|---|
| A alone, 16 procs (17:11) | 15.1 | - | 15.1 |
| A and B together, 16 procs each (17:17) | 16.6 | 17.2 | 33.8 |
| B alone, 16 procs (17:22) | - | 21.2 | 21.2 |
| A alone, 8 procs (17:27) | 11.5 | - | 11.5 |
| A and B together, 8 procs each (17:31) | 10.0 | 11.1 | 21.1 |
| A and B together, 16 procs each, repeat (17:35) | 20.0 | 20.7 | 40.7 |

Running two nodes at once did not reduce either node's throughput (A: 15.1 alone vs 16.6 and 20.0 together; B: 21.2 alone vs
17.2 and 20.7 together; the variation between steps is the same size as the variation between neighbouring solo runs), and
the total doubled. At the aggregate levels reached here (up to 40.7 MB/s from two nodes) there was no sign of a
shared per-site cap. Higher aggregate loads across nodes were not tested (the fast regime was not active during this job).

## 4. Alternative sources (CPU node dec0868, job 8209244, 15:36-15:54)

Whole-file `curl`, different file per stream (`-m 60`; bytes are what arrived; each stream is a different 0.70-0.81 GB file):

| source | streams | wall s | agg MB/s | per-stream MB/s |
|---|---|---|---|---|
| AWS HTTPS (s3.amazonaws.com) | 1 | 60.1 (cut off at 60 s) | 0.6 | 0.6 |
| AWS HTTPS | 8 | 60.1 (cut off) | 29.9 | 0.8 1.0 1.0 1.0 1.6 2.0 9.9 19.8 |
| GCS (storage.googleapis.com) | 1 | 33.2 | 21.1 | 21.1 |
| GCS | 8 | 50.1 | 120.1 | 14.9 17.8 19.1 34.6 56.6 57.7 64.9 80.7 |

The same obstore range-fetch loop (pipeline selection) per source, 1 and 8 processes, run in this order:

| time | source / store | procs | files | wall s | agg MB/s | fetch s (min/med/max) |
|---|---|---|---|---|---|---|
| 15:38:55 | GCS, `GCSStore` anonymous | 1 | 2 | 33.0 | 17.2 | 6.0/16.5/27.0 |
| 15:39:32 | GCS | 8 | 26 | 43.2 | 175.6 | 3.4/8.2/22.2 |
| 15:40:18 | AWS, `HTTPStore` | 1 | 1 | 70.6 | 4.0 | 70.6 |
| 15:41:37 | AWS, `HTTPStore` | 8 | 8 | 106.2 | 21.6 | 70.2/88.3/106.2 |
| 15:43:27 | AWS, `S3Store` | 1 | 4 | 10.7 | 106.3 | 2.0/2.3/4.1 |
| 15:43:41 | AWS, `S3Store` | 8 | 32 | 23.0 | 407.1 | 2.0/3.4/13.3 |

During the slow AWS period (curl at 15:36-15:38 and the HTTPStore runs) GCS delivered 120-176 MB/s with 8 streams or
processes versus about 22-30 MB/s from AWS; single-stream curl was 21.1 MB/s from GCS and 0.6 MB/s from AWS. Then
AWS switched to the fast regime (15:43, 407 MB/s) and was faster than GCS had been. AWS `HTTPStore` and `S3Store` differ
only in the slow period by noise (4.0 vs 1.3-3.1 MB/s for one process); they were not measured in the same regime, so no
difference between them can be claimed. GCS and AWS were never measured in the same minute, and GCS was not measured in a
period when AWS was fast, nor with `threads>1` or at 16/32 processes (the job reached walltime before those).

GCS mirror versus AWS (`compare-sources`, same job): for f00, f01, f12, f24, f31 the `.idx` byte-offset columns are identical,
the `select_messages` plan (names, levels, offsets, ends) built from each `.idx` is identical, and the object sizes are identical
(e.g. f12: 790,950,105 bytes on both; ETags match in the HTTP headers). The idx *text* is not identical: a few variable names
differ (AWS `CLMR`, GCS `CLWMR`), which does not affect the selected variables. SHA-256 of the first 20 selected ranges plus the last
range of f12: 21 of 21 identical.

## 5. Variability

Default configuration, 8 processes (the third planned repeat did not run: job 8208955 was killed at its walltime before it):
16.5 and 32.9 MB/s at 15:06 and 15:14 (same job), 407.1 MB/s at 15:43 (job 8209244), 11.5 MB/s at 17:26 (node A, 2-node job),
11.9 and 10.6 MB/s at 17:51 and 17:59 (GPU job). Run-to-run variation within a regime was about 2x (16.5 vs 32.9; 11.9 vs 10.6);
between regimes about 40x. 16 processes: 15.1, 16.6/17.2, 20.0/20.7, 21.2 MB/s (2-node job).

## Inference (not directly measured)

* In the slow regime throughput was about proportional to the number of simultaneous connections: 1 process, 10 connections
  (obstore's limit per `get_ranges` call) gave 1.3-3.1 MB/s; 8 processes gave about 8x that; with 8 threads each, a process
  can open up to about 70 connections and got 12-14 MB/s per process (97.7-99.4 MB/s for 8 processes). That is about 0.1-0.2 MB/s per
  connection in every case (e.g. 32 processes x 10 = 320 connections: 56 MB/s). This looks like a per-connection (per-flow)
  limit on the path to the AWS endpoint, not a per-process, per-node or shared-gateway limit: the 2-node test showed no
  sharing, and CPU and GPU nodes were identical. The measurements do not show what causes the per-connection limit (for example
  a shaper or loss on the route to S3 us-east-1 at busy times).
* The fast regime (single process 106 MB/s) shows the node and the process are able to sustain at least 407 MB/s when the path is not in that state.
* GCS did not show the per-connection limit while AWS did, at least in the period of 15:36-15:41.

## Recommendation for the pipeline (based on the numbers above)

1. Increase per-process concurrency: split each file's ranges over several threads (8 threads x `get_ranges`; ~70 concurrent
   requests). Measured 9x on the GPU node (97.7 vs 10.6-11.9 MB/s at 8 procs) and 2-6x on the CPU node (64-99 MB/s vs 16.5-32.9 MB/s); it did not
   help when the link was fast (91 vs 106 MB/s, one pair), so it is cheap insurance, not a cost.
2. Run about 8 processes per node: 8 procs x 8 threads reached ~100 MB/s in the slow regime; the process count by itself scales
   linearly in the slow regime (32 procs: 56-73 MB/s) so more processes are a worse lever than more connections per process,
   and 8 processes already hit 407 MB/s in the fast regime. Other jobs on other nodes can run at the same time (no cross-node sharing seen).
3. Try the Google mirror as the primary source (offsets, sizes and the 21 checked ranges are identical): it gave 120-176 MB/s with
   8 processes while AWS was giving 22-30 MB/s. Do not read it as "GCS is always faster": AWS was faster than that when it was in its fast
   state. A fallback to AWS on error/timeout is cheap because the keys and offsets are the same.
4. No need to stage whole files: the selected ranges are about 0.30 GB out of 0.70-0.81 GB per file, and the fast regime reached
   106 MB/s per process (3 s per file). Staging would cost 2.5x the bytes. If an existing GLADE copy (below) covers the dates needed, it would remove the network entirely; its coverage was not checked.
5. Keep obstore's default 30 s per-request `timeout` in mind: in the slow regime whole-file `get_ranges` calls ran 100-250 s;
   a request that hits the timeout is retried silently by obstore, and the benchmark does not log retries, so whether any timeouts occurred is unknown.

## GLADE search (read-only, `find -maxdepth 3`)

* `/glade/campaign/ral/nral0003/grib/HRRR-wrfnat/` (year directories `2020`, `2021` listed; found by name; contents not crawled) and
  `/glade/campaign/ral/nral0003/grib/HRRR-AK-wrfnat`.
* `/glade/campaign/ral/nsap/zhangyx/HRRR_Conus` (permission denied), `/glade/campaign/ral/jntp/HRRRE`, `/glade/campaign/ral/wsap/brummet/HRRR-HI`.
* `/glade/campaign/mmm/parc/schwartz/HRRRE` and `HRRR_v4experimental_geo_em.d01.nc`, `/glade/campaign/mmm/parc/sobash/HRRRv4`.
* Scratch (other users): `/glade/derecho/scratch/ananyo/herbie_data/hrrr` (dates 20221230, 20221231, 20230101, 20260920, with `subset_*` wrfnat files),
  `/glade/derecho/scratch/ananyo/e2s_cache/hrrr`, `/glade/derecho/scratch/jmantovani/metdata/HRRR/2022033000/hrrr.t00z.wrfnatf*.grib2`,
  `/glade/derecho/scratch/hshin/GTG4/hrrr`.
* Nothing under `/glade/campaign/collections` matched within depth 3. Whether the `nral0003` archive contains 2023-12-26 or later dates
  was not checked (only the years 2020 and 2021 appeared in a depth-1 listing of the first six entries).

## Caveats

* The network state changed during the day, so cross-test comparisons are only valid between runs minutes apart; the claims above
  are limited to such pairs. No test was repeated three times back to back.
* Our own jobs' concurrency: the sources and 2-node jobs were first submitted together with the CPU sweep and started at once; they
  were deleted after about a minute (qdel) and re-run later in a dependency chain. The first sweep point (1 process, 3.1 MB/s,
  15:00:02) may have overlapped those, but later points did not and were no faster, so it did not change the picture.
* Another of our jobs (`bl_scaling`, 8207605 14:32-15:32 and 8208611 from 15:39) was running on other nodes during some of the
  tests; whether it or other users' traffic caused the slow regime is not known.
* Not measured: GCS with threads or at 16/32 processes; 3 back-to-back repeats; `pool_max_idle_per_host`, `timeout`,
  async API; TCP-level diagnostics (RTT, loss, which S3 IPs were used); fast-regime GPU node; 2-node test in the fast regime.
