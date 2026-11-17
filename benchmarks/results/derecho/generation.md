# Generating the forecasts: pilot and 2025-26 events (Derecho, 2026-11-17 UTC)

Every hourly HRRR run in eleven date windows was run through `ptype.hrrr_fast` on Derecho CPU nodes and uploaded
to `s3://wwe-ptype`. Job figures are from PBS accounting (`qhist`, 652 jobs named `ptype_20*`); bucket figures are
from `aws s3 ls --recursive` after the last job; the index figures are from `ptype/inits.json`.
Commit `7dd74ff`, one 128-core node per job, 15 workers x 8 threads, source `gcs`, `upload_backend: boto3`.

## What is in the bucket

| | Runs | Forecast hours | Gridded stores | Sounding stores | Objects |
|---|---|---|---|---|---|
| 2023-12-21 to 27 (pilot) | 168 | 4,032 | 15.7 GB | 538.6 GB | 32,088 |
| 2025-26 (ten windows) | 1,872 | 44,928 | 148.8 GB | 5,952.2 GB | 357,555 |
| Total | 2,040 | 48,960 | 164.5 GB | 6,490.8 GB | 389,644 (with the index) |

- 6,655 GB in all, 3.26 GB per run on average: 3.18 GB of soundings and 0.08 GB of gridded fields.
  The sounding store is 97.5% of the bytes.
- The index lists 2,040 runs on 85 days, 24 per day on every day, and each run lists all of its forecast hours
  (0-48 for the 00, 06, 12 and 18 UTC runs, 0-18 for the rest).
- Windows (UTC dates of initialization): 2023-12-21 to 27; 2025-11-07 to 11; 2025-11-22 to 12-03;
  2025-12-10 to 20; 2025-12-26 to 30; 2026-01-21 to 27; 2026-01-29 to 02-02; 2026-02-13 to 24;
  2026-03-08 to 17; 2026-03-30 to 04-05; 2026-05-03 to 06.

Not checked: that every object of every store is present and readable (only the index and the totals were
checked for the 2025-26 runs; the pilot stores were opened from the website), and the storage class of the
2025-26 objects (the pilot's were all Standard-IA except the index).

## What it cost

| Phase (2026-11-17 UTC unless noted) | Jobs | Node-hours | Exit 0 | Walltime | Exit 1 | Project |
|---|---|---|---|---|---|---|
| Pilot, 03:05-05:08 | 17 | 4.6 | 16 | 1 | 0 | NAML0001 |
| First launch, 05:14-06:17, about 96 nodes at once | 188 | 83.3 | 28 | 157 | 3 | NCIS0013 |
| Throttled to 10 jobs, 06:12-10:27 | 144 | 14.0 | 144 | 0 | 0 | NCIS0013 |
| AWS login expired, 10:22-13:02 | 290 | 0.5 | 0 | 0 | 290 | NCIS0013 |
| Final, 10 jobs, 14:46-15:28 | 13 | 4.1 | 13 | 0 | 0 | NAML0001 |
| Total | 652 | 106.6 | 201 | 158 | 293 | 97.9 NCIS0013, 8.7 NAML0001 |

106.6 node-hours is 13,645 core-hours.

The two phases that started from nothing and ran cleanly give the cost of doing it right:

| | Runs | Node-hours | Node-hours per run | Node-seconds per forecast hour |
|---|---|---|---|---|
| Pilot | 168 (4,032 forecast hours) | 4.6 | 0.027 | 4.1 |
| Final | 127 (3,043 forecast hours) | 4.1 | 0.032 | 4.9 |

The pilot figure includes its first job, which hit the walltime while uploading with `aws s3 sync` (see below).
Some of the final 127 runs may have had hours left on scratch by jobs killed earlier, so 0.032 may be low for a
cold start. At 0.03 node-hours per run the 1,872 runs of 2025-26 would have taken about 56 node-hours; they
took 101.9. The "throttled" row cannot be read as a rate: most of its runs had already been computed on
scratch by the first launch and only needed finalizing and uploading.

## What went wrong, and what was changed

1. **Serial upload was the bottleneck (pilot).** The first pilot job computed 10 runs in 7 minutes and then ran
   out of its walltime uploading them one at a time with `aws s3 sync` at 5-15 MB/s. `Boto3Uploader`
   (8 files x 8 parts, 4 runs at once) replaced it; the next job finished 11 runs, compute and upload, in 378 s.
2. **The launcher resubmitted runs that were still queued.** Derecho's `qstat` is cached, so jobs submitted a
   minute earlier did not show as active and 10 duplicates went in. The launcher now treats anything in its
   ledger from the last 10 minutes as in flight.
3. **Too many nodes at once (first launch).** All 188 jobs were submitted together and about 96 ran
   concurrently. Downloads from the Google mirror failed with connection resets (about 15,000 retries in the
   logs) and uploads failed with SSL EOF and write timeouts; 157 jobs hit their 30-minute walltime and 83.3
   node-hours produced 170 uploaded runs. Batches of 4 to 11 jobs the same night uploaded every run. Each
   node opens several hundred connections (15 workers x 8 fetch threads, plus the upload pool), so node count
   multiplies the load on the outbound path. The earlier two-node test did not show this. The cause on the
   network side was not investigated with NCAR. **Run no more than about 10 of these jobs at a time**:
   `--max-jobs` defaults to 10 and `benchmarks/pbs/throttle_submit.sh` keeps the queue topped up to that.
4. **Expired AWS login.** The SSO login used by the jobs expired at about 10:20 UTC. The loop then running
   tested credentials with `aws sts get-caller-identity`, which kept passing for 3.5 hours, so it went on
   submitting: 290 jobs, each of which stopped at its start-up credential check after a few seconds (0.5
   node-hours in all, nothing computed or uploaded). The committed script tests the bucket instead and stops
   on the first failure.
5. **Project overspent.** After the login was renewed `qsub` refused NCIS0013 ("Account is in state:
   Overspent"); the last 127 runs were charged to NAML0001. The committed script stops when `qsub` fails.
6. **A wrong upload during a manual check.** A hand-run uploader call with an empty source path copied 302
   repository files (74 MB) under one sounding store's prefix. They were deleted after checking the count and
   size against the store's own 55 objects, and the uploader now refuses a source that is not the named
   `.zarr` directory with a `zarr.json` in it.

## Website round trip

The website (`ai2es/ptype-web`) reads these stores directly. Checked against the pilot run 2023-12-26 00 UTC:
the map layers draw from the gridded store at every pyramid level, a clicked sounding is one range request,
and re-running the network in the browser on the stored sounding matches the map within 0.017 in probability
at six precipitating cells once the stored earth-relative winds are rotated back to the grid (0.20 without the
rotation). One writer bug found this way is still open: the `wind_rotation` root attribute of the sounding
store is written as the string form of a Python dict, so the website computes the angle from the grid
attributes instead.
