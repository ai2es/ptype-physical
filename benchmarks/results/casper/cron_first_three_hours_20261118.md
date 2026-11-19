# Cron's first three hours, and the first 48-hour realtime cycle (2026-11-18 23:00 to 2026-11-19 02:00 UTC)

The crontab line was installed on `cron.hpc.ucar.edu` at 22:57 UTC on 2026-11-18 (every five minutes, under
`flock`). Everything below was read from the launcher's `cron.log`, the ledger, the job logs, `qstat -xf`,
listings of the HRRR and `wwe-ptype` buckets, and the public index. Times are UTC.

## Before the first pass

- The cron host sees `/glade/work`, has `qsub` and `flock`, and had no crontab. The Casper virtual environment
  runs there (launcher `--dry-run`).
- That dry run planned a realtime job **and** a catch-up job for 17Z and 18Z in one pass: both runs were
  inside the 6-hour realtime lookback and past the 4-hour give-up time. Fixed in `b49917b` (runs a pass starts
  a realtime job for are kept out of its catch-up job); the dry run then planned realtime 17Z, 18Z, 21Z, 22Z
  and catch-up 16Z, 15Z.

## What cron submitted

36 passes between 23:00 and 01:55, one line each in `cron.log`, so no pass was skipped.

| Pass | Job | Runs |
|---|---|---|
| 23:00 | realtime x3 | 18Z (49 h), 21Z, 22Z |
| 23:00 to 23:40 | catch-up x9, two runs each | 17Z back to 00Z of 2026-11-18 |
| 23:40 | realtime | 23Z |
| 00:45 | realtime | 2026-11-19 00Z (49 h) |
| 01:45 | realtime | 01Z |

- Every run appears once in the ledger.
- Every run whose log I read ended with `run uploaded and indexed`, `failed: []` and `missing: []`: catch-up
  runs 01Z to 17Z of 2026-11-18 (06Z and 12Z with 49 hours, the others 19) and realtime 18Z, 21Z, 22Z, 23Z and
  00Z. The catch-up of 2026-11-18 00Z was still running at the 23:46 reading and I did not read its log after.
- The last catch-up job was submitted by the 23:40 pass, 40 minutes after the first.
- The 00Z job was submitted by the 00:45 pass, not 00:40: the launcher's window excludes a run that is exactly
  at the lead time. It cost nothing here (first file at 00:56) and would cost nothing on a normal day (first
  file at about 00:53).

## The 48-hour cycle: 2026-11-19 00Z (job 6744928)

| | |
|---|---|
| Hours published | 49 of 49 |
| Lag from the AWS `.idx` to the hour being in the index | 20 to 59 s, median 38 s |
| Hour 0 readable | 00:57:40 (57.7 min after initialization; its `.idx` arrived at 00:56:57, later than the usual 53 min) |
| Hour 18 readable | 01:20:13 (80.2 min) |
| Hour 48 readable | 01:54:09 (114.2 min; `.idx` at 01:53:20) |
| Consolidated store uploaded and indexed | 01:54:31, 22 s after the last hour |
| Final store | 161 objects, all `STANDARD_IA`; index entry `complete: true`, 49 hours, `frozen_cells` 124206 |
| Walltime, cores, memory | 1 h 09 min 21 s on 6 cores (6.9 core-hours); CPU time 42 min 35 s; peak 12.1 GB of 24 GB |
| Consolidating the gridded store | 15.4 s |

One worker kept up throughout: no hour waited behind another for more than a minute.

## A late delivery from NOAA: 23Z (job 6744326)

Hours 0 to 4 of the 23Z HRRR arrived on schedule (23:51 to 00:01). Hours 5 to 18 all arrived between 00:54 and
00:56, about 50 minutes late. The job published hours 0 to 4 at 53.9 to 61.9 minutes after initialization,
waited, and published the other 14 between 00:55:33 and 00:59:18 in the order the files arrived (about 17 s
each on one worker). The run was consolidated and indexed at 00:59:31, five minutes after the first late file.

While it waited, the website showed the run stopping at F04, and during the burst it had gaps (for example
F05 and F12 missing with F18 present), because hours come out in arrival order.

## Staging cleanup (first deletes with the realtime key)

| Removed at | Staging store | Objects |
|---|---|---|
| 00:45:24 (by the 00Z job) | 18Z | 3,233 |
| 01:45:30 (by the 01Z job) | 19Z | 1,313 |

At 02:00 the staging prefix held 20Z to 23Z of 2026-11-18 and 00Z and 01Z of 2026-11-19, which matches
the rule (remove a finished run's staging store once it is six hours past initialization).

## Website

Checked on the deployed site at 23:50 to 23:58: the backfilled and realtime runs load, and a page opened before
23Z was in the index showed 23Z F00 a few minutes after it was published, without a reload.

## Not covered

- A full day: this is three hours of cron. Queue waits at busier times of day are not measured.
- A cycle that never completes (the give-up path) and a job killed mid-run, against the bucket.
- The retention script, which has not been run.
- Core-hours per day: one 18-hour cycle cost 6.0 core-hours (20Z, earlier test) and one 48-hour cycle 6.9;
  a day of 20 + 4 cycles would be about 150 core-hours if every job waited as these did, which is an
  extrapolation from two jobs, not a measurement.
