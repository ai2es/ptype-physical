# Download source and fetch threads, end to end (Derecho CPU node dec0941, job 8210303)

One run (2023-12-26 00 UTC, f00-f18), 15 workers x 8 compute threads, torch on CPU, no upload, commit `b48317b`.
The five cases ran back to back in one job (`benchmarks/pbs/smoke_fetch.pbs`), 01:16-01:33 UTC on 2026-11-17.
Wall time is first to last log record of the driver; it includes worker start-up and finalize.

| start (UTC) | source | fetch threads | hours written | wall s | until workers ready s | fetch s median | fetch s max | inference s median | Bourgouin s median | write s median |
|---|---|---|---|---|---|---|---|---|---|---|
| 01:16:28 | s3 | 8 | 19 | 201 | 46 | 46.8 | 54.6 | 7.2 | 12.1 | 5.5 |
| 01:20:02 | gcs | 8 | 19 | 67 | 6 | 8.2 | 11.4 | 6.5 | 3.3 | 5.3 |
| 01:21:21 | s3 | 1 | 19 | 440 | 14 | 192.4 | 212.7 | 4.7 | 3.3 | 4.9 |
| 01:28:54 | gcs | 1 | 19 | 111 | 6 | 33.4 | 38.9 | 7.1 | 3.3 | 5.0 |
| 01:30:57 | s3 | 8 | 19 | 138 | 5 | 44.4 | 57.7 | 3.3 | 3.4 | 5.0 |

All five cases produced the same SHA-256 over every chunk and shard file of both stores
(`83f24443c4d9...`, see `job.log`), so the source and the thread count do not change the output.

The first case paid for the job's one-time numba compilation (46 s until the workers were ready and a 12 s
Bourgouin median, against 5-14 s and 3.3 s afterwards), so compare the repeat at 01:30 with the other cases.

Each log has one warning: the read of the index on `wwe-ptype` was refused (permission denied). The runs used
`--no-upload`, so nothing depended on it; why the read was refused in this job was not investigated.
