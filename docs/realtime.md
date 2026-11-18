# Realtime precipitation-type forecasts on Casper

Every HRRR cycle is processed on Casper as its files arrive and published to `s3://wwe-ptype`, the bucket the
archive is in and the website reads. This page is how it is put together and how to run it. Measurements behind
the choices: `benchmarks/results/casper/arrival.md` and `measure.md`.

## How it works

```
cron.hpc.ucar.edu (every 5 min)
  benchmarks/pbs/realtime_cron.sh
    applications/hrrr_ptype_realtime_submit.py     reads the index over HTTPS, submits to Casper
      one job per new run, 45 min after initialization
        applications/hrrr_ptype_run.py --realtime  (ptype/hrrr_fast/realtime.py)
          list the HRRR bucket every 30 s
          each hour whose GRIB file and idx are both listed -> worker -> upload that hour -> index
          all hours in -> consolidate the gridded store -> upload to the archive key -> switch the index entry
      one catch-up job for runs still unfinished 4 h after initialization (batch mode)
```

- **Trigger.** The HRRR's analysis file appears 52 to 54 minutes after initialization and the rest follow one
  every one to two minutes. The launcher does not watch the bucket; it submits each run's job at 45 minutes
  and the job waits for the files, so queue wait costs nothing unless it exceeds about eight minutes.
- **Readiness.** An hour is read only when both its `.grib2` and `.idx` are listed. AWS writes the idx after
  the GRIB file and Google usually before it, so neither object alone is a safe signal on both.
- **Publishing.** Each finished hour's objects are uploaded and the run's entry in `ptype/inits.json` is
  rewritten with the hours so far. While a run is arriving its gridded store is at
  `ptype/realtime-staging/<archive path>` in the per-hour layout; the sounding store is at its permanent key
  from the start. When the run is complete the consolidated gridded store goes to the archive key and the
  entry's `store` is switched to it in one write. The module docstring of `ptype/hrrr_fast/realtime.py` says
  why two keys are needed.
- **Index entry.** Realtime entries carry `realtime: true`, `complete` and `frozen_cells`. The website ignores
  fields it does not know.
- **When things go wrong.** A job that dies leaves the hours it published on the site with `complete: false`.
  The launcher's catch-up pass picks up any run of the last 24 hours that is missing or incomplete once it is
  more than 4 h 10 min old, two runs per job, three tries per run. Gaps older than 24 hours need a batch run by
  hand (`applications/hrrr_ptype_submit.py`).
- **Status.** `ptype/realtime_status.json` holds the latest run, the hours published and the last error.

## Setting it up

1. **Python environment** (once). Casper has no system Python 3.12, so the Derecho environment does not load
   there. `venv_casper` is built on a uv-managed Python with the Derecho environment's package versions:

   ```bash
   cd /glade/work/dgagne/ptype_fast && export UV_CACHE_DIR=$PWD/uv_cache UV_PYTHON_INSTALL_DIR=$PWD/uv_python
   uv pip freeze --python venv/bin/python | grep -v -i -E "^(torch|nvidia|triton|ptype|-e )" > casper_constraints.txt
   uv venv --python 3.12 --python-preference only-managed venv_casper
   uv pip install --python venv_casper/bin/python -c casper_constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match "torch==2.15.0+cpu" numpy numba obstore eccodes "zarr>=3.4" cast-value-rs pyproj onnxruntime pyyaml psutil boto3 pytest
   ```

2. **AWS key**: `docs/aws_realtime.md`. You create it; the pipeline is given only the profile name.

3. **Settings**, in `~/.ptype_realtime.env` (not in the repository):

   ```bash
   PTYPE_ACCOUNT=NAML0001
   PTYPE_AWS_PROFILE=wwe-ptype-realtime
   ```

4. **Try one pass by hand** before installing cron. This prints what would be submitted and changes nothing:

   ```bash
   PTYPE_SUBMIT_ARGS=--dry-run /glade/work/dgagne/ptype_fast/ptype-physical/benchmarks/pbs/realtime_cron.sh; tail -5 /glade/derecho/scratch/$USER/ptype_fast/realtime/jobs/cron.log
   ```

5. **Cron**, on `cron.hpc.ucar.edu` (login nodes refuse `crontab`):

   ```
   */5 * * * * flock -n $HOME/.ptype_realtime.lock /glade/work/dgagne/ptype_fast/ptype-physical/benchmarks/pbs/realtime_cron.sh
   ```

   To stop realtime, remove that line. Jobs already running finish their run.

Logs are in `/glade/derecho/scratch/$USER/ptype_fast/realtime/jobs/`: `cron.log` (one line per pass),
`rt_<run>.log` and `rt_<run>.summary.json` per job, and `submitted.jsonl`, the record of every submission.

## Retention

`applications/hrrr_ptype_expire.py` removes realtime runs older than 30 days whose `frozen_cells` is below
100, where `frozen_cells` is the largest number of grid cells in any one forecast hour that the HRRR's own
categorical type calls snow, ice pellets or freezing rain. Archive runs are never touched. It prints what it
would do and changes nothing unless given `--delete`:

```bash
AWS_PROFILE=<a profile allowed to delete> python applications/hrrr_ptype_expire.py
```

Expect it to prune little in the cold season: on 2026-11-17 the 12Z and 13Z cycles had 64,000 to 68,000
frozen cells. Storage then grows by about 3.3 GB a run, 78 GB a day.

## Cost

A realtime job is 6 cores and 24 GB and lives from 45 minutes after initialization until its last hour is
published: about 50 minutes for an 18 h cycle and 75 for a 48 h cycle. From those figures a day is about
130 core-hours. That is an estimate from measured per-hour times, not a measured day.

## Not verified yet

- The cron host: that it can see `/glade/work` and submit to Casper.
- Uploads with the realtime key (it does not exist yet); everything else has run on Casper.
- Queue wait at busy times of day (four jobs started in 3 s each, all within 15 minutes of each other).
