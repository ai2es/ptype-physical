"""Submit one Casper job per new HRRR run; meant to be called every few minutes by cron.

Crontab line (on ``cron.hpc.ucar.edu``; login nodes refuse ``crontab``)::

    */5 * * * * flock -n $HOME/.ptype_realtime.lock /glade/work/dgagne/ptype_fast/ptype-physical/benchmarks/pbs/realtime_cron.sh

Each pass does two things and exits:

1. **Realtime.** For every run initialized in the last ``--lookback-hours`` that is at least ``--lead-minutes``
   old, is not complete in the index and has never been submitted, submit a job that runs
   ``hrrr_ptype_run.py --realtime`` for it. The first wrfnat file of a cycle appears 52-54 minutes after
   initialization (``benchmarks/results/casper/arrival.md``); submitting at 45 minutes lets the job sit out any
   queue wait before the data is there, and the job itself waits for the files. A run is submitted for realtime
   once only (the ledger remembers it for good): a second realtime job for the same run could publish over the
   first.
2. **Catch-up.** Runs older than the realtime give-up time but younger than ``--catchup-hours`` that are still
   missing or incomplete (the realtime job died, the cluster was down, a file arrived late) go into ONE batch job
   (``hrrr_ptype_run.py --runs-file``), at most ``--max-catchup`` runs per pass and ``--max-attempts`` tries per
   run. The batch driver treats a ``complete: false`` entry as unfinished and finalizes the run.

The index is read anonymously over HTTPS with the website's ``Referer`` (the bucket policy allows exactly that),
so this script needs no AWS credentials and cannot be stopped by an expired login. If the index cannot be read
the pass still submits realtime jobs (the ledger prevents duplicates) but skips catch-up, which needs to know
what is already done.

The scheduler helpers and the ledger format are the batch launcher's (``hrrr_ptype_submit.py``).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrrr_ptype_submit as launcher

INDEX_URL = "https://wwe-ptype.s3.us-east-1.amazonaws.com/ptype/inits.json"
REFERER = "https://ai2es.github.io/ptype-web/"
PBS_SERVER = "casper-pbs"
VENV = f"{launcher.PTYPE_FAST}/venv_casper"
SCRATCH = "/glade/derecho/scratch/$USER/ptype_fast/realtime"


def max_forecast_hour(hour: int) -> int:
    """HRRR runs to 48 h at 00/06/12/18 UTC and 18 h otherwise."""
    return 48 if hour % 6 == 0 else 18


def read_index(url: str = INDEX_URL, referer: str = REFERER, timeout: float = 20.0) -> dict:
    """The run index, read the way the website reads it (no credentials)."""
    req = urllib.request.Request(url, headers={"Referer": referer})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def complete_runs(idx: dict) -> set[str]:
    """Run ids the index lists as finished (``complete`` absent means an archive run, which is finished)."""
    return {launcher.run_id_from_init(e["init"]) for e in idx.get("inits", []) if e.get("complete", True) is not False}


def ledger_records(ledger: Path) -> list[dict]:
    """Every submission ever recorded."""
    if not ledger.exists():
        return []
    return [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]


def hourly_runs(start: datetime, end: datetime) -> list[datetime]:
    """Initialization times on the hour with ``start < init <= end``, oldest first."""
    t = end.replace(minute=0, second=0, microsecond=0)
    out = []
    while t > start:
        out.append(t)
        t -= timedelta(hours=1)
    return out[::-1]


def realtime_candidates(now: datetime, done: set[str], records: Sequence[dict], lead: timedelta, lookback: timedelta) -> list[str]:
    """Runs to start a realtime job for now: old enough, recent enough, not done, never submitted."""
    seen = {r for rec in records for r in rec["runs"]}
    out = []
    for init in hourly_runs(now - lookback, now - lead):
        rid = f"{init:%Y%m%d%H}"
        if rid not in done and rid not in seen:
            out.append(rid)
    return out


def catchup_candidates(
    now: datetime, done: set[str], records: Sequence[dict], in_flight: set[str], give_up: timedelta, horizon: timedelta, max_runs: int, max_attempts: int
) -> list[str]:
    """Unfinished runs past the realtime give-up time, newest first (the newest matter most to someone looking now)."""
    tries: dict[str, int] = {}
    for rec in records:
        if rec.get("kind") == "catchup":
            for r in rec["runs"]:
                tries[r] = tries.get(r, 0) + 1
    out = []
    # Ten minutes past the give-up time, so the realtime job has finished writing its partial entry.
    for init in reversed(hourly_runs(now - horizon, now - give_up - timedelta(minutes=10))):
        rid = f"{init:%Y%m%d%H}"
        if rid in done or rid in in_flight or tries.get(rid, 0) >= max_attempts:
            continue
        out.append(rid)
        if len(out) >= max_runs:
            break
    return out


def render_script(
    name: str,
    log_file: Path,
    command: Sequence[str],
    *,
    account: str,
    queue: str,
    ncpus: int,
    mem_gb: int,
    walltime: str,
    aws_profile: str,
    comment: str,
    extra_args: str = "",
) -> str:
    """The PBS script of one Casper job.

    ``common.sh`` supplies the checkout, weights and numba cache; ``VENV`` points it at the Casper environment
    (the Derecho one needs a system Python that Casper does not have). ``AWS_PROFILE`` names the long-lived,
    bucket-scoped key (``docs/aws_realtime.md``): batch jobs do not inherit the submitting shell, and a single
    sign-on profile would expire. ``-m a`` mails the owner if PBS aborts the job.
    """
    cmd = " ".join(a if "$" in a else shlex.quote(a) for a in command) + (f" {extra_args}" if extra_args else "")
    return f"""#!/bin/bash
#PBS -N {name[:15]}
#PBS -A {account}
#PBS -q {queue}
#PBS -l select=1:ncpus={ncpus}:mem={mem_gb}GB
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -m a
#PBS -o {log_file}
# Generated by applications/hrrr_ptype_realtime_submit.py at {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}
# {comment}
export VENV={VENV}
export RESULTS={SCRATCH}/results
source {launcher.REPO}/benchmarks/pbs/common.sh
export AWS_PROFILE={shlex.quote(aws_profile)}
{cmd}
"""


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jobs-dir", required=True, help="where scripts, logs and the ledger (submitted.jsonl) go")
    p.add_argument("--account", required=True, help="PBS project code to charge")
    p.add_argument("--aws-profile", required=True, help="name of the AWS profile the jobs upload with (the key itself is never read here)")
    p.add_argument("--queue", default=f"casper@{PBS_SERVER}")
    p.add_argument("--config", default="config/hrrr_fast_casper.yml")
    # One worker of four threads does a forecast hour in 16-20 s on either Casper CPU type and peaks at 4.3 GB
    # (benchmarks/results/casper/measure.md); files arrive a minute or more apart, so that keeps up with room to spare.
    p.add_argument("--ncpus", type=int, default=6)
    p.add_argument("--mem-gb", type=int, default=24)
    p.add_argument("--processes", type=int, default=1, help="worker processes of a job")
    p.add_argument("--threads", type=int, default=4, help="threads per worker")
    p.add_argument("--extra-args", default="", help="appended to the hrrr_ptype_run.py command line (for example a local upload root in a test)")
    p.add_argument("--lead-minutes", type=float, default=45.0, help="submit a run's job this long after its initialization")
    p.add_argument("--lookback-hours", type=float, default=6.0, help="oldest run a realtime job is still started for")
    p.add_argument("--give-up-hours", type=float, default=4.0, help="how long after initialization a realtime job waits for files")
    p.add_argument("--catchup-hours", type=float, default=24.0, help="oldest unfinished run the catch-up job retries")
    p.add_argument("--max-catchup", type=int, default=2, help="runs per catch-up job")
    p.add_argument("--max-attempts", type=int, default=3, help="catch-up tries per run")
    p.add_argument("--no-catchup", action="store_true")
    p.add_argument("--index-url", default=INDEX_URL)
    p.add_argument("--now", default=None, help="pretend it is this time, YYYYMMDDHHMM UTC (tests, replays)")
    p.add_argument("--dry-run", action="store_true", help="print what would be submitted; write and submit nothing")
    return p


def walltime_for(lead_minutes: float, give_up_hours: float) -> str:
    """Walltime of a realtime job: long enough to reach the give-up time, plus a margin to finish up.

    The job starts about ``lead_minutes`` after initialization and waits for files until ``give_up_hours``
    after it, so anything shorter lets PBS end the job while it is still (rightly) waiting. That happened in
    the first live test: the walltime was sized for a normal cycle (last file at 75-110 minutes) and the cycle
    of 2026-11-18 16Z delivered its last file at 100 minutes. Ten minutes more and PBS would have ended the
    job, leaving the run incomplete until the catch-up pass more than four hours after initialization. A job leaves
    as soon as its run is complete, so on a normal day the long walltime costs nothing; the 15-minute margin
    covers consolidating and uploading the gridded store.
    """
    seconds = give_up_hours * 3600 - lead_minutes * 60 + 15 * 60
    seconds = int(-(-max(seconds, 1800) // 300) * 300)  # whole five minutes, at least half an hour
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def main(argv: list[str] | None = None, run_cmd: Callable[[Sequence[str]], str] = launcher._run_cmd, index_reader: Callable[[str], dict] | None = None) -> list[dict]:
    """One pass; returns the records of what was (or, with ``--dry-run``, would be) submitted."""
    a = build_parser().parse_args(argv)
    now = datetime.strptime(a.now, "%Y%m%d%H%M").replace(tzinfo=UTC) if a.now else datetime.now(UTC)
    jobs_dir = Path(a.jobs_dir)
    ledger = jobs_dir / "submitted.jsonl"
    records = ledger_records(ledger)
    try:
        done: set[str] | None = complete_runs((index_reader or read_index)(a.index_url))
    except Exception as err:  # noqa: BLE001 - an unreadable index must not stop realtime jobs
        print(f"warning: could not read the index ({type(err).__name__}: {err}); realtime jobs only, no catch-up", file=sys.stderr)
        done = None
    stamp = f"{now:%Y%m%dT%H%M}"
    common = ["--config", a.config, "--weights", launcher.WEIGHTS, "--out-dir", f"{SCRATCH}/runs"]
    planned: list[dict] = []

    for rid in realtime_candidates(now, done or set(), records, timedelta(minutes=a.lead_minutes), timedelta(hours=a.lookback_hours)):
        max_fh = max_forecast_hour(int(rid[8:10]))
        log = jobs_dir / f"rt_{rid}.log"
        cmd = ["python", "applications/hrrr_ptype_run.py", *common, "--realtime", "--init", rid, "--processes", str(a.processes), "--threads", str(a.threads)]
        cmd += ["--give-up-hours", str(a.give_up_hours), "--summary-json", str(log.with_suffix(".summary.json"))]
        script = render_script(
            f"rt_{rid[4:]}", log, cmd, account=a.account, queue=a.queue, ncpus=a.ncpus, mem_gb=a.mem_gb, walltime=walltime_for(a.lead_minutes, a.give_up_hours),
            aws_profile=a.aws_profile, extra_args=a.extra_args, comment=f"realtime run {rid}, forecast hours 0-{max_fh}",
        )  # fmt: skip
        planned.append({"kind": "realtime", "runs": [rid], "script": jobs_dir / f"rt_{rid}.pbs", "text": script})

    if done is not None and not a.no_catchup:
        # The scheduler lives on Casper; ask it by name so the same call works from the cron host and from Derecho.
        active = launcher.active_job_ids(lambda cmd: run_cmd([*cmd, f"@{PBS_SERVER}"]))
        # Runs this pass is starting a realtime job for are not in the ledger yet. Without them a run inside both
        # windows (older than the give-up time, younger than the lookback) got a realtime job and a catch-up job
        # in the same pass, which is what the first pass after any gap longer than the give-up time looks like.
        flying = launcher.in_flight_runs(ledger, active, now) | {r for job in planned for r in job["runs"]}
        late = catchup_candidates(now, done, records, flying, timedelta(hours=a.give_up_hours), timedelta(hours=a.catchup_hours), a.max_catchup, a.max_attempts)
        if late:
            runs_file = jobs_dir / f"catchup_{stamp}.runs"
            log = jobs_dir / f"catchup_{stamp}.log"
            cmd = ["python", "applications/hrrr_ptype_run.py", *common, "--runs-file", str(runs_file), "--processes", str(a.processes), "--threads", str(a.threads)]
            cmd += ["--summary-json", str(log.with_suffix(".summary.json"))]
            script = render_script(
                f"cu_{stamp[4:]}", log, cmd, account=a.account, queue=a.queue, ncpus=a.ncpus, mem_gb=a.mem_gb, walltime="02:00:00",
                aws_profile=a.aws_profile, extra_args=a.extra_args, comment=f"catch-up of {len(late)} unfinished runs: {' '.join(late)}",
            )  # fmt: skip
            lines = "".join(f"{r} {max_forecast_hour(int(r[8:10]))}\n" for r in late)
            planned.append({"kind": "catchup", "runs": late, "script": jobs_dir / f"catchup_{stamp}.pbs", "text": script, "runs_file": runs_file, "runs_text": lines})

    out = []
    for job in planned:
        rec = {"kind": job["kind"], "runs": job["runs"], "time": now.isoformat()}
        if a.dry_run:
            print(f"would submit {job['kind']} job for {' '.join(job['runs'])}")
            out.append(rec)
            continue
        jobs_dir.mkdir(parents=True, exist_ok=True)
        if "runs_file" in job:
            job["runs_file"].write_text(job["runs_text"])
        job["script"].write_text(job["text"])
        try:
            rec["job_id"] = launcher.submit(job["script"], run_cmd)
        except Exception as err:  # noqa: BLE001 - one refused job must not stop the others; the next pass retries it
            print(f"qsub failed for {' '.join(job['runs'])}: {type(err).__name__}: {str(err)[:300]}", file=sys.stderr)
            continue
        with open(ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"submitted {rec['job_id']} ({job['kind']}) for {' '.join(job['runs'])}")
        out.append(rec)
    return out


if __name__ == "__main__":
    main()
