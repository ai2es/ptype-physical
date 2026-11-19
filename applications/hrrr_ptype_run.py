"""Run the fast HRRR precipitation-type pipeline for one or more model runs on one node.

Examples::

    # one run, all forecast hours of its cycle, settings from the YAML file
    python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600

    # a run list written by ``python -m ptype.hrrr_fast.runlist ... --out runs.txt``
    python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --runs-file runs.txt --processes 16 --threads 8

    # what would be fetched (reads only the small .idx files)
    python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600 --dry-run

Command-line options override the YAML file. The exit status is 0 when every run ended uploaded/indexed (or
complete with ``--no-upload``, or skipped because it is already indexed), and 1 otherwise, so a PBS job log
shows a failure clearly. Re-running the same command resumes: completed hours (per-hour markers) and runs
already in the index are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout without installing

from ptype.hrrr_fast import pipeline
from ptype.hrrr_fast.config import PipelineConfig

OK_STATUSES = {"indexed", "complete", "skipped_indexed"}


def build_parser() -> argparse.ArgumentParser:
    """Argument parser (separate so tests can inspect it)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/hrrr_fast.yml", help="YAML settings (ptype.hrrr_fast.config.PipelineConfig)")
    p.add_argument("--init", action="append", default=[], help="initialization time YYYYMMDDHH (repeatable)")
    p.add_argument("--runs-file", default=None, help="file of 'YYYYMMDDHH maxfh' lines (ptype.hrrr_fast.runlist --out)")
    p.add_argument("--forecast-hours", default=None, help="override the hours of every run, e.g. 0-18 or 11,12,13")
    p.add_argument("--processes", default=None, help="worker processes, an integer or 'auto' (0 = in-process)")
    p.add_argument("--threads", type=int, default=None, help="threads per worker process")
    p.add_argument("--engine", default=None, choices=["torch", "onnx", "numpy", "numba"])
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument("--weights", default=None, help="exported ptype_weights.npz")
    p.add_argument("--source", default=None, help="'s3' (NOAA AWS bucket), 'gcs' (Google mirror) or a local mirror directory")
    p.add_argument("--out-dir", default=None, help="work directory for run directories (node-local or scratch)")
    p.add_argument("--bourgouin-top-m", type=float, default=None)
    p.add_argument("--fetch-threads", type=int, default=None, help="download threads per worker, each fetching a share of the byte ranges")
    p.add_argument("--max-concurrent-fetches", type=int, default=None, help="node-wide cap on simultaneous GRIB downloads")
    p.add_argument("--upload", default=None, choices=["s3", "local", "none"])
    p.add_argument("--upload-root", default=None, help="destination of --upload local")
    p.add_argument("--keep-local", action="store_true", default=None, help="keep the run directory after upload")
    p.add_argument("--no-upload", action="store_true", help="leave the stores on disk; do not upload or touch the index")
    p.add_argument("--ignore-index", action="store_true", help="process runs even if the index lists them")
    p.add_argument("--dry-run", action="store_true", help="print the plan and estimated bytes from the idx files only")
    p.add_argument("--log-level", default=None)
    p.add_argument("--plain-log", action="store_true", help="human-readable log lines instead of JSON")
    p.add_argument("--summary-json", default=None, help="write the final manifests to this file")
    p.add_argument("--realtime", action="store_true", help="one --init only: process the run as its files arrive and publish each hour (ptype.hrrr_fast.realtime)")
    p.add_argument("--poll-seconds", type=float, default=30.0, help="--realtime: seconds between listings of the HRRR bucket")
    p.add_argument("--give-up-hours", type=float, default=4.0, help="--realtime: stop waiting for files this long after initialization")
    return p


def config_from_args(a: argparse.Namespace) -> PipelineConfig:
    """Merge the YAML file with the command-line overrides."""
    processes = a.processes
    if processes is not None and processes != "auto":
        processes = int(processes)
    return PipelineConfig.from_yaml(
        a.config,
        processes=processes,
        threads_per_process=a.threads,
        engine=a.engine,
        device=a.device,
        weights=a.weights,
        source=a.source,
        work_dir=a.out_dir,
        bourgouin_top_m=a.bourgouin_top_m,
        max_concurrent_fetches=a.max_concurrent_fetches,
        fetch_threads=a.fetch_threads,
        upload=a.upload,
        upload_root=a.upload_root,
        keep_local=a.keep_local,
        log_level=a.log_level,
        log_json=False if a.plain_log else None,
    )


def runs_from_args(a: argparse.Namespace) -> list[pipeline.RunSpec]:
    """Runs from ``--init`` and/or ``--runs-file`` (duplicates removed, order kept)."""
    hours = pipeline.parse_hours(a.forecast_hours) if a.forecast_hours else None
    runs = pipeline.make_runs(a.init, hours)
    if a.runs_file:
        runs += pipeline.read_runs_file(a.runs_file, hours)
    seen, out = set(), []
    for r in runs:
        if r.run_id not in seen:
            seen.add(r.run_id)
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    pipeline.configure_logging(cfg.log_level, cfg.log_json)
    runs = runs_from_args(a)
    if not runs:
        print("no runs given (--init or --runs-file)", file=sys.stderr)
        return 2
    if a.dry_run:
        pipeline.dry_run(cfg, runs)
        return 0
    if a.realtime:
        if len(runs) != 1:
            print("--realtime takes exactly one --init", file=sys.stderr)
            return 2
        from datetime import timedelta

        from ptype.hrrr_fast import realtime

        results = [realtime.run_realtime(cfg, runs[0], poll_s=a.poll_seconds, give_up_after=timedelta(hours=a.give_up_hours), no_upload=a.no_upload)]
    else:
        results = pipeline.run_pipeline(cfg, runs, no_upload=a.no_upload, ignore_index=a.ignore_index)
    if a.summary_json:
        Path(a.summary_json).write_text(json.dumps(results, indent=1, default=str))
    bad = [r.get("run_id") for r in results if r.get("status") not in OK_STATUSES]
    for r in results:
        print(
            json.dumps({"run": r.get("run_id"), "status": r.get("status"), "written": r.get("written_hours"), "missing": r.get("missing_hours"), "failed": r.get("failed_hours")})
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
