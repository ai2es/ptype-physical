"""Remove old realtime runs that had almost no frozen precipitation. A dry run unless ``--delete`` is given.

Rule (docs/realtime.md): a run is removed when ALL of these hold

* its index entry says ``realtime: true`` (archive runs, which have no such flag, are never touched);
* it was initialized more than ``--days`` (30) days ago;
* its ``frozen_cells`` is below ``--min-frozen`` (100). ``frozen_cells`` is the largest number of grid cells, in
  any one forecast hour of the run, that the HRRR's own categorical type calls snow, ice pellets or freezing
  rain. A realtime entry without the field is kept and reported, not guessed at.

Everything else is kept indefinitely. In the cold season nearly every run passes the test, so this mainly
prunes summer.

Deleting is permanent, so:

* without ``--delete`` the script only prints what it would remove and how much it would free;
* object keys are derived from the run's initialization time (``index.store_paths``), never taken from the
  index entry, and every delete is checked against the ``ptype/`` or ``soundings/`` prefix and the ``.zarr``
  suffix (``pipeline._check_delete``), so a damaged index cannot point the script at anything else;
* a run's index entry is removed before its stores, so the website never lists a run whose data is going.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptype.hrrr_fast import index, pipeline, realtime

_INIT_FMT = "%Y-%m-%dT%H:%MZ"


def classify(entries: list[dict[str, Any]], now: datetime, days: float, min_frozen: int) -> tuple[list[dict], list[dict]]:
    """``(expire, unknown)``: realtime entries to remove, and old realtime entries that lack ``frozen_cells``."""
    cutoff = now - timedelta(days=days)
    expire, unknown = [], []
    for e in entries:
        if e.get("realtime") is not True:
            continue
        if datetime.strptime(e["init"], _INIT_FMT).replace(tzinfo=UTC) >= cutoff:
            continue
        if "frozen_cells" not in e:
            unknown.append(e)
        elif int(e["frozen_cells"]) < min_frozen:
            expire.append(e)
    return expire, unknown


def store_keys(init_string: str) -> list[tuple[str, str]]:
    """``(key, prefix it must lie under)`` of every store a realtime run can have: archive, staging and soundings."""
    init = datetime.strptime(init_string, _INIT_FMT).replace(tzinfo=UTC)
    pk, sk = index.store_paths(init)
    return [(pk, index.PTYPE_PREFIX), (realtime.staging_key(init), realtime.STAGING_PREFIX), (sk, index.SOUNDING_PREFIX)]


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="config/hrrr_fast_casper.yml", help="pipeline settings (bucket, region); AWS_PROFILE selects the credentials")
    p.add_argument("--days", type=float, default=30.0, help="only runs older than this are considered")
    p.add_argument("--min-frozen", type=int, default=100, help="runs with at least this many frozen cells in some hour are kept")
    p.add_argument("--delete", action="store_true", help="actually remove the runs; without it nothing is changed")
    p.add_argument("--local-root", default=None, help="act on a local copy of the bucket instead of S3 (tests)")
    p.add_argument("--now", default=None, help="pretend it is this time, YYYYMMDDHHMM UTC (tests)")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the rule once; returns what was (or would be) removed."""
    a = build_parser().parse_args(argv)
    now = datetime.strptime(a.now, "%Y%m%d%H%M").replace(tzinfo=UTC) if a.now else datetime.now(UTC)
    if a.local_root:
        uploader: Any = pipeline.LocalUploader(a.local_root)
        idx: Any = pipeline.LocalIndex(Path(a.local_root) / index.INDEX_KEY)
    else:
        cfg = pipeline.PipelineConfig.from_yaml(a.config)
        session = pipeline.resolve_aws_session(os.environ.get("AWS_PROFILE"), cfg.region, verify=True)
        uploader = pipeline.Boto3Uploader(cfg.bucket, cfg.region, session, cfg.storage_class)
        idx = pipeline.ObstoreIndex(pipeline.output_s3_store(cfg.bucket, cfg.region, session))
    entries = idx.read().get("inits", [])
    expire, unknown = classify(entries, now, a.days, a.min_frozen)
    n_realtime = sum(1 for e in entries if e.get("realtime") is True)
    print(f"{len(entries)} runs in the index, {n_realtime} realtime; {len(expire)} to expire (older than {a.days:g} days, fewer than {a.min_frozen} frozen cells)")
    for e in unknown:
        print(f"kept, no frozen_cells recorded: {e['init']}")
    removed: list[str] = []
    objects = 0
    for e in expire:
        print(f"{'removing' if a.delete else 'would remove'} {e['init']} (frozen_cells={e['frozen_cells']}, hours={len(e.get('forecast_hours', []))})")
        if not a.delete:
            continue
        idx.remove({e["init"]})
        for key, prefix in store_keys(e["init"]):
            objects += uploader.delete_store(key, prefix)
        removed.append(e["init"])
    if a.delete:
        print(f"removed {len(removed)} runs, {objects} objects")
    else:
        print("dry run: nothing was changed (pass --delete to remove)")
    return {"expire": [e["init"] for e in expire], "unknown": [e["init"] for e in unknown], "removed": removed, "objects": objects}


if __name__ == "__main__":
    main()
