"""Realtime driver: process ONE HRRR run while its files are still arriving, publishing each forecast hour as it is done.

The batch driver (``pipeline.run_pipeline``) assumes every file of a run exists, computes them all and then
uploads the run once. A live HRRR cycle does not work like that: its wrfnat files reach the public buckets one
forecast hour at a time, the analysis about 53 minutes after initialization and the last hour 20 to 55 minutes
later (measured, ``benchmarks/results/casper/arrival.md``). Waiting for the last file would hold back the first
hours by that long, so this driver

1. polls the bucket listing (:meth:`pipeline.GribSource.available`) and hands each hour to the worker pool the
   moment its GRIB file and idx are both there;
2. after each finished hour uploads that hour's objects and rewrites the run's entry in ``ptype/inits.json``
   with the hours written so far, so the website shows the hour within a few minutes of the HRRR publishing it;
3. when every hour is in (or the give-up time passes) hands over to the batch code path
   (``pipeline.finalize_run``): consolidate the gridded store, upload it to its permanent key, and switch the
   index entry to it.

Why the gridded store lives at two keys. Per-hour writes need the "staging" layout (one small object per array,
level and hour: disjoint keys, no locks; see ``writers``), but the archive wants the consolidated layout (one
shard per array, so every object clears S3 Standard-IA's 128 KiB billing minimum). The two layouts cannot share
a key: the consolidated shard of an array is named ``c/0/0/0``, which is also the staging chunk of forecast
hour 0, and the array's ``zarr.json`` differs. Overwriting in place would leave readers with metadata and data
that disagree for as long as the upload takes. So hours are published under
``ptype/realtime-staging/<same path>`` (Standard class, short-lived), the finished store is uploaded to the
normal archive key, and the index entry's ``store`` is switched from one to the other in a single conditional
write. Staging stores are deleted later, by :func:`cleanup_staging` at the start of a following job, once no
index entry points at them and they are old enough that no open browser tab still holds the old entry.

The sounding store needs none of this: it is already one shard per hour, so each hour is uploaded straight to
its permanent key.

Entry fields added for realtime runs (``index.merge_entry(..., extra=...)``): ``realtime: true``,
``complete`` (false while hours are arriving or if the run was cut short; the batch driver treats such a run as
not done, so a later catch-up pass completes it) and ``frozen_cells`` (the largest number of grid cells in any
one forecast hour that the HRRR's own categorical type calls snow, ice pellets or freezing rain; the retention
rule in ``applications/hrrr_ptype_expire.py`` reads it).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import index, pipeline
from .config import PipelineConfig

logger = logging.getLogger(__name__)

#: Bucket prefix of the per-hour gridded stores of runs that are still arriving.
STAGING_PREFIX = f"{index.PTYPE_PREFIX}/realtime-staging"
#: Name of the status document written next to the index.
STATUS_NAME = "realtime_status.json"
#: A data object of forecast-hour index k: ``.../c/<k>/<y>/<x>`` (gridded) or ``sounding/c/<k>/0/0``.
_HOUR_OBJECT = re.compile(r"(?:^|/)c/(\d+)/\d+/\d+$")
_STORE_STAMP = re.compile(r"hrrr_ptype_(\d{8})_(\d{2})00\.zarr$")


def staging_key(init: datetime) -> str:
    """Bucket key of a run's staging gridded store: the archive path, under :data:`STAGING_PREFIX`."""
    pk, _ = index.store_paths(init)
    return f"{STAGING_PREFIX}/{pk.removeprefix(index.PTYPE_PREFIX + '/')}"


def store_files(store: Path) -> tuple[list[str], dict[int, list[str]]]:
    """``(static, by_hour)``: the store's metadata and coordinate files, and its data files by forecast-hour index.

    Paths are relative to ``store``, POSIX style. Anything that is not a three-dimensional chunk key is static
    (``zarr.json`` documents and the one-dimensional coordinate chunks), which is only true right after the
    store is created; later a walk can also see another hour's temporary file, so callers take ``static`` once
    at creation and afterwards use only ``by_hour`` for hours they know are finished.
    """
    static: list[str] = []
    by_hour: dict[int, list[str]] = {}
    for path in sorted(p for p in Path(store).rglob("*") if p.is_file()):
        rel = path.relative_to(store).as_posix()
        m = _HOUR_OBJECT.search(rel)
        if m:
            by_hour.setdefault(int(m.group(1)), []).append(rel)
        else:
            static.append(rel)
    return static, by_hour


def cleanup_staging(uploader, index_backend, now: datetime, min_age: timedelta = timedelta(hours=6)) -> list[str]:
    """Delete staging stores that nothing points at any more; returns the keys removed.

    A staging store is removed when its run started more than ``min_age`` ago AND the index does not reference
    it (the run was finalized and its entry switched to the archive key, or the entry is gone). The age keeps a
    store alive for browser tabs that still hold an index from before the switch. Runs that were cut short keep
    pointing at their staging store, so they are kept until a catch-up pass completes them. Failures are logged
    and never stop the job: cleanup is housekeeping.
    """
    removed: list[str] = []
    if uploader is None or index_backend is None or not hasattr(uploader, "list_stores"):
        return removed
    try:
        referenced = {f"{index.PTYPE_PREFIX}/{e.get('store')}" for e in index_backend.read().get("inits", [])}
        for key in uploader.list_stores(STAGING_PREFIX):
            m = _STORE_STAMP.search(key)
            if not m or key in referenced:
                continue
            started = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=UTC)
            if now - started < min_age:
                continue
            n = uploader.delete_store(key, STAGING_PREFIX)
            removed.append(key)
            logger.info("staging store removed", extra={"key": key, "objects": n})
    except Exception as err:  # noqa: BLE001 - housekeeping must not fail the run
        logger.warning("staging cleanup failed", extra={"error": f"{type(err).__name__}: {str(err)[:300]}"})
    return removed


@dataclasses.dataclass
class _Publisher:
    """Uploads finished hours and keeps the run's index entry and the status file current."""

    st: pipeline.RunState
    uploader: Any
    index_backend: Any
    now: Callable[[], datetime]
    published: set[int] = dataclasses.field(default_factory=set)
    static_sent: bool = False
    static: tuple[list[str], list[str]] = ((), ())  # (gridded, sounding) metadata and coordinate files
    last_error: str | None = None

    def frozen_cells(self) -> int:
        """Largest per-hour count of HRRR snow / ice pellets / freezing rain cells among the hours written."""
        return max((int(r.get("info", {}).get("frozen_cells", 0)) for r in self.st.manifest["hours"].values() if r.get("status") == "done"), default=0)

    def extra(self, complete: bool) -> dict[str, Any]:
        return {"realtime": True, "complete": complete, "frozen_cells": self.frozen_cells()}

    def publish(self) -> None:
        """Send every written hour that has not been sent, then rewrite the index entry. Errors are kept for the next try."""
        st = self.st
        written = set(st.done_hours())
        todo = sorted(written - self.published)
        if self.uploader is None or not todo:
            return
        skey = staging_key(st.spec.init)
        _, sounding_key = index.store_paths(st.spec.init)
        try:
            t0 = time.perf_counter()
            if not self.static_sent:
                # Metadata first this once: the stores are created complete, with every hour slot allocated, so
                # they are valid (all fill) before any data is there; the index is what announces an hour.
                self.uploader.upload_files(st.ptype_path, skey, self.static[0], storage_class="STANDARD")
                self.uploader.upload_files(st.sounding_path, sounding_key, self.static[1])
                self.static_sent = True
            _, g_hours = store_files(st.ptype_path)
            _, s_hours = store_files(st.sounding_path)
            for fxx in todo:
                k = st.spec.store_hours.index(fxx)
                self.uploader.upload_files(st.ptype_path, skey, g_hours.get(k, []), storage_class="STANDARD")
                self.uploader.upload_files(st.sounding_path, sounding_key, s_hours.get(k, []))
            entry = index.make_entry(st.spec.init, sorted(self.published | set(todo)))
            entry["store"] = skey.removeprefix(index.PTYPE_PREFIX + "/")
            entry["extra"] = self.extra(complete=False)
            self.index_backend.update(entry)
            self.published |= set(todo)
            self.last_error = None
            lag = {str(h): round((self.now() - st.spec.init).total_seconds() / 60, 1) for h in todo}
            st.manifest.setdefault("published", {}).update({h: {"minutes_after_init": m, "time": self.now().isoformat()} for h, m in lag.items()})
            logger.info("hours published", extra={"run": st.spec.run_id, "hours": todo, "seconds": round(time.perf_counter() - t0, 2), "minutes_after_init": lag})
        except Exception as err:  # noqa: BLE001 - keep computing; the next publish retries everything unsent
            self.last_error = f"{type(err).__name__}: {str(err)[:300]}"
            logger.warning("publish failed; will retry with the next hour", extra={"run": st.spec.run_id, "hours": todo, "error": self.last_error})
        self.status("running")

    def status(self, state: str) -> None:
        """Write the small status document the site (or a person) can read to see that realtime is alive."""
        # Nothing is published with uploads off, and a status file would say otherwise.
        if self.uploader is None or self.index_backend is None or not hasattr(self.index_backend, "write_json"):
            return
        st = self.st
        doc = {
            "updated": self.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run": index.init_string(st.spec.init),
            "state": state,
            "hours_expected": len(st.spec.hours),
            "hours_published": sorted(self.published),
            "last_error": self.last_error,
        }
        try:
            self.index_backend.write_json(STATUS_NAME, doc)
        except Exception as err:  # noqa: BLE001 - a status write must never stop the run
            logger.warning("status write failed", extra={"error": f"{type(err).__name__}: {str(err)[:200]}"})


def run_realtime(
    cfg: PipelineConfig,
    spec: pipeline.RunSpec,
    *,
    poll_s: float = 30.0,
    give_up_after: timedelta = timedelta(hours=4),
    max_attempts: int = 3,
    no_upload: bool = False,
    source=None,
    engine=None,
    uploader=None,
    index_backend=None,
    available: Callable[[], set[int]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Process one run as its files arrive; returns the run's final manifest.

    Args:
        cfg: Configuration (``upload_backend`` must be ``boto3`` or ``upload: local``; the CLI uploader cannot send named files).
        spec: The run and the forecast hours wanted.
        poll_s: Seconds between bucket listings while waiting for files.
        give_up_after: Stop waiting this long after initialization; hours that never arrived are recorded as missing
            and the run is left ``complete: false`` in the index for a catch-up pass.
        max_attempts: Tries per hour. A listed hour that then reads as missing or fails is tried again on a later poll.
        no_upload: Compute and keep the stores locally; publish nothing.
        source, engine, uploader, index_backend: Injected for tests; default from the config.
        available: Returns the forecast hours that can be read now; default lists the source bucket.
        now, sleep: Clock and sleep, injected by tests so no test waits.

    Returns:
        The manifest, with ``status`` one of ``indexed`` (complete and switched to the archive key),
        ``published_partial`` (cut short; staging entry left in the index), ``complete`` (``no_upload``),
        ``missing`` (no file ever appeared) or one of the batch failure states of ``pipeline.finalize_run``.
    """
    if uploader is None and index_backend is None:
        uploader, index_backend = pipeline.make_backends(cfg, no_upload)
    if no_upload:
        uploader = None
    if uploader is not None:
        if not hasattr(uploader, "upload_files"):
            raise ValueError("realtime publishing needs upload_backend: boto3 (or upload: local); the aws CLI backend cannot upload named files")
        uploader.preflight()
        cleanup_staging(uploader, index_backend, now())
    parent_source = source if source is not None else pipeline.make_source(cfg)
    if available is None:
        available = lambda: parent_source.available(spec.init, spec.hours)
    give_up_at = spec.init + give_up_after
    extra = {"run": spec.run_id}

    # Nothing can be created before the first file exists: the stores take their grid from it.
    while not available():
        if now() >= give_up_at:
            logger.warning("no file of this run appeared before the give-up time", extra=extra)
            return {"run_id": spec.run_id, "status": "missing", "init": index.init_string(spec.init)}
        sleep(poll_s)

    prov = pipeline.provenance(cfg)
    st = pipeline.prepare_run(spec, cfg, parent_source, prov)
    if st.grid is None:  # the listed file vanished or could not be read; treat like "not there yet" would loop forever
        return st.manifest
    pub = _Publisher(st, uploader, index_backend, now)
    pub.static = (store_files(st.ptype_path)[0], store_files(st.sounding_path)[0])
    # A restarted job finds its earlier hours on disk; they are published again (cheap overwrites) so the index
    # entry and the bucket agree with the local stores whatever state the previous job died in.
    already_final = st.finalized_marker.exists()
    if not already_final:
        pub.publish()

    processes = pipeline.auto_processes(cfg, len(spec.hours)) if cfg.processes == "auto" else int(cfg.processes)
    logger.info("realtime run starting", extra=extra | {"hours": len(spec.hours), "pending": len(st.pending), "processes": processes, "give_up_at": give_up_at.isoformat()})
    executor = pipeline.make_executor(cfg, processes, parent_source, engine)
    grid = dataclasses.asdict(st.grid)
    futures: dict[Future, int] = {}
    attempts: dict[int, int] = {}
    in_flight: set[int] = set()
    try:
        while st.pending:
            ready = (available() & st.pending) - in_flight
            for fxx in sorted(ready):
                if attempts.get(fxx, 0) >= max_attempts:
                    continue
                attempts[fxx] = attempts.get(fxx, 0) + 1
                in_flight.add(fxx)
                task = pipeline.HourTask(spec.run_id, spec.init, fxx, spec.store_hours.index(fxx), str(st.ptype_path), str(st.sounding_path), str(st.marker(fxx)), grid)
                futures[executor.submit(pipeline._run_task, task)] = fxx
            exhausted = {h for h in st.pending if attempts.get(h, 0) >= max_attempts and h not in in_flight}
            if not futures:
                if st.pending <= exhausted:
                    break  # every remaining hour has used its attempts
                if now() >= give_up_at:
                    break
                sleep(poll_s)
                continue
            done, _ = wait(list(futures), timeout=poll_s, return_when=FIRST_COMPLETED)
            newly_done = False
            for f in done:
                fxx = futures.pop(f)
                in_flight.discard(fxx)
                try:
                    res = f.result()
                except BrokenProcessPool as err:
                    res = {"run": spec.run_id, "fxx": fxx, "status": "failed", "error": f"worker pool broken: {err}"}
                except Exception as err:  # noqa: BLE001
                    res = {"run": spec.run_id, "fxx": fxx, "status": "failed", "error": f"{type(err).__name__}: {err}"}
                st.manifest["hours"][str(fxx)] = res
                if res.get("status") == "done":
                    st.pending.discard(fxx)
                    newly_done = True
                # "missing" (listed, then not readable) and "failed" stay pending and are tried again on a later poll.
            st.save()
            if newly_done:
                pub.publish()
    finally:
        executor.shutdown(wait=True)
        pipeline.set_fetch_slots(None)

    for fxx in sorted(st.pending):
        rec = st.manifest["hours"].get(str(fxx), {})
        if rec.get("status") != "failed":  # a real failure keeps its record; finalize_run then refuses to upload
            st.manifest["hours"][str(fxx)] = {
                "status": "missing",
                "error": "not available before the give-up time" if attempts.get(fxx, 0) == 0 else rec.get("error", "not readable"),
            }
    complete = not st.pending
    st.pending = set()
    st.save()
    if not already_final:
        pub.publish()  # anything a failed upload left behind

    if not complete and uploader is not None:
        # Leave the hours we have on the site, at the staging key, and say so; a catch-up pass (batch mode) fills
        # the gaps from the files that arrive later and finalizes. Finalizing now would lock the gaps in.
        st.manifest["status"] = "published_partial" if pub.published else "empty"
        st.manifest["written_hours"] = st.done_hours()
        st.manifest["missing_hours"] = sorted(int(h) for h, r in st.manifest["hours"].items() if r.get("status") == "missing")
        st.manifest["finished"] = now().isoformat()
        st.save()
        pub.status("partial")
        logger.warning(
            "run cut short; left at the staging key for a catch-up pass", extra=extra | {"written": len(st.manifest["written_hours"]), "missing": st.manifest["missing_hours"]}
        )
        return st.manifest

    manifest = pipeline.finalize_run(st, cfg, uploader, index_backend, no_upload, entry_extra=pub.extra(complete=complete))
    if manifest.get("status") == "indexed":
        pub.published = set(manifest.get("written_hours", []))
    pub.status(str(manifest.get("status")))
    return manifest


def realtime_spec(init: str, forecast_hours: Sequence[int] | None = None) -> pipeline.RunSpec:
    """The :class:`pipeline.RunSpec` for one ``YYYYMMDDHH`` run (all hours of the cycle unless given)."""
    return pipeline.make_runs([init], forecast_hours)[0]
