"""End-to-end driver of the fast HRRR precipitation-type pipeline: one model run in, two Zarr stores out.

Unit of work and data flow
--------------------------
A **run** is one HRRR initialization time; its forecast hours (0-18, or 0-48 for the 00/06/12/18 UTC
cycles) are independent of each other, so the forecast hour is the unit that is scheduled. For every
forecast hour one worker process does, start to finish:

1. **fetch** - download the wgrib2 ``.idx`` sidecar and only the GRIB2 messages needed (hybrid-level TMP,
   SPFH, PRES, HGT, UGRD, VGRD on the lowest levels plus a dozen surface fields) with one
   ``obstore.get_ranges`` call (``reader.fetch_ranges``). This moves ~250 MB instead of the ~790 MB file.
2. **decode** - ecCodes in a thread pool (ecCodes releases the GIL; 6x at 8 threads, while process pools
   lose to pickling costs; ``benchmarks/results/reader_local.md``).
3. **columns** - heights above ground, ``columns.build_soundings`` (0 m policies 2 m T/Td, 10 m wind,
   surface pressure, as in the training data) on the 0-5000 m network grid with GRID-relative winds, and
   ``columns.network_inputs``.
4. **inference** - the evidential network through ``engine.make_engine`` (scaler and evidential head are
   folded into the engine), giving four class probabilities and the uncertainty ``u``.
5. **bourgouin** - the revised Bourgouin method (Birk et al. 2021) on its own, possibly deeper, height
   grid (``bourgouin_top_m``), T and Td only.
6. **rotate** - the stored soundings get EARTH-relative winds (what a person reading a hodograph expects);
   the network already consumed the grid-relative copy.
7. **write** - one forecast-hour slice of the gridded multiscale store and of the sounding store, then a
   small per-hour marker file that says "this hour is complete".

Predictions are made at every grid point; ``precip_mask`` is stored alongside but never used to blank
anything, so the website (or a later analysis) can decide what to show.

Parallelism inside one node
---------------------------
Hours are distributed over a pool of worker **processes** (spawned, not forked: forking a parent that has
started numba/OpenMP/torch thread pools can deadlock the child). Each worker loads the model once and
handles whole hours, using ``threads_per_process`` threads for every multi-threaded stage (ecCodes decode,
numba kernels, BLAS/torch). The thread budget is fixed through ``NUMBA_NUM_THREADS``, ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS`` (set in the parent before the
workers start, because BLAS reads them at import time) and ``torch.set_num_threads`` so that
``processes x threads_per_process`` matches the cores and nothing is oversubscribed. Whole hours per
process (rather than splitting one hour over all cores) is what gives throughput: the stages of one hour
scale sub-linearly (fetch is network bound, decode has ~200 messages, zarr writes are partly serial), but
independent hours overlap perfectly. Memory is bounded by the number of processes, each holding at most
one hour (measured peak in ``benchmarks/results/pipeline_local.md``).

On a GPU node each worker is pinned to one GPU with ``CUDA_VISIBLE_DEVICES`` (round-robin over the
workers) and runs the torch engine there. This was chosen over a single GPU "inference server" fed through
shared memory because inference is a small fraction of an hour's work (the CPU stages dominate) and the
server would add a 640 MB inter-process copy and a single point of failure per hour; with per-worker GPUs
nothing crosses a process boundary and a crashed worker loses only its own hour.

Robustness
----------
* A run directory (``<work_dir>/<YYYYMMDDHH>/``) holds the two stores, ``status/fFF.json`` markers and
  ``manifest.json``. Stores are created with every forecast-hour slot up front, so hours can be written in
  any order by any process (``writers`` guarantees disjoint objects per hour).
* An hour counts as done only when its marker exists; the marker is written after both stores were
  written, so a crash mid-hour simply means the hour is redone (writes are whole-chunk overwrites).
* Missing HRRR files (404) are not retried; they are recorded in the manifest and the run is finalised with
  the hours that exist. Transient fetch errors are retried with jittered exponential backoff.
* After all hours of a run are terminal: upload both stores, THEN update the index (only if the upload
  succeeded), then optionally delete the local copy. Runs already in the index are skipped.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, prange

from ptype.hrrr_fast import columns, index, reader, runlist, writers
from ptype.hrrr_fast.bourgouin import CLASS_NAMES, revised_bourgouin_from_sounding
from ptype.hrrr_fast.config import PipelineConfig

logger = logging.getLogger("ptype.hrrr_fast.pipeline")

#: HRRR categorical precipitation-type flags stored next to the ML output.
CATEGORICAL = ("CRAIN", "CSNOW", "CICEP", "CFRZR")
#: Hybrid-level variables needed for the network sounding (all six) ...
WIND_VARIABLES = ("UGRD", "VGRD")
#: ... and the ones also needed above 5 km for Bourgouin (no winds there: the method does not use them).
THERMO_VARIABLES = ("TMP", "SPFH", "PRES", "HGT")
#: Stage names in the order they run; the manifest and the benchmark report these.
STAGES = ("fetch", "decode", "columns", "inference", "bourgouin", "rotate", "write")
#: Top of the network height grid (the model was trained on 0-5000 m every 250 m).
NETWORK_TOP_M = float(columns.NETWORK_HEIGHTS[-1])

#: Minimum height above ground (m) of hybrid levels 1..50 over the whole CONUS grid in
#: hrrr.20231226 t00z f12 (``benchmarks/pipeline_benchmark.py bourgouin-depth``). Used only for the FIRST
#: guess of how many levels reach a given height everywhere; the pipeline checks every hour and reads more
#: levels automatically when the guess is too low (lower tropopause in the cold season, high terrain).
MIN_AGL_BY_LEVEL_M = (
    8, 32, 74, 140, 234, 352, 488, 633, 787, 950, 1123, 1304, 1496, 1702, 1927, 2179, 2462, 2768, 3087, 3413,
    3749, 4092, 4442, 4797, 5155, 5515, 5877, 6243, 6611, 6989, 7395, 7839, 8322, 8844, 9409, 10024, 10700,
    11454, 12286, 13149, 13973, 14752, 15542, 16378, 17267, 18221, 19264, 20420, 21733, 23301,
)  # fmt: skip

_STD_LOG_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


# =============================================================================================
# Logging
# =============================================================================================
class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep in a PBS log and to load with pandas for post-mortems.

    Anything passed with ``extra={...}`` (run, fxx, stage, seconds, ...) becomes a top-level key, so a log
    line is self-describing and no message parsing is needed.
    """

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "pid": record.process,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD_LOG_ATTRS and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    """Configure the root logger (called in the parent and again in every spawned worker)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


# =============================================================================================
# Runs
# =============================================================================================
def parse_init(text: str) -> datetime:
    """``YYYYMMDDHH`` -> aware UTC datetime."""
    return datetime.strptime(text.strip(), "%Y%m%d%H").replace(tzinfo=UTC)


@dataclass(frozen=True)
class RunSpec:
    """One model run to process.

    Attributes:
        init: Initialization time (aware, UTC).
        hours: Forecast hours to process in this invocation.
    """

    init: datetime
    hours: tuple[int, ...]

    @property
    def run_id(self) -> str:
        """``YYYYMMDDHH``, the run directory name and the token used on the command line."""
        return f"{self.init:%Y%m%d%H}"

    @property
    def store_hours(self) -> list[int]:
        """Forecast-hour axis of the stores: always 0..max for the cycle (plus any extra hours asked for).

        Deriving the axis from the init cycle, not from the hours of this invocation, keeps the store
        layout identical whatever subset is processed, so a later invocation can fill in more hours of the
        same store (``fh_index`` never changes meaning).
        """
        return sorted(set(range(runlist.max_forecast_hour(self.init.hour) + 1)) | set(self.hours))


def make_runs(inits: Iterable[str], forecast_hours: Sequence[int] | None = None) -> list[RunSpec]:
    """Runs for ``YYYYMMDDHH`` strings; hours default to 0..max for each cycle (``runlist.max_forecast_hour``)."""
    out = []
    for s in inits:
        init = parse_init(s)
        hours = tuple(forecast_hours) if forecast_hours is not None else tuple(range(runlist.max_forecast_hour(init.hour) + 1))
        out.append(RunSpec(init, hours))
    return out


def read_runs_file(path: str | os.PathLike, forecast_hours: Sequence[int] | None = None) -> list[RunSpec]:
    """Read ``YYYYMMDDHH maxfh`` lines (as written by ``python -m ptype.hrrr_fast.runlist --out``).

    Blank lines and ``#`` comments are ignored. ``forecast_hours`` (the ``--forecast-hours`` override)
    replaces the 0..maxfh range of every line.
    """
    runs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        init = parse_init(parts[0])
        maxfh = int(parts[1]) if len(parts) > 1 else runlist.max_forecast_hour(init.hour)
        hours = tuple(forecast_hours) if forecast_hours is not None else tuple(range(maxfh + 1))
        runs.append(RunSpec(init, hours))
    return runs


def parse_hours(text: str) -> list[int]:
    """``"0-18"``, ``"11,12,13"`` or ``"0-3,12"`` -> sorted list of hours."""
    hours: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        hours.update(range(int(lo), int(hi or lo) + 1))
    return sorted(hours)


# =============================================================================================
# Threads
# =============================================================================================
def thread_env(threads: int) -> dict[str, str]:
    """Environment that limits every thread pool a worker can start to ``threads``.

    These must be in the environment BEFORE numpy/numba/torch are imported in the worker (OpenBLAS, MKL,
    Accelerate and the numba thread pool size themselves at import/first use), which is why the parent sets
    them before spawning the workers instead of each worker setting them after start-up.
    """
    t = str(int(threads))
    return {
        "NUMBA_NUM_THREADS": t,
        "OMP_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
        "NUMEXPR_MAX_THREADS": t,
    }


def available_cores() -> int:
    """Cores this job may use: PBS ``NCPUS`` first (the scheduler's grant), then the CPU affinity mask."""
    for var in ("NCPUS", "PBS_NCPUS"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def detect_gpus() -> int:
    """Number of visible NVIDIA GPUs without initialising CUDA in the parent (``nvidia-smi -L``)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return len([g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def auto_processes(cfg: PipelineConfig, n_tasks: int | None = None) -> int:
    """Worker count: limited by cores (``cores // threads``) and by memory (``usable / per-worker peak``).

    The memory term uses the MEASURED per-worker peak (``memory_per_process_gb``) so the pool cannot push a
    235 GB Derecho node into swap/OOM; the core term avoids oversubscription. No more workers than tasks.
    """
    import psutil

    by_cores = max(1, available_cores() // cfg.threads_per_process)
    usable = psutil.virtual_memory().available * cfg.memory_fraction
    by_mem = max(1, int(usable // (cfg.memory_per_process_gb * 2**30)))
    p = min(by_cores, by_mem)
    if n_tasks is not None:
        p = min(p, max(1, n_tasks))
    return p


def peak_rss_bytes() -> int:
    """Lifetime peak resident set size of this process (``ru_maxrss`` is KiB on Linux, bytes on macOS)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


# =============================================================================================
# Sources and retries
# =============================================================================================
class MissingInputError(Exception):
    """The HRRR file (or its idx) does not exist: not yet published, or a gap in the archive. Not retried."""


def is_transient(exc: BaseException) -> bool:
    """Whether a fetch error is worth retrying.

    Network/IO and obstore service errors are; programming and data errors (KeyError for a message absent
    from the idx, ValueError from a bad shape, ...) are not, because retrying cannot fix them and would only
    delay the failure report.
    """
    if isinstance(exc, MissingInputError | FileNotFoundError):
        return False
    try:
        from obstore.exceptions import BaseError, NotFoundError

        if isinstance(exc, NotFoundError):
            return False
        if isinstance(exc, BaseError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, OSError | TimeoutError | ConnectionError)


def call_with_retries(
    fn: Callable[[], Any],
    *,
    retries: int,
    backoff_s: float,
    max_backoff_s: float,
    what: str,
    log_extra: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` and retry transient failures with full-jitter exponential backoff.

    Full jitter (``uniform(0, min(cap, base * 2**attempt))``) spreads retries of many workers that failed
    together (e.g. a brief S3 throttle) so they do not hammer the service again in lockstep.

    Raises:
        MissingInputError: Immediately, for a missing object.
        Exception: The last error once ``retries`` is exhausted, or immediately for a non-transient error.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except FileNotFoundError as err:
            raise MissingInputError(f"{what}: not found") from err
        except Exception as err:
            if type(err).__name__ == "NotFoundError":
                raise MissingInputError(f"{what}: not found") from err
            if attempt >= retries or not is_transient(err):
                raise
            delay = random.uniform(0.0, min(max_backoff_s, backoff_s * 2**attempt))
            logger.warning("transient error, retrying", extra={"what": what, "attempt": attempt + 1, "delay_s": round(delay, 2), "error": repr(err), **(log_extra or {})})
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class ReadStats:
    """Cost of one read call: seconds and bytes (payload = message bytes; requested = after coalescing).

    ``fetch_wait_s`` is the time spent waiting for a download slot (``max_concurrent_fetches``), already
    included in ``fetch_s``; ``fetch_start_unix``/``fetch_end_unix`` bracket the byte-range downloads in wall-clock
    time so a network study can see how downloads of different workers overlapped.
    """

    fetch_s: float = 0.0
    decode_s: float = 0.0
    fetch_wait_s: float = 0.0
    n_messages: int = 0
    n_requests: int = 0
    payload_bytes: int = 0
    requested_bytes: int = 0
    fetch_start_unix: float | None = None
    fetch_end_unix: float | None = None

    def add(self, other: ReadStats) -> None:
        """Accumulate another read (used when extra levels are fetched): sums, and the outer time bracket."""
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name == "fetch_start_unix":
                setattr(self, f.name, min(v for v in (a, b) if v is not None) if (a, b) != (None, None) else None)
            elif f.name == "fetch_end_unix":
                setattr(self, f.name, max(v for v in (a, b) if v is not None) if (a, b) != (None, None) else None)
            else:
                setattr(self, f.name, a + b)


# ---------------------------------------------------------------------------------------------
# Download slots (max_concurrent_fetches)
# ---------------------------------------------------------------------------------------------
#: Cross-process semaphore limiting simultaneous byte-range downloads on this node, or None (no cap). Set in
#: each worker by the pool initializer (and in the parent for ``processes: 0``).
_FETCH_SLOTS = None


def set_fetch_slots(semaphore) -> None:
    """Install (or with ``None`` remove) the node-wide download semaphore for this process."""
    global _FETCH_SLOTS
    _FETCH_SLOTS = semaphore


@contextlib.contextmanager
def fetch_slot(stats: ReadStats | None = None):
    """Hold one download slot while the body runs; record the wait and the download's wall-clock bracket.

    Why a cap independent of the number of workers: on Derecho 15 workers downloading at once made the median
    fetch 55.5 s per hour against 2.4 s for one worker alone, i.e. the node link (or the per-connection S3
    throughput) is shared, not multiplied. Compute stages do not have that limit, so the cap lets e.g. 15
    workers compute while only a few download, without changing the process layout. The slot is acquired per
    attempt (inside the retry loop), so a worker sleeping in a retry back-off does not block the others.
    """
    t0 = time.perf_counter()
    sem = _FETCH_SLOTS
    if sem is not None:
        sem.acquire()
    try:
        if stats is not None:
            stats.fetch_wait_s += time.perf_counter() - t0
            now = time.time()
            stats.fetch_start_unix = now if stats.fetch_start_unix is None else min(stats.fetch_start_unix, now)
        yield
    finally:
        if stats is not None:
            stats.fetch_end_unix = time.time()
        if sem is not None:
            sem.release()


def level_spec(wind: tuple[int, int], thermo: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Per-variable 0-based level ranges for ``reader.select_messages(n_levels=...)``; empty ranges are dropped."""
    return {v: wind for v in WIND_VARIABLES} | {v: thermo for v in THERMO_VARIABLES}


class GribHour:
    """One HRRR wrfnat file: its idx entries, and reads of any set of hybrid levels.

    Reads are incremental: :meth:`read` takes, per variable group, the range of 0-based level indices to
    add, so a top-up read after the height check fetches only the missing levels.
    """

    def __init__(self, store, key: str, entries: list[reader.IdxEntry], fxx: int, cfg: PipelineConfig):
        self.store = store
        self.key = key
        self.entries = entries
        self.fxx = fxx
        self.cfg = cfg

    def plan(self, wind: tuple[int, int], thermo: tuple[int, int], surface: bool) -> reader.ReadPlan:
        """The read plan for wind levels ``wind = (lo, hi)``, thermo levels ``thermo`` and (optionally) surface fields.

        ``reader.select_messages`` with a per-variable mapping gives each variable its own level range (the
        winds are only needed to 5 km, T/q/p/z up to the Bourgouin top), raises if any planned level is
        absent from the idx, and records the ranges in the plan so ``reader.decode`` sizes each cube itself.
        """
        return reader.select_messages(self.entries, n_levels=level_spec(wind, thermo), fxx=self.fxx, surface_fields=None if surface else {}, include_apcp=False)

    def planned_bytes(self, wind: tuple[int, int], thermo: tuple[int, int], surface: bool = True) -> tuple[int, int, int]:
        """``(payload bytes, requested bytes after coalescing, requests)`` without fetching (dry run)."""
        plan = self.plan(wind, thermo, surface)
        ends = [m.entry.end if m.entry.end is not None else m.entry.offset + 50_000_000 for m in plan.messages]
        merged = reader.merged_ranges(plan.starts, ends, self.cfg.coalesce)
        return plan.total_bytes(), sum(e - s for s, e in merged), len(merged)

    def read(self, wind: tuple[int, int], thermo: tuple[int, int], surface: bool, threads: int, log_extra: dict | None = None):
        """Fetch and decode the planned messages; returns ``(fields, grid, stats)``.

        Hybrid variables come back as ``(hi - lo, ny, nx)`` cubes for the requested level range only (the
        caller concatenates top-ups). ``reader.decode`` writes each message from the ecCodes threads straight
        into its slot of a cube it allocated for that variable's own range, and checks first that every
        planned level slot is filled exactly once.
        """
        plan = self.plan(wind, thermo, surface)
        stats = ReadStats(n_messages=len(plan.messages), payload_bytes=plan.total_bytes())

        def fetch():
            with fetch_slot(stats):
                return reader.fetch_ranges(self.store, self.key, plan, coalesce=self.cfg.coalesce, threads=self.cfg.fetch_threads)

        t0 = time.perf_counter()
        bufs = call_with_retries(
            fetch,
            retries=self.cfg.fetch_retries,
            backoff_s=self.cfg.fetch_backoff_s,
            max_backoff_s=self.cfg.fetch_max_backoff_s,
            what=f"ranges {self.key}",
            log_extra=log_extra,
        )
        stats.fetch_s = time.perf_counter() - t0
        ends = [m.entry.offset + len(b) for m, b in zip(plan.messages, bufs, strict=True)]
        merged = reader.merged_ranges(plan.starts, ends, self.cfg.coalesce)
        stats.n_requests = len(merged)
        stats.requested_bytes = sum(e - s for s, e in merged)
        t0 = time.perf_counter()
        grid = reader.grid_metadata(bufs[0])
        out = reader.decode(plan, bufs, decoder="eccodes", workers=threads, pool="thread")
        del bufs
        stats.decode_s = time.perf_counter() - t0
        return out, grid, stats


class GribSource:
    """HRRR wrfnat files from a public bucket (AWS or the Google mirror, anonymous) or a local mirror directory."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.store = reader.open_store(None if cfg.source == "s3" else cfg.source)

    def open_hour(self, init: datetime, fxx: int, log_extra: dict | None = None) -> GribHour:
        """Fetch and parse the idx (retrying transient errors); :class:`MissingInputError` if absent."""
        key = reader.hrrr_key(init, init.hour, fxx)
        text = call_with_retries(
            lambda: reader.fetch_idx(self.store, key),
            retries=self.cfg.fetch_retries,
            backoff_s=self.cfg.fetch_backoff_s,
            max_backoff_s=self.cfg.fetch_max_backoff_s,
            what=f"idx {key}",
            log_extra=log_extra,
        )
        return GribHour(self.store, key, reader.parse_idx(text), fxx, self.cfg)

    def available(self, init: datetime, hours: Sequence[int]) -> set[int]:
        """Forecast hours of a run whose GRIB file AND ``.idx`` are both in the bucket (one listing).

        Both are required because the two public copies publish them in opposite orders: on AWS the idx lands
        0.5-1.2 minutes after its GRIB file, on Google it is created before it (up to 12.5 minutes, measured
        on four runs; benchmarks/results/casper/arrival.md). Objects appear whole on both, so "both listed"
        means the hour can be read.
        """
        import obstore

        first = reader.hrrr_key(init, init.hour, 0)
        directory, stem = first[: first.rindex("/")], first[: first.rindex("f00.grib2")]
        # A listing prefix must end at a "/" (object stores match whole path segments), and the day's directory
        # holds every cycle and product (thousands of objects). Keys come back sorted, so start just before this
        # cycle's wrfnat files and stop at the first key past them: one page instead of the whole day.
        # (A local mirror lists in directory order, not sorted, so there every name is looked at.)
        from obstore.store import LocalStore

        ordered = not isinstance(self.store, LocalStore)
        names: set[str] = set()
        past = False
        for batch in obstore.list(self.store, prefix=directory, offset=stem):
            for m in batch:
                if m["path"].startswith(stem):
                    names.add(m["path"])
                elif ordered:
                    past = True
                    break
            if past:
                break
        out = set()
        for fxx in hours:
            key = reader.hrrr_key(init, init.hour, fxx)
            if key in names and key + ".idx" in names:
                out.add(fxx)
        return out

    def grid(self, init: datetime, hours: Sequence[int]) -> reader.GridMetadata | None:
        """Grid of a run from ONE small message (surface height) of its first available hour.

        The parent needs the grid to create the stores before any worker writes; reading it from the file
        (rather than hard-coding HRRR constants) means a grid change is picked up, and the ~1.5 MB read
        doubles as an availability check. ``None`` if no hour of the run exists.
        """
        for fxx in hours:
            try:
                hour = self.open_hour(init, fxx)
            except MissingInputError:
                continue
            plan = reader.select_messages(hour.entries, n_levels=1, fxx=fxx, hybrid_variables=(), surface_fields={"HGT_sfc": ("HGT", "surface")}, include_apcp=False)
            buf = call_with_retries(
                lambda hour=hour, plan=plan: reader.fetch_ranges(self.store, hour.key, plan, coalesce=0),
                retries=self.cfg.fetch_retries,
                backoff_s=self.cfg.fetch_backoff_s,
                max_backoff_s=self.cfg.fetch_max_backoff_s,
                what=f"grid {hour.key}",
            )
            return reader.grid_metadata(buf[0])
        return None


# ---------------------------------------------------------------------------------------------
# Synthetic source (tests and orchestration dry runs; no GRIB, no network)
# ---------------------------------------------------------------------------------------------
_SYN_PROJ4 = "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +R=6371229.0 +units=m +no_defs"


def synthetic_grid(ny: int = 24, nx: int = 32) -> reader.GridMetadata:
    """A small south-first Lambert grid in the HRRR projection (rows are the southern edge first)."""
    import pyproj

    x0, y0 = -300000.0, -200000.0
    lon, lat = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(_SYN_PROJ4), "EPSG:4326", always_xy=True).transform(x0, y0)
    return reader.GridMetadata(
        nx=nx, ny=ny, dx=3000.0, dy=3000.0, lon_0=-97.5, lat_0=38.5, lat_1=38.5, lat_2=38.5, earth_radius=6371229.0, shape_of_the_earth=6,
        lat_first=float(lat), lon_first=float(lon), x_first=x0, y_first=y0, i_scans_negatively=0, j_scans_positively=1,
        j_points_consecutive=0, row0_is_south=True, proj4=_SYN_PROJ4,
    )  # fmt: skip


def synthetic_level_agl(k: np.ndarray, terrain: np.ndarray) -> np.ndarray:
    """Height above ground of 0-based hybrid level ``k``: ~5.4 km at level 25, ~13 km at level 40.

    Levels are squeezed over high terrain (as terrain-following coordinates are), so the high-terrain
    columns are the ones that need more levels, like the Rockies in HRRR.
    """
    return (12.0 * (k + 1.0) ** 1.9) * (1.0 - 0.15 * terrain / 2000.0)


class SyntheticHour:
    """Deterministic fields of one fake forecast hour, any number of levels (see :class:`SyntheticSource`)."""

    def __init__(self, source: SyntheticSource, init: datetime, fxx: int):
        self.source = source
        self.init = init
        self.fxx = fxx

    def planned_bytes(self, wind, thermo, surface=True):
        n = (wind[1] - wind[0]) * 2 + (thermo[1] - thermo[0]) * 4 + (12 if surface else 0)
        return n * 1000, n * 1000, 1

    def read(self, wind, thermo, surface, threads, log_extra=None):
        src = self.source
        key = (self.init.isoformat(), self.fxx)
        src.reads[key] = src.reads.get(key, 0) + 1
        if src.fail_first and src.reads[key] <= src.fail_first:
            raise ConnectionError(f"synthetic transient failure {src.reads[key]}")
        stats = ReadStats(n_requests=1)
        t0 = time.perf_counter()
        with fetch_slot(stats):  # same download-slot discipline as the GRIB source
            if src.fetch_sleep_s:
                time.sleep(src.fetch_sleep_s)
        stats.fetch_s = time.perf_counter() - t0
        g = src.grid_meta
        ny, nx = g.ny, g.nx
        rng = np.random.default_rng(int(self.init.timestamp()) // 3600 * 100 + self.fxx)
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        terrain = (2000.0 * np.exp(-(((ii - nx * 0.7) / (nx * 0.15)) ** 2 + ((jj - ny * 0.5) / (ny * 0.3)) ** 2))).astype(np.float32)
        t_sfc = (273.15 + 8.0 * np.sin(ii / nx * 3.0 + self.fxx * 0.3) - 6.0 * jj / ny).astype(np.float64)
        out: dict[str, np.ndarray] = {}

        def cube(lo, hi, fn):
            k = np.arange(lo, hi, dtype=np.float64)[:, None, None]
            z = synthetic_level_agl(k, terrain[None].astype(np.float64))
            return fn(k, z).astype(np.float32)

        def temp(k, z):
            # A warm nose near 1.5 km in part of the domain gives freezing-rain/ice-pellet columns.
            nose = 6.0 * np.exp(-(((z - 1500.0) / 500.0) ** 2)) * (ii[None] > nx // 2)
            return t_sfc[None] - 4.0 - 0.0065 * z + nose

        for name, fn in (
            ("TMP", temp),
            ("SPFH", lambda k, z: 0.006 * np.exp(-z / 3000.0) * (0.6 + 0.4 * np.cos(ii[None] / 5.0))),
            ("PRES", lambda k, z: 100000.0 * np.exp(-z / 8000.0) - 0.0 * k),
            ("HGT", lambda k, z: z + terrain[None]),
        ):
            if thermo[1] > thermo[0]:
                out[name] = cube(*thermo, fn)
        for name, fn in (("UGRD", lambda k, z: 5.0 + 0.004 * z + 0.0 * k), ("VGRD", lambda k, z: -3.0 + 0.002 * z + 0.0 * k)):
            if wind[1] > wind[0]:
                out[name] = cube(*wind, fn)
        if surface:
            out["HGT_sfc"] = terrain
            out["PRES_sfc"] = (100000.0 * np.exp(-1.0 / 8000.0) * np.ones((ny, nx))).astype(np.float32)
            out["TMP_2m"] = (t_sfc - 4.0).astype(np.float32)
            out["DPT_2m"] = (t_sfc - 6.0).astype(np.float32)
            out["UGRD_10m"] = np.full((ny, nx), 4.0, np.float32)
            out["VGRD_10m"] = np.full((ny, nx), -2.0, np.float32)
            flags = rng.random((4, ny, nx)) < 0.25
            for k, f in zip(CATEGORICAL, flags, strict=True):
                out[k] = f.astype(np.float32)
            out["PRATE"] = rng.gamma(0.5, 1e-4, size=(ny, nx)).astype(np.float32)
        n = sum(1 for v in out.values() for _ in range(v.shape[0] if v.ndim == 3 else 1))
        stats.n_messages, stats.payload_bytes, stats.requested_bytes = n, n * 1000, n * 1000
        return out, g, stats


class SyntheticSource:
    """Fake HRRR source with the same interface as :class:`GribSource`.

    ``source: "synthetic"`` in the config, optionally with options ``"synthetic:missing=3,5;fail=1;sleep=0.2"``:
    ``missing`` hours raise :class:`MissingInputError`, every hour's first ``fail`` reads raise a transient
    ``ConnectionError`` (to exercise the retry path), and ``sleep`` simulates a download of that many seconds
    inside a download slot (to exercise ``max_concurrent_fetches``).
    """

    def __init__(self, missing: Iterable[int] = (), fail_first: int = 0, ny: int = 24, nx: int = 32, fetch_sleep_s: float = 0.0, cfg: PipelineConfig | None = None):
        self.missing = set(missing)
        self.fetch_sleep_s = float(fetch_sleep_s)
        self.fail_first = int(fail_first)
        self.grid_meta = synthetic_grid(ny, nx)
        self.reads: dict[tuple[str, int], int] = {}
        self.cfg = cfg

    @classmethod
    def from_spec(cls, spec: str, cfg: PipelineConfig | None = None) -> SyntheticSource:
        """Parse ``synthetic[:missing=3,5;fail=1;ny=24;nx=32;sleep=0.2]``."""
        kwargs: dict[str, Any] = {}
        _, _, opts = spec.partition(":")
        for item in filter(None, opts.split(";")):
            k, _, v = item.partition("=")
            if k == "missing":
                kwargs["missing"] = parse_hours(v) if v else []
            elif k == "fail":
                kwargs["fail_first"] = int(v)
            elif k in ("ny", "nx"):
                kwargs[k] = int(v)
            elif k == "sleep":
                kwargs["fetch_sleep_s"] = float(v)
            else:
                raise ValueError(f"unknown synthetic option {k!r}")
        return cls(cfg=cfg, **kwargs)

    def open_hour(self, init: datetime, fxx: int, log_extra: dict | None = None) -> SyntheticHour:
        if fxx in self.missing:
            raise MissingInputError(f"synthetic f{fxx:02d} missing")
        return SyntheticHour(self, init, fxx)

    def available(self, init: datetime, hours: Sequence[int]) -> set[int]:
        return {h for h in hours if h not in self.missing}

    def grid(self, init: datetime, hours: Sequence[int]) -> reader.GridMetadata | None:
        return self.grid_meta if any(h not in self.missing for h in hours) else None


def make_source(cfg: PipelineConfig):
    """The configured source object."""
    if cfg.source.startswith("synthetic"):
        return SyntheticSource.from_spec(cfg.source, cfg)
    return GribSource(cfg)


# =============================================================================================
# Reading with the automatic level check
# =============================================================================================
def levels_for_height(top_m: float, margin: int = 1) -> int:
    """First guess of how many hybrid levels reach ``top_m`` AGL everywhere (+ ``margin``), from :data:`MIN_AGL_BY_LEVEL_M`."""
    for k, h in enumerate(MIN_AGL_BY_LEVEL_M):
        if h >= top_m:
            return min(len(MIN_AGL_BY_LEVEL_M), k + 1 + margin)
    return len(MIN_AGL_BY_LEVEL_M)


@dataclass
class LevelState:
    """Per-worker level counts; grow when an hour needed more, so later hours start with enough."""

    n_wind: int
    n_thermo: int


def read_hour_fields(hour, cfg: PipelineConfig, levels: LevelState, threads: int, log_extra: dict | None = None):
    """Read one hour with enough hybrid levels for the network (5 km) and Bourgouin (``bourgouin_top_m``) grids.

    The requirement is checked on the decoded heights: the top level read must be at least the top target
    height at EVERY column, because ``build_soundings`` clamps (repeats the top value) above the top level,
    which would silently corrupt the profile there. If a column falls short, ``level_step`` more levels of
    the affected variable group are fetched (only the new levels) and the check repeats, up to
    ``max_levels``. Each top-up is logged and counted in the returned info.

    Returns:
        ``(fields, grid, stats, info)``; ``fields["TMP"|"SPFH"|"PRES"|"HGT"]`` have ``n_thermo`` levels,
        ``UGRD``/``VGRD`` have ``n_wind`` levels (``n_wind <= n_thermo``).
    """
    n_wind = min(levels.n_wind, cfg.max_levels)
    n_thermo = min(max(levels.n_thermo, n_wind), cfg.max_levels)
    retry = {"retries": cfg.fetch_retries, "backoff_s": cfg.fetch_backoff_s, "max_backoff_s": cfg.fetch_max_backoff_s, "log_extra": log_extra}
    fields, grid, stats = call_with_retries(lambda: hour.read((0, n_wind), (0, n_thermo), True, threads, log_extra), what="read", **retry)
    info = {"rereads": 0, "initial_levels": [n_wind, n_thermo]}
    while True:
        z_top_min = lambda n: float(np.min(fields["HGT"][n - 1] - fields["HGT_sfc"]))
        need_wind = z_top_min(n_wind) < NETWORK_TOP_M and n_wind < cfg.max_levels
        need_thermo = z_top_min(n_thermo) < cfg.bourgouin_top_m and n_thermo < cfg.max_levels
        if not (need_wind or need_thermo):
            break
        new_wind = min(cfg.max_levels, n_wind + cfg.level_step) if need_wind else n_wind
        new_thermo = min(cfg.max_levels, max(n_thermo + (cfg.level_step if need_thermo else 0), new_wind))
        logger.warning(
            "top level too low at some columns; reading more levels",
            extra={
                **(log_extra or {}),
                "wind_levels": [n_wind, new_wind],
                "thermo_levels": [n_thermo, new_thermo],
                "z_top_min_wind": z_top_min(n_wind),
                "z_top_min_thermo": z_top_min(n_thermo),
            },
        )
        extra, _, st = call_with_retries(lambda w=(n_wind, new_wind), t=(n_thermo, new_thermo): hour.read(w, t, False, threads, log_extra), what="read more levels", **retry)
        stats.add(st)
        for name, arr in extra.items():
            fields[name] = np.concatenate([fields[name], arr], axis=0)
        n_wind, n_thermo = new_wind, new_thermo
        info["rereads"] += 1
    levels.n_wind, levels.n_thermo = max(levels.n_wind, n_wind), max(levels.n_thermo, n_thermo)
    info["levels"] = [n_wind, n_thermo]
    info["z_top_min_wind_m"] = float(np.min(fields["HGT"][n_wind - 1] - fields["HGT_sfc"]))
    info["z_top_min_thermo_m"] = float(np.min(fields["HGT"][n_thermo - 1] - fields["HGT_sfc"]))
    return fields, grid, stats, info


# =============================================================================================
# Science of one hour
# =============================================================================================
def bourgouin_heights(top_m: float, dz_m: float = 250.0) -> np.ndarray:
    """0, dz, ..., top (m AGL). With dz = 250 the first 21 levels ARE the network heights."""
    return np.arange(0.0, top_m + dz_m / 2.0, dz_m)


def grid_lonlat(grid: reader.GridMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Longitude/latitude of every grid point (array order) from the grid's own PROJ string."""
    import pyproj

    x, y = grid.xy_coords()
    xx, yy = np.meshgrid(x, y)
    tf = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(grid.proj4), "EPSG:4326", always_xy=True)
    return tf.transform(xx, yy)


@njit(parallel=True, cache=True)
def _rotate_sounding_kernel(snd, cosa, sina):
    # Same arithmetic as columns._rotate_kernel (u_e = c u + s v, v_e = -s u + c v in float64), but on the
    # column-major (n, 5, heights) sounding IN PLACE, so no (heights, ny, nx) transposed copies are needed.
    n = snd.shape[0]
    nh = snd.shape[2]
    for c in prange(n):
        cs = cosa[c]
        sn = sina[c]
        for t in range(nh):
            uu = np.float64(snd[c, 2, t])
            vv = np.float64(snd[c, 3, t])
            snd[c, 2, t] = cs * uu + sn * vv
            snd[c, 3, t] = -sn * uu + cs * vv


def rotate_sounding_to_earth(snd: np.ndarray, rotation: tuple[np.ndarray, np.ndarray]) -> None:
    """Rotate the u/v groups of a ``(n, 5, h)`` sounding from grid- to earth-relative, in place.

    Rotating after vertical interpolation is exact: the angle is constant in a column and interpolation is
    linear (``columns.build_soundings`` docstring).
    """
    cosa = np.ascontiguousarray(rotation[0], dtype=np.float64).reshape(-1)
    sina = np.ascontiguousarray(rotation[1], dtype=np.float64).reshape(-1)
    if snd.shape[0] != cosa.size or snd.shape[1] != 5:
        raise ValueError("sounding must be (ny*nx, 5, heights) matching the rotation arrays")
    _rotate_sounding_kernel(snd, cosa, sina)


def _surface_kwargs(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "surface_temperature": cfg.surface_temperature,
        "surface_pressure": cfg.surface_pressure,
        "pressure_interp": cfg.pressure_interp,
        "clip_surface_dewpoint": cfg.clip_surface_dewpoint,
    }


def run_bourgouin(fields: dict[str, np.ndarray], z_agl: np.ndarray, snd: np.ndarray, cfg: PipelineConfig, return_diagnostics: bool = False):
    """Revised Bourgouin on the ``0..bourgouin_top_m`` grid; returns ``(probs, cls[, diag], n_short)``.

    * Top = 5000 m: the network sounding already holds T, Td and p on exactly that grid, so it is used as is.
    * Deeper: T, Td and p are interpolated to the deeper grid with the SAME kernel and 0 m policies as the
      network sounding (``build_soundings``; winds are not needed, so the temperature cube is passed as a
      dummy wind and its output ignored). Because each target height is computed independently, the
      0-5000 m part is bit-identical to the network sounding. The work is done in blocks of grid rows so the
      temporary ``(columns, 5, levels)`` array stays ~200 MB instead of ~2 GB for the whole grid.

    ``n_short`` counts columns whose top native level is below the grid top (should be 0).
    """
    heights = bourgouin_heights(cfg.bourgouin_top_m, cfg.bourgouin_dz_m)
    if heights.size == snd.shape[2] and np.allclose(heights, columns.NETWORK_HEIGHTS[: heights.size]):
        res = revised_bourgouin_from_sounding(snd, heights, return_diagnostics=return_diagnostics)
        return (*res, 0)
    _, ny, nx = z_agl.shape
    t, q, p = fields["TMP"], fields["SPFH"], fields["PRES"]
    ncol = ny * nx
    probs = np.empty((ncol, 4), np.float32)
    cls = np.empty(ncol, np.uint8)
    diag = np.empty((ncol, 6), np.float32) if return_diagnostics else None
    rows = max(1, int(cfg.bourgouin_rows_per_chunk))
    buf = np.empty((rows * nx, 5, heights.size), np.float32)
    n_short = 0
    for j0 in range(0, ny, rows):
        j1 = min(ny, j0 + rows)
        sl = slice(j0, j1)
        c0, c1 = j0 * nx, j1 * nx
        out = buf[: c1 - c0]
        tt = t[:, sl]
        _, ns = columns.build_soundings(
            z_agl[:, sl], tt, q[:, sl], p[:, sl], tt, tt,
            t2m_k=fields["TMP_2m"][sl], td2m_k=fields["DPT_2m"][sl], psfc_pa=fields["PRES_sfc"][sl],
            target_heights=heights, surface_wind="interp", out=out, **_surface_kwargs(cfg),
        )  # fmt: skip
        n_short += ns
        res = revised_bourgouin_from_sounding(out, heights, return_diagnostics=return_diagnostics)
        probs[c0:c1], cls[c0:c1] = res[0], res[1]
        if return_diagnostics:
            diag[c0:c1] = res[2]
    return (probs, cls, diag, n_short) if return_diagnostics else (probs, cls, n_short)


def sounding_codecs(cfg: PipelineConfig) -> tuple[list | None, list | None]:
    """``(filters, compressors)`` for the sounding store; ``(None, None)`` means the writers' default.

    The precision decision (0.1-unit int16, chosen by the project scientist) lives in ONE place,
    ``writers.default_sounding_encoding``; ``sounding_encoding: default`` therefore passes nothing and lets the
    writers decide. Naming an encoding (``int16-0.1``, ``int32-0.01``, ``lossless``) pins it explicitly through
    ``writers.sounding_encoding``, e.g. to reproduce or compare older stores.
    """
    if cfg.sounding_encoding == "default":
        return None, None
    return writers.sounding_encoding(cfg.sounding_encoding)


def ptype_variables() -> dict[str, writers.VariableSpec]:
    """Gridded variables of a run store: the writers' defaults plus the Bourgouin fields.

    Bourgouin probabilities are the paper's independent, un-normalised fractions (0-1), stored like the ML
    probabilities (uint8 steps of 1/254, ``mean`` for coarse levels). ``BOURG_class`` is categorical, so
    coarse levels use ``mode`` (a mean of class codes is meaningless); 254 = no type, 255 = missing/invalid.
    """
    v = writers.default_ptype_variables()
    names = {"rain": "rain", "snow": "snow", "icep": "ice pellets", "frzr": "freezing rain"}
    for c in CLASS_NAMES:
        v[f"BOURG_{c}"] = writers.VariableSpec(
            attrs={"long_name": f"revised Bourgouin (Birk et al. 2021) probability of {names[c]}", "units": "1", "comment": "independent probabilities; the four do not sum to 1"},
            aggregation="mean",
            filters=writers.quantized_float(1.0 / 254.0, "uint8"),
        )
    v["BOURG_class"] = writers.VariableSpec(
        dtype="uint8",
        fill_value=255,
        aggregation="mode",
        attrs={"long_name": "revised Bourgouin dominant type", "flag_values": [0, 1, 2, 3, 254], "flag_meanings": "rain snow icep frzr none", "missing_value": 255},
    )
    return v


def compute_hour(fields: dict[str, np.ndarray], grid, engine, cfg: PipelineConfig, n_wind: int, rotation, timings: dict[str, float]):
    """Columns -> inference -> Bourgouin -> earth-relative winds for one decoded hour.

    ``fields`` is CONSUMED (arrays are deleted as soon as they are no longer needed, and ``HGT`` is turned
    into height above ground in place) to keep the per-worker peak memory low.

    Returns:
        ``(gridded, sounding, info)``: ``gridded`` maps store variable names to ``(ny, nx)`` arrays,
        ``sounding`` is the ``(ny*nx, 105)`` earth-relative sounding matrix, ``info`` holds sanity counts.
    """
    if not grid.row0_is_south:
        raise ValueError("north-first grids are not supported: the sounding store needs location = j * ni + i with j south to north")
    ny, nx = grid.ny, grid.nx
    t0 = time.perf_counter()
    z = fields.pop("HGT")
    z -= fields["HGT_sfc"][None]  # height above ground, in place (no second cube)
    snd, n_short = columns.build_soundings(
        z[:n_wind], fields["TMP"][:n_wind], fields["SPFH"][:n_wind], fields["PRES"][:n_wind], fields.pop("UGRD"), fields.pop("VGRD"),
        fields["TMP_2m"], fields["DPT_2m"], fields["UGRD_10m"], fields["VGRD_10m"], fields["PRES_sfc"],
        surface_wind=cfg.surface_wind, **_surface_kwargs(cfg),
    )  # fmt: skip
    x = columns.network_inputs(snd)
    timings["columns"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs, u = engine.predict(x, batch_size=cfg.batch_size)
    del x
    timings["inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    b_probs, b_cls, n_short_b = run_bourgouin(fields, z, snd, cfg)
    del z
    for k in ("TMP", "SPFH", "PRES"):
        fields.pop(k, None)
    timings["bourgouin"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rotate_sounding_to_earth(snd, rotation)
    timings["rotate"] = time.perf_counter() - t0

    cat = {k: (fields[k] == 1).astype(np.uint8) for k in CATEGORICAL}
    mask = np.zeros((ny, nx), np.uint8)
    for v in cat.values():
        mask |= v
    gridded: dict[str, np.ndarray] = {f"ML_{c}": probs[:, i].reshape(ny, nx) for i, c in enumerate(CLASS_NAMES)}
    gridded["ML_u"] = u.reshape(ny, nx)
    gridded |= {f"BOURG_{c}": b_probs[:, i].reshape(ny, nx) for i, c in enumerate(CLASS_NAMES)}
    gridded["BOURG_class"] = b_cls.reshape(ny, nx)
    gridded["precip_mask"] = mask
    gridded |= cat
    gridded["prate"] = fields["PRATE"].astype(np.float32, copy=False)
    m = mask.reshape(-1).astype(bool)
    info = {
        "n_short_network": int(n_short),
        "n_short_bourgouin": int(n_short_b),
        "precip_fraction": float(m.mean()),
        # Cells the HRRR itself calls snow, ice pellets or freezing rain: the retention rule keeps realtime runs
        # with enough of them (docs/realtime.md).
        "frozen_cells": int(np.count_nonzero(cat["CSNOW"] | cat["CICEP"] | cat["CFRZR"])),
        "ml_prob_sum_max_dev": float(np.max(np.abs(probs.sum(axis=1) - 1.0))) if probs.size else 0.0,
        "ml_class_freq_in_mask": {c: float(np.mean(probs[m].argmax(axis=1) == i)) if m.any() else 0.0 for i, c in enumerate(CLASS_NAMES)},
        "bourg_class_freq_in_mask": {c: float(np.mean(b_cls[m] == i)) if m.any() else 0.0 for i, c in enumerate(CLASS_NAMES)},
        "nan_ml": int(np.isnan(probs).sum()),
    }
    return gridded, snd.reshape(snd.shape[0], -1), info


# =============================================================================================
# Worker
# =============================================================================================
@dataclass(frozen=True)
class HourTask:
    """Everything a worker needs to process one forecast hour (picklable)."""

    run_id: str
    init: datetime
    fxx: int
    fh_index: int
    ptype_path: str
    sounding_path: str
    marker_path: str
    grid: dict[str, Any]


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via a temporary file + rename, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


class WorkerContext:
    """Per-process state: config, source, the engine (loaded ONCE), level guesses and grid caches."""

    def __init__(self, cfg: PipelineConfig, worker_index: int = 0, source=None, engine=None):
        self.cfg = cfg
        self.worker_index = worker_index
        self.threads = cfg.threads_per_process
        self.source = source if source is not None else make_source(cfg)
        self._apply_threads()
        t0 = time.perf_counter()
        self.engine = engine if engine is not None else self._make_engine()
        self.engine_load_s = time.perf_counter() - t0
        self.levels = LevelState(cfg.n_levels, max(cfg.n_levels, levels_for_height(cfg.bourgouin_top_m)))
        self._rotation_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.hours_done = 0

    def _apply_threads(self) -> None:
        """Set the numba/torch thread counts in this process (the env vars cover BLAS/OpenMP at import)."""
        import numba

        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))

    def _make_engine(self):
        from ptype.hrrr_fast.engine import make_engine

        device = self.cfg.device
        if device.startswith("cuda"):
            device = "cuda:0"  # CUDA_VISIBLE_DEVICES already restricts this process to its one GPU
        opts = dict(self.cfg.engine_options)
        opts.setdefault("threads", self.threads)
        engine = make_engine(self.cfg.engine, self.cfg.weights, device=device, **opts)
        if self.cfg.engine == "torch":
            import torch

            with contextlib.suppress(RuntimeError):  # can only be set before torch starts inter-op work
                torch.set_num_interop_threads(1)
        return engine

    def rotation(self, grid) -> tuple[np.ndarray, np.ndarray]:
        """cos/sin of the grid-to-earth angle, computed once per grid per process (pyproj inverse ~0.15 s)."""
        key = grid.proj4 + f"{grid.x_first},{grid.y_first},{grid.nx},{grid.ny}"
        if key not in self._rotation_cache:
            lon, _ = grid_lonlat(grid)
            self._rotation_cache[key] = columns.grid_rotation_cos_sin(lon, lon_0=grid.lon_0, lat_std=grid.lat_1)
        return self._rotation_cache[key]


def process_hour(ctx: WorkerContext, task: HourTask) -> dict[str, Any]:
    """Process one forecast hour end to end; never raises (errors are returned as ``status: failed``)."""
    cfg = ctx.cfg
    extra = {"run": task.run_id, "fxx": task.fxx, "worker": ctx.worker_index}
    res: dict[str, Any] = {"run": task.run_id, "fxx": task.fxx, "fh_index": task.fh_index, "worker": ctx.worker_index, "worker_hour": ctx.hours_done, "pid": os.getpid()}
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    try:
        import psutil

        res["rss_start_bytes"] = psutil.Process().memory_info().rss
        t0 = time.perf_counter()
        try:
            hour = ctx.source.open_hour(task.init, task.fxx, extra)
            idx_s = time.perf_counter() - t0
            fields, grid, stats, linfo = read_hour_fields(hour, cfg, ctx.levels, cfg.effective_decode_threads, extra)
        except MissingInputError as err:
            # idx or GRIB object absent (not yet published / archive gap): recorded, not retried, not an error.
            res |= {"status": "missing", "error": str(err), "timings": {"fetch": time.perf_counter() - t0}}
            logger.warning("input file missing", extra=extra | {"error": str(err)})
            ctx.hours_done += 1
            return res
        timings["fetch"] = idx_s + stats.fetch_s
        timings["decode"] = stats.decode_s
        expected = reader.GridMetadata(**task.grid)
        if (grid.nx, grid.ny, grid.proj4, round(grid.x_first, 3), round(grid.y_first, 3)) != (
            expected.nx,
            expected.ny,
            expected.proj4,
            round(expected.x_first, 3),
            round(expected.y_first, 3),
        ):
            raise ValueError(f"grid of f{task.fxx:02d} differs from the run store's grid")
        gridded, sounding, info = compute_hour(fields, grid, ctx.engine, cfg, linfo["levels"][0], ctx.rotation(grid), timings)
        del fields
        t0 = time.perf_counter()
        writers.write_ptype_hour(task.ptype_path, task.fh_index, gridded)
        writers.write_sounding_hour(task.sounding_path, task.fh_index, sounding)
        timings["write"] = time.perf_counter() - t0
        del gridded, sounding
        res |= {
            "status": "done",
            "levels": linfo,
            "bytes": dataclasses.asdict(stats),
            "info": info,
        }
        if info["n_short_network"] or info["n_short_bourgouin"]:
            logger.warning("columns whose top level is below the grid top (values clamped)", extra=extra | {"n_short": [info["n_short_network"], info["n_short_bourgouin"]]})
    except Exception as err:  # one bad hour must not take the worker (and the other hours) down
        logger.exception("hour failed", extra=extra)
        res |= {"status": "failed", "error": f"{type(err).__name__}: {err}"}
    timings["total"] = time.perf_counter() - t_start
    res["timings"] = timings
    res["peak_rss_bytes"] = peak_rss_bytes()
    res["finished"] = datetime.now(UTC).isoformat()
    ctx.hours_done += 1
    if res["status"] == "done":
        _atomic_write_json(Path(task.marker_path), res)  # the marker is the commit point of the hour
        logger.info("hour done", extra=extra | {"seconds": {k: round(v, 3) for k, v in timings.items()}, "peak_rss_gb": round(res["peak_rss_bytes"] / 2**30, 2)})
    return res


_WORKER: WorkerContext | None = None


def _init_worker(cfg_dict: dict[str, Any], counter, n_gpus: int, fetch_slots=None) -> None:
    """Pool initializer: pick this worker's GPU, install the node-wide download semaphore, load the model once."""
    global _WORKER
    set_fetch_slots(fetch_slots)
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    cfg = PipelineConfig(**cfg_dict)
    if cfg.device.startswith("cuda") and n_gpus > 0:
        # Must happen before torch initialises CUDA in this process; the parent never touches CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % n_gpus)
    configure_logging(cfg.log_level, cfg.log_json)
    _WORKER = WorkerContext(cfg, worker_index=idx)
    logger.info(
        "worker ready", extra={"worker": idx, "engine": _WORKER.engine.describe(), "engine_load_s": round(_WORKER.engine_load_s, 3), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    )


def _run_task(task: HourTask) -> dict[str, Any]:
    assert _WORKER is not None, "worker not initialised"
    return process_hour(_WORKER, task)


class InlineExecutor:
    """Executor that runs tasks in the calling process (``processes: 0``): same loop, no pickling."""

    def __init__(self, ctx: WorkerContext):
        self.ctx = ctx

    def submit(self, fn, task):
        f: Future = Future()
        try:
            f.set_result(process_hour(self.ctx, task))
        except Exception as err:  # noqa: BLE001  # pragma: no cover - process_hour does not raise
            f.set_exception(err)
        return f

    def shutdown(self, wait: bool = True) -> None:
        pass


# =============================================================================================
# Upload and index backends
# =============================================================================================
class LocalUploader:
    """Copy stores into a local directory laid out like the bucket (tests and air-gapped runs)."""

    name = "local"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def preflight(self) -> None:
        """Make sure the destination exists (fail before computing anything)."""
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_store(self, local: Path, key: str) -> list[str]:
        """Copy ``local`` to ``<root>/<key>`` (merging into an existing copy, never deleting)."""
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, dest, dirs_exist_ok=True)
        return [f"copytree {local} {dest}"]

    def upload_files(self, local: Path, key: str, rel_paths: Sequence[str], storage_class: str | None = None) -> list[str]:
        """Copy only the named files of ``local`` (paths relative to it) to ``<root>/<key>``."""
        for rel in rel_paths:
            dest = self.root / key / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(local) / rel, dest)
        return [f"copy {len(rel_paths)} files {local} {self.root / key}"]

    def list_stores(self, prefix: str) -> list[str]:
        """Keys of the ``.zarr`` stores under ``prefix``."""
        base = self.root / prefix
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*.zarr") if p.is_dir()) if base.exists() else []

    def delete_store(self, key: str, required_prefix: str) -> int:
        """Remove the store at ``key``; refuses anything outside ``required_prefix``. Returns the number of files removed."""
        _check_delete(key, required_prefix)
        dest = self.root / key
        n = sum(1 for p in dest.rglob("*") if p.is_file()) if dest.exists() else 0
        shutil.rmtree(dest, ignore_errors=True)
        return n


def _check_delete(key: str, required_prefix: str) -> None:
    """Deletion is only ever asked for one store under one named prefix; refuse anything else.

    The archive and the index live in the same bucket as the realtime staging stores, so a wrong key here
    would be unrecoverable. The caller states the prefix it means to clean and the key must be a ``.zarr``
    store strictly inside it.
    """
    p = required_prefix.strip("/") + "/"
    k = key.strip("/")
    if not p.strip("/") or not k.startswith(p) or not k.endswith(".zarr") or ".." in k.split("/") or k.endswith("inits.json"):
        raise ValueError(f"refusing to delete {key!r}: not a .zarr store under {required_prefix!r}")


class AwsCliUploader:
    """Upload with ``aws s3 sync`` (credentials from ``$AWS_PROFILE`` / ``~/.aws/credentials`` on Derecho).

    Why the CLI and not obstore/boto3 in-process: the AWS CLI is the documented, already-working path on
    Derecho (``docs/aws_upload.md``), handles multipart, parallelism and retries for thousands of chunk files,
    and ``sync`` makes a re-upload after a partial failure cheap (unchanged files are skipped).

    Two passes per store set explicit content types: ``zarr.json`` documents as ``application/json`` and every
    chunk/shard object as ``application/octet-stream`` (chunk keys have no extension, so the CLI's guess would
    be arbitrary). No ``Content-Encoding`` is set: chunks are already compressed by Zarr. ``--delete`` is never
    used (a wrong destination prefix would otherwise silently delete good data). Objects get the configured
    storage class (Standard-IA by default); the index is NOT uploaded here, see :meth:`sync_commands`.
    """

    name = "s3"

    def __init__(self, bucket: str, region: str, aws_cli: str = "aws", storage_class: str | None = "STANDARD_IA", runner: Callable[..., Any] = subprocess.run):
        self.bucket = bucket
        self.region = region
        self.aws_cli = aws_cli
        self.storage_class = storage_class
        self.runner = runner

    def preflight(self) -> None:
        """``aws sts get-caller-identity``: expired temporary keys fail here, not after hours of compute."""
        cmd = [self.aws_cli, "sts", "get-caller-identity", "--region", self.region]
        r = self.runner(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"AWS credentials check failed (AWS_PROFILE={os.environ.get('AWS_PROFILE')!r}): {r.stderr.strip()[:500]}")

    def sync_commands(self, local: Path, key: str) -> list[list[str]]:
        """The ``aws s3 sync`` commands for one store.

        ``--exclude "*inits.json"`` is a belt-and-braces guard: the index must never be written by the sync
        (it would get the IA storage class and could race with ``update_index``); it is only ever written by
        ``index.update_index`` (STANDARD class, conditional put).
        """
        if key.rstrip("/").endswith("inits.json") or not key.endswith(".zarr"):
            raise ValueError(f"refusing to sync to {key!r}: only .zarr store prefixes are uploaded with sync")
        base = [self.aws_cli, "s3", "sync", str(local), f"s3://{self.bucket}/{key}", "--region", self.region, "--only-show-errors", "--no-progress"]
        if self.storage_class:
            base += ["--storage-class", self.storage_class]
        guard = ["--exclude", "*inits.json"]
        chunks = [*base, "--exclude", "*zarr.json", *guard, "--content-type", "application/octet-stream"]
        meta = [*base, "--exclude", "*", "--include", "*zarr.json", *guard, "--content-type", "application/json"]
        return [chunks, meta]  # chunks first, metadata last: a store is only discoverable once its data is there

    def upload_store(self, local: Path, key: str) -> list[str]:
        """Run the sync commands; raise on the first failure (the index must then not be updated)."""
        done = []
        for cmd in self.sync_commands(local, key):
            r = self.runner(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()[:1000]}")
            done.append(" ".join(cmd))
        return done


class Boto3Uploader:
    """Upload a store with boto3, many files and many parts at a time.

    Why this replaced ``aws s3 sync`` as the default: in the first pilot job on Derecho a node computed 231
    forecast hours in 7 minutes and then spent 50 minutes uploading, one run at a time at 5-15 MB/s (a 6.8 GB
    run took 1187 s). ``aws s3 sync`` holds 10 connections and that cannot be raised without editing the
    user's AWS config. The network study (``benchmarks/results/derecho/network.md``) found throughput to AWS
    limited per connection, so the remedy is the same as for downloads: more connections. Here
    ``file_workers`` files are sent at once and each multipart upload uses ``part_concurrency`` connections.

    Behaviour kept from the CLI path: data objects first and ``zarr.json`` documents last (a store only becomes
    discoverable once its data is there), explicit content types, the configured storage class, nothing is ever
    deleted, and the index is never written here. A file is skipped when the bucket already holds an object of
    the same size that is not older than the local file (the rule ``aws s3 sync`` uses), which makes a re-run
    after a partial upload cheap.
    """

    name = "s3"
    PART_BYTES = 16 * 1024 * 1024

    def __init__(self, bucket: str, region: str, session: Any, storage_class: str | None = "STANDARD_IA", file_workers: int = 8, part_concurrency: int = 8, client: Any = None):
        self.bucket = bucket
        self.region = region
        self.storage_class = storage_class
        self.file_workers = max(1, file_workers)
        self.part_concurrency = max(1, part_concurrency)
        if client is None:
            from botocore.config import Config

            # The pool must hold every connection this uploader can open, for several runs uploading at once,
            # or boto3 silently queues requests behind a pool of 10.
            client = session.client(
                "s3", region_name=region, config=Config(max_pool_connections=4 * self.file_workers * self.part_concurrency, retries={"max_attempts": 8, "mode": "adaptive"})
            )
        self.client = client

    def preflight(self) -> None:
        """Nothing to do: ``resolve_aws_session(verify=True)`` already checked the identity of this session."""

    def _remote(self, key: str) -> dict[str, tuple[int, float]]:
        """``{object key: (size, last-modified epoch)}`` of what is already under the store prefix."""
        out: dict[str, tuple[int, float]] = {}
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=key.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                out[obj["Key"]] = (int(obj["Size"]), obj["LastModified"].timestamp())
        return out

    def _put(self, path: Path, key: str, storage_class: str | None = "default") -> None:
        from boto3.s3.transfer import TransferConfig

        extra = {"ContentType": "application/json" if path.name == "zarr.json" else "application/octet-stream"}
        storage_class = self.storage_class if storage_class == "default" else storage_class
        if storage_class:
            extra["StorageClass"] = storage_class
        cfg = TransferConfig(multipart_threshold=self.PART_BYTES, multipart_chunksize=self.PART_BYTES, max_concurrency=self.part_concurrency, use_threads=True)
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra, Config=cfg)

    def upload_store(self, local: Path, key: str) -> list[str]:
        """Upload every file of ``local`` under ``key``; raise on the first failure (the index must then not be updated)."""
        if key.rstrip("/").endswith("inits.json") or not key.endswith(".zarr"):
            raise ValueError(f"refusing to upload to {key!r}: only .zarr store prefixes are uploaded")
        local = Path(local)
        # The source must be a real store whose name matches the destination. An empty path means "the current
        # directory" to pathlib; without this check a caller with an unset variable uploaded a source checkout
        # under a store prefix (it happened once, by hand, on Derecho).
        if not str(local) or local.name != key.rstrip("/").rsplit("/", 1)[-1] or not (local / "zarr.json").is_file():
            raise ValueError(f"refusing to upload {str(local)!r} to {key!r}: the source must be the Zarr store directory of that name")
        remote = self._remote(key)
        data, meta, skipped, sent_bytes = [], [], 0, 0
        for path in sorted(p for p in local.rglob("*") if p.is_file()):
            if path.name == "inits.json":
                continue  # the index is only ever written by index.update_index
            okey = f"{key.rstrip('/')}/{path.relative_to(local).as_posix()}"
            stat = path.stat()
            have = remote.get(okey)
            if have is not None and have[0] == stat.st_size and have[1] >= stat.st_mtime:
                skipped += 1
                continue
            sent_bytes += stat.st_size
            (meta if path.name == "zarr.json" else data).append((path, okey))
        for batch in (data, meta):  # data first, metadata last
            with ThreadPoolExecutor(max_workers=self.file_workers, thread_name_prefix="upload") as pool:
                for fut in [pool.submit(self._put, p, k) for p, k in batch]:
                    fut.result()
        return [f"boto3 upload s3://{self.bucket}/{key}: {len(data)} data + {len(meta)} metadata objects, {sent_bytes} bytes, {skipped} unchanged skipped"]

    def upload_files(self, local: Path, key: str, rel_paths: Sequence[str], storage_class: str | None = "default") -> list[str]:
        """Upload only the named files of ``local`` (paths relative to it) under ``key``; data before ``zarr.json``.

        The realtime driver publishes one forecast hour at a time while other hours are still being written
        into the same local store, so it names the finished files instead of walking the directory (a walk
        would pick up another hour's half-written temporary file). No listing is made: a re-sent hour simply
        overwrites its own objects.
        """
        if key.rstrip("/").endswith("inits.json") or not key.endswith(".zarr"):
            raise ValueError(f"refusing to upload to {key!r}: only .zarr store prefixes are uploaded")
        local = Path(local)
        if not str(local) or local.name != key.rstrip("/").rsplit("/", 1)[-1] or not (local / "zarr.json").is_file():
            raise ValueError(f"refusing to upload {str(local)!r} to {key!r}: the source must be the Zarr store directory of that name")
        items = [(local / rel, f"{key.rstrip('/')}/{rel}") for rel in rel_paths if Path(rel).name != "inits.json"]
        data = [it for it in items if it[0].name != "zarr.json"]
        meta = [it for it in items if it[0].name == "zarr.json"]
        for batch in (data, meta):
            with ThreadPoolExecutor(max_workers=self.file_workers, thread_name_prefix="upload") as pool:
                for fut in [pool.submit(self._put, p, k, storage_class) for p, k in batch]:
                    fut.result()
        return [f"boto3 upload s3://{self.bucket}/{key}: {len(data)} data + {len(meta)} metadata objects (named files)"]

    def list_stores(self, prefix: str) -> list[str]:
        """Keys of the ``.zarr`` stores under ``prefix`` (from the root ``zarr.json`` of each)."""
        out = set()
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if ".zarr/" in k:
                    out.add(k[: k.index(".zarr/") + 5])
        return sorted(out)

    def delete_store(self, key: str, required_prefix: str) -> int:
        """Delete every object of the store at ``key``; refuses anything outside ``required_prefix``. Returns the count."""
        _check_delete(key, required_prefix)
        keys = list(self._remote(key))
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True})
        return len(keys)


class LocalIndex:
    """``inits.json`` in a local directory, updated under an exclusive ``fcntl`` lock.

    ``index.update_index`` relies on conditional puts, which obstore's ``LocalStore`` does not implement
    (``NotImplementedError``, see ``docs/aws_upload.md``), so the local backend merges with the same
    ``index.merge_entry`` under a file lock and replaces the file atomically.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text()) if self.path.exists() else index.empty_index()

    def update(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = index.merge_entry(self.read(), entry["init"], entry["forecast_hours"], entry["store"], entry.get("soundings"), extra=entry.get("extra"))
            _atomic_write_json(self.path, merged)
        return merged

    def write_json(self, name: str, obj: Any) -> None:
        """Write a small JSON document next to the index (the realtime status file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.path.with_name(name), obj)

    def remove(self, inits: set[str]) -> dict[str, Any]:
        """Drop runs from the index (retention), under the same lock as :meth:`update`."""
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            merged = index.remove_entries(self.read(), inits)
            _atomic_write_json(self.path, merged)
        return merged


class ObstoreIndex:
    """``inits.json`` in an obstore store (S3, or ``MemoryStore`` in tests) via ``index.update_index``."""

    def __init__(self, store, key: str = index.INDEX_KEY):
        self.store = store
        self.key = key

    def read(self) -> dict[str, Any]:
        import obstore

        try:
            return json.loads(bytes(obstore.get(self.store, self.key).bytes()))
        except FileNotFoundError:
            return index.empty_index()

    def update(self, entry: dict[str, Any]) -> dict[str, Any]:
        return index.update_index(self.store, self.key, entry)

    def write_json(self, name: str, obj: Any) -> None:
        """Write a small JSON document next to the index (the realtime status file); last writer wins."""
        import obstore

        key = f"{self.key.rsplit('/', 1)[0]}/{name}"
        obstore.put(self.store, key, json.dumps(obj, indent=2).encode(), attributes={"Content-Type": "application/json", "Cache-Control": "no-cache"})

    def remove(self, inits: set[str]) -> dict[str, Any]:
        """Drop runs from the index (retention)."""
        return index.remove_from_index(self.store, self.key, inits)


class CredentialsError(RuntimeError):
    """AWS credentials for the output bucket are missing, unreadable or rejected (message never contains secrets)."""


#: Timeouts/retries for the quick index probe (``--no-upload``, launcher): a failure must cost seconds, not the
#: ~15 s of obstore's default 10 retries. The upload path keeps obstore's default retries for the real writes.
QUICK_CLIENT_OPTIONS = {"connect_timeout": "3s", "timeout": "10s"}
QUICK_RETRY_CONFIG = {
    "max_retries": 1,
    "retry_timeout": timedelta(seconds=10),
    "backoff": {"init_backoff": timedelta(milliseconds=200), "max_backoff": timedelta(seconds=1), "base": 2},
}


def resolve_aws_session(profile: str | None, region: str, *, verify: bool, timeout_s: float = 10.0, session_factory: Callable[..., Any] | None = None):
    """Resolve credentials for the OUTPUT bucket once, in the parent, from an AWS profile; optionally verify them.

    Why: obstore's ``S3Store`` does not read ``~/.aws/credentials`` or profiles. Without an explicit credential
    provider it falls back to the EC2 instance-metadata service (``169.254.169.254``), which does not exist on
    Derecho and costs ~15 s of retries before failing. On Derecho the keys are temporary (with a session token)
    and live in the shared credentials file under ``$AWS_PROFILE``; boto3 reads that file (and SSO caches,
    credential processes ...), so a ``boto3.Session`` is the single source of credentials.

    With ``verify`` (upload enabled) one STS ``GetCallerIdentity`` call with short timeouts and no retries checks
    that the keys are accepted, so a job with expired keys fails at start-up instead of after hours of compute.
    Only the caller ARN is logged; access keys, secrets and tokens are never put in messages or log records.

    Args:
        profile: Profile name (``$AWS_PROFILE``); ``None`` uses boto3's default chain.
        region: Bucket region.
        verify: Make the STS call.
        timeout_s: Connect/read timeout of that call.
        session_factory: For tests; defaults to ``boto3.Session``.

    Returns:
        The ``boto3.Session``.

    Raises:
        CredentialsError: Profile missing, no credentials, or credentials rejected/unverifiable.
    """
    import boto3
    import botocore.exceptions as bexc
    from botocore.config import Config

    factory = session_factory or boto3.Session
    where = f"profile {profile!r}" if profile else "the default AWS credential chain (AWS_PROFILE not set)"
    try:
        session = factory(profile_name=profile, region_name=region) if profile else factory(region_name=region)
        creds = session.get_credentials()
    except bexc.ProfileNotFound:
        raise CredentialsError(f"AWS {where} not found in ~/.aws/config or ~/.aws/credentials") from None
    except bexc.BotoCoreError as err:
        raise CredentialsError(f"could not load AWS credentials from {where}: {type(err).__name__}") from None
    if creds is None:
        raise CredentialsError(f"no AWS credentials in {where}")
    if verify:
        cfg = Config(connect_timeout=timeout_s, read_timeout=timeout_s, retries={"max_attempts": 1, "mode": "standard"})
        try:
            ident = session.client("sts", region_name=region, config=cfg).get_caller_identity()
        except bexc.ClientError as err:
            code = err.response.get("Error", {}).get("Code", "unknown")
            raise CredentialsError(f"AWS credentials from {where} were rejected ({code}); refresh them (docs/aws_upload.md)") from None
        except bexc.BotoCoreError as err:
            raise CredentialsError(f"could not verify AWS credentials from {where}: {type(err).__name__}") from None
        logger.info("AWS credentials verified", extra={"aws_profile": profile, "aws_arn": ident.get("Arn")})
    return session


def output_s3_store(bucket: str, region: str, session, *, quick: bool = False):
    """obstore ``S3Store`` for the output bucket with credentials from ``session`` (see :func:`resolve_aws_session`).

    The credential provider is ``obstore.auth.boto3.Boto3CredentialProvider`` over the already-resolved
    session, exactly what ``index.main`` builds from ``--profile``; ``index.s3_store`` is used if that module
    provides it (proposed refactor) so there is one construction for both. ``quick`` sets short timeouts and a
    single retry for probes. The HRRR input store is a different, anonymous store (``reader.open_store``,
    ``skip_signature=True``) and never gets these credentials.
    """
    from obstore.auth.boto3 import Boto3CredentialProvider
    from obstore.store import S3Store

    opts: dict[str, Any] = {"credential_provider": Boto3CredentialProvider(session)}
    if quick:
        opts |= {"client_options": QUICK_CLIENT_OPTIONS, "retry_config": QUICK_RETRY_CONFIG}
    make = getattr(index, "s3_store", None)
    if make is not None:
        return make(bucket, region, session=session, **{k: v for k, v in opts.items() if k != "credential_provider"})
    return S3Store(bucket, region=region, **opts)


def make_backends(cfg: PipelineConfig, no_upload: bool = False):
    """``(uploader, index)`` for the configured upload mode; both ``None`` for ``upload: none``.

    For ``upload: s3`` the credentials are resolved here, once. With uploads enabled a missing profile or
    rejected keys raise :class:`CredentialsError` immediately. With ``no_upload`` the index is only read to skip
    finished runs, so a credentials problem is a quick warning and no index is used.
    """
    if cfg.upload == "none":
        return None, None
    if cfg.upload == "local":
        return (None if no_upload else LocalUploader(cfg.upload_root)), LocalIndex(Path(cfg.upload_root) / index.INDEX_KEY)
    profile = os.environ.get("AWS_PROFILE")
    if no_upload:
        try:
            session = resolve_aws_session(profile, cfg.region, verify=False)
        except CredentialsError as err:
            logger.warning("index not readable without AWS credentials; not skipping indexed runs", extra={"error": str(err)})
            return None, None
        return None, ObstoreIndex(output_s3_store(cfg.bucket, cfg.region, session, quick=True))
    session = resolve_aws_session(profile, cfg.region, verify=True)
    if cfg.upload_backend == "awscli":
        uploader: Any = AwsCliUploader(cfg.bucket, cfg.region, cfg.aws_cli, cfg.storage_class)
    else:
        uploader = Boto3Uploader(cfg.bucket, cfg.region, session, cfg.storage_class, cfg.upload_file_workers, cfg.upload_part_concurrency)
    return uploader, ObstoreIndex(output_s3_store(cfg.bucket, cfg.region, session))


# =============================================================================================
# Run state, manifest, provenance
# =============================================================================================
_PACKAGES = ("numpy", "numba", "zarr", "cast-value-rs", "eccodes", "eccodeslib", "obstore", "pyproj", "torch", "onnxruntime", "pyyaml", "psutil", "boto3")


def provenance(cfg: PipelineConfig) -> dict[str, Any]:
    """Code commit, package versions, host and config: enough to reproduce or explain a run later."""
    repo = Path(__file__).resolve().parents[2]
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=20).stdout.strip() or None
        dirty = bool(
            subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], check=False, capture_output=True, text=True, timeout=20).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    versions = {}
    for p in _PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "code_commit": commit,
        "code_dirty": dirty,
        "host": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "config": cfg.to_dict(),
    }


def store_names(init: datetime) -> tuple[str, str, str, str]:
    """``(ptype_key, sounding_key, ptype_dirname, sounding_dirname)`` from ``index.store_paths``."""
    pk, sk = index.store_paths(init)
    return pk, sk, Path(pk).name, Path(sk).name


@dataclass
class RunState:
    """Book-keeping of one run inside an invocation (parent process only)."""

    spec: RunSpec
    run_dir: Path
    ptype_path: Path
    sounding_path: Path
    manifest: dict[str, Any]
    pending: set[int] = field(default_factory=set)
    grid: reader.GridMetadata | None = None

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def marker(self, fxx: int) -> Path:
        return self.run_dir / "status" / f"f{fxx:02d}.json"

    @property
    def finalized_marker(self) -> Path:
        """Written after ``writers.finalize_ptype_store`` succeeded; makes finalize happen once per run."""
        return self.run_dir / "status" / "finalized.json"

    def done_hours(self) -> list[int]:
        """Hours whose marker exists (the ground truth of what is written)."""
        d = self.run_dir / "status"
        return sorted(int(p.stem[1:]) for p in d.glob("f[0-9]*.json")) if d.exists() else []

    def save(self) -> None:
        self.manifest["updated"] = datetime.now(UTC).isoformat()
        _atomic_write_json(self.manifest_path, self.manifest)


def _stores_match(st: RunState) -> bool:
    """Whether both stores exist, finished creation, and have this run's forecast-hour axis."""
    if not (st.run_dir / "status" / "stores_created").exists():
        return False
    try:
        import zarr

        g = zarr.open_group(st.ptype_path, mode="r")
        s = zarr.open_array(st.sounding_path, path="forecast_hour", mode="r")
        return list(g.attrs["hrrr_ptype"]["forecast_hours"]) == st.spec.store_hours and [int(v) for v in s[:]] == st.spec.store_hours
    except Exception:  # noqa: BLE001 - any read problem means "recreate"
        return False


def prepare_run(spec: RunSpec, cfg: PipelineConfig, source, prov: dict[str, Any]) -> RunState:
    """Create (or reopen) the run directory and stores, and work out which hours still need processing.

    Store creation writes only metadata and coordinates and is guarded by a ``status/stores_created`` flag
    written after both stores exist, so an interrupted creation is redone (and its stale markers removed)
    rather than leaving a half-initialised store that a worker would write into.
    """
    _, _, pname, sname = store_names(spec.init)
    run_dir = Path(cfg.work_dir) / spec.run_id
    status = run_dir / "status"
    st = RunState(spec, run_dir, run_dir / pname, run_dir / sname, {})
    if st.manifest_path.exists():
        st.manifest = json.loads(st.manifest_path.read_text())
    st.manifest.setdefault("init", spec.init.strftime("%Y-%m-%dT%H:%MZ"))
    st.manifest.setdefault("run_id", spec.run_id)
    st.manifest.setdefault("hours", {})
    st.manifest["store_hours"] = spec.store_hours
    st.manifest["requested_hours"] = sorted(set(st.manifest.get("requested_hours", [])) | set(spec.hours))
    st.manifest.setdefault("invocations", []).append(
        {"started": datetime.now(UTC).isoformat(), "hours": list(spec.hours), **{k: prov[k] for k in ("code_commit", "code_dirty", "host", "pbs_jobid")}}
    )
    st.manifest["provenance"] = prov
    grid_file = run_dir / "grid.json"
    if grid_file.exists():
        st.grid = reader.GridMetadata(**json.loads(grid_file.read_text()))
    else:
        st.grid = source.grid(spec.init, spec.hours)
        if st.grid is not None:
            _atomic_write_json(grid_file, dataclasses.asdict(st.grid))
    if st.grid is None:
        for h in spec.hours:
            st.manifest["hours"][str(h)] = {"status": "missing", "error": "no file of this run was found"}
        st.manifest["status"] = "missing"
        st.save()
        return st
    if not _stores_match(st):
        if status.exists():
            shutil.rmtree(status)
        status.mkdir(parents=True)
        attrs = {
            "network_heights_m": [float(h) for h in columns.NETWORK_HEIGHTS],
            "bourgouin_top_m": cfg.bourgouin_top_m,
            "bourgouin_dz_m": cfg.bourgouin_dz_m,
            "ml_wind_frame": "grid_relative (network input)",
            "engine": cfg.engine,
            "code_commit": prov.get("code_commit"),
            "surface_policy": _surface_kwargs(cfg) | {"surface_wind": cfg.surface_wind},
        }
        writers.create_ptype_store(
            st.ptype_path, st.grid, spec.store_hours, variables=ptype_variables(), n_levels=cfg.pyramid_levels,
            init_time=spec.init.strftime("%Y-%m-%dT%H:%MZ"), attrs=attrs,
        )  # fmt: skip
        writers.create_sounding_store(
            st.sounding_path, st.grid, spec.store_hours, heights_m=columns.NETWORK_HEIGHTS, variable_groups=columns.SOUNDING_VARIABLES,
            init_time=spec.init.strftime("%Y-%m-%dT%H:%M:%SZ"), location_chunk=cfg.location_chunk,
            filters=sounding_codecs(cfg)[0], compressors=sounding_codecs(cfg)[1],
        )  # fmt: skip
        (status / "stores_created").write_text(datetime.now(UTC).isoformat())
        logger.info("stores created", extra={"run": spec.run_id, "ptype": str(st.ptype_path), "soundings": str(st.sounding_path)})
    done = set(st.done_hours())
    for h in done:
        st.manifest["hours"][str(h)] = json.loads(st.marker(h).read_text())  # the marker wins over older records
    st.pending = {h for h in spec.hours if h not in done}
    if st.pending and st.finalized_marker.exists():
        # After finalize the per-hour gridded objects have been consolidated (and the staging copy removed), so a
        # worker can no longer add an hour to this store safely. Such hours (typically archive gaps that were
        # missing when the run was finalized) are left out, loudly; delete the run directory to rebuild the run.
        logger.warning(
            "run store already finalized; not adding hours (delete the run directory to reprocess the run)",
            extra={"run": spec.run_id, "hours": sorted(st.pending)},
        )
        for h in st.pending:
            st.manifest["hours"].setdefault(str(h), {"status": "missing", "error": "not written before the store was finalized"})
        st.pending = set()
    skipped = sorted(set(spec.hours) & done)
    if skipped:
        logger.info("hours already complete, skipping", extra={"run": spec.run_id, "hours": skipped})
    st.manifest["status"] = "running"
    st.save()
    return st


def finalize_ptype(st: RunState, cfg: PipelineConfig) -> bool:
    """Run ``writers.finalize_ptype_store`` ONCE for the run's gridded store; record result and time; True on success.

    Why here: workers write the gridded store hour by hour into small per-hour objects (the only layout that lets
    many processes write one store without locks). Once every hour is written, the parent consolidates them into
    shards so each uploaded object exceeds S3 Standard-IA's 128 KB minimum billable size. It must run after the
    last hour and before the upload, in one process, hence in the parent's finalisation step. A success marker
    (``status/finalized.json``) makes a re-run after an upload failure go straight to the upload. Any error is a
    failed run: nothing is uploaded and the index is not touched.
    """
    m = st.manifest
    if not cfg.finalize_ptype:
        m["finalize"] = {"skipped": "finalize_ptype is false"}
        return True
    if st.finalized_marker.exists():
        m.setdefault("finalize", json.loads(st.finalized_marker.read_text()))
        return True
    fn = getattr(writers, "finalize_ptype_store", None)
    t0 = time.perf_counter()
    try:
        if fn is None:
            raise RuntimeError("ptype.hrrr_fast.writers.finalize_ptype_store is not available in this checkout (update the writers module, or set finalize_ptype: false)")
        result = fn(st.ptype_path, remove_staging=True)
    except Exception as err:
        m["finalize"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "seconds": time.perf_counter() - t0}
        logger.exception("finalize_ptype_store failed; not uploading", extra={"run": st.spec.run_id})
        return False
    rec = {"ok": True, "seconds": time.perf_counter() - t0, "result": result, "time": datetime.now(UTC).isoformat()}
    _atomic_write_json(st.finalized_marker, rec)
    m["finalize"] = json.loads(json.dumps(rec, default=str))
    logger.info("gridded store finalized", extra={"run": st.spec.run_id, "seconds": round(rec["seconds"], 2)})
    return True


def finalize_run(st: RunState, cfg: PipelineConfig, uploader, index_backend, no_upload: bool, entry_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finalize the gridded store, upload both stores, THEN update the index, THEN optionally remove the local copy.

    The order is the safety property: the website only learns about a run from the index, so the index must
    never point at data that is not (completely) uploaded. Any upload error leaves the index untouched and
    the local data in place for the next attempt. A run with FAILED hours is not uploaded at all (a re-run
    retries them); MISSING hours are expected (archive gaps, not-yet-published files) and the index entry
    then lists only the hours actually written.
    """
    m = st.manifest
    written = st.done_hours()
    failed = sorted(int(h) for h, r in m["hours"].items() if r.get("status") == "failed" and int(h) not in written)
    missing = sorted(int(h) for h, r in m["hours"].items() if r.get("status") == "missing" and int(h) not in written)
    m["written_hours"], m["failed_hours"], m["missing_hours"] = written, failed, missing
    extra = {"run": st.spec.run_id, "written": len(written), "failed": failed, "missing": missing}
    if failed:
        m["status"] = "incomplete"
        logger.error("run has failed hours; not uploading (re-run to retry them)", extra=extra)
    elif not written:
        m["status"] = "empty"
        logger.warning("run has no written hours; nothing to upload", extra=extra)
    elif not finalize_ptype(st, cfg):
        m["status"] = "finalize_failed"
    elif no_upload or uploader is None:
        m["status"] = "complete"
        logger.info("run complete (upload disabled)", extra=extra)
    else:
        pk, sk, _, _ = store_names(st.spec.init)
        up: dict[str, Any] = {"backend": uploader.name, "started": datetime.now(UTC).isoformat()}
        try:
            t0 = time.perf_counter()
            up["commands"] = uploader.upload_store(st.ptype_path, pk) + uploader.upload_store(st.sounding_path, sk)
            up["seconds"] = time.perf_counter() - t0
            up["ok"] = True
        except Exception as err:
            up |= {"ok": False, "error": f"{type(err).__name__}: {err}"}
            m["upload"] = up
            m["status"] = "upload_failed"
            logger.exception("upload failed; index not updated", extra=extra)
            st.save()
            return m
        m["upload"] = up
        try:
            entry = index.make_entry(st.spec.init, written)
            if entry_extra:
                entry["extra"] = dict(entry_extra)
            index_backend.update(entry)
            m["index"] = {"ok": True, "entry": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in entry.items()}, "time": datetime.now(UTC).isoformat()}
            m["status"] = "indexed"
            logger.info("run uploaded and indexed", extra=extra | {"upload_s": round(up["seconds"], 2)})
        except Exception as err:
            m["index"] = {"ok": False, "error": f"{type(err).__name__}: {err}"}
            m["status"] = "index_failed"
            logger.exception("index update failed", extra=extra)
            st.save()
            return m
        if not cfg.keep_local:
            for p in (st.ptype_path, st.sounding_path, st.run_dir / "status"):
                shutil.rmtree(p, ignore_errors=True)
            m["local_removed"] = True
    m["finished"] = datetime.now(UTC).isoformat()
    st.save()
    return m


# =============================================================================================
# Orchestration
# =============================================================================================
def _indexed_inits(index_backend, strict: bool = False) -> set[str]:
    """Init strings (``2024-04-30T00:00Z``) already in the index.

    A missing index object is an empty index (handled by the backend). Any other read failure is fatal with
    ``strict`` (uploads enabled: the run would only fail later when writing the index) and a warning otherwise.
    Only the exception type and message are logged; obstore/botocore messages carry no credentials.
    """
    if index_backend is None:
        return set()
    try:
        # A realtime run that is still arriving (or was cut short) is listed with ``complete: false`` so the site
        # can show its hours; it is not finished, and a later batch pass must be able to complete it.
        return {e["init"] for e in index_backend.read().get("inits", []) if e.get("complete", True) is not False}
    except Exception as err:
        if strict:
            raise
        logger.warning("could not read the index; not skipping any run", extra={"error": f"{type(err).__name__}: {str(err)[:300]}"})
        return set()


def make_executor(cfg: PipelineConfig, processes: int, source, engine):
    """The pool that runs :func:`process_hour`: worker processes, or the calling process when ``processes <= 0``.

    Shared by the batch driver and the realtime driver (``ptype.hrrr_fast.realtime``) so both set up threads,
    GPUs and the node-wide download cap the same way. Call ``executor.shutdown()`` and
    ``set_fetch_slots(None)`` when done.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    # One semaphore for the whole node, shared by every worker (inherited at spawn through initargs).
    fetch_slots = ctx.Semaphore(cfg.max_concurrent_fetches) if cfg.max_concurrent_fetches else None
    if processes <= 0:
        set_fetch_slots(fetch_slots)
        return InlineExecutor(WorkerContext(cfg, 0, source=source, engine=engine))
    os.environ.update(thread_env(cfg.threads_per_process))
    n_gpus = (cfg.gpus if cfg.gpus is not None else detect_gpus()) if cfg.device.startswith("cuda") else 0
    if cfg.device.startswith("cuda") and n_gpus == 0:
        raise RuntimeError("device cuda requested but no GPU is visible")
    return ProcessPoolExecutor(max_workers=processes, mp_context=ctx, initializer=_init_worker, initargs=(cfg.to_dict(), ctx.Value("i", 0), n_gpus, fetch_slots))


def run_pipeline(
    cfg: PipelineConfig,
    runs: Sequence[RunSpec],
    *,
    no_upload: bool = False,
    ignore_index: bool = False,
    source=None,
    engine=None,
    uploader=None,
    index_backend=None,
) -> list[dict[str, Any]]:
    """Process runs: skip indexed runs, schedule pending hours on the worker pool, finalise each finished run.

    Hours of up to ``max_runs_in_flight`` runs are queued together, so workers never idle at the end of a run
    while it is uploaded: finalisation (upload + index) happens in a background thread of the parent.

    Args:
        cfg: Configuration.
        runs: Runs to process.
        no_upload: Leave stores on disk (no upload, no index update).
        ignore_index: Process runs even if the index already lists them (hours with markers are still skipped).
        source, engine: Injected source/engine for in-process tests (``processes: 0`` only).
        uploader, index_backend: Injected backends; default from the config.

    Returns:
        The final manifest of every run.
    """
    if uploader is None and index_backend is None:
        uploader, index_backend = make_backends(cfg, no_upload)
    if no_upload:
        uploader = None
    if uploader is not None:
        uploader.preflight()
    # With uploads the index must be readable (we will write it); without, an unreadable index is a warning.
    indexed = set() if ignore_index else _indexed_inits(index_backend, strict=uploader is not None)
    todo = []
    results = []
    for r in runs:
        if index.init_string(r.init) in indexed:
            logger.info("run already in the index; skipping", extra={"run": r.run_id})
            results.append({"run_id": r.run_id, "status": "skipped_indexed"})
        else:
            todo.append(r)
    if not todo:
        return results
    prov = provenance(cfg)
    parent_source = source if source is not None else make_source(cfg)
    n_tasks = sum(len(r.hours) for r in todo)
    processes = auto_processes(cfg, n_tasks) if cfg.processes == "auto" else int(cfg.processes)
    logger.info(
        "starting", extra={"runs": len(todo), "hours": n_tasks, "processes": processes, "threads_per_process": cfg.threads_per_process, "engine": cfg.engine, "device": cfg.device}
    )
    executor = make_executor(cfg, processes, parent_source, engine)
    # Several runs finalize and upload at once: workers finish a node's runs far faster than one upload stream
    # can ship them, so a single finalize thread left the node idle behind the uploads (see Boto3Uploader).
    finalizer = ThreadPoolExecutor(max_workers=max(1, cfg.upload_workers), thread_name_prefix="finalize")
    fin_futures: list[Future] = []
    queue = deque(todo)
    active: dict[str, RunState] = {}
    futures: dict[Future, tuple[str, int]] = {}
    lock = threading.Lock()

    def finish(st: RunState) -> None:
        with lock:
            active.pop(st.spec.run_id, None)
        fin_futures.append(finalizer.submit(finalize_run, st, cfg, uploader, index_backend, no_upload))

    try:
        while queue or futures:
            while queue and len(active) < max(1, cfg.max_runs_in_flight):
                spec = queue.popleft()
                try:
                    st = prepare_run(spec, cfg, parent_source, prov)
                except Exception as err:  # one unpreparable run must not stop the others
                    logger.exception("could not prepare run", extra={"run": spec.run_id})
                    results.append({"run_id": spec.run_id, "status": "prepare_failed", "error": f"{type(err).__name__}: {err}"})
                    continue
                if st.grid is None:
                    logger.warning("no input file of this run exists", extra={"run": spec.run_id})
                    results.append(st.manifest)
                    continue
                active[spec.run_id] = st
                if not st.pending:
                    finish(st)
                    continue
                g = dataclasses.asdict(st.grid)
                for fxx in sorted(st.pending):
                    task = HourTask(spec.run_id, spec.init, fxx, spec.store_hours.index(fxx), str(st.ptype_path), str(st.sounding_path), str(st.marker(fxx)), g)
                    futures[executor.submit(_run_task, task)] = (spec.run_id, fxx)
            if not futures:
                continue
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for f in done:
                run_id, fxx = futures.pop(f)
                try:
                    res = f.result()
                except BrokenProcessPool as err:
                    res = {"run": run_id, "fxx": fxx, "status": "failed", "error": f"worker pool broken: {err}"}
                except Exception as err:  # noqa: BLE001
                    res = {"run": run_id, "fxx": fxx, "status": "failed", "error": f"{type(err).__name__}: {err}"}
                st = active[run_id]
                st.manifest["hours"][str(fxx)] = res
                st.pending.discard(fxx)
                st.save()
                if not st.pending:
                    finish(st)
    finally:
        executor.shutdown(wait=True)
        set_fetch_slots(None)
        finalizer.shutdown(wait=True)
    for f in fin_futures:
        results.append(f.result())
    return results


def dry_run(cfg: PipelineConfig, runs: Sequence[RunSpec], source=None, out=None) -> dict[str, Any]:
    """Print the plan: per hour, whether the file exists and the bytes the first read would fetch (idx only).

    Uses the first-guess level counts (``n_levels`` for the network, :func:`levels_for_height` for the
    Bourgouin top); an automatic top-up during the real run would add a little.
    """
    out = out if out is not None else sys.stdout
    source = source if source is not None else make_source(cfg)
    n_wind = cfg.n_levels
    n_thermo = max(n_wind, levels_for_height(cfg.bourgouin_top_m))
    tot = {"hours": 0, "missing": 0, "payload": 0, "requested": 0}
    print(f"levels: wind 1..{n_wind}, thermo 1..{n_thermo}; bourgouin top {cfg.bourgouin_top_m:g} m", file=out)
    for r in runs:
        payload = requested = n_ok = 0
        missing = []
        for fxx in r.hours:
            try:
                hour = source.open_hour(r.init, fxx)
            except MissingInputError:
                missing.append(fxx)
                continue
            p, q, _ = hour.planned_bytes((0, n_wind), (0, n_thermo))
            payload += p
            requested += q
            n_ok += 1
        print(f"{r.run_id}: {n_ok} hours available, missing {missing or '-'}; payload {payload / 2**20:.1f} MiB, requested {requested / 2**20:.1f} MiB", file=out)
        tot["hours"] += n_ok
        tot["missing"] += len(missing)
        tot["payload"] += payload
        tot["requested"] += requested
    print(f"total: {tot['hours']} hours, {tot['missing']} missing, {tot['requested'] / 2**30:.2f} GiB to fetch", file=out)
    return tot


def summarize_timings(manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-stage median/mean seconds and peak RSS over the done hours of some manifests (for reports)."""
    rows = [r for m in manifests for r in m.get("hours", {}).values() if r.get("status") == "done"]
    out: dict[str, Any] = {"n_hours": len(rows)}
    for s in (*STAGES, "total"):
        v = [r["timings"].get(s, 0.0) for r in rows]
        if v:
            out[s] = {"median": float(np.median(v)), "mean": float(np.mean(v)), "max": float(np.max(v))}
    if rows:
        out["peak_rss_gb"] = max(r.get("peak_rss_bytes", 0) for r in rows) / 2**30
        out["payload_mib_mean"] = float(np.mean([r["bytes"]["payload_bytes"] for r in rows])) / 2**20
    return out


def fmt_seconds(x: float) -> str:
    """Seconds with 2 decimals (used in the markdown tables)."""
    return f"{x:.2f}" if math.isfinite(x) else "-"
