"""Read only the HRRR native-level (wrfnat) GRIB2 messages that precipitation-type inference needs.

A full HRRR ``wrfnat`` file is ~750-800 MB and holds ~1130 GRIB2 messages, but the
precipitation-type model only needs the temperature/moisture/wind/pressure/height profile on
the lowest N hybrid levels plus a dozen surface fields. NOAA publishes a wgrib2-style ``.idx``
sidecar next to every file listing the byte offset of every message, so the cheap path is:

1. download the small ``.idx`` text file and parse it (:func:`fetch_idx`, :func:`parse_idx`),
2. pick the messages we need and turn them into byte ranges (:func:`select_messages`),
3. fetch all of those byte ranges with ONE :func:`obstore.get_ranges` call, which coalesces
   nearby ranges into fewer HTTP requests and runs them concurrently (:func:`fetch_ranges`),
4. decode the GRIB2 messages straight from memory, never touching disk (:func:`decode`).

The same code path also reads a local directory laid out like the bucket
(``<root>/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfnatfFF.grib2`` plus ``.idx``) because
:class:`obstore.store.LocalStore` implements the same ``get_ranges`` interface.

Three interchangeable decoders are supported behind one interface (``decoder=`` argument):
``"eccodes"`` (``codes_new_from_message``), ``"pygrib"`` (``pygrib.fromstring``) and
``"gribberish"`` (``parse_grib_array``). All three return values in the file's native scan
order as ``(ny, nx)``; for HRRR that means row 0 is the SOUTHERN edge of the grid
(``jScansPositively = 1``). :func:`grid_metadata` reports the scan flags so downstream code
never has to assume the orientation.

Output layout of :func:`decode` (all ``float32``):

* 3D hybrid-level fields ``TMP``, ``SPFH``, ``PRES``, ``HGT``, ``UGRD``, ``VGRD`` with shape
  ``(n_levels, ny, nx)``; index 0 is hybrid level 1, the level nearest the ground, so the
  vertical axis runs surface-upward.
* 2D fields with shape ``(ny, nx)``: ``HGT_sfc`` (orography), ``PRES_sfc``, ``TMP_2m``,
  ``DPT_2m``, ``UGRD_10m``, ``VGRD_10m``, ``CRAIN``, ``CSNOW``, ``CICEP``, ``CFRZR``,
  ``PRATE`` and ``APCP_1h``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

HRRR_BUCKET = "noaa-hrrr-bdp-pds"
HRRR_REGION = "us-east-1"

#: Variables read on every hybrid level 1..N. HRRR wrfnat has no hybrid-level dewpoint, so the
#: moisture variable is specific humidity; dewpoint has to be derived from SPFH and PRES.
HYBRID_VARIABLES: tuple[str, ...] = ("TMP", "SPFH", "PRES", "HGT", "UGRD", "VGRD")

#: Instantaneous single-level fields: output name -> (idx variable, idx level string).
#: The level strings are exactly what wgrib2 writes into the HRRR wrfnat .idx files
#: (verified on hrrr.20231226 t00z f00/f01/f12; APCP step strings also on f24/f48).
SURFACE_FIELDS: dict[str, tuple[str, str]] = {
    "HGT_sfc": ("HGT", "surface"),
    "PRES_sfc": ("PRES", "surface"),
    "TMP_2m": ("TMP", "2 m above ground"),
    "DPT_2m": ("DPT", "2 m above ground"),
    "UGRD_10m": ("UGRD", "10 m above ground"),
    "VGRD_10m": ("VGRD", "10 m above ground"),
    "CRAIN": ("CRAIN", "surface"),
    "CSNOW": ("CSNOW", "surface"),
    "CICEP": ("CICEP", "surface"),
    "CFRZR": ("CFRZR", "surface"),
    "PRATE": ("PRATE", "surface"),
}

#: Output name of the 1-hour accumulated precipitation (selected by its accumulation window).
APCP_NAME = "APCP_1h"

_UNIT_HOURS = {"min": 1.0 / 60.0, "hour": 1.0, "day": 24.0}
_HYBRID_RE = re.compile(r"^(\d+) hybrid level$")
_ACC_RE = re.compile(r"^(\d+)-(\d+) (min|hour|day) (acc|ave|max|min) fcst$")
_INST_RE = re.compile(r"^(\d+) (min|hour|day) fcst$")


# --------------------------------------------------------------------------------------------
# idx parsing and message selection
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IdxEntry:
    """One line of a wgrib2-style ``.idx`` inventory.

    Attributes:
        number: Message number as written in the idx (a string because wgrib2 writes
            sub-messages as ``"12.1"``; HRRR does not use them but we do not want to crash on them).
        offset: Byte offset of the start of the GRIB message in the file.
        end: Exclusive end offset, i.e. the offset of the next message that starts at a different
            byte. ``None`` for the last message, whose end is the file size and is only known
            after a HEAD request (see :func:`fetch_ranges`).
        date: Reference time field, e.g. ``"d=2023122600"``.
        variable: Short name, e.g. ``"TMP"``.
        level: Level description, e.g. ``"1 hybrid level"`` or ``"2 m above ground"``.
        step: Forecast-time description, e.g. ``"12 hour fcst"``, ``"anl"`` or ``"11-12 hour acc fcst"``.
        extra: Anything after the step field (usually empty).
    """

    number: str
    offset: int
    end: int | None
    date: str
    variable: str
    level: str
    step: str
    extra: str = ""

    @property
    def size(self) -> int | None:
        """Message length in bytes, or ``None`` if this is the last message of the file."""
        return None if self.end is None else self.end - self.offset


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse the text of a wgrib2 ``.idx`` file.

    Each line looks like ``1054:703732743:d=2023122600:APCP:surface:0-12 hour acc fcst:``.
    The idx only stores start offsets, so a message's end is the start of the next message that
    begins at a *different* offset (sub-messages ``N.1``, ``N.2`` share an offset). Lines are sorted
    by offset before computing ends because nothing in the format guarantees file order.

    Args:
        text: Full idx text.

    Returns:
        Entries in file order with ``end`` filled in (``None`` for the last message).

    Raises:
        ValueError: If a non-empty line does not have at least six ``:``-separated fields.
    """
    raw: list[tuple[str, int, str, str, str, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6:
            raise ValueError(f"idx line {line_no} has fewer than 6 fields: {line!r}")
        # Anything after the step is joined back so that ':' inside extra descriptors survives.
        extra = ":".join(parts[6:]).strip(":")
        raw.append((parts[0], int(parts[1]), parts[2], parts[3], parts[4], parts[5], extra))
    raw.sort(key=lambda r: r[1])
    distinct_offsets = sorted({r[1] for r in raw})
    next_offset = {o: (distinct_offsets[i + 1] if i + 1 < len(distinct_offsets) else None) for i, o in enumerate(distinct_offsets)}
    return [IdxEntry(number=n, offset=o, end=next_offset[o], date=d, variable=v, level=lv, step=s, extra=e) for n, o, d, v, lv, s, e in raw]


def parse_step(step: str) -> tuple[str, float, float]:
    """Decode the idx step field into ``(kind, start_hours, end_hours)``.

    Needed because HRRR switches units for accumulations: the run-total APCP at f24 is written as
    ``0-1 day acc fcst`` and at f00 as ``0-0 day acc fcst``, while the hourly bucket is
    ``23-24 hour acc fcst``. Converting to hours lets one rule pick the 1-hour window everywhere.

    Args:
        step: The idx step string, e.g. ``"anl"``, ``"12 hour fcst"``, ``"11-12 hour acc fcst"``.

    Returns:
        ``kind`` is ``"inst"`` for analyses/instantaneous forecasts, otherwise the statistic
        (``"acc"``, ``"ave"``, ``"max"``, ``"min"``). For instantaneous fields start == end.

    Raises:
        ValueError: For step strings this parser does not understand.
    """
    step = step.strip()
    if step == "anl":
        return "inst", 0.0, 0.0
    m = _INST_RE.match(step)
    if m:
        h = int(m.group(1)) * _UNIT_HOURS[m.group(2)]
        return "inst", h, h
    m = _ACC_RE.match(step)
    if m:
        scale = _UNIT_HOURS[m.group(3)]
        return m.group(4), int(m.group(1)) * scale, int(m.group(2)) * scale
    raise ValueError(f"unrecognised idx step field: {step!r}")


@dataclass(frozen=True)
class PlannedMessage:
    """One GRIB message to fetch and where its decoded values go.

    Attributes:
        name: Output dictionary key (``"TMP"``, ``"PRES_sfc"``, ``"APCP_1h"``, ...).
        level_index: Index along the vertical axis for hybrid fields (0 = hybrid level 1, nearest
            the ground) or ``None`` for 2D fields.
        entry: The idx entry the bytes come from.
    """

    name: str
    level_index: int | None
    entry: IdxEntry


#: Per-variable hybrid level request: a count ``n`` (levels 1..n, i.e. 0-based indices ``0..n-1``) or a
#: half-open 0-based index range ``(lo, hi)`` (hybrid levels ``lo+1 .. hi``), used for top-up reads.
LevelSpec = int | tuple[int, int]


@dataclass
class ReadPlan:
    """The messages to read from one file, sorted by byte offset.

    Sorting by offset is deliberate: ``obstore.get_ranges`` coalesces neighbouring ranges, and the
    returned list is in the same order as the request, so keeping the plan in file order makes the
    request, the coalescing and the decode bookkeeping all line up.

    Attributes:
        messages: Planned messages in ascending file offset.
        n_levels: Highest hybrid level requested for any variable (the uniform count when all are equal).
        fxx: Forecast hour the plan was built for.
        hybrid_variables: Hybrid variables in the plan.
        level_ranges: Per hybrid variable, the half-open 0-based level-index range ``(lo, hi)`` it was
            planned for. :func:`decode` sizes each cube to ``hi - lo`` levels from this, so variables
            requested on different numbers of levels each get a cube of their own depth. ``None`` (for
            plans assembled by hand) means ``(0, n_levels)`` for every variable in ``hybrid_variables``,
            which is the original uniform behaviour.
    """

    messages: list[PlannedMessage]
    n_levels: int
    fxx: int
    hybrid_variables: tuple[str, ...] = field(default=HYBRID_VARIABLES)
    level_ranges: dict[str, tuple[int, int]] | None = None

    @property
    def starts(self) -> list[int]:
        """Start byte offsets, one per message."""
        return [m.entry.offset for m in self.messages]

    def total_bytes(self) -> int:
        """Sum of message sizes (exact bytes needed, before any coalescing)."""
        return sum(m.entry.size or 0 for m in self.messages)

    def expected_level_ranges(self) -> dict[str, tuple[int, int]]:
        """The ``(lo, hi)`` level-index range every hybrid variable must fill in :func:`decode`."""
        if self.level_ranges is not None:
            return dict(self.level_ranges)
        return {v: (0, self.n_levels) for v in self.hybrid_variables}


def _normalize_level_spec(n_levels: int | dict[str, LevelSpec], hybrid_variables: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Turn the ``n_levels`` argument of :func:`select_messages` into ``{variable: (lo, hi)}``.

    The int form keeps the original contract (every variable on levels 1..n, n >= 1). The mapping form
    lets each variable have its own depth; a count of 0 or an empty range drops that variable, which is
    what an incremental top-up needs when only some variables need more levels.
    """
    if isinstance(n_levels, int):
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        return {v: (0, n_levels) for v in hybrid_variables}
    ranges: dict[str, tuple[int, int]] = {}
    for var, spec in n_levels.items():
        if isinstance(spec, int):
            lo, hi = 0, spec
        else:
            lo, hi = (int(x) for x in spec)
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid level spec for {var}: {spec!r} (need count >= 0 or 0 <= lo <= hi)")
        if hi > lo:
            ranges[var] = (lo, hi)
    return ranges


def _select_apcp_1h(entries: Sequence[IdxEntry], fxx: int) -> IdxEntry:
    """Pick the APCP message whose accumulation window is the last hour.

    HRRR writes two APCP messages for fxx >= 2: the run total (``0-fxx``) and the hourly bucket
    (``fxx-1 - fxx``). At f01 both windows are the same so only one message exists, and at f00 the
    only message is the empty ``0-0 day acc fcst`` window (all zeros). Matching on the parsed step
    (not on message order) is what keeps this correct for every lead time.
    """
    want = (0.0, 0.0) if fxx == 0 else (float(fxx - 1), float(fxx))
    hits = []
    for e in entries:
        if e.variable != "APCP" or e.level != "surface":
            continue
        kind, start, end = parse_step(e.step)
        if kind == "acc" and math.isclose(start, want[0]) and math.isclose(end, want[1]):
            hits.append(e)
    if len(hits) != 1:
        raise KeyError(f"expected exactly one APCP {want[0]:g}-{want[1]:g} h window for f{fxx:02d}, found {len(hits)}")
    return hits[0]


def select_messages(
    entries: Sequence[IdxEntry],
    n_levels: int | dict[str, LevelSpec],
    fxx: int,
    hybrid_variables: Sequence[str] = HYBRID_VARIABLES,
    surface_fields: dict[str, tuple[str, str]] | None = None,
    include_apcp: bool = True,
) -> ReadPlan:
    """Choose the messages needed for precipitation-type inference.

    Instantaneous fields are matched on ``(variable, level)`` *and* on an instantaneous step
    (``anl`` or ``N hour fcst``). The step check matters because the same variable/level pair can
    also appear as a time statistic (e.g. ``WEASD:surface`` appears both instantaneous and as an
    accumulation), and silently taking the wrong one would be a subtle bug.

    Args:
        entries: Output of :func:`parse_idx`.
        n_levels: Either an int ``n`` (every variable in ``hybrid_variables`` on hybrid levels 1..n, the
            level nearest the ground first), or a mapping ``{variable: spec}`` giving each hybrid variable its
            own depth, where ``spec`` is a count ``n`` (levels 1..n) or a 0-based half-open index range
            ``(lo, hi)`` (hybrid levels ``lo+1..hi``, for top-up reads). With a mapping, its keys are the
            hybrid variables and ``hybrid_variables`` is ignored; variables with a count of 0 or an empty
            range are left out. Example: ``{"TMP": 34, "SPFH": 34, "PRES": 34, "HGT": 34, "UGRD": 27, "VGRD": 27}``.
        fxx: Forecast hour of the file; used to pick the 1-hour APCP window.
        hybrid_variables: Variables to read on each hybrid level.
        surface_fields: Mapping of output name to ``(variable, level)``; defaults to :data:`SURFACE_FIELDS`.
        include_apcp: Whether to add the 1-hour APCP message.

    Returns:
        A :class:`ReadPlan` sorted by byte offset.

    Raises:
        KeyError: If a required message is missing or ambiguous.
        ValueError: If an int ``n_levels`` is not positive or a mapping spec is malformed.
    """
    ranges = _normalize_level_spec(n_levels, hybrid_variables)
    surface_fields = SURFACE_FIELDS if surface_fields is None else surface_fields
    wanted_hybrid = set(ranges)
    by_key: dict[tuple[str, str], list[IdxEntry]] = {}
    planned: list[PlannedMessage] = []
    for e in entries:
        m = _HYBRID_RE.match(e.level)
        if m and e.variable in wanted_hybrid:
            lo, hi = ranges[e.variable]
            index = int(m.group(1)) - 1
            if lo <= index < hi:
                planned.append(PlannedMessage(e.variable, index, e))
            continue
        if parse_step_kind_is_inst(e.step):
            by_key.setdefault((e.variable, e.level), []).append(e)

    # Every requested (variable, level) must be present exactly once; a gap would otherwise
    # surface much later as an array of zeros in the decoded cube.
    found = {(p.name, p.level_index) for p in planned}
    missing = [f"{v}:{lv + 1} hybrid level" for v, (lo, hi) in ranges.items() for lv in range(lo, hi) if (v, lv) not in found]
    if missing:
        raise KeyError(f"missing hybrid-level messages: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if len(found) != len(planned):
        raise KeyError("duplicate hybrid-level messages in idx")

    for name, key in surface_fields.items():
        hits = by_key.get(key, [])
        if len(hits) != 1:
            raise KeyError(f"expected one instantaneous {key[0]}:{key[1]} message for {name}, found {len(hits)}")
        planned.append(PlannedMessage(name, None, hits[0]))
    if include_apcp:
        planned.append(PlannedMessage(APCP_NAME, None, _select_apcp_1h(entries, fxx)))
    planned.sort(key=lambda p: p.entry.offset)
    return ReadPlan(
        messages=planned,
        # The int form keeps reporting the requested count (unchanged behaviour); the mapping form
        # reports the deepest variable.
        n_levels=n_levels if isinstance(n_levels, int) else max((hi for _, hi in ranges.values()), default=0),
        fxx=fxx,
        hybrid_variables=tuple(ranges),
        level_ranges=ranges,
    )


def parse_step_kind_is_inst(step: str) -> bool:
    """Return True for ``anl`` / ``N hour fcst`` steps and False for statistics or unknown steps."""
    try:
        return parse_step(step)[0] == "inst"
    except ValueError:
        return False


# --------------------------------------------------------------------------------------------
# Stores and byte-range fetching
# --------------------------------------------------------------------------------------------
def hrrr_key(init: date | datetime, cycle: int, fxx: int, product: str = "wrfnat") -> str:
    """Object key of an HRRR CONUS GRIB2 file, identical in the bucket and in a local mirror.

    Args:
        init: Initialization date.
        cycle: Initialization hour (0-23).
        fxx: Forecast hour.
        product: HRRR product name; this module is written for ``wrfnat``.

    Returns:
        e.g. ``"hrrr.20231226/conus/hrrr.t00z.wrfnatf12.grib2"``.
    """
    return f"hrrr.{init:%Y%m%d}/conus/hrrr.t{cycle:02d}z.{product}f{fxx:02d}.grib2"


def open_store(source: str | Path | None = None):
    """Return an obstore store for the public HRRR bucket or for a local mirror directory.

    Args:
        source: ``None`` or ``"s3"`` for the anonymous AWS bucket; otherwise a local directory
            containing ``hrrr.YYYYMMDD/conus/...``.

    Returns:
        An ``obstore`` store. Anonymous S3 access needs ``skip_signature=True``; without it obstore
        would look for credentials and fail (or, worse, sign requests with unrelated credentials).
    """
    from obstore.store import LocalStore, S3Store

    if source is None or source == "s3":
        return S3Store(HRRR_BUCKET, region=HRRR_REGION, skip_signature=True)
    return LocalStore(prefix=Path(source).expanduser().resolve())


def _as_store(store_or_path):
    """Accept either an obstore store or a local directory path."""
    if isinstance(store_or_path, (str, Path)):
        return open_store(store_or_path)
    return store_or_path


def fetch_idx(store_or_path, key: str) -> str:
    """Download and return the ``.idx`` text that sits next to ``key``."""
    import obstore

    store = _as_store(store_or_path)
    return bytes(obstore.get(store, key + ".idx").bytes()).decode("ascii")


def fetch_ranges(store_or_path, key: str, plan: ReadPlan, coalesce: int = 1024 * 1024) -> list[memoryview]:
    """Fetch every planned message with a single ``obstore.get_ranges`` call.

    One call is used (rather than one ``get_range`` per message) because obstore then merges
    ranges that are less than ``coalesce`` bytes apart into one HTTP request and issues the
    resulting requests concurrently; the merged response is sliced back into per-message buffers
    without copying. The hybrid-level messages we want are interleaved with ~14 unwanted variables
    per level, so the gap size relative to ``coalesce`` decides how many requests are made and how
    many surplus bytes are downloaded (see ``benchmarks/reader_benchmark.py``).

    Args:
        store_or_path: obstore store or local directory.
        key: Object key of the GRIB2 file.
        plan: Output of :func:`select_messages`.
        coalesce: Maximum gap in bytes between ranges that obstore will merge; ``0`` disables merging.

    Returns:
        One buffer per planned message, in plan order. The obstore ``Bytes`` objects are returned
        as zero-copy ``memoryview``s.
    """
    import obstore

    store = _as_store(store_or_path)
    ends: list[int] = []
    file_size: int | None = None
    for m in plan.messages:
        if m.entry.end is None:
            # Only the very last message in the file has an unknown end; one HEAD resolves it.
            if file_size is None:
                file_size = int(obstore.head(store, key)["size"])
            ends.append(file_size)
        else:
            ends.append(m.entry.end)
    out = obstore.get_ranges(store, key, starts=plan.starts, ends=ends, coalesce=coalesce)
    return [memoryview(b) for b in out]


def merged_ranges(starts: Sequence[int], ends: Sequence[int], coalesce: int) -> list[tuple[int, int]]:
    """Reproduce object_store's ``merge_ranges`` to report how many requests/bytes a fetch costs.

    obstore does not expose transfer statistics, so the benchmark uses this mirror of the Rust
    algorithm (sort by start; extend the current request while the next range starts at most
    ``coalesce`` bytes after the current end) to count requests and requested bytes.

    Args:
        starts: Range starts.
        ends: Exclusive range ends.
        coalesce: Merge threshold in bytes.

    Returns:
        The merged ``(start, end)`` requests.
    """
    ranges = sorted(zip(starts, ends, strict=True))
    out: list[tuple[int, int]] = []
    for s, e in ranges:
        if out and s - out[-1][1] <= coalesce:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# --------------------------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------------------------
def _decode_eccodes(buf) -> np.ndarray:
    """Decode one message with ecCodes from memory.

    ``codes_get_values`` returns the values in storage (scan) order as float64; we reshape with
    the grid's own ``Nj``/``Ni`` so the result is ``(ny, nx)`` in native scan order. Points flagged
    missing by the bitmap come back as ``missingValue`` and are converted to NaN so every decoder
    reports missing data the same way.
    """
    import eccodes

    gid = eccodes.codes_new_from_message(bytes(buf))
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        vals = eccodes.codes_get_values(gid)
        if eccodes.codes_get(gid, "bitmapPresent"):
            missing = eccodes.codes_get_double(gid, "missingValue")
            vals = np.where(vals == missing, np.nan, vals)
    finally:
        eccodes.codes_release(gid)
    return vals.reshape(nj, ni)


def _decode_pygrib(buf) -> np.ndarray:
    """Decode one message with pygrib from memory.

    ``pygrib.fromstring`` wraps ecCodes; ``.values`` returns a masked array when a bitmap is
    present, which we fill with NaN to match the other decoders.
    """
    import pygrib

    vals = pygrib.fromstring(bytes(buf)).values
    if np.ma.isMaskedArray(vals):
        vals = vals.astype(np.float64).filled(np.nan)
    return np.asarray(vals)


def _decode_gribberish(buf) -> np.ndarray:
    """Decode one message with gribberish (pure Rust, no ecCodes).

    ``parse_grib_array`` returns a FLAT float64 array (gribberish 1.9.0) in native scan order when
    ``north_up=False``; ``north_up=True`` would flip a south-first grid, which we avoid so all
    decoders agree. The ``(ny, nx)`` shape comes from ``parse_grib_message_metadata``, which reads
    only the section headers. We deliberately do not use gribberish's ``crs`` (it reports
    ``EPSG:2154`` for the HRRR grid in 1.9.0, which is wrong); :func:`grid_metadata` uses ecCodes.
    """
    import gribberish

    data = bytes(buf)
    shape = gribberish.parse_grib_message_metadata(data, 0).grid_shape
    return np.asarray(gribberish.parse_grib_array(data, 0, north_up=False)).reshape(shape)


DECODERS: dict[str, Callable[[object], np.ndarray]] = {
    "eccodes": _decode_eccodes,
    "pygrib": _decode_pygrib,
    "gribberish": _decode_gribberish,
}


def decode_message(buf, decoder: str = "eccodes") -> np.ndarray:
    """Decode one GRIB2 message to a float32 ``(ny, nx)`` array in native scan order."""
    return DECODERS[decoder](buf).astype(np.float32, copy=False)


def _decode_task(args: tuple[bytes, str]) -> np.ndarray:
    """Top-level (picklable) wrapper used by the process pool."""
    buf, decoder = args
    return decode_message(buf, decoder)


def _check_plan_levels(plan: ReadPlan) -> dict[str, tuple[int, int]]:
    """Verify that the plan fills every expected level slot exactly once; return the level ranges.

    Done before any decoding so a malformed plan fails fast with a message naming the gaps, rather
    than after seconds of decoding (or, worse, returning cubes with levels that were never written).
    """
    ranges = plan.expected_level_ranges()
    seen: dict[str, set[int]] = {v: set() for v in ranges}
    for pm in plan.messages:
        if pm.level_index is None:
            continue
        if pm.name not in ranges:
            raise ValueError(f"plan has a hybrid message for {pm.name!r} but no level range for it (known: {sorted(ranges)})")
        lo, hi = ranges[pm.name]
        if not lo <= pm.level_index < hi:
            raise ValueError(f"{pm.name} level index {pm.level_index} is outside its planned range [{lo}, {hi})")
        if pm.level_index in seen[pm.name]:
            raise ValueError(f"duplicate message for {pm.name} level index {pm.level_index}")
        seen[pm.name].add(pm.level_index)
    missing = [f"{v}:{i + 1} hybrid level" for v, (lo, hi) in ranges.items() for i in range(lo, hi) if i not in seen[v]]
    if missing:
        raise ValueError(f"plan is missing {len(missing)} hybrid-level message(s): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    return ranges


def decode(
    plan: ReadPlan,
    messages: Sequence,
    decoder: str = "eccodes",
    workers: int = 1,
    pool: str = "thread",
) -> dict[str, np.ndarray]:
    """Decode fetched messages into named float32 arrays.

    Each hybrid variable gets a cube sized to ITS OWN planned level range (``plan.level_ranges``), so a
    plan that reads TMP on 34 levels and UGRD on 27 returns ``TMP.shape[0] == 34`` and
    ``UGRD.shape[0] == 27``. Cube level ``k`` holds 0-based level index ``lo + k`` (hybrid level
    ``lo + k + 1``); for the usual ``lo = 0`` that is simply surface-upward from hybrid level 1.

    Cubes are allocated once, NaN-filled (never ``np.empty``) and each decoded message is written into
    its slot, so the thread path never concatenates or copies a 3D cube and no uninitialised memory
    can leak out. The plan is checked before decoding: every expected level slot must have exactly one
    message. With ``pool="process"`` the message bytes are pickled to the workers and the decoded
    arrays pickled back, which costs ~8 MB of IPC per 3D slice.

    Args:
        plan: The plan the messages were fetched with (gives names, level indices and level ranges).
        messages: Buffers from :func:`fetch_ranges`, in plan order.
        decoder: ``"eccodes"``, ``"pygrib"`` or ``"gribberish"``.
        workers: Number of workers; ``1`` decodes serially in the calling thread.
        pool: ``"thread"`` or ``"process"``.

    Returns:
        ``{name: array}``; 3D hybrid fields are ``(hi - lo, ny, nx)`` ordered surface-upward,
        2D fields are ``(ny, nx)``.

    Raises:
        ValueError: On a message-count mismatch, unknown decoder/pool, or a plan whose hybrid messages do
            not cover every expected level exactly once.
    """
    if len(messages) != len(plan.messages):
        raise ValueError(f"got {len(messages)} messages for a plan of {len(plan.messages)}")
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}; choose from {sorted(DECODERS)}")
    if workers > 1 and pool not in ("thread", "process"):
        raise ValueError(f"pool must be 'thread' or 'process', not {pool!r}")
    ranges = _check_plan_levels(plan)

    out: dict[str, np.ndarray] = {}

    def store(pm: PlannedMessage, arr: np.ndarray) -> None:
        if pm.level_index is None:
            out[pm.name] = arr
            return
        lo, hi = ranges[pm.name]
        cube = out.get(pm.name)
        if cube is None:
            # The grid shape is only known once a message is decoded, hence the lazy allocation.
            cube = out[pm.name] = np.full((hi - lo, *arr.shape), np.nan, dtype=np.float32)
        cube[pm.level_index - lo] = arr

    if workers <= 1:
        for pm, buf in zip(plan.messages, messages, strict=True):
            store(pm, decode_message(buf, decoder))
    elif pool == "thread":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = ex.map(lambda b: decode_message(b, decoder), messages)
            for pm, arr in zip(plan.messages, results, strict=True):
                store(pm, arr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(_decode_task, [(bytes(b), decoder) for b in messages], chunksize=max(1, len(messages) // (4 * workers)))
            for pm, arr in zip(plan.messages, results, strict=True):
                store(pm, arr)
    # _check_plan_levels guarantees every slot had a message; this guards the invariant cheaply.
    absent = [v for v in ranges if v not in out]
    if absent:
        raise ValueError(f"no messages decoded for hybrid variables {absent}")
    return out


# --------------------------------------------------------------------------------------------
# Grid metadata
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GridMetadata:
    """Lambert conformal grid description read from a GRIB2 grid definition section.

    Attributes:
        nx: Points along a row (``Ni``/``Nx``).
        ny: Points along a column (``Nj``/``Ny``).
        dx: Grid spacing in x, metres.
        dy: Grid spacing in y, metres.
        lon_0: Orientation longitude ``LoV`` in degrees east, in [-180, 180).
        lat_0: Latitude where dx/dy are true (``LaD``); used as the projection origin latitude.
        lat_1: First standard parallel (``Latin1``).
        lat_2: Second standard parallel (``Latin2``).
        earth_radius: Sphere radius in metres (HRRR uses ``shapeOfTheEarth = 6``, 6371229 m).
        shape_of_the_earth: GRIB code table 3.2 value.
        lat_first: Latitude of the first stored grid point.
        lon_first: Longitude of the first stored grid point, in [-180, 180).
        x_first: Projected x of the first grid point, metres, for the spherical LCC above.
        y_first: Projected y of the first grid point, metres.
        i_scans_negatively: GRIB scanning-mode flag bit 1.
        j_scans_positively: GRIB scanning-mode flag bit 2; 1 means rows go south to north.
        j_points_consecutive: GRIB scanning-mode flag bit 3.
        row0_is_south: Convenience flag, ``True`` when array row 0 is the southern edge.
        proj4: PROJ string matching the parameters above.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    lon_0: float
    lat_0: float
    lat_1: float
    lat_2: float
    earth_radius: float
    shape_of_the_earth: int
    lat_first: float
    lon_first: float
    x_first: float
    y_first: float
    i_scans_negatively: int
    j_scans_positively: int
    j_points_consecutive: int
    row0_is_south: bool
    proj4: str

    def xy_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """1D projected x and y coordinates of array columns/rows (respecting the scan flags)."""
        sx = -1.0 if self.i_scans_negatively else 1.0
        sy = 1.0 if self.j_scans_positively else -1.0
        return self.x_first + sx * self.dx * np.arange(self.nx), self.y_first + sy * self.dy * np.arange(self.ny)


def _wrap180(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


def lcc_forward(lon, lat, lon_0: float, lat_0: float, lat_1: float, lat_2: float, radius: float):
    """Spherical Lambert conformal conic forward projection (Snyder 1987, eqs. 15-1 to 15-4).

    Implemented here so the reader can describe the grid with only ecCodes installed; the
    benchmark checks it against pyproj.
    """
    phi1, phi2, phi0 = (math.radians(v) for v in (lat_1, lat_2, lat_0))
    if math.isclose(lat_1, lat_2):
        n = math.sin(phi1)
    else:
        n = math.log(math.cos(phi1) / math.cos(phi2)) / math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2))
    f = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n
    rho0 = radius * f / math.tan(math.pi / 4 + phi0 / 2) ** n
    lam = np.radians((np.asarray(lon, dtype=np.float64) - lon_0 + 180.0) % 360.0 - 180.0)
    rho = radius * f / np.tan(np.pi / 4 + np.radians(np.asarray(lat, dtype=np.float64)) / 2) ** n
    return rho * np.sin(n * lam), rho0 - rho * np.cos(n * lam)


def grid_metadata(message) -> GridMetadata:
    """Read the Lambert conformal grid definition of one GRIB2 message with ecCodes.

    Every value comes from the message's grid definition section rather than from hard-coded
    HRRR constants, so a grid change (or a different Lambert model) is picked up automatically.

    Args:
        message: Bytes of any one message from the file (all HRRR wrfnat messages share a grid).

    Returns:
        A :class:`GridMetadata`.

    Raises:
        ValueError: If the grid is not Lambert conformal or the earth is not a sphere of known radius.
    """
    import eccodes

    gid = eccodes.codes_new_from_message(bytes(message))
    try:
        g = eccodes.codes_get
        grid_type = g(gid, "gridType")
        if grid_type != "lambert":
            raise ValueError(f"expected a Lambert conformal grid, got {grid_type!r}")
        shape = int(g(gid, "shapeOfTheEarth"))
        if shape == 6:
            radius = 6371229.0
        elif shape == 0:
            radius = 6367470.0
        elif shape == 1:
            radius = float(g(gid, "radius"))
        else:
            raise ValueError(f"unsupported shapeOfTheEarth={shape} (need a sphere)")
        meta = {
            "nx": int(g(gid, "Nx")),
            "ny": int(g(gid, "Ny")),
            "dx": float(g(gid, "DxInMetres")),
            "dy": float(g(gid, "DyInMetres")),
            "lon_0": _wrap180(float(g(gid, "LoVInDegrees"))),
            "lat_0": float(g(gid, "LaDInDegrees")),
            "lat_1": float(g(gid, "Latin1InDegrees")),
            "lat_2": float(g(gid, "Latin2InDegrees")),
            "lat_first": float(g(gid, "latitudeOfFirstGridPointInDegrees")),
            "lon_first": _wrap180(float(g(gid, "longitudeOfFirstGridPointInDegrees"))),
            "i_scans_negatively": int(g(gid, "iScansNegatively")),
            "j_scans_positively": int(g(gid, "jScansPositively")),
            "j_points_consecutive": int(g(gid, "jPointsAreConsecutive")),
        }
    finally:
        eccodes.codes_release(gid)
    x0, y0 = lcc_forward(meta["lon_first"], meta["lat_first"], meta["lon_0"], meta["lat_0"], meta["lat_1"], meta["lat_2"], radius)
    proj4 = f"+proj=lcc +lat_0={meta['lat_0']} +lon_0={meta['lon_0']} +lat_1={meta['lat_1']} +lat_2={meta['lat_2']} +R={radius} +units=m +no_defs"
    return GridMetadata(
        earth_radius=radius,
        shape_of_the_earth=shape,
        x_first=float(x0),
        y_first=float(y0),
        row0_is_south=bool(meta["j_scans_positively"]),
        proj4=proj4,
        **meta,
    )


# --------------------------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------------------------
def read_hrrr_fields(
    init: date | datetime,
    cycle: int,
    fxx: int,
    n_levels: int,
    source: str | Path | None = None,
    decoder: str = "eccodes",
    workers: int = 8,
    pool: str = "thread",
    coalesce: int = 1024 * 1024,
) -> tuple[dict[str, np.ndarray], GridMetadata]:
    """Fetch idx, plan, fetch byte ranges, decode, and describe the grid for one HRRR wrfnat file.

    Args:
        init: Initialization date.
        cycle: Initialization hour.
        fxx: Forecast hour.
        n_levels: Hybrid levels to read (1..n_levels).
        source: ``None``/``"s3"`` for the public bucket or a local mirror directory.
        decoder: Decoder name (see :data:`DECODERS`).
        workers: Decode workers.
        pool: ``"thread"`` or ``"process"``.
        coalesce: obstore range-coalescing threshold in bytes.

    Returns:
        ``(fields, grid)``.
    """
    store = open_store(source)
    key = hrrr_key(init, cycle, fxx)
    plan = select_messages(parse_idx(fetch_idx(store, key)), n_levels=n_levels, fxx=fxx)
    msgs = fetch_ranges(store, key, plan, coalesce=coalesce)
    return decode(plan, msgs, decoder=decoder, workers=workers, pool=pool), grid_metadata(msgs[0])
