"""Zarr v3 writers for HRRR precipitation-type output: one gridded multiscale store and one sounding store per run.

Both stores are built for many writers working in parallel on the same run, one forecast hour each:

* ``create_*_store`` writes ALL metadata up front (group and array ``zarr.json`` files, coordinate
  arrays, every forecast-hour slot). Nothing is ever appended or resized later, so readers always see a
  complete, valid store and hours that have not been written yet simply read as the fill value
  (NaN for floats, 255 for the uint8 flags).
* ``write_*_hour`` writes the data of ONE forecast hour and touches only chunk (or shard) objects whose
  key contains that hour's index. Every chunk/shard has extent 1 along ``forecast_hour`` and the write
  always covers whole chunks, so zarr never has to read-modify-write an object another process could be
  writing, and no metadata document is rewritten. Different hours therefore map to disjoint sets of
  object keys; a local ``zarr.storage.LocalStore`` writes each object to a temporary file and renames it
  into place, and an S3 PUT is atomic per key, so concurrent writers of different hours need no lock.
  (Two writers of the SAME hour would race, last writer wins; the scheduler must not do that.)

Gridded store (read by the website's ``src/data/ptype.ts``)
-----------------------------------------------------------
The layout is the one ``topozarr.create_pyramid(...).write(...)`` produces, which is what the site and
``tools/make_sample_data.py`` use: a root group whose attributes hold the ``zarr_conventions``
(multiscales, proj, spatial), ``proj:code``/``proj:wkt2``, ``spatial:*`` and ``multiscales.layout``
(``asset``, ``derived_from``, ``transform``, ``resampling_method``, ``spatial:transform``,
``spatial:shape``), and one group per level (``"0"``, ``"1"``, ...) holding ``x``, ``y``,
``forecast_hour``, ``spatial_ref`` and the data arrays on ``(forecast_hour, y, x)``. topozarr itself
cannot be used for the writes: it builds a pyramid from a complete in-memory/dask Dataset in one call
(no per-hour region writes), applies ONE resampling method to every variable, and its default shards at
coarse levels span several forecast hours (e.g. ``(2, 264, 225)``), which would make two hours share an
object. The attributes are therefore written here; ``tests/test_hrrr_fast_writers.py`` diffs them
against a store written by topozarr for the same grid.

Coarser levels halve both spatial dimensions (floor, cascading from the previous level exactly like
topozarr) with a per-variable aggregation recorded in each array's ``resampling_method`` attribute:

* ``mean`` for probabilities, ``ML_u`` and precipitation rate: the coarse cell shows the areal average,
  which keeps the four probabilities summing to 1 and conserves the area-mean rate.
* ``max`` for 0/1 flags (``precip_mask``, ``CRAIN``/``CSNOW``/``CICEP``/``CFRZR``): "any fine cell
  flagged". At a 24 km cell a 3 km freezing-rain band covers a minority of the fine cells, so ``mode``
  or ``nearest`` would erase exactly the narrow, high-impact features; ``max`` keeps them at the cost of
  widening flagged areas by up to one coarse cell at their edges. Because the site shades probabilities
  only where the mask is set, ``max`` also guarantees that a coarse cell is never blanked while some of
  its fine cells have precipitation.
* ``mode`` for categorical class codes (e.g. a Bourgouin class added later): ``max``/``mean`` of class
  codes are meaningless. Ties go to the smallest code; fill values are ignored.
* ``nearest`` (top-left cell, topozarr's decimation) is available but not used by default.

Sounding store (read by ``src/data/soundings.ts``)
--------------------------------------------------
``sounding (forecast_hour, location, variable)`` float32 with ``location = j * ni + i`` (``i`` west to
east, ``j`` south to north) and ``variable`` = the five groups (T, Td, u, v, p) x 21 heights. Winds are
stored EARTH-relative; the root attributes give the rotation needed to get the grid-relative winds the
network was trained/run on. Inner chunks are ``(1, location_chunk, n_var)`` so a click decodes one small
chunk; with ``sharding=True`` all inner chunks of one hour live in one shard object (an index of
``n_chunks * 16 + 4`` bytes at its end), which turns ~1,860 objects per hour into one. zarrita 0.7.5
reads sharded arrays with range requests (``zarrita/src/codecs/sharding.ts``: suffix read of the index,
then one range read per inner chunk; ``FetchStore`` issues a HEAD to learn the object size unless
``useSuffixRequest`` is set).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import zarr
from zarr.codecs import BloscCodec, CastValue, ScaleOffset, TransposeCodec, ZstdCodec

__all__ = [
    "Aggregation",
    "VariableSpec",
    "block_reduce",
    "coarsen_shapes",
    "create_ptype_store",
    "create_sounding_store",
    "default_ptype_variables",
    "default_sounding_encoding",
    "level_coords",
    "quantized_float",
    "sounding_grid_attrs",
    "write_ptype_hour",
    "write_sounding_hour",
]

Aggregation = Literal["mean", "max", "mode", "nearest"]

#: Zarr conventions written by topozarr 0.3.0 (``topozarr.metadata``); copied verbatim so that the
#: root/level attributes are byte-for-byte the same structure the site was developed against.
_CONVENTIONS = {
    "multiscales": {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
        "uuid": "d35379db-88df-4056-af3a-620245f8e347",
        "name": "multiscales",
        "description": "Multiscale layout of zarr datasets",
    },
    "proj": {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
        "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
        "name": "proj",
        "description": "Coordinate reference system information for geospatial data",
    },
    "spatial": {
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
        "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
        "name": "spatial",
        "description": "Spatial coordinate information",
    },
}

#: Default spatial chunk of the gridded store (applied per level as ``min(level size, chunk)``).
#: The site always reads one whole level of one hour for each of 5 variables, so with this chunk every
#: level is ONE object per variable (5 GETs per map frame); see benchmarks/results/compression_local.md
#: for the request/byte counts of the alternatives.
DEFAULT_GRID_CHUNK = (1059, 1799)

#: Default number of pyramid levels: 3, 6, 12 and 24 km. The site's ``chooseLevel`` picks the coarsest
#: level with dx <= 2 screen pixels; at its minimum zoom (3) that is 25-35 km over CONUS latitudes, so a
#: 48 km level would never be chosen.
DEFAULT_LEVELS = 4

#: Default zstd level. Level 3 is zstd's own default; see the benchmark for 1/3/9.
DEFAULT_ZSTD_LEVEL = 3


# ---------------------------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------------------------
def quantized_float(step: float, dtype: str = "uint8", offset: float = 0.0) -> list:
    """Array-to-array filters that store a float32 array as a scaled integer.

    Uses the zarr-extensions ``scale_offset`` + ``cast_value`` codecs (zarr-python >= 3.4 with
    ``cast-value-rs``; decoded by zarrita 0.7.5, ``src/codecs/scale_offset.ts`` and
    ``src/codecs/cast_value.ts``). The ARRAY keeps ``data_type: float32``, so a browser gets a
    ``Float32Array`` back with no application-side scaling, and missing chunks still read as NaN. The
    largest integer of the target type is reserved for NaN through ``scalar_map`` (so e.g. uint8 with
    ``step=1/254`` covers 0..1 in 254 steps; 255 means missing).

    Args:
        step: Quantisation step (``1 / scale``); values are rounded to the nearest multiple (ties to even).
        dtype: Stored integer type: ``uint8``, ``uint16``, ``int16`` or ``int32``.
        offset: Subtracted before scaling (``stored = round((x - offset) / step)``).

    Returns:
        ``[ScaleOffset, CastValue]`` to pass as ``filters``.
    """
    info = np.iinfo(dtype)
    sentinel = int(info.max) if info.min == 0 else int(info.min)
    scale = 1.0 / step
    scale_json: float | int = round(scale) if math.isclose(scale, round(scale), rel_tol=0, abs_tol=1e-9) else scale
    so = ScaleOffset(offset=offset, scale=scale_json) if offset else ScaleOffset(scale=scale_json)
    cv = CastValue(data_type=dtype, rounding="nearest-even", out_of_range="clamp", scalar_map={"encode": [("NaN", sentinel)], "decode": [(sentinel, "NaN")]})
    return [so, cv]


@dataclass(frozen=True)
class VariableSpec:
    """How one gridded variable is stored.

    Attributes:
        dtype: Array data type (what readers get back). Probabilities stay ``float32`` even when stored
            quantised, because the site converts whatever it reads with ``Float32Array.from``.
        fill_value: Value of never-written chunks, i.e. of forecast hours that are not written yet.
        aggregation: How coarser pyramid levels are derived (see the module docstring).
        attrs: Array attributes (units, long_name, ...).
        filters: Zarr v3 array-to-array codecs (e.g. :func:`quantized_float`).
        compressors: Zarr v3 bytes-to-bytes codecs.
    """

    dtype: str = "float32"
    fill_value: float | int = float("nan")
    aggregation: Aggregation = "mean"
    attrs: Mapping[str, Any] = field(default_factory=dict)
    filters: Sequence[Any] = ()
    compressors: Sequence[Any] = (ZstdCodec(level=DEFAULT_ZSTD_LEVEL),)


def _prob_spec(long_name: str) -> VariableSpec:
    return VariableSpec(attrs={"long_name": long_name, "units": "1"}, aggregation="mean", filters=quantized_float(1.0 / 254.0, "uint8"))


def _flag_spec(long_name: str) -> VariableSpec:
    return VariableSpec(dtype="uint8", fill_value=255, aggregation="max", attrs={"long_name": long_name, "units": "1", "flag_values": [0, 1], "missing_value": 255})


def default_ptype_variables() -> dict[str, VariableSpec]:
    """The variables every run store holds; extra variables can be added by passing more specs.

    Probabilities and ``ML_u`` are stored as uint8 steps of 1/254 behind a float32 array (see
    benchmarks/results/compression_local.md for the size/error trade-off); flags are uint8 with 255 =
    missing; precipitation rate is float32 + zstd (it spans orders of magnitude, so a fixed step is a
    poor fit).
    """
    return {
        "ML_rain": _prob_spec("ML probability of rain"),
        "ML_snow": _prob_spec("ML probability of snow"),
        "ML_icep": _prob_spec("ML probability of ice pellets"),
        "ML_frzr": _prob_spec("ML probability of freezing rain"),
        "ML_u": replace(_prob_spec("ML evidential uncertainty u = K / S"), attrs={"long_name": "ML evidential uncertainty u = K / S", "units": "1"}),
        "precip_mask": _flag_spec("any of HRRR CRAIN/CSNOW/CICEP/CFRZR == 1"),
        "CRAIN": _flag_spec("HRRR categorical rain"),
        "CSNOW": _flag_spec("HRRR categorical snow"),
        "CICEP": _flag_spec("HRRR categorical ice pellets"),
        "CFRZR": _flag_spec("HRRR categorical freezing rain"),
        "prate": VariableSpec(attrs={"long_name": "HRRR precipitation rate", "units": "kg m-2 s-1"}, aggregation="mean"),
    }


# ---------------------------------------------------------------------------------------------
# Grid / pyramid geometry
# ---------------------------------------------------------------------------------------------
def coarsen_shapes(ny: int, nx: int, n_levels: int) -> list[tuple[int, int]]:
    """Level shapes, halving with floor at each step like topozarr (1059x1799 -> 529x899 -> ...)."""
    shapes = [(ny, nx)]
    for _ in range(1, n_levels):
        py, px = shapes[-1]
        shapes.append((py // 2, px // 2))
    return shapes


def level_coords(grid, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Projected cell-centre x/y of a pyramid level: the mean of the fine cells each coarse cell covers.

    This equals what topozarr writes (it coarsens the coordinate arrays with a block mean), and keeps
    every level's ``spatial:transform`` origin at the level-0 corner.
    """
    x0, y0 = grid.xy_coords()
    f = 2**level
    nyl, nxl = coarsen_shapes(len(y0), len(x0), level + 1)[-1]
    x = x0[: nxl * f].reshape(nxl, f).mean(axis=1)
    y = y0[: nyl * f].reshape(nyl, f).mean(axis=1)
    return x, y


def _transform(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Affine ``[a, b, c, d, e, f]`` with (c, f) the outer corner of cell (0, 0), as topozarr computes it."""
    rx = float(x[1] - x[0])
    ry = float(y[1] - y[0])
    return [rx, 0.0, float(x[0]) - 0.5 * rx, 0.0, ry, float(y[0]) - 0.5 * ry]


def _bbox(t: list[float], ny: int, nx: int) -> list[float]:
    rx, _, c, _, ry, f = t
    return [min(c, c + rx * nx), min(f, f + ry * ny), max(c, c + rx * nx), max(f, f + ry * ny)]


def _crs_attrs(grid) -> dict[str, Any]:
    import pyproj

    crs = pyproj.CRS.from_proj4(grid.proj4)
    return {"proj:code": str(crs), "proj:wkt2": crs.to_wkt()}


def sounding_grid_attrs(grid) -> dict[str, Any]:
    """The ``grid`` attribute block the site's sounding reader (and make_sample_data.py) uses.

    ``ll_x``/``ll_y`` are the projected coordinates of the centre of cell (i=0, j=0), the south-west
    corner because rows are stored south to north.
    """
    x, y = grid.xy_coords()
    return {
        "projection": "lambert_conformal",
        "ni": int(grid.nx),
        "nj": int(grid.ny),
        "lon_0": float(grid.lon_0),
        "lat_0": float(grid.lat_0),
        "lat_std": [float(grid.lat_1), float(grid.lat_2)],
        "earth_radius": float(grid.earth_radius),
        "dx": float(grid.dx),
        "dy": float(grid.dy),
        "ll_x": float(np.min(x)),
        "ll_y": float(np.min(y)),
        # The first stored GRIB point is the south-west corner only for south-first, west-first grids (HRRR).
        "ll_lon": float(grid.lon_first) if grid.row0_is_south and not grid.i_scans_negatively else None,
        "ll_lat": float(grid.lat_first) if grid.row0_is_south and not grid.i_scans_negatively else None,
        "location_index": "j * ni + i, with i west to east and j south to north",
    }


# ---------------------------------------------------------------------------------------------
# Block reduction
# ---------------------------------------------------------------------------------------------
def _is_missing(a: np.ndarray, fill) -> np.ndarray:
    if np.issubdtype(a.dtype, np.floating):
        return np.isnan(a) if fill is None or (isinstance(fill, float) and math.isnan(fill)) else (np.isnan(a) | (a == fill))
    return np.zeros(a.shape, dtype=bool) if fill is None else (a == fill)


def block_reduce(a: np.ndarray, method: Aggregation, fill_value=None) -> np.ndarray:
    """Reduce a 2-D array by 2 in both dimensions (trailing odd row/column dropped, as topozarr does).

    Fill values (and NaN for floats) are ignored; a block with no valid cell becomes the fill value.

    Args:
        a: ``(ny, nx)`` array.
        method: ``mean``, ``max``, ``mode`` or ``nearest``.
        fill_value: The array's fill value.

    Returns:
        ``(ny // 2, nx // 2)`` array of the same dtype.
    """
    ny, nx = a.shape[0] // 2, a.shape[1] // 2
    b = a[: 2 * ny, : 2 * nx].reshape(ny, 2, nx, 2)
    if method == "nearest":
        return np.ascontiguousarray(b[:, 0, :, 0])
    miss = _is_missing(b, fill_value)
    valid_n = 4 - miss.sum(axis=(1, 3))
    empty = valid_n == 0
    fill_out = fill_value if fill_value is not None else (np.nan if np.issubdtype(a.dtype, np.floating) else 0)
    if method == "mean":
        s = np.where(miss, 0.0, b.astype(np.float64)).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / valid_n
        if np.issubdtype(a.dtype, np.integer):
            out = np.rint(out)
        out = np.where(empty, fill_out, out)
        return out.astype(a.dtype)
    if method == "max":
        low = -np.inf if np.issubdtype(a.dtype, np.floating) else np.iinfo(a.dtype).min
        out = np.where(miss, low, b).max(axis=(1, 3))
        return np.where(empty, fill_out, out).astype(a.dtype)
    if method == "mode":
        values = np.unique(b[~miss])
        if values.size == 0:
            return np.full((ny, nx), fill_out, dtype=a.dtype)
        counts = np.stack([((b == v) & ~miss).sum(axis=(1, 3)) for v in values])  # ascending values -> ties to smallest
        out = values[np.argmax(counts, axis=0)]
        return np.where(empty, fill_out, out).astype(a.dtype)
    raise ValueError(f"unknown aggregation {method!r}")


# ---------------------------------------------------------------------------------------------
# Gridded multiscale store
# ---------------------------------------------------------------------------------------------
def _open_root(path, mode: str):
    return zarr.open_group(path, mode=mode, zarr_format=3)


def create_ptype_store(
    path,
    grid,
    forecast_hours: Sequence[int],
    variables: Mapping[str, VariableSpec] | None = None,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    n_levels: int = DEFAULT_LEVELS,
    chunks: tuple[int, int] = DEFAULT_GRID_CHUNK,
    init_time: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    """Create the gridded multiscale store of one run with every forecast-hour slot allocated.

    Writes only metadata and the small coordinate arrays (``x``, ``y``, ``forecast_hour``,
    ``spatial_ref`` in every level group); data chunks are written later by :func:`write_ptype_hour`.

    Args:
        path: Local path or any zarr ``StoreLike`` (e.g. ``zarr.storage.ObjectStore`` on S3).
        grid: :class:`ptype.hrrr_fast.reader.GridMetadata` of the HRRR grid (defines x/y, CRS, row order).
        forecast_hours: Forecast hours of the run, e.g. ``range(0, 19)``; the index into this list is
            the ``fh_index`` of :func:`write_ptype_hour`.
        variables: Name -> :class:`VariableSpec`; defaults to :func:`default_ptype_variables`. Pass a
            merged dict to add fields (e.g. Bourgouin probabilities or class with ``aggregation="mode"``).
        encoding: Optional per-variable overrides of :class:`VariableSpec` fields, e.g.
            ``{"ML_rain": {"filters": [], "compressors": [ZstdCodec(level=9)]}}``.
        n_levels: Pyramid levels including the native 3 km level.
        chunks: Spatial chunk ``(cy, cx)``; each level uses ``min(level shape, chunk)``. The
            forecast_hour chunk is always 1 (required for concurrent per-hour writes).
        init_time: ISO initialization time, stored in the root ``hrrr_ptype`` attribute.
        attrs: Extra entries for the root ``hrrr_ptype`` attribute.
        overwrite: Replace an existing store at ``path``.
    """
    specs = dict(default_ptype_variables() if variables is None else variables)
    for name, over in (encoding or {}).items():
        if name not in specs:
            raise KeyError(f"encoding given for unknown variable {name!r}")
        specs[name] = replace(specs[name], **over)
    fh = np.asarray(list(forecast_hours), dtype=np.int32)
    shapes = coarsen_shapes(grid.ny, grid.nx, n_levels)
    if min(shapes[-1]) < 2:
        raise ValueError(f"{n_levels} levels is too many for a {grid.ny} x {grid.nx} grid")
    crs = _crs_attrs(grid)
    coords = [level_coords(grid, lvl) for lvl in range(n_levels)]
    transforms = [_transform(x, y) for x, y in coords]
    root_method = "mean"
    layout = []
    for lvl in range(n_levels):
        entry: dict[str, Any] = {"asset": str(lvl)}
        if lvl > 0:
            entry |= {"derived_from": str(lvl - 1), "transform": {"scale": [1.0, 2.0, 2.0], "translation": [0.0, 0.0, 0.0]}, "resampling_method": root_method}
        entry |= {"spatial:transform": transforms[lvl], "spatial:shape": list(shapes[lvl])}
        layout.append(entry)
    root_attrs = {
        "zarr_conventions": [_CONVENTIONS["multiscales"], _CONVENTIONS["proj"], _CONVENTIONS["spatial"]],
        **crs,
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:transform": transforms[0],
        "spatial:bbox": _bbox(transforms[0], *shapes[0]),
        "spatial:shape": list(shapes[0]),
        "multiscales": {"layout": layout, "resampling_method": root_method},
        "hrrr_ptype": {
            "init_time": init_time,
            "forecast_hours": fh.tolist(),
            "variables": {k: {"resampling_method": s.aggregation, "dtype": s.dtype} for k, s in specs.items()},
            "row_order": "south_to_north" if grid.row0_is_south else "north_to_south",
            **(dict(attrs) if attrs else {}),
        },
    }
    root = _open_root(path, "w" if overwrite else "w-")
    root.attrs.update(root_attrs)
    grid_block = sounding_grid_attrs(grid)
    for lvl in range(n_levels):
        ny_l, nx_l = shapes[lvl]
        x, y = coords[lvl]
        g = root.create_group(
            str(lvl),
            attributes={
                "grid": grid_block,
                "coordinates": "spatial_ref",
                "zarr_conventions": [_CONVENTIONS["proj"], _CONVENTIONS["spatial"]],
                **crs,
                "spatial:dimensions": ["y", "x"],
                "spatial:registration": "pixel",
                "spatial:transform": transforms[lvl],
                "spatial:shape": [ny_l, nx_l],
            },
        )
        coord_codecs = {"compressors": [ZstdCodec(level=DEFAULT_ZSTD_LEVEL)]}
        g.create_array("x", data=x, dimension_names=["x"], attributes={"units": "m", "standard_name": "projection_x_coordinate"}, **coord_codecs)
        g.create_array("y", data=y, dimension_names=["y"], attributes={"units": "m", "standard_name": "projection_y_coordinate"}, **coord_codecs)
        g.create_array("forecast_hour", data=fh, dimension_names=["forecast_hour"], attributes={"units": "hours"}, **coord_codecs)
        sr = g.create_array("spatial_ref", shape=(), dtype="int64", fill_value=0, attributes={"crs_wkt": crs["proj:wkt2"]}, **coord_codecs)
        sr[...] = 0
        cy, cx = min(chunks[0], ny_l), min(chunks[1], nx_l)
        for name, s in specs.items():
            g.create_array(
                name,
                shape=(len(fh), ny_l, nx_l),
                chunks=(1, cy, cx),
                dtype=s.dtype,
                fill_value=s.fill_value,
                filters=list(s.filters),
                compressors=list(s.compressors),
                dimension_names=["forecast_hour", "y", "x"],
                attributes={"coordinates": "spatial_ref", "resampling_method": s.aggregation, **dict(s.attrs)},
            )


def write_ptype_hour(path, fh_index: int, fields: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Write one forecast hour of any subset of the store's variables at every pyramid level.

    Only chunks ``<level>/<name>/c/<fh_index>/...`` are written, each completely, so processes writing
    different ``fh_index`` values of the same store never touch the same object (module docstring).

    Args:
        path: The store created by :func:`create_ptype_store`.
        fh_index: Index into the store's ``forecast_hour`` coordinate.
        fields: Name -> ``(ny, nx)`` array in the reader's native row order (the order the store's ``y``
            coordinate describes). Values are cast to each array's dtype; boolean masks become 0/1.

    Returns:
        ``{name: number of levels written}``.

    Raises:
        KeyError: For a name that is not in the store.
        ValueError: For a wrong shape or an out-of-range ``fh_index``.
    """
    root = _open_root(path, "r+")
    layout = root.attrs["multiscales"]["layout"]
    ny, nx = layout[0]["spatial:shape"]
    written: dict[str, int] = {}
    for name, data in fields.items():
        a0 = root[f"0/{name}"]
        if not isinstance(a0, zarr.Array):
            raise KeyError(name)
        if not 0 <= fh_index < a0.shape[0]:
            raise ValueError(f"fh_index {fh_index} outside 0..{a0.shape[0] - 1}")
        arr = np.asarray(data)
        if arr.shape != (ny, nx):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(ny, nx)}")
        method = a0.attrs.get("resampling_method", "mean")
        fill = a0.fill_value
        cur = arr.astype(a0.dtype, copy=False)
        for lvl, entry in enumerate(layout):
            if lvl > 0:
                cur = block_reduce(cur, method, fill)
            a = a0 if lvl == 0 else root[f"{entry['asset']}/{name}"]
            a[fh_index] = cur
        written[name] = len(layout)
    return written


# ---------------------------------------------------------------------------------------------
# Sounding store
# ---------------------------------------------------------------------------------------------
def default_sounding_encoding() -> tuple[list, list]:
    """``(filters, compressors)`` of the sounding array: 0.01-unit int32, transposed, blosc-zstd bitshuffle.

    Why (numbers from benchmarks/results/compression_local.md, one real hour): lossless float32 + zstd
    only reaches 738 MB/hour (the low mantissa bits of interpolated values are noise). Rounding to 0.01
    in every unit (degC, m/s, hPa; max error 0.005) as int32 lets the bit shuffle put the constant high
    bits together, and transposing each ``(1, 1024, 105)`` chunk to variable-major makes neighbouring
    bytes come from neighbouring grid columns, which are smooth: 220 MB/hour, and the network changes
    the dominant class in 0.007 % of columns. One scale must serve every unit because ``scale_offset``
    takes a scalar, and p * 100 does not fit int16, hence int32 (``int16`` with step 0.1 is the smaller
    alternative: 134 MB/hour, class change in 0.06 % of columns).
    """
    return [*quantized_float(0.01, "int32"), TransposeCodec(order=(0, 2, 1))], [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)]


def create_sounding_store(
    path,
    grid,
    forecast_hours: Sequence[int],
    *,
    heights_m: Sequence[float],
    variable_groups: Sequence[str],
    init_time: str | None = None,
    location_chunk: int = 1024,
    sharding: bool = True,
    filters: Sequence[Any] | None = None,
    compressors: Sequence[Any] | None = None,
    overwrite: bool = True,
) -> None:
    """Create the per-run sounding store with every forecast-hour slot allocated.

    Args:
        path: Local path or zarr ``StoreLike``.
        grid: :class:`ptype.hrrr_fast.reader.GridMetadata`.
        forecast_hours: Forecast hours of the run.
        heights_m: Heights above ground of each level inside a variable group.
        variable_groups: Group names in storage order (``columns.SOUNDING_VARIABLES``).
        init_time: ISO initialization time (also written as a 1-element ``init_time`` array, seconds
            since the epoch, so the store describes itself).
        location_chunk: Locations per inner chunk; ~1024 keeps one click to one small range read.
        sharding: Put all inner chunks of one hour into one shard object (see module docstring).
        filters: Array-to-array codecs; ``None`` uses :func:`default_sounding_encoding`.
        compressors: Bytes-to-bytes codecs of each inner chunk; ``None`` uses :func:`default_sounding_encoding`.
        overwrite: Replace an existing store.
    """
    d_filters, d_compressors = default_sounding_encoding()
    filters = d_filters if filters is None else filters
    compressors = d_compressors if compressors is None else compressors
    n_loc = grid.nx * grid.ny
    n_var = len(heights_m) * len(variable_groups)
    fh = np.asarray(list(forecast_hours), dtype=np.int32)
    n_chunks = math.ceil(n_loc / location_chunk)
    lon_0 = float(grid.lon_0)
    lat_std = float(grid.lat_1)
    root = _open_root(path, "w" if overwrite else "w-")
    root.attrs.update(
        {
            "grid": sounding_grid_attrs(grid),
            "heights_m": [float(h) if not float(h).is_integer() else int(h) for h in heights_m],
            "variable_groups": list(variable_groups),
            "variable_layout": "concatenate(variable_groups), each ordered surface to top",
            "init_time": init_time,
            "init_time_units": "seconds since 1970-01-01T00:00:00Z",
            "wind_frame": "earth_relative",
            "wind_rotation": {
                "note": "u_ms/v_ms are EARTH-relative (east/north). The network takes GRID-relative winds; rotate back before running it.",
                "lon_0": lon_0,
                "lat_std": lat_std,
                "cone_constant": math.sin(math.radians(lat_std)),
                "alpha": "alpha = cone_constant * radians(wrap180(lon - lon_0)), lon = longitude of the grid cell",
                "earth_to_grid": "u_grid = cos(alpha) * u_earth - sin(alpha) * v_earth; v_grid = sin(alpha) * u_earth + cos(alpha) * v_earth",
                "grid_to_earth": "u_earth = cos(alpha) * u_grid + sin(alpha) * v_grid; v_earth = -sin(alpha) * u_grid + cos(alpha) * v_grid",
            },
        }
    )
    shards = (1, n_chunks * location_chunk, n_var) if sharding else None
    root.create_array(
        "sounding",
        shape=(len(fh), n_loc, n_var),
        chunks=(1, location_chunk, n_var),
        shards=shards,
        dtype="float32",
        fill_value=np.nan,
        filters=list(filters),
        compressors=list(compressors),
        dimension_names=["forecast_hour", "location", "variable"],
    )
    root.create_array("forecast_hour", data=fh, dimension_names=["forecast_hour"])
    epoch = [] if init_time is None else [int(np.datetime64(init_time.rstrip("Z"), "s").astype(np.int64))]
    root.create_array("init_time", data=np.asarray(epoch, dtype=np.int64), dimension_names=["init_time"])


def write_sounding_hour(path, fh_index: int, sounding: np.ndarray) -> None:
    """Write one forecast hour of soundings (one shard object, or its inner chunks if unsharded).

    Args:
        path: Store created by :func:`create_sounding_store`.
        fh_index: Index into the store's ``forecast_hour``.
        sounding: ``(n_locations, n_variables)`` float32 with ``location = j * ni + i`` (j south to
            north) and EARTH-relative winds. ``columns.build_soundings(...).reshape(n, -1)`` of a
            south-first HRRR grid already has this order.
    """
    arr = zarr.open_array(path, path="sounding", mode="r+", zarr_format=3)
    s = np.asarray(sounding)
    if s.shape != arr.shape[1:]:
        raise ValueError(f"sounding has shape {s.shape}, expected {arr.shape[1:]}")
    if not 0 <= fh_index < arr.shape[0]:
        raise ValueError(f"fh_index {fh_index} outside 0..{arr.shape[0] - 1}")
    arr[fh_index] = s.astype(np.float32, copy=False)
