"""Tests for ptype.hrrr_fast.writers: round trips, concurrent per-hour writes, site-facing metadata and read cost.

The "site read" tests do not use zarr to read: they replay what the web app's zarrita does (fetch the
JSON documents, compute the chunk keys of one forecast hour, read a shard index with a suffix read and
one inner chunk with a range read) against the files on disk, count objects/bytes, and decode the
bytes with numcodecs directly. That checks both the request pattern and the on-disk representation
independently of zarr-python's reader.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("cast_value_rs")
pyproj = pytest.importorskip("pyproj")

from ptype.hrrr_fast import writers
from ptype.hrrr_fast.reader import GridMetadata

PROJ4 = "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +R=6371229.0 +units=m +no_defs"
NY, NX = 100, 130
HOURS = list(range(8))
HEIGHTS = [250 * k for k in range(21)]
GROUPS = ["temperature_c", "dewpoint_c", "u_ms", "v_ms", "pressure_hpa"]
ML = ["ML_rain", "ML_snow", "ML_icep", "ML_frzr", "ML_u"]
FLAGS = ["precip_mask", "CRAIN", "CSNOW", "CICEP", "CFRZR"]


def make_grid(ny: int = NY, nx: int = NX) -> GridMetadata:
    """A small south-first Lambert grid with the HRRR projection."""
    x0, y0 = -300000.0, -200000.0
    lon, lat = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(PROJ4), "EPSG:4326", always_xy=True).transform(x0, y0)
    return GridMetadata(
        nx=nx, ny=ny, dx=3000.0, dy=3000.0, lon_0=-97.5, lat_0=38.5, lat_1=38.5, lat_2=38.5, earth_radius=6371229.0, shape_of_the_earth=6,
        lat_first=float(lat), lon_first=float(lon), x_first=x0, y_first=y0, i_scans_negatively=0, j_scans_positively=1, j_points_consecutive=0,
        row0_is_south=True, proj4=PROJ4,
    )  # fmt: skip


def hour_fields(fh: int, ny: int = NY, nx: int = NX) -> dict[str, np.ndarray]:
    """Deterministic, spatially smooth-ish fields for one hour (so every process can regenerate them)."""
    rng = np.random.default_rng(1000 + fh)
    logits = rng.normal(size=(4, ny, nx)).astype(np.float64)
    p = np.exp(logits) / np.exp(logits).sum(axis=0)
    out = {k: p[i].astype(np.float32) for i, k in enumerate(ML[:4])}
    out["ML_u"] = rng.uniform(0.01, 0.99, size=(ny, nx)).astype(np.float32)
    flags = rng.random((4, ny, nx)) < 0.2
    for k, f in zip(FLAGS[1:], flags, strict=True):
        out[k] = f.astype(np.uint8)
    out["precip_mask"] = flags.any(axis=0).astype(np.uint8)
    out["prate"] = rng.gamma(0.5, 1e-4, size=(ny, nx)).astype(np.float32)
    return out


def hour_sounding(fh: int, ny: int = NY, nx: int = NX) -> np.ndarray:
    """``(ny*nx, 105)`` sounding with realistic magnitudes (T, Td degC; u, v m/s; p hPa)."""
    rng = np.random.default_rng(2000 + fh)
    n = ny * nx
    t = rng.uniform(-40, 30, size=(n, 21))
    td = t - rng.uniform(0, 20, size=(n, 21))
    u = rng.normal(0, 15, size=(n, 21))
    v = rng.normal(0, 15, size=(n, 21))
    p = np.linspace(1000, 550, 21)[None, :] + rng.normal(0, 5, size=(n, 21))
    return np.concatenate([t, td, u, v, p], axis=1).astype(np.float32)


def _create(root: Path, extra: dict | None = None, n_levels: int = 3) -> tuple[Path, Path]:
    grid = make_grid()
    variables = writers.default_ptype_variables() | (extra or {})
    g = root / "grid.zarr"
    s = root / "sounding.zarr"
    writers.create_ptype_store(g, grid, HOURS, variables=variables, n_levels=n_levels, init_time="2023-12-26T00:00Z")
    writers.create_sounding_store(s, grid, HOURS, heights_m=HEIGHTS, variable_groups=GROUPS, init_time="2023-12-26T00:00Z")
    return g, s


# ---------------------------------------------------------------------------------------------
# Block reduction
# ---------------------------------------------------------------------------------------------
def test_block_reduce_methods():
    a = np.array([[1, 1, 2, 3, 9], [1, 0, 2, 2, 9], [5, 5, 4, 4, 9]], dtype=np.uint8)  # odd row/col dropped
    np.testing.assert_array_equal(writers.block_reduce(a, "max", 255), [[1, 3]])
    np.testing.assert_array_equal(writers.block_reduce(a, "mode", 255), [[1, 2]])
    np.testing.assert_array_equal(writers.block_reduce(a, "nearest", 255), [[1, 2]])
    np.testing.assert_array_equal(writers.block_reduce(a, "mean", 255), [[1, 2]])  # 0.75 -> 1, 2.25 -> 2
    f = np.array([[0.2, np.nan], [0.4, np.nan]], dtype=np.float32)
    assert writers.block_reduce(f, "mean")[0, 0] == pytest.approx(0.3)
    g = np.full((2, 2), np.nan, dtype=np.float32)
    assert np.isnan(writers.block_reduce(g, "mean")[0, 0])
    m = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    assert writers.block_reduce(m, "max", 255)[0, 0] == 255  # all-missing block stays missing
    t = np.array([[3, 3], [1, 1]], dtype=np.uint8)
    assert writers.block_reduce(t, "mode", 255)[0, 0] == 1  # tie -> smallest code


def test_coarsen_shapes_match_topozarr_floor():
    assert writers.coarsen_shapes(1059, 1799, 4) == [(1059, 1799), (529, 899), (264, 449), (132, 224)]


# ---------------------------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------------------------
def test_grid_round_trip_all_levels_and_missing_hours(tmp_path):
    extra = {
        "bourgouin_class": writers.VariableSpec(dtype="uint8", fill_value=255, aggregation="mode", attrs={"long_name": "class"}),
        "bourgouin_rain": writers.VariableSpec(attrs={"units": "1"}),
    }
    g, _ = _create(tmp_path, extra)
    f = hour_fields(3)
    rng = np.random.default_rng(7)
    f["bourgouin_class"] = rng.integers(0, 4, size=(NY, NX)).astype(np.uint8)
    f["bourgouin_rain"] = rng.random((NY, NX)).astype(np.float32)
    assert writers.write_ptype_hour(g, 3, f) == {k: 3 for k in f}
    root = zarr.open_group(g, mode="r")
    layout = root.attrs["multiscales"]["layout"]
    for name, data in f.items():
        a0 = root[f"0/{name}"]
        spec = (writers.default_ptype_variables() | extra)[name]
        tol = 0.5 / 254 + 1e-6 if spec.filters else 0.0
        np.testing.assert_allclose(a0[3].astype(np.float64), data.astype(np.float64), atol=tol, rtol=0)
        cur = data.astype(a0.dtype)
        for lvl in range(1, len(layout)):
            cur = writers.block_reduce(cur, a0.attrs["resampling_method"], a0.fill_value)
            np.testing.assert_allclose(root[f"{lvl}/{name}"][3].astype(np.float64), cur.astype(np.float64), atol=tol, rtol=0)
        other = a0[5]
        if np.issubdtype(a0.dtype, np.floating):
            assert np.isnan(other).all()
        else:
            assert (other == a0.fill_value).all()
    # probabilities still sum to ~1 after averaging (mean is linear) at the coarsest level, up to quantisation
    s = sum(root[f"{len(layout) - 1}/{k}"][3].astype(np.float64) for k in ML[:4])
    assert np.max(np.abs(s - 1)) < 4 * 0.5 / 254 + 1e-6
    # the precip mask aggregates with "any": a coarse cell is set iff any fine cell is set
    m = f["precip_mask"][: NY // 2 * 2, : NX // 2 * 2].reshape(NY // 2, 2, NX // 2, 2).max(axis=(1, 3))
    np.testing.assert_array_equal(root["1/precip_mask"][3], m)


def test_quantized_float_keeps_nan(tmp_path):
    g, _ = _create(tmp_path)
    f = {"ML_rain": hour_fields(0)["ML_rain"].copy()}
    f["ML_rain"][:10] = np.nan
    f["ML_rain"][10, 0] = 1.0
    writers.write_ptype_hour(g, 0, f)
    back = zarr.open_group(g, mode="r")["0/ML_rain"][0]
    assert np.isnan(back[:10]).all() and not np.isnan(back[10:]).any()
    assert back[10, 0] == 1.0


def test_sounding_round_trip_and_location_order(tmp_path):
    _, s = _create(tmp_path)
    snd = hour_sounding(2)
    writers.write_sounding_hour(s, 2, snd)
    arr = zarr.open_array(s, path="sounding", mode="r")
    back = arr[2]
    assert np.max(np.abs(back.astype(np.float64) - snd.astype(np.float64))) <= 0.05 + 1e-4  # 0.1-unit int16 default
    assert np.isnan(arr[1]).all()
    # location = j * ni + i with j south -> north: same index as the row-major (y, x) grid of a south-first grid
    attrs = zarr.open_group(s, mode="r").attrs
    j, i = 37, 101
    np.testing.assert_allclose(arr[2, j * attrs["grid"]["ni"] + i], snd.reshape(NY, NX, -1)[j, i], atol=0.0501)
    # sharded: exactly one object per written hour
    objs = sorted(p.relative_to(s).as_posix() for p in (s / "sounding" / "c").rglob("*") if p.is_file())
    assert objs == ["sounding/c/2/0/0"]


# ---------------------------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------------------------
def _worker(args):
    g, s, hours = args
    for fh in hours:
        writers.write_ptype_hour(g, fh, hour_fields(fh))
        writers.write_sounding_hour(s, fh, hour_sounding(fh))
    return hours


def test_concurrent_processes_write_different_hours(tmp_path):
    g, s = _create(tmp_path)
    files_before = {p.relative_to(tmp_path) for p in tmp_path.rglob("zarr.json")}
    meta_before = {p: p.read_bytes() for p in tmp_path.rglob("zarr.json")}
    # Interleave hours across 4 processes; nobody holds a lock.
    chunks = [HOURS[k::4] for k in range(4)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        done = pool.map(_worker, [(g, s, h) for h in chunks])
    assert sorted(h for d in done for h in d) == HOURS
    root = zarr.open_group(g, mode="r")
    snd = zarr.open_array(s, path="sounding", mode="r")
    for fh in HOURS:
        f = hour_fields(fh)
        for name in ML:
            np.testing.assert_allclose(root[f"0/{name}"][fh], f[name], atol=0.5 / 254 + 1e-6, rtol=0)
        for name in FLAGS:
            np.testing.assert_array_equal(root[f"0/{name}"][fh], f[name])
            np.testing.assert_array_equal(root[f"2/{name}"][fh], writers.block_reduce(writers.block_reduce(f[name], "max", 255), "max", 255))
        np.testing.assert_allclose(snd[fh], hour_sounding(fh), atol=0.0501, rtol=0)
    # metadata documents were written once by create_* and never touched by the writers
    assert {p.relative_to(tmp_path) for p in tmp_path.rglob("zarr.json")} == files_before
    assert all(p.read_bytes() == b for p, b in meta_before.items())
    assert not list(tmp_path.rglob("*.partial"))


# ---------------------------------------------------------------------------------------------
# Site-facing metadata
# ---------------------------------------------------------------------------------------------
def test_root_attrs_are_what_the_site_reads(tmp_path):
    g, s = _create(tmp_path)
    grid = make_grid()
    attrs = json.loads((g / "zarr.json").read_text())["attributes"]
    assert "+proj=lcc" in attrs["proj:code"]
    layout = attrs["multiscales"]["layout"]
    for lvl, entry in enumerate(layout):
        assert entry["asset"] == str(lvl)
        # src/data/ptype.ts: x = a*col + c, y = e*row + f at cell edges; llX = c + a/2, llY = f + dy/2 (e > 0)
        a, _, c, _, e, f = entry["spatial:transform"]
        nj, ni = entry["spatial:shape"]
        x, y = writers.level_coords(grid, lvl)
        assert (nj, ni) == (len(y), len(x))
        assert e > 0  # rows stored south to north: the site will not flip
        assert c + a / 2 == pytest.approx(x[0]) and f + abs(e) / 2 == pytest.approx(y[0])
        assert a == pytest.approx(3000.0 * 2**lvl)
        assert (g / str(lvl) / "forecast_hour" / "zarr.json").exists()
        fh = zarr.open_array(g, path=f"{lvl}/forecast_hour", mode="r")[:]
        np.testing.assert_array_equal(fh, HOURS)
    sa = json.loads((s / "zarr.json").read_text())["attributes"]
    for key in ("ni", "nj", "dx", "dy", "ll_x", "ll_y", "lon_0", "lat_0", "lat_std", "earth_radius"):
        assert key in sa["grid"]
    assert sa["grid"]["ni"] == NX and sa["grid"]["nj"] == NY
    assert (sa["grid"]["ll_x"], sa["grid"]["ll_y"]) == (grid.x_first, grid.y_first)
    assert sa["heights_m"] == HEIGHTS and sa["variable_groups"] == GROUPS
    assert sa["wind_frame"] == "earth_relative" and sa["wind_rotation"]["lon_0"] == -97.5 and sa["wind_rotation"]["lat_std"] == 38.5


def test_metadata_identical_to_topozarr(tmp_path):
    """Root and level attributes equal what topozarr writes for the same grid (except our extra blocks)."""
    pytest.importorskip("topozarr")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xproj")
    from topozarr import create_pyramid

    grid = make_grid()
    x, y = grid.xy_coords()
    ds = xr.Dataset({"ML_rain": (("forecast_hour", "y", "x"), np.zeros((2, NY, NX), np.float32))}, coords={"x": x, "y": y, "forecast_hour": np.array([0, 1], "int32")})
    ds = ds.proj.assign_crs(spatial_ref=pyproj.CRS.from_proj4(PROJ4))
    create_pyramid(ds, levels=3, method="mean").write(str(tmp_path / "topo.zarr"))
    g, _ = _create(tmp_path)
    ref = json.loads((tmp_path / "topo.zarr" / "zarr.json").read_text())["attributes"]
    ours = json.loads((g / "zarr.json").read_text())["attributes"]
    assert set(ours) - set(ref) == {"hrrr_ptype"}
    assert {k: ours[k] for k in ref} == ref
    for lvl in range(3):
        r = json.loads((tmp_path / "topo.zarr" / str(lvl) / "zarr.json").read_text())["attributes"]
        o = json.loads((g / str(lvl) / "zarr.json").read_text())["attributes"]
        assert set(o) - set(r) == {"grid"}
        assert {k: o[k] for k in r} == r
        for c in ("x", "y"):
            np.testing.assert_array_equal(zarr.open_array(tmp_path / "topo.zarr", path=f"{lvl}/{c}")[:], zarr.open_array(g, path=f"{lvl}/{c}")[:])


# ---------------------------------------------------------------------------------------------
# Site read pattern, without JavaScript
# ---------------------------------------------------------------------------------------------
class SiteMimic:
    """Plays the requests zarrita makes against a local store and counts objects and bytes."""

    def __init__(self, root: Path):
        self.root = root
        self.log: list[tuple[str, str, int]] = []

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        if not p.is_file():
            self.log.append(("GET404", key, 0))
            return None
        b = p.read_bytes()
        self.log.append(("GET", key, len(b)))
        return b

    def get_range(self, key: str, offset: int, length: int) -> bytes:
        with open(self.root / key, "rb") as f:
            f.seek(offset)
            b = f.read(length)
        self.log.append(("RANGE", key, len(b)))
        return b

    def get_suffix(self, key: str, n: int) -> bytes:
        size = (self.root / key).stat().st_size
        return self.get_range(key, size - n, n)

    def json(self, key: str) -> dict:
        return json.loads(self.get(key))


def _decode_bytes(meta: dict, raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Independent decoder for the codec chains the writers use (zstd/blosc, scale_offset+cast_value, transpose)."""
    from numcodecs import Blosc, Zstd

    codecs = meta["codecs"]
    if codecs[0]["name"] == "sharding_indexed":
        codecs = codecs[0]["configuration"]["codecs"]
    for c in reversed(codecs):
        if c["name"] == "zstd":
            raw = Zstd().decode(raw)
        elif c["name"] == "blosc":
            raw = Blosc().decode(raw)
    dtype = np.dtype(meta["data_type"])
    order = None
    scale = None
    for c in codecs:
        if c["name"] == "cast_value":
            dtype = np.dtype(c["configuration"]["data_type"])
        elif c["name"] == "transpose":
            order = c["configuration"]["order"]
        elif c["name"] == "scale_offset":
            scale = c["configuration"]["scale"]
    stored_shape = tuple(shape[k] for k in order) if order else shape
    a = np.frombuffer(raw, dtype=dtype).reshape(stored_shape)
    if order:
        a = a.transpose(np.argsort(order))
    if scale is not None:
        sentinel = np.iinfo(dtype).max if np.iinfo(dtype).min == 0 else np.iinfo(dtype).min
        a = np.where(a == sentinel, np.nan, a.astype(np.float64) / scale).astype(np.float32)
    return a


def test_site_read_one_forecast_hour_and_one_sounding(tmp_path):
    g, s = _create(tmp_path)
    for fh in (1, 4):
        writers.write_ptype_hour(g, fh, hour_fields(fh))
        writers.write_sounding_hour(s, fh, hour_sounding(fh))

    # --- map frame: open root -> pick a level -> forecast_hour -> 5 arrays at one hour (src/data/ptype.ts)
    for level in (0, 2):
        site = SiteMimic(g)
        attrs = site.json("zarr.json")["attributes"]
        entry = attrs["multiscales"]["layout"][level]
        lvl = entry["asset"]
        fh_meta = site.json(f"{lvl}/forecast_hour/zarr.json")
        fh = _decode_bytes(fh_meta, site.get(f"{lvl}/forecast_hour/c/0"), (len(HOURS),))
        fh_index = int(np.flatnonzero(fh == 4)[0])
        nj, ni = entry["spatial:shape"]
        expected = hour_fields(4)
        for name in ML:
            meta = site.json(f"{lvl}/{name}/zarr.json")
            _, cy, cx = meta["chunk_grid"]["configuration"]["chunk_shape"]
            assert (cy, cx) == (nj, ni)  # default chunk: one object per level
            out = np.empty((nj, ni), np.float32)
            for j in range(math.ceil(nj / cy)):
                for i in range(math.ceil(ni / cx)):
                    raw = site.get(f"{lvl}/{name}/c/{fh_index}/{j}/{i}")
                    out[j * cy : (j + 1) * cy, i * cx : (i + 1) * cx] = _decode_bytes(meta, raw, (1, cy, cx))[0]
            ref = expected[name]
            for _ in range(level):
                ref = writers.block_reduce(ref, "mean")
            np.testing.assert_allclose(out, ref, atol=0.5 / 254 + 1e-6, rtol=0)
        data = [e for e in site.log if "/c/" in e[1] and "forecast_hour" not in e[1]]
        assert len(data) == 5  # one data object per variable per frame
        assert len(site.log) == 1 + 2 + 2 * 5  # root, forecast_hour (doc + chunk), (doc + chunk) per variable

    # --- sounding click: root attrs -> location -> shard index (suffix) -> one inner chunk (src/data/soundings.ts)
    site = SiteMimic(s)
    attrs = site.json("zarr.json")["attributes"]
    meta = site.json("sounding/zarr.json")
    shard = meta["codecs"][0]["configuration"]
    _, n_loc_chunk, n_var = shard["chunk_shape"]
    n_inner = meta["chunk_grid"]["configuration"]["chunk_shape"][1] // n_loc_chunk
    j, i = 61, 17
    loc = j * attrs["grid"]["ni"] + i
    key = "sounding/c/4/0/0"
    index_raw = site.get_suffix(key, n_inner * 16 + 4)  # uint64 (offset, nbytes) pairs + crc32c
    index = np.frombuffer(index_raw[:-4], dtype="<u8").reshape(n_inner, 2)
    off, nbytes = index[loc // n_loc_chunk]
    chunk = _decode_bytes(meta, site.get_range(key, int(off), int(nbytes)), (1, n_loc_chunk, n_var))[0]
    np.testing.assert_allclose(chunk[loc % n_loc_chunk], hour_sounding(4)[loc], atol=0.0501, rtol=0)
    ranges = [e for e in site.log if e[0] == "RANGE"]
    assert len(ranges) == 2  # index + one inner chunk
    assert ranges[1][2] < (s / key).stat().st_size / 4  # a click reads a small part of the hour's shard
    assert ranges[1][2] < n_loc_chunk * n_var * 4  # and less than the raw float32 chunk


# ---------------------------------------------------------------------------------------------
# int16 range handling
# ---------------------------------------------------------------------------------------------
def test_sounding_int16_range_is_checked_not_clamped(tmp_path):
    _, s = _create(tmp_path)
    snd = hour_sounding(0)
    edge = snd.copy()
    edge[0, :3] = [3276.7, -3276.7, np.nan]  # the representable extremes and NaN are accepted
    writers.write_sounding_hour(s, 0, edge)
    back = zarr.open_array(s, path="sounding", mode="r")[0, 0, :3]
    np.testing.assert_allclose(back[:2], [3276.7, -3276.7], atol=1e-3)
    assert np.isnan(back[2])
    for bad in (101325.0, -3276.8, np.inf):  # pressure in Pa, one step past the sentinel side, infinity
        x = snd.copy()
        x[5, 104] = bad
        with pytest.raises(ValueError, match="representable|infinite"):
            writers.write_sounding_hour(s, 1, x)
    assert np.isnan(zarr.open_array(s, path="sounding", mode="r")[1]).all()  # nothing was written
    # realistic HRRR extremes in every group fit with margin: p 1085 hPa -> 10850, T -90 / 60 degC, wind 150 m/s
    ok = snd.copy()
    ok[:, 84:] = 1085.0
    ok[:, :21] = -90.0
    ok[:, 42:63] = 150.0
    writers.write_sounding_hour(s, 2, ok)


def test_probability_range_is_checked(tmp_path):
    g, _ = _create(tmp_path)
    f = {"ML_rain": hour_fields(0)["ML_rain"].copy()}
    f["ML_rain"][0, 0] = 1.0 + 1e-6  # rounds to step 254: fine
    writers.write_ptype_hour(g, 0, f)
    f["ML_rain"][0, 0] = 1.01  # step 256.5 -> would collide with / exceed the NaN sentinel 255
    with pytest.raises(ValueError, match="representable"):
        writers.write_ptype_hour(g, 1, f)


def test_selectable_sounding_encodings(tmp_path):
    grid = make_grid()
    snd = hour_sounding(0)
    for kind, tol in (("int32-0.01", 0.0051), ("lossless", 0.0)):
        p = tmp_path / f"{kind}.zarr"
        f, c = writers.sounding_encoding(kind)
        writers.create_sounding_store(p, grid, HOURS, heights_m=HEIGHTS, variable_groups=GROUPS, filters=f, compressors=c)
        writers.write_sounding_hour(p, 0, snd)
        np.testing.assert_allclose(zarr.open_array(p, path="sounding", mode="r")[0], snd, atol=tol, rtol=0)


# ---------------------------------------------------------------------------------------------
# Finalize (run-spanning shards)
# ---------------------------------------------------------------------------------------------
def _staged_values(g: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(g, mode="r")
    return {f"{lvl}/{k}": a[...] for lvl in root.group_keys() for k, a in root[lvl].arrays()}


def test_finalize_round_trip_idempotent_and_missing_hours(tmp_path):
    g, _ = _create(tmp_path)
    written = [0, 1, 2, 4, 5, 7]  # hours 3 and 6 never written
    for fh in written:
        writers.write_ptype_hour(g, fh, hour_fields(fh))
    before = _staged_values(g)
    root_before = json.loads((g / "zarr.json").read_text())["attributes"]
    levels_before = {lvl: json.loads((g / str(lvl) / "zarr.json").read_text())["attributes"] for lvl in range(3)}

    rep = writers.finalize_ptype_store(g)
    assert not rep["already_finalized"] and rep["missing_hours"] == [3, 6]
    assert all(v == [3, 6] for v in rep["missing_by_array"].values())
    assert rep["upload"] == [str(g)]
    assert not (tmp_path / "grid.zarr.staging").exists() and not (tmp_path / "grid.zarr.finalize-tmp").exists()
    after = _staged_values(g)
    assert before.keys() == after.keys()
    for k in before:
        np.testing.assert_array_equal(after[k], before[k])  # bit-identical, including NaN/255 for hours 3 and 6
    # site-facing metadata unchanged; only our own hrrr_ptype block records the layout
    root_after = json.loads((g / "zarr.json").read_text())["attributes"]
    assert {k: v for k, v in root_after.items() if k != "hrrr_ptype"} == {k: v for k, v in root_before.items() if k != "hrrr_ptype"}
    assert root_after["hrrr_ptype"]["layout"] == "finalized"
    for lvl in range(3):
        assert json.loads((g / str(lvl) / "zarr.json").read_text())["attributes"] == levels_before[lvl]
    # one data object per array per level, inner chunk = one hour, shard = all hours
    data_objs = [p for p in g.rglob("*") if p.is_file() and "c" in p.relative_to(g).parts and p.relative_to(g).parts[1] not in ("x", "y", "forecast_hour", "spatial_ref")]
    assert len(data_objs) == 3 * len(writers.default_ptype_variables())
    meta = json.loads((g / "0" / "ML_rain" / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [len(HOURS), NY, NX]
    assert meta["codecs"][0]["name"] == "sharding_indexed" and meta["codecs"][0]["configuration"]["chunk_shape"] == [1, NY, NX]
    assert rep["objects"] == len([p for p in g.rglob("*") if p.is_file()])

    # idempotent: a second call changes nothing
    stamp = {p: p.stat().st_mtime_ns for p in g.rglob("*") if p.is_file()}
    rep2 = writers.finalize_ptype_store(g)
    assert rep2["already_finalized"] and {p: p.stat().st_mtime_ns for p in g.rglob("*") if p.is_file()} == stamp
    with pytest.raises(ValueError, match="finalized"):
        writers.write_ptype_hour(g, 3, hour_fields(3))


def test_finalize_recovers_from_crashes(tmp_path):
    g, _ = _create(tmp_path)
    for fh in HOURS:
        writers.write_ptype_hour(g, fh, hour_fields(fh))
    before = _staged_values(g)
    # crash during the build: a stale, incomplete temp dir is discarded and rebuilt
    (tmp_path / "grid.zarr.finalize-tmp" / "junk").mkdir(parents=True)
    writers.finalize_ptype_store(g, remove_staging=False)
    assert (tmp_path / "grid.zarr.staging").exists()  # kept on request, sibling of the final store
    # crash between the two renames: final store sits in the temp name, path missing
    os_rename = __import__("os").rename
    os_rename(g, tmp_path / "grid.zarr.finalize-tmp")
    rep = writers.finalize_ptype_store(g)
    assert rep["already_finalized"] and not (tmp_path / "grid.zarr.staging").exists()
    after = _staged_values(g)
    for k in before:
        np.testing.assert_array_equal(after[k], before[k])


def test_concurrent_staging_then_finalize_and_site_read(tmp_path):
    g, s = _create(tmp_path)
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_worker, [(g, s, HOURS[k::4]) for k in range(4)])
    writers.finalize_ptype_store(g)
    # replay the site's frame read on the finalized store: per array, doc + shard-index suffix + one inner chunk
    for level, fh_value in ((0, 6), (2, 1)):
        site = SiteMimic(g)
        entry = site.json("zarr.json")["attributes"]["multiscales"]["layout"][level]
        lvl = entry["asset"]
        fh = _decode_bytes(site.json(f"{lvl}/forecast_hour/zarr.json"), site.get(f"{lvl}/forecast_hour/c/0"), (len(HOURS),))
        k = int(np.flatnonzero(fh == fh_value)[0])
        nj, ni = entry["spatial:shape"]
        for name in ML:
            meta = site.json(f"{lvl}/{name}/zarr.json")
            n_inner = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
            idx = np.frombuffer(site.get_suffix(f"{lvl}/{name}/c/0/0/0", n_inner * 16 + 4)[:-4], dtype="<u8").reshape(n_inner, 2)
            off, nb = idx[k]
            out = _decode_bytes(meta, site.get_range(f"{lvl}/{name}/c/0/0/0", int(off), int(nb)), (1, nj, ni))[0]
            ref = hour_fields(fh_value)[name]
            for _ in range(level):
                ref = writers.block_reduce(ref, "mean")
            np.testing.assert_allclose(out, ref, atol=0.5 / 254 + 1e-6, rtol=0)
        assert len([e for e in site.log if e[0] == "RANGE"]) == 2 * 5  # index + one inner chunk per array
        assert len(site.log) == 1 + 2 + 3 * 5


def test_check_representable_edge_cases_and_memory(tmp_path):
    import tracemalloc

    _, s = _create(tmp_path)
    arr = zarr.open_array(s, path="sounding", mode="r")
    writers.check_representable(arr, np.full((NY * NX, 105), np.nan, np.float32), "all-NaN")  # all NaN -> sentinel, fine
    writers.check_representable(arr, np.empty((0, 105), np.float32), "empty")
    x = hour_sounding(0)
    x[7, 3] = -np.inf
    with pytest.raises(ValueError, match="infinite"):
        writers.check_representable(arr, x, "sounding")
    x[7, 3] = np.nan
    x[8, 4] = -3277.0  # the minimum alone is out of range
    with pytest.raises(ValueError, match="representable"):
        writers.check_representable(arr, x, "sounding")
    # O(1) extra memory: no float64 copy or masks of the input (40 MB of float32 here)
    big = np.random.default_rng(0).uniform(-50, 50, size=(10_000_000,)).astype(np.float32)
    tracemalloc.start()
    writers.check_representable(arr, big, "big")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 1_000_000
