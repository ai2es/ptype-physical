"""Storage size, speed and error of candidate Zarr encodings for the HRRR ptype run stores, on one real hour.

Input: the directory written by ``benchmarks/make_sample_hour.py`` (``*.npy`` + ``meta.json``).
Every number is measured on that hour by writing it with ``ptype.hrrr_fast.writers`` under ``--work``
(outside the repo):

* stored bytes = sum of the sizes of the chunk/shard objects on disk (metadata reported separately),
* write time = ``write_ptype_hour`` / ``write_sounding_hour`` wall time,
* read time = Python read+decode the way the site reads (gridded: level 0, one hour, the 5 ML arrays;
  soundings: the whole hour, and a single location),
* max / mean absolute error against the input,
* for lossy SOUNDING encodings, the effect on the network: decoded soundings -> winds rotated back to
  grid-relative -> engine -> change in probabilities and in the dominant class (whole grid and inside
  ``precip_mask``), relative to the same pipeline on the lossless float32 store.

Sections: ``grid`` (encodings of ML_*, flags and prate), ``chunks`` (request/byte count per map frame for
candidate spatial chunkings), ``mask`` (bytes if values outside precip_mask were blanked) and
``sounding``. Results are printed as Markdown tables and saved as JSON next to the stores.

Example::

    PYTHONPATH=. python benchmarks/compression_benchmark.py --sample <scratch>/sample_hour \\
        --work <scratch>/compression --weights <scratch>/weights/ptype_weights.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from ptype.hrrr_fast import columns, reader, writers
from ptype.hrrr_fast.engine import make_engine
from zarr.codecs import BloscCodec, GzipCodec, TransposeCodec, ZstdCodec
from zarr.codecs.numcodecs import BitRound

ML = ("ML_rain", "ML_snow", "ML_icep", "ML_frzr", "ML_u")
FLAGS = ("precip_mask", "CRAIN", "CSNOW", "CICEP", "CFRZR")
GROUPS = ("T", "Td", "u", "v", "p")


def dir_bytes(path: Path, only_chunks: bool = True) -> tuple[int, int]:
    """(object count, bytes) under ``path``; chunk objects are the files below a ``c`` directory."""
    n = b = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        is_chunk = "c" in f.relative_to(path).parts[:-1]
        if is_chunk == only_chunks:
            n += 1
            b += f.stat().st_size
    return n, b


def lossless(name: str):
    """Bytes-to-bytes codec lists that zarrita 0.7.5 can decode (zstd, blosc, gzip) plus 'none'."""
    return {
        "none": [],
        "zstd1": [ZstdCodec(level=1)],
        "zstd3": [ZstdCodec(level=3)],
        "zstd9": [ZstdCodec(level=9)],
        "blosc-zstd5-shuffle": [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=4)],
        "blosc-zstd5-bitshuffle": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)],
        "gzip6": [GzipCodec(level=6)],
    }[name]


# ---------------------------------------------------------------------------------------------
# Gridded encodings
# ---------------------------------------------------------------------------------------------
def grid_variants() -> dict[str, dict]:
    """Encodings tried for the probability/u arrays (filters + compressors), all browser-decodable."""
    v = {k: {"filters": [], "compressors": lossless(k)} for k in ("none", "zstd1", "zstd3", "zstd9", "blosc-zstd5-shuffle", "blosc-zstd5-bitshuffle", "gzip6")}
    v["uint8-1/254+zstd3"] = {"filters": writers.quantized_float(1 / 254, "uint8"), "compressors": [ZstdCodec(level=3)]}
    v["uint8-1/254+zstd9"] = {"filters": writers.quantized_float(1 / 254, "uint8"), "compressors": [ZstdCodec(level=9)]}
    v["uint16-0.001+zstd3"] = {"filters": writers.quantized_float(0.001, "uint16"), "compressors": [ZstdCodec(level=3)]}
    v["uint16-0.001+blosc-bitshuffle"] = {
        "filters": writers.quantized_float(0.001, "uint16"),
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=2)],
    }
    v["bitround9+zstd3"] = {"filters": [BitRound(keepbits=9)], "compressors": [ZstdCodec(level=3)]}
    v["f32-round0.001+zstd3"] = {"filters": [], "compressors": [ZstdCodec(level=3)], "preround": 0.001}
    return v


def bench_grid(sample: Path, grid, work: Path, repeat: int) -> list[dict]:
    """Write the 5 ML arrays (and separately prate and the flags) with each encoding; measure."""
    rows = []
    data = {k: np.load(sample / f"{k}.npy") for k in ML}
    for name, enc in grid_variants().items():
        rows.append(_grid_case(work, grid, "ML(5)", name, data, enc, repeat))
    prate = {"prate": np.load(sample / "PRATE.npy")}
    for name in ("none", "zstd1", "zstd3", "zstd9", "blosc-zstd5-shuffle", "blosc-zstd5-bitshuffle"):
        rows.append(_grid_case(work, grid, "prate", name, prate, {"filters": [], "compressors": lossless(name)}, repeat))
    flags = {k: np.load(sample / f"{k}.npy") for k in FLAGS}
    for name in ("none", "zstd1", "zstd3", "zstd9", "gzip6"):
        rows.append(_grid_case(work, grid, "flags(5) uint8", name, flags, {"filters": [], "compressors": lossless(name), "dtype": "uint8"}, repeat, agg="max", fill=255))
    return rows


def _grid_case(work: Path, grid, group: str, name: str, data: dict, enc: dict, repeat: int, agg: str = "mean", fill=float("nan")) -> dict:
    path = work / "grid" / f"{group.split('(')[0].split()[0]}__{name.replace('/', '_')}.zarr"
    shutil.rmtree(path, ignore_errors=True)
    step = enc.get("preround")
    specs = {k: writers.VariableSpec(dtype=enc.get("dtype", "float32"), fill_value=fill, aggregation=agg, filters=enc["filters"], compressors=enc["compressors"]) for k in data}
    writers.create_ptype_store(path, grid, [12], variables=specs)
    fields = {k: (np.round(v / step) * step).astype(np.float32) if step else v for k, v in data.items()}
    t0 = time.perf_counter()
    writers.write_ptype_hour(path, 0, fields)
    t_write = time.perf_counter() - t0
    # Only the data arrays' chunks (the x/y/forecast_hour coordinate chunks are excluded).
    sizes = {(lvl, k): dir_bytes(path / str(lvl) / k) for lvl in range(len(zarr.open_group(path, mode="r").attrs["multiscales"]["layout"])) for k in data}
    n_obj = sum(v[0] for v in sizes.values())
    nbytes = sum(v[1] for v in sizes.values())
    n0 = sum(v[1] for (lvl, _), v in sizes.items() if lvl == 0)
    t_read = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        g = zarr.open_group(path, mode="r")
        back = {k: g[f"0/{k}"][0] for k in data}
        t_read.append(time.perf_counter() - t0)
    err = np.concatenate([np.abs(back[k].astype(np.float64) - data[k].astype(np.float64)).ravel() for k in data])
    return {
        "group": group,
        "encoding": name,
        "bytes_all_levels": nbytes,
        "bytes_level0": n0,
        "objects": n_obj,
        "write_s": t_write,
        "read_level0_s": min(t_read),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
    }


def bench_chunks(sample: Path, grid, work: Path, chunkings: list[tuple[int, int]], enc_name: str) -> list[dict]:
    """Requests and bytes for the site's read (5 ML arrays, one hour, one level) per chunking and level."""
    enc = grid_variants()[enc_name]
    data = {k: np.load(sample / f"{k}.npy") for k in ML}
    rows = []
    for cy, cx in chunkings:
        path = work / "chunks" / f"{cy}x{cx}.zarr"
        shutil.rmtree(path, ignore_errors=True)
        specs = {k: writers.VariableSpec(filters=enc["filters"], compressors=enc["compressors"]) for k in data}
        writers.create_ptype_store(path, grid, [12], variables=specs, chunks=(cy, cx))
        writers.write_ptype_hour(path, 0, data)
        root = zarr.open_group(path, mode="r")
        for lvl, entry in enumerate(root.attrs["multiscales"]["layout"]):
            n, b = 0, 0
            for k in data:
                nn, bb = dir_bytes(path / str(lvl) / k)
                n += nn
                b += bb
            a = root[f"{lvl}/ML_rain"]
            rows.append({"chunk": f"{cy}x{cx}", "level": lvl, "shape": entry["spatial:shape"], "level_chunk": list(a.chunks[1:]), "chunk_requests": n, "bytes": b})
    return rows


def bench_mask(sample: Path, grid, work: Path) -> list[dict]:
    """Bytes of the 5 ML arrays as stored (everywhere) vs. blanked to NaN outside precip_mask."""
    mask = np.load(sample / "precip_mask.npy").astype(bool)
    data = {k: np.load(sample / f"{k}.npy") for k in ML}
    rows = []
    for enc_name in ("zstd3", "uint8-1/254+zstd3", "uint16-0.001+zstd3"):
        enc = grid_variants()[enc_name]
        for blank in (False, True):
            path = work / "mask" / f"{enc_name.replace('/', '_')}_{'masked' if blank else 'full'}.zarr"
            shutil.rmtree(path, ignore_errors=True)
            specs = {k: writers.VariableSpec(filters=enc["filters"], compressors=enc["compressors"]) for k in data}
            writers.create_ptype_store(path, grid, [12], variables=specs)
            writers.write_ptype_hour(path, 0, {k: np.where(mask, v, np.nan).astype(np.float32) if blank else v for k, v in data.items()})
            levels = range(len(zarr.open_group(path, mode="r").attrs["multiscales"]["layout"]))
            b_all = sum(dir_bytes(path / str(lvl) / k)[1] for lvl in levels for k in data)
            b0 = sum(dir_bytes(path / "0" / k)[1] for k in data)
            rows.append({"encoding": enc_name, "blanked_outside_mask": blank, "bytes_all_levels": b_all, "bytes_level0": b0})
    return rows


# ---------------------------------------------------------------------------------------------
# Sounding encodings
# ---------------------------------------------------------------------------------------------
T_ORDER = (0, 2, 1)  # store each inner chunk variable-major: neighbouring locations become neighbouring bytes


def sounding_variants() -> dict[str, dict]:
    """Sounding encodings: lossless float32 and lossy quantisations, all decodable by zarrita 0.7.5."""
    tr = [TransposeCodec(order=T_ORDER)]
    v: dict[str, dict] = {k: {"filters": [], "compressors": lossless(k)} for k in ("none", "zstd1", "zstd3", "zstd9", "blosc-zstd5-shuffle", "blosc-zstd5-bitshuffle", "gzip6")}
    v["transpose+zstd3"] = {"filters": tr, "compressors": [ZstdCodec(level=3)]}
    v["transpose+blosc-zstd5-shuffle"] = {"filters": tr, "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=4)]}
    v["transpose+blosc-zstd5-bitshuffle"] = {"filters": tr, "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)]}
    # Lossy. One scale applies to the whole array (scale_offset takes a scalar), so the two integer
    # options are "0.1 in every unit" (int16: p*10 <= ~10,900 fits) and "0.01 in every unit" (int32,
    # because p*100 does not fit int16). The mixed precisions are pre-rounded float32 (no codec needed).
    v["int16-0.1+zstd3"] = {"filters": writers.quantized_float(0.1, "int16"), "compressors": [ZstdCodec(level=3)]}
    v["int16-0.1+transpose+zstd3"] = {"filters": [*writers.quantized_float(0.1, "int16"), *tr], "compressors": [ZstdCodec(level=3)]}
    v["int16-0.1+transpose+blosc-bitshuffle"] = {
        "filters": [*writers.quantized_float(0.1, "int16"), *tr],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=2)],
    }
    v["int32-0.01+transpose+zstd3"] = {"filters": [*writers.quantized_float(0.01, "int32"), *tr], "compressors": [ZstdCodec(level=3)]}
    v["int32-0.01+transpose+blosc-bitshuffle"] = {
        "filters": [*writers.quantized_float(0.01, "int32"), *tr],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)],
    }
    v["f32-round(T,Td,wind 0.01; p 0.1)+transpose+zstd3"] = {"filters": tr, "compressors": [ZstdCodec(level=3)], "preround": (0.01, 0.01, 0.01, 0.01, 0.1)}
    v["f32-round(all 0.1)+transpose+zstd3"] = {"filters": tr, "compressors": [ZstdCodec(level=3)], "preround": (0.1, 0.1, 0.1, 0.1, 0.1)}
    v["bitround(keep 12)+transpose+zstd3"] = {"filters": [BitRound(keepbits=12), *tr], "compressors": [ZstdCodec(level=3)]}
    v["bitround(keep 12)+transpose+blosc-bitshuffle"] = {
        "filters": [BitRound(keepbits=12), *tr],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)],
    }
    v["bitround(keep 10)+transpose+blosc-bitshuffle"] = {
        "filters": [BitRound(keepbits=10), *tr],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=4)],
    }
    v["int16-0.1+transpose+blosc-shuffle"] = {
        "filters": [*writers.quantized_float(0.1, "int16"), *tr],
        "compressors": [BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", typesize=2)],
    }
    return v


def _preround(s: np.ndarray, steps, nh: int) -> np.ndarray:
    out = s.copy()
    for g, step in enumerate(steps):
        sl = out[:, g * nh : (g + 1) * nh]
        sl[...] = np.round(sl / step) * step
    return out


def _probs_from_earth(snd: np.ndarray, lon_flat: np.ndarray, grid, engine, nh: int) -> np.ndarray:
    """Network probabilities from an EARTH-relative (n, 105) sounding: rotate winds back, build inputs, predict."""
    n = snd.shape[0]
    s3 = snd.reshape(n, 5, nh)
    ue = np.ascontiguousarray(s3[:, 2, :].T)[:, None, :]
    ve = np.ascontiguousarray(s3[:, 3, :].T)[:, None, :]
    ug, vg = columns.rotate_earth_to_grid(ue, ve, lon_flat[None, :], lon_0=grid.lon_0, lat_std=grid.lat_1)
    x = np.empty((n, 4 * nh), dtype=np.float32)
    x[:, : 2 * nh] = snd[:, : 2 * nh]
    x[:, 2 * nh : 3 * nh] = ug[:, 0, :].T
    x[:, 3 * nh :] = vg[:, 0, :].T
    return engine.predict(x)[0]


def bench_sounding(sample: Path, grid, work: Path, weights: str, keep: bool, only: list[str] | None, location_chunk: int) -> list[dict]:
    """Write the sounding with each encoding; bytes, timings, per-group errors and network effect."""
    meta = json.loads((sample / "meta.json").read_text())
    nh = len(meta["heights_m"])
    snd = np.load(sample / "sounding_earth.npy")
    lon_flat = np.load(sample / "lon.npy").reshape(-1)
    mask = np.load(sample / "precip_mask.npy").reshape(-1).astype(bool)
    engine = make_engine("numpy", weights)
    t0 = time.perf_counter()
    p_ref = _probs_from_earth(snd, lon_flat, grid, engine, nh)
    print(f"reference network pass on the float32 sounding: {time.perf_counter() - t0:.2f} s", flush=True)
    cls_ref = p_ref.argmax(axis=1)
    rows = []
    for name, enc in sounding_variants().items():
        if only and name not in only:
            continue
        path = work / "sounding" / f"{name.replace('/', '_').replace(' ', '').replace(';', '_')}.zarr"
        shutil.rmtree(path, ignore_errors=True)
        writers.create_sounding_store(
            path,
            grid,
            [12],
            heights_m=meta["heights_m"],
            variable_groups=meta["variable_groups"],
            location_chunk=location_chunk,
            filters=enc["filters"],
            compressors=enc["compressors"],
        )
        src = _preround(snd, enc["preround"], nh) if "preround" in enc else snd
        t0 = time.perf_counter()
        writers.write_sounding_hour(path, 0, src)
        t_write = time.perf_counter() - t0
        del src
        n_obj, nbytes = dir_bytes(path)
        arr = zarr.open_array(path, path="sounding", mode="r")
        t0 = time.perf_counter()
        back = arr[0]
        t_read = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = arr[0, 987654, :]
        t_one = time.perf_counter() - t0
        row = {"encoding": name, "bytes": nbytes, "objects": n_obj, "write_s": t_write, "read_hour_s": t_read, "read_one_location_s": t_one}
        diff = np.abs(back.astype(np.float64) - snd.astype(np.float64)).reshape(-1, 5, nh)
        for g, gname in enumerate(GROUPS):
            row[f"max_err_{gname}"] = float(diff[:, g].max())
            row[f"mean_err_{gname}"] = float(diff[:, g].mean())
        if float(diff.max()) > 0:
            p = _probs_from_earth(back, lon_flat, grid, engine, nh)
            dp = np.abs(p - p_ref)
            changed = p.argmax(axis=1) != cls_ref
            row |= {
                "net_max_abs_dprob": float(dp.max()),
                "net_mean_abs_dprob": float(dp.mean()),
                "net_class_change_frac": float(changed.mean()),
                "net_class_change_frac_in_mask": float(changed[mask].mean()),
                "net_class_change_count": int(changed.sum()),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del back
        if not keep:
            shutil.rmtree(path, ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------------------------
# Default encodings: bytes per forecast hour and arithmetic extrapolation
# ---------------------------------------------------------------------------------------------
#: Forecast hours per store/archive used for the extrapolation: an 18 h run has f00..f18 (19 hours), a
#: 48 h run f00..f48 (49 hours); the archive is 85 days x (4 x 49 + 20 x 19 = 576) = 48,960 hours.
EXTRAPOLATE = {"18 h run (19 hours)": 19, "48 h run (49 hours)": 49, "one day (4 x 49 + 20 x 19 = 576 hours)": 576, "archive: 85 days (48,960 hours)": 85 * 576}


def bench_defaults(sample: Path, grid, work: Path) -> list[dict]:
    """Write one hour of a run with the writers' DEFAULT encodings and extrapolate its bytes."""
    meta = json.loads((sample / "meta.json").read_text())
    fields = {k: np.load(sample / f"{k}.npy") for k in (*ML, *FLAGS)}
    fields["prate"] = np.load(sample / "PRATE.npy")
    gpath, spath = work / "defaults" / "grid.zarr", work / "defaults" / "sounding.zarr"
    t0 = time.perf_counter()
    writers.create_ptype_store(gpath, grid, range(19), init_time="2023-12-26T00:00Z")
    t_gcreate = time.perf_counter() - t0
    t0 = time.perf_counter()
    writers.write_ptype_hour(gpath, 12, fields)
    t_gwrite = time.perf_counter() - t0
    snd = np.load(sample / "sounding_earth.npy")
    t0 = time.perf_counter()
    writers.create_sounding_store(spath, grid, range(19), heights_m=meta["heights_m"], variable_groups=meta["variable_groups"], init_time="2023-12-26T00:00Z")
    t_screate = time.perf_counter() - t0
    t0 = time.perf_counter()
    writers.write_sounding_hour(spath, 12, snd)
    t_swrite = time.perf_counter() - t0
    g_obj, g_bytes = dir_bytes(gpath)
    s_obj, s_bytes = dir_bytes(spath)
    gm_obj, gm_bytes = dir_bytes(gpath, only_chunks=False)
    sm_obj, sm_bytes = dir_bytes(spath, only_chunks=False)
    # coordinate chunks (x, y, forecast_hour, init_time) are per store, not per hour
    coord = (
        sum(dir_bytes(p)[1] for p in gpath.glob("*/[xy]")) + sum(dir_bytes(p)[1] for p in gpath.glob("*/forecast_hour")) + sum(dir_bytes(p)[1] for p in gpath.glob("*/spatial_ref"))
    )
    coord_s = dir_bytes(spath / "forecast_hour")[1] + dir_bytes(spath / "init_time")[1]
    per_hour = {"grid": g_bytes - coord, "sounding": s_bytes - coord_s}
    rows = [
        {
            "item": "measured: one hour",
            "grid_bytes": per_hour["grid"],
            "sounding_bytes": per_hour["sounding"],
            "total_bytes": per_hour["grid"] + per_hour["sounding"],
            "note": f"grid write {t_gwrite:.2f} s ({g_obj} chunk objects incl. coords), sounding write {t_swrite:.2f} s ({s_obj} objects incl. coords); "
            f"create {t_gcreate:.2f} s / {t_screate:.3f} s; per-store metadata {gm_bytes + sm_bytes:,} B in {gm_obj + sm_obj} documents, coordinate chunks {coord + coord_s:,} B",
        }
    ]
    for label, n in EXTRAPOLATE.items():
        rows.append(
            {
                "item": f"x {n:,}: {label}",
                "grid_bytes": per_hour["grid"] * n,
                "sounding_bytes": per_hour["sounding"] * n,
                "total_bytes": (per_hour["grid"] + per_hour["sounding"]) * n,
                "note": "arithmetic",
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Object sizes: per-hour (staging) vs finalized layout, S3 Standard-IA billed bytes
# ---------------------------------------------------------------------------------------------
def _size_stats(sizes: list[int]) -> dict:
    a = np.asarray(sizes, dtype=np.int64)
    small = a[a < writers.IA_MIN_OBJECT_BYTES]
    return {
        "objects": int(a.size),
        "bytes": int(a.sum()),
        "min": int(a.min()),
        "median": float(np.median(a)),
        "max": int(a.max()),
        "objects_under_128k": int(small.size),
        "bytes_under_128k": int(small.sum()),
        "billed_bytes_ia": int(np.maximum(a, writers.IA_MIN_OBJECT_BYTES).sum()),
    }


def _sizes(path: Path) -> list[int]:
    return [f.stat().st_size for f in path.rglob("*") if f.is_file()]


def bench_objects(sample: Path, grid, work: Path, n_hours_list: list[int], keep: bool) -> list[dict]:
    """Build whole runs from the sample hour REPEATED n times; object counts/sizes before and after finalize.

    Every hour holds identical data, so per-object compressibility is real but hour-to-hour variation (and
    hence a run-spanning shard's size) is only as representative as this one hour.
    """
    meta = json.loads((sample / "meta.json").read_text())
    fields = {k: np.load(sample / f"{k}.npy") for k in (*ML, *FLAGS)}
    fields["prate"] = np.load(sample / "PRATE.npy")
    snd = np.load(sample / "sounding_earth.npy")
    rows = []
    for n in n_hours_list:
        run = work / f"objects_{n}h"
        shutil.rmtree(run, ignore_errors=True)
        gpath, spath = run / "grid.zarr", run / "sounding.zarr"
        writers.create_ptype_store(gpath, grid, range(n), init_time="2023-12-26T00:00Z")
        writers.create_sounding_store(spath, grid, range(n), heights_m=meta["heights_m"], variable_groups=meta["variable_groups"], init_time="2023-12-26T00:00Z")
        t0 = time.perf_counter()
        for fh in range(n):
            writers.write_ptype_hour(gpath, fh, fields)
        t_g = time.perf_counter() - t0
        t0 = time.perf_counter()
        for fh in range(n):
            writers.write_sounding_hour(spath, fh, snd)
        t_s = time.perf_counter() - t0
        staged = _sizes(gpath)
        sound = _sizes(spath)
        t0 = time.perf_counter()
        rep = writers.finalize_ptype_store(gpath)
        t_f = time.perf_counter() - t0
        final = _sizes(gpath)
        common = {"hours": n, "write_grid_s": t_g, "write_sounding_s": t_s, "finalize_s": t_f}
        rows.append(common | {"layout": "per-hour (staging)", "store": "gridded"} | _size_stats(staged))
        rows.append(common | {"layout": "finalized", "store": "gridded"} | _size_stats(final))
        rows.append(common | {"layout": "both", "store": "sounding"} | _size_stats(sound))
        rows.append(common | {"layout": "per-hour (staging)", "store": "gridded+sounding"} | _size_stats(staged + sound))
        rows.append(common | {"layout": "finalized", "store": "gridded+sounding"} | _size_stats(final + sound))
        rows.append({"hours": n, "layout": "finalized", "store": "small objects (gridded)", "detail": rep["small_objects"]})
        print(json.dumps(rows[-6:-1]), flush=True)
        if not keep:
            shutil.rmtree(run, ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------
def md_table(rows: list[dict], cols: list[str]) -> str:
    """Render rows as a Markdown table (floats to 4 significant digits, ints with separators)."""

    def fmt(v):
        if isinstance(v, bool) or v is None:
            return str(v)
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--sections", default="grid,chunks,mask,sounding,defaults")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--keep", action="store_true", help="keep the sounding stores (for the zarrita read test)")
    p.add_argument("--only", default=None, help="comma-separated sounding encodings to run")
    p.add_argument("--location-chunk", type=int, default=1024)
    p.add_argument("--chunkings", default="1059x1799,530x900,512x512,353x360,256x256")
    p.add_argument("--chunk-encoding", default="uint8-1/254+zstd3")
    p.add_argument("--object-hours", default="19,49", help="run lengths (hours) for --sections objects")
    args = p.parse_args()

    sample = Path(args.sample)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    grid = reader.GridMetadata(**json.loads((sample / "meta.json").read_text())["grid"])
    sections = args.sections.split(",")
    results: dict[str, list[dict]] = {}
    if "grid" in sections:
        results["grid"] = bench_grid(sample, grid, work, args.repeat)
        print(
            md_table(results["grid"], ["group", "encoding", "bytes_all_levels", "bytes_level0", "objects", "write_s", "read_level0_s", "max_abs_err", "mean_abs_err"]),
            "\n",
            flush=True,
        )
    if "chunks" in sections:
        ch = [tuple(int(v) for v in c.split("x")) for c in args.chunkings.split(",")]
        results["chunks"] = bench_chunks(sample, grid, work, ch, args.chunk_encoding)
        print(md_table(results["chunks"], ["chunk", "level", "shape", "level_chunk", "chunk_requests", "bytes"]), "\n", flush=True)
    if "mask" in sections:
        results["mask"] = bench_mask(sample, grid, work)
        print(md_table(results["mask"], ["encoding", "blanked_outside_mask", "bytes_all_levels", "bytes_level0"]), "\n", flush=True)
    if "sounding" in sections:
        only = args.only.split("|") if args.only else None
        results["sounding"] = bench_sounding(sample, grid, work, args.weights, args.keep, only, args.location_chunk)
        cols = ["encoding", "bytes", "objects", "write_s", "read_hour_s", "read_one_location_s"]
        cols += [f"max_err_{g}" for g in GROUPS] + ["net_max_abs_dprob", "net_mean_abs_dprob", "net_class_change_frac", "net_class_change_frac_in_mask"]
        print(md_table(results["sounding"], cols), flush=True)
    if "defaults" in sections:
        results["defaults"] = bench_defaults(sample, grid, work)
        print(md_table(results["defaults"], ["item", "grid_bytes", "sounding_bytes", "total_bytes", "note"]), flush=True)
    if "objects" in sections:
        results["objects"] = bench_objects(sample, grid, work, [int(h) for h in args.object_hours.split(",")], args.keep)
        cols = ["hours", "store", "layout", "objects", "bytes", "min", "median", "max", "objects_under_128k", "bytes_under_128k", "billed_bytes_ia"]
        print(md_table([r for r in results["objects"] if "objects" in r], cols), flush=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (work / f"results_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nsaved {work / f'results_{stamp}.json'}")


if __name__ == "__main__":
    main()
