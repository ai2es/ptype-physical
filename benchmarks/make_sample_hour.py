"""Produce one REAL forecast hour of everything the HRRR ptype writers store, for tests and benchmarks.

Pipeline (each stage timed): read HRRR wrfnat GRIB2 messages (from a cached byte-range pickle written by
``benchmarks/reader_benchmark.py fetch --cache`` or straight from the NOAA bucket) -> decode -> grid
metadata -> heights above ground -> ``columns.build_soundings`` (default surface policies: 2 m T/Td,
10 m wind, surface pressure at 0 m; grid-relative winds, which is what the network is fed) ->
``columns.network_inputs`` -> engine -> probabilities and evidential ``u``; then the soundings are
rotated to EARTH-relative winds for storage, and the HRRR categorical fields, PRATE, 1-h APCP and
``precip_mask`` (any of CRAIN/CSNOW/CICEP/CFRZR == 1) are kept.

Outputs go to ``--out`` as plain ``.npy`` files (memory-mappable, so the 800 MB sounding array can
be opened lazily by the tests) plus ``meta.json`` with the grid metadata, timings and sanity stats.
Keep ``--out`` outside the repository.

Example::

    PYTHONPATH=. python benchmarks/make_sample_hour.py --cache <scratch>/reader_data/f12_n30.pkl \\
        --weights <scratch>/weights/ptype_weights.npz --out <scratch>/writers_data/sample_hour
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
from ptype.hrrr_fast import columns, reader
from ptype.hrrr_fast.engine import make_engine

CATEGORICAL = ("CRAIN", "CSNOW", "CICEP", "CFRZR")
CLASSES = ("rain", "snow", "icep", "frzr")


def load_messages(args) -> tuple[reader.ReadPlan, list]:
    """Return the read plan for ``n_levels`` and the matching message buffers.

    A cached pickle written for MORE levels is a superset of what we need, so the plan for
    ``n_levels`` is re-derived from the cached idx and messages are looked up by byte offset.
    """
    if args.cache:
        with open(args.cache, "rb") as f:
            c = pickle.load(f)  # our own cache file in the scratch directory
        entries = reader.parse_idx(c["idx"])
        if c["n_levels"] < args.n_levels:
            raise SystemExit(f"cache holds {c['n_levels']} levels, need {args.n_levels}")
        full = reader.select_messages(entries, n_levels=c["n_levels"], fxx=c["fxx"])
        by_offset = {m.entry.offset: b for m, b in zip(full.messages, c["messages"], strict=True)}
        plan = reader.select_messages(entries, n_levels=args.n_levels, fxx=c["fxx"])
        return plan, [by_offset[m.entry.offset] for m in plan.messages]
    store = reader.open_store(args.source)
    key = reader.hrrr_key(date.fromisoformat(args.date), args.cycle, args.fxx)
    plan = reader.select_messages(reader.parse_idx(reader.fetch_idx(store, key)), n_levels=args.n_levels, fxx=args.fxx)
    return plan, reader.fetch_ranges(store, key, plan)


def grid_lonlat(grid: reader.GridMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Longitude/latitude of every grid point (array order) from the grid's own PROJ string."""
    import pyproj

    x, y = grid.xy_coords()
    xx, yy = np.meshgrid(x, y)
    tf = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(grid.proj4), "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(xx, yy)
    return lon, lat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=None, help="pickle from reader_benchmark.py fetch --cache (skips the network)")
    p.add_argument("--source", default=None, help="None/'s3' for the NOAA bucket or a local mirror directory")
    p.add_argument("--date", default="2023-12-26")
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--fxx", type=int, default=12)
    p.add_argument("--n-levels", type=int, default=27)
    p.add_argument("--weights", required=True)
    p.add_argument("--engine", default="numpy")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    plan, msgs = load_messages(args)
    t["load_messages_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    f = reader.decode(plan, msgs, decoder="eccodes", workers=args.decode_workers, pool="thread")
    t["decode_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = reader.grid_metadata(msgs[0])
    lon, lat = grid_lonlat(grid)
    t["grid_lonlat_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    z_agl = f["HGT"] - f["HGT_sfc"][None]
    t["height_agl_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    snd, n_short = columns.build_soundings(z_agl, f["TMP"], f["SPFH"], f["PRES"], f["UGRD"], f["VGRD"], f["TMP_2m"], f["DPT_2m"], f["UGRD_10m"], f["VGRD_10m"], f["PRES_sfc"])
    t["build_soundings_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = columns.network_inputs(snd)
    t["network_inputs_s"] = time.perf_counter() - t0

    engine = make_engine(args.engine, args.weights, threads=args.threads)
    t0 = time.perf_counter()
    probs, u = engine.predict(x)
    t["engine_predict_s"] = time.perf_counter() - t0
    del x

    # Earth-relative winds for the stored sounding. The angle is constant in a column, so rotating the
    # interpolated grid-relative winds is exact (see columns.build_soundings, "rotating after interpolation").
    t0 = time.perf_counter()
    ny, nx = grid.ny, grid.nx
    nh = snd.shape[2]
    ug = np.ascontiguousarray(snd[:, 2, :].T).reshape(nh, ny, nx)
    vg = np.ascontiguousarray(snd[:, 3, :].T).reshape(nh, ny, nx)
    ue, ve = columns.rotate_grid_to_earth(ug, vg, lon, lon_0=grid.lon_0, lat_std=grid.lat_1)
    snd_earth = snd.copy()
    snd_earth[:, 2, :] = ue.reshape(nh, -1).T
    snd_earth[:, 3, :] = ve.reshape(nh, -1).T
    t["rotate_to_earth_s"] = time.perf_counter() - t0

    # Row order: the reader returns native scan order. location = j * ni + i must have j south -> north.
    if not grid.row0_is_south:
        raise SystemExit("rows are north->south; flip before saving (not needed for HRRR, jScansPositively=1)")

    t0 = time.perf_counter()
    cat = {k: (f[k] == 1) for k in CATEGORICAL}
    precip_mask = np.zeros((ny, nx), dtype=bool)
    for v in cat.values():
        precip_mask |= v
    save = {
        "ML_rain": probs[:, 0].reshape(ny, nx),
        "ML_snow": probs[:, 1].reshape(ny, nx),
        "ML_icep": probs[:, 2].reshape(ny, nx),
        "ML_frzr": probs[:, 3].reshape(ny, nx),
        "ML_u": u.reshape(ny, nx),
        **{k: v.astype(np.uint8) for k, v in cat.items()},
        "precip_mask": precip_mask.astype(np.uint8),
        "PRATE": f["PRATE"].astype(np.float32),
        "APCP_1h": f["APCP_1h"].astype(np.float32),
        "lon": lon,
        "lat": lat,
        "sounding_earth": snd_earth.reshape(snd.shape[0], -1),
    }
    for k, v in save.items():
        np.save(out / f"{k}.npy", np.ascontiguousarray(v))
    t["save_s"] = time.perf_counter() - t0

    # Sanity statistics.
    psum = probs.sum(axis=1)
    cls = probs.argmax(axis=1)
    m = precip_mask.reshape(-1)
    freq_in = {CLASSES[k]: float(np.mean(cls[m] == k)) for k in range(4)}
    hrrr_in = {k: float(np.mean(cat[k].reshape(-1)[m])) for k in CATEGORICAL}
    stats = {
        "n_columns": int(probs.shape[0]),
        "n_short_columns": int(n_short),
        "prob_sum_max_abs_dev_from_1": float(np.max(np.abs(psum - 1.0))),
        "probs_min": float(probs.min()),
        "probs_max": float(probs.max()),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "precip_fraction_of_grid": float(m.mean()),
        "precip_columns": int(m.sum()),
        "ml_argmax_class_freq_inside_mask": freq_in,
        "hrrr_flag_freq_inside_mask": hrrr_in,
        "ml_argmax_class_freq_whole_grid": {CLASSES[k]: float(np.mean(cls == k)) for k in range(4)},
        "nan_in_sounding": int(np.isnan(snd_earth).sum()),
        "max_abs_wind_speed_change_from_rotation": float(np.max(np.abs(np.hypot(ue, ve) - np.hypot(ug, vg)))),
    }
    meta = {
        "source": {"date": args.date, "cycle": args.cycle, "fxx": plan.fxx, "n_levels": plan.n_levels, "cache": args.cache},
        "engine": engine.describe(),
        "grid": asdict(grid),
        "heights_m": [float(h) for h in columns.NETWORK_HEIGHTS],
        "variable_groups": list(columns.SOUNDING_VARIABLES),
        "timings": t,
        "stats": stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"timings": {k: round(v, 3) for k, v in t.items()}, "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
