"""End-to-end difference between the original (wrfprs, netCDF) and the new (wrfnat, Zarr) pipeline for one hour.

Inputs are the original run's netCDF (``MILES_ptype_hrrr_<date>_<hh>00_f<fh>.nc``; a lat/lon-box subset of the
HRRR grid) and the new pipeline's gridded Zarr store (full 1059 x 1799 grid, level 0). The two grids share one
Lambert projection, so the subset is matched by projecting the original's latitude/longitude into the Zarr's
projected x/y axes and rounding to the nearest grid index. The match is then *verified*, not assumed: the
projection residual (metres) must be a small fraction of the 3 km spacing, and the Zarr cell centres are projected
back to latitude/longitude and compared with the original's coordinates.

Reported: per-class probability and uncertainty differences (mean, median, 95th, 99th percentile, max of the absolute
difference), dominant-class agreement over all matched points and over HRRR-precipitation points, and a 4 x 4 confusion
table of dominant classes. The Zarr stores probabilities as uint8 steps of 1/254, so the comparison also reports the
same statistics for the *original's own* probabilities passed through that quantisation: that is the floor any
comparison against the store has, whatever the two pipelines do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["rain", "snow", "icep", "frzr"]
HRRR_FLAGS = ["crain", "csnow", "cicep", "cfrzr"]
STEP = 254.0  # the store's probability quantisation, writers.quantized_float(1 / 254)


def match_grids(lat: np.ndarray, lon: np.ndarray, x: np.ndarray, y: np.ndarray, proj_string: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map every original grid point to its index on the Zarr grid and verify the mapping.

    Args:
        lat: Original latitude, shape ``(ny, nx)``.
        lon: Original longitude, degrees east (0-360 or -180-180), same shape.
        x: Zarr projected x of the cell centres (m), shape ``(NX,)``, evenly spaced.
        y: Zarr projected y of the cell centres (m), shape ``(NY,)``, evenly spaced.
        proj_string: PROJ string of the Zarr grid (the store's ``proj:code``).

    Returns:
        ``(iy, ix, report)``: integer index arrays of the original's shape, and a dict of verification numbers.

    Raises:
        ValueError: If a point is more than 1 % of a cell away from a Zarr cell centre or falls outside the Zarr grid.
    """
    from pyproj import CRS, Transformer

    crs = CRS.from_user_input(proj_string)
    to_xy = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon180 = (lon + 180.0) % 360.0 - 180.0
    px, py = to_xy.transform(lon180, lat)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    fx, fy = (px - x[0]) / dx, (py - y[0]) / dy
    ix, iy = np.rint(fx).astype(np.int64), np.rint(fy).astype(np.int64)
    # Residual in metres between the original point and the centre of the cell it was assigned to.
    res_m = np.hypot((fx - ix) * abs(dx), (fy - iy) * abs(dy))
    if ix.min() < 0 or iy.min() < 0 or ix.max() >= len(x) or iy.max() >= len(y):
        raise ValueError("original grid points fall outside the Zarr grid")
    if res_m.max() > 0.01 * abs(dx):
        raise ValueError(f"original points are not on the Zarr grid: max residual {res_m.max():.1f} m")
    # Independent second check: back-project the Zarr cell centres to lat/lon and compare with the original's coordinates.
    lon_z, lat_z = to_ll.transform(x[ix], y[iy])
    dlon = np.abs((lon_z - lon180 + 180.0) % 360.0 - 180.0)
    dlat = np.abs(lat_z - lat)
    # Distinct cells and a one-to-one index pattern (contiguous block, unit steps) rule out a shuffled or strided match.
    unit_steps = bool(np.all(np.diff(ix, axis=1) == 1) and np.all(np.diff(iy, axis=0) == 1) and np.all(np.diff(iy, axis=1) == 0) and np.all(np.diff(ix, axis=0) == 0))
    report = {
        "original_shape": list(lat.shape),
        "zarr_shape": [len(y), len(x)],
        "iy_range": [int(iy.min()), int(iy.max())],
        "ix_range": [int(ix.min()), int(ix.max())],
        "contiguous_block_unit_steps": unit_steps,
        "max_projection_residual_m": float(res_m.max()),
        "max_abs_dlat_deg": float(dlat.max()),
        "max_abs_dlon_deg": float(dlon.max()),
        "orig_corner_lat_lon_first": [float(lat[0, 0]), float(lon180[0, 0])],
        "orig_corner_lat_lon_last": [float(lat[-1, -1]), float(lon180[-1, -1])],
        "zarr_corner_lat_lon_first": [float(lat_z[0, 0]), float(lon_z[0, 0])],
        "zarr_corner_lat_lon_last": [float(lat_z[-1, -1]), float(lon_z[-1, -1])],
    }
    return iy, ix, report


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean, median, 95th, 99th percentile and max of ``|a - b|`` (float64)."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).ravel()
    q = np.percentile(d, [50, 95, 99])
    return {"mean": float(d.mean()), "median": float(q[0]), "p95": float(q[1]), "p99": float(q[2]), "max": float(d.max())}


def agreement(a_cls: np.ndarray, b_cls: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Fraction of points with the same dominant class (optionally inside ``mask``) and the 4 x 4 confusion counts (rows = a)."""
    if mask is not None:
        a_cls, b_cls = a_cls[mask], b_cls[mask]
    conf = np.zeros((4, 4), dtype=np.int64)
    np.add.at(conf, (a_cls.ravel(), b_cls.ravel()), 1)
    return {"n": int(a_cls.size), "agreement": float((a_cls == b_cls).mean()) if a_cls.size else float("nan"), "confusion_rows_original_cols_new": conf.tolist()}


def main() -> None:
    """Load both outputs, match the grids, compute the statistics and write the JSON report."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original-nc", required=True)
    p.add_argument("--new-store", required=True, help="gridded Zarr store (hrrr_ptype_<init>.zarr)")
    p.add_argument("--forecast-hour", type=int, required=True)
    p.add_argument("--out", required=True, help="JSON report path")
    a = p.parse_args()

    import xarray as xr
    import zarr

    ds = xr.open_dataset(a.original_nc)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    orig_p = np.stack([ds[f"ML_{c}"].values[0] for c in CLASSES], axis=-1).astype(np.float64)
    orig_u = ds["ML_u"].values[0].astype(np.float64)
    orig_flags = np.stack([ds[v].values[0] for v in HRRR_FLAGS], axis=-1)

    root = zarr.open_group(a.new_store, mode="r")
    lvl = root["0"]
    fh = np.asarray(lvl["forecast_hour"][:])
    idx = int(np.flatnonzero(fh == a.forecast_hour)[0])
    iy, ix, grid_report = match_grids(lat, lon, np.asarray(lvl["x"][:]), np.asarray(lvl["y"][:]), root.attrs["proj:code"])

    def take(name: str) -> np.ndarray:
        # Read the whole hour once and index the matched block; the subset is contiguous so this is a plain slice in effect.
        return np.asarray(lvl[name][idx])[iy, ix]

    new_p = np.stack([take(f"ML_{c}") for c in CLASSES], axis=-1).astype(np.float64)
    new_u = take("ML_u").astype(np.float64)
    new_mask = take("precip_mask").astype(bool)
    new_flags = np.stack([take(v.upper()) for v in HRRR_FLAGS], axis=-1)
    if not (np.isfinite(new_p).all() and np.isfinite(new_u).all() and np.isfinite(orig_p).all()):
        raise ValueError("non-finite values in the compared fields")

    orig_cls, new_cls = orig_p.argmax(axis=-1), new_p.argmax(axis=-1)
    orig_mask = (orig_flags == 1).any(axis=-1)
    quant_p = np.rint(orig_p * STEP) / STEP  # the original's own probabilities after the store's quantisation
    quant_cls = quant_p.argmax(axis=-1)

    rep: dict = {"grid_match": grid_report, "forecast_hour": a.forecast_hour, "n_points": int(orig_cls.size)}
    rep["hrrr_flags_agree_fraction"] = {v: float((orig_flags[..., i] == new_flags[..., i]).mean()) for i, v in enumerate(HRRR_FLAGS)}
    rep["precip_points"] = {
        "new_wrfnat_mask": int(new_mask.sum()),
        "original_wrfprs_mask": int(orig_mask.sum()),
        "in_both": int((new_mask & orig_mask).sum()),
        "in_either": int((new_mask | orig_mask).sum()),
    }
    for label, m in (("all_points", None), ("precip_points_wrfnat_mask", new_mask)):
        sel = slice(None) if m is None else m
        block = {c: diff_stats(orig_p[sel][..., i], new_p[sel][..., i]) for i, c in enumerate(CLASSES)}
        block["uncertainty"] = diff_stats(orig_u[sel], new_u[sel])
        block["quantisation_floor"] = {c: diff_stats(orig_p[sel][..., i], quant_p[sel][..., i]) for i, c in enumerate(CLASSES)}
        block["dominant_class"] = agreement(orig_cls, new_cls, m)
        block["dominant_class_original_vs_its_own_quantised"] = agreement(orig_cls, quant_cls, m)
        rep[label] = block
    rep["precip_points_wrfprs_mask"] = {"dominant_class": agreement(orig_cls, new_cls, orig_mask)}
    rep["class_counts_all_points"] = {"original": np.bincount(orig_cls.ravel(), minlength=4).tolist(), "new": np.bincount(new_cls.ravel(), minlength=4).tolist()}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
