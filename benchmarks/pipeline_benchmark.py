"""Benchmarks of the end-to-end HRRR ptype pipeline (``ptype.hrrr_fast.pipeline``).

Three subcommands:

``mirror``
    Download only the byte ranges the pipeline needs (idx-planned GRIB messages) for some forecast hours into a
    SPARSE local copy laid out like the bucket (``hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfnatfFF.grib2`` + ``.idx``).
    The pipeline reads a local mirror through obstore's ``LocalStore`` with the same code path as S3, so a
    scaling study can repeat the same hours many times without re-downloading and without the network
    dominating (``source: <mirror dir>``). ``--from-cache`` writes a mirror file from a pickle written by
    ``benchmarks/reader_benchmark.py fetch --cache`` instead of fetching.

``run``
    Run N forecast hours end to end for one or more ``processes x threads`` layouts (``--layouts 1x1 1x2 2x2 ...``)
    with upload disabled, and report wall time per forecast hour (wall / N, i.e. node throughput), the per-stage
    breakdown (fetch / decode / columns / inference / bourgouin / rotate / write; median over hours) and peak
    RSS per worker, as JSON and markdown. Runs unchanged on Derecho for the 1..128-core scaling study.

``bourgouin-depth``
    For one hour, how deep the Bourgouin height grid must be: for each top (default 5, 8, 10, 12 km) the
    fraction of columns whose chosen precipitation-generation layer touches the top of the grid (i.e. may be
    cut off), and the change in ProbIce and the dominant class relative to the deepest grid, plus the extra
    bytes and seconds that the extra hybrid levels cost.

Examples::

    PYTHONPATH=. python benchmarks/pipeline_benchmark.py mirror --init 2023122600 --hours 11-13 --out $MIRROR
    PYTHONPATH=. python benchmarks/pipeline_benchmark.py run --config config/hrrr_fast.yml --source $MIRROR \\
        --init 2023122600 --hours 11-13 --layouts 1x1 1x2 1x4 2x4 --engine torch --out results/pipeline_scaling
    PYTHONPATH=. python benchmarks/pipeline_benchmark.py bourgouin-depth --cache f12_n50.pkl --weights w.npz --out depth
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from ptype.hrrr_fast import columns, pipeline, reader
from ptype.hrrr_fast.config import PipelineConfig


# ---------------------------------------------------------------------------------------------
# mirror
# ---------------------------------------------------------------------------------------------
def _write_sparse(path: Path, size: int, pieces: list[tuple[int, bytes]]) -> None:
    """Write ``pieces`` at their offsets into a file of ``size`` bytes; untouched regions stay holes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "r+b" if path.exists() else "wb") as f:
        f.truncate(max(size, f.seek(0, 2)))
        for off, data in pieces:
            f.seek(off)
            f.write(data)


def cmd_mirror(a) -> None:
    """Fetch the planned messages of each hour (wind 1..n_levels, thermo 1..thermo_levels, surface) into a sparse mirror."""
    out = Path(a.out)
    cfg = PipelineConfig(source="s3", coalesce=a.coalesce)
    if a.from_cache:
        with open(a.from_cache, "rb") as f:
            c = pickle.load(f)  # our own cache file (reader_benchmark.py fetch --cache)
        entries = reader.parse_idx(c["idx"])
        full = reader.select_messages(entries, n_levels=c["n_levels"], fxx=c["fxx"])
        pieces = [(m.entry.offset, bytes(b)) for m, b in zip(full.messages, c["messages"], strict=True)]
        key = a.key or c["key"]
        size = max(o + len(b) for o, b in pieces)
        _write_sparse(out / key, size, pieces)
        (out / (key + ".idx")).write_text(c["idx"])
        print(json.dumps({"key": key, "messages": len(pieces), "bytes": sum(len(b) for _, b in pieces)}))
        return
    src = pipeline.GribSource(cfg)
    import obstore

    for run in pipeline.make_runs(a.init, pipeline.parse_hours(a.hours)):
        for fxx in run.hours:
            t0 = time.perf_counter()
            hour = src.open_hour(run.init, fxx)
            plan = hour.plan((0, a.n_levels), (0, a.thermo_levels), True)
            bufs = reader.fetch_ranges(src.store, hour.key, plan, coalesce=a.coalesce)
            dt = time.perf_counter() - t0
            size = int(obstore.head(src.store, hour.key)["size"])
            _write_sparse(out / hour.key, size, [(m.entry.offset, bytes(b)) for m, b in zip(plan.messages, bufs, strict=True)])
            (out / (hour.key + ".idx")).write_text(reader.fetch_idx(src.store, hour.key))
            nbytes = sum(len(b) for b in bufs)
            print(json.dumps({"key": hour.key, "messages": len(bufs), "bytes": nbytes, "seconds": round(dt, 2), "MiB_per_s": round(nbytes / 2**20 / dt, 1)}))


# ---------------------------------------------------------------------------------------------
# run (scaling)
# ---------------------------------------------------------------------------------------------
def _machine() -> dict:
    import psutil

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cores_available": pipeline.available_cores(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "loadavg_start": list(os.getloadavg()),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
    }


def cmd_run(a) -> None:
    """Run the pipeline for each layout on the same hours; write JSON and markdown."""
    pipeline.configure_logging(a.log_level, True)
    base = PipelineConfig.from_yaml(
        a.config, source=a.source, weights=a.weights, engine=a.engine, device=a.device, bourgouin_top_m=a.bourgouin_top_m, upload="none", keep_local=True, log_level=a.log_level
    )
    hours = pipeline.parse_hours(a.hours)
    runs = pipeline.make_runs(a.init, hours)
    n_hours = sum(len(r.hours) for r in runs)
    work = Path(a.work_dir)
    rows = []
    for layout in a.layouts:
        p, t = (int(v) for v in layout.lower().split("x"))
        wd = work / f"layout_{p}x{t}"
        shutil.rmtree(wd, ignore_errors=True)
        cfg = base.replace(processes=p, threads_per_process=t, work_dir=str(wd))
        load0 = os.getloadavg()[0]
        t0 = time.perf_counter()
        manifests = pipeline.run_pipeline(cfg, runs, no_upload=True)
        wall = time.perf_counter() - t0
        s = pipeline.summarize_timings(manifests)
        done = s["n_hours"]
        row = {
            "layout": layout,
            "processes": p,
            "threads": t,
            "cores": p * t,
            "hours": n_hours,
            "hours_done": done,
            "wall_s": wall,
            "wall_per_hour_s": wall / max(done, 1),
            "stages_median_s": {k: s[k]["median"] for k in (*pipeline.STAGES, "total") if k in s},
            "peak_rss_gb_max_worker": s.get("peak_rss_gb"),
            "payload_mib_per_hour": s.get("payload_mib_mean"),
            "finalize_s": sum(float(m.get("finalize", {}).get("seconds", 0.0)) for m in manifests),
            "loadavg_1min_before": load0,
            "levels": sorted({tuple(r["levels"]["levels"]) for m in manifests for r in m.get("hours", {}).values() if r.get("status") == "done"}),
            "rereads": sum(r["levels"]["rereads"] for m in manifests for r in m.get("hours", {}).values() if r.get("status") == "done"),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not a.keep:
            shutil.rmtree(wd, ignore_errors=True)
    result = {"date": datetime.now(UTC).isoformat(), "machine": _machine(), "config": base.to_dict(), "inits": a.init, "hours": a.hours, "rows": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
    out.with_suffix(".md").write_text(run_markdown(result))
    print(run_markdown(result))


def run_markdown(result: dict) -> str:
    """Markdown table of a ``run`` result."""
    m = result["machine"]
    lines = [
        f"Machine: {m['host']} ({m['platform']}), {m['cores_available']} cores available, {m['memory_gb']} GB. Load average at start {m['loadavg_start']}.",
        f"Engine {result['config']['engine']} on {result['config']['device']}; source {result['config']['source']}; inits {result['inits']} hours {result['hours']};"
        + f" Bourgouin top {result['config']['bourgouin_top_m']:g} m.",
        "",
        "| layout PxT | cores | hours done | wall s | wall s / hour | fetch | decode | columns | inference | bourgouin | rotate | write | hour total | peak RSS GB / worker | MiB fetched / hour | finalize s (per run) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        s = r["stages_median_s"]
        cells = [pipeline.fmt_seconds(s.get(k, float("nan"))) for k in (*pipeline.STAGES, "total")]
        rss = f"{r['peak_rss_gb_max_worker']:.2f}" if r["peak_rss_gb_max_worker"] else "-"
        mib = f"{r['payload_mib_per_hour']:.0f}" if r["payload_mib_per_hour"] else "-"
        lines.append(
            f"| {r['layout']} | {r['cores']} | {r['hours_done']}/{r['hours']} | {r['wall_s']:.1f} | {r['wall_per_hour_s']:.2f} | "
            + " | ".join(cells)
            + f" | {rss} | {mib} | {r.get('finalize_s', 0.0):.2f} |"
        )
    lines.append("")
    lines.append(
        "Stage columns are medians over hours of the per-hour stage seconds inside a worker; 'wall s / hour' is the node throughput (wall / hours,"
        " including worker start-up, model load and the parent's finalize of the gridded store)."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# bourgouin-depth
# ---------------------------------------------------------------------------------------------
def cmd_depth(a) -> None:
    """Measure the effect of the Bourgouin grid top on one hour (see module docstring)."""
    tops = [float(t) for t in a.tops]
    deepest = max(tops)
    with open(a.cache, "rb") as f:
        c = pickle.load(f)  # our own cache file
    entries = reader.parse_idx(c["idx"])
    fxx = c["fxx"]
    key = a.key or c["key"]
    full = reader.select_messages(entries, n_levels=c["n_levels"], fxx=fxx)
    by_off = {m.entry.offset: b for m, b in zip(full.messages, c["messages"], strict=True)}
    cfg0 = PipelineConfig(source="s3", n_levels=a.n_levels, coalesce=a.coalesce)
    # Decode thermo levels 1..n_levels (max in the cache) once; heights decide how many each top needs.
    hour = pipeline.GribHour(None, c["key"], entries, fxx, cfg0)
    plan = hour.plan((0, a.n_levels), (0, c["n_levels"]), True)
    # reader.decode sizes every cube to the plan's max level count; only the first n_levels of UGRD/VGRD are filled.
    t0 = time.perf_counter()
    fields = reader.decode(plan, [by_off[m.entry.offset] for m in plan.messages], workers=a.threads)
    decode_all_s = time.perf_counter() - t0
    z_agl = fields["HGT"] - fields["HGT_sfc"][None]
    min_agl = z_agl.reshape(z_agl.shape[0], -1).min(axis=1)
    mask = np.zeros(fields["CRAIN"].shape, bool)
    for k in pipeline.CATEGORICAL:
        mask |= fields[k] == 1
    mask = mask.reshape(-1)
    snd, n_short = columns.build_soundings(
        z_agl[: a.n_levels], fields["TMP"][: a.n_levels], fields["SPFH"][: a.n_levels], fields["PRES"][: a.n_levels], fields["UGRD"][: a.n_levels], fields["VGRD"][: a.n_levels],
        fields["TMP_2m"], fields["DPT_2m"], fields["UGRD_10m"], fields["VGRD_10m"], fields["PRES_sfc"],
    )  # fmt: skip
    if n_short:
        raise SystemExit(f"{n_short} columns below 5000 m with {a.n_levels} levels")
    # Bytes: size of every thermo message on levels n_levels+1..k (what the extra levels cost to fetch).
    sizes = {(m.name, m.level_index): m.entry.size for m in full.messages if m.level_index is not None}
    store = reader.open_store(None) if a.fetch_s3 else None
    results = {}
    import numba

    numba.set_num_threads(a.threads)
    for top in tops:
        k_need = int(np.argmax(min_agl >= top)) + 1 if np.any(min_agl >= top) else None
        if k_need is None:
            raise SystemExit(f"no level reaches {top} m everywhere")
        n_th = max(a.n_levels, k_need)
        cfg = cfg0.replace(bourgouin_top_m=top, bourgouin_rows_per_chunk=a.rows)
        sub = {k: fields[k][:n_th] for k in ("TMP", "SPFH", "PRES")} | {k: fields[k] for k in ("TMP_2m", "DPT_2m", "PRES_sfc")}
        best = None
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            probs, cls, diag, ns = pipeline.run_bourgouin(sub, np.ascontiguousarray(z_agl[:n_th]), snd, cfg, return_diagnostics=True)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        extra_msgs = [(v, k) for v in pipeline.THERMO_VARIABLES for k in range(a.n_levels, n_th)]
        extra_bytes = int(sum(sizes[m] for m in extra_msgs))
        fetch_s = None
        if store is not None and extra_msgs:
            msgs = sorted((m for m in full.messages if (m.name, m.level_index) in set(extra_msgs)), key=lambda m: m.entry.offset)
            p = reader.ReadPlan(messages=msgs, n_levels=n_th, fxx=fxx)
            t0 = time.perf_counter()
            reader.fetch_ranges(store, key, p, coalesce=a.coalesce)
            fetch_s = time.perf_counter() - t0
        extra_decode = None
        if extra_msgs:
            msgs = [m for m in full.messages if (m.name, m.level_index) in set(extra_msgs)]
            t0 = time.perf_counter()
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(a.threads) as ex:
                list(ex.map(lambda m: reader.decode_message(by_off[m.entry.offset]), msgs))
            extra_decode = time.perf_counter() - t0
        gen_top = diag[:, 5]
        touches = np.isclose(gen_top, top)
        results[top] = {
            "levels_needed": k_need,
            "thermo_levels_read": n_th,
            "min_agl_top_level_m": float(min_agl[n_th - 1]),
            "n_short": int(ns),
            "bourgouin_s": best,
            "extra_payload_bytes": extra_bytes,
            "extra_fetch_s": fetch_s,
            "extra_decode_s": extra_decode,
            "touch_all": float(touches.mean()),
            "touch_mask": float(touches[mask].mean()),
            "prob_ice": diag[:, 0].copy(),
            "cls": cls.copy(),
            "probs": probs.copy(),
        }
        print(json.dumps({k: v for k, v in results[top].items() if not isinstance(v, np.ndarray)}), flush=True)
    ref = results[deepest]
    rows = []
    for top in tops:
        r = results[top]
        dpi = np.abs(r["prob_ice"] - ref["prob_ice"])
        dcls = r["cls"] != ref["cls"]
        dprob = np.abs(r["probs"] - ref["probs"]).max(axis=1)
        rows.append(
            {
                "top_m": top,
                **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)},
                "probice_changed_all": float((dpi > 1e-3).mean()),
                "probice_changed_mask": float((dpi[mask] > 1e-3).mean()),
                "probice_mean_abs_change_mask_pct": float(dpi[mask].mean()),
                "probice_changed_by_50pct_mask": float((dpi[mask] >= 50).mean()),
                "class_changed_all": float(dcls.mean()),
                "class_changed_mask": float(dcls[mask].mean()),
                "max_prob_change_mask_gt_0.1": float((dprob[mask] > 0.1).mean()),
                "max_prob_change_mask_gt_0.01": float((dprob[mask] > 0.01).mean()),
                "max_prob_change_mask_max": float(dprob[mask].max()),
            }
        )
    out = {
        "date": datetime.now(UTC).isoformat(),
        "machine": _machine(),
        "file": key,
        "n_columns": int(mask.size),
        "n_mask": int(mask.sum()),
        "decode_all_levels_s": decode_all_s,
        "min_agl_by_level_m": [float(v) for v in min_agl],
        "rows": rows,
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(json.dumps(rows, indent=1))


def main(argv=None) -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mirror", help="download needed byte ranges into a sparse local mirror")
    m.add_argument("--init", nargs="*", default=[], help="YYYYMMDDHH (repeatable)")
    m.add_argument("--hours", default="0")
    m.add_argument("--n-levels", type=int, default=27)
    m.add_argument("--thermo-levels", type=int, default=34)
    m.add_argument("--coalesce", type=int, default=1024 * 1024)
    m.add_argument("--from-cache", default=None)
    m.add_argument("--key", default=None, help="bucket key to file the cached hour under (default: the key stored in the cache)")
    m.add_argument("--out", required=True)
    r = sub.add_parser("run", help="end-to-end timing for processes x threads layouts")
    r.add_argument("--config", default="config/hrrr_fast.yml")
    r.add_argument("--source", default=None)
    r.add_argument("--weights", default=None)
    r.add_argument("--engine", default=None)
    r.add_argument("--device", default=None)
    r.add_argument("--bourgouin-top-m", type=float, default=None)
    r.add_argument("--init", nargs="+", required=True)
    r.add_argument("--hours", required=True, help="e.g. 0-18 or 11-13")
    r.add_argument("--layouts", nargs="+", default=["1x1"], help="PxT, e.g. 1x1 2x4 16x8")
    r.add_argument("--work-dir", required=True)
    r.add_argument("--keep", action="store_true", help="keep the run directories")
    r.add_argument("--log-level", default="WARNING")
    r.add_argument("--out", required=True, help="output path stem (.json and .md are written)")
    d = sub.add_parser("bourgouin-depth", help="effect of the Bourgouin grid top on one cached hour")
    d.add_argument("--cache", required=True, help="pickle with >= 40 levels (reader_benchmark.py fetch --n-levels 50 --cache)")
    d.add_argument("--tops", nargs="+", default=["5000", "8000", "10000", "12000"])
    d.add_argument("--n-levels", type=int, default=27)
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--rows", type=int, default=128)
    d.add_argument("--repeat", type=int, default=2)
    d.add_argument("--coalesce", type=int, default=1024 * 1024)
    d.add_argument("--key", default=None, help="bucket key of the cached file (default: the key stored in the cache)")
    d.add_argument("--fetch-s3", action="store_true", help="also time fetching the extra levels from the NOAA bucket")
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"mirror": cmd_mirror, "run": cmd_run, "bourgouin-depth": cmd_depth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
