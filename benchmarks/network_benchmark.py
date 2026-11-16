"""Measure what limits download throughput of HRRR byte-range reads on one compute node.

The HRRR pipeline turned out to be network-bound on Derecho, but one fetch alone is fast while many
concurrent fetches are slow. This script separates the candidate limits (per connection, per process,
per node, shared gateway) by running controlled loads and recording only measured quantities.

Sub-commands:

``sweep``
    Start N worker processes (spawned, so they share nothing), each fetching the *same message selection the
    pipeline uses* from a *different* forecast hour with ``obstore.get_ranges``, discarding the bytes. Workers
    build their stores and read the ``.idx`` files before a common start time, so the timed region is only the
    range downloads. Reports aggregate throughput, per-process throughput and per-fetch times.
``curl``
    Whole-file downloads with ``curl`` (N parallel streams of different files) as a reference for raw link
    speed that involves neither obstore nor range requests.
``compare-sources``
    Check that the Google Cloud mirror has identical ``.idx`` text, file sizes and bytes (SHA-256 of every
    selected range) as the AWS bucket, so the mirror can be used with the same offsets.

Every test appends one record to the ``--out`` JSON list. Only measured values are written; throughput is
decimal MB/s (1e6 bytes) unless a key says ``mib``.

Example (on a compute node)::

    python benchmarks/network_benchmark.py sweep --source aws --procs 1 2 4 8 --out net.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptype.hrrr_fast import reader

#: Message counts per hybrid variable used by the production pipeline.
PIPELINE_LEVELS = {"TMP": 34, "SPFH": 34, "PRES": 34, "HGT": 34, "UGRD": 27, "VGRD": 27}
DATE = "2023-12-26"
CYCLE = 0
N_HOURS = 32  # f00..f31
AWS_HTTP_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
GCS_BUCKET = "high-resolution-rapid-refresh"
GCS_HTTP_BASE = "https://storage.googleapis.com/high-resolution-rapid-refresh"
SOURCES = ("aws", "aws-http", "gcs", "gcs-http")


def _parse_option(text: str) -> tuple[str, object]:
    """Parse ``key=value`` for ``client_options``; ``true``/``false`` become bool because obstore's flags are bool."""
    key, _, value = text.partition("=")
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    return key, value


def make_store(source: str, client_options: dict | None = None):
    """Build an anonymous obstore store for one of the sources.

    Args:
        source: One of :data:`SOURCES`.
        client_options: Passed to obstore as ``client_options`` (connection pool, HTTP version, timeouts).

    Returns:
        An obstore store; anonymous access means ``skip_signature=True`` (no credentials are read or sent).
    """
    from obstore.store import GCSStore, HTTPStore, S3Store

    kw = {"client_options": client_options} if client_options else {}
    if source == "aws":
        return S3Store(reader.HRRR_BUCKET, region=reader.HRRR_REGION, skip_signature=True, **kw)
    if source == "aws-http":
        return HTTPStore.from_url(AWS_HTTP_BASE, **kw)
    if source == "gcs":
        return GCSStore(GCS_BUCKET, skip_signature=True, **kw)
    if source == "gcs-http":
        return HTTPStore.from_url(GCS_HTTP_BASE, **kw)
    raise ValueError(f"unknown source {source!r}")


def hour_key(fxx: int) -> str:
    """Object key of the 2023-12-26 00Z wrfnat file for forecast hour ``fxx``."""
    return f"hrrr.20231226/conus/hrrr.t{CYCLE:02d}z.wrfnatf{fxx:02d}.grib2"


def prepare_ranges(store, fxx: int) -> tuple[str, list[int], list[int]]:
    """Fetch the idx and return ``(key, starts, ends)`` for the pipeline's message selection of one hour.

    The last message has no end in the idx; one HEAD resolves it exactly as :func:`reader.fetch_ranges` does.
    This is done before the timed region so only range downloads are timed.
    """
    import obstore

    key = hour_key(fxx)
    entries = reader.parse_idx(reader.fetch_idx(store, key))
    plan = reader.select_messages(entries, n_levels=PIPELINE_LEVELS, fxx=fxx)
    ends = []
    for m in plan.messages:
        ends.append(int(obstore.head(store, key)["size"]) if m.entry.end is None else m.entry.end)
    return key, plan.starts, ends


def fetch_discard(store, key: str, starts: list[int], ends: list[int], threads: int, coalesce: int) -> None:
    """Download the ranges and drop the bytes.

    With ``threads > 1`` the ranges are split into contiguous groups, one ``get_ranges`` call per thread. obstore
    already issues up to 10 requests in parallel inside one call, so threads multiply that concurrency.
    """
    import obstore

    if threads <= 1:
        obstore.get_ranges(store, key, starts=starts, ends=ends, coalesce=coalesce)
        return
    bounds = [round(i * len(starts) / threads) for i in range(threads + 1)]
    groups = [(starts[a:b], ends[a:b]) for a, b in itertools.pairwise(bounds) if b > a]
    with ThreadPoolExecutor(len(groups)) as pool:
        list(pool.map(lambda g: obstore.get_ranges(store, key, starts=g[0], ends=g[1], coalesce=coalesce), groups))


def _worker(i: int, cfg: dict, ready: mp.Queue, start_val, results: mp.Queue) -> None:
    """One downloading process: prepare, signal ready, wait for the common start, fetch files until the time is up.

    Distinct forecast hours per process (and per file) are chosen by ``cfg`` so that no two fetches in one test
    request the same object. Prepared (idx, HEAD) work is done before ``ready`` so it is not timed.
    """
    store = make_store(cfg["source"], cfg["client_options"])
    n = cfg["n_procs"]
    hours = [(cfg["fxx_offset"] + i + k * n) % N_HOURS for k in range(cfg["max_files"])]
    prepared = [(fxx, *prepare_ranges(store, fxx)) for fxx in hours]
    ready.put(i)
    while start_val.value == 0.0:
        time.sleep(0.01)
    start = start_val.value
    while time.time() < start:
        time.sleep(0.001)
    events = []
    for fxx, key, starts, ends in prepared:
        if events and time.time() - start >= cfg["duration"]:
            break
        t0 = time.time()
        fetch_discard(store, key, starts, ends, cfg["threads"], cfg["coalesce"])
        t1 = time.time()
        merged = reader.merged_ranges(starts, ends, cfg["coalesce"])
        events.append(
            {
                "fxx": fxx,
                "t0": t0,
                "t1": t1,
                "useful_bytes": sum(e - s for s, e in zip(starts, ends, strict=True)),
                "wire_bytes": sum(e - s for s, e in merged),
                "n_ranges": len(starts),
                "n_requests": len(merged),
            }
        )
    results.put((i, events))


def _summarize(cfg: dict, start: float, per_proc: dict[int, list[dict]], ready_late_s: float) -> dict:
    """Turn per-file events into the record written to JSON (aggregate over [common start, last finish])."""
    events = [e for ev in per_proc.values() for e in ev]
    end = max(e["t1"] for e in events)
    wall = end - start
    useful = sum(e["useful_bytes"] for e in events)
    wire = sum(e["wire_bytes"] for e in events)
    durs = [e["t1"] - e["t0"] for e in events]
    proc_rates = [sum(e["wire_bytes"] for e in ev) / (max(e["t1"] for e in ev) - start) / 1e6 for ev in per_proc.values()]
    return {
        **{k: cfg[k] for k in ("source", "n_procs", "threads", "coalesce", "client_options", "max_files", "duration", "fxx_offset", "label")},
        "start_epoch": start,
        "start_late_s": ready_late_s,
        "wall_s": wall,
        "n_files": len(events),
        "useful_bytes": useful,
        "wire_bytes": wire,
        "agg_wire_MBps": wire / wall / 1e6,
        "agg_useful_MBps": useful / wall / 1e6,
        "agg_wire_MiBps": wire / wall / 2**20,
        "per_proc_wire_MBps": {"min": min(proc_rates), "median": statistics.median(proc_rates), "max": max(proc_rates)},
        "fetch_s": {"min": min(durs), "median": statistics.median(durs), "max": max(durs), "all": sorted(round(d, 2) for d in durs)},
        "requests_per_file": events[0]["n_requests"],
        "host": socket.gethostname(),
        "job": os.environ.get("PBS_JOBID", ""),
        "unix_time": time.time(),
    }


def append_record(path: str | None, record: dict) -> None:
    """Append ``record`` to the JSON list at ``path`` (created if absent); rewritten each time so partial runs survive."""
    if not path:
        return
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else []
    data.append(record)
    p.write_text(json.dumps(data, indent=1))


def run_sweep_one(cfg: dict) -> dict:
    """Run one concurrency level: spawn workers, wait for readiness, release them at a common start, collect."""
    import obstore

    ctx = mp.get_context("spawn")
    ready, results = ctx.Queue(), ctx.Queue()
    start_val = ctx.Value("d", 0.0)
    procs = [ctx.Process(target=_worker, args=(i, cfg, ready, start_val, results)) for i in range(cfg["n_procs"])]
    for p in procs:
        p.start()
    for _ in range(cfg["n_procs"]):
        ready.get(timeout=600)
    now = time.time()
    start = max(cfg["start_at"] or 0.0, now + 2.0)
    late = max(0.0, now + 2.0 - cfg["start_at"]) if cfg["start_at"] else 0.0
    start_val.value = start
    per_proc = {}
    for _ in range(cfg["n_procs"]):
        i, ev = results.get(timeout=1200)
        per_proc[i] = ev
    for p in procs:
        p.join()
    rec = _summarize(cfg, start, per_proc, late)
    rec["obstore"] = obstore.__version__
    return rec


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run the requested concurrency levels one after another and print a Markdown row per level."""
    opts = dict(_parse_option(o) for o in args.client_option)
    print("| label | source | procs | threads | coalesce | files | wall s | agg wire MB/s | agg useful MB/s | per-proc MB/s (min/med/max) | fetch s (min/med/max) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in args.procs:
        cfg = {
            "source": args.source,
            "n_procs": n,
            "threads": args.threads,
            "coalesce": args.coalesce,
            "client_options": opts,
            "max_files": max(1, min(args.max_files, N_HOURS // n)),
            "duration": args.duration,
            "fxx_offset": args.fxx_offset,
            "start_at": args.start_at,
            "label": args.label,
        }
        rec = run_sweep_one(cfg)
        append_record(args.out, rec)
        pp, fs = rec["per_proc_wire_MBps"], rec["fetch_s"]
        print(
            f"| {args.label} | {args.source} | {n} | {args.threads} | {args.coalesce} | {rec['n_files']} | {rec['wall_s']:.1f} | {rec['agg_wire_MBps']:.1f} | {rec['agg_useful_MBps']:.1f} "
            f"| {pp['min']:.1f}/{pp['median']:.1f}/{pp['max']:.1f} | {fs['min']:.1f}/{fs['median']:.1f}/{fs['max']:.1f} |",
            flush=True,
        )
        if args.start_at:
            args.start_at = 0.0  # a fixed start time only makes sense for the first level


def cmd_curl(args: argparse.Namespace) -> None:
    """Whole-file ``curl`` downloads, N parallel streams of different forecast hours, bytes discarded.

    ``-m`` bounds each stream so a slow link cannot overrun the job; ``size_download`` is what actually arrived, so
    throughput is exact bytes over wall time even when a stream is cut off.
    """
    base = AWS_HTTP_BASE if args.source == "aws" else GCS_HTTP_BASE
    for n in args.procs:
        cmds = [
            ["curl", "-s", "-o", "/dev/null", "-m", str(args.max_time), "-w", "%{size_download} %{time_total}", f"{base}/{hour_key((args.fxx_offset + i) % N_HOURS)}"]
            for i in range(n)
        ]
        t0 = time.time()
        ps = [subprocess.Popen(c, stdout=subprocess.PIPE, text=True) for c in cmds]
        outs = [p.communicate()[0].split() for p in ps]
        wall = time.time() - t0
        sizes = [int(o[0]) for o in outs]
        times = [float(o[1]) for o in outs]
        rec = {
            "kind": "curl",
            "source": args.source,
            "n_procs": n,
            "max_time": args.max_time,
            "label": args.label,
            "wall_s": wall,
            "bytes": sum(sizes),
            "agg_MBps": sum(sizes) / wall / 1e6,
            "per_stream_MBps": sorted(round(s / t / 1e6, 1) for s, t in zip(sizes, times, strict=True)),
            "host": socket.gethostname(),
            "job": os.environ.get("PBS_JOBID", ""),
            "unix_time": time.time(),
        }
        append_record(args.out, rec)
        print(f"curl {args.source} streams={n} wall={wall:.1f}s bytes={rec['bytes']} agg={rec['agg_MBps']:.1f} MB/s per-stream={rec['per_stream_MBps']}", flush=True)


def cmd_compare_sources(args: argparse.Namespace) -> None:
    """Compare AWS and GCS: idx text, object size, and SHA-256 of every selected range for ``--hours``."""
    import obstore

    aws, gcs = make_store("aws"), make_store("gcs")
    out = []
    for fxx in args.hours:
        key = hour_key(fxx)
        idx_a, idx_g = reader.fetch_idx(aws, key), reader.fetch_idx(gcs, key)
        size_a, size_g = int(obstore.head(aws, key)["size"]), int(obstore.head(gcs, key)["size"])
        # Variable names can differ between mirrors (e.g. CLMR vs CLWMR), so also compare the byte-offset column and the
        # plan the pipeline would build from each idx: those are what matter for range reads.
        offsets = [[ln.split(":")[1] for ln in t.splitlines()] for t in (idx_a, idx_g)]
        plans = [reader.select_messages(reader.parse_idx(t), n_levels=PIPELINE_LEVELS, fxx=fxx) for t in (idx_a, idx_g)]
        rec = {
            "fxx": fxx,
            "idx_text_identical": idx_a == idx_g,
            "idx_offsets_identical": offsets[0] == offsets[1],
            "idx_lines": len(idx_a.splitlines()),
            "pipeline_plan_identical": [(m.name, m.level_index, m.entry.offset, m.entry.end) for m in plans[0].messages]
            == [(m.name, m.level_index, m.entry.offset, m.entry.end) for m in plans[1].messages],
            "size_aws": size_a,
            "size_gcs": size_g,
        }
        if fxx in args.hash_hours:
            _, starts, ends = prepare_ranges(aws, fxx)
            ra = obstore.get_ranges(aws, key, starts=starts, ends=ends)
            rg = obstore.get_ranges(gcs, key, starts=starts, ends=ends)
            same = [hashlib.sha256(bytes(a)).digest() == hashlib.sha256(bytes(g)).digest() for a, g in zip(ra, rg, strict=True)]
            rec.update({"ranges_compared": len(same), "ranges_identical": sum(same)})
        out.append(rec)
        print(rec, flush=True)
    append_record(args.out, {"kind": "compare-sources", "results": out, "host": socket.gethostname()})


def main() -> None:
    """Parse arguments and dispatch."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help="JSON list file to append records to")
    common.add_argument("--label", default="", help="free-text tag stored with each record")
    common.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    common.add_argument("--fxx-offset", type=int, default=0, help="shift the forecast hours used (to keep two nodes on different files)")
    s = sub.add_parser("sweep", parents=[common])
    s.add_argument("--source", choices=SOURCES, default="aws")
    s.add_argument("--threads", type=int, default=1, help="threads per process, each calling get_ranges on a share of the ranges")
    s.add_argument("--coalesce", type=int, default=1024 * 1024)
    s.add_argument("--duration", type=float, default=30.0, help="seconds after which a process starts no further file")
    s.add_argument("--max-files", type=int, default=4, help="cap on files per process")
    s.add_argument("--start-at", type=float, default=0.0, help="unix time at which all processes start (for synchronising nodes)")
    s.add_argument("--client-option", action="append", default=[], help="obstore client_options entry key=value (repeatable)")
    s.set_defaults(func=cmd_sweep)
    c = sub.add_parser("curl", parents=[common])
    c.add_argument("--source", choices=("aws", "gcs"), default="aws")
    c.add_argument("--max-time", type=int, default=60)
    c.set_defaults(func=cmd_curl)
    k = sub.add_parser("compare-sources")
    k.add_argument("--out")
    k.add_argument("--hours", type=int, nargs="+", default=[0, 1, 12, 24, 31])
    k.add_argument("--hash-hours", type=int, nargs="+", default=[12])
    k.set_defaults(func=cmd_compare_sources)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
