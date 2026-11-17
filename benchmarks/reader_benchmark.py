"""Benchmark and validate ``ptype.hrrr_fast.reader`` on real HRRR wrfnat files.

Sub-commands (each prints Markdown tables that are copied into ``benchmarks/results/reader_local.md``):

``fetch``
    idx download time, and range-fetch time / request count / bytes for one file under several
    ``coalesce`` settings. obstore does not report transfer statistics, so request counts and bytes
    are measured two ways: (a) by replaying the identical ``get_ranges`` call against a local HTTP
    server that logs every ``Range`` header (exact count of the requests object_store issues), and
    (b) with :func:`ptype.hrrr_fast.reader.merged_ranges`, which mirrors object_store's merge rule.
``decode``
    Decode the selected messages with every decoder, serially and with thread/process pools of
    1, 2, 4 and 8 workers. A thread pool that scales means the decoder releases the GIL.
``compare``
    Check that eccodes and gribberish reproduce pygrib values exactly (max |difference| per field,
    NaN-mask agreement), and that grid metadata matches pyproj.
``agl``
    Height above ground of every hybrid level (HGT(level) - HGT(surface)) in two files, to choose N.

The ``--cache`` option stores fetched message bytes as a pickle so ``decode``/``compare`` can be
rerun without the network. Keep caches outside the repository.

Example (from the repository root)::

    python benchmarks/reader_benchmark.py fetch --date 2023-12-26 --cycle 0 --fxx 12 --n-levels 30
    python benchmarks/reader_benchmark.py decode --cache /path/f12_n30.pkl --n-levels 30
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptype.hrrr_fast import reader

COALESCE_VALUES = (0, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024)


def _mib(n: float) -> str:
    return f"{n / 2**20:,.1f}"


# --------------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------------
class _RangeLogger(BaseHTTPRequestHandler):
    """Serve zero bytes for any Range of a file of fixed size and log every request.

    Only the request pattern matters for counting, so no real data is needed: the server pretends
    to hold a file of ``size`` bytes and answers each ``Range: bytes=a-b`` with ``b-a+1`` zeros.
    """

    size = 0
    log: ClassVar[list[tuple[str, int, int]]] = []
    lock = threading.Lock()

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _range(self) -> tuple[int, int]:
        spec = self.headers.get("Range")
        if not spec:
            return 0, self.size - 1
        a, b = spec.split("=")[1].split("-")
        return int(a), min(int(b), self.size - 1)

    def do_HEAD(self):
        with self.lock:
            self.log.append(("HEAD", 0, 0))
        self.send_response(200)
        self.send_header("Content-Length", str(self.size))
        self.send_header("Last-Modified", "Tue, 26 Dec 2023 00:00:00 GMT")
        self.send_header("ETag", '"x"')
        self.end_headers()

    def do_GET(self):
        a, b = self._range()
        with self.lock:
            self.log.append(("GET", a, b + 1))
        n = b - a + 1
        self.send_response(206 if self.headers.get("Range") else 200)
        self.send_header("Content-Length", str(n))
        self.send_header("Content-Range", f"bytes {a}-{b}/{self.size}")
        self.send_header("Last-Modified", "Tue, 26 Dec 2023 00:00:00 GMT")
        self.send_header("ETag", '"x"')
        self.end_headers()
        chunk = bytes(min(n, 8 * 2**20))
        while n > 0:
            k = min(n, len(chunk))
            self.wfile.write(chunk[:k])
            n -= k


def count_requests(plan: reader.ReadPlan, file_size: int, coalesce: int) -> tuple[int, int]:
    """Replay ``fetch_ranges`` against a logging HTTP server; return (GET requests, bytes requested)."""
    from obstore.store import HTTPStore

    _RangeLogger.size = file_size
    _RangeLogger.log = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeLogger)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        store = HTTPStore(f"http://127.0.0.1:{srv.server_address[1]}", client_options={"allow_http": True})
        reader.fetch_ranges(store, "file.grib2", plan, coalesce=coalesce)
    finally:
        srv.shutdown()
    gets = [(a, b) for kind, a, b in _RangeLogger.log if kind == "GET"]
    return len(gets), sum(b - a for a, b in gets)


def cmd_fetch(args) -> None:
    """Time idx + range fetches for several coalesce values."""
    import obstore

    store = reader.open_store(args.source)
    key = reader.hrrr_key(args.date, args.cycle, args.fxx)
    file_size = int(obstore.head(store, key)["size"])
    idx_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        idx = reader.fetch_idx(store, key)
        idx_times.append(time.perf_counter() - t0)
    entries = reader.parse_idx(idx)
    plan = reader.select_messages(entries, n_levels=args.n_levels, fxx=args.fxx)
    ends = [m.entry.end for m in plan.messages]
    print(f"key={key}  messages in file={len(entries)}  full file={file_size:,} B ({_mib(file_size)} MiB)  idx bytes={len(idx):,}")
    print(f"idx fetch s: {', '.join(f'{t:.3f}' for t in idx_times)} (median {statistics.median(idx_times):.3f})")
    print(f"N={args.n_levels}: {len(plan.messages)} messages, {plan.total_bytes():,} B needed ({100 * plan.total_bytes() / file_size:.1f}% of file)\n")
    print("| coalesce | GET requests (logged) | bytes requested (logged) | merged_ranges reqs / bytes | % of file | fetch s (each run) | median s | MiB/s |")
    print("|---|---|---|---|---|---|---|---|")
    cached = None
    for c in args.coalesce:
        n_req, n_bytes = count_requests(plan, file_size, c)
        merged = reader.merged_ranges(plan.starts, ends, c)
        m_bytes = sum(b - a for a, b in merged)
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            msgs = reader.fetch_ranges(store, key, plan, coalesce=c)
            times.append(time.perf_counter() - t0)
            assert sum(len(m) for m in msgs) == plan.total_bytes()
            cached = msgs
        med = statistics.median(times)
        print(
            f"| {c:,} | {n_req} | {n_bytes:,} | {len(merged)} / {m_bytes:,} | {100 * n_bytes / file_size:.1f} | "
            f"{', '.join(f'{t:.2f}' for t in times)} | {med:.2f} | {n_bytes / 2**20 / med:.1f} |"
        )
    if args.cache and cached is not None:
        with open(args.cache, "wb") as f:
            pickle.dump({"key": key, "fxx": args.fxx, "idx": idx, "n_levels": args.n_levels, "messages": [bytes(m) for m in cached]}, f)
        print(f"\ncached {len(cached)} messages to {args.cache}")


# --------------------------------------------------------------------------------------------
# decode / compare helpers
# --------------------------------------------------------------------------------------------
def _load_cache(path: str, n_levels: int | None = None) -> tuple[reader.ReadPlan, list[bytes]]:
    """Load cached messages; optionally re-plan for fewer levels using the cached superset."""
    with open(path, "rb") as f:
        c = pickle.load(f)
    entries = reader.parse_idx(c["idx"])
    full_plan = reader.select_messages(entries, n_levels=c["n_levels"], fxx=c["fxx"])
    by_offset = {m.entry.offset: b for m, b in zip(full_plan.messages, c["messages"], strict=True)}
    plan = full_plan if n_levels is None else reader.select_messages(entries, n_levels=n_levels, fxx=c["fxx"])
    return plan, [by_offset[m.entry.offset] for m in plan.messages]


def cmd_decode(args) -> None:
    """Time decode with each decoder for serial, thread and process pools."""
    plan, msgs = _load_cache(args.cache, args.n_levels)
    print(f"{len(msgs)} messages, {sum(map(len, msgs)):,} B compressed, N={plan.n_levels}; best-of-{args.repeat} wall seconds\n")
    configs = [("serial", 1)] + [(p, w) for p in ("thread", "process") for w in args.workers]
    print("| decoder | " + " | ".join(f"{p} {w}" if p != "serial" else "serial" for p, w in configs) + " |")
    print("|---" * (len(configs) + 1) + "|")
    for dec in args.decoders:
        reader.decode_message(msgs[0], dec)  # import the decoder once so the first timing is not an import
        row = []
        for pool, w in configs:
            times = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                reader.decode(plan, msgs, decoder=dec, workers=w if pool != "serial" else 1, pool=pool if pool != "serial" else "thread")
                times.append(time.perf_counter() - t0)
            row.append(f"{min(times):.2f}")
        print(f"| {dec} | " + " | ".join(row) + " |", flush=True)


def cmd_compare(args) -> None:
    """Compare eccodes and gribberish to pygrib, message by message."""
    plan, msgs = _load_cache(args.cache, args.n_levels)
    ref = reader.decode(plan, msgs, decoder="pygrib", workers=8)
    others = {d: reader.decode(plan, msgs, decoder=d, workers=8) for d in ("eccodes", "gribberish")}
    print("| field | shape | pygrib NaNs | " + " | ".join(f"{d} max abs diff | {d} NaN mask equal" for d in others) + " |")
    print("|---" * (3 + 2 * len(others)) + "|")
    for name, r in ref.items():
        cells = []
        for out in others.values():
            o = out[name]
            same_nan = bool(np.array_equal(np.isnan(r), np.isnan(o)))
            diff = np.nanmax(np.abs(r.astype(np.float64) - o.astype(np.float64))) if np.isfinite(r).any() else 0.0
            cells += [f"{diff:.3g}", str(same_nan)]
        print(f"| {name} | {r.shape} | {int(np.isnan(r).sum())} | " + " | ".join(cells) + " |")
    # float32 vs float64 view: are the float64 decodes identical too?
    import eccodes
    import gribberish
    import pygrib

    worst = {"eccodes": 0.0, "gribberish": 0.0}
    for b in msgs:
        p = np.ma.filled(pygrib.fromstring(b).values.astype(np.float64), np.nan).ravel()
        gid = eccodes.codes_new_from_message(b)
        e = eccodes.codes_get_values(gid)
        eccodes.codes_release(gid)
        g = np.asarray(gribberish.parse_grib_array(b, 0, north_up=False))
        worst["eccodes"] = max(worst["eccodes"], float(np.nanmax(np.abs(p - e))))
        worst["gribberish"] = max(worst["gribberish"], float(np.nanmax(np.abs(p - g))))
    print(f"\nfloat64 (before float32 cast) worst max abs diff vs pygrib over all {len(msgs)} messages: {worst}")

    g = reader.grid_metadata(msgs[0])
    print(f"\ngrid metadata: {g}")
    try:
        import pyproj

        p = pyproj.Proj(g.proj4)
        x, y = p(g.lon_first, g.lat_first)
        print(f"pyproj first point x,y = {x:.3f}, {y:.3f}; reader = {g.x_first:.3f}, {g.y_first:.3f}")
        lats, lons = pygrib.fromstring(msgs[0]).latlons()
        xs, ys = g.xy_coords()
        px, py = p(lons, lats)
        print(f"max |pyproj(x,y of pygrib lat/lon) - reader xy_coords| = {np.abs(px - xs[None, :]).max():.4f} m, {np.abs(py - ys[:, None]).max():.4f} m")
        print(f"pygrib lat row0 mean {lats[0].mean():.2f}, last row mean {lats[-1].mean():.2f} (row0_is_south={g.row0_is_south})")
    except ImportError:
        print("pyproj not installed; skipped projection check")


# --------------------------------------------------------------------------------------------
# agl
# --------------------------------------------------------------------------------------------
def _hgt_plan(entries, fxx):
    return reader.select_messages(entries, n_levels=50, fxx=fxx, hybrid_variables=("HGT",), surface_fields={"HGT_sfc": ("HGT", "surface")}, include_apcp=False)


def cmd_agl(args) -> None:
    """Height AGL statistics per hybrid level for several files."""
    store = reader.open_store(args.source)
    stats = {}
    for spec in args.files:
        d, cyc, fxx = spec.split(",")
        dd = date.fromisoformat(d)
        key = reader.hrrr_key(dd, int(cyc), int(fxx))
        plan = _hgt_plan(reader.parse_idx(reader.fetch_idx(store, key)), int(fxx))
        out = reader.decode(plan, reader.fetch_ranges(store, key, plan), decoder="eccodes", workers=8)
        agl = out["HGT"] - out["HGT_sfc"][None]
        stats[key] = [(float(a.min()), float(np.median(a)), float(a.max())) for a in agl]
        first = next((i for i, s in enumerate(stats[key]) if s[0] >= args.threshold), None)
        print(f"{key}: smallest level index (0-based) with min AGL >= {args.threshold:g} m: {first} (hybrid level {None if first is None else first + 1})")
    keys = list(stats)
    print("\n| index (0-based) | hybrid level | " + " | ".join(f"{k.split('/')[0]} {k.split('.')[-2]} min / median / max (m)" for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for i in range(50):
        print(f"| {i} | {i + 1} | " + " | ".join(f"{stats[k][i][0]:.0f} / {stats[k][i][1]:.0f} / {stats[k][i][2]:.0f}" for k in keys) + " |")


def main(argv=None) -> None:
    """Parse arguments and dispatch to a sub-command."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--date", type=date.fromisoformat, default=date(2023, 12, 26))
    f.add_argument("--cycle", type=int, default=0)
    f.add_argument("--fxx", type=int, default=12)
    f.add_argument("--n-levels", type=int, default=30)
    f.add_argument("--source", default=None, help="local mirror directory (default: public S3 bucket)")
    f.add_argument("--coalesce", type=int, nargs="+", default=list(COALESCE_VALUES))
    f.add_argument("--repeat", type=int, default=2)
    f.add_argument("--cache", default=None)
    d = sub.add_parser("decode")
    d.add_argument("--cache", required=True)
    d.add_argument("--n-levels", type=int, default=None)
    d.add_argument("--decoders", nargs="+", default=["eccodes", "pygrib", "gribberish"])
    d.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    d.add_argument("--repeat", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("--cache", required=True)
    c.add_argument("--n-levels", type=int, default=None)
    a = sub.add_parser("agl")
    a.add_argument("--files", nargs="+", default=["2023-12-26,0,12", "2025-07-15,18,6"], help="date,cycle,fxx triples")
    a.add_argument("--threshold", type=float, default=5000.0)
    a.add_argument("--source", default=None)
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "decode": cmd_decode, "compare": cmd_compare, "agl": cmd_agl}[args.cmd](args)


if __name__ == "__main__":
    main()
