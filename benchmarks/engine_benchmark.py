"""Throughput and max-batch-size benchmark for the HRRR precipitation-type inference engines.

Two measurements for one (engine, device, threads) configuration:

(a) **throughput**: rows/s for batch sizes 2^10, 2^11, ... on a synthetic array with the size of one
    full HRRR CONUS forecast hour (1,905,141 rows x 84 float32). Each size gets one untimed warm-up
    pass over its first batch (JIT / cuDNN / allocator / torch.compile), then ``--repeats`` timed passes
    over the entire array. Median and best rows/s are reported.
(b) **max batch**: a single forward pass over ``2^k`` rows for k = 10, 11, ... until the engine runs out
    of memory (``torch.cuda.OutOfMemoryError`` / ``RuntimeError`` / ``MemoryError`` / onnxruntime
    allocation failures) or ``--max-rows`` is reached. Peak process RSS (sampled with psutil every 5 ms)
    and peak GPU memory (``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``) are recorded per size.

Everything is explicit: ``--device`` and ``--threads`` are never inferred, and the detected hardware
(logical CPUs, scheduler affinity, physical cores, RAM, GPU names) is printed and saved in the JSON so
results from a laptop, a Derecho CPU node and a Derecho A100 node can be compared honestly.

Examples::

    # Derecho CPU node (128 cores), all engines that matter
    python benchmarks/engine_benchmark.py --engine torch --device cpu --threads 128 --weights w/ptype_weights.npz --out results/torch_cpu128
    python benchmarks/engine_benchmark.py --engine onnx  --device cpu --threads 128 --weights w/ptype_weights.npz --out results/onnx_cpu128
    python benchmarks/engine_benchmark.py --engine numba --device cpu --threads 128 --weights w/ptype_weights.npz --out results/numba_cpu128

    # A100 40 GB node
    python benchmarks/engine_benchmark.py --engine torch --device cuda --threads 8 --tf32 --weights w/ptype_weights.npz --out results/torch_a100
    python benchmarks/engine_benchmark.py --engine torch --device cuda --threads 8 --dtype float16 --weights w/ptype_weights.npz --out results/torch_a100_fp16

Outputs ``<out>.json`` and ``<out>.md`` (``--md-append`` appends to an existing markdown file so several
runs can share one report).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

# Allow running as a script from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptype.hrrr_fast.engine import ENGINE_NAMES, N_FEATURES, make_engine

HRRR_ROWS = 1_905_141  # 1059 x 1799 CONUS grid points, one forecast hour (the size requested by the pipeline design)
OOM_WORDS = ("out of memory", "alloc", "memory", "oom", "bad_alloc", "failed to allocate")


def synthetic_rows(n: int, seed: int = 0) -> np.ndarray:
    """Cheap realistic-range synthetic input of shape (n, 84).

    Built by tiling a 65,536-row random block: inference cost is independent of the values, and tiling
    is ~100x faster than drawing 160M normals, which matters for the multi-GB max-batch arrays.
    """
    rng = np.random.default_rng(seed)
    block = 65536
    t0 = rng.uniform(-30, 35, (block, 1))
    z = np.arange(0, 5250, 250) / 1000.0
    t = t0 - rng.uniform(2, 9, (block, 1)) * z + rng.normal(0, 0.5, (block, 21))
    td = t - rng.uniform(0, 20, (block, 21))
    u = rng.normal(2, 10, (block, 21))
    v = rng.normal(1, 10, (block, 21))
    base = np.concatenate([t, td, u, v], axis=1).astype(np.float32)
    reps = -(-n // block)
    return np.tile(base, (reps, 1))[:n]


def detect_hardware() -> dict:
    """Describe the machine so results are self-documenting."""
    hw: dict = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        # A busy shared machine silently depresses throughput; record the 1/5/15-min load averages at start.
        "loadavg_1_5_15": [round(v, 2) for v in psutil.getloadavg()],
        "env": {
            k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PBS_JOBID") if k in os.environ
        },
    }
    if hasattr(os, "sched_getaffinity"):
        hw["cpu_affinity_count"] = len(os.sched_getaffinity(0))  # what a PBS/cgroup job is actually allowed to use
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["cuda_available"] = bool(torch.cuda.is_available())
        hw["mps_available"] = bool(torch.backends.mps.is_available())
        if torch.cuda.is_available():
            hw["gpus"] = [
                {"name": torch.cuda.get_device_name(i), "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)} for i in range(torch.cuda.device_count())
            ]
            hw["cuda_runtime"] = torch.version.cuda
    except ImportError:
        pass
    for mod in ("numpy", "onnxruntime", "numba", "keras"):
        try:
            hw[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 - optional packages: record why a version is missing instead of hiding it
            hw[mod] = f"unavailable ({type(exc).__name__})"
    try:
        import onnxruntime as ort

        hw["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_info

        hw["blas"] = [{"api": i["user_api"], "lib": i.get("internal_api"), "threads": i["num_threads"]} for i in threadpool_info()]
    except ImportError:
        pass
    return hw


class RssSampler:
    """Background thread recording peak RSS of this process.

    ``ru_maxrss`` is only a lifetime high-water mark; sampling gives a per-measurement peak that can
    be compared across increasing sizes. 5 ms resolution is plenty for passes lasting seconds.
    """

    def __init__(self):
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.005)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def maxrss_bytes() -> int:
    """Lifetime peak RSS from the kernel (bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def is_oom(exc: BaseException) -> bool:
    """Classify an exception as an out-of-memory condition for this engine family."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
            return True
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    # RuntimeError covers MPS ("MPS backend out of memory"), CPU allocator failures, and onnxruntime's
    # RuntimeException/Fail ("Failed to allocate memory ..."), which have no dedicated exception class.
    if isinstance(exc, RuntimeError) or type(exc).__name__ in ("RuntimeException", "Fail", "InvalidArgument", "OrtException"):
        return any(w in str(exc).lower() for w in OOM_WORDS)
    return False


def sync_device(device: str) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.startswith("cuda") or device == "mps":
        import torch

        (torch.cuda if device.startswith("cuda") else torch.mps).synchronize()


def gpu_peaks(device: str) -> tuple[float | None, float | None]:
    """(max allocated, max reserved) GPU bytes since the last reset, or Nones off-CUDA."""
    if device.startswith("cuda"):
        import torch

        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    return None, None


def reset_gpu_peaks(device: str) -> None:
    """Reset CUDA peak counters and free cached blocks so each size is measured independently."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_throughput(engine, x: np.ndarray, device: str, log2_min: int, log2_max: int, repeats: int, warmup: int) -> list[dict]:
    """Sweep batch sizes 2^log2_min..2^log2_max (stopping once a batch covers the whole array)."""
    rows = []
    for k in range(log2_min, log2_max + 1):
        bs = 2**k
        try:
            for _ in range(warmup):
                engine.predict(x[: min(bs, len(x))], batch_size=bs)
            sync_device(device)
            reset_gpu_peaks(device)
            times = []
            with RssSampler() as rss:
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    engine.predict(x, batch_size=bs)
                    sync_device(device)
                    times.append(time.perf_counter() - t0)
        except Exception as exc:
            if is_oom(exc):
                rows.append({"batch_size": bs, "status": f"OOM: {str(exc)[:120]}"})
                break
            raise
        alloc, reserved = gpu_peaks(device)
        rows.append(
            {
                "batch_size": bs,
                "status": "ok",
                "times_s": times,
                "rows_per_s_median": len(x) / float(np.median(times)),
                "rows_per_s_best": len(x) / min(times),
                "full_hour_s_median": float(np.median(times)),
                "peak_rss_gb": rss.peak / 1e9,
                "peak_gpu_alloc_gb": None if alloc is None else alloc / 1e9,
                "peak_gpu_reserved_gb": None if reserved is None else reserved / 1e9,
            }
        )
        print(
            f"  batch 2^{k:<2d}={bs:>9,d}  median {rows[-1]['rows_per_s_median']:>12,.0f} rows/s  best {rows[-1]['rows_per_s_best']:>12,.0f}  RSS {rss.peak / 1e9:5.2f} GB",
            flush=True,
        )
        if bs >= len(x):
            break
    return rows


def run_maxbatch(engine, device: str, log2_min: int, max_rows: int, ram_fraction: float) -> list[dict]:
    """Double the single-pass batch until OOM, the row cap, or the host-RAM safety guard."""
    rows = []
    k = log2_min
    while 2**k <= max_rows:
        n = 2**k
        # Rough host footprint of one pass: input + up to ~3 live 200-wide float32 activations + output (CPU engines).
        est = n * 4 * (N_FEATURES + 3 * 200 + 8)
        avail = psutil.virtual_memory().available
        if not device.startswith("cuda") and est > ram_fraction * avail:
            rows.append({"rows": n, "status": f"SKIPPED: estimated {est / 1e9:.1f} GB > {ram_fraction:.0%} of {avail / 1e9:.1f} GB free RAM"})
            break
        gc.collect()
        reset_gpu_peaks(device)
        x = None
        try:
            x = synthetic_rows(n)
            rss_start = psutil.Process().memory_info().rss
            with RssSampler() as rss:
                t0 = time.perf_counter()
                engine.predict(x, batch_size=n)
                sync_device(device)
                dt = time.perf_counter() - t0
        except Exception as exc:
            if is_oom(exc):
                rows.append({"rows": n, "status": f"OOM: {type(exc).__name__}: {str(exc)[:120]}"})
                break
            raise
        finally:
            del x
            gc.collect()
        alloc, reserved = gpu_peaks(device)
        rows.append(
            {
                "rows": n,
                "status": "ok",
                "seconds": dt,
                "rows_per_s": n / dt,
                "peak_rss_gb": rss.peak / 1e9,
                "rss_over_start_gb": (rss.peak - rss_start) / 1e9,
                "peak_gpu_alloc_gb": None if alloc is None else alloc / 1e9,
                "peak_gpu_reserved_gb": None if reserved is None else reserved / 1e9,
            }
        )
        print(f"  rows 2^{k:<2d}={n:>11,d}  {n / dt:>12,.0f} rows/s  RSS {rss.peak / 1e9:6.2f} GB  GPU {'-' if alloc is None else f'{alloc / 1e9:.2f} GB'}", flush=True)
        k += 1
    return rows


def fmt(v, spec=".2f") -> str:
    """Format optional numbers for markdown."""
    return "-" if v is None else format(v, spec)


def to_markdown(result: dict) -> str:
    """Render one run as markdown."""
    a = result["args"]
    cfg = result["engine_config"]
    lines = [f"### {a['label'] or a['engine']}: engine `{a['engine']}`, device `{a['device']}`, threads `{a['threads']}`", "", f"Engine config: `{json.dumps(cfg)}`", ""]
    if result.get("failed"):
        lines += [f"**Run failed:** `{result['failed']}`", ""]
    if result.get("throughput"):
        lines += [
            f"Throughput on {a['n_rows']:,} rows ({'one HRRR CONUS forecast hour' if a['n_rows'] == HRRR_ROWS else 'reduced smoke-test size'}), {a['repeats']} timed repeats after warm-up.",
            "",
            "| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["throughput"]:
            if r["status"] != "ok":
                lines.append(f"| {r['batch_size']:,} | {r['status']} | | | | |")
            else:
                lines.append(
                    f"| {r['batch_size']:,} | {r['rows_per_s_median']:,.0f} | {r['rows_per_s_best']:,.0f} | {r['full_hour_s_median']:.2f} | {fmt(r['peak_rss_gb'])} | {fmt(r['peak_gpu_alloc_gb'])} |"
                )
        lines.append("")
    if result.get("maxbatch"):
        ok = [r for r in result["maxbatch"] if r["status"] == "ok"]
        last = result["maxbatch"][-1]
        lines += [
            f"Max single-pass batch (doubling from 2^{a['maxbatch_log2_min']}, cap {a['max_rows']:,} rows): "
            + f"largest successful = **{ok[-1]['rows'] if ok else 0:,}** rows; stopped by: {last['status'] if last['status'] != 'ok' else 'row cap'}.",
            "",
            "| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in result["maxbatch"]:
            lines.append(
                f"| {r['rows']:,} | {r['status'][:60]} | {fmt(r.get('rows_per_s'), ',.0f')} | {fmt(r.get('peak_rss_gb'))} | {fmt(r.get('rss_over_start_gb'))} | {fmt(r.get('peak_gpu_alloc_gb'))} | {fmt(r.get('peak_gpu_reserved_gb'))} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, choices=ENGINE_NAMES)
    p.add_argument("--weights", required=True, help="ptype_weights.npz from applications/export_ptype_weights.py")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N | mps (numpy/numba are CPU only)")
    p.add_argument("--threads", type=int, required=True, help="CPU threads (torch intra-op / ORT intra-op / numba / BLAS). Always explicit.")
    p.add_argument("--keras-path", default=None, help="Required for --engine keras")
    p.add_argument("--onnx-path", default=None)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="torch autocast dtype")
    p.add_argument("--tf32", action="store_true", help="torch: allow TF32 matmul on Ampere+")
    p.add_argument("--compile", action="store_true", help="torch: use torch.compile")
    p.add_argument("--no-pin", action="store_true", help="torch/cuda: disable pinned-memory double buffering")
    p.add_argument("--mode", default="both", choices=["throughput", "maxbatch", "both"])
    p.add_argument("--n-rows", type=int, default=HRRR_ROWS)
    p.add_argument("--log2-min", type=int, default=10, help="smallest batch size = 2^this (throughput)")
    p.add_argument("--log2-max", type=int, default=21, help="largest throughput batch = 2^this (2^21 > 1.9M rows = whole hour in one pass)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--maxbatch-log2-min", type=int, default=10)
    p.add_argument("--max-rows", type=int, default=2**22, help="row cap for the max-batch search (raise on big machines, e.g. 67108864 on an A100 node)")
    p.add_argument("--ram-fraction", type=float, default=0.6, help="CPU only: skip a size whose estimated footprint exceeds this fraction of free RAM")
    p.add_argument("--out", required=True, help="output prefix; writes <out>.json and <out>.md")
    p.add_argument("--json-out", default=None, help="JSON path (default <out>.json); give each run its own when sharing one markdown via --md-append")
    p.add_argument("--md-append", action="store_true", help="append markdown to an existing <out>.md instead of overwriting")
    p.add_argument("--label", default="")
    return p.parse_args(argv)


def main(argv=None) -> None:
    """Run the requested measurements and write JSON + markdown."""
    args = parse_args(argv)
    hw = detect_hardware()
    print("Hardware:", json.dumps(hw, indent=2), flush=True)
    opts = {"threads": args.threads, "dtype": args.dtype, "tf32": args.tf32, "compile": args.compile, "pin_memory": not args.no_pin}
    if args.keras_path:
        opts["keras_path"] = args.keras_path
    if args.onnx_path:
        opts["onnx_path"] = args.onnx_path
    t0 = time.perf_counter()
    engine = make_engine(args.engine, args.weights, device=args.device, **opts)
    cfg = engine.describe()
    print(f"Engine ready in {time.perf_counter() - t0:.1f}s: {cfg}", flush=True)
    result: dict = {"args": vars(args), "hardware": hw, "engine_config": cfg, "maxrss_after_load_gb": maxrss_bytes() / 1e9}
    try:
        _run_modes(args, engine, result)
    except Exception as exc:  # noqa: BLE001 - any engine failure must still produce a report
        # e.g. torch.compile needing a C++ toolchain: record the failure in the report rather than losing the run.
        result["failed"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        print("RUN FAILED:", result["failed"], flush=True)
    result["maxrss_lifetime_gb"] = maxrss_bytes() / 1e9
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out or out.with_suffix(".json")).write_text(json.dumps(result, indent=2, default=str))
    md = to_markdown(result)
    with open(out.with_suffix(".md"), "a" if args.md_append else "w") as f:
        f.write(md)
    print(md)


def _run_modes(args, engine, result: dict) -> None:
    """Execute the selected measurements, storing results in ``result``."""

    if args.mode in ("throughput", "both"):
        print(f"Throughput on {args.n_rows:,} rows", flush=True)
        x = synthetic_rows(args.n_rows)
        result["throughput"] = run_throughput(engine, x, args.device, args.log2_min, args.log2_max, args.repeats, args.warmup)
        del x
        gc.collect()
    if args.mode in ("maxbatch", "both"):
        print("Max batch search", flush=True)
        result["maxbatch"] = run_maxbatch(engine, args.device, args.maxbatch_log2_min, args.max_rows, args.ram_fraction)


if __name__ == "__main__":
    main()
