"""Split a run list into single-node PBS jobs on Derecho and submit them (CPU or GPU nodes, by availability).

Workflow::

    python -m ptype.hrrr_fast.runlist config/hrrr_fast_events.yml --out runs.txt     # all runs of the events
    python applications/hrrr_ptype_submit.py --runs-file runs.txt --jobs-dir $SCRATCH/ptype_fast/jobs --dry-run
    python applications/hrrr_ptype_submit.py --runs-file runs.txt --jobs-dir $SCRATCH/ptype_fast/jobs  # submit <= 10 jobs

Run it again (by hand, or from cron on a login node) to submit the next batch: runs already in the index and runs
belonging to jobs that are still queued or running are left out, so nothing is processed twice and a failed job's
runs are picked up again automatically.

Steps and the reasoning behind each:

1. **Read the runs** (``YYYYMMDDHH maxfh`` lines).
2. **Remove finished runs**: those listed in ``ptype/inits.json`` (read from S3 with ``$AWS_PROFILE``, or a local
   copy), else from a local done-list. The index is the ground truth of what the website can show.
3. **Remove in-flight runs**: every submission is appended to ``<jobs-dir>/submitted.jsonl``; runs of ledger jobs
   that ``qstat`` still reports (queued/running/held) are skipped.
4. **Choose node types** from availability at submit time (``qstat -Q``, ``pbsnodes -a``), rule in
   :func:`assign_node_types`; ``--node-type`` overrides it.
5. **Size jobs**: either ``--runs-per-job`` or from the measured node throughput ``seconds_per_forecast_hour`` per
   node type and a target walltime (default and cap 2 h). Short jobs start sooner (backfill), lose less on a node
   failure, and keep each job's memory/credential lifetime small; the pipeline's resume makes many short jobs as
   good as one long one.
6. **Write one PBS script per job** (plus its runs file) and submit at most ``--max-jobs`` (default 10), so a typo
   cannot flood the queue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PTYPE_FAST = "/glade/work/dgagne/ptype_fast"
REPO = f"{PTYPE_FAST}/ptype-physical"
VENV = f"{PTYPE_FAST}/venv"
WEIGHTS = f"{PTYPE_FAST}/weights/ptype_weights.npz"
SCRATCH = "/glade/derecho/scratch/$USER/ptype_fast"

#: Node shapes on Derecho. ``memory_gb`` is what a job can use (CPU nodes 256 GB physical, ~235 GB usable).
NODE_TYPES = {
    "cpu": {"select": "select=1:ncpus=128:mpiprocs=1", "cores": 128, "memory_gb": 235.0, "gpus": 0},
    "gpu": {"select": "select=1:ncpus=64:mpiprocs=1:ngpus=4", "cores": 64, "memory_gb": 470.0, "gpus": 4},
}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Run:
    """One model run: ``run_id`` = ``YYYYMMDDHH`` and the number of forecast-hour files (maxfh + 1)."""

    run_id: str
    max_fh: int

    @property
    def n_hours(self) -> int:
        return self.max_fh + 1


def read_runs(path: str | os.PathLike) -> list[Run]:
    """Parse ``YYYYMMDDHH maxfh`` lines (blank lines and ``#`` comments ignored)."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if not re.fullmatch(r"\d{10}", parts[0]):
            raise ValueError(f"bad run line {line!r}")
        hour = int(parts[0][8:10])
        max_fh = int(parts[1]) if len(parts) > 1 else (48 if hour in (0, 6, 12, 18) else 18)
        out.append(Run(parts[0], max_fh))
    return out


def run_id_from_init(init: str) -> str:
    """``2024-04-30T00:00Z`` (index format) -> ``2024043000``."""
    return datetime.strptime(init, "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d%H")


def load_done(index_source: str | None, done_list: str | None, reader: Callable[[str], dict] | None = None) -> tuple[set[str], str]:
    """Run ids already finished, and where that information came from.

    ``index_source`` is ``s3://bucket/key`` or a local path to an ``inits.json``. If it cannot be read (no
    credentials on this node, network down), fall back to ``done_list`` (one ``YYYYMMDDHH`` per line, or a local
    ``inits.json``) rather than failing, and say so: submitting a few duplicate runs is cheaper than not submitting.
    """
    reader = reader or _read_index
    if index_source:
        try:
            idx = reader(index_source)
            return {run_id_from_init(e["init"]) for e in idx.get("inits", [])}, f"index {index_source}"
        except Exception as err:  # noqa: BLE001 - any failure means "use the fallback"
            print(f"warning: could not read the index {index_source}: {err}; using the done-list", file=sys.stderr)
    if done_list and Path(done_list).exists():
        text = Path(done_list).read_text()
        if text.lstrip().startswith("{"):
            return {run_id_from_init(e["init"]) for e in json.loads(text).get("inits", [])}, f"local index {done_list}"
        return {ln.split()[0] for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}, f"done-list {done_list}"
    return set(), "none"


def _read_index(source: str) -> dict:
    """Read an index from ``s3://bucket/key`` (obstore + boto3 profile) or a local path."""
    if source.startswith("s3://"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ptype.hrrr_fast.pipeline import ObstoreIndex, output_s3_store, resolve_aws_session

        bucket, _, key = source[5:].partition("/")
        region = os.environ.get("AWS_REGION", "us-east-1")
        # Explicit profile credentials and short timeouts: a failure falls back to the done-list in seconds.
        session = resolve_aws_session(os.environ.get("AWS_PROFILE"), region, verify=False)
        return ObstoreIndex(output_s3_store(bucket, region, session, quick=True), key).read()
    return json.loads(Path(source).read_text())


# ---------------------------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------------------------
def parse_qstat_queues(text: str) -> dict[str, dict[str, int]]:
    """Parse ``qstat -Q``: ``{queue: {"queued": Que, "running": Run, "held": Hld, "total": Tot}}``.

    Columns are located from the header line (``Queue Max Tot Ena Str Que Run Hld Wat Trn Ext Type``) rather than
    by position, so an extra column in another PBS version does not shift the numbers.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = next((ln.split() for ln in lines if ln.split()[:1] == ["Queue"]), None)
    if header is None:
        raise ValueError("qstat -Q output has no header line")
    col = {name: i for i, name in enumerate(header)}
    out = {}
    for ln in lines:
        parts = ln.split()
        if parts[0] in ("Queue",) or set(parts[0]) <= {"-"} or len(parts) < len(header):
            continue
        names = {"Que": "queued", "Run": "running", "Hld": "held", "Tot": "total"}
        out[parts[0]] = {v: int(parts[col[k]]) for k, v in names.items() if k in col}
    return out


def parse_pbsnodes(text: str, qlist: str | None = None) -> dict[str, dict[str, int]]:
    """Count CPU and GPU nodes and how many are completely free from ``pbsnodes -a``.

    A node is "free" when its ``state`` is ``free`` and no CPU is assigned (a partially used node cannot take an
    exclusive whole-node job). A node is a GPU node when ``resources_available.ngpus > 0``. ``qlist`` optionally
    keeps only nodes whose ``resources_available.Qlist`` contains that queue name (e.g. to leave out dev nodes).
    """
    nodes: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if not ln[0].isspace():
            cur = {"name": ln.strip()}
            nodes.append(cur)
        elif cur is not None and "=" in ln:
            k, _, v = ln.strip().partition("=")
            cur[k.strip()] = v.strip()
    out = {"cpu": {"total": 0, "free": 0}, "gpu": {"total": 0, "free": 0}}
    for n in nodes:
        if qlist and qlist not in n.get("resources_available.Qlist", "").split(","):
            continue
        kind = "gpu" if int(n.get("resources_available.ngpus", "0") or 0) > 0 else "cpu"
        out[kind]["total"] += 1
        states = {s.strip() for s in n.get("state", "").split(",")}
        if states == {"free"} and int(n.get("resources_assigned.ncpus", "0") or 0) == 0:
            out[kind]["free"] += 1
    return out


def _run_cmd(cmd: Sequence[str]) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=True, timeout=120).stdout


@dataclass
class ClusterState:
    """Free nodes and queue backlog per node type at submit time."""

    free: dict[str, int] = field(default_factory=lambda: {"cpu": 0, "gpu": 0})
    total: dict[str, int] = field(default_factory=lambda: {"cpu": 0, "gpu": 0})
    queued: dict[str, int] = field(default_factory=lambda: {"cpu": 0, "gpu": 0})
    note: str = ""


def cluster_state(run_cmd: Callable[[Sequence[str]], str] = _run_cmd, cpu_queue: str = "cpu", gpu_queue: str = "gpu", qlist: str | None = None) -> ClusterState:
    """Query ``pbsnodes -a`` and ``qstat -Q``; on any failure return an empty state (the default type is then used).

    On Derecho, jobs are submitted to the routing queue ``main`` and run in execution queues per node type; the
    backlog per type is read from those queues (``--cpu-queue``/``--gpu-queue``; ``cpu`` and ``gpu`` by default -
    check ``qstat -Q`` once and adjust if the names differ).
    """
    st = ClusterState()
    try:
        nodes = parse_pbsnodes(run_cmd(["pbsnodes", "-a"]), qlist)
        st.free = {k: v["free"] for k, v in nodes.items()}
        st.total = {k: v["total"] for k, v in nodes.items()}
    except Exception as err:  # noqa: BLE001
        st.note += f"pbsnodes failed: {err}; "
    try:
        queues = parse_qstat_queues(run_cmd(["qstat", "-Q"]))
        st.queued = {"cpu": queues.get(cpu_queue, {}).get("queued", 0), "gpu": queues.get(gpu_queue, {}).get("queued", 0)}
    except Exception as err:  # noqa: BLE001
        st.note += f"qstat -Q failed: {err}; "
    return st


def assign_node_types(n_slots: int, state: ClusterState, override: str | None = None, default: str = "cpu") -> list[str]:
    """Node type for each of up to ``n_slots`` jobs, in submission order.

    Rule (explicit so it can be reasoned about and overridden):

    1. ``override`` (``--node-type cpu|gpu``) wins for every job.
    2. Otherwise fill nodes that are free RIGHT NOW: first free CPU nodes, then free GPU nodes. A job that can start
       immediately beats any throughput difference between the node types, and CPU nodes come first because the
       pipeline is CPU-bound (fetch/decode/columns/Bourgouin dominate; inference is a small part) and a CPU node has
       twice the cores of a GPU node.
    3. Jobs beyond the free nodes go to the type whose execution queue has FEWER queued jobs (ties: ``default``),
       since that queue is likely to start them sooner.
    """
    if override:
        return [override] * n_slots
    types: list[str] = []
    types += ["cpu"] * min(n_slots, state.free.get("cpu", 0))
    types += ["gpu"] * min(n_slots - len(types), state.free.get("gpu", 0))
    if len(types) < n_slots:
        qc, qg = state.queued.get("cpu", 0), state.queued.get("gpu", 0)
        rest = default if qc == qg else ("cpu" if qc < qg else "gpu")
        types += [rest] * (n_slots - len(types))
    return types


def active_job_ids(run_cmd: Callable[[Sequence[str]], str] = _run_cmd, user: str | None = None) -> set[str] | None:
    """Ids of this user's jobs that PBS still knows as queued/running/held (``qstat -u``); ``None`` if unknown."""
    user = user or os.environ.get("USER", "")
    try:
        text = run_cmd(["qstat", "-u", user])
    except Exception:  # noqa: BLE001
        return None
    ids = set()
    for ln in text.splitlines():
        m = re.match(r"^(\d+)(\.\S+)?\s", ln.strip() + " ")
        if m:
            parts = ln.split()
            state = parts[-2] if len(parts) >= 2 else ""
            if state in ("Q", "R", "H", "B", "E", "W", "S", "T"):
                ids.add(m.group(1))
    return ids


def in_flight_runs(ledger: Path, active: set[str] | None) -> set[str]:
    """Runs of ledger jobs that are still active. If the job list is unknown, every ledger run counts as in flight.

    Assuming "in flight" when ``qstat`` fails is the safe direction: a skipped run is picked up next time, whereas a
    duplicate job would process (and upload) the same run twice concurrently.
    """
    if not ledger.exists():
        return set()
    runs = set()
    for ln in ledger.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if active is None or rec["job_id"].split(".")[0] in active:
            runs.update(rec["runs"])
    return runs


# ---------------------------------------------------------------------------------------------
# Job sizing and scripts
# ---------------------------------------------------------------------------------------------
@dataclass
class Sizing:
    """How big a job may be (see :func:`plan_jobs`)."""

    runs_per_job: int | None = None
    seconds_per_forecast_hour: dict[str, float] = field(default_factory=lambda: {"cpu": 5.0, "gpu": 8.0})
    walltime_s: int = 2 * 3600
    max_walltime_s: int = 2 * 3600
    overhead_s: int = 300
    safety: float = 0.75


def _round_up(seconds: float, step: int = 300) -> int:
    return int(math.ceil(seconds / step) * step)


@dataclass
class Job:
    """One PBS job: its node type, runs and requested walltime."""

    name: str
    node_type: str
    runs: list[Run]
    walltime_s: int

    @property
    def n_hours(self) -> int:
        return sum(r.n_hours for r in self.runs)


def plan_jobs(runs: Sequence[Run], node_types: Sequence[str], sizing: Sizing, stamp: str = "") -> list[Job]:
    """Greedily pack runs (in order) into jobs, one job per entry of ``node_types``.

    Without ``runs_per_job`` a job takes runs while ``overhead + hours * seconds_per_forecast_hour[type]`` stays
    within ``safety * walltime`` (the safety factor absorbs slow S3 periods and the first-hour JIT/model load). The
    requested walltime is that estimate divided by ``safety``, rounded up to 5 minutes and capped at
    ``max_walltime_s``. A single run that alone exceeds the budget still gets its own job at the cap: the pipeline
    resumes, so a job that times out loses at most the hours in progress.
    """
    jobs: list[Job] = []
    i = 0
    for k, kind in enumerate(node_types):
        if i >= len(runs):
            break
        spfh = sizing.seconds_per_forecast_hour[kind]
        take: list[Run] = []
        if sizing.runs_per_job:
            take = list(runs[i : i + sizing.runs_per_job])
        else:
            budget = sizing.safety * sizing.walltime_s - sizing.overhead_s
            used = 0.0
            while i + len(take) < len(runs):
                nxt = runs[i + len(take)]
                if take and used + nxt.n_hours * spfh > budget:
                    break
                take.append(nxt)
                used += nxt.n_hours * spfh
        i += len(take)
        est = sizing.overhead_s + sum(r.n_hours for r in take) * spfh
        wall = min(sizing.max_walltime_s, max(_round_up(est / sizing.safety), 900))
        # The timestamp keeps a resubmission of the same runs from overwriting the earlier job's script and log.
        jobs.append(Job(f"ptype_{take[0].run_id}{stamp}_{k:02d}", kind, take, wall))
    return jobs


def worker_layout(kind: str, threads: int, memory_per_process_gb: float, memory_fraction: float = 0.8) -> tuple[int, int]:
    """``(processes, threads)`` for a node type: ``cores // threads`` workers, fewer if memory would not fit."""
    node = NODE_TYPES[kind]
    by_cores = max(1, node["cores"] // threads)
    by_mem = max(1, int(node["memory_gb"] * memory_fraction // memory_per_process_gb))
    return min(by_cores, by_mem), threads


def _hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render_script(
    job: Job, runs_file: Path, log_file: Path, *, config: str, threads: int, memory_per_process_gb: float, engine: str, account: str, queue: str, extra_args: str = ""
) -> str:
    """The PBS script of one job.

    The environment block mirrors ``benchmarks/pbs/common.sh`` (and sources it, so a change there applies here):
    same venv, checkout, weights and a per-job numba cache. ``AWS_PROFILE`` is written into the script explicitly
    (batch jobs do not inherit the submitting shell's environment) so the upload uses the documented credentials.
    """
    node = NODE_TYPES[job.node_type]
    procs, thr = worker_layout(job.node_type, threads, memory_per_process_gb)
    device = "cuda" if job.node_type == "gpu" else "cpu"
    aws_profile = os.environ.get("AWS_PROFILE", "ai2es-testbed-user")
    run_args = [
        "python", "applications/hrrr_ptype_run.py",
        "--config", config,
        "--runs-file", str(runs_file),
        "--processes", str(procs),
        "--threads", str(thr),
        "--engine", engine,
        "--device", device,
        "--weights", WEIGHTS,
        "--out-dir", f"{SCRATCH}/runs",
        "--summary-json", str(log_file.with_suffix(".summary.json")),
    ]  # fmt: skip
    cmd = " ".join(shlex.quote(a) if "$" not in a else a for a in run_args) + (f" {extra_args}" if extra_args else "")
    return f"""#!/bin/bash
#PBS -N {job.name[:15]}
#PBS -A {account}
#PBS -q {queue}
#PBS -l {node["select"]}
#PBS -l walltime={_hms(job.walltime_s)}
#PBS -j oe
#PBS -o {log_file}
# Generated by applications/hrrr_ptype_submit.py at {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}
# {len(job.runs)} runs, {job.n_hours} forecast hours, {job.node_type} node: {procs} processes x {thr} threads.
source {REPO}/benchmarks/pbs/common.sh
export AWS_PROFILE={shlex.quote(aws_profile)}
nvidia-smi -L 2>/dev/null || true
{cmd}
"""


def submit(script: Path, run_cmd: Callable[[Sequence[str]], str] = _run_cmd) -> str:
    """``qsub script`` -> job id."""
    return run_cmd(["qsub", str(script)]).strip()


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-file", required=True)
    p.add_argument("--jobs-dir", required=True, help="where scripts, runs files, logs and the ledger go")
    p.add_argument("--index", default="s3://wwe-ptype/ptype/inits.json", help="s3://bucket/key or local inits.json")
    p.add_argument("--done-list", default=None, help="fallback: YYYYMMDDHH per line, or a local inits.json")
    p.add_argument("--config", default="config/hrrr_fast.yml")
    p.add_argument("--engine", default="torch")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-per-process-gb", type=float, default=6.0, help="measured peak RSS per worker + margin")
    p.add_argument("--runs-per-job", type=int, default=None)
    p.add_argument("--seconds-per-forecast-hour-cpu", type=float, default=5.0, help="measured node throughput, CPU node")
    p.add_argument("--seconds-per-forecast-hour-gpu", type=float, default=8.0, help="measured node throughput, GPU node")
    p.add_argument("--walltime", default="02:00:00", help="target walltime per job (HH:MM:SS)")
    p.add_argument("--max-walltime", default="02:00:00")
    p.add_argument("--max-jobs", type=int, default=10)
    p.add_argument("--node-type", choices=["cpu", "gpu"], default=None, help="override the availability rule")
    p.add_argument("--cpu-queue", default="cpu", help="execution queue whose backlog stands for CPU nodes")
    p.add_argument("--gpu-queue", default="gpu", help="execution queue whose backlog stands for GPU nodes")
    p.add_argument("--account", default="NAML0001")
    p.add_argument("--queue", default="main")
    p.add_argument("--extra-args", default="", help="appended to the hrrr_ptype_run.py command line")
    p.add_argument("--dry-run", action="store_true", help="write scripts and print the plan; do not qsub")
    return p


def _seconds(hms: str) -> int:
    h, m, s = (int(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def main(argv: list[str] | None = None, run_cmd: Callable[[Sequence[str]], str] = _run_cmd, index_reader: Callable[[str], dict] | None = None) -> list[Job]:
    """Plan (and unless ``--dry-run`` submit) the next batch of jobs; returns the planned jobs."""
    a = build_parser().parse_args(argv)
    jobs_dir = Path(a.jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    ledger = jobs_dir / "submitted.jsonl"
    runs = read_runs(a.runs_file)
    done, done_src = load_done(a.index, a.done_list, index_reader)
    flying = in_flight_runs(ledger, active_job_ids(run_cmd))
    todo = [r for r in runs if r.run_id not in done and r.run_id not in flying]
    print(f"runs: {len(runs)} listed, {len(done & {r.run_id for r in runs})} done ({done_src}), {len(flying)} in flight, {len(todo)} to submit")
    if not todo:
        return []
    state = cluster_state(run_cmd, a.cpu_queue, a.gpu_queue) if not a.node_type else ClusterState(note="override")
    types = assign_node_types(a.max_jobs, state, a.node_type)
    print(f"cluster: free {state.free}, queued {state.queued} {state.note}-> node types {types}")
    sizing = Sizing(
        runs_per_job=a.runs_per_job,
        seconds_per_forecast_hour={"cpu": a.seconds_per_forecast_hour_cpu, "gpu": a.seconds_per_forecast_hour_gpu},
        walltime_s=_seconds(a.walltime),
        max_walltime_s=_seconds(a.max_walltime),
    )
    jobs = plan_jobs(todo, types, sizing, stamp=datetime.now(UTC).strftime("_%m%d%H%M%S"))
    for job in jobs:
        runs_file = jobs_dir / f"{job.name}.runs"
        runs_file.write_text("".join(f"{r.run_id} {r.max_fh}\n" for r in job.runs))
        script = jobs_dir / f"{job.name}.pbs"
        script.write_text(
            render_script(
                job, runs_file, jobs_dir / f"{job.name}.log", config=a.config, threads=a.threads, memory_per_process_gb=a.memory_per_process_gb,
                engine=a.engine, account=a.account, queue=a.queue, extra_args=a.extra_args,
            )
        )  # fmt: skip
        line = f"{job.name}: {job.node_type}, {len(job.runs)} runs ({job.runs[0].run_id}..{job.runs[-1].run_id}), {job.n_hours} hours, walltime {_hms(job.walltime_s)}"
        if a.dry_run:
            print(f"[dry-run] {line} -> {script}")
            continue
        job_id = submit(script, run_cmd)
        with open(ledger, "a") as f:
            f.write(
                json.dumps({"job_id": job_id, "name": job.name, "node_type": job.node_type, "runs": [r.run_id for r in job.runs], "time": datetime.now(UTC).isoformat()}) + "\n"
            )
        print(f"{job_id} {line}")
    return jobs


if __name__ == "__main__":
    main()
