"""Tests of the end-to-end pipeline driver, the run CLI and the PBS launcher. No network, no AWS, no GPU.

The pipeline is exercised with :class:`ptype.hrrr_fast.pipeline.SyntheticSource` (deterministic fake HRRR fields
for any number of hybrid levels, optional missing hours and transient failures) and a tiny randomly initialised
network exported in the same ``.npz`` format as the real weights, so every stage except GRIB decoding runs for
real: soundings, engine, revised Bourgouin, rotation, both Zarr stores, markers, manifest, upload and index.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("cast_value_rs")
pytest.importorskip("pyproj")
pytest.importorskip("numba")

from ptype.hrrr_fast import columns, index, pipeline
from ptype.hrrr_fast.bourgouin import revised_bourgouin_from_sounding
from ptype.hrrr_fast.config import PipelineConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "applications"))
import hrrr_ptype_run
import hrrr_ptype_submit as submit

INIT = "2023122600"


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def weights(tmp_path_factory) -> str:
    """A tiny 84-16-4 network in the export format of applications/export_ptype_weights.py."""
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("weights") / "ptype_weights.npz"
    np.savez(
        path,
        n_layers=np.int64(2),
        W0=rng.normal(0, 0.2, (84, 16)).astype(np.float32),
        b0=np.zeros(16, np.float32),
        W1=rng.normal(0, 0.5, (16, 4)).astype(np.float32),
        b1=np.zeros(4, np.float32),
        negative_slope=np.float64(0.2),
        center=np.zeros(84),
        scale=np.full(84, 10.0),
        classes=np.array(["rain", "snow", "icep", "frzr"]),
        heights_m=np.arange(0, 5001, 250),
        feature_names=np.array([f"f{i}" for i in range(84)]),
    )
    return str(path)


def make_cfg(tmp_path: Path, weights: str, **kw) -> PipelineConfig:
    base = {
        "source": "synthetic",
        "work_dir": str(tmp_path / "work"),
        "weights": weights,
        "engine": "numpy",
        "processes": 0,
        "threads_per_process": 2,
        "upload": "local",
        "upload_root": str(tmp_path / "bucket"),
        "keep_local": True,
        "fetch_backoff_s": 0.001,
        "log_json": True,
    }
    return PipelineConfig(**(base | kw))


class RecordingIndex:
    """Index backend that records calls (and can fail)."""

    def __init__(self, calls: list, fail: bool = False):
        self.calls = calls
        self.fail = fail
        self.data = index.empty_index()

    def read(self):
        return self.data

    def update(self, entry):
        self.calls.append(("index", entry["forecast_hours"]))
        if self.fail:
            raise RuntimeError("index down")
        self.data = index.merge_entry(self.data, entry["init"], entry["forecast_hours"], entry["store"], entry.get("soundings"))
        return self.data


class RecordingUploader:
    """Uploader that records calls (and can fail on the second store)."""

    name = "fake"

    def __init__(self, calls: list, fail: bool = False):
        self.calls = calls
        self.fail = fail

    def preflight(self):
        self.calls.append(("preflight",))

    def upload_store(self, local, key):
        self.calls.append(("upload", key))
        if self.fail and key.startswith("soundings/"):
            raise RuntimeError("network down")
        return [key]


# ---------------------------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------------------------
def test_config_yaml_overrides_and_validation(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("engine: onnx\nthreads_per_process: 4\nwork_dir: $HOME/x\n")
    cfg = PipelineConfig.from_yaml(p, engine=None, processes=3)
    assert cfg.engine == "onnx" and cfg.processes == 3 and cfg.threads_per_process == 4
    assert "$HOME" not in cfg.work_dir
    assert cfg.storage_class == "STANDARD_IA"
    p.write_text("engin: onnx\n")
    with pytest.raises(KeyError):
        PipelineConfig.from_yaml(p)
    with pytest.raises(ValueError):
        PipelineConfig(bourgouin_top_m=4000)
    with pytest.raises(ValueError):
        PipelineConfig(bourgouin_top_m=8100)
    with pytest.raises(ValueError):
        PipelineConfig(upload="local")


def test_repo_config_loads():
    cfg = PipelineConfig.from_yaml(REPO / "config" / "hrrr_fast.yml")
    assert cfg.n_levels == 27 and cfg.bucket == "wwe-ptype" and cfg.storage_class == "STANDARD_IA"


def test_runs_and_hours(tmp_path):
    assert pipeline.parse_hours("0-3,12") == [0, 1, 2, 3, 12]
    f = tmp_path / "runs.txt"
    f.write_text("# comment\n2023122600 48\n2023122601 18\n\n")
    runs = pipeline.read_runs_file(f)
    assert [r.run_id for r in runs] == ["2023122600", "2023122601"]
    assert runs[0].hours == tuple(range(49)) and runs[1].hours == tuple(range(19))
    # The store axis depends only on the cycle, so any subset of hours maps to the same fh_index.
    sub = pipeline.read_runs_file(f, [11, 12])
    assert sub[0].store_hours == list(range(49)) and sub[1].store_hours == list(range(19))
    assert sub[0].init == datetime(2023, 12, 26, 0, tzinfo=UTC)


def test_thread_env_and_layout():
    env = pipeline.thread_env(8)
    assert {env[k] for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} == {"8"}


def test_retries():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    sleeps = []
    assert pipeline.call_with_retries(flaky, retries=5, backoff_s=1.0, max_backoff_s=2.0, what="t", sleep=sleeps.append) == "ok"
    assert calls["n"] == 3 and len(sleeps) == 2 and all(0 <= s <= 2.0 for s in sleeps)

    def missing():
        raise FileNotFoundError("nope")

    with pytest.raises(pipeline.MissingInputError):
        pipeline.call_with_retries(missing, retries=5, backoff_s=0, max_backoff_s=0, what="t", sleep=sleeps.append)

    def bug():
        calls["n"] += 1
        raise KeyError("not transient")

    calls["n"] = 0
    with pytest.raises(KeyError):
        pipeline.call_with_retries(bug, retries=5, backoff_s=0, max_backoff_s=0, what="t", sleep=lambda s: None)
    assert calls["n"] == 1  # not retried

    def always():
        raise TimeoutError("slow")

    with pytest.raises(TimeoutError):
        pipeline.call_with_retries(always, retries=2, backoff_s=0, max_backoff_s=0, what="t", sleep=lambda s: None)


def test_levels_for_height_matches_measured_table():
    assert pipeline.levels_for_height(5000, margin=0) == 25
    assert pipeline.levels_for_height(8000, margin=0) == 33
    assert pipeline.levels_for_height(12000, margin=0) == 39
    assert pipeline.levels_for_height(99999) == 50


def test_read_tops_up_levels_until_every_column_is_deep_enough(weights, tmp_path):
    cfg = make_cfg(tmp_path, weights, n_levels=15, bourgouin_top_m=10000)
    src = pipeline.SyntheticSource()
    hour = src.open_hour(pipeline.parse_init(INIT), 3)
    levels = pipeline.LevelState(15, 15)
    fields, _, _, info = pipeline.read_hour_fields(hour, cfg, levels, 1)
    assert info["rereads"] >= 2
    n_w, n_t = info["levels"]
    z = fields["HGT"] - fields["HGT_sfc"]
    assert fields["UGRD"].shape[0] == n_w and fields["TMP"].shape[0] == n_t
    assert z[n_w - 1].min() >= 5000 and z[n_t - 1].min() >= 10000
    assert z[n_w - 2].min() < 5000 or n_w == 15  # it stopped as soon as the requirement was met (within one step)
    # The worker remembers the counts so the next hour starts high enough.
    assert (levels.n_wind, levels.n_thermo) == (n_w, n_t)
    _, _, _, info2 = pipeline.read_hour_fields(src.open_hour(pipeline.parse_init(INIT), 4), cfg, levels, 1)
    assert info2["rereads"] == 0


def _decoded_hour(n_wind=27, n_thermo=34):
    src = pipeline.SyntheticSource(ny=20, nx=30)
    fields, grid, _ = src.open_hour(pipeline.parse_init(INIT), 2).read((0, n_wind), (0, n_thermo), True, 1)
    return fields, grid


def test_deep_bourgouin_chunks_equal_one_pass_and_share_the_network_levels(weights, tmp_path):
    fields, _ = _decoded_hour()
    z = fields["HGT"] - fields["HGT_sfc"][None]
    w = slice(0, 27)
    snd, _ = columns.build_soundings(z[w], fields["TMP"][w], fields["SPFH"][w], fields["PRES"][w], fields["UGRD"], fields["VGRD"],
                                     fields["TMP_2m"], fields["DPT_2m"], fields["UGRD_10m"], fields["VGRD_10m"], fields["PRES_sfc"])  # fmt: skip
    heights = pipeline.bourgouin_heights(8000)
    full, _ = columns.build_soundings(z, fields["TMP"], fields["SPFH"], fields["PRES"], fields["TMP"], fields["TMP"], fields["TMP_2m"], fields["DPT_2m"],
                                      psfc_pa=fields["PRES_sfc"], target_heights=heights, surface_wind="interp")  # fmt: skip
    # T, Td and p on 0..5000 m of the deep grid are bit-identical to the network sounding.
    for g in (0, 1, 4):
        np.testing.assert_array_equal(full[:, g, :21], snd[:, g, :])
    ref_p, ref_c, ref_d = revised_bourgouin_from_sounding(full, heights, return_diagnostics=True)
    cfg = make_cfg(tmp_path, weights, bourgouin_top_m=8000, bourgouin_rows_per_chunk=7)  # 7 does not divide 20 rows
    p, c, d, n_short = pipeline.run_bourgouin(fields, np.ascontiguousarray(z), snd, cfg, return_diagnostics=True)
    np.testing.assert_array_equal(p, ref_p)
    np.testing.assert_array_equal(c, ref_c)
    np.testing.assert_array_equal(d, ref_d)
    assert n_short == 0
    # Top = 5000 m uses the network sounding directly.
    p5, c5, _ = pipeline.run_bourgouin(fields, z, snd, make_cfg(tmp_path, weights, bourgouin_top_m=5000))
    ref5 = revised_bourgouin_from_sounding(snd)
    np.testing.assert_array_equal(p5, ref5[0])
    np.testing.assert_array_equal(c5, ref5[1])


def test_in_place_rotation_matches_columns_rotation():
    rng = np.random.default_rng(1)
    ny, nx, nh = 6, 9, 21
    lon = rng.uniform(-125, -70, (ny, nx))
    snd = rng.normal(0, 10, (ny * nx, 5, nh)).astype(np.float32)
    ug = np.ascontiguousarray(snd[:, 2, :].T).reshape(nh, ny, nx)
    vg = np.ascontiguousarray(snd[:, 3, :].T).reshape(nh, ny, nx)
    ue, ve = columns.rotate_grid_to_earth(ug, vg, lon)
    other = snd[:, [0, 1, 4], :].copy()
    pipeline.rotate_sounding_to_earth(snd, columns.grid_rotation_cos_sin(lon))
    np.testing.assert_array_equal(snd[:, 2, :], ue.reshape(nh, -1).T)
    np.testing.assert_array_equal(snd[:, 3, :], ve.reshape(nh, -1).T)
    np.testing.assert_array_equal(snd[:, [0, 1, 4], :], other)


# ---------------------------------------------------------------------------------------------
# Whole pipeline (in-process)
# ---------------------------------------------------------------------------------------------
def test_pipeline_end_to_end_with_missing_hour_and_transient_errors(weights, tmp_path):
    cfg = make_cfg(tmp_path, weights, source="synthetic:missing=2;fail=1")
    runs = pipeline.make_runs([INIT], [0, 1, 2, 3])
    res = pipeline.run_pipeline(cfg, runs)
    m = res[0]
    assert m["status"] == "indexed"
    assert m["written_hours"] == [0, 1, 3] and m["missing_hours"] == [2] and m["failed_hours"] == []
    idx = json.loads((tmp_path / "bucket" / "ptype" / "inits.json").read_text())
    assert idx["inits"][0]["init"] == "2023-12-26T00:00Z" and idx["inits"][0]["forecast_hours"] == [0, 1, 3]
    pk, sk = index.store_paths(pipeline.parse_init(INIT))
    g = zarr.open_group(tmp_path / "bucket" / pk, mode="r")
    assert list(g["0/forecast_hour"][:]) == list(range(49))  # 00 UTC cycle: 0..48
    for name in ("ML_rain", "ML_u", "BOURG_snow", "BOURG_class", "precip_mask", "CRAIN", "prate"):
        assert name in g["0"] and name in g["3"]
    ml = np.stack([g[f"0/ML_{c}"][1] for c in ("rain", "snow", "icep", "frzr")])
    assert np.all(np.isfinite(ml)) and np.abs(ml.sum(axis=0) - 1).max() < 0.01
    assert np.all(np.isnan(g["0/ML_rain"][2]))  # missing hour stays at the fill value
    bc = g["0/BOURG_class"][3]
    assert set(np.unique(bc)) <= {0, 1, 2, 3, 254}
    s = zarr.open_array(tmp_path / "bucket" / sk, path="sounding", mode="r")
    assert s.shape[1:] == (24 * 32, 105) and np.all(np.isfinite(s[1])) and np.all(np.isnan(s[2]))
    man = json.loads((tmp_path / "work" / INIT / "manifest.json").read_text())
    h = man["hours"]["1"]
    assert set(pipeline.STAGES) <= set(h["timings"]) and h["peak_rss_bytes"] > 0 and h["bytes"]["payload_bytes"] > 0
    assert man["provenance"]["packages"]["numpy"] and "config" in man["provenance"]
    # Predictions are made everywhere: nothing is blanked by the mask.
    mask = g["0/precip_mask"][1].astype(bool)
    assert (~mask).any() and np.all(np.isfinite(g["0/ML_snow"][1][~mask]))


def test_resume_skips_indexed_runs_and_completed_hours(weights, tmp_path):
    # Staging layout (no finalize), where hours can still be (re)written; a finalized store refuses new hours,
    # see test_interrupted_run_resumes_missing_hours_then_finalizes_once.
    cfg = make_cfg(tmp_path, weights, finalize_ptype=False)
    src = pipeline.SyntheticSource()
    runs = pipeline.make_runs([INIT], [0, 1, 2])
    pipeline.run_pipeline(cfg, runs, source=src)
    assert len(src.reads) == 3
    # Already indexed: nothing is read at all.
    res = pipeline.run_pipeline(cfg, runs, source=src)
    assert res == [{"run_id": INIT, "status": "skipped_indexed"}] and sum(src.reads.values()) == 3
    # Ignoring the index: markers make every hour a no-op ...
    pipeline.run_pipeline(cfg, runs, source=src, ignore_index=True)
    assert sum(src.reads.values()) == 3
    # ... and deleting one marker redoes exactly that hour.
    (tmp_path / "work" / INIT / "status" / "f01.json").unlink()
    res = pipeline.run_pipeline(cfg, runs, source=src, ignore_index=True)
    assert sum(src.reads.values()) == 4 and src.reads[(pipeline.parse_init(INIT).isoformat(), 1)] == 2
    assert res[0]["written_hours"] == [0, 1, 2]


def test_failed_hour_blocks_upload_and_index(weights, tmp_path):
    calls: list = []

    class Broken(pipeline.SyntheticSource):
        def open_hour(self, init, fxx, log_extra=None):
            if fxx == 1:
                raise ValueError("corrupt GRIB")
            return super().open_hour(init, fxx, log_extra)

    cfg = make_cfg(tmp_path, weights)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1]), source=Broken(), uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls))
    assert res[0]["status"] == "incomplete" and res[0]["failed_hours"] == [1]
    assert calls == [("preflight",)]  # no upload, no index
    assert not (tmp_path / "work" / INIT / "status" / "f01.json").exists()


@pytest.fixture
def finalize_calls(monkeypatch):
    """Record every finalize_ptype_store call (delegating to the real one when the writers module has it)."""
    from ptype.hrrr_fast import writers

    calls: list = []
    real = getattr(writers, "finalize_ptype_store", None)

    def wrapper(path, *, remove_staging=True):
        calls.append(("finalize", Path(path).name))
        if real is None:  # stub until the writers module provides it
            return {"stub": True}
        return real(path, remove_staging=remove_staging)

    monkeypatch.setattr(writers, "finalize_ptype_store", wrapper, raising=False)
    return calls


def test_index_only_after_successful_upload_and_local_cleanup(weights, tmp_path, finalize_calls):
    calls = finalize_calls
    cfg = make_cfg(tmp_path, weights, keep_local=False)
    runs = pipeline.make_runs([INIT], [0])
    res = pipeline.run_pipeline(cfg, runs, uploader=RecordingUploader(calls, fail=True), index_backend=RecordingIndex(calls))
    assert res[0]["status"] == "upload_failed"
    assert [c[0] for c in calls] == ["preflight", "finalize", "upload", "upload"]  # finalize BEFORE upload, index never
    assert res[0]["finalize"]["seconds"] >= 0
    assert (tmp_path / "work" / INIT / "hrrr_ptype_20231226_0000.zarr").exists()  # kept for the next attempt
    calls.clear()
    res = pipeline.run_pipeline(cfg, runs, uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls))
    # finalize ran once (first attempt, before the failed upload); the retry goes straight to the upload.
    assert [c[0] for c in calls] == ["preflight", "upload", "upload", "index"]
    assert res[0]["status"] == "indexed" and res[0]["local_removed"] and res[0]["finalize"]["ok"]
    assert not (tmp_path / "work" / INIT / "hrrr_ptype_20231226_0000.zarr").exists()
    assert (tmp_path / "work" / INIT / "manifest.json").exists()


def test_index_failure_is_recorded(weights, tmp_path):
    calls: list = []
    cfg = make_cfg(tmp_path, weights, keep_local=False)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0]), uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls, fail=True))
    assert res[0]["status"] == "index_failed" and (tmp_path / "work" / INIT / "hrrr_ptype_20231226_0000.zarr").exists()


def test_obstore_index_backend_with_memorystore(weights, tmp_path):
    from obstore.store import MemoryStore

    store = MemoryStore()
    calls: list = []
    backend = pipeline.ObstoreIndex(store)
    cfg = make_cfg(tmp_path, weights)
    pipeline.run_pipeline(cfg, pipeline.make_runs([INIT, "2023122601"], [0]), uploader=RecordingUploader(calls), index_backend=backend)
    inits = [e["init"] for e in backend.read()["inits"]]
    assert inits == ["2023-12-26T00:00Z", "2023-12-26T01:00Z"]


def test_no_upload_leaves_stores_and_index_alone(weights, tmp_path):
    cfg = make_cfg(tmp_path, weights)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0]), no_upload=True)
    assert res[0]["status"] == "complete"
    assert not (tmp_path / "bucket" / "ptype" / "inits.json").exists()


def test_all_hours_missing(weights, tmp_path):
    cfg = make_cfg(tmp_path, weights, source="synthetic:missing=0-3")
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1, 2, 3]))
    assert res[0]["status"] == "missing"


def test_local_index_concurrent_updates(tmp_path):
    idx = pipeline.LocalIndex(tmp_path / "ptype" / "inits.json")
    errors = []

    def add(h):
        try:
            idx.update(index.make_entry(datetime(2024, 1, 1, h, tzinfo=UTC), [0, 1]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=add, args=(h,)) for h in range(24)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and len(idx.read()["inits"]) == 24


def test_worker_pool_spawn(weights, tmp_path):
    """Two spawned worker processes (the production path): every hour done, each by a worker that loaded the model once."""
    cfg = make_cfg(tmp_path, weights, processes=2, threads_per_process=1)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1, 2, 3]))
    assert res[0]["written_hours"] == [0, 1, 2, 3]
    workers = {h["worker"] for h in res[0]["hours"].values()}
    assert workers <= {0, 1}


# ---------------------------------------------------------------------------------------------
# AWS CLI upload backend (subprocess mocked; nothing is sent anywhere)
# ---------------------------------------------------------------------------------------------
class FakeRun:
    def __init__(self, fail_on: str | None = None):
        self.cmds: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        rc = 1 if self.fail_on and self.fail_on in " ".join(cmd) else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="boom" if rc else "")


def test_aws_sync_commands_storage_class_and_index_exclusion(tmp_path):
    run = FakeRun()
    up = pipeline.AwsCliUploader("wwe-ptype", "us-east-1", storage_class="STANDARD_IA", runner=run)
    up.upload_store(tmp_path / "hrrr_ptype_20231226_0000.zarr", "ptype/2023/12/26/hrrr_ptype_20231226_0000.zarr")
    assert len(run.cmds) == 2
    for cmd in run.cmds:
        assert cmd[:3] == ["aws", "s3", "sync"]
        assert cmd[4] == "s3://wwe-ptype/ptype/2023/12/26/hrrr_ptype_20231226_0000.zarr"
        assert cmd[cmd.index("--storage-class") + 1] == "STANDARD_IA"
        assert "--delete" not in cmd
        assert any(cmd[i] == "--exclude" and cmd[i + 1] == "*inits.json" for i in range(len(cmd) - 1))
        assert cmd[cmd.index("--region") + 1] == "us-east-1"
    types = {cmd[cmd.index("--content-type") + 1] for cmd in run.cmds}
    assert types == {"application/json", "application/octet-stream"}
    # The index is never a sync destination.
    with pytest.raises(ValueError):
        up.sync_commands(tmp_path, "ptype/inits.json")
    # storage_class None -> bucket default (no flag)
    assert "--storage-class" not in pipeline.AwsCliUploader("b", "us-east-1", storage_class=None).sync_commands(tmp_path, "ptype/x.zarr")[0]


def test_aws_preflight_and_failure_propagate(tmp_path):
    with pytest.raises(RuntimeError, match="credentials"):
        pipeline.AwsCliUploader("b", "us-east-1", runner=FakeRun(fail_on="get-caller-identity")).preflight()
    up = pipeline.AwsCliUploader("b", "us-east-1", runner=FakeRun(fail_on="sync"))
    with pytest.raises(RuntimeError):
        up.upload_store(tmp_path, "ptype/x.zarr")


def test_index_written_with_standard_class_via_obstore_not_sync(weights, tmp_path):
    """With the S3 uploader, sync only touches .zarr prefixes; the index goes through the index backend."""
    from obstore.store import MemoryStore

    run = FakeRun()
    up = pipeline.AwsCliUploader("wwe-ptype", "us-east-1", runner=run)
    backend = pipeline.ObstoreIndex(MemoryStore())
    cfg = make_cfg(tmp_path, weights)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0]), uploader=up, index_backend=backend)
    assert res[0]["status"] == "indexed"
    dests = [c[4] for c in run.cmds if c[1:3] == ["s3", "sync"]]
    assert dests and all(d.endswith(".zarr") for d in dests) and not any("inits.json" in d for d in dests)
    assert backend.read()["inits"][0]["forecast_hours"] == [0]


# ---------------------------------------------------------------------------------------------
# Run CLI
# ---------------------------------------------------------------------------------------------
def test_run_cli_args_and_dry_run(tmp_path, weights, capsys):
    runs_file = tmp_path / "runs.txt"
    runs_file.write_text("2023122600 2\n2023122601 1\n")
    a = hrrr_ptype_run.build_parser().parse_args(
        [
            "--config",
            str(REPO / "config/hrrr_fast.yml"),
            "--init",
            INIT,
            "--runs-file",
            str(runs_file),
            "--processes",
            "4",
            "--threads",
            "2",
            "--engine",
            "numpy",
            "--out-dir",
            str(tmp_path),
        ]
    )
    cfg = hrrr_ptype_run.config_from_args(a)
    assert cfg.processes == 4 and cfg.threads_per_process == 2 and cfg.engine == "numpy" and cfg.work_dir == str(tmp_path)
    runs = hrrr_ptype_run.runs_from_args(a)
    assert [r.run_id for r in runs] == [INIT, "2023122601"] and runs[0].hours == tuple(range(49))  # --init gives the full cycle
    tot = pipeline.dry_run(make_cfg(tmp_path, weights, source="synthetic:missing=1"), pipeline.make_runs([INIT], [0, 1, 2]))
    assert tot["hours"] == 2 and tot["missing"] == 1 and tot["requested"] > 0
    assert "missing [1]" in capsys.readouterr().out


def test_run_cli_main_inline(tmp_path, weights):
    code = hrrr_ptype_run.main(
        ["--config", str(REPO / "config/hrrr_fast.yml"), "--init", INIT, "--forecast-hours", "0-1", "--source", "synthetic", "--weights", weights, "--engine", "numpy",
         "--processes", "0", "--threads", "1", "--out-dir", str(tmp_path / "w"), "--no-upload", "--summary-json", str(tmp_path / "s.json")]
    )  # fmt: skip
    assert code == 0 and json.loads((tmp_path / "s.json").read_text())[0]["written_hours"] == [0, 1]


# ---------------------------------------------------------------------------------------------
# PBS launcher (qstat / pbsnodes / qsub mocked)
# ---------------------------------------------------------------------------------------------
QSTAT_Q = """
Queue              Max   Tot Ena Str   Que   Run   Hld   Wat   Trn   Ext Type
---------------- ----- ----- --- --- ----- ----- ----- ----- ----- ----- ----
main                 0     0 yes yes     0     0     0     0     0     0 Rout
cpu                  0  1534 yes yes   812   650    72     0     0     0 Exec
gpu                  0    61 yes yes    12    49     0     0     0     0 Exec
"""

PBSNODES = """dec0001
     Mom = dec0001.hsn.de.hpc.ucar.edu
     state = free
     resources_available.ncpus = 128
     resources_available.ngpus = 0
     resources_available.Qlist = cpu
     resources_assigned.ncpus = 0

dec0002
     state = job-busy
     resources_available.ncpus = 128
     resources_assigned.ncpus = 128

dec0003
     state = free
     resources_available.ncpus = 128
     resources_assigned.ncpus = 64

deg0001
     state = free
     resources_available.ncpus = 64
     resources_available.ngpus = 4
     resources_available.Qlist = gpu
     resources_assigned.ncpus = 0

deg0002
     state = down,offline
     resources_available.ngpus = 4
"""

QSTAT_U = """
desched1:
                                                            Req'd  Req'd   Elap
Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
5551234.desched dgagne   cpu      ptype_2023     --    1 128  235gb 02:00 Q   --
"""


def fake_cmd(log: list):
    def run(cmd):
        log.append(list(cmd))
        if cmd[:2] == ["qstat", "-Q"]:
            return QSTAT_Q
        if cmd[:2] == ["pbsnodes", "-a"]:
            return PBSNODES
        if cmd[:2] == ["qstat", "-u"]:
            return QSTAT_U
        if cmd[0] == "qsub":
            return f"{7000000 + sum(c[0] == 'qsub' for c in log)}.desched1\n"
        raise AssertionError(cmd)

    return run


def test_parse_scheduler_output():
    q = submit.parse_qstat_queues(QSTAT_Q)
    assert q["cpu"] == {"queued": 812, "running": 650, "held": 72, "total": 1534} and q["gpu"]["queued"] == 12
    n = submit.parse_pbsnodes(PBSNODES)
    assert n == {"cpu": {"total": 3, "free": 1}, "gpu": {"total": 2, "free": 1}}
    assert submit.parse_pbsnodes(PBSNODES, qlist="gpu") == {"cpu": {"total": 0, "free": 0}, "gpu": {"total": 1, "free": 1}}
    assert submit.active_job_ids(fake_cmd([])) == {"5551234"}


def test_node_type_rule():
    st = submit.ClusterState(free={"cpu": 2, "gpu": 1}, queued={"cpu": 800, "gpu": 10})
    assert submit.assign_node_types(5, st) == ["cpu", "cpu", "gpu", "gpu", "gpu"]  # free nodes first, then the shorter queue
    st = submit.ClusterState(free={"cpu": 0, "gpu": 0}, queued={"cpu": 5, "gpu": 50})
    assert submit.assign_node_types(2, st) == ["cpu", "cpu"]
    assert submit.assign_node_types(3, st, override="gpu") == ["gpu"] * 3
    assert submit.assign_node_types(1, submit.ClusterState()) == ["cpu"]  # ties / unknown -> default


def test_job_sizing():
    runs = [submit.Run(f"20231226{h:02d}", 48 if h % 6 == 0 else 18) for h in range(12)]  # 2x49 + 10x19 = 288 hours
    sz = submit.Sizing(seconds_per_forecast_hour={"cpu": 10.0, "gpu": 20.0}, walltime_s=3600, max_walltime_s=7200, overhead_s=300, safety=0.75)
    jobs = submit.plan_jobs(runs, ["cpu"] * 10, sz)
    # budget = 0.75 * 3600 - 300 = 2400 s = 240 forecast hours at 10 s
    assert sum(len(j.runs) for j in jobs) == 12
    assert all(j.n_hours * 10 <= 2400 for j in jobs)
    assert all(j.walltime_s <= 7200 and j.walltime_s % 300 == 0 for j in jobs)
    gpu = submit.plan_jobs(runs, ["gpu"] * 10, sz)
    assert len(gpu) > len(jobs)  # slower node type -> smaller jobs
    fixed = submit.plan_jobs(runs, ["cpu"] * 3, submit.Sizing(runs_per_job=5))
    assert [len(j.runs) for j in fixed] == [5, 5, 2]
    # A run that alone exceeds the budget still gets a job, at the walltime cap.
    big = submit.plan_jobs([submit.Run("2023122600", 48)], ["cpu"], submit.Sizing(seconds_per_forecast_hour={"cpu": 500.0, "gpu": 500.0}))
    assert len(big) == 1 and big[0].walltime_s == 7200
    assert submit.worker_layout("cpu", 8, 6.0) == (16, 8) and submit.worker_layout("gpu", 8, 6.0) == (8, 8)
    assert submit.worker_layout("cpu", 4, 12.0) == (15, 4)  # memory-limited: 235 * 0.8 // 12


def test_render_script(tmp_path):
    job = submit.Job("ptype_2023122600_00", "gpu", [submit.Run("2023122600", 48)], 5400)
    s = submit.render_script(
        job, tmp_path / "j.runs", tmp_path / "j.log", config="config/hrrr_fast.yml", threads=8, memory_per_process_gb=6.0, engine="torch", account="NAML0001", queue="main"
    )
    assert "#PBS -A NAML0001" in s and "#PBS -q main" in s and "select=1:ncpus=64:mpiprocs=1:ngpus=4" in s and "walltime=01:30:00" in s
    assert "source /glade/work/dgagne/ptype_fast/ptype-physical/benchmarks/pbs/common.sh" in s
    assert "--device cuda" in s and "--processes 8" in s and "/glade/work/dgagne/ptype_fast/weights/ptype_weights.npz" in s
    assert "/glade/derecho/scratch/$USER/ptype_fast/runs" in s and "export AWS_PROFILE=" in s
    cpu = submit.render_script(
        submit.Job("x", "cpu", job.runs, 3600), tmp_path / "j.runs", tmp_path / "j.log", config="c", threads=8, memory_per_process_gb=6.0, engine="torch", account="A", queue="main"
    )
    assert "select=1:ncpus=128:mpiprocs=1\n" in cpu and "--device cpu" in cpu and "--processes 16" in cpu


def test_launcher_main_dry_run_and_submit(tmp_path):
    runs_file = tmp_path / "runs.txt"
    runs_file.write_text("".join(f"20231226{h:02d} {48 if h % 6 == 0 else 18}\n" for h in range(24)))
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    # Job 5551234 is still queued (QSTAT_U) -> its runs are in flight; 7777 finished -> its runs are eligible again.
    (jobs_dir / "submitted.jsonl").write_text(
        json.dumps({"job_id": "5551234.desched1", "runs": ["2023122602"]}) + "\n" + json.dumps({"job_id": "7777.desched1", "runs": ["2023122603"]}) + "\n"
    )
    index_doc = {"inits": [{"init": "2023-12-26T00:00Z"}, {"init": "2023-12-26T01:00Z"}]}
    log: list = []
    args = ["--runs-file", str(runs_file), "--jobs-dir", str(jobs_dir), "--index", "s3://wwe-ptype/ptype/inits.json", "--max-jobs", "3", "--runs-per-job", "4"]
    jobs = submit.main([*args, "--dry-run"], run_cmd=fake_cmd(log), index_reader=lambda src: index_doc)
    ids = [r.run_id for j in jobs for r in j.runs]
    assert "2023122600" not in ids and "2023122601" not in ids and "2023122602" not in ids and "2023122603" in ids
    assert len(jobs) == 3 and [j.node_type for j in jobs] == ["cpu", "gpu", "gpu"]  # 1 free cpu, 1 free gpu, gpu queue shorter
    assert not any(c[0] == "qsub" for c in log) and len(list(jobs_dir.glob("*.pbs"))) == 3
    jobs = submit.main(args, run_cmd=fake_cmd(log), index_reader=lambda src: index_doc)
    assert sum(c[0] == "qsub" for c in log) == 3
    ledger = [json.loads(ln) for ln in (jobs_dir / "submitted.jsonl").read_text().splitlines()]
    assert len(ledger) == 5 and ledger[-1]["job_id"].endswith(".desched1")
    # Next invocation: the new jobs are not in qstat output -> treated as finished; but if qstat fails, every ledger run is in flight.
    assert submit.in_flight_runs(jobs_dir / "submitted.jsonl", None) >= {r.run_id for j in jobs for r in j.runs}


def test_launcher_index_fallback(tmp_path):
    done = tmp_path / "done.txt"
    done.write_text("2023122600\n2023122601\n")

    def broken(src):
        raise OSError("no credentials")

    ids, src = submit.load_done("s3://wwe-ptype/ptype/inits.json", str(done), broken)
    assert ids == {"2023122600", "2023122601"} and src.startswith("done-list")
    local = tmp_path / "inits.json"
    local.write_text(json.dumps({"inits": [{"init": "2024-04-30T06:00Z"}]}))
    assert submit.load_done(str(local), None)[0] == {"2024043006"}


# ---------------------------------------------------------------------------------------------
# Finalize step (writers.finalize_ptype_store)
# ---------------------------------------------------------------------------------------------
def test_finalize_failure_blocks_upload_and_index(weights, tmp_path, monkeypatch):
    from ptype.hrrr_fast import writers

    def boom(path, *, remove_staging=True):
        raise OSError("disk full")

    monkeypatch.setattr(writers, "finalize_ptype_store", boom, raising=False)
    calls: list = []
    res = pipeline.run_pipeline(make_cfg(tmp_path, weights), pipeline.make_runs([INIT], [0]), uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls))
    assert res[0]["status"] == "finalize_failed" and "disk full" in res[0]["finalize"]["error"]
    assert calls == [("preflight",)]
    # Missing function -> the same, with a clear message.
    monkeypatch.delattr(writers, "finalize_ptype_store", raising=False)
    res = pipeline.run_pipeline(make_cfg(tmp_path / "b", weights), pipeline.make_runs([INIT], [0]), uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls))
    assert res[0]["status"] == "finalize_failed" and "not available" in res[0]["finalize"]["error"]


def test_interrupted_run_resumes_missing_hours_then_finalizes_once(weights, tmp_path, finalize_calls):
    class Broken(pipeline.SyntheticSource):
        def open_hour(self, init, fxx, log_extra=None):
            if fxx == 1:
                raise ValueError("node died")
            return super().open_hour(init, fxx, log_extra)

    calls: list = []
    cfg = make_cfg(tmp_path, weights)
    runs = pipeline.make_runs([INIT], [0, 1])
    res = pipeline.run_pipeline(cfg, runs, source=Broken(), uploader=RecordingUploader(calls), index_backend=RecordingIndex(calls))
    assert res[0]["status"] == "incomplete" and finalize_calls == []
    src = pipeline.SyntheticSource()
    idx = RecordingIndex(calls)
    res = pipeline.run_pipeline(cfg, runs, source=src, uploader=RecordingUploader(calls), index_backend=idx)
    assert list(src.reads) == [(pipeline.parse_init(INIT).isoformat(), 1)]  # only the missing hour
    assert res[0]["status"] == "indexed" and res[0]["written_hours"] == [0, 1] and len(finalize_calls) == 1
    # Finalized and indexed: skipped entirely next time.
    assert pipeline.run_pipeline(cfg, runs, source=src, uploader=RecordingUploader(calls), index_backend=idx)[0]["status"] == "skipped_indexed"
    # A finalized store is never written into again: extra hours are refused (left out, with a warning).
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1, 2]), source=src, uploader=RecordingUploader(calls), index_backend=idx, ignore_index=True)
    assert res[0]["written_hours"] == [0, 1] and (pipeline.parse_init(INIT).isoformat(), 2) not in src.reads and len(finalize_calls) == 1


def test_real_finalize_produces_readable_store(weights, tmp_path):
    from ptype.hrrr_fast import writers

    if not hasattr(writers, "finalize_ptype_store"):
        pytest.skip("writers.finalize_ptype_store not available yet")
    res = pipeline.run_pipeline(make_cfg(tmp_path, weights), pipeline.make_runs([INIT], [0, 2]))
    assert res[0]["status"] == "indexed" and res[0]["finalize"]["ok"]
    pk, _ = index.store_paths(pipeline.parse_init(INIT))
    g = zarr.open_group(tmp_path / "bucket" / pk, mode="r")
    assert g.attrs["hrrr_ptype"].get("layout") == "finalized"
    assert np.all(np.isfinite(g["0/ML_rain"][2])) and np.all(np.isnan(g["0/ML_rain"][1]))
    assert not list((tmp_path / "bucket" / Path(pk).parent).glob("*.staging")) and not list((tmp_path / "bucket" / Path(pk).parent).glob("*finalize-tmp"))


# ---------------------------------------------------------------------------------------------
# Output-bucket credentials (boto3 profile -> obstore credential provider); no network
# ---------------------------------------------------------------------------------------------
FAKE_KEY, FAKE_SECRET, FAKE_TOKEN = "AKIAFAKEKEYFORTESTS", "fakeSecretValue/ShouldNeverBeLogged", "fakeSessionTokenNeverLogged"


@pytest.fixture
def aws_files(tmp_path, monkeypatch):
    """A private ~/.aws with one profile holding temporary keys (like Derecho), and no other credential sources."""
    creds = tmp_path / "credentials"
    creds.write_text(f"[testprof]\naws_access_key_id = {FAKE_KEY}\naws_secret_access_key = {FAKE_SECRET}\naws_session_token = {FAKE_TOKEN}\n")
    conf = tmp_path / "config"
    conf.write_text("")
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(conf))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return tmp_path


def _no_secret_in_logs(caplog):
    text = " ".join(json.dumps(r.__dict__, default=str) for r in caplog.records)
    assert FAKE_SECRET not in text and FAKE_TOKEN not in text and FAKE_KEY not in text


def test_profile_credentials_reach_the_output_store(aws_files):
    pytest.importorskip("boto3")
    session = pipeline.resolve_aws_session("testprof", "us-east-1", verify=False)
    store = pipeline.output_s3_store("wwe-ptype", "us-east-1", session)
    assert store.config == {"bucket": "wwe-ptype", "region": "us-east-1"}  # no skip_signature: requests are signed
    cred = store.credential_provider()
    assert (cred["access_key_id"], cred["secret_access_key"], cred["token"]) == (FAKE_KEY, FAKE_SECRET, FAKE_TOKEN)
    quick = pipeline.output_s3_store("wwe-ptype", "us-east-1", session, quick=True)
    assert quick.client_options == pipeline.QUICK_CLIENT_OPTIONS and quick.retry_config["max_retries"] == 1
    assert FAKE_SECRET not in repr(store)
    # The HRRR input store stays anonymous and never gets credentials.
    src = pipeline.GribSource(PipelineConfig(source="s3"))
    assert src.store.config.get("skip_signature") == "true" and src.store.credential_provider is None


def test_missing_profile_and_rejected_keys(aws_files, caplog):
    boto3 = pytest.importorskip("boto3")
    from botocore.exceptions import ClientError

    with pytest.raises(pipeline.CredentialsError, match="'nope' not found"):
        pipeline.resolve_aws_session("nope", "us-east-1", verify=False)

    class FakeSTS:
        def get_caller_identity(self):
            raise ClientError({"Error": {"Code": "ExpiredToken", "Message": "The security token included in the request is expired"}}, "GetCallerIdentity")

    class ExpiredSession(boto3.Session):
        def client(self, *a, **kw):
            return FakeSTS()

    caplog.set_level("DEBUG")
    with pytest.raises(pipeline.CredentialsError, match="rejected \\(ExpiredToken\\)") as exc:
        pipeline.resolve_aws_session("testprof", "us-east-1", verify=True, session_factory=ExpiredSession)
    assert FAKE_SECRET not in str(exc.value) and exc.value.__cause__ is None and exc.value.__suppress_context__

    class OkSTS:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::123456789012:user/test", "Account": "123456789012"}

    class OkSession(boto3.Session):
        def client(self, *a, **kw):
            return OkSTS()

    pipeline.resolve_aws_session("testprof", "us-east-1", verify=True, session_factory=OkSession)
    assert any(getattr(r, "aws_arn", None) == "arn:aws:iam::123456789012:user/test" for r in caplog.records)
    _no_secret_in_logs(caplog)


def test_upload_fails_fast_on_bad_credentials_and_no_upload_only_warns(aws_files, weights, tmp_path, monkeypatch, caplog):
    pytest.importorskip("boto3")
    import time as _time

    cfg = make_cfg(tmp_path, weights, upload="s3")
    src = pipeline.SyntheticSource()
    monkeypatch.setenv("AWS_PROFILE", "missingprofile")
    with pytest.raises(pipeline.CredentialsError):
        pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0]), source=src)
    assert src.reads == {}  # failed before any input was read
    caplog.set_level("WARNING")
    t0 = _time.perf_counter()
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0]), source=src, no_upload=True)
    assert _time.perf_counter() - t0 < 60 and res[0]["status"] == "complete"
    assert any("not skipping indexed runs" in r.getMessage() for r in caplog.records)
    _no_secret_in_logs(caplog)


def test_unreadable_index_is_fatal_only_when_uploading(caplog):
    class Broken:
        def read(self):
            raise PermissionError("403 Forbidden")

    caplog.set_level("WARNING")
    assert pipeline._indexed_inits(Broken(), strict=False) == set()
    assert any("could not read the index" in r.getMessage() for r in caplog.records)
    with pytest.raises(PermissionError):
        pipeline._indexed_inits(Broken(), strict=True)


# ---------------------------------------------------------------------------------------------
# Node-wide download cap (max_concurrent_fetches)
# ---------------------------------------------------------------------------------------------
def test_fetch_slot_limits_concurrency_across_threads():
    import multiprocessing as mp

    pipeline.set_fetch_slots(mp.get_context("spawn").Semaphore(2))
    active, peak, lock = [0], [0], threading.Lock()

    def job():
        st = pipeline.ReadStats()
        with pipeline.fetch_slot(st):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            import time as _t

            _t.sleep(0.05)
            with lock:
                active[0] -= 1
        assert st.fetch_start_unix is not None and st.fetch_end_unix >= st.fetch_start_unix

    try:
        ts = [threading.Thread(target=job) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        pipeline.set_fetch_slots(None)
    assert peak[0] == 2


def test_max_concurrent_fetches_across_worker_processes(weights, tmp_path):
    """Three spawned workers, one download slot: the (simulated, 0.3 s) downloads never overlap in time."""
    cfg = make_cfg(tmp_path, weights, source="synthetic:sleep=0.3", processes=3, threads_per_process=1, max_concurrent_fetches=1)
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1, 2, 3, 4, 5]), no_upload=True)
    hours = res[0]["hours"].values()
    assert len({h["worker"] for h in hours}) >= 2
    spans = sorted((h["bytes"]["fetch_start_unix"], h["bytes"]["fetch_end_unix"]) for h in hours)
    assert all(b0 >= a1 - 1e-3 for (_, a1), (b0, _) in itertools.pairwise(spans))
    assert sum(h["bytes"]["fetch_wait_s"] for h in hours) > 0.2  # somebody had to wait for the slot


def test_config_rejects_bad_fetch_cap():
    with pytest.raises(ValueError):
        PipelineConfig(max_concurrent_fetches=0)
    assert PipelineConfig().max_concurrent_fetches is None and PipelineConfig().memory_per_process_gb == 6.0


def test_in_flight_counts_recent_submissions_qstat_does_not_list(tmp_path):
    """A job submitted seconds ago is in flight even when qstat (cached on Derecho) does not list it yet."""
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 11, 17, 3, 5, tzinfo=UTC)
    ledger = tmp_path / "submitted.jsonl"
    recs = [
        {"job_id": "1.srv", "runs": ["2023122100"], "time": (now - timedelta(seconds=2)).isoformat()},  # new, not in qstat yet
        {"job_id": "2.srv", "runs": ["2023122101"], "time": (now - timedelta(hours=3)).isoformat()},  # old and finished
        {"job_id": "3.srv", "runs": ["2023122102"], "time": (now - timedelta(hours=3)).isoformat()},  # old, still queued
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert submit.in_flight_runs(ledger, {"3"}, now=now) == {"2023122100", "2023122102"}
    assert submit.in_flight_runs(ledger, set(), now=now + timedelta(hours=1)) == set()
