"""Tests of the realtime driver (``ptype.hrrr_fast.realtime``). No network, no AWS, no waiting.

The synthetic source and tiny network of ``test_hrrr_fast_pipeline`` are reused. Arrival of HRRR files is
simulated by a scripted ``available`` callable and a fake clock that advances only when the driver "sleeps",
so a four-hour wait takes no time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("cast_value_rs")
pytest.importorskip("pyproj")
pytest.importorskip("numba")

from ptype.hrrr_fast import index, pipeline, realtime

from test_hrrr_fast_pipeline import INIT, make_cfg, weights  # noqa: F401 - weights is a fixture

T0 = datetime(2023, 12, 26, 0, tzinfo=UTC)


class Clock:
    """A clock that moves only when the driver sleeps, plus a schedule of when each forecast hour 'arrives'."""

    def __init__(self, arrivals: dict[int, float], start_min: float = 50.0):
        self.t = T0 + timedelta(minutes=start_min)
        self.arrivals = arrivals  # forecast hour -> minutes after initialization
        self.sleeps = 0

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += timedelta(seconds=seconds)

    def available(self) -> set[int]:
        minutes = (self.t - T0).total_seconds() / 60
        return {h for h, m in self.arrivals.items() if m <= minutes}


def read_index(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "bucket" / "ptype" / "inits.json").read_text())


def run(cfg, hours, clock, **kw):
    spec = realtime.realtime_spec(INIT, hours)
    return realtime.run_realtime(cfg, spec, poll_s=30, available=clock.available, now=clock.now, sleep=clock.sleep, **kw)


def test_hours_are_published_as_they_arrive_then_the_entry_moves_to_the_archive_key(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights, keep_local=False)
    clock = Clock({0: 53, 1: 54, 2: 56, 3: 58})
    seen: list[tuple[list[int], str, object]] = []
    backend = pipeline.LocalIndex(tmp_path / "bucket" / index.INDEX_KEY)
    update = backend.update

    def recording_update(entry):
        out = update(entry)
        e = out["inits"][0]
        seen.append((e["forecast_hours"], e["store"], e.get("complete")))
        return out

    backend.update = recording_update
    m = run(cfg, [0, 1, 2, 3], clock, uploader=pipeline.LocalUploader(tmp_path / "bucket"), index_backend=backend)
    assert m["status"] == "indexed" and m["written_hours"] == [0, 1, 2, 3]
    # The index grew one arrival at a time at the staging key, and only the last write is the archive key.
    assert [s[0] for s in seen] == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert all(s[1].startswith("realtime-staging/") and s[2] is False for s in seen[:-1])
    pk, sk = index.store_paths(T0)
    assert seen[-1][1] == pk.removeprefix("ptype/") and seen[-1][2] is True
    e = read_index(tmp_path)["inits"][0]
    assert e["realtime"] is True and e["complete"] is True and isinstance(e["frozen_cells"], int)
    # First hour is published right after it arrives, not at the end of the run.
    assert m["published"]["0"]["minutes_after_init"] < 54 and m["published"]["3"]["minutes_after_init"] >= 58
    # Staging holds per-hour objects that read correctly; the archive holds the consolidated store.
    staged = zarr.open_group(tmp_path / "bucket" / realtime.staging_key(T0), mode="r")
    final = zarr.open_group(tmp_path / "bucket" / pk, mode="r")
    assert staged.attrs["hrrr_ptype"]["layout"] == "staging" and final.attrs["hrrr_ptype"]["layout"] == "finalized"
    np.testing.assert_array_equal(staged["0/ML_snow"][2], final["0/ML_snow"][2])
    s = zarr.open_array(tmp_path / "bucket" / sk, path="sounding", mode="r")
    assert np.all(np.isfinite(s[3])) and np.all(np.isnan(s[4]))
    assert not (tmp_path / "work" / INIT / Path(pk).name).exists()  # local copy removed after indexing
    status = json.loads((tmp_path / "bucket" / "ptype" / realtime.STATUS_NAME).read_text())
    assert status["state"] == "indexed" and status["hours_published"] == [0, 1, 2, 3] and status["last_error"] is None


def test_waits_for_the_first_file_and_reports_a_run_that_never_arrives(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    clock = Clock({})
    m = run(cfg, [0, 1], clock, give_up_after=timedelta(hours=2))
    assert m["status"] == "missing" and clock.now() >= T0 + timedelta(hours=2) and clock.sleeps > 100
    assert not (tmp_path / "bucket" / "ptype" / "inits.json").exists()


def test_a_run_cut_short_stays_at_the_staging_key_and_a_batch_pass_completes_it(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    clock = Clock({0: 53, 1: 55, 3: 60})  # hour 2 never arrives
    m = run(cfg, [0, 1, 2, 3], clock, give_up_after=timedelta(hours=2))
    assert m["status"] == "published_partial" and m["written_hours"] == [0, 1, 3] and m["missing_hours"] == [2]
    e = read_index(tmp_path)["inits"][0]
    assert e["forecast_hours"] == [0, 1, 3] and e["complete"] is False and e["store"].startswith("realtime-staging/")
    assert json.loads((tmp_path / "bucket" / "ptype" / realtime.STATUS_NAME).read_text())["state"] == "partial"
    # The batch driver does not treat the unfinished entry as done: it fills the gap and finalizes.
    res = pipeline.run_pipeline(cfg, pipeline.make_runs([INIT], [0, 1, 2, 3]))
    assert res[0]["status"] == "indexed" and res[0]["written_hours"] == [0, 1, 2, 3]
    e = read_index(tmp_path)["inits"][0]
    assert e["forecast_hours"] == [0, 1, 2, 3] and not e["store"].startswith("realtime-staging/")


def test_a_listed_hour_that_cannot_be_read_is_retried_then_recorded(weights, tmp_path):  # noqa: F811
    # The listing says hour 1 is there but the read finds it missing every time (a truncated upload, say).
    cfg = make_cfg(tmp_path, weights, source="synthetic:missing=1")
    clock = Clock({0: 53, 1: 54, 2: 55})
    m = run(cfg, [0, 1, 2], clock, give_up_after=timedelta(hours=2), max_attempts=3)
    assert m["status"] == "published_partial" and m["written_hours"] == [0, 2] and m["missing_hours"] == [1]
    assert clock.now() < T0 + timedelta(hours=2)  # gave up on the attempts, did not sit out the clock


def test_restart_republishes_what_is_on_disk_and_processes_only_the_rest(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    m1 = run(cfg, [0, 1, 2], Clock({0: 53, 1: 54}), give_up_after=timedelta(hours=1))  # job dies with hour 2 outstanding
    assert m1["written_hours"] == [0, 1]
    (tmp_path / "bucket" / "ptype" / "inits.json").unlink()  # as if the first job never got its index write through
    src = pipeline.SyntheticSource()
    m2 = run(cfg, [0, 1, 2], Clock({0: 53, 1: 54, 2: 70}, start_min=65), source=src)
    assert m2["status"] == "indexed" and m2["written_hours"] == [0, 1, 2]
    assert {k[1] for k in src.reads} == {2}  # hours 0 and 1 were not read again
    assert read_index(tmp_path)["inits"][0]["forecast_hours"] == [0, 1, 2]


def test_publish_failure_does_not_stop_the_run_and_is_made_up_later(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    up = pipeline.LocalUploader(tmp_path / "bucket")
    real, calls = up.upload_files, {"n": 0}

    def flaky(local, key, rel_paths, storage_class=None):
        calls["n"] += 1
        if calls["n"] <= 2:  # the first publish (two static uploads) fails
            raise ConnectionError("s3 down")
        return real(local, key, rel_paths, storage_class)

    up.upload_files = flaky
    m = run(cfg, [0, 1, 2], Clock({0: 53, 1: 54, 2: 55}), uploader=up, index_backend=pipeline.LocalIndex(tmp_path / "bucket" / index.INDEX_KEY))
    assert m["status"] == "indexed" and m["written_hours"] == [0, 1, 2]
    assert read_index(tmp_path)["inits"][0]["forecast_hours"] == [0, 1, 2]


def test_no_upload_keeps_everything_local(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    m = run(cfg, [0, 1], Clock({0: 53, 1: 54}), no_upload=True)
    assert m["status"] == "complete" and not (tmp_path / "bucket").exists()


def test_store_files_separates_metadata_from_hours(weights, tmp_path):  # noqa: F811
    cfg = make_cfg(tmp_path, weights)
    run(cfg, [0, 2], Clock({0: 53, 2: 54}), no_upload=True)
    # no_upload finalizes the gridded store, so look at the sounding store, which keeps one shard per hour.
    static, by_hour = realtime.store_files(next((tmp_path / "work" / INIT).glob("hrrr_soundings_*.zarr")))
    assert "zarr.json" in static and "sounding/zarr.json" in static and "forecast_hour/c/0" in static
    assert sorted(by_hour) == [0, 2] and by_hour[2] == ["sounding/c/2/0/0"]


def test_cleanup_removes_only_old_unreferenced_staging_stores(tmp_path):
    up = pipeline.LocalUploader(tmp_path / "bucket")
    idx = pipeline.LocalIndex(tmp_path / "bucket" / index.INDEX_KEY)

    def make(init: datetime) -> str:
        key = realtime.staging_key(init)
        (tmp_path / "bucket" / key / "0").mkdir(parents=True)
        (tmp_path / "bucket" / key / "zarr.json").write_text("{}")
        return key

    old_done, old_partial, recent = make(T0), make(T0 + timedelta(hours=1)), make(T0 + timedelta(hours=9))
    archive = tmp_path / "bucket" / index.store_paths(T0)[0]
    archive.mkdir(parents=True)
    (archive / "zarr.json").write_text("{}")
    idx.update(index.make_entry(T0, [0, 1]))  # finalized: points at the archive key
    partial = index.make_entry(T0 + timedelta(hours=1), [0])
    partial["store"] = old_partial.removeprefix("ptype/")
    idx.update(partial)  # cut short: still points at its staging store
    removed = realtime.cleanup_staging(up, idx, now=T0 + timedelta(hours=10))
    assert removed == [old_done]
    assert not (tmp_path / "bucket" / old_done).exists() and (tmp_path / "bucket" / old_partial).exists() and (tmp_path / "bucket" / recent).exists()
    assert archive.exists() and (tmp_path / "bucket" / "ptype" / "inits.json").exists()


@pytest.mark.parametrize(
    "key", ["ptype/2023/12/26/hrrr_ptype_20231226_0000.zarr", "ptype/inits.json", "ptype/realtime-staging", "ptype/realtime-staging/../2023/x.zarr", "soundings/a.zarr"]
)
def test_delete_refuses_anything_outside_the_staging_prefix(tmp_path, key):
    up = pipeline.LocalUploader(tmp_path / "bucket")
    with pytest.raises(ValueError, match="refusing to delete"):
        up.delete_store(key, realtime.STAGING_PREFIX)


def test_available_needs_both_the_grib_file_and_its_idx(tmp_path):
    mirror = tmp_path / "mirror" / "hrrr.20231226" / "conus"
    mirror.mkdir(parents=True)
    for name in (
        "hrrr.t00z.wrfnatf00.grib2",
        "hrrr.t00z.wrfnatf00.grib2.idx",
        "hrrr.t00z.wrfnatf01.grib2",
        "hrrr.t00z.wrfnatf02.grib2.idx",
        "hrrr.t01z.wrfnatf03.grib2",
        "hrrr.t01z.wrfnatf03.grib2.idx",
    ):
        (mirror / name).write_text("x")
    src = pipeline.GribSource(pipeline.PipelineConfig(source=str(tmp_path / "mirror"), weights="unused"))
    assert src.available(T0, range(5)) == {0}  # 01 has no idx, 02 has no GRIB file, 03 belongs to another cycle


def test_cli_realtime_flag(weights, tmp_path, monkeypatch):  # noqa: F811
    import hrrr_ptype_run

    cfg_file = tmp_path / "cfg.yml"
    cfg_file.write_text(json.dumps(make_cfg(tmp_path, weights).to_dict()))
    assert hrrr_ptype_run.main(["--config", str(cfg_file), "--realtime", "--init", INIT, "--init", "2023122601"]) == 2
    rc = hrrr_ptype_run.main(
        ["--config", str(cfg_file), "--realtime", "--init", INIT, "--forecast-hours", "0-1", "--poll-seconds", "0.01", "--summary-json", str(tmp_path / "s.json")]
    )
    assert rc == 0 and json.loads((tmp_path / "s.json").read_text())[0]["status"] == "indexed"


# ---------------------------------------------------------------------------------------------
# The cron launcher (applications/hrrr_ptype_realtime_submit.py)
# ---------------------------------------------------------------------------------------------
import hrrr_ptype_realtime_submit as rts


def launcher_args(tmp_path, now, *extra):
    return ["--jobs-dir", str(tmp_path / "jobs"), "--account", "TEST0001", "--aws-profile", "some-profile", "--now", now, *extra]


def fake_qsub(log):
    def run_cmd(cmd):
        log.append(list(cmd))
        if cmd[0] == "qsub":
            return f"{1000 + sum(1 for c in log if c[0] == 'qsub')}.casper-pbs\n"
        if cmd[0] == "qstat":
            return ""  # nothing queued or running
        raise AssertionError(cmd)

    return run_cmd


def entry(run_id, complete=None):
    e = {"init": datetime.strptime(run_id, "%Y%m%d%H").strftime("%Y-%m-%dT%H:%MZ"), "forecast_hours": [0], "store": "x.zarr"}
    if complete is not None:
        e["complete"] = complete
    return e


def test_launcher_submits_each_new_run_once_at_the_lead_time(tmp_path):
    log: list = []
    idx = {"inits": []}
    # 12:44: the 12Z run is 44 minutes old, one short of the lead time. Older runs inside the lookback start.
    first = rts.main(launcher_args(tmp_path, "202311261244", "--lookback-hours", "2", "--no-catchup"), fake_qsub(log), lambda _u: idx)
    assert [r["runs"] for r in first] == [["2023112611"]]
    assert rts.main(launcher_args(tmp_path, "202311261245", "--lookback-hours", "2", "--no-catchup"), fake_qsub(log), lambda _u: idx)[0]["runs"] == ["2023112612"]
    # Nothing new five minutes later: both are in the ledger for good, whatever qstat says.
    assert rts.main(launcher_args(tmp_path, "202311261250", "--lookback-hours", "2", "--no-catchup"), fake_qsub(log), lambda _u: idx) == []
    script = (tmp_path / "jobs" / "rt_2023112612.pbs").read_text()
    assert "#PBS -A TEST0001" in script and "#PBS -q casper@casper-pbs" in script and "#PBS -l walltime=02:00:00" in script
    assert "--realtime --init 2023112612" in script and "export AWS_PROFILE=some-profile" in script and "venv_casper" in script
    assert "#PBS -l walltime=01:15:00" in (tmp_path / "jobs" / "rt_2023112611.pbs").read_text()  # an 18 h cycle


def test_launcher_skips_runs_the_index_lists_as_complete(tmp_path):
    idx = {"inits": [entry("2023112611"), entry("2023112612", complete=False)]}
    out = rts.main(launcher_args(tmp_path, "202311261300", "--lookback-hours", "2", "--no-catchup"), fake_qsub([]), lambda _u: idx)
    assert [r["runs"] for r in out] == [["2023112612"]]  # 11Z is done (archive style entry); 12Z is still arriving


def test_launcher_catches_up_unfinished_runs_newest_first_with_a_try_limit(tmp_path):
    log: list = []
    idx = {"inits": [entry("2023112606"), entry("2023112605", complete=False)]}

    def at(now):
        return launcher_args(tmp_path, now, "--lookback-hours", "0.5", "--catchup-hours", "8", "--max-catchup", "2", "--max-attempts", "2")

    out = rts.main(at("202311261200"), fake_qsub(log), lambda _u: idx)
    # Realtime gives up at +4 h, so the newest run catch-up may take is 07Z (4 h 10 min before 12:00 is 07:50).
    assert [(r["kind"], r["runs"]) for r in out] == [("catchup", ["2023112607", "2023112605"])]  # 06Z is complete
    runs_file = next((tmp_path / "jobs").glob("catchup_*.runs")).read_text()
    assert runs_file == "2023112607 18\n2023112605 18\n"
    assert "--runs-file" in next((tmp_path / "jobs").glob("catchup_*.pbs")).read_text()
    # Half an hour later the first job is gone (qstat empty, grace period over) and 08Z has passed the give-up
    # time: the two newest unfinished runs are taken, so 07Z gets its second try. At 13:00 07Z has used both
    # tries and is left alone, and 05Z has dropped out of the 8-hour window.
    again = rts.main(at("202311261230"), fake_qsub(log), lambda _u: idx)
    assert again[0]["runs"] == ["2023112608", "2023112607"]
    third = rts.main(at("202311261300"), fake_qsub(log), lambda _u: idx)
    assert [r["runs"] for r in third if r["kind"] == "catchup"] == [["2023112608"]]


def test_launcher_without_an_index_still_starts_realtime_but_not_catchup(tmp_path, capsys):
    def broken(_url):
        raise OSError("no route")

    out = rts.main(launcher_args(tmp_path, "202311261300", "--lookback-hours", "1.5"), fake_qsub([]), broken)
    assert [(r["kind"], r["runs"]) for r in out] == [("realtime", ["2023112612"])]
    assert "could not read the index" in capsys.readouterr().err


def test_launcher_dry_run_writes_nothing(tmp_path):
    out = rts.main(launcher_args(tmp_path, "202311261300", "--lookback-hours", "1.5", "--dry-run"), fake_qsub([]), lambda _u: {"inits": []})
    assert out and not (tmp_path / "jobs").exists()


def test_launcher_survives_a_refused_qsub_and_retries_next_pass(tmp_path):
    def refuse(cmd):
        if cmd[0] == "qsub":
            raise RuntimeError("qsub: Unknown queue")
        return ""

    args = launcher_args(tmp_path, "202311261300", "--lookback-hours", "1.5", "--no-catchup")
    assert rts.main(args, refuse, lambda _u: {"inits": []}) == []
    assert rts.main(args, fake_qsub([]), lambda _u: {"inits": []})[0]["runs"] == ["2023112612"]


# ---------------------------------------------------------------------------------------------
# Retention (applications/hrrr_ptype_expire.py)
# ---------------------------------------------------------------------------------------------
import hrrr_ptype_expire as expire


def retention_bucket(tmp_path):
    """A local bucket with an archive run, and realtime runs that are old/new and snowy/dry."""
    up = pipeline.LocalUploader(tmp_path / "bucket")
    idx = pipeline.LocalIndex(tmp_path / "bucket" / index.INDEX_KEY)
    runs = {
        "archive": (datetime(2023, 12, 26, 0, tzinfo=UTC), None),  # no realtime flag at all
        "old_dry": (datetime(2026, 7, 1, 0, tzinfo=UTC), {"realtime": True, "complete": True, "frozen_cells": 12}),
        "old_snowy": (datetime(2026, 7, 1, 1, tzinfo=UTC), {"realtime": True, "complete": True, "frozen_cells": 100}),
        "old_unknown": (datetime(2026, 7, 1, 2, tzinfo=UTC), {"realtime": True, "complete": True}),
        "new_dry": (datetime(2026, 8, 20, 0, tzinfo=UTC), {"realtime": True, "complete": True, "frozen_cells": 0}),
    }
    for init, extra in runs.values():
        for key, _ in expire.store_keys(index.init_string(init)):
            d = tmp_path / "bucket" / key
            (d / "0").mkdir(parents=True)
            (d / "zarr.json").write_text("{}")
            (d / "0" / "data").write_text("x")
        e = index.make_entry(init, [0, 1])
        if extra:
            e["extra"] = extra
        idx.update(e)
    return up, idx, {k: index.init_string(v[0]) for k, v in runs.items()}


def test_expire_is_a_dry_run_unless_told_to_delete(tmp_path, capsys):
    _, _, names = retention_bucket(tmp_path)
    before = sorted(p.as_posix() for p in (tmp_path / "bucket").rglob("*"))
    out = expire.main(["--local-root", str(tmp_path / "bucket"), "--now", "202609010000"])
    assert out["expire"] == [names["old_dry"]] and out["unknown"] == [names["old_unknown"]] and out["removed"] == []
    assert sorted(p.as_posix() for p in (tmp_path / "bucket").rglob("*")) == before
    assert "dry run: nothing was changed" in capsys.readouterr().out


def test_expire_removes_only_old_realtime_runs_below_the_threshold(tmp_path):
    _, idx, names = retention_bucket(tmp_path)
    out = expire.main(["--local-root", str(tmp_path / "bucket"), "--now", "202609010000", "--delete"])
    assert out["removed"] == [names["old_dry"]] and out["objects"] == 6  # three stores of two files each
    left = {e["init"] for e in idx.read()["inits"]}
    assert left == {names["archive"], names["old_snowy"], names["old_unknown"], names["new_dry"]}
    for key, _ in expire.store_keys(names["old_dry"]):
        assert not (tmp_path / "bucket" / key).exists()
    for name in ("archive", "old_snowy", "old_unknown", "new_dry"):
        for key, _ in expire.store_keys(names[name]):
            assert (tmp_path / "bucket" / key / "zarr.json").exists()


def test_index_remove_entries_is_pure_and_keeps_the_rest():
    idx = index.merge_entry(index.merge_entry(index.empty_index(), T0, [0], "a.zarr", None), T0 + timedelta(hours=1), [0], "b.zarr", None)
    out = index.remove_entries(idx, {index.init_string(T0)})
    assert [e["store"] for e in out["inits"]] == ["b.zarr"] and len(idx["inits"]) == 2
