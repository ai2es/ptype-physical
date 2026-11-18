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
