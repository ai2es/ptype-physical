"""Tests for ptype.hrrr_fast.index.

Conditional puts: obstore's ``MemoryStore`` (the real object_store in-memory
implementation) enforces create/update-if-match in-process, so the concurrency
test below genuinely exercises the retry path.  ``LocalStore`` does not implement
update-if-match and is not used for it.  Nothing here talks to S3; S3's own
If-Match support is not tested.
"""

import json
import threading
from datetime import UTC, datetime

import obstore
import pytest
from obstore.store import MemoryStore
from ptype.hrrr_fast.index import empty_index, init_string, make_entry, merge_entry, store_paths, update_index

T0 = datetime(2024, 4, 30, 0, 0, tzinfo=UTC)
KEY = "ptype/inits.json"


def test_store_paths():
    assert store_paths(T0) == (
        "ptype/2024/04/30/hrrr_ptype_20240430_0000.zarr",
        "soundings/2024/04/30/hrrr_soundings_20240430_0000.zarr",
    )
    # naive is treated as UTC
    assert store_paths(datetime(2024, 4, 30, 6)) == store_paths(datetime(2024, 4, 30, 6, tzinfo=UTC))


def test_merge_format_and_idempotence():
    a = merge_entry(empty_index(), T0, [3, 1, 2], "2024/04/30/x.zarr", "soundings/s.zarr", updated=T0)
    assert a == {
        "version": 1,
        "updated": "2024-04-30T00:00:00Z",
        "inits": [{"init": "2024-04-30T00:00Z", "forecast_hours": [1, 2, 3], "store": "2024/04/30/x.zarr", "soundings": "soundings/s.zarr"}],
    }
    b = merge_entry(a, T0, [1, 2, 3], "2024/04/30/x.zarr", "soundings/s.zarr", updated=T0)
    assert a == b


def test_merge_replaces_not_unions_and_sorts():
    idx = merge_entry(empty_index(), datetime(2024, 5, 1, tzinfo=UTC), [1], "b", None)
    idx = merge_entry(idx, T0, [1, 2, 3], "a", "s")
    idx2 = merge_entry(idx, T0, [1], "a", None)
    assert [e["init"] for e in idx2["inits"]] == ["2024-04-30T00:00Z", "2024-05-01T00:00Z"]
    assert idx2["inits"][0]["forecast_hours"] == [1]
    assert "soundings" not in idx2["inits"][0]
    assert len(idx["inits"]) == 2 and idx["inits"][0]["forecast_hours"] == [1, 2, 3]  # input not mutated


def test_merge_keeps_legacy_entries_and_unknown_keys():
    legacy = {"version": 1, "extra": 5, "inits": [{"init": "2024-04-29T00:00Z", "max_forecast_hour": 18}]}
    out = merge_entry(legacy, T0, [1], "a", None)
    assert out["extra"] == 5
    assert out["inits"][0] == {"init": "2024-04-29T00:00Z", "max_forecast_hour": 18}


def test_update_creates_then_updates():
    store = MemoryStore()
    update_index(store, KEY, make_entry(T0, [1, 2]))
    update_index(store, KEY, make_entry(datetime(2024, 4, 30, 1, tzinfo=UTC), [1]))
    data = json.loads(bytes(obstore.get(store, KEY).bytes()))
    assert [e["init"] for e in data["inits"]] == ["2024-04-30T00:00Z", "2024-04-30T01:00Z"]
    assert data["inits"][0]["store"] == "2024/04/30/hrrr_ptype_20240430_0000.zarr"
    assert data["inits"][0]["soundings"].startswith("soundings/2024/04/30/")


def test_memorystore_enforces_conditional_put():
    """Guard the premise of the concurrency test: the store must reject a stale ETag."""
    store = MemoryStore()
    first = obstore.put(store, "k", b"1", mode="create")
    obstore.put(store, "k", b"2", mode={"e_tag": first["e_tag"]})
    with pytest.raises(obstore.exceptions.PreconditionError):
        obstore.put(store, "k", b"3", mode={"e_tag": first["e_tag"]})
    with pytest.raises(obstore.exceptions.AlreadyExistsError):
        obstore.put(store, "k", b"3", mode="create")


def test_concurrent_updates_lose_nothing():
    store = MemoryStore()
    n = 40
    barrier = threading.Barrier(n)
    errors = []

    def work(i):
        init = datetime(2024, 1, 1, tzinfo=UTC).replace(day=1 + i // 24, hour=i % 24)
        try:
            barrier.wait()
            update_index(store, KEY, make_entry(init, [1, 2, 3]), max_retries=200)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    data = json.loads(bytes(obstore.get(store, KEY).bytes()))
    assert len(data["inits"]) == n
    expected = {init_string(datetime(2024, 1, 1 + i // 24, i % 24, tzinfo=UTC)) for i in range(n)}
    assert {e["init"] for e in data["inits"]} == expected
