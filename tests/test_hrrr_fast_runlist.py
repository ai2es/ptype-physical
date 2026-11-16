"""Tests for the HRRR fast-inference run list.

The shipped events file is checked against hand-computed totals. Those totals are the
contract for what gets downloaded, so a change to the events file should be a deliberate
update to the expected numbers here rather than a silent drift.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml
from ptype.hrrr_fast.runlist import build_run_list, event_window, max_forecast_hour, merge_windows, summarize

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "hrrr_fast_events.yml"


def utc(year, month, day, hour=0):
    return datetime(year, month, day, hour, tzinfo=UTC)


@pytest.mark.parametrize("init_hour", [0, 6, 12, 18])
def test_long_range_cycles_reach_forecast_hour_48(init_hour):
    assert max_forecast_hour(init_hour) == 48


@pytest.mark.parametrize("init_hour", [1, 2, 5, 7, 11, 13, 17, 19, 23])
def test_short_range_cycles_stop_at_forecast_hour_18(init_hour):
    assert max_forecast_hour(init_hour) == 18


def test_event_window_edges():
    first, last = event_window(date(2025, 11, 9), date(2025, 11, 11), lead_days=2)
    assert first == utc(2025, 11, 7, 0)
    assert last == utc(2025, 11, 11, 23)
    assert first.tzinfo is not None


def test_event_window_crosses_month_and_year_boundaries():
    first, _ = event_window(date(2026, 1, 1), date(2026, 1, 3), lead_days=2)
    assert first == utc(2025, 12, 30, 0)


def test_merge_windows_merges_overlapping_and_touching():
    windows = [
        (utc(2025, 1, 5), utc(2025, 1, 9, 23)),
        (utc(2025, 1, 1), utc(2025, 1, 4, 23)),
        (utc(2025, 1, 10), utc(2025, 1, 12)),
    ]
    assert merge_windows(windows) == [(utc(2025, 1, 1), utc(2025, 1, 12))]


def test_merge_windows_keeps_separated_windows_sorted():
    windows = [
        (utc(2025, 2, 1), utc(2025, 2, 2)),
        (utc(2025, 1, 1, 0), utc(2025, 1, 1, 5)),
        (utc(2025, 1, 1, 7), utc(2025, 1, 1, 9)),
    ]
    assert merge_windows(windows) == [
        (utc(2025, 1, 1, 0), utc(2025, 1, 1, 5)),
        (utc(2025, 1, 1, 7), utc(2025, 1, 1, 9)),
        (utc(2025, 2, 1), utc(2025, 2, 2)),
    ]


def test_build_run_list_forecast_hours_follow_cycle_rule():
    config = {"lead_days": 0, "events": [{"event_start": date(2025, 1, 1), "event_end": date(2025, 1, 1)}]}
    runs = build_run_list(config)
    assert len(runs) == 24
    assert all(run.init.tzinfo is not None for run in runs)
    assert [run.init for run in runs] == sorted(run.init for run in runs)
    by_hour = {run.init.hour: run for run in runs}
    assert by_hour[0].forecast_hours == tuple(range(49))
    assert by_hour[6].forecast_hours == tuple(range(49))
    assert by_hour[1].forecast_hours == tuple(range(19))
    assert by_hour[23].forecast_hours == tuple(range(19))


def test_build_run_list_honours_init_interval():
    config = {
        "lead_days": 0,
        "init_interval_hours": 6,
        "events": [{"event_start": date(2025, 1, 1), "event_end": date(2025, 1, 2)}],
    }
    runs = build_run_list(config)
    assert [run.init.hour for run in runs[:4]] == [0, 6, 12, 18]
    assert len(runs) == 8


def test_build_run_list_merges_overlapping_events_without_duplicates():
    config = {
        "lead_days": 2,
        "events": [
            {"event_start": date(2025, 1, 10), "event_end": date(2025, 1, 12)},
            {"event_start": date(2025, 1, 11), "event_end": date(2025, 1, 14)},
        ],
    }
    runs = build_run_list(config)
    inits = [run.init for run in runs]
    assert len(inits) == len(set(inits))
    assert inits[0] == utc(2025, 1, 8, 0)
    assert inits[-1] == utc(2025, 1, 14, 23)
    assert len(inits) == 7 * 24


def test_summarize_splits_blocks_at_gaps():
    config = {
        "lead_days": 0,
        "events": [
            {"event_start": date(2025, 1, 1), "event_end": date(2025, 1, 1)},
            {"event_start": date(2025, 1, 3), "event_end": date(2025, 1, 3)},
        ],
    }
    runs = build_run_list(config)
    summary = summarize(runs)
    assert summary["n_runs"] == 48
    assert summary["n_forecast_hour_files"] == 2 * 576
    assert summary["n_days"] == 2
    assert len(summary["windows"]) == 2
    assert summary["windows"][0]["first_init"] == utc(2025, 1, 1, 0)
    assert summary["windows"][0]["last_init"] == utc(2025, 1, 1, 23)
    assert summary["windows"][0]["n_days"] == 1
    assert summary["windows"][1]["first_init"] == utc(2025, 1, 3, 0)


def test_shipped_events_file_totals():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert len(config["events"]) == 16
    assert config["lead_days"] == 2
    assert config["init_interval_hours"] == 1

    runs = build_run_list(config)
    summary = summarize(runs, config["init_interval_hours"])

    # Per UTC day: 4 long-range cycles * 49 files + 20 short-range cycles * 19 files.
    files_per_day = 4 * 49 + 20 * 19
    assert files_per_day == 576
    assert summary["n_days"] == 85
    assert summary["n_runs"] == 85 * 24 == 2040
    assert summary["n_forecast_hour_files"] == 85 * files_per_day == 48960
