"""Build the list of HRRR forecast runs needed to cover a set of storm events.

The fast HRRR inference pipeline should only download and process the model runs that
can contribute to the event windows. Each event window is padded by a lead time so that
forecast hours verifying at the start of the event were initialized beforehand, then
overlapping or touching windows are merged so no run is listed twice.

Forecast-hour coverage depends on the init cycle. The 00, 06, 12 and 18 UTC cycles are
the long-range HRRR runs and are archived out to 48 hours; the hourly cycles only run
out to 18 hours. Counting forecast-hour files therefore needs the cycle-aware rule in
`max_forecast_hour` rather than one number for every run.

Run from the repository root:

    python -m ptype.hrrr_fast.runlist config/hrrr_fast_events.yml --out runs.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import yaml

LONG_RANGE_INIT_HOURS = (0, 6, 12, 18)
LONG_RANGE_MAX_FORECAST_HOUR = 48
SHORT_RANGE_MAX_FORECAST_HOUR = 18

Window = tuple[datetime, datetime]


@dataclass(frozen=True)
class Run:
    """One HRRR model initialization and the forecast hours to process from it.

    Attributes:
        init: Model initialization time, timezone-aware in UTC.
        forecast_hours: Every forecast hour from 0 through the maximum for this cycle,
            inclusive. Hour 0 (the analysis) is counted as a file like any other forecast hour.
    """

    init: datetime
    forecast_hours: tuple[int, ...]


def max_forecast_hour(init_hour: int) -> int:
    """Return the longest forecast hour available for an init cycle.

    The 48-hour cutoff applies only to the 00, 06, 12 and 18 UTC cycles, which are the
    long-range runs. Using 18 hours for every cycle would undercount the files per day
    (576 with this rule, versus 4 * 19 + 20 * 19 = 456 with a flat 18-hour limit).

    Args:
        init_hour: Initialization hour in UTC, 0 through 23.

    Returns:
        48 for the 00, 06, 12 and 18 UTC cycles, otherwise 18.
    """
    return LONG_RANGE_MAX_FORECAST_HOUR if init_hour in LONG_RANGE_INIT_HOURS else SHORT_RANGE_MAX_FORECAST_HOUR


def event_window(event_start: date, event_end: date, lead_days: int) -> Window:
    """Return the first and last model init times that can serve an event.

    The first init is 00 UTC `lead_days` before the event start, so the earliest valid
    forecast hour of the event is covered by a run that was already available for it. The
    last init is 23 UTC on the final event day, so runs initialized late in that day are
    still included.

    Args:
        event_start: First day of the event (UTC).
        event_end: Last day of the event (UTC), inclusive.
        lead_days: Number of days of lead time to include before the event start.

    Returns:
        Tuple of (first_init, last_init), both timezone-aware UTC datetimes.
    """
    first_init = datetime.combine(event_start - timedelta(days=lead_days), time(0), tzinfo=UTC)
    last_init = datetime.combine(event_end, time(23), tzinfo=UTC)
    return first_init, last_init


def merge_windows(windows: Iterable[Window], step: timedelta = timedelta(hours=1)) -> list[Window]:
    """Merge event windows that overlap or touch, returning them sorted by start.

    Two windows are merged when the next one starts no later than one `step` after the
    previous one ends. That makes adjacent hours (for example 23 UTC and 00 UTC of the next
    day) merge too, because both would otherwise be separate blocks of the same run list.
    Merging before enumeration is what keeps the total run count free of double counting.

    Args:
        windows: (start, end) pairs with inclusive endpoints.
        step: Spacing between consecutive init times.

    Returns:
        Disjoint, sorted list of (start, end) windows.
    """
    merged: list[Window] = []
    for start, end in sorted(windows, key=lambda window: window[0]):
        if merged and start <= merged[-1][1] + step:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _init_interval_hours(config: dict) -> int:
    """Return the init spacing from the config, requiring it to divide the day evenly.

    A spacing that does not divide 24 would make the hour-of-day filter produce irregular
    cycles on different days, which breaks the per-cycle forecast-hour rule.
    """
    interval = int(config.get("init_interval_hours", 1))
    if interval < 1 or 24 % interval != 0:
        raise ValueError(f"init_interval_hours must be a positive divisor of 24, got {interval}")
    return interval


def build_run_list(config: dict) -> list[Run]:
    """Build the sorted, de-duplicated list of runs for every event in a config.

    Args:
        config: Parsed YAML with an `events` list (each with `event_start` and `event_end`
            dates), plus optional `lead_days` (default 2) and `init_interval_hours`
            (default 1).

    Returns:
        Runs sorted by init time. Each run lists forecast hours 0 through its maximum.
    """
    lead_days = int(config.get("lead_days", 2))
    interval = _init_interval_hours(config)
    windows = [event_window(event["event_start"], event["event_end"], lead_days) for event in config["events"]]

    inits: set[datetime] = set()
    for first, last in merge_windows(windows):
        current = first
        while current <= last:
            if current.hour % interval == 0:
                inits.add(current)
            current += timedelta(hours=1)

    return [Run(init=init, forecast_hours=tuple(range(max_forecast_hour(init.hour) + 1))) for init in sorted(inits)]


def summarize(runs: Sequence[Run], init_interval_hours: int = 1) -> dict:
    """Summarize a run list as totals plus one entry per contiguous block of runs.

    A block ends where the gap between consecutive inits is larger than the init spacing,
    so the per-window rows match the merged event windows the runs came from.

    Args:
        runs: Runs sorted by init time, as returned by `build_run_list`.
        init_interval_hours: Spacing between inits used to split the blocks.

    Returns:
        Dict with `n_runs`, `n_forecast_hour_files`, `n_days` (distinct UTC init dates) and
        `windows`, a list of dicts holding `first_init`, `last_init`, `n_days` and `n_runs`.
    """
    spacing = timedelta(hours=init_interval_hours)
    blocks: list[list[Run]] = []
    for run in runs:
        if blocks and run.init - blocks[-1][-1].init <= spacing:
            blocks[-1].append(run)
        else:
            blocks.append([run])

    windows = []
    for block in blocks:
        windows.append(
            {
                "first_init": block[0].init,
                "last_init": block[-1].init,
                "n_days": len({run.init.date() for run in block}),
                "n_runs": len(block),
            }
        )

    return {
        "n_runs": len(runs),
        "n_forecast_hour_files": sum(len(run.forecast_hours) for run in runs),
        "n_days": len({run.init.date() for run in runs}),
        "windows": windows,
    }


def _format_table(summary: dict) -> str:
    """Render the per-window rows of a summary as a fixed-width text table."""
    lines = [f"{'first_init':<17}{'last_init':<17}{'days':>6}{'runs':>7}"]
    for window in summary["windows"]:
        lines.append(f"{window['first_init']:%Y-%m-%d %H}Z".ljust(17) + f"{window['last_init']:%Y-%m-%d %H}Z".ljust(17) + f"{window['n_days']:>6}{window['n_runs']:>7}")
    lines.append("")
    lines.append(f"runs: {summary['n_runs']}  forecast-hour files: {summary['n_forecast_hour_files']}  days: {summary['n_days']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: print the run summary and optionally write the run list.

    Args:
        argv: Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description="Build the HRRR fast-inference run list from an events YAML file.")
    parser.add_argument("config", type=Path, help="Path to the events YAML file.")
    parser.add_argument("--out", type=Path, help="Write one 'YYYYMMDDHH maxfh' line per run to this file.")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    runs = build_run_list(config)
    summary = summarize(runs, _init_interval_hours(config))
    print(_format_table(summary))

    if args.out is not None:
        args.out.write_text("".join(f"{run.init:%Y%m%d%H} {max(run.forecast_hours)}\n" for run in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
