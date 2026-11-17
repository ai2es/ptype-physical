"""Maintain ``ptype/inits.json``, the list of runs the ptype website reads.

Many batch jobs finish at about the same time and each wants to add its own run
to one shared JSON object.  S3 has no append, so a plain read-modify-write would
let the last writer silently drop the others' entries.  ``update_index`` avoids
that with optimistic concurrency: it remembers the ETag of the object it read and
writes back only if the ETag is unchanged (``If-Match``), or only if the object
still does not exist (``If-None-Match: *``).  On a precondition failure it
re-reads, re-merges and tries again.

Index format (version 1, backward compatible with what the website already reads)::

    {
      "version": 1,
      "updated": "2024-04-30T01:12:09Z",
      "inits": [
        {"init": "2024-04-30T00:00Z", "forecast_hours": [1, 2, 3],
         "store": "2024/04/30/hrrr_ptype_20240430_0000.zarr",
         "soundings": "soundings/2024/04/30/hrrr_soundings_20240430_0000.zarr"}
      ]
    }

``store`` is relative to the ``ptype/`` prefix (the website joins it to its ptype
root, so this must not change).  ``soundings`` is relative to the *bucket root*
(or an absolute ``https://`` URL), because the soundings live beside ``ptype/``,
not under it; a path relative to ``ptype/`` would need ``../`` and would break
URL joining.  The key is optional so old readers and runs without soundings
keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import obstore
from obstore.exceptions import AlreadyExistsError, NotFoundError, PreconditionError

if TYPE_CHECKING:
    from obstore.store import ObjectStore

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
PTYPE_PREFIX = "ptype"
SOUNDING_PREFIX = "soundings"
INDEX_KEY = f"{PTYPE_PREFIX}/inits.json"
_INIT_FMT = "%Y-%m-%dT%H:%MZ"


def _as_utc(init: datetime) -> datetime:
    """Return ``init`` as an aware UTC datetime; naive values are taken to be UTC (HRRR times always are)."""
    return init.replace(tzinfo=UTC) if init.tzinfo is None else init.astimezone(UTC)


def init_string(init: datetime) -> str:
    """Format an init time the way the website parses it (``2024-04-30T00:00Z``).

    A fixed-width string is used as the entry identity so that sorting entries
    by it is also chronological sorting.
    """
    return _as_utc(init).strftime(_INIT_FMT)


def store_paths(init: datetime) -> tuple[str, str]:
    """Return the bucket keys ``(ptype_key, sounding_key)`` for a run.

    One function owns the layout so the uploader, the index and the docs cannot
    drift apart.  Keys are relative to the bucket root and carry no trailing slash.
    """
    t = _as_utc(init)
    day = t.strftime("%Y/%m/%d")
    stamp = t.strftime("%Y%m%d_%H00")
    return (
        f"{PTYPE_PREFIX}/{day}/hrrr_ptype_{stamp}.zarr",
        f"{SOUNDING_PREFIX}/{day}/hrrr_soundings_{stamp}.zarr",
    )


def make_entry(init: datetime, forecast_hours: list[int], with_soundings: bool = True) -> dict[str, Any]:
    """Build the argument for ``update_index`` using the standard layout.

    The ptype path is stripped of its ``ptype/`` prefix because that is what the
    website expects; the sounding path keeps its bucket-root-relative form.
    """
    ptype_key, sounding_key = store_paths(init)
    return {
        "init": init,
        "forecast_hours": forecast_hours,
        "store": ptype_key.removeprefix(f"{PTYPE_PREFIX}/"),
        "soundings": sounding_key if with_soundings else None,
    }


def empty_index() -> dict[str, Any]:
    """Return a new, empty version-1 index."""
    return {"version": INDEX_VERSION, "inits": []}


def merge_entry(
    index: dict[str, Any],
    init: datetime,
    forecast_hours: list[int],
    store: str,
    soundings: str | None,
    updated: datetime | None = None,
) -> dict[str, Any]:
    """Return a new index with this run added, or replacing the run with the same init.

    The function is pure (``index`` is not modified) and idempotent: merging the
    same run twice gives the same result apart from ``updated``.  An existing
    entry for the same init is replaced *as a whole*; forecast hours and
    sounding paths are not unioned.  Unioning would hide a re-run that produced
    fewer hours (for example after a bad hour was dropped), and the batch job
    that writes the entry always knows the complete state of its stores.  Entries
    for other inits, and unknown top-level keys, are preserved.  ``soundings``
    is omitted from the entry when ``None``.
    """
    key = init_string(init)
    entry: dict[str, Any] = {"init": key, "forecast_hours": sorted({int(h) for h in forecast_hours}), "store": store}
    if soundings is not None:
        entry["soundings"] = soundings
    kept = [e for e in index.get("inits", []) if e.get("init") != key]
    out = {k: v for k, v in index.items() if k not in ("inits", "version", "updated")}
    out["version"] = INDEX_VERSION
    out["updated"] = _as_utc(updated or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["inits"] = sorted([*kept, entry], key=lambda e: e["init"])
    return out


def _read(store: ObjectStore, key: str) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Read the index and the precondition for overwriting exactly that version.

    Returns ``(empty_index, None)`` if the object does not exist.  Only keys with
    a value go into the precondition because obstore rejects ``None`` values.
    """
    try:
        result = obstore.get(store, key)
    except (FileNotFoundError, NotFoundError):  # obstore raises the builtin FileNotFoundError for a missing key
        return empty_index(), None
    body = bytes(result.bytes())
    version = {k: v for k, v in {"e_tag": result.meta["e_tag"], "version": result.meta["version"]}.items() if v is not None}
    if not version:
        msg = f"store returned no ETag/version for {key}; cannot do a safe conditional update"
        raise RuntimeError(msg)
    return json.loads(body), version


def update_index(
    store: ObjectStore,
    key: str,
    entry: dict[str, Any],
    max_retries: int = 12,
    base_delay: float = 0.05,
    max_delay: float = 2.0,
) -> dict[str, Any]:
    """Add ``entry`` to the index at ``key`` without losing concurrent updates.

    ``entry`` has the keys ``init``, ``forecast_hours``, ``store`` and
    (optionally) ``soundings``; see ``make_entry``.  Each attempt reads the
    object and its ETag, merges, and writes with ``mode="create"`` (object must
    not exist) or ``mode={"e_tag": ...}`` (object must be unchanged).  A losing
    writer gets ``PreconditionError``/``AlreadyExistsError``, sleeps a jittered,
    exponentially growing time (jitter keeps jobs that failed together from
    colliding again together) and starts over.

    Returns the index that was written.  Raises ``RuntimeError`` after
    ``max_retries`` lost races so a stuck job fails loudly instead of hanging.
    """
    for attempt in range(max_retries + 1):
        current, version = _read(store, key)
        merged = merge_entry(current, entry["init"], entry["forecast_hours"], entry["store"], entry.get("soundings"))
        payload = json.dumps(merged, indent=2).encode()
        try:
            obstore.put(
                store,
                key,
                payload,
                mode="create" if version is None else version,
                attributes={"Content-Type": "application/json", "Cache-Control": "no-cache"},
            )
        except (PreconditionError, AlreadyExistsError):
            delay = random.uniform(0, min(max_delay, base_delay * 2**attempt))
            logger.info("index update lost a race (attempt %d); retrying in %.2fs", attempt + 1, delay)
            time.sleep(delay)
            continue
        return merged
    msg = f"could not update {key} after {max_retries + 1} attempts"
    raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> None:
    """Command line: add one run to the index in an S3 bucket.

    Example: ``python -m ptype.hrrr_fast.index add wwe-ptype 2024-04-30T00:00 1-18``.
    Credentials come from ``--profile``/``$AWS_PROFILE`` via boto3, else from obstore's own environment handling (``AWS_ACCESS_KEY_ID`` etc.).
    """
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("command", choices=["add"])
    p.add_argument("bucket")
    p.add_argument("init", help="UTC init time, e.g. 2024-04-30T00:00")
    p.add_argument("hours", help="forecast hours, e.g. 1-18 or 1,2,3")
    p.add_argument("--no-soundings", action="store_true")
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile (default: $AWS_PROFILE); read through boto3")
    a = p.parse_args(argv)
    from obstore.store import S3Store

    hours: list[int] = []
    for part in a.hours.split(","):
        lo, _, hi = part.partition("-")
        hours.extend(range(int(lo), int(hi or lo) + 1))
    kwargs: dict[str, Any] = {"region": a.region} if a.region else {}
    if a.profile:
        # obstore's S3Store does not read ~/.aws/credentials or profiles itself, so go through boto3,
        # which understands profiles, the credentials file and SSO caches.
        import boto3
        from obstore.auth.boto3 import Boto3CredentialProvider

        kwargs["credential_provider"] = Boto3CredentialProvider(boto3.Session(profile_name=a.profile))
    s3 = S3Store(a.bucket, **kwargs)
    init = datetime.fromisoformat(a.init)
    result = update_index(s3, INDEX_KEY, make_entry(init, hours, not a.no_soundings))
    print(f"index now lists {len(result['inits'])} runs")


if __name__ == "__main__":
    main()
