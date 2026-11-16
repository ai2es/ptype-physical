"""Configuration of the fast HRRR precipitation-type pipeline (``ptype.hrrr_fast.pipeline``).

One flat dataclass holds every setting so that the YAML file (``config/hrrr_fast.yml``), the command
line (``applications/hrrr_ptype_run.py``) and the PBS launcher (``applications/hrrr_ptype_submit.py``)
all talk about the same names. Flat (rather than nested sections) because every option has to be
overridable from the command line and recorded verbatim in each run's manifest; a flat dict makes both
trivial and keeps the manifest diffable between runs.

Environment variables (``$USER``, ``$TMPDIR`` ...) and ``~`` are expanded in path-valued settings, so the
same YAML works for every user on Derecho.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Settings whose values are filesystem paths (expanded with ``os.path.expandvars``/``expanduser``).
PATH_FIELDS = ("work_dir", "weights", "upload_root", "source")


@dataclass
class PipelineConfig:
    """All settings of a pipeline invocation.

    Attributes:
        source: ``"s3"`` for the public NOAA bucket ``noaa-hrrr-bdp-pds`` (anonymous), a local directory
            laid out like the bucket (``hrrr.YYYYMMDD/conus/...grib2`` + ``.idx``; sparse mirrors made by
            ``benchmarks/pipeline_benchmark.py mirror`` work), or ``"synthetic"`` (tests only).
        work_dir: Where run directories are built (node-local or scratch disk). One sub-directory per init.
        weights: Exported ``ptype_weights.npz`` (``applications/export_ptype_weights.py``); the ONNX
            engine looks for ``ptype_model.onnx`` next to it.
        engine: ``torch``, ``onnx``, ``numpy`` or ``numba`` (see ``ptype.hrrr_fast.engine``).
        device: ``cpu`` or ``cuda``. With ``cuda`` every worker process is pinned to one GPU by
            ``CUDA_VISIBLE_DEVICES`` round-robin (see ``docs/hrrr_fast.md``).
        engine_options: Extra keyword arguments for ``make_engine`` (``dtype``, ``tf32``, ``compile`` ...).
        batch_size: Rows per forward pass; ``None`` uses the engine default.
        processes: Worker processes; ``0`` runs everything in the calling process (tests/debugging),
            ``"auto"`` uses ``min(cores // threads_per_process, usable_memory // memory_per_process_gb)``.
        threads_per_process: Threads each worker may use for numba kernels, BLAS/torch and ecCodes decode.
        decode_threads: ecCodes decode threads per worker; ``None`` means ``threads_per_process``.
        memory_per_process_gb: Peak resident memory of one worker for one forecast hour, used to size
            ``processes="auto"`` and the PBS launcher. Measured, see ``benchmarks/results/pipeline_local.md``.
        memory_fraction: Fraction of the node's available memory the workers may use together.
        max_runs_in_flight: Runs whose hours may be queued at the same time; bounds local disk use while
            keeping workers busy across run boundaries.
        gpus: Number of GPUs to spread workers over when ``device == "cuda"``; ``None`` = detect.
        n_levels: Hybrid levels read first for the network (wind) variables. 27 reaches 5000 m AGL with a
            two-level margin over the Rockies in the measured files (level 25 is the first whose minimum
            AGL height exceeds 5000 m, ``benchmarks/results/reader_local.md``).
        bourgouin_top_m: Top of the height grid used for the revised Bourgouin method (T and Td only).
        bourgouin_dz_m: Spacing of that grid. 250 m like the network grid, so the 0-5000 m part is
            bit-identical to the network sounding and the paper's layer-depth tests keep their meaning.
        bourgouin_rows_per_chunk: Grid rows per Bourgouin chunk; bounds the temporary ``(columns, 5, levels)``
            array (128 rows x 1799 columns x 5 x 49 levels x 4 B = 226 MB for a 12 km top).
        max_levels: Never read more hybrid levels than this (HRRR wrfnat has 50).
        level_step: Levels added per automatic re-read when the top level read is too low somewhere.
        surface_temperature, surface_wind, surface_pressure, pressure_interp, clip_surface_dewpoint: The
            0 m policies of ``columns.build_soundings``; defaults follow the training data (2 m T/Td,
            10 m wind, surface pressure, pressure linear in height, no dewpoint clip).
        coalesce: obstore range-coalescing threshold in bytes (1 MiB transfers no extra bytes on HRRR
            wrfnat; ``benchmarks/results/reader_local.md``).
        fetch_retries: Attempts after the first for a transient fetch error (missing files are not retried).
        fetch_backoff_s: First retry delay; doubles each attempt with full jitter, capped at ``fetch_max_backoff_s``.
        fetch_max_backoff_s: Cap of the retry delay.
        pyramid_levels: Levels of the gridded multiscale store (3, 6, 12, 24 km).
        location_chunk: Locations per inner chunk of the sounding store.
        sounding_encoding: ``"default"`` uses ``writers.default_sounding_encoding()`` (the agreed default, 0.1-unit
            int16); ``"int16-0.1"``, ``"int32-0.01"`` or ``"lossless"`` pin one through ``writers.sounding_encoding``
            (sizes/errors in ``benchmarks/results/compression_local.md``).
        finalize_ptype: Call ``writers.finalize_ptype_store`` once per run, in the parent, after every hour is
            written and before upload (consolidates the per-hour gridded objects into shards so that objects exceed
            Standard-IA's 128 KB minimum billable size).
        upload: ``"s3"`` (``aws s3 sync`` with ``$AWS_PROFILE``), ``"local"`` (copy into ``upload_root``) or
            ``"none"`` (stores stay in ``work_dir``; the index is not touched).
        upload_root: Destination directory of the ``local`` backend; the index is ``<upload_root>/ptype/inits.json``.
        bucket: Destination bucket of the ``s3`` backend.
        region: Bucket region.
        aws_cli: The ``aws`` executable.
        storage_class: S3 storage class of the uploaded Zarr objects (``aws s3 sync --storage-class``).
            ``STANDARD_IA`` because archived cases are read rarely; ``None`` uses the bucket default. The
            index ``ptype/inits.json`` never goes through the sync (it is written by ``index.update_index``
            without a storage class, i.e. STANDARD), because it is rewritten after every run and IA bills a
            30-day minimum per object version.
        keep_local: Keep the run directory after a successful upload.
        log_level: Python logging level name.
        log_json: Emit one JSON object per log line (structured logging) instead of plain text.
    """

    source: str = "s3"
    work_dir: str = "./hrrr_ptype_runs"
    weights: str = "ptype_weights.npz"
    engine: str = "torch"
    device: str = "cpu"
    engine_options: dict[str, Any] = field(default_factory=dict)
    batch_size: int | None = None
    processes: int | str = "auto"
    threads_per_process: int = 8
    decode_threads: int | None = None
    memory_per_process_gb: float = 12.0
    memory_fraction: float = 0.8
    max_runs_in_flight: int = 2
    gpus: int | None = None
    n_levels: int = 27
    bourgouin_top_m: float = 8000.0
    bourgouin_dz_m: float = 250.0
    bourgouin_rows_per_chunk: int = 128
    max_levels: int = 50
    level_step: int = 3
    surface_temperature: str = "2m"
    surface_wind: str = "10m"
    surface_pressure: str = "surface"
    pressure_interp: str = "linear"
    clip_surface_dewpoint: bool = False
    coalesce: int = 1024 * 1024
    fetch_retries: int = 5
    fetch_backoff_s: float = 2.0
    fetch_max_backoff_s: float = 60.0
    pyramid_levels: int = 4
    location_chunk: int = 1024
    sounding_encoding: str = "default"
    finalize_ptype: bool = True
    upload: str = "s3"
    upload_root: str | None = None
    bucket: str = "wwe-ptype"
    region: str = "us-east-1"
    aws_cli: str = "aws"
    storage_class: str | None = "STANDARD_IA"
    keep_local: bool = False
    log_level: str = "INFO"
    log_json: bool = True

    def __post_init__(self) -> None:
        """Expand paths and validate values early, before any worker is started or byte is fetched."""
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if isinstance(value, str) and value not in ("s3", "synthetic"):
                setattr(self, name, os.path.expanduser(os.path.expandvars(value)))
        if self.upload not in ("s3", "local", "none"):
            raise ValueError(f"upload must be s3, local or none, not {self.upload!r}")
        if self.upload == "local" and not self.upload_root:
            raise ValueError("upload: local needs upload_root")
        if self.bourgouin_top_m < 5000.0:
            raise ValueError("bourgouin_top_m must be >= 5000 (the network grid is 0-5000 m)")
        top_steps = self.bourgouin_top_m / self.bourgouin_dz_m
        if abs(top_steps - round(top_steps)) > 1e-9 or abs(5000.0 / self.bourgouin_dz_m - round(5000.0 / self.bourgouin_dz_m)) > 1e-9:
            raise ValueError("bourgouin_top_m and 5000 must be multiples of bourgouin_dz_m")
        if not (isinstance(self.processes, int) or self.processes == "auto"):
            raise ValueError(f"processes must be an integer or 'auto', not {self.processes!r}")
        if self.threads_per_process < 1:
            raise ValueError("threads_per_process must be >= 1")
        if self.sounding_encoding not in ("default", "int16-0.1", "int32-0.01", "lossless"):
            raise ValueError(f"sounding_encoding must be default, int16-0.1, int32-0.01 or lossless, not {self.sounding_encoding!r}")
        if not 1 <= self.n_levels <= self.max_levels:
            raise ValueError("need 1 <= n_levels <= max_levels")

    @classmethod
    def from_yaml(cls, path: str | os.PathLike | None, **overrides: Any) -> PipelineConfig:
        """Load a YAML file (optional) and apply overrides; ``None`` overrides are ignored.

        Ignoring ``None`` lets the CLI pass every option unconditionally (argparse defaults are ``None``)
        so that only options the user actually typed replace the file's values.

        Raises:
            KeyError: For a key that is not a setting (a typo would otherwise be silently ignored).
        """
        data: dict[str, Any] = {}
        if path is not None:
            data = yaml.safe_load(Path(path).read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise KeyError(f"unknown config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the manifest (JSON-serialisable)."""
        return dataclasses.asdict(self)

    def replace(self, **changes: Any) -> PipelineConfig:
        """Copy with some settings changed (validated again)."""
        return dataclasses.replace(self, **changes)

    @property
    def effective_decode_threads(self) -> int:
        """ecCodes decode threads per worker (defaults to the worker's thread budget)."""
        return int(self.decode_threads or self.threads_per_process)
