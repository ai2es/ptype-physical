"""Instrumented launcher for the ORIGINAL HRRR p-type inference (scripts/run_inference.py), unmodified.

What this wrapper adds (and nothing else):
  1. ``import mlguess.keras.models`` / ``callbacks`` so keras can deserialize ``CategoricalDNN``, and
     replaces ``ptype.inference.load_model`` with a call that passes
     ``custom_objects={"loss": mlguess.keras.losses.evidential_cat_loss}`` (the saved model's compile
     config references a nested function named ``loss``; plain ``load_model(path)`` raises TypeError).
     Same loading call as applications/export_ptype_weights.py.
  2. Per-stage wall-clock timers around the original ptype.inference functions (and ``model.predict``),
     appended as JSON lines to ``<out>/stages.jsonl`` (one record per call, with pid and forecast hour).
     The timers wrap the functions; arguments and results pass through untouched.
  3. Optional ``--save-inputs DIR``: dumps the exact array handed to ``transform_data`` (before scaling),
     its column order, lat/lon and grid shape as .npy / .json.
  4. Writes a per-run override of the config (out_path, forecast range, n_processors) to the output dir and
     runs ``scripts/run_inference.py`` as ``__main__`` via runpy, so the script's own Pool/multiprocessing
     logic is exercised as is.
"""

import argparse
import json
import os
import resource
import runpy
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import keras
import mlguess.keras.callbacks
import mlguess.keras.models  # noqa: F401
import numpy as np
import ptype.inference as inf
from mlguess.keras.losses import evidential_cat_loss

ARGS = None
STATE = {"fxx": None}


def log(stage, t0, t1, **extra):
    rec = {
        "pid": os.getpid(),
        "stage": stage,
        "t_start": t0,
        "t_end": t1,
        "seconds": t1 - t0,
        "fxx": STATE["fxx"],
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    rec.update(extra)
    with open(os.path.join(ARGS.out_dir, "stages.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def timed(name, fn, before=None, after=None):
    def inner(*a, **k):
        if before:
            before(a, k)
        t0 = time.time()
        res = fn(*a, **k)
        t1 = time.time()
        if after:
            after(res, a, k)
        log(name, t0, t1)
        return res

    return inner


class TimedModel:
    """Proxy that times model.predict and forwards everything else."""

    def __init__(self, model):
        self._m = model

    def predict(self, *a, **k):
        t0 = time.time()
        res = self._m.predict(*a, **k)
        log("predict", t0, time.time())
        return res

    def __getattr__(self, name):
        return getattr(self._m, name)


def patch():
    def load_model_fixed(path):
        return keras.models.load_model(path, custom_objects={"loss": evidential_cat_loss})

    inf.load_model = load_model_fixed

    orig_load_saved = inf.load_saved_model

    def load_saved(*a, **k):
        t0 = time.time()
        model, scaler, feats = orig_load_saved(*a, **k)
        log("load_model_scaler", t0, time.time())
        return TimedModel(model), scaler, feats

    inf.load_saved_model = load_saved

    def set_fxx(a, k):
        STATE["fxx"] = k.get("forecast_hour")

    inf.download_data = timed("download", inf.download_data, before=set_fxx)

    def keep_ds(res, a, k):
        STATE["ds"] = res[0]

    inf.load_data = timed("load_data(cfgrib+flatten+subset)", inf.load_data, after=keep_ds)
    inf.convert_and_interpolate = timed("convert_and_interpolate", inf.convert_and_interpolate)

    orig_transform = inf.transform_data

    def transform(input_data, transformer, input_features):
        if ARGS.save_inputs:
            d = ARGS.save_inputs
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "input_matrix.npy"), np.asarray(input_data))
            with open(os.path.join(d, "input_columns.json"), "w") as f:
                json.dump(list(input_features), f)
            ds = STATE["ds"]
            np.save(os.path.join(d, "lat.npy"), ds["latitude"].values)
            np.save(os.path.join(d, "lon.npy"), ds["longitude"].values)
            grid_shape = {
                "y": int(ds["y"].size),
                "x": int(ds["x"].size),
                "n_rows": len(input_data),
                "input_dtype": str(np.asarray(input_data).dtype),
                "input_shape": list(np.asarray(input_data).shape),
            }
            with open(os.path.join(d, "grid_shape.json"), "w") as f:
                json.dump(grid_shape, f)
        t0 = time.time()
        res = orig_transform(input_data, transformer, input_features)
        log("transform_data(scaler)", t0, time.time())
        return res

    inf.transform_data = transform

    inf.grid_predictions = timed("grid_predictions", inf.grid_predictions)
    inf.save_data = timed("save_data", inf.save_data)


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--out-dir", required=True, help="run directory: derived config, stages.jsonl, netCDF output")
    p.add_argument("--fxx-start", type=int, required=True)
    p.add_argument("--fxx-end", type=int, required=True)
    p.add_argument("--n-procs", type=int, default=1)
    p.add_argument("--save-inputs", default=None)
    ARGS = p.parse_args()
    os.makedirs(ARGS.out_dir, exist_ok=True)
    with open(ARGS.config) as f:
        cfg = yaml.safe_load(f)
    cfg["out_path"] = os.path.join(ARGS.out_dir, "out") + "/"
    cfg["forecast_range"]["start"] = ARGS.fxx_start
    cfg["forecast_range"]["end"] = ARGS.fxx_end
    cfg["n_processors"] = ARGS.n_procs
    derived = os.path.join(ARGS.out_dir, "config_derived.yml")
    with open(derived, "w") as f:
        yaml.safe_dump(cfg, f)
    patch()
    t0 = time.time()
    sys.argv = ["run_inference.py", "-c", derived]
    err = None
    try:
        runpy.run_path(os.path.join(REPO, "scripts", "run_inference.py"), run_name="__main__")
    except BaseException as e:  # record, then re-raise
        err = repr(e)
        raise
    finally:
        t1 = time.time()
        summ = {
            "wall_seconds": t1 - t0,
            "n_procs": ARGS.n_procs,
            "fxx": [ARGS.fxx_start, ARGS.fxx_end],
            "error": err,
            "children_maxrss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0,
            "self_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        with open(os.path.join(ARGS.out_dir, "summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
        print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
