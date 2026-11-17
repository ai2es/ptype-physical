"""Network correctness on identical inputs: original Keras path vs the ``ptype.hrrr_fast`` torch engine.

The input is the exact matrix the ORIGINAL script handed to its model for one forecast hour (dumped by
``benchmarks/baseline_run.py --save-inputs``; it is dumped *before* ``transform_data``, i.e. UNSCALED, in
physical units). Both paths then start from the same rows:

* original: ``bridgescaler`` ``transformer.transform(DataFrame)`` followed by
  ``CategoricalDNN.predict(x, return_uncertainties=True, batch_size=predict_batch_size)``, exactly the two
  calls ``scripts/run_inference.py`` makes (the model is loaded with the ``custom_objects`` fix of the
  wrapper, because the plain call raises, see ``benchmarks/results/derecho/baseline.md``);
* new: ``ptype.hrrr_fast.engine.make_engine("torch", weights)`` fed the unscaled float32 rows (the engine
  folds the scaler into layer 0 and contains the evidential head).

The original predictions were not saved by the baseline runs (only quantised copies inside the netCDF), so
they are recomputed here and written next to the report for reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

CLASSES = ["rain", "snow", "icep", "frzr"]


def original_predict(inputs: np.ndarray, columns: list[str], keras_path: str, scaler_path: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the original scaler + Keras model the way ``scripts/run_inference.py`` does.

    Args:
        inputs: Unscaled ``(n, 84)`` matrix exactly as the original script built it (float64).
        columns: The 84 scaler column names in matrix order (``input_columns.json``).
        keras_path: Path of the ``.keras`` model.
        scaler_path: Path of the bridgescaler JSON.
        batch_size: ``predict_batch_size`` of the original config.

    Returns:
        ``(probs, u)`` as float32 arrays of shape ``(n, 4)`` and ``(n,)``.
    """
    import keras
    import mlguess.keras.callbacks  # noqa: F401  (registers custom classes so the model can be deserialised)
    import mlguess.keras.models  # noqa: F401
    import pandas as pd
    from bridgescaler import load_scaler
    from mlguess.keras.losses import evidential_cat_loss

    model = keras.models.load_model(keras_path, custom_objects={"loss": evidential_cat_loss})
    scaler = load_scaler(scaler_path)
    # Same call as ptype.inference.transform_data, repeated here instead of imported because importing
    # ptype.inference pulls in herbie/cfgrib/pygrib, which this check does not need.
    x = scaler.transform(pd.DataFrame(inputs, columns=columns))
    preds = model.predict(x, return_uncertainties=True, batch_size=batch_size, verbose=0)
    probs = preds[0].detach().cpu().numpy() if hasattr(preds[0], "detach") else np.asarray(preds[0])
    u = preds[1].detach().cpu().numpy() if hasattr(preds[1], "detach") else np.asarray(preds[1])
    return probs.astype(np.float32), u.reshape(-1).astype(np.float32)


def summarize(ref_p: np.ndarray, new_p: np.ndarray, ref_u: np.ndarray, new_u: np.ndarray) -> dict:
    """Absolute differences per output and dominant-class agreement.

    Args:
        ref_p: Original probabilities ``(n, 4)``.
        new_p: New-engine probabilities ``(n, 4)``.
        ref_u: Original uncertainty ``(n,)``.
        new_u: New-engine uncertainty ``(n,)``.

    Returns:
        Nested dict of max and mean absolute differences and the agreement fractions.
    """
    # NaN inputs (if any) give NaN outputs in both paths; they are counted and excluded rather than poisoning the maxima.
    ok = np.isfinite(ref_p).all(axis=1) & np.isfinite(new_p).all(axis=1) & np.isfinite(ref_u) & np.isfinite(new_u)
    out: dict = {"n_rows": int(ref_p.shape[0]), "n_rows_compared": int(ok.sum())}
    ref_p, new_p, ref_u, new_u = ref_p[ok], new_p[ok], ref_u[ok], new_u[ok]
    # Differences are formed in float64 so the comparison itself adds no rounding.
    d = np.abs(ref_p.astype(np.float64) - new_p.astype(np.float64))
    for i, c in enumerate(CLASSES):
        out[c] = {"max_abs": float(d[:, i].max()), "mean_abs": float(d[:, i].mean())}
    out["all_four_probabilities"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
    du = np.abs(ref_u.astype(np.float64) - new_u.astype(np.float64))
    out["uncertainty"] = {"max_abs": float(du.max()), "mean_abs": float(du.mean())}
    a, b = ref_p.argmax(axis=1), new_p.argmax(axis=1)
    out["dominant_class_agreement"] = float((a == b).mean())
    out["dominant_class_disagree_rows"] = int((a != b).sum())
    if (a != b).any():
        # Disagreements can only happen where the top two probabilities are nearly tied; report how near.
        top2 = np.sort(ref_p[a != b], axis=1)[:, -2:]
        out["disagree_top2_gap_max"] = float((top2[:, 1] - top2[:, 0]).max())
    out["ref_class_counts"] = {c: int((a == i).sum()) for i, c in enumerate(CLASSES)}
    out["prob_sum_dev_ref_max"] = float(np.abs(ref_p.astype(np.float64).sum(axis=1) - 1.0).max())
    out["prob_sum_dev_new_max"] = float(np.abs(new_p.astype(np.float64).sum(axis=1) - 1.0).max())
    return out


def main() -> None:
    """Parse arguments, run both paths on the dumped matrix and write the JSON report."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-dir", required=True, help="directory with input_matrix.npy and input_columns.json")
    p.add_argument("--keras-path", required=True)
    p.add_argument("--scaler-path", required=True)
    p.add_argument("--weights", required=True, help="exported ptype_weights.npz")
    p.add_argument("--batch-size", type=int, default=2048, help="predict_batch_size of the original config")
    p.add_argument("--threads", type=int, default=8, help="torch threads for both paths (set deliberately: the Keras torch backend otherwise starts 128)")
    p.add_argument("--out", required=True, help="output directory (JSON report and the original predictions)")
    a = p.parse_args()

    import torch

    torch.set_num_threads(a.threads)
    from ptype.hrrr_fast.engine import make_engine

    d = Path(a.inputs_dir)
    inputs = np.load(d / "input_matrix.npy")
    columns = json.loads((d / "input_columns.json").read_text())
    print(f"inputs {inputs.shape} {inputs.dtype}; NaN rows: {int(np.isnan(inputs).any(axis=1).sum())}", flush=True)

    t0 = time.time()
    ref_p, ref_u = original_predict(inputs, columns, a.keras_path, a.scaler_path, a.batch_size)
    print(f"original path done in {time.time() - t0:.1f} s", flush=True)

    engine = make_engine("torch", a.weights, device="cpu", threads=a.threads)
    t0 = time.time()
    new_p, new_u = engine.predict(inputs.astype(np.float32))
    print(f"new engine done in {time.time() - t0:.1f} s", flush=True)

    rep = summarize(ref_p, new_p, ref_u, new_u)
    rep["inputs_dtype_in_file"] = str(inputs.dtype)
    rep["nan_input_rows"] = int(np.isnan(inputs).any(axis=1).sum())
    rep["engine"] = engine.describe()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "original_probs.npy", ref_p)
    np.save(out / "original_u.npy", ref_u)
    (out / "network_correctness.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    os.sync()


if __name__ == "__main__":
    main()
