"""Export the Keras precipitation-type model to framework-neutral files.

Loads the Keras ``CategoricalDNN`` and the bridgescaler ``GroupRobustScaler`` **once** (this is the
only step that needs Keras/TensorFlow-style dependencies) and writes

* ``ptype_weights.npz``: dense kernels/biases, leaky-ReLU slope, scaler centers/scales expanded to
  one value per input feature, class names, heights, feature names. Readable with numpy alone.
* ``ptype_model.onnx``: a Gemm/LeakyRelu graph that takes **raw physical-unit** inputs
  ``input[batch, 84]`` and returns ``probs[batch, 4]`` and ``u[batch]``; the scaler is folded into
  the first Gemm and the evidential head (relu, +1, normalise, K/S) is part of the graph.

Production inference environments can then drop Keras entirely and use ``ptype.hrrr_fast.engine``.

Usage::

    KERAS_BACKEND=torch python applications/export_ptype_weights.py \\
        --model ptype_model_20240909.keras --scaler ptype_scaler_20240909.json --out-dir weights/
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from ptype.hrrr_fast.engine import N_FEATURES, PtypeWeights

CLASSES = ["rain", "snow", "icep", "frzr"]
HEIGHTS_M = np.arange(0, 5250, 250)


def load_keras(path: str):
    """Load the Keras model. mlguess must be imported first so its custom classes are registered."""
    import keras
    import mlguess.keras.callbacks
    import mlguess.keras.models  # noqa: F401
    from mlguess.keras.losses import evidential_cat_loss

    return keras.models.load_model(path, custom_objects={"loss": evidential_cat_loss})


def leaky_slope(layer) -> float:
    """Read the negative slope Keras applies for ``activation='leaky_relu'``.

    The saved model stores only the activation *name*, so the slope is the default argument of
    ``keras.activations.leaky_relu`` (0.2 in Keras 3, unlike the 0.3 of ``keras.layers.LeakyReLU``).
    Reading it from the installed function, and then checking it numerically in ``main``, avoids
    hard-coding a value that a Keras upgrade could change.
    """
    fn = layer.activation
    return float(inspect.signature(fn).parameters["negative_slope"].default)


def extract(model, scaler) -> dict:
    """Pull weights, slope and expanded scaler constants out of the loaded objects."""
    dense = [layer for layer in model.layers if type(layer).__name__ == "Dense"]
    slopes = set()
    out: dict = {}
    for i, layer in enumerate(dense):
        kernel, bias = [np.asarray(w, dtype=np.float32) for w in layer.get_weights()]
        out[f"W{i}"], out[f"b{i}"] = kernel, bias
        act = layer.activation.__name__
        if i == len(dense) - 1:
            if act != "linear":
                raise ValueError(f"final layer must be linear, got {act}")
        elif act == "leaky_relu":
            slopes.add(leaky_slope(layer))
        else:
            raise ValueError(f"unsupported activation {act!r} in {layer.name}")
    if len(slopes) != 1:
        raise ValueError(f"hidden layers must share one leaky slope, got {slopes}")
    out["n_layers"] = np.int64(len(dense))
    out["negative_slope"] = np.float64(slopes.pop())

    # Expand per-group (center, scale) to one value per feature, following the scaler's own group order.
    groups = scaler.groups_
    group_index = np.concatenate([np.full(len(g), gi) for gi, g in enumerate(groups)])
    # This model's scaler has group_index_ == [0, 1, 2, 3] (parameter i belongs to groups_[i]); fail loudly
    # rather than silently mis-assigning constants if a future scaler orders them differently.
    if not np.array_equal(np.asarray(scaler.group_index_).ravel(), np.arange(len(groups))):
        raise ValueError(f"unexpected scaler.group_index_ {scaler.group_index_}")
    center_g = np.asarray(scaler.center_, dtype=np.float64).ravel()
    scale_g = np.asarray(scaler.scale_, dtype=np.float64).ravel()
    features = [f for g in groups for f in g]
    if len(features) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, scaler has {len(features)}")
    out["center_group"], out["scale_group"] = center_g, scale_g
    out["center"], out["scale"] = center_g[group_index], scale_g[group_index]
    out["feature_names"] = np.asarray(features)
    out["classes"] = np.asarray(CLASSES)
    out["heights_m"] = HEIGHTS_M
    return out


def build_onnx(weights: PtypeWeights):
    """Build the ONNX graph: raw inputs -> (probs, u).

    Gemm + LeakyRelu are fused by onnxruntime into efficient kernels. The scaler is folded into
    layer 0 (see ``PtypeWeights.folded``), and the evidential head is expressed with Relu/Add/
    ReduceSum/Div so no post-processing is needed outside the session.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    kernels, biases = weights.folded()
    nodes, inits = [], []
    prev = "input"
    for i, (k, b) in enumerate(zip(kernels, biases, strict=True)):
        inits += [numpy_helper.from_array(k, f"W{i}"), numpy_helper.from_array(b, f"B{i}")]
        last = i == len(kernels) - 1
        z = "logits" if last else f"z{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"], [z]))
        if not last:
            nodes.append(helper.make_node("LeakyRelu", [z], [f"a{i}"], alpha=weights.negative_slope))
            prev = f"a{i}"
    inits += [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
        numpy_helper.from_array(np.array(float(weights.n_classes), dtype=np.float32), "k_classes"),
    ]
    nodes += [
        helper.make_node("Relu", ["logits"], ["evidence"]),
        helper.make_node("Add", ["evidence", "one"], ["alpha"]),
        helper.make_node("ReduceSum", ["alpha", "axes1"], ["S"], keepdims=1),
        helper.make_node("Div", ["alpha", "S"], ["probs"]),
        helper.make_node("Div", ["k_classes", "S"], ["u2d"]),
        helper.make_node("Squeeze", ["u2d", "axes1"], ["u"]),
    ]
    graph = helper.make_graph(
        nodes,
        "ptype_evidential_dnn_raw_inputs",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", N_FEATURES])],
        [
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", weights.n_classes]),
            helper.make_tensor_value_info("u", TensorProto.FLOAT, ["batch"]),
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # readable by every onnxruntime >= 1.14
    onnx.checker.check_model(model)
    return model


def verify(model, weights_path: Path, onnx_path: Path, n: int = 20000) -> dict:
    """Check the exported npz/onnx reproduce Keras on random realistic-range inputs."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    # Wide physical ranges; the exact distribution matters little for an export sanity check.
    x = np.concatenate([rng.uniform(-45, 35, (n, 21)), rng.uniform(-50, 25, (n, 21)), rng.normal(5, 15, (n, 21)), rng.normal(3, 15, (n, 21))], axis=1).astype(np.float32)
    w = PtypeWeights.load(weights_path)
    scaled = ((x - w.center) / w.scale).astype(np.float32)
    probs, u, _, _ = model.predict(scaled, batch_size=8192, verbose=0, return_uncertainties=True)
    probs = np.asarray(probs.cpu() if hasattr(probs, "cpu") else probs)
    u = np.asarray(u.cpu() if hasattr(u, "cpu") else u).ravel()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    p_o, u_o = sess.run(["probs", "u"], {"input": x})
    return {"onnx_max_abs_dprob": float(np.abs(p_o - probs).max()), "onnx_max_abs_du": float(np.abs(u_o - u).max())}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Keras .keras file")
    ap.add_argument("--scaler", required=True, help="bridgescaler JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip the Keras-vs-ONNX sanity check")
    args = ap.parse_args()

    from bridgescaler import load_scaler

    model = load_keras(args.model)
    scaler = load_scaler(args.scaler)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = extract(model, scaler)
    npz_path, onnx_path = out_dir / "ptype_weights.npz", out_dir / "ptype_model.onnx"
    np.savez(npz_path, **arrays)

    import onnx

    onnx.save(build_onnx(PtypeWeights.load(npz_path)), onnx_path)
    shapes = [tuple(arrays[f"W{i}"].shape) for i in range(int(arrays["n_layers"]))]
    print(f"layers (in,out): {shapes}; leaky slope {float(arrays['negative_slope'])}")
    print(f"wrote {npz_path} ({npz_path.stat().st_size / 1e3:.0f} kB), {onnx_path} ({onnx_path.stat().st_size / 1e3:.0f} kB)")
    if not args.no_verify:
        res = verify(model, npz_path, onnx_path)
        print(json.dumps(res))
        assert res["onnx_max_abs_dprob"] < 1e-4 and res["onnx_max_abs_du"] < 1e-4, "ONNX export does not match Keras"


if __name__ == "__main__":
    main()
