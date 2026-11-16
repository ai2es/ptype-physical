"""Parity tests: every inference engine versus the Keras model on raw physical-unit inputs.

Needs the Keras model + scaler (the only place Keras is required). Paths come from
``PTYPE_KERAS_MODEL`` / ``PTYPE_SCALER`` / ``PTYPE_WEB_FIXTURES`` and default to the ptype-web
checkout; the tests skip when the files or optional runtimes are unavailable.

Tolerance rationale
-------------------
All engines run float32 arithmetic through the same 6-layer network, but they differ from Keras in
(a) folding the scaler into layer 0 (one extra float32 rounding of each weight), (b) summation
order inside matmul, and (c) ONNX/BLAS kernel fusion. These perturb logits by ~1e-6 relative;
since ``prob = alpha / S`` and ``u = 4 / S`` with ``S >= 4`` the output error stays well below
1e-5. We assert ``PROB_TOL = U_TOL = 5e-5``: ~10x above the measured worst case (so the tests are
not flaky across BLAS/CPU vendors) yet ~20x tighter than the 1e-3 that would matter for a
categorical map. Half-precision (autocast) is checked separately with a loose 2e-2 tolerance.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from ptype.hrrr_fast.engine import ENGINE_NAMES, PtypeWeights, make_engine

REPO = Path(__file__).resolve().parents[1]
WEB = Path("/Users/dgagne/ptype-web")
MODEL = Path(os.environ.get("PTYPE_KERAS_MODEL", WEB / "backend/ptype_model_20240909.keras"))
SCALER = Path(os.environ.get("PTYPE_SCALER", WEB / "backend/ptype_scaler_20240909.json"))
FIXTURE_CANDIDATES = [
    Path(p)
    for p in (
        os.environ.get("PTYPE_WEB_FIXTURES", ""),
        WEB / "web/src/model/fixtures.json",
        *sorted(WEB.glob(".claude/worktrees/*/web/src/model/fixtures.json")),
    )
    if str(p)
]
PROB_TOL = 5e-5
U_TOL = 5e-5
N_ROWS = 100_000

pytestmark = pytest.mark.skipif(not (MODEL.exists() and SCALER.exists()), reason="Keras model/scaler not available")


def _import_export_module():
    """Import applications/export_ptype_weights.py (a script, not a package) by path."""
    spec = importlib.util.spec_from_file_location("export_ptype_weights", REPO / "applications/export_ptype_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def realistic_soundings(n: int, seed: int = 0) -> np.ndarray:
    """Synthetic 84-feature rows spanning winter-to-summer HRRR-like profiles.

    Surface temperature -30..+35 C, lapse rates 2-9 C/km with random inversions/warm noses, dewpoint
    depression 0-25 C (saturated layers are common in precipitation), winds with shear. Uniform ranges
    are intentionally wider than the training mean so tails of the scaler are exercised too.
    """
    rng = np.random.default_rng(seed)
    z = np.arange(0, 5250, 250) / 1000.0  # km
    t0 = rng.uniform(-30, 35, (n, 1))
    lapse = rng.uniform(2, 9, (n, 1))
    # warm nose / inversion: Gaussian bump of up to +8 C centred 0.5..3 km
    bump_amp = rng.uniform(-2, 8, (n, 1)) * (rng.random((n, 1)) < 0.4)
    bump_c = rng.uniform(0.5, 3.0, (n, 1))
    t = t0 - lapse * z + bump_amp * np.exp(-(((z - bump_c) / 0.7) ** 2)) + rng.normal(0, 0.5, (n, 21))
    dep = rng.uniform(0, 25, (n, 1)) * rng.uniform(0.2, 1.0, (n, 21))
    td = t - dep
    u = rng.normal(2, 8, (n, 1)) + rng.normal(0, 4, (n, 1)) * z + rng.normal(0, 1.5, (n, 21))
    v = rng.normal(1, 8, (n, 1)) + rng.normal(0, 4, (n, 1)) * z + rng.normal(0, 1.5, (n, 21))
    return np.concatenate([t, td, u, v], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Run the real export script's functions once; returns (npz_path, onnx_path, keras_model)."""
    exp = _import_export_module()
    from bridgescaler import load_scaler

    model = exp.load_keras(str(MODEL))
    out = tmp_path_factory.mktemp("weights")
    arrays = exp.extract(model, load_scaler(str(SCALER)))
    npz = out / "ptype_weights.npz"
    np.savez(npz, **arrays)
    import onnx

    onnx.save(exp.build_onnx(PtypeWeights.load(npz)), out / "ptype_weights.onnx")
    return npz, out / "ptype_weights.onnx", model


@pytest.fixture(scope="module")
def data_and_reference(exported):
    """100k realistic rows and the Keras-engine reference outputs."""
    npz, _, _ = exported
    x = realistic_soundings(N_ROWS)
    ref = make_engine("keras", str(npz), device="cpu", keras_path=str(MODEL)).predict(x, batch_size=8192)
    return x, ref


def _engine_cases():
    cases = [("numpy", {}), ("numba", {}), ("onnx", {}), ("torch", {}), ("torch", {"compile": True})]
    return [pytest.param(n, o, id=f"{n}-{'-'.join(map(str, o)) or 'default'}") for n, o in cases]


def test_export_contents(exported):
    """The npz has the architecture, slope and expanded scaler that the engines assume."""
    npz, _, _ = exported
    w = PtypeWeights.load(npz)
    assert [k.shape for k in w.kernels] == [(84, 84), (84, 200), (200, 200), (200, 200), (200, 200), (200, 4)]
    assert w.negative_slope == pytest.approx(0.2)
    assert w.center.shape == (84,) and w.scale.shape == (84,)
    assert w.classes == ["rain", "snow", "icep", "frzr"]
    assert len(w.feature_names) == 84 and len(w.heights_m) == 21
    # four groups, each constant across its 21 levels
    assert all(np.ptp(w.center[i * 21 : (i + 1) * 21]) == 0 for i in range(4))


@pytest.mark.parametrize("name,opts", _engine_cases())
def test_engine_matches_keras(name, opts, exported, data_and_reference, record_property):
    """Each engine reproduces Keras probs/u on 100k rows within the documented tolerance."""
    if name == "onnx":
        pytest.importorskip("onnxruntime")
    if name in ("torch",):
        pytest.importorskip("torch")
    npz, onnx_path, _ = exported
    x, (p_ref, u_ref) = data_and_reference
    eng = make_engine(name, str(npz), onnx_path=str(onnx_path), **opts)
    try:
        probs, u = eng.predict(x, batch_size=30000)  # 100k rows in 30k batches -> a ragged 10k tail
    except Exception as exc:
        # torch.compile on CPU needs a working C++/OpenMP toolchain; that is an environment property, not an engine bug.
        if opts.get("compile") and "Inductor" in type(exc).__name__:
            pytest.skip(f"torch.compile toolchain unavailable: {type(exc).__name__}")
        raise
    assert probs.dtype == np.float32 and u.dtype == np.float32
    assert probs.shape == (N_ROWS, 4) and u.shape == (N_ROWS,)
    dp, du = float(np.abs(probs - p_ref).max()), float(np.abs(u - u_ref).max())
    record_property("max_abs_dprob", dp)
    record_property("max_abs_du", du)
    print(f"\nPARITY {name} {opts}: max|dprob|={dp:.3e} max|du|={du:.3e}")
    assert dp < PROB_TOL and du < U_TOL
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", ["numpy", "numba", "onnx", "torch"])
def test_ragged_batches_equal_single_batch(name, exported, data_and_reference):
    """A batch size that does not divide n must give the same answer as one big batch."""
    npz, onnx_path, _ = exported
    x, _ = data_and_reference
    x = x[:5003]
    eng = make_engine(name, str(npz), onnx_path=str(onnx_path))
    p1, u1 = eng.predict(x, batch_size=1000)
    p2, u2 = eng.predict(x, batch_size=10**6)
    np.testing.assert_allclose(p1, p2, atol=1e-6)
    np.testing.assert_allclose(u1, u2, atol=1e-6)


def test_torch_half_precision_if_accelerator(exported, data_and_reference):
    """float16 autocast on an accelerator stays within 2e-2 of Keras (half-precision mantissa)."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if dev is None:
        pytest.skip("no accelerator")
    npz, _, _ = exported
    x, (p_ref, u_ref) = data_and_reference
    probs, u = make_engine("torch", str(npz), device=dev, dtype="float16").predict(x[:20000], batch_size=8192)
    dp, du = float(np.abs(probs - p_ref[:20000]).max()), float(np.abs(u - u_ref[:20000]).max())
    print(f"\nPARITY torch {dev} float16: max|dprob|={dp:.3e} max|du|={du:.3e}")
    assert dp < 2e-2 and du < 2e-2


def test_torch_mps_float32_if_available(exported, data_and_reference):
    """MPS float32 matches Keras within the standard tolerance."""
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    npz, _, _ = exported
    x, (p_ref, u_ref) = data_and_reference
    probs, u = make_engine("torch", str(npz), device="mps").predict(x[:20000], batch_size=8192)
    dp, du = float(np.abs(probs - p_ref[:20000]).max()), float(np.abs(u - u_ref[:20000]).max())
    print(f"\nPARITY torch mps float32: max|dprob|={dp:.3e} max|du|={du:.3e}")
    assert dp < PROB_TOL and du < U_TOL


def test_input_validation(exported):
    """Wrong shape or dtype is rejected instead of silently converted."""
    npz, _, _ = exported
    eng = make_engine("numpy", str(npz))
    with pytest.raises(ValueError):
        eng.predict(np.zeros((3, 83), dtype=np.float32))
    with pytest.raises(TypeError):
        eng.predict(np.zeros((3, 84), dtype=np.float64))
    with pytest.raises(ValueError):
        make_engine("nope", str(npz))
    assert set(ENGINE_NAMES) == {"keras", "torch", "onnx", "numpy", "numba"}


@pytest.mark.skipif(not any(p.exists() for p in FIXTURE_CANDIDATES), reason="web fixtures.json not found")
@pytest.mark.parametrize("name", ["numpy", "numba", "onnx", "torch"])
def test_web_fixtures(name, exported):
    """Reproduce the Keras outputs stored in web/src/model/fixtures.json (same raw inputs, tol 1e-4 as the JS test)."""
    npz, onnx_path, _ = exported
    fx = json.loads(next(p for p in FIXTURE_CANDIDATES if p.exists()).read_text())
    x = np.asarray(fx["inputs"], dtype=np.float32)
    probs, u = make_engine(name, str(npz), onnx_path=str(onnx_path)).predict(x, batch_size=16)
    dp = float(np.abs(probs - np.asarray(fx["prob"])).max())
    du = float(np.abs(u - np.asarray(fx["u"]).ravel()).max())
    print(f"\nFIXTURES {name}: n={len(x)} max|dprob|={dp:.3e} max|du|={du:.3e}")
    assert dp < 1e-4 and du < 1e-4
