"""Framework-neutral inference engines for the evidential precipitation-type network.

All engines share one contract::

    probs, u = engine.predict(x, batch_size=...)

where ``x`` is a ``float32`` array of shape ``(n, 84)`` holding **raw physical-unit** inputs
(temperature degC x21, dewpoint degC x21, U wind m/s x21, V wind m/s x21 at 0..5000 m every
250 m) and the return values are ``probs`` ``(n, 4)`` float32 (rain, snow, icep, frzr) and
``u`` ``(n,)`` float32 (evidential uncertainty).

Why scaling and post-processing live inside the engine
------------------------------------------------------
* The GroupRobustScaler is ``(x - center) / scale`` with one (center, scale) pair per variable
  group. That is an affine map, so it can be folded exactly into the first Dense layer:
  ``((x - c) / s) @ W + b == x @ (W / s[:, None]) + (b - (c / s) @ W)``. The folding is done in
  float64 and rounded once to float32, so the compute graph consumes raw inputs and the scaler
  costs nothing at run time (no extra pass over a 1.9M x 84 array, no pandas DataFrame).
* The evidential head (``mlguess.keras.models.CategoricalDNN.calc_uncertainty``) is::

      evidence = relu(y); alpha = evidence + 1; S = sum(alpha, axis=1, keepdims=True)
      u = num_classes / S; prob = alpha / S

  It is cheap element-wise work, and doing it inside the graph/kernel means only the final
  ``(n, 4)`` and ``(n,)`` arrays ever leave the device.

Only :class:`KerasEngine` (the correctness baseline) needs Keras. Everything else needs only
the ``.npz`` written by ``applications/export_ptype_weights.py`` plus its own runtime
(``torch``, ``onnxruntime`` or just ``numpy`` / ``numba``). Heavy imports are deferred to the
engine constructors so importing this module is cheap and dependency-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ENGINE_NAMES",
    "N_FEATURES",
    "Engine",
    "KerasEngine",
    "NumbaEngine",
    "NumpyEngine",
    "OnnxEngine",
    "PtypeWeights",
    "TorchEngine",
    "make_engine",
]

N_FEATURES = 84


# --------------------------------------------------------------------------------------
# Weights container
# --------------------------------------------------------------------------------------
@dataclass
class PtypeWeights:
    """Dense-stack weights plus scaler constants loaded from the exported ``.npz``.

    Attributes:
        kernels: Per-layer weight matrices in ``(in, out)`` layout (Keras convention), float32.
        biases: Per-layer bias vectors, float32.
        negative_slope: Leaky-ReLU slope used by every hidden layer (the final layer is linear).
        center: Scaler centers expanded to one value per input feature, float64, shape (84,).
        scale: Scaler scales expanded to one value per input feature, float64, shape (84,).
        classes: Output class names in column order.
        heights_m: Height AGL (m) of each level inside a variable group.
        feature_names: The 84 input column names in order.
    """

    kernels: list[np.ndarray]
    biases: list[np.ndarray]
    negative_slope: float
    center: np.ndarray
    scale: np.ndarray
    classes: list[str]
    heights_m: np.ndarray
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | os.PathLike) -> PtypeWeights:
        """Read an ``.npz`` written by ``applications/export_ptype_weights.py``.

        ``allow_pickle`` stays False: the file contains only numeric and unicode arrays, so it
        is safe to load from any environment.
        """
        with np.load(path, allow_pickle=False) as z:
            n_layers = int(z["n_layers"])
            return cls(
                kernels=[np.ascontiguousarray(z[f"W{i}"], dtype=np.float32) for i in range(n_layers)],
                biases=[np.ascontiguousarray(z[f"b{i}"], dtype=np.float32) for i in range(n_layers)],
                negative_slope=float(z["negative_slope"]),
                center=np.asarray(z["center"], dtype=np.float64),
                scale=np.asarray(z["scale"], dtype=np.float64),
                classes=[str(c) for c in z["classes"]],
                heights_m=np.asarray(z["heights_m"]),
                feature_names=[str(c) for c in z["feature_names"]],
            )

    @property
    def n_classes(self) -> int:
        """Number of output classes K (the evidential ``u = K / S`` numerator)."""
        return int(self.kernels[-1].shape[1])

    def folded(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (kernels, biases) with the input scaler folded into layer 0.

        Folding is done in float64 and rounded once so the only error versus scaling first is
        one float32 rounding of each folded weight (~1e-7 relative).
        """
        w0 = self.kernels[0].astype(np.float64)
        b0 = self.biases[0].astype(np.float64)
        w0f = w0 / self.scale[:, None]
        b0f = b0 - (self.center / self.scale) @ w0
        kernels = [np.ascontiguousarray(w0f, dtype=np.float32)] + [k.copy() for k in self.kernels[1:]]
        biases = [np.ascontiguousarray(b0f, dtype=np.float32)] + [b.copy() for b in self.biases[1:]]
        return kernels, biases


def _check_input(x: np.ndarray) -> np.ndarray:
    """Validate shape/dtype once at the API boundary.

    We refuse silent float64 -> float32 conversion of a huge array (it would double peak memory
    unnoticed); callers should hand in float32 as produced by the HRRR reader.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise ValueError(f"x must be a numpy array of shape (n, {N_FEATURES}), got {getattr(x, 'shape', type(x))}")
    if x.dtype != np.float32:
        raise TypeError(f"x must be float32, got {x.dtype}")
    return x


class Engine:
    """Abstract base: ``predict(x, batch_size) -> (probs, u)``."""

    name = "base"
    default_batch_size = 65536

    def predict(self, x: np.ndarray, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities and evidential uncertainty from raw inputs.

        Args:
            x: float32 array (n, 84) in physical units.
            batch_size: Rows per forward pass. ``None`` uses the engine default.

        Returns:
            ``(probs, u)``: float32 arrays of shape (n, 4) and (n,).
        """
        x = _check_input(x)
        bs = int(batch_size or self.default_batch_size)
        if bs < 1:
            raise ValueError("batch_size must be >= 1")
        return self._predict(x, bs)

    def _predict(self, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def describe(self) -> dict:
        """Small dict of settings for benchmark logs."""
        return {"engine": self.name}


# --------------------------------------------------------------------------------------
# Keras baseline
# --------------------------------------------------------------------------------------
class KerasEngine(Engine):
    """Baseline: ``CategoricalDNN.predict`` with the scaler applied in numpy.

    This is what the production web backend does today (minus pandas), and is the reference
    every other engine is compared with. Scaling is done in numpy float32 exactly like
    ``bridgescaler``'s ``(x - center) / scale``.
    """

    name = "keras"
    default_batch_size = 65536

    def __init__(self, weights_path: str, keras_path: str, device: str = "cpu", threads: int | None = None, **_: object):
        """Load the Keras model; ``weights_path`` supplies the scaler constants.

        Args:
            weights_path: Exported ``.npz`` (only center/scale are used).
            keras_path: Path to the ``.keras`` file.
            device: ``cpu`` / ``cuda`` / ``mps``. Keras' torch backend otherwise silently picks
                the best accelerator it finds, which would make a "CPU baseline" run on the GPU.
            threads: ``torch.set_num_threads`` (Keras' torch backend computes with torch).
        """
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras
        import mlguess.keras.callbacks
        import mlguess.keras.models  # noqa: F401
        from mlguess.keras.losses import evidential_cat_loss

        w = PtypeWeights.load(weights_path)
        self.center = w.center.astype(np.float32)
        self.scale = w.scale.astype(np.float32)
        if threads:
            import torch

            torch.set_num_threads(int(threads))
        self.keras = keras
        self.device = device
        with keras.device(device):
            self.model = keras.models.load_model(keras_path, custom_objects={"loss": evidential_cat_loss})

    def describe(self):
        return {"engine": self.name, "device": self.device}

    def _predict(self, x, batch_size):
        scaled = (x - self.center) / self.scale
        with self.keras.device(self.device):
            probs, u, _, _ = self.model.predict(scaled, batch_size=batch_size, verbose=0, return_uncertainties=True)
        # With the torch backend the outputs are torch tensors (possibly on an accelerator).
        probs = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
        u = u.detach().cpu().numpy() if hasattr(u, "detach") else np.asarray(u)
        return probs.astype(np.float32, copy=False), u.reshape(-1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------
def _build_torch_module(weights: PtypeWeights):
    """Build the network as ``nn.Sequential`` wrapped with the evidential head.

    A plain Sequential of Linear + LeakyReLU is the form ``torch.compile`` and TorchScript
    handle best; Dropout is omitted because it is the identity at inference.
    """
    import torch
    from torch import nn

    kernels, biases = weights.folded()
    layers: list[nn.Module] = []
    for i, (k, b) in enumerate(zip(kernels, biases, strict=True)):
        lin = nn.Linear(k.shape[0], k.shape[1])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(k.T.copy()))  # torch stores (out, in)
            lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
        if i < len(kernels) - 1:
            layers.append(nn.LeakyReLU(weights.negative_slope))
    net = nn.Sequential(*layers)
    k_classes = float(weights.n_classes)

    class PtypeNet(nn.Module):
        """Raw physical inputs -> (probs, u). Head runs in float32 even under autocast."""

        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            y = self.net(x).float()  # float16 logits -> float32 before the (1+evidence)/S arithmetic
            alpha = torch.relu(y) + 1.0
            s = alpha.sum(dim=1, keepdim=True)
            return alpha / s, (k_classes / s).squeeze(1)

    return PtypeNet()


class TorchEngine(Engine):
    """PyTorch engine with scaling folded into layer 0 and the evidential head in-graph.

    Options:
        device: ``cpu``, ``cuda`` (``cuda:N``) or ``mps``.
        dtype: ``float32`` (default), ``float16`` or ``bfloat16``. Non-float32 uses
            ``torch.autocast`` so weights stay float32 masters and the head stays float32; this
            helps only on accelerators with fast half-precision matmul (A100 tensor cores).
            Expect ~1e-3 probability error for float16 (see tests); the default is exact-ish.
        tf32: Allow TF32 matmuls on Ampere+ (float32 inputs, 10-bit mantissa): ~1e-3 error,
            often 2-5x faster than strict float32 on A100.
        compile: Wrap the module in ``torch.compile`` (fuses leaky-relu/bias, head). Pads the
            last batch to ``batch_size`` so a single compiled shape is reused.
        threads: ``torch.set_num_threads`` for CPU intra-op parallelism (global setting).
        pin_memory: On CUDA, stage batches through two pinned buffers and copy with
            ``non_blocking=True`` so H2D/D2H overlap compute and the next host memcpy.
    """

    name = "torch"
    default_batch_size = 65536

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        tf32: bool = False,
        compile: bool = False,
        threads: int | None = None,
        pin_memory: bool = True,
        compile_mode: str | None = None,
        **_: object,
    ):
        import torch

        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.autocast_dtype = None if dtype == "float32" else getattr(torch, dtype)
        self.compile = bool(compile)
        self.pin = bool(pin_memory) and self.device.type == "cuda"
        if tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self.tf32 = bool(tf32)
        module = _build_torch_module(PtypeWeights.load(weights_path)).to(self.device).eval()
        self.module = torch.compile(module, mode=compile_mode) if self.compile else module

    def describe(self):
        return {
            "engine": self.name,
            "device": str(self.device),
            "dtype": self.dtype_name,
            "tf32": self.tf32,
            "compile": self.compile,
            "threads": self.torch.get_num_threads(),
            "pinned": self.pin,
        }

    def _forward(self, xb):
        """One forward pass on a device tensor; autocast only when a half dtype was requested."""
        if self.autocast_dtype is None:
            return self.module(xb)
        with self.torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            return self.module(xb)

    def _predict(self, x, batch_size):
        torch = self.torch
        n = x.shape[0]
        probs = np.empty((n, 4), dtype=np.float32)
        u = np.empty(n, dtype=np.float32)
        if n == 0:
            return probs, u
        with torch.inference_mode():
            if self.pin:
                self._predict_pinned(x, batch_size, probs, u)
            else:
                for i in range(0, n, batch_size):
                    chunk = x[i : i + batch_size]
                    m = chunk.shape[0]
                    if self.compile and m < batch_size:
                        # One static shape avoids a recompile for the ragged tail.
                        pad = np.zeros((batch_size, N_FEATURES), dtype=np.float32)
                        pad[:m] = chunk
                        chunk = pad
                    p, uu = self._forward(torch.from_numpy(chunk).to(self.device))
                    probs[i : i + m] = p[:m].cpu().numpy()
                    u[i : i + m] = uu[:m].cpu().numpy()
        return probs, u

    def _predict_pinned(self, x, batch_size, probs, u):
        """Double-buffered pinned-memory pipeline for CUDA.

        Pageable host memory forces a synchronous staged copy; pinned memory lets the DMA engine
        run asynchronously while the CPU memcpys the next batch and the GPU computes. Two input
        and two output buffers are used round-robin; a CUDA event per slot guards reuse so a
        buffer is never overwritten while its previous H2D/D2H is still in flight.
        """
        torch = self.torch
        n = x.shape[0]
        bs = min(batch_size, n)
        in_buf = [torch.empty((bs, N_FEATURES), dtype=torch.float32, pin_memory=True) for _ in range(2)]
        p_buf = [torch.empty((bs, 4), dtype=torch.float32, pin_memory=True) for _ in range(2)]
        u_buf = [torch.empty(bs, dtype=torch.float32, pin_memory=True) for _ in range(2)]
        done = [torch.cuda.Event() for _ in range(2)]
        have_event = [False, False]
        pending: list[tuple[int, int] | None] = [None, None]

        def finalize(slot: int) -> None:
            job = pending[slot]
            if job is None:
                return
            done[slot].synchronize()
            i0, m = job
            probs[i0 : i0 + m] = p_buf[slot][:m].numpy()
            u[i0 : i0 + m] = u_buf[slot][:m].numpy()
            pending[slot] = None

        slot = 0
        for i0 in range(0, n, bs):
            if have_event[slot]:
                finalize(slot)  # frees this slot's input and output buffers
            chunk = x[i0 : i0 + bs]
            m = chunk.shape[0]
            in_buf[slot].numpy()[:m] = chunk
            if m < bs and self.compile:
                in_buf[slot].numpy()[m:] = 0.0
            xb = in_buf[slot].to(self.device, non_blocking=True)
            p, uu = self._forward(xb)
            p_buf[slot].copy_(p, non_blocking=True)
            u_buf[slot].copy_(uu, non_blocking=True)
            done[slot].record()
            have_event[slot] = True
            pending[slot] = (i0, m)
            slot ^= 1
        finalize(0)
        finalize(1)


# --------------------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------------------
class OnnxEngine(Engine):
    """ONNX Runtime engine; the exported graph already contains scaling and the evidential head.

    Args:
        weights_path: The exported ``.npz`` *or* the ``.onnx`` itself; if an ``.npz`` is given
            the sibling ``<stem>.onnx`` / ``ptype_model.onnx`` is used unless ``onnx_path`` is set.
        device: ``cpu`` or ``cuda`` (``cuda:N``) -> CUDAExecutionProvider with CPU fallback.
        threads: ``intra_op_num_threads`` (0 / None lets ORT choose = physical cores).
        onnx_path: Explicit path to the ``.onnx``.
    """

    name = "onnx"
    default_batch_size = 65536

    def __init__(self, weights_path: str, device: str = "cpu", threads: int | None = None, onnx_path: str | None = None, **_: object):
        import onnxruntime as ort

        p = Path(onnx_path or weights_path)
        if p.suffix == ".npz":
            cand = [p.with_suffix(".onnx"), p.with_name("ptype_model.onnx")]
            p = next((c for c in cand if c.exists()), cand[0])
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or 0)
        so.inter_op_num_threads = 1  # a purely sequential graph gains nothing from inter-op threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers: list = ["CPUExecutionProvider"]
        self.device = device
        if device.startswith("cuda"):
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(f"CUDAExecutionProvider unavailable; have {ort.get_available_providers()}")
            providers = [("CUDAExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]
        self.threads = so.intra_op_num_threads
        self.sess = ort.InferenceSession(str(p), so, providers=providers)
        self.providers = self.sess.get_providers()

    def describe(self):
        return {"engine": self.name, "device": self.device, "threads": self.threads, "providers": self.providers}

    def _predict(self, x, batch_size):
        n = x.shape[0]
        probs = np.empty((n, 4), dtype=np.float32)
        u = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch_size):
            chunk = np.ascontiguousarray(x[i : i + batch_size])
            p, uu = self.sess.run(["probs", "u"], {"input": chunk})
            probs[i : i + chunk.shape[0]] = p
            u[i : i + chunk.shape[0]] = uu
        return probs, u


# --------------------------------------------------------------------------------------
# NumPy / Numba (zero framework dependency)
# --------------------------------------------------------------------------------------
class NumpyEngine(Engine):
    """Pure-numpy engine: chunked float32 ``matmul`` into preallocated buffers.

    Chunks are kept small by default (16k rows -> ~13 MB per 200-wide activation buffer) so the
    activations stay near cache; ``np.matmul(..., out=)`` and in-place ops avoid allocating
    per layer. Leaky-ReLU is ``max(z, slope * z)`` which is exact for 0 < slope < 1.
    Thread count is the BLAS thread count (controlled through ``threadpoolctl`` when present,
    otherwise via OMP/OPENBLAS/VECLIB env vars set before numpy is imported).
    """

    name = "numpy"
    default_batch_size = 16384

    def __init__(self, weights_path: str, threads: int | None = None, **_: object):
        w = PtypeWeights.load(weights_path)
        self.kernels, self.biases = w.folded()
        self.slope = np.float32(w.negative_slope)
        self.k = np.float32(w.n_classes)
        self.threads = threads
        self._limit = None
        if threads:
            try:
                from threadpoolctl import threadpool_limits

                self._limit = threadpool_limits(limits=int(threads), user_api="blas")
            except ImportError:  # pragma: no cover - optional dependency
                pass

    def describe(self):
        return {"engine": self.name, "threads": self.threads}

    def _predict(self, x, batch_size):
        n = x.shape[0]
        probs = np.empty((n, 4), dtype=np.float32)
        u = np.empty(n, dtype=np.float32)
        widths = [k.shape[1] for k in self.kernels]
        bufs = [np.empty((min(batch_size, max(n, 1)), w), dtype=np.float32) for w in widths]
        scratch = [np.empty_like(b) for b in bufs[:-1]]
        for i in range(0, n, batch_size):
            h = x[i : i + batch_size]
            m = h.shape[0]
            for li, (k, b) in enumerate(zip(self.kernels, self.biases, strict=True)):
                z = bufs[li][:m]
                np.matmul(h, k, out=z)
                z += b
                if li < len(self.kernels) - 1:
                    t = scratch[li][:m]
                    np.multiply(z, self.slope, out=t)
                    np.maximum(z, t, out=z)
                h = z
            alpha = h  # (m, 4) logits -> alpha in place
            np.maximum(alpha, 0.0, out=alpha)
            alpha += 1.0
            s = alpha.sum(axis=1, keepdims=True)
            np.divide(alpha, s, out=probs[i : i + m])
            np.divide(self.k, s[:, 0], out=u[i : i + m])
        return probs, u


class NumbaEngine(Engine):
    """Numba-JIT fused kernel: all six layers + evidential head per tile of 8 rows, parallel over tiles.

    Why tiles: a single-row matrix-vector product re-reads every weight for every row; with 8 rows per
    tile each weight row ``W[k, :]`` is loaded once and reused for 8 rows while it is hot in L1
    (measured 1.06M vs 0.66M rows/s with 14 threads for 8-row tiles vs single rows; 4, 16 and 32-row
    tiles were within 3% of 8). Inner loops are contiguous ``axpy`` (``z[r, :] += x[r, k] * W[k, :]``) so
    LLVM auto-vectorises them; ``fastmath`` permits the reassociation that needs. No intermediate
    activation ever reaches main memory and no BLAS/framework is needed. Compiled once per machine
    (``cache=True``). It is slower than a tuned BLAS (torch/ONNX) on large batches; its value is
    zero dependencies beyond numpy+numba and deterministic memory (no per-batch allocation).
    """

    name = "numba"
    default_batch_size = 1 << 20

    def __init__(self, weights_path: str, threads: int | None = None, **_: object):
        import numba

        w = PtypeWeights.load(weights_path)
        self.kernels, self.biases = w.folded()
        if len(self.kernels) != 6:
            raise ValueError("NumbaEngine kernel is specialised for 6 dense layers (84-84-200-200-200-200-4)")
        self.slope = float(w.negative_slope)
        self.k = float(w.n_classes)
        if threads:
            numba.set_num_threads(int(threads))
        self.threads = numba.get_num_threads()
        self._kernel = _get_numba_kernel()

    def describe(self):
        return {"engine": self.name, "threads": self.threads}

    def _predict(self, x, batch_size):
        n = x.shape[0]
        probs = np.empty((n, 4), dtype=np.float32)
        u = np.empty(n, dtype=np.float32)
        k, b = self.kernels, self.biases
        for i in range(0, n, batch_size):
            chunk = np.ascontiguousarray(x[i : i + batch_size])
            m = chunk.shape[0]
            self._kernel(chunk, k[0], b[0], k[1], b[1], k[2], b[2], k[3], b[3], k[4], b[4], k[5], b[5], self.slope, self.k, probs[i : i + m], u[i : i + m])
        return probs, u


_NUMBA_KERNEL = None
_TILE = 8


def _get_numba_kernel():
    """Compile (once per process) and return the fused, tiled numba kernel."""
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is not None:
        return _NUMBA_KERNEL
    import numba
    from numba import prange

    tile = _TILE

    @numba.njit(cache=True, fastmath=True)  # serial: called from inside the parallel prange below
    def dense_tile(h_in, w, b, h_out, slope, activate, m):
        # h_out[:m] = act(h_in[:m] @ w + b); loop order keeps the innermost loop contiguous in w and h_out.
        n_in, n_out = w.shape
        for r in range(m):
            for j in range(n_out):
                h_out[r, j] = b[j]
        for k in range(n_in):
            for r in range(m):
                a = h_in[r, k]
                for j in range(n_out):
                    h_out[r, j] += a * w[k, j]
        if activate:
            for r in range(m):
                for j in range(n_out):
                    v = h_out[r, j]
                    h_out[r, j] = v if v > 0.0 else slope * v

    @numba.njit(cache=True, parallel=True, fastmath=True)
    def kernel(x, w0, b0, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, slope, k_classes, probs, u):
        n = x.shape[0]
        for t in prange((n + tile - 1) // tile):
            r0 = t * tile
            m = min(tile, n - r0)
            # Per-tile scratch (allocated once per tile, ~7 kB each). Two 200-wide buffers ping-pong through the
            # three 200->200 layers; the 84-wide and 4-wide ones serve the first and last layer.
            h0 = np.empty((tile, w0.shape[1]), dtype=np.float32)
            h1 = np.empty((tile, w1.shape[1]), dtype=np.float32)
            h2 = np.empty((tile, w2.shape[1]), dtype=np.float32)
            y = np.empty((tile, w5.shape[1]), dtype=np.float32)
            dense_tile(x[r0 : r0 + m], w0, b0, h0, slope, True, m)
            dense_tile(h0, w1, b1, h1, slope, True, m)
            dense_tile(h1, w2, b2, h2, slope, True, m)
            dense_tile(h2, w3, b3, h1, slope, True, m)
            dense_tile(h1, w4, b4, h2, slope, True, m)
            dense_tile(h2, w5, b5, y, slope, False, m)
            for r in range(m):
                s = 0.0
                for j in range(y.shape[1]):
                    a = max(0.0, y[r, j]) + 1.0
                    y[r, j] = a
                    s += a
                for j in range(y.shape[1]):
                    probs[r0 + r, j] = y[r, j] / s
                u[r0 + r] = k_classes / s

    _NUMBA_KERNEL = kernel
    return kernel


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
ENGINE_NAMES = ("keras", "torch", "onnx", "numpy", "numba")


def make_engine(name: str, weights_path: str, device: str = "cpu", **opts) -> Engine:
    """Create an engine by name.

    Args:
        name: One of ``keras``, ``torch``, ``onnx``, ``numpy``, ``numba``.
        weights_path: Exported ``.npz`` (``onnx`` also accepts a ``.onnx`` path).
        device: ``cpu`` / ``cuda[:N]`` / ``mps`` where the engine supports it.
        **opts: Engine-specific options (``threads``, ``dtype``, ``tf32``, ``compile``,
            ``pin_memory``, ``keras_path``, ``onnx_path`` ...). Unknown options are ignored
            by engines that do not use them so one CLI can drive every engine.

    Raises:
        ValueError: For an unknown name or an unsupported device for that engine.
    """
    name = name.lower()
    if name in ("numpy", "numba") and device != "cpu":
        raise ValueError(f"{name} engine is CPU-only (got device={device!r})")
    if name == "keras":
        if not opts.get("keras_path"):
            raise ValueError("keras engine needs keras_path=...")
        return KerasEngine(weights_path, device=device, **opts)
    if name == "torch":
        return TorchEngine(weights_path, device=device, **opts)
    if name == "onnx":
        return OnnxEngine(weights_path, device=device, **opts)
    if name == "numpy":
        return NumpyEngine(weights_path, **opts)
    if name == "numba":
        return NumbaEngine(weights_path, **opts)
    raise ValueError(f"unknown engine {name!r}; choose from {ENGINE_NAMES}")
