# Local (laptop) engine benchmark results

**These are laptop numbers from a shared Apple-silicon Mac, NOT Derecho numbers.** They exist to smoke-test
`benchmarks/engine_benchmark.py` and to give first relative orderings. Do not extrapolate absolute rows/s
to a 128-core Derecho CPU node or an A100.

## Machine
- Apple M4 Pro, 14 logical cores (10 performance + 4 efficiency), 48 GB unified memory, macOS 26.7.2 (Darwin 25.6.0), arm64.
- **The machine was shared with other agents running heavy jobs during these runs.** The load averages recorded at the
  start of each run (1/5/15 min, in the per-run JSON) ranged from 6 to 52 on 14 cores, so every CPU number is depressed
  and noisy (three full sweeps of the same torch-CPU configuration gave best-median throughput of 2.61M, 2.76M and 3.08M rows/s).
  Treat differences below ~30% as noise.
- Python 3.12.12; numpy 2.5.5 (BLAS thread count set through threadpoolctl),
  torch 2.15.0, onnxruntime 1.32.0 (CPUExecutionProvider only on macOS), numba 0.68.1, keras 3.15.1 (torch backend), miles-guess 2025.1.0,
  bridgescaler 0.9.0, onnx 1.23.2, psutil, threadpoolctl 3.8.0.
- Model: 84-84-200-200-200-200-4 dense, leaky-ReLU 0.2, 145,544 parameters; weights exported once with
  `applications/export_ptype_weights.py`.

## Commands
```
export KERAS_BACKEND=torch
PYTHONPATH=. python applications/export_ptype_weights.py --model ptype_model_20240909.keras --scaler ptype_scaler_20240909.json --out-dir weights/
B="python benchmarks/engine_benchmark.py --weights weights/ptype_weights.npz --out benchmarks/results/engine_local --md-append"
$B --engine numpy --threads 14 --label numpy-blas --json-out r1.json
$B --engine numba --threads 14 --json-out r2.json
$B --engine onnx  --threads 14 --json-out r3.json
$B --engine torch --threads 14 --json-out r4.json
$B --engine torch --threads 14 --compile --label torch-compile-cpu --json-out r5.json
$B --engine keras --threads 14 --keras-path ptype_model_20240909.keras --json-out r6.json
$B --engine torch --device mps --threads 14 --json-out r7.json
$B --engine torch --device mps --threads 14 --dtype float16 --label torch-mps-fp16 --json-out r8.json
$B --engine keras --device mps --threads 14 --keras-path ptype_model_20240909.keras --json-out r9.json
```
Defaults used: 1,905,141 synthetic rows, batch sizes 2^10..2^21, 1 warm-up pass + 3 timed passes, max-batch search from 2^10 with
a 2^22-row cap. All engines receive raw physical-unit float32 input; scaling and the evidential head are inside each engine.

## Caveats on what was and was not measured
- **No out-of-memory limit was reached locally.** Every max-batch search ended at the 2^22-row cap, or (MPS/Keras-MPS, where memory is
  unified with the OS) at the RAM safety guard (estimated footprint above 60% of free RAM). "Max batch" values here are therefore lower bounds set
  by the cap/guard, not true limits. True OOM behaviour (CUDA 40 GB, 128-core node RAM) is still to be measured on Derecho.
- Peak RSS is the process peak sampled every 5 ms. The "RSS above pre-pass" column in the max-batch tables subtracts the RSS at the
  start of that pass (the allocator may keep pages from earlier passes, which inflates absolute RSS for later rows). GPU columns are empty on this Mac because
  the benchmark only instruments CUDA peak-memory counters (MPS memory is not recorded).
- torch.compile on CPU **failed** on this Mac (Inductor C++ compile error: the Homebrew clang/OpenMP toolchain); recorded below as a failed run.
  It is untested anywhere else, including CUDA.
- CUDA paths (pinned double-buffered transfer, TF32, float16 autocast, ORT CUDAExecutionProvider) are **implemented but have not been run**
  (no NVIDIA GPU here). float16 autocast was exercised on MPS only (parity 3.9e-3 max |dprob| vs Keras).

## Summary (best median batch size per configuration)
| configuration | device | threads | best batch | rows/s (median of 3) | s per 1.9M-row hour | largest successful single pass (rows) | peak RSS GB at that size |
|---|---|---:|---:|---:|---:|---|---:|
| numpy-blas | cpu | 14 | 16,384 | 2,254,449 | 0.85 | 4,194,304 (row cap 2^22) | 25.0 |
| numba | cpu | 14 | 2,097,152 | 1,064,440 | 1.79 | 4,194,304 (row cap 2^22) | 3.0 |
| onnx | cpu | 14 | 8,192 | 2,399,920 | 0.79 | 4,194,304 (row cap 2^22) | 24.0 |
| torch (cpu, fp32) | cpu | 14 | 16,384 | 3,084,296 | 0.62 | 4,194,304 (row cap 2^22) | 11.5 |
| torch-compile-cpu | cpu | 14 | failed: InductorError: CppCompileError: C++ compile e | | | | |
| keras (cpu) | cpu | 14 | 131,072 | 1,677,119 | 1.14 | 4,194,304 (row cap 2^22) | 19.4 |
| torch (mps, fp32) | mps | - | 32,768 | 6,673,978 | 0.29 | 4,194,304 (row cap 2^22) | 3.3 |
| torch-mps-fp16 | mps | - | 32,768 | 9,024,189 | 0.21 | 4,194,304 (row cap 2^22) | 3.3 |
| keras (mps) | mps | - | 32,768 | 3,002,657 | 0.63 | 2,097,152 (SKIPPED) | 6.8 |

## Full tables

### numpy-blas: engine `numpy`, device `cpu`, threads `14`

Engine config: `{"engine": "numpy", "threads": 14}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,903,045 | 1,928,104 | 1.00 | 1.47 | - |
| 2,048 | 1,942,741 | 1,953,546 | 0.98 | 1.51 | - |
| 4,096 | 2,195,212 | 2,197,127 | 0.87 | 1.60 | - |
| 8,192 | 2,204,332 | 2,216,247 | 0.86 | 1.73 | - |
| 16,384 | 2,254,449 | 2,255,853 | 0.85 | 1.86 | - |
| 32,768 | 2,242,019 | 2,245,477 | 0.85 | 2.08 | - |
| 65,536 | 2,203,052 | 2,227,508 | 0.86 | 2.50 | - |
| 131,072 | 2,207,478 | 2,214,543 | 0.86 | 3.34 | - |
| 262,144 | 2,152,058 | 2,162,630 | 0.89 | 5.03 | - |
| 524,288 | 1,962,222 | 1,979,562 | 0.97 | 8.39 | - |
| 1,048,576 | 1,527,780 | 1,545,183 | 1.25 | 15.10 | - |
| 2,097,152 | 1,183,406 | 1,214,677 | 1.61 | 21.87 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 907,299 | 7.73 | 0.00 | - | - |
| 2,048 | ok | 1,077,730 | 7.73 | 0.00 | - | - |
| 4,096 | ok | 874,637 | 7.73 | 0.00 | - | - |
| 8,192 | ok | 1,005,884 | 7.73 | 0.00 | - | - |
| 16,384 | ok | 1,494,834 | 7.73 | 0.00 | - | - |
| 32,768 | ok | 1,162,486 | 7.73 | 0.00 | - | - |
| 65,536 | ok | 1,294,821 | 7.73 | 0.00 | - | - |
| 131,072 | ok | 1,306,881 | 7.73 | 0.00 | - | - |
| 262,144 | ok | 1,383,651 | 7.73 | 0.00 | - | - |
| 524,288 | ok | 1,338,694 | 7.73 | 0.00 | - | - |
| 1,048,576 | ok | 1,259,205 | 14.44 | 6.71 | - | - |
| 2,097,152 | ok | 1,118,850 | 23.27 | 14.83 | - | - |
| 4,194,304 | ok | 430,486 | 25.03 | 15.89 | - | - |

### numba: engine `numba`, device `cpu`, threads `14`

Engine config: `{"engine": "numba", "threads": 14}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 878,813 | 883,788 | 2.17 | 1.51 | - |
| 2,048 | 949,462 | 957,556 | 2.01 | 1.51 | - |
| 4,096 | 981,606 | 999,094 | 1.94 | 1.53 | - |
| 8,192 | 959,953 | 966,275 | 1.98 | 1.55 | - |
| 16,384 | 930,151 | 950,424 | 2.05 | 1.55 | - |
| 32,768 | 972,914 | 977,736 | 1.96 | 1.56 | - |
| 65,536 | 1,024,283 | 1,024,534 | 1.86 | 1.57 | - |
| 131,072 | 1,003,370 | 1,032,737 | 1.90 | 1.58 | - |
| 262,144 | 1,014,344 | 1,021,237 | 1.88 | 1.58 | - |
| 524,288 | 1,021,623 | 1,033,733 | 1.86 | 1.59 | - |
| 1,048,576 | 1,051,247 | 1,061,180 | 1.81 | 1.59 | - |
| 2,097,152 | 1,064,440 | 1,074,628 | 1.79 | 1.59 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 476,584 | 0.95 | 0.00 | - | - |
| 2,048 | ok | 778,954 | 0.95 | 0.00 | - | - |
| 4,096 | ok | 1,032,876 | 0.95 | 0.00 | - | - |
| 8,192 | ok | 955,066 | 0.95 | 0.00 | - | - |
| 16,384 | ok | 964,429 | 0.95 | 0.00 | - | - |
| 32,768 | ok | 1,031,097 | 0.95 | 0.00 | - | - |
| 65,536 | ok | 949,827 | 0.95 | 0.00 | - | - |
| 131,072 | ok | 1,175,667 | 0.95 | 0.00 | - | - |
| 262,144 | ok | 1,109,309 | 1.04 | 0.00 | - | - |
| 524,288 | ok | 1,086,071 | 1.22 | 0.00 | - | - |
| 1,048,576 | ok | 999,577 | 1.57 | 0.00 | - | - |
| 2,097,152 | ok | 1,070,642 | 2.27 | 0.00 | - | - |
| 4,194,304 | ok | 1,118,499 | 2.98 | 0.00 | - | - |

### onnx: engine `onnx`, device `cpu`, threads `14`

Engine config: `{"engine": "onnx", "device": "cpu", "threads": 14, "providers": ["CPUExecutionProvider"]}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,758,762 | 1,815,187 | 1.08 | 1.47 | - |
| 2,048 | 2,197,666 | 2,228,830 | 0.87 | 1.49 | - |
| 4,096 | 2,392,214 | 2,721,748 | 0.80 | 1.51 | - |
| 8,192 | 2,399,920 | 2,627,061 | 0.79 | 1.52 | - |
| 16,384 | 2,357,424 | 2,412,191 | 0.81 | 1.57 | - |
| 32,768 | 1,876,362 | 1,959,909 | 1.02 | 1.67 | - |
| 65,536 | 1,871,393 | 2,059,725 | 1.02 | 1.86 | - |
| 131,072 | 1,921,700 | 2,027,328 | 0.99 | 2.25 | - |
| 262,144 | 2,020,240 | 2,083,643 | 0.94 | 3.01 | - |
| 524,288 | 1,965,901 | 2,064,948 | 0.97 | 4.59 | - |
| 1,048,576 | 2,039,004 | 2,041,888 | 0.93 | 8.56 | - |
| 2,097,152 | 2,128,749 | 2,132,273 | 0.89 | 14.96 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 1,577,204 | 14.36 | 0.00 | - | - |
| 2,048 | ok | 2,297,682 | 14.36 | 0.00 | - | - |
| 4,096 | ok | 2,660,029 | 14.36 | 0.00 | - | - |
| 8,192 | ok | 1,621,349 | 14.36 | 0.00 | - | - |
| 16,384 | ok | 2,971,660 | 14.36 | 0.00 | - | - |
| 32,768 | ok | 1,953,383 | 14.36 | 0.00 | - | - |
| 65,536 | ok | 1,515,293 | 14.36 | 0.00 | - | - |
| 131,072 | ok | 2,061,114 | 14.36 | 0.00 | - | - |
| 262,144 | ok | 2,133,732 | 14.45 | 0.00 | - | - |
| 524,288 | ok | 2,009,152 | 14.62 | 0.00 | - | - |
| 1,048,576 | ok | 2,157,970 | 14.62 | 0.00 | - | - |
| 2,097,152 | ok | 2,072,241 | 15.68 | 0.35 | - | - |
| 4,194,304 | ok | 1,671,283 | 23.96 | 7.57 | - | - |

### torch: engine `torch`, device `cpu`, threads `14`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": false, "threads": 14, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 977,937 | 992,758 | 1.95 | 1.47 | - |
| 2,048 | 1,083,365 | 1,085,774 | 1.76 | 1.51 | - |
| 4,096 | 1,430,501 | 1,435,669 | 1.33 | 1.57 | - |
| 8,192 | 1,519,292 | 1,535,420 | 1.25 | 1.69 | - |
| 16,384 | 3,084,296 | 3,139,445 | 0.62 | 1.71 | - |
| 32,768 | 2,185,403 | 2,195,306 | 0.87 | 1.76 | - |
| 65,536 | 2,335,419 | 2,337,370 | 0.82 | 1.87 | - |
| 131,072 | 2,439,084 | 2,443,784 | 0.78 | 2.08 | - |
| 262,144 | 2,478,014 | 2,493,141 | 0.77 | 2.50 | - |
| 524,288 | 2,535,106 | 2,566,065 | 0.75 | 3.37 | - |
| 1,048,576 | 2,095,328 | 2,104,419 | 0.91 | 5.06 | - |
| 2,097,152 | 2,041,651 | 2,079,832 | 0.93 | 6.46 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 572,814 | 2.75 | 0.00 | - | - |
| 2,048 | ok | 852,904 | 2.75 | 0.00 | - | - |
| 4,096 | ok | 1,037,849 | 2.75 | 0.00 | - | - |
| 8,192 | ok | 1,459,317 | 2.75 | 0.00 | - | - |
| 16,384 | ok | 2,505,502 | 2.75 | 0.00 | - | - |
| 32,768 | ok | 2,061,383 | 2.75 | 0.00 | - | - |
| 65,536 | ok | 2,262,576 | 2.75 | 0.00 | - | - |
| 131,072 | ok | 2,373,832 | 2.75 | 0.00 | - | - |
| 262,144 | ok | 2,387,969 | 2.84 | 0.09 | - | - |
| 524,288 | ok | 2,517,761 | 3.02 | 0.18 | - | - |
| 1,048,576 | ok | 1,891,327 | 5.05 | 2.03 | - | - |
| 2,097,152 | ok | 1,944,720 | 7.43 | 3.36 | - | - |
| 4,194,304 | ok | 1,951,155 | 11.52 | 6.71 | - | - |

### torch-compile-cpu: engine `torch`, device `cpu`, threads `14`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": true, "threads": 14, "pinned": false}`

**Run failed:** `InductorError: CppCompileError: C++ compile error`

### keras: engine `keras`, device `cpu`, threads `14`

Engine config: `{"engine": "keras", "device": "cpu"}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 611,935 | 621,290 | 3.11 | 3.02 | - |
| 2,048 | 715,178 | 726,097 | 2.66 | 3.07 | - |
| 4,096 | 1,046,876 | 1,048,345 | 1.82 | 3.13 | - |
| 8,192 | 1,167,218 | 1,175,331 | 1.63 | 3.25 | - |
| 16,384 | 1,321,268 | 1,326,976 | 1.44 | 3.31 | - |
| 32,768 | 1,442,778 | 1,455,959 | 1.32 | 3.33 | - |
| 65,536 | 1,444,156 | 1,499,342 | 1.32 | 3.49 | - |
| 131,072 | 1,677,119 | 1,700,120 | 1.14 | 3.81 | - |
| 262,144 | 1,573,246 | 1,609,050 | 1.21 | 4.53 | - |
| 524,288 | 1,597,243 | 1,654,273 | 1.19 | 5.97 | - |
| 1,048,576 | 1,430,317 | 1,433,637 | 1.33 | 8.30 | - |
| 2,097,152 | 1,391,332 | 1,438,314 | 1.37 | 10.95 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 217,591 | 4.35 | 0.00 | - | - |
| 2,048 | ok | 462,033 | 4.35 | 0.00 | - | - |
| 4,096 | ok | 593,534 | 4.35 | 0.00 | - | - |
| 8,192 | ok | 916,887 | 4.35 | 0.00 | - | - |
| 16,384 | ok | 992,143 | 4.39 | 0.04 | - | - |
| 32,768 | ok | 1,307,395 | 4.39 | 0.00 | - | - |
| 65,536 | ok | 1,289,625 | 4.39 | 0.00 | - | - |
| 131,072 | ok | 1,513,579 | 4.44 | 0.04 | - | - |
| 262,144 | ok | 1,533,813 | 4.53 | 0.09 | - | - |
| 524,288 | ok | 1,627,066 | 4.71 | 0.19 | - | - |
| 1,048,576 | ok | 1,398,529 | 7.58 | 2.88 | - | - |
| 2,097,152 | ok | 1,395,285 | 12.25 | 6.46 | - | - |
| 4,194,304 | ok | 1,360,433 | 19.41 | 12.90 | - | - |

### torch: engine `torch`, device `mps`, threads `14`

Engine config: `{"engine": "torch", "device": "mps", "dtype": "float32", "tf32": false, "compile": false, "threads": 14, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,178,167 | 1,178,379 | 1.62 | 1.60 | - |
| 2,048 | 2,025,280 | 2,078,958 | 0.94 | 1.62 | - |
| 4,096 | 2,976,171 | 2,982,774 | 0.64 | 1.63 | - |
| 8,192 | 4,625,642 | 4,650,370 | 0.41 | 1.64 | - |
| 16,384 | 6,164,723 | 6,323,617 | 0.31 | 1.64 | - |
| 32,768 | 6,673,978 | 6,731,765 | 0.29 | 1.66 | - |
| 65,536 | 4,675,422 | 4,764,314 | 0.41 | 1.66 | - |
| 131,072 | 4,913,606 | 5,054,337 | 0.39 | 1.67 | - |
| 262,144 | 5,219,945 | 5,256,163 | 0.36 | 1.68 | - |
| 524,288 | 5,374,266 | 5,423,440 | 0.35 | 1.71 | - |
| 1,048,576 | 5,324,185 | 5,378,478 | 0.36 | 1.74 | - |
| 2,097,152 | 5,462,367 | 5,509,366 | 0.35 | 1.78 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 340,115 | 1.12 | 0.00 | - | - |
| 2,048 | ok | 937,014 | 1.12 | 0.00 | - | - |
| 4,096 | ok | 1,313,452 | 1.12 | 0.00 | - | - |
| 8,192 | ok | 1,729,958 | 1.12 | 0.00 | - | - |
| 16,384 | ok | 2,110,456 | 1.12 | 0.00 | - | - |
| 32,768 | ok | 1,768,642 | 1.12 | 0.00 | - | - |
| 65,536 | ok | 2,368,243 | 1.12 | 0.00 | - | - |
| 131,072 | ok | 4,638,869 | 1.12 | 0.00 | - | - |
| 262,144 | ok | 4,995,975 | 1.21 | 0.00 | - | - |
| 524,288 | ok | 5,269,142 | 1.38 | 0.00 | - | - |
| 1,048,576 | ok | 4,987,230 | 1.74 | 0.01 | - | - |
| 2,097,152 | ok | 4,816,895 | 2.48 | 0.03 | - | - |
| 4,194,304 | ok | 4,708,127 | 3.26 | 0.08 | - | - |

### torch-mps-fp16: engine `torch`, device `mps`, threads `14`

Engine config: `{"engine": "torch", "device": "mps", "dtype": "float16", "tf32": false, "compile": false, "threads": 14, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,120,113 | 1,132,353 | 1.70 | 1.60 | - |
| 2,048 | 1,930,673 | 1,985,846 | 0.99 | 1.60 | - |
| 4,096 | 2,711,971 | 3,184,484 | 0.70 | 1.62 | - |
| 8,192 | 4,809,564 | 4,835,934 | 0.40 | 1.62 | - |
| 16,384 | 6,975,647 | 7,030,227 | 0.27 | 1.62 | - |
| 32,768 | 9,024,189 | 9,090,720 | 0.21 | 1.62 | - |
| 65,536 | 6,529,994 | 6,579,894 | 0.29 | 1.63 | - |
| 131,072 | 6,781,966 | 6,786,332 | 0.28 | 1.65 | - |
| 262,144 | 6,986,213 | 7,035,151 | 0.27 | 1.65 | - |
| 524,288 | 7,170,554 | 7,204,967 | 0.27 | 1.70 | - |
| 1,048,576 | 7,180,628 | 7,229,710 | 0.27 | 1.73 | - |
| 2,097,152 | 7,338,225 | 7,431,952 | 0.26 | 1.78 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **4,194,304** rows; stopped by: row cap.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 536,336 | 1.12 | 0.00 | - | - |
| 2,048 | ok | 858,520 | 1.12 | 0.00 | - | - |
| 4,096 | ok | 492,517 | 1.12 | 0.00 | - | - |
| 8,192 | ok | 838,886 | 1.12 | 0.00 | - | - |
| 16,384 | ok | 1,648,822 | 1.12 | 0.00 | - | - |
| 32,768 | ok | 2,674,066 | 1.12 | 0.00 | - | - |
| 65,536 | ok | 2,667,539 | 1.12 | 0.00 | - | - |
| 131,072 | ok | 4,200,554 | 1.12 | 0.00 | - | - |
| 262,144 | ok | 6,742,257 | 1.21 | 0.00 | - | - |
| 524,288 | ok | 7,083,194 | 1.39 | 0.01 | - | - |
| 1,048,576 | ok | 6,855,817 | 1.74 | 0.00 | - | - |
| 2,097,152 | ok | 7,070,482 | 2.49 | 0.04 | - | - |
| 4,194,304 | ok | 6,917,433 | 3.27 | 0.08 | - | - |

### keras: engine `keras`, device `mps`, threads `14`

Engine config: `{"engine": "keras", "device": "mps"}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 662,816 | 680,037 | 2.87 | 2.91 | - |
| 2,048 | 1,211,304 | 1,217,406 | 1.57 | 2.92 | - |
| 4,096 | 1,526,707 | 1,536,513 | 1.25 | 2.96 | - |
| 8,192 | 2,130,325 | 2,202,579 | 0.89 | 2.97 | - |
| 16,384 | 2,745,951 | 2,772,467 | 0.69 | 3.06 | - |
| 32,768 | 3,002,657 | 3,042,854 | 0.63 | 3.08 | - |
| 65,536 | 2,661,984 | 2,680,427 | 0.72 | 3.10 | - |
| 131,072 | 2,684,177 | 2,691,483 | 0.71 | 3.21 | - |
| 262,144 | 2,517,873 | 2,557,238 | 0.76 | 3.48 | - |
| 524,288 | 2,318,789 | 2,472,844 | 0.82 | 4.07 | - |
| 1,048,576 | 2,203,386 | 2,207,373 | 0.86 | 5.15 | - |
| 2,097,152 | 1,935,177 | 1,946,143 | 0.98 | 5.87 | - |

Max single-pass batch (doubling from 2^10, cap 4,194,304 rows): largest successful = **2,097,152** rows; stopped by: SKIPPED: estimated 11.6 GB > 60% of 16.3 GB free RAM.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 256,101 | 3.25 | 0.00 | - | - |
| 2,048 | ok | 463,860 | 3.25 | 0.00 | - | - |
| 4,096 | ok | 348,909 | 3.25 | 0.00 | - | - |
| 8,192 | ok | 681,817 | 3.25 | 0.00 | - | - |
| 16,384 | ok | 302,057 | 3.25 | 0.00 | - | - |
| 32,768 | ok | 1,078,241 | 3.25 | 0.00 | - | - |
| 65,536 | ok | 1,861,223 | 3.27 | 0.02 | - | - |
| 131,072 | ok | 2,037,845 | 3.31 | 0.04 | - | - |
| 262,144 | ok | 1,997,216 | 3.41 | 0.09 | - | - |
| 524,288 | ok | 2,014,999 | 3.59 | 0.19 | - | - |
| 1,048,576 | ok | 1,982,649 | 3.96 | 0.38 | - | - |
| 2,097,152 | ok | 1,869,198 | 6.81 | 2.18 | - | - |
| 4,194,304 | SKIPPED: estimated 11.6 GB > 60% of 16.3 GB free RAM | - | - | - | - | - |

