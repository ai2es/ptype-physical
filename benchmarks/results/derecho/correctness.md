# Correctness of the fast pipeline against the original, 2023-12-26 00 UTC f12

Both checks use the same HRRR run and forecast hour. Everything below was measured on Derecho by the commands at
the end; raw outputs are `correctness/network_correctness.json` and `correctness/compare_f12.json` in this directory.
Jobs: 8210874 (network check, then the new pipeline's f12) and 8210890 (comparison), one node each, repository commit
`e1e27fe` / `dde297a`.

## 1. Network on identical inputs (original Keras path vs new torch engine)

Input: `baseline/f12_inputs/input_matrix.npy`, 1,902,284 x 84, float64, **unscaled** (`baseline_run.py --save-inputs` dumps it
before `transform_data`; `input_columns.json` lists the names, T, Td, U, V at 0-5000 m; no NaN rows). The original's
predictions had not been saved (the netCDF only holds copies rounded to 4 digits), so they were recomputed: bridgescaler
`transform(DataFrame)` then `model.predict(x, return_uncertainties=True, batch_size=2048)` on the 20240909 `.keras` model
loaded with the wrapper's `custom_objects` fix, torch backend, CPU, 8 threads. The new path is
`make_engine("torch", weights, device="cpu")` fed the float32 unscaled rows (scaler folded into layer 0).
Probabilities and uncertainty are compared directly as float32 (no 1/254 quantisation involved; that is only in the Zarr
store, section 2).

| output | max abs difference | mean abs difference |
|---|---|---|
| P(rain) | 2.77e-06 | 2.51e-08 |
| P(snow) | 4.05e-06 | 3.00e-08 |
| P(ice pellets) | 3.25e-06 | 9.06e-09 |
| P(freezing rain) | 2.83e-06 | 1.04e-08 |
| all four probabilities | **4.05e-06** | 1.86e-08 |
| uncertainty u | **2.62e-06** | 1.89e-08 |

Dominant class (argmax of the four probabilities): **identical on all 1,902,284 rows** (0 disagreements; original class
counts rain 1,264,563 / snow 596,991 / ice pellets 9,062 / freezing rain 31,668). Probabilities sum to 1 within 1.5e-07 (original) and
1.4e-07 (new). Timing in this job, for orientation only: original path 56.8 s (model load, scaling and predict), new engine 2.5 s.

## 2. End to end: original (wrfprs, netCDF) vs new (wrfnat, Zarr), same hour

Original: `baseline/runs/smoke_f12/out/hrrr/20231226/0000/MILES_ptype_hrrr_2023-12-26_0000_f12.nc` (job 8207343, the run
that produced the input matrix above; 1058 x 1798). New: level 0 of `hrrr_ptype_20231226_0000.zarr`, forecast-hour slot 12,
from `hrrr_ptype_run.py` (1 worker x 8 threads, 0 level re-reads, 292.8 MB fetched).

**Grid matching.** The original is the lat/lon box `[-122.8,-60.9,21.0,47.9]` sliced with an exclusive end, so it is the
first 1058 rows and 1798 columns of the full 1059 x 1799 HRRR grid. This was not assumed: the original's latitude/longitude
were projected (pyproj, the store's `proj:code`) into the store's x/y axes and rounded, giving `iy` 0..1057 and `ix` 0..1797
as a contiguous block with unit steps (offset 0, 0). Verification: maximum distance between each original point and the
centre of its assigned cell 2.46 m (cell spacing 3000 m); projecting the assigned cell centres back to lat/lon differs
from the original's coordinates by at most 1.9e-06 deg in latitude and 2.3e-05 deg in longitude; first point
(21.13812, -122.71954) and last point (47.82795, -60.96909) agree between the two.

All numbers are absolute differences new minus original in probability units (0-1); n = 1,902,284 matched points. The
store keeps probabilities and `u` as uint8 steps of 1/254, so every comparison against it has a floor: passing the
**original's own** probabilities through that quantisation gives the "quantisation floor" rows (mean about 0.00105,
max 0.00197 = half a step). Differences at or below that size cannot be told apart from storage rounding.

### All matched points

| variable | mean | median | 95th | 99th | max |
|---|---|---|---|---|---|
| P(rain) | 0.00355 | 0.00125 | 0.01416 | 0.05623 | 0.79290 |
| P(snow) | 0.00431 | 0.00116 | 0.01849 | 0.06588 | 0.87448 |
| P(ice pellets) | 0.00207 | 0.00110 | 0.00364 | 0.02810 | 0.54105 |
| P(freezing rain) | 0.00218 | 0.00110 | 0.00394 | 0.03309 | 0.76262 |
| uncertainty u | 0.00260 | 0.00122 | 0.01006 | 0.02623 | 0.33242 |
| *quantisation floor, rain* | 0.00105 | 0.00107 | 0.00189 | 0.00195 | 0.00197 |
| *quantisation floor, snow / icep / frzr* | 0.00102 / 0.00103 / 0.00103 | 0.00101 / 0.00104 / 0.00104 | 0.00186 | 0.00195 | 0.00197 |

### Points where HRRR has precipitation (any of CRAIN/CSNOW/CICEP/CFRZR = 1): 441,326 points (23.2 %)

The new store's `precip_mask` (wrfnat) and the original's flags (wrfprs) mark exactly the same 441,326 points, and all
four categorical flags agree at every one of the 1,902,284 points.

| variable | mean | median | 95th | 99th | max |
|---|---|---|---|---|---|
| P(rain) | 0.00454 | 0.00128 | 0.01776 | 0.08571 | 0.79290 |
| P(snow) | 0.00545 | 0.00119 | 0.01914 | 0.10765 | 0.87448 |
| P(ice pellets) | 0.00269 | 0.00113 | 0.00543 | 0.04252 | 0.54105 |
| P(freezing rain) | 0.00366 | 0.00116 | 0.00961 | 0.07279 | 0.76262 |
| uncertainty u | 0.00297 | 0.00125 | 0.01185 | 0.03415 | 0.29273 |

### Dominant class

| set | points | same class | different class |
|---|---|---|---|
| all matched points | 1,902,284 | **99.522 %** (1,893,199) | 9,085 |
| HRRR-precipitation points | 441,326 | **99.271 %** (438,107) | 3,219 |
| original's own probabilities, quantised vs unquantised (the floor), all points | 1,902,284 | 99.986 % | 260 |
| same floor, precipitation points | 441,326 | 99.989 % | 50 |

Confusion tables of dominant class (rows original / wrfprs, columns new / wrfnat):

All 1,902,284 points

| original \ new | rain | snow | ice pellets | freezing rain |
|---|---|---|---|---|
| rain | 1,262,033 | 1,482 | 81 | 967 |
| snow | 3,267 | 592,058 | 1,018 | 649 |
| ice pellets | 158 | 514 | 8,072 | 319 |
| freezing rain | 124 | 277 | 229 | 31,036 |

HRRR-precipitation points (441,326)

| original \ new | rain | snow | ice pellets | freezing rain |
|---|---|---|---|---|
| rain | 279,734 | 455 | 15 | 766 |
| snow | 619 | 137,770 | 500 | 381 |
| ice pellets | 12 | 50 | 1,614 | 205 |
| freezing rain | 47 | 59 | 110 | 18,989 |

Dominant-class counts over all points, original vs new: rain 1,264,563 vs 1,265,582; snow 596,992 vs 594,331; ice pellets 9,063 vs
9,400; freezing rain 31,666 vs 32,971. The original counts here come from the netCDF (rounded to 4 digits), so they differ by
1-2 points from the recomputed ones in section 1.

The differences are not zero, as expected for different input levels and 0 m handling (see `docs/hrrr_fast.md`); this
document only measures them and does not attribute them further.

## Not verified / limits

- One forecast hour (f12 of one run), one winter case with 23 % precipitation coverage; no other hours, seasons or inits.
- Only the original's netCDF output (`smoke_f12`) was compared; `single_f12_A_default` and the scaling runs wrote f12 files from
  the same code and data but were not compared with each other here.
- The original netCDF is written with `least_significant_digit=4` (code in `ptype.inference.save_data`), so its probabilities
  are rounded to about 1e-4; this was not measured separately and is small against the 1/254 store step.
- The comparison is against the Zarr store (uint8 steps), not against the new engine's float32 output for wrfnat inputs, which the
  pipeline does not keep; the quantisation floor above bounds that part.
- Section 1 shows the network itself is equivalent; the end-to-end differences therefore come from the inputs and storage,
  but no experiment here separates the contributions (not attempted).

## Commands

```
# on Derecho, repo /glade/work/dgagne/ptype_fast/ptype-physical, after `git pull`; BL=/glade/derecho/scratch/dgagne/ptype_fast/baseline
qsub benchmarks/pbs/correctness.pbs          # job 8210874, runs:
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
python benchmarks/correctness_network.py --inputs-dir $BL/f12_inputs \
    --keras-path $PTYPE_FAST/model/ptype_model_20240909.keras --scaler-path $PTYPE_FAST/model/ptype_scaler_20240909.json \
    --weights $WEIGHTS/ptype_weights.npz --batch-size 2048 --threads 8 --out $SCRATCH_DIR/correctness/network
python applications/hrrr_ptype_run.py --config config/hrrr_fast.yml --init 2023122600 --forecast-hours 12 \
    --processes 1 --threads 8 --weights $WEIGHTS/ptype_weights.npz --out-dir $SCRATCH_DIR/correctness/new_f12 --no-upload \
    --summary-json $SCRATCH_DIR/correctness/new_f12_summary.json
qsub benchmarks/pbs/correctness_compare.pbs  # job 8210890, runs:
python benchmarks/compare_original_vs_new.py \
    --original-nc $BL/runs/smoke_f12/out/hrrr/20231226/0000/MILES_ptype_hrrr_2023-12-26_0000_f12.nc \
    --new-store $SCRATCH_DIR/correctness/new_f12/2023122600/hrrr_ptype_20231226_0000.zarr --forecast-hour 12 \
    --out $SCRATCH_DIR/correctness/compare_f12.json
```
