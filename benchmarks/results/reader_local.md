# HRRR wrfnat byte-range reader: local benchmark

Everything below was measured with `benchmarks/reader_benchmark.py` and `ptype/hrrr_fast/reader.py` on 2026-11-16 (UTC).

## Machine and software

- Apple M4 Pro, 14 cores, 48 GiB RAM (`sysctl`: `hw.ncpu=14`, `hw.memsize=51539607552`), macOS 26.7.2 (25G332).
- Home/office network to AWS us-east-1. Other agents were running on the same machine at the same time (the load average was 4.0 before the decode benchmark and 10.0 after it), so the timings are indicative, not clean.
- Python 3.12.12 (uv venv). numpy 2.5.5, obstore 0.12.1, eccodes (python) 2.50.1 with eccodeslib 2.50.0.34 (ecCodes 2.50.0), pygrib 2.1.8 (bundles ecCodes 2.44.0), gribberish 1.9.0, pyproj 3.8.0, pandas 3.0.3, torch 2.15.0, xarray 2026.9.0, pytest 9.2.0, ruff 0.17.5.

## Data

- Main file: `s3://noaa-hrrr-bdp-pds/hrrr.20231226/conus/hrrr.t00z.wrfnatf12.grib2`. It exists, so no fallback was needed. It holds 1136 messages and is 790,950,105 B (754.3 MiB). Its idx is 70,908 B.
- Warm-season file for the N study: `hrrr.20250715/conus/hrrr.t18z.wrfnatf06.grib2`, which holds 1136 messages and is 771,424,628 B.
- The f00, f01, f24 and f48 idx files of `hrrr.20231226` t00z were also inspected for the APCP and 2 m/10 m names. Those files hold 1133 messages, compared with 1136 at f12.

### idx names (exact strings, verified in wrfnat)

| field | idx variable:level | step at f12 / f00 |
|---|---|---|
| 2 m temperature | `TMP:2 m above ground` | `12 hour fcst` / `anl` |
| 2 m dewpoint | `DPT:2 m above ground` (also `SPFH:2 m above ground`, `RH:2 m above ground`, `POT:2 m above ground`) | same |
| 10 m wind | `UGRD:10 m above ground`, `VGRD:10 m above ground` | same |
| orography / surface pressure | `HGT:surface`, `PRES:surface` | same |
| precip-type flags | `CRAIN:surface`, `CSNOW:surface`, `CICEP:surface`, `CFRZR:surface` | same |
| precip rate | `PRATE:surface` | same |
| APCP f00 | `APCP:surface:0-0 day acc fcst` (the only APCP message) | |
| APCP f01 | `APCP:surface:0-1 hour acc fcst` (the only APCP message) | |
| APCP f12 | `0-12 hour acc fcst` **and** `11-12 hour acc fcst` | |
| APCP f24 / f48 | `0-1 day acc fcst` + `23-24 hour acc fcst` / `0-2 day acc fcst` + `47-48 hour acc fcst` | |

Hybrid levels: `N hybrid level`, N = 1..50, with 20 variables per level. Level 1 is nearest the ground: HGT increases with level everywhere (see the N table). The wrfnat file has no hybrid-level dewpoint or RH.

## 1. Fetch: idx plus one `obstore.get_ranges` call, N = 30 (192 messages)

Command:

```
python benchmarks/reader_benchmark.py fetch --date 2023-12-26 --cycle 0 --fxx 12 --n-levels 30 --repeat 2 --cache <scratch>/f12_n30.pkl
```

- idx fetch: 0.150 s and 0.155 s.
- Bytes needed: 271,698,718 B, which is 34.4 % of the file. Each hybrid 3D message is about 1.2–2.3 MB.
- Request counts: the benchmark replays the same `get_ranges` call against a local HTTP server that logs every `Range` header. The "logged" columns come from that server. They agree exactly with `reader.merged_ranges`, a Python mirror of object_store's merge rule.

| coalesce (B) | GET requests | bytes requested | % of file | fetch s (run 1, run 2) | median s | MiB/s |
|---|---|---|---|---|---|---|
| 0 | 67 | 271,698,718 | 34.4 | 18.75, 14.77 | 16.76 | 15.5 |
| 65,536 | 67 | 271,698,718 | 34.4 | 18.13, 15.86 | 17.00 | 15.2 |
| 1,048,576 (default) | 66 | 272,030,280 | 34.4 | 19.84, 17.62 | 18.73 | 13.9 |
| 4,194,304 | 6 | 442,474,046 | 55.9 | 31.50, 21.64 | 26.57 | 15.9 |
| 16,777,216 | 2 | 460,042,516 | 58.2 | 19.54, 35.09 | 27.32 | 16.1 |
| 67,108,864 | 2 | 460,042,516 | 58.2 | 26.98, 24.51 | 25.74 | 17.0 |

What this shows:

- With `coalesce=0` there are still only 67 requests, not 192. object_store merges ranges whose gap is ≤ `coalesce`, and a gap of 0 qualifies. On each level, HGT, TMP, SPFH, UGRD and VGRD are adjacent messages, so they form one request, and PRES is a second.
- On this link the fetch is bandwidth-bound at about 14–17 MiB/s whatever the request count. So the best setting is the one that transfers the fewest bytes: `coalesce` between 0 and 1 MiB. At 4 MiB and above, 60–70 % more bytes are pulled and the fetch is slower.
- On a high-bandwidth link inside AWS, request count may matter more. That was not measured.

## 2. Decode: 192 messages (N = 30), best of 3 wall seconds

Command:

```
python benchmarks/reader_benchmark.py decode --cache <scratch>/f12_n50.pkl --n-levels 30 --repeat 3
```

Each timing is for one `reader.decode(...)` call. Process pools include pool start-up (macOS spawn) and the cost of pickling about 1.4 MB in and 7.6 MB out per message.

| decoder | serial | thread 1 | thread 2 | thread 4 | thread 8 | process 1 | process 2 | process 4 | process 8 |
|---|---|---|---|---|---|---|---|---|---|
| eccodes | 2.03 | 2.11 | 1.03 | 0.53 | **0.32** | 2.41 | 4.68 | 2.52 | 2.43 |
| pygrib | 2.14 | 2.20 | 1.97 | 1.86 | 1.87 | 2.49 | 5.21 | 3.02 | 2.88 |
| gribberish | 2.97 | 3.29 | 3.02 | 3.06 | 3.90 | 3.61 | 4.17 | 2.87 | 3.84 |

How each decoder behaves with threads:

- **eccodes releases the GIL.** Its cffi calls drop the GIL, and threads scale almost linearly: 6.3× at 8 threads.
- **pygrib and gribberish do not release the GIL.** Neither gets faster with threads.
- Process pools lose for all three decoders because the IPC cost exceeds the decode cost.

The eccodes float32 API (`codes_get_float_array`) differs from float64-then-cast by up to 0.0078 (tested on 60 messages). The reader therefore decodes in float64 and casts, to stay bit-identical with pygrib.

## Decoder agreement (exactness)

Command:

```
python benchmarks/reader_benchmark.py compare --cache <scratch>/f12_n50.pkl
```

This covers all 312 messages of the N = 50 set from f12.

| field | shape | eccodes max abs diff vs pygrib | gribberish max abs diff vs pygrib | NaN masks equal |
|---|---|---|---|---|
| TMP, SPFH, PRES, HGT, UGRD, VGRD | (50, 1059, 1799) | 0 | 0 | yes (no missing values) |
| HGT_sfc, PRES_sfc, TMP_2m, DPT_2m, UGRD_10m, VGRD_10m, PRATE, APCP_1h, CRAIN, CSNOW, CICEP, CFRZR | (1059, 1799) | 0 | 0 | yes (no missing values) |

- In float64, before the float32 cast, eccodes is identical to pygrib. gribberish differs by at most 7.3e-12, which disappears in float32.
- All three decoders return the same native scan order: row 0 is the south edge, and the pygrib latitude mean is 23.28 on row 0 and 51.00 on the last row.
- gribberish 1.9.0 `parse_grib_array` returns a flat array. The reader reshapes it using `parse_grib_message_metadata(...).grid_shape`.
- gribberish's metadata reports `crs = EPSG:2154` for this grid, which is wrong (that is French Lambert-93). Its proj string is fine. Do not use the gribberish CRS.
- None of these messages has a bitmap, so the missing-value path was **not** exercised on real data.

### Grid metadata (read from the GRIB grid definition)

The grid is 1799 × 1059 at 3000 m spacing. The Lambert parameters are lon_0 = −97.5 (LoV 262.5), lat_0 = lat_1 = lat_2 = 38.5, and the earth is a sphere with R = 6,371,229 m (`shapeOfTheEarth = 6`). The first point is at 21.138123 N, −122.719528 E, which projects to x = −2,697,520.143 m, y = −1,587,306.153 m. The scan flags are iScansNegatively = 0, jScansPositively = 1, jPointsAreConsecutive = 0, so row 0 is south.

The reader's own spherical LCC math matches pyproj to 0.0000 m, both at the first point and over every grid point of `xy_coords()`.

## 3. Choosing N: height above ground of each hybrid level

Command:

```
python benchmarks/reader_benchmark.py agl
```

The two files are 2023-12-26 t00z f12 and 2025-07-15 t18z f06. AGL = HGT(level) − HGT(surface), with statistics over all 1,905,141 grid points.

**Smallest level whose minimum AGL is ≥ 5000 m:**

- Cold season: index 24, which is hybrid level 25 (min 5156 m).
- Warm season: index 23, which is hybrid level 24 (min 5368 m).

**N = 25 covers ≥ 5 km AGL everywhere in both files.**

| index | level | 20231226 f12 min / median / max (m) | 20250715 f06 min / median / max (m) |
|---|---|---|---|
| 0 | 1 | 9 / 10 / 11 | 10 / 11 / 12 |
| 1 | 2 | 33 / 38 / 41 | 36 / 41 / 43 |
| 2 | 3 | 74 / 86 / 93 | 82 / 93 / 98 |
| 3 | 4 | 141 / 165 / 179 | 157 / 178 / 189 |
| 4 | 5 | 235 / 280 / 304 | 263 / 302 / 321 |
| 5 | 6 | 353 / 431 / 467 | 397 / 463 / 492 |
| 6 | 7 | 488 / 613 / 662 | 550 / 655 / 698 |
| 7 | 8 | 634 / 820 / 884 | 715 / 874 / 932 |
| 8 | 9 | 787 / 1051 / 1134 | 889 / 1118 / 1192 |
| 9 | 10 | 950 / 1314 / 1417 | 1073 / 1398 / 1487 |
| 10 | 11 | 1123 / 1614 / 1740 | 1268 / 1718 / 1821 |
| 11 | 12 | 1305 / 1953 / 2105 | 1474 / 2077 / 2197 |
| 12 | 13 | 1497 / 2338 / 2517 | 1694 / 2481 / 2620 |
| 13 | 14 | 1703 / 2771 / 2983 | 1929 / 2936 / 3094 |
| 14 | 15 | 1928 / 3260 / 3507 | 2186 / 3448 / 3626 |
| 15 | 16 | 2180 / 3809 / 4092 | 2472 / 4024 / 4220 |
| 16 | 17 | 2463 / 4414 / 4735 | 2792 / 4655 / 4870 |
| 17 | 18 | 2769 / 5042 / 5410 | 3137 / 5310 / 5542 |
| 18 | 19 | 3087 / 5658 / 6067 | 3492 / 5952 / 6197 |
| 19 | 20 | 3414 / 6248 / 6701 | 3854 / 6568 / 6823 |
| 20 | 21 | 3749 / 6812 / 7307 | 4224 / 7158 / 7421 |
| 21 | 22 | 4093 / 7354 / 7885 | 4601 / 7726 / 7996 |
| 22 | 23 | 4443 / 7872 / 8437 | 4984 / 8269 / 8555 |
| 23 | 24 | 4797 / 8366 / 8962 | 5368 / 8788 / 9095 |
| 24 | 25 | 5156 / 8836 / 9460 | 5753 / 9283 / 9613 |
| 25 | 26 | 5516 / 9282 / 9933 | 6135 / 9752 / 10104 |
| 26 | 27 | 5878 / 9708 / 10384 | 6516 / 10201 / 10573 |
| 27 | 28 | 6244 / 10121 / 10820 | 6898 / 10634 / 11023 |
| 28 | 29 | 6612 / 10518 / 11239 | 7281 / 11049 / 11453 |
| 29 | 30 | 6990 / 10910 / 11650 | 7670 / 11455 / 11872 |
| 30–49 | 31–50 | 7396 … 23301 / 11316 … 27107 / 12070 … 27463 | 8082 … 23899 / 11872 … 27655 / 12301 … 28104 |

The full 50-row table is printed by the command above.

## 4. miles-credit `HRRRDataset` on the same forecast hour

The test used `git clone --depth 1 https://github.com/NCAR/miles-credit` at commit `396b8a4edc74ed594c5e1bd404a89258134d37ac` (2026-11-14). It was not pip-installed; the clone was added to `sys.path`. The extra packages it needed were torch, cftime, xarray, pyyaml and `pandas<3.0.4`; miles-credit pins `pandas<3.0.4`, and 3.1.1 had to be downgraded to 3.0.3. There were no other installation problems.

The minimum data_config was:

```python
{"source": {"HRRR_NAT": {"dataset_type": "hrrr", "product": "wrfnat", "mode": "remote", "forecast_hour": 12,
                         "levels": list(range(1, 31)),
                         "variables": {"prognostic": {"vars_3D": ["T", "U", "V", "Q", "P", "GH"],
                                                      "vars_2D": ["sp", "orog", "t2m", "d2m", "u10m", "v10m", "prate", "tp"]}}}},
 "start_datetime": "2023-12-26 00:00", "end_datetime": "2023-12-26 00:00", "timestep": "1h", "forecast_len": 0}
```

The sample was loaded with `ds[(pd.Timestamp("2023-12-26 00:00"), 0)]`.

### Results

**It works for the thermodynamic profile.**
- It returns `(30, 1, 1059, 1799)` tensors for T/U/V/Q/P/GH and `(1, 1, 1059, 1799)` tensors for the 2D fields.
- T is identical to this reader's TMP (max abs diff 0.0).
- Load time was 30.95 s and 26.29 s over two calls; the reader's fetch alone took about 15–19 s for the same bytes (section 1). These were single runs on a shared network, so this is not a careful comparison.
- The first `import credit.datasets.gen_2.hrrr` took 37.6 s (cold torch import).

**It cannot provide CRAIN, CSNOW, CICEP or CFRZR.**
- They are not in `VAR_REGISTRY` (`credit/datasets/gen_2/hrrr.py` lines 165–237).
- `_register_field` raises `KeyError: "Variable 'crain' is not in VAR_REGISTRY. ..."` (line 873–874). The same happens for `CRAIN`, `csnow`, `cicep` and `cfrzr`.
- They could be added by mutating `VAR_REGISTRY` at import time, as documented at lines 59–66; this was not tried.

**`tp` returns the wrong APCP window.**
- The 2D lookup matches only variable and level and takes `matching[0]` (lines 1020–1025), so `tp` gets the first APCP message, which is the run total `0-12 hour acc fcst`.
- This was verified: its `tp` equals the decoded 0–12 h message exactly (max diff 0.0). It differs from the 11–12 h bucket by up to 455.97 kg m⁻²; the max of `tp` is 455.97, while the max of the 1-hour APCP is 96.07.
- No configuration selects the 1-hour bucket.

**Performance notes from the code:**
- It calls `get_ranges` with the default coalesce (line 393).
- It decodes with `pygrib.fromstring` serially (line 1046) and then reads `.values` in an 8-thread pool (lines 1069–1080). Per section 2, pygrib does not release the GIL, so that pool gives no speedup.
- It computes `latlons()` once to build the spatial slice (line 1055).

**It has no grid metadata API** besides lat/lon arrays.

## Recommendation

| choice | value | evidence |
|---|---|---|
| decoder | **eccodes** | bit-identical to pygrib; fastest serially (2.03 s) and the only one that scales with threads |
| pool | **thread, 8 workers** | 0.32 s for 192 messages, versus 1.87 s for pygrib and 2.87 s for gribberish at their best; process pools 2.4–5.2 s |
| coalesce | **0 to 1 MiB** (0 or 64 KiB is minimal) | 66–67 requests and 272 MB; at ≥ 4 MiB, 442–460 MB and slower on a bandwidth-bound link |
| N | **25** | first level with min AGL ≥ 5 km in both files (cold 5156 m, warm 5753 m at level 25) |

For N = 30, end to end, the fetch (about 15–19 s here) dominates over decode (0.3 s). Using N = 25 instead of 30 cuts about 1/6 of the hybrid bytes; that saving was not separately measured.

## Not verified

- The missing-value/bitmap path, because no selected message had a bitmap.
- The local-directory (`LocalStore`) read path against a real mirrored file. It shares the same `get_ranges` code, and the benchmark exercised obstore against a local HTTP server, but no full wrfnat file was downloaded.
- Fetch performance from inside AWS.
- Process-pool decode with a persistent pool or shared memory.
