# HRRR ptype run stores: layout, browser decodability and compression (local, one real hour)

Measured on 2026-11-16 with `benchmarks/make_sample_hour.py`, `benchmarks/compression_benchmark.py` and
`ptype/hrrr_fast/writers.py`. Machine: Apple M4 Pro, 14 cores, macOS 26.7.2. Other agents were running on
the machine at the same time (load average 3.4-8.0 during these runs), so timings are indicative only;
byte counts and errors are exact. Python 3.12.12, numpy 2.5.5, numba 0.68.1, zarr 3.4.2, numcodecs 0.17.1,
cast-value-rs 0.5.1, topozarr 0.3.0, xarray 2026.9.0, pyproj 3.8.0. Browser side: the web project's installed
zarrita 0.7.5, @zarrita/storage 0.2.0 and numcodecs (JS) 0.3.2, run under Node v25.2.1.

Everything below is from one forecast hour (hrrr.20231226 t00z f12). **It is not necessarily representative**:
precipitation covered 23.2 % of the grid in this winter case, and the size of every field that varies with
the weather (probabilities, flags, precipitation rate, soundings) depends on that. The extrapolations are
arithmetic on this one hour.

## 1. Sample hour (`make_sample_hour.py`)

Input: the reader agent's cached byte ranges (`reader_data/f12_n30.pkl`, 30 levels), re-planned to N = 27
hybrid levels. Stages, wall seconds (run 1 = numba cache cold for this venv, run 2 = warm):

| stage | run 1 | run 2 |
|---|---|---|
| load cached messages (pickle) | 0.046 | 0.045 |
| decode, ecCodes, 8 threads | 0.489 | 0.448 |
| grid lon/lat (pyproj inverse of the reader's grid) | 0.186 | 0.148 |
| heights AGL (`HGT - HGT_sfc`) | 0.020 | 0.014 |
| `columns.build_soundings` (2 m T/Td, 10 m wind, p_sfc at 0 m; grid-relative) | 2.658 | 0.257 |
| `columns.network_inputs` | 0.156 | 0.016 |
| engine `numpy` predict, 1,905,141 columns | 1.173 | 0.761 |
| rotate sounding winds to earth-relative | 0.843 | 0.442 |
| save `.npy` | 0.735 | 0.663 |

Sanity (identical in both runs): probabilities sum to 1 within 2.4e-7; probabilities 0.00048-0.99857;
u 0.0019-0.967 (mean 0.059); 0 columns whose top level is below 5000 m AGL; no NaN in the sounding;
rotation changes wind speed by at most 3.8e-6 m/s. `precip_mask` (any of CRAIN/CSNOW/CICEP/CFRZR == 1):
441,326 columns = 23.17 % of the grid. Inside the mask:

| | rain | snow | icep | frzr |
|---|---|---|---|---|
| ML dominant class (argmax) | 63.5 % | 31.3 % | 0.51 % | 4.6 % |
| HRRR flag set (flags can overlap) | 65.5 % | 26.1 % | 1.7 % | 9.0 % |

Row order: the reader returns HRRR in native scan order with `jScansPositively = 1`, i.e. row 0 is the
southern edge, so `build_soundings` column `c = j * nx + i` is already `location = j * ni + i` with j south to
north; no flip is needed (the script refuses north-first input).

## 2. What zarrita 0.7.5 (the site's version) can decode, from its source

`web/node_modules/zarrita/src/codecs.ts`, `createDefaultRegistry()`:

| codec name in metadata | where | notes |
|---|---|---|
| `zstd`, `numcodecs.zstd` | `import("numcodecs/zstd")` (numcodecs JS 0.3.2, WASM, `dist/zstd.js`) | |
| `blosc`, `numcodecs.blosc` | `import("numcodecs/blosc")` | `dist/blosc.js`: compressors `blosclz, lz4, lz4hc, snappy, zlib, zstd`; shuffle is read from the blosc header when decoding |
| `gzip`, `zlib` (and `numcodecs.` forms) | `src/codecs/gzip.ts`, `zlib.ts` | |
| `lz4`, `numcodecs.lz4` | numcodecs JS | |
| `bytes`, `transpose`, `crc32c`, `vlen-utf8`, `json2` | `src/codecs/*.ts` | |
| `scale_offset` | `src/codecs/scale_offset.ts` | zarr-extensions spec; decode `x / scale + offset`, keeps the dtype |
| `cast_value` | `src/codecs/cast_value.ts` | zarr-extensions spec incl. `scalar_map` (used here for NaN) |
| `bitround`, `numcodecs.bitround` | `src/codecs/bitround.ts` | decode is a no-op (the rounding is already in the data) |
| `numcodecs.shuffle`, `numcodecs.delta` | `src/codecs/shuffle.ts`, `delta.ts` | v2-compat names |
| `sharding_indexed` | `src/codecs/sharding.ts`, `src/util.ts:400` | needs a store with `getRange`; reads the index as a SUFFIX of `16 * n_inner` bytes plus what the index codecs add (crc32c: +4), i.e. `index_location = "end"` only; then one range read per inner chunk; index cached per shard per opened array |

Not in the registry: `numcodecs.fixedscaleoffset`, `numcodecs.quantize`, `numcodecs.packbits` (and any other
numcodecs filter) — so scaled integers must use `scale_offset` + `cast_value`, which zarr-python 3.4.2 writes
(it needs the `cast-value-rs` package).
`@zarrita/storage/src/fetch.ts`: `FetchStore.getRange` with a suffix issues a `HEAD` (for `Content-Length`)
and then a `Range: bytes=start-end` GET, unless the store is created with `useSuffixRequest: true`, in which
case it sends `Range: bytes=-N`.

**Node read test (actually run).** A logging HTTP server with Range/HEAD support served the stores; a
throwaway script imported the web project's installed `zarrita` and did what `src/data/ptype.ts` does (open
root, take `multiscales.layout[level]`, read `forecast_hour`, read the 5 ML arrays at one hour) and what
`src/data/soundings.ts` does (read `[fh, location, null]`). Results compared with zarr-python reads of the
same stores:

* Gridded, all 13 encodings of section 4 (none, zstd 1/3/9, blosc-zstd shuffle/bitshuffle, gzip, uint8 and
  uint16 `scale_offset`+`cast_value`, bitround, pre-rounded float32): every one decoded to a `Float32Array`
  equal to Python's values (mean within 1e-6, sample value exact). 13 requests per frame (root, level
  `forecast_hour` doc + chunk, 5 x (array doc + chunk)). A NaN written into a uint8-quantised array came back
  as NaN (179,900 of 179,900 at level 0; 44,950 at level 1), and an unwritten hour read as all-NaN (the chunk
  GETs return 404).
* Sounding, all 21 encodings of section 5, sharded: every one decoded bit-identical to Python for two
  locations, with both `FetchStore` modes. Requests for the first click: root doc, `sounding` doc,
  `forecast_hour` doc + chunk, then `HEAD` + index range (29,780 B) + one inner-chunk range; with
  `useSuffixRequest` the `HEAD` disappears (`Range: bytes=-29780`). A second click in the same hour is ONE
  range request (index cached). In a 19-hour store with one hour written, reading another hour gave all-NaN
  (HEAD 404 -> missing chunk).

## 3. Layout chosen

**Gridded store** (one per run, `hrrr_ptype_YYYYMMDD_HHMM.zarr`): metadata written by `writers.py` because
topozarr 0.3.0 writes a whole dataset in one call (no per-hour region writes), uses one resampling method for
all variables, and its default shards at coarse levels span forecast hours (`(2, 264, 225)` at level 2 in a
reference store written here), which would make hours share objects. Equivalence was verified by writing a
topozarr store for the same grid and diffing: root attributes identical on every topozarr key
(`zarr_conventions`, `proj:code`, `proj:wkt2`, `spatial:*`, `multiscales`) — the only extra root key is
`hrrr_ptype`; level-group attributes identical except the extra `grid` block (which
`tools/make_sample_data.py` also adds); `x`/`y` coordinates of all 4 levels identical (max |diff| 0). This is
also a unit test (`test_metadata_identical_to_topozarr`).

Levels 3/6/12/24 km (1059x1799, 529x899, 264x449, 132x224). The site's minimum zoom is 3
(`src/map/MapView.tsx`), where `chooseLevel` wants dx <= 2 px, i.e. about 25-35 km over CONUS latitudes, so a
48 km level would not be used.

Chunk choice — measured with the uint8 encoding, 5 ML arrays, one hour (`--sections chunks`); the site reads a
whole level of one hour for each of 5 arrays:

| chunk (cy x cx) | L0 GETs / bytes | L1 GETs / bytes | L2 GETs / bytes | L3 GETs / bytes |
|---|---|---|---|---|
| 1059x1799 (whole level) | 5 / 2,258,641 | 5 / 643,471 | 5 / 184,258 | 5 / 52,575 |
| 530x900 | 20 / 2,241,305 | 5 / 643,471 | 5 / 184,258 | 5 / 52,575 |
| 512x512 | 60 / 2,225,748 | 20 / 656,295 | 5 / 184,258 | 5 / 52,575 |
| 353x360 (topozarr's inner chunk) | 75 / 2,229,670 | 30 / 654,980 | 10 / 189,061 | 5 / 52,575 |
| 256x256 | 200 / 2,215,152 | 60 / 650,993 | 20 / 190,752 | 5 / 52,575 |

(GETs count data chunks only; each array also costs one `zarr.json` GET because `readVar` re-opens it.)
Bytes differ by < 2 % between chunkings, requests by up to 40x. Chosen: whole-level chunks `(1, ny_l, nx_l)`,
no sharding (sharding would ADD a HEAD + index read per array and save nothing, since there is already one
object per array per level per hour). Trade-off: if the site later reads only the viewport, smaller chunks
would help at level 0 only (arithmetic with `chooseLevel` at 38 N: level 0 is chosen from zoom ~5.4; at
zoom 6 a 1400-px-wide map spans ~2,700 km of the 5,400 km wide domain).

Per-variable aggregation to coarser levels (cascading 2x2 like topozarr, recorded in each array's
`resampling_method` attribute): `mean` for probabilities, `ML_u` and `prate`; `max` ("any") for
`precip_mask` and the four HRRR flags (keeps 3 km freezing-rain/ice-pellet bands that `mode`/`nearest` would
erase at 24 km, and never blanks a coarse cell that contains precipitation; cost: flagged areas widen by up to
one coarse cell); `mode` available for class codes (e.g. a Bourgouin class).

**Sounding store** (one per run): `sounding (forecast_hour, location, variable)` float32, inner chunk
`(1, 1024, 105)`, shard `(1, 1861 x 1024, 105)` = one object per hour (index 1861 x 16 + 4 = 29,780 B). Without
sharding an hour is 1,861 objects (1,861 x 48,960 = 91,117,560 objects for the archive vs 48,960 with
shards). Root attributes: the site's `grid` block, `heights_m`, `variable_groups`, plus `wind_frame:
"earth_relative"` and `wind_rotation` (lon_0 -97.5, lat_std 38.5, cone constant, formulas both ways).

**Concurrency.** Every chunk/shard has extent 1 along `forecast_hour`, `write_*_hour` always writes whole
chunks of one hour (no read-modify-write) and never rewrites a `zarr.json`; all metadata and coordinates are
written once by `create_*_store`. So different hours touch disjoint object keys; zarr's `LocalStore` writes
via a temp file + rename and S3 PUTs are atomic per key. Tested: 4 spawned processes writing 8 interleaved
hours of both stores, every hour verified, metadata bytes unchanged, no leftover `.partial` files. (S3 itself
was not tested.)

## 4. Gridded fields, one hour, all 4 levels (`--sections grid`)

Bytes = sum of chunk objects (coordinates excluded); read = Python, level 0 of the 5 arrays (best of 3).

| group | encoding | bytes all levels | bytes L0 | write s | read L0 s | max abs err | mean abs err |
|---|---|---|---|---|---|---|---|
| ML (5) | none | 50,576,320 | 38,102,820 | 0.139 | 0.0093 | 0 | 0 |
| ML (5) | zstd 1 | 44,563,521 | 33,545,427 | 0.196 | 0.036 | 0 | 0 |
| ML (5) | zstd 3 | 44,292,465 | 33,326,797 | 0.207 | 0.036 | 0 | 0 |
| ML (5) | zstd 9 | 43,933,695 | 33,017,840 | 0.255 | 0.037 | 0 | 0 |
| ML (5) | blosc-zstd5 byte shuffle | 35,222,854 | 26,292,406 | 0.370 | 0.028 | 0 | 0 |
| ML (5) | blosc-zstd5 bit shuffle | 34,763,323 | 25,900,132 | 0.354 | 0.036 | 0 | 0 |
| ML (5) | gzip 6 | 44,250,792 | 33,308,222 | 1.468 | 0.132 | 0 | 0 |
| ML (5) | **uint8 step 1/254 + zstd 3** | **3,138,945** | **2,258,641** | 0.183 | 0.023 | 0.00197 | 0.00102 |
| ML (5) | uint8 step 1/254 + zstd 9 | 2,960,203 | 2,123,300 | 0.255 | 0.022 | 0.00197 | 0.00102 |
| ML (5) | uint16 step 0.001 + zstd 3 | 6,114,647 | 4,437,945 | 0.216 | 0.031 | 0.0005 | 0.00025 |
| ML (5) | uint16 step 0.001 + blosc bit shuffle | 5,152,255 | 3,695,570 | 0.320 | 0.035 | 0.0005 | 0.00025 |
| ML (5) | bitround keepbits 9 + zstd 3 | 19,455,429 | 14,391,652 | 0.343 | 0.042 | 0.00049 | 0.000056 |
| ML (5) | float32 rounded to 0.001 + zstd 3 | 9,849,893 | 6,032,229 | 0.220 | 0.027 | 0.0005 | 0.00025 |
| prate | none | 10,115,264 | 7,620,564 | 0.027 | 0.0021 | 0 | 0 |
| prate | zstd 1 / **3** / 9 | 276,829 / **259,571** / 222,201 | 146,379 / 138,659 / 113,700 | 0.030 / 0.030 / 0.044 | 0.003 | 0 | 0 |
| prate | blosc-zstd5 shuffle / bitshuffle | 441,475 / 533,796 | 240,547 / 236,629 | 0.072 / 0.076 | 0.005 | 0 | 0 |
| flags (5, uint8) | none | 12,644,080 | 9,525,705 | 0.124 | 0.0050 | 0 | 0 |
| flags (5, uint8) | zstd 1 / **3** / 9 | 223,470 / **227,463** / 179,292 | 144,940 / 147,676 / 116,265 | 0.115 / 0.114 / 0.138 | 0.006 | 0 | 0 |
| flags (5, uint8) | gzip 6 | 201,777 | 132,963 | 0.171 | 0.016 | 0 | 0 |

"uint8 step 1/254" stores `round(p * 254)` (255 = NaN) behind a float32 array, so the site still gets
`Float32Array` probabilities; 254 (not 255) steps because one code is reserved for NaN.
Lossless float32 hardly compresses (the low mantissa bits of network outputs are noise).

Values outside `precip_mask` NOT blanked (as required) vs a hypothetical blanked store (`--sections mask`,
5 ML arrays, all levels): float32+zstd3 44,292,465 vs 11,124,590 B (4.0x); uint8+zstd3 3,138,945 vs
1,143,822 B (2.7x); uint16+zstd3 6,114,647 vs 1,935,578 B (3.2x). For information only.

## 5. Soundings, one hour (`--sections sounding`), 1,905,141 x 105 float32 = 800 MB raw

Read hour = whole hour in Python; read 1 location = one inner chunk. Network effect: decoded soundings,
winds rotated back to grid-relative, numpy engine, compared with the same pipeline on the lossless store
(dominant-class change over all 1,905,141 columns and inside the 441,326 precip columns). Pressure is not a
network input, so it does not affect the network columns.

| encoding | bytes | write s | read hour s | max err T / Td / u / v / p | max abs dP | mean abs dP | class change all / in mask |
|---|---|---|---|---|---|---|---|
| none | 800,408,673 | 0.76 | 0.25 | 0 | | | |
| zstd 1 / 3 / 9 | 741,751,258 / 738,203,771 / 736,788,903 | 0.92 / 1.04 / 1.47 | 0.36 / 0.39 / 0.37 | 0 | | | |
| blosc-zstd5 shuffle / bitshuffle | 583,656,586 / 653,244,222 | 1.14 / 1.15 | 0.46 / 0.48 | 0 | | | |
| gzip 6 | 728,911,901 | 3.15 | 1.73 | 0 | | | |
| transpose + zstd 3 | 723,523,602 | 1.20 | 0.61 | 0 | | | |
| transpose + blosc shuffle / bitshuffle | 560,676,162 / 541,457,332 | 1.32 / 1.23 | 0.58 / 0.65 | 0 | | | |
| int16 step 0.1 + zstd 3 | 217,521,952 | 1.44 | 0.79 | 0.05 / 0.05 / 0.05 / 0.05 / 0.05 | 0.106 | 2.8e-4 | 0.060 % / 0.106 % |
| int16 step 0.1 + transpose + zstd 3 | 185,737,507 | 1.63 | 0.98 | same | same | same | same |
| int16 step 0.1 + transpose + blosc shuffle | 145,135,658 | 1.36 | 0.94 | same | same | same | same |
| int16 step 0.1 + transpose + blosc bitshuffle | 134,062,757 | 1.45 | 0.99 | same | same | same | same |
| int32 step 0.01 + transpose + zstd 3 | 350,647,106 | 1.74 | 1.04 | 0.005 (all) | 0.0087 | 2.8e-5 | 0.0069 % / 0.0113 % |
| **int32 step 0.01 + transpose + blosc bitshuffle** | **219,660,299** | 1.64 | 1.03 | 0.005 (all) | 0.0087 | 2.8e-5 | 0.0069 % / 0.0113 % |
| float32 rounded (T, Td, wind 0.01; p 0.1) + transpose + zstd 3 | 410,750,427 | 1.35 | 0.58 | 0.005 x4 / 0.05 | 0.0087 | 2.8e-5 | 0.0069 % / 0.0113 % |
| float32 rounded (all 0.1) + transpose + zstd 3 | 236,526,305 | 1.24 | 0.56 | 0.05 (all) | 0.106 | 2.8e-4 | 0.060 % / 0.106 % |
| bitround keep 12 + transpose + zstd 3 | 444,887,813 | 1.67 | 0.66 | 0.0039 / 0.0078 / 0.0039 / 0.0039 / 0.10 | 0.00089 | 3.6e-6 | 0.00068 % / 0.00091 % |
| bitround keep 12 + transpose + blosc bitshuffle | 268,193,801 | 1.20 | 0.61 | same | same | same | same |
| bitround keep 10 + transpose + blosc bitshuffle | 219,527,641 | 1.20 | 0.67 | 0.016 / 0.031 / 0.016 / 0.016 / 0.5 | 0.0030 | 1.4e-5 | 0.0028 % / 0.0025 % |

Class changes as counts: int16 0.1: 1,137 columns; int32 0.01: 132; bitround 12: 13.
One `scale_offset` scale serves the whole array (it takes a scalar), so integer storage is either "0.1 in
every unit" (p x 10 fits int16) or "0.01 in every unit" (p x 100 needs int32). The transpose codec stores each
`(1, 1024, 105)` chunk variable-major so neighbouring bytes are neighbouring grid columns.
Bytes per click with the chosen encoding (Node log): index 29,780 B once per hour, then ~116-123 kB per click.

## 6. Default encodings: one hour and extrapolation (`--sections defaults`)

Defaults: ML_* uint8 1/254 + zstd 3; flags uint8 + zstd 3; prate float32 + zstd 3; sounding int32 0.01 +
transpose + blosc-zstd5 bitshuffle, sharded. Measured one hour: gridded 3,625,979 B (56 chunk objects incl.
coordinates), sounding 219,660,286 B (1 shard); write 0.32 s (gridded, 11 arrays x 4 levels) and 1.27 s
(sounding); per-store metadata 84,859 B in 69 documents. Arithmetic, ASSUMING this hour is representative:

| scope | hours | gridded | soundings | total |
|---|---|---|---|---|
| one hour (measured) | 1 | 3.63 MB | 219.7 MB | 223.3 MB |
| 18 h run (f00-f18) | 19 | 68.9 MB | 4.17 GB | 4.24 GB |
| 48 h run (f00-f48) | 49 | 177.7 MB | 10.76 GB | 10.94 GB |
| archive, 85 days x 576 h | 48,960 | 177.5 GB | 10.75 TB | 10.93 TB |

Same arithmetic for other sounding encodings (gridded defaults included): int16 0.1 + transpose + bitshuffle
137,688,736 B/h -> 2.62 GB per 18 h run, 6.74 TB archive; bitround 12 + bitshuffle 13.31 TB; lossless float32
transpose + bitshuffle 26.69 TB; lossless float32 zstd 3 36.32 TB.

## 7. Recommendation

Facts (above): soundings are ~98 % of the bytes. Lossless float32 soundings do not compress well (best
541 MB/hour). Quantising to 0.01 in every unit, transposing chunks and bit-shuffling gives 220 MB/hour with
max error 0.005 and 132 of 1.9 M columns changing dominant class; 0.1 gives 134 MB/hour, max error 0.05 and
1,137 columns changing class (max probability change 0.106). Every option measured is readable by zarrita
0.7.5 (Node test). Gridded uint8 probabilities are 14x smaller than float32 + zstd with max error 0.002.

Judgement:
* Gridded: uint8 (step 1/254) + zstd 3 for ML_*; error 0.002 is invisible in a colour-shaded map. If the
  probabilities are also used numerically downstream, uint16 0.001 + blosc bitshuffle costs +2 MB/hour.
* Soundings: int32 0.01 + transpose + blosc-zstd bitshuffle (the default). It keeps every value within 0.005
  of the float32 input, so a profile in the skew-T and a browser re-run of the network are essentially the
  stored ones, for 30 % of the lossless float32 + zstd 3 size (41 % of the best lossless option). Choose int16 0.1 if the archive budget matters more (-39 %,
  ~4.2 TB less over the 85 days) and a 0.1 % class change rate in precipitating columns is acceptable.
* zstd 3 rather than 9 for the gridded arrays: 9 is 5.7 % smaller for uint8 ML, 14 % for prate and 21 % for
  the flags (264,283 B/hour together, 7 % of the gridded hour, 0.1 % of the whole hour) at 1.2-1.5x the write
  time; either is defensible, the difference is negligible next to the soundings.

## 8. Things the site must change, and what was not verified

* `src/data/soundings.ts` opens ONE store with `(init_time, forecast_hour, location, variable)` at a fixed
  URL; the per-run store is `(forecast_hour, location, variable)` and needs a per-run URL (e.g. next to the
  gridded store) and indexing `[fi, location, null]`.
* Winds in the sounding store are EARTH-relative (the docs say grid-relative). `src/model/evidential.ts`
  feeds `profile.u`/`profile.v` to the model, which takes grid-relative winds: rotate back with the
  `wind_rotation` attributes (or keep grid-relative storage if that is simpler for the site).
* Sharded reads need `HEAD` with a readable `Content-Length` (CORS `Access-Control-Expose-Headers`) or
  `useSuffixRequest: true` (suffix `Range`). Only a local server was used; S3/CloudFront were not tested.
* `ptype.hrrr_fast.writers` needs `zarr>=3.4` and `cast-value-rs` (for `cast_value`); `pyproject.toml` was not
  changed by this work.
* Not verified: a real browser (Node uses the same zarrita code), concurrent writers on S3 (only LocalStore),
  other seasons/hours, timings on an idle machine.
