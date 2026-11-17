# Object sizes of a run, per-hour vs finalized layout, and S3 Standard-IA billed bytes

Measured on 2026-11-16 with `benchmarks/compression_benchmark.py --sections objects` (results JSON
`results_20261116T131255.json` in the scratch work dir), same machine and software as
`compression_local.md` (Apple M4 Pro, zarr 3.4.2; other agents running, load average 5-6).

**Data: the one real sample hour (hrrr.20231226 t00z f12) written to EVERY forecast hour of the run.**
Compression of each object is therefore real, but hour-to-hour variation is not represented: real runs will
have different (and different-sized) hours, and precipitation coverage in the sample is 23 %.

Encodings are the writer defaults: ML_* uint8 1/254 + zstd 3, flags uint8 + zstd 3, prate float32 + zstd 3,
four pyramid levels; soundings int16 step 0.1 + transpose + blosc-zstd bitshuffle, one shard per hour.
"Billed" counts every object as max(size, 128 KiB = 131,072 B), the Standard-IA minimum (AWS documents it
as "128 KB"; KiB is assumed here). All objects are counted, including every `zarr.json` and coordinate
chunk, because all of them are uploaded.

## Layouts

* **Per-hour (staging)**: what `write_ptype_hour` produces; one object per array, level and hour
  (`<level>/<array>/c/<hour>/0/0`). This is what was uploaded before this change.
* **Finalized**: `finalize_ptype_store` rewrites every gridded data array at every level as a Zarr v3
  sharded array, shard = `(n_hours, ny_l, nx_l)`, inner chunk = `(1, ny_l, nx_l)`: one object per array per
  level per run (`<level>/<array>/c/0/0/0`), shard index of `16 * n_hours + 4` bytes at its end. Root,
  level-group and coordinate objects are copied unchanged (only `hrrr_ptype.layout` changes to
  `"finalized"`).
* Soundings are already one shard per hour (~134 MB) in both layouts.

## Measured

| run | store | layout | objects | bytes | min | median | max | objects < 128 KiB | bytes in them | IA billed bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| 19 h | gridded | per-hour | 913 | 68,987,108 | 41 | 12,513 | 576,041 | 742 | 14,973,699 | 151,268,833 |
| 19 h | gridded | finalized | 121 | 69,026,155 | 41 | 1,944 | 10,945,087 | 89 | 595,950 | 80,095,613 |
| 19 h | sounding | (same) | 25 | 2,547,197,505 | 17 | 134,062,744 | 134,062,744 | 6 | 5,369 | 2,547,978,568 |
| 19 h | both | per-hour | 938 | 2,616,184,613 | 17 | 12,870 | 134,062,744 | 748 | 14,979,068 | 2,699,247,401 |
| 19 h | both | finalized | 146 | 2,616,223,660 | 17 | 1,952 | 134,062,744 | 95 | 601,319 | 2,628,074,181 |
| 49 h | gridded | per-hour | 2,233 | 177,767,014 | 85 | 15,554 | 576,041 | 1,792 | 38,469,275 | 374,178,763 |
| 49 h | gridded | finalized | 121 | 177,827,181 | 85 | 1,944 | 28,226,797 | 86 | 722,078 | 188,377,295 |
| 49 h | sounding | (same) | 55 | 6,569,079,869 | 17 | 134,062,744 | 134,062,744 | 6 | 5,413 | 6,569,860,888 |
| 49 h | both | per-hour | 2,288 | 6,746,846,883 | 17 | 15,554 | 134,062,744 | 1,798 | 38,474,688 | 6,944,039,651 |
| 49 h | both | finalized | 176 | 6,746,907,050 | 17 | 79,140 | 134,062,744 | 92 | 727,491 | 6,758,238,183 |

Gridded billed overhead (billed / actual - 1): per-hour 119 % (19 h) and 110 % (49 h); finalized 16 % and
5.9 %. Whole run (gridded + soundings): per-hour 3.2 % / 2.9 %, finalized 0.45 % / 0.17 %.
Finalization adds 39,047 B (19 h) / 60,167 B (49 h): the shard indexes plus slightly different compression.
Timings (sequential, one process, indicative): staging writes 7.5 s (gridded) + 43.7 s (soundings) for 19 h,
21.6 s + 131.7 s for 49 h; `finalize_ptype_store` 2.24 s (19 h) and 5.67 s (49 h).

## What stays under 128 KiB after finalize, and why

Per finalized gridded store (identical lists for 19 h and 49 h apart from the data objects):

* **65 `zarr.json` documents** (root, 4 level groups, 4 x 15 arrays; 105 kB together, largest 6.9 kB).
  The site opens the root, the level `forecast_hour` and every array it reads by fetching these
  documents, so they must exist as separate objects.
* **12 coordinate chunks** (`x`, `y`, `forecast_hour`, `spatial_ref` at 4 levels; 14 kB together).
* **Data shards of the sparse uint8 flags at coarse levels**: 19 h: `0/CICEP`, `1/{CFRZR,CICEP,CSNOW}`,
  `2/{CFRZR,CICEP,CSNOW}`, `3/{CFRZR,CICEP,CRAIN,CSNOW,precip_mask}` (12 objects, 7-93 kB); 49 h:
  `1/CICEP`, `2/{CFRZR,CICEP,CSNOW}`, `3/{CFRZR,CICEP,CRAIN,CSNOW,precip_mask}` (9 objects, 18-123 kB).
  Mostly-zero flags compress to almost nothing even over a whole run. On a run with little precipitation more
  flag shards, and possibly coarse prate shards, will fall below 128 KiB.
* Soundings: 6 small objects per store (3 `zarr.json`, the `forecast_hour` and `init_time` chunks).

All ML_* and prate shards are above 128 KiB in both runs.

## Archive, by arithmetic only

85 days x (4 runs of 49 h + 20 runs of 19 h) per day, using the measured run totals above:

| layout | objects | actual bytes | IA billed bytes |
|---|---|---|---|
| per-hour, gridded + soundings | 2,372,520 | 6.741 TB | 6.950 TB |
| finalized, gridded + soundings | 308,040 | 6.742 TB | 6.766 TB |
| per-hour, gridded only | 2,311,320 | 0.178 TB | 0.384 TB |
| finalized, gridded only | 246,840 | 0.178 TB | 0.200 TB |

This assumes every hour looks like the sample hour.

## Options not implemented (they would change what is stored, not what the site reads)

* **Drop the coarse levels of the HRRR flag arrays** (or all of the flags): the site currently takes the
  categorical fields from NOAA's GRIB, not from this store, so their pyramids are unused by the site. Dropping
  levels 1-3 of the five flags would remove 15 small or near-small objects per run (and their 15 `zarr.json`).
  Trade-off: any future site feature that masks coarse levels with `precip_mask` would need it back.
* **Pack the four flags into one uint8 bit field** (`precip_mask` is their OR): 5 arrays become 1 per level,
  which removes 16 data objects and 16 documents per run. Trade-off: readers need bit unpacking, and `max`
  aggregation of a bit field must be done per bit (bitwise OR), not as a numeric max.
* **Consolidated metadata** would not remove the documents: zarrita 0.7.5 opens arrays by fetching their own
  `zarr.json`, so they have to stay. At 65 x 128 KiB = 8.5 MB billed per run (0.3 % of a 19 h run) this is
  not worth contorting the layout for.
