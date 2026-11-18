# When HRRR wrfnat files reach the public buckets

Measured 2026-11-18 from the object listings of four cycles: `LastModified` on AWS
(`noaa-hrrr-bdp-pds`) and `timeCreated` on Google (`high-resolution-rapid-refresh`). Values are minutes after
the cycle's initialization time; in brackets, the `.idx` of the same forecast hour.

| Cycle | Bucket | F00 | F01 | F06 | F12 | F18 | F24 | F36 | F48 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 12Z (48 h) | AWS | 53 (53) | 54 (54) | 60 (60) | 67 (68) | 75 (75) | 81 (81) | 94 (95) | 108 (109) |
| | Google | 52 (51) | 53 (52) | 59 (58) | 67 (66) | 74 (73) | 81 (80) | 94 (93) | 107 (107) |
| 2026-11-17 00Z (48 h) | AWS | 53 (54) | 54 (56) | 62 (62) | 68 (68) | 76 (77) | 83 (84) | 96 (96) | 110 (111) |
| | Google | 53 (52) | 54 (53) | 68 (68) | 69 (68) | 75 (74) | 82 (81) | 95 (94) | 109 (108) |
| 2026-11-17 13Z (18 h) | AWS | 53 (54) | 56 (56) | 65 (66) | 76 (77) | 88 (89) | | | |
| | Google | 53 (52) | 55 (54) | 64 (63) | 76 (75) | 87 (86) | | | |
| 2026-11-16 21Z (18 h) | AWS | 52 (53) | 54 (55) | 64 (64) | 74 (74) | 86 (86) | | | |
| | Google | 52 (51) | 54 (53) | 63 (62) | 74 (73) | 85 (84) | | | |

What this fixes in the design:

- The analysis is there 52 to 54 minutes after initialization. The last hour is 75 minutes (48 h cycles) to
  88 minutes (18 h cycles) for F18, and about 110 minutes for F48. Waiting for a whole cycle before publishing
  would delay its first hours by 20 to 55 minutes.
- Files arrive about one a minute in a 48 h cycle and one every two minutes in an 18 h cycle.
- The two buckets are within a minute of each other, with exceptions (Google's 00Z F06 was 6 minutes behind).
- **Order of the idx.** On AWS the `.idx` was written 0.5 to 1.2 minutes *after* its GRIB file in every one of
  the 136 hours looked at. On Google it was usually created *before* the GRIB file (by up to 12.5 minutes)
  and sometimes after it (by up to 1.4 minutes). So an idx alone does not mean the GRIB file can be read. The realtime driver treats an hour as
  ready only when both objects are listed (`pipeline.GribSource.available`), which is right on both.

Four cycles on two days is a small sample; the driver does not depend on these numbers, only the job
submission lead time (40 minutes) and walltimes do.
