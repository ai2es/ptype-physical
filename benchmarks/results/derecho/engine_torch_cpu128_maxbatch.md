### torch: engine `torch`, device `cpu`, threads `128`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": false, "threads": 128, "pinned": false}`

Max single-pass batch (doubling from 2^10, cap 1,073,741,824 rows): largest successful = **67,108,864** rows; stopped by: SKIPPED: estimated 371.5 GB > 80% of 250.6 GB free RAM.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 2,847 | 0.84 | 0.00 | - | - |
| 2,048 | ok | 375,370 | 0.90 | 0.00 | - | - |
| 4,096 | ok | 436,734 | 0.90 | 0.00 | - | - |
| 8,192 | ok | 319,364 | 0.90 | 0.00 | - | - |
| 16,384 | ok | 350,211 | 0.90 | 0.00 | - | - |
| 32,768 | ok | 349,855 | 0.94 | 0.03 | - | - |
| 65,536 | ok | 263,569 | 0.98 | 0.07 | - | - |
| 131,072 | ok | 707,078 | 1.11 | 0.23 | - | - |
| 262,144 | ok | 575,545 | 1.37 | 0.43 | - | - |
| 524,288 | ok | 643,171 | 1.87 | 0.84 | - | - |
| 1,048,576 | ok | 765,561 | 2.89 | 1.68 | - | - |
| 2,097,152 | ok | 935,435 | 4.92 | 3.36 | - | - |
| 4,194,304 | ok | 1,181,637 | 8.98 | 6.72 | - | - |
| 8,388,608 | ok | 1,323,991 | 17.10 | 13.42 | - | - |
| 16,777,216 | ok | 1,381,759 | 33.34 | 26.85 | - | - |
| 33,554,432 | ok | 1,493,403 | 65.83 | 53.69 | - | - |
| 67,108,864 | ok | 1,385,968 | 130.75 | 107.34 | - | - |
| 134,217,728 | SKIPPED: estimated 371.5 GB > 80% of 250.6 GB free RAM | - | - | - | - | - |

