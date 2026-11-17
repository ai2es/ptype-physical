### onnx: engine `onnx`, device `cpu`, threads `128`

Engine config: `{"engine": "onnx", "device": "cpu", "threads": 128, "providers": ["CPUExecutionProvider"]}`

Max single-pass batch (doubling from 2^10, cap 1,073,741,824 rows): largest successful = **33,554,432** rows; stopped by: SKIPPED: estimated 185.8 GB > 80% of 139.3 GB free RAM.

| rows | status | rows/s | peak RSS GB | RSS above pre-pass GB | peak GPU alloc GB | peak GPU reserved GB |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | ok | 266,382 | 0.85 | 0.00 | - | - |
| 2,048 | ok | 533,471 | 0.92 | 0.00 | - | - |
| 4,096 | ok | 800,258 | 0.92 | 0.00 | - | - |
| 8,192 | ok | 911,136 | 0.93 | 0.00 | - | - |
| 16,384 | ok | 930,229 | 0.95 | 0.00 | - | - |
| 32,768 | ok | 855,134 | 1.01 | 0.03 | - | - |
| 65,536 | ok | 725,530 | 1.09 | 0.08 | - | - |
| 131,072 | ok | 891,510 | 1.22 | 0.16 | - | - |
| 262,144 | ok | 912,652 | 1.60 | 0.33 | - | - |
| 524,288 | ok | 910,583 | 2.36 | 0.67 | - | - |
| 1,048,576 | ok | 978,568 | 3.84 | 1.31 | - | - |
| 2,097,152 | ok | 948,630 | 7.71 | 3.54 | - | - |
| 4,194,304 | ok | 957,254 | 15.45 | 7.07 | - | - |
| 8,388,608 | ok | 918,009 | 30.95 | 14.15 | - | - |
| 16,777,216 | ok | 606,856 | 61.97 | 28.34 | - | - |
| 33,554,432 | ok | 851,956 | 123.95 | 56.68 | - | - |
| 67,108,864 | SKIPPED: estimated 185.8 GB > 80% of 139.3 GB free RAM | - | - | - | - | - |

