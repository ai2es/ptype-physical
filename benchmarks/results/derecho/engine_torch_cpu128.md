### torch: engine `torch`, device `cpu`, threads `128`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": false, "threads": 128, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 463,294 | 477,668 | 4.11 | 1.52 | - |
| 2,048 | 492,634 | 509,693 | 3.87 | 1.52 | - |
| 4,096 | 628,039 | 628,567 | 3.03 | 1.52 | - |
| 8,192 | 686,761 | 686,840 | 2.77 | 1.54 | - |
| 16,384 | 842,385 | 894,931 | 2.26 | 1.55 | - |
| 32,768 | 840,429 | 845,805 | 2.27 | 1.62 | - |
| 65,536 | 710,580 | 712,521 | 2.68 | 1.71 | - |
| 131,072 | 669,233 | 674,911 | 2.85 | 1.78 | - |
| 262,144 | 676,868 | 693,059 | 2.81 | 2.01 | - |
| 524,288 | 760,446 | 772,942 | 2.51 | 2.46 | - |
| 1,048,576 | 864,412 | 871,099 | 2.20 | 3.24 | - |
| 2,097,152 | 1,043,781 | 1,050,560 | 1.83 | 4.60 | - |

