### torch: engine `torch`, device `cpu`, threads `1`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": false, "threads": 1, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 272,771 | 273,031 | 6.98 | 1.53 | - |
| 2,048 | 280,211 | 280,259 | 6.80 | 1.53 | - |
| 4,096 | 284,550 | 284,595 | 6.70 | 1.53 | - |
| 8,192 | 284,758 | 284,860 | 6.69 | 1.55 | - |
| 16,384 | 265,731 | 266,538 | 7.17 | 1.57 | - |
| 32,768 | 243,177 | 245,256 | 7.83 | 1.61 | - |
| 65,536 | 231,155 | 231,363 | 8.24 | 1.68 | - |
| 131,072 | 230,053 | 230,587 | 8.28 | 1.75 | - |
| 262,144 | 230,585 | 232,142 | 8.26 | 1.97 | - |
| 524,288 | 230,742 | 232,835 | 8.26 | 2.43 | - |
| 1,048,576 | 229,902 | 232,824 | 8.29 | 3.21 | - |
| 2,097,152 | 231,552 | 233,505 | 8.23 | 4.57 | - |

