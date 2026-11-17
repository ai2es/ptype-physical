### torch: engine `torch`, device `cpu`, threads `8`

Engine config: `{"engine": "torch", "device": "cpu", "dtype": "float32", "tf32": false, "compile": false, "threads": 8, "pinned": false}`

Throughput on 1,905,141 rows (one HRRR CONUS forecast hour), 3 timed repeats after warm-up.

| batch size | median rows/s | best rows/s | median s per pass | peak RSS GB | peak GPU alloc GB |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 619,963 | 621,867 | 3.07 | 1.52 | - |
| 2,048 | 702,690 | 703,976 | 2.71 | 1.52 | - |
| 4,096 | 788,353 | 789,812 | 2.42 | 1.52 | - |
| 8,192 | 738,831 | 751,411 | 2.58 | 1.54 | - |
| 16,384 | 799,144 | 816,147 | 2.38 | 1.57 | - |
| 32,768 | 771,252 | 772,066 | 2.47 | 1.61 | - |
| 65,536 | 802,667 | 803,504 | 2.37 | 1.68 | - |
| 131,072 | 756,131 | 757,815 | 2.52 | 1.75 | - |
| 262,144 | 732,788 | 733,027 | 2.60 | 1.98 | - |
| 524,288 | 821,183 | 821,973 | 2.32 | 2.41 | - |
| 1,048,576 | 893,888 | 895,487 | 2.13 | 3.21 | - |
| 2,097,152 | 921,768 | 925,236 | 2.07 | 4.57 | - |

