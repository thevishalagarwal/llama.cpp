# llama.cpp/examples/llama-bench

Performance testing tool for llama.cpp.

## Table of contents

1. [Syntax](#syntax)
2. [Examples](#examples)
    1. [Text generation with different models](#text-generation-with-different-models)
    2. [Prompt processing with different batch sizes](#prompt-processing-with-different-batch-sizes)
    3. [Different numbers of threads](#different-numbers-of-threads)
    4. [Different numbers of layers offloaded to the GPU](#different-numbers-of-layers-offloaded-to-the-gpu)
3. [Output formats](#output-formats)
    1. [Markdown](#markdown)
    2. [CSV](#csv)
    3. [JSON](#json)
    4. [JSONL](#jsonl)
    5. [SQL](#sql)

## Syntax

```
usage: ./llama-bench [options]

options:
  -h, --help
  -m, --model <filename>                    (default: models/7B/ggml-model-q4_0.gguf)
  -p, --n-prompt <n>                        (default: 512)
  -n, --n-gen <n>                           (default: 128)
  -pg <pp,tg>                               (default: )
  -b, --batch-size <n>                      (default: 2048)
  -ub, --ubatch-size <n>                    (default: 512)
  -ctk, --cache-type-k <t>                  (default: f16)
  -ctv, --cache-type-v <t>                  (default: f16)
  -t, --threads <n>                         (default: 8)
  -C, --cpu-mask <hex,hex>                  (default: 0x0)
  --cpu-strict <0|1>                        (default: 0)
  --poll <0...100>                          (default: 50)
  -ngl, --n-gpu-layers <n>                  (default: 99)
  -rpc, --rpc <rpc_servers>                 (default: )
  -sm, --split-mode <none|layer|row>        (default: layer)
  -mg, --main-gpu <i>                       (default: 0)
  -nkvo, --no-kv-offload <0|1>              (default: 0)
  -fa, --flash-attn <0|1>                   (default: 0)
  -mmp, --mmap <0|1>                        (default: 1)
  --numa <distribute|isolate|numactl>       (default: disabled)
  -embd, --embeddings <0|1>                 (default: 0)
  -ts, --tensor-split <ts0/ts1/..>          (default: 0)
  -r, --repetitions <n>                     (default: 5)
  --prio <0|1|2|3>                          (default: 0)
  --delay <0...N> (seconds)                 (default: 0)
  -o, --output <csv|json|jsonl|md|sql>      (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql> (default: none)
  -v, --verbose                             (default: 0)

Multiple values can be given for each parameter by separating them with ',' or by specifying the parameter multiple times.
```

llama-bench can perform three types of tests:

- Prompt processing (pp): processing a prompt in batches (`-p`)
- Text generation (tg): generating a sequence of tokens (`-n`)
- Prompt processing + text generation (pg): processing a prompt followed by generating a sequence of tokens (`-pg`)

With the exception of `-r`, `-o` and `-v`, all options can be specified multiple times to run multiple tests. Each pp and tg test is run with all combinations of the specified options. To specify multiple values for an option, the values can be separated by commas (e.g. `-n 16,32`), or the option can be specified multiple times (e.g. `-n 16 -n 32`).

Each test is repeated the number of times given by `-r`, and the results are averaged. The results are given in average tokens per second (t/s) and standard deviation. Some output formats (e.g. json) also include the individual results of each repetition.

For a description of the other options, see the [main example](../main/README.md).

Note:

- When using SYCL backend, there would be hang issue in some cases. Please set `--mmp 0`.

## Examples

### Prompt processing and text generation

```sh
$ ./llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p 0 -n 0 -pg 100,100 -pg 500,100
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp100+tg100 |  14303.91 ± 362.95 |  455.50 ± 11.99 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |   pp500+tg100 |  28026.49 ± 970.98 |   440.05 ± 4.57 |


### Text generation with different models

```sh
$ ./llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 0 -n 128,256,512
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg128 |        0.00 ± 0.00 |   469.99 ± 2.69 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg256 |        0.00 ± 0.00 |   454.10 ± 9.76 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg512 |        0.00 ± 0.00 |  444.62 ± 11.83 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg128 |        0.00 ± 0.00 |   219.82 ± 0.37 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg256 |        0.00 ± 0.00 |   215.15 ± 2.04 |
| llama 3B Q4_K - Medium         |     3.21 B | CUDA       |  99 |         tg512 |        0.00 ± 0.00 |   211.12 ± 1.43 |

### Prompt processing with different batch sizes

```sh
$ ./llama-bench -n 0 -p 1024 -b 128,256,512,1024
```

| model                          |     params | backend    | ngl | n_batch |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     128 |        pp1024 |  16751.82 ± 667.31 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     256 |        pp1024 |  23255.17 ± 446.86 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |     512 |        pp1024 |  25544.36 ± 571.16 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |    1024 |        pp1024 |  25610.04 ± 606.37 |     0.00 ± 0.00 |

### Different numbers of threads

```sh
$ ./llama-bench -n 0 -n 16 -p 64 -t 1,2,4,8,16,32
```

| model                          |     params | backend    | ngl | threads |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       1 |          pp64 |  10322.32 ± 193.62 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       1 |          tg16 |        0.00 ± 0.00 |   444.22 ± 6.66 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       2 |          pp64 |   7313.31 ± 145.33 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       2 |          tg16 |        0.00 ± 0.00 |  468.99 ± 12.30 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       4 |          pp64 | 10111.46 ± 1261.15 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       4 |          tg16 |        0.00 ± 0.00 |  464.07 ± 18.15 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       8 |          pp64 |  9605.79 ± 1684.50 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |       8 |          tg16 |        0.00 ± 0.00 |  469.92 ± 16.23 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      16 |          pp64 |  10336.80 ± 740.34 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      16 |          tg16 |        0.00 ± 0.00 |  472.06 ± 10.29 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      32 |          pp64 |  8819.08 ± 1529.51 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |      32 |          tg16 |        0.00 ± 0.00 |  458.20 ± 15.14 |

### Different numbers of layers offloaded to the GPU

```sh
$ ./llama-bench -ngl 10,20,30,31,32,33,34,35
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  10 |         pp512 |  12082.67 ± 403.77 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  10 |         tg128 |        0.00 ± 0.00 |   106.37 ± 2.72 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  20 |         pp512 | 16742.49 ± 8252.51 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  20 |         tg128 |        0.00 ± 0.00 |   454.01 ± 8.64 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  30 |         pp512 |  29580.40 ± 106.86 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  30 |         tg128 |        0.00 ± 0.00 |   457.68 ± 9.88 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  31 |         pp512 |  29594.52 ± 154.46 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  31 |         tg128 |        0.00 ± 0.00 |   465.27 ± 9.24 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  32 |         pp512 |  29503.27 ± 174.82 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  32 |         tg128 |        0.00 ± 0.00 |   467.16 ± 2.22 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  33 |         pp512 |  29479.41 ± 180.78 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  33 |         tg128 |        0.00 ± 0.00 |   465.67 ± 6.10 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  34 |         pp512 |   29446.50 ± 59.09 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  34 |         tg128 |        0.00 ± 0.00 |   470.60 ± 2.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  35 |         pp512 |  29369.74 ± 229.29 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  35 |         tg128 |        0.00 ± 0.00 |   467.41 ± 6.81 |

## Output formats

By default, llama-bench outputs the results in markdown format. The results can be output in other formats by using the `-o` option.

### Markdown

```sh
$ ./llama-bench -o md
```

| model                          |     params | backend    | ngl |          test |         prompt t/s |         gen t/s |
| ------------------------------ | ---------: | ---------- | --: | ------------: | -----------------: | --------------: |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         pp512 |   27663.05 ± 90.18 |     0.00 ± 0.00 |
| llama 1B Q4_K - Medium         |     1.24 B | CUDA       |  99 |         tg128 |        0.00 ± 0.00 |   467.13 ± 5.21 |

### CSV

```sh
$ ./llama-bench -o csv
```

```csv
build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,use_mmap,embeddings,n_prompt,n_gen,test_time,avg_prompt_ns,stddev_prompt_ns,avg_prompt_ts,stddev_prompt_ts,avg_gen_ns,stddev_gen_ns,avg_gen_ts,stddev_gen_ts
"df46ea53","5099","AMD Ryzen 7 7800X3D 8-Core Processor           ","NVIDIA GeForce RTX 4080","CUDA","models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","llama 1B Q4_K - Medium","799862912","1235814432","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","1","0","512","0","2025-04-18T07:40:28Z","18599580","87220","27527.987050","128.945972","60","54","0.000000","0.000000"
"df46ea53","5099","AMD Ryzen 7 7800X3D 8-Core Processor           ","NVIDIA GeForce RTX 4080","CUDA","models/Llama-3.2-1B-Instruct-Q4_K_M.gguf","llama 1B Q4_K - Medium","799862912","1235814432","2048","512","8","0x0","0","50","f16","f16","99","layer","0","0","0","0.00","1","0","0","128","2025-04-18T07:40:28Z","0","0","0.000000","0.000000","273595440","4100226","467.926504","6.877051"
```

### JSON

```sh
$ ./llama-bench -o json
```

```json
[
  {
    "build_commit": "df46ea53",
    "build_number": 5099,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "model_type": "llama 1B Q4_K - Medium",
    "model_size": 799862912,
    "model_n_params": 1235814432,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "n_prompt": 512,
    "n_gen": 0,
    "test_time": "2025-04-18T07:41:24Z",
    "avg_prompt_ns": 19432500,
    "stddev_prompt_ns": 1155276,
    "avg_prompt_ts": 26420.253006,
    "stddev_prompt_ts": 1527.724050,
    "avg_gen_ns": 60,
    "stddev_gen_ns": 89,
    "avg_gen_ts": 0.000000,
    "stddev_gen_ts": 0.000000,
    "samples_prompt_ns": [ 18723500, 18641500, 18476200, 21034200, 20287100 ],
    "samples_prompt_ts": [ 27345.3, 27465.6, 27711.3, 24341.3, 25237.7 ]
    "samples_gen_ns": [ 0, 100, 0, 200, 0 ],
    "samples_gen_ts": [ 0 ]
  },
  {
    "build_commit": "df46ea53",
    "build_number": 5099,
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "model_type": "llama 1B Q4_K - Medium",
    "model_size": 799862912,
    "model_n_params": 1235814432,
    "n_batch": 2048,
    "n_ubatch": 512,
    "n_threads": 8,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": false,
    "tensor_split": "0.00",
    "use_mmap": true,
    "embeddings": false,
    "n_prompt": 0,
    "n_gen": 128,
    "test_time": "2025-04-18T07:41:24Z",
    "avg_prompt_ns": 20,
    "stddev_prompt_ns": 44,
    "avg_prompt_ts": 0.000000,
    "stddev_prompt_ts": 0.000000,
    "avg_gen_ns": 279581280,
    "stddev_gen_ns": 7013491,
    "avg_gen_ts": 458.054981,
    "stddev_gen_ts": 11.337387,
    "samples_prompt_ns": [ 0, 0, 0, 0, 100 ],
    "samples_prompt_ts": [ 0 ]
    "samples_gen_ns": [ 290465300, 280112200, 280351000, 274751300, 272226600 ],
    "samples_gen_ts": [ 440.672, 456.96, 456.571, 465.876, 470.197 ]
  }
]
```


### JSONL

```sh
$ ./llama-bench -o jsonl
```

```json lines
{"build_commit": "df46ea53", "build_number": 5099, "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ", "gpu_info": "NVIDIA GeForce RTX 4080", "backends": "CUDA", "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", "model_type": "llama 1B Q4_K - Medium", "model_size": 799862912, "model_n_params": 1235814432, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "use_mmap": true, "embeddings": false, "n_prompt": 512, "n_gen": 0, "test_time": "2025-04-18T07:42:16Z", "avg_prompt_ns": 18637700, "stddev_prompt_ns": 164536, "avg_prompt_ts": 27472.914745, "stddev_prompt_ts": 242.676976, "avg_gen_ns": 0, "stddev_gen_ns": 0, "avg_gen_ts": 0.000000, "stddev_gen_ts": 0.000000, "samples_prompt_ns": [ 18782700, 18654200, 18812300, 18466100, 18473200 ],"samples_prompt_ts": [ 27259.1, 27446.9, 27216.2, 27726.5, 27715.8 ]"samples_gen_ns": [ 0, 0, 0, 0, 0 ],"samples_gen_ts": [ 0 ]}
{"build_commit": "df46ea53", "build_number": 5099, "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor           ", "gpu_info": "NVIDIA GeForce RTX 4080", "backends": "CUDA", "model_filename": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf", "model_type": "llama 1B Q4_K - Medium", "model_size": 799862912, "model_n_params": 1235814432, "n_batch": 2048, "n_ubatch": 512, "n_threads": 8, "cpu_mask": "0x0", "cpu_strict": false, "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99, "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false, "flash_attn": false, "tensor_split": "0.00", "use_mmap": true, "embeddings": false, "n_prompt": 0, "n_gen": 128, "test_time": "2025-04-18T07:42:17Z", "avg_prompt_ns": 100, "stddev_prompt_ns": 122, "avg_prompt_ts": 0.000000, "stddev_prompt_ts": 0.000000, "avg_gen_ns": 273159220, "stddev_gen_ns": 2025528, "avg_gen_ts": 468.611680, "stddev_gen_ts": 3.456568, "samples_prompt_ns": [ 300, 100, 100, 0, 0 ],"samples_prompt_ts": [ 0 ]"samples_gen_ns": [ 276312400, 272096600, 271459600, 274053300, 271874200 ],"samples_gen_ts": [ 463.244, 470.421, 471.525, 467.062, 470.806 ]}
```


### SQL

SQL output is suitable for importing into a SQLite database. The output can be piped into the `sqlite3` command line tool to add the results to a database.

```sh
$ ./llama-bench -o sql
```

```sql
CREATE TABLE IF NOT EXISTS test (
  build_commit TEXT,
  build_number INTEGER,
  cpu_info TEXT,
  gpu_info TEXT,
  backends TEXT,
  model_filename TEXT,
  model_type TEXT,
  model_size INTEGER,
  model_n_params INTEGER,
  n_batch INTEGER,
  n_ubatch INTEGER,
  n_threads INTEGER,
  cpu_mask TEXT,
  cpu_strict INTEGER,
  poll INTEGER,
  type_k TEXT,
  type_v TEXT,
  n_gpu_layers INTEGER,
  split_mode TEXT,
  main_gpu INTEGER,
  no_kv_offload INTEGER,
  flash_attn INTEGER,
  tensor_split TEXT,
  use_mmap INTEGER,
  embeddings INTEGER,
  n_prompt INTEGER,
  n_gen INTEGER,
  test_time TEXT,
  avg_prompt_ns INTEGER,
  stddev_prompt_ns INTEGER,
  avg_prompt_ts REAL,
  stddev_prompt_ts REAL,
  avg_gen_ns INTEGER,
  stddev_gen_ns INTEGER,
  avg_gen_ts REAL,
  stddev_gen_ts REAL
);

INSERT INTO test (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, use_mmap, embeddings, n_prompt, n_gen, test_time, avg_prompt_ns, stddev_prompt_ns, avg_prompt_ts, stddev_prompt_ts, avg_gen_ns, stddev_gen_ns, avg_gen_ts, stddev_gen_ts) VALUES ('df46ea53', '5099', 'AMD Ryzen 7 7800X3D 8-Core Processor           ', 'NVIDIA GeForce RTX 4080', 'CUDA', 'models/Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'llama 1B Q4_K - Medium', '799862912', '1235814432', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', '1', '0', '512', '0', '2025-04-18T07:42:43Z', '18543960', '131206', '27611.175041', '195.547424', '60', '54', '0.000000', '0.000000');
INSERT INTO test (build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type, model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll, type_k, type_v, n_gpu_layers, split_mode, main_gpu, no_kv_offload, flash_attn, tensor_split, use_mmap, embeddings, n_prompt, n_gen, test_time, avg_prompt_ns, stddev_prompt_ns, avg_prompt_ts, stddev_prompt_ts, avg_gen_ns, stddev_gen_ns, avg_gen_ts, stddev_gen_ts) VALUES ('df46ea53', '5099', 'AMD Ryzen 7 7800X3D 8-Core Processor           ', 'NVIDIA GeForce RTX 4080', 'CUDA', 'models/Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'llama 1B Q4_K - Medium', '799862912', '1235814432', '2048', '512', '8', '0x0', '0', '50', 'f16', 'f16', '99', 'layer', '0', '0', '0', '0.00', '1', '0', '0', '128', '2025-04-18T07:42:43Z', '20', '44', '0.000000', '0.000000', '274190080', '2765950', '466.867210', '4.680900');
```
