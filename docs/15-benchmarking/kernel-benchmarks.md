# Kernel Benchmarks

vLLM provides micro-benchmarks for its custom CUDA kernels, covering CUTLASS GEMM operations (W8A8 quantized and sparse), fused normalization kernels, and a wide range of other GPU primitives. These benchmarks help identify the fastest kernel variant for a given hardware and shape combination.

## Directory Layout

```
benchmarks/
├── cutlass_benchmarks/
│   ├── sparse_benchmarks.py      # CUTLASS sparse INT8/FP8 GEMM
│   ├── w8a8_benchmarks.py        # CUTLASS W8A8 INT8/FP8 GEMM
│   ├── utils.py                  # Tensor generation helpers
│   └── weight_shapes.py          # Model weight shape presets
├── fused_kernels/
│   └── layernorm_rms_benchmarks.py  # Fused RMSNorm + quantization
└── kernels/
    ├── benchmark_layernorm.py
    ├── benchmark_rmsnorm.py
    ├── benchmark_paged_attention.py
    ├── benchmark_moe.py
    ├── benchmark_fp8_gemm.py
    ├── benchmark_int8_gemm.py
    ├── benchmark_marlin.py
    ├── benchmark_machete.py
    └── ...  (30+ additional kernel benchmarks)
```

## CUTLASS W8A8 Benchmarks

`benchmarks/cutlass_benchmarks/w8a8_benchmarks.py` benchmarks CUTLASS-based quantized GEMM kernels against PyTorch baselines for both INT8 and FP8 data types.

### Kernels Benchmarked

**INT8 kernels** (`bench_int8`):

| Kernel | Description |
|--------|-------------|
| `pytorch_bf16_bf16_bf16_matmul-no-scales` | PyTorch BF16 baseline |
| `pytorch_fp16_fp16_fp16_matmul-no-scales` | PyTorch FP16 baseline |
| `cutlass_i8_i8_bf16_scaled_mm` | CUTLASS INT8 scaled GEMM |
| `cutlass_i8_i8_bf16_scaled_mm_bias` | CUTLASS INT8 with bias |
| `cutlass_i8_i8_bf16_scaled_mm_azp` | CUTLASS INT8 with activation zero-point |
| `cutlass_i8_i8_bf16_scaled_mm_azp_bias` | CUTLASS INT8 with AZP + bias |
| `cutlass_i8_i8_bf16_scaled_mm_azp_pt` | CUTLASS INT8 with per-token AZP |
| `cutlass_i8_i8_bf16_scaled_mm_azp_pt_bias` | CUTLASS INT8 with per-token AZP + bias |

**FP8 kernels** (`bench_fp8`):

| Kernel | Description |
|--------|-------------|
| `pytorch_fp8_fp8_bf16_scaled_mm` | PyTorch FP8 scaled GEMM |
| `pytorch_fp8_fp8_bf16_scaled_mm_fast_accum` | PyTorch FP8 with fast accumulation |
| `cutlass_fp8_fp8_bf16_scaled_mm` | CUTLASS FP8 → BF16 |
| `cutlass_fp8_fp8_fp16_scaled_mm` | CUTLASS FP8 → FP16 |
| `cutlass_fp8_fp8_bf16_scaled_mm_bias` | CUTLASS FP8 with bias |
| `triton_fp8_fp8_fp16_scaled_mm_blockwise` | Triton block-wise FP8 |
| `cutlass_fp8_fp8_fp16_scaled_mm_blockwise` | CUTLASS block-wise FP8 |

### Running W8A8 Benchmarks

```bash
cd benchmarks/cutlass_benchmarks

# Benchmark INT8 square matrices
python w8a8_benchmarks.py --dtype int8 --square-bench \
    --dim-start 128 --dim-end 4096 --dim-increment 128

# Benchmark FP8 with specific M/K/N dimensions
python w8a8_benchmarks.py --dtype float8_e4m3fn --range-bench \
    --m-constant 1 --dim-start 1024 --dim-end 8192 --dim-increment 1024

# Benchmark model weight shapes (e.g., Llama-3-8B)
python w8a8_benchmarks.py --dtype int8 --model-bench \
    --models meta-llama/Meta-Llama-3-8B \
    --batch-sizes 1 16 32 64 128 256 512

# Benchmark only specific kernels
python w8a8_benchmarks.py --dtype float8_e4m3fn --square-bench \
    --kernels cutlass_fp8_fp8_bf16_scaled_mm triton_fp8_fp8_fp16_scaled_mm_blockwise
```

### Weight Shape Presets

`weight_shapes.py` defines standard weight shapes for popular models. These are used with `--model-bench` to test realistic GEMM dimensions:

```python
WEIGHT_SHAPES = {
    "meta-llama/Meta-Llama-3-8B": [
        (1, 4096, 14336),   # FFN gate/up
        (1, 14336, 4096),   # FFN down
        (1, 4096, 4096),    # Attention QKV
        ...
    ],
    ...
}
```

## CUTLASS Sparse Benchmarks

`benchmarks/cutlass_benchmarks/sparse_benchmarks.py` benchmarks CUTLASS 2:4 structured sparse GEMM kernels. These are used for sparse quantization where 50% of weights are pruned in a structured pattern.

### Kernels Benchmarked

| Kernel | Description |
|--------|-------------|
| `pytorch_bf16_bf16_bf16_matmul-no-scales` | Dense BF16 baseline |
| `pytorch_fp16_fp16_fp16_matmul-no-scales` | Dense FP16 baseline |
| `cutlass_i8_i8_bf16_scaled_mm` | Dense CUTLASS INT8 |
| `cutlass_i8_i8_bf16_scaled_sparse_mm` | **Sparse** CUTLASS INT8 (2:4) |

The sparse benchmark verifies correctness by comparing sparse output against the dense reference before timing.

### Running Sparse Benchmarks

```bash
cd benchmarks/cutlass_benchmarks

# Sparse INT8 square benchmark
python sparse_benchmarks.py --dtype int8 --square-bench \
    --dim-start 256 --dim-end 4096 --dim-increment 256

# Sparse INT8 model weight shapes
python sparse_benchmarks.py --dtype int8 --model-bench \
    --models meta-llama/Meta-Llama-3-8B \
    --batch-sizes 1 16 32 64 128
```

## Fused Kernel Benchmarks

`benchmarks/fused_kernels/layernorm_rms_benchmarks.py` benchmarks fused RMSNorm + quantization kernels against their unfused equivalents.

### Benchmark Parameters

```python
@dataclass
class bench_params_t:
    num_tokens: int       # Powers of 2 from 1 to 1024
    hidden_size: int      # 1024 to 8192 in steps of 1024
    add_residual: bool    # With or without residual addition
    dtype: torch.dtype    # bfloat16 or float32
    group_size: list[int] # [1, 64] or [1, 128] for block-wise quant
```

### Implementations Compared

| Implementation | Description |
|----------------|-------------|
| `unfused_int8_impl` | Separate RMSNorm + `scaled_int8_quant` |
| `unfused_fp8_impl` | Separate RMSNorm + `scaled_fp8_quant` |
| `fused_int8_impl` | `rms_norm_dynamic_per_token_quant` (INT8) |
| `fused_fp8_impl` | `rms_norm_dynamic_per_token_quant` (FP8) |
| `unfused_groupwise_fp8_impl` | Separate RMSNorm + `per_token_group_quant_fp8` |
| `fused_groupwise_impl` | `rms_norm_per_block_quant` (block-wise FP8) |

The fused kernels combine normalization and quantization in a single GPU pass, reducing memory bandwidth usage.

### Running Fused Kernel Benchmarks

```bash
cd benchmarks/fused_kernels
python layernorm_rms_benchmarks.py
# Results saved to rms_norm_dpt_quant-<timestamp>.pkl
```

## Additional Kernel Benchmarks (`benchmarks/kernels/`)

The `kernels/` directory contains 30+ individual kernel micro-benchmarks:

| Script | Kernel |
|--------|--------|
| `benchmark_paged_attention.py` | Paged attention decode |
| `benchmark_layernorm.py` | LayerNorm variants |
| `benchmark_rmsnorm.py` | RMSNorm variants |
| `benchmark_moe.py` | Mixture-of-Experts routing |
| `benchmark_fp8_gemm.py` | FP8 GEMM |
| `benchmark_int8_gemm.py` | INT8 GEMM |
| `benchmark_marlin.py` | Marlin GPTQ kernels |
| `benchmark_machete.py` | Machete mixed-precision kernels |
| `benchmark_rope.py` | Rotary position embeddings |
| `benchmark_mrope.py` | Multi-dimensional RoPE |
| `benchmark_lora.py` | LoRA GEMM kernels |
| `benchmark_mla_k_concat.py` | MLA key concatenation |
| `benchmark_reshape_and_cache.py` | KV cache reshape |
| `benchmark_quant.py` | Quantization kernels |
| `benchmark_activation.py` | Activation functions (SiLU, GELU) |
| `benchmark_moe_permute_unpermute.py` | MoE token permutation |
| `benchmark_fused_topk.py` | Fused top-k selection |
| `benchmark_device_communicators.py` | NCCL/custom all-reduce |
| `benchmark_grouped_gemm_cutlass.py` | Grouped GEMM (MoE) |
| `benchmark_cutlass_moe_fp8.py` | CUTLASS FP8 MoE GEMM |
| `benchmark_cutlass_moe_nvfp4.py` | CUTLASS NVFP4 MoE GEMM |
| `benchmark_w8a8_block_fp8.py` | Block-wise W8A8 FP8 |
| `benchmark_block_fp8_gemm.py` | Block FP8 GEMM |
| `benchmark_nvfp4_gemm.py` | NVFP4 GEMM |
| `benchmark_nvfp4_quant.py` | NVFP4 quantization |
| `benchmark_per_token_quant_fp8.py` | Per-token FP8 quantization |
| `benchmark_per_token_group_quant.py` | Per-token group quantization |
| `benchmark_silu_mul_fp8_quant.py` | Fused SiLU + FP8 quant |
| `benchmark_fused_collective.py` | Fused collective operations |
| `benchmark_shapes.py` | Shape-based GEMM sweep |

### Example: Paged Attention Benchmark

```bash
cd benchmarks/kernels
python benchmark_paged_attention.py \
    --version v2 \
    --batch-size 32 \
    --context-len 2048 \
    --num-query-heads 32 \
    --num-kv-heads 8 \
    --head-size 128 \
    --block-size 16 \
    --dtype float16
```

### Example: MoE Benchmark

```bash
python benchmark_moe.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --tp-size 2 \
    --batch-size 64
```

## Benchmark Infrastructure

All CUTLASS benchmarks use `torch.utils.benchmark.Timer` with `blocked_autorange` for statistically robust timing:

```python
def bench_fn(label, sub_label, description, fn, *args, **kwargs):
    return TBenchmark.Timer(
        stmt="fn(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "fn": fn},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=1)  # Run for at least 1 second
```

Results are collected as `TMeasurement` objects and compared with `TBenchmark.Compare`:

```python
compare = TBenchmark.Compare(timers)
compare.print()
```

Results can also be pickled for later analysis:

```python
with open(f"results-{timestamp}.pkl", "wb") as f:
    pkl.dump(timers, f)
```

## Interpreting Results

The `TBenchmark.Compare` output shows:

```
[------------- scaled-torch.int8-gemm -------------]
                                  |  pytorch_bf16  |  cutlass_i8_i8_bf16
1 threads: -----------------------------------------------
      MKN=(1x4096x14336)         |    1234.5 us   |    456.7 us
      MKN=(16x4096x14336)        |    2345.6 us   |    789.0 us
      MKN=(128x4096x14336)       |    8901.2 us   |   2345.6 us
```

- **Lower is better** — all times are in microseconds
- Compare CUTLASS kernels against PyTorch baselines to quantify speedup
- Look for crossover points where one kernel becomes faster than another

## Related Pages

- [Quantization](../14-quantization/README.md) — W8A8, FP8, and sparse quantization
- [Attention Benchmarks](attention-benchmarks.md) — Attention backend benchmarks
- [Performance Tuning Guide](performance-tuning.md) — Applying benchmark insights
