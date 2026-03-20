# KV Cache Configuration

`CacheConfig` controls the KV (key-value) cache — the memory structure that stores attention keys and values for all tokens in active sequences. Proper KV cache configuration is critical for both memory efficiency and throughput.

**Source:** `vllm/config/cache.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — KV Cache section.

---

## Memory Allocation

### `gpu_memory_utilization`

```
Type:    float (0 < value ≤ 1)
Default: 0.9
CLI:     --gpu-memory-utilization
```

Fraction of GPU memory to allocate for the model executor (weights + KV cache + activations). The KV cache receives whatever memory remains after model weights and activations are allocated.

- `0.9` — Use 90% of GPU memory (default)
- `0.95` — Maximize KV cache size (less headroom for other processes)
- `0.7` — Leave more headroom (useful when sharing GPU with other workloads)

**This is a per-instance limit.** Multiple vLLM instances on the same GPU each apply their own limit independently.

```bash
vllm serve mymodel --gpu-memory-utilization 0.85
```

### `kv_cache_memory_bytes`

```
Type:    int | None
Default: None (use gpu_memory_utilization)
CLI:     --kv-cache-memory-bytes
```

Explicit KV cache size per GPU in bytes. When set, **overrides** `gpu_memory_utilization` for KV cache sizing. Supports human-readable suffixes on the CLI: `16g`, `32G`, `8192m`.

Use this for fine-grained control when you know exactly how much memory to allocate:

```bash
# Allocate exactly 16 GiB for KV cache
vllm serve mymodel --kv-cache-memory-bytes 16g
```

### `num_gpu_blocks_override`

```
Type:    int | None
Default: None
CLI:     --num-gpu-blocks-override
```

Override the profiled GPU block count. Primarily used for testing preemption behavior. Not recommended for production use.

---

## Block Size

### `block_size`

```
Type:    Literal[1, 8, 16, 32, 64, 128, 256]
Default: platform-dependent (typically 16 or 32)
CLI:     --block-size
```

Size of a contiguous KV cache block in tokens. The KV cache is managed in fixed-size blocks (similar to virtual memory pages).

**Trade-offs:**
- **Smaller blocks** (e.g., 16) — Less internal fragmentation, better memory utilization for short sequences
- **Larger blocks** (e.g., 128) — Better throughput for long sequences, less block management overhead

The default is set by the platform's `check_and_update_config()` method based on the hardware and attention backend.

---

## KV Cache Data Type

### `cache_dtype`

```
Type:    Literal["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla"]
Default: "auto"
CLI:     --kv-cache-dtype
```

Data type for KV cache storage:

| Value | Description | Platform |
|---|---|---|
| `auto` | Same as model dtype | All |
| `bfloat16` | BF16 storage | All |
| `fp8` | FP8 (alias for `fp8_e4m3`) | CUDA 11.8+, ROCm |
| `fp8_e4m3` | FP8 E4M3 format | CUDA 11.8+, ROCm |
| `fp8_e5m2` | FP8 E5M2 format | CUDA 11.8+ |
| `fp8_inc` | FP8 for Intel Gaudi (HPU) | Intel Gaudi |
| `fp8_ds_mla` | FP8 for DeepSeek MLA | CUDA |

FP8 KV cache reduces memory usage by ~50% compared to FP16/BF16, enabling larger batch sizes or longer contexts. Some accuracy loss may occur without proper scaling factors.

```bash
# FP8 KV cache for 2× memory efficiency
vllm serve mymodel --kv-cache-dtype fp8
```

### `calculate_kv_scales`

```
Type:    bool
Default: False
CLI:     --calculate-kv-scales
```

Dynamically calculate `k_scale` and `v_scale` when `kv_cache_dtype` is FP8. If `False`, scales are loaded from the model checkpoint (if available) or default to 1.0.

Enable this when the model checkpoint doesn't include FP8 KV cache scales:

```bash
vllm serve mymodel \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales
```

---

## Prefix Caching

Prefix caching (Automatic Prefix Caching / APC) reuses KV cache blocks for common prompt prefixes across requests, dramatically improving throughput for workloads with shared prefixes (e.g., system prompts, few-shot examples).

### `enable_prefix_caching`

```
Type:    bool
Default: True (enabled by default in V1)
CLI:     --enable-prefix-caching / --no-enable-prefix-caching
```

Enable automatic prefix caching. When enabled, KV cache blocks for repeated prompt prefixes are reused across requests.

**Best for:**
- Chat applications with long system prompts
- Few-shot inference with shared examples
- Document Q&A with repeated context

```bash
# Explicitly enable (default in V1)
vllm serve mymodel --enable-prefix-caching

# Disable
vllm serve mymodel --no-enable-prefix-caching
```

### `prefix_caching_hash_algo`

```
Type:    Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
Default: "sha256"
CLI:     --prefix-caching-hash-algo
```

Hash algorithm for identifying cacheable prefix blocks:

| Algorithm | Speed | Security | Notes |
|---|---|---|---|
| `sha256` | Moderate | Cryptographic | Default. Most secure. |
| `sha256_cbor` | Moderate | Cryptographic | Cross-language reproducible (canonical CBOR serialization). |
| `xxhash` | Fast | Non-cryptographic | Requires `xxhash` package. Risk of hash collisions. |
| `xxhash_cbor` | Fast | Non-cryptographic | Reproducible + fast. Requires `xxhash` package. |

!!! warning
    Non-cryptographic hash algorithms (`xxhash*`) theoretically increase the risk of hash collisions, which could cause undefined behavior or information leakage in multi-tenant environments. Evaluate your security requirements before using them.

---

## Mamba Cache (Hybrid Models)

For hybrid Mamba/attention models (e.g., Jamba, Zamba), vLLM maintains a separate Mamba state cache.

### `mamba_cache_dtype`

```
Type:    Literal["auto", "float32", "float16"]
Default: "auto"
CLI:     --mamba-cache-dtype
```

Data type for the Mamba cache (both conv state and SSM state). `"auto"` infers from the model config.

### `mamba_ssm_cache_dtype`

```
Type:    Literal["auto", "float32", "float16"]
Default: "auto"
CLI:     --mamba-ssm-cache-dtype
```

Data type for the Mamba SSM state only. Overrides `mamba_cache_dtype` for the SSM state. Conv state still uses `mamba_cache_dtype`.

### `mamba_block_size`

```
Type:    int | None
Default: None
CLI:     --mamba-block-size
```

Block size for the Mamba cache in tokens. Must be a multiple of 8 (to align with `causal_conv1d` kernel requirements). Can only be set when prefix caching is enabled.

### `mamba_cache_mode`

```
Type:    Literal["none", "all", "align"]
Default: "none"
CLI:     --mamba-cache-mode
```

Caching strategy for Mamba layers:

| Mode | Description |
|---|---|
| `none` | No Mamba state caching (default when prefix caching is disabled) |
| `all` | Cache Mamba state at every `i × block_size` position (default when prefix caching is enabled) |
| `align` | Cache only at the last token of each scheduler step and at `i × block_size` positions |

---

## KV Cache Offloading

KV cache offloading moves KV cache blocks to CPU memory when GPU memory is full, enabling longer effective context lengths at the cost of PCIe bandwidth.

### `kv_offloading_size`

```
Type:    float | None
Default: None (disabled)
CLI:     --kv-offloading-size
```

Size of the KV cache offloading buffer in GiB. When TP > 1, this is the total buffer size across all TP ranks. Setting this enables KV cache offloading.

```bash
# Offload up to 32 GiB of KV cache to CPU
vllm serve mymodel --kv-offloading-size 32
```

### `kv_offloading_backend`

```
Type:    Literal["native", "lmcache"]
Default: "native"
CLI:     --kv-offloading-backend
```

Backend for KV cache offloading:
- `native` — vLLM's built-in CPU offloading
- `lmcache` — LMCache backend (requires LMCache installation)

---

## KV Sharing

### `kv_sharing_fast_prefill`

```
Type:    bool
Default: False
CLI:     --kv-sharing-fast-prefill
```

Enable fast-prefill optimization for KV-sharing model architectures (e.g., YOCO — You Only Cache Once). In these architectures, some layers can skip tokens during prefill. This flag enables the necessary attention metadata overrides.

!!! note
    This feature is work in progress. No prefill optimization currently takes place with this flag enabled.

---

## CPU Backend KV Cache

### `cpu_kvcache_space_bytes`

```
Type:    int | None
Default: None (4 GiB)
CLI:     (set via VLLM_CPU_KVCACHE_SPACE env var)
```

CPU key-value cache space in bytes (CPU backend only). Defaults to 4 GiB when not set.

---

## Memory Sizing Guide

### Estimating KV cache memory

The KV cache memory per token per layer is:

```
bytes_per_token_per_layer = 2 × num_kv_heads × head_dim × dtype_bytes
```

For a Llama-3.1-8B model (32 KV heads, 128 head dim, FP16):
```
= 2 × 8 × 128 × 2 = 4,096 bytes per token per layer
× 32 layers = 131,072 bytes per token total
≈ 128 KB per token
```

With 40 GiB GPU and 30 GiB for weights:
```
Available for KV cache ≈ 10 GiB = 10,737,418,240 bytes
Max tokens ≈ 10,737,418,240 / 131,072 ≈ 81,920 tokens
```

### Practical configurations

```bash
# Maximize KV cache (high throughput)
vllm serve mymodel \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching

# Conservative (leave headroom for other processes)
vllm serve mymodel \
  --gpu-memory-utilization 0.75 \
  --kv-cache-memory-bytes 8g

# Long context with CPU offloading
vllm serve mymodel \
  --gpu-memory-utilization 0.90 \
  --kv-offloading-size 64 \
  --max-model-len 1000000
```
