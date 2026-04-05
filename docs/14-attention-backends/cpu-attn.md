# CPU Attention Backend

The CPU attention backend (`vllm/v1/attention/backends/cpu_attn.py`) enables vLLM to run inference on CPU-only systems. It is the only backend available on the CPU platform and is automatically selected — no configuration is needed.

## Overview

The CPU backend uses vLLM's custom C++ `cpu_attention_with_kv_cache` op for decode and optionally PyTorch's `scaled_dot_product_attention` (SDPA) for prefill on non-x86/ARM/s390x architectures.

## Backend Class

```python
class CPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
```

### Supported Configurations

| Property | Value |
|---|---|
| Head sizes | 32, 64, 80, 96, 112, 128, 160, 192, 224, 256 |
| KV cache dtypes | `auto`, `bfloat16` (no FP8) |
| Attention types | Decoder, Encoder, Encoder-only, Encoder-Decoder |
| Cascade attention | Not supported |
| MLA | Not supported |
| Sparse attention | Not supported |

### KV Cache Shape

```python
# Shape: (2, num_blocks, num_kv_heads, block_size, head_size)
# Note: different dimension ordering from GPU backends
```

## ISA Selection

The CPU backend selects the optimal instruction set architecture (ISA) at runtime based on the platform and configuration:

```python
def _get_attn_isa(dtype, block_size, head_size=None) -> str:
    if head_size % 16 == 0 and head_size % 32 != 0:
        return "vec16"
    if supports_amx and dtype == bfloat16 and block_size % 32 == 0:
        return "amx"      # Intel AMX (Advanced Matrix Extensions)
    elif block_size % 32 == 0:
        if supports_arm:
            return "neon"  # ARM NEON FMLA / BFMMLA
        elif supports_vxe:
            return "vxe"   # IBM s390x VXE
        else:
            return "vec"   # Generic AVX-512 / AVX2
    else:
        return "vec16"     # 16-element vector fallback
```

| ISA | Platform | Condition |
|---|---|---|
| `amx` | x86 with Intel AMX | `bfloat16`, `block_size % 32 == 0` |
| `neon` | ARM | `block_size % 32 == 0` |
| `vxe` | IBM s390x | `block_size % 32 == 0` |
| `vec` | x86 (AVX-512/AVX2) | `block_size % 32 == 0` |
| `vec16` | Any | `head_size % 16 == 0` and `head_size % 32 != 0`, or small blocks |

## Prefill vs Decode Split

The CPU backend has two strategies depending on the CPU architecture:

### Mixed Batch (x86, ARM, s390x)

For `_CPU_ARCH_PREFER_MIXED_BATCH` architectures, decode and prefill tokens are processed together using `cpu_attention_with_kv_cache`. No batch reordering is needed.

### Split Batch (other architectures)

For other architectures, the builder reorders the batch so decode requests come first, then uses:
- `cpu_attention_with_kv_cache` for decode tokens
- PyTorch SDPA (`scaled_dot_product_attention`) for prefill tokens

```python
if self.use_sdpa_prefill and causal:
    # Reorder: decode first, then prefill
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(common_attn_metadata, decode_threshold=1)
    )
    # SDPA handles prefill portion
    # cpu_attention_with_kv_cache handles decode portion
```

## Metadata

```python
@dataclass
class CPUAttentionMetadata:
    isa: str                          # Selected ISA string
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    scheduler_metadata: torch.Tensor | None  # From cpu_attn_get_scheduler_metadata
    causal: bool = True

    # SDPA-specific fields
    use_sdpa_prefill: bool = False
    num_decode_tokens: int = 0
    sdpa_attn_masks: list[torch.Tensor | None] | None = None
    sdpa_start_loc: torch.Tensor | None = None
```

The `scheduler_metadata` tensor is computed by `ops.cpu_attn_get_scheduler_metadata()`, which pre-computes scheduling information for the C++ attention kernel.

## Forward Pass

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, ...):
    # 1. Write new K/V to cache
    ops.cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, isa)

    # 2. SDPA for prefill (if use_sdpa_prefill)
    if attn_metadata.use_sdpa_prefill:
        self._run_sdpa_forward(query[decode_tokens:], key[decode_tokens:], ...)

    # 3. C++ kernel for decode (and prefill in mixed-batch mode)
    ops.cpu_attention_with_kv_cache(
        query=query[:num_actual_tokens],
        key_cache=key_cache,
        value_cache=value_cache,
        output=output[:num_actual_tokens],
        query_start_loc=attn_metadata.query_start_loc,
        seq_lens=attn_metadata.seq_lens,
        scale=self.scale,
        causal=attn_metadata.causal,
        alibi_slopes=self.alibi_slopes,
        sliding_window=self.sliding_window,
        block_table=attn_metadata.block_table,
        softcap=self.logits_soft_cap,
        scheduler_metadata=attn_metadata.scheduler_metadata,
        s_aux=self.sinks,
    )
```

### Encoder Attention

For encoder-only and encoder-decoder models, the forward pass uses SDPA directly without the KV cache:

```python
if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
    return self._run_sdpa_forward(query, key, value, output, attn_metadata, attn_type)
```

### SDPA Forward

The `_run_sdpa_forward()` method:
1. Builds attention masks for ALiBi or sliding window (cached in `sdpa_attn_masks`)
2. Transposes Q/K/V from `[tokens, heads, head_size]` to `[heads, tokens, head_size]`
3. Calls `torch.nn.functional.scaled_dot_product_attention()` per sequence

## Limitations

- **No FP8 KV cache**: `is_quantized_kv_cache()` raises `NotImplementedError`
- **No MLA**: CPU platform raises `NotImplementedError` for MLA
- **No sparse attention**: CPU platform raises `NotImplementedError` for sparse
- **No cascade attention**: `use_cascade_attention()` always returns `False`
- **No CUDA graphs**: CPU inference does not use CUDA graphs
- **Logits soft-cap warning**: Soft-cap is not supported for encoder/encoder-only attention types (outputs may be slightly off)

## KV Cache Space Configuration

CPU KV cache space is controlled by `VLLM_CPU_KVCACHE_SPACE` (in GiB). If not set, vLLM defaults to 50% of available NUMA node memory:

```bash
# Set 16 GiB KV cache
VLLM_CPU_KVCACHE_SPACE=16 vllm serve meta-llama/Llama-3.1-8B
```

## See Also

- [Backend Selection](./backend-selection.md) — CPU always uses CPU_ATTN
- [Hardware Guide](../09-hardware/README.md) — CPU inference setup
