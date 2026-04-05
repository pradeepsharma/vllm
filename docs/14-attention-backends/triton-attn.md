# Triton Attention Backend

The Triton attention backend (`vllm/v1/attention/backends/triton_attn.py`) provides a pure-Triton implementation of paged attention that works across CUDA and ROCm platforms without requiring FlashAttention or FlashInfer. It is the default fallback on ROCm and is also used on CUDA when neither FlashAttention nor FlashInfer is available.

## Overview

The backend uses two custom Triton kernels from `vllm/v1/attention/ops/`:

- **`triton_unified_attention.py`** — `unified_attention()`: handles both prefill and decode in a single kernel dispatch, selecting between a 2D and 3D kernel based on batch size
- **`triton_prefill_attention.py`** — `context_attention_fwd()`: used for the prefix pass in cascade attention
- **`triton_reshape_and_cache_flash.py`** — `triton_reshape_and_cache_flash()`: writes new K/V tokens into the paged cache

## Backend Class

```python
class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"]
    forward_includes_kv_cache_update: bool = False
```

### Supported Configurations

| Property | Value |
|---|---|
| Head sizes | `head_size >= 32` (no upper limit) |
| Block sizes | Multiples of 16 |
| KV cache dtypes | `auto`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| Compute capability | All (no restriction) |
| Attention types | Decoder, Encoder, Encoder-only, Encoder-Decoder |
| Attention sinks | Yes |
| ALiBi sqrt | Yes |
| Multimodal prefix | Yes |

The Triton backend is the most permissive in terms of hardware support — it works on any GPU that supports Triton.

### KV Cache Shape

```python
# Shape: (num_blocks, 2, block_size, num_kv_heads, head_size)
# Note: blocks-first layout (same as FlashInfer)
```

## 2D vs 3D Kernel Dispatch

The unified attention kernel has two variants optimized for different batch sizes:

### 3D Kernel (small batches)

Used when `num_seqs < seq_threshold_3D`. The 3D kernel launches a grid of `(num_seqs, num_heads_q, num_par_softmax_segments)` and uses parallel softmax segments to handle long sequences efficiently.

```python
# Pre-allocated buffers for 3D kernel
self.softmax_segm_output  # [seq_threshold_3D, num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, headdim_padded]
self.softmax_segm_max     # [seq_threshold_3D, num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS]
self.softmax_segm_expsum  # [seq_threshold_3D, num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS]
```

`NUM_PAR_SOFTMAX_SEGMENTS = 16` by default.

### 2D Kernel (large batches)

Used when `num_seqs >= seq_threshold_3D`. The 2D kernel launches a grid of `(num_q_blocks, num_kv_heads)` and is more efficient when the batch is large enough to saturate the GPU.

The threshold is computed as:

```python
seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
# MIN_LAUNCH_GRID_SIZE_2D = 128
```

When CUDA graphs are enabled, the threshold is snapped to the nearest captured graph size to ensure each graph covers the correct execution path.

## Metadata

```python
@dataclass
class TritonAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # 2D/3D dispatch parameters
    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # Cascade attention
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Multimodal prefix ranges
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
```

### Multimodal Prefix Range

The `mm_prefix_range` field supports models where image tokens require full (non-causal) attention while text tokens use causal attention. It maps sequence indices to lists of `(start, end)` token ranges that should receive full attention.

The `mm_prefix_range_tensor` property converts this dict to a padded tensor of shape `(num_seqs, max_ranges, 2)` for efficient Triton kernel access.

## Forward Pass

```python
unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    alibi_slopes=self.alibi_slopes,
    logits_soft_cap=self.logits_soft_cap,
    block_table=block_table,
    sliding_window=self.sliding_window,
    seq_threshold_3D=seq_threshold_3D,
    num_par_softmax_segments=num_par_softmax_segments,
    softmax_segm_output=softmax_segm_output,
    softmax_segm_max=softmax_segm_max,
    softmax_segm_expsum=softmax_segm_expsum,
    k_scale=layer._k_scale_float,
    v_scale=layer._v_scale_float,
    mm_prefix_range=mm_prefix_range_tensor,
    sinks=self.sinks,
    use_alibi_sqrt=self.use_alibi_sqrt,
)
```

### FP8 KV Cache

When `kv_cache_dtype` starts with `"fp8"`, the key and value caches are reinterpreted as the platform's FP8 dtype:

```python
key_cache = key_cache.view(self.fp8_dtype)
value_cache = value_cache.view(self.fp8_dtype)
```

Note: The Triton backend currently requires `q_scale == 1.0` (no query quantization).

### Encoder Attention

For encoder-only and encoder-decoder models, the forward pass skips the KV cache and calls `context_attention_fwd()` directly with the Q, K, V tensors.

## CUDA Graph Support

The Triton backend supports `AttentionCGSupport.ALWAYS` — full CUDA graphs for all batch shapes. During graph capture, `seq_lens` is set to 1 to avoid slow graph capture with large sequence lengths:

```python
def build_for_cudagraph_capture(self, common_attn_metadata):
    attn_metadata = self.build(0, common_attn_metadata)
    attn_metadata.seq_lens.fill_(1)  # Avoid slow capture
    return attn_metadata
```

## Triton Kernel Details

### `unified_attention` (`triton_unified_attention.py`)

The kernel implements:
- Causal and non-causal attention
- Sliding window attention
- ALiBi positional biases (including sqrt variant)
- Logits soft-capping (Gemma-style)
- FP8 dequantization
- Attention sinks
- Multimodal prefix full-attention ranges

Key Triton JIT functions:
- `kernel_unified_attention_2d` — 2D grid kernel for large batches
- `kernel_unified_attention_3d` — 3D grid kernel for small batches
- `apply_softcap(S, x)` — tanh-based soft cap: `x * tanh(S/x)`
- `find_seq_idx(...)` — binary search for sequence index

### `context_attention_fwd` (`triton_prefill_attention.py`)

Used for the prefix pass in cascade attention. Implements standard varlen attention without paged KV cache.

### `triton_reshape_and_cache_flash` (`triton_reshape_and_cache_flash.py`)

Writes new K/V tokens into the paged KV cache using slot mapping:

```python
triton_reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping,
    kv_cache_dtype, k_scale, v_scale
)
```

## ROCm-Specific Notes

On ROCm, the Triton backend is the primary fallback when AITER ops are not available. The `rocm_aiter_ops` import is present in the file but only used when AITER is enabled. The Triton kernels themselves are platform-agnostic and compile for both CUDA and HIP.

## See Also

- [Backend Selection](./backend-selection.md) — When Triton attention is chosen
- [ROCm Backends](./rocm-attn.md) — ROCm-specific backends that extend Triton
- [FlashAttention Backend](./flash-attn.md) — Higher-performance CUDA alternative
