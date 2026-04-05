# Flex Attention Backend

The Flex Attention backend (`vllm/v1/attention/backends/flex_attention.py`) wraps PyTorch's `torch.nn.attention.flex_attention` API, enabling custom attention mask and score modification functions. It is particularly useful for models with non-standard attention patterns such as document-level masking, sliding window, or multimodal prefix attention.

## Overview

PyTorch's FlexAttention allows users to define custom `mask_mod` and `score_mod` functions that are JIT-compiled into efficient Triton kernels. vLLM's `FlexAttentionBackend` integrates this with the paged KV cache system.

```python
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks,
)

# Pre-compiled versions
create_block_mask_compiled = torch.compile(create_block_mask, fullgraph=True, mode="reduce-overhead")
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)
```

## Backend Class

```python
class FlexAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]
```

### Supported Configurations

| Property | Value |
|---|---|
| Head sizes | All (no restriction) |
| Block sizes | Any |
| KV cache dtypes | `auto`, `bfloat16` (no FP8) |
| Attention types | Decoder, Encoder-only |
| Cascade attention | Not supported |
| Multimodal prefix | Yes |

### KV Cache Shape

```python
# Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
```

## Metadata

```python
@dataclass
class FlexAttentionMetadata:
    causal: bool
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # Cascade attention
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Block mapping
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor  # [max_reqs, total_blocks]
    decode_offset: torch.Tensor
    num_blocks_per_seq: torch.Tensor

    # FlexAttention-specific
    block_mask: BlockMask | None = None
    score_mod: _score_mod_signature | None = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod
    doc_ids: torch.Tensor | None = None
    q_block_size: int = 16
    kv_block_size: int = 16
    sliding_window: int | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
```

## Physical-to-Logical Block Mapping

FlexAttention operates on **logical** token indices, but vLLM's paged KV cache uses **physical** block indices. The backend maintains a `physical_to_logical` mapping tensor of shape `[max_reqs, total_blocks]`:

```python
def physical_to_logical_mapping(
    block_table: torch.Tensor,  # [max_reqs, max_num_blocks] — logical → physical
    seq_lens: torch.Tensor,
    block_size: int,
    total_blocks: int,
) -> torch.Tensor:
    """Creates inverse mapping: physical block → logical block index.
    Returns -1 for unused physical blocks."""
```

This inverse mapping is needed because FlexAttention's `mask_mod` receives logical indices, but the KV cache is addressed by physical block.

### Garbage Value Protection

The block table may contain garbage values in unused positions. The mapping function masks these out using `seq_lens` and `block_size` to ensure only valid block references are processed.

### Reused Physical Blocks

For sliding window or hybrid attention, the same physical block may appear at multiple logical positions. The mapping uses `scatter_reduce_` with `reduce="amax"` to keep the **latest** (maximum) logical index for each physical block.

## Mask Mod Functions

The backend builds `BlockMask` objects using composed mask functions:

### Default Causal Mask

```python
def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```

### Sliding Window Mask

```python
def sliding_window_mask_mod(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx) <= sliding_window
```

### Document Mask (Batch Packing)

For packed batches where multiple documents are concatenated, a document ID mask ensures tokens only attend within their document:

```python
def document_mask_mod(b, h, q_idx, kv_idx):
    return doc_ids[q_idx] == doc_ids[kv_idx]
```

### Multimodal Prefix Mask

For models where image tokens receive full (non-causal) attention:

```python
def mm_prefix_mask_mod(b, h, q_idx, kv_idx):
    # Image tokens: full attention
    # Text tokens: causal attention
    ...
```

Masks are composed using `and_masks()` and `or_masks()`:

```python
final_mask = and_masks(causal_mask_mod, sliding_window_mask_mod)
```

## Forward Pass

The FlexAttention forward pass:

1. **Build BlockMask**: `create_block_mask_compiled(mask_mod, B, H, Q_LEN, KV_LEN, ...)`
2. **Run attention**: `flex_attention_compiled(query, key, value, block_mask=block_mask, score_mod=score_mod)`

The key challenge is that FlexAttention expects a **dense** KV tensor, but vLLM uses a paged KV cache. The backend handles this by:
- Gathering KV entries from the paged cache using the physical-to-logical mapping
- Passing the gathered dense KV to `flex_attention`

## Score Mod Functions

Score mods modify attention logits before softmax. Common uses:

- **ALiBi**: `score += alibi_slopes[h] * (q_idx - kv_idx)`
- **Soft-cap**: `score = tanh(score / cap) * cap`
- **Temperature scaling**: `score *= temperature`

## Compilation

FlexAttention uses `torch.compile` with `fullgraph=True` and `mode="reduce-overhead"`. The recompile limit is set to 16 to handle different mask configurations:

```python
torch._dynamo.config.recompile_limit = 16
```

## Limitations

- **No FP8 KV cache**: Only `auto` and `bfloat16` KV cache dtypes
- **No cascade attention**: `use_cascade_attention()` always returns `False`
- **No MLA**: Standard attention only
- **Compilation overhead**: First call triggers JIT compilation

## Use Cases

FlexAttention is most useful for:

1. **Custom positional encodings** (e.g., ALiBi with custom slopes)
2. **Document-level attention** in batch-packed training
3. **Sliding window + global attention** combinations
4. **Multimodal models** with mixed causal/non-causal attention
5. **Research models** with experimental attention patterns

## See Also

- [Backend Selection](./backend-selection.md) — When FlexAttention is chosen
- [FlashAttention Backend](./flash-attn.md) — Higher-performance standard attention
- [Triton Attention](./triton-attn.md) — Triton-based alternative with similar flexibility
