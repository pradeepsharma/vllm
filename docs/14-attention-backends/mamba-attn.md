# Mamba SSM Backends

vLLM supports State Space Models (SSMs) through a family of Mamba attention backends. These backends do not perform traditional attention — instead they manage the recurrent state (SSM state) for Mamba1, Mamba2, and related architectures.

## Overview

Mamba backends are selected via `get_mamba_attn_backend()` in `selector.py`, which maps the model's `mamba_type` string to a `MambaAttentionBackendEnum` member:

```python
MAMBA_TYPE_TO_BACKEND_MAP = {
    "mamba1": "MAMBA1",
    "mamba2": "MAMBA2",
    "short_conv": "SHORT_CONV",
    "linear_attention": "LINEAR",
    "gdn_attention": "GDN_ATTN",
    "custom": "CUSTOM",
}
```

## Base Class: `BaseMambaAttentionMetadata`

All Mamba backends share a common metadata structure defined in `mamba_attn.py`:

```python
@dataclass
class BaseMambaAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # Prefill-only tensors (None if no prefill)
    has_initial_states_p: torch.Tensor | None   # Which seqs have cached states
    query_start_loc_p: torch.Tensor | None       # Cumulative token offsets
    num_computed_tokens_p: torch.Tensor | None   # Tokens already computed
    state_indices_tensor_p: torch.Tensor | None  # State slot indices

    # Decode-only tensors (None if no decode)
    state_indices_tensor_d: torch.Tensor | None  # State slot indices
    query_start_loc_d: torch.Tensor | None       # [num_decodes + 1]

    # Speculative decoding
    num_accepted_tokens: torch.Tensor | None     # [batch]

    # Prefix caching (mamba_cache_mode="all")
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None

    seq_lens: torch.Tensor

    # Chunked prefill
    cu_chunk_seqlen_p: torch.Tensor | None       # [nchunks+1]
    last_chunk_indices_p: torch.Tensor | None    # [batch]

    # Triton causal_conv1d metadata
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None
```

## Base Builder: `BaseMambaAttentionMetadataBuilder`

The base builder in `mamba_attn.py` handles:

### Batch Reordering

Mamba backends reorder the batch so decode requests come before prefill requests. The `reorder_batch_threshold` controls when this happens (default: 1, meaning always reorder when there are any decodes).

### State Index Management

Each sequence has a "state slot" in the SSM state cache. The builder computes `state_indices_tensor_d` and `state_indices_tensor_p` to map sequences to their state slots.

For `mamba_cache_mode="all"` (full prefix caching), state indices are 2D tensors `[num_seqs, max_num_blocks]` tracking which blocks contain valid state.

### Chunked Prefill

The builder computes chunk metadata ensuring:
1. Each chunk contains tokens from a **single sequence only**
2. For every sequence, the Mamba state can be retrieved every `chunk_size` tokens

```python
def _compute_chunk_metadata(self, chunk_size, num_prefills, ...):
    # Splits sequences into chunks at both sequence boundaries
    # and physical chunk boundaries
    cu_chunk_seqlen = []
    seq_idx = []
    last_chunk_indices = []
    ...
```

### CUDA Graph Support

Mamba backends use `AttentionCGSupport.UNIFORM_BATCH` — CUDA graphs are captured for uniform decode-only batches. Mixed prefill-decode batches run in eager mode.

---

## Mamba1 Backend

**File:** `vllm/v1/attention/backends/mamba1_attn.py`

`Mamba1AttentionBackend` is a thin wrapper around the base builder. For `mamba_cache_mode="all"`, it computes chunk metadata using the KV cache block size as the chunk size:

```python
class Mamba1AttentionMetadataBuilder(BaseMambaAttentionMetadataBuilder):
    metadata_cls = Mamba1AttentionMetadata

    def build(self, common_prefix_len, common_attn_metadata, ...):
        common = self._compute_common_metadata(common_attn_metadata)

        if common.num_prefills > 0 and mamba_cache_mode == "all":
            cu_chunk_seqlen_p, _, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.kv_cache_spec.block_size,  # chunk_size = block_size
                    common,
                    common_attn_metadata,
                )
            )
            return replace(common, cu_chunk_seqlen_p=..., last_chunk_indices_p=...)

        return common
```

Mamba1 uses the block size as the chunk size because the SSM state is stored per-block.

---

## Mamba2 Backend

**File:** `vllm/v1/attention/backends/mamba2_attn.py`

`Mamba2AttentionBackend` extends the base with Mamba2-specific chunk metadata. Mamba2 uses a fixed `chunk_size` from the model config (typically 256).

### Additional Metadata

```python
@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False  # Whether to prepare initial states
    chunk_size: int = 0                # From model config
    seq_idx_p: torch.Tensor | None = None  # Sequence index per chunk
```

### Chunk Metadata Computation

Mamba2 uses `compute_varlen_chunk_metadata()` to build chunk-aligned metadata:

```python
def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      cu_chunk_seqlens:    (nchunks+1,) int32 — exclusive prefix-sum of chunk lengths
      last_chunk_indices:  (B,) int32 — index of last chunk per sequence
      seq_idx_chunks:      (nchunks,) int32 — sequence index per chunk
    """
```

The function splits sequences at both sequence boundaries and physical chunk boundaries, ensuring no chunk crosses a sequence boundary.

### Initial State Preparation

When `prep_initial_states=True`, the Mamba2 kernel needs to load the SSM state from the previous iteration before processing the current chunk. This happens when any sequence in the prefill batch has cached states (`has_initial_states_p` is True for at least one sequence).

---

## Speculative Decoding Support

All Mamba backends support speculative decoding. When `num_spec_tokens > 0`:

- `state_indices_tensor_d` has shape `[num_decodes, 1 + num_spec_tokens]` to track the state for each speculative token
- `num_accepted_tokens` tensor tracks how many tokens were accepted per sequence
- `supports_update_block_table = False` (block table updates are disabled during spec decode)

```python
if self.num_spec_tokens > 0:
    self.decode_num_accepted_tokens = torch.empty(
        (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
    )
    self.supports_update_block_table = False
```

---

## Prefix Caching (`mamba_cache_mode`)

The `mamba_cache_mode` config controls how SSM states are cached:

| Mode | Description |
|---|---|
| `"none"` (default) | No prefix caching; state is recomputed each time |
| `"all"` | Full prefix caching; state is stored per-block and reused |

In `"all"` mode, the builder tracks:
- `block_idx_last_scheduled_token` — which block contains the last scheduled token
- `block_idx_last_computed_token` — which block contains the last computed token
- `block_idx_first_scheduled_token_p` — first scheduled token block for prefill

---

## Other SSM Backends

### SHORT_CONV

**File:** `vllm/v1/attention/backends/short_conv_attn.py`

Used for models with short causal convolution layers (e.g., Hawk/Griffin). Manages 1D convolution state rather than SSM state.

### LINEAR

**File:** `vllm/v1/attention/backends/linear_attn.py`

Used for linear attention variants. Manages the linear attention state (key-value outer product accumulator).

### GDN_ATTN

**File:** `vllm/v1/attention/backends/gdn_attn.py`

Used for GDN (Generalized Divisive Normalization) attention variants.

---

## Hybrid Models

For models that mix attention and Mamba layers (e.g., Jamba, Zamba), vLLM uses both an attention backend and a Mamba backend simultaneously. The scheduler assigns each layer to the appropriate backend based on its type.

The `HybridAttentionMambaModelConfig` in the model config handles the block size initialization for hybrid models, ensuring compatibility between the attention and Mamba KV cache layouts.

## See Also

- [Backend Selection](./backend-selection.md) — Mamba backend selection
- [MLA Backends](./mla-attn.md) — Multi-head Latent Attention
- [FlashAttention Backend](./flash-attn.md) — Attention layers in hybrid models
