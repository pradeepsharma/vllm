# Punica Kernels

vLLM's multi-LoRA serving is powered by the **Punica** kernel system, which implements batched GEMM operations that can simultaneously apply different LoRA adapters to different tokens within the same forward pass. This is the key innovation that makes serving many adapters efficiently possible without separate forward passes per adapter.

The implementation is based on the paper:
> Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). **Punica: Multi-Tenant LoRA Serving.** https://arxiv.org/abs/2310.18547

Source files: `vllm/lora/punica_wrapper/` and `vllm/lora/ops/`.

## The Core Problem

Standard LoRA inference applies a single adapter to all tokens:
```
output += x @ A @ B * scale
```

Multi-LoRA serving requires applying **different** adapters to different tokens in the same batch:
```
for token i:
    output[i] += x[i] @ A[lora_id[i]] @ B[lora_id[i]] * scale
```

Doing this naively with separate matrix multiplications per adapter would be extremely slow. Punica solves this with a custom Triton kernel that processes all adapters in a single fused operation.

## Two-Step LoRA Computation

The LoRA delta computation is split into two steps:

### Step 1: Shrink (lora_a projection)

```
y[i] += x[i] @ lora_a[lora_id[i]] * scale
```

Projects the input from `d_in` dimensions down to `rank` dimensions. This is the "compression" step.

### Step 2: Expand (lora_b projection)

```
output[i] += y[i] @ lora_b[lora_id[i]]
```

Projects from `rank` dimensions back up to `d_out` dimensions. This is the "expansion" step.

The two-step design allows the intermediate `rank`-dimensional buffer to be shared across tensor-parallel ranks (all-gathered after the shrink step for column-parallel layers).

## `PunicaWrapperBase`

Defined in `vllm/lora/punica_wrapper/punica_base.py`. The abstract base class that all platform-specific wrappers implement.

### State Tensors

The wrapper maintains per-batch metadata tensors:

```python
class PunicaWrapperBase(PunicaWrapperABC):
    def __init__(self, max_num_batched_tokens, max_batches, device, **kwargs):
        self._token_lora_indices = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        self._sampler_indices = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        self._sampler_indices_padded = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        self._embeddings_indices = torch.empty(
            2, max_num_batched_tokens, dtype=torch.long, device=device
        )
```

These tensors are updated each forward pass via `update_metadata()` and tell the kernel which adapter slot to use for each token.

### Abstract Interface

```python
class PunicaWrapperABC(ABC):
    @abstractmethod
    def update_metadata(
        self, mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int, vocab_size: int,
    ) -> None:
        """Update the lora-related metadata"""

    @abstractmethod
    def add_shrink(
        self, y, x, lora_a_stacked, scale, **kwargs
    ) -> torch.Tensor | None:
        """Performs GEMM for multiple slices of lora_a.
        Semantics: y[i] += (x @ lora_a_stacked[i]) * scale
        """

    @abstractmethod
    def add_expand(
        self, y, x, lora_b_stacked, output_slices,
        offset_start=0, add_inputs=True, **kwargs
    ) -> torch.Tensor | None:
        """Performs GEMM for multiple slices of lora_b."""

    @abstractmethod
    def add_lora_embedding(
        self, y, x, lora_b_stacked, add_inputs=True, **kwargs
    ) -> torch.Tensor | None:
        """Applies lora for VocabParallelEmbeddingWithLoRA."""

    @abstractmethod
    def add_lora_linear(
        self, y, x, lora_a_stacked, lora_b_stacked,
        scale, output_slices, *, buffer=None, **kwargs
    ) -> torch.Tensor | None:
        """Applicable to linear-related lora."""

    @abstractmethod
    def add_lora_logits(
        self, y, x, lora_a_stacked, lora_b_stacked, scale,
        *, buffer=None, **kwargs
    ) -> torch.Tensor | None:
        """Applies lora for LogitsProcessorWithLoRA."""
```

## `PunicaWrapperGPU`

Defined in `vllm/lora/punica_wrapper/punica_gpu.py`. The GPU implementation using Triton kernels.

### Initialization

```python
@final
class PunicaWrapperGPU(PunicaWrapperBase):
    def __init__(self, max_num_batched_tokens, max_batches, device, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        self.lora_config = kwargs["lora_config"]
        self.max_loras = self.lora_config.max_loras

        # Compute captured LoRA counts for cudagraph specialization
        captured_lora_counts = get_captured_lora_counts(
            self.max_loras, self.lora_config.specialize_active_lora
        )

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens,
            device=device, captured_lora_counts=captured_lora_counts,
        )
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens,
            device=device, captured_lora_counts=captured_lora_counts,
        )
```

### `add_shrink`

```python
def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
    """
    Semantics:
    for i in range(len(lora_a_stacked)):
        y[i] += (x @ lora_a_stacked[i]) * scale
    """
    x = x.view(-1, x.shape[-1])
    lora_shrink(
        x, lora_a_stacked, y,
        *self.token_mapping_meta.meta_args(
            x.size(0), self.lora_config.specialize_active_lora
        ),
        scale,
    )
```

### `update_metadata`

```python
def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, **kwargs):
    self.is_prefill = mapping.is_prefill
    self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
    # Prepare cuda kernel metadata tensors
    self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
    self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
```

## Triton Kernel: `_lora_shrink_kernel`

The shrink kernel is defined in `vllm/lora/ops/triton_ops/lora_shrink_op.py`:

```python
@triton.jit
def _lora_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    M, N, K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    # strides...
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
```

Key design features:
- **Sorted token dispatch**: Tokens are sorted by their LoRA ID so that all tokens for the same adapter are processed contiguously, maximizing cache efficiency.
- **Early exit**: CTAs (thread blocks) for inactive adapters (`lora_id == -1`) exit immediately.
- **Column-major ordering**: Within groups, CTAs are ordered column-major for better L2 cache reuse.
- **Split-K**: Supports split-K reduction for better parallelism on small M (decode phase).

## Triton Kernel: `_lora_expand_kernel`

The expand kernel is defined in `vllm/lora/ops/triton_ops/lora_expand_op.py`:

```python
@triton.jit
def _lora_expand_kernel(
    input_ptr, lora_ptr, out_ptr,
    M, N, K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_loc,
    # strides...
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
```

The expand kernel adds the LoRA output to the existing base model output (`ADD_INPUTS=True`), implementing the residual connection.

## `LoRAKernelMeta`

Defined in `vllm/lora/ops/triton_ops/lora_kernel_metadata.py`. Manages the metadata tensors that the Triton kernels need:

- `token_indices_sorted_by_lora_ids`: Token indices sorted by their LoRA assignment, enabling contiguous memory access per adapter.
- `num_tokens_per_lora`: Number of tokens assigned to each adapter slot.
- `lora_token_start_loc`: Starting offset in the sorted token array for each adapter.
- `lora_ids`: The active adapter slot IDs.

These tensors are prepared once per forward pass in `prepare_tensors()`.

## MoE LoRA Kernels

For Mixture-of-Experts models, dedicated fused kernels handle the expert routing and LoRA computation together:

```python
# vllm/lora/ops/triton_ops/fused_moe_lora_op.py
# vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
```

These kernels fuse the MoE expert dispatch with the LoRA shrink/expand operations, avoiding separate passes over the expert weights.

## Platform-Specific Implementations

The Punica wrapper is selected at runtime based on the current platform:

```python
def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    punica_wrapper_qualname = current_platform.get_punica_wrapper()
    punica_wrapper_cls = resolve_obj_by_qualname(punica_wrapper_qualname)
    return punica_wrapper_cls(*args, **kwargs)
```

| Platform | Implementation | File |
|----------|---------------|------|
| NVIDIA GPU | `PunicaWrapperGPU` | `punica_gpu.py` |
| Intel XPU | `PunicaWrapperXPU` | `punica_xpu.py` |
| CPU | `PunicaWrapperCPU` | `punica_cpu.py` |

## CUDA Graph Specialization

When `specialize_active_lora=True` in `LoRAConfig`, separate CUDA graphs are captured for different numbers of active LoRA adapters:

```python
def get_captured_lora_counts(max_loras: int, specialize: bool) -> list[int]:
    if not specialize:
        return [max_loras + 1]
    # Powers of 2 up to max_loras, plus max_loras + 1
    return [
        n for n in range(1, max_loras + 2)
        if (n & (n - 1)) == 0 or n == max_loras + 1
    ]
```

For example, with `max_loras=8` and `specialize=True`, graphs are captured for 1, 2, 4, 8, and 9 active adapters. The kernel dispatches to the appropriate graph based on the actual number of active adapters in the current batch.

## Performance Characteristics

| Scenario | Performance Notes |
|----------|------------------|
| Single adapter, many tokens | Near-identical to standard LoRA inference |
| Multiple adapters, balanced load | Efficient batched GEMM, minimal overhead |
| Many adapters, few tokens each | Overhead from metadata preparation; consider increasing batch size |
| Decode phase (1 token/request) | Split-K helps; consider `specialize_active_lora=True` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_LORA_DISABLE_PDL` | `0` | Disable Persistent Data Loading (PDL) in Triton kernels |

## See Also

- [LoRA Layers](lora-layers.md) — How layers call the Punica wrapper
- [LoRAModelManager](lora-model-manager.md) — Adapter slot management
- [LoRAConfig](lora-config.md) — `specialize_active_lora` and other options
