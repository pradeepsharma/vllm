# Attention Backends

vLLM's attention system is built around a pluggable backend architecture that selects the most efficient attention kernel for each combination of hardware, model type, and serving configuration. This section documents every backend in detail.

## Overview

The attention backend system lives under `vllm/v1/attention/`. It consists of:

- **`selector.py`** — Public API for choosing a backend at model-load time
- **`backend.py`** — Abstract base classes (`AttentionBackend`, `AttentionImpl`, `AttentionMetadataBuilder`)
- **`backends/registry.py`** — Enum registry of all known backends with override support
- **`backends/<name>.py`** — Concrete backend implementations
- **`ops/`** — Low-level Triton and CUDA kernels shared across backends

## Pages in This Section

| Page | What It Covers |
|------|---------------|
| [Backend Selection](./backend-selection.md) | `selector.py`, `registry.py`, decision tree |
| [FlashAttention](./flash-attn.md) | FA2/FA3/FA4, prefill vs decode, cascade attention |
| [FlashInfer](./flashinfer.md) | Workspace management, paged attention, TRTLLM path |
| [Triton Attention](./triton-attn.md) | Custom Triton kernels, 2D/3D dispatch |
| [ROCm / AITER Backends](./rocm-attn.md) | ROCm-specific backends, AITER ops |
| [CPU Attention](./cpu-attn.md) | x86/ARM/s390x CPU attention |
| [Mamba SSM Backends](./mamba-attn.md) | Mamba1, Mamba2, SSM state management |
| [MLA (Multi-head Latent Attention)](./mla-attn.md) | DeepSeek MLA, FlashMLA, CutlassMLA |
| [Flex Attention](./flex-attention.md) | PyTorch FlexAttention, custom mask mods |
| [Tree Attention](./tree-attn.md) | Speculative decoding tree attention |

## Quick Reference: Backend Names

| Backend Name | Enum | Primary Use |
|---|---|---|
| `FLASH_ATTN` | `AttentionBackendEnum.FLASH_ATTN` | CUDA, default for Ampere/Hopper |
| `FLASHINFER` | `AttentionBackendEnum.FLASHINFER` | CUDA, default for Blackwell |
| `TRITON_ATTN` | `AttentionBackendEnum.TRITON_ATTN` | CUDA/ROCm fallback |
| `ROCM_ATTN` | `AttentionBackendEnum.ROCM_ATTN` | ROCm prefill-decode split |
| `ROCM_AITER_FA` | `AttentionBackendEnum.ROCM_AITER_FA` | ROCm AITER Flash Attention |
| `ROCM_AITER_UNIFIED_ATTN` | `AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN` | ROCm AITER unified |
| `CPU_ATTN` | `AttentionBackendEnum.CPU_ATTN` | CPU inference |
| `FLASHMLA` | `AttentionBackendEnum.FLASHMLA` | MLA on Hopper |
| `CUTLASS_MLA` | `AttentionBackendEnum.CUTLASS_MLA` | MLA on Blackwell |
| `FLASHINFER_MLA` | `AttentionBackendEnum.FLASHINFER_MLA` | MLA on Blackwell (low head count) |
| `TRITON_MLA` | `AttentionBackendEnum.TRITON_MLA` | MLA fallback |
| `FLEX_ATTENTION` | `AttentionBackendEnum.FLEX_ATTENTION` | Custom mask/score mods |
| `TREE_ATTN` | `AttentionBackendEnum.TREE_ATTN` | Speculative decoding |
| `MAMBA1` | `MambaAttentionBackendEnum.MAMBA1` | Mamba1 SSM |
| `MAMBA2` | `MambaAttentionBackendEnum.MAMBA2` | Mamba2 SSM |

## Cross-References

- [Compilation & Optimization](../10-compilation/README.md) — how attention backends interact with `torch.compile`
- [Hardware Support](../09-hardware/README.md) — hardware requirements for each backend
- [Quantization: KV Cache](../14-quantization/kv-cache.md) — FP8 KV cache with attention backends
- [Speculative Decoding](../10-speculative-decoding/README.md) — tree attention for EAGLE
- [Distributed Inference](../07-distributed/README.md) — attention in distributed settings
- [Architecture Overview](../03-architecture/README.md) — where attention fits in the engine
- [Environment Variables](../06-configuration/environment-variables.md) — `VLLM_ATTENTION_BACKEND` override
