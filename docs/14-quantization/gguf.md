# GGUF Quantization

GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp and the broader llama.cpp ecosystem. vLLM can load GGUF files directly, enabling use of the large library of community-quantized models available in this format.

## Overview

GGUF support is implemented in `vllm/model_executor/layers/quantization/gguf.py`. It uses the `gguf` Python library for parsing GGUF files and implements custom CUDA dequantization kernels for the various GGUF quantization types.

## GGUF Quantization Types

GGUF supports a wide range of quantization formats, organized into four categories:

### Unquantized Types

| Type | Description |
|------|-------------|
| `F32` | 32-bit float |
| `F16` | 16-bit float |
| `BF16` | BFloat16 |

### Standard Quantization Types

| Type | Bits | Description |
|------|------|-------------|
| `Q4_0` | 4 | 4-bit, no zero-point |
| `Q4_1` | 4 | 4-bit, with zero-point |
| `Q5_0` | 5 | 5-bit, no zero-point |
| `Q5_1` | 5 | 5-bit, with zero-point |
| `Q8_0` | 8 | 8-bit, no zero-point |
| `Q8_1` | 8 | 8-bit, with zero-point |

### K-Quant Types (Block Quantization)

K-quants use a hierarchical block structure with mixed precision for improved accuracy:

| Type | Effective Bits | Description |
|------|---------------|-------------|
| `Q2_K` | ~2.6 | 2-bit with K-quant blocks |
| `Q3_K` | ~3.4 | 3-bit with K-quant blocks |
| `Q4_K` | ~4.5 | 4-bit with K-quant blocks |
| `Q5_K` | ~5.5 | 5-bit with K-quant blocks |
| `Q6_K` | ~6.6 | 6-bit with K-quant blocks |

### I-Matrix Quantization Types

I-matrix (importance matrix) quantization uses activation statistics to improve accuracy:

| Type | Effective Bits | Description |
|------|---------------|-------------|
| `IQ1_M` | ~1.75 | 1-bit importance-matrix |
| `IQ1_S` | ~1.56 | 1-bit importance-matrix (small) |
| `IQ2_XXS` | ~2.06 | 2-bit importance-matrix (extra extra small) |
| `IQ2_XS` | ~2.31 | 2-bit importance-matrix (extra small) |
| `IQ2_S` | ~2.5 | 2-bit importance-matrix (small) |
| `IQ3_XXS` | ~3.06 | 3-bit importance-matrix |
| `IQ3_S` | ~3.44 | 3-bit importance-matrix (small) |
| `IQ4_XS` | ~4.25 | 4-bit importance-matrix (extra small) |
| `IQ4_NL` | ~4.5 | 4-bit importance-matrix (non-linear) |

> **Note:** I-matrix quantization types currently use dequantization-based inference (no MMQ kernel). Standard and K-quant types have optimized MMQ (mixed-precision matrix quantization) kernels.

## GGUFConfig

```python
class GGUFConfig(QuantizationConfig):
    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []

    @classmethod
    def get_min_capability(cls) -> int:
        return 60  # Maxwell or newer

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # GGUF dequantization kernels use half precision (fp16) internally
        # bfloat16 has precision issues on Blackwell devices
        if current_platform.has_device_capability(100):
            return [torch.half, torch.float32]
        return [torch.half, torch.bfloat16, torch.float32]
```

### Layer Method Dispatch

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        if is_layer_skipped_gguf(prefix, self.unquantized_modules, ...):
            return UnquantizedLinearMethod()
        return GGUFLinearMethod(self)
    elif isinstance(layer, VocabParallelEmbedding):
        if is_layer_skipped_gguf(prefix, self.unquantized_modules, ...):
            return UnquantizedEmbeddingMethod()
        return GGUFEmbeddingMethod(self)
    elif isinstance(layer, FusedMoE):
        return GGUFMoEMethod(self, layer.moe_config)
    return None
```

GGUF supports quantized embeddings in addition to linear layers, which is important for models where the embedding table is quantized.

## Loading GGUF Files

### Direct GGUF Loading

vLLM can load `.gguf` files directly without conversion:

```python
from vllm import LLM, SamplingParams

# Load a GGUF file directly
llm = LLM(
    model="path/to/model.gguf",
    tokenizer="meta-llama/Llama-3-8B",  # Tokenizer from HF Hub
)

outputs = llm.generate(
    ["What is the meaning of life?"],
    SamplingParams(max_tokens=100)
)
```

### Loading from Hugging Face Hub

Many GGUF models are hosted on Hugging Face Hub:

```python
# Load a specific GGUF file from a HF repo
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_loader_extra_config={
        "gguf_file": "llama-2-7b-chat.Q4_K_M.gguf"
    },
    tokenizer="meta-llama/Llama-2-7b-chat-hf",
)
```

### Specifying the GGUF File

When a repository contains multiple GGUF files (different quantization levels), specify which one to use:

```python
llm = LLM(
    model="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    model_loader_extra_config={
        "gguf_file": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    },
    tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
)
```

## Dequantization Kernels

vLLM implements CUDA dequantization kernels for GGUF types via `vllm._custom_ops`. The dequantization happens on-the-fly during inference:

```
GGUF weight (quantized) → CUDA dequant kernel → FP16 weight → GEMM → output
```

This is a weight-only quantization approach: weights are stored in GGUF format but dequantized to FP16 before each matrix multiply.

### MMQ Kernels

For standard and K-quant types, vLLM uses Mixed-precision Matrix Quantization (MMQ) kernels that perform the dequantization and matrix multiply in a single fused operation, avoiding the overhead of materializing the full FP16 weight matrix.

## Fused Layer Handling

GGUF models may have some layers unquantized (stored as F32/F16/BF16). The `is_layer_skipped_gguf` function handles fused layers (like `qkv_proj` which fuses `q_proj`, `k_proj`, `v_proj`) by checking each shard:

```python
def is_layer_skipped_gguf(prefix, unquantized_modules, fused_mapping):
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        # Check each shard of the fused layer
        shard_prefixes = [prefix.replace(proj_name, shard) 
                          for shard in fused_mapping[proj_name]]
        # All shards must have the same quantization status
        is_skipped = all(
            any(module_name in shard for module_name in unquantized_modules)
            for shard in shard_prefixes
        )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)
    return is_skipped
```

## Precision Notes

- GGUF dequantization kernels produce FP16 output internally
- On Blackwell (SM100) devices, BF16 has precision issues with GGUF; FP16 or FP32 is recommended
- The `get_supported_act_dtypes()` method automatically handles this:

```python
if current_platform.has_device_capability(100):
    logger.warning_once("GGUF has precision issues with bfloat16 on Blackwell.")
    return [torch.half, torch.float32]
return [torch.half, torch.bfloat16, torch.float32]
```

## Choosing a GGUF Quantization Level

| Type | Size (7B) | Quality | Use Case |
|------|-----------|---------|----------|
| Q2_K | ~2.8 GB | Low | Extreme memory constraint |
| Q3_K_M | ~3.3 GB | Low-Medium | Very tight memory |
| Q4_0 | ~3.8 GB | Medium | Good balance |
| Q4_K_M | ~4.1 GB | Medium-High | Recommended default |
| Q5_K_M | ~4.8 GB | High | Better accuracy |
| Q6_K | ~5.5 GB | Very High | Near-lossless |
| Q8_0 | ~7.2 GB | Excellent | Near-lossless |
| F16 | ~14 GB | Reference | No quantization |

The `Q4_K_M` variant is generally recommended as the best balance of size and quality.

## Related Pages

- [Quantization Overview](README.md)
- [BitsAndBytes](bitsandbytes.md) — Alternative 4-bit format
