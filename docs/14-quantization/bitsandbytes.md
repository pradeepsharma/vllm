# BitsAndBytes Quantization

BitsAndBytes (BnB) provides two quantization modes for LLM inference: 4-bit NF4 (NormalFloat4) and 8-bit INT8 with mixed-precision decomposition. It is the most accessible quantization method in vLLM, requiring no pre-quantized checkpoint — quantization happens automatically during model loading.

**Reference:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## Overview

BitsAndBytes is implemented in `vllm/model_executor/layers/quantization/bitsandbytes.py` and wraps the `bitsandbytes` Python library.

### Version Requirements

```python
# Minimum versions:
# NVIDIA: bitsandbytes >= 0.48.1
# AMD ROCm: bitsandbytes >= 0.49.2
```

## BitsAndBytesConfig

```python
class BitsAndBytesConfig(QuantizationConfig):
    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: list[str] | None = None,
        llm_int8_threshold: float = 6.0,
    ) -> None:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_8bit` | bool | `False` | Enable 8-bit INT8 quantization |
| `load_in_4bit` | bool | `True` | Enable 4-bit NF4 quantization |
| `bnb_4bit_compute_dtype` | str | `"float32"` | Compute dtype for 4-bit dequantization |
| `bnb_4bit_quant_storage` | str | `"uint8"` | Storage dtype (only `"uint8"` supported) |
| `bnb_4bit_quant_type` | str | `"fp4"` | Quantization type: `"fp4"` or `"nf4"` |
| `bnb_4bit_use_double_quant` | bool | `False` | Quantize the quantization constants (double quantization) |
| `llm_int8_skip_modules` | list[str] | `None` | Modules to keep in FP16 for 8-bit mode |
| `llm_int8_threshold` | float | `6.0` | Outlier threshold for INT8 mixed-precision |

### Hardware Requirements

```python
@classmethod
def get_min_capability(cls) -> int:
    return 70  # Volta or newer
```

BitsAndBytes supports NVIDIA SM70+ and AMD ROCm (with appropriate library version).

## 4-bit NF4 Quantization

NF4 (NormalFloat4) is a data type specifically designed for normally distributed weights. It uses a non-uniform quantization grid that is optimal for weights following a normal distribution, providing better accuracy than uniform INT4.

### How NF4 Works

NF4 defines 16 quantization levels that are equally spaced in the quantile space of a standard normal distribution. This means more quantization levels are allocated near zero (where most weights cluster) and fewer at the extremes.

```
NF4 levels: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

### Double Quantization

When `bnb_4bit_use_double_quant=True`, the quantization constants (scales) are themselves quantized to 8-bit, saving an additional ~0.4 bits per parameter:

```
Memory per parameter:
- NF4 alone:          4.5 bits (4-bit weight + 0.5-bit scale overhead)
- NF4 + double quant: 4.1 bits
```

### Weight Storage

4-bit weights are stored using `bitsandbytes.nn.Params4bit`:

```python
from bitsandbytes.nn import Params4bit

qweight = Params4bit(
    data=weight,
    requires_grad=False,
    compress_statistics=bnb_4bit_use_double_quant,
    quant_type=bnb_4bit_quant_type,  # "nf4" or "fp4"
    quant_storage=torch.uint8,
)
```

## 8-bit INT8 Quantization

The 8-bit mode uses LLM.int8(), a mixed-precision decomposition approach:

1. **Identify outliers:** Activation values exceeding `llm_int8_threshold` (default 6.0) are treated as outliers
2. **Decompose:** The matrix multiply is split into:
   - Outlier columns: computed in FP16
   - Non-outlier columns: computed in INT8
3. **Combine:** Results are added together

This approach preserves accuracy for models with activation outliers (common in large models) while still achieving ~2× memory reduction.

### Weight Storage

8-bit weights are stored using `bitsandbytes.nn.Int8Params`:

```python
from bitsandbytes.nn import Int8Params

qweight = Int8Params(
    data=weight,
    requires_grad=False,
    has_fp16_weights=llm_int8_has_fp16_weight,
)
```

## Layer Method Dispatch

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        if is_layer_skipped_bnb(prefix, self.llm_int8_skip_modules):
            return UnquantizedLinearMethod()
        return BitsAndBytesLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return BitsAndBytesMoEMethod(self, layer.moe_config)
    return None
```

### Layer Skipping Logic

The `is_layer_skipped_bnb` function checks both exact component matches and prefix matches:

```python
def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]):
    components = prefix.split(".")
    # Check if any skip module exactly matches any component
    substr_check = any(module_name in components for module_name in llm_int8_skip_modules)
    # Also check prefix matches
    set_components = set("." .join(components[:i+1]) for i in range(len(components)))
    prefix_check = len(set(llm_int8_skip_modules) & set_components) != 0
    return substr_check or prefix_check
```

## Usage Examples

### 4-bit NF4 Loading

```python
from vllm import LLM, SamplingParams

# Load with 4-bit NF4 quantization
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)

outputs = llm.generate(
    ["Tell me about quantum computing."],
    SamplingParams(max_tokens=200)
)
```

### 8-bit INT8 Loading

```python
# Load with 8-bit INT8 quantization
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    # The quantization_config in the model card controls 8-bit vs 4-bit
)
```

### From a Pre-Quantized BnB Checkpoint

If the model was saved with `save_pretrained()` after BnB quantization, vLLM auto-detects the config:

```python
# Model has quantization_config in config.json with quant_type="nf4"
llm = LLM(model="path/to/bnb-quantized-model")
```

### Skipping Specific Layers

```python
# Keep lm_head and embed_tokens in FP16
# (configured via quantization_config in the model)
# llm_int8_skip_modules: ["lm_head", "embed_tokens"]
```

## Memory Comparison

| Mode | Memory vs FP16 | Accuracy |
|------|---------------|----------|
| FP16 baseline | 1× | Reference |
| INT8 (8-bit) | ~0.5× | Very close |
| NF4 (4-bit) | ~0.25× | Good |
| NF4 + double quant | ~0.22× | Good |

For a 7B model:
- FP16: ~14 GB
- INT8: ~7 GB
- NF4: ~3.5 GB

## Compute Dtype

The `bnb_4bit_compute_dtype` controls the precision used during the dequantization and matrix multiply:

| Value | Speed | Accuracy |
|-------|-------|----------|
| `"float32"` | Slowest | Highest |
| `"float16"` | Fast | Good |
| `"bfloat16"` | Fast | Good (recommended) |

For inference, `"bfloat16"` is typically the best choice.

## Limitations

- BitsAndBytes does not support tensor parallelism (TP > 1) in all configurations
- The 4-bit mode requires dequantization before each matrix multiply (no native INT4 GEMM)
- Performance is generally lower than GPTQ-Marlin or AWQ-Marlin for the same bit-width
- Only `uint8` is supported as `bnb_4bit_quant_storage`

## Related Pages

- [AWQ](awq.md) — Higher-performance 4-bit quantization
- [GPTQ](gptq.md) — Alternative weight-only quantization
- [Quantization Overview](README.md)
