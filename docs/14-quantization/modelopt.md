# ModelOpt (NVIDIA) Quantization

NVIDIA ModelOpt (Model Optimization) is NVIDIA's toolkit for post-training quantization and model optimization. vLLM supports loading models quantized with ModelOpt, covering FP8, FP4 (NvFP4), MXFP8, and mixed-precision configurations.

## Overview

ModelOpt quantization is implemented in `vllm/model_executor/layers/quantization/modelopt.py`. It supports four distinct quantization algorithms, each with its own config class:

| Config Class | Key | Algorithm |
|-------------|-----|-----------|
| `ModelOptFp8Config` | `modelopt` | FP8 (per-tensor, per-channel/token, per-block) |
| `ModelOptNvFp4Config` | `modelopt_fp4` | NVIDIA FP4 (NvFP4) |
| `ModelOptMxFp8Config` | `modelopt_mxfp8` | Microscaling FP8 (MXFP8) |
| `ModelOptMixedPrecisionConfig` | `modelopt_mixed` | Mixed precision |

## Supported Quantization Algorithms

```python
QUANT_ALGOS = [
    "FP8",                    # Per-tensor weight + optional static activation scale
    "FP8_PER_CHANNEL_PER_TOKEN",  # Per-channel weight + per-token activation
    "FP8_PB_WO",              # Per-block weight-only FP8
    "NVFP4",                  # NVIDIA FP4
    "MXFP8",                  # Microscaling FP8
    "MIXED_PRECISION",        # Mixed precision
]
KV_CACHE_QUANT_ALGOS = ["FP8"]
```

## ModelOptFp8Config

The primary FP8 config, supporting three FP8 variants:

```python
class ModelOptFp8Config(ModelOptQuantConfigBase):
    def __init__(
        self,
        quant_method: str,                    # "FP8", "FP8_PER_CHANNEL_PER_TOKEN", "FP8_PB_WO"
        is_checkpoint_fp8_serialized: bool,   # Whether checkpoint has FP8 weights
        kv_cache_quant_method: str | None,    # "FP8" or None
        exclude_modules: list[str],           # Modules to skip
    ) -> None:
```

### Hardware Requirements

```python
@classmethod
def get_min_capability(cls) -> int:
    return 89  # Ada Lovelace or newer (SM89+)
```

### FP8 Variants

#### FP8 (Per-Tensor)

`ModelOptFp8LinearMethod` — Static per-tensor weight and activation scales:

```python
# Weight: float8_e4m3fn, per-tensor scale
# Activation: float8_e4m3fn, per-tensor static scale
# Uses: torch._scaled_mm for hardware-accelerated GEMM
```

Limitations:
- Only per-tensor quantization (no per-channel)
- Only `float8_e4m3fn` dtype
- Requires static activation scales in checkpoint

#### FP8_PER_CHANNEL_PER_TOKEN

`ModelOptFp8PcPtLinearMethod` — Per-channel weight scales + per-token activation scales:

```python
# Weight: float8_e4m3fn, per-channel scale (one per output channel)
# Activation: float8_e4m3fn, per-token dynamic scale
# Better accuracy than per-tensor for models with varying channel magnitudes
```

#### FP8_PB_WO (Per-Block Weight-Only)

`ModelOptFp8PbWoLinearMethod` — Block-wise weight-only FP8:

```python
# Weight: float8_e4m3fn, per-block scale (e.g., 128×128 blocks)
# Activation: BF16/FP16 (no activation quantization)
# Uses CUTLASS block FP8 kernels when available
```

## ModelOptNvFp4Config

NVIDIA FP4 quantization using the NvFP4 format:

```python
class ModelOptNvFp4Config(ModelOptQuantConfigBase):
    # quant_algo: "NVFP4"
    # Uses NvFP4 linear and MoE methods
    # Requires SM89+ (Ada Lovelace or newer)
```

NvFP4 is NVIDIA's proprietary 4-bit floating-point format, different from the OCP MXFP4 standard. It provides:
- 4-bit values with block-level scaling
- Hardware acceleration on Ada Lovelace and Hopper GPUs
- Higher throughput than FP8 at the cost of some accuracy

## ModelOptMxFp8Config

Microscaling FP8 quantization:

```python
class ModelOptMxFp8Config(ModelOptQuantConfigBase):
    # quant_algo: "MXFP8"
    # Block size: MXFP8_BLOCK_SIZE (typically 32)
    # Scale dtype: MXFP8_SCALE_DTYPE (E8M0)
    # Value dtype: MXFP8_VALUE_DTYPE (E4M3)
```

MXFP8 uses the OCP MX specification with 8-bit values and block-level E8M0 scales. It provides better accuracy than standard FP8 due to fine-grained scaling.

## ModelOptMixedPrecisionConfig

Mixed precision allows different layers to use different quantization:

```python
class ModelOptMixedPrecisionConfig(ModelOptQuantConfigBase):
    # quant_algo: "MIXED_PRECISION"
    # Reads per-layer quantization from the checkpoint config
    # Supports mixing FP8, NvFP4, and unquantized layers
```

## Config File Format

ModelOpt uses `hf_quant_config.json` as its primary config file:

```json
{
  "quantization": {
    "quant_algo": "FP8",
    "kv_cache_quant_algo": "FP8",
    "exclude_modules": ["lm_head"]
  }
}
```

It also supports the compressed-tensors style format in `config.json`:

```json
{
  "quantization_config": {
    "quant_method": "modelopt",
    "quant_algo": "FP8",
    "kv_cache_scheme": {
      "type": "float",
      "num_bits": 8
    },
    "ignore": ["lm_head"]
  }
}
```

## Module Exclusion

ModelOpt uses wildcard patterns for module exclusion:

```python
def is_layer_excluded(self, prefix: str) -> bool:
    # 1. Exact matching with fused layer support
    if is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping):
        return True

    # 2. Substring matching for legacy checkpoints
    for exclude_module in self.exclude_modules:
        if exclude_module in prefix:
            return True

    # 3. Wildcard pattern matching (fnmatch)
    for wildcard_pattern in self.exclude_modules:
        if fnmatch(prefix, wildcard_pattern):
            return True

    return False
```

Vision tower layers are automatically excluded:

```python
# Hard-coded exclusion for vision components
if "vision_tower" in prefix or "vision_model" in prefix:
    return UnquantizedLinearMethod()
```

## KV Cache Quantization

ModelOpt FP8 supports KV cache quantization via `ModelOptFp8KVCacheMethod`:

```python
class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """Supports loading kv-cache scaling factors from FP8 checkpoints."""
    def __init__(self, quant_config: "ModelOptQuantConfigBase"):
        super().__init__(quant_config)
```

When `kv_cache_quant_algo="FP8"` is set in the config, the KV cache is stored in FP8 format.

## Weight Processing

### FP8 Per-Tensor

```python
def process_weights_after_loading(self, layer):
    weight = layer.weight
    max_w_scale = layer.weight_scale.max()
    if not (layer.weight_scale == layer.weight_scale[0]).all():
        # Requantize with unified max scale for tensor parallel shards
        max_w_scale, weight = requantize_with_max_scale(
            layer.weight, layer.weight_scale, layer.logical_widths
        )
    layer.weight = Parameter(weight.t(), requires_grad=False)
    layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
    layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
```

The requantization step ensures that tensor-parallel shards use a consistent scale factor.

## Usage Examples

### Loading a ModelOpt FP8 Model

```python
from vllm import LLM, SamplingParams

# NVIDIA TensorRT-LLM / ModelOpt quantized model
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")

outputs = llm.generate(
    ["Explain neural networks."],
    SamplingParams(max_tokens=200)
)
```

### ModelOpt FP8 with KV Cache

```python
# FP8 weights + FP8 KV cache (if kv_cache_quant_algo="FP8" in config)
llm = LLM(
    model="nvidia/Llama-3.1-70B-Instruct-FP8",
    tensor_parallel_size=4,
)
```

### ModelOpt NvFP4

```python
# NvFP4 quantized model
llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-NvFP4")
```

## Comparison: ModelOpt vs Standard FP8

| Feature | ModelOpt FP8 | Standard FP8 (`fp8`) |
|---------|-------------|---------------------|
| Config file | `hf_quant_config.json` | `config.json` |
| Per-tensor | ✅ | ✅ |
| Per-channel/token | ✅ | ✅ |
| Block-wise | ✅ (FP8_PB_WO) | ✅ |
| Dynamic activation | ❌ | ✅ |
| KV cache | ✅ | ✅ |
| NvFP4 | ✅ | ❌ |
| MXFP8 | ✅ | ❌ |
| Min capability | SM89 | SM75 (Marlin fallback) |

## Related Pages

- [FP8 Quantization](fp8.md)
- [KV Cache Quantization](kv-cache.md)
- [MXFP4](mxfp4.md)
- [Quantization Overview](README.md)
