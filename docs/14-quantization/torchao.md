# TorchAO Quantization

TorchAO is PyTorch's native quantization and sparsity library, providing a flexible, composable API for applying various quantization techniques. vLLM integrates TorchAO to support its growing ecosystem of quantization configurations.

## Overview

TorchAO support is implemented in `vllm/model_executor/layers/quantization/torchao.py`. It wraps the `torchao` Python library and supports both pre-quantized checkpoints and on-the-fly quantization.

**Minimum version:** `torchao >= 0.10.0`

## TorchAOConfig

```python
class TorchAOConfig(QuantizationConfig):
    def __init__(
        self,
        torchao_config,                              # AOBaseConfig object
        skip_modules: list[str] | None = None,       # Modules to skip
        is_checkpoint_torchao_serialized: bool = False,
    ) -> None:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `torchao_config` | `AOBaseConfig` | TorchAO quantization configuration object |
| `skip_modules` | list[str] | Module names/prefixes to skip quantization |
| `is_checkpoint_torchao_serialized` | bool | Whether checkpoint was saved with TorchAO |

### Hardware Requirements

```python
@classmethod
def get_min_capability(cls) -> int:
    return 75  # Turing or newer

def get_supported_act_dtypes(self) -> list[torch.dtype]:
    return [torch.float32, torch.float16, torch.bfloat16]
```

## Configuration Loading

### From HuggingFace Config

TorchAO models store their config in `config.json` under `quantization_config`:

```json
{
  "quantization_config": {
    "quant_method": "torchao",
    "quant_type": {
      "default": {
        "_type": "Float8DynamicActivationFloat8WeightConfig",
        "_data": {
          "granularity": {"_type": "PerRow"}
        }
      }
    }
  }
}
```

The config is parsed using `torchao.core.config.config_from_dict`:

```python
@classmethod
def from_config(cls, config: dict[str, Any]) -> "TorchAOConfig":
    from torchao.core.config import config_from_dict

    hf_config = cls.get_from_keys_or(config, ["quant_type"], None)
    quant_type = hf_config["default"]
    ao_config = config_from_dict(quant_type)

    skip_modules = config.get("modules_to_not_convert", []) or []
    return cls(ao_config, skip_modules, is_checkpoint_torchao_serialized)
```

### From Config File

```python
@classmethod
def from_config_file(cls, config_file: str) -> "TorchAOConfig":
    """Load from a JSON config file."""
    with open(config_file) as f:
        config_dict = json.loads(f.read())
    hf_config = {"quant_type": {"default": config_dict}}
    return cls.from_config(hf_config)
```

### From Config Dict JSON String

```python
@classmethod
def from_config_dict_json(cls, config_dict_json: str) -> "TorchAOConfig":
    config_dict = json.loads(config_dict_json)
    hf_config = {"quant_type": {"default": config_dict}}
    return cls.from_config(hf_config)
```

## Per-Module Configuration

TorchAO supports per-module quantization via `ModuleFqnToConfig`:

```python
def get_quant_method(self, layer, prefix):
    from torchao.quantization import ModuleFqnToConfig

    if isinstance(self.torchao_config, ModuleFqnToConfig):
        module_fqn_to_config = self.torchao_config.module_fqn_to_config
        c = None
        if prefix in module_fqn_to_config:
            c = module_fqn_to_config[prefix]
        else:
            # Try regex patterns (prefixed with "re:")
            for pattern in module_fqn_to_config:
                if pattern.startswith("re:") and re.fullmatch(pattern[3:], prefix):
                    c = module_fqn_to_config[pattern]
                    break
            else:
                c = module_fqn_to_config.get("_default", None)

        if c is not None:
            return TorchAOLinearMethod(TorchAOConfig(c, ...))
        return UnquantizedLinearMethod()

    return TorchAOLinearMethod(self)
```

This allows different layers to use different quantization configurations:

```python
from torchao.quantization import ModuleFqnToConfig, Int8DynamicActivationInt8WeightConfig

config = ModuleFqnToConfig({
    "model.layers.0.self_attn.q_proj": Int8DynamicActivationInt8WeightConfig(),
    "re:model.layers.[0-9]+.mlp.*": Float8DynamicActivationFloat8WeightConfig(),
    "_default": None,  # Skip all other layers
})
```

## Module Skipping Logic

The `should_skip` function provides robust prefix-based skipping:

```python
def should_skip(prefix: str, skip_modules: list[str]) -> bool:
    """
    Examples:
    should_skip("model.layers.1.q_proj", ["model.layers.1.q_proj"]) → True
    should_skip("model.layers.10.o_proj", ["o_proj"])               → True
    should_skip("visual.model.layers.1.q_proj", ["visual"])         → True
    should_skip("model.layers.1.q_proj", ["layers.1"])              → True
    should_skip("model.layers.11.q_proj", ["layers.1"])             → False
    """
    for s in skip_modules:
        if prefix == s:
            return True
        if f".{s}." in f".{prefix}.":
            return True
    return False
```

## Weight Quantization

For serialized TorchAO checkpoints, weights are already quantized. For online quantization, `torchao_quantize_param_data` applies quantization during loading:

```python
def torchao_quantize_param_data(param: torch.Tensor, torchao_config: Any):
    from torchao.core.config import AOBaseConfig
    from torchao.quantization import quantize_

    with torch.device("meta"):
        dummy_linear = torch.nn.Sequential(
            torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
        )
    quantize_(dummy_linear, torchao_config)
    # Apply quantization to the actual parameter
    ...
```

## Hardware-Optimized Packing

For TorchAO >= 0.15.0, vLLM uses hardware-optimized tensor packing:

```python
if torchao_version_at_least("0.15.0"):
    from torchao.prototype.tensor_conversion.api import (
        convert_to_packed_tensor_based_on_current_hardware,
    )
else:
    convert_to_packed_tensor_based_on_current_hardware = lambda t: t
```

## Supported TorchAO Configurations

TorchAO supports many quantization configurations. Common ones include:

| Config Class | Description |
|-------------|-------------|
| `Int8DynamicActivationInt8WeightConfig` | INT8 W8A8 dynamic |
| `Int8WeightOnlyConfig` | INT8 weight-only |
| `Int4WeightOnlyConfig` | INT4 weight-only |
| `Float8DynamicActivationFloat8WeightConfig` | FP8 W8A8 dynamic |
| `Float8WeightOnlyConfig` | FP8 weight-only |
| `UIntXWeightOnlyConfig` | Unsigned INT-X weight-only |
| `ModuleFqnToConfig` | Per-module configuration |

## Usage Examples

### Loading a Pre-Quantized TorchAO Model

```python
from vllm import LLM, SamplingParams

# Model quantized with TorchAO and uploaded to HF Hub
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct-torchao")

outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(max_tokens=50)
)
```

### Online Quantization with TorchAO

```python
from vllm import LLM
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

# Create config from a JSON file
config = TorchAOConfig.from_config_file("torchao_config.json")

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="torchao",
    # quantization_config will be read from model's config.json
)
```

### Creating a TorchAO Config File

```python
import json
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.core.config import config_to_dict

config = Float8DynamicActivationFloat8WeightConfig()
with open("torchao_config.json", "w") as f:
    f.write(json.dumps(config_to_dict(config)))
```

## Comparison with Other Methods

| Feature | TorchAO | FP8 (native) | AWQ | GPTQ |
|---------|---------|-------------|-----|------|
| Flexibility | Very high | Medium | Low | Low |
| Bit-widths | Many | 8 | 4 | 2/3/4/8 |
| Per-module config | ✅ | ❌ | ❌ | ✅ (dynamic) |
| Serialized checkpoints | ✅ | ✅ | ✅ | ✅ |
| Online quantization | ✅ | ✅ | ❌ | ❌ |
| MoE support | ❌ | ✅ | ✅ | ✅ |

## Related Pages

- [FP8 Quantization](fp8.md)
- [Quantization Overview](README.md)
