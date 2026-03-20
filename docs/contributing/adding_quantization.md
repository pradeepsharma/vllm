# Contributing a Quantization Backend

This guide explains how to implement and contribute a new quantization backend to vLLM. Quantization backends allow vLLM to load and run quantized models efficiently, often with custom CUDA kernels for accelerated inference.

---

## Overview

vLLM's quantization system is built around two abstract base classes:

- **`QuantizationConfig`** — describes the quantization scheme and creates quantize methods
- **`QuantizeMethodBase`** — implements the actual quantized computation for a layer

When a model is loaded with a quantization method, vLLM:
1. Detects the quantization config from the model's `quantize_config.json` or user flags
2. Instantiates the `QuantizationConfig` for that method
3. For each linear layer, calls `get_quant_method()` to get a `QuantizeMethodBase`
4. The `QuantizeMethodBase` creates quantized weights and handles the forward pass

---

## Supported Quantization Methods

The current list of built-in methods is defined in `vllm/model_executor/layers/quantization/__init__.py`:

```python
QuantizationMethods = Literal[
    "awq", "fp8", "gptq", "gptq_marlin", "awq_marlin",
    "compressed-tensors", "bitsandbytes", "torchao",
    "mxfp4", "gguf", "inc", "moe_wna16", ...
]
```

---

## Step 1: Create the Quantization Config File

Create `vllm/model_executor/layers/quantization/myquant.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class MyQuantConfig(QuantizationConfig):
    """Configuration for MyQuant quantization."""

    def __init__(
        self,
        weight_bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act

    @classmethod
    def get_name(cls) -> str:
        return "myquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability (e.g., 80 = Ampere)
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Files to search for in the model directory
        return ["quantize_config.json", "quant_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MyQuantConfig":
        weight_bits = cls.get_from_keys(config, ["bits", "w_bit", "weight_bits"])
        group_size = cls.get_from_keys(config, ["group_size", "q_group_size"])
        desc_act = cls.get_from_keys_or(config, ["desc_act", "desc_act"], False)
        return cls(weight_bits=weight_bits, group_size=group_size, desc_act=desc_act)

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> "MyQuantLinearMethod | None":
        from vllm.model_executor.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod(self)
        return None
```

---

## Step 2: Implement the Quantize Method

The `QuantizeMethodBase` handles weight creation and the forward pass:

```python
class MyQuantLinearMethod(QuantizeMethodBase):
    """Quantized linear method for MyQuant."""

    def __init__(self, quant_config: MyQuantConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create quantized weight tensors and register them on the layer."""
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size
        weight_bits = self.quant_config.weight_bits

        # Number of groups
        num_groups = input_size_per_partition // group_size

        # Packed weight (e.g., 8 int4 values packed into one int32)
        pack_factor = 32 // weight_bits
        qweight = torch.nn.Parameter(
            torch.empty(
                input_size_per_partition // pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, {"weight_loader": self.weight_loader, **extra_weight_attrs})

        # Scales
        scales = torch.nn.Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, {"weight_loader": self.weight_loader, **extra_weight_attrs})

        # Zero points
        qzeros = torch.nn.Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, {"weight_loader": self.weight_loader, **extra_weight_attrs})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """Load a weight shard into the parameter."""
        param.data.copy_(loaded_weight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the quantized linear transformation."""
        # Option 1: Use a custom CUDA kernel
        out = my_quant_kernel(
            x,
            layer.qweight,
            layer.scales,
            layer.qzeros,
            self.quant_config.group_size,
        )

        # Option 2: Dequantize then multiply (slower, for reference)
        # weight = dequantize(layer.qweight, layer.scales, layer.qzeros)
        # out = torch.nn.functional.linear(x, weight, bias)

        if bias is not None:
            out = out + bias
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Optional: post-process weights after all shards are loaded.

        Use this for operations like:
        - Reordering weights for kernel efficiency
        - Converting to a different layout
        - Fusing scales and zero points
        """
        pass
```

---

## Step 3: Register the Quantization Method

### Add to the Literal Type

In `vllm/model_executor/layers/quantization/__init__.py`, add your method to `QuantizationMethods`:

```python
QuantizationMethods = Literal[
    "awq",
    "fp8",
    # ... existing methods ...
    "myquant",  # Add your method here
]
```

### Add to `get_quantization_config`

In the same file, add your config to the `method_to_config` dictionary:

```python
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    # ... existing imports ...
    from .myquant import MyQuantConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
        # ... existing entries ...
        "myquant": MyQuantConfig,
    }
    # ...
```

### Add Platform Support (Optional)

If your method is only supported on specific platforms, add it to the platform's `supported_quantization` list. For example, in `vllm/platforms/cuda.py`:

```python
class CudaPlatform(Platform):
    supported_quantization: list[str] = [
        "awq", "fp8", "gptq", ..., "myquant"
    ]
```

---

## Step 4: Handle Auto-Detection

vLLM can automatically detect the quantization method from the model's config files. Implement `get_config_filenames()` to return the files your method uses:

```python
@staticmethod
def get_config_filenames() -> list[str]:
    return ["quantize_config.json", "quant_config.json"]
```

If your method needs to override the user-specified quantization (e.g., to use a more optimized variant), implement `override_quantization_method`:

```python
@classmethod
def override_quantization_method(
    cls, hf_quant_cfg: dict, user_quant: str | None
) -> str | None:
    # Example: auto-upgrade gptq to gptq_marlin when supported
    if user_quant == "gptq" and cls._is_marlin_compatible(hf_quant_cfg):
        return "gptq_marlin"
    return None
```

---

## Step 5: Write Tests

Create `tests/quantization/test_myquant.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm import LLM, SamplingParams


@pytest.mark.parametrize("model", [
    "your-org/mymodel-4bit-myquant",
])
def test_myquant_generation(model: str):
    """Test that MyQuant quantized model generates correct output."""
    llm = LLM(model=model, quantization="myquant")
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=10, temperature=0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


def test_myquant_config_parsing():
    """Test that MyQuantConfig correctly parses quantization configs."""
    from vllm.model_executor.layers.quantization.myquant import MyQuantConfig

    config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
    }
    quant_config = MyQuantConfig.from_config(config)
    assert quant_config.weight_bits == 4
    assert quant_config.group_size == 128
    assert quant_config.desc_act is False


def test_myquant_linear_layer():
    """Test the quantized linear layer forward pass."""
    from vllm.model_executor.layers.quantization.myquant import (
        MyQuantConfig,
        MyQuantLinearMethod,
    )

    config = MyQuantConfig(weight_bits=4, group_size=128)
    method = MyQuantLinearMethod(config)

    # Create a mock layer and test forward pass
    # ...
```

### Running Quantization Tests

```bash
cd tests
pytest -v -s quantization/test_myquant.py
```

---

## Step 6: Update Documentation

1. **Add to `docs/features/quantization/index.md`** — add your method to the comparison table
2. **Create `docs/features/quantization/myquant.md`** — document usage, requirements, and examples
3. **Update `docs/configuration/quantization_config.md`** — document any new config options

---

## Advanced Topics

### Using the Plugin System

For quantization methods that live outside the vLLM repository, use the plugin registration API:

```python
# In your external package's __init__.py or plugin entry point:
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

@register_quantization_config("my_external_quant")
class MyExternalQuantConfig(QuantizationConfig):
    # ... implementation ...
    pass
```

Register the plugin in your `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
my_quant_plugin = "my_package.quant:register_my_quant"
```

### Meta Device Loading

For large models, you can create weights on the meta device to reduce peak memory during loading. Set `uses_meta_device = True` and implement `process_weights_after_loading`:

```python
class MyQuantLinearMethod(QuantizeMethodBase):
    uses_meta_device: bool = True

    def create_weights(self, layer, ...):
        # Weights are created on meta device
        qweight = torch.nn.Parameter(
            torch.empty(..., device="meta"),
            requires_grad=False,
        )
        layer.register_parameter("qweight", qweight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Quantize the loaded float weights here
        layer.qweight = quantize_weights(layer.weight)
        del layer.weight  # Free the float weights
```

### MoE (Mixture of Experts) Support

For quantization methods that support MoE layers, implement a separate `QuantizeMethodBase` for fused MoE:

```python
class MyQuantFusedMoEMethod(QuantizeMethodBase):
    """Quantized method for fused MoE layers."""

    def create_weights(self, layer, num_experts, ...):
        # Create per-expert quantized weights
        ...

    def apply(self, layer, x, router_logits, ...):
        # Apply quantized MoE computation
        ...
```

Register it in `get_quant_method`:

```python
def get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.fused_moe import FusedMoE
    if isinstance(layer, FusedMoE):
        return MyQuantFusedMoEMethod(self)
    from vllm.model_executor.layers.linear import LinearBase
    if isinstance(layer, LinearBase):
        return MyQuantLinearMethod(self)
    return None
```

### KV Cache Quantization

To support KV cache quantization, implement `get_cache_scale`:

```python
def get_cache_scale(self, name: str) -> str | None:
    """Return the scale parameter name for a KV cache tensor."""
    if name.endswith(".k_cache"):
        return name.replace(".k_cache", ".k_scale")
    if name.endswith(".v_cache"):
        return name.replace(".v_cache", ".v_scale")
    return None
```

---

## PR Checklist

Before submitting your quantization PR:

- [ ] `QuantizationConfig` subclass implemented with all abstract methods
- [ ] `QuantizeMethodBase` subclass implemented with `create_weights` and `apply`
- [ ] Method registered in `QuantizationMethods` literal and `get_quantization_config`
- [ ] Platform support declared (if hardware-specific)
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No mypy errors

---

## Getting Help

- Look at existing implementations: `awq.py`, `gptq.py`, `fp8.py`, `bitsandbytes.py`
- Ask in `#quantization` on [vLLM Slack](https://slack.vllm.ai)
- Reference the [Quantization Features](../features/quantization/index.md) documentation
