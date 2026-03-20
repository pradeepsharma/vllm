---
description: >
  Guide to TorchAO quantization in vLLM, covering INT4, INT8, FP8, and other
  PyTorch-native quantization formats with torch.compile support.
---

# TorchAO

**TorchAO** is PyTorch's native architecture optimization library. It provides
high-performance data types, quantization techniques, and kernels for inference
and training, with first-class support for `torch.compile`, FSDP, and other
PyTorch features.

vLLM integrates TorchAO through the `torchao` quantization method, which
supports loading pre-quantized TorchAO checkpoints from Hugging Face and
applying in-flight quantization using any TorchAO configuration.

---

## Hardware requirements

| Platform | Supported |
|---|---|
| NVIDIA GPU (SM 7.5+, Turing and newer) | ✅ |
| AMD GPU | ❌ |
| Intel GPU | ❌ |
| CPU | ❌ |

Supported activation dtypes: `float32`, `float16`, `bfloat16`.

---

## Installation

Install the latest TorchAO nightly build for best performance:

```bash
# Choose the CUDA version that matches your system (cu126, cu128, etc.)
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

Or install the stable release:

```bash
pip install torchao>=0.10.0
```

---

## Supported quantization types

TorchAO supports a wide range of quantization configurations:

| Config class | Description | Bits |
|---|---|---|
| `Int8WeightOnlyConfig` | INT8 weight-only | 8 |
| `Int4WeightOnlyConfig` | INT4 weight-only | 4 |
| `Int8DynamicActivationInt8WeightConfig` | INT8 W8A8 dynamic | 8 |
| `Float8WeightOnlyConfig` | FP8 weight-only | 8 |
| `Float8DynamicActivationFloat8WeightConfig` | FP8 W8A8 dynamic | 8 |
| `UIntXWeightOnlyConfig` | Unsigned INT (2–7 bit) weight-only | 2–7 |
| `ModuleFqnToConfig` | Per-module configuration | Various |

---

## Loading a pre-quantized TorchAO checkpoint

Many TorchAO-quantized models are available on Hugging Face. vLLM
automatically detects the TorchAO quantization config from the model's
`config.json`.

```python
from vllm import LLM, SamplingParams

# Load a pre-quantized TorchAO INT8 weight-only model
llm = LLM(model="jerryzh168/llama3-8b-int8wo")

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Tell me about quantization."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Quantizing a model with TorchAO

### Using the transformers integration

The easiest way to quantize a model is through the `transformers` library's
TorchAO integration:

```python
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
SAVE_DIR = "Meta-Llama-3-8B-TorchAO-INT8wo"

# Configure INT8 weight-only quantization
quantization_config = TorchAoConfig(Int8WeightOnlyConfig())

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Save to Hugging Face Hub or local directory
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR, safe_serialization=False)
```

### INT4 weight-only quantization

```python
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig

quantization_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=128))

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
```

### FP8 dynamic activation quantization

```python
from transformers import TorchAoConfig, AutoModelForCausalLM
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow

quantization_config = TorchAoConfig(
    Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
```

---

## Using a TorchAO config file

You can save a TorchAO configuration to a JSON file and load it with vLLM:

```python
import json
from torchao.quantization import Int8WeightOnlyConfig
from torchao.core.config import config_to_dict

# Save config to file
config = Int8WeightOnlyConfig()
with open("torchao_config.json", "w") as f:
    f.write(json.dumps(config_to_dict(config)))
```

```python
from vllm import LLM
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig

# Load from config file
torchao_config = TorchAOConfig.from_config_file("torchao_config.json")
```

---

## Per-module quantization

TorchAO supports applying different quantization configurations to different
modules using `ModuleFqnToConfig`:

```python
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    ModuleFqnToConfig,
)

# Apply INT4 to most layers, INT8 to the last few layers
config = ModuleFqnToConfig(
    module_fqn_to_config={
        "_default": Int4WeightOnlyConfig(group_size=128),
        # Use regex for pattern matching (prefix with "re:")
        "re:.*layers\\.3[0-1]\\..*": Int8WeightOnlyConfig(),
    }
)
```

---

## Running a quantized model

### Command line

```bash
# Auto-detect TorchAO quantization from config
vllm serve ./Meta-Llama-3-8B-TorchAO-INT8wo

# Explicit TorchAO quantization
vllm serve ./Meta-Llama-3-8B-TorchAO-INT8wo \
    --quantization torchao
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./Meta-Llama-3-8B-TorchAO-INT8wo")

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}")
    print()
```

---

## Skipping modules

You can skip quantization for specific modules using `modules_to_not_convert`
in the quantization config or by specifying `skip_modules` in the TorchAO
config:

```python
from transformers import TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

# Skip the LM head and specific layers
quantization_config = TorchAoConfig(
    Int8WeightOnlyConfig(),
    modules_to_not_convert=["lm_head", "model.layers.0"],
)
```

The skip logic supports:
- Exact module names: `"model.layers.1.q_proj"`
- Partial names: `"lm_head"` (matches any module containing `lm_head`)
- Prefix matches: `"visual"` (matches any module under `visual.*`)

---

## TorchAO vs. other methods

| Feature | TorchAO | GPTQ | AWQ | BitsAndBytes |
|---|---|---|---|---|
| Calibration required | Optional | Yes | Yes | No |
| `torch.compile` support | ✅ | ❌ | ❌ | ❌ |
| FSDP support | ✅ | ❌ | ❌ | ❌ |
| Supported bits | 2–8 | 2, 3, 4, 8 | 4 | 4, 8 |
| Marlin kernel | ❌ | ✅ | ✅ | ❌ |
| Min. GPU | Turing (SM 7.5) | Pascal (SM 6.0) | Turing (SM 7.5) | Volta (SM 7.0) |

TorchAO is the best choice for PyTorch-native workflows that use
`torch.compile` or FSDP. For pure inference throughput, GPTQ or AWQ with
the Marlin kernel may deliver better performance.

---

## Best practices

- **Install TorchAO nightly** for the latest performance improvements and
  bug fixes.
- **Use `Int8WeightOnlyConfig` as a starting point.** It provides a good
  balance between accuracy and memory reduction with no calibration required.
- **Use `Float8DynamicActivationFloat8WeightConfig` on Hopper/Ada GPUs** for
  the highest throughput.
- **Use `ModuleFqnToConfig` for mixed-precision models.** Apply aggressive
  quantization to most layers and preserve precision in sensitive layers.
- **Use `safe_serialization=False`** when saving TorchAO models, as some
  TorchAO tensor types are not compatible with safetensors format.

---

## Troubleshooting

**`ImportError: Please install torchao>=0.10.0`**
: Install TorchAO:
  ```bash
  pip install torchao>=0.10.0
  ```

**`AssertionError: quant_type must be specified`**
: The model's `config.json` is missing the `quant_type` field. Ensure the
  model was saved with a valid TorchAO quantization config.

**`Expected only one key 'default' in quant_type dictionary`**
: The `quant_type` dictionary in `config.json` must have exactly one key
  named `"default"`. Check the model's quantization config.

---

## Related pages

- [BitsAndBytes](bitsandbytes.md) — no-calibration 4-bit and 8-bit quantization
- [FP8](fp8.md) — FP8 quantization for Ada/Hopper GPUs
- [Quantization overview](index.md) — method comparison and hardware matrix
