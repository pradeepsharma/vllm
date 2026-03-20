---
description: >
  Guide to BitsAndBytes quantization in vLLM, covering 4-bit NF4/FP4 and 8-bit
  LLM.int8() quantization, in-flight quantization, and loading pre-quantized
  checkpoints.
---

# BitsAndBytes

**BitsAndBytes** is a quantization library that provides 4-bit and 8-bit
quantization for transformer models. Unlike GPTQ and AWQ, BitsAndBytes does
**not require calibration data** — it quantizes weights on the fly when the
model is loaded. This makes it the easiest way to reduce memory usage for
models that do not have pre-quantized checkpoints.

vLLM supports BitsAndBytes through the `bitsandbytes` quantization method,
which reads the quantization configuration from the model's `config.json` file
or applies in-flight quantization when explicitly requested.

---

## Hardware requirements

| Platform | Supported |
|---|---|
| NVIDIA GPU (SM 7.0+, Volta and newer) | ✅ |
| AMD GPU | ❌ |
| Intel GPU | ❌ |
| CPU | ❌ |

---

## Installation

```bash
# NVIDIA CUDA
pip install bitsandbytes>=0.48.1

# AMD ROCm
pip install bitsandbytes>=0.49.2
```

---

## Quantization modes

### 4-bit quantization (recommended)

4-bit quantization stores weights in 4-bit format (NF4 or FP4) and
dequantizes them to a compute dtype (typically BF16 or FP16) before each
matrix multiplication. This delivers a **4× memory reduction** compared to
BF16.

**NF4** (NormalFloat 4-bit) is the default and recommended format. It is
designed for normally distributed weights and delivers better accuracy than
standard INT4.

**Double quantization** further reduces memory by quantizing the quantization
constants themselves. It saves approximately 0.4 bits per parameter.

### 8-bit quantization (LLM.int8())

8-bit quantization uses the LLM.int8() algorithm, which identifies outlier
features in activations and handles them in FP16 while quantizing the rest
to INT8. This delivers a **2× memory reduction** with minimal accuracy loss.

---

## Loading a pre-quantized checkpoint

Many models on Hugging Face include a `quantization_config` section in their
`config.json` that specifies BitsAndBytes settings. vLLM automatically detects
and applies these settings.

```python
from vllm import LLM
import torch

# unsloth/tinyllama-bnb-4bit is a pre-quantized 4-bit BitsAndBytes checkpoint
llm = LLM(
    model="unsloth/tinyllama-bnb-4bit",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

You can find BitsAndBytes quantized models on Hugging Face:
[huggingface.co/models?search=bitsandbytes](https://huggingface.co/models?search=bitsandbytes)

---

## In-flight 4-bit quantization

For models without a pre-quantized checkpoint, you can apply 4-bit
quantization on the fly by specifying `quantization="bitsandbytes"`:

```python
from vllm import LLM
import torch

llm = LLM(
    model="huggyllama/llama-7b",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

!!! warning
    In-flight quantization loads the model at full precision before
    quantizing. You need enough GPU memory to hold the full BF16/FP16 model
    during loading.

### Command line

```bash
# In-flight 4-bit quantization
vllm serve huggyllama/llama-7b \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16
```

---

## In-flight 8-bit quantization

```python
from vllm import LLM
import torch

llm = LLM(
    model="huggyllama/llama-7b",
    dtype=torch.bfloat16,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)
```

To use 8-bit instead of 4-bit, configure the model's quantization config
to set `load_in_8bit=True` and `load_in_4bit=False`.

---

## Configuration options

BitsAndBytes configuration is controlled by the `quantization_config` section
in `config.json`. Key options:

| Option | Default | Description |
|---|---|---|
| `load_in_4bit` | `true` | Enable 4-bit quantization |
| `load_in_8bit` | `false` | Enable 8-bit quantization |
| `bnb_4bit_quant_type` | `"nf4"` | Quantization type: `"nf4"` or `"fp4"` |
| `bnb_4bit_compute_dtype` | `"float32"` | Compute dtype for dequantization |
| `bnb_4bit_use_double_quant` | `false` | Enable double quantization |
| `llm_int8_threshold` | `6.0` | Outlier threshold for LLM.int8() |
| `llm_int8_skip_modules` | `[]` | Modules to skip (e.g., `["lm_head"]`) |

---

## Generating a quantized checkpoint

You can create a BitsAndBytes quantized checkpoint using the `transformers`
library and save it to Hugging Face Hub or a local directory:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-BnB-4bit"

# Configure 4-bit NF4 quantization with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Save the quantized model
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

---

## Generating text

```python
from vllm import LLM, SamplingParams
import torch

llm = LLM(
    model="unsloth/tinyllama-bnb-4bit",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

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

## OpenAI-compatible server

```bash
# Serve a pre-quantized BitsAndBytes model
vllm serve unsloth/tinyllama-bnb-4bit \
    --dtype bfloat16

# In-flight 4-bit quantization
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype bfloat16
```

---

## BitsAndBytes vs. GPTQ vs. AWQ

| Feature | BitsAndBytes | GPTQ | AWQ |
|---|---|---|---|
| Calibration required | ❌ No | ✅ Yes | ✅ Yes |
| Supported bits | 4, 8 | 2, 3, 4, 8 | 4 |
| Throughput | Moderate | High (Marlin) | High (Marlin) |
| Memory reduction | 4× (4-bit) | 4× (4-bit) | 4× (4-bit) |
| Min. GPU | Volta (SM 7.0) | Pascal (SM 6.0) | Turing (SM 7.5) |
| Pre-quantized models | Many | Many (5000+) | Many (6500+) |

BitsAndBytes is the easiest option when you need to quickly reduce memory
usage without calibration. For production deployments where throughput matters,
GPTQ with the Marlin kernel or AWQ with the Marlin kernel will deliver better
performance.

---

## Best practices

- **Use NF4 for best accuracy.** NF4 is designed for normally distributed
  weights and outperforms FP4 in most benchmarks.
- **Enable double quantization** to save an additional ~0.4 bits per parameter
  with minimal accuracy impact.
- **Use BF16 as the compute dtype.** BF16 provides better numerical stability
  than FP32 for most models.
- **Skip the LM head.** The language model head is sensitive to quantization.
  Add it to `llm_int8_skip_modules` for 8-bit mode.
- **For production, prefer GPTQ or AWQ.** BitsAndBytes is convenient but
  delivers lower throughput than Marlin-based kernels.

---

## Troubleshooting

**`ImportError: Please install bitsandbytes>=0.48.1`**
: Install the required version:
  ```bash
  pip install bitsandbytes>=0.48.1
  ```

**Out-of-memory during loading**
: BitsAndBytes loads the model at full precision before quantizing. Ensure
  you have enough GPU memory for the full model. Consider using a pre-quantized
  checkpoint instead.

**Low throughput compared to GPTQ/AWQ**
: BitsAndBytes dequantizes weights on the fly during each forward pass, which
  adds overhead. For higher throughput, use a GPTQ or AWQ checkpoint with the
  Marlin kernel.

---

## Related pages

- [GPTQ](gptq.md) — higher-throughput 4-bit quantization with Marlin kernel
- [AWQ](awq.md) — activation-aware 4-bit quantization
- [TorchAO](torchao.md) — PyTorch-native quantization
- [Quantization overview](index.md) — method comparison and hardware matrix
