---
description: >
  Complete guide to GPTQ and GPTQ-Marlin quantization in vLLM, covering
  2/3/4/8-bit weight quantization, the Marlin kernel, and dynamic per-layer
  quantization with GPTQModel.
---

# GPTQ

**GPTQ** (Generative Pre-trained Transformer Quantization) is a
post-training quantization method that compresses model weights to 2, 3, 4,
or 8 bits while keeping activations in FP16 or BF16. It uses a second-order
optimization technique (based on the Optimal Brain Surgeon framework) to
minimize the quantization error layer by layer.

vLLM supports two GPTQ backends:

- **`gptq`** — the original GPTQ kernel. Supports 2/3/4/8-bit weights on
  GPUs with compute capability ≥ 6.0 (Pascal and newer).
- **`gptq_marlin`** — the high-performance Marlin kernel. Supports 4-bit and
  8-bit symmetric quantization on GPUs with compute capability ≥ 7.5
  (Turing and newer). Delivers significantly higher throughput than the
  original kernel and is selected automatically when the model is compatible.

!!! tip
    For new quantization workflows, use [GPTQModel](gptqmodel.md) or
    [LLM Compressor](llm_compressor.md) to produce GPTQ checkpoints.
    Over 5,000 pre-quantized GPTQ models are available on
    [Hugging Face](https://huggingface.co/models?search=gptq).

---

## Hardware requirements

| Backend | Min. compute capability | Supported bits | Activation dtype |
|---|---|---|---|
| `gptq` | SM 6.0 (Pascal) | 2, 3, 4, 8 | FP16 |
| `gptq_marlin` | SM 7.5 (Turing) | 4, 8 (symmetric) | FP16, BF16 |

!!! note
    The 4-bit `gptq_gemm` kernel has known accuracy issues. vLLM automatically
    uses `gptq_marlin` for 4-bit models on supported hardware. You can force
    the original kernel with `--quantization gptq`, but `gptq_marlin` is
    strongly recommended.

---

## Quick start

### Serving a pre-quantized GPTQ model

```bash
# Serve a 4-bit GPTQ model from Hugging Face
vllm serve TheBloke/Llama-2-7B-Chat-GPTQ \
    --quantization gptq_marlin
```

vLLM auto-detects the quantization format from the model's
`quantize_config.json` file. You can omit `--quantization` entirely for
models that include this file:

```bash
vllm serve TheBloke/Llama-2-7B-Chat-GPTQ
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq_marlin",  # or "gptq" for the original kernel
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["Tell me about quantization."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Quantizing a model with GPTQModel

[GPTQModel](https://github.com/ModelCloud/GPTQModel) is the recommended tool
for creating new GPTQ checkpoints. It supports dynamic per-layer quantization
and produces checkpoints that are fully compatible with vLLM's Marlin kernel.

### Installation

```bash
pip install -U gptqmodel --no-build-isolation
```

### Basic 4-bit quantization

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SAVE_PATH = "Llama-3.2-1B-Instruct-GPTQ-4bit"

# Load calibration data (1024 samples from C4)
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

# Configure 4-bit symmetric quantization with group size 128
quant_config = QuantizeConfig(bits=4, group_size=128)

# Load and quantize
model = GPTQModel.load(MODEL_ID, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(SAVE_PATH)
```

### 8-bit quantization

```python
quant_config = QuantizeConfig(bits=8, group_size=128)
model = GPTQModel.load(MODEL_ID, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save("Llama-3.2-1B-Instruct-GPTQ-8bit")
```

### Dynamic per-layer quantization

GPTQModel supports **dynamic quantization**, which lets you apply different
quantization settings to different layers. This is useful for preserving
accuracy in sensitive layers while aggressively compressing others.

```python
from gptqmodel import GPTQModel, QuantizeConfig

# Base config: 4-bit for most layers
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    dynamic={
        # Layers 16–31: use 8-bit for better accuracy
        r"+:.*\.(?:1[6-9]|2[0-9]|3[01])\..*": {"bits": 8},
        # Skip all MoE layers entirely (no quantization)
        r"-:.*\.moe\..*": {},
    },
)

model = GPTQModel.load(MODEL_ID, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save("Llama-3.2-1B-Instruct-GPTQ-dynamic")
```

The `dynamic` dictionary uses regex keys with a `+:` prefix for positive
matches (apply override) and `-:` prefix for negative matches (skip
quantization entirely).

---

## Quantizing with LLM Compressor

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is the
vLLM project's recommended quantization toolkit. It supports GPTQ-style
quantization via the `GPTQModifier`.

```bash
pip install llmcompressor
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-W4A16-G128"

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Prepare calibration data
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(preprocess).map(tokenize, remove_columns=ds.column_names)

# Apply GPTQ W4A16 quantization
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

---

## Running a quantized model

### Command line

```bash
# Auto-detect quantization from config
vllm serve ./Meta-Llama-3-8B-Instruct-W4A16-G128

# Explicitly specify the backend
vllm serve ./Meta-Llama-3-8B-Instruct-W4A16-G128 \
    --quantization gptq_marlin
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./Meta-Llama-3-8B-Instruct-W4A16-G128")

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

## Evaluating accuracy

Install `lm-evaluation-harness` for standardized benchmarks:

```bash
pip install "lm-eval[api]>=0.4.11"
```

```bash
MODEL=./Meta-Llama-3-8B-Instruct-W4A16-G128

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL,add_bos_token=True \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --limit 250
```

!!! note
    Quantized models can be sensitive to the presence of the `bos` token.
    Always include `add_bos_token=True` when running evaluations.

---

## Configuration reference

The `quantize_config.json` file in a GPTQ checkpoint controls how vLLM
loads the model. Key fields:

| Field | Description | Example |
|---|---|---|
| `bits` | Weight bit width | `4` |
| `group_size` | Quantization group size (`-1` = per-channel) | `128` |
| `desc_act` | Activation ordering (improves accuracy, slower) | `false` |
| `sym` | Symmetric quantization | `true` |
| `lm_head` | Whether the LM head is quantized | `false` |
| `dynamic` | Per-layer quantization overrides | `{}` |

---

## Tensor parallelism

GPTQ models support tensor parallelism. Use `--tensor-parallel-size` to
distribute the model across multiple GPUs:

```bash
vllm serve TheBloke/Llama-2-70B-Chat-GPTQ \
    --quantization gptq_marlin \
    --tensor-parallel-size 4
```

---

## Best practices

- **Use `gptq_marlin` on Turing+ GPUs.** It is significantly faster than the
  original `gptq` kernel and is selected automatically for compatible models.
- **Use group size 128.** This is the standard default that balances accuracy
  and memory efficiency.
- **Use 512+ calibration samples.** More samples generally improve accuracy,
  especially for instruction-tuned models.
- **Match calibration data to your use case.** If you are deploying a
  code-generation model, calibrate on code data rather than general text.
- **Enable `desc_act` for higher accuracy.** Activation ordering improves
  quantization quality but may slightly reduce throughput.

---

## Troubleshooting

**Model loads but produces garbled output**
: Ensure you are using the correct `--quantization` flag. Try `gptq_marlin`
  instead of `gptq` on Turing+ GPUs.

**`ValueError: Unsupported quantization config`**
: The model's `quantize_config.json` may use a bit width or symmetry setting
  not supported by `gptq_marlin` (which requires 4-bit or 8-bit symmetric).
  Fall back to `--quantization gptq`.

**Out-of-memory during quantization**
: Reduce `batch_size` in GPTQModel, or use a smaller calibration dataset.

---

## Related pages

- [GPTQModel guide](gptqmodel.md) — dynamic per-layer quantization
- [LLM Compressor](llm_compressor.md) — vLLM's recommended quantization toolkit
- [AWQ](awq.md) — alternative 4-bit weight-only quantization
- [Quantization overview](index.md) — method comparison and hardware matrix
