---
description: >
  Complete guide to INT8 W8A8 quantization in vLLM using compressed-tensors,
  covering static and dynamic activation schemes, SmoothQuant, and calibration
  with LLM Compressor.
---

# INT8 W8A8

**INT8 W8A8** quantization stores both model weights and activations in 8-bit
integers. It delivers a **2× memory reduction** compared to BF16 and enables
hardware-accelerated integer matrix multiplications on NVIDIA GPUs with
compute capability ≥ 7.5 (Turing and newer).

vLLM implements INT8 W8A8 through the **compressed-tensors** format, which is
produced by [LLM Compressor](llm_compressor.md). Pre-quantized INT8 checkpoints
are available on Hugging Face:
[neuralmagic/int8-llms-for-vllm](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415).

---

## Hardware requirements

| Platform | Supported | Notes |
|---|---|---|
| NVIDIA Turing (SM 7.5) | ✅ | |
| NVIDIA Ampere (SM 8.x) | ✅ | |
| NVIDIA Ada Lovelace (SM 8.9) | ✅ | |
| NVIDIA Hopper (SM 9.0) | ✅ | |
| NVIDIA Blackwell (SM 10.x) | ❌ | Use [FP8](fp8.md) instead |
| x86 CPU | ✅ | Via Intel Extension for PyTorch |
| AMD GPU | ❌ | |

!!! warning
    INT8 W8A8 is **not supported on Blackwell GPUs** (compute capability ≥ 10.0,
    e.g., RTX 6000 Blackwell). Use [FP8 quantization](fp8.md) on Blackwell,
    or run on Hopper / Ada / Ampere.

---

## Quick start

### Serving a pre-quantized INT8 model

```bash
# Quantization is auto-detected from the checkpoint config
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8")

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["Tell me about quantization."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Quantizing a model with LLM Compressor

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is the
recommended tool for producing INT8 W8A8 checkpoints. It combines
**SmoothQuant** (to balance quantization difficulty between weights and
activations) with **GPTQ** (to minimize weight quantization error).

### Installation

```bash
pip install llmcompressor
pip install vllm "lm-eval[api]>=0.4.11"
```

### Step 1: Load the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### Step 2: Prepare calibration data

INT8 activation quantization requires calibration data to estimate activation
scales. Use data that closely matches your deployment distribution.

```python
from datasets import load_dataset

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
```

### Step 3: Apply quantization

The recommended recipe combines SmoothQuant with GPTQ W8A8:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"

recipe = [
    # SmoothQuant migrates quantization difficulty from activations to weights
    SmoothQuantModifier(smoothing_strength=0.8),
    # GPTQ minimizes weight quantization error using second-order optimization
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

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

### Step 4: Evaluate accuracy

```bash
MODEL=./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL,add_bos_token=True \
    --tasks gsm8k \
    --num_fewshot 5 \
    --limit 250 \
    --batch_size auto
```

!!! note
    Always include `add_bos_token=True` when evaluating quantized models.
    Quantized models can be sensitive to the presence of the `bos` token.

---

## Understanding the compressed-tensors format

vLLM's INT8 implementation uses the **compressed-tensors** format, which
supports multiple quantization schemes:

| Scheme | Weights | Activations | Notes |
|---|---|---|---|
| `W8A8` | INT8, per-channel | INT8, per-token (dynamic) | Recommended |
| `W8A8` static | INT8, per-channel | INT8, per-tensor (static) | Faster, less accurate |
| `W8A16` | INT8, per-channel | FP16 (unquantized) | Weight-only |

The scheme is stored in the checkpoint's `quantization_config` and is
automatically detected by vLLM.

---

## SmoothQuant explained

Activations in transformer models often have **outliers** — a small number of
channels with very large values. These outliers make activation quantization
difficult because they force a large quantization range, reducing precision
for the majority of values.

**SmoothQuant** addresses this by mathematically migrating the quantization
difficulty from activations to weights. It multiplies activations by a
per-channel smoothing factor $s$ and divides weights by the same factor:

$$
Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W^T)
$$

This makes activations easier to quantize while keeping the mathematical
equivalence of the computation. The `smoothing_strength` parameter (0–1)
controls how aggressively the difficulty is migrated. A value of 0.8 is a
good default for most models.

---

## Running a quantized model

### Command line

```bash
# Auto-detect quantization from config
vllm serve ./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token

# With tensor parallelism
vllm serve ./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token \
    --tensor-parallel-size 2
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")

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

## Best practices

- **Use 512+ calibration samples.** More samples generally improve accuracy.
  Increase to 1024 if you observe accuracy degradation.
- **Match calibration data to your deployment use case.** For a code model,
  calibrate on code data. For a chat model, use chat-formatted data.
- **Use the chat template.** Apply the model's chat template when preparing
  calibration data to match the format used during fine-tuning.
- **Use `smoothing_strength=0.8` as a starting point.** Adjust between 0.5
  and 0.9 if accuracy is unsatisfactory.
- **Do not quantize `lm_head`.** The language model head is sensitive to
  quantization and is typically excluded.
- **Consider FP8 for Hopper/Ada GPUs.** FP8 W8A8 delivers higher throughput
  than INT8 on Ada Lovelace and Hopper hardware.

---

## INT8 vs. FP8 comparison

| Feature | INT8 W8A8 | FP8 W8A8 |
|---|---|---|
| Min. GPU | Turing (SM 7.5) | Ada (SM 8.9) |
| Memory reduction | ~2× vs. BF16 | ~2× vs. BF16 |
| Throughput gain | Up to 1.3× | Up to 1.6× |
| Calibration required | Yes | Optional (dynamic mode) |
| Blackwell support | ❌ | ✅ |
| AMD support | ❌ | ✅ (MI300X) |

For Hopper and Ada GPUs, FP8 is generally preferred due to higher throughput.
For Turing and Ampere GPUs, INT8 is the best option for W8A8 quantization.

---

## Troubleshooting

**`INT8 is not supported on compute capability >= 10.0`**
: Blackwell GPUs do not support INT8 W8A8. Use [FP8](fp8.md) instead.

**Low accuracy after quantization**
: Try increasing `NUM_CALIBRATION_SAMPLES` to 1024 or more. Also ensure the
  calibration data matches your deployment distribution.

**`SmoothQuantModifier` fails with NaN**
: Reduce `smoothing_strength` to 0.5 or 0.6. Some models are sensitive to
  aggressive smoothing.

---

## Related pages

- [FP8](fp8.md) — higher-throughput 8-bit quantization for Ada/Hopper GPUs
- [INT4 W4A16](int4.md) — 4-bit weight-only quantization for maximum memory
  savings
- [LLM Compressor](llm_compressor.md) — recommended quantization toolkit
- [Quantization overview](index.md) — method comparison and hardware matrix
