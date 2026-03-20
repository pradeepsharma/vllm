---
description: >
  Complete guide to FP8 quantization in vLLM, covering W8A8 and W8A16 modes,
  static and dynamic activation schemes, block-wise quantization, FP8 KV cache,
  and online dynamic quantization.
---

# FP8

**FP8** (8-bit floating point) quantization stores model weights and optionally
activations in 8-bit floating-point format. Compared to BF16, FP8 W8A8
delivers up to a **1.6× throughput improvement** and a **2× memory reduction**
with minimal accuracy loss.

vLLM supports two FP8 formats defined by IEEE 754:

- **E4M3** — 1 sign bit, 4 exponent bits, 3 mantissa bits. Range: ±448.
  Used for weights and activations in most FP8 deployments.
- **E5M2** — 1 sign bit, 5 exponent bits, 2 mantissa bits. Range: ±57344.
  Higher dynamic range but lower precision. Used for gradients in training.

And two operating modes:

- **W8A8** — weights and activations are both quantized to FP8. Enables
  hardware-accelerated FP8 matrix multiplications. Requires Ada Lovelace
  (SM 8.9) or Hopper (SM 9.0) GPUs, or AMD MI300X.
- **W8A16** — only weights are quantized to FP8; activations remain in FP16.
  Uses the FP8 Marlin kernel. Supported on Turing (SM 7.5) and newer GPUs.

---

## Hardware requirements

| Mode | Min. compute capability | Notes |
|---|---|---|
| W8A8 (native FP8) | SM 8.9 (Ada Lovelace) | Full hardware acceleration |
| W8A8 (native FP8) | SM 9.0 (Hopper) | Full hardware acceleration |
| W8A8 | AMD MI300X (ROCm) | Uses E4M3FNUZ format |
| W8A16 (Marlin) | SM 7.5 (Turing) | Weight-only, FP16 activations |

!!! note
    On AMD ROCm, vLLM uses the `E4M3FNUZ` variant of FP8, which has a
    slightly different range than the CUDA `E4M3FN` format. Scale factors
    are automatically adjusted by a factor of 2.

---

## Quick start

### Serving a pre-quantized FP8 model

```bash
# Serve a pre-quantized FP8 checkpoint — quantization is auto-detected
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-FP8
```

A curated collection of FP8 checkpoints is available on Hugging Face:
[neuralmagic/fp8-llms-for-vllm](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127).

### Online dynamic quantization (no checkpoint required)

You can quantize any BF16/FP16 model to FP8 on the fly without a pre-quantized
checkpoint. All `Linear` layers (except `lm_head`) have their weights quantized
to FP8 E4M3 with a per-tensor scale. Activations are quantized dynamically
per-tensor during each forward pass.

```python
from vllm import LLM, SamplingParams

# Quantize on the fly — no pre-quantized checkpoint needed
llm = LLM(model="facebook/opt-125m", quantization="fp8")

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["Hello, my name is"], sampling_params)
print(outputs[0].outputs[0].text)
```

```bash
# Command-line equivalent
vllm serve facebook/opt-125m --quantization fp8
```

!!! warning
    Online dynamic quantization loads the model at full precision before
    quantizing. You need enough GPU memory to hold the full BF16/FP16 model.
    Latency improvements are limited compared to a pre-quantized checkpoint
    because scales are computed dynamically.

---

## Activation schemes

FP8 supports two activation quantization schemes:

### Dynamic activation (recommended for accuracy)

Scale factors for activations are computed on the fly for each forward pass.
No calibration data is required. This is the default for online quantization
and for most pre-quantized checkpoints.

```python
from vllm import LLM

# Dynamic activation scheme (default)
llm = LLM(
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8-Dynamic",
    quantization="fp8",
)
```

### Static activation

Scale factors are computed once during calibration and stored in the
checkpoint. This is faster at inference time but requires a calibration step.

```python
from vllm import LLM

# Static activation scheme — scales are loaded from the checkpoint
llm = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8")
```

---

## Quantizing a model with LLM Compressor

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is the
recommended tool for producing FP8 checkpoints.

### Installation

```bash
pip install llmcompressor
```

### Dynamic FP8 quantization (no calibration data)

This is the simplest approach. It uses per-channel static weight quantization
and dynamic per-token activation quantization. No calibration data is needed.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-FP8-Dynamic"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# FP8_DYNAMIC: static per-channel weights + dynamic per-token activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### Static FP8 quantization (with calibration data)

Static quantization requires calibration data to compute activation scales.
It delivers better throughput than dynamic quantization.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-FP8-Static"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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

# FP8: static per-channel weights + static per-tensor activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",
    ignore=["lm_head"],
)

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

## Block-wise FP8 quantization

Block-wise (also called **fine-grained**) FP8 quantization applies separate
scale factors to small blocks of weights (e.g., 128×128). This improves
accuracy compared to per-tensor or per-channel scaling, at the cost of
slightly more memory for the scale tensors.

Block-wise quantization is supported for pre-quantized checkpoints with
`weight_block_size` set in the quantization config. It requires dynamic
activation quantization.

```python
from vllm import LLM

# Block-wise FP8 checkpoint (weight_block_size=[128, 128] in config)
llm = LLM(model="neuralmagic/DeepSeek-V3-FP8")
```

!!! note
    Block-wise FP8 quantization is only supported for pre-quantized
    checkpoints. Online dynamic quantization always uses per-tensor scaling.

---

## FP8 KV cache

You can quantize the attention KV cache to FP8 independently of the model
weights. This reduces KV cache memory usage, enabling longer context windows
and higher throughput.

See the dedicated [KV cache quantization guide](kv_cache_quantization.md) for
full details. A quick example:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,  # Estimate scales from a warmup batch
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["London is the capital of"], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Running a quantized model

### Command line

```bash
# Auto-detect quantization from checkpoint config
vllm serve ./Meta-Llama-3-8B-Instruct-FP8-Dynamic

# Explicit FP8 quantization
vllm serve ./Meta-Llama-3-8B-Instruct-FP8-Dynamic \
    --quantization fp8
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./Meta-Llama-3-8B-Instruct-FP8-Dynamic")

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

```bash
pip install "lm-eval[api]>=0.4.11"
```

```bash
MODEL=./Meta-Llama-3-8B-Instruct-FP8-Dynamic

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL,add_bos_token=True \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --limit 250
```

Expected results for `Meta-Llama-3-8B-Instruct-FP8-Dynamic` on GSM8K:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

---

## Configuration reference

FP8 checkpoints use a `quantization_config` section in `config.json`:

| Field | Description | Example |
|---|---|---|
| `quant_method` | Must contain `"fp8"` | `"fp8"` |
| `activation_scheme` | `"dynamic"` or `"static"` | `"dynamic"` |
| `ignored_layers` | Layers to skip | `["lm_head"]` |
| `weight_block_size` | Block size for fine-grained quantization | `[128, 128]` |

---

## Tensor parallelism

FP8 models support tensor parallelism:

```bash
vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-FP8 \
    --tensor-parallel-size 4
```

---

## Best practices

- **Use pre-quantized checkpoints for production.** Online dynamic quantization
  is convenient but delivers lower throughput than a properly calibrated
  static checkpoint.
- **Use `FP8_DYNAMIC` for the best accuracy/speed tradeoff.** Static
  per-channel weights with dynamic per-token activations is the recommended
  scheme for most models.
- **Use block-wise quantization for large models.** Fine-grained scaling
  (e.g., 128×128 blocks) significantly improves accuracy for models with
  large weight tensors, such as DeepSeek-V3.
- **Combine FP8 weights with FP8 KV cache** for maximum memory savings.
- **On AMD ROCm**, scale factors are automatically doubled to account for the
  `E4M3FNUZ` format difference. No manual adjustment is needed.

---

## Troubleshooting

**`FP8 computation is not supported on this GPU`**
: FP8 W8A8 requires Ada Lovelace (SM 8.9) or Hopper (SM 9.0). On older GPUs,
  vLLM falls back to FP8 W8A16 using the Marlin kernel (Turing+).

**Model produces NaN outputs**
: This can happen with static activation quantization if the calibration data
  does not cover the full activation range. Try switching to dynamic
  activation quantization (`activation_scheme: dynamic`).

**Slow inference despite FP8**
: Ensure you are using a pre-quantized checkpoint with static scales. Online
  dynamic quantization has limited throughput gains because scales are
  computed per-forward-pass.

---

## Related pages

- [KV cache quantization](kv_cache_quantization.md) — quantize the KV cache
  to FP8 for longer context windows
- [LLM Compressor](llm_compressor.md) — recommended tool for producing FP8
  checkpoints
- [INT8 W8A8](int8.md) — alternative 8-bit quantization for Turing+ GPUs
- [Quantization overview](index.md) — method comparison and hardware matrix
