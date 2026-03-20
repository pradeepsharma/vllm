---
description: >
  Complete guide to AWQ (Activation-aware Weight Quantization) in vLLM,
  covering 4-bit weight quantization, the Marlin kernel, and how to quantize
  models with LLM Compressor or AutoAWQ.
---

# AWQ

**AWQ** (Activation-aware Weight Quantization) is a post-training quantization
method that compresses model weights to 4 bits while keeping activations in
FP16 or BF16. Unlike GPTQ, AWQ identifies and protects the small fraction of
weights that are most important to model accuracy (based on activation
magnitudes), then quantizes the rest aggressively.

vLLM supports two AWQ backends:

- **`awq`** — the original AWQ GEMM kernel. Supports 4-bit weights on GPUs
  with compute capability ≥ 7.5 (Turing and newer).
- **`awq_marlin`** — the high-performance Marlin kernel. Delivers
  significantly higher throughput and is selected automatically when the model
  is compatible (Turing+ GPUs).

!!! tip
    For new quantization workflows, use [LLM Compressor](llm_compressor.md).
    Over 6,500 pre-quantized AWQ models are available on
    [Hugging Face](https://huggingface.co/models?search=awq).

!!! warning
    The [AutoAWQ](auto_awq.md) library is deprecated. Its functionality has
    been adopted by [LLM Compressor](llm_compressor.md). For new projects,
    use LLM Compressor instead.

---

## Hardware requirements

| Backend | Min. compute capability | Supported bits | Activation dtype |
|---|---|---|---|
| `awq` | SM 7.5 (Turing) | 4 | FP16 |
| `awq_marlin` | SM 7.5 (Turing) | 4 | FP16, BF16 |

!!! note
    AWQ is not supported on Volta GPUs (SM 7.0) or older. Use
    [GPTQ](gptq.md) for broader GPU compatibility.

---

## Quick start

### Serving a pre-quantized AWQ model

```bash
# Serve a 4-bit AWQ model — quantization is auto-detected from config
vllm serve TheBloke/Llama-2-7b-Chat-AWQ
```

You can also specify the backend explicitly:

```bash
vllm serve TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq_marlin
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    quantization="awq_marlin",  # or "awq" for the original kernel
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["Tell me about quantization."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Quantizing a model with LLM Compressor

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is the
recommended tool for creating AWQ checkpoints. It produces models that are
fully compatible with vLLM's Marlin kernel.

### Installation

```bash
pip install llmcompressor
```

### Basic 4-bit AWQ quantization

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-AWQ-W4A16"

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

# Apply AWQ W4A16 quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
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

## Quantizing with AutoAWQ (legacy)

!!! warning
    AutoAWQ is deprecated. Use [LLM Compressor](llm_compressor.md) for new
    projects. The instructions below are provided for reference only.

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "mistralai/Mistral-7B-Instruct-v0.2"
quant_path = "mistral-instruct-v0.2-awq"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize and save
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

---

## Running a quantized model

### Command line

```bash
# Auto-detect quantization from config
vllm serve ./Meta-Llama-3-8B-Instruct-AWQ-W4A16

# Explicitly use the Marlin kernel
vllm serve ./Meta-Llama-3-8B-Instruct-AWQ-W4A16 \
    --quantization awq_marlin
```

### Python API

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

llm = LLM(model="./Meta-Llama-3-8B-Instruct-AWQ-W4A16")
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
MODEL=./Meta-Llama-3-8B-Instruct-AWQ-W4A16

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL,add_bos_token=True \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto \
    --limit 250
```

---

## Configuration reference

AWQ checkpoints include a `quant_config.json` or `quantize_config.json` file.
Key fields:

| Field | Description | Example |
|---|---|---|
| `w_bit` / `bits` | Weight bit width | `4` |
| `q_group_size` / `group_size` | Quantization group size | `128` |
| `zero_point` | Whether zero-point quantization is used | `true` |
| `modules_to_not_convert` | Layers to skip | `["lm_head"]` |

---

## AWQ vs. GPTQ comparison

| Feature | AWQ | GPTQ |
|---|---|---|
| Supported bits | 4 only | 2, 3, 4, 8 |
| Min. GPU | Turing (SM 7.5) | Pascal (SM 6.0) |
| Activation dtype | FP16, BF16 | FP16 (Marlin: BF16) |
| Calibration data | Required | Required |
| Zero-point support | Yes | Yes |
| Marlin kernel | Yes (`awq_marlin`) | Yes (`gptq_marlin`) |
| MoE support | Via Marlin | Via MoeWNA16 |

Both methods deliver similar accuracy at 4-bit precision. AWQ is often
slightly faster at low batch sizes due to its activation-aware weight
selection. GPTQ supports more bit widths and has broader GPU compatibility.

---

## Tensor parallelism

AWQ models support tensor parallelism:

```bash
vllm serve TheBloke/Llama-2-70B-Chat-AWQ \
    --quantization awq_marlin \
    --tensor-parallel-size 4
```

---

## Best practices

- **Use `awq_marlin` on Turing+ GPUs.** It is significantly faster than the
  original `awq` kernel and is selected automatically for compatible models.
- **Use group size 128.** This is the standard default.
- **Use 512+ calibration samples** from data that matches your deployment
  distribution.
- **Do not quantize `lm_head`.** The language model head is sensitive to
  quantization and is typically excluded.

---

## Troubleshooting

**`ValueError: Currently, only 4-bit weight quantization is supported for AWQ`**
: AWQ only supports 4-bit weights. For other bit widths, use [GPTQ](gptq.md).

**Model produces low-quality output**
: Try using `awq_marlin` instead of `awq`. Also verify that the calibration
  data matches your deployment use case.

**Out-of-memory during quantization**
: Reduce the number of calibration samples or use a smaller batch size.

---

## Related pages

- [AutoAWQ (legacy)](auto_awq.md) — legacy AutoAWQ library documentation
- [GPTQ](gptq.md) — alternative weight-only quantization with more bit widths
- [LLM Compressor](llm_compressor.md) — recommended quantization toolkit
- [Quantization overview](index.md) — method comparison and hardware matrix
