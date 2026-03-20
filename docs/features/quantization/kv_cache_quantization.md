---
description: >
  Guide to KV cache quantization in vLLM, covering FP8 KV cache with per-tensor
  and per-attention-head scaling, scale calibration approaches, and integration
  with FP8 model weights.
---

# KV cache quantization

The **KV cache** (Key-Value cache) stores the attention keys and values for
all tokens in the context window. For long-context workloads, the KV cache
can consume a significant fraction of GPU memory — often more than the model
weights themselves.

Quantizing the KV cache to FP8 reduces its memory footprint by approximately
**2×** compared to BF16, enabling:

- **Longer context windows** — more tokens fit in the same GPU memory
- **Higher throughput** — more concurrent requests can be served
- **Lower cost** — fewer GPUs needed for the same workload

!!! note
    When using the Flash Attention 3 backend with FP8 KV cache, attention
    operations are performed in the quantized (FP8) domain. In this
    configuration, queries are also quantized to FP8.

---

## Supported formats

| Format | Description | Hardware |
|---|---|---|
| `fp8` / `fp8_e4m3` | FP8 E4M3 format | CUDA 11.8+, ROCm (AMD) |
| `fp8_e5m2` | FP8 E5M2 format | CUDA 11.8+ |
| `auto` | Use model's default dtype | All |

On AMD ROCm, vLLM uses the `E4M3FNUZ` variant of FP8, which has a slightly
different range than the CUDA `E4M3FN` format. Scale factors are automatically
adjusted.

---

## Quantization strategies

### Per-tensor quantization

A single scale factor is applied to each Q, K, and V tensor:
- `q_scale`: shape `[1]`
- `k_scale`: shape `[1]`
- `v_scale`: shape `[1]`

This is the default and most widely supported strategy.

### Per-attention-head quantization

Each scale factor corresponds to one attention head:
- `q_scale`: shape `[num_heads]`
- `k_scale`: shape `[num_kv_heads]`
- `v_scale`: shape `[num_kv_heads]`

!!! note
    Per-attention-head quantization is only available with the
    **Flash Attention backend** and requires calibration via
    [LLM Compressor](llm_compressor.md).

---

## Scale calibration approaches

vLLM supports three approaches for computing KV cache scale factors:

### 1. No calibration (default scales = 1.0)

All scale factors are set to 1.0. This is the simplest approach and works
well for many models, but may reduce accuracy for models with large activation
ranges.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,  # Use scale = 1.0 (default)
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)
outputs = llm.generate(["London is the capital of"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 2. Random token calibration (on-the-fly)

Scale factors are estimated from a single batch of random tokens during
warmup and then fixed for the rest of inference. This is a good middle ground
between no calibration and full dataset calibration.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,  # Estimate scales from warmup batch
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)
outputs = llm.generate(["London is the capital of"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 3. Dataset calibration with LLM Compressor (recommended)

For the highest accuracy, calibrate scale factors using a representative
dataset. This requires [LLM Compressor](https://github.com/vllm-project/llm-compressor).

```bash
pip install llmcompressor
```

```python
"""
Quantize Llama attention + KV cache to FP8 using dataset calibration.
Supports both per-tensor and per-attention-head strategies.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
STRATEGY = "tensor"       # or "attn_head" for per-head quantization
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048

def process_and_tokenize(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],
                input_activations=fp8_args,  # Quantize queries (q_scale)
            )
        },
        kv_cache_scheme=fp8_args,  # Quantize KV cache (k/v_scale)
    )

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Prepare calibration dataset
ds = load_dataset(DATASET_ID, split=f"train_sft[:{NUM_CALIB_SAMPLES}]")
ds = ds.shuffle(seed=42)
ds = ds.map(
    lambda ex: process_and_tokenize(ex, tokenizer),
    remove_columns=ds.column_names,
)

# Apply quantization
recipe = build_recipe(STRATEGY)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_CALIB_SAMPLES,
)

# Save the quantized model
save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvcache-fp8-{STRATEGY}"
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
```

For more examples, see the
[llm-compressor KV cache examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).

---

## Combining KV cache and weight quantization

You can combine FP8 KV cache quantization with FP8 weight quantization for
maximum memory savings:

```python
from vllm import LLM, SamplingParams

# FP8 weights + FP8 KV cache
llm = LLM(
    model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain attention mechanisms."], sampling_params)
print(outputs[0].outputs[0].text)
```

```bash
# Command-line equivalent
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-FP8 \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales
```

---

## Loading a checkpoint with calibrated KV scales

If your checkpoint includes calibrated KV cache scales (produced by LLM
Compressor), vLLM loads them automatically:

```python
from vllm import LLM

# Checkpoint includes calibrated k_scale and v_scale
llm = LLM(
    model="./Llama-3.1-8B-Instruct-kvcache-fp8-tensor",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,  # Use scales from checkpoint
)
```

The scale parameters in the checkpoint follow this naming convention:

| Parameter | Description |
|---|---|
| `*.attn.k_scale` | Key cache scale factor |
| `*.attn.v_scale` | Value cache scale factor |
| `*.attn.q_scale` | Query scale factor (Flash Attention 3 only) |
| `*.attn.prob_scale` | Attention probability scale (Flash Attention 3 only) |

---

## Command-line reference

| Flag | Description |
|---|---|
| `--kv-cache-dtype fp8` | Enable FP8 KV cache quantization |
| `--kv-cache-dtype fp8_e4m3` | Explicitly use E4M3 format |
| `--kv-cache-dtype fp8_e5m2` | Use E5M2 format (higher range, lower precision) |
| `--kv-cache-dtype auto` | Use model's default dtype (no quantization) |
| `--calculate-kv-scales` | Estimate scales from a warmup batch |

---

## Memory savings

The memory savings from FP8 KV cache quantization depend on the model
architecture and context length:

| Context length | BF16 KV cache | FP8 KV cache | Savings |
|---|---|---|---|
| 4K tokens | ~1 GB | ~0.5 GB | ~50% |
| 32K tokens | ~8 GB | ~4 GB | ~50% |
| 128K tokens | ~32 GB | ~16 GB | ~50% |

*Approximate values for a 7B-parameter model with 32 attention heads.*

The actual savings depend on the number of KV heads, the hidden dimension,
and the number of layers.

---

## Best practices

- **Use dataset calibration for production.** Random token calibration is
  convenient but may not capture the full activation range for your use case.
- **Use per-attention-head quantization for maximum accuracy.** It provides
  finer-grained scaling at the cost of requiring the Flash Attention backend.
- **Combine with FP8 weight quantization** for maximum memory savings on
  Ada Lovelace and Hopper GPUs.
- **Use `calculate_kv_scales=True` as a quick start.** It provides better
  accuracy than default scales (1.0) without requiring calibration data.
- **Monitor for accuracy degradation.** If you observe quality issues, switch
  from `fp8_e4m3` to `fp8_e5m2` (higher dynamic range) or use calibrated
  scales.

---

## Troubleshooting

**`Using KV cache scaling factor 1.0 for fp8_e4m3`**
: This warning appears when no calibrated scales are found in the checkpoint
  and `calculate_kv_scales=False`. Either set `calculate_kv_scales=True` or
  use a checkpoint with calibrated scales.

**`Only support per-tensor scaling factor for fp8 KV cache`**
: Per-attention-head quantization requires the Flash Attention backend. Ensure
  you are using Flash Attention 3 and that the checkpoint was calibrated with
  the `attn_head` strategy.

**Accuracy degradation with FP8 KV cache**
: Try switching to `calculate_kv_scales=True` or using dataset-calibrated
  scales. Also consider using `fp8_e5m2` instead of `fp8_e4m3` for models
  with large activation ranges.

---

## Related pages

- [FP8](fp8.md) — FP8 weight quantization
- [LLM Compressor](llm_compressor.md) — recommended tool for calibrating KV
  cache scales
- [Quantization overview](index.md) — method comparison and hardware matrix
