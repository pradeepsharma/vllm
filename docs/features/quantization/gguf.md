---
description: >
  Guide to loading and serving GGUF-format quantized models in vLLM, including
  supported quantization types, HuggingFace loading, local file loading, and
  tensor parallelism.
---

# GGUF

**GGUF** (GPT-Generated Unified Format) is a binary file format developed by
the [llama.cpp](https://github.com/ggerganov/llama.cpp) project. It packages
model weights, tokenizer data, and metadata into a single self-contained file.
GGUF supports a wide range of quantization types, from 2-bit to 8-bit, using
mixed-precision schemes that quantize different parts of the model at different
bit widths.

vLLM can load GGUF files directly, making it easy to serve models from the
large ecosystem of GGUF-quantized models available on Hugging Face.

!!! warning
    GGUF support in vLLM is experimental and may be incompatible with some
    features. If you encounter issues, please report them to the vLLM team.

!!! warning
    vLLM currently supports only **single-file** GGUF models. If you have a
    multi-file GGUF model (split across multiple `.gguf` files), use the
    [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) tool to
    merge them into a single file before loading.

---

## Supported quantization types

GGUF supports many quantization types. The most commonly used ones are:

| Type | Description | Bits (approx.) |
|---|---|---|
| `Q2_K` | 2-bit, k-quant | ~2.6 |
| `Q3_K_S` / `Q3_K_M` / `Q3_K_L` | 3-bit, k-quant (small/medium/large) | ~3.4 |
| `Q4_0` | 4-bit, legacy | 4.0 |
| `Q4_K_S` / `Q4_K_M` | 4-bit, k-quant (small/medium) | ~4.5 |
| `Q5_K_S` / `Q5_K_M` | 5-bit, k-quant (small/medium) | ~5.5 |
| `Q6_K` | 6-bit, k-quant | ~6.6 |
| `Q8_0` | 8-bit, legacy | 8.0 |
| `F16` | 16-bit float (no quantization) | 16.0 |
| `BF16` | BF16 float (no quantization) | 16.0 |
| `IQ2_XXS` / `IQ2_XS` | 2-bit, importance-matrix quant | ~2.1–2.3 |
| `IQ3_XXS` / `IQ3_XS` | 3-bit, importance-matrix quant | ~3.1–3.3 |
| `IQ4_XS` / `IQ4_NL` | 4-bit, importance-matrix quant | ~4.3 |

**K-quant types** (e.g., `Q4_K_M`) use mixed precision: some layers are
quantized at a higher bit width to preserve accuracy. The `_M` (medium) and
`_L` (large) variants use more high-precision layers than `_S` (small).

**Importance-matrix quants** (e.g., `IQ4_XS`) use an importance matrix to
identify which weights are most critical and quantize them at higher precision.

!!! note
    GGUF dequantization kernels use FP16 internally. On Blackwell GPUs
    (SM 10.x), BF16 has precision issues; use FP16 or FP32 instead.

---

## Hardware requirements

| Platform | Supported |
|---|---|
| NVIDIA GPU (SM 6.0+) | ✅ |
| AMD GPU (ROCm) | ✅ |
| x86 CPU | ❌ (GPU required) |

---

## Loading from Hugging Face

vLLM supports loading GGUF models directly from Hugging Face using the
`repo_id:quant_type` format.

### Command line

```bash
# Load Q4_K_M quantization from unsloth/Qwen3-0.6B-GGUF
# Use the tokenizer from the base model to avoid conversion issues
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
    --tokenizer Qwen/Qwen3-0.6B
```

!!! tip
    Always use the tokenizer from the base (unquantized) model rather than
    the GGUF model. Tokenizer conversion from GGUF is slow and can be
    unreliable, especially for models with large vocabularies.

### With tensor parallelism

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
    --tokenizer Qwen/Qwen3-0.6B \
    --tensor-parallel-size 2
```

### With a custom HuggingFace config

If Hugging Face does not support your model's architecture, you can provide
a compatible config manually:

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
    --tokenizer Qwen/Qwen3-0.6B \
    --hf-config-path Qwen/Qwen3-0.6B
```

---

## Loading a local GGUF file

You can download a GGUF file and load it directly from disk:

```bash
# Download the file
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf

# Serve from local path
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf \
    --tokenizer Qwen/Qwen3-0.6B
```

---

## Python API

```python
from vllm import LLM, SamplingParams

# Create a sampling params object
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

# Load from Hugging Face using repo_id:quant_type format
llm = LLM(
    model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
    tokenizer="Qwen/Qwen3-0.6B",
)

# Generate text
outputs = llm.generate(["Tell me about the history of AI."], sampling_params)
print(outputs[0].outputs[0].text)
```

### Chat API

```python
from vllm import LLM, SamplingParams

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Write an essay about the importance of higher education."},
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(
    model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
    tokenizer="Qwen/Qwen3-0.6B",
)

outputs = llm.chat(conversation, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

---

## Choosing a quantization type

The right quantization type depends on your memory budget and accuracy
requirements:

| Priority | Recommended type | Notes |
|---|---|---|
| Best accuracy | `Q6_K` or `Q8_0` | Near-lossless quality |
| Balanced (recommended) | `Q4_K_M` | Good accuracy, ~4.5 bits/weight |
| Memory-constrained | `Q3_K_M` | Noticeable quality drop |
| Extreme compression | `Q2_K` or `IQ2_XXS` | Significant quality loss |

For most use cases, **`Q4_K_M`** is the recommended starting point. It
provides a good balance between model quality and memory efficiency.

---

## Finding GGUF models

Many popular models are available in GGUF format on Hugging Face. Some
well-known providers:

- [unsloth](https://huggingface.co/unsloth) — optimized GGUF models
- [TheBloke](https://huggingface.co/TheBloke) — large collection of GGUF models
- [bartowski](https://huggingface.co/bartowski) — recent model releases

Search for GGUF models at:
[huggingface.co/models?search=gguf](https://huggingface.co/models?search=gguf)

---

## Limitations

- **Single-file only.** Multi-file GGUF models must be merged before loading.
- **No CPU inference.** vLLM requires a GPU to run GGUF models.
- **Experimental status.** Some vLLM features may not work with GGUF models.
- **Blackwell BF16 precision.** On Blackwell GPUs, use FP16 or FP32 to avoid
  precision issues.
- **Tokenizer conversion.** Always use the base model tokenizer rather than
  the GGUF tokenizer for best results.

---

## Troubleshooting

**`Only single-file GGUF models are supported`**
: Merge your multi-file GGUF model using
  [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135):
  ```bash
  ./gguf-split --merge model-00001-of-00002.gguf merged-model.gguf
  ```

**Slow tokenizer loading**
: Use `--tokenizer <base-model-id>` to load the tokenizer from the base model
  instead of converting it from the GGUF file.

**`HuggingFace does not support this model architecture`**
: Provide a compatible config with `--hf-config-path <base-model-id>`.

**Precision issues on Blackwell**
: Add `--dtype float16` or `--dtype float32` to avoid BF16 precision issues
  on Blackwell GPUs.

---

## Related pages

- [GPTQ](gptq.md) — alternative weight-only quantization with Marlin kernel
- [AWQ](awq.md) — 4-bit weight-only quantization
- [BitsAndBytes](bitsandbytes.md) — no-calibration 4-bit and 8-bit quantization
- [Quantization overview](index.md) — method comparison and hardware matrix
