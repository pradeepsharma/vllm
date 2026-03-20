---
description: >
  Guide to MXFP4 (Microscaling FP4) quantization in vLLM for Mixture-of-Experts
  models, covering supported backends, hardware requirements, and how to load
  MXFP4-quantized checkpoints.
---

# MXFP4

**MXFP4** (Microscaling FP4) is a 4-bit floating-point quantization format
defined by the [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
It uses a shared microscale factor for small blocks of weights (typically 32
elements), which provides better accuracy than standard INT4 at the same bit
width.

vLLM's MXFP4 support is primarily focused on **Mixture-of-Experts (MoE)**
models, where the expert weight matrices are quantized to MXFP4. This delivers
significant memory savings and throughput improvements for large MoE models
such as Mixtral and DeepSeek.

!!! note
    MXFP4 linear layer quantization (non-MoE) is not yet implemented in vLLM.
    For MXFP4 linear layers, see [AMD Quark](quark.md) which provides this
    capability.

---

## Hardware requirements

| Platform | Supported | Backend |
|---|---|---|
| NVIDIA Hopper (SM 9.0) | ✅ | FlashInfer (BF16), Marlin, Triton |
| NVIDIA Blackwell (SM 10.x) | ✅ | FlashInfer (MXFP8 CUTLASS/TRTLLM), Marlin |
| NVIDIA Ampere (SM 8.x) | ✅ | Marlin |
| AMD MI300X (gfx950) | ✅ | CK (Aiter), Triton |
| Intel GPU (XPU) | ✅ | Marlin |

!!! note
    For best performance on Hopper and Blackwell GPUs, install FlashInfer:
    ```bash
    pip install vllm[flashinfer]
    ```

---

## Backend selection

vLLM automatically selects the best available backend for MXFP4 MoE:

| GPU | FlashInfer available | Backend selected |
|---|---|---|
| SM 10.x (Blackwell) | Yes | FlashInfer MXFP4+MXFP8 CUTLASS |
| SM 9.0 (Hopper) | Yes | FlashInfer MXFP4+BF16 |
| SM 9.0 / SM 10.x | No | Marlin or Triton |
| SM 8.x (Ampere) | — | Marlin |
| AMD gfx950 | — | CK (Aiter) |
| Other AMD | — | Triton |

You can override the backend selection with environment variables:

| Variable | Effect |
|---|---|
| `VLLM_MXFP4_USE_MARLIN=1` | Force Marlin backend |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1` | Force FlashInfer BF16 backend |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` | Force FlashInfer MXFP8 TRTLLM backend |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1` | Force FlashInfer MXFP8 CUTLASS backend |

---

## Quick start

### Loading a pre-quantized MXFP4 model

MXFP4 quantization is typically applied to MoE models using tools like
[LLM Compressor](llm_compressor.md) or [NVIDIA ModelOpt](modelopt.md).
The quantization format is stored in the checkpoint's `quantization_config`.

```bash
# Serve an MXFP4-quantized MoE model
vllm serve neuralmagic/DeepSeek-V3-MXFP4 \
    --tensor-parallel-size 8
```

### Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="neuralmagic/DeepSeek-V3-MXFP4",
    tensor_parallel_size=8,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain mixture-of-experts models."], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Quantizing a model with LLM Compressor

[LLM Compressor](https://github.com/vllm-project/llm-compressor) supports
MXFP4 quantization for MoE models through the compressed-tensors format.

```bash
pip install llmcompressor
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
SAVE_DIR = "Mixtral-8x7B-Instruct-MXFP4"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Apply MXFP4 quantization to MoE expert layers
recipe = QuantizationModifier(
    targets=["MixtralSparseMoeBlock"],
    scheme="MXFP4",
)

oneshot(model=model, recipe=recipe)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

---

## MXFP4 format details

MXFP4 uses the **E2M1** floating-point format:
- 1 sign bit
- 2 exponent bits
- 1 mantissa bit

Weights are stored in blocks of 32 elements. Each block shares a single
**microscale factor** stored in FP8 (E8M0 format). This shared scaling
provides better accuracy than per-tensor or per-channel scaling at the same
bit width.

The effective storage is approximately **4.5 bits per weight** (4 bits for
the value + 0.5 bits for the shared scale).

---

## Supported models

MXFP4 quantization is most beneficial for large MoE models:

- **DeepSeek-V2 / V3** — large MoE models with many experts
- **Mixtral 8x7B / 8x22B** — Mistral's MoE models
- **Qwen MoE** — Alibaba's MoE models

For dense (non-MoE) models, consider [FP8](fp8.md) or [GPTQ](gptq.md) instead.

---

## Tensor parallelism

MXFP4 MoE models support tensor parallelism. Large models typically require
multiple GPUs:

```bash
# DeepSeek-V3 requires 8 GPUs at MXFP4 precision
vllm serve neuralmagic/DeepSeek-V3-MXFP4 \
    --tensor-parallel-size 8

# Mixtral 8x7B fits on 2 GPUs at MXFP4 precision
vllm serve ./Mixtral-8x7B-Instruct-MXFP4 \
    --tensor-parallel-size 2
```

---

## LoRA support

MXFP4 supports LoRA adapters, but only through the Marlin or Triton backends.
FlashInfer backends do not currently support LoRA with MXFP4.

```python
from vllm import LLM

llm = LLM(
    model="./Mixtral-8x7B-Instruct-MXFP4",
    enable_lora=True,
    # LoRA forces Marlin or Triton backend
)
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_MXFP4_USE_MARLIN` | `0` | Force Marlin backend for MXFP4 MoE |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | `0` | Use FlashInfer BF16 backend on SM90 |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | `0` | Use FlashInfer MXFP8 TRTLLM backend |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS` | `0` | Use FlashInfer MXFP8 CUTLASS backend |

---

## Limitations

- **MoE only.** MXFP4 linear layer quantization is not yet implemented.
  Only MoE expert layers are quantized.
- **No attention quantization.** MXFP4 attention is not yet implemented.
- **Dimension alignment.** The CK backend on AMD requires
  `intermediate_size_per_partition` to be a multiple of the alignment
  requirement. If not met, vLLM falls back to Triton.
- **SM 11.x (Blackwell Ultra).** Some SM 11.x devices have known issues with
  Triton kernels. Use Marlin backend if you encounter problems.

---

## Troubleshooting

**`No compatible MXFP4 MoE backend found`**
: Ensure you are running on a supported GPU (SM 8.0+). Install FlashInfer
  for best performance on Hopper and Blackwell:
  ```bash
  pip install vllm[flashinfer]
  ```

**`CK MXFP4 MoE GEMM does not support intermediate_size_per_partition`**
: The model's intermediate size is not a multiple of the CK alignment
  requirement. vLLM will automatically fall back to Triton if available.
  If Triton is not available, adjust `tensor_parallel_size` to change the
  per-partition size.

**Low throughput on Hopper without FlashInfer**
: Install FlashInfer for significantly better performance:
  ```bash
  pip install vllm[flashinfer]
  ```

---

## Related pages

- [FP8](fp8.md) — 8-bit floating-point quantization for dense models
- [NVIDIA ModelOpt](modelopt.md) — NVIDIA's quantization toolkit with MXFP4
  support
- [AMD Quark](quark.md) — AMD's quantization toolkit with MXFP4 linear support
- [Quantization overview](index.md) — method comparison and hardware matrix
