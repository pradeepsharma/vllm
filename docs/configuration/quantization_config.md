# Quantization Configuration

vLLM supports a wide range of quantization methods for reducing model memory footprint and improving inference throughput. Quantization is configured primarily through the `--quantization` flag and the model's built-in quantization config.

**Related config:** `ModelConfig.quantization` in `vllm/config/model.py`  
**CLI flag:** `--quantization`

---

## Overview

Quantization reduces the precision of model weights (and sometimes activations) from FP16/BF16 to lower-bit formats. This reduces:
- **Memory usage** — Smaller weights fit more models on fewer GPUs
- **Memory bandwidth** — Faster weight loading during inference
- **Compute** — Some formats enable faster matrix multiplications

```
FP16 (16-bit) → INT8 (8-bit) → FP8 (8-bit) → INT4 (4-bit) → INT2 (2-bit)
Memory:  2×      1×              1×              0.5×           0.25×
```

---

## Supported Quantization Methods

### `--quantization` / `-q`

```
Type:    str | None
Default: None (auto-detect from model config)
CLI:     --quantization
```

Specify the quantization method. If `None`, vLLM reads `quantization_config` from the model's `config.json`. If that is also absent, the model is assumed to be unquantized.

---

## Weight-Only Quantization

### AWQ — Activation-aware Weight Quantization

```bash
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq
```

- **Bits:** INT4
- **Activation:** FP16
- **Memory reduction:** ~4× vs FP16
- **Notes:** Requires AWQ-quantized model weights. Fast on NVIDIA GPUs with Marlin kernel.

### GPTQ — Generalized Post-Training Quantization

```bash
vllm serve TheBloke/Llama-2-7B-GPTQ --quantization gptq
```

- **Bits:** INT4 or INT8
- **Activation:** FP16
- **Memory reduction:** ~4× (INT4) or ~2× (INT8) vs FP16
- **Notes:** Widely supported. Multiple GPTQ variants: `gptq`, `gptq_marlin`, `gptq_marlin_24`.

### GGUF

```bash
vllm serve /path/to/model.gguf \
  --load-format gguf \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

- **Bits:** Q2_K through Q8_0 (various)
- **Notes:** Requires `--load-format gguf`. Quantization method auto-detected from file.

### BitsAndBytes

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes
```

- **Bits:** INT4 (NF4) or INT8
- **Notes:** Dynamic quantization at load time. No pre-quantized weights needed.

---

## FP8 Quantization

FP8 provides a good balance between accuracy and efficiency, especially on NVIDIA H100/H200 and AMD MI300X GPUs with native FP8 compute support.

### FP8 Weight Quantization

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --quantization fp8
```

- **Bits:** FP8 (E4M3 or E5M2)
- **Activation:** FP8 or FP16
- **Memory reduction:** ~2× vs FP16
- **Notes:** Best performance on H100/H200 (SM90) and MI300X.

### FP8 KV Cache

FP8 KV cache is separate from weight quantization and can be combined with any weight format:

```bash
vllm serve mymodel \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales
```

See [Cache Config](cache_config.md) for details.

---

## Activation Quantization

### SmoothQuant

```bash
vllm serve mymodel --quantization smoothquant
```

- **Bits:** INT8 weights + INT8 activations
- **Notes:** Requires pre-quantized model with SmoothQuant calibration data.

---

## Sparse Quantization

### SqueezeLLM

```bash
vllm serve mymodel --quantization squeezellm
```

- **Bits:** INT4 with sparse outliers
- **Notes:** Combines weight quantization with sparse representation of outlier values.

---

## Advanced Quantization Methods

### Marlin

```bash
vllm serve mymodel --quantization marlin
```

- **Bits:** INT4
- **Notes:** Highly optimized Marlin kernel for NVIDIA GPUs. Often used automatically for AWQ/GPTQ models.

### Marlin 2:4 Sparse

```bash
vllm serve mymodel --quantization marlin24
```

- **Bits:** INT4 with 2:4 sparsity
- **Notes:** Combines INT4 quantization with NVIDIA 2:4 structured sparsity.

### DeepSpeedFP

```bash
vllm serve mymodel --quantization deepspeedfp
```

- **Notes:** DeepSpeed FP quantization format.

### EETQ

```bash
vllm serve mymodel --quantization eetq
```

- **Bits:** INT8
- **Notes:** Efficient INT8 quantization with per-channel scales.

### ExLlamaV2

```bash
vllm serve mymodel --quantization exl2
```

- **Bits:** Variable (2–8 bits per weight)
- **Notes:** ExLlamaV2 format with mixed-precision quantization.

### QuIP#

```bash
vllm serve mymodel --quantization quip
```

- **Notes:** Quantization with Incoherence Processing.

### HQQ

```bash
vllm serve mymodel --quantization hqq
```

- **Notes:** Half-Quadratic Quantization.

### FBGEMM FP8

```bash
vllm serve mymodel --quantization fbgemm_fp8
```

- **Notes:** FBGEMM-based FP8 quantization.

### NF4

```bash
vllm serve mymodel --quantization bitsandbytes
```

- **Bits:** NF4 (4-bit NormalFloat)
- **Notes:** Used by BitsAndBytes 4-bit quantization.

---

## FP4 Quantization (NVIDIA Blackwell)

NVIDIA Blackwell (B100/B200) GPUs support native FP4 compute:

```bash
vllm serve mymodel --quantization nvfp4
```

- **Bits:** FP4 (E2M1)
- **Notes:** Requires Blackwell GPU (SM100+). ~4× memory reduction vs FP16.

---

## Quantization Selection Guide

| Use Case | Recommended Method | Notes |
|---|---|---|
| Best accuracy | FP8 (`fp8`) | Minimal accuracy loss, 2× memory reduction |
| Best throughput (H100) | FP8 or AWQ Marlin | Native FP8 compute on H100 |
| Maximum compression | GPTQ INT4 or AWQ | ~4× memory reduction |
| No pre-quantized model | BitsAndBytes | Dynamic quantization at load time |
| GGUF ecosystem | GGUF | Wide community support |
| Blackwell GPU | FP4 (`nvfp4`) | Native FP4 compute |

---

## Deprecated Quantization Methods

Some older quantization methods are deprecated. To use them, set:

```bash
vllm serve mymodel \
  --quantization <deprecated_method> \
  --allow-deprecated-quantization
```

---

## Combining Quantization with Other Features

### Quantization + LoRA

```bash
vllm serve mymodel \
  --quantization awq \
  --enable-lora \
  --max-lora-rank 64
```

### Quantization + Prefix Caching

```bash
vllm serve mymodel \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### Quantization + Tensor Parallelism

```bash
vllm serve mymodel \
  --quantization gptq \
  --tensor-parallel-size 4
```

---

## Environment Variables for Quantization

| Variable | Default | Description |
|---|---|---|
| `VLLM_USE_TRITON_AWQ` | `0` | Use Triton AWQ kernels instead of CUDA |
| `VLLM_DISABLED_KERNELS` | `""` | Comma-separated list of kernels to disable |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `0` | Use atomic add in Marlin kernel |
| `VLLM_MARLIN_INPUT_DTYPE` | `None` | Override Marlin input dtype (`int8`, `fp8`) |
| `VLLM_USE_DEEP_GEMM` | `1` | Use DeepGEMM for FP8 GEMM |
| `VLLM_MOE_USE_DEEP_GEMM` | `1` | Use DeepGEMM for MoE FP8 GEMM |
| `VLLM_MXFP4_USE_MARLIN` | `None` | Use Marlin for MXFP4 |

See [Environment Variables](environment_variables.md) for the complete list.
