---
description: >
  Overview of all quantization methods supported by vLLM, including a hardware
  compatibility matrix and guidance on choosing the right method for your use case.
---

# Quantization overview

Quantization reduces model precision to shrink memory footprint and increase
throughput. A 7B-parameter model stored in BF16 occupies roughly 14 GB of GPU
memory; the same model in INT4 fits in about 3.5 GB — a 4× reduction that
makes it possible to run larger models on fewer GPUs, or to serve more
concurrent requests on the same hardware.

vLLM supports a wide range of quantization formats, from industry-standard
methods such as GPTQ and AWQ to cutting-edge floating-point formats such as
FP8 and MXFP4.

!!! tip
    To get started quickly, see [LLM Compressor](llm_compressor.md) — the
    recommended tool for producing FP8, INT8, and INT4 checkpoints that work
    out of the box with vLLM.

---

## Supported methods

| Method | Format | Bits | Activation | Best for |
|---|---|---|---|---|
| [GPTQ](gptq.md) | INT | 2 / 3 / 4 / 8 | W-only (A16) | Broad GPU support, large model zoo |
| [AWQ](awq.md) | INT | 4 | W-only (A16) | Fast inference, Turing+ GPUs |
| [FP8](fp8.md) | Float | 8 | W8A8 or W8A16 | Hopper / Ada, highest throughput |
| [INT8 W8A8](int8.md) | INT | 8 | W8A8 | Turing+ GPUs, good accuracy |
| [INT4 W4A16](int4.md) | INT | 4 | W-only (A16) | Memory-constrained deployments |
| [GGUF](gguf.md) | Mixed | 2–8 | W-only | llama.cpp-compatible models |
| [BitsAndBytes](bitsandbytes.md) | INT / NF4 | 4 / 8 | W-only | No-calibration quantization |
| [TorchAO](torchao.md) | Various | 4 / 8 | Various | PyTorch-native workflows |
| [MXFP4](mxfp4.md) | MX Float | 4 | W4A8 (MoE) | Hopper / Blackwell MoE models |
| [KV cache quantization](kv_cache_quantization.md) | FP8 | 8 | KV cache | Longer context, higher throughput |
| [LLM Compressor](llm_compressor.md) | Various | 4 / 8 | Various | Recommended quantization toolkit |
| [AutoAWQ](auto_awq.md) | INT | 4 | W-only | Legacy AWQ checkpoints |
| [GPTQModel](gptqmodel.md) | INT | 4 / 8 | W-only | Dynamic per-layer quantization |
| [NVIDIA ModelOpt](modelopt.md) | FP8 / FP4 | 4 / 8 | Various | NVIDIA-optimized models |
| [AMD Quark](quark.md) | Various | 4 / 8 | Various | AMD GPU-optimized models |
| [Intel Neural Compressor](inc.md) | INT | 4 / 8 | Various | Intel CPU / GPU deployments |

---

## Hardware compatibility

The table below shows which quantization methods are supported on each hardware
platform. Use it to narrow down your options before choosing a method.

<style>
td:not(:first-child) { text-align: center !important; }
td { padding: 0.5rem !important; white-space: nowrap; }
th { padding: 0.5rem !important; min-width: 0 !important; }
th:not(:first-child) { writing-mode: vertical-lr; transform: rotate(180deg); }
</style>

| Method | Volta (SM 7.0) | Turing (SM 7.5) | Ampere (SM 8.x) | Ada (SM 8.9) | Hopper (SM 9.0) | Blackwell (SM 10.x) | AMD GPU | Intel GPU | x86 CPU |
|---|---|---|---|---|---|---|---|---|---|
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| AWQ Marlin | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPTQ Marlin | ❌ | ✅* | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| FP8 W8A8 | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| FP8 W8A16 (Marlin) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| INT8 W8A8 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅† | ✅ | ❌ | ❌ |
| BitsAndBytes | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| TorchAO | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| MXFP4 (MoE) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅‡ | ✅ | ❌ |

- ✅ Supported  ❌ Not supported
- \* Turing does not support Marlin MXFP4.
- † GGUF has precision issues with BF16 on Blackwell; FP16 or FP32 is recommended.
- ‡ AMD MXFP4 MoE support requires ROCm with gfx950 (MI300X) and the Aiter backend.

!!! note
    For Google TPU quantization support, see the
    [TPU inference documentation](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/).

---

## Choosing a method

Use the following decision tree to pick the right quantization method:

1. **Do you need the absolute highest throughput on Hopper or Ada GPUs?**
   → Use [FP8 W8A8](fp8.md). It delivers up to 1.6× throughput improvement
   with minimal accuracy loss.

1. **Do you need to run a very large model on limited GPU memory?**
   → Use [GPTQ](gptq.md) (4-bit) or [AWQ](awq.md) (4-bit). Both offer a
   4× memory reduction with good accuracy.

1. **Do you want zero-calibration quantization?**
   → Use [BitsAndBytes](bitsandbytes.md) for 4-bit or 8-bit quantization
   without any calibration data.

1. **Do you have a llama.cpp-compatible GGUF file?**
   → Use [GGUF](gguf.md) directly. vLLM can load GGUF files from Hugging Face
   or from a local path.

1. **Are you running a Mixture-of-Experts model on Hopper or Blackwell?**
   → Use [MXFP4](mxfp4.md) for the best MoE throughput on SM90/SM100.

1. **Do you want a PyTorch-native workflow with `torch.compile` support?**
   → Use [TorchAO](torchao.md).

1. **Are you targeting Intel CPUs or Intel GPUs?**
   → Use [Intel Neural Compressor (INC)](inc.md).

---

## Quantization method details

### Weight-only vs. weight-and-activation

Most quantization methods fall into one of two categories:

- **Weight-only (WNA16):** Only the model weights are quantized. Activations
  remain in FP16 or BF16. This reduces memory usage and can improve throughput
  in memory-bandwidth-bound workloads (low batch sizes). Examples: GPTQ, AWQ,
  BitsAndBytes, GGUF.

- **Weight-and-activation (W8A8, W4A8):** Both weights and activations are
  quantized. This enables integer or floating-point matrix multiplications in
  hardware, delivering the highest throughput gains at larger batch sizes.
  Examples: FP8 W8A8, INT8 W8A8, MXFP4.

### Static vs. dynamic activation quantization

When activations are quantized, the scale factors can be computed in two ways:

- **Static:** Scale factors are computed once during calibration and stored in
  the checkpoint. This is faster at inference time but requires a calibration
  step.
- **Dynamic:** Scale factors are computed on the fly for each token or tensor
  during inference. This requires no calibration but adds a small overhead.

### Group size

Many weight-only methods support a **group size** parameter (e.g., 128). A
smaller group size means more scale factors are stored, which improves accuracy
at the cost of slightly higher memory usage. A group size of 128 is a common
default that balances accuracy and efficiency.

---

## Out-of-tree quantization plugins

vLLM supports registering custom quantization methods using the
`@register_quantization_config` decorator. See the
[quantization README](README.md#out-of-tree-quantization-plugins) for the full
plugin API.

---

## Related pages

- [LLM Compressor](llm_compressor.md) — recommended tool for producing
  quantized checkpoints
- [KV cache quantization](kv_cache_quantization.md) — quantize the attention
  KV cache to FP8 for longer context windows
- [Engine arguments](../../configuration/engine_args.md) — `--quantization`
  and related flags
