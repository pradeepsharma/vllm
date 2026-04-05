# ModelConfig

`ModelConfig` controls everything related to the model itself — which model to load, its data type, context length, quantization method, tokenizer settings, and multimodal capabilities. It is defined in `vllm/config/model.py`.

## Overview

```python
from vllm.config import ModelConfig

model_config = ModelConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    trust_remote_code=False,
)
```

## Core Fields

### Model Identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"Qwen/Qwen3-0.6B"` | HuggingFace model name or local path. Also used as the `model_name` tag in Prometheus metrics when `served_model_name` is not set. |
| `model_weights` | `str` | `""` | Original model weights path when the model is pulled from object storage (e.g., RunAI). |
| `revision` | `str \| None` | `None` | Specific model version — branch name, tag, or commit ID. |
| `code_revision` | `str \| None` | `None` | Specific revision for model code on HuggingFace Hub. |
| `hf_config_path` | `str \| None` | `None` | Path to HuggingFace config. Defaults to model path. |
| `hf_token` | `bool \| str \| None` | `None` | HuggingFace authentication token. Set `True` to use the cached token from `hf auth login`. |
| `trust_remote_code` | `bool` | `False` | Allow execution of remote code when loading the model and tokenizer. **Enable only for trusted sources.** |
| `served_model_name` | `str \| list[str] \| None` | `None` | Model name(s) exposed via the API. If multiple names are provided, the server responds to any of them. |

### Data Type

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | `ModelDType \| torch.dtype` | `"auto"` | Data type for model weights and activations. |

Valid `dtype` values:

| Value | Description |
|-------|-------------|
| `"auto"` | FP16 for FP32/FP16 models; BF16 for BF16 models |
| `"half"` / `"float16"` | FP16 precision. Recommended for AWQ quantization. |
| `"bfloat16"` | BF16 — good balance of precision and range |
| `"float"` / `"float32"` | Full FP32 precision |

### Context Length

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_model_len` | `int` | `None` | Maximum sequence length (prompt + output tokens). Auto-derived from model config if unset. Supports human-readable suffixes: `1k` → 1000, `1K` → 1024, `25.6k` → 25600. Set to `-1` or `"auto"` to automatically find the largest length that fits in GPU memory. |
| `spec_target_max_model_len` | `int \| None` | `None` | Maximum length for speculative decoding draft models. |

### Quantization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quantization` | `str \| None` | `None` | Quantization method. If `None`, checks `quantization_config` in the model's `config.json`. Supported methods include `awq`, `gptq`, `fp8`, `bitsandbytes`, `gguf`, and more. |
| `allow_deprecated_quantization` | `bool` | `False` | Allow deprecated quantization methods. |

### Tokenizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tokenizer` | `str` | `None` | HuggingFace tokenizer name or path. Defaults to `model`. |
| `tokenizer_mode` | `TokenizerMode \| str` | `"auto"` | Tokenizer mode selection. |
| `tokenizer_revision` | `str \| None` | `None` | Specific tokenizer version. |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer initialization. Requires `prompt_token_ids` input; output will contain token IDs only. |

Valid `tokenizer_mode` values:

| Value | Description |
|-------|-------------|
| `"auto"` | Use `mistral_common` for Mistral models if available, otherwise HF fast tokenizer |
| `"hf"` | HuggingFace fast tokenizer |
| `"slow"` | HuggingFace slow tokenizer |
| `"mistral"` | Always use `mistral_common` tokenizer |
| `"deepseek_v32"` | DeepSeek V3.2 tokenizer |

### Model Runner

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runner` | `RunnerOption` | `"auto"` | Model runner type: `"auto"`, `"generate"`, `"pooling"`, or `"draft"`. |
| `convert` | `ConvertOption` | `"auto"` | Convert the model using adapters (e.g., adapt a generation model for pooling tasks). |
| `model_impl` | `ModelImpl` | `"auto"` | Which model implementation to use: `"auto"`, `"vllm"`, `"transformers"`, or `"terratorch"`. |

### Inference Behavior

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `0` | Random seed for reproducibility across tensor parallel workers. |
| `enforce_eager` | `bool` | `False` | Disable CUDA graph capture and always run in eager mode. Useful for debugging. |
| `max_logprobs` | `int` | `20` | Maximum number of log probabilities returned when `logprobs` is specified. Set to `-1` for no cap (may cause OOM). |
| `logprobs_mode` | `LogprobsMode` | `"raw_logprobs"` | Content of returned logprobs: `"raw_logprobs"`, `"processed_logprobs"`, `"raw_logits"`, or `"processed_logits"`. |
| `disable_sliding_window` | `bool` | `False` | Disable sliding window attention, capping to the sliding window size. |
| `disable_cascade_attn` | `bool` | `False` | Disable cascade attention in V1 engine. |
| `enable_sleep_mode` | `bool` | `False` | Enable sleep mode (CUDA/HIP platforms only). |
| `enable_return_routed_experts` | `bool` | `False` | Return routed expert indices in MoE models. |

### Generation Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generation_config` | `str` | `"auto"` | Path to generation config folder. `"auto"` loads from model path; `"vllm"` uses vLLM defaults. |
| `override_generation_config` | `dict` | `{}` | Override specific generation config parameters, e.g., `{"temperature": 0.5}`. |
| `config_format` | `str \| ConfigFormat` | `"auto"` | Model config format: `"auto"`, `"hf"`, or `"mistral"`. |
| `hf_overrides` | `HfOverrides` | `{}` | Dict of arguments forwarded to the HuggingFace config, or a callable that modifies it. |

### Multimodal

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `multimodal_config` | `MultiModalConfig \| None` | `None` | Multimodal configuration. Auto-inferred from model architecture if `None`. |
| `allowed_local_media_path` | `str` | `""` | Allow API requests to read local images/videos from this directory. **Security risk — use only in trusted environments.** |
| `allowed_media_domains` | `list[str] \| None` | `None` | Restrict media URLs to these domains only. |
| `enable_prompt_embeds` | `bool` | `False` | Enable passing text embeddings as inputs via `prompt_embeds`. **Only enable for trusted users.** |

### Logits Processing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logits_processors` | `list[str \| type] \| None` | `None` | Fully-qualified class names or class definitions for logits processors. |
| `io_processor_plugin` | `str \| None` | `None` | IOProcessor plugin name to load at model startup. |

### Pooling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pooler_config` | `PoolerConfig \| None` | `None` | Controls output pooling behavior for pooling models. |

## Configuration Examples

### Loading a Local Model

```python
ModelConfig(
    model="/path/to/local/model",
    dtype="bfloat16",
    max_model_len=32768,
)
```

### Quantized Model with Custom Tokenizer

```python
ModelConfig(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
    tokenizer="meta-llama/Llama-2-70b-hf",
    max_model_len=4096,
)
```

### Embedding / Pooling Model

```python
ModelConfig(
    model="BAAI/bge-large-en-v1.5",
    runner="pooling",
    dtype="float16",
)
```

### Model with Remote Code

```python
ModelConfig(
    model="microsoft/phi-4",
    trust_remote_code=True,
    dtype="bfloat16",
)
```

### Multimodal Model

```python
ModelConfig(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    dtype="bfloat16",
    max_model_len=32768,
    # limit_mm_per_prompt is passed as InitVar
)
```

## Derived Properties

`ModelConfig` exposes several computed properties after initialization:

- `hf_config` — The loaded HuggingFace `PretrainedConfig` object
- `hf_text_config` — The text model config (same as `hf_config` for text-only models)
- `is_encoder_decoder` — Whether the model uses encoder-decoder architecture
- `get_hidden_size()` — Model hidden dimension size
- `is_quantized()` — Whether the model uses quantization
- `is_model_moe()` — Whether the model is a Mixture-of-Experts model
- `is_nvfp4_quantized()` — Whether the model uses NVFP4 quantization

## Hash Computation

`ModelConfig.compute_hash()` produces a hash of all fields that affect the computation graph. Fields excluded from the hash include tokenizer settings, seed, `served_model_name`, `hf_overrides`, and multimodal processor settings — since these don't change the model's computation structure.

## Related Pages

- [VllmConfig](vllm-config.md) — the parent container
- [CacheConfig](cache-config.md) — KV cache memory management
- [SpeculativeConfig](speculative-config.md) — draft model configuration
- [Additional Configs](additional-configs.md) — LoRA and multimodal settings
