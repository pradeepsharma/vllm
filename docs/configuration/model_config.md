# Model Configuration

`ModelConfig` controls how vLLM loads, initializes, and runs the language model. It covers model identity, tokenizer settings, data types, quantization, context length, and multimodal configuration.

**Source:** `vllm/config/model.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — Model & Tokenizer section.

---

## Core Model Identity

### `model`

```
Type:    str
Default: "Qwen/Qwen3-0.6B"
CLI:     --model
```

The Hugging Face model ID or local path. This is the primary identifier for the model and is also used as the `model_name` tag in Prometheus metrics when `served_model_name` is not set.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
vllm serve /path/to/local/model
```

### `served_model_name`

```
Type:    str | list[str] | None
Default: None
CLI:     --served-model-name
```

The model name(s) exposed in the OpenAI-compatible API. If multiple names are provided, the server responds to any of them. The first name is used in the `model` field of responses and in Prometheus metrics.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama-3.1-8b my-model
```

### `revision`

```
Type:    str | None
Default: None
CLI:     --revision
```

Specific model version to use — a branch name, tag, or commit ID. If unspecified, uses the default (usually `main`).

### `hf_config_path`

```
Type:    str | None
Default: None
CLI:     --hf-config-path
```

Path to a custom HF config directory. Useful when the config is stored separately from the weights.

---

## Tokenizer

### `tokenizer`

```
Type:    str | None
Default: None (uses model path)
CLI:     --tokenizer
```

Name or path of the HF tokenizer. Defaults to the model path if not specified.

### `tokenizer_mode`

```
Type:    Literal["auto", "hf", "slow", "mistral", "deepseek_v32"]
Default: "auto"
CLI:     --tokenizer-mode
```

Controls which tokenizer implementation to use:

| Value | Behavior |
|---|---|
| `auto` | Uses `mistral_common` for Mistral models if available, otherwise HF fast tokenizer |
| `hf` | Always use the HF fast tokenizer |
| `slow` | Always use the HF slow tokenizer |
| `mistral` | Always use `mistral_common` tokenizer |
| `deepseek_v32` | Always use the DeepSeek V3.2 tokenizer |

Custom values can be registered via plugins.

### `tokenizer_revision`

```
Type:    str | None
Default: None
CLI:     --tokenizer-revision
```

Specific tokenizer version (branch, tag, or commit ID).

### `skip_tokenizer_init`

```
Type:    bool
Default: False
CLI:     --skip-tokenizer-init
```

Skip tokenizer and detokenizer initialization. When enabled, the engine expects `prompt_token_ids` (not text) as input, and outputs will contain token IDs only.

---

## Data Types

### `dtype`

```
Type:    Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
Default: "auto"
CLI:     --dtype
```

Data type for model weights and activations:

| Value | Description |
|---|---|
| `auto` | FP16 for FP32/FP16 models; BF16 for BF16 models |
| `half` / `float16` | FP16 — recommended for AWQ quantization |
| `bfloat16` | BF16 — better numerical range than FP16 |
| `float` / `float32` | FP32 — highest precision, highest memory usage |

### `quantization`

```
Type:    str | None
Default: None
CLI:     --quantization
```

Quantization method for model weights. If `None`, vLLM checks the model's `quantization_config` in its config file. Supported methods include:

- `awq` — Activation-aware Weight Quantization
- `gptq` — Generalized Post-Training Quantization
- `fp8` — FP8 weight quantization
- `bitsandbytes` — BitsAndBytes 4-bit/8-bit
- `squeezellm` — SqueezeLLM
- `marlin` — Marlin INT4/FP8
- `gguf` — GGUF format (via `--load-format gguf`)
- And many more — see [Quantization Config](quantization_config.md)

### `allow_deprecated_quantization`

```
Type:    bool
Default: False
CLI:     --allow-deprecated-quantization
```

Allow deprecated quantization methods that may be removed in future versions.

---

## Context Length

### `max_model_len`

```
Type:    int | None
Default: None (auto-detect from model config)
CLI:     --max-model-len
```

Maximum sequence length (prompt + generated tokens). Supports human-readable suffixes on the CLI:

| Input | Value |
|---|---|
| `8192` | 8,192 tokens |
| `8k` | 8,000 tokens |
| `8K` | 8,192 tokens |
| `128K` | 131,072 tokens |
| `-1` or `auto` | Auto-detect maximum that fits in GPU memory |

!!! warning
    Setting `max_model_len` larger than the model's trained context length may produce degraded outputs. Set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to bypass the safety check.

### `spec_target_max_model_len`

```
Type:    int | None
Default: None
```

Maximum context length for speculative decoding draft models. Only relevant when using speculative decoding.

---

## Model Loading

### `config_format`

```
Type:    Literal["auto", "hf", "mistral"]
Default: "auto"
CLI:     --config-format
```

Format of the model config file:

- `auto` — Try HF format first, then Mistral format
- `hf` — Load HuggingFace format config
- `mistral` — Load Mistral format config

### `hf_token`

```
Type:    bool | str | None
Default: None
CLI:     --hf-token
```

HuggingFace Hub authentication token. `True` uses the token from `~/.cache/huggingface/token` (set via `hf auth login`).

### `hf_overrides`

```
Type:    dict[str, Any] | Callable
Default: {}
CLI:     --hf-overrides
```

Override HuggingFace config fields. Can be a dict of key-value pairs or a callable that modifies the config object.

```bash
# Override rope_scaling
vllm serve mymodel --hf-overrides '{"rope_scaling": {"type": "linear", "factor": 2.0}}'
```

### `trust_remote_code`

```
Type:    bool
Default: False
CLI:     --trust-remote-code
```

Allow execution of remote code from the model repository. Required for some custom model architectures.

!!! warning
    Only enable this for models from trusted sources.

---

## Model Behavior

### `enforce_eager`

```
Type:    bool
Default: False
CLI:     --enforce-eager
```

Disable CUDA graph capture and always run in PyTorch eager mode. Useful for debugging or when CUDA graphs cause issues. Reduces throughput significantly.

### `enable_sleep_mode`

```
Type:    bool
Default: False
CLI:     --enable-sleep-mode
```

Enable sleep mode for the engine (CUDA and HIP platforms only). Allows the engine to release GPU memory when idle.

### `model_impl`

```
Type:    Literal["auto", "vllm", "transformers", "terratorch"]
Default: "auto"
CLI:     --model-impl
```

Which model implementation to use:

| Value | Description |
|---|---|
| `auto` | Use vLLM implementation if available, fall back to Transformers |
| `vllm` | Force vLLM implementation |
| `transformers` | Force HuggingFace Transformers implementation |
| `terratorch` | Use TerraTorch implementation (geospatial models) |

### `runner`

```
Type:    Literal["auto", "generate", "pooling", "draft"]
Default: "auto"
CLI:     --runner
```

Model runner type. Each vLLM instance supports only one runner type:

- `generate` — Text generation (default for causal LMs)
- `pooling` — Embedding/classification/reward models
- `draft` — Draft model for speculative decoding

### `convert`

```
Type:    Literal["auto", "none", "embed", "classify"]
Default: "auto"
CLI:     --convert
```

Convert a generation model to a pooling task using adapters.

---

## Logprobs

### `max_logprobs`

```
Type:    int
Default: 20
CLI:     --max-logprobs
```

Maximum number of log probabilities to return when `logprobs` is specified in `SamplingParams`. `-1` means no cap (may cause OOM for large vocabularies).

### `logprobs_mode`

```
Type:    Literal["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"]
Default: "raw_logprobs"
CLI:     --logprobs-mode
```

Controls what values are returned in logprobs:

| Value | Description |
|---|---|
| `raw_logprobs` | Log probabilities before any logit processors |
| `processed_logprobs` | Log probabilities after all processors (temperature, top-k/p) |
| `raw_logits` | Raw logit values before processors |
| `processed_logits` | Logit values after all processors |

---

## Generation Config

### `generation_config`

```
Type:    str
Default: "auto"
CLI:     --generation-config
```

Path to the generation config folder:

- `"auto"` — Load from model path
- `"vllm"` — Use vLLM defaults (no generation config loaded)
- `/path/to/folder` — Load from specified folder

If `max_new_tokens` is set in the generation config, it becomes a server-wide limit on output tokens.

### `override_generation_config`

```
Type:    dict[str, Any]
Default: {}
CLI:     --override-generation-config
```

Override specific generation config fields. Merged with the loaded config when `generation_config="auto"`.

```bash
vllm serve mymodel --override-generation-config '{"temperature": 0.7, "top_p": 0.9}'
```

---

## Sliding Window & Attention

### `disable_sliding_window`

```
Type:    bool
Default: False
CLI:     --disable-sliding-window
```

Disable sliding window attention, capping effective context to the sliding window size. Ignored for models that don't use sliding window.

### `disable_cascade_attn`

```
Type:    bool
Default: False
CLI:     --disable-cascade-attn
```

Disable cascade attention in V1. Cascade attention is mathematically equivalent but may cause numerical differences. Disabled automatically when the heuristic determines it's not beneficial.

### `override_attention_dtype`

```
Type:    str | None
Default: None
CLI:     --override-attention-dtype
```

Override the dtype used for attention computation (e.g., `"float32"` for higher precision).

---

## Multimodal

### `multimodal_config`

```
Type:    MultiModalConfig | None
Default: None (auto-detected from architecture)
```

Multimodal configuration. Typically auto-detected; use `MultiModalConfig` directly for fine-grained control.

Key multimodal CLI flags (passed via `EngineArgs`):

| Flag | Description |
|---|---|
| `--language-model-only` | Disable multimodal processing |
| `--limit-mm-per-prompt` | Max items per modality per prompt |
| `--mm-processor-kwargs` | Extra kwargs for the MM processor |
| `--mm-processor-cache-gb` | MM processor cache size in GiB |
| `--skip-mm-profiling` | Skip MM profiling during warmup |
| `--video-pruning-rate` | Frame pruning rate for video inputs |

---

## Pooling

### `pooler_config`

```
Type:    PoolerConfig | None
Default: None
CLI:     --pooler-config
```

Configuration for output pooling in embedding/classification models. Passed as JSON:

```bash
vllm serve BAAI/bge-m3 \
  --runner pooling \
  --pooler-config '{"pooling_type": "MEAN", "normalize": true}'
```

---

## Custom Logits Processors

### `logits_processors`

```
Type:    list[str | type] | None
Default: None
CLI:     --logits-processors
```

One or more fully-qualified class names of custom logits processors to apply globally to all requests.

```bash
vllm serve mymodel \
  --logits-processors mypackage.processors.RepetitionPenaltyProcessor
```

---

## Practical Examples

### Load a model with BF16 and custom context length

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --tensor-parallel-size 4
```

### Load a GGUF model

```bash
vllm serve /path/to/model.gguf \
  --load-format gguf \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

### Embedding model

```bash
vllm serve BAAI/bge-m3 \
  --runner pooling \
  --dtype float16
```

### Model with HF config overrides

```bash
vllm serve mymodel \
  --hf-overrides '{"max_position_embeddings": 65536}' \
  --max-model-len 65536 \
  --trust-remote-code
```
