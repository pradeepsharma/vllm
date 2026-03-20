# Models Overview

vLLM supports a broad and growing ecosystem of model architectures — from classic decoder-only transformers to cutting-edge multimodal, mixture-of-experts, and state-space models. This section documents everything you need to know about working with models in vLLM.

---

## Model Categories

vLLM organizes supported models into five primary categories:

| Category | Description | Guide |
|---|---|---|
| **Generative (Text Generation)** | Decoder-only causal language models for text completion and chat | [Generative Models](generative_models.md) |
| **Embedding & Pooling** | Models that produce dense vector representations for retrieval and ranking | [Embedding Models](embedding_models.md) |
| **Multimodal** | Vision-language, audio-language, and omni models that process images, video, and audio | [Multimodal Models](multimodal_models.md) |
| **Encoder-Decoder** | Sequence-to-sequence architectures (e.g., Whisper for ASR) | [Encoder-Decoder Models](encoder_decoder_models.md) |
| **Speculative Decoding** | Draft models and MTP heads used to accelerate inference | [Supported Models](supported_models.md) |

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-format-list-bulleted: **Supported Models**

    ---

    Complete table of 150+ supported architectures with task types, modalities, and quantization support.

    [:octicons-arrow-right-24: View full list](supported_models.md)

-   :material-plus-box: **Adding a New Model**

    ---

    Step-by-step guide to implementing and registering a new model architecture in vLLM.

    [:octicons-arrow-right-24: Adding a model](adding_model.md)

-   :material-database: **ModelRegistry API**

    ---

    Reference documentation for the `ModelRegistry` — how models are registered, resolved, and loaded.

    [:octicons-arrow-right-24: ModelRegistry reference](model_registry.md)

-   :material-image-multiple: **Multimodal Models**

    ---

    Working with vision, audio, and video inputs — processors, placeholders, and multi-image support.

    [:octicons-arrow-right-24: Multimodal guide](multimodal_models.md)

-   :material-vector-combine: **Embedding Models**

    ---

    Dense and sparse embeddings, cross-encoders, late-interaction models, and pooling strategies.

    [:octicons-arrow-right-24: Embedding guide](embedding_models.md)

    [:octicons-arrow-right-24: Pooling models (technical)](pooling_models.md)

-   :material-swap-horizontal: **Encoder-Decoder Models**

    ---

    Sequence-to-sequence architectures including Whisper ASR and conditional generation models.

    [:octicons-arrow-right-24: Encoder-decoder guide](encoder_decoder_models.md)

</div>

---

## How vLLM Loads Models

vLLM uses a **lazy model registry** that maps HuggingFace `architectures` strings to internal implementation classes. When you pass a model path, vLLM:

1. Reads `config.json` from the model directory to extract the `architectures` field.
2. Looks up the architecture string in the `ModelRegistry`.
3. Lazily imports the corresponding Python module and class.
4. Instantiates the model and loads weights using the configured **load format**.

```python
from vllm import LLM

# vLLM automatically resolves the architecture from config.json
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

See [ModelRegistry API](model_registry.md) for details on how the registry works and how to extend it.

---

## Quantization Support

vLLM supports a wide range of quantization formats that work across most model architectures:

| Format | Description |
|---|---|
| `awq` | Activation-aware Weight Quantization (INT4) |
| `gptq` | GPTQ post-training quantization |
| `bitsandbytes` | BitsAndBytes INT4/INT8 quantization |
| `fp8` | FP8 weight and activation quantization |
| `gguf` | GGUF format (llama.cpp compatible) |
| `squeezellm` | SqueezeLLM sparse quantization |
| `marlin` | Marlin INT4×FP16 kernel |
| `aqlm` | Additive Quantization of Language Models |

Quantization is configured via the `quantization` parameter in `LLM()` or `--quantization` in the CLI. See the [configuration guide](../configuration/optimization.md) for details.

---

## Model Loading Formats

vLLM supports multiple weight loading strategies:

| Format | Loader | Use Case |
|---|---|---|
| `auto` / `safetensors` | `DefaultModelLoader` | Standard HuggingFace models |
| `hf` | `DefaultModelLoader` | Explicit HuggingFace format |
| `pt` | `DefaultModelLoader` | PyTorch `.bin` checkpoints |
| `gguf` | `GGUFModelLoader` | GGUF quantized models |
| `bitsandbytes` | `BitsAndBytesModelLoader` | BnB quantized loading |
| `tensorizer` | `TensorizerLoader` | Fast serialized loading |
| `sharded_state` | `ShardedStateLoader` | Pre-sharded checkpoints |
| `runai_streamer` | `RunaiModelStreamerLoader` | Run:ai streamed loading |
| `dummy` | `DummyModelLoader` | Testing without real weights |

---

## Transformers Backend

For models not natively implemented in vLLM, you can use the **Transformers backend** by setting `--model-impl transformers`. This allows any HuggingFace-compatible model to run through vLLM's serving infrastructure, though with potentially lower performance than a native vLLM implementation.

```bash
vllm serve my-custom-model --model-impl transformers
```

---

## Related Resources

- [Engine Arguments](../configuration/engine_args.md) — full list of model-related CLI flags
- [Model Resolution](../configuration/model_resolution.md) — how vLLM resolves model configs
- [Multimodal Processing Design](../design/mm_processing.md) — internals of multimodal input handling
- [Plugin System](../design/plugin_system.md) — registering custom models via plugins
