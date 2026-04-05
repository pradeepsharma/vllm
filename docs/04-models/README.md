# Models

vLLM supports 150+ model architectures across text generation, multimodal, embedding, and transcription tasks. This section covers the complete model ecosystem: what's supported, how the registry works, model interfaces, and how to add new architectures.

## In This Section

| Page | Description |
|---|---|
| [Supported Models](supported-models.md) | Complete list of all 150+ supported architectures organized by type and family |
| [Model Registry Internals](registry-internals.md) | How `_ModelRegistry`, `_RegisteredModel`, and `_LazyRegisteredModel` work |
| [Model Interfaces](model-interfaces.md) | `SupportsLoRA`, `SupportsPP`, `SupportsMultiModal`, `SupportsQuant`, `SupportsTranscription` |
| [Adding a New Model](adding-new-model.md) | Step-by-step guide to implementing and registering a new architecture |
| [Multimodal Models](multimodal-models.md) | Vision encoders, audio encoders, cross-modal attention, and multimodal APIs |
| [Model Loading](model-loading.md) | HuggingFace Hub, local paths, safetensors, GGUF, dummy weights, and `hf-overrides` |

## Quick Reference

### Supported Model Categories

- **Text Generation** (decoder-only): Llama, Mistral, Qwen, DeepSeek, Gemma, GPT, Falcon, Phi, OLMo, Granite, and 100+ more
- **Multimodal**: LLaVA, Qwen2-VL, Gemma3, InternVL, Phi4-MM, Whisper, and 50+ more
- **Embedding/Pooling**: BERT, RoBERTa, ModernBERT, ColBERT, CLIP, SigLIP, and more
- **Speculative Decoding**: EAGLE, Medusa, MTP draft models

### Loading a Model

```bash
# From HuggingFace Hub
vllm serve meta-llama/Llama-3.1-8B-Instruct

# From local path
vllm serve /path/to/model

# GGUF format
vllm serve bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M

# With config override
vllm serve my-model --hf-overrides '{"architectures": ["LlamaForCausalLM"]}'
```

### Registering a Custom Model

```python
from vllm import ModelRegistry

# Lazy registration (preferred)
ModelRegistry.register_model(
    "MyCustomForCausalLM",
    "my_package.models:MyCustomModel"
)
```
