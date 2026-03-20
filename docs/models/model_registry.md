# ModelRegistry API

The `ModelRegistry` is vLLM's central catalog of supported model architectures. It maps HuggingFace `architectures` strings (from `config.json`) to the Python classes that implement them, and provides methods to inspect model capabilities without importing the model.

---

## Overview

```python
from vllm.model_executor.models import ModelRegistry
```

The `ModelRegistry` is a singleton instance of `_ModelRegistry` that is pre-populated with all built-in vLLM model implementations. It supports:

- **Lazy loading** — model modules are only imported when needed, avoiding CUDA initialization in the main process
- **Capability inspection** — query whether a model supports multimodal inputs, pipeline parallelism, LoRA, etc.
- **External registration** — register custom model classes at runtime
- **Transformers backend fallback** — automatically route unsupported architectures to the HuggingFace Transformers backend

---

## Architecture Resolution

When vLLM loads a model, it resolves the architecture string from `config.json` through the following priority order:

```
1. Explicit --model-impl transformers  →  Transformers backend
2. Explicit --model-impl terratorch    →  Terratorch backend
3. Architecture in ModelRegistry       →  Native vLLM implementation
4. Architecture in Transformers        →  Transformers backend (auto-fallback)
5. Architecture not found              →  ValueError
```

The resolution logic also handles:
- **Architecture normalization** — matching `runner_type` and `convert_type` suffixes
- **Alias resolution** — multiple architecture strings can map to the same implementation (e.g., `AquilaModel` and `AquilaForCausalLM` both map to `LlamaForCausalLM`)

---

## Core Methods

### `register_model`

Register a custom model architecture at runtime.

```python
ModelRegistry.register_model(
    model_arch: str,
    model_cls: type[nn.Module] | str,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model_arch` | `str` | The architecture string from `config.json` (e.g., `"MyModelForCausalLM"`) |
| `model_cls` | `type[nn.Module]` or `str` | The model class, or a lazy import string in `"module:ClassName"` format |

**Examples:**

```python
from vllm.model_executor.models import ModelRegistry

# Register with a direct class reference
from my_package.models import MyModelForCausalLM
ModelRegistry.register_model("MyModelForCausalLM", MyModelForCausalLM)

# Register with a lazy import string (avoids CUDA initialization)
ModelRegistry.register_model(
    "MyModelForCausalLM",
    "my_package.models:MyModelForCausalLM",
)
```

!!! warning "Overwriting existing registrations"
    If you register an architecture that is already registered, a warning is logged and the new class overwrites the existing one. This can be used to override built-in implementations.

!!! tip "Lazy import format"
    Use the `"module:ClassName"` string format when registering models in plugin code. This avoids importing the model module (and initializing CUDA) until the model is actually needed.

---

### `get_supported_archs`

Return the set of all registered architecture strings.

```python
ModelRegistry.get_supported_archs() -> Set[str]
```

**Example:**

```python
archs = ModelRegistry.get_supported_archs()
print("LlamaForCausalLM" in archs)  # True
print(len(archs))  # 300+
```

---

### `inspect_model_cls`

Inspect a model's capabilities without fully loading it. Returns a `_ModelInfo` dataclass and the resolved architecture string.

```python
ModelRegistry.inspect_model_cls(
    architectures: str | list[str],
    model_config: ModelConfig,
) -> tuple[_ModelInfo, str]
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `architectures` | `str` or `list[str]` | Architecture string(s) from `config.json` |
| `model_config` | `ModelConfig` | The vLLM model configuration |

**Returns:** A tuple of `(_ModelInfo, resolved_arch_string)`.

The `_ModelInfo` dataclass contains:

| Field | Type | Description |
|---|---|---|
| `architecture` | `str` | The resolved class name |
| `is_text_generation_model` | `bool` | Whether the model generates text |
| `is_pooling_model` | `bool` | Whether the model produces embeddings |
| `attn_type` | `AttnTypeStr` | Attention mechanism type |
| `default_seq_pooling_type` | `SequencePoolingType` | Default sequence-level pooling |
| `default_tok_pooling_type` | `TokenPoolingType` | Default token-level pooling |
| `supports_cross_encoding` | `bool` | Whether the model supports cross-encoding |
| `supports_late_interaction` | `bool` | Whether the model supports late interaction (ColBERT-style) |
| `supports_multimodal` | `bool` | Whether the model accepts multimodal inputs |
| `supports_multimodal_raw_input_only` | `bool` | Whether the model requires raw (non-embedded) multimodal inputs |
| `requires_raw_input_tokens` | `bool` | Whether the model processes raw input tokens |
| `supports_multimodal_encoder_tp_data` | `bool` | Whether multimodal encoder supports tensor-parallel data mode |
| `supports_pp` | `bool` | Whether the model supports pipeline parallelism |
| `has_inner_state` | `bool` | Whether the model has recurrent inner state (e.g., Mamba) |
| `is_attention_free` | `bool` | Whether the model has no attention layers |
| `is_hybrid` | `bool` | Whether the model mixes attention and SSM layers |
| `has_noops` | `bool` | Whether the model has no-op layers |
| `supports_mamba_prefix_caching` | `bool` | Whether Mamba prefix caching is supported |
| `supports_transcription` | `bool` | Whether the model supports speech transcription |
| `supports_transcription_only` | `bool` | Whether the model is transcription-only (no text generation) |

---

### `resolve_model_cls`

Resolve and load the model class for a given architecture. Unlike `inspect_model_cls`, this actually imports the module.

```python
ModelRegistry.resolve_model_cls(
    architectures: str | list[str],
    model_config: ModelConfig,
) -> tuple[type[nn.Module], str]
```

**Returns:** A tuple of `(model_class, resolved_arch_string)`.

---

### Capability Query Methods

These convenience methods wrap `inspect_model_cls` to check specific capabilities:

```python
# Check if a model generates text
ModelRegistry.is_text_generation_model(architectures, model_config) -> bool

# Check if a model produces embeddings/pooled outputs
ModelRegistry.is_pooling_model(architectures, model_config) -> bool

# Check if a model is a cross-encoder (reranker)
ModelRegistry.is_cross_encoder_model(architectures, model_config) -> bool

# Check if a model accepts multimodal inputs
ModelRegistry.is_multimodal_model(architectures, model_config) -> bool

# Check if a model supports pipeline parallelism
ModelRegistry.is_pp_supported_model(architectures, model_config) -> bool

# Check if a model has recurrent inner state (e.g., Mamba)
ModelRegistry.model_has_inner_state(architectures, model_config) -> bool

# Check if a model has no attention layers (e.g., pure SSM)
ModelRegistry.is_attention_free_model(architectures, model_config) -> bool

# Check if a model mixes attention and SSM layers
ModelRegistry.is_hybrid_model(architectures, model_config) -> bool

# Check if a model supports speech transcription
ModelRegistry.is_transcription_model(architectures, model_config) -> bool

# Check if a model is transcription-only (no text generation)
ModelRegistry.is_transcription_only_model(architectures, model_config) -> bool
```

---

## Lazy Loading and Caching

vLLM uses a **lazy loading** strategy to avoid importing model modules in the main process. This is important because importing PyTorch model code can trigger CUDA initialization, which causes errors in forked subprocesses.

### How Lazy Loading Works

1. The registry stores `_LazyRegisteredModel` entries containing only `(module_name, class_name)` strings.
2. When `inspect_model_cls` is called, vLLM runs the inspection in a **subprocess** to avoid CUDA initialization.
3. The result is cached to disk in `$VLLM_CACHE_ROOT/modelinfos/` as a JSON file.
4. On subsequent calls, the cached result is loaded if the model file hasn't changed (verified by hash).

### Cache Location

Model info is cached at:
```
$VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json
```

The default `VLLM_CACHE_ROOT` is `~/.cache/vllm`. You can override it with the `VLLM_CACHE_ROOT` environment variable.

### Cache Invalidation

The cache is automatically invalidated when the model's Python source file changes (detected by SHA hash). You can also manually clear the cache:

```bash
rm -rf ~/.cache/vllm/modelinfos/
```

---

## Registering Models via Plugins

The recommended way to register custom models for production use is through vLLM's plugin system. This ensures your model is registered before any model loading occurs.

### Creating a Plugin

```python
# my_vllm_plugin/__init__.py

def register():
    """Called by vLLM's plugin loader at startup."""
    from vllm.model_executor.models import ModelRegistry

    # Use lazy import to avoid CUDA initialization
    ModelRegistry.register_model(
        "MyCustomModelForCausalLM",
        "my_vllm_plugin.models:MyCustomModelForCausalLM",
    )
```

### Registering the Plugin

Add your plugin to `pyproject.toml` (or `setup.py`):

```toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_vllm_plugin:register"
```

After installing your package, vLLM will automatically call `register()` at startup.

### Inline Registration

For quick testing or one-off use, you can register models inline before creating an `LLM` instance:

```python
from vllm.model_executor.models import ModelRegistry
from my_package import MyCustomModel

# Register before creating LLM
ModelRegistry.register_model("MyCustomModelForCausalLM", MyCustomModel)

from vllm import LLM
llm = LLM(model="path/to/my-custom-model")
```

---

## Internal Architecture Dictionaries

The registry is built from five internal dictionaries that are merged into `_VLLM_MODELS`:

| Dictionary | Contents |
|---|---|
| `_TEXT_GENERATION_MODELS` | Decoder-only causal language models |
| `_EMBEDDING_MODELS` | Embedding and pooling models (text and multimodal) |
| `_CROSS_ENCODER_MODELS` | Sequence classification / reranking models |
| `_MULTIMODAL_MODELS` | Vision-language, audio-language, and omni models |
| `_SPECULATIVE_DECODING_MODELS` | Draft models and MTP heads |
| `_TRANSFORMERS_SUPPORTED_MODELS` | Models routed to Transformers backend by default |
| `_TRANSFORMERS_BACKEND_MODELS` | Internal Transformers backend wrapper classes |

Each entry has the format:
```python
"ArchitectureString": ("module_filename", "ClassName"),
```

---

## Previously Supported Models

The registry also maintains a `_PREVIOUSLY_SUPPORTED_MODELS` dictionary that maps removed architecture strings to the last vLLM version that supported them. When a user tries to load a removed model, vLLM raises a helpful error message:

```
ValueError: Model architecture BartForConditionalGeneration was supported in
vLLM until v0.10.2, and is not supported anymore. Please use an older version
of vLLM if you want to use this model architecture.
```

---

## Example: Checking Model Capabilities

```python
from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry

# Create a minimal ModelConfig for inspection
model_config = ModelConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="auto",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    tokenizer_mode="auto",
    trust_remote_code=False,
    dtype="auto",
    seed=0,
)

# Inspect capabilities
model_info, arch = ModelRegistry.inspect_model_cls(
    ["LlamaForCausalLM"],
    model_config,
)

print(f"Architecture: {arch}")
print(f"Text generation: {model_info.is_text_generation_model}")
print(f"Pooling: {model_info.is_pooling_model}")
print(f"Multimodal: {model_info.supports_multimodal}")
print(f"Pipeline parallel: {model_info.supports_pp}")
print(f"Has inner state: {model_info.has_inner_state}")
```

---

## Example: Listing All Supported Architectures

```python
from vllm.model_executor.models import ModelRegistry

archs = sorted(ModelRegistry.get_supported_archs())
for arch in archs:
    print(arch)
```

---

## Related

- [Adding a New Model](adding_model.md) — how to implement and register a new architecture
- [Supported Models](supported_models.md) — complete list of registered architectures
- [Plugin System](../design/plugin_system.md) — registering models via vLLM plugins
