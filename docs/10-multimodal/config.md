# MultiModalConfig

`MultiModalConfig` controls all aspects of multimodal model behavior in vLLM. It is
defined in `vllm/config/multimodal.py` and can be configured via CLI flags, environment
variables, or the Python API.

## Configuration Reference

```python
@config
class MultiModalConfig:
    """Controls the behavior of multimodal models."""
```

### Core Settings

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `language_model_only` | `bool` | `False` | `--language-model-only` | Disable all multimodal inputs (set all limits to 0) |
| `limit_per_prompt` | `MMDummyOptions` | `{}` | `--limit-mm-per-prompt` | Max items per prompt per modality (default: 999) |
| `enable_mm_embeds` | `bool` | `False` | `--enable-mm-embeds` | Allow pre-computed embeddings as input |

### Cache Settings

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `mm_processor_cache_gb` | `float` | `4.0` | `--mm-processor-cache-gb` | Cache size in GiB (0 = disabled) |
| `mm_processor_cache_type` | `"lru"` or `"shm"` | `"lru"` | `--mm-processor-cache-type` | Cache backend type |
| `mm_shm_cache_max_object_size_mb` | `int` | `128` | `--mm-shm-cache-max-object-size-mb` | Max object size for SHM cache (MiB) |

### Processor Settings

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `mm_processor_kwargs` | `dict` or `None` | `None` | `--mm-processor-kwargs` | Override HF processor kwargs |
| `media_io_kwargs` | `dict` | `{}` | `--media-io-kwargs` | Per-modality media loading kwargs |
| `interleave_mm_strings` | `bool` | `False` | `--interleave-mm-strings` | Enable interleaved multimodal prompts with string format |

### Encoder Settings

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `mm_encoder_only` | `bool` | `False` | `--mm-encoder-only` | Skip language model (disaggregated encoder) |
| `mm_encoder_tp_mode` | `"weights"` or `"data"` | `"weights"` | `--mm-encoder-tp-mode` | Tensor parallelism mode for encoder |
| `mm_encoder_attn_backend` | `AttentionBackendEnum` or `None` | `None` | `--mm-encoder-attn-backend` | Override attention backend for encoder |

### Video Settings

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `video_pruning_rate` | `float` or `None` | `None` | `--video-pruning-rate` | EVS pruning rate in [0, 1) |
| `skip_mm_profiling` | `bool` | `False` | `--skip-mm-profiling` | Skip multimodal memory profiling at startup |

## `limit_per_prompt` Format

The `limit_per_prompt` field accepts multiple formats:

### Legacy Format (count only)

```bash
vllm serve my-model --limit-mm-per-prompt '{"image": 4, "video": 2}'
```

```python
llm = LLM(model="my-model", limit_mm_per_prompt={"image": 4, "video": 2})
```

### Configurable Format (with options)

```bash
vllm serve my-model --limit-mm-per-prompt '{
    "image": {"count": 4, "width": 512, "height": 512},
    "video": {"count": 2, "num_frames": 32, "width": 512, "height": 512},
    "audio": {"count": 1, "length": 16000}
}'
```

### Mixed Format

```bash
vllm serve my-model --limit-mm-per-prompt '{
    "image": 4,
    "video": {"count": 1, "num_frames": 32}
}'
```

### Dummy Options Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `ImageDummyOptions` | `count`, `width`, `height` | Image profiling options |
| `VideoDummyOptions` | `count`, `num_frames`, `width`, `height` | Video profiling options |
| `AudioDummyOptions` | `count`, `length` | Audio profiling options |

The `count` field defaults to `999` if not specified. The dimension fields (`width`,
`height`, `num_frames`, `length`) control the size of dummy inputs used during memory
profiling.

## `mm_processor_kwargs`

Override HuggingFace processor arguments for all requests:

```bash
# Phi-3-Vision: set number of image crops
vllm serve microsoft/Phi-3-vision-128k-instruct \
    --mm-processor-kwargs '{"num_crops": 4}'
```

```python
llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    mm_processor_kwargs={"num_crops": 4},
)
```

## `media_io_kwargs`

Control media loading behavior per modality:

```bash
# Set number of video frames globally
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --media-io-kwargs '{"video": {"num_frames": 32}}'

# Set FPS for dynamic video backend
vllm serve my-model \
    --media-io-kwargs '{"video": {"fps": 2, "max_duration": 60}}'

# Set RGBA background color for images
vllm serve my-model \
    --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'
```

Per-request override (via API):

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[...],
    extra_body={"media_io_kwargs": {"video": {"num_frames": 16}}},
)
```

## `mm_encoder_tp_mode`

Controls how tensor parallelism is applied to the multimodal encoder:

| Mode | Description | Use Case |
|------|-------------|----------|
| `"weights"` | Split encoder weights across TP ranks (default) | Standard TP behavior |
| `"data"` | Split input data across TP ranks, full weights on each rank | High-throughput batch processing |

```bash
vllm serve my-model --mm-encoder-tp-mode data --tensor-parallel-size 4
```

> **Note**: `"data"` mode is only supported on a per-model basis and falls back to
> `"weights"` if the encoder does not support data parallelism.

## `enable_mm_embeds`

When enabled, allows passing pre-computed embeddings instead of raw media:

```bash
vllm serve my-model --enable-mm-embeds
```

In the OpenAI API, use `"type": "image_embeds"` content blocks:

```python
response = client.chat.completions.create(
    model="my-model",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this."},
            {
                "type": "image_embeds",
                "image_embeds": {"embeds": base64_encoded_tensor},
            },
        ],
    }],
)
```

> **Warning**: The engine may crash if embeddings with incorrect shapes are passed.
> Only enable this for trusted users.

## `language_model_only`

Disables all multimodal inputs by setting all modality limits to 0. Equivalent to
`--limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'`:

```bash
vllm serve llava-hf/llava-1.5-7b-hf --language-model-only
```

This is useful for running a vision-language model in text-only mode for benchmarking or
when multimodal inputs are not needed.

## `video_pruning_rate`

Enables EVS (Efficient Video Sampling) to prune redundant video tokens:

```bash
# Prune 30% of video tokens
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --video-pruning-rate 0.3
```

Valid range: `[0.0, 1.0)`. See [EVS Documentation](evs.md) for details.

## `mm_processor_cache_type`

| Type | Description | When to Use |
|------|-------------|-------------|
| `"lru"` | Mirrored LRU cache between P0 and P1 | Default, good for most cases |
| `"shm"` | Shared-memory FIFO ring buffer | Low-latency, single-writer scenarios |

The SHM cache uses a ring buffer and is more efficient for high-throughput scenarios
but requires careful sizing:

```bash
vllm serve my-model \
    --mm-processor-cache-type shm \
    --mm-processor-cache-gb 8 \
    --mm-shm-cache-max-object-size-mb 256
```

## `compute_hash` Method

`MultiModalConfig` provides a `compute_hash()` method that returns a hash of all fields
that affect the computation graph structure. This is used for cache invalidation:

```python
def compute_hash(self) -> str:
    factors = [
        self.mm_encoder_attn_backend.name if self.mm_encoder_attn_backend else None,
        self.mm_encoder_tp_mode,
    ]
    return safe_hash(str(factors).encode()).hexdigest()
```

## `get_limit_per_prompt` Method

Convenience method to get the limit for a specific modality:

```python
mm_config = model_config.get_multimodal_config()

# Returns 999 if not specified, 0 if language_model_only=True
limit = mm_config.get_limit_per_prompt("image")
```

## Complete Example

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    # Modality limits
    limit_mm_per_prompt={
        "image": {"count": 4, "width": 512, "height": 512},
        "video": {"count": 1, "num_frames": 32},
    },
    # Cache
    mm_processor_cache_gb=8.0,
    mm_processor_cache_type="lru",
    # Processor
    mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
    media_io_kwargs={"video": {"num_frames": 32}},
    # Video pruning
    video_pruning_rate=0.3,
    # Encoder
    mm_encoder_tp_mode="weights",
)
```

## Related Pages

- [Multimodal Overview](README.md) — overview and quick start
- [Multimodal Cache](cache.md) — cache architecture
- [Encoder Budget](encoder-budget.md) — encoder token budget
- [EVS — Efficient Video Sampling](evs.md) — video token pruning
