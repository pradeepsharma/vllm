# Additional Configuration Classes

This page documents the supplementary configuration classes in vLLM: `LoRAConfig`, `MultiModalConfig`, `ObservabilityConfig`, `OffloadConfig`, and `KVTransferConfig`. Each is an optional component of `VllmConfig`.

## LoRAConfig

`LoRAConfig` controls Low-Rank Adaptation (LoRA) support, enabling efficient fine-tuned adapter loading without duplicating the base model. Defined in `vllm/config/lora.py`.

### When to Use

Enable LoRA when you need to serve multiple fine-tuned variants of the same base model simultaneously, or when you want to dynamically load/unload adapters at runtime.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_lora_rank` | `MaxLoRARanks` | `16` | Maximum LoRA rank. Valid values: `1, 8, 16, 32, 64, 128, 256, 320, 512`. |
| `max_loras` | `int` | `1` | Maximum number of LoRA adapters active in a single batch. |
| `max_cpu_loras` | `int \| None` | `None` | Maximum LoRA adapters stored in CPU memory. Must be ≥ `max_loras`. Defaults to `max_loras`. |
| `fully_sharded_loras` | `bool` | `False` | Use fully sharded LoRA layers. At high sequence lengths, max rank, or large TP size, this is likely faster. By default, only half of the LoRA computation is sharded. |
| `lora_dtype` | `torch.dtype \| LoRADType` | `"auto"` | Data type for LoRA weights. `"auto"` uses the base model dtype. Options: `"auto"`, `"float16"`, `"bfloat16"`. |
| `default_mm_loras` | `dict[str, str] \| None` | `None` | Maps modalities to LoRA model paths for multimodal models. Applied automatically when the specified modality is present. |
| `enable_tower_connector_lora` | `bool` | `False` | Enable LoRA for the vision encoder tower and connector in multimodal models. Experimental; currently supports some Qwen VL series models. |
| `specialize_active_lora` | `bool` | `False` | Construct LoRA kernel grids by the number of active adapters. When `True`, separate CUDA graphs are captured for different active LoRA counts (powers of 2 up to `max_loras`). Improves performance for variable LoRA usage at the cost of startup time and memory. |

### Configuration Examples

```python
from vllm.config import LoRAConfig

# Basic LoRA — serve up to 4 adapters simultaneously
lora_config = LoRAConfig(
    max_lora_rank=64,
    max_loras=4,
    max_cpu_loras=16,  # Cache 16 adapters in CPU memory
)
```

```python
# High-rank LoRA with full sharding for large TP
lora_config = LoRAConfig(
    max_lora_rank=256,
    max_loras=2,
    fully_sharded_loras=True,
    lora_dtype="bfloat16",
)
```

### Using LoRA with the LLM API

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,
)

outputs = llm.generate(
    "Hello, my name is",
    lora_request=LoRARequest("my-adapter", 1, "/path/to/lora"),
)
```

### Hash Computation

`LoRAConfig.compute_hash()` includes `max_lora_rank`, `max_loras`, `fully_sharded_loras`, `lora_dtype`, and `enable_tower_connector_lora` — all of which affect the computation graph structure.

---

## MultiModalConfig

`MultiModalConfig` controls the behavior of multimodal models — how many images/videos/audio clips are allowed per prompt, caching of multimodal processor outputs, and tensor parallelism for the encoder. Defined in `vllm/config/multimodal.py`.

> **Note**: `MultiModalConfig` is typically auto-inferred from the model architecture. You usually configure it via `ModelConfig` init vars (`limit_mm_per_prompt`, `mm_processor_kwargs`, etc.) rather than constructing it directly.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `language_model_only` | `bool` | `False` | Disable all multimodal inputs by setting all modality limits to 0. Equivalent to `--limit-mm-per-prompt 0` for every modality. |
| `limit_per_prompt` | `MMDummyOptions` | `{}` | Maximum number of inputs per modality per prompt. Defaults to 999 for each modality. |
| `enable_mm_embeds` | `bool` | `False` | Enable passing precomputed multimodal embeddings as inputs. **Only enable for trusted users** — incorrect embedding shapes can crash the engine. |
| `media_io_kwargs` | `dict[str, dict]` | `{}` | Additional args for media processing, keyed by modality. E.g., `{"video": {"num_frames": 40}}`. |
| `mm_processor_kwargs` | `dict \| None` | `None` | Arguments forwarded to the model's multimodal processor (e.g., `{"num_crops": 4}` for Phi-3-Vision). |
| `mm_processor_cache_gb` | `float` | `4` | Size (GiB) of the multimodal processor cache. Avoids re-processing past inputs. Set to `0` to disable. |
| `mm_processor_cache_type` | `MMCacheType` | `"lru"` | Cache type: `"lru"` (mirrored LRU) or `"shm"` (shared memory FIFO). |
| `mm_shm_cache_max_object_size_mb` | `int` | `128` | Max object size (MiB) in the shared memory cache. Only effective when `mm_processor_cache_type="shm"`. |
| `mm_encoder_only` | `bool` | `False` | Skip the language component. Used in disaggregated encoder processes. |
| `mm_encoder_tp_mode` | `MMEncoderTPMode` | `"weights"` | How to apply TP to the multimodal encoder: `"weights"` (split weights across TP ranks) or `"data"` (split input data across TP ranks). |

### Limit Per Prompt Format

```python
# Legacy format (count only)
limit_per_prompt = {"image": 16, "video": 2}

# Configurable format (with options)
limit_per_prompt = {
    "video": {"count": 1, "num_frames": 32, "width": 512, "height": 512},
    "image": {"count": 5, "width": 512, "height": 512},
}

# Mixed format
limit_per_prompt = {
    "image": 16,
    "video": {"count": 1, "num_frames": 32},
}
```

### Configuration via LLM API

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    limit_mm_per_prompt={"image": 4, "video": 1},
    mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
    mm_processor_cache_gb=8,
)
```

---

## ObservabilityConfig

`ObservabilityConfig` controls metrics collection and distributed tracing. Defined in `vllm/config/observability.py`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `show_hidden_metrics_for_version` | `str \| None` | `None` | Re-enable deprecated Prometheus metrics hidden since the specified version (e.g., `"0.7"`). Temporary escape hatch during migration. |
| `otlp_traces_endpoint` | `str \| None` | `None` | OpenTelemetry traces endpoint URL. Requires `opentelemetry-sdk` and related packages. |
| `collect_detailed_traces` | `list[str] \| None` | `None` | Collect detailed traces for specified modules: `"model"`, `"worker"`, or `"all"`. Requires `otlp_traces_endpoint`. May have performance impact. |
| `kv_cache_metrics` | `bool` | `False` | Enable KV cache residency metrics (lifetime, idle time, reuse gaps). Uses sampling to minimize overhead. Requires log stats to be enabled. |
| `kv_cache_metrics_sample` | `float` | `0.01` | Sampling rate for KV cache metrics (0.0, 1.0]. Default 1% of blocks. |
| `cudagraph_metrics` | `bool` | `False` | Enable CUDA graph metrics (padded/unpadded tokens, dispatch modes, frequencies). |
| `enable_layerwise_nvtx_tracing` | `bool` | `False` | Enable per-layer NVTX tracing with input/output shape annotations. Does not work with CUDA graphs enabled. |
| `enable_mfu_metrics` | `bool` | `False` | Enable Model FLOPs Utilization (MFU) metrics. |
| `enable_mm_processor_stats` | `bool` | `False` | Enable timing statistics for multimodal processor operations (internal use). |
| `enable_logging_iteration_details` | `bool` | `False` | Log detailed iteration information (context/generation request counts, token counts, CPU time). |

### Computed Properties

| Property | Description |
|----------|-------------|
| `show_hidden_metrics` | Whether hidden metrics should be shown (based on version comparison). |
| `collect_model_forward_time` | Whether to collect model forward time (True if `"model"` or `"all"` in `collect_detailed_traces`). |
| `collect_model_execute_time` | Whether to collect model execute time (True if `"worker"` or `"all"` in `collect_detailed_traces`). |

### Configuration Examples

```python
from vllm.config import ObservabilityConfig

# Enable OpenTelemetry tracing
obs_config = ObservabilityConfig(
    otlp_traces_endpoint="http://jaeger:4317",
    collect_detailed_traces=["model", "worker"],
)
```

```python
# Enable KV cache and MFU metrics
obs_config = ObservabilityConfig(
    kv_cache_metrics=True,
    kv_cache_metrics_sample=0.05,  # 5% sampling
    enable_mfu_metrics=True,
)
```

---

## OffloadConfig

`OffloadConfig` controls CPU weight offloading — moving model weights to CPU memory and loading them on-demand during inference. This allows running models larger than GPU memory at the cost of inference speed. Defined in `vllm/config/offload.py`.

### Offload Backends

| Backend | Description |
|---------|-------------|
| `"auto"` | Automatically select based on which sub-config has non-default values |
| `"uva"` | UVA (Unified Virtual Addressing) zero-copy offloading — simple but requires fast CPU-GPU interconnect |
| `"prefetch"` | Async prefetch with group-based layer offloading — hides transfer latency |

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offload_backend` | `OffloadBackend` | `"auto"` | Which offloading backend to use. |
| `uva` | `UVAOffloadConfig` | `UVAOffloadConfig()` | UVA backend configuration. |
| `prefetch` | `PrefetchOffloadConfig` | `PrefetchOffloadConfig()` | Prefetch backend configuration. |

### UVAOffloadConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cpu_offload_gb` | `float` | `0` | Space in GiB to offload to CPU per GPU. `0` = no offloading. Think of it as virtual GPU memory expansion. |
| `cpu_offload_params` | `set[str]` | `set()` | Parameter name segments to target for offloading. If empty, offloads non-selectively until the memory limit is reached. Segment matching: `"experts"` matches `"mlp.experts.w2_weight"`. |

### PrefetchOffloadConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offload_group_size` | `int` | `0` | Group every N layers together. `0` = disabled. |
| `offload_num_in_group` | `int` | `1` | Number of layers to offload per group. Must be ≤ `offload_group_size`. |
| `offload_prefetch_step` | `int` | `1` | Number of layers to prefetch ahead. Higher values hide more latency but use more GPU memory. |
| `offload_params` | `set[str]` | `set()` | Parameter name segments to target. If empty, offloads ALL parameters of each offloaded layer. |

### Configuration Examples

```python
from vllm import LLM

# UVA offloading — expand effective GPU memory by 10 GiB
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    cpu_offload_gb=10,
)
```

```python
from vllm.config import OffloadConfig, PrefetchOffloadConfig

# Prefetch offloading — offload last 2 layers of every 8-layer group
offload_config = OffloadConfig(
    offload_backend="prefetch",
    prefetch=PrefetchOffloadConfig(
        offload_group_size=8,
        offload_num_in_group=2,
        offload_prefetch_step=2,
        offload_params={"experts"},  # Only offload expert weights
    ),
)
```

---

## KVTransferConfig

`KVTransferConfig` enables disaggregated prefill/decode — a deployment pattern where separate vLLM instances handle prefill and decode phases, transferring KV cache between them. Defined in `vllm/config/kv_transfer.py`.

### Overview

In disaggregated serving:
- **Prefill instance** (`kv_role="kv_producer"`) processes the prompt and sends KV cache to the decode instance
- **Decode instance** (`kv_role="kv_consumer"`) receives KV cache and generates tokens
- **Both** (`kv_role="kv_both"`) handles both roles

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kv_connector` | `str \| None` | `None` | KV connector implementation name for transmitting KV caches between instances. |
| `engine_id` | `str \| None` | `None` | Engine ID for KV transfers. Auto-generated UUID if not set. |
| `kv_buffer_device` | `str` | `"cuda"` | Device for buffering KV cache: `"cuda"` or `"cpu"`. |
| `kv_buffer_size` | `float` | `1e9` | Buffer size in bytes (~1 GB). |
| `kv_role` | `KVRole \| None` | `None` | Role of this instance: `"kv_producer"`, `"kv_consumer"`, or `"kv_both"`. Required when `kv_connector` is set. |
| `kv_rank` | `int \| None` | `None` | Rank of this instance in the KV transfer setup. Typically `0` for prefill, `1` for decode. |
| `kv_parallel_size` | `int` | `1` | Number of parallel instances for KV transfer. For `P2pNcclConnector`, set to `2`. |
| `kv_ip` | `str` | `"127.0.0.1"` | IP address for the KV connector distributed connection. |
| `kv_port` | `int` | `14579` | Port for the KV connector distributed connection. |
| `kv_connector_extra_config` | `dict` | `{}` | Extra configuration for the connector implementation. |
| `kv_connector_module_path` | `str \| None` | `None` | Python module path to dynamically load the KV connector from. V1 only. |
| `enable_permute_local_kv` | `bool` | `False` | Experimental: enable HND to NHD KV transfer permutation. |
| `kv_load_failure_policy` | `Literal["recompute", "fail"]` | `"fail"` | Policy for KV cache load failures: `"recompute"` (reschedule for recomputation) or `"fail"` (immediately fail the request). |

### Properties

| Property | Description |
|----------|-------------|
| `is_kv_transfer_instance` | `True` if `kv_connector` is set and `kv_role` is valid. |
| `is_kv_producer` | `True` if this instance produces KV cache. |
| `is_kv_consumer` | `True` if this instance consumes KV cache. |

### Configuration Example

```python
from vllm.config import KVTransferConfig

# Prefill instance
prefill_config = KVTransferConfig(
    kv_connector="PyNcclConnector",
    kv_role="kv_producer",
    kv_rank=0,
    kv_parallel_size=2,
    kv_ip="192.168.1.100",
    kv_port=14579,
)

# Decode instance
decode_config = KVTransferConfig(
    kv_connector="PyNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,
    kv_parallel_size=2,
    kv_ip="192.168.1.100",
    kv_port=14579,
)
```

### CLI Usage

```bash
# Start prefill instance
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector": "PyNcclConnector", "kv_role": "kv_producer", "kv_rank": 0, "kv_parallel_size": 2}'

# Start decode instance
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector": "PyNcclConnector", "kv_role": "kv_consumer", "kv_rank": 1, "kv_parallel_size": 2}'
```

---

## Related Pages

- [VllmConfig](vllm-config.md) — the parent container
- [ModelConfig](model-config.md) — model and multimodal settings
- [CacheConfig](cache-config.md) — KV cache memory management
- [Environment Variables](environment-variables.md) — `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `VLLM_LORA_RESOLVER_CACHE_DIR`
