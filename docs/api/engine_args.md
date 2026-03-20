# EngineArgs / AsyncEngineArgs

```python
from vllm import EngineArgs, AsyncEngineArgs
```

`EngineArgs` and `AsyncEngineArgs` are dataclasses that collect every configuration knob for the vLLM engine. They are the single source of truth for engine configuration and are used by both [`LLM`](llm.md) and [`AsyncLLMEngine`](async_llm_engine.md).

---

## EngineArgs

```python
@dataclass
class EngineArgs:
    model: str
    ...
```

`EngineArgs` is a flat dataclass. Every field has a sensible default; you only need to set the fields relevant to your use case.

### Creating an engine from EngineArgs

```python
from vllm import LLMEngine, EngineArgs

args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)
engine = LLMEngine.from_engine_args(args)
```

### CLI integration

`EngineArgs` fields map 1-to-1 to CLI flags for `vllm serve` and `vllm run`. Use `EngineArgs.add_cli_args(parser)` to add all flags to an `argparse.ArgumentParser`:

```python
import argparse
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)
args = parser.parse_args()
engine_args = EngineArgs.from_cli_args(args)
```

---

## Parameter Reference

### Model & Tokenizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | *(required)* | HuggingFace model name or local path. |
| `tokenizer` | `str \| None` | `None` | Override tokenizer path. Defaults to `model`. |
| `tokenizer_mode` | `str` | `"auto"` | `"auto"` uses the fast tokenizer when available; `"slow"` always uses the slow tokenizer. |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer and detokenizer initialisation. |
| `trust_remote_code` | `bool` | `False` | Allow execution of remote code from HuggingFace Hub. |
| `revision` | `str \| None` | `None` | Model revision (branch, tag, or commit SHA). |
| `code_revision` | `str \| None` | `None` | Revision for the model code on HuggingFace Hub. |
| `tokenizer_revision` | `str \| None` | `None` | Tokenizer revision. |
| `hf_token` | `bool \| str \| None` | `None` | HuggingFace authentication token. |
| `hf_overrides` | `dict \| Callable` | `{}` | Config overrides forwarded to the HuggingFace config. |
| `hf_config_path` | `str \| None` | `None` | Path to a custom HuggingFace config file. |
| `config_format` | `str` | `"auto"` | Config format (`"auto"`, `"hf"`, `"mistral"`). |

### Runner & Conversion

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runner` | `str` | `"auto"` | Execution runner. `"auto"` selects the best runner. Use `"generate"` or `"pooling"`. |
| `convert` | `str` | `"auto"` | Model conversion mode. Use `"embed"`, `"classify"`, or `"score"` to force a pooling task. |

### Precision & Quantization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | `str` | `"auto"` | Model weight and activation dtype: `"auto"`, `"float32"`, `"float16"`, `"bfloat16"`. |
| `quantization` | `str \| None` | `None` | Quantization method: `"awq"`, `"gptq"`, `"fp8"`, etc. `None` auto-detects from model config. |
| `allow_deprecated_quantization` | `bool` | `False` | Allow deprecated quantization methods. |
| `kv_cache_dtype` | `str` | `"auto"` | KV cache dtype: `"auto"`, `"fp8"`, `"fp8_e5m2"`, `"fp8_e4m3"`. |

### Parallelism

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism. |
| `pipeline_parallel_size` | `int` | `1` | Number of pipeline stages. |
| `data_parallel_size` | `int` | `1` | Number of data-parallel replicas. |
| `distributed_executor_backend` | `str \| None` | `None` | Distributed executor: `"ray"`, `"mp"` (multiprocessing), or `"external_launcher"`. |
| `enable_expert_parallel` | `bool` | `False` | Enable expert parallelism for MoE models. |
| `max_parallel_loading_workers` | `int \| None` | `None` | Maximum number of workers for parallel weight loading. |
| `disable_custom_all_reduce` | `bool` | `False` | Disable the custom all-reduce kernel. |
| `master_addr` | `str` | `"localhost"` | Master node address for distributed training. |
| `master_port` | `int` | `0` | Master node port (0 = auto-select). |
| `nnodes` | `int` | `1` | Number of nodes. |
| `node_rank` | `int` | `0` | Rank of this node. |

### Memory Management

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory reserved for model weights, activations, and KV cache. |
| `kv_cache_memory_bytes` | `int \| None` | `None` | Explicit KV cache size per GPU in bytes. Overrides `gpu_memory_utilization` for KV cache sizing when set. |
| `cpu_offload_gb` | `float` | `0` | GiB of CPU memory for weight offloading. |
| `offload_group_size` | `int` | `0` | Prefetch offloading: group every N layers. `0` disables. |
| `offload_num_in_group` | `int` | `1` | Prefetch offloading: layers to offload per group. |
| `offload_prefetch_step` | `int` | `1` | Prefetch offloading: layers to prefetch ahead. |
| `offload_params` | `set[str]` | `set()` | Prefetch offloading: parameter name segments to offload. |
| `num_gpu_blocks_override` | `int \| None` | `None` | Override the number of GPU KV cache blocks. |
| `block_size` | `int` | `16` | KV cache block size in tokens. |

### Scheduling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_num_seqs` | `int \| None` | `None` | Maximum number of sequences per scheduling step. `None` uses the engine default. |
| `max_num_batched_tokens` | `int \| None` | `None` | Maximum number of tokens per scheduling step. `None` uses the engine default. |
| `max_model_len` | `int` | *(from config)* | Maximum sequence length (prompt + output). |
| `enable_chunked_prefill` | `bool \| None` | `None` | Enable chunked prefill. `None` auto-selects. |
| `max_num_partial_prefills` | `int` | `1` | Maximum number of partial prefills per step. |
| `max_long_partial_prefills` | `int` | `1` | Maximum number of long partial prefills per step. |
| `long_prefill_token_threshold` | `int` | `0` | Token threshold for classifying a prefill as "long". |
| `disable_chunked_mm_input` | `bool` | `False` | Disable chunked multimodal input. |

### Prefix Caching

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_prefix_caching` | `bool \| None` | `None` | Enable automatic prefix caching (APC). `None` auto-selects. |
| `prefix_caching_hash_algo` | `str` | `"builtin"` | Hash algorithm for prefix cache keys. |

### LoRA

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_lora` | `bool` | `False` | Enable LoRA adapter support. |
| `max_loras` | `int` | `1` | Maximum number of LoRA adapters loaded simultaneously. |
| `max_lora_rank` | `int` | `16` | Maximum LoRA rank. |
| `max_cpu_loras` | `int \| None` | `None` | Maximum number of LoRA adapters cached on CPU. |
| `lora_dtype` | `str \| None` | `None` | LoRA weight dtype. |
| `fully_sharded_loras` | `bool` | `False` | Shard LoRA weights across tensor-parallel ranks. |
| `default_mm_loras` | `dict[str, str] \| None` | `None` | Default LoRA adapters for specific modalities. |

### Speculative Decoding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `speculative_config` | `dict \| None` | `None` | Speculative decoding configuration dict. Keys include `model`, `num_speculative_tokens`, `draft_tensor_parallel_size`, etc. |

### Multimodal

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `limit_mm_per_prompt` | `dict[str, int]` | `{}` | Maximum number of multimodal items per prompt per modality (e.g. `{"image": 4}`). |
| `mm_processor_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to the multimodal processor. |
| `mm_processor_cache_gb` | `float` | `0` | GiB of memory for the multimodal processor cache. |
| `allowed_local_media_path` | `str` | `""` | Directory from which local media files may be read. |
| `allowed_media_domains` | `list[str] \| None` | `None` | Allowlist of domains for remote media URLs. |
| `language_model_only` | `bool` | `False` | Treat the model as language-model-only (ignore multimodal components). |
| `enable_mm_embeds` | `bool` | `False` | Enable multimodal embedding inputs. |

### Structured Outputs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `structured_outputs_config` | `StructuredOutputsConfig` | *(default)* | Structured output (guided decoding) configuration. |
| `reasoning_parser` | `str` | `""` | Reasoning parser plugin name. |

### Observability

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disable_log_stats` | `bool` | `False` | Disable Prometheus stat logging. |
| `otlp_traces_endpoint` | `str \| None` | `None` | OpenTelemetry traces endpoint URL. |
| `collect_detailed_traces` | `list[str] \| None` | `None` | Modules for which to collect detailed traces. |
| `kv_cache_metrics` | `bool` | `False` | Enable KV cache metrics. |
| `enable_mfu_metrics` | `bool` | `False` | Enable MFU (model FLOP utilisation) metrics. |
| `show_hidden_metrics_for_version` | `str \| None` | `None` | Show hidden metrics for a specific vLLM version. |

### Compilation & Execution

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enforce_eager` | `bool` | `False` | Disable CUDA graph capture. |
| `compilation_config` | `CompilationConfig` | *(default)* | Compilation optimisation configuration. |
| `attention_config` | `AttentionConfig` | *(default)* | Attention backend configuration. |
| `cudagraph_capture_sizes` | `list[int] \| None` | `None` | Batch sizes for CUDA graph capture. |
| `max_cudagraph_capture_size` | `int \| None` | `None` | Maximum batch size for CUDA graph capture. |

### Loading

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_format` | `str` | `"auto"` | Weight loading format: `"auto"`, `"pt"`, `"safetensors"`, `"npcache"`, `"dummy"`, `"tensorizer"`, `"bitsandbytes"`. |
| `download_dir` | `str \| None` | `None` | Directory for downloading model weights. |
| `model_loader_extra_config` | `dict` | `{}` | Extra config forwarded to the model loader. |
| `ignore_patterns` | `str \| list[str]` | `[]` | File patterns to ignore when loading weights. |

### Miscellaneous

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `0` | Global random seed. |
| `max_logprobs` | `int` | `20` | Maximum number of log probabilities to return. |
| `disable_sliding_window` | `bool` | `False` | Disable sliding window attention. |
| `enable_return_routed_experts` | `bool` | `False` | Return routed expert indices for MoE models. |
| `served_model_name` | `str \| list[str] \| None` | `None` | Model name(s) exposed by the API server. |

---

## AsyncEngineArgs

```python
@dataclass
class AsyncEngineArgs(EngineArgs):
    enable_log_requests: bool = False
```

`AsyncEngineArgs` extends `EngineArgs` with one additional field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_log_requests` | `bool` | `False` | Log request IDs and parameters at INFO level (DEBUG level logs prompt inputs). |

### Creating an AsyncLLMEngine from AsyncEngineArgs

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,
    enable_log_requests=True,
)
engine = AsyncLLMEngine.from_engine_args(args)
```

---

## Common Patterns

### Minimal configuration

```python
from vllm import EngineArgs

args = EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
```

### Multi-GPU tensor parallelism

```python
args = EngineArgs(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
)
```

### Quantized model

```python
args = EngineArgs(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
)
```

### Embedding model

```python
args = EngineArgs(
    model="BAAI/bge-base-en-v1.5",
    runner="pooling",
    convert="embed",
)
```

### LoRA-enabled serving

```python
args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64,
)
```

### Memory-constrained deployment

```python
args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.75,
    max_model_len=4096,
    enforce_eager=True,
)
```

### Speculative decoding

```python
args = EngineArgs(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "num_speculative_tokens": 5,
    },
)
```

---

## CLI Flag Mapping

Every `EngineArgs` field maps to a CLI flag for `vllm serve`. The mapping follows Python's `snake_case` → CLI `--kebab-case` convention:

| Python field | CLI flag |
|---|---|
| `model` | `--model` |
| `tensor_parallel_size` | `--tensor-parallel-size` |
| `gpu_memory_utilization` | `--gpu-memory-utilization` |
| `dtype` | `--dtype` |
| `quantization` | `--quantization` / `-q` |
| `max_model_len` | `--max-model-len` |
| `enable_prefix_caching` | `--enable-prefix-caching` |
| `enable_lora` | `--enable-lora` |
| `disable_log_stats` | `--disable-log-stats` |

See `vllm serve --help` for the complete list of CLI flags.
