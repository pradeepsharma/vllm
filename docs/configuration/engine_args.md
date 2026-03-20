# EngineArgs Reference

`EngineArgs` is the primary entry point for configuring a vLLM engine. Every CLI flag exposed by `vllm serve` and `vllm run` maps to a field in this dataclass. Fields are grouped by the sub-config they belong to.

**Source:** `vllm/engine/arg_utils.py`

---

## Model & Tokenizer

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `model` | `--model` | `str` | `"Qwen/Qwen3-0.6B"` | Name or path of the Hugging Face model. Also used as the `model_name` tag in metrics when `served_model_name` is not set. |
| `tokenizer` | `--tokenizer` | `str \| None` | `None` | Name or path of the HF tokenizer. Defaults to the model path. |
| `tokenizer_mode` | `--tokenizer-mode` | `str` | `"auto"` | Tokenizer mode: `auto`, `hf`, `slow`, `mistral`, `deepseek_v32`. |
| `trust_remote_code` | `--trust-remote-code` | `bool` | `False` | Trust remote code when downloading model/tokenizer. |
| `dtype` | `--dtype` | `str` | `"auto"` | Model weight dtype: `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`. |
| `seed` | `--seed` | `int` | `0` | Random seed for reproducibility. |
| `max_model_len` | `--max-model-len` | `int \| None` | `None` | Maximum context length (prompt + output). Supports human-readable suffixes: `8k`, `128K`. `-1` or `auto` = auto-detect. |
| `revision` | `--revision` | `str \| None` | `None` | Model version (branch, tag, or commit ID). |
| `code_revision` | `--code-revision` | `str \| None` | `None` | Model code revision on HF Hub. |
| `tokenizer_revision` | `--tokenizer-revision` | `str \| None` | `None` | Tokenizer revision on HF Hub. |
| `hf_token` | `--hf-token` | `bool \| str \| None` | `None` | HF Hub token. `True` uses the cached token from `hf auth login`. |
| `hf_overrides` | `--hf-overrides` | `dict` | `{}` | Dict of HF config overrides, or a callable to update the config. |
| `hf_config_path` | `--hf-config-path` | `str \| None` | `None` | Path to a custom HF config directory. |
| `config_format` | `--config-format` | `str` | `"auto"` | Config format: `auto`, `hf`, `mistral`. |
| `served_model_name` | `--served-model-name` | `str \| list[str] \| None` | `None` | Model name(s) exposed in the API. First name used in metrics. |
| `runner` | `--runner` | `str` | `"auto"` | Model runner type: `auto`, `generate`, `pooling`, `draft`. |
| `convert` | `--convert` | `str` | `"auto"` | Convert model using adapters: `auto`, `none`, `embed`, `classify`. |
| `model_impl` | `--model-impl` | `str` | `"auto"` | Model implementation: `auto`, `vllm`, `transformers`, `terratorch`. |
| `generation_config` | `--generation-config` | `str` | `"auto"` | Path to generation config folder. `"vllm"` uses vLLM defaults. |
| `override_generation_config` | `--override-generation-config` | `dict` | `{}` | Override generation config fields, e.g. `{"temperature": 0.5}`. |
| `quantization` | `--quantization` | `str \| None` | `None` | Quantization method: `awq`, `gptq`, `fp8`, `bitsandbytes`, etc. |
| `allow_deprecated_quantization` | `--allow-deprecated-quantization` | `bool` | `False` | Allow deprecated quantization methods. |
| `enforce_eager` | `--enforce-eager` | `bool` | `False` | Disable CUDA graphs; always run in eager mode. |
| `skip_tokenizer_init` | `--skip-tokenizer-init` | `bool` | `False` | Skip tokenizer initialization. Requires `prompt_token_ids` input. |
| `enable_prompt_embeds` | `--enable-prompt-embeds` | `bool` | `False` | Enable passing text embeddings as inputs via `prompt_embeds`. |
| `disable_sliding_window` | `--disable-sliding-window` | `bool` | `False` | Disable sliding window attention. |
| `disable_cascade_attn` | `--disable-cascade-attn` | `bool` | `False` | Disable cascade attention (V1 only). |
| `max_logprobs` | `--max-logprobs` | `int` | `20` | Maximum number of log probabilities to return. `-1` = unlimited. |
| `logprobs_mode` | `--logprobs-mode` | `str` | `"raw_logprobs"` | Logprobs content: `raw_logprobs`, `processed_logprobs`, `raw_logits`, `processed_logits`. |
| `enable_sleep_mode` | `--enable-sleep-mode` | `bool` | `False` | Enable sleep mode (CUDA/HIP only). |
| `enable_return_routed_experts` | `--enable-return-routed-experts` | `bool` | `False` | Return routed expert indices in outputs. |
| `override_attention_dtype` | `--override-attention-dtype` | `str \| None` | `None` | Override dtype for attention computation. |
| `logits_processors` | `--logits-processors` | `list[str]` | `None` | Fully-qualified class names of custom logits processors. |
| `pooler_config` | `--pooler-config` | `dict \| None` | `None` | Pooler config for embedding/classification models (JSON). |
| `model_weights` | `--model-weights` | `str` | `""` | Original model weights path (used with object storage). |
| `allowed_local_media_path` | `--allowed-local-media-path` | `str` | `""` | Allow API to read local media from this directory (security risk). |
| `allowed_media_domains` | `--allowed-media-domains` | `list[str] \| None` | `None` | Restrict multimodal media URLs to these domains. |

---

## Loading

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `load_format` | `--load-format` | `str` | `"auto"` | Model load format: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `tensorizer`, `bitsandbytes`, `gguf`, `runai_streamer`. |
| `download_dir` | `--download-dir` | `str \| None` | `None` | Directory to download and cache model weights. |
| `safetensors_load_strategy` | `--safetensors-load-strategy` | `str` | `"auto"` | Strategy for loading safetensors files. |
| `model_loader_extra_config` | `--model-loader-extra-config` | `dict` | `{}` | Extra config passed to the model loader (JSON). |
| `ignore_patterns` | `--ignore-patterns` | `str \| list[str]` | `[]` | Glob patterns for files to ignore when loading. |
| `use_tqdm_on_load` | `--use-tqdm-on-load` | `bool` | `True` | Show tqdm progress bar during model loading. |
| `pt_load_map_location` | `--pt-load-map-location` | `str \| dict` | `""` | `map_location` argument for `torch.load`. |

---

## Parallelism

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | `int` | `1` | Number of tensor parallel shards. |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | `int` | `1` | Number of pipeline parallel stages. |
| `data_parallel_size` | `--data-parallel-size` | `int` | `1` | Number of data parallel replicas. |
| `distributed_executor_backend` | `--distributed-executor-backend` | `str \| None` | `None` | Distributed backend: `ray`, `mp`, `uni`, `external_launcher`. |
| `enable_expert_parallel` | `--enable-expert-parallel` | `bool` | `False` | Use expert parallelism for MoE layers. |
| `all2all_backend` | `--all2all-backend` | `str` | `"allgather_reducescatter"` | All2All backend for MoE EP: `naive`, `allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `mori`, `flashinfer_all2allv`. |
| `enable_eplb` | `--enable-eplb` | `bool` | `False` | Enable expert parallelism load balancing. |
| `eplb_config` | `--eplb-config` | `dict` | `EPLBConfig()` | EPLB configuration (JSON). |
| `expert_placement_strategy` | `--expert-placement-strategy` | `str` | `"linear"` | Expert placement: `linear` or `round_robin`. |
| `max_parallel_loading_workers` | `--max-parallel-loading-workers` | `int \| None` | `None` | Max workers for parallel model loading. |
| `disable_custom_all_reduce` | `--disable-custom-all-reduce` | `bool` | `False` | Disable custom all-reduce kernel; fall back to NCCL. |
| `ray_workers_use_nsight` | `--ray-workers-use-nsight` | `bool` | `False` | Profile Ray workers with Nsight. |
| `worker_cls` | `--worker-cls` | `str` | `"auto"` | Fully-qualified worker class name. |
| `worker_extension_cls` | `--worker-extension-cls` | `str` | `""` | Worker extension class for `collective_rpc` injection. |
| `master_addr` | `--master-addr` | `str` | `"127.0.0.1"` | Distributed master address (multi-node MP mode). |
| `master_port` | `--master-port` | `int` | `29501` | Distributed master port (multi-node MP mode). |
| `nnodes` | `--nnodes` | `int` | `1` | Number of nodes for multi-node inference. |
| `node_rank` | `--node-rank` | `int` | `0` | Node rank for multi-node inference. |
| `distributed_timeout_seconds` | `--distributed-timeout-seconds` | `int \| None` | `None` | Timeout for distributed ops (e.g., `init_process_group`). |
| `prefill_context_parallel_size` | `--prefill-context-parallel-size` | `int` | `1` | Prefill context parallel groups. |
| `decode_context_parallel_size` | `--decode-context-parallel-size` | `int` | `1` | Decode context parallel groups (reuses TP GPUs). |
| `dcp_comm_backend` | `--dcp-comm-backend` | `str` | `"ag_rs"` | DCP communication backend: `ag_rs` or `a2a`. |
| `enable_elastic_ep` | `--enable-elastic-ep` | `bool` | `False` | Enable elastic expert parallelism. |
| `enable_dbo` | `--enable-dbo` | `bool` | `False` | Enable dual batch overlap. |
| `data_parallel_backend` | `--data-parallel-backend` | `str` | `"mp"` | DP backend: `mp` or `ray`. |

---

## KV Cache

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `float` | `0.9` | Fraction of GPU memory to use for the model executor (0–1). |
| `kv_cache_memory_bytes` | `--kv-cache-memory-bytes` | `int \| None` | `None` | Explicit KV cache size per GPU in bytes. Overrides `gpu_memory_utilization`. Supports `k/m/g` suffixes. |
| `kv_cache_dtype` | `--kv-cache-dtype` | `str` | `"auto"` | KV cache dtype: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, `bfloat16`. |
| `block_size` | `--block-size` | `int` | platform default | KV cache block size in tokens: `1`, `8`, `16`, `32`, `64`, `128`, `256`. |
| `enable_prefix_caching` | `--enable-prefix-caching` / `--no-enable-prefix-caching` | `bool \| None` | `None` | Enable automatic prefix caching (APC). |
| `prefix_caching_hash_algo` | `--prefix-caching-hash-algo` | `str` | `"sha256"` | Hash algorithm for prefix caching: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`. |
| `num_gpu_blocks_override` | `--num-gpu-blocks-override` | `int \| None` | `None` | Override profiled GPU block count (testing only). |
| `calculate_kv_scales` | `--calculate-kv-scales` | `bool` | `False` | Dynamically calculate FP8 KV cache scales. |
| `mamba_cache_dtype` | `--mamba-cache-dtype` | `str` | `"auto"` | Mamba cache dtype: `auto`, `float32`, `float16`. |
| `mamba_ssm_cache_dtype` | `--mamba-ssm-cache-dtype` | `str` | `"auto"` | Mamba SSM state dtype (overrides `mamba_cache_dtype` for SSM). |
| `mamba_block_size` | `--mamba-block-size` | `int \| None` | `None` | Mamba cache block size (must be multiple of 8). |
| `mamba_cache_mode` | `--mamba-cache-mode` | `str` | `"none"` | Mamba cache strategy: `none`, `all`, `align`. |
| `kv_offloading_size` | `--kv-offloading-size` | `float \| None` | `None` | KV cache CPU offload buffer size in GiB. |
| `kv_offloading_backend` | `--kv-offloading-backend` | `str` | `"native"` | KV offloading backend: `native`, `lmcache`. |
| `kv_sharing_fast_prefill` | `--kv-sharing-fast-prefill` | `bool` | `False` | Enable fast-prefill optimization for KV-sharing models (e.g., YOCO). |

---

## Scheduler

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `max_num_batched_tokens` | `--max-num-batched-tokens` | `int \| None` | `None` | Max tokens processed per iteration. Supports `k/m/g` suffixes. |
| `max_num_seqs` | `--max-num-seqs` | `int \| None` | `None` | Max sequences per iteration. |
| `enable_chunked_prefill` | `--enable-chunked-prefill` / `--no-enable-chunked-prefill` | `bool \| None` | `None` | Enable chunked prefill (split long prompts across iterations). |
| `disable_chunked_mm_input` | `--disable-chunked-mm-input` | `bool` | `False` | Prevent partial scheduling of multimodal items in chunked prefill. |
| `max_num_partial_prefills` | `--max-num-partial-prefills` | `int` | `1` | Max concurrent partially-prefilled sequences (chunked prefill). |
| `max_long_partial_prefills` | `--max-long-partial-prefills` | `int` | `1` | Max concurrent long partial prefills (allows short prompts to jump queue). |
| `long_prefill_token_threshold` | `--long-prefill-token-threshold` | `int` | `0` | Token count above which a prompt is considered "long". |
| `scheduling_policy` | `--scheduling-policy` | `str` | `"fcfs"` | Scheduling policy: `fcfs` (first-come-first-served) or `priority`. |
| `scheduler_cls` | `--scheduler-cls` | `str \| None` | `None` | Custom scheduler class path (e.g., `mymod.MyScheduler`). |
| `disable_hybrid_kv_cache_manager` | `--disable-hybrid-kv-cache-manager` | `bool \| None` | `None` | Disable hybrid KV cache manager for mixed attention models. |
| `async_scheduling` | `--async-scheduling` / `--no-async-scheduling` | `bool \| None` | `None` | Enable async scheduling to reduce GPU idle gaps. |
| `stream_interval` | `--stream-interval` | `int` | `1` | Token buffer size for streaming. `1` = send each token immediately. |

---

## Speculative Decoding

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `speculative_config` | `--speculative-config` | `dict \| None` | `None` | Full speculative decoding config as JSON. See [Speculative Config](speculative_config.md). |

---

## LoRA

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `enable_lora` | `--enable-lora` | `bool` | `False` | Enable LoRA adapter serving. |
| `max_loras` | `--max-loras` | `int` | `1` | Max LoRA adapters in a single batch. |
| `max_lora_rank` | `--max-lora-rank` | `int` | `16` | Max LoRA rank: `1`, `8`, `16`, `32`, `64`, `128`, `256`, `320`, `512`. |
| `fully_sharded_loras` | `--fully-sharded-loras` | `bool` | `False` | Use fully sharded LoRA layers (better for high TP/rank). |
| `max_cpu_loras` | `--max-cpu-loras` | `int \| None` | `None` | Max LoRA adapters in CPU memory. Must be ≥ `max_loras`. |
| `lora_dtype` | `--lora-dtype` | `str` | `"auto"` | LoRA weight dtype: `auto`, `float16`, `bfloat16`. |
| `default_mm_loras` | `--default-mm-loras` | `dict \| None` | `None` | Modality → LoRA path mapping for multimodal models. |
| `enable_tower_connector_lora` | `--enable-tower-connector-lora` | `bool` | `False` | Enable LoRA for vision encoder/connector (experimental). |
| `specialize_active_lora` | `--specialize-active-lora` | `bool` | `False` | Capture separate CUDA graphs per active LoRA count. |

---

## Multimodal

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `language_model_only` | `--language-model-only` | `bool` | `False` | Disable multimodal processing even for MM models. |
| `limit_mm_per_prompt` | `--limit-mm-per-prompt` | `dict` | `{}` | Max multimodal items per prompt per modality, e.g. `{"image": 4}`. |
| `enable_mm_embeds` | `--enable-mm-embeds` | `bool` | `False` | Enable passing pre-computed multimodal embeddings. |
| `mm_processor_kwargs` | `--mm-processor-kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to the multimodal processor. |
| `mm_processor_cache_gb` | `--mm-processor-cache-gb` | `float` | `0` | Multimodal processor cache size in GiB. |
| `mm_processor_cache_type` | `--mm-processor-cache-type` | `str \| None` | `None` | Cache type for MM processor: `lru`, `shm`. |
| `mm_shm_cache_max_object_size_mb` | `--mm-shm-cache-max-object-size-mb` | `int` | `0` | Max object size in MB for shared-memory MM cache. |
| `mm_encoder_only` | `--mm-encoder-only` | `bool` | `False` | Run only the multimodal encoder (no LM). |
| `mm_encoder_tp_mode` | `--mm-encoder-tp-mode` | `str` | `"all_ranks"` | TP mode for MM encoder: `all_ranks`, `first_rank_only`. |
| `mm_encoder_attn_backend` | `--mm-encoder-attn-backend` | `str \| None` | `None` | Attention backend for MM encoder. |
| `skip_mm_profiling` | `--skip-mm-profiling` | `bool` | `False` | Skip multimodal profiling during warmup. |
| `video_pruning_rate` | `--video-pruning-rate` | `float \| None` | `None` | Frame pruning rate for video inputs (0–1). |
| `interleave_mm_strings` | `--interleave-mm-strings` | `bool` | `False` | Interleave text and multimodal tokens in the prompt. |
| `media_io_kwargs` | `--media-io-kwargs` | `dict` | `{}` | Per-modality kwargs for media I/O, e.g. `{"image": {"timeout": 10}}`. |

---

## Compilation & CUDA Graphs

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `compilation_config` | `--compilation-config` / `-cc` | `dict` | `CompilationConfig()` | Full compilation config as JSON. See [Compilation Config](compilation_config.md). |
| `cudagraph_capture_sizes` | `--cudagraph-capture-sizes` | `list[int] \| None` | `None` | Explicit CUDA graph capture sizes. |
| `max_cudagraph_capture_size` | `--max-cudagraph-capture-size` | `int \| None` | `None` | Maximum batch size for CUDA graph capture. |
| `optimization_level` | `--optimization-level` | `int` | `2` | Optimization level O0–O3. |
| `performance_mode` | `--performance-mode` | `str` | `"balanced"` | Performance mode: `balanced`, `interactivity`, `throughput`. |

---

## Attention

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `attention_config` | `--attention-config` | `dict` | `AttentionConfig()` | Attention backend configuration (JSON). |
| `attention_backend` | `--attention-backend` | `str \| None` | `None` | Attention backend override (e.g., `flash_attn`, `flashinfer`). |
| `kernel_config` | `--kernel-config` | `dict` | `KernelConfig()` | Kernel configuration (JSON). |
| `enable_flashinfer_autotune` | `--enable-flashinfer-autotune` | `bool` | `True` | Enable FlashInfer kernel autotuning. |
| `moe_backend` | `--moe-backend` | `str` | `"auto"` | MoE kernel backend. |

---

## Observability

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `disable_log_stats` | `--disable-log-stats` | `bool` | `False` | Disable periodic stats logging. |
| `show_hidden_metrics_for_version` | `--show-hidden-metrics-for-version` | `str \| None` | `None` | Re-enable deprecated Prometheus metrics hidden since this version. |
| `otlp_traces_endpoint` | `--otlp-traces-endpoint` | `str \| None` | `None` | OpenTelemetry traces endpoint URL. |
| `collect_detailed_traces` | `--collect-detailed-traces` | `list[str] \| None` | `None` | Collect detailed traces for: `model`, `worker`, `all`. Requires `--otlp-traces-endpoint`. |
| `kv_cache_metrics` | `--kv-cache-metrics` | `bool` | `False` | Enable KV cache residency metrics. |
| `kv_cache_metrics_sample` | `--kv-cache-metrics-sample` | `float` | `0.01` | Sampling rate for KV cache metrics (0–1]. |
| `cudagraph_metrics` | `--cudagraph-metrics` | `bool` | `False` | Enable CUDA graph dispatch metrics. |
| `enable_layerwise_nvtx_tracing` | `--enable-layerwise-nvtx-tracing` | `bool` | `False` | Enable per-layer NVTX tracing (incompatible with CUDA graphs). |
| `enable_mfu_metrics` | `--enable-mfu-metrics` | `bool` | `False` | Enable Model FLOPs Utilization (MFU) metrics. |
| `enable_logging_iteration_details` | `--enable-logging-iteration-details` | `bool` | `False` | Log per-iteration details (request counts, token counts, CPU time). |
| `aggregate_engine_logging` | `--aggregate-engine-logging` | `bool` | `False` | Aggregate engine logs across DP ranks. |

---

## Structured Outputs

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `structured_outputs_config` | `--structured-outputs-config` | `dict` | `StructuredOutputsConfig()` | Structured outputs configuration (JSON). |
| `reasoning_parser` | `--reasoning-parser` | `str` | `""` | Reasoning parser name (e.g., `deepseek_r1`). |
| `reasoning_parser_plugin` | `--reasoning-parser-plugin` | `str \| None` | `None` | Custom reasoning parser plugin path. |

---

## CPU Offloading

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `offload_backend` | `--offload-backend` | `str` | `"none"` | Weight offloading backend. |
| `cpu_offload_gb` | `--cpu-offload-gb` | `float` | `0` | CPU memory to use for weight offloading in GiB. |
| `cpu_offload_params` | `--cpu-offload-params` | `set[str]` | `{}` | Parameter names to offload to CPU. |
| `offload_group_size` | `--offload-group-size` | `int` | `1` | Number of layers per offload group. |
| `offload_num_in_group` | `--offload-num-in-group` | `int` | `1` | Number of layers to keep in GPU per group. |
| `offload_prefetch_step` | `--offload-prefetch-step` | `int` | `1` | Prefetch step for offloaded layers. |
| `offload_params` | `--offload-params` | `set[str]` | `{}` | Parameter names for prefetch offloading. |

---

## KV Transfer (Disaggregated Prefill)

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `kv_transfer_config` | `--kv-transfer-config` | `dict \| None` | `None` | KV cache transfer config for disaggregated prefill (JSON). |
| `kv_events_config` | `--kv-events-config` | `dict \| None` | `None` | KV cache event publishing config (JSON). |
| `ec_transfer_config` | `--ec-transfer-config` | `dict \| None` | `None` | EC cache transfer config (JSON). |

---

## Miscellaneous

| Argument | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `profiler_config` | `--profiler-config` | `dict` | `ProfilerConfig()` | Profiler configuration (JSON). |
| `weight_transfer_config` | `--weight-transfer-config` | `dict \| None` | `None` | Weight transfer config for RL training (JSON). |
| `additional_config` | `--additional-config` | `dict` | `{}` | Platform-specific additional config (JSON). |
| `tokens_only` | `--tokens-only` | `bool` | `False` | Return only token IDs (no text decoding). |
| `shutdown_timeout` | `--shutdown-timeout` | `int` | `0` | Grace period in seconds for in-flight requests on shutdown. |
| `fail_on_environ_validation` | `--fail-on-environ-validation` | `bool` | `False` | Fail on environment variable validation errors. |
| `disable_log_stats` | `--disable-log-stats` | `bool` | `False` | Disable periodic stats logging. |
| `io_processor_plugin` | `--io-processor-plugin` | `str \| None` | `None` | IOProcessor plugin name to load at startup. |

---

## Usage Examples

### Basic model serving

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768
```

### FP8 quantized serving with prefix caching

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-batched-tokens 65536
```

### Speculative decoding with EAGLE

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{"model": "lmzheng/sglang-EAGLE-LLaMA3.1-Instruct-70B", "num_speculative_tokens": 5}'
```

### LoRA serving

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64
```

### Multi-node tensor parallel

```bash
# Node 0
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 10.0.0.1 \
  --master-port 29501

# Node 1
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 10.0.0.1 \
  --master-port 29501
```
