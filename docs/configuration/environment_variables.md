# Environment Variables

vLLM uses environment variables for low-level configuration that doesn't fit neatly into CLI flags, or for settings that must be available before the engine starts. All `VLLM_*` variables are read at import time.

**Source:** `vllm/envs.py`

---

## Variable Precedence

Environment variables provide defaults that can be overridden by explicit CLI flags or Python constructor arguments:

```
Explicit Python arg > CLI flag > Environment variable > Compiled-in default
```

---

## Installation & Build

| Variable | Default | Description |
|---|---|---|
| `VLLM_TARGET_DEVICE` | `"cuda"` | Target device: `cuda`, `rocm`, `cpu`. Set at build time. |
| `VLLM_MAIN_CUDA_VERSION` | `"12.9"` | Main CUDA version. Follows PyTorch but can be overridden. |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `"highest"` | PyTorch float32 matmul precision: `highest`, `high`, `medium`. |
| `MAX_JOBS` | `None` | Maximum parallel compilation jobs (defaults to CPU count). |
| `NVCC_THREADS` | `None` | Number of threads for nvcc. Reduces `MAX_JOBS` to avoid oversubscription. |
| `VLLM_USE_PRECOMPILED` | `False` | Use precompiled binaries (`*.so`). |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | `False` | Skip `+precompiled` suffix in version string. |
| `VLLM_DOCKER_BUILD_CONTEXT` | `False` | Mark that setup.py is running in a Docker build context. |
| `CMAKE_BUILD_TYPE` | `None` | CMake build type: `Debug`, `Release`, `RelWithDebInfo`. |
| `VERBOSE` | `False` | Print verbose logs during installation. |

---

## Paths & Cache

| Variable | Default | Description |
|---|---|---|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Root directory for vLLM cache files. Respects `XDG_CACHE_HOME`. |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | Root directory for vLLM config files. Respects `XDG_CONFIG_HOME`. |
| `VLLM_ASSETS_CACHE` | `~/.cache/vllm/assets` | Cache for downloaded assets. |
| `VLLM_ASSETS_CACHE_MODEL_CLEAN` | `False` | Clean model files in assets cache on startup. |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA persistent cache directory (TPU only). |
| `VLLM_MODEL_REDIRECT_PATH` | `None` | JSON or TSV file mapping model IDs to local paths. |

---

## Networking & RPC

| Variable | Default | Description |
|---|---|---|
| `VLLM_HOST_IP` | `""` | IP address of the current node (multi-node inference). Set differently on each node. |
| `VLLM_PORT` | `None` | Communication port. If set and multiple ports are needed, subsequent ports increment from this value. |
| `VLLM_RPC_BASE_PATH` | `tempfile.gettempdir()` | IPC path for frontend-backend communication in multiprocessing mode. |
| `VLLM_RPC_TIMEOUT` | `10000` (ms) | Timeout for ZMQ client operations (milliseconds). |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` (seconds) | HTTP keep-alive timeout for the API server. |
| `VLLM_LOOPBACK_IP` | `""` | Loopback IP address override. |

---

## API Server

| Variable | Default | Description |
|---|---|---|
| `VLLM_API_KEY` | `None` | API key for the vLLM API server. |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `False` | Log API server responses for debugging. |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `False` | Keep the API server alive even after the engine errors. |
| `VLLM_SERVER_DEV_MODE` | `False` | Enable development mode (extra endpoints like `/reset_prefix_cache`). |
| `VLLM_ENABLE_RESPONSES_API_STORE` | `False` | Enable Responses API storage. |

---

## Logging

| Variable | Default | Description |
|---|---|---|
| `VLLM_CONFIGURE_LOGGING` | `1` | Enable vLLM logging configuration. Set to `0` to disable. |
| `VLLM_LOGGING_LEVEL` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `VLLM_LOGGING_PREFIX` | `""` | Prefix prepended to all log messages. |
| `VLLM_LOGGING_STREAM` | `"ext://sys.stdout"` | Log output stream. |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | Path to a custom Python logging config file. |
| `VLLM_LOGGING_COLOR` | `"auto"` | Color output: `auto` (when terminal), `1` (always), `0` (never). |
| `NO_COLOR` | `0` | Standard Unix flag to disable ANSI color codes. |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Interval in seconds between stats log entries. |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | Interval for logging batch size. `-1` = disabled. |
| `VLLM_TRACE_FUNCTION` | `0` | Enable function call tracing (debugging). |
| `VLLM_LOG_MODEL_INSPECTION` | `False` | Log model inspection details. |
| `VLLM_DEBUG_MFU_METRICS` | `False` | Enable debug MFU metrics logging. |

---

## Model Loading

| Variable | Default | Description |
|---|---|---|
| `VLLM_USE_MODELSCOPE` | `False` | Load models from ModelScope instead of HuggingFace Hub. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `False` | Allow `max_model_len` greater than the model's configured maximum. |
| `S3_ACCESS_KEY_ID` | `None` | S3 access key ID (for tensorizer S3 loading). |
| `S3_SECRET_ACCESS_KEY` | `None` | S3 secret access key. |
| `S3_ENDPOINT_URL` | `None` | S3 endpoint URL. |

---

## Distributed & Parallelism

| Variable | Default | Description |
|---|---|---|
| `LOCAL_RANK` | `0` | Local rank of the process in distributed setting. |
| `CUDA_VISIBLE_DEVICES` | `None` | Visible CUDA devices. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `"fork"` | Worker multiprocessing method: `fork` or `spawn`. |
| `VLLM_NCCL_SO_PATH` | `None` | Path to the NCCL library file. |
| `LD_LIBRARY_PATH` | `None` | Library search path (used to find NCCL). |
| `VLLM_NCCL_INCLUDE_PATH` | `None` | NCCL include path. |
| `VLLM_SKIP_P2P_CHECK` | `True` | Skip P2P connectivity check (set to `0` if custom allreduce hangs). |
| `VLLM_DISABLE_PYNCCL` | `False` | Disable PyNCCL; use `torch.distributed` instead. |
| `VLLM_PP_LAYER_PARTITION` | `None` | Pipeline stage layer partition strategy. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `True` | Enable multiprocessing in LLM for V1 code path. |
| `VLLM_USE_NCCL_SYMM_MEM` | `False` | Use NCCL symmetric memory. |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `True` | Use symmetric memory for allreduce. |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | `False` | Use FlashInfer for allreduce. |

---

## Data Parallel

| Variable | Default | Description |
|---|---|---|
| `VLLM_DP_RANK` | `0` | Data parallel rank of this process. |
| `VLLM_DP_RANK_LOCAL` | `VLLM_DP_RANK` | Local data parallel rank (SPMD mode). |
| `VLLM_DP_SIZE` | `1` | Data parallel world size. |
| `VLLM_DP_MASTER_IP` | `"127.0.0.1"` | Master node IP for data parallel. |
| `VLLM_DP_MASTER_PORT` | `0` | Master node port for data parallel. |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` | `False` | Randomize dummy inputs during DP warmup runs. |

---

## Ray Integration

| Variable | Default | Description |
|---|---|---|
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `"auto"` | Ray Compiled Graph channel type: `auto`, `nccl`, `shm`. |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | `False` | Enable GPU communication overlap in Ray Compiled Graph. |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | `True` | Use Ray Communicator wrapping for pipeline parallelism. |
| `VLLM_RAY_PER_WORKER_GPUS` | `1.0` | GPUs per Ray worker. Fractions allow multiple actors per GPU. |
| `VLLM_RAY_BUNDLE_INDICES` | `""` | Comma-separated bundle indices for Ray workers. |
| `VLLM_RAY_DP_PACK_STRATEGY` | `"strict"` | DP rank packing strategy for Ray: `strict`, `fill`, `span`. |
| `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` | `""` | Additional env var prefixes to copy to Ray workers. |
| `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` | `""` | Additional individual env vars to copy to Ray workers. |

---

## Compilation & CUDA Graphs

| Variable | Default | Description |
|---|---|---|
| `VLLM_DISABLE_COMPILE_CACHE` | `False` | Disable the compilation cache. |
| `VLLM_USE_AOT_COMPILE` | auto | Enable AOT (Ahead-of-Time) compilation. Auto-enabled on PyTorch ≥ 2.10. |
| `VLLM_USE_BYTECODE_HOOK` | `True` | Enable bytecode hook in TorchCompile wrapper. |
| `VLLM_FORCE_AOT_LOAD` | `False` | Force loading AOT compiled models; fail if not found. |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `False` | Load from mega AOT artifact without re-splitting graph modules. |
| `VLLM_USE_STANDALONE_COMPILE` | `True` | Use Inductor standalone compile (torch ≥ 2.9). |
| `VLLM_ENABLE_PREGRAD_PASSES` | `False` | Enable Inductor pre-grad passes (disabled by default for performance). |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `"binary"` | Cache save format: `binary` (multiprocess safe) or `unpacked`. |
| `VLLM_DEBUG_DUMP_PATH` | `None` | Directory for debug FX graph dumps. Overrides `CompilationConfig.debug_dump_path`. |
| `VLLM_PATTERN_MATCH_DEBUG` | `None` | FX node name for pattern match debugging. |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `True` | Enable Inductor max autotune. |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | `True` | Enable coordinate descent tuning in Inductor. |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `False` | Enable CUDA graph garbage collection. |
| `VLLM_ENABLE_PREGRAD_PASSES` | `False` | Enable Inductor pre-grad passes. |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `False` | Estimate CUDA graph memory during profiling. |

---

## KV Cache & Memory

| Variable | Default | Description |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | `None` (4 GiB) | CPU KV cache space in GiB (CPU backend only). |
| `VLLM_KV_CACHE_LAYOUT` | `None` | KV cache layout: `NHD` or `HND`. |
| `Q_SCALE_CONSTANT` | `200` | Divisor for dynamic query scale factor (FP8 KV cache). |
| `K_SCALE_CONSTANT` | `200` | Divisor for dynamic key scale factor (FP8 KV cache). |
| `V_SCALE_CONSTANT` | `100` | Divisor for dynamic value scale factor (FP8 KV cache). |
| `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | `True` | Use integer block hashes for KV events. |

---

## CPU Backend

| Variable | Default | Description |
|---|---|---|
| `VLLM_CPU_OMP_THREADS_BIND` | `"auto"` | CPU core IDs for OpenMP threads. Format: `"0-31"`, `"0,1,2"`, `"0-31,33"`. Ranks separated by `\|`. |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `None` | CPU cores to exclude from OpenMP thread binding. |
| `VLLM_CPU_SGL_KERNEL` | `False` | Use SGL kernels (optimized for small batch on CPU). |

---

## Multimodal

| Variable | Default | Description |
|---|---|---|
| `VLLM_IMAGE_FETCH_TIMEOUT` | `5` | Timeout in seconds for fetching images. |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `30` | Timeout in seconds for fetching videos. |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `10` | Timeout in seconds for fetching audio. |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` | `True` | Allow HTTP redirects when fetching media URLs. |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | `8` | Thread pool size for media loading. Set to `1` to disable parallel loading. |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | Maximum audio file size in MB for speech-to-text. |
| `VLLM_VIDEO_LOADER_BACKEND` | `"opencv"` | Video loading backend: `opencv`, `identity`, or custom. |
| `VLLM_MEDIA_CONNECTOR` | `"http"` | Media connector: `http` or custom. |
| `VLLM_MM_HASHER_ALGORITHM` | `"blake3"` | Hash algorithm for multimodal content: `blake3`, `sha256`, `sha512`. |

---

## MoE (Mixture of Experts)

| Variable | Default | Description |
|---|---|---|
| `VLLM_FUSED_MOE_CHUNK_SIZE` | `16384` | Chunk size for fused MoE computation. |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | `True` | Enable fused MoE activation chunking. |
| `VLLM_MOE_DP_CHUNK_SIZE` | `256` | Token dispatch quantum for MoE with DP+EP. |
| `VLLM_ENABLE_MOE_DP_CHUNK` | `True` | Enable MoE DP chunking. |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK` | `True` | Use fused grouped top-k for MoE. |
| `VLLM_MLA_DISABLE` | `False` | Disable MLA (Multi-head Latent Attention) optimizations. |

---

## FlashInfer

| Variable | Default | Description |
|---|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER` | `None` | Use FlashInfer sampler. `None` = auto-detect. |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | `394 MiB` | FlashInfer workspace buffer size in bytes. |
| `VLLM_FLASHINFER_MOE_BACKEND` | `"latency"` | FlashInfer MoE backend: `throughput`, `latency`, `masked_gemm`. Requires SM100+. |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `"trtllm"` | FlashInfer allreduce backend: `auto`, `trtllm`, `mnnvl`. |
| `VLLM_HAS_FLASHINFER_CUBIN` | `False` | Whether FlashInfer cubin is available. |
| `VLLM_USE_FLASHINFER_MOE_FP16` | `False` | Use FlashInfer for FP16 MoE. |
| `VLLM_USE_FLASHINFER_MOE_FP8` | `False` | Use FlashInfer for FP8 MoE. |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `False` | Use FlashInfer for FP4 MoE. |
| `VLLM_USE_FLASHINFER_MOE_INT4` | `False` | Use FlashInfer for INT4 MoE. |
| `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` | `True` | Use FlashInfer for block-scale FP8 GEMM. |

---

## Quantization Kernels

| Variable | Default | Description |
|---|---|---|
| `VLLM_USE_TRITON_AWQ` | `False` | Use Triton AWQ kernels instead of CUDA. |
| `VLLM_DISABLED_KERNELS` | `[]` | Comma-separated list of kernels to disable (e.g., `MacheteLinearKernel`). |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `False` | Use atomic add in GPTQ/AWQ Marlin kernel. |
| `VLLM_MARLIN_INPUT_DTYPE` | `None` | Override Marlin input dtype: `int8`, `fp8`. |
| `VLLM_MXFP4_USE_MARLIN` | `None` | Use Marlin for MXFP4 quantization. |
| `VLLM_USE_DEEP_GEMM` | `True` | Use DeepGEMM for FP8 GEMM. |
| `VLLM_MOE_USE_DEEP_GEMM` | `True` | Use DeepGEMM for MoE FP8 GEMM. |
| `VLLM_USE_DEEP_GEMM_E8M0` | `True` | Use E8M0 scaling with DeepGEMM on Blackwell. |
| `VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES` | `True` | Use TMA-aligned scale tensors with DeepGEMM. |
| `VLLM_DEEP_GEMM_WARMUP` | `"relax"` | DeepGEMM warmup mode: `skip`, `full`, `relax`. |
| `VLLM_NVFP4_GEMM_BACKEND` | `None` | NVFP4 GEMM backend override. |
| `VLLM_USE_NVFP4_CT_EMULATIONS` | `False` | Use NVFP4 compute type emulations. |
| `VLLM_DEEPEPLL_NVFP4_DISPATCH` | `False` | Use DeepEPLL kernels for NVFP4 dispatch (Blackwell only). |
| `VLLM_USE_FBGEMM` | `False` | Use FBGEMM kernels. |
| `VLLM_USE_OINK_OPS` | `False` | Use OINK ops. |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | `False` | Compute NaN values in logits (debugging). |

---

## ROCm / AMD GPU

| Variable | Default | Description |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | `False` | Enable AITER (AMD Inference Toolkit for Efficient Reasoning) ops. |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `False` | Use AITER paged attention. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `True` | Use AITER linear ops. |
| `VLLM_ROCM_USE_AITER_MOE` | `True` | Use AITER MoE ops. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `True` | Use AITER RMSNorm. |
| `VLLM_ROCM_USE_AITER_MLA` | `True` | Use AITER MLA. |
| `VLLM_ROCM_USE_AITER_MHA` | `True` | Use AITER MHA. |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `False` | Use AITER FP4 ASM GEMM. |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `False` | Use AITER Triton RoPE. |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` | Use AITER Triton FP8 BMM kernel. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `True` | Use AITER Triton FP4 BMM kernel. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `False` | Use AITER unified attention for V1. |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | `False` | Use AITER fusion shared experts ops. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `True` | Use AITER Triton GEMM kernels. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | Use ROCm skinny GEMM kernels. |
| `VLLM_ROCM_FP8_PADDING` | `True` | Pad FP8 weights to 256 bytes for ROCm. |
| `VLLM_ROCM_MOE_PADDING` | `True` | Pad weights for the MoE kernel on ROCm. |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | `True` | Use custom paged attention kernel for MI3xx. |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `False` | Use shuffled KV cache layout on ROCm. |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | `256` | Chunk size (MB) for sleeping memory allocations on ROCm. |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | `False` | Use FP8 MFMA paged attention on ROCm. |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `"NONE"` | Quick allreduce quantization for MI3xx: `FP`, `INT8`, `INT6`, `INT4`, `NONE`. |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | `True` | Cast BF16 to FP16 in quick allreduce (ROCm). |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `None` | Max data size (MB) for quick allreduce on ROCm. |

---

## TPU / XLA

| Variable | Default | Description |
|---|---|---|
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA persistent cache directory. |
| `VLLM_XLA_CHECK_RECOMPILATION` | `False` | Assert on XLA recompilation after each step. |
| `VLLM_XLA_USE_SPMD` | `False` | Enable SPMD mode for TPU backend. |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `0` | Gap between padding buckets for TPU forward pass. |
| `VLLM_TPU_MOST_MODEL_LEN` | `None` | Most common model length for TPU bucket optimization. |
| `VLLM_TPU_USING_PATHWAYS` | `False` | Whether using Pathways (auto-detected from `JAX_PLATFORMS`). |

---

## LoRA

| Variable | Default | Description |
|---|---|---|
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `False` | Allow loading/unloading LoRA adapters at runtime. |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | `None` | Local directory for unrecognized LoRA adapters. |
| `VLLM_LORA_RESOLVER_HF_REPO_LIST` | `None` | Comma-separated HF repos containing LoRA adapters. |
| `VLLM_LORA_DISABLE_PDL` | `False` | Disable LoRA PDL (Persistent Data Layout). |

---

## Structured Outputs

| Variable | Default | Description |
|---|---|---|
| `VLLM_V1_USE_OUTLINES_CACHE` | `False` | Enable Outlines cache for V1 (unbounded disk cache — not safe for untrusted users). |
| `VLLM_XGRAMMAR_CACHE_MB` | `0` | XGrammar cache size in MB. `0` = use default (512 MB). |
| `VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS` | `1` | Timeout for tool parsing regex operations. |
| `VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY` | `False` | Automatically retry on JSON tool parsing errors. |
| `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` | `False` | Use experimental parser context. |

---

## Disaggregated Prefill / KV Transfer

| Variable | Default | Description |
|---|---|---|
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `"localhost"` | Host for NIXL handshake between remote agents. |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | Port for NIXL handshake. |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | `480` | Timeout (seconds) for NIXL request abort. |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `8998` | Port for Mooncake handshake. |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | `480` | Timeout (seconds) for Mooncake request abort. |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `False` | Enable MORIIO connector read mode. |
| `VLLM_MORIIO_QP_PER_TRANSFER` | `1` | Queue pairs per transfer for MORIIO. |
| `VLLM_MORIIO_POST_BATCH_SIZE` | `-1` | Post batch size for MORIIO. |
| `VLLM_MORIIO_NUM_WORKERS` | `1` | Number of MORIIO workers. |

---

## DeepEP (Expert Parallelism)

| Variable | Default | Description |
|---|---|---|
| `VLLM_DEEPEP_BUFFER_SIZE_MB` | `1024` | DeepEP buffer size in MB. |
| `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` | `False` | Force intra-node communication for DeepEP high-throughput. |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | `False` | Use MNNVL for DeepEP low-latency mode. |
| `VLLM_DBO_COMM_SMS` | `20` | Number of SMs for DBO communication. |

---

## Dual Batch Overlap

| Variable | Default | Description |
|---|---|---|
| `VLLM_DBO_COMM_SMS` | `20` | Number of streaming multiprocessors for DBO communication. |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `False` | Disable shared experts stream. |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | `256` | Token threshold for shared experts stream. |

---

## Serialization

| Variable | Default | Description |
|---|---|---|
| `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` | `256` | Tensor size threshold (bytes) for zero-copy msgpack serialization. |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `False` | Allow insecure pickle-based serialization. |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | `16` | Maximum chunk size in MB for message queue. |

---

## Usage Statistics

| Variable | Default | Description |
|---|---|---|
| `VLLM_NO_USAGE_STATS` | `False` | Disable usage statistics collection. |
| `VLLM_DO_NOT_TRACK` | `False` | Opt out of usage tracking (also respects `DO_NOT_TRACK`). |
| `VLLM_USAGE_STATS_SERVER` | `"https://stats.vllm.ai"` | Usage statistics server URL. |
| `VLLM_USAGE_SOURCE` | `"production"` | Usage source tag. |

---

## Miscellaneous

| Variable | Default | Description |
|---|---|---|
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | Timeout in seconds for each engine iteration. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | Timeout in seconds for engine core startup. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for model execution. |
| `VLLM_PLUGINS` | `None` | Comma-separated list of plugins to load. Empty string = no plugins. |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | `60` | Interval (seconds) for ring buffer full warnings. |
| `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` | `True` | Allow chunked local attention with hybrid KV cache. |
| `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` | `False` | Skip random suffix in internal request IDs. |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `128` | Max requests per asyncio task for V1 output processing. |
| `VLLM_DISABLE_LOG_LOGO` | `False` | Disable the vLLM logo in startup logs. |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `False` | Enable CUDA compatibility mode. |
| `VLLM_CUDA_COMPATIBILITY_PATH` | `None` | Path for CUDA compatibility libraries. |
| `VLLM_CUDART_SO_PATH` | `None` | Path to `libcudart.so` (when auto-detection fails). |
| `VLLM_GC_DEBUG` | `""` | Garbage collection debug mode. |
| `VLLM_DEBUG_WORKSPACE` | `False` | Enable debug workspace. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` | `False` | Disable pinned memory for weight offloading. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` | `False` | Disable UVA (Unified Virtual Addressing) for weight offloading. |
| `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` | `False` | Enable elastic EP scale-up launch. |
| `VLLM_ELASTIC_EP_DRAIN_REQUESTS` | `False` | Drain requests during elastic EP scaling. |
| `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` | `"VLLM_OBJECT_STORAGE_SHM_BUFFER"` | Shared memory buffer name for object storage. |
| `VLLM_TUNED_CONFIG_FOLDER` | `None` | Folder containing tuned kernel configurations. |
| `VLLM_USE_V2_MODEL_RUNNER` | `False` | Use V2 model runner. |
| `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` | `163840` | Maximum tokens per expert for FP4 MoE. |
| `VLLM_CUSTOM_SCOPES_FOR_PROFILING` | `False` | Enable custom scopes for profiling. |
| `VLLM_NVTX_SCOPES_FOR_PROFILING` | `False` | Enable NVTX scopes for profiling. |

---

## Setting Environment Variables

### Shell

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_GPU_MEMORY_UTILIZATION=0.85
vllm serve mymodel
```

### Inline

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve mymodel
```

### Docker

```dockerfile
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_NO_USAGE_STATS=1
```

### Kubernetes

```yaml
env:
  - name: VLLM_LOGGING_LEVEL
    value: "INFO"
  - name: VLLM_NO_USAGE_STATS
    value: "1"
  - name: VLLM_HOST_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
```

---

## Privacy: Disabling Usage Statistics

vLLM collects anonymous usage statistics by default. To opt out:

```bash
export VLLM_NO_USAGE_STATS=1
# or
export VLLM_DO_NOT_TRACK=1
# or
export DO_NOT_TRACK=1  # standard Unix convention
```
