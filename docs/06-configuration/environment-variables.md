# Environment Variables

vLLM uses a large set of environment variables to control runtime behavior, hardware selection, logging, performance tuning, and feature flags. All variables are defined in `vllm/envs.py` and accessed via the `vllm.envs` module.

## How Environment Variables Work

vLLM reads environment variables lazily through a dictionary of callables:

```python
import vllm.envs as envs

# Access a variable
device = envs.VLLM_TARGET_DEVICE  # e.g., "cuda"
log_level = envs.VLLM_LOGGING_LEVEL  # e.g., "INFO"
```

Variables are evaluated once and cached. The `TYPE_CHECKING` block at the top of `envs.py` provides type annotations for IDE support.

## Installation-Time Variables

These variables affect how vLLM is built and installed:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TARGET_DEVICE` | `"cuda"` | Target device for vLLM: `"cuda"`, `"rocm"`, `"cpu"`. Set at build time. |
| `VLLM_MAIN_CUDA_VERSION` | `"12.9"` | Main CUDA version. Follows PyTorch but can be overridden. |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `"highest"` | PyTorch float32 matmul precision: `"highest"`, `"high"`, `"medium"`. |
| `MAX_JOBS` | `None` | Maximum parallel compilation jobs. Defaults to CPU count. |
| `NVCC_THREADS` | `None` | Number of threads for nvcc. If set, `MAX_JOBS` is reduced to avoid CPU oversubscription. |
| `VLLM_USE_PRECOMPILED` | `False` | Use precompiled binaries (`*.so`) instead of building from source. |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | `False` | Skip adding `+precompiled` suffix to version string. |
| `VLLM_DOCKER_BUILD_CONTEXT` | `False` | Mark that setup.py is running in a Docker build context. |
| `CMAKE_BUILD_TYPE` | `None` | CMake build type: `"Debug"`, `"Release"`, `"RelWithDebInfo"`. |
| `VERBOSE` | `False` | Print verbose logs during installation. |

## Path and Cache Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Root directory for vLLM cache files. Respects `XDG_CACHE_HOME`. |
| `VLLM_CONFIG_ROOT` | `~/.config/vllm` | Root directory for vLLM configuration files. Respects `XDG_CONFIG_HOME`. |
| `VLLM_ASSETS_CACHE` | `~/.cache/vllm/assets` | Cache directory for downloaded assets. |
| `VLLM_ASSETS_CACHE_MODEL_CLEAN` | `False` | Clean model files in the assets cache path. |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA persistent cache directory (TPU only). |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | `None` | Local directory for unrecognized LoRA adapters. Requires plugins and `VLLM_ALLOW_RUNTIME_LORA_UPDATING`. |

## Networking and Distributed Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST_IP` | `""` | IP address of the current node for distributed inference. Set differently on each node in multi-node setups. |
| `VLLM_PORT` | `None` | Communication port for distributed setup. If set and multiple ports are needed, subsequent ports increment from this value. |
| `VLLM_RPC_BASE_PATH` | `tempfile.gettempdir()` | IPC path for frontend API server ↔ backend engine communication in multiprocessing mode. |
| `VLLM_RPC_TIMEOUT` | `10000` (ms) | Timeout in milliseconds for ZMQ client waiting for backend responses. |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `5` (seconds) | Timeout for keeping HTTP connections alive in the API server. |
| `VLLM_NCCL_SO_PATH` | `None` | Path to the NCCL library file. Needed for NCCL ≥ 2.19 bug workaround. |
| `LD_LIBRARY_PATH` | `None` | Used to find the NCCL library when `VLLM_NCCL_SO_PATH` is not set. |
| `VLLM_SKIP_P2P_CHECK` | `True` | Skip P2P capability check. Set to `0` if the program hangs with custom allreduce (potential driver bug). |
| `VLLM_DISABLE_PYNCCL` | `False` | Disable PyNCCL and use `torch.distributed` instead. |
| `VLLM_LOOPBACK_IP` | `""` | Loopback IP address override. |

## Worker and Process Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_WORKER_MULTIPROC_METHOD` | `"fork"` | Multiprocessing method for workers: `"fork"` or `"spawn"`. |
| `LOCAL_RANK` | `0` | Local rank of the process in distributed setup. Used to determine GPU device ID. |
| `CUDA_VISIBLE_DEVICES` | `None` | Controls visible CUDA devices. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `True` | Enable multiprocessing in LLM for the V1 code path. |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | Timeout in seconds for each engine iteration. |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | Timeout in seconds for engine cores to become ready during startup. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for model execution. |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `False` | Keep the OpenAI API server alive even after the engine errors and stops serving. |

## Data Parallel Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DP_RANK` | `0` | Rank of this process in the data parallel group. |
| `VLLM_DP_RANK_LOCAL` | (= `VLLM_DP_RANK`) | Local rank in the data parallel group. |
| `VLLM_DP_SIZE` | `1` | World size of the data parallel group. |
| `VLLM_DP_MASTER_IP` | `"127.0.0.1"` | IP address of the data parallel master node. |
| `VLLM_DP_MASTER_PORT` | `0` | Port of the data parallel master node. |
| `VLLM_MOE_DP_CHUNK_SIZE` | `256` | Token dispatch quantum for MoE models with Data-Parallel + Expert-Parallel. |
| `VLLM_ENABLE_MOE_DP_CHUNK` | `True` | Enable MoE DP chunking. |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` | `False` | Randomize inputs during dummy runs in Data Parallel mode. |

## Ray Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_RAY_PER_WORKER_GPUS` | `1.0` | GPUs per worker in Ray. Fractional values allow multiple actors per GPU. |
| `VLLM_RAY_BUNDLE_INDICES` | `""` | Comma-separated bundle indices for Ray workers. |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `"auto"` | Channel type for Ray Compiled Graph pipeline-parallel communication: `"auto"`, `"nccl"`, or `"shm"`. |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | `False` | Enable GPU communication overlap in Ray's Compiled Graph (experimental). |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | `True` | Use Ray Communicator wrapping vLLM's pipeline parallelism communicator. |
| `VLLM_RAY_DP_PACK_STRATEGY` | `"strict"` | Strategy for packing DP ranks in Ray: `"strict"`, `"fill"`, or `"span"`. |
| `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` | `""` | Extra environment variable prefixes to copy to Ray workers. |
| `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY` | `""` | Extra environment variables to copy to Ray workers. |

## Logging Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CONFIGURE_LOGGING` | `True` | If `0`, vLLM will not configure logging. |
| `VLLM_LOGGING_LEVEL` | `"INFO"` | Default logging level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`. |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | Path to a custom logging configuration file. |
| `VLLM_LOGGING_PREFIX` | `""` | Prefix prepended to all log messages. |
| `VLLM_LOGGING_STREAM` | `"ext://sys.stdout"` | Logging output stream. |
| `VLLM_LOGGING_COLOR` | `"auto"` | Colored logging: `"auto"` (colors when terminal), `"1"` (always), `"0"` (never). |
| `NO_COLOR` | `False` | Standard Unix flag to disable ANSI color codes. |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Interval in seconds for logging statistics. |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | Interval for logging batch size. `-1` = disabled. |
| `VLLM_TRACE_FUNCTION` | `0` | Enable function call tracing (`1` = enabled). Useful for debugging. |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `False` | Log API server responses for debugging. |
| `VLLM_LOG_MODEL_INSPECTION` | `False` | Log model inspection details. |

## Compilation and Performance Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DISABLE_COMPILE_CACHE` | `False` | Disable the torch.compile cache. |
| `VLLM_USE_AOT_COMPILE` | (auto) | Enable Ahead-of-Time compilation. Auto-enabled on PyTorch ≥ 2.10 when compile cache is enabled. |
| `VLLM_USE_BYTECODE_HOOK` | `True` | Enable bytecode hook in `TorchCompileWithNoGuardsWrapper`. |
| `VLLM_FORCE_AOT_LOAD` | `False` | Force loading AOT compiled models from disk. Hard error if loading fails. |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `False` | Load compiled models from cached standalone compile artifacts without re-splitting. |
| `VLLM_USE_STANDALONE_COMPILE` | `True` | Enable Inductor standalone compile (ignored on PyTorch ≤ 2.7). |
| `VLLM_ENABLE_PREGRAD_PASSES` | `False` | Enable Inductor pre-grad passes (disabled by default as they negatively impact cold compile times). |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `"binary"` | Format for saving torch compile cache: `"binary"` (multiprocess-safe) or `"unpacked"` (human-readable). |
| `VLLM_DEBUG_DUMP_PATH` | `None` | Directory for dumping FX graphs. Overrides `CompilationConfig.debug_dump_path`. |
| `VLLM_PATTERN_MATCH_DEBUG` | `None` | FX node name for debugging pattern matching in custom passes. |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `True` | Enable Inductor max autotune. |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | `True` | Enable Inductor coordinate descent tuning. |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `False` | Enable CUDA graph garbage collection. |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `False` | Estimate CUDA graph memory during profiling. |

## Model Loading Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_MODELSCOPE` | `False` | Load models from ModelScope instead of HuggingFace Hub. |
| `VLLM_MODEL_REDIRECT_PATH` | `None` | Path for model redirects. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `False` | Allow `max_model_len` greater than the model's configured maximum. Set to `1` to enable. |
| `VLLM_PP_LAYER_PARTITION` | `None` | Pipeline stage layer partition strategy. |

## API Server Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_API_KEY` | `None` | API key for the vLLM API server. |
| `VLLM_SERVER_DEV_MODE` | `False` | Enable development mode with additional endpoints (e.g., `/reset_prefix_cache`). |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `128` | Maximum requests per asyncio task when processing per-token outputs in V1 AsyncLLM. |
| `VLLM_ENABLE_RESPONSES_API_STORE` | `False` | Enable Responses API store. |

## LoRA Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `False` | Allow loading/unloading LoRA adapters at runtime. |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | `None` | Local directory for unrecognized LoRA adapters. |
| `VLLM_LORA_RESOLVER_HF_REPO_LIST` | `None` | Comma-separated HuggingFace repos containing LoRA adapters for dynamic download. |
| `VLLM_LORA_DISABLE_PDL` | `False` | Disable LoRA PDL (Persistent Data Layout). |

## Multimodal Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE_FETCH_TIMEOUT` | `5` | Timeout in seconds for fetching images. |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `30` | Timeout in seconds for fetching videos. |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `10` | Timeout in seconds for fetching audio. |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` | `True` | Allow HTTP redirects when fetching media URLs. |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | `8` | Thread pool size for parallel media loading. Set to `1` to disable parallelism. |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | Maximum audio file size in MB for speech-to-text requests. |
| `VLLM_VIDEO_LOADER_BACKEND` | `"opencv"` | Video loading backend: `"opencv"` or `"identity"` (raw bytes). |
| `VLLM_MEDIA_CONNECTOR` | `"http"` | Media connector implementation. Default supports HTTP fetching. |
| `VLLM_MM_HASHER_ALGORITHM` | `"blake3"` | Hash algorithm for multimodal content: `"blake3"`, `"sha256"`, or `"sha512"`. Use `sha256`/`sha512` for FIPS compliance. |

## CPU Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CPU_KVCACHE_SPACE` | `None` | CPU KV cache space in GiB (CPU backend only). Defaults to 4 GiB if not set. |
| `VLLM_CPU_OMP_THREADS_BIND` | `"auto"` | CPU core IDs bound by OpenMP threads. Format: `"0-31"`, `"0,1,2"`, `"0-31,33"`. Ranks separated by `\|`. |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | `None` | Number of CPU cores not used by OMP threads. |
| `VLLM_CPU_SGL_KERNEL` | `False` | Use SGL kernels optimized for small batch sizes (CPU backend). |

## ROCm / AMD GPU Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ROCM_USE_AITER` | `False` | Master switch to enable AITER ops on ROCm. |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `False` | Use AITER paged attention. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `True` | Use AITER linear ops (scaled_mm). |
| `VLLM_ROCM_USE_AITER_MOE` | `True` | Use AITER MoE ops. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `True` | Use AITER RMSNorm op. |
| `VLLM_ROCM_USE_AITER_MLA` | `True` | Use AITER MLA ops. |
| `VLLM_ROCM_USE_AITER_MHA` | `True` | Use AITER MHA ops. |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `False` | Use AITER FP4 ASM GEMM. |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `False` | Use AITER Triton RoPE. |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` | Use AITER Triton FP8 BMM kernel. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `True` | Use AITER Triton FP4 BMM kernel. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `False` | Use AITER Triton unified attention for V1. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `True` | Use AITER Triton GEMM kernels. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | Use ROCm skinny GEMM kernels. |
| `VLLM_ROCM_FP8_PADDING` | `True` | Pad FP8 weights to 256 bytes for ROCm. |
| `VLLM_ROCM_MOE_PADDING` | `True` | Pad MoE kernel weights for ROCm. |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | `True` | Use custom paged attention kernel for MI3xx cards. |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `False` | Use shuffled KV cache layout. |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | `256` | Chunk size (MB) for sleeping memory allocations under ROCm. |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `"NONE"` | Quick allreduce quantization level for MI3xx: `"FP"`, `"INT8"`, `"INT6"`, `"INT4"`, `"NONE"`. |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | `True` | Cast BF16 to FP16 for quick allreduce (ROCm lacks BF16 ASM instructions). |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `None` | Max data size (MB) for custom quick allreduce. Larger data uses custom allreduce or RCCL. |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | `False` | Use FP8 MFMA paged attention on ROCm. |

## TPU / XLA Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA persistent cache directory. |
| `VLLM_XLA_CHECK_RECOMPILATION` | `False` | Assert on XLA recompilation after each execution step. |
| `VLLM_XLA_USE_SPMD` | `False` | Enable SPMD mode for TPU backend. |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `0` | TPU bucket padding gap. |
| `VLLM_TPU_MOST_MODEL_LEN` | `None` | Most common model length for TPU. |
| `VLLM_TPU_USING_PATHWAYS` | `False` | Use Pathways for TPU. |

## MoE and Quantization Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FUSED_MOE_CHUNK_SIZE` | `16384` | Chunk size for fused MoE kernel. |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | `True` | Enable fused MoE activation chunking. |
| `VLLM_MLA_DISABLE` | `False` | Disable MLA attention optimizations. |
| `VLLM_USE_DEEP_GEMM` | `True` | Use DeepGEMM for GEMM operations. |
| `VLLM_MOE_USE_DEEP_GEMM` | `True` | Use DeepGEMM for MoE GEMM operations. |
| `VLLM_USE_DEEP_GEMM_E8M0` | `True` | Use DeepGEMM E8M0 format. |
| `VLLM_USE_TRITON_AWQ` | `False` | Use Triton implementations of AWQ. |
| `VLLM_DISABLED_KERNELS` | `[]` | Comma-separated list of quantization kernels to disable (e.g., `MacheteLinearKernel`). |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `False` | Use atomic add in Marlin kernels. |
| `VLLM_USE_FBGEMM` | `False` | Use FBGEMM for GEMM operations. |
| `Q_SCALE_CONSTANT` | `200` | Divisor for dynamic query scale factor in FP8 KV cache. |
| `K_SCALE_CONSTANT` | `200` | Divisor for dynamic key scale factor in FP8 KV cache. |
| `V_SCALE_CONSTANT` | `100` | Divisor for dynamic value scale factor in FP8 KV cache. |

## FlashInfer Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_FLASHINFER_SAMPLER` | `None` | Use FlashInfer sampler. `None` = auto-detect. |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | `394 MiB` | FlashInfer workspace buffer size in bytes. |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `"trtllm"` | FlashInfer allreduce backend: `"auto"`, `"trtllm"`, or `"mnnvl"`. |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `True` | Use symmetric memory for allreduce. |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | `False` | Use FlashInfer for allreduce. |
| `VLLM_USE_NCCL_SYMM_MEM` | `False` | Use NCCL symmetric memory. |
| `VLLM_NCCL_INCLUDE_PATH` | `None` | Path to NCCL include directory. |

## Usage Statistics Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_NO_USAGE_STATS` | `False` | Disable usage statistics collection. Set to `1` to opt out. |
| `VLLM_DO_NOT_TRACK` | `False` | Disable tracking. Also respects the standard `DO_NOT_TRACK` env var. |
| `VLLM_USAGE_STATS_SERVER` | `"https://stats.vllm.ai"` | Usage statistics server URL. |
| `VLLM_USAGE_SOURCE` | `"production"` | Usage source label. |

## S3 / Object Storage Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_ACCESS_KEY_ID` | `None` | S3 access key ID for tensorizer model loading. |
| `S3_SECRET_ACCESS_KEY` | `None` | S3 secret access key. |
| `S3_ENDPOINT_URL` | `None` | S3 endpoint URL. |

## Miscellaneous Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PLUGINS` | `None` | Comma-separated list of plugin names to load. `None` = load all; `""` = load none. |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | `60` | Interval in seconds to log a warning when the ring buffer is full. |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `False` | Allow insecure serialization (e.g., pickle). |
| `VLLM_MSGPACK_ZERO_COPY_THRESHOLD` | `256` | Threshold in bytes for zero-copy msgpack serialization. |
| `VLLM_MQ_MAX_CHUNK_BYTES_MB` | `16` | Maximum chunk size in MB for message queue. |
| `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` | `False` | Disable request ID randomization. |
| `VLLM_DISABLE_LOG_LOGO` | `False` | Disable the vLLM logo in logs. |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `False` | Enable CUDA compatibility mode. |
| `VLLM_CUDA_COMPATIBILITY_PATH` | `None` | Path for CUDA compatibility libraries. |
| `VLLM_CUDART_SO_PATH` | `None` | Path to `libcudart.so` when `find_loaded_library()` doesn't work. |
| `VLLM_NCCL_INCLUDE_PATH` | `None` | Path to NCCL include directory. |
| `VLLM_GC_DEBUG` | `""` | Enable garbage collection debugging. |
| `VLLM_DEBUG_WORKSPACE` | `False` | Enable debug workspace. |
| `VLLM_DEBUG_MFU_METRICS` | `False` | Enable debug MFU metrics. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` | `False` | Disable pinned memory for weight offloading. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA` | `False` | Disable UVA for weight offloading. |
| `VLLM_XGRAMMAR_CACHE_MB` | `0` | XGrammar cache size in MB. |
| `VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS` | `1` | Timeout for tool parsing regex operations. |
| `VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY` | `False` | Automatically retry on JSON tool parsing errors. |
| `VLLM_V1_USE_OUTLINES_CACHE` | `False` | Use Outlines cache for structured outputs in V1. |
| `VLLM_KV_CACHE_LAYOUT` | `None` | KV cache layout: `"NHD"` or `"HND"`. |
| `VLLM_COMPUTE_NANS_IN_LOGITS` | `False` | Compute NaN checks in logits. |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `False` | Disable shared experts stream. |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` | `256` | Token threshold for shared experts stream. |
| `VLLM_USE_V2_MODEL_RUNNER` | `False` | Use V2 model runner. |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `"localhost"` | NIXL side channel host. |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | NIXL side channel port. |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | `480` | NIXL abort request timeout in seconds. |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | `8998` | Mooncake bootstrap port. |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | `480` | Mooncake abort request timeout in seconds. |

## Common Configuration Patterns

### Disable Usage Tracking

```bash
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
```

### Debug Logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1
```

### Custom Cache Location

```bash
export VLLM_CACHE_ROOT=/fast-ssd/vllm-cache
export VLLM_CONFIG_ROOT=/etc/vllm
```

### Multi-Node Setup

```bash
# On each node, set the node's IP
export VLLM_HOST_IP=192.168.1.$(hostname -I | awk '{print $1}' | cut -d. -f4)

# Increase timeout for slow model downloads
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

### Disable Compilation Cache (Development)

```bash
export VLLM_DISABLE_COMPILE_CACHE=1
```

### ROCm with AITER Ops

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
```

### Opt Out of Telemetry

```bash
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
```

## Related Pages

- [VllmConfig](vllm-config.md) — programmatic configuration
- [CompilationConfig](compilation-config.md) — compilation-related env vars
- [ParallelConfig](parallel-config.md) — distributed inference env vars
- [CacheConfig](cache-config.md) — cache-related env vars
