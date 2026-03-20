---
description: >
  Complete reference for `vllm serve` — all flags, argument groups, and
  examples for launching the vLLM OpenAI-compatible API server.
---

# `vllm serve` — API Server Reference

`vllm serve` launches a local OpenAI-compatible HTTP API server that accepts chat completions, text completions, embeddings, and other requests. It is the primary way to deploy vLLM as a production inference service.

```
vllm serve [model_tag] [options]
```

The `model_tag` positional argument is optional — if omitted, vLLM defaults to `Qwen/Qwen3-0.6B`. You can also specify the model in a `--config` YAML file.

---

## :zap: Quick Examples

```bash
# Serve the default model
vllm serve

# Serve a specific model
vllm serve meta-llama/Llama-3.2-8B-Instruct

# Serve on a custom port
vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8100

# Serve over a Unix domain socket
vllm serve meta-llama/Llama-3.2-8B-Instruct --uds /tmp/vllm.sock

# Tensor parallelism across 4 GPUs
vllm serve meta-llama/Llama-3.2-70B-Instruct --tensor-parallel-size 4

# Load options from a YAML config file
vllm serve --config my_config.yaml

# Require an API key
vllm serve meta-llama/Llama-3.2-8B-Instruct --api-key my-secret-key

# Enable LoRA adapters
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --enable-lora \
    --lora-modules adapter1=/path/to/lora

# Enable tool calling
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

---

## :mag: Exploring Options

`vllm serve` has hundreds of flags organized into named groups. Use the built-in help system to explore them:

```bash
# List all argument groups
vllm serve --help=listgroup

# Show all flags in a specific group
vllm serve --help=ModelConfig
vllm serve --help=Frontend
vllm serve --help=CacheConfig
vllm serve --help=ParallelConfig
vllm serve --help=SchedulerConfig

# Search flags by keyword
vllm serve --help=max
vllm serve --help=lora
vllm serve --help=quantization

# Show all flags at once
vllm serve --help=all

# Browse with a pager (less/more)
vllm serve --help=page
```

---

## :page_facing_up: Config File

You can store frequently used options in a YAML file and pass it with `--config`:

```yaml
# vllm_config.yaml
model: meta-llama/Llama-3.2-8B-Instruct
tensor_parallel_size: 2
port: 8100
max_model_len: 8192
gpu_memory_utilization: 0.90
enable_prefix_caching: true
```

```bash
vllm serve --config vllm_config.yaml
```

YAML keys use underscores (matching Python attribute names). CLI flags use hyphens. Both forms are accepted.

---

## :gear: JSON Argument Syntax

Several flags accept JSON objects. vLLM supports a convenient dot-notation shorthand so you don't have to escape JSON on the command line:

```bash
# These are equivalent:
vllm serve --compilation-config '{"level": 3}'
vllm serve --compilation-config.level 3

# Nested keys:
vllm serve --speculative-config '{"method": "ngram", "num_speculative_tokens": 5}'
vllm serve --speculative-config.method ngram --speculative-config.num_speculative_tokens 5
```

List elements can be appended with `+`:

```bash
vllm serve --lora-modules+ '{"name": "adapter1", "path": "/path/to/lora"}'
vllm serve --lora-modules+ '{"name": "adapter2", "path": "/path/to/lora2"}'
```

---

## :clipboard: Top-Level Arguments

These arguments appear before the argument groups and control server-level behavior.

| Flag | Type | Default | Description |
|---|---|---|---|
| `model_tag` | `str` | — | Model to serve (positional, optional). Takes precedence over `--model` if both are set. |
| `--headless` | flag | `false` | Run engine workers without starting an API server. Used in multi-node data-parallel setups where a separate process manages the API layer. |
| `--api-server-count`, `-asc` | `int` | `data_parallel_size` | Number of API server processes to run. Defaults to `data_parallel_size`. Set to `0` with `--headless`. |
| `--config` | `str` | — | Path to a YAML config file. Keys map to CLI flag names (underscores or hyphens). |

---

## :globe_with_meridians: Frontend Arguments

Network, security, CORS, and HTTP server settings.

### Network

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `None` (all interfaces) | Hostname or IP address to bind to. |
| `--port` | `int` | `8000` | TCP port to listen on. |
| `--uds` | `str` | — | Unix domain socket path. When set, `--host` and `--port` are ignored. |
| `--root-path` | `str` | — | FastAPI `root_path` for apps behind a path-based routing proxy. |

### TLS / SSL

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ssl-keyfile` | `str` | — | Path to the SSL private key file. |
| `--ssl-certfile` | `str` | — | Path to the SSL certificate file. |
| `--ssl-ca-certs` | `str` | — | Path to the CA certificates file. |
| `--ssl-cert-reqs` | `int` | `0` (CERT_NONE) | Whether client certificate is required (stdlib `ssl` module constant). |
| `--ssl-ciphers` | `str` | — | SSL cipher suites for TLS 1.2 and below. Example: `ECDHE-RSA-AES256-GCM-SHA384`. |
| `--enable-ssl-refresh` | flag | `false` | Automatically refresh the SSL context when certificate files change on disk. |

### CORS

| Flag | Type | Default | Description |
|---|---|---|---|
| `--allow-credentials` | flag | `false` | Allow credentials in CORS requests. |
| `--allowed-origins` | `JSON` | `["*"]` | JSON list of allowed CORS origins. |
| `--allowed-methods` | `JSON` | `["*"]` | JSON list of allowed HTTP methods. |
| `--allowed-headers` | `JSON` | `["*"]` | JSON list of allowed HTTP headers. |

### Authentication

| Flag | Type | Default | Description |
|---|---|---|---|
| `--api-key` | `str` (repeatable) | — | One or more API keys. If set, every request must include a matching `Authorization: Bearer <key>` header. |

### HTTP Server

| Flag | Type | Default | Description |
|---|---|---|---|
| `--uvicorn-log-level` | `str` | `info` | Uvicorn log level. Choices: `critical`, `error`, `warning`, `info`, `debug`, `trace`. |
| `--disable-uvicorn-access-log` | flag | `false` | Disable uvicorn access logging entirely. |
| `--disable-access-log-for-endpoints` | `str` | — | Comma-separated list of endpoint paths to exclude from access logs (e.g., `/health,/metrics`). |
| `--middleware` | `str` (repeatable) | — | Import path of an ASGI middleware to add. Functions are added via `@app.middleware('http')`; classes via `app.add_middleware()`. |
| `--enable-request-id-headers` | flag | `false` | Add `X-Request-Id` header to all responses. |
| `--disable-fastapi-docs` | flag | `false` | Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoints. |
| `--enable-offline-docs` | flag | `false` | Enable offline FastAPI documentation using vendored static assets (for air-gapped environments). |
| `--h11-max-incomplete-event-size` | `int` | `4194304` | Maximum size (bytes) of an incomplete HTTP event for the h11 parser. Helps mitigate header abuse. |
| `--h11-max-header-count` | `int` | `256` | Maximum number of HTTP headers allowed per request for the h11 parser. |
| `--use-gpu-for-pooling-score` | flag | `false` | Run pooling score MaxSim on GPU in the API server process. Can significantly improve late-interaction scoring performance. |

---

## :speech_balloon: Chat & Template Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--chat-template` | `str` | — | Path to a Jinja2 chat template file, or an inline template string. Overrides the model's built-in template. |
| `--chat-template-content-format` | `str` | `auto` | How to render message content in the chat template. `"string"` renders as a plain string; `"openai"` renders as a list of dicts (OpenAI schema). |
| `--trust-request-chat-template` | flag | `false` | Allow clients to supply their own chat template in requests. When `false`, only the server-side template is used. |
| `--default-chat-template-kwargs` | `JSON` | — | Default keyword arguments passed to the chat template renderer. Merged with per-request `chat_template_kwargs`, with request values taking precedence. Example: `'{"enable_thinking": false}'`. |
| `--response-role` | `str` | `assistant` | Role name returned when `request.add_generation_prompt=true`. |

---

## :wrench: Tool Calling Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-auto-tool-choice` | flag | `false` | Enable automatic tool choice for supported models. Requires `--tool-call-parser`. |
| `--tool-call-parser` | `str` | — | Parser to convert model-generated tool calls into OpenAI API format. Built-in options include `llama3_json`, `hermes`, `mistral`, `granite-20b-fc`, `internlm`, `jamba`, `pythonic`, `xlam`, and others. Register custom parsers via `--tool-parser-plugin`. |
| `--tool-parser-plugin` | `str` | — | Import path of a plugin that registers additional tool call parsers. |
| `--tool-server` | `str` | — | Comma-separated list of `host:port` pairs for external tool servers. Use `demo` for the built-in browser and Python code interpreter demo tools. |
| `--exclude-tools-when-tool-choice-none` | flag | `false` | Exclude tool definitions from prompts when `tool_choice='none'`. |

---

## :floppy_disk: LoRA Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lora-modules` | `str` (repeatable) | — | LoRA module configurations. Accepts `name=path` format or JSON: `'{"name": "adapter", "path": "/path", "base_model_name": "id"}'`. |

---

## :bar_chart: Logging & Observability Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--log-config-file` | `str` | `$VLLM_LOGGING_CONFIG_PATH` | Path to a JSON logging config file for both vLLM and uvicorn. |
| `--max-log-len` | `int` | `None` (unlimited) | Maximum number of prompt characters or token IDs printed in log messages. |
| `--enable-log-requests` | flag | `false` | Log request information (ID, parameters, LoRA request at INFO; prompt inputs at DEBUG). |
| `--enable-log-outputs` | flag | `false` | Log model outputs (generations). Requires `--enable-log-requests`. |
| `--enable-log-deltas` | flag | `true` | Log output deltas during streaming. Only relevant when `--enable-log-outputs` is set. |
| `--log-error-stack` | flag | `false` | Log the full stack trace of error responses. |
| `--disable-log-stats` | flag | `false` | Disable engine statistics logging. |
| `--aggregate-engine-logging` | flag | `false` | Aggregate engine log messages. |

---

## :electric_plug: Usage Tracking Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-prompt-tokens-details` | flag | `false` | Include `prompt_tokens_details` in usage responses. |
| `--enable-server-load-tracking` | flag | `false` | Track `server_load_metrics` in the application state. |
| `--enable-force-include-usage` | flag | `false` | Include usage statistics on every request response. |
| `--enable-tokenizer-info-endpoint` | flag | `false` | Enable the `/tokenizer_info` endpoint (may expose chat templates and tokenizer config). |
| `--return-tokens-as-token-ids` | flag | `false` | When `--max-logprobs` is set, represent tokens as `token_id:{id}` strings so non-JSON-encodable tokens can be identified. |
| `--tokens-only` | flag | `false` | Enable only the Tokens In/Out endpoint. Intended for Disaggregated Everything setups. |

---

## :brain: Model Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `Qwen/Qwen3-0.6B` | HuggingFace model ID or local path. |
| `--model-weights` | `str` | — | Override path for model weights (separate from config). |
| `--served-model-name` | `str` (repeatable) | — | Model name(s) exposed in the `/v1/models` API response. |
| `--tokenizer` | `str` | — | HuggingFace tokenizer ID or local path. Defaults to the model's tokenizer. |
| `--hf-config-path` | `str` | — | Override path for the HuggingFace config directory. |
| `--tokenizer-mode` | `str` | `auto` | Tokenizer mode: `auto`, `hf`, `slow`, `mistral`, `deepseek_v32`, `qwen_vl`, or a custom plugin value. |
| `--trust-remote-code` | flag | `false` | Allow execution of remote code from HuggingFace Hub (e.g., custom model implementations). |
| `--allowed-local-media-path` | `str` | — | Local filesystem path from which the server is allowed to read media files. |
| `--allowed-media-domains` | `str` (repeatable) | — | Domains from which the server is allowed to fetch remote media. |
| `--dtype` | `str` | `auto` | Model weight and activation dtype. Options: `auto`, `half`, `float16`, `bfloat16`, `float`, `float32`. |
| `--seed` | `int` | `0` | Random seed for reproducibility. |
| `--max-model-len` | `int` | — | Maximum sequence length (prompt + output tokens). Defaults to the model's configured maximum. |
| `--max-logprobs` | `int` | `20` | Maximum number of log probabilities to return per token. |
| `--revision` | `str` | — | HuggingFace model revision (branch, tag, or commit hash). |
| `--code-revision` | `str` | — | HuggingFace revision for model code (if different from weights). |
| `--hf-token` | `str` or flag | — | HuggingFace API token for accessing gated models. |
| `--quantization` | `str` | — | Quantization method. Options include `awq`, `gptq`, `squeezellm`, `fp8`, `bitsandbytes`, `gguf`, and others. |
| `--enforce-eager` | flag | `false` | Disable CUDA graph capture and always run in eager mode. Slower but uses less memory. |
| `--max-num-seqs` | `int` | — | Maximum number of sequences processed per iteration. |
| `--max-num-batched-tokens` | `int` | — | Maximum number of tokens processed per iteration. |
| `--generation-config` | `str` | `auto` | Path to a `generation_config.json` file, or `auto` to load from the model directory, or `vllm` to use vLLM defaults. |
| `--override-generation-config` | `JSON` | — | JSON dict of generation config overrides (e.g., `'{"temperature": 0.7}'`). |
| `--runner` | `str` | `auto` | Model runner type. Options: `auto`, `generate`, `pooling`, `draft`. |

---

## :floppy_disk: Load Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--download-dir` | `str` | — | Directory for downloading and caching model weights. Defaults to `~/.cache/huggingface`. |
| `--load-format` | `str` | `auto` | Weight loading format. Options: `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `tensorizer`, `sharded_state`, `gguf`, `bitsandbytes`, `mistral`, `runai_streamer`. |
| `--config-format` | `str` | `auto` | Config loading format. Options: `auto`, `hf`, `mistral`. |
| `--safetensors-load-strategy` | `str` | `auto` | Strategy for loading safetensors files. |
| `--ignore-patterns` | `str` (repeatable) | — | Glob patterns for weight files to ignore during loading. |
| `--model-loader-extra-config` | `JSON` | — | Extra configuration passed to the model loader (loader-specific). |
| `--use-tqdm-on-load` | flag | `true` | Show a tqdm progress bar during model weight loading. |

---

## :arrows_counterclockwise: Parallelism Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--tensor-parallel-size` | `int` | `1` | Number of GPUs for tensor parallelism (splits model layers across GPUs). |
| `--pipeline-parallel-size` | `int` | `1` | Number of pipeline stages (splits model depth across GPUs/nodes). |
| `--data-parallel-size` | `int` | `1` | Number of data-parallel replicas (each holds a full model copy). |
| `--data-parallel-size-local` | `int` | — | Number of data-parallel replicas on this node (for multi-node setups). |
| `--data-parallel-rank` | `int` | — | Data-parallel rank of this process (for external load balancing). |
| `--data-parallel-start-rank` | `int` | — | Starting data-parallel rank for hybrid load balancing. |
| `--data-parallel-address` | `str` | — | Address of the data-parallel coordinator. |
| `--data-parallel-rpc-port` | `int` | — | RPC port for data-parallel coordination. |
| `--data-parallel-external-lb` | flag | `false` | Use an external load balancer for data parallelism. |
| `--data-parallel-hybrid-lb` | flag | `false` | Use hybrid (internal + external) load balancing for data parallelism. |
| `--data-parallel-backend` | `str` | — | Backend for data-parallel communication. |
| `--enable-expert-parallel` | flag | `false` | Enable expert parallelism for MoE models. |
| `--enable-elastic-ep` | flag | `false` | Enable elastic expert parallelism. |
| `--enable-dbo` | flag | `false` | Enable disaggregated batch orchestration. |
| `--distributed-executor-backend` | `str` | `ray` or `mp` | Backend for distributed execution. Options: `ray`, `mp` (multiprocessing). |
| `--max-parallel-loading-workers` | `int` | — | Maximum number of parallel weight-loading workers. |
| `--master-addr` | `str` | — | Master node address for distributed training. |
| `--master-port` | `int` | — | Master node port for distributed training. |
| `--nnodes` | `int` | `1` | Number of nodes in the cluster. |
| `--node-rank` | `int` | `0` | Rank of this node in the cluster. |
| `--worker-cls` | `str` | — | Custom worker class import path. |
| `--worker-extension-cls` | `str` | — | Custom worker extension class import path. |

---

## :package: Cache Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--gpu-memory-utilization` | `float` | `0.90` | Fraction of GPU memory to use for the KV cache (0.0–1.0). |
| `--kv-cache-memory-bytes` | `int` | — | Fixed KV cache size in bytes (overrides `--gpu-memory-utilization`). |
| `--block-size` | `int` | `16` | Token block size for paged attention. |
| `--enable-prefix-caching` | flag | `true` (v1) | Enable automatic prefix caching (APC) to reuse KV cache across requests with shared prefixes. |
| `--prefix-caching-hash-algo` | `str` | — | Hash algorithm for prefix caching. |
| `--num-gpu-blocks-override` | `int` | — | Override the number of GPU KV cache blocks (for testing). |
| `--kv-cache-dtype` | `str` | `auto` | Data type for KV cache storage. Options: `auto`, `fp8`, `fp8_e5m2`, `fp8_e4m3`. |
| `--calculate-kv-scales` | flag | `false` | Calculate KV cache scaling factors dynamically. |
| `--cpu-offload-gb` | `float` | `0` | Amount of CPU RAM (GB) to use for KV cache offloading. |
| `--kv-offloading-size` | `float` | — | Size (GB) of the KV offloading buffer. |
| `--kv-offloading-backend` | `str` | — | Backend for KV cache offloading. |

---

## :calendar: Scheduler Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--scheduling-policy` | `str` | `fcfs` | Request scheduling policy. Options: `fcfs` (first-come-first-served), `priority`. |
| `--scheduler-cls` | `str` | — | Custom scheduler class import path. |
| `--enable-chunked-prefill` | flag | — | Enable chunked prefill to process long prompts in multiple iterations. |
| `--max-num-partial-prefills` | `int` | — | Maximum number of partial prefill requests per iteration. |
| `--max-long-partial-prefills` | `int` | — | Maximum number of long partial prefill requests per iteration. |
| `--long-prefill-token-threshold` | `int` | — | Token threshold above which a prefill is considered "long". |
| `--async-scheduling` | flag | — | Enable asynchronous scheduling. |
| `--stream-interval` | `int` | — | Token interval for streaming responses. |

---

## :camera: Speculative Decoding

| Flag | Type | Default | Description |
|---|---|---|---|
| `--speculative-config` | `JSON` | — | Speculative decoding configuration. Example: `'{"method": "ngram", "num_speculative_tokens": 5}'`. Supports methods: `ngram`, `draft_model`, `medusa`, `mlp_speculator`, `eagle`, `deepseek_mtp`. |

---

## :microscope: Quantization Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--quantization` | `str` | — | Quantization method. Options: `aqlm`, `awq`, `deepspeedfp`, `tpu_int8`, `fp8`, `ptpc_fp8`, `fbgemm_fp8`, `modelopt`, `nvfp4`, `marlin`, `gguf`, `gptq_marlin_24`, `gptq_marlin`, `awq_marlin`, `gptq`, `compressed-tensors`, `bitsandbytes`, `qqq`, `hqq`, `experts_int8`, `neuron_quant`, `ipex`, `quip`, `qoq`, `mxfp8`, `torchao`, `bitnet`, `vptq`, `kv_cache_int8`, `nvfp4_w4a4`, `nvfp4_w4a8`, `nvfp4_w4a16`, `nvfp4_w4a16_mxfp8_kv`, `nvfp4_w4a8_mxfp8_kv`. |
| `--allow-deprecated-quantization` | flag | `false` | Allow deprecated quantization methods. |
| `--kv-cache-dtype` | `str` | `auto` | KV cache quantization dtype. |

---

## :globe_with_meridians: Multimodal Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--limit-mm-per-prompt` | `JSON` | — | Maximum number of multimodal items per prompt per type. Example: `'{"image": 4, "video": 1}'`. |
| `--mm-processor-kwargs` | `JSON` | — | Keyword arguments passed to the multimodal processor. |
| `--mm-processor-cache-gb` | `float` | — | Size (GB) of the multimodal processor cache. |
| `--language-model-only` | flag | `false` | Treat the model as language-model-only (disable multimodal processing). |
| `--enable-mm-embeds` | flag | `false` | Enable multimodal embedding inputs. |
| `--video-pruning-rate` | `float` | — | Frame pruning rate for video inputs (0.0–1.0). |

---

## :link: LoRA Engine Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-lora` | flag | `false` | Enable LoRA adapter support. |
| `--max-loras` | `int` | `1` | Maximum number of LoRA adapters loaded simultaneously. |
| `--max-lora-rank` | `int` | `16` | Maximum LoRA rank. |
| `--max-cpu-loras` | `int` | — | Maximum number of LoRA adapters cached in CPU memory. |
| `--lora-dtype` | `str` | `auto` | Data type for LoRA adapter weights. |
| `--fully-sharded-loras` | flag | `false` | Shard LoRA weights across tensor-parallel ranks. |

---

## :telescope: Observability Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--otlp-traces-endpoint` | `str` | — | OpenTelemetry traces endpoint URL. |
| `--collect-detailed-traces` | `str` (repeatable) | — | Modules for which to collect detailed traces. |
| `--kv-cache-metrics` | flag | `false` | Enable KV cache utilization metrics. |
| `--enable-mfu-metrics` | flag | `false` | Enable Model FLOP Utilization (MFU) metrics. |
| `--enable-logging-iteration-details` | flag | `false` | Log per-iteration scheduling details. |
| `--show-hidden-metrics-for-version` | `str` | — | Expose hidden Prometheus metrics introduced in the specified version. |

---

## :computer: Compilation Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--compilation-config` | `JSON` | — | Compilation configuration. Example: `'{"level": 3}'`. Level 0 = no compilation, level 3 = full torch.compile. |
| `--enforce-eager` | flag | `false` | Disable CUDA graph capture (equivalent to `compilation_config.level=0`). |
| `--cudagraph-capture-sizes` | `int` (repeatable) | — | Specific batch sizes for CUDA graph capture. |
| `--max-cudagraph-capture-size` | `int` | — | Maximum batch size for CUDA graph capture. |
| `--enable-flashinfer-autotune` | flag | `false` | Enable FlashInfer kernel autotuning. |

---

## :arrows_counterclockwise: KV Transfer Arguments (Disaggregated Prefill)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--kv-transfer-config` | `JSON` | — | KV cache transfer configuration for disaggregated prefill/decode setups. |
| `--kv-events-config` | `JSON` | — | KV events configuration. |

---

## :bulb: Common Patterns

### Multi-GPU serving

```bash
# 4-way tensor parallelism (single node)
vllm serve meta-llama/Llama-3.2-70B-Instruct --tensor-parallel-size 4

# Pipeline parallelism across 2 nodes
vllm serve meta-llama/Llama-3.2-70B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4

# Data parallelism (2 replicas, each on 1 GPU)
vllm serve meta-llama/Llama-3.2-8B-Instruct --data-parallel-size 2
```

### Memory optimization

```bash
# Reduce GPU memory usage
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --gpu-memory-utilization 0.80 \
    --max-model-len 4096

# Enable CPU offloading
vllm serve meta-llama/Llama-3.2-8B-Instruct --cpu-offload-gb 10

# Use quantization
vllm serve meta-llama/Llama-3.2-8B-Instruct --quantization awq
```

### Production hardening

```bash
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --api-key "$MY_API_KEY" \
    --ssl-keyfile /etc/ssl/private/server.key \
    --ssl-certfile /etc/ssl/certs/server.crt \
    --max-log-len 100 \
    --disable-fastapi-docs \
    --enable-request-id-headers
```

### Headless mode (multi-node data parallelism)

In headless mode, the process runs engine workers without an API server. A separate process (or external load balancer) handles the API layer.

```bash
# Node 0: API server + engine
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --data-parallel-size 4 \
    --data-parallel-external-lb

# Nodes 1–3: headless engine workers
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --headless \
    --data-parallel-size 4 \
    --data-parallel-rank 1  # (2, 3 on other nodes)
```

---

## :link: Related

- [CLI Overview](index.md) — All vLLM CLI commands
- [OpenAI-Compatible Server guide](../serving/openai_compatible_server.md) — HTTP API endpoints
- [Configuration reference](../configuration/index.md) — Full engine config documentation
- [Deployment guide](../deployment/docker.md) — Docker and Kubernetes
