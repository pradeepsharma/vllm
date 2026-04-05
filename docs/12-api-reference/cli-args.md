# vllm serve — CLI Arguments

The `vllm serve` command starts the OpenAI-compatible HTTP server. It accepts a large number of arguments that control the model, serving behavior, parallelism, memory, and security.

**Source:** `vllm/entrypoints/openai/cli_args.py`, `vllm/engine/arg_utils.py`

---

## Quick Start

```bash
# Minimal invocation
vllm serve meta-llama/Llama-3.1-8B-Instruct

# With common options
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --api-key my-secret-key
```

---

## Argument Groups

Arguments are organized into the following groups:

1. [Positional & Top-Level](#positional--top-level)
2. [Frontend / Server Arguments](#frontend--server-arguments)
3. [Model Arguments](#model-arguments)
4. [Load Arguments](#load-arguments)
5. [Parallelism Arguments](#parallelism-arguments)
6. [Cache & Memory Arguments](#cache--memory-arguments)
7. [Scheduler Arguments](#scheduler-arguments)
8. [LoRA Arguments](#lora-arguments)
9. [Structured Outputs & Reasoning](#structured-outputs--reasoning)
10. [Observability Arguments](#observability-arguments)

---

## Positional & Top-Level

| Argument | Default | Description |
|----------|---------|-------------|
| `model_tag` | `null` | The model tag to serve (optional if specified in config file). |
| `--config` | `null` | Path to a YAML config file. All CLI options can be specified in the file. |
| `--headless` | `false` | Run in headless mode for multi-node data parallel setups. |
| `--api-server-count`, `-asc` | `null` | Number of API server processes to run. Defaults to `data_parallel_size`. |

---

## Frontend / Server Arguments

These arguments control the HTTP server behavior, security, and CORS.

### Network

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `null` | Host name to bind to. `null` binds to all interfaces. |
| `--port` | `8000` | Port number. |
| `--uds` | `null` | Unix domain socket path. If set, `--host` and `--port` are ignored. |
| `--root-path` | `null` | FastAPI `root_path` when the app is behind a path-based routing proxy. |

### Authentication

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | `null` | One or more API keys required in the `Authorization: Bearer` header. Can be specified multiple times. Also reads from `VLLM_API_KEY` environment variable. |

### CORS

| Argument | Default | Description |
|----------|---------|-------------|
| `--allowed-origins` | `["*"]` | JSON list of allowed CORS origins. Example: `'["https://app.example.com"]'`. |
| `--allowed-methods` | `["*"]` | JSON list of allowed HTTP methods. |
| `--allowed-headers` | `["*"]` | JSON list of allowed HTTP headers. |
| `--allow-credentials` | `false` | Allow CORS credentials. |

### SSL / TLS

| Argument | Default | Description |
|----------|---------|-------------|
| `--ssl-keyfile` | `null` | Path to the SSL private key file. |
| `--ssl-certfile` | `null` | Path to the SSL certificate file. |
| `--ssl-ca-certs` | `null` | Path to the CA certificates file. |
| `--ssl-cert-reqs` | `0` | Client certificate requirement (stdlib `ssl` module constant). `0` = `CERT_NONE`. |
| `--ssl-ciphers` | `null` | SSL cipher suites for TLS 1.2 and below. Example: `'ECDHE-RSA-AES256-GCM-SHA384'`. |
| `--enable-ssl-refresh` | `false` | Automatically reload SSL certificates when files change. |

### Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--uvicorn-log-level` | `"info"` | Log level for uvicorn: `critical`, `error`, `warning`, `info`, `debug`, `trace`. |
| `--disable-uvicorn-access-log` | `false` | Disable uvicorn access logging. |
| `--disable-access-log-for-endpoints` | `null` | Comma-separated list of endpoint paths to exclude from access logs (e.g., `"/health,/metrics"`). |
| `--log-config-file` | `null` | Path to a JSON logging config file for both vLLM and uvicorn. |
| `--max-log-len` | `null` | Maximum number of prompt characters to print in logs. `null` = unlimited. |
| `--enable-log-outputs` | `false` | Log model outputs (requires `--enable-log-requests`). |
| `--enable-log-deltas` | `true` | Log output deltas (relevant only with `--enable-log-outputs`). |
| `--log-error-stack` | `false` | Log stack traces for error responses. |

### HTTP Server Tuning

| Argument | Default | Description |
|----------|---------|-------------|
| `--h11-max-incomplete-event-size` | `4194304` | Maximum size (bytes) of an incomplete HTTP event for the h11 parser (4 MB). |
| `--h11-max-header-count` | `256` | Maximum number of HTTP headers allowed per request. |
| `--enable-request-id-headers` | `false` | Add `X-Request-Id` header to all responses. |
| `--disable-fastapi-docs` | `false` | Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoints. |
| `--enable-offline-docs` | `false` | Enable offline FastAPI documentation using vendored static assets. |

### Middleware

| Argument | Default | Description |
|----------|---------|-------------|
| `--middleware` | `[]` | ASGI middleware to add. Specify as import paths. Can be repeated. Functions are added with `@app.middleware('http')`, classes with `app.add_middleware()`. |

### Chat & Templates

| Argument | Default | Description |
|----------|---------|-------------|
| `--chat-template` | `null` | Path to a Jinja2 chat template file, or the template as a single-line string. |
| `--chat-template-content-format` | `"auto"` | Format for rendering message content: `"auto"`, `"string"`, or `"openai"`. |
| `--trust-request-chat-template` | `false` | Allow clients to override the chat template in requests. |
| `--default-chat-template-kwargs` | `null` | JSON object of default kwargs for the chat template renderer. Example: `'{"enable_thinking": false}'`. |
| `--response-role` | `"assistant"` | Role name returned when `add_generation_prompt=true`. |

### Tool Calling

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-auto-tool-choice` | `false` | Enable automatic tool choice for supported models. Requires `--tool-call-parser`. |
| `--tool-call-parser` | `null` | Tool call parser to use. Built-in options: `hermes`, `mistral`, `llama3_json`, `granite-20b-fc`, `internlm`, `jamba`, `pythonic`, `xlam`, `kimi_k2`, and others. |
| `--tool-parser-plugin` | `""` | Import path to a custom tool parser plugin. |
| `--tool-server` | `null` | Comma-separated `host:port` pairs for external tool servers, or `"demo"` for built-in demo tools. |
| `--exclude-tools-when-tool-choice-none` | `false` | Exclude tool definitions from prompts when `tool_choice="none"`. |

### LoRA Modules

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-modules` | `null` | LoRA module configurations. Old format: `name=path`. New format: `'{"name": "...", "path": "...", "base_model_name": "..."}'`. Can be repeated. |

### Miscellaneous Frontend

| Argument | Default | Description |
|----------|---------|-------------|
| `--return-tokens-as-token-ids` | `false` | Represent tokens as `"token_id:{id}"` strings in logprobs output. |
| `--disable-frontend-multiprocessing` | `false` | Run the frontend server in the same process as the engine (disables multiprocessing). |
| `--enable-prompt-tokens-details` | `false` | Include `prompt_tokens_details` in usage statistics. |
| `--enable-server-load-tracking` | `false` | Track server load metrics in app state. |
| `--enable-force-include-usage` | `false` | Include usage statistics in every response. |
| `--enable-tokenizer-info-endpoint` | `false` | Enable the `/tokenizer_info` endpoint (may expose chat templates). |
| `--tokens-only` | `false` | Enable only the Tokens In/Out endpoint (for disaggregated serving). |
| `--use-gpu-for-pooling-score` | `false` | Run pooling score MaxSim on GPU for improved late-interaction scoring. |

---

## Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | **required** | HuggingFace model ID or local path. |
| `--served-model-name` | `null` | One or more names to register the model under. Clients use these names in API requests. |
| `--tokenizer` | `null` | Tokenizer to use. Defaults to the model's tokenizer. |
| `--tokenizer-mode` | `"auto"` | Tokenizer mode: `"auto"`, `"slow"`, `"mistral"`. |
| `--trust-remote-code` | `false` | Trust remote code in model/tokenizer definitions. |
| `--dtype` | `"auto"` | Model data type: `"auto"`, `"half"`, `"float16"`, `"bfloat16"`, `"float"`, `"float32"`. |
| `--kv-cache-dtype` | `"auto"` | KV cache data type: `"auto"`, `"fp8"`, `"fp8_e5m2"`, `"fp8_e4m3"`. |
| `--max-model-len` | `null` | Maximum context length. Defaults to the model's maximum. |
| `--quantization`, `-q` | `null` | Quantization method: `awq`, `gptq`, `squeezellm`, `bitsandbytes`, `fp8`, `gguf`, etc. |
| `--enforce-eager` | `false` | Disable CUDA graph capture and always use eager mode. |
| `--max-logprobs` | `20` | Maximum number of log probabilities to return. |
| `--seed` | `0` | Random seed for reproducibility. |
| `--revision` | `null` | HuggingFace model revision (branch, tag, or commit hash). |
| `--tokenizer-revision` | `null` | HuggingFace tokenizer revision. |
| `--hf-token` | `null` | HuggingFace API token for private models. Also reads from `HF_TOKEN` environment variable. |
| `--skip-tokenizer-init` | `false` | Skip tokenizer initialization (for embedding-only use cases). |
| `--disable-sliding-window` | `false` | Disable sliding window attention. |
| `--config-format` | `"auto"` | Model config format: `"auto"`, `"hf"`, `"mistral"`. |

---

## Load Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--load-format` | `"auto"` | Model weight format: `"auto"`, `"pt"`, `"safetensors"`, `"npcache"`, `"dummy"`, `"tensorizer"`, `"bitsandbytes"`, `"gguf"`. |
| `--download-dir` | `null` | Directory to download and cache model weights. |
| `--model-loader-extra-config` | `{}` | Extra config for the model loader (JSON). |
| `--ignore-patterns` | `[]` | File patterns to ignore when loading model weights. |

---

## Parallelism Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--tensor-parallel-size` | `-tp` | `1` | Number of tensor parallel shards. |
| `--pipeline-parallel-size` | `-pp` | `1` | Number of pipeline parallel stages. |
| `--data-parallel-size` | | `1` | Number of data parallel replicas. |
| `--distributed-executor-backend` | | `null` | Distributed executor: `"ray"`, `"mp"` (multiprocessing), or a custom class. |
| `--master-addr` | | `"127.0.0.1"` | Master node address for distributed training. |
| `--master-port` | | `0` | Master node port. |
| `--nnodes` | `-n` | `1` | Number of nodes. |
| `--node-rank` | `-r` | `0` | Rank of this node. |
| `--enable-expert-parallel` | | `false` | Enable expert parallelism for MoE models. |

---

## Cache & Memory Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu-memory-utilization` | `0.90` | Fraction of GPU memory to use for the model and KV cache (0.0–1.0). |
| `--block-size` | `16` | KV cache block size in tokens. |
| `--enable-prefix-caching` | `null` | Enable automatic prefix caching (APC). `null` = auto-detect. |
| `--cpu-offload-gb` | `0` | Amount of CPU memory (GB) to use for offloading model weights. |
| `--num-gpu-blocks-override` | `null` | Override the number of GPU KV cache blocks. |
| `--kv-cache-memory-bytes` | `null` | Explicit KV cache memory budget in bytes. |
| `--swap-space` | `4` | CPU swap space (GB) per GPU for beam search. |

---

## Scheduler Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-num-seqs` | `null` | Maximum number of sequences to process per iteration. |
| `--max-num-batched-tokens` | `null` | Maximum number of tokens to process per iteration. |
| `--enable-chunked-prefill` | `null` | Enable chunked prefill for long prompts. |
| `--scheduling-policy` | `"fcfs"` | Scheduling policy: `"fcfs"` (first-come-first-served) or `"priority"`. |
| `--max-num-partial-prefills` | `1` | Maximum number of partial prefills per step. |

---

## LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-lora` | `false` | Enable LoRA adapter support. |
| `--max-loras` | `1` | Maximum number of LoRA adapters to load simultaneously. |
| `--max-lora-rank` | `16` | Maximum LoRA rank. |
| `--max-cpu-loras` | `null` | Maximum number of LoRA adapters to store in CPU memory. |
| `--lora-dtype` | `null` | Data type for LoRA weights: `"auto"`, `"float16"`, `"bfloat16"`. |
| `--fully-sharded-loras` | `false` | Fully shard LoRA computation across tensor parallel workers. |

---

## Structured Outputs & Reasoning

| Argument | Default | Description |
|----------|---------|-------------|
| `--reasoning-parser` | `"auto"` | Parser for reasoning model outputs: `"deepseek_r1"`, `"qwen3"`, `"granite"`, etc. |
| `--reasoning-parser-plugin` | `null` | Import path to a custom reasoning parser plugin. |

---

## Observability Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--disable-log-stats` | `false` | Disable logging of engine statistics. |
| `--enable-log-requests` | `false` | Log all incoming requests. |
| `--otlp-traces-endpoint` | `null` | OpenTelemetry traces endpoint URL. |
| `--collect-detailed-traces` | `null` | Modules to collect detailed traces for. |
| `--kv-cache-metrics` | `false` | Enable KV cache metrics. |
| `--enable-mfu-metrics` | `false` | Enable Model FLOP Utilization (MFU) metrics. |

---

## Configuration File

All CLI arguments can be specified in a YAML configuration file:

```yaml
# serve_config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
host: 0.0.0.0
port: 8000
tensor_parallel_size: 2
gpu_memory_utilization: 0.90
max_model_len: 8192
dtype: bfloat16
enable_prefix_caching: true
api_key:
  - my-secret-key
allowed_origins: '["https://app.example.com"]'
chat_template: /path/to/template.jinja
enable_auto_tool_choice: true
tool_call_parser: hermes
lora_modules:
  - name: sql-lora
    path: /path/to/sql-lora
```

```bash
vllm serve --config serve_config.yaml
```

---

## Common Recipes

### Production Deployment

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /etc/ssl/private/server.key \
  --ssl-certfile /etc/ssl/certs/server.crt \
  --api-key $VLLM_API_KEY \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  --disable-log-stats false \
  --uvicorn-log-level warning
```

### Development / Debug

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --enable-log-requests \
  --log-error-stack \
  --disable-fastapi-docs false
```

### Tool-Calling Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template /path/to/tool-template.jinja
```

### Multi-LoRA Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --lora-modules sql-lora=/path/to/sql \
  --lora-modules code-lora=/path/to/code \
  --lora-modules math-lora=/path/to/math
```

---

## Related Pages

- [Authentication & SSL](auth-ssl.md) — API key and SSL configuration details
- [POST /v1/chat/completions](chat-completions.md) — Chat completions API
- [POST /v1/models](models.md) — Model listing
- [Batch Inference](batch-inference.md) — Offline batch processing
