---
description: >
  Complete reference for the vLLM command-line interface — serve models,
  run batch jobs, benchmark performance, and collect environment diagnostics.
---

# vLLM CLI Reference

The `vllm` command-line tool is the primary entry point for running and managing vLLM. It exposes a set of subcommands that cover the full lifecycle of model serving: launching an OpenAI-compatible API server, running offline batch inference, benchmarking throughput and latency, and collecting environment diagnostics.

```bash
vllm --help
```

---

## :compass: Command Overview

| Command | Description |
|---|---|
| [`vllm serve`](vllm_serve.md) | Launch an OpenAI-compatible HTTP API server |
| [`vllm run-batch`](vllm_run_batch.md) | Process a JSONL batch file offline and write results |
| [`vllm bench`](vllm_bench.md) | Benchmark latency, throughput, and online serving |
| [`vllm collect-env`](vllm_collect_env.md) | Collect system and environment diagnostics |
| `vllm chat` | Interactive chat via a running API server |
| `vllm complete` | Text completion via a running API server |
| `vllm launch` | Launch individual vLLM components (e.g., rendering server) |

---

## :rocket: Quick Start

### Start a server

```bash
# Serve the default model (Qwen/Qwen3-0.6B)
vllm serve

# Serve a specific model
vllm serve meta-llama/Llama-3.2-8B-Instruct

# Serve on a custom port with tensor parallelism
vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8100 --tensor-parallel-size 2
```

### Run batch inference

```bash
vllm run-batch \
    -i requests.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Benchmark performance

```bash
# Latency benchmark
vllm bench latency --model meta-llama/Llama-3.2-1B-Instruct

# Online serving throughput benchmark
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --host 127.0.0.1 --port 8000
```

### Collect environment info

```bash
vllm collect-env
```

---

## :books: Subcommand Details

### `vllm serve`

Starts a local OpenAI-compatible HTTP API server. Accepts a model identifier as a positional argument (or reads it from a config file). Supports hundreds of configuration flags organized into logical groups.

```bash
vllm serve [model_tag] [options]
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `model_tag` | `Qwen/Qwen3-0.6B` | Model to serve (positional, optional) |
| `--port` | `8000` | HTTP port |
| `--host` | `None` (all interfaces) | Bind address |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--data-parallel-size` | `1` | Number of data-parallel replicas |
| `--api-key` | — | Require this key in request headers |
| `--config` | — | Load options from a YAML config file |

**Explore options interactively:**

```bash
# List all argument groups
vllm serve --help=listgroup

# Show all flags in a specific group
vllm serve --help=ModelConfig
vllm serve --help=Frontend

# Search flags by keyword
vllm serve --help=max

# Show all flags at once
vllm serve --help=all

# Browse with a pager
vllm serve --help=page
```

→ [Full `vllm serve` reference](vllm_serve.md)

---

### `vllm run-batch`

Reads a JSONL batch file (OpenAI Batch API format), processes each request offline using the vLLM engine, and writes results to an output JSONL file. Supports local paths and HTTP(S) URLs for both input and output.

```bash
vllm run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>
```

**Supported endpoints in batch files:**

- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/score`
- `/rerank`

→ [Full `vllm run-batch` reference](vllm_run_batch.md)

---

### `vllm bench`

A suite of benchmarking subcommands for measuring vLLM performance. Requires the `[bench]` extra:

```bash
pip install vllm[bench]
```

Available bench subcommands:

| Subcommand | Description |
|---|---|
| `vllm bench latency` | Measure single-batch generation latency |
| `vllm bench throughput` | Measure offline inference throughput |
| `vllm bench serve` | Measure online serving throughput |
| `vllm bench startup` | Measure model startup time |
| `vllm bench mm-processor` | Benchmark multimodal processor latency |
| `vllm bench sweep` | Parameter sweep across configurations |

→ [Full `vllm bench` reference](vllm_bench.md)

---

### `vllm collect-env`

Collects comprehensive system and environment information useful for bug reports and debugging. Outputs Python version, OS, CUDA version, GPU models, installed packages, and vLLM-specific environment variables.

```bash
vllm collect-env
```

→ [Full `vllm collect-env` reference](vllm_collect_env.md)

---

### `vllm chat`

Sends chat completion requests to a running `vllm serve` instance. Supports interactive multi-turn conversations and single-shot queries.

```bash
# Interactive chat (connects to localhost:8000 by default)
vllm chat

# Single-shot query
vllm chat --quick "What is the capital of France?"

# Connect to a remote server
vllm chat --url http://my-server:8000/v1
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8000/v1` | API server URL |
| `--model-name` | first available | Model to use |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--system-prompt` | — | System prompt for the conversation |
| `-q`, `--quick` | — | Send a single message and exit |

---

### `vllm complete`

Sends text completion requests to a running `vllm serve` instance.

```bash
# Interactive completion
vllm complete

# Single-shot completion
vllm complete --quick "The future of AI is"
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--url` | `http://localhost:8000/v1` | API server URL |
| `--model-name` | first available | Model to use |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--max-tokens` | — | Maximum tokens to generate |
| `-q`, `--quick` | — | Send a single prompt and exit |

---

### `vllm launch`

Launches individual vLLM components. Currently supports:

```bash
# Launch a GPU-less rendering server (preprocessing and postprocessing only)
vllm launch render [model_tag] [options]
```

The `render` subcommand starts a FastAPI server that handles tokenization, chat template rendering, and response formatting without loading model weights — useful in disaggregated serving architectures.

---

## :gear: Global Options

These options apply to the top-level `vllm` command:

| Flag | Description |
|---|---|
| `-v`, `--version` | Print the installed vLLM version and exit |
| `--help` | Show help and exit |

---

## :bulb: Tips

### Config file support

`vllm serve` accepts a `--config` flag pointing to a YAML file. This lets you store frequently used options without repeating them on the command line:

```yaml
# vllm_config.yaml
model: meta-llama/Llama-3.2-8B-Instruct
tensor_parallel_size: 2
port: 8100
max_model_len: 8192
```

```bash
vllm serve --config vllm_config.yaml
```

### JSON argument syntax

Several flags accept JSON objects. vLLM supports a convenient dot-notation shorthand:

```bash
# These are equivalent:
vllm serve --compilation-config '{"level": 3}'
vllm serve --compilation-config.level 3
```

List elements can be appended with `+`:

```bash
vllm serve --lora-modules+ '{"name": "adapter1", "path": "/path/to/lora"}'
```

### Environment variable overrides

Many engine settings can also be controlled via environment variables (prefixed `VLLM_`). Run `vllm collect-env` to see which variables are active in your environment.

---

## :link: Related Resources

- [OpenAI-Compatible Server guide](../serving/openai_compatible_server.md) — HTTP API endpoints and usage
- [Configuration reference](../configuration/index.md) — Engine config groups and YAML format
- [Deployment guide](../deployment/docker.md) — Docker and Kubernetes deployment
- [Benchmarking guide](../benchmarking/) — Interpreting benchmark results
