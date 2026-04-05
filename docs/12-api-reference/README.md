# API Reference

Complete reference documentation for all vLLM HTTP, gRPC, and protocol APIs.

## OpenAI-Compatible APIs

vLLM's primary interface is an OpenAI-compatible REST API. See the [Serving Guide](../05-serving/README.md) for setup instructions.

## Additional Protocol APIs

| Page | Description |
|------|-------------|
| [Anthropic Messages API](anthropic-messages-api.md) | `POST /v1/messages` — Anthropic-compatible chat with streaming |
| [gRPC API](grpc-api.md) | Binary gRPC interface for high-performance engine communication |
| [SageMaker Endpoints](sagemaker-api.md) | `/ping` and `/invocations` for AWS SageMaker hosting |
| [MCP Server Protocol](mcp-server.md) | Model Context Protocol integration for external tools |

## Pooling & Embedding APIs

| Page | Description |
|------|-------------|
| [Pooling Endpoints](pooling-endpoints.md) | `/embed`, `/classify`, `/score`, `/rerank` — embedding, classification, cross-encoding |

## Serving Utilities

| Page | Description |
|------|-------------|
| [Serving Utilities](serving-utilities.md) | `/tokenize`, `/detokenize`, `/tokenizer_info` |
| [LoRA Management](lora-management.md) | `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter` |

## Operational Endpoints

| Page | Description |
|------|-------------|
| [Health, Version & Load](health-version-load.md) | `/health`, `/version`, `/load`, `/server_info`, `/metrics` |
| [Cache Management](cache-management.md) | `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache` |
| [Profiling](profiling.md) | `/start_profile`, `/stop_profile` |
| [Sleep / Wake](sleep-wake.md) | `/sleep`, `/wake_up`, `/is_sleeping` |

## Advanced / Distributed APIs

| Page | Description |
|------|-------------|
| [RLHF Endpoints](rlhf-endpoints.md) | `/pause`, `/resume`, `/update_weights`, `/init_weight_transfer_engine` |
| [Disaggregated Prefill](disaggregated-prefill.md) | `/abort_requests`, `/inference/v1/generate` |
| [Elastic Expert Parallelism](elastic-expert-parallelism.md) | `/scale_elastic_ep`, `/is_scaling_elastic_ep` |

---

## Endpoint Availability Matrix

| Endpoint Group | Always Available | Requires Flag |
|---------------|-----------------|---------------|
| OpenAI Chat/Completions | ✅ | — |
| Anthropic Messages | ✅ | — |
| Pooling (embed/classify/score) | ✅ (task-dependent) | `--task` |
| SageMaker `/ping`, `/invocations` | ✅ | — |
| `/tokenize`, `/detokenize` | ✅ | — |
| `/tokenizer_info` | ❌ | `--enable-tokenizer-info-endpoint` |
| LoRA management | ❌ | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` |
| Cache management | ❌ | `VLLM_SERVER_DEV_MODE=true` |
| Profiling | ❌ | `--profiler-config` |
| Sleep/Wake | ❌ | `VLLM_SERVER_DEV_MODE=true` |
| RLHF endpoints | ❌ | `VLLM_SERVER_DEV_MODE=true` |
| Disaggregated prefill | ✅ | `--tokens-only` for `/abort_requests` |
| Elastic EP | ✅ | — |
| gRPC server | ✅ | Separate server process |
