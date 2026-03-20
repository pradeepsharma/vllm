---
description: >
  Complete documentation sitemap for vLLM — every page organized by section
  with direct links for quick navigation.
hide:
  - toc
---

# Documentation Sitemap

This page lists every documentation page in the vLLM docs, organized by section.
Use it to quickly find any topic or to get an overview of what is covered.

---

## :material-rocket-launch: Getting Started

| Page | Description |
|---|---|
| [Overview](getting_started/index.md) | Introduction, prerequisites, and path selection |
| [Quickstart](getting_started/quickstart.md) | Install vLLM and run your first model in minutes |
| [First Inference](getting_started/first-inference.md) | Step-by-step first inference walkthrough |
| **Installation** | |
| [Installation Overview](getting_started/installation/index.md) | Choose the right installation method |
| [NVIDIA CUDA](getting_started/installation/gpu-cuda.md) | Install on NVIDIA GPUs (A100, H100, H200…) |
| [AMD ROCm](getting_started/installation/gpu-rocm.md) | Install on AMD GPUs (MI250, MI300…) |
| [Intel XPU](getting_started/installation/xpu.md) | Install on Intel XPU (Arc, Flex, Max) |
| [CPU (x86 / ARM / s390x)](getting_started/installation/cpu.md) | CPU-only inference on x86, ARM, and IBM Z |
| [Google TPU](getting_started/installation/tpu.md) | Install on Google Cloud TPUs |
| [Intel Gaudi (HPU)](getting_started/installation/hpu.md) | Install on Intel Gaudi accelerators |
| [Ascend NPU](getting_started/installation/npu.md) | Install on Huawei Ascend NPUs |
| [Apple macOS](getting_started/installation/macos.md) | Install on Apple Silicon (M1/M2/M3) |
| [Build from Source](getting_started/installation/from-source.md) | Build vLLM from source for development |

---

## :material-book-open-variant: Usage

| Page | Description |
|---|---|
| [Overview](usage/index.md) | Usage guide index and quick reference |
| [Offline Inference](usage/offline_inference.md) | Run inference without a server using the `LLM` class |
| [Sampling Parameters](usage/sampling_params.md) | Temperature, top-p, top-k, and all sampling options |
| [Pooling Parameters](usage/pooling_params.md) | Pooling configuration for embedding models |
| [Output Types](usage/output_types.md) | Understanding `RequestOutput` and `CompletionOutput` |
| [Beam Search](usage/beam_search.md) | Beam search decoding configuration |
| [Log Probabilities](usage/logprobs.md) | Requesting and interpreting log probabilities |
| [Tokenization](usage/tokenization.md) | Tokenizer configuration, special tokens, and chat templates |
| [Chat Templates](usage/chat_templates.md) | Configuring and customizing chat templates |
| [Tool Call Templates](usage/tool_templates.md) | Tool call and function calling templates |
| [V1 Engine Guide](usage/v1_guide.md) | Migration guide and new features in the V1 engine |
| [Reproducibility](usage/reproducibility.md) | Ensuring deterministic outputs |
| [Usage Statistics](usage/usage_stats.md) | Anonymous usage statistics collection |
| [Troubleshooting](usage/troubleshooting.md) | Common issues and solutions |
| [FAQ](usage/faq.md) | Frequently asked questions |

---

## :material-server: Serving

| Page | Description |
|---|---|
| [Overview](serving/index.md) | Serving guide index |
| [OpenAI-Compatible Server](serving/openai_compatible_server.md) | Full reference for the OpenAI-compatible API server |
| [Chat Completions](serving/chat_completions.md) | `/v1/chat/completions` endpoint reference |
| [Completions](serving/completions.md) | `/v1/completions` endpoint reference |
| [Embeddings](serving/embeddings.md) | `/v1/embeddings` endpoint reference |
| [Responses API](serving/responses_api.md) | `/v1/responses` endpoint reference |
| [Realtime API](serving/realtime_api.md) | WebSocket-based realtime streaming API |
| [Speech to Text](serving/speech_to_text.md) | Audio transcription endpoint |
| [Anthropic API](serving/anthropic_api.md) | Anthropic-compatible API compatibility layer |
| [gRPC Server](serving/grpc_server.md) | High-performance gRPC inference server |
| [Multimodal Serving](serving/multimodal_serving.md) | Serving vision-language and multimodal models |
| [Batch Inference](serving/batch_inference.md) | Offline batch processing with the server |
| [Streaming](serving/streaming.md) | Server-sent events and streaming responses |
| [Parallelism & Scaling](serving/parallelism_scaling.md) | Overview of all parallelism strategies |
| [Data Parallel Deployment](serving/data_parallel_deployment.md) | Data parallelism for independent request batches |
| [Context Parallel Deployment](serving/context_parallel_deployment.md) | Context parallelism for long-context workloads |
| [Expert Parallel Deployment](serving/expert_parallel_deployment.md) | Expert parallelism for MoE models |
| [Distributed Troubleshooting](serving/distributed_troubleshooting.md) | Debugging distributed inference issues |
| [Amazon SageMaker](serving/sagemaker.md) | Deploying vLLM on Amazon SageMaker |
| **Integrations** | |
| [LangChain](serving/integrations/langchain.md) | Using vLLM with LangChain |
| [LlamaIndex](serving/integrations/llamaindex.md) | Using vLLM with LlamaIndex |
| [Claude Code](serving/integrations/claude_code.md) | Using vLLM with Claude Code |

---

## :material-chip: Models

| Page | Description |
|---|---|
| [Overview](models/index.md) | Model support overview and selection guide |
| [Supported Models](models/supported_models.md) | Complete list of supported model architectures |
| [Generative Models](models/generative_models.md) | Decoder-only LLMs for text generation |
| [Embedding Models](models/embedding_models.md) | Models for text embeddings and similarity |
| [Pooling Models](models/pooling_models.md) | Models with pooling for classification and reranking |
| [Multimodal Models](models/multimodal_models.md) | Vision-language and audio-language models |
| [Encoder-Decoder Models](models/encoder_decoder_models.md) | Seq2seq models (T5, BART, Whisper…) |
| [Adding a Model](models/adding_model.md) | How to add a new model architecture to vLLM |
| [Model Registry](models/model_registry.md) | How the model registry works |
| **Hardware Support** | |
| [CPU Models](models/hardware_supported_models/cpu.md) | Models supported on CPU backends |
| [XPU Models](models/hardware_supported_models/xpu.md) | Models supported on Intel XPU |
| **Model Extensions** | |
| [Tensorizer](models/extensions/tensorizer.md) | Fast model loading with CoreWeave Tensorizer |
| [RunAI Model Streamer](models/extensions/runai_model_streamer.md) | Streaming model loading with RunAI |
| [FastSafetensor](models/extensions/fastsafetensor.md) | Accelerated safetensor loading |

---

## :material-puzzle: Features

### Quantization

| Page | Description |
|---|---|
| [Overview](features/quantization/index.md) | Quantization methods comparison and selection guide |
| [GPTQ](features/quantization/gptq.md) | GPTQ 4-bit and 8-bit quantization |
| [AWQ](features/quantization/awq.md) | Activation-aware Weight Quantization |
| [FP8](features/quantization/fp8.md) | FP8 W8A8 quantization |
| [INT8 (SmoothQuant)](features/quantization/int8.md) | INT8 weight and activation quantization |
| [INT4](features/quantization/int4.md) | INT4 weight quantization |
| [GGUF](features/quantization/gguf.md) | GGUF format support |
| [BitsAndBytes](features/quantization/bitsandbytes.md) | BitsAndBytes 4-bit and 8-bit quantization |
| [TorchAO](features/quantization/torchao.md) | TorchAO quantization toolkit |
| [MXFP4](features/quantization/mxfp4.md) | Microscaling FP4 quantization |
| [KV Cache Quantization](features/quantization/kv_cache_quantization.md) | FP8 and INT8 KV cache quantization |
| [GPTQModel](features/quantization/gptqmodel.md) | GPTQModel integration |
| [LLM Compressor](features/quantization/llm_compressor.md) | Neural Magic LLM Compressor |
| [ModelOpt](features/quantization/modelopt.md) | NVIDIA ModelOpt quantization |
| [Quark](features/quantization/quark.md) | AMD Quark quantization |
| [INC](features/quantization/inc.md) | Intel Neural Compressor |
| [AutoAWQ](features/quantization/auto_awq.md) | AutoAWQ quantization |

### Speculative Decoding

| Page | Description |
|---|---|
| [Overview](features/speculative_decoding/README.md) | Speculative decoding overview and method comparison |
| [EAGLE](features/speculative_decoding/eagle.md) | EAGLE and EAGLE-2 speculative decoding |
| [Medusa](features/speculative_decoding/medusa.md) | Medusa multi-head speculative decoding |
| [N-gram](features/speculative_decoding/ngram.md) | N-gram prompt lookup decoding |
| [Draft Model](features/speculative_decoding/draft_model.md) | Smaller draft model speculation |
| [Parallel Draft Model](features/speculative_decoding/parallel_draft_model.md) | Parallel draft model execution |
| [MLP Speculator](features/speculative_decoding/mlp.md) | MLP-based token speculation |
| [MTP](features/speculative_decoding/mtp.md) | Multi-Token Prediction |
| [Suffix Decoding](features/speculative_decoding/suffix.md) | Suffix-based speculation |
| [Speculators Hub](features/speculative_decoding/speculators.md) | Pre-trained speculator models |

### Other Features

| Page | Description |
|---|---|
| [LoRA Adapters](features/lora.md) | Multi-LoRA serving and adapter management |
| [Multimodal Inputs](features/multimodal_inputs.md) | Images, audio, and video inputs |
| [Structured Outputs](features/structured_outputs.md) | JSON schema, regex, and grammar-constrained generation |
| [Tool Calling](features/tool_calling.md) | Function calling and tool use |
| [Disaggregated Prefill](features/disagg_prefill.md) | Separate prefill and decode across nodes |
| [Disaggregated Encoder](features/disagg_encoder.md) | Separate encoder execution |
| [Automatic Prefix Caching](features/automatic_prefix_caching.md) | Reuse KV cache across requests |
| [Custom Logits Processors](features/custom_logitsprocs.md) | Custom token sampling logic |
| [Reasoning Outputs](features/reasoning_outputs.md) | Chain-of-thought and reasoning token support |
| [Interleaved Thinking](features/interleaved_thinking.md) | Interleaved thinking mode |
| [Prompt Embeddings](features/prompt_embeds.md) | Direct embedding inputs |
| [Sleep Mode](features/sleep_mode.md) | Resource-saving sleep and wake |
| [Batch Invariance](features/batch_invariance.md) | Deterministic outputs regardless of batch size |
| [Custom Arguments](features/custom_arguments.md) | Extending vLLM with custom CLI arguments |
| [Mooncake Connector](features/mooncake_connector_usage.md) | Mooncake distributed KV cache connector |
| [NIXL Connector](features/nixl_connector_usage.md) | NIXL high-performance transfer connector |

---

## :material-tune: Configuration

| Page | Description |
|---|---|
| [Overview](configuration/index.md) | Configuration reference index |
| [Engine Arguments](configuration/engine_args.md) | All `EngineArgs` parameters |
| [Serve Arguments](configuration/serve_args.md) | All `vllm serve` arguments |
| [Model Config](configuration/model_config.md) | Model loading and architecture configuration |
| [Parallel Config](configuration/parallel_config.md) | Parallelism configuration |
| [Scheduler Config](configuration/scheduler_config.md) | Scheduler tuning parameters |
| [Cache Config](configuration/cache_config.md) | KV cache configuration |
| [Quantization Config](configuration/quantization_config.md) | Quantization configuration |
| [LoRA Config](configuration/lora_config.md) | LoRA adapter configuration |
| [Speculative Config](configuration/speculative_config.md) | Speculative decoding configuration |
| [Compilation Config](configuration/compilation_config.md) | torch.compile configuration |
| [Observability Config](configuration/observability_config.md) | Metrics, tracing, and logging configuration |
| [Model Resolution](configuration/model_resolution.md) | How vLLM resolves model names and paths |
| [Optimization](configuration/optimization.md) | Performance optimization settings |
| [Conserving Memory](configuration/conserving_memory.md) | Memory reduction techniques |
| [Environment Variables](configuration/environment_variables.md) | All `VLLM_*` environment variables |

---

## :material-cloud-upload: Deployment

| Page | Description |
|---|---|
| [Overview](deployment/index.md) | Deployment guide index and checklist |
| [Docker](deployment/docker.md) | Running vLLM in Docker containers |
| [Kubernetes](deployment/kubernetes.md) | Deploying vLLM on Kubernetes |
| [Multi-Node](deployment/multi_node.md) | Multi-node deployment without Ray |
| [Ray Cluster](deployment/ray_cluster.md) | Multi-node deployment with Ray |
| [Elastic Expert Parallelism](deployment/elastic_ep.md) | Dynamic expert parallelism scaling |
| [Load Balancing](deployment/load_balancing.md) | Load balancing multiple vLLM instances |
| [SSL / TLS](deployment/ssl_tls.md) | Securing vLLM with SSL/TLS |
| [Security](deployment/security.md) | Security hardening and best practices |
| [Production Checklist](deployment/production_checklist.md) | Pre-production readiness checklist |
| [Monitoring](deployment/monitoring.md) | Prometheus, Grafana, and observability |
| **Frameworks** | |
| [Helm](deployment/frameworks/helm.md) | Helm chart for Kubernetes |
| [SkyPilot](deployment/frameworks/skypilot.md) | Cloud deployment with SkyPilot |
| [HF Inference Endpoints](deployment/frameworks/hf_inference_endpoints.md) | Hugging Face Inference Endpoints |
| [dStack](deployment/frameworks/dstack.md) | GPU cloud with dStack |
| [BentoML](deployment/frameworks/bentoml.md) | BentoML serving framework |
| [Cerebrium](deployment/frameworks/cerebrium.md) | Serverless GPU with Cerebrium |
| [RunPod](deployment/frameworks/runpod.md) | RunPod serverless GPU |
| [Modal](deployment/frameworks/modal.md) | Modal serverless platform |
| [Anyscale](deployment/frameworks/anyscale.md) | Anyscale platform |
| [LiteLLM](deployment/frameworks/litellm.md) | LiteLLM proxy |
| [Dify](deployment/frameworks/dify.md) | Dify AI application platform |
| [Haystack](deployment/frameworks/haystack.md) | Haystack NLP framework |
| [AutoGen](deployment/frameworks/autogen.md) | Microsoft AutoGen |
| [Open WebUI](deployment/frameworks/open-webui.md) | Open WebUI chat interface |
| [Streamlit](deployment/frameworks/streamlit.md) | Streamlit apps |
| [Chatbox](deployment/frameworks/chatbox.md) | Chatbox desktop client |
| [Lobe Chat](deployment/frameworks/lobe-chat.md) | Lobe Chat interface |
| [Anything LLM](deployment/frameworks/anything-llm.md) | AnythingLLM platform |
| [Triton](deployment/frameworks/triton.md) | NVIDIA Triton Inference Server |
| [LWS](deployment/frameworks/lws.md) | LeaderWorkerSet for Kubernetes |
| [RAG](deployment/frameworks/retrieval_augmented_generation.md) | Retrieval-Augmented Generation patterns |
| **Integrations** | |
| [KubeRay](deployment/integrations/kuberay.md) | KubeRay operator |
| [KServe](deployment/integrations/kserve.md) | KServe model serving |
| [KAITO](deployment/integrations/kaito.md) | Kubernetes AI Toolchain Operator |
| [KubeAI](deployment/integrations/kubeai.md) | KubeAI platform |
| [AIBrix](deployment/integrations/aibrix.md) | AIBrix inference platform |
| [Dynamo](deployment/integrations/dynamo.md) | NVIDIA Dynamo |
| [LlamaStack](deployment/integrations/llamastack.md) | Meta LlamaStack |
| [llm-d](deployment/integrations/llm-d.md) | llm-d distributed inference |
| [llmaz](deployment/integrations/llmaz.md) | llmaz cloud-native LLM serving |
| [Kthena](deployment/integrations/kthena.md) | Kthena inference platform |
| [Production Stack](deployment/integrations/production-stack.md) | vLLM Production Stack |

---

## :material-sitemap: Design & Architecture

| Page | Description |
|---|---|
| [Architecture Overview](design/arch_overview.md) | High-level architecture and entrypoints |
| [V1 Engine](design/v1_engine.md) | V1 engine internals: AsyncLLM, EngineCore, workers |
| [Scheduler](design/scheduler.md) | Unified scheduling algorithm and request states |
| [KV Cache Management](design/kv_cache_management.md) | Block pool, prefix caching, and eviction |
| [Distributed Inference](design/distributed_inference.md) | Parallelism groups and worker communication |
| [Speculative Decoding Design](design/speculative_decoding_design.md) | Internal speculative decoding architecture |
| [PagedAttention](design/paged_attention.md) | PagedAttention memory management |
| [Prefix Caching](design/prefix_caching.md) | Automatic prefix caching design |
| [Hybrid KV Cache Manager](design/hybrid_kv_cache_manager.md) | Hybrid attention KV cache management |
| [Attention Backends](design/attention_backends.md) | FlashAttention, FlashInfer, and other backends |
| [CUDA Graphs](design/cuda_graphs.md) | CUDA graph capture and replay |
| [torch.compile](design/torch_compile.md) | torch.compile integration |
| [torch.compile (Multimodal)](design/torch_compile_multimodal.md) | torch.compile for multimodal models |
| [Optimization Levels](design/optimization_levels.md) | Compilation and optimization level selection |
| [Fusions](design/fusions.md) | Kernel fusion strategies |
| [Custom Ops](design/custom_op.md) | Custom operator registration |
| [Multiprocessing](design/multiprocessing.md) | Multi-process architecture |
| [Model Runner V2](design/model_runner_v2.md) | Model runner V2 design |
| [Multimodal Processing](design/mm_processing.md) | Multimodal input processing pipeline |
| [Logits Processors](design/logits_processors.md) | Logits processor design |
| [Metrics](design/metrics.md) | Metrics collection and export |
| [MoE Kernel Features](design/moe_kernel_features.md) | Mixture-of-Experts kernel features |
| [Fused MoE Modular Kernel](design/fused_moe_modular_kernel.md) | Fused MoE kernel architecture |
| [DBO](design/dbo.md) | Disaggregated batch orchestration |
| [HuggingFace Integration](design/huggingface_integration.md) | HuggingFace model loading integration |
| [P2P NCCL Connector](design/p2p_nccl_connector.md) | P2P NCCL KV transfer connector |
| [Plugin System](design/plugin_system.md) | vLLM plugin architecture |
| [I/O Processor Plugins](design/io_processor_plugins.md) | Input/output processor plugin system |
| [LoRA Resolver Plugins](design/lora_resolver_plugins.md) | LoRA adapter resolver plugins |
| [Debug vLLM Compile](design/debug_vllm_compile.md) | Debugging torch.compile issues |

---

## :material-console: CLI Reference

| Page | Description |
|---|---|
| [Overview](cli/index.md) | CLI reference index |
| [vllm serve](cli/vllm_serve.md) | Start the OpenAI-compatible API server |
| [vllm run-batch](cli/vllm_run_batch.md) | Run offline batch inference |
| [vllm bench](cli/vllm_bench.md) | Benchmark throughput and latency |
| [vllm collect-env](cli/vllm_collect_env.md) | Collect environment information for bug reports |

---

## :material-code-braces: API Reference

| Page | Description |
|---|---|
| [Overview](api/index.md) | Python API reference index |
| [LLM](api/llm.md) | `vllm.LLM` — synchronous offline inference |
| [AsyncLLMEngine](api/async_llm_engine.md) | `vllm.AsyncLLMEngine` — async inference engine |
| [LLMEngine](api/llm_engine.md) | `vllm.LLMEngine` — synchronous engine |
| [EngineArgs](api/engine_args.md) | `vllm.EngineArgs` — engine configuration |
| [SamplingParams](api/sampling_params.md) | `vllm.SamplingParams` — sampling configuration |
| [PoolingParams](api/pooling_params.md) | `vllm.PoolingParams` — pooling configuration |
| [Outputs](api/outputs.md) | Output types: `RequestOutput`, `CompletionOutput` |
| [OpenAI Protocol](api/openai_protocol.md) | OpenAI protocol data models |

---

## :material-flask: Benchmarking

| Page | Description |
|---|---|
| [Overview](benchmarking/index.md) | Benchmarking guide index |
| [Performance Benchmarks](benchmarking/performance_benchmarks.md) | End-to-end performance benchmark results |
| [Latency Benchmarks](benchmarking/latency_benchmarks.md) | Latency measurement methodology |
| [Throughput Benchmarks](benchmarking/throughput_benchmarks.md) | Throughput measurement methodology |
| [LM Eval Harness](benchmarking/lm_eval_harness.md) | Accuracy evaluation with lm-evaluation-harness |
| [Profiling](benchmarking/profiling.md) | GPU profiling with Nsight and PyTorch Profiler |
| [Benchmark CLI](benchmarking/cli.md) | `vllm bench` CLI reference |
| [Dashboard](benchmarking/dashboard.md) | Performance dashboard setup |
| [Sweeps](benchmarking/sweeps.md) | Automated benchmark sweeps |

---

## :material-code-block-tags: Examples

| Page | Description |
|---|---|
| [Overview](examples/index.md) | Examples index |
| [Offline Inference](examples/offline_inference/index.md) | Offline inference code examples |
| [Online Serving](examples/online_serving/index.md) | Online serving code examples |
| [Pooling](examples/pooling/index.md) | Pooling and embedding examples |
| [Other Examples](examples/others/index.md) | Miscellaneous examples |

---

## :material-dumbbell: Training Integration

| Page | Description |
|---|---|
| [Overview](training/index.md) | Training integration overview |
| [Weight Transfer API](training/weight_transfer_api.md) | Live weight transfer between training and inference |
| [RLHF](training/rlhf.md) | RLHF integration patterns |
| [TRL](training/trl.md) | TRL (Transformer Reinforcement Learning) integration |

---

## :material-source-pull: Contributing

| Page | Description |
|---|---|
| [Overview](contributing/index.md) | Contributing guide index |
| [Development Setup](contributing/development_setup.md) | Set up a development environment |
| [Code Style](contributing/code_style.md) | Code formatting and style guidelines |
| [Testing](contributing/testing.md) | Running and writing tests |
| [CI/CD](contributing/ci_cd.md) | Continuous integration and deployment |
| [Release Process](contributing/release_process.md) | How vLLM releases are made |
| [Adding a Model](contributing/adding_model.md) | Step-by-step guide to adding a new model |
| [Adding Quantization](contributing/adding_quantization.md) | Adding a new quantization method |
| [Adding a Platform](contributing/adding_platform.md) | Adding support for a new hardware platform |
| [Incremental Build](contributing/incremental_build.md) | Speeding up builds with incremental compilation |
| [Profiling](contributing/profiling.md) | Profiling vLLM for performance analysis |
| [Deprecation Policy](contributing/deprecation_policy.md) | How deprecations are handled |
| [Vulnerability Management](contributing/vulnerability_management.md) | Security vulnerability reporting |

---

## :material-account-group: Community

| Page | Description |
|---|---|
| [Overview](community/index.md) | Community resources and links |
| [Support](community/support.md) | Getting help and support channels |
| [RFCs](community/rfcs.md) | Request for Comments process |
| [Meetups](community/meetups.md) | Community meetups and events |
| [Sponsors](community/sponsors.md) | Project sponsors |
| [Contact Us](community/contact_us.md) | How to contact the team |

---

## :material-scale-balance: Governance

| Page | Description |
|---|---|
| [Overview](governance/index.md) | Project governance overview |
| [Process](governance/process.md) | Decision-making process |
| [Committers](governance/committers.md) | Committer list and responsibilities |
| [Collaboration](governance/collaboration.md) | Collaboration guidelines |

---

!!! tip
    Use the search bar (press `/` or `S`) to find any topic instantly.
    The [Home page](index.md) also provides a feature matrix and quick navigation table.
