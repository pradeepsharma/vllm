---
hide:
  - navigation
  - toc
description: >
  vLLM — easy, fast, and cheap LLM serving for everyone. High-throughput inference
  engine with PagedAttention, continuous batching, and broad hardware support.
---

# Welcome to vLLM

<figure markdown="span">
  ![vLLM logo (light mode)](./assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM" class="logo-light" width="55%" }
  ![vLLM logo (dark mode)](./assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM" class="logo-dark" width="55%" }
</figure>

<p style="text-align:center; font-size:1.25rem; font-weight:600;">
Easy, fast, and cheap LLM serving for everyone
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star vllm-project/vllm on GitHub">Star</a>
&nbsp;
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch vllm-project/vllm on GitHub">Watch</a>
&nbsp;
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork vllm-project/vllm on GitHub">Fork</a>
</p>

---

vLLM is a fast and easy-to-use library for LLM inference and serving. Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into a thriving community-driven project with contributions from academia and industry worldwide.

---

## :rocket: Key features

<div class="grid cards" markdown>

-   :zap: **State-of-the-art throughput**

    ---

    Achieve industry-leading serving throughput with [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html), continuous batching, and CUDA/HIP graph execution — all enabled by default.

    [:octicons-arrow-right-24: Performance overview](getting_started/index.md)

-   :brain: **Broad model support**

    ---

    Run Transformer LLMs, Mixture-of-Expert models, multimodal models, and embedding models. Seamlessly load weights from Hugging Face or ModelScope.

    [:octicons-arrow-right-24: Supported models](models/supported_models.md)

-   :globe_with_meridians: **OpenAI-compatible API**

    ---

    Drop-in replacement for OpenAI's API. Serve chat completions, completions, and embeddings with zero application-level changes.

    [:octicons-arrow-right-24: OpenAI-compatible server](serving/openai_compatible_server.md)

-   :bar_chart: **Flexible quantization**

    ---

    Reduce memory footprint and increase throughput with GPTQ, AWQ, INT4, INT8, and FP8 quantization — no accuracy sacrifice required.

    [:octicons-arrow-right-24: Quantization guide](features/quantization/index.md)

-   :arrows_counterclockwise: **Distributed inference**

    ---

    Scale across multiple GPUs and nodes with tensor, pipeline, data, and expert parallelism. Supports disaggregated prefill for maximum efficiency.

    [:octicons-arrow-right-24: Parallelism & scaling](serving/parallelism_scaling.md)

-   :electric_plug: **Wide hardware support**

    ---

    Runs on NVIDIA CUDA, AMD ROCm, Intel XPU, Apple Silicon, ARM, PowerPC, IBM Z, and Google TPU — plus third-party hardware plugins.

    [:octicons-arrow-right-24: Installation guide](getting_started/installation/index.md)

</div>

---

## :compass: Where to start

Your path through the documentation depends on what you want to do:

| I want to… | Start here |
|---|---|
| **Run a model right now** | [Quickstart](getting_started/quickstart.md) |
| **Choose the right installation method** | [Getting Started Overview](getting_started/index.md) |
| **Serve an OpenAI-compatible API** | [OpenAI-Compatible Server](serving/openai_compatible_server.md) |
| **Run offline batch inference** | [Offline Inference](usage/offline_inference.md) |
| **Deploy to production (Docker / K8s)** | [Deployment](deployment/index.md) |
| **Understand vLLM's architecture** | [Architecture Overview](design/arch_overview.md) |
| **Contribute code or models** | [Contributing Guide](contributing/index.md) |
| **Browse all documentation** | [Sitemap](sitemap.md) |

---

## :sparkles: What's new

- **vLLM V1 Engine** — A fully re-architected core with near-zero CPU overhead, unified optimizations, and zero-config defaults. [Read the guide →](usage/v1_guide.md)
- **Disaggregated Prefill** — Separate prefill and decode phases across nodes for maximum throughput. [Learn more →](features/disagg_prefill.md)
- **Speculative Decoding** — Accelerate generation with EAGLE, Medusa, n-gram, and draft models. [Learn more →](features/speculative_decoding/README.md)
- **Structured Outputs** — Constrained generation with JSON schemas, regex, and grammars. [Learn more →](features/structured_outputs.md)
- **Multi-LoRA Serving** — Serve hundreds of LoRA adapters from a single base model. [Learn more →](features/lora.md)
- **Elastic Expert Parallelism** — Dynamically scale MoE expert parallelism at runtime. [Learn more →](deployment/elastic_ep.md)

---

## :white_check_mark: Feature matrix

The table below summarizes vLLM's major capabilities and where to find documentation for each.

### Inference modes

| Feature | Status | Documentation |
|---|---|---|
| Offline batch inference | ✅ Stable | [Offline Inference](usage/offline_inference.md) |
| Online serving (OpenAI API) | ✅ Stable | [OpenAI-Compatible Server](serving/openai_compatible_server.md) |
| Chat completions | ✅ Stable | [Chat Completions](serving/chat_completions.md) |
| Text completions | ✅ Stable | [Completions](serving/completions.md) |
| Embeddings | ✅ Stable | [Embeddings](serving/embeddings.md) |
| Responses API | ✅ Stable | [Responses API](serving/responses_api.md) |
| Realtime API (WebSocket) | ✅ Stable | [Realtime API](serving/realtime_api.md) |
| Speech-to-text | ✅ Stable | [Speech to Text](serving/speech_to_text.md) |
| Anthropic API compatibility | ✅ Stable | [Anthropic API](serving/anthropic_api.md) |
| gRPC server | ✅ Stable | [gRPC Server](serving/grpc_server.md) |
| Batch inference (offline) | ✅ Stable | [Batch Inference](serving/batch_inference.md) |
| Streaming responses | ✅ Stable | [Streaming](serving/streaming.md) |

### Model types

| Model Type | Status | Documentation |
|---|---|---|
| Generative (decoder-only) LLMs | ✅ Stable | [Generative Models](models/generative_models.md) |
| Embedding models | ✅ Stable | [Embedding Models](models/embedding_models.md) |
| Pooling / reranking models | ✅ Stable | [Pooling Models](models/pooling_models.md) |
| Multimodal models (vision-language) | ✅ Stable | [Multimodal Models](models/multimodal_models.md) |
| Encoder-decoder models | ✅ Stable | [Encoder-Decoder Models](models/encoder_decoder_models.md) |
| Mixture-of-Experts (MoE) | ✅ Stable | [Supported Models](models/supported_models.md) |

### Quantization

| Method | Bits | Status | Documentation |
|---|---|---|---|
| GPTQ | 4-bit / 8-bit | ✅ Stable | [GPTQ](features/quantization/gptq.md) |
| AWQ | 4-bit | ✅ Stable | [AWQ](features/quantization/awq.md) |
| FP8 (W8A8) | 8-bit float | ✅ Stable | [FP8](features/quantization/fp8.md) |
| INT8 (SmoothQuant) | 8-bit int | ✅ Stable | [INT8](features/quantization/int8.md) |
| INT4 | 4-bit int | ✅ Stable | [INT4](features/quantization/int4.md) |
| GGUF | Variable | ✅ Stable | [GGUF](features/quantization/gguf.md) |
| BitsAndBytes | 4-bit / 8-bit | ✅ Stable | [BitsAndBytes](features/quantization/bitsandbytes.md) |
| TorchAO | Variable | ✅ Stable | [TorchAO](features/quantization/torchao.md) |
| MXFP4 | 4-bit MX | ✅ Stable | [MXFP4](features/quantization/mxfp4.md) |
| KV cache quantization | FP8 / INT8 | ✅ Stable | [KV Cache Quant](features/quantization/kv_cache_quantization.md) |

### Parallelism & scaling

| Strategy | Status | Documentation |
|---|---|---|
| Tensor parallelism (TP) | ✅ Stable | [Distributed Inference](design/distributed_inference.md) |
| Pipeline parallelism (PP) | ✅ Stable | [Distributed Inference](design/distributed_inference.md) |
| Data parallelism (DP) | ✅ Stable | [Data Parallel Deployment](serving/data_parallel_deployment.md) |
| Expert parallelism (EP) | ✅ Stable | [Expert Parallel Deployment](serving/expert_parallel_deployment.md) |
| Context parallelism (CP) | ✅ Stable | [Context Parallel Deployment](serving/context_parallel_deployment.md) |
| Multi-node (Ray) | ✅ Stable | [Ray Cluster](deployment/ray_cluster.md) |
| Multi-node (Docker) | ✅ Stable | [Multi-Node](deployment/multi_node.md) |
| Disaggregated prefill | ✅ Stable | [Disaggregated Prefill](features/disagg_prefill.md) |
| Elastic expert parallelism | ✅ Stable | [Elastic EP](deployment/elastic_ep.md) |

### Speculative decoding

| Method | Status | Documentation |
|---|---|---|
| EAGLE / EAGLE-2 | ✅ Stable | [EAGLE](features/speculative_decoding/eagle.md) |
| Medusa | ✅ Stable | [Medusa](features/speculative_decoding/medusa.md) |
| N-gram matching | ✅ Stable | [N-gram](features/speculative_decoding/ngram.md) |
| Draft model | ✅ Stable | [Draft Model](features/speculative_decoding/draft_model.md) |
| MLP Speculator | ✅ Stable | [MLP Speculator](features/speculative_decoding/mlp.md) |
| MTP (Multi-Token Prediction) | ✅ Stable | [MTP](features/speculative_decoding/mtp.md) |
| Suffix decoding | ✅ Stable | [Suffix Decoding](features/speculative_decoding/suffix.md) |

### Hardware platforms

| Platform | Status | Documentation |
|---|---|---|
| NVIDIA CUDA (A100, H100, H200…) | ✅ Stable | [CUDA Installation](getting_started/installation/gpu-cuda.md) |
| AMD ROCm (MI250, MI300…) | ✅ Stable | [ROCm Installation](getting_started/installation/gpu-rocm.md) |
| Intel XPU (Gaudi, Arc…) | ✅ Stable | [XPU Installation](getting_started/installation/xpu.md) |
| Intel Gaudi (HPU) | ✅ Stable | [HPU Installation](getting_started/installation/hpu.md) |
| CPU (x86, ARM, s390x) | ✅ Stable | [CPU Installation](getting_started/installation/cpu.md) |
| Google TPU | ✅ Stable | [TPU Installation](getting_started/installation/tpu.md) |
| Ascend NPU | ✅ Stable | [NPU Installation](getting_started/installation/npu.md) |
| Apple macOS (Metal) | ✅ Stable | [macOS Installation](getting_started/installation/macos.md) |

---

## :books: Documentation sections

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Installation guides for every platform, quickstart tutorial, and a decision tree to choose your path.

    [:octicons-arrow-right-24: Getting Started](getting_started/index.md)

-   :material-server: **Serving**

    ---

    OpenAI-compatible server, offline inference, distributed deployment, and parallelism strategies.

    [:octicons-arrow-right-24: Serving](serving/index.md)

-   :material-tune: **Configuration**

    ---

    Engine arguments, environment variables, optimization settings, and memory management.

    [:octicons-arrow-right-24: Configuration](configuration/index.md)

-   :material-puzzle: **Features**

    ---

    LoRA, quantization, speculative decoding, multimodal inputs, structured outputs, prefix caching, and more.

    [:octicons-arrow-right-24: Features](features/quantization/index.md)

-   :material-chip: **Models**

    ---

    Supported model architectures, generative and pooling models, and hardware compatibility matrices.

    [:octicons-arrow-right-24: Models](models/index.md)

-   :material-cloud-upload: **Deployment**

    ---

    Docker, Kubernetes, Helm, cloud platforms, and integration with popular frameworks.

    [:octicons-arrow-right-24: Deployment](deployment/index.md)

-   :material-pencil: **Contributing**

    ---

    Developer guide, CI/CD, model contribution, profiling, and governance.

    [:octicons-arrow-right-24: Contributing](contributing/index.md)

-   :material-flask: **Benchmarking**

    ---

    Benchmark CLI, performance dashboards, and sweep configurations.

    [:octicons-arrow-right-24: Benchmarking](benchmarking/index.md)

-   :material-book-open-variant: **API Reference**

    ---

    Python API for `LLM`, `AsyncLLMEngine`, `SamplingParams`, and all output types.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-school: **Examples**

    ---

    Runnable code examples for offline inference, online serving, pooling, and more.

    [:octicons-arrow-right-24: Examples](examples/index.md)

-   :material-dumbbell: **Training Integration**

    ---

    Weight transfer API, RLHF integration, and TRL compatibility.

    [:octicons-arrow-right-24: Training](training/index.md)

-   :material-map: **Sitemap**

    ---

    Complete index of all documentation pages organized by section.

    [:octicons-arrow-right-24: Sitemap](sitemap.md)

</div>

---

## :handshake: Community & resources

<div class="grid cards" markdown>

-   :fontawesome-brands-github: **GitHub**

    ---

    Report bugs, request features, and browse the source code.

    [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

-   :fontawesome-brands-slack: **Developer Slack**

    ---

    Chat with the core team and contributors in real time.

    [slack.vllm.ai](https://slack.vllm.ai)

-   :material-forum: **User Forum**

    ---

    Ask questions and share knowledge with the broader vLLM community.

    [discuss.vllm.ai](https://discuss.vllm.ai)

-   :fontawesome-brands-x-twitter: **Twitter / X**

    ---

    Follow [@vllm_project](https://x.com/vllm_project) for announcements and updates.

    [x.com/vllm_project](https://x.com/vllm_project)

-   :material-calendar: **Events & Meetups**

    ---

    Join in-person and virtual meetups around the world.

    [vllm.ai/events](https://vllm.ai/events)

-   :material-newspaper: **Blog**

    ---

    Deep dives, release notes, and research from the vLLM team.

    [blog.vllm.ai](https://blog.vllm.ai)

</div>

---

## :scroll: Citation

If you use vLLM in your research, please cite the original paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng
          and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

---

<p style="text-align:center; color: var(--md-default-fg-color--light); font-size:0.85rem;">
vLLM is an open-source project released under the Apache 2.0 License.<br>
Compute resources for development and testing are generously provided by our <a href="community/sponsors.md">sponsors</a>.
</p>
