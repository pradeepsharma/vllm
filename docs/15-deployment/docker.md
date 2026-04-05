# Docker Deployment

vLLM ships official Docker images that bundle the OpenAI-compatible server, all CUDA dependencies, and the vLLM Python package. This page covers pulling and running those images, passing environment variables, and enabling GPU access.

## Official Images

| Image Tag | Description |
|-----------|-------------|
| `vllm/vllm-openai:latest` | Latest stable release — OpenAI-compatible server |
| `vllm/vllm-openai:<version>` | Pinned release (e.g., `v0.8.0`) |
| `vllm/vllm-openai:nightly` | Nightly build with latest PyTorch |

The images are built from `docker/Dockerfile` using a multi-stage build. The final image is based on `nvidia/cuda:<version>-base-ubuntu22.04` and includes only the runtime CUDA libraries needed for JIT compilation (FlashInfer, DeepGEMM, EP kernels).

## Prerequisites

- **NVIDIA Container Toolkit** installed on the host (`nvidia-container-toolkit`)
- Docker Engine ≥ 20.10
- NVIDIA driver compatible with the CUDA version in the image (CUDA 12.9 by default)

## Basic `docker run`

### Single GPU

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Key flags:
- `--runtime nvidia --gpus all` — expose all host GPUs to the container
- `-v ~/.cache/huggingface:/root/.cache/huggingface` — mount the HuggingFace model cache to avoid re-downloading
- `--ipc=host` — required for shared memory used by PyTorch multiprocessing
- `-p 8000:8000` — expose the API server port

### Specific GPU Selection

```bash
docker run --runtime nvidia --gpus '"device=0,1"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2
```

Or use `CUDA_VISIBLE_DEVICES` inside the container:

```bash
docker run --runtime nvidia --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2
```

### With HuggingFace Token (Gated Models)

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=hf_your_token_here \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Environment Variables

The following environment variables are recognized by vLLM and can be passed via `-e` flags:

### Networking

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST_IP` | `""` | IP address for inter-process communication (required for multi-node) |
| `VLLM_PORT` | `None` | Override the internal RPC port |

### Model & Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace Hub token for gated models |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Root directory for vLLM caches |
| `VLLM_XLA_CACHE_PATH` | `~/.cache/vllm/xla_cache` | XLA compilation cache (TPU) |

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | Worker spawn method (`fork` or `spawn`) |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | `16384` | Chunk size for fused MoE kernels |
| `VLLM_USE_DEEP_GEMM` | `1` | Enable DeepGEMM for FP8 matmul |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `1` | Enable V1 multiprocessing engine |

### CUDA Compatibility

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | Enable CUDA forward compatibility for older drivers |
| `CUDA_VISIBLE_DEVICES` | — | Restrict visible GPUs |
| `TORCH_CUDA_ARCH_LIST` | `7.0 7.5 8.0 8.9 9.0 10.0 12.0` | CUDA architectures for compilation |

### Logging & Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_LOGGING_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Seconds between stats log lines |
| `VLLM_USAGE_SOURCE` | `production` | Usage stats source tag |

> **Note:** The Dockerfile sets `VLLM_USAGE_SOURCE=production-docker-image` automatically for the `vllm-openai` image target.

## GPU Passthrough Details

### NVIDIA Runtime

The `--runtime nvidia` flag (or `--gpus` with Docker ≥ 19.03) invokes the NVIDIA Container Runtime, which:
1. Mounts NVIDIA device files (`/dev/nvidia*`) into the container
2. Injects CUDA libraries from the host driver
3. Sets `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`

### GPU Memory Sharing

For large models that require shared memory between GPU processes:

```bash
docker run --runtime nvidia --gpus all \
  --shm-size 16g \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4
```

Use `--shm-size` when `--ipc=host` is not appropriate (e.g., in Kubernetes pods).

### CUDA Forward Compatibility

For datacenter GPUs with older drivers, enable CUDA forward compatibility:

```bash
docker run --runtime nvidia --gpus all \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Docker Compose Example

```yaml
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - VLLM_LOGGING_LEVEL=INFO
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command:
      - --model
      - meta-llama/Meta-Llama-3.1-8B-Instruct
      - --gpu-memory-utilization
      - "0.9"
```

## SageMaker Image Variant

The Dockerfile also builds a `vllm-sagemaker` target that wraps the server with the SageMaker entrypoint script:

```dockerfile
FROM vllm-openai-base AS vllm-sagemaker
COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]
```

See [AWS SageMaker Deployment](sagemaker.md) for details.

## KV Connector Dependencies

To include optional KV connector libraries (e.g., LMCache, NIXL) in the image:

```bash
docker build \
  --build-arg INSTALL_KV_CONNECTORS=true \
  -f docker/Dockerfile \
  -t vllm/vllm-openai:kv-connectors \
  .
```

## Related Pages

- [Multi-Stage Docker Builds](docker-bake.md) — `docker-bake.hcl` and image variants
- [Multi-Node Deployment](multi-node.md) — Ray cluster with Docker
- [AWS SageMaker](sagemaker.md) — SageMaker-specific image and entrypoint
- [SSL/TLS Configuration](ssl-tls.md) — HTTPS for the API server
