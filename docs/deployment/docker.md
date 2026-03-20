---
description: >
  Complete guide to deploying vLLM with Docker — pre-built images, building
  from source, multi-platform builds with Bake, and SageMaker variants.
---

# Docker deployment

vLLM ships official Docker images that are ready to run the OpenAI-compatible
API server. This page covers every image variant, how to build your own, and
how to customise the build for private registries or air-gapped environments.

---

## Quick start

Pull the latest production image and serve a model in one command:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.3
```

The server is ready when you see:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Official image variants

| Image tag | Base | Purpose |
|---|---|---|
| `vllm/vllm-openai:latest` | CUDA 12.9 + Ubuntu 22.04 | Production OpenAI-compatible server |
| `vllm/vllm-openai:<version>` | CUDA 12.9 + Ubuntu 22.04 | Pinned release (recommended for production) |
| `vllm-sagemaker` | Same as above | AWS SageMaker endpoint |
| `vllm:test` | Build stage | CI / unit-test image |

!!! tip
    Always pin a specific version tag in production. Using `latest` can cause
    unexpected behaviour when a new release changes defaults.

---

## Pre-built images

Pre-built images are published to Docker Hub at
[hub.docker.com/r/vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai).

### Pulling a specific version

```bash
docker pull vllm/vllm-openai:v0.9.0
```

### Running with a Hugging Face token

For gated models (e.g., Llama 3), pass your token as an environment variable:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3-8B-Instruct
```

### GPU selection

Use `--gpus` to control which GPUs are visible to the container:

```bash
# All GPUs
docker run --runtime nvidia --gpus all ...

# Specific GPU by index
docker run --runtime nvidia --gpus device=0 ...

# Multiple specific GPUs
docker run --runtime nvidia --gpus '"device=0,1"' ...
```

### Shared memory for tensor parallelism

When using tensor parallelism across multiple GPUs, increase the shared memory
limit so PyTorch NCCL can communicate:

```bash
docker run --runtime nvidia --gpus all \
    --shm-size=10g \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4
```

---

## Build image from source

Building from source lets you include local code changes or customise the
image for your environment.

### Standard build

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Build the production OpenAI image
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    -t vllm:local \
    .
```

### Build arguments reference

The Dockerfile exposes many `ARG` values to customise the build. The most
commonly used ones are listed below.

#### Version pins

| Argument | Default | Description |
|---|---|---|
| `CUDA_VERSION` | `12.9.1` | CUDA toolkit version |
| `PYTHON_VERSION` | `3.12` | Python version |
| `FLASHINFER_VERSION` | `0.6.4` | FlashInfer kernel version |
| `BITSANDBYTES_VERSION_X86` | `0.46.1` | BitsAndBytes for x86-64 |
| `BITSANDBYTES_VERSION_ARM64` | `0.42.0` | BitsAndBytes for ARM64 |

#### Base images

| Argument | Default | Description |
|---|---|---|
| `BUILD_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04` | Build stage base |
| `FINAL_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04` | Runtime base |

!!! note
    The build stage uses Ubuntu 20.04 to maximise glibc compatibility with
    older Linux distributions. The final runtime image uses Ubuntu 22.04.

#### Private registry support

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:12.9.1-devel-ubuntu20.04 \
    --build-arg FINAL_BASE_IMAGE=registry.acme.org/mirror/nvidia/cuda:12.9.1-base-ubuntu22.04 \
    -t vllm:local \
    .
```

#### Private Python package indexes

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg PIP_INDEX_URL=https://pypi.acme.org/simple \
    --build-arg PIP_EXTRA_INDEX_URL=https://pypi.acme.org/simple \
    -t vllm:local \
    .
```

#### PyTorch nightly builds

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg PYTORCH_NIGHTLY=1 \
    -t vllm:nightly \
    .
```

#### CUDA architecture list

Control which GPU architectures are compiled into the CUDA kernels:

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg torch_cuda_arch_list="8.0 8.9 9.0" \
    -t vllm:sm90 \
    .
```

The default list is `7.0 7.5 8.0 8.9 9.0 10.0 12.0`.

#### Build parallelism

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg max_jobs=8 \
    --build-arg nvcc_threads=4 \
    -t vllm:local \
    .
```

#### KV connector support

To include optional KV connector libraries (e.g., LMCache):

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg INSTALL_KV_CONNECTORS=true \
    -t vllm:kv-connectors \
    .
```

#### Proxy support

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-openai \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    -t vllm:local \
    .
```

---

## Build stages

The Dockerfile uses a multi-stage build to maximise layer caching and
parallelism. Understanding the stages helps when debugging build failures.

```
base              ← System deps + Python venv + PyTorch (~2 GB, rarely changes)
├── csrc-build    ← Compiles C++/CUDA extensions (.so files)
├── extensions-build ← Builds DeepGEMM + DeepEP wheels (runs in parallel)
├── build         ← Assembles the final vLLM wheel
├── dev           ← Development image with test/lint deps
└── vllm-base     ← Runtime image (Ubuntu 22.04 + Python + CUDA runtime)
    ├── test      ← CI test image
    ├── vllm-openai-base ← OpenAI server deps
    │   ├── vllm-openai  ← Production image (ENTRYPOINT: vllm serve)
    │   └── vllm-sagemaker ← SageMaker image
```

### Targeting a specific stage

```bash
# Build only the development image
docker build -f docker/Dockerfile --target dev -t vllm:dev .

# Build the test image
docker build -f docker/Dockerfile --target test -t vllm:test .
```

---

## Multi-platform builds with Docker Bake

`docker/docker-bake.hcl` defines a declarative build configuration for
multi-platform and multi-target builds using `docker buildx bake`.

### Default build (openai target)

```bash
cd docker
docker buildx bake -f docker-bake.hcl -f versions.json
```

### Build the test image

```bash
docker buildx bake -f docker/docker-bake.hcl test
```

### Preview the resolved configuration

```bash
docker buildx bake --print
```

### Bake variables

| Variable | Default | Description |
|---|---|---|
| `MAX_JOBS` | `16` | Parallel compilation jobs |
| `NVCC_THREADS` | `8` | NVCC compiler threads |
| `TORCH_CUDA_ARCH_LIST` | `8.0 8.9 9.0 10.0` | GPU architectures |
| `COMMIT` | `""` | Git commit SHA for OCI labels |

Override variables on the command line:

```bash
docker buildx bake \
    --set "*.args.MAX_JOBS=32" \
    --set "*.args.TORCH_CUDA_ARCH_LIST=9.0 10.0" \
    -f docker/docker-bake.hcl \
    -f docker/versions.json
```

### Version management

`docker/versions.json` is auto-generated from the `ARG` defaults in the
Dockerfile. Do not edit it manually. To regenerate after changing `ARG`
defaults:

```bash
python tools/generate_versions_json.py
```

Query a specific version:

```bash
jq -r '.variable.CUDA_VERSION.default' docker/versions.json
# 12.9.1
```

---

## AWS SageMaker image

The `vllm-sagemaker` target wraps the OpenAI server with a SageMaker-compatible
entrypoint script.

```bash
docker build \
    -f docker/Dockerfile \
    --target vllm-sagemaker \
    -t vllm:sagemaker \
    .
```

The SageMaker image uses `sagemaker-entrypoint.sh` as its `ENTRYPOINT`, which
translates SageMaker environment variables into `vllm serve` arguments.

---

## CUDA forward compatibility

On datacenter GPUs with older drivers, enable CUDA forward compatibility:

```bash
docker run --runtime nvidia --gpus all \
    -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.3
```

See the [NVIDIA CUDA Compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/)
for details on supported driver/toolkit combinations.

---

## Environment variables

Key environment variables set in the production image:

| Variable | Default | Description |
|---|---|---|
| `VLLM_USAGE_SOURCE` | `production-docker-image` | Telemetry source tag |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | Enable CUDA forward compat |
| `UV_HTTP_TIMEOUT` | `500` | uv package manager timeout (s) |
| `UV_INDEX_STRATEGY` | `unsafe-best-match` | uv index resolution strategy |
| `LD_LIBRARY_PATH` | `/usr/local/nvidia/lib64:...` | CUDA library path |

---

## Troubleshooting

### Out-of-memory during build

Reduce parallel jobs:

```bash
docker build \
    -f docker/Dockerfile \
    --build-arg max_jobs=2 \
    --build-arg nvcc_threads=2 \
    --target vllm-openai \
    -t vllm:local \
    .
```

### Slow builds

Use `sccache` with an S3 bucket for distributed compilation caching:

```bash
docker build \
    -f docker/Dockerfile \
    --build-arg USE_SCCACHE=1 \
    --build-arg SCCACHE_BUCKET_NAME=my-sccache-bucket \
    --build-arg SCCACHE_REGION_NAME=us-east-1 \
    --target vllm-openai \
    -t vllm:local \
    .
```

### Wheel size check failure

The build checks that the vLLM wheel does not exceed 500 MB. To skip this
check during development:

```bash
docker build \
    -f docker/Dockerfile \
    --build-arg RUN_WHEEL_CHECK=false \
    --target vllm-openai \
    -t vllm:local \
    .
```

---

## Next steps

- [Kubernetes deployment](kubernetes.md) — deploy vLLM on Kubernetes with Helm
- [SSL/TLS setup](ssl_tls.md) — enable HTTPS for the API server
- [Load balancing](load_balancing.md) — distribute traffic across multiple instances
- [Production checklist](production_checklist.md) — harden your deployment
