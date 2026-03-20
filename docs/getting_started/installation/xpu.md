---
description: >
  Install vLLM for Intel GPU (XPU) — Intel Data Center GPU and Intel Arc GPU
  with oneAPI 2025.3 and PyTorch XPU backend.
toc_depth: 3
---

# Intel GPU (XPU) Installation

vLLM supports Intel GPUs through the Intel oneAPI software stack and the
PyTorch XPU backend. This enables inference on Intel Data Center GPUs
(Ponte Vecchio, Battlemage) and Intel Arc consumer GPUs.

---

## Requirements

| Requirement | Value |
|---|---|
| **Operating System** | Linux (Ubuntu 24.04 recommended) |
| **Python** | 3.12 (required — pre-built kernels are Python 3.12 specific) |
| **Intel GPU Driver** | Latest [Intel GPU driver](https://dgpu-docs.intel.com/driver/installation.html) |
| **Intel oneAPI** | oneAPI Base Toolkit 2025.3 or later |
| **oneCCL** | 2021.15 or later (for distributed inference) |

**Supported hardware:**

| Hardware | Notes |
|---|---|
| Intel Data Center GPU Max Series (Ponte Vecchio) | Full support |
| Intel Arc GPU (Battlemage / BMG) | Requires oneCCL 2021.15+ |
| Intel Data Center GPU Flex Series | Basic support |

!!! warning "Python 3.12 required"
    The pre-built `vllm-xpu-kernels` wheel is compiled for Python 3.12 only.
    Other Python versions require building the kernels from source.

---

## Option 1 — Docker (Recommended)

The Docker image is the easiest way to get started with Intel XPU support,
as it bundles all drivers, oneAPI, and dependencies.

### Pull the pre-built image

Intel publishes pre-built vLLM XPU images on Docker Hub:

```bash
docker pull intel/vllm:latest
```

All available tags: [hub.docker.com/r/intel/vllm/tags](https://hub.docker.com/r/intel/vllm/tags)

For release notes and version details, see the
[Intel AI Containers repository](https://github.com/intel/ai-containers/blob/main/vllm).

### Run the container

```bash
docker run -it \
    --rm \
    --network=host \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host \
    --privileged \
    -e "HF_TOKEN=${HF_TOKEN}" \
    intel/vllm:latest \
    --model facebook/opt-125m
```

| Flag | Purpose |
|---|---|
| `--device /dev/dri:/dev/dri` | Expose Intel GPU devices |
| `-v /dev/dri/by-path:/dev/dri/by-path` | GPU device path mapping |
| `--ipc=host` | Shared memory for multi-GPU setups |
| `--privileged` | Required for full GPU access |

### Build a custom Docker image

```bash
docker build -f docker/Dockerfile.xpu \
    -t vllm-xpu-env \
    --shm-size=4g \
    .
```

The Dockerfile installs:

- Intel oneAPI DPC++ compiler 2025.3
- Intel GPU UMD (User Mode Driver) from GitHub releases
- oneCCL 2021.15 (for distributed inference)
- PyTorch XPU backend
- `vllm-xpu-kernels` (custom XPU kernels)
- NIXL and UCX (for KV cache transfer)

```bash
docker run -it \
    --rm \
    --network=host \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host \
    --privileged \
    vllm-xpu-env
```

---

## Option 2 — Build from Source

### Step 1 — Install Intel GPU driver

Follow the [Intel GPU driver installation guide](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers).

### Step 2 — Install Intel oneAPI

Download and install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
version 2025.3 or later.

```bash
# Add Intel oneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
    https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install -y intel-oneapi-compiler-dpcpp-cpp-2025.3
```

### Step 3 — Install oneCCL (for distributed inference)

```bash
# Download and install oneCCL 2021.15
ONECCL_INSTALLER="intel-oneccl-2021.15.7.8_offline.sh"
wget "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.7/${ONECCL_INSTALLER}"
bash "${ONECCL_INSTALLER}" -a --silent --eula accept
rm "${ONECCL_INSTALLER}"

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force
```

### Step 4 — Create a Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Step 5 — Install vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install XPU requirements (includes PyTorch XPU and vllm-xpu-kernels)
uv pip install --upgrade pip
uv pip install -r requirements/xpu.txt

# Build and install vLLM for XPU
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CMAKE_PREFIX_PATH="$(python -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH}"

uv pip install --no-build-isolation .
```

**XPU requirements include:**

| Package | Purpose |
|---|---|
| `torch==2.10.0+xpu` | PyTorch with XPU backend |
| `torchaudio`, `torchvision` | Audio/vision processing |
| `vllm_xpu_kernels` | Custom XPU kernels (pre-built wheel) |
| `ray>=2.9` | Distributed execution |
| `numba` | N-gram speculative decoding |

---

## Verify the Installation

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

# Check XPU device
python3 -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
python3 -c "import torch; print(f'XPU device count: {torch.xpu.device_count()}')"

# Run a basic inference test
python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m \
    --block-size 64 \
    --enforce-eager
```

---

## Running Inference

### Offline inference

```bash
source /opt/intel/oneapi/setvars.sh --force

python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m \
    --block-size 64 \
    --enforce-eager
```

### Online serving

```bash
source /opt/intel/oneapi/setvars.sh --force

vllm serve facebook/opt-13b \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --port 8000
```

### Tensor parallel (multi-GPU)

```bash
source /opt/intel/oneapi/setvars.sh --force

# Tensor parallel with Ray backend
python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m \
    --block-size 64 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray

# Tensor parallel with multiprocessing backend
python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m \
    --block-size 64 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp
```

### Pipeline parallel (online serving)

```bash
vllm serve facebook/opt-13b \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --distributed-executor-backend mp \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 8
```

---

## Distributed Backend

| PyTorch Version | Distributed Backend |
|---|---|
| < 2.8 | `torch-ccl` (Intel oneCCL) |
| ≥ 2.8 | `xccl` (built-in XPU backend) |

By default, a Ray instance is launched automatically if none is detected.
For multi-node setups, start a Ray cluster manually:

```bash
# See examples/online_serving/run_cluster.sh
bash examples/online_serving/run_cluster.sh
```

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `VLLM_TARGET_DEVICE` | Set to `xpu` for XPU builds |
| `VLLM_WORKER_MULTIPROC_METHOD` | Set to `spawn` (required for XPU) |
| `ZE_AFFINITY_MASK` | Control which XPU devices are visible (e.g., `0,1`) |

---

## Supported Features

XPU platform supports:

- **Tensor parallel** inference and serving
- **Pipeline parallel** (beta, single-node with `mp` backend)
- **FP8 quantization**
- **GPTQ quantization**
- **Triton attention backend** (`--attention-backend=TRITON_ATTN`)

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for the full list.

---

## Troubleshooting

??? question "oneAPI environment not sourced"
    Always source the oneAPI environment before running vLLM:
    ```bash
    source /opt/intel/oneapi/setvars.sh --force
    source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force
    ```

??? question "Triton version conflict"
    The XPU build requires `triton-xpu`. If you see Triton errors:
    ```bash
    pip uninstall triton triton-xpu
    pip install triton-xpu==3.6.0
    ```

??? question "oneCCL conflict with torch bundled version"
    Remove the torch-bundled oneCCL to avoid conflicts:
    ```bash
    pip uninstall oneccl oneccl-devel
    ```

??? question "GPU not detected"
    Verify the Intel GPU driver is installed:
    ```bash
    clinfo | grep "Platform Name"
    ```

---

## Community & Support

- **Slack**: `#sig-xpu` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub Issues**: Add `[XPU]` to the issue title

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Parallelism & Scaling](../../serving/parallelism_scaling.md)** — Multi-GPU tensor parallel
