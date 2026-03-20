---
description: >
  Install vLLM for Huawei Ascend NPU — using the vllm-ascend plugin on top
  of the CANN software stack for Ascend 910B and 910C series.
toc_depth: 3
---

# Huawei Ascend NPU Installation

vLLM supports Huawei Ascend NPUs through the **vllm-ascend** hardware plugin.
This plugin lives outside the main vLLM repository and is maintained by
Huawei and the community.

!!! note "Plugin-based architecture"
    Huawei Ascend support uses vLLM's
    [hardware plugin system](../../design/plugin_system.md). The `vllm-ascend`
    plugin is installed on top of a standard vLLM installation and provides
    all Ascend-specific kernels and optimizations.

---

## Requirements

| Requirement | Value |
|---|---|
| **Hardware** | Huawei Ascend 910B or 910C series |
| **Operating System** | Linux (Ubuntu 22.04 or OpenEuler recommended) |
| **Python** | 3.10 – 3.12 |
| **CANN** | Ascend CANN toolkit (latest version recommended) |
| **Ascend Toolkit** | `/usr/local/Ascend/ascend-toolkit/` |
| **ATB** | `/usr/local/Ascend/nnal/atb/` |

!!! tip "Ascend software stack"
    Install the Ascend CANN toolkit from
    [Huawei Ascend documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC3alpha001/softwareinst/instg/instg_0001.html)
    before proceeding.

---

## Option 1 — Docker (Recommended)

The Docker-based installation is the recommended approach for Ascend NPU.

### Build the vLLM Ascend image

The build process fetches the `vllm-ascend` repository and installs it
alongside vLLM:

```bash
# Set the Ascend base image (provided by Huawei)
BASE_IMAGE_NAME="your-ascend-base-image:latest"

docker build \
    --build-arg BASE_IMAGE_NAME="${BASE_IMAGE_NAME}" \
    -t vllm-ascend \
    -f - . <<'EOF'
FROM ${BASE_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION="ascend910b1"

RUN apt-get update -y && \
    apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN pip install pytest>=6.0 modelscope

WORKDIR /workspace/vllm

# Install vLLM common dependencies
COPY requirements/common.txt /workspace/vllm/requirements/common.txt
RUN pip install -r requirements/common.txt

COPY . .

# Install vLLM with empty device target (no GPU/CPU kernels)
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install -v -e /workspace/vllm/ \
    --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton

# Clone and install vllm-ascend
WORKDIR /workspace
ARG VLLM_ASCEND_REPO=https://github.com/vllm-project/vllm-ascend.git
ARG VLLM_ASCEND_TAG=main
RUN git clone --depth 1 ${VLLM_ASCEND_REPO} --branch ${VLLM_ASCEND_TAG} /workspace/vllm-ascend

RUN pip install -r /workspace/vllm-ascend/requirements.txt

RUN export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib && \
    python3 -m pip install -v -e /workspace/vllm-ascend/ \
        --extra-index https://download.pytorch.org/whl/cpu/

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_USE_MODELSCOPE=True

WORKDIR /workspace/vllm-ascend
CMD ["/bin/bash"]
EOF
```

### Run the container

```bash
# Determine your NPU device indices
# Ascend NPU devices are at /dev/davinci0, /dev/davinci1, etc.

docker run \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -e "HF_TOKEN=${HF_TOKEN}" \
    --name vllm-ascend-container \
    vllm-ascend \
    bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    vllm serve Qwen/Qwen3-0.6B --port 8000
    '
```

**Required device mounts:**

| Device / Volume | Purpose |
|---|---|
| `/dev/davinciX` | NPU compute device (one per card) |
| `/dev/davinci_manager` | NPU management interface |
| `/dev/devmm_svm` | Shared virtual memory |
| `/dev/hisi_hdc` | Host-device communication |
| `/usr/local/dcmi` | Device management interface library |
| `/usr/local/bin/npu-smi` | NPU system management interface |
| `/usr/local/Ascend/driver/lib64/` | Ascend driver libraries |

---

## Option 2 — Build from Source

### Prerequisites

Ensure the Ascend CANN toolkit is installed and the environment is set up:

```bash
# Source the Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Verify NPU devices
npu-smi info
```

### Install vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install common dependencies
pip install -r requirements/common.txt

# Install vLLM with empty device target
VLLM_TARGET_DEVICE="empty" python3 -m pip install -v -e . \
    --extra-index https://download.pytorch.org/whl/cpu/

# Remove triton (not needed for Ascend)
python3 -m pip uninstall -y triton
```

### Install the vllm-ascend plugin

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# Install plugin dependencies
pip install -r requirements.txt

# Install the plugin (with Ascend environment sourced)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi

python3 -m pip install -v -e . \
    --extra-index https://download.pytorch.org/whl/cpu/
```

### Verify the installation

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python3 -c "
import vllm
print(f'vLLM {vllm.__version__} installed')
import torch_npu
print(f'NPU available: {torch_npu.npu.is_available()}')
"
```

---

## Running Inference

### Offline inference

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m
```

### Online serving

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve Qwen/Qwen3-0.6B \
    --port 8000 \
    --dtype bfloat16
```

### Using ModelScope (for China mainland users)

```bash
export VLLM_USE_MODELSCOPE=True

vllm serve Qwen/Qwen3-0.6B \
    --port 8000 \
    --dtype bfloat16
```

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `VLLM_TARGET_DEVICE` | Set to `empty` for Ascend builds |
| `VLLM_WORKER_MULTIPROC_METHOD` | Set to `spawn` (required for Ascend) |
| `VLLM_USE_MODELSCOPE` | Set to `True` to use ModelScope instead of HuggingFace |
| `SOC_VERSION` | Ascend SoC version (e.g., `ascend910b1`) |
| `PIP_EXTRA_INDEX_URL` | Huawei Ascend PyPI mirror for Ascend-specific packages |

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for Ascend NPU-specific feature support.

---

## Troubleshooting

??? question "Ascend environment not sourced"
    Always source the Ascend environment before running vLLM:
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

??? question "NPU device not found"
    Verify the NPU driver is installed and devices are accessible:
    ```bash
    npu-smi info
    ls /dev/davinci*
    ```

??? question "Plugin version mismatch"
    Check the [vllm-ascend repository](https://github.com/vllm-project/vllm-ascend)
    for the compatible vLLM version.

??? question "Triton import error"
    Remove triton after installing vLLM:
    ```bash
    python3 -m pip uninstall -y triton
    ```

---

## Community & Support

- **Slack**: `#sig-ascend` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub**: [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- **Huawei Ascend Docs**: [hiascend.com](https://www.hiascend.com/)

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Plugin System](../../design/plugin_system.md)** — How hardware plugins work
