---
description: >
  Install vLLM for Intel Gaudi HPU (Habana Processing Unit) — using the
  vllm-gaudi plugin on top of the Gaudi software stack.
toc_depth: 3
---

# Intel Gaudi HPU Installation

vLLM supports Intel Gaudi HPUs (Habana Processing Units) through the
**vllm-gaudi** hardware plugin. This plugin lives outside the main vLLM
repository and is maintained by Intel/Habana.

!!! note "Plugin-based architecture"
    Intel Gaudi support uses vLLM's
    [hardware plugin system](../../design/plugin_system.md). The `vllm-gaudi`
    plugin is installed on top of a standard vLLM installation and provides
    all Gaudi-specific kernels and optimizations.

---

## Requirements

| Requirement | Value |
|---|---|
| **Hardware** | Intel Gaudi 2 or Gaudi 3 accelerator |
| **Operating System** | Linux (Ubuntu 22.04 recommended) |
| **Python** | 3.10 – 3.12 |
| **Habana Software Stack** | SynapseAI 1.19+ |
| **Docker Runtime** | `habana` runtime (for Docker-based installs) |

!!! tip "Gaudi software stack"
    Install the Habana software stack from the
    [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
    before proceeding.

---

## Option 1 — Docker (Recommended)

The Docker-based installation is the recommended approach for Gaudi, as it
bundles the correct SynapseAI version and all dependencies.

### Pull the Gaudi base image

```bash
docker pull gaudi-base-image:latest
```

!!! note
    The `gaudi-base-image` is provided by Intel/Habana. See the
    [Intel Gaudi Docker Hub](https://hub.docker.com/u/vault) for available tags.

### Build the vLLM Gaudi image

```bash
cat <<'EOF' | docker build -t vllm-gaudi -f - .
FROM gaudi-base-image:latest

COPY ./ /workspace/vllm
WORKDIR /workspace/vllm

ENV no_proxy=localhost,127.0.0.1
ENV PT_HPU_ENABLE_LAZY_COLLECTIVES=true

# Install build dependencies (excluding torch, which comes from the base image)
RUN bash -c 'pip install -r <(sed "/^torch/d" requirements/build.txt)'

# Install vLLM in "empty" device mode (no GPU/CPU kernels)
RUN VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .

# Install the vllm-gaudi plugin
RUN pip install git+https://github.com/vllm-project/vllm-gaudi.git

# Install test utilities
RUN python3 -m pip install -e tests/vllm_test_utils
EOF
```

### Run the container

```bash
docker run --rm \
    --runtime=habana \
    --network=host \
    -e HABANA_VISIBLE_DEVICES=all \
    -e VLLM_SKIP_WARMUP=true \
    -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
    -e PT_HPU_LAZY_MODE=1 \
    -e "HF_TOKEN=${HF_TOKEN}" \
    vllm-gaudi \
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

| Flag | Purpose |
|---|---|
| `--runtime=habana` | Use the Habana container runtime |
| `HABANA_VISIBLE_DEVICES=all` | Expose all Gaudi devices |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` | Enable lazy collective operations |
| `PT_HPU_LAZY_MODE=1` | Enable Gaudi lazy execution mode |
| `VLLM_SKIP_WARMUP=true` | Skip warmup for faster startup (testing only) |

---

## Option 2 — Build from Source

### Prerequisites

Ensure the Habana software stack is installed on your system:

```bash
# Verify Gaudi devices are accessible
hl-smi

# Check SynapseAI version
python3 -c "import habana_frameworks.torch; print('Habana frameworks available')"
```

### Install vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install build dependencies (skip torch — use Habana's PyTorch)
pip install -r <(sed '/^torch/d' requirements/build.txt)

# Install vLLM with empty device target
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
```

### Install the vllm-gaudi plugin

```bash
pip install git+https://github.com/vllm-project/vllm-gaudi.git
```

!!! tip "Version compatibility"
    The `vllm-gaudi` plugin tracks vLLM's main branch. If you are using a
    specific vLLM version, check the plugin's
    [compatibility notes](https://github.com/vllm-project/vllm-gaudi)
    for the matching plugin version.

### Verify the installation

```bash
python3 -c "
import vllm
print(f'vLLM {vllm.__version__} installed')
import habana_frameworks.torch
print('Habana frameworks available')
"
```

---

## Running Inference

### Offline inference

```bash
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_LAZY_MODE=1

python3 examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m
```

### Online serving

```bash
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_LAZY_MODE=1

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype bfloat16
```

### Multi-card tensor parallel

```bash
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_LAZY_MODE=1

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --port 8000
```

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `HABANA_VISIBLE_DEVICES` | Gaudi devices to expose (e.g., `all`, `0,1`) |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES` | Enable lazy collective operations (`true`) |
| `PT_HPU_LAZY_MODE` | Enable Gaudi lazy execution mode (`1`) |
| `VLLM_TARGET_DEVICE` | Set to `empty` for Gaudi builds |
| `VLLM_SKIP_WARMUP` | Skip warmup phase (`true` for testing) |

---

## Version Compatibility

The `vllm-gaudi` plugin maintains a compatibility file that pins the vLLM
community commit it is tested against:

```
https://raw.githubusercontent.com/vllm-project/vllm-gaudi/vllm/last-good-commit-for-vllm-gaudi/VLLM_COMMUNITY_COMMIT
```

- If the file contains `latest`, the current vLLM main branch is used.
- If it contains a commit SHA, that specific commit is used for compatibility.

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for Gaudi-specific feature support.

---

## Troubleshooting

??? question "Habana runtime not found"
    Install the Habana container runtime:
    ```bash
    # Follow the Habana installation guide
    # https://docs.habana.ai/en/latest/Installation_Guide/index.html
    ```

??? question "Plugin version mismatch"
    If you see API compatibility errors, check the
    [vllm-gaudi repository](https://github.com/vllm-project/vllm-gaudi) for
    the compatible vLLM version and install accordingly.

??? question "Out-of-memory on Gaudi"
    Reduce `--max-model-len` or use a smaller model. Gaudi HBM is shared
    between model weights and the KV cache.

---

## Community & Support

- **Slack**: `#sig-hpu` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub**: [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
- **Intel Gaudi Docs**: [docs.habana.ai](https://docs.habana.ai/)

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Plugin System](../../design/plugin_system.md)** — How hardware plugins work
