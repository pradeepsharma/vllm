---
description: >
  Install vLLM for Google Cloud TPUs — pip, Docker, and from-source instructions
  for TPU v4, v5e, v5p, and v6e (Trillium).
toc_depth: 3
---

# Google TPU Installation

vLLM supports Google Cloud TPUs through the PyTorch/XLA stack. TPU support
enables high-throughput inference on Google's custom AI accelerators.

!!! note "Dedicated TPU documentation"
    For the most up-to-date TPU-specific documentation, including advanced
    configuration and troubleshooting, see the
    [vLLM on TPU project documentation](https://docs.vllm.ai/projects/tpu/en/latest/).

---

## Requirements

| Requirement | Value |
|---|---|
| **Platform** | Google Cloud TPU VM (v4, v5e, v5p, v6e / Trillium) |
| **Operating System** | Linux (TPU VM) |
| **Python** | 3.12 |
| **PyTorch/XLA** | Nightly build (provided by base image) |

!!! note "TPU VM only"
    vLLM TPU support requires a **TPU VM** (not a TPU node). TPU VMs have
    direct access to the TPU hardware. See
    [Google Cloud TPU documentation](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
    for how to create a TPU VM.

---

## Option 1 — Install via pip

The simplest way to install vLLM for TPU is via the `vllm-tpu` package:

```bash
pip install vllm-tpu
```

!!! tip
    The `vllm-tpu` package bundles the correct PyTorch/XLA version and all
    TPU-specific dependencies. It is the recommended installation method for
    production use.

---

## Option 2 — Docker

### Build the TPU Docker image

The TPU Dockerfile starts from Google's official PyTorch/XLA nightly image:

```bash
# Build the vLLM TPU image
docker build -f docker/Dockerfile.tpu -t vllm-tpu .
```

The Dockerfile uses the following base image by default:

```
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.12_tpuvm_<NIGHTLY_DATE>
```

You can override the nightly date:

```bash
docker build -f docker/Dockerfile.tpu \
    --build-arg NIGHTLY_DATE=20250730 \
    -t vllm-tpu .
```

### Run the TPU container

```bash
docker run --privileged --net host --shm-size=16G -it \
    -e "HF_TOKEN=${HF_TOKEN}" \
    --name tpu-vllm \
    vllm-tpu /bin/bash
```

| Flag | Purpose |
|---|---|
| `--privileged` | Required for TPU device access |
| `--net host` | Required for TPU networking |
| `--shm-size=16G` | Shared memory for large model weights |

---

## Option 3 — Build from Source

### Prerequisites

Start from Google's official PyTorch/XLA TPU VM image or a TPU VM with the
correct software stack:

```bash
# On a TPU VM, verify the TPU is accessible
pip install tpu-info
tpu-info
```

### Install dependencies

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Remove existing PyTorch/XLA (the base image version may conflict)
pip uninstall -y torch torch_xla torchvision

# Install TPU-specific requirements
export VLLM_TARGET_DEVICE=tpu
pip install -r requirements/tpu.txt
```

**TPU requirements include:**

| Package | Purpose |
|---|---|
| `torch`, `torch_xla` | PyTorch with XLA backend |
| `ray[default]`, `ray[data]` | Distributed execution |
| `nixl` | Network interconnect library |
| `tpu-inference` | TPU inference utilities |
| `cmake`, `jinja2` | Build tools |

### Install vLLM

```bash
# Install in editable mode (for development)
pip install -e .

# Install test utilities
pip install -e tests/vllm_test_utils
```

---

## Verify the Installation

```bash
# Check TPU hardware
tpu-info

# Run a basic inference test
python3 -c "
import vllm
from vllm import LLM, SamplingParams

llm = LLM(model='facebook/opt-125m')
outputs = llm.generate(['Hello, my name is'], SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
"
```

---

## Running Inference

### Offline inference

```bash
python3 examples/offline_inference/tpu.py
```

### Online serving

```bash
vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --port 8000
```

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `VLLM_TARGET_DEVICE` | Set to `tpu` for TPU builds |
| `VLLM_XLA_CHECK_RECOMPILATION` | Set to `1` to detect unexpected XLA recompilations |
| `VLLM_XLA_CACHE_PATH` | Path for XLA compilation cache (leave empty to disable) |
| `HF_TOKEN` | Hugging Face authentication token for gated models |

---

## Performance Notes

- **XLA compilation**: The first inference request triggers XLA compilation,
  which can take several minutes. Subsequent requests use the cached compilation.
- **Recompilation**: Changing the input shape (sequence length, batch size)
  triggers recompilation. Use fixed shapes or `--max-model-len` to minimize this.
- **Tensor parallelism**: TPU pods support tensor parallelism across multiple
  TPU chips. Use `--tensor-parallel-size` to match your TPU topology.

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for TPU-specific feature support.

---

## Troubleshooting

??? question "TPU not detected"
    Ensure you are running on a TPU VM (not a regular GCE VM). Run
    `ls /dev/accel*` to verify TPU device files are present.

??? question "XLA recompilation warnings"
    Set `VLLM_XLA_CHECK_RECOMPILATION=1` to identify which operations trigger
    recompilation. Use consistent input shapes to minimize recompilation.

??? question "Out-of-memory on TPU"
    Reduce `--max-model-len` or use a smaller model. TPU HBM is shared between
    model weights and the KV cache.

??? question "Docker: TPU not accessible in container"
    Ensure `--privileged` is set. TPU VMs require privileged access for the
    TPU device driver.

---

## Community & Support

- **Slack**: `#sig-tpu` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub Issues**: Add `[TPU]` to the issue title
- **TPU Docs**: [docs.vllm.ai/projects/tpu](https://docs.vllm.ai/projects/tpu/en/latest/)

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Parallelism & Scaling](../../serving/parallelism_scaling.md)** — Multi-chip tensor parallel
