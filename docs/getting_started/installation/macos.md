---
description: >
  Install vLLM on macOS Apple Silicon (M1, M2, M3, M4) — build from source
  for CPU inference, or use vllm-metal for GPU-accelerated inference via MLX.
toc_depth: 3
---

# macOS Apple Silicon Installation

vLLM has experimental support for macOS with Apple Silicon (M-series chips).
The CPU backend runs natively on Apple Silicon using the ARM AArch64 code path,
supporting FP32 and FP16 data types.

!!! tip "GPU-accelerated inference"
    For GPU-accelerated inference on Apple Silicon using Metal/MLX, check out
    [vllm-metal](https://github.com/vllm-project/vllm-metal), a
    community-maintained hardware plugin that uses MLX as the compute backend.

---

## Requirements

| Requirement | Value |
|---|---|
| **Hardware** | Apple Silicon (M1, M2, M3, M4 or later) |
| **Operating System** | macOS Sonoma (14) or later |
| **Xcode** | Xcode 15.4 or later with Command Line Tools |
| **Compiler** | Apple Clang ≥ 15.0.0 |
| **Python** | 3.10 – 3.13 (3.12 recommended) |
| **Data types** | FP32, FP16 |

!!! note
    On macOS, `VLLM_TARGET_DEVICE` is automatically set to `cpu`. There is no
    need to set it manually.

---

## Install Xcode Command Line Tools

If you haven't already, install the Xcode Command Line Tools:

```bash
xcode-select --install
```

Verify the installation:

```bash
clang --version
# Expected: Apple clang version 15.0.0 or later
```

---

## Option 1 — Install via pip (Recommended)

### Create a Python environment

We recommend [uv](https://docs.astral.sh/uv/) for fast environment management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.12
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Install vLLM

```bash
# Install dependencies
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match

# Install vLLM
uv pip install -e . --no-build-isolation
```

!!! tip "Why `--index-strategy unsafe-best-match`?"
    This flag allows `uv` to search all configured package indexes (PyTorch CPU
    index and PyPI) to find the best compatible versions. Without it, you may
    encounter `typing-extensions` version conflicts. The term "unsafe" refers
    to the resolution strategy, not security — both PyTorch and PyPI are trusted
    sources.

### Verify the installation

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
```

---

## Option 2 — Build from Source (Full Build)

Use this approach if you need to modify C++ code or want a standalone wheel:

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install dependencies
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match

# Build and install
uv pip install -e . --no-build-isolation
```

For the macOS CI workflow, vLLM uses the following build configuration:

```bash
# Set parallel build level
export CMAKE_BUILD_PARALLEL_LEVEL=4

# Install build dependencies
uv pip install -r requirements/cpu-build.txt --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match

# Install vLLM
uv pip install -e . --no-build-isolation
```

---

## Verify with a Smoke Test

Run a quick smoke test to verify the installation:

```bash
# Start the server in the background
vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --load-format dummy \
    --enforce-eager \
    --port 8000 &

SERVER_PID=$!

# Wait for the server to start (up to 60 seconds)
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Server started successfully"
        break
    fi
    sleep 2
done

# Test the health endpoint
curl -f http://localhost:8000/health

# Test text completion
curl -f http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Hello, my name is",
        "max_tokens": 20
    }'

# Stop the server
kill $SERVER_PID
```

---

## Running Inference

### Offline inference

```bash
python3 -c "
from vllm import LLM, SamplingParams

llm = LLM(model='facebook/opt-125m')
outputs = llm.generate(
    ['Hello, my name is', 'The capital of France is'],
    SamplingParams(max_tokens=50)
)
for output in outputs:
    print(output.outputs[0].text)
"
```

### Online serving

```bash
vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 2048 \
    --port 8000
```

Then query the server:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

---

## Runtime Configuration

### Key environment variables

| Variable | Description | Default |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | KV cache size in GiB | `0` |
| `VLLM_CPU_OMP_THREADS_BIND` | CPU cores for OpenMP threads | `auto` |

### Recommended configuration for Apple Silicon

```bash
# Use BF16 for best performance on Apple Silicon
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096
```

---

## GPU-Accelerated Inference with vllm-metal

For significantly better performance on Apple Silicon, use the
[vllm-metal](https://github.com/vllm-project/vllm-metal) plugin, which
uses Apple's Metal GPU API via the MLX framework:

```bash
pip install vllm-metal
```

!!! note
    `vllm-metal` is a community-maintained plugin and may not support all
    vLLM features. See the
    [vllm-metal repository](https://github.com/vllm-project/vllm-metal)
    for supported models and features.

---

## Troubleshooting

??? question "Build fails: standard C++ headers not found"
    This usually means the Xcode Command Line Tools are not properly installed.
    Remove and reinstall them:
    ```bash
    sudo rm -rf /Library/Developer/CommandLineTools
    xcode-select --install
    ```

??? question "C++17 compatibility errors (`constexpr` not a type)"
    Your compiler may be defaulting to an older C++ standard. Edit
    `cmake/cpu_extension.cmake` and add before `set(CMAKE_CXX_STANDARD_REQUIRED ON)`:
    ```cmake
    set(CMAKE_CXX_STANDARD 17)
    ```
    Verify your compiler supports C++17:
    ```bash
    clang++ -std=c++17 -pedantic -dM -E -x c++ /dev/null | grep __cplusplus
    # Expected: #define __cplusplus 201703L
    ```

??? question "typing-extensions version conflict"
    Use `--index-strategy unsafe-best-match` when installing with `uv`:
    ```bash
    uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
    ```

??? question "Slow inference performance"
    CPU inference on macOS is significantly slower than GPU inference.
    Consider:
    - Using a smaller model (1B–3B parameters)
    - Reducing `--max-model-len`
    - Using the [vllm-metal](https://github.com/vllm-project/vllm-metal)
      plugin for Metal GPU acceleration

??? question "Out-of-memory errors"
    Apple Silicon uses unified memory shared between CPU and GPU. Reduce
    `--max-model-len` or use a quantized model to reduce memory usage.

---

## Supported Features

macOS Apple Silicon uses the CPU backend. See the
[Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for CPU-specific feature support.

**Supported data types:** FP32, FP16

**Not supported on macOS:**
- CUDA-specific features (FlashInfer, CUDA graphs)
- ROCm features
- Multi-GPU tensor parallel (single device only)

---

## Community & Support

- **Slack**: `#sig-cpu` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub Issues**: Add `[macOS]` to the issue title
- **vllm-metal**: [github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[CPU Installation](cpu.md)** — Full CPU installation guide (all platforms)
- **[Configuration](../../configuration/engine_args.md)** — Tune engine arguments
