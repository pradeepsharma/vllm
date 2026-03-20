# Testing Guide

vLLM has a comprehensive test suite covering correctness, performance, distributed behavior, and hardware-specific functionality. This guide explains how to run tests, understand the test organization, and write new tests.

---

## Test Infrastructure

vLLM uses **pytest** as its primary test framework, with several plugins:

| Plugin | Purpose |
|---|---|
| `pytest-asyncio` | Testing async code |
| `pytest-forked` | Run tests in isolated subprocesses |
| `pytest-rerunfailures` | Retry flaky tests |
| `pytest-shard` | Shard tests across parallel workers |
| `pytest-timeout` | Enforce test time limits |
| `pytest-cov` | Code coverage reporting |

### Installing Test Dependencies

```bash
pip install -r requirements/test.txt
```

Or as part of the full development setup:

```bash
pip install -r requirements/dev.txt
```

---

## Test Organization

Tests live in the `tests/` directory, organized by area:

```
tests/
├── basic_correctness/        # Core generation correctness
│   ├── test_basic_correctness.py
│   ├── test_cpu_offload.py
│   └── test_cumem.py
├── distributed/              # Multi-GPU communication tests
│   ├── test_comm_ops.py
│   ├── test_pynccl.py
│   └── test_shm_broadcast.py
├── engine/                   # Engine-level tests
├── entrypoints/              # API server tests
│   └── openai/               # OpenAI-compatible API tests
├── kernels/                  # CUDA kernel tests
├── lora/                     # LoRA adapter tests
├── models/                   # Per-model tests
│   ├── language/             # Language model tests
│   │   ├── generation/       # Text generation tests
│   │   └── pooling/          # Embedding/pooling tests
│   ├── multimodal/           # Multimodal model tests
│   └── quantization/         # Quantized model tests
├── quantization/             # Quantization method tests
├── samplers/                 # Sampling strategy tests
├── v1/                       # V1 engine tests
│   ├── distributed/          # V1 distributed tests
│   ├── engine/               # V1 engine tests
│   └── worker/               # V1 worker tests
└── conftest.py               # Shared fixtures
```

---

## Running Tests

### Basic Usage

```bash
# Run from the tests/ directory
cd tests

# Run a specific test file
pytest -v -s basic_correctness/test_basic_correctness.py

# Run a specific test function
pytest -v -s basic_correctness/test_basic_correctness.py::test_vllm_gc_ed

# Run tests matching a pattern
pytest -v -s -k "test_llama"
```

### Common pytest Flags

| Flag | Description |
|---|---|
| `-v` | Verbose output (show test names) |
| `-s` | Show stdout/stderr (don't capture) |
| `-k "pattern"` | Run tests matching the pattern |
| `-x` | Stop after first failure |
| `--tb=short` | Short traceback format |
| `--timeout=300` | Set per-test timeout in seconds |
| `-n auto` | Run tests in parallel (requires pytest-xdist) |

### Running by Marker

vLLM uses pytest markers to categorize tests:

```bash
# Run only core model tests (run in every PR)
pytest -v -s models/language -m "core_model"

# Run CPU-specific tests
pytest -v -s -m "cpu_test"

# Run distributed tests
pytest -v -s -m "distributed"

# Skip slow tests
pytest -v -s -m "not slow_test"

# Run optional tests (skipped by default)
pytest -v -s --optional models/language
```

### Available Markers

| Marker | Description |
|---|---|
| `core_model` | Run in every PR (fast, critical models) |
| `slow_test` | Slow tests, run in nightly CI |
| `distributed` | Requires multiple GPUs |
| `cpu_model` | Run in CPU test suite |
| `cpu_test` | CPU-only test |
| `hybrid_model` | Models with Mamba/SSM layers |
| `optional` | Skipped by default; use `--optional` to include |
| `split` | Run as part of a sharded test group |

---

## Test Areas

### Basic Correctness

Tests that vLLM produces correct outputs for standard generation tasks:

```bash
cd tests
export VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -v -s basic_correctness/test_basic_correctness.py
pytest -v -s basic_correctness/test_cpu_offload.py
pytest -v -s basic_correctness/test_cumem.py
```

### Model Tests

Model tests verify that vLLM's output matches the HuggingFace reference implementation:

```bash
# Language model generation tests
pytest -v -s models/language/generation -m "core_model and not slow_test"

# Embedding/pooling tests
pytest -v -s models/language/pooling -m "core_model"

# Multimodal model tests
pytest -v -s models/multimodal -m "core_model"
```

### Distributed Tests

Distributed tests require multiple GPUs. Use `torchrun` or set `num_devices`:

```bash
# 2-GPU tests
CUDA_VISIBLE_DEVICES=0,1 pytest -v -s distributed/test_comm_ops.py

# 4-GPU tests
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -v -s distributed/

# V1 distributed tests
TP_SIZE=1 DP_SIZE=2 pytest -v -s v1/distributed/test_async_llm_dp.py
```

### Quantization Tests

```bash
pytest -v -s quantization/
pytest -v -s models/quantization/
```

### Kernel Tests

```bash
pytest -v -s kernels/
```

### Entrypoint Tests

```bash
# OpenAI API server tests
pytest -v -s entrypoints/openai/

# LLM class tests
pytest -v -s entrypoints/llm/
```

---

## Writing Tests

### Test File Conventions

- Place test files in the appropriate subdirectory of `tests/`
- Name test files `test_<feature>.py`
- Name test functions `test_<specific_behavior>`
- Use descriptive names that explain what is being tested

### Basic Test Structure

```python
# tests/models/language/generation/test_my_model.py
# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm import LLM, SamplingParams


@pytest.mark.parametrize("model", ["my-org/my-model-7b"])
@pytest.mark.core_model
def test_my_model_generation(model: str, vllm_runner):
    """Test that my model generates coherent text."""
    with vllm_runner(model) as llm:
        outputs = llm.generate_greedy(
            ["The capital of France is"],
            max_tokens=10,
        )
    assert len(outputs) == 1
    assert len(outputs[0][1]) > 0  # Non-empty output
```

### Using the `vllm_runner` Fixture

The `vllm_runner` fixture (defined in `conftest.py`) provides a convenient context manager for testing:

```python
def test_with_runner(vllm_runner):
    with vllm_runner(
        model="facebook/opt-125m",
        dtype="float16",
        max_model_len=512,
    ) as llm:
        # Generate with greedy decoding
        outputs = llm.generate_greedy(["Hello"], max_tokens=20)
        
        # Generate with sampling
        outputs = llm.generate(
            ["Hello"],
            SamplingParams(temperature=0.8, max_tokens=20),
        )
```

### Comparing with HuggingFace

For model correctness tests, compare vLLM output against HuggingFace:

```python
from tests.models.utils import check_logprobs_close

def test_matches_hf(hf_runner, vllm_runner):
    model = "facebook/opt-125m"
    prompts = ["The quick brown fox"]
    
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(prompts, max_tokens=10)
    
    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(prompts, max_tokens=10)
    
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
```

### Async Tests

For testing async code (e.g., the async engine):

```python
import pytest

@pytest.mark.asyncio
async def test_async_generation():
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    
    engine_args = AsyncEngineArgs(model="facebook/opt-125m")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async for output in engine.generate("Hello", request_id="test-1"):
        pass  # Process streaming output
```

### Parametrize Tests

Use `@pytest.mark.parametrize` to test multiple configurations:

```python
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_model_dtypes(dtype: str, tp_size: int, vllm_runner):
    with vllm_runner(
        "facebook/opt-125m",
        dtype=dtype,
        tensor_parallel_size=tp_size,
    ) as llm:
        outputs = llm.generate_greedy(["Hello"], max_tokens=5)
        assert outputs
```

### Skipping Tests

```python
import pytest
import torch

# Skip if not enough GPUs
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)
def test_tensor_parallel():
    ...

# Skip on specific platforms
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA"
)
def test_cuda_kernel():
    ...
```

### Fixtures

Common fixtures are defined in `tests/conftest.py`. Key fixtures:

| Fixture | Description |
|---|---|
| `vllm_runner` | Context manager for vLLM LLM instances |
| `hf_runner` | Context manager for HuggingFace model instances |
| `example_prompts` | Standard set of test prompts |
| `tmp_path` | Temporary directory (pytest built-in) |

---

## Test Markers in Detail

### `core_model`

Tests marked `core_model` run in every PR CI pipeline. Use this for:
- The most important/popular models
- Tests that are fast (< 5 minutes)
- Tests that cover critical functionality

```python
@pytest.mark.core_model
def test_llama_generation(vllm_runner):
    ...
```

### `slow_test`

Tests marked `slow_test` run only in nightly CI:

```python
@pytest.mark.slow_test
def test_full_benchmark(vllm_runner):
    ...
```

### `distributed`

Tests requiring multiple GPUs:

```python
@pytest.mark.distributed
def test_tensor_parallel_inference(vllm_runner):
    ...
```

---

## Debugging Test Failures

### Enable Debug Logging

```bash
VLLM_LOGGING_LEVEL=DEBUG pytest -v -s tests/basic_correctness/test_basic_correctness.py
```

### Trace Function Calls

For debugging hangs or crashes:

```bash
VLLM_TRACE_FUNCTION=1 pytest -v -s tests/basic_correctness/test_basic_correctness.py
```

### Run with `collect_env`

When reporting test failures, include environment information:

```bash
python vllm/collect_env.py
```

### Isolate Failures

```bash
# Run a single test with maximum verbosity
pytest -v -s --tb=long tests/path/to/test_file.py::test_specific_function

# Run in a forked subprocess to isolate memory issues
pytest --forked tests/path/to/test_file.py
```

---

## CI Test Pipeline

Tests run in Buildkite CI across multiple test areas. Each area has its own YAML configuration in `.buildkite/test_areas/`:

| Test Area | File | Description |
|---|---|---|
| Basic Correctness | `basic_correctness.yaml` | Core generation tests |
| Distributed | `distributed.yaml` | Multi-GPU tests |
| Models (Language) | `models_language.yaml` | Language model tests |
| Models (Multimodal) | `models_multimodal.yaml` | Multimodal model tests |
| Quantization | `quantization.yaml` | Quantization method tests |
| Kernels | `kernels.yaml` | CUDA kernel tests |
| Entrypoints | `entrypoints.yaml` | API server tests |
| Engine | `engine.yaml` | Engine-level tests |
| LoRA | `lora.yaml` | LoRA adapter tests |
| Samplers | `samplers.yaml` | Sampling strategy tests |

See [CI/CD Pipeline](ci_cd.md) for details on how the CI pipeline works.

---

## Performance Testing

For performance-sensitive changes, run benchmarks before and after:

```bash
# Throughput benchmark
python benchmarks/benchmark_throughput.py \
    --model facebook/opt-125m \
    --num-prompts 1000

# Latency benchmark
python benchmarks/benchmark_latency.py \
    --model facebook/opt-125m \
    --batch-size 1
```

See [Benchmarking](../benchmarking/index.md) for comprehensive benchmarking guidance.
