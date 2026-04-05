# Hardware CI

vLLM uses Buildkite for hardware-specific continuous integration. Hardware tests run on real physical devices — NVIDIA GPUs, AMD GPUs, Intel GPUs, Intel CPUs, ARM CPUs, and Google TPUs — to validate platform-specific code paths that cannot be tested in software emulation.

## CI Structure

Hardware tests are defined in `.buildkite/hardware_tests/` as YAML pipeline files. Each file corresponds to a hardware category and contains one or more Buildkite steps.

```
.buildkite/
├── hardware_tests/
│   ├── amd.yaml          # AMD ROCm GPU tests
│   ├── ascend_npu.yaml   # Huawei Ascend NPU tests
│   ├── cpu.yaml          # CPU tests (x86, ARM)
│   ├── gh200.yaml        # NVIDIA GH200 (Grace Hopper) tests
│   └── intel.yaml        # Intel GPU (XPU) and HPU tests
└── scripts/
    └── hardware_ci/
        ├── run-amd-test.sh
        ├── run-cpu-test.sh
        ├── run-cpu-test-arm.sh
        ├── run-cpu-test-ppc64le.sh
        ├── run-cpu-test-s390x.sh
        ├── run-gh200-test.sh
        ├── run-hpu-test.sh
        ├── run-npu-test.sh
        ├── run-tpu-v1-test.sh
        ├── run-tpu-v1-test-part2.sh
        └── run-xpu-test.sh
```

## AMD ROCm CI

**File**: `.buildkite/hardware_tests/amd.yaml`

The AMD CI pipeline builds a ROCm Docker image targeting `gfx942` (MI300X) and `gfx950` (MI325X) architectures:

```yaml
group: Hardware - AMD Build
steps:
  - label: "AMD: :docker: build image"
    key: image-build-amd
    device: amd_cpu
    commands:
    - >
      docker build
      --build-arg max_jobs=16
      --build-arg REMOTE_VLLM=1
      --build-arg ARG_PYTORCH_ROCM_ARCH='gfx942;gfx950'
      --build-arg VLLM_BRANCH=$BUILDKITE_COMMIT
      --tag "rocm/vllm-ci:${BUILDKITE_COMMIT}"
      -f docker/Dockerfile.rocm
      --target test
      --no-cache
      --progress plain .
    - docker push "rocm/vllm-ci:${BUILDKITE_COMMIT}"
```

The build runs on an `amd_cpu` agent (a CPU-only machine that can build ROCm images) and pushes the result to a registry for subsequent test steps.

**Retry policy**: The AMD build retries up to 3 times on agent loss or machine failures.

**Test script**: `.buildkite/scripts/hardware_ci/run-amd-test.sh` (17 KB) covers:
- Offline inference
- Attention kernel tests
- Quantization tests
- Distributed inference
- MoE (Mixture of Experts) tests
- AITER kernel tests

## CPU CI

**File**: `.buildkite/hardware_tests/cpu.yaml`

The CPU CI runs multiple parallel test groups on `intel_cpu` and `arm_cpu` agents:

### Test Steps

| Label | Device | Timeout | Tests |
|-------|--------|---------|-------|
| CPU-Kernel Tests | `intel_cpu` | 20m | CPU attention, fused MoE, oneDNN |
| CPU-Language Generation | `intel_cpu` | 30m | Language generation, pooling models |
| CPU-Quantization | `intel_cpu` | 20m | Compressed tensors, WNA16 |
| CPU-Distributed | `intel_cpu` | 10m | Distributed smoke tests |
| CPU-Multi-Modal (×2) | `intel_cpu` | 45m | Multimodal generation (sharded) |
| Arm CPU Test | `arm_cpu` | 2h | Full ARM test suite |

All CPU tests use `soft_fail: true` and `depends_on: []` (run independently).

### Source File Dependencies

Each step declares `source_file_dependencies` to enable selective test execution:

```yaml
- label: CPU-Kernel Tests
  source_file_dependencies:
  - csrc/cpu/
  - cmake/cpu_extension.cmake
  - CMakeLists.txt
  - vllm/_custom_ops.py
  - tests/kernels/attention/test_cpu_attn.py
  - tests/kernels/moe/test_cpu_fused_moe.py
  - tests/kernels/test_onednn.py
  commands:
    - |
      bash .buildkite/scripts/hardware_ci/run-cpu-test.sh 20m "
      pytest -x -v -s tests/kernels/attention/test_cpu_attn.py
      pytest -x -v -s tests/kernels/moe/test_cpu_fused_moe.py
      pytest -x -v -s tests/kernels/test_onednn.py"
```

### Distributed Smoke Test

```bash
# .buildkite/scripts/hardware_ci/run-cpu-distributed-smoke-test.sh
bash .buildkite/scripts/hardware_ci/run-cpu-distributed-smoke-test.sh
```

### Multi-Modal Sharding

The multimodal test step uses Buildkite's parallelism feature to shard tests across 2 agents:

```yaml
- label: CPU-Multi-Modal Model Tests %N
  parallelism: 2
  commands:
    - |
      pytest -x -v -s tests/models/multimodal/generation \
        --ignore=tests/models/multimodal/generation/test_pixtral.py \
        -m cpu_model \
        --num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT \
        --shard-id=$$BUILDKITE_PARALLEL_JOB
```

## Intel GPU (XPU) CI

**File**: `.buildkite/hardware_tests/intel.yaml`

```yaml
group: Hardware
steps:
  - label: "Intel HPU Test"
    soft_fail: true
    device: intel_hpu
    commands:
    - bash .buildkite/scripts/hardware_ci/run-hpu-test.sh

  - label: "Intel GPU Test"
    depends_on: []
    soft_fail: true
    device: intel_gpu
    commands:
    - bash .buildkite/scripts/hardware_ci/run-xpu-test.sh
```

### XPU Test Coverage

The XPU test script (`run-xpu-test.sh`) validates:

```bash
# Basic inference
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m --block-size 64 --enforce-eager

# Compilation modes
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m --block-size 64 -O3 -cc.cudagraph_mode=NONE

# Tensor parallelism (Ray and MP)
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m --block-size 64 --enforce-eager -tp 2 \
  --distributed-executor-backend ray

# Triton attention backend
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m --block-size 64 --enforce-eager \
  --attention-backend=TRITON_ATTN

# FP8 quantization
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m --block-size 64 --enforce-eager --quantization fp8

# MoE with expert parallelism
python3 examples/offline_inference/basic/generate.py \
  --model ibm-research/PowerMoE-3b --block-size 64 --enforce-eager \
  -tp 2 --enable-expert-parallel
```

Unit tests:

```bash
cd tests
pytest -v -s v1/core
pytest -v -s v1/engine
pytest -v -s v1/sample
pytest -v -s v1/worker
pytest -v -s v1/structured_output
pytest -v -s v1/spec_decode
pytest -v -s v1/kv_connector/unit
```

## GH200 (Grace Hopper) CI

**File**: `.buildkite/hardware_tests/gh200.yaml`

```yaml
group: Hardware
steps:
  - label: "GH200 Test"
    soft_fail: true
    device: gh200
    optional: true
    commands:
    - nvidia-smi
    - bash .buildkite/scripts/hardware_ci/run-gh200-test.sh
```

The GH200 test is marked `optional: true` — it does not block the pipeline if the GH200 agent is unavailable.

The test script builds vLLM for ARM64 with `9.0+PTX` CUDA architecture:

```bash
DOCKER_BUILDKIT=1 docker build . \
  --file docker/Dockerfile \
  --target vllm-openai \
  --platform "linux/arm64" \
  -t gh200-test \
  --build-arg max_jobs=66 \
  --build-arg nvcc_threads=2 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg torch_cuda_arch_list="9.0+PTX"
```

Then runs offline inference:

```bash
docker run -e HF_TOKEN \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --gpus=all \
  gh200-test bash -c '
    python3 examples/offline_inference/basic/generate.py \
      --model meta-llama/Llama-3.2-1B
  '
```

## Ascend NPU CI

**File**: `.buildkite/hardware_tests/ascend_npu.yaml`

```yaml
group: Hardware
steps:
  - label: "Ascend NPU Test"
    soft_fail: true
    timeout_in_minutes: 20
    device: ascend_npu
    commands:
    - bash .buildkite/scripts/hardware_ci/run-npu-test.sh
```

Huawei Ascend NPU tests run with a 20-minute timeout.

## TPU CI

TPU tests are not in the `hardware_tests/` directory but run via dedicated scripts:

| Script | Description |
|--------|-------------|
| `run-tpu-v1-test.sh` | Main TPU v1 test suite |
| `run-tpu-v1-test-part2.sh` | Additional TPU v1 tests |

Both scripts:
1. Build `docker/Dockerfile.tpu`
2. Run inside a privileged container with `--net host --shm-size=16G`
3. Install `tpu-info`, `pytest`, `lm-eval`, and `hf-transfer`
4. Set `VLLM_XLA_CHECK_RECOMPILATION=1` to catch unexpected recompilations
5. Run tests independently with pass/fail tracking

## CI Environment Variables

| Variable | Used In | Description |
|----------|---------|-------------|
| `BUILDKITE_COMMIT` | AMD, XPU | Git commit SHA for image tagging |
| `HF_TOKEN` | All | HuggingFace API token for model downloads |
| `VLLM_CPU_KVCACHE_SPACE` | CPU | KV cache size in GiB |
| `VLLM_CPU_CI_ENV` | CPU | Use eager mode (skip compilation) |
| `CORE_RANGE` | ARM | CPU core range for binding |
| `OMP_CORE_RANGE` | ARM | OpenMP thread binding range |
| `ZE_AFFINITY_MASK` | XPU | Intel GPU device selection |
| `VLLM_XLA_CHECK_RECOMPILATION` | TPU | Detect unexpected XLA recompilations |
| `DOCKER_BUILDKIT` | AMD | Enable BuildKit for Docker builds |

## Adding New Hardware Tests

To add a new hardware test:

1. Create a YAML file in `.buildkite/hardware_tests/`:

```yaml
group: Hardware - My Platform
steps:
  - label: "My Platform Test"
    soft_fail: true
    device: my_device_type
    no_plugin: true
    commands:
    - bash .buildkite/scripts/hardware_ci/run-my-platform-test.sh
```

2. Create the test script in `.buildkite/scripts/hardware_ci/`:

```bash
#!/bin/bash
set -ex

# Build image
docker build -t my-platform-test -f docker/Dockerfile.my_platform .

# Run tests
docker run --rm my-platform-test bash -c '
  python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m
'
```

3. Register the device type with the Buildkite agent configuration.

## Related Pages

- [Platform Overview](overview.md)
- [NVIDIA CUDA Platform](cuda.md)
- [AMD ROCm Platform](rocm.md)
- [Intel XPU Platform](xpu.md)
- [CPU Platform](cpu.md)
- [Google TPU Platform](tpu.md)
- [ARM/ppc64le/s390x](alt-arch.md)
