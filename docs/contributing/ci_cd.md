# CI/CD Pipeline

vLLM uses a two-tier CI/CD system: **Buildkite** for GPU-accelerated tests and **GitHub Actions** for lightweight checks. This guide explains how the pipeline works, how to interpret CI results, and how to add new CI jobs.

---

## Overview

| System | Purpose | Trigger |
|---|---|---|
| **GitHub Actions** | Linting, pre-commit, macOS smoke tests | Every PR and push to `main` |
| **Buildkite** | GPU tests, distributed tests, model tests | Every PR and push to `main` |

---

## GitHub Actions Workflows

GitHub Actions workflows live in `.github/workflows/`. They handle lightweight checks that don't require GPU hardware.

### Pre-commit (`.github/workflows/pre-commit.yml`)

Runs on every PR and push to `main`. Executes all pre-commit hooks:

```yaml
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@...
    - uses: actions/setup-python@...
      with:
        python-version: "3.12"
    - uses: pre-commit/action@...
      with:
        extra_args: --all-files --hook-stage manual
```

This runs:
- **Ruff** — Python linting and formatting
- **mypy** — Static type checking
- **typos** — Spell checking
- **markdownlint** — Markdown linting
- **actionlint** — GitHub Actions workflow linting

**If this fails:** Run `pre-commit run --all-files --hook-stage manual` locally, fix the issues, and push again.

### macOS Smoke Test (`.github/workflows/macos-smoke-test.yml`)

Runs on push to `main` and manual triggers. Tests that vLLM installs and serves correctly on Apple Silicon (macOS):

1. Installs vLLM with CPU dependencies
2. Starts `vllm serve` with a small model
3. Tests the `/health` and `/v1/completions` endpoints

### Other Workflows

| Workflow | Purpose |
|---|---|
| `stale.yml` | Marks stale issues and PRs after inactivity |
| `reminder_comment.yml` | Posts reminders on PRs about contribution guidelines |
| `cleanup_pr_body.yml` | Removes template text from PR descriptions |
| `issue_autolabel.yml` | Automatically labels issues based on content |
| `add_label_automerge.yml` | Manages auto-merge labels |

---

## Buildkite CI Pipeline

Buildkite handles GPU-intensive testing. The pipeline is configured in `.buildkite/` and runs on NVIDIA GPU hardware.

### Pipeline Structure

```
.buildkite/
├── ci_config.yaml          # Pipeline configuration
├── image_build/            # Docker image build jobs
│   ├── image_build.yaml    # Main GPU image build
│   ├── image_build.sh      # Build script
│   ├── image_build_cpu.sh  # CPU image build
│   └── image_build_hpu.sh  # HPU image build
├── test_areas/             # Test job definitions (25 areas)
│   ├── basic_correctness.yaml
│   ├── distributed.yaml
│   ├── models_language.yaml
│   └── ...
├── hardware_tests/         # Hardware-specific tests
│   ├── amd.yaml            # AMD ROCm tests
│   ├── cpu.yaml            # CPU tests
│   ├── intel.yaml          # Intel XPU tests
│   └── ...
├── release-pipeline.yaml   # Release wheel building
└── test-amd.yaml           # AMD-specific test pipeline
```

### Pipeline Configuration (`ci_config.yaml`)

```yaml
name: vllm_ci
job_dirs:
  - ".buildkite/image_build"    # Image build jobs
  - ".buildkite/test_areas"     # Test jobs
  - ".buildkite/hardware_tests" # Hardware-specific jobs

# Files that trigger a full rebuild of all tests
run_all_patterns:
  - "docker/Dockerfile"
  - "CMakeLists.txt"
  - "requirements/common.txt"
  - "requirements/cuda.txt"
  - "requirements/build.txt"
  - "setup.py"
  - "csrc/"
  - "cmake/"

# Excluded from run_all_patterns
run_all_exclude_patterns:
  - "docker/Dockerfile."    # Platform-specific Dockerfiles
  - "csrc/cpu/"
  - "csrc/rocm/"
```

When any file matching `run_all_patterns` is changed, **all** test jobs run. Otherwise, only jobs whose `source_file_dependencies` match the changed files run.

### Image Build

Before tests run, Docker images are built:

1. **Main GPU image** (`image-build`) — CUDA-enabled image with vLLM installed
2. **CPU image** (`image-build-cpu`) — CPU-only image
3. **HPU image** (`image-build-hpu`) — Intel Gaudi image
4. **AMD image** (`image-build-amd`) — ROCm image

All test jobs `depends_on: image-build`, ensuring the image is ready before tests start.

---

## Test Areas

Each test area is defined in a YAML file in `.buildkite/test_areas/`. Here's the complete list:

| Test Area | File | Description |
|---|---|---|
| Basic Correctness | `basic_correctness.yaml` | Core generation correctness |
| Attention | `attention.yaml` | Attention backend tests |
| Benchmarks | `benchmarks.yaml` | Performance benchmarks |
| Compile | `compile.yaml` | torch.compile tests |
| CUDA | `cuda.yaml` | CUDA-specific tests |
| Distributed | `distributed.yaml` | Multi-GPU tests (2, 4, 8 GPUs) |
| E2E Integration | `e2e_integration.yaml` | End-to-end integration tests |
| Engine | `engine.yaml` | Engine and V1 engine tests |
| Entrypoints | `entrypoints.yaml` | API server tests |
| Expert Parallelism | `expert_parallelism.yaml` | MoE expert parallelism |
| Kernels | `kernels.yaml` | CUDA kernel tests |
| LM Eval | `lm_eval.yaml` | Language model evaluation |
| LoRA | `lora.yaml` | LoRA adapter tests |
| Misc | `misc.yaml` | Miscellaneous tests |
| Model Executor | `model_executor.yaml` | Model executor tests |
| Models (Basic) | `models_basic.yaml` | Basic model tests |
| Models (Distributed) | `models_distributed.yaml` | Distributed model tests |
| Models (Language) | `models_language.yaml` | Language model tests |
| Models (Multimodal) | `models_multimodal.yaml` | Multimodal model tests |
| Plugins | `plugins.yaml` | Plugin system tests |
| PyTorch | `pytorch.yaml` | PyTorch integration tests |
| Quantization | `quantization.yaml` | Quantization method tests |
| Ray Compat | `ray_compat.yaml` | Ray compatibility tests |
| Samplers | `samplers.yaml` | Sampling strategy tests |
| Weight Loading | `weight_loading.yaml` | Weight loading tests |

### Test Area YAML Format

```yaml
group: Basic Correctness
depends_on:
  - image-build
steps:
- label: Basic Correctness
  timeout_in_minutes: 30
  source_file_dependencies:
  - vllm/                              # Run if any vllm/ file changes
  - tests/basic_correctness/test_basic_correctness
  commands:
  - export VLLM_WORKER_MULTIPROC_METHOD=spawn
  - pytest -v -s basic_correctness/test_basic_correctness.py
  mirror:
    amd:
      device: mi325_1
      depends_on:
      - image-build-amd
```

Key fields:
- `group` — Display name in Buildkite UI
- `depends_on` — Jobs that must complete first
- `source_file_dependencies` — Files/directories that trigger this job
- `timeout_in_minutes` — Maximum job duration
- `num_devices` — Number of GPUs required (default: 1)
- `mirror.amd` — AMD-equivalent test configuration
- `optional: true` — Job is skipped by default (not required for PR merge)

---

## Hardware Tests

Hardware-specific tests run on dedicated hardware queues:

### AMD ROCm Tests

Defined in `.buildkite/hardware_tests/amd.yaml` and `.buildkite/test-amd.yaml`. Tests run on AMD MI300x and MI325x GPUs.

The AMD image is built from `docker/Dockerfile.rocm` targeting `gfx942;gfx950` architectures.

### CPU Tests

Defined in `.buildkite/hardware_tests/cpu.yaml`. Tests run on CPU-only machines, covering:
- x86_64 CPU inference
- ARM64 CPU inference
- CPU-specific quantization (WNA16)

### Intel Tests

Defined in `.buildkite/hardware_tests/intel.yaml`. Tests Intel XPU (GPU) support.

### Ascend NPU Tests

Defined in `.buildkite/hardware_tests/ascend_npu.yaml`. Tests Huawei Ascend NPU support.

### GH200 Tests

Defined in `.buildkite/hardware_tests/gh200.yaml`. Tests NVIDIA GH200 (Grace Hopper) support.

---

## Understanding CI Results

### Buildkite Dashboard

Access the Buildkite dashboard at [buildkite.com/vllm](https://buildkite.com/vllm). Each PR shows:
- Build status (passing/failing/running)
- Individual step results
- Test logs and artifacts

### Interpreting Failures

**Pre-commit failure:**
```
ruff check failed
```
→ Run `pre-commit run --all-files --hook-stage manual` locally and fix issues.

**Test failure:**
```
FAILED tests/basic_correctness/test_basic_correctness.py::test_vllm_gc_ed
```
→ Run the specific test locally to reproduce. Check if it's a flaky test using [Buildkite Test Suites](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main).

**Image build failure:**
→ Usually indicates a dependency issue. Check the Docker build logs.

**Timeout:**
→ The test exceeded `timeout_in_minutes`. Either the test is too slow or there's a hang.

### Flaky Tests

If a test fails intermittently, it may be a flaky test. To report:
1. Open a [CI Failure issue](https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml)
2. Include the test name, Buildkite link, and history from [Buildkite Test Suites](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main)

---

## Adding New CI Jobs

### Adding a Test to an Existing Area

Edit the appropriate YAML file in `.buildkite/test_areas/`:

```yaml
- label: My New Test
  timeout_in_minutes: 20
  source_file_dependencies:
  - vllm/my_feature/
  - tests/my_feature/
  commands:
  - pytest -v -s my_feature/test_my_feature.py
```

### Creating a New Test Area

1. Create `.buildkite/test_areas/my_area.yaml`:

```yaml
group: My Area
depends_on:
  - image-build
steps:
- label: My Area Tests
  timeout_in_minutes: 30
  source_file_dependencies:
  - vllm/my_module/
  - tests/my_area/
  commands:
  - pytest -v -s my_area/
```

2. The CI system automatically discovers YAML files in `job_dirs` (configured in `ci_config.yaml`).

### Adding AMD Mirror Tests

To run your test on AMD hardware as well, add a `mirror` section:

```yaml
- label: My Test
  commands:
  - pytest -v -s my_test.py
  mirror:
    amd:
      device: mi325_1
      depends_on:
      - image-build-amd
      commands:
      - pytest -v -s my_test.py  # Same or different commands for AMD
```

### Making Tests Optional

Tests that are slow or cover niche scenarios can be marked optional:

```yaml
- label: Extended Test
  optional: true  # Not required for PR merge
  timeout_in_minutes: 120
  commands:
  - pytest -v -s extended_test.py
```

Optional tests are skipped by default but can be manually triggered.

---

## Release Pipeline

The release pipeline (`.buildkite/release-pipeline.yaml`) builds Python wheels for distribution on PyPI. It runs when a release tag is pushed.

### Wheel Build Matrix

| Architecture | CUDA Version | Platform |
|---|---|---|
| x86_64 | 12.9 | Linux (manylinux_2_31) |
| x86_64 | 13.0 | Linux (manylinux_2_35) |
| x86_64 | CPU | Linux |
| aarch64 | 12.9 | Linux (manylinux_2_35) |
| aarch64 | 13.0 | Linux (manylinux_2_35) |
| aarch64 | CPU | Linux |

### Release Trigger

Wheels are built when an RC tag is pushed:

```bash
git tag v0.9.0-rc1
git push origin v0.9.0-rc1
```

The final release tag (`v0.9.0`) does not trigger a build — it's used for release notes and GitHub assets only.

---

## CI Best Practices

### For Contributors

1. **Run pre-commit locally** before pushing — saves CI time
2. **Check `source_file_dependencies`** — ensure your changes trigger the right tests
3. **Mark slow tests as `optional`** — don't block PRs on slow tests
4. **Use `timeout_in_minutes`** — always set a reasonable timeout
5. **Add AMD mirrors** for tests that should run on both NVIDIA and AMD

### For Maintainers

1. **Monitor flaky tests** using Buildkite Test Suites analytics
2. **Use `soft_fail: true`** for new hardware tests during initial integration
3. **Keep image build times reasonable** — large images slow down all tests
4. **Review `run_all_patterns`** — changes here affect all PRs

---

## Environment Variables in CI

| Variable | Description |
|---|---|
| `BUILDKITE_COMMIT` | Current commit SHA |
| `BUILDKITE_BRANCH` | Current branch name |
| `BUILDKITE_PARALLEL_JOB_COUNT` | Total parallel job count (for sharding) |
| `BUILDKITE_PARALLEL_JOB` | Current job index (for sharding) |
| `REGISTRY` | Docker registry URL |
| `REPO` | Docker repository name |
| `IMAGE_TAG` | Docker image tag |

---

## Related Resources

- [Buildkite Documentation](https://buildkite.com/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [vLLM Buildkite Test Suites](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main)
- [Release Process](release_process.md)
