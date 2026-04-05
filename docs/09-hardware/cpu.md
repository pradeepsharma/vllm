# CPU Platform

vLLM's CPU platform (`CpuPlatform`) enables inference on standard CPUs without any GPU hardware. It supports x86_64, ARM (AArch64), ppc64le (IBM POWER), s390x (IBM Z), and RISC-V architectures. The CPU backend uses Intel's oneDNN library for optimized matrix operations and OpenMP for parallelism.

## Source File

`vllm/platforms/cpu.py`

## Supported Architectures

| Architecture | Enum | Platforms | ISA Extensions |
|-------------|------|-----------|---------------|
| x86_64 | `CpuArchEnum.X86` | Intel/AMD servers, desktops | AVX-512, AMX, AVX2 |
| AArch64 | `CpuArchEnum.ARM` | AWS Graviton, Apple Silicon, Ampere | NEON, BF16 (optional) |
| ppc64le | `CpuArchEnum.POWERPC` | IBM POWER9/POWER10 | VSX |
| s390x | `CpuArchEnum.S390X` | IBM Z (mainframe) | Vector extensions |
| RISC-V | `CpuArchEnum.RISCV` | Sophgo SG2044 | Scalar only |

## Key Platform Properties

```python
class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"
    device_control_env_var = "CPU_VISIBLE_MEMORY_NODES"
```

## Platform Detection

The CPU platform is activated when:
1. The vLLM package version contains `"cpu"` (CPU-specific build), **or**
2. The system is macOS (`sys.platform.startswith("darwin")`)

```python
def cpu_platform_plugin() -> str | None:
    is_cpu = vllm_version_matches_substr("cpu")
    if not is_cpu:
        is_cpu = sys.platform.startswith("darwin")
    return "vllm.platforms.cpu.CpuPlatform" if is_cpu else None
```

## Supported Data Types

Data type support varies by CPU architecture:

```python
@property
def supported_dtypes(self) -> list[torch.dtype]:
    if self.get_cpu_architecture() == CpuArchEnum.POWERPC:
        return [torch.bfloat16, torch.float32]
    elif self.get_cpu_architecture() == CpuArchEnum.ARM and sys.platform.startswith("darwin"):
        # Apple Silicon: check for BF16 hardware support
        if subprocess.check_output(["sysctl -n hw.optional.arm.FEAT_BF16"], shell=True).strip() == b"1":
            return [torch.bfloat16, torch.float16, torch.float32]
        return [torch.float16, torch.float32]
    elif self.get_cpu_architecture() == CpuArchEnum.RISCV:
        # RISC-V: FP32 only (scheduler bug workaround for FP16)
        return [torch.float32]
    # x86 and AArch64: full support
    return [torch.bfloat16, torch.float16, torch.float32]
```

## x86 ISA Extensions

The CPU Dockerfile (`docker/Dockerfile.cpu`) supports fine-grained ISA control via build arguments:

| Build Argument | Default | Description |
|---------------|---------|-------------|
| `VLLM_CPU_DISABLE_AVX512` | `false` | Disable AVX-512 (use AVX2 fallback) |
| `VLLM_CPU_AVX2` | `false` | Force AVX2 (cross-compilation) |
| `VLLM_CPU_AVX512` | `false` | Force AVX-512 (cross-compilation) |
| `VLLM_CPU_AVX512BF16` | `false` | Enable AVX-512 BF16 instructions |
| `VLLM_CPU_AVX512VNNI` | `false` | Enable AVX-512 VNNI (INT8 dot product) |
| `VLLM_CPU_AMXBF16` | `true` | Enable AMX BF16 (Intel 4th Gen Xeon+) |
| `VLLM_CPU_ARM_BF16` | `false` | Enable ARM BF16 (cross-compilation) |

### AMX (Advanced Matrix Extensions)

AMX is Intel's hardware accelerator for matrix operations, available on 4th Generation Intel Xeon Scalable processors (Sapphire Rapids) and newer. It provides significant speedups for BF16 matrix multiplications used in transformer attention and FFN layers.

AMX is enabled by default (`VLLM_CPU_AMXBF16=1`). To disable:

```bash
docker build --build-arg VLLM_CPU_AMXBF16=0 -f docker/Dockerfile.cpu .
```

### AVX-512 Kernel Selection

At runtime, vLLM selects the appropriate kernel library based on CPU capabilities:

```python
@classmethod
def import_kernels(cls) -> None:
    if Platform.get_cpu_architecture() in (CpuArchEnum.X86,):
        if torch._C._cpu._is_avx512_supported():
            import vllm._C        # AVX-512 optimized kernels
        else:
            import vllm._C_AVX2   # AVX2 fallback kernels
    else:
        import vllm._C            # Generic kernels for ARM/POWER/Z
```

## Attention Backend

The CPU platform uses a dedicated CPU attention backend:

```python
@classmethod
def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
    if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
        logger.info("Cannot use %s backend on CPU.", selected_backend)
    if attn_selector_config.use_mla:
        raise NotImplementedError("MLA is not supported on CPU.")
    if attn_selector_config.use_sparse:
        raise NotImplementedError("Sparse Attention is not supported on CPU.")
    return AttentionBackendEnum.CPU_ATTN.get_path()
```

> **Note**: MLA (Multi-head Latent Attention) and Sparse Attention are not supported on CPU.

## KV Cache Memory Management

The CPU platform uses system RAM for KV cache. Memory allocation is controlled by `VLLM_CPU_KVCACHE_SPACE`:

```python
@classmethod
def get_device_total_memory(cls, device_id: int = 0) -> int:
    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE
    if kv_cache_space is None:
        # Default: 50% of total RAM divided by NUMA nodes
        num_numa_nodes = len(nodes) or 1
        free_cpu_memory = psutil.virtual_memory().total // num_numa_nodes
        kv_cache_space = int(free_cpu_memory * 0.5)
    else:
        kv_cache_space *= GiB_bytes
    return kv_cache_space
```

Set KV cache space explicitly:

```bash
VLLM_CPU_KVCACHE_SPACE=32 vllm serve meta-llama/Llama-3.1-8B  # 32 GiB
```

## NUMA-Aware CPU Binding

The CPU platform supports NUMA-aware thread binding for optimal memory locality:

```python
# Bind to specific CPU cores
VLLM_CPU_OMP_THREADS_BIND=0-15 vllm serve ...

# Disable binding (let OS schedule)
VLLM_CPU_OMP_THREADS_BIND=nobind vllm serve ...
```

The `get_allowed_cpu_core_node_list()` method uses `lscpu` to enumerate logical CPUs, physical cores, and NUMA nodes, then filters by `CPU_VISIBLE_MEMORY_NODES`.

## OpenMP Configuration

The CPU platform configures OpenMP for optimal performance:

```python
os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # Disable async compilation
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"  # No streams on CPU
```

For Intel OpenMP (`libiomp5.so`):

```python
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_TPAUSE"] = "0"
os.environ["KMP_FORKJOIN_BARRIER_PATTERN"] = "dist,dist"
```

## ARM/POWER libgomp Workaround

On ARM and POWER platforms, PyTorch's `libgomp` must be preloaded to ensure all CPU cores are utilized:

```python
# Automatically adds PyTorch's libgomp to LD_PRELOAD on ARM/POWER
if platform.system() == "Linux" and get_cpu_architecture() in (CpuArchEnum.ARM, CpuArchEnum.POWERPC):
    pytorch_libgomp_so = find_pytorch_libgomp()
    os.environ["LD_PRELOAD"] += ":" + pytorch_libgomp_so
```

## Compilation Backend

The CPU platform uses `torch.compile` with the `inductor` backend:

```python
compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
compilation_config.backend = "inductor"  # or "eager" in CI
compilation_config.inductor_compile_config.update({
    "dce": True,
    "size_asserts": False,
    "nan_asserts": False,
    "epilogue_fusion": True,
})
```

In CI environments (`VLLM_CPU_CI_ENV=1`), eager mode is used to avoid long compilation times.

## Distributed Inference

The CPU platform uses Gloo for distributed communication:

```python
dist_backend: str = "gloo"
```

Multi-process execution uses the `mp` (multiprocessing) backend:

```python
parallel_config.distributed_executor_backend = "mp"
parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"
```

> **Note**: Ray distributed executor is not supported on CPU — it falls back to `mp` automatically.

## Block Size

The default KV cache block size on CPU is 128 tokens (optimized for cache line alignment):

```python
if cache_config.block_size is None:
    cache_config.block_size = 128
```

Blocks should be multiples of 32 for optimal performance.

## Inference Mode

The CPU platform uses `torch.no_grad()` instead of `torch.inference_mode()`:

```python
@classmethod
def inference_mode(cls):
    return torch.no_grad()
```

## Limitations

| Feature | Status |
|---------|--------|
| MLA Attention | ❌ Not supported |
| Sparse Attention | ❌ Not supported |
| FP8 KV Cache | ❌ Not supported (falls back to auto) |
| Chunked Prefill + FP8 KV | ❌ Incompatible |
| CUDA Graphs | ❌ Not applicable |
| Pin Memory | ❌ Not available |
| Dual-Batch Overlap (DBO) | ❌ Disabled |
| Async Scheduling | ❌ Disabled |

## Docker Images

### x86_64 / ARM64 (Unified)

```bash
# x86_64
docker build -f docker/Dockerfile.cpu -t vllm-cpu .

# ARM64 (AArch64)
docker buildx build --platform=linux/arm64 -f docker/Dockerfile.cpu -t vllm-cpu-arm .

# Disable AVX-512 (for older x86 CPUs)
docker build --build-arg VLLM_CPU_DISABLE_AVX512=true -f docker/Dockerfile.cpu .
```

### ppc64le (IBM POWER)

```bash
podman build -t cpu-test-ubi9-ppc -f docker/Dockerfile.ppc64le .
```

The ppc64le Dockerfile uses Red Hat UBI9 as the base image and builds OpenBLAS from source targeting POWER9.

### s390x (IBM Z)

```bash
docker build -t cpu-test -f docker/Dockerfile.s390x .
```

The s390x Dockerfile uses Red Hat UBI9 and builds Apache Arrow from source.

## Related Pages

- [Platform Overview](overview.md)
- [Hardware CI](hardware-ci.md)
- [ARM/ppc64le/s390x](alt-arch.md)
