# Contributing a Hardware Platform

This guide explains how to add support for a new hardware accelerator (platform) to vLLM. vLLM uses a plugin-based platform abstraction that allows new hardware backends to be integrated without modifying the core codebase.

---

## Overview

vLLM's platform system provides a hardware abstraction layer through the `Platform` base class in `vllm/platforms/interface.py`. Each platform:

- Detects whether the hardware is available
- Declares supported quantization methods
- Provides hardware-specific attention backends
- Configures distributed communication backends
- Handles device-specific initialization

### Built-in Platforms

| Platform | Class | Hardware |
|---|---|---|
| CUDA | `CudaPlatform` | NVIDIA GPUs |
| ROCm | `RocmPlatform` | AMD GPUs |
| TPU | `TpuPlatform` | Google TPUs |
| XPU | `XPUPlatform` | Intel GPUs |
| CPU | `CpuPlatform` | x86/ARM CPUs, macOS |

### Out-of-Tree Platforms

New platforms can be contributed either:
1. **In-tree** — directly in `vllm/platforms/` (for widely-used hardware)
2. **Out-of-tree** — as a separate Python package using the plugin system (preferred for niche hardware)

---

## Architecture

### Platform Detection

vLLM uses a plugin system to detect the current platform at runtime. The detection flow is:

```
vllm.platforms.current_platform
    → resolve_current_platform_cls_qualname()
    → Load platform plugins (built-in + out-of-tree)
    → Each plugin function returns a class qualname or None
    → First activated plugin wins
```

Built-in platform plugins are registered in `vllm/platforms/__init__.py`:

```python
builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}
```

Out-of-tree platforms register via the `vllm.platform_plugins` entry point group.

---

## Step 1: Create the Platform Class

Create `vllm/platforms/myplatform.py` (or in your external package):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class MyPlatform(Platform):
    """Platform implementation for MyHardware accelerators."""

    _enum = PlatformEnum.OOT  # Use OOT for out-of-tree platforms
    device_name: str = "myplatform"
    device_type: str = "myplatform"

    # PyTorch dispatch key for custom ops
    # See: https://github.com/pytorch/pytorch/blob/main/torchgen/model.py
    dispatch_key: str = "CPU"  # Use "CPU" as fallback if not registered in PyTorch

    # Ray device key (empty string if Ray is not supported)
    ray_device_key: str = ""

    # Environment variable to control visible devices
    device_control_env_var: str = "MY_VISIBLE_DEVICES"

    # Supported quantization methods
    supported_quantization: list[str] = [
        "fp8",
        "gptq",
        "awq",
        # Add methods your hardware supports
    ]

    # Distributed communication backend
    dist_backend: str = "gloo"  # or "nccl", "xccl", etc.

    @classmethod
    def get_device_capability(
        cls, device_id: int = 0
    ) -> DeviceCapability | None:
        """Return the compute capability of the device."""
        # Return None if the concept doesn't apply to your hardware
        return DeviceCapability(major=1, minor=0)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Return the name of the device."""
        return "MyHardware Accelerator"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Return total device memory in bytes."""
        # Query your hardware's memory
        return 16 * 1024 ** 3  # 16 GB example

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """Whether async output processing is supported."""
        return False

    @classmethod
    def inference_mode(cls):
        """Context manager for inference mode (disables gradient computation)."""
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Validate and update the vLLM config for this platform.

        Called before the engine starts. Use this to:
        - Enforce hardware-specific constraints
        - Set default values appropriate for your hardware
        - Raise errors for unsupported configurations
        """
        parallel_config = vllm_config.parallel_config

        # Example: enforce single-GPU for now
        if parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "MyPlatform does not yet support tensor parallelism. "
                "Set tensor_parallel_size=1."
            )

        # Example: set a default dtype
        model_config = vllm_config.model_config
        if model_config.dtype == torch.float32:
            logger.warning(
                "float32 is not recommended on MyPlatform. "
                "Consider using bfloat16 for better performance."
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int,
        use_v1: bool,
    ) -> str:
        """Return the attention backend class for this platform."""
        # Return the fully-qualified class name of your attention backend
        return "my_package.attention.MyAttentionBackend"

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """Verify that the model architecture is supported on this platform."""
        supported_archs = [
            "LlamaForCausalLM",
            "MistralForCausalLM",
            # Add supported architectures
        ]
        if model_arch not in supported_archs:
            raise ValueError(
                f"Model architecture {model_arch!r} is not supported on "
                f"MyPlatform. Supported architectures: {supported_archs}"
            )
```

---

## Step 2: Implement the Attention Backend

The attention backend is the most critical component for a new platform. It implements the core attention computation.

Create `my_package/attention.py`:

```python
from dataclasses import dataclass
from typing import Any

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)


class MyAttentionBackend(AttentionBackend):
    """Attention backend for MyPlatform."""

    @staticmethod
    def get_name() -> str:
        return "MY_PLATFORM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MyAttentionImpl"]:
        return MyAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["MyAttentionMetadata"]:
        return MyAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MyAttentionMetadataBuilder"]:
        return MyAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Return the shape of the KV cache tensor."""
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap KV cache blocks between source and destination."""
        # Implement block swapping for your hardware
        ...

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy KV cache blocks."""
        # Implement block copying for your hardware
        ...


@dataclass
class MyAttentionMetadata(AttentionMetadata):
    """Metadata for MyPlatform attention."""
    # Add platform-specific metadata fields
    pass


class MyAttentionImpl(AttentionImpl):
    """Attention implementation for MyPlatform."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        blocksparse_params: dict | None = None,
        logits_soft_cap: float | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # ... store other parameters

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MyAttentionMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention for the given query, key, value tensors."""
        # Implement your hardware's attention computation
        # This is the core of the platform implementation
        ...
```

---

## Step 3: Register the Platform Plugin

### Option A: In-Tree Platform

For platforms contributed directly to vLLM, add a detection function in `vllm/platforms/__init__.py`:

```python
def myplatform_platform_plugin() -> str | None:
    is_myplatform = False
    logger.debug("Checking if MyPlatform is available.")
    try:
        import my_hardware_sdk
        if my_hardware_sdk.device_count() > 0:
            is_myplatform = True
            logger.debug("Confirmed MyPlatform is available.")
    except Exception as e:
        logger.debug("MyPlatform is not available because: %s", str(e))

    return "vllm.platforms.myplatform.MyPlatform" if is_myplatform else None


builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
    "myplatform": myplatform_platform_plugin,  # Add here
}
```

### Option B: Out-of-Tree Platform (Plugin Package)

For platforms distributed as separate packages, register via Python entry points.

In your package's `pyproject.toml`:

```toml
[project.entry-points."vllm.platform_plugins"]
myplatform = "my_package.platform:myplatform_plugin"
```

Where `myplatform_plugin` is a callable that returns the platform class qualname:

```python
# my_package/platform.py

def myplatform_plugin() -> str | None:
    """Detect if MyPlatform hardware is available."""
    try:
        import my_hardware_sdk
        if my_hardware_sdk.device_count() > 0:
            return "my_package.platform.MyPlatform"
    except ImportError:
        pass
    return None
```

---

## Step 4: Add Build Support

### Docker Image

Create a Dockerfile for your platform in `docker/`:

```dockerfile
# docker/Dockerfile.myplatform
FROM my_hardware_base_image:latest

# Install vLLM dependencies
COPY requirements/common.txt requirements/common.txt
RUN pip install -r requirements/common.txt

# Install platform-specific dependencies
RUN pip install my_hardware_sdk

# Install vLLM
COPY . /workspace/vllm
RUN cd /workspace/vllm && pip install -e .
```

### Build Script

Add a build script in `.buildkite/image_build/`:

```bash
#!/bin/bash
# .buildkite/image_build/image_build_myplatform.sh

REGISTRY=$1
REPO=$2
COMMIT=$3

docker build \
    --tag ${REGISTRY}/${REPO}:myplatform-${COMMIT} \
    --file docker/Dockerfile.myplatform \
    .

docker push ${REGISTRY}/${REPO}:myplatform-${COMMIT}
```

---

## Step 5: Add CI Tests

Create a hardware test configuration in `.buildkite/hardware_tests/myplatform.yaml`:

```yaml
group: MyPlatform Tests
steps:
  - label: "MyPlatform Basic Tests"
    agents:
      queue: myplatform_queue
    commands:
      - pytest -v -s tests/basic_correctness/test_basic_correctness.py
      - pytest -v -s tests/models/language -m "core_model"
    timeout_in_minutes: 60
    soft_fail: true  # Use soft_fail during initial development
```

---

## Step 6: Write Tests

Create platform-specific tests in `tests/`:

```python
# tests/platforms/test_myplatform.py
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.mark.skipif(
    not _is_myplatform_available(),
    reason="MyPlatform hardware not available"
)
class TestMyPlatform:
    def test_platform_detection(self):
        """Test that MyPlatform is correctly detected."""
        from vllm.platforms import current_platform
        assert current_platform.device_type == "myplatform"

    def test_basic_generation(self, vllm_runner):
        """Test basic text generation on MyPlatform."""
        with vllm_runner("facebook/opt-125m") as llm:
            outputs = llm.generate_greedy(["Hello, world!"], max_tokens=10)
        assert len(outputs) == 1
        assert len(outputs[0][1]) > 0

    def test_supported_quantization(self):
        """Test that supported quantization methods work."""
        from vllm.platforms import current_platform
        assert "fp8" in current_platform.supported_quantization
```

---

## Step 7: Update Documentation

1. **Add an installation guide** at `docs/getting_started/installation/myplatform.md`
2. **Update `docs/getting_started/installation/index.md`** to include your platform
3. **Add to the hardware tests section** in `docs/contributing/ci_cd.md`

---

## Platform Interface Reference

### Required Methods

| Method | Description |
|---|---|
| `get_device_capability()` | Return compute capability (or `None`) |
| `get_device_name()` | Return human-readable device name |
| `get_device_total_memory()` | Return total memory in bytes |
| `is_async_output_supported()` | Whether async output is supported |
| `inference_mode()` | Context manager for inference |
| `check_and_update_config()` | Validate/update VllmConfig |

### Optional Methods

| Method | Description |
|---|---|
| `get_attn_backend_cls()` | Return attention backend class |
| `verify_model_arch()` | Validate model architecture |
| `verify_quantization()` | Validate quantization method |
| `get_cpu_architecture()` | Return CPU architecture enum |
| `init_distributed_environment()` | Initialize distributed comms |

### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `_enum` | `PlatformEnum` | Platform enum value |
| `device_name` | `str` | Short device name |
| `device_type` | `str` | Device type string |
| `dispatch_key` | `str` | PyTorch dispatch key |
| `ray_device_key` | `str` | Ray device key |
| `device_control_env_var` | `str` | Visible devices env var |
| `supported_quantization` | `list[str]` | Supported quant methods |
| `dist_backend` | `str` | Distributed backend |
| `simple_compile_backend` | `str` | torch.compile backend |

---

## Common Challenges

### Attention Kernel Implementation

The attention kernel is typically the most hardware-specific component. Options:

1. **Use FlashAttention-compatible kernels** if your hardware supports them
2. **Implement a custom kernel** using your hardware's SDK
3. **Fall back to PyTorch SDPA** for correctness (lower performance)

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
    # Option 3: PyTorch SDPA fallback
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, scale=self.scale
    )
```

### KV Cache Management

Implement `swap_blocks` and `copy_blocks` for paged attention:

```python
@staticmethod
def swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst):
    """Move KV cache blocks between CPU and device memory."""
    src_key_cache, src_val_cache = src_kv_cache[0], src_kv_cache[1]
    dst_key_cache, dst_val_cache = dst_kv_cache[0], dst_kv_cache[1]
    
    for src_id, dst_id in src_to_dst.items():
        dst_key_cache[dst_id] = src_key_cache[src_id].to(dst_key_cache.device)
        dst_val_cache[dst_id] = src_val_cache[src_id].to(dst_val_cache.device)
```

### Distributed Communication

For multi-device support, configure the distributed backend:

```python
class MyPlatform(Platform):
    dist_backend: str = "gloo"  # or implement custom backend

    @classmethod
    def init_distributed_environment(
        cls,
        distributed_init_method: str,
        rank: int,
        local_rank: int,
        world_size: int,
    ) -> None:
        """Initialize the distributed process group."""
        import torch.distributed as dist
        dist.init_process_group(
            backend=cls.dist_backend,
            init_method=distributed_init_method,
            rank=rank,
            world_size=world_size,
        )
```

---

## Getting Help

- Study existing platform implementations: `vllm/platforms/cuda.py`, `vllm/platforms/rocm.py`
- Ask in `#hardware-platforms` on [vLLM Slack](https://slack.vllm.ai)
- Open a GitHub issue with the `[Platform]` prefix to discuss your implementation
- Look at out-of-tree platform examples in the vLLM ecosystem
