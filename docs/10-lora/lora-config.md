# LoRAConfig

`LoRAConfig` is the server-wide configuration class that controls how vLLM manages LoRA adapters. It is defined in `vllm/config/lora.py` and is passed to the engine at startup. All LoRA-related command-line arguments are ultimately stored in a `LoRAConfig` instance.

## Class Definition

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    max_lora_rank: MaxLoRARanks = 16
    max_loras: int = Field(default=1, ge=1)
    fully_sharded_loras: bool = False
    max_cpu_loras: int | None = None
    lora_dtype: torch.dtype | LoRADType = "auto"
    default_mm_loras: dict[str, str] | None = None
    enable_tower_connector_lora: bool = False
    specialize_active_lora: bool = False
```

## Fields

### `max_lora_rank`

**Type:** `Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]`  
**Default:** `16`

The maximum LoRA rank (`r`) that any loaded adapter may use. This value determines the size of the pre-allocated `lora_a_stacked` and `lora_b_stacked` tensors in each LoRA layer. Adapters with a rank greater than `max_lora_rank` will be rejected at load time.

Supported rank values: `1, 8, 16, 32, 64, 128, 256, 320, 512`.

> **Important:** Set `max_lora_rank` to the highest rank among all adapters you plan to serve. A higher rank increases GPU memory usage proportionally.

### `max_loras`

**Type:** `int`  
**Default:** `1`  
**Constraint:** `>= 1`

The maximum number of LoRA adapters that can be **active on the GPU simultaneously** (i.e., present in GPU memory and available for use in a single batch). This controls the number of GPU slots allocated in each LoRA layer's stacked weight tensors.

Setting `max_loras=4` means up to 4 different adapters can be used across the tokens in a single forward pass.

### `fully_sharded_loras`

**Type:** `bool`  
**Default:** `False`

By default, only half of the LoRA computation is sharded with tensor parallelism (the `lora_b` expand step is all-gathered). When `fully_sharded_loras=True`, both the shrink (`lora_a`) and expand (`lora_b`) steps are fully sharded across tensor-parallel ranks.

This can improve performance at high sequence lengths, high ranks, or large tensor-parallel sizes, at the cost of additional all-reduce communication.

### `max_cpu_loras`

**Type:** `int | None`  
**Default:** `None` (auto-set to `max_loras`)

The maximum number of LoRA adapters to keep in **CPU memory**. Must be `>= max_loras`. When using `LRUCacheLoRAModelManager`, adapters are loaded from disk into CPU memory first, then transferred to GPU slots as needed. The CPU cache acts as a second-level cache between disk and GPU.

```python
@model_validator(mode="after")
def _validate_lora_config(self) -> Self:
    if self.max_cpu_loras is None:
        self.max_cpu_loras = self.max_loras
    elif self.max_cpu_loras < self.max_loras:
        raise ValueError(
            f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
            f"max_loras ({self.max_loras})."
        )
    return self
```

### `lora_dtype`

**Type:** `torch.dtype | Literal["auto", "float16", "bfloat16"]`  
**Default:** `"auto"`

The data type for LoRA weight tensors (`lora_a_stacked`, `lora_b_stacked`). When set to `"auto"`, the dtype is inherited from the base model's dtype at validation time:

```python
def verify_with_model_config(self, model_config: ModelConfig):
    if self.lora_dtype in (None, "auto"):
        self.lora_dtype = model_config.dtype
    elif isinstance(self.lora_dtype, str):
        self.lora_dtype = getattr(torch, self.lora_dtype)
```

### `default_mm_loras`

**Type:** `dict[str, str] | None`  
**Default:** `None`

A dictionary mapping modality names to LoRA adapter paths. This is only applicable to multimodal models and is used when a model should always apply a specific LoRA adapter whenever a given modality (e.g., `"image"`) is present in the request.

> **Note:** If a request includes multiple modalities each with their own default LoRA, the default LoRAs are not applied because vLLM currently supports only one adapter per prompt.

### `enable_tower_connector_lora`

**Type:** `bool`  
**Default:** `False`

When `True`, enables LoRA support for the vision tower (encoder) and connector modules of multimodal models. This is an experimental feature currently supporting select models such as the Qwen VL series.

### `specialize_active_lora`

**Type:** `bool`  
**Default:** `False`

When `True`, separate CUDA graphs are captured for different counts of active LoRA adapters (powers of 2 up to `max_loras`). This can improve performance for variable LoRA usage patterns at the cost of increased startup time and memory usage. Only takes effect when CUDA graph specialization is enabled.

## Configuration via CLI

All `LoRAConfig` fields map to command-line arguments:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 4 \
  --max-cpu-loras 16 \
  --lora-dtype bfloat16 \
  --fully-sharded-loras
```

## Configuration via Python

```python
from vllm import LLM
from vllm.config import LoRAConfig

# Using LLM constructor kwargs (maps to LoRAConfig fields)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=4,
    max_cpu_loras=16,
    lora_dtype="bfloat16",
)
```

## Memory Sizing

The GPU memory consumed by LoRA weight slots can be estimated as:

```
memory_per_layer ≈ max_loras × max_lora_rank × (d_in + d_out) × sizeof(lora_dtype)
```

For a Llama-3.1-8B model with `max_loras=4`, `max_lora_rank=64`, `lora_dtype=float16`:
- Each attention projection (hidden_size=4096): `4 × 64 × (4096 + 4096) × 2 bytes ≈ 4 MB`
- Total across all LoRA-enabled layers: typically 100–500 MB depending on model architecture

## Validation Summary

| Constraint | Error |
|-----------|-------|
| `max_cpu_loras < max_loras` | `ValueError` |
| `lora_int_id < 1` | `ValueError` (in `LoRARequest`) |
| Adapter rank > `max_lora_rank` | `ValueError` (at load time) |
| DoRA adapter | `ValueError` (unsupported) |
| Adapter with bias | `ValueError` (unsupported) |

## Hash Computation

`LoRAConfig` contributes to the computation graph hash used for CUDA graph caching:

```python
def compute_hash(self) -> str:
    factors = [
        self.max_lora_rank,
        self.max_loras,
        self.fully_sharded_loras,
        self.lora_dtype,
        self.enable_tower_connector_lora,
    ]
    return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
```

Changing any of these fields requires re-capturing CUDA graphs.

## See Also

- [LoRARequest](lora-request.md) — Per-request adapter descriptor
- [LoRAModelManager](lora-model-manager.md) — Adapter caching and lifecycle
- [LoRA Layers](lora-layers.md) — How layers are patched
