# LoRA Configuration

`LoRAConfig` controls Low-Rank Adaptation (LoRA) adapter serving in vLLM. vLLM supports serving multiple LoRA adapters simultaneously, dynamically switching between them per request.

**Source:** `vllm/config/lora.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — LoRA section.

---

## Overview

LoRA adapters are small, trainable weight matrices added to a frozen base model. vLLM can serve multiple LoRA adapters simultaneously, selecting the appropriate adapter per request. This enables:

- **Multi-tenant serving** — Different users/tasks with different fine-tuned adapters
- **Task specialization** — Switch between adapters for different domains
- **Memory efficiency** — Share one base model across many adapters

```
Base Model (frozen) + LoRA Adapter A → Task A output
Base Model (frozen) + LoRA Adapter B → Task B output
```

---

## Enabling LoRA

### `enable_lora`

```
Type:    bool
Default: False
CLI:     --enable-lora
```

Enable LoRA adapter serving. Must be set to `True` to use any LoRA features.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64
```

---

## Capacity Configuration

### `max_loras`

```
Type:    int (≥ 1)
Default: 1
CLI:     --max-loras
```

Maximum number of LoRA adapters that can be active in a single batch. This determines how many different adapters can be served simultaneously.

- Higher values allow more concurrent adapters but use more GPU memory
- Each active adapter requires memory proportional to `max_lora_rank`

```bash
# Serve up to 8 different adapters simultaneously
vllm serve mymodel --enable-lora --max-loras 8
```

### `max_cpu_loras`

```
Type:    int | None
Default: None (same as max_loras)
CLI:     --max-cpu-loras
```

Maximum number of LoRA adapters to keep in CPU memory. Must be ≥ `max_loras`. Adapters in CPU memory can be swapped to GPU as needed.

Setting this higher than `max_loras` allows a larger pool of adapters to be available without keeping them all on GPU:

```bash
# Keep 4 adapters on GPU, 32 in CPU memory
vllm serve mymodel \
  --enable-lora \
  --max-loras 4 \
  --max-cpu-loras 32
```

---

## Rank Configuration

### `max_lora_rank`

```
Type:    Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
Default: 16
CLI:     --max-lora-rank
```

Maximum LoRA rank supported. This must be ≥ the rank of any adapter you intend to serve. Higher ranks provide more expressive adapters but use more memory and compute.

Common rank values:
- `8` — Lightweight adapters, minimal overhead
- `16` — Default, good balance
- `32` — More expressive, moderate overhead
- `64` — High-quality fine-tuning
- `128`+ — Very high-quality, significant overhead

```bash
vllm serve mymodel --enable-lora --max-lora-rank 64
```

---

## Data Type

### `lora_dtype`

```
Type:    Literal["auto", "float16", "bfloat16"]
Default: "auto"
CLI:     --lora-dtype
```

Data type for LoRA weight matrices. `"auto"` uses the base model's dtype.

```bash
vllm serve mymodel \
  --enable-lora \
  --lora-dtype float16
```

---

## Sharding

### `fully_sharded_loras`

```
Type:    bool
Default: False
CLI:     --fully-sharded-loras
```

By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling fully sharded LoRA shards all LoRA layers across all TP ranks.

**When to enable:** High sequence lengths, high LoRA ranks, or large tensor parallel sizes. Can improve throughput in these scenarios.

```bash
vllm serve mymodel \
  --enable-lora \
  --tensor-parallel-size 4 \
  --fully-sharded-loras \
  --max-lora-rank 128
```

---

## CUDA Graph Specialization

### `specialize_active_lora`

```
Type:    bool
Default: False
CLI:     --specialize-active-lora
```

Capture separate CUDA graphs for different counts of active LoRA adapters (powers of 2 up to `max_loras`). This removes the overhead of running LoRA ops when fewer adapters are active than the maximum.

**Trade-off:** Increased startup time and memory usage for better runtime performance with variable LoRA usage.

Only takes effect when `cudagraph_specialize_lora=True` in `CompilationConfig`.

---

## Multimodal LoRA

### `default_mm_loras`

```
Type:    dict[str, str] | None
Default: None
CLI:     --default-mm-loras
```

Dictionary mapping modality names to LoRA model paths. Used when a multimodal model always expects a specific LoRA adapter when a given modality is present.

```bash
vllm serve mymodel \
  --enable-lora \
  --default-mm-loras '{"image": "/path/to/image-lora", "audio": "/path/to/audio-lora"}'
```

**Note:** If a request provides multiple modalities each with their own LoRA, only one adapter is applied (current limitation).

### `enable_tower_connector_lora`

```
Type:    bool
Default: False
CLI:     --enable-tower-connector-lora
```

Enable LoRA support for the vision encoder (tower) and connector of multimodal models. Currently supports select models including the Qwen VL series.

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --enable-lora \
  --enable-tower-connector-lora
```

---

## Runtime LoRA Management

### Loading adapters at startup

Use the `--lora-modules` flag (server-level) to pre-load adapters:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules \
    sql-lora=/path/to/sql-lora \
    code-lora=/path/to/code-lora
```

### Dynamic adapter loading

With `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, adapters can be loaded/unloaded at runtime via the API:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
vllm serve mymodel --enable-lora
```

```python
# Load a new adapter
import requests
requests.post("http://localhost:8000/v1/load_lora_adapter", json={
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter"
})

# Unload an adapter
requests.post("http://localhost:8000/v1/unload_lora_adapter", json={
    "lora_name": "my-adapter"
})
```

### Using adapters in requests

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="sql-lora",  # Use the adapter name as the model
    messages=[{"role": "user", "content": "Write a SQL query to..."}]
)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `0` | Allow loading/unloading LoRA adapters at runtime |
| `VLLM_LORA_RESOLVER_CACHE_DIR` | `None` | Local directory for unrecognized LoRA adapters |
| `VLLM_LORA_RESOLVER_HF_REPO_LIST` | `None` | Comma-separated HF repos containing LoRA adapters |
| `VLLM_LORA_DISABLE_PDL` | `0` | Disable LoRA PDL (Persistent Data Layout) |

---

## Memory Estimation

LoRA adapter memory per adapter:

```
bytes = 2 × max_lora_rank × (hidden_size + intermediate_size) × num_layers × dtype_bytes
```

For Llama-3.1-8B (hidden=4096, intermediate=14336, 32 layers, rank=64, FP16):
```
≈ 2 × 64 × (4096 + 14336) × 32 × 2 ≈ 4.7 GB per adapter
```

In practice, vLLM uses unified memory pools and the actual overhead is lower.

---

## Complete Example

```bash
# Serve with 4 concurrent LoRA adapters, rank 64, fully sharded
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-cpu-loras 16 \
  --max-lora-rank 64 \
  --fully-sharded-loras \
  --lora-dtype bfloat16 \
  --tensor-parallel-size 2 \
  --lora-modules \
    sql-adapter=/adapters/sql \
    code-adapter=/adapters/code \
    math-adapter=/adapters/math \
    chat-adapter=/adapters/chat
```
