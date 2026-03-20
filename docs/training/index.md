# Training Integration

vLLM is designed not only for production inference but also as a high-performance
**rollout engine** inside reinforcement learning (RL) training pipelines. This section
covers how to integrate vLLM with training frameworks, how weights are transferred
from a trainer to the inference engine, and which third-party libraries already
provide ready-made integrations.

---

## Why Use vLLM for RL Training?

Online RL algorithms — GRPO, PPO, Online DPO, RLOO, and similar methods — require
the policy model to **generate completions** at every training step. These rollouts
are the most latency-sensitive part of the pipeline. vLLM accelerates them with:

- **Continuous batching** — maximises GPU utilisation across concurrent requests.
- **PagedAttention** — efficient KV-cache management reduces memory waste.
- **Tensor / pipeline / data parallelism** — scales to multi-GPU and multi-node
  clusters.
- **Sleep / wake-up mode** — frees GPU memory while the trainer runs, then
  reclaims it for the next rollout phase.
- **Native weight transfer** — pushes updated model weights from the trainer
  directly into the inference workers without restarting the server.

---

## Architecture Overview

A typical online RL loop with vLLM looks like this:

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Training Loop                        │
│                                                             │
│  ┌──────────────┐   weights    ┌──────────────────────────┐ │
│  │   Trainer    │ ──────────►  │   vLLM Inference Engine  │ │
│  │  (GPU 0)     │              │   (GPU 1 … N)            │ │
│  │              │ ◄──────────  │                          │ │
│  │  loss.backward│  rollouts   │  generate(prompts)       │ │
│  └──────────────┘              └──────────────────────────┘ │
│                                                             │
│  1. Trainer generates rollouts via vLLM                     │
│  2. Trainer computes rewards and updates weights            │
│  3. Updated weights are pushed to vLLM workers             │
│  4. Repeat                                                  │
└─────────────────────────────────────────────────────────────┘
```

vLLM supports two physical topologies:

| Topology | Description | Best for |
|---|---|---|
| **Separate GPUs** | Trainer and inference engine on different GPUs | Large models, stable throughput |
| **Colocated** | Trainer and inference engine share the same GPU(s) | Single-GPU setups, memory-constrained environments |

---

## Deployment Modes

### Offline (Programmatic) API

Use the [`LLM`](../api/llm.md) class directly inside a Python training script.
The trainer and inference engine run in the same process or via Ray actors.

```python
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)
```

See [Weight Transfer API](weight_transfer_api.md) for the full API reference.

### Online (HTTP Server) Mode

Start a vLLM server with the RLHF endpoints enabled, then drive weight updates
over HTTP from any training process — including processes on separate nodes.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve meta-llama/Llama-3.1-8B \
    --weight-transfer-config '{"backend": "nccl"}'
```

The server exposes additional REST endpoints under the same port:

| Endpoint | Method | Purpose |
|---|---|---|
| `/pause` | POST | Pause generation before a weight update |
| `/resume` | POST | Resume generation after a weight update |
| `/is_paused` | GET | Query pause status |
| `/init_weight_transfer_engine` | POST | Initialise the weight transfer channel |
| `/update_weights` | POST | Trigger weight reception on workers |
| `/get_world_size` | GET | Query the inference world size |

!!! note "Dev-mode flag"
    The RLHF HTTP endpoints are only available when
    `VLLM_SERVER_DEV_MODE=1` is set. This flag is intentionally opt-in
    because the endpoints expose privileged operations.

---

## Sleep / Wake-up Mode

When the trainer is running its backward pass, the inference engine is idle.
**Sleep mode** offloads GPU memory so the trainer has more headroom:

```python
# Before training step — free GPU memory
llm.sleep(level=1)   # offload weights to CPU, discard KV cache

# ... trainer runs backward pass and updates weights ...

# After training step — restore GPU memory
llm.wake_up(tags=["weights", "kv_cache", "scheduling"])
```

| Sleep level | Effect |
|---|---|
| `0` | Pause scheduling; requests queue but are not processed |
| `1` | Offload model weights to CPU; discard KV cache |
| `2` | Discard all GPU memory (weights + KV cache) |

!!! tip
    Use `level=0` when you want to pause generation mid-stream without
    losing in-flight requests. Use `level=1` or `level=2` for full
    training/inference alternation.

---

## Weight Transfer Backends

vLLM ships two built-in backends for pushing updated weights from the trainer
to the inference workers:

| Backend | Transport | Topology | Notes |
|---|---|---|---|
| `nccl` | NCCL broadcast | Separate or colocated GPUs | Recommended for multi-GPU setups |
| `ipc` | CUDA IPC handles | Same node only | Zero-copy; ideal for colocated single-GPU |

See [Weight Transfer API](weight_transfer_api.md) for detailed documentation of
both backends, configuration options, and code examples.

---

## Third-Party Integrations

### TRL (Transformers Reinforcement Learning)

[TRL](https://huggingface.co/docs/trl) provides first-class vLLM support for
online trainers (GRPO, Online DPO, RLOO, Nash-MD, XPO). Enable it with a single
flag:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",   # or "colocate"
)
```

See [TRL Integration](trl.md) for full details.

### RLHF Libraries

Many open-source RL libraries use vLLM for fast rollouts:

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [verl](https://github.com/volcengine/verl)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [ms-swift](https://github.com/modelscope/ms-swift)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Unsloth](https://github.com/unslothai/unsloth)

See [RLHF Overview](rlhf.md) for links to examples and notebooks.

---

## Quick-Start Examples

| Example | Backend | Topology |
|---|---|---|
| [RLHF — separate GPUs (NCCL)](../examples/offline_inference/rlhf.md) | NCCL | Separate |
| [RLHF — colocated GPUs (IPC)](../examples/offline_inference/rlhf_colocate.md) | IPC | Colocated |
| [RLHF — HTTP server + NCCL](../examples/online_serving/rlhf_http_nccl.md) | NCCL | Separate |
| [RLHF — HTTP server + IPC](../examples/online_serving/rlhf_http_ipc.md) | IPC | Colocated |
| [Async RLHF with weight sync](../examples/offline_inference/rlhf_async.md) | NCCL | Separate |

---

## In This Section

| Page | Description |
|---|---|
| [TRL Integration](trl.md) | Using vLLM with Hugging Face TRL for GRPO and other online trainers |
| [RLHF Overview](rlhf.md) | RLHF libraries and basic examples |
| [Weight Transfer API](weight_transfer_api.md) | Full API reference for the weight transfer subsystem |
