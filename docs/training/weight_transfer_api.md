# Weight Transfer API

The weight transfer subsystem lets a **trainer process** push updated model
weights into running vLLM inference workers without restarting the server.
This is the core primitive that enables online reinforcement learning (RL)
training loops where the policy model is updated after every rollout batch.

---

## Overview

```
Trainer process                    vLLM inference workers
──────────────                     ──────────────────────
model.named_parameters()
        │
        ▼
trainer_send_weights()  ──────►  receive_weights()
  (NCCL broadcast /               (load into model)
   CUDA IPC handles)
```

The API is split into two sides:

- **Trainer side** — calls `trainer_send_weights()` (a static method on the
  engine class) to push tensors out.
- **Worker side** — calls `init_weight_transfer_engine()` once to set up the
  channel, then `update_weights()` for each weight-sync step.

Both sides must agree on the same **backend** (`nccl` or `ipc`) and the same
buffer parameters (for NCCL packed mode).

---

## Configuration

### `WeightTransferConfig`

```python
from vllm.config import WeightTransferConfig

config = WeightTransferConfig(backend="nccl")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | `"nccl"` \| `"ipc"` | `"nccl"` | Transport backend to use |

Pass the config when constructing the `LLM` object or the vLLM server:

=== "Python API"

    ```python
    from vllm import LLM
    from vllm.config import WeightTransferConfig

    llm = LLM(
        model="facebook/opt-125m",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )
    ```

=== "HTTP server"

    ```bash
    VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
        --weight-transfer-config '{"backend": "nccl"}'
    ```

---

## Worker-Side API

### `LLM.init_weight_transfer_engine()`

Initialise the weight transfer channel on all inference workers. Call this
**once** before the first weight update.

```python
llm.init_weight_transfer_engine(request)
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `request` | `WeightTransferInitRequest` \| `dict` | Backend-specific initialisation info |

The `init_info` dict inside the request is backend-specific — see the
[NCCL backend](#nccl-backend) and [IPC backend](#ipc-backend) sections below.

---

### `LLM.update_weights()`

Trigger weight reception on all inference workers. The workers will block
until all weights have been received and loaded into the model.

```python
llm.update_weights(request)
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `request` | `WeightTransferUpdateRequest` \| `dict` | Backend-specific weight metadata |

The `update_info` dict inside the request is backend-specific.

---

### `LLM.sleep()` / `LLM.wake_up()`

Free GPU memory while the trainer runs, then reclaim it before the next
rollout phase.

```python
# Free GPU memory before training step
llm.sleep(level=1)

# ... trainer runs backward pass ...

# Restore GPU memory before next rollout
llm.wake_up(tags=["weights", "kv_cache", "scheduling"])
```

**`sleep()` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `int` | `1` | Sleep level (0, 1, or 2) |
| `mode` | `"abort"` \| `"wait"` \| `"keep"` | `"abort"` | How to handle in-flight requests |

**Sleep levels**

| Level | Effect |
|---|---|
| `0` | Pause scheduling; requests queue but are not processed |
| `1` | Offload model weights to CPU; discard KV cache |
| `2` | Discard all GPU memory (weights + KV cache) |

**`wake_up()` parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tags` | `list[str]` \| `None` | `None` | Memory tags to restore. `None` restores all. Valid values: `"weights"`, `"kv_cache"`, `"scheduling"` |

---

## HTTP Server API

When running vLLM as an HTTP server with `VLLM_SERVER_DEV_MODE=1`, the
following endpoints are available on the same port as the OpenAI-compatible API.

### `POST /pause`

Pause generation before a weight update.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `"abort"` \| `"wait"` \| `"keep"` | `"abort"` | How to handle in-flight requests |
| `clear_cache` | `bool` | `true` | Whether to clear KV/prefix caches after draining (deprecated) |

**Response**

```json
{"status": "paused"}
```

---

### `POST /resume`

Resume generation after a weight update.

**Response**

```json
{"status": "resumed"}
```

---

### `GET /is_paused`

Query the current pause status.

**Response**

```json
{"is_paused": false}
```

---

### `POST /init_weight_transfer_engine`

Initialise the weight transfer channel on all workers.

**Request body**

```json
{
  "init_info": { ... }
}
```

The `init_info` object is backend-specific — see the sections below.

**Response**

```json
{"message": "Weight transfer initialized"}
```

---

### `POST /update_weights`

Trigger weight reception on all workers. The request blocks until all weights
have been received and loaded.

**Request body**

```json
{
  "update_info": { ... }
}
```

The `update_info` object is backend-specific.

**Response**

```json
{"message": "Weights updated"}
```

---

### `GET /get_world_size`

Query the inference world size (useful for computing NCCL rank offsets).

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_dp` | `bool` | `true` | If `true`, returns `TP × PP × DP`; if `false`, returns `TP × PP` |

**Response**

```json
{"world_size": 2}
```

---

## NCCL Backend

The NCCL backend uses **NCCL broadcast** operations to transfer weights from
the trainer (rank 0) to all inference workers. It works across separate GPUs
and across nodes.

### When to use

- Trainer and inference workers are on **different GPUs** (recommended for
  production).
- Multi-node setups where IPC is not available.
- You want maximum bandwidth utilisation via GPU-to-GPU NVLink / InfiniBand.

### Initialisation

**Trainer side**

```python
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

master_address = get_ip()
master_port = get_open_port()
world_size = inference_world_size + 1  # +1 for the trainer

# Trainer is always rank 0
model_update_group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

**Worker side (Python API)**

```python
llm.init_weight_transfer_engine(
    dict(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,        # workers start at rank 1
            world_size=world_size,
        )
    )
)
```

**Worker side (HTTP)**

```python
import requests

requests.post(
    "http://localhost:8000/init_weight_transfer_engine",
    json={
        "init_info": {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
    },
)
```

### `NCCLWeightTransferInitInfo` fields

| Field | Type | Description |
|---|---|---|
| `master_address` | `str` | IP address of the rendezvous host (typically the trainer) |
| `master_port` | `int` | TCP port for the rendezvous |
| `rank_offset` | `int` | First rank assigned to inference workers (usually `1`) |
| `world_size` | `int` | Total number of processes (trainer + all inference workers) |

### Sending weights (trainer side)

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLTrainerSendWeightsArgs,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=model_update_group,
    packed=True,   # recommended: batch tensors for efficiency
)

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

### Receiving weights (worker side)

```python
# Python API — triggers reception on all workers
llm.update_weights(
    dict(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)
```

```python
# HTTP — triggers reception on all workers
requests.post(
    "http://localhost:8000/update_weights",
    json={
        "update_info": {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
        }
    },
)
```

### `NCCLWeightTransferUpdateInfo` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `names` | `list[str]` | — | Parameter names (from `model.named_parameters()`) |
| `dtype_names` | `list[str]` | — | Dtype strings, e.g. `"bfloat16"` |
| `shapes` | `list[list[int]]` | — | Tensor shapes |
| `is_checkpoint_format` | `bool` | `True` | `True` if weights are in checkpoint format and need layer-wise processing |
| `packed` | `bool` | `False` | Enable packed tensor broadcasting (recommended) |
| `packed_buffer_size_bytes` | `int` | `1073741824` (1 GB) | Buffer size per packed broadcast |
| `packed_num_buffers` | `int` | `2` | Number of double-buffering slots |

### `NCCLTrainerSendWeightsArgs` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `group` | `PyNcclCommunicator` | — | NCCL communicator returned by `trainer_init()` |
| `src` | `int` | `0` | Source rank (trainer is always rank 0) |
| `post_iter_func` | `Callable` \| `None` | `None` | Optional transform applied to each `(name, tensor)` before broadcasting |
| `packed` | `bool` | `False` | Enable packed tensor broadcasting |
| `stream` | `torch.cuda.Stream` \| `None` | `None` | CUDA stream (unpacked mode only) |
| `packed_buffer_size_bytes` | `int` | `1073741824` | Must match `NCCLWeightTransferUpdateInfo.packed_buffer_size_bytes` |
| `packed_num_buffers` | `int` | `2` | Must match `NCCLWeightTransferUpdateInfo.packed_num_buffers` |

### Packed tensor broadcasting

By default, each tensor is broadcast individually. With `packed=True`, multiple
tensors are concatenated into a single large buffer before each broadcast,
significantly reducing NCCL call overhead for models with many small parameters.

```
Without packing:  broadcast(w1) → broadcast(w2) → broadcast(w3) → ...
With packing:     broadcast([w1 | w2 | w3 | ...])  (up to buffer_size_bytes)
```

Both `trainer_send_weights()` and `update_weights()` must use the same
`packed_buffer_size_bytes` and `packed_num_buffers` values.

---

## IPC Backend

The IPC backend uses **CUDA Inter-Process Communication (IPC) handles** to
share GPU memory between the trainer and inference workers **without copying**.
The trainer creates IPC handles for each weight tensor; the workers map those
handles directly into their address space.

### When to use

- Trainer and inference workers are **colocated on the same GPU**.
- You want zero-copy weight transfer with minimal overhead.
- Single-node setups where NCCL is not required.

!!! warning "Same-node requirement"
    CUDA IPC handles are only valid within the same physical node. For
    multi-node setups, use the NCCL backend instead.

!!! warning "Security"
    The HTTP transport for IPC uses `pickle` to serialise IPC handles.
    You must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to enable this.
    Never expose the vLLM HTTP server to untrusted networks when using IPC.

### Initialisation

The IPC backend requires no initialisation parameters. The `init_info` dict
is empty:

**Worker side (Python API)**

```python
llm.init_weight_transfer_engine(dict(init_info=dict()))
```

**Worker side (HTTP)**

```python
requests.post(
    "http://localhost:8000/init_weight_transfer_engine",
    json={"init_info": {}},
)
```

### Sending weights (trainer side)

**Ray mode** — trainer and workers communicate via Ray RPC:

```python
import ray
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCTrainerSendWeightsArgs,
)

trainer_args = IPCTrainerSendWeightsArgs(
    mode="ray",
    llm_handle=llm,   # Ray actor handle to the LLM
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

**HTTP mode** — trainer sends IPC handles via HTTP POST:

```python
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCTrainerSendWeightsArgs,
)

trainer_args = IPCTrainerSendWeightsArgs(
    mode="http",
    url="http://localhost:8000",
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

### `IPCTrainerSendWeightsArgs` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `"ray"` \| `"http"` | — | Transport mode |
| `llm_handle` | Ray actor handle | `None` | Required for `mode="ray"` |
| `url` | `str` | `None` | Base URL, e.g. `"http://localhost:8000"`. Required for `mode="http"` |

### `IPCWeightTransferUpdateInfo` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `names` | `list[str]` | — | Parameter names |
| `dtype_names` | `list[str]` | — | Dtype strings |
| `shapes` | `list[list[int]]` | — | Tensor shapes |
| `ipc_handles` | `list[dict]` | `None` | IPC handles (Ray transport) |
| `ipc_handles_pickled` | `str` | `None` | Base64-encoded pickled IPC handles (HTTP transport) |
| `is_checkpoint_format` | `bool` | `True` | Whether weights are in checkpoint format |

!!! note
    Exactly one of `ipc_handles` or `ipc_handles_pickled` must be provided.
    The `trainer_send_weights()` method populates the correct field automatically
    based on the selected `mode`.

---

## Engine Factory

The `WeightTransferEngineFactory` manages engine registration and instantiation.
It uses lazy loading so that engine modules are only imported when actually needed.

```python
from vllm.distributed.weight_transfer import WeightTransferEngineFactory

# Create an engine from config
engine = WeightTransferEngineFactory.create_engine(
    config=WeightTransferConfig(backend="nccl"),
    parallel_config=parallel_config,
)
```

### Registering a custom backend

You can register your own weight transfer backend at runtime:

```python
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
from mypackage.rdma_engine import RDMAWeightTransferEngine

WeightTransferEngineFactory.register_engine(
    "rdma",
    RDMAWeightTransferEngine,
)
```

Or with lazy loading (the module is only imported when the engine is first used):

```python
WeightTransferEngineFactory.register_engine(
    "rdma",
    "mypackage.rdma_engine",
    "RDMAWeightTransferEngine",
)
```

Then use it like any built-in backend:

```python
llm = LLM(
    model="...",
    weight_transfer_config=WeightTransferConfig(backend="rdma"),
)
```

---

## Base Classes

### `WeightTransferEngine`

Abstract base class for all weight transfer backends. Subclass this to
implement a custom backend.

```python
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from dataclasses import dataclass
from collections.abc import Callable, Iterator
import torch

@dataclass
class MyInitInfo(WeightTransferInitInfo):
    endpoint: str

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # Set up your transport channel
        ...

    def receive_weights(
        self,
        update_info: MyUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        # Receive tensors and call load_weights() incrementally
        ...

    def shutdown(self) -> None:
        ...

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict | Any,
    ) -> None:
        # Send tensors from the trainer side
        ...
```

**Abstract methods**

| Method | Description |
|---|---|
| `init_transfer_engine(init_info)` | Set up the transport channel (called once) |
| `receive_weights(update_info, load_weights)` | Receive tensors and load them into the model |
| `shutdown()` | Clean up resources |
| `trainer_send_weights(iterator, trainer_args)` | Static — send tensors from the trainer |

**Helper methods**

| Method | Description |
|---|---|
| `parse_init_info(init_dict)` | Deserialise `init_info` dict into `TInitInfo` |
| `parse_update_info(update_dict)` | Deserialise `update_info` dict into `TUpdateInfo` |

---

## Complete Examples

### Separate GPUs — NCCL (Ray)

This example runs the trainer on GPU 0 and the vLLM inference engine on GPUs 1–2
using Ray for process management.

```python
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

MODEL = "facebook/opt-125m"


@ray.remote(num_gpus=1)
class TrainModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda:0")
        self.port = get_open_port()
        self.address = get_ip()

    def get_address_and_port(self):
        return self.address, self.port

    def get_weight_metadata(self):
        names, dtype_names, shapes = [], [], []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_nccl_group(self, world_size):
        self.group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.address,
                master_port=self.port,
                world_size=world_size,
            )
        )

    def broadcast_weights(self):
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self.group, packed=True),
        )


ray.init()

trainer = TrainModel.remote()

pg = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg.ready())

llm = ray.remote(num_cpus=0, num_gpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
)(LLM).remote(
    model=MODEL,
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
)

# --- Rollout phase ---
outputs = ray.get(llm.generate.remote(["Hello, my name is"], SamplingParams(temperature=0)))

# --- Weight sync ---
ray.get(llm.sleep.remote(level=0))

address, port = ray.get(trainer.get_address_and_port.remote())
world_size = ray.get(llm.get_world_size.remote()) + 1

# Initialise NCCL on both sides simultaneously
ray.get([
    llm.init_weight_transfer_engine.remote(
        dict(init_info=dict(
            master_address=address,
            master_port=port,
            rank_offset=1,
            world_size=world_size,
        ))
    ),
    trainer.init_nccl_group.remote(world_size),
])

names, dtype_names, shapes = ray.get(trainer.get_weight_metadata.remote())

ray.get([
    llm.update_weights.remote(
        dict(update_info=dict(names=names, dtype_names=dtype_names,
                              shapes=shapes, packed=True))
    ),
    trainer.broadcast_weights.remote(),
])

ray.get(llm.wake_up.remote(tags=["scheduling"]))

# --- Next rollout phase with updated weights ---
outputs = ray.get(llm.generate.remote(["Hello, my name is"], SamplingParams(temperature=0)))
```

---

### Colocated GPU — IPC (Ray)

This example colocates the trainer and inference engine on the same GPU using
CUDA IPC for zero-copy weight transfer.

```python
import os
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCTrainerSendWeightsArgs,
)

MODEL = "facebook/opt-125m"


class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        super().__init__(*args, **kwargs)


@ray.remote
class TrainModel:
    def __init__(self, llm_handle):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda:0")
        self.llm_handle = llm_handle

    def init_weight_transfer(self):
        ray.get(
            self.llm_handle.init_weight_transfer_engine.remote(
                dict(init_info=dict())
            )
        )

    def broadcast_weights(self):
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            trainer_args=IPCTrainerSendWeightsArgs(
                mode="ray", llm_handle=self.llm_handle
            ),
        )


ray.init()

pg = placement_group([{"GPU": 1, "CPU": 0}])
ray.get(pg.ready())

llm = ray.remote(num_cpus=0, num_gpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
    )
)(MyLLM).remote(
    model=MODEL,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.7,
    weight_transfer_config=WeightTransferConfig(backend="ipc"),
    load_format="dummy",
)

trainer = TrainModel.options(
    num_gpus=0.1, num_cpus=0,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
    ),
).remote(llm)

# --- Rollout phase ---
outputs = ray.get(llm.generate.remote(["Hello"], SamplingParams(temperature=0)))

# --- Weight sync ---
ray.get(llm.sleep.remote(level=0))
ray.get(trainer.init_weight_transfer.remote())
ray.get(trainer.broadcast_weights.remote())
ray.get(llm.wake_up.remote(tags=["scheduling"]))

# --- Next rollout phase ---
outputs = ray.get(llm.generate.remote(["Hello"], SamplingParams(temperature=0)))
```

---

### HTTP Server — NCCL

Start the server:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
    --enforce-eager \
    --weight-transfer-config '{"backend": "nccl"}' \
    --load-format dummy
```

Trainer script:

```python
import threading
import requests
from transformers import AutoModelForCausalLM
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLTrainerSendWeightsArgs,
)
from vllm.utils.network_utils import get_ip, get_open_port

BASE_URL = "http://localhost:8000"
MODEL = "facebook/opt-125m"

# Query inference world size
world_size = requests.get(f"{BASE_URL}/get_world_size").json()["world_size"] + 1

master_address = get_ip()
master_port = get_open_port()

# Load training model
train_model = AutoModelForCausalLM.from_pretrained(MODEL)
train_model.to(f"cuda:{world_size - 1}")

# Initialise NCCL on both sides simultaneously
def init_server():
    requests.post(f"{BASE_URL}/init_weight_transfer_engine", json={
        "init_info": {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
    })

t = threading.Thread(target=init_server)
t.start()

group = NCCLWeightTransferEngine.trainer_init(dict(
    master_address=master_address,
    master_port=master_port,
    world_size=world_size,
))
t.join()

# Pause generation, sync weights, resume
requests.post(f"{BASE_URL}/pause")

names, dtype_names, shapes = [], [], []
for name, p in train_model.named_parameters():
    names.append(name)
    dtype_names.append(str(p.dtype).split(".")[-1])
    shapes.append(list(p.shape))

def update_server():
    requests.post(f"{BASE_URL}/update_weights", json={
        "update_info": {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
        }
    }, timeout=300)

t = threading.Thread(target=update_server)
t.start()

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=train_model.named_parameters(),
    trainer_args=NCCLTrainerSendWeightsArgs(group=group, packed=True),
)
t.join()

requests.post(f"{BASE_URL}/resume")
```

---

### HTTP Server — IPC

Start the server (reduced GPU memory to leave room for the training model):

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    vllm serve facebook/opt-125m \
    --enforce-eager \
    --weight-transfer-config '{"backend": "ipc"}' \
    --load-format dummy \
    --gpu-memory-utilization 0.5
```

Trainer script:

```python
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import requests
from transformers import AutoModelForCausalLM
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCTrainerSendWeightsArgs,
)

BASE_URL = "http://localhost:8000"
MODEL = "facebook/opt-125m"

train_model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda:0")

# Initialise (no-op for IPC)
requests.post(f"{BASE_URL}/init_weight_transfer_engine", json={"init_info": {}})

# Pause, sync, resume
requests.post(f"{BASE_URL}/pause")

IPCWeightTransferEngine.trainer_send_weights(
    iterator=train_model.named_parameters(),
    trainer_args=IPCTrainerSendWeightsArgs(mode="http", url=BASE_URL),
)

requests.post(f"{BASE_URL}/resume")
```

---

## Troubleshooting

### NCCL hangs during initialisation

The NCCL rendezvous requires both the trainer and all inference workers to
connect simultaneously. Make sure you call `init_weight_transfer_engine()` on
the workers and `trainer_init()` on the trainer **at the same time** (e.g., in
parallel threads or Ray futures).

### `ValueError: Invalid weight transfer backend`

The backend name in `WeightTransferConfig` does not match any registered engine.
Built-in backends are `"nccl"` and `"ipc"`. Check for typos.

### IPC: `ValueError: IPC handle not found for GPU UUID`

The trainer and the inference worker are not on the same physical GPU. CUDA IPC
handles are node-local and GPU-specific. Use the NCCL backend for cross-GPU
setups.

### IPC: `ValueError: Refusing to deserialize ipc_handles_pickled`

Set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` in the environment of both the vLLM
server and the trainer script when using IPC over HTTP.

### `RuntimeError: NCCL weight transfer not initialized`

`receive_weights()` was called before `init_transfer_engine()`. Always call
`init_weight_transfer_engine()` (worker side) and `trainer_init()` (trainer
side) before the first `update_weights()` / `trainer_send_weights()` call.

### Packed buffer size mismatch

The `packed_buffer_size_bytes` and `packed_num_buffers` values in
`NCCLTrainerSendWeightsArgs` must exactly match those in
`NCCLWeightTransferUpdateInfo`. A mismatch causes the producer and consumer to
disagree on buffer boundaries, leading to data corruption or hangs.

---

## See Also

- [Training Integration Overview](index.md)
- [TRL Integration](trl.md)
- [RLHF Overview](rlhf.md)
- [Distributed Inference](../design/distributed_inference.md)
- [Sleep Mode](../features/sleep_mode.md)
