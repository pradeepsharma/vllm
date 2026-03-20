# Disaggregated Prefilling

!!! note "Experimental"
    This feature is experimental and subject to change.

Disaggregated prefilling separates the **prefill** and **decode** phases of LLM inference into distinct vLLM instances. A dedicated prefill instance processes the prompt and computes KV cache, then transfers that KV cache to a decode instance which generates tokens.

---

## Why Disaggregated Prefilling?

### Separate TTFT and ITL Tuning

In a standard vLLM deployment, prefill and decode share the same GPU resources. This creates a tension: optimising for low time-to-first-token (TTFT) can hurt inter-token latency (ITL) and vice versa.

Disaggregated prefilling lets you:

- Assign different parallelism strategies (TP, PP) to prefill and decode instances independently.
- Scale prefill capacity separately from decode capacity.
- Tune TTFT without affecting ITL, and vice versa.

### Eliminate Prefill Interruptions

Without disaggregated prefilling, vLLM may interleave prefill jobs with ongoing decode steps. This causes "prefill spikes" that increase tail ITL. Disaggregated prefilling eliminates this by ensuring decode instances never process prefill requests.

!!! note
    Disaggregated prefilling does **not** improve throughput. It trades throughput for more predictable latency.

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Router                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Prefill Instance  │
        │  (vLLM + Connector) │
        │                     │
        │  1. Run prefill     │
        │  2. Save KV cache   │
        └──────────┬──────────┘
                   │  KV Transfer
                   │  (NCCL / NIXL / LMCache / etc.)
        ┌──────────▼──────────┐
        │   Decode Instance   │
        │  (vLLM + Connector) │
        │                     │
        │  1. Load KV cache   │
        │  2. Run decode      │
        └─────────────────────┘
```

### Connector Architecture (v1)

Every vLLM process involved in disaggregated prefilling has a **KV Connector** with two roles:

**Scheduler Connector** (runs in the scheduler process):
- `get_num_new_matched_tokens()` — query how many tokens exist in the remote KV cache
- `update_state_after_alloc()` — update state after KV buffer allocation
- `update_connector_output()` — update state after worker-side results arrive
- `request_finished()` — called when a request completes; optionally takes ownership of KV blocks for async transfer
- `take_events()` — return new KV cache events since the last call

**Worker Connector** (runs in each GPU worker process):
- `start_load_kv()` — begin loading KV cache from the connector (may be async)
- `wait_for_layer_load(layer_name)` — block until a specific layer's KV is loaded
- `save_kv_layer()` — begin saving KV for a layer (may be async)
- `wait_for_save()` — block until all saves complete
- `get_finished()` — return IDs of requests that have completed async transfer

### Layer-by-Layer Pipeline

The worker connector integrates with the attention module to enable layer-by-layer KV pipelining:

```
Forward pass (prefill instance):
  Layer 0 attention → save_kv_layer(0) → Layer 1 attention → save_kv_layer(1) → ...

Forward pass (decode instance):
  start_load_kv() → Layer 0 attention ← wait_for_layer_load(0) → Layer 1 ...
```

This overlap of KV transfer with computation reduces end-to-end latency.

---

## Available Connectors

vLLM ships with six production-ready connectors:

### ExampleConnector

A reference implementation using shared local storage. Suitable for single-node testing.

```bash
# See examples/offline_inference/disaggregated-prefill-v1/run.sh
--kv-transfer-config '{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"/tmp/kv_storage"}}'
```

### NixlConnector

Fully asynchronous send/receive using the NIXL transport library. Supports UCX and GDS backends for high-performance transfers.

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda"}'

# With specific backends:
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX","GDS"]}}'
```

For detailed setup, see [NixlConnector Usage Guide](nixl_connector_usage.md).

### P2pNcclConnector

Peer-to-peer KV transfer using NCCL. Good for GPU-to-GPU transfers within a cluster.

```bash
# See examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_both"}'
```

### LMCacheConnectorV1

Integration with [LMCache](https://github.com/LMCache/LMCache) for KV cache management. Uses NIXL for the underlying transport.

```bash
# See examples/others/lmcache/disagg_prefill_lmcache_v1/
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### MooncakeConnector

Integration with Moonshot AI's Mooncake KV transfer system.

```bash
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_both"}'
```

For detailed setup, see [MooncakeConnector Usage Guide](mooncake_connector_usage.md).

### OffloadingConnector

Offloads KV data to CPU memory. Useful for reducing GPU memory pressure without a separate prefill instance.

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":1000000000}}'
```

### MultiConnector

Chains multiple connectors together. Useful for combining, e.g., NIXL for remote transfer and a local offloading connector.

```bash
--kv-transfer-config '{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {"kv_connector": "ExampleConnector", "kv_role": "kv_both",
       "kv_connector_extra_config": {"shared_storage_path": "local_storage"}}
    ]
  }
}'
```

---

## KV Transfer Configuration

The `--kv-transfer-config` argument accepts a JSON object with the following fields:

| Field | Description |
|-------|-------------|
| `kv_connector` | Connector class name (e.g., `"NixlConnector"`) |
| `kv_role` | Role of this instance: `"kv_producer"`, `"kv_consumer"`, or `"kv_both"` |
| `kv_buffer_device` | Device for KV buffer: `"cuda"` or `"cpu"` |
| `kv_connector_extra_config` | Connector-specific configuration dict |

### Roles

| Role | Description |
|------|-------------|
| `kv_producer` | Prefill instance — computes and sends KV cache |
| `kv_consumer` | Decode instance — receives and uses KV cache |
| `kv_both` | Instance acts as both producer and consumer |

---

## Usage Example

### Online Serving (Two Instances)

```bash
# Start the prefill instance
vllm serve meta-llama/Llama-3-8b \
    --port 8100 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Start the decode instance
vllm serve meta-llama/Llama-3-8b \
    --port 8200 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

A request router (e.g., nginx, custom proxy) directs incoming requests to the prefill instance, which processes the prompt and transfers the KV cache to the decode instance.

See [examples/online_serving/disaggregated_prefill.sh](../../examples/online_serving/disaggregated_prefill.sh) for a complete example.

### Offline Inference

```bash
# See examples/offline_inference/disaggregated-prefill-v1/run.sh
```

---

## Metrics and Observability

KV connectors expose Prometheus metrics for monitoring transfer performance. Connectors that implement `KVConnectorStats` report:

- Transfer throughput (bytes/second)
- Transfer latency (milliseconds)
- Queue depths
- Error counts

Metrics are collected per-worker and aggregated by the scheduler connector. Access them via the standard vLLM metrics endpoint:

```bash
curl http://localhost:8000/metrics | grep kv_transfer
```

---

## Implementation Details

### Code Location

All disaggregated prefilling code lives under `vllm/distributed/kv_transfer/`:

```
vllm/distributed/kv_transfer/
├── kv_transfer_state.py          # Global KV connector singleton
├── kv_connector/
│   ├── factory.py                # Connector instantiation
│   ├── v1/
│   │   ├── base.py               # KVConnectorBase_V1 abstract class
│   │   ├── nixl_connector.py     # NixlConnector
│   │   ├── p2p/                  # P2pNcclConnector
│   │   ├── mooncake/             # MooncakeConnector
│   │   ├── lmcache_connector.py  # LMCacheConnectorV1
│   │   ├── offloading_connector.py
│   │   ├── multi_connector.py    # MultiConnector
│   │   └── metrics.py            # Stats and Prometheus metrics
```

### Hybrid Memory Allocator (HMA)

Connectors that implement `SupportsHMA` can work with vLLM's hybrid memory allocator, which allows KV blocks to be allocated across GPU and CPU memory. The connector takes ownership of blocks asynchronously and signals completion via `get_finished()`.

### KV Events

The `take_events()` method returns `KVCacheEvent` objects that describe KV cache state changes (blocks stored, blocks evicted, etc.). These events can be consumed by external systems for cache-aware routing.

---

## Benchmarks

See [benchmarks/disagg_benchmarks](../../benchmarks/disagg_benchmarks) for disaggregated prefilling benchmarks.

---

## Third-Party Connector Development

vLLM relies on third-party connectors for production-grade disaggregated prefilling. Three implementation approaches are supported:

### 1. Fully-Customised Connector

Implement `KVConnectorBase_V1` directly. This gives maximum control but requires tracking vLLM API changes.

```python
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

class MyConnector(KVConnectorBase_V1):
    def start_load_kv(self, forward_context, **kwargs):
        # Load KV from your storage system
        ...

    def wait_for_layer_load(self, layer_name):
        # Block until layer is loaded
        ...

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        # Save KV to your storage system
        ...

    def wait_for_save(self):
        # Block until all saves complete
        ...
```

### 2. Database-Like Connector

Implement `LookupBuffer` with `insert` and `drop_select` semantics (similar to SQL). This is the v0 API.

### 3. Distributed P2P Connector

Implement `Pipe` with `send_tensor` and `recv_tensor` APIs, similar to `torch.distributed`. This is the v0 API.

!!! note
    New connectors should target the v1 API (`KVConnectorBase_V1`). The v0 API (`LookupBuffer`, `Pipe`) is maintained for backward compatibility.

---

## See Also

- [NixlConnector Usage Guide](nixl_connector_usage.md)
- [MooncakeConnector Usage Guide](mooncake_connector_usage.md)
- [Distributed Inference Design](../design/distributed_inference.md)
- [Multi-Node Setup](../deployment/multi_node.md)
