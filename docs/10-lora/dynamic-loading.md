# Dynamic LoRA Loading and Unloading

vLLM supports loading and unloading LoRA adapters at runtime via a REST API, without restarting the server. This is useful for development workflows where you want to test new adapters or swap out adapters without downtime.

> **Warning:** Dynamic LoRA loading is intended for **local development only**. It is disabled by default and must be explicitly enabled. It is not recommended for production deployments.

## Enabling Dynamic Loading

Set the `VLLM_ALLOW_RUNTIME_LORA_UPDATING` environment variable before starting the server:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --max-loras 4 \
  --max-cpu-loras 16
```

When enabled, the server logs a warning:
```
WARNING: LoRA dynamic loading & unloading is enabled in the API server.
This should ONLY be used for local development!
```

> **Note:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING` cannot be used with `--api-server-count > 1` (multiple API server processes), as adapter state would not be synchronized across processes.

## API Endpoints

The dynamic loading API is implemented in `vllm/entrypoints/serve/lora/api_router.py` and registered at startup when `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`.

### `POST /v1/load_lora_adapter`

Loads a new LoRA adapter into the server.

**Request Body:**

```json
{
  "lora_name": "sql-lora",
  "lora_path": "/path/to/sql-lora-adapter",
  "load_inplace": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | `string` | ✅ | Unique name for the adapter. Used as the `model` field in subsequent requests. |
| `lora_path` | `string` | ✅ | Filesystem path to the adapter directory (must contain `adapter_config.json`). |
| `load_inplace` | `boolean` | ❌ | If `true`, replaces an existing adapter with the same name. Default: `false`. |

**Success Response:** HTTP 200

```
Success: LoRA adapter 'sql-lora' added successfully.
```

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 400 Bad Request | Missing `lora_name` or `lora_path` |
| 400 Bad Request | Adapter with same name already loaded (and `load_inplace=false`) |
| 400 Bad Request | Adapter rank exceeds `max_lora_rank` |
| 404 Not Found | No adapter found at `lora_path` |

### `POST /v1/unload_lora_adapter`

Unloads a previously loaded LoRA adapter.

**Request Body:**

```json
{
  "lora_name": "sql-lora"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lora_name` | `string` | ✅ | Name of the adapter to unload. |
| `lora_int_id` | `integer \| null` | ❌ | Optional integer ID (not currently used for lookup). |

**Success Response:** HTTP 200

```
Success: LoRA adapter 'sql-lora' removed successfully.
```

**Error Responses:**

| Status | Condition |
|--------|-----------|
| 400 Bad Request | Missing `lora_name` |
| 404 Not Found | Adapter with given name not loaded |

## Usage Examples

### Loading an Adapter

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sql-lora",
    "lora_path": "/models/adapters/sql-lora"
  }'
```

### Using the Loaded Adapter

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sql-lora",
    "prompt": "SELECT * FROM users WHERE age >",
    "max_tokens": 50
  }'
```

### Replacing an Adapter In-Place

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sql-lora",
    "lora_path": "/models/adapters/sql-lora-v2",
    "load_inplace": true
  }'
```

### Unloading an Adapter

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sql-lora"
  }'
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Load adapter
response = requests.post(
    f"{BASE_URL}/v1/load_lora_adapter",
    json={
        "lora_name": "sql-lora",
        "lora_path": "/models/adapters/sql-lora",
    }
)
print(response.text)  # "Success: LoRA adapter 'sql-lora' added successfully."

# Use adapter
response = requests.post(
    f"{BASE_URL}/v1/completions",
    json={
        "model": "sql-lora",
        "prompt": "SELECT * FROM orders WHERE",
        "max_tokens": 50,
    }
)
print(response.json()["choices"][0]["text"])

# Unload adapter
response = requests.post(
    f"{BASE_URL}/v1/unload_lora_adapter",
    json={"lora_name": "sql-lora"}
)
print(response.text)  # "Success: LoRA adapter 'sql-lora' removed successfully."
```

## Protocol Classes

Defined in `vllm/entrypoints/serve/lora/protocol.py`:

```python
class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str
    load_inplace: bool = False

class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: int | None = Field(default=None)
```

## Implementation Details

The API router is defined in `vllm/entrypoints/serve/lora/api_router.py`:

```python
def attach_router(app: FastAPI):
    if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        return  # Do nothing if not enabled

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.error.code)
        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: UnloadLoRAAdapterRequest, raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        ...

    app.include_router(router)
```

The `OpenAIServingModels.load_lora_adapter()` method:
1. Acquires a per-adapter-name lock to prevent concurrent loads of the same adapter.
2. Validates the request (name and path must be non-empty; name must not already exist unless `load_inplace=True`).
3. Creates a `LoRARequest` with an auto-incremented integer ID.
4. Calls `engine_client.add_lora(lora_request)` to preload the adapter into the engine.
5. Stores the `LoRARequest` in `self.lora_requests` for future lookups.

## Preloading Adapters with `add_lora()`

Beyond the REST API, you can preload adapters programmatically using the async engine client:

```python
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.lora.request import LoRARequest

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=64,
)

async def main():
    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        # Preload adapters before requests arrive
        lora_requests = [
            LoRARequest(lora_name=f"adapter-{i}", lora_int_id=i,
                        lora_path=f"/models/adapter-{i}")
            for i in range(1, 5)
        ]
        results = await asyncio.gather(*[llm.add_lora(lr) for lr in lora_requests])
        print(f"Loaded: {results}")  # [True, True, True, True]

asyncio.run(main())
```

Preloading adapters before requests arrive eliminates the cold-start latency of loading from disk on the first request.

## Listing Loaded Adapters

The `/v1/models` endpoint lists all loaded adapters alongside the base model:

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "max_model_len": 131072
    },
    {
      "id": "sql-lora",
      "object": "model",
      "parent": "meta-llama/Llama-3.1-8B-Instruct"
    },
    {
      "id": "code-lora",
      "object": "model",
      "parent": "meta-llama/Llama-3.1-8B-Instruct"
    }
  ]
}
```

## SageMaker Integration

The dynamic loading endpoints are also registered with SageMaker's model hosting container standards via decorators:

```python
@sagemaker_standards.register_load_adapter_handler(
    request_shape={
        "lora_name": "body.name",
        "lora_path": "body.src",
        "load_inplace": "body.load_inplace || `false`",
    },
)
@router.post("/v1/load_lora_adapter")
async def load_lora_adapter(...):
    ...
```

This allows SageMaker's adapter management infrastructure to call the vLLM endpoints using SageMaker's standard request format.

## See Also

- [LoRARequest](lora-request.md) — The adapter descriptor
- [LoRAConfig](lora-config.md) — `max_loras`, `max_cpu_loras` configuration
- [LoRAModelManager](lora-model-manager.md) — LRU eviction when cache is full
- [LoRA Resolvers](lora-resolvers.md) — Automatic adapter discovery by name
