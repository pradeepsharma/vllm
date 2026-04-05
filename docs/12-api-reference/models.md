# GET /v1/models

The Models endpoint lists all models currently available on the vLLM server, including the base model and any loaded LoRA adapters.

**Source:** `vllm/entrypoints/openai/models/`

---

## Endpoint

```
GET /v1/models
```

No request body is required.

---

## Response Schema

The response is defined in `vllm/entrypoints/openai/engine/protocol.py` as `ModelList`.

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1714000000,
      "owned_by": "vllm",
      "root": "meta-llama/Llama-3.1-8B-Instruct",
      "parent": null,
      "max_model_len": 131072,
      "permission": [
        {
          "id": "modelperm-abc123",
          "object": "model_permission",
          "created": 1714000000,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    },
    {
      "id": "my-lora-adapter",
      "object": "model",
      "created": 1714000000,
      "owned_by": "vllm",
      "root": "meta-llama/Llama-3.1-8B-Instruct",
      "parent": "meta-llama/Llama-3.1-8B-Instruct",
      "max_model_len": 131072,
      "permission": [...]
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `object` | `string` | Always `"list"`. |
| `data` | `array` | List of `ModelCard` objects. |

### ModelCard Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Model identifier (the name used in API requests). |
| `object` | `string` | Always `"model"`. |
| `created` | `int` | Unix timestamp when the model was registered. |
| `owned_by` | `string` | Always `"vllm"`. |
| `root` | `string \| null` | The base model path. For LoRA adapters, this is the adapter path. |
| `parent` | `string \| null` | For LoRA adapters, the base model name. `null` for base models. |
| `max_model_len` | `int \| null` | Maximum context length supported by the model. |
| `permission` | `array` | List of `ModelPermission` objects describing allowed operations. |

### ModelPermission Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Permission ID (format: `modelperm-{uuid}`). |
| `object` | `string` | Always `"model_permission"`. |
| `allow_sampling` | `bool` | Whether sampling is allowed (always `true`). |
| `allow_logprobs` | `bool` | Whether log probabilities are allowed (always `true`). |
| `allow_view` | `bool` | Whether the model can be viewed (always `true`). |
| `organization` | `string` | Organization scope (always `"*"`). |

---

## Examples

### List Models with Python

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

models = client.models.list()
for model in models.data:
    print(f"ID: {model.id}")
    print(f"  Max context: {model.max_model_len}")
    print(f"  Parent: {model.parent or 'base model'}")
```

### Raw HTTP Request

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
```

### Example Output with LoRA Adapters

When the server is started with `--lora-modules`, the response includes both the base model and all loaded adapters:

```bash
# Start server with LoRA adapters
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --lora-modules sql-lora=/path/to/sql-lora \
  --lora-modules code-lora=/path/to/code-lora
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 131072,
      "parent": null
    },
    {
      "id": "sql-lora",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 131072,
      "parent": "meta-llama/Llama-3.1-8B-Instruct"
    },
    {
      "id": "code-lora",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 131072,
      "parent": "meta-llama/Llama-3.1-8B-Instruct"
    }
  ]
}
```

---

## Served Model Names

By default, the model is listed under its HuggingFace model ID. You can customize the name using `--served-model-name`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama-8b my-custom-name
```

This registers the model under multiple names, all pointing to the same weights. Clients can use any of the registered names in API requests.

---

## LoRA Module Configuration

LoRA adapters are registered at server startup using `--lora-modules`. Two formats are supported:

**Old format (name=path):**
```bash
vllm serve base-model --lora-modules adapter-name=/path/to/adapter
```

**New JSON format (with base model override):**
```bash
vllm serve base-model \
  --lora-modules '{"name": "adapter-name", "path": "/path/to/adapter", "base_model_name": "base-model"}'
```

The `base_model_name` field in the JSON format sets the `parent` field in the model listing response.

---

## Related Pages

- [CLI Arguments](cli-args.md) — `--served-model-name`, `--lora-modules` flags
- [POST /v1/chat/completions](chat-completions.md) — Using models in requests
- [Error Handling](error-handling.md) — HTTP status codes
