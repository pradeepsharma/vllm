# AWS SageMaker Deployment

vLLM provides native support for Amazon SageMaker's container hosting requirements. SageMaker expects two specific HTTP endpoints — `/ping` and `/invocations` — which vLLM implements via the `model_hosting_container_standards` library.

Source: `vllm/entrypoints/sagemaker/api_router.py`, `examples/online_serving/sagemaker-entrypoint.sh`

## SageMaker Container Standards

SageMaker imposes specific requirements on inference containers:

| Requirement | Detail |
|-------------|--------|
| **Port** | Container must listen on port `8080` |
| **`GET /ping`** | Health check — must return HTTP 200 when ready |
| **`POST /ping`** | Also accepted for health checks |
| **`POST /invocations`** | Inference endpoint — receives request body, returns predictions |
| **Content-Type** | Must accept `application/json` |

## The `vllm-sagemaker` Image

The `docker/Dockerfile` builds a dedicated `vllm-sagemaker` target:

```dockerfile
FROM vllm-openai-base AS vllm-sagemaker
COPY examples/online_serving/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]
```

The entrypoint script translates `SM_VLLM_*` environment variables into `vllm serve` CLI arguments and forces port `8080`:

```bash
#!/bin/bash
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Port 8080 required by SageMaker
ARGS=(--port 8080)

# Translate SM_VLLM_* env vars to CLI args
while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

exec standard-supervisor vllm serve "${ARGS[@]}"
```

### Environment Variable Mapping

Set `SM_VLLM_<OPTION>` to pass any `vllm serve` argument. The prefix is stripped, the name is lowercased, and underscores become dashes:

| Environment Variable | CLI Argument |
|---------------------|--------------|
| `SM_VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct` | `--model meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `SM_VLLM_TENSOR_PARALLEL_SIZE=4` | `--tensor-parallel-size 4` |
| `SM_VLLM_GPU_MEMORY_UTILIZATION=0.9` | `--gpu-memory-utilization 0.9` |
| `SM_VLLM_MAX_MODEL_LEN=8192` | `--max-model-len 8192` |
| `SM_VLLM_TRUST_REMOTE_CODE=` (empty value) | `--trust-remote-code` (flag only) |

## `/ping` Endpoint

The `/ping` endpoint is registered with `@sagemaker_standards.register_ping_handler` and internally calls the same `health()` function as the `/health` endpoint:

```python
@router.post("/ping", response_class=Response)
@router.get("/ping", response_class=Response)
@sagemaker_standards.register_ping_handler
async def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return await health(raw_request)
```

SageMaker calls `/ping` periodically to determine if the container is healthy. The endpoint returns:
- **HTTP 200** — container is healthy and ready to serve
- **HTTP 503** — engine is not ready (during startup or after an error)

## `/invocations` Endpoint

The `/invocations` endpoint routes requests to the appropriate handler based on the request body type:

```python
@router.post("/invocations", ...)
@sagemaker_standards.register_invocation_handler
@sagemaker_standards.stateful_session_manager()
@sagemaker_standards.inject_adapter_id(adapter_path="model")
async def invocations(raw_request: Request):
    """For SageMaker, routes requests based on the request type."""
    body = await raw_request.json()

    for request_validator, endpoint in valid_endpoints:
        try:
            request = request_validator.validate_python(body)
        except pydantic.ValidationError:
            continue
        return await endpoint(request, raw_request)
```

### Supported Request Types

The `/invocations` endpoint auto-detects the request type by attempting Pydantic validation in priority order:

| Priority | Request Type | Task |
|----------|-------------|------|
| 1 | `ChatCompletionRequest` | `generate` |
| 2 | `CompletionRequest` | `generate` |
| 3 | `EmbeddingRequest` | `embed` |
| 4 | `ClassificationRequest` | `classify` |
| 5 | `RerankRequest` | `score` |
| 6 | `ScoreRequest` | `score` |
| 7 | `PoolingRequest` | pooling tasks |

### Decorator Behaviors

| Decorator | Effect |
|-----------|--------|
| `@sagemaker_standards.register_invocation_handler` | Registers with SageMaker container standards |
| `@sagemaker_standards.stateful_session_manager()` | Enables session-based routing for stateful inference |
| `@sagemaker_standards.inject_adapter_id(adapter_path="model")` | Injects SageMaker adapter ID into the `model` field for multi-model endpoints |

## Deploying to SageMaker

### Step 1: Push the Image to ECR

```bash
# Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build \
  --target vllm-sagemaker \
  -f docker/Dockerfile \
  -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/vllm-sagemaker:latest \
  .

docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/vllm-sagemaker:latest
```

### Step 2: Create a SageMaker Model

```python
import boto3

sagemaker_client = boto3.client("sagemaker", region_name="us-east-1")

sagemaker_client.create_model(
    ModelName="vllm-llama-8b",
    PrimaryContainer={
        "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/vllm-sagemaker:latest",
        "Environment": {
            "SM_VLLM_MODEL": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
            "SM_VLLM_MAX_MODEL_LEN": "8192",
            "HF_TOKEN": "hf_your_token_here",
        },
    },
    ExecutionRoleArn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
)
```

### Step 3: Create an Endpoint Configuration

```python
sagemaker_client.create_endpoint_config(
    EndpointConfigName="vllm-llama-8b-config",
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": "vllm-llama-8b",
            "InstanceType": "ml.g5.2xlarge",  # 1x A10G GPU
            "InitialInstanceCount": 1,
        }
    ],
)
```

### Step 4: Create the Endpoint

```python
sagemaker_client.create_endpoint(
    EndpointName="vllm-llama-8b-endpoint",
    EndpointConfigName="vllm-llama-8b-config",
)

# Wait for the endpoint to be in service
waiter = sagemaker_client.get_waiter("endpoint_in_service")
waiter.wait(EndpointName="vllm-llama-8b-endpoint")
print("Endpoint is ready!")
```

### Step 5: Invoke the Endpoint

```python
import json
import boto3

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="vllm-llama-8b-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }),
)

result = json.loads(response["Body"].read())
print(result["choices"][0]["message"]["content"])
```

## SageMaker Instance Types

| Instance Type | GPU | VRAM | Recommended Use |
|---------------|-----|------|-----------------|
| `ml.g4dn.xlarge` | 1× T4 | 16 GB | Small models (≤7B) |
| `ml.g5.2xlarge` | 1× A10G | 24 GB | 7B–13B models |
| `ml.g5.12xlarge` | 4× A10G | 96 GB | 70B models (TP=4) |
| `ml.p4d.24xlarge` | 8× A100 | 320 GB | 70B+ models (TP=8) |
| `ml.p4de.24xlarge` | 8× A100 80GB | 640 GB | Large models |

## Stateful Sessions

The `@sagemaker_standards.stateful_session_manager()` decorator enables session-based routing. This is useful for multi-turn conversations where requests must be routed to the same instance:

```python
# Create a new session
response = requests.post(
    endpoint_url + "/invocations",
    json={"requestType": "NEW_SESSION"},
)
session_id = response.headers.get("X-Amzn-SageMaker-New-Session-Id")

# Use the session for subsequent requests
response = requests.post(
    endpoint_url + "/invocations",
    headers={"X-Amzn-SageMaker-Session-Id": session_id},
    json={
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
)
```

## LoRA Adapter Injection

The `@sagemaker_standards.inject_adapter_id(adapter_path="model")` decorator enables SageMaker's multi-model endpoint feature. When SageMaker routes a request to a specific adapter, the adapter ID is automatically injected into the `model` field of the request body, enabling dynamic LoRA adapter selection.

See [LoRA Management API](../12-api-reference/lora-management.md) for details on loading adapters.

## Related Pages

- [SageMaker API Reference](../12-api-reference/sagemaker-api.md) — detailed endpoint documentation
- [Docker Deployment](docker.md) — building the container image
- [LoRA Adapters](../10-lora/README.md) — dynamic adapter loading
