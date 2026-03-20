# AWS SageMaker Integration

vLLM integrates with [AWS SageMaker](https://aws.amazon.com/sagemaker/) for managed ML model deployment. The SageMaker integration adds two required endpoints (`/ping` and `/invocations`) and automatically routes requests to the appropriate vLLM handler based on the request body schema.

---

## Overview

SageMaker requires all inference containers to implement:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET, POST | Health check — must return `200 OK` when ready |
| `/invocations` | POST | Inference endpoint — processes all requests |

The `/invocations` endpoint is a **smart router**: it inspects the request body and dispatches to the appropriate handler (chat completions, embeddings, scoring, etc.) based on which Pydantic model validates successfully.

---

## Quick Start

### Docker Image

Build a SageMaker-compatible Docker image:

```dockerfile
FROM vllm/vllm-openai:latest

# SageMaker requires the server to listen on port 8080
ENV PORT=8080

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

### Deploy to SageMaker

```python
import boto3
import sagemaker
from sagemaker.model import Model

session = sagemaker.Session()
role = sagemaker.get_execution_role()

model = Model(
    image_uri="<your-ecr-image-uri>",
    model_data=None,  # Model is baked into the image or loaded from HF
    role=role,
    env={
        "MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "VLLM_API_KEY": "my-secret-key",
    },
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name="vllm-llama-endpoint",
)
```

---

## Endpoints

### GET/POST /ping

Health check endpoint. Returns `200 OK` when the engine is ready to serve requests.

```bash
curl https://<endpoint>/ping
# → 200 OK
```

### POST /invocations

The main inference endpoint. Accepts any valid vLLM request body and routes it automatically.

```bash
curl https://<endpoint>/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Request Routing

The `/invocations` endpoint tries to validate the request body against each supported schema in priority order:

| Priority | Schema | Endpoint |
|----------|--------|----------|
| 1 | `ChatCompletionRequest` | Chat completions |
| 2 | `CompletionRequest` | Text completions |
| 3 | `EmbeddingRequest` | Embeddings |
| 4 | `ClassificationRequest` | Classification |
| 5 | `RerankRequest` | Re-ranking |
| 6 | `ScoreRequest` | Scoring |
| 7 | `PoolingRequest` | Raw pooling |

The first schema that validates successfully handles the request. If no schema matches, a `400 Bad Request` error is returned.

### Routing by Task

The available handlers depend on which tasks the model supports:

| Task | Handlers Available |
|------|--------------------|
| `generate` | Chat completions, Text completions |
| `embed` | Embeddings, Scoring |
| `classify` | Classification |
| `score` | Scoring, Re-ranking |
| Pooling tasks | Raw pooling |

---

## Chat Completions via SageMaker

```python
import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="vllm-llama-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }),
)

result = json.loads(response["Body"].read())
print(result["choices"][0]["message"]["content"])
```

---

## Embeddings via SageMaker

```python
response = runtime.invoke_endpoint(
    EndpointName="vllm-embedding-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "BAAI/bge-base-en-v1.5",
        "input": "The quick brown fox jumps over the lazy dog.",
    }),
)

result = json.loads(response["Body"].read())
embedding = result["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

---

## Stateful Sessions

The SageMaker integration supports stateful sessions via the `model_hosting_container_standards` library. This enables:

- **Session affinity**: Route requests from the same session to the same instance
- **LoRA adapter injection**: Automatically inject LoRA adapter IDs from session metadata

The `@sagemaker_standards.stateful_session_manager()` decorator manages session state, and `@sagemaker_standards.inject_adapter_id(adapter_path="model")` automatically sets the model field from session context.

---

## LoRA Adapters

Serve multiple LoRA adapters on a single endpoint:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules \
    '{"name": "adapter-v1", "path": "s3://my-bucket/adapter-v1"}' \
    '{"name": "adapter-v2", "path": "s3://my-bucket/adapter-v2"}'
```

SageMaker's session manager can inject the appropriate adapter ID based on the session context.

---

## Environment Variables

Configure vLLM behavior via environment variables in your SageMaker model:

| Variable | Description |
|----------|-------------|
| `MODEL_ID` | HuggingFace model ID or path |
| `VLLM_API_KEY` | API key for authentication |
| `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism |
| `MAX_MODEL_LEN` | Maximum context length |
| `DTYPE` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | Max audio file size (MB) |

---

## Multi-Model Endpoints

SageMaker supports multi-model endpoints (MME) where multiple models share the same instance. vLLM's LoRA support enables this pattern:

```python
# Deploy with multiple LoRA adapters
model_env = {
    "MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
    "ENABLE_LORA": "true",
    "LORA_MODULES": json.dumps([
        {"name": "customer-service", "path": "s3://bucket/cs-adapter"},
        {"name": "code-assistant", "path": "s3://bucket/code-adapter"},
    ]),
}
```

---

## Health Check Configuration

SageMaker polls `/ping` to determine if the endpoint is healthy. The endpoint returns:

- `200 OK` — Engine is ready
- `503 Service Unavailable` — Engine is not ready or has errored

Configure the health check interval and timeout in your SageMaker endpoint configuration:

```python
endpoint_config = {
    "HealthCheckConfig": {
        "HealthCheckIntervalInSeconds": 30,
        "HealthCheckTimeoutInSeconds": 10,
        "UnhealthyThresholdCount": 3,
    }
}
```

---

## Streaming on SageMaker

SageMaker supports streaming responses via `invoke_endpoint_with_response_stream`:

```python
response = runtime.invoke_endpoint_with_response_stream(
    EndpointName="vllm-llama-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "stream": True,
    }),
)

event_stream = response["Body"]
for event in event_stream:
    chunk = event.get("PayloadPart", {}).get("Bytes", b"")
    if chunk:
        line = chunk.decode("utf-8").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            import json
            data = json.loads(line[6:])
            content = data["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

---

## Cost Optimization

### Instance Selection

| Instance Type | GPUs | VRAM | Recommended For |
|---------------|------|------|-----------------|
| `ml.g5.2xlarge` | 1× A10G (24GB) | 24GB | 7B–13B models |
| `ml.g5.12xlarge` | 4× A10G (96GB) | 96GB | 34B–70B models |
| `ml.p4d.24xlarge` | 8× A100 (320GB) | 320GB | 70B+ models |
| `ml.p5.48xlarge` | 8× H100 (640GB) | 640GB | Largest models |

### Auto-Scaling

```python
import boto3

autoscaling = boto3.client("application-autoscaling")

autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/vllm-llama-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

autoscaling.put_scaling_policy(
    PolicyName="vllm-scaling-policy",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/vllm-llama-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

---

## Troubleshooting

### Endpoint Not Healthy

Check CloudWatch logs for the endpoint:

```bash
aws logs tail /aws/sagemaker/Endpoints/vllm-llama-endpoint --follow
```

Common issues:
- Model too large for instance VRAM → Use larger instance or reduce `max-model-len`
- Missing HuggingFace token → Set `HUGGING_FACE_HUB_TOKEN` environment variable
- Port mismatch → SageMaker requires port 8080

### Request Routing Failures

If `/invocations` returns `400 Bad Request` with "Cannot find suitable handler":
- Verify the request body matches one of the supported schemas
- Check that the model supports the requested task (e.g., embedding models don't support chat completions)

---

## See Also

- [OpenAI-Compatible Server](openai_compatible_server.md) — Full server configuration
- [Chat Completions](chat_completions.md) — Chat request format
- [Embeddings](embeddings.md) — Embedding request format
- [API Reference](api_reference.md) — Complete endpoint reference
