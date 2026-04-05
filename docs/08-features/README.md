# Features

vLLM provides a rich set of advanced inference features beyond basic text generation. This section covers structured output, tool calling, prefix caching, and other capabilities that make vLLM suitable for complex production workloads.

## Structured Output

Constrained generation ensures model outputs conform to a specified format — JSON Schema, regex pattern, context-free grammar, or a fixed set of choices. vLLM supports multiple structured output backends:

| Backend | Key | Description |
|---------|-----|-------------|
| [xgrammar](structured-output/backend-xgrammar.md) | `xgrammar` | Default backend; fast grammar-based constraints |
| [outlines](structured-output/backend-outlines.md) | `outlines` | Regex and JSON Schema via finite-state machines |
| [guidance](structured-output/backend-guidance.md) | `guidance` | Microsoft Guidance library integration |
| [lm-format-enforcer](structured-output/backend-lm-format-enforcer.md) | `lm-format-enforcer` | Token-level format enforcement |

### Quick Start

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# JSON Schema constraint
sampling_params = SamplingParams(
    guided_decoding={
        "json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }
)

outputs = llm.generate(["Extract: John is 30 years old."], sampling_params)
print(outputs[0].outputs[0].text)  # {"name": "John", "age": 30}
```

### Via OpenAI API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "List 3 colors as JSON array"}],
    extra_body={"guided_json": {"type": "array", "items": {"type": "string"}}},
)
```

## Structured Output Pages

| Page | Description |
|------|-------------|
| [Overview](structured-output/overview.md) | Architecture, backend selection, performance |
| [xgrammar Backend](structured-output/backend-xgrammar.md) | Grammar compilation, token mask caching |
| [outlines Backend](structured-output/backend-outlines.md) | FSM-based constraints, regex support |
| [guidance Backend](structured-output/backend-guidance.md) | Microsoft Guidance integration |
| [lm-format-enforcer Backend](structured-output/backend-lm-format-enforcer.md) | Token-level enforcement |
| [Response Format](structured-output/response-format.md) | `response_format` field reference |
| [Guided Decoding Backend](structured-output/guided-decoding-backend.md) | Backend selection and configuration |
| [Code Examples](structured-output/code-examples.md) | Complete working examples |

## Prefix Caching

Automatic prefix caching reuses KV cache blocks for identical prompt prefixes across requests. This is especially valuable for:

- **System prompts** — shared across all requests in a chat application
- **RAG context** — the same retrieved documents appear in many queries
- **Few-shot examples** — fixed demonstration examples in the prompt

Prefix caching is enabled by default in the V1 engine. The cache hit rate is reported in Prometheus metrics as `vllm:gpu_prefix_cache_hit_rate`.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,  # default in V1
)
```

## Tool Calling

vLLM supports OpenAI-compatible tool calling (function calling) for models that have been fine-tuned for it. Tools are specified in the `tools` field of the chat completion request:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto",
)
```

## Chunked Prefill

Chunked prefill splits long prompt processing across multiple scheduling steps, interleaving prefill tokens with decode tokens. This prevents long prompts from monopolizing the GPU and reduces time-to-first-token for concurrent requests.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

## Related Sections

- [Speculative Decoding](../10-speculative-decoding/README.md) — EAGLE, Medusa, MTP, n-gram
- [LoRA Adapters](../10-lora/README.md) — multi-adapter serving
- [Multimodal Support](../10-multimodal/README.md) — images, audio, video
- [Quantization](../14-quantization/README.md) — FP8, AWQ, GPTQ, MXFP4
- [Configuration Reference](../06-configuration/README.md) — all config options
- [API Reference](../12-api-reference/README.md) — endpoint documentation
