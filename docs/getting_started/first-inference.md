---
description: >
  Run your first vLLM inference — offline batch generation, chat completions,
  and the OpenAI-compatible server, with working code examples.
---

# First Inference Walkthrough

This guide walks you through running your first inference with vLLM, from a
simple offline generation to a full OpenAI-compatible server. By the end, you
will have generated text, used the chat interface, and queried a live HTTP
endpoint.

**Prerequisites:** vLLM installed. If not, see the
[Installation Overview](installation/index.md).

---

## Step 1 — Verify Your Installation

Before running inference, confirm vLLM is installed correctly:

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} ready')"
```

You should see output like:

```
vLLM 0.x.y ready
```

If you see an import error, revisit the [installation guide](installation/index.md).

---

## Step 2 — Offline Batch Inference

The simplest way to use vLLM is the `LLM` class for offline (batch) inference.
This loads a model and generates text for a list of prompts in one call.

```python
from vllm import LLM, SamplingParams

# Load the model (downloads from Hugging Face on first run)
llm = LLM(model="facebook/opt-125m")

# Define prompts and sampling parameters
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# Generate text
outputs = llm.generate(prompts, params)

# Print results
for output in outputs:
    print(f"Prompt:    {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print()
```

**Expected output:**

```
Prompt:    'Hello, my name is'
Generated: ' John. I am a software engineer...'

Prompt:    'The capital of France is'
Generated: ' Paris, a city known for...'

Prompt:    'The future of AI is'
Generated: ' bright, with advances in...'
```

### Key classes

| Class | Purpose |
|---|---|
| [`LLM`][vllm.LLM] | Main class for offline inference |
| [`SamplingParams`][vllm.SamplingParams] | Controls generation behavior |
| `RequestOutput` | Returned by `llm.generate()`, contains generated tokens |

### Common sampling parameters

| Parameter | Description | Default |
|---|---|---|
| `temperature` | Randomness (0 = greedy, 1 = full sampling) | `1.0` |
| `top_p` | Nucleus sampling probability | `1.0` |
| `top_k` | Top-K sampling | `-1` (disabled) |
| `max_tokens` | Maximum tokens to generate | `16` |
| `stop` | Stop strings (generation halts when matched) | `[]` |
| `n` | Number of output sequences per prompt | `1` |

!!! important "Default sampling parameters"
    By default, vLLM applies the model's recommended sampling parameters from
    `generation_config.json` on Hugging Face. To use vLLM's own defaults
    instead, set `generation_config="vllm"` when creating the `LLM` instance.

---

## Step 3 — Chat Interface (Instruct Models)

For instruction-tuned and chat models, use `llm.chat()` instead of
`llm.generate()`. It automatically applies the model's chat template:

```python
from vllm import LLM, SamplingParams

# Load a chat/instruct model
llm = LLM(model="Qwen/Qwen3-0.6B")

# Single conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the speed of light?"},
]

outputs = llm.chat(messages, SamplingParams(temperature=0.7, max_tokens=200))
print(outputs[0].outputs[0].text)
```

### Batch chat inference

Process multiple conversations in one call:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")

conversations = [
    [{"role": "user", "content": "Tell me a joke."}],
    [{"role": "user", "content": "What is the capital of Japan?"}],
    [{"role": "user", "content": "Write a haiku about autumn."}],
]

outputs = llm.chat(conversations, SamplingParams(temperature=0.8, max_tokens=128))

for conv, output in zip(conversations, outputs):
    print(f"Q: {conv[0]['content']}")
    print(f"A: {output.outputs[0].text}")
    print()
```

!!! note "Chat template"
    `llm.chat()` automatically applies the model's chat template. If you use
    `llm.generate()` with a chat model, you must apply the template manually
    using the tokenizer's `apply_chat_template()` method.

---

## Step 4 — Greedy Decoding (Deterministic Output)

For reproducible, deterministic outputs, use `temperature=0.0`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = llm.generate(
    ["The meaning of life is"],
    params,
)
print(outputs[0].outputs[0].text)
```

With `temperature=0.0`, the same prompt always produces the same output.

---

## Step 5 — Start the OpenAI-Compatible Server

vLLM provides an OpenAI-compatible HTTP server that you can query with any
OpenAI client library or `curl`.

### Start the server

```bash
vllm serve Qwen/Qwen3-0.6B \
    --port 8000 \
    --max-model-len 4096
```

The server starts and prints:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

!!! tip "First startup"
    The first startup downloads the model from Hugging Face (if not cached)
    and compiles CUDA kernels. This can take a few minutes. Subsequent
    startups are much faster.

### Check the server health

```bash
curl http://localhost:8000/health
```

Expected response: `{}`

### List available models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Step 6 — Query the Server

### Text completion (v1/completions)

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.8
    }'
```

### Chat completion (v1/chat/completions)

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the speed of light?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }'
```

### Using the OpenAI Python client

```python
from openai import OpenAI

# Point the client at your local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",   # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement simply."},
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Streaming responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Stream the response token by token
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Count from 1 to 10."}],
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

---

## Step 7 — Embeddings (Optional)

vLLM also supports embedding models for semantic search and retrieval:

```python
from vllm import LLM

# Load an embedding model
llm = LLM(model="intfloat/e5-small", runner="pooling")

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
    "The stock market fell sharply today.",
]

outputs = llm.embed(texts)

for text, output in zip(texts, outputs):
    embedding = output.outputs.embedding
    print(f"Text: {text[:50]!r}")
    print(f"Embedding: dim={len(embedding)}, first 4={embedding[:4]}")
    print()
```

---

## Common Patterns

### Load a gated model (requires HF token)

```bash
export HF_TOKEN="hf_your_token_here"
```

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    # HF_TOKEN is read from the environment automatically
)
```

### Limit GPU memory usage

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.85,   # Use 85% of GPU memory (default: 0.90)
)
```

### Limit context length

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=8192,   # Limit to 8K tokens (reduces memory usage)
)
```

### Use multiple GPUs (tensor parallel)

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,   # Distribute across 4 GPUs
)
```

### Use quantization

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",   # or "awq", "gptq", "bitsandbytes"
)
```

---

## Troubleshooting

??? question "Model download is slow"
    Enable fast transfer:
    ```bash
    pip install hf-transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ```

??? question "CUDA out of memory"
    Try one or more of:
    - Reduce `gpu_memory_utilization` (e.g., `0.80`)
    - Reduce `max_model_len`
    - Use a quantized model (`quantization="fp8"` or `"awq"`)
    - Use a smaller model

??? question "Server takes too long to start"
    The first startup compiles CUDA kernels. Subsequent startups reuse the
    cache and are much faster. You can also use `--enforce-eager` to skip
    CUDA graph capture (slower inference, faster startup).

??? question "Port already in use"
    Change the port: `vllm serve ... --port 8001`

??? question "Model not found on Hugging Face"
    - Check the model name spelling
    - Set `HF_TOKEN` for gated models
    - Use `VLLM_USE_MODELSCOPE=True` for ModelScope models

??? question "Tokenizer warnings about chat template"
    Use `llm.chat()` instead of `llm.generate()` for chat/instruct models.
    `llm.chat()` applies the chat template automatically.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-server: **OpenAI-Compatible Server**

    ---

    Full server documentation: authentication, multi-model serving, TLS,
    and all supported endpoints.

    [:octicons-arrow-right-24: Server guide](../serving/openai_compatible_server.md)

-   :material-code-braces: **Offline Batch Inference**

    ---

    High-throughput batch inference without a server, including multimodal
    inputs and structured outputs.

    [:octicons-arrow-right-24: Offline inference](../usage/offline_inference.md)

-   :material-tune: **Configuration**

    ---

    Tune engine arguments, memory settings, and optimization levels for
    your workload.

    [:octicons-arrow-right-24: Engine arguments](../configuration/engine_args.md)

-   :material-scale-balance: **Parallelism & Scaling**

    ---

    Scale to multiple GPUs and nodes with tensor, pipeline, and data
    parallelism.

    [:octicons-arrow-right-24: Parallelism guide](../serving/parallelism_scaling.md)

</div>
