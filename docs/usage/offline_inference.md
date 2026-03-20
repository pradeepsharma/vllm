# Offline Inference

Offline inference lets you run vLLM directly in your Python code — no server, no HTTP overhead. You load a model once, submit a batch of prompts, and get results back as Python objects. This is ideal for:

- **Batch processing** large datasets (evaluation, data augmentation, annotation)
- **Research and experimentation** where you iterate quickly on prompts and parameters
- **Pipeline integration** where vLLM is one step in a larger Python workflow

The primary entry point is the [`LLM`][vllm.LLM] class.

---

## Installation

=== "NVIDIA CUDA"

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
    ```

=== "CPU / Other"

    See the [full installation guide](../getting_started/installation/index.md) for CPU, TPU, and other hardware.

---

## The LLM Class

[`LLM`][vllm.LLM] is the main class for offline inference. It initializes the engine, loads model weights, and allocates GPU memory — all in one call.

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
```

### Key Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *(required)* | HuggingFace model name or local path |
| `tokenizer` | `str \| None` | `None` | Override tokenizer (defaults to `model`) |
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism |
| `dtype` | `str` | `"auto"` | Weight dtype: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"` |
| `quantization` | `str \| None` | `None` | Quantization method: `"awq"`, `"gptq"`, `"fp8"`, etc. |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory to reserve for model + KV cache |
| `max_model_len` | `int \| None` | `None` | Override the model's maximum context length |
| `trust_remote_code` | `bool` | `False` | Allow executing remote code from HuggingFace |
| `seed` | `int` | `0` | Global random seed |
| `enforce_eager` | `bool` | `False` | Disable CUDA graphs (useful for debugging) |
| `runner` | `str` | `"auto"` | `"generate"` for text generation, `"pooling"` for embeddings |

For the complete list, see the [Engine Arguments reference](../configuration/engine_args.md).

---

## Text Generation with `generate()`

`llm.generate()` is the core method for text completion. It accepts raw text prompts (or token IDs) and returns a list of [`RequestOutput`][vllm.RequestOutput] objects.

### Basic Example

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print()
```

### Greedy Decoding (Deterministic)

Set `temperature=0.0` for fully deterministic, greedy output:

```python
params = SamplingParams(temperature=0.0, max_tokens=100)
outputs = llm.generate("Explain the theory of relativity:", params)
print(outputs[0].outputs[0].text)
```

### Multiple Outputs per Prompt

Use `n` to generate several independent completions for each prompt:

```python
params = SamplingParams(n=4, temperature=0.9, max_tokens=50)
outputs = llm.generate("Once upon a time", params)

for i, completion in enumerate(outputs[0].outputs):
    print(f"Completion {i}: {completion.text!r}")
```

### Per-Prompt Sampling Parameters

Pass a list of `SamplingParams` — one per prompt — for fine-grained control:

```python
prompts = ["Write a haiku about rain.", "Summarize quantum mechanics."]
params = [
    SamplingParams(temperature=1.2, max_tokens=30),   # creative for haiku
    SamplingParams(temperature=0.0, max_tokens=200),  # deterministic for summary
]
outputs = llm.generate(prompts, params)
```

### Token ID Inputs

You can pass pre-tokenized inputs using a `dict` with `"prompt_token_ids"`:

```python
outputs = llm.generate(
    {"prompt_token_ids": [1, 2, 3, 4, 5]},
    SamplingParams(max_tokens=50),
)
```

### Disabling the Progress Bar

```python
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
```

---

## Chat Completion with `chat()`

`llm.chat()` applies the model's chat template automatically, making it the right choice for instruction-tuned and chat models.

### Single Conversation

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = llm.chat(messages, SamplingParams(temperature=0.7, max_tokens=256))
print(outputs[0].outputs[0].text)
```

### Batch Chat (Multiple Conversations)

Pass a list of conversations for efficient batch processing:

```python
conversations = [
    [
        {"role": "user", "content": "Tell me a joke."},
    ],
    [
        {"role": "system", "content": "You are a Python expert."},
        {"role": "user", "content": "How do I reverse a list in Python?"},
    ],
    [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3 + 3?"},
    ],
]

outputs = llm.chat(conversations, SamplingParams(temperature=0.8, max_tokens=128))

for conv, output in zip(conversations, outputs):
    print(f"Last user message: {conv[-1]['content']!r}")
    print(f"Response: {output.outputs[0].text!r}")
    print()
```

### Multi-Turn Conversations

Build up conversation history across turns:

```python
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=512)

history = [{"role": "system", "content": "You are a helpful coding assistant."}]

# Turn 1
history.append({"role": "user", "content": "Write a Python function to compute Fibonacci numbers."})
output = llm.chat(history, params)
response = output[0].outputs[0].text
history.append({"role": "assistant", "content": response})
print("Assistant:", response)

# Turn 2
history.append({"role": "user", "content": "Now add memoization to it."})
output = llm.chat(history, params)
print("Assistant:", output[0].outputs[0].text)
```

### Custom Chat Template

Override the model's default chat template:

```python
with open("my_template.jinja") as f:
    template = f.read()

outputs = llm.chat(
    messages,
    sampling_params,
    chat_template=template,
)
```

### Continuing a Partial Response

Use `continue_final_message=True` to have the model continue an in-progress assistant turn:

```python
messages = [
    {"role": "user", "content": "List the planets in order from the sun:"},
    {"role": "assistant", "content": "1. Mercury\n2. Venus\n3."},  # partial
]
outputs = llm.chat(
    messages,
    SamplingParams(temperature=0.0, max_tokens=100),
    add_generation_prompt=False,
    continue_final_message=True,
)
print(outputs[0].outputs[0].text)  # continues from "3."
```

---

## Embeddings with `embed()`

Use `embed()` with a pooling model to generate dense embedding vectors.

```python
from vllm import LLM

# Use runner="pooling" for embedding models
llm = LLM(model="intfloat/e5-small", runner="pooling")

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
    "The stock market fell sharply today.",
]

outputs = llm.embed(texts)

for text, output in zip(texts, outputs):
    embedding = output.outputs.embedding  # list[float]
    print(f"Text: {text!r}")
    print(f"Embedding dim: {len(embedding)}, first 4 values: {embedding[:4]}")
    print()
```

### Semantic Similarity

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

llm = LLM(model="intfloat/e5-small", runner="pooling")
outputs = llm.embed(texts)
embeddings = [o.outputs.embedding for o in outputs]

# Compare first two texts
sim = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity (fox sentences): {sim:.4f}")  # high similarity

sim = cosine_similarity(embeddings[0], embeddings[2])
print(f"Similarity (fox vs. stocks): {sim:.4f}")  # low similarity
```

### Matryoshka Embeddings (Reduced Dimensions)

Some models support matryoshka representation — truncating embeddings to a smaller dimension:

```python
from vllm import LLM
from vllm.pooling_params import PoolingParams

llm = LLM(model="nomic-ai/nomic-embed-text-v1.5", runner="pooling")

# Full dimension
outputs_full = llm.embed(texts)
print(f"Full dim: {len(outputs_full[0].outputs.embedding)}")  # e.g., 768

# Reduced to 256 dimensions
params = PoolingParams(dimensions=256)
outputs_small = llm.embed(texts, pooling_params=params)
print(f"Reduced dim: {len(outputs_small[0].outputs.embedding)}")  # 256
```

---

## Scoring / Reranking with `score()`

Use `score()` with a cross-encoder model to compute relevance scores between query-document pairs.

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")

query = "What is the capital of France?"
documents = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Paris is a major European city known for the Eiffel Tower.",
]

# 1-to-N: one query against multiple documents
outputs = llm.score(query, documents)

for doc, output in zip(documents, outputs):
    print(f"Score: {output.outputs.score:.4f} | {doc}")
```

### N-to-N Scoring

```python
queries = ["What is AI?", "How does photosynthesis work?"]
docs    = ["AI is artificial intelligence.", "Plants use sunlight to make food."]

# Paired: queries[i] scored against docs[i]
outputs = llm.score(queries, docs)
for q, d, o in zip(queries, docs, outputs):
    print(f"{q!r} vs {d!r} → {o.outputs.score:.4f}")
```

---

## Classification with `classify()`

Use `classify()` with a classification model to get class probability distributions.

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")

texts = [
    "I love this product! It works perfectly.",
    "This is the worst purchase I've ever made.",
    "The package arrived on time.",
]

outputs = llm.classify(texts)

for text, output in zip(texts, outputs):
    probs = output.outputs.probs  # list[float], one per class
    predicted_class = probs.index(max(probs))
    print(f"Text: {text!r}")
    print(f"Probs: {[f'{p:.3f}' for p in probs]}, Predicted class: {predicted_class}")
    print()
```

---

## Reward Models with `reward()`

Use `reward()` with a reward model to score prompt-response pairs for RLHF pipelines.

```python
from vllm import LLM

llm = LLM(model="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", runner="pooling")

# Format: "<prompt><response>" as a single string, or use chat template
prompts = [
    "User: What is 2+2?\nAssistant: 4",
    "User: What is 2+2?\nAssistant: I don't know.",
]

outputs = llm.reward(prompts)
for prompt, output in zip(prompts, outputs):
    # output.outputs.data is a tensor with reward scores
    print(f"Reward: {output.outputs.data.item():.4f}")
```

---

## Enqueue / Wait Pattern

For advanced use cases, you can decouple request submission from result collection using `enqueue()` and `wait_for_completion()`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0.8, max_tokens=100)

# Submit requests without blocking
request_ids = llm.enqueue(
    ["Hello, my name is", "The future of AI is"],
    params,
)
print(f"Enqueued {len(request_ids)} requests: {request_ids}")

# ... do other work here ...

# Collect all results
outputs = llm.wait_for_completion()
for output in outputs:
    print(output.outputs[0].text)
```

---

## Multi-GPU Inference

Scale to multiple GPUs with tensor parallelism:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # use 4 GPUs
)
```

For pipeline parallelism across nodes, see the [Parallelism & Scaling](../serving/parallelism_scaling.md) guide.

---

## LoRA Adapters

Load and use LoRA adapters at inference time:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    enable_lora=True,
    max_lora_rank=64,
)

lora_request = LoRARequest(
    lora_name="my-adapter",
    lora_int_id=1,
    lora_path="/path/to/lora/adapter",
)

outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0.8, max_tokens=100),
    lora_request=lora_request,
)
```

---

## Prefix Caching

Enable automatic prefix caching to speed up requests that share a common prefix (e.g., a long system prompt):

```python
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_prefix_caching=True,
)

system_prompt = "You are an expert Python developer. " * 100  # long shared prefix

conversations = [
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": "Write a binary search function."}],
    [{"role": "system", "content": system_prompt},
     {"role": "user", "content": "Write a merge sort function."}],
]

# The system prompt tokens are cached after the first request
outputs = llm.chat(conversations, SamplingParams(max_tokens=256))
```

---

## Structured Outputs

Constrain generation to valid JSON, a regex pattern, or a grammar:

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# JSON schema constraint
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
}

params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    structured_outputs=StructuredOutputsParams(json=schema),
)

outputs = llm.generate(
    'Generate a JSON object for a person named Alice who is 30 and lives in Paris.',
    params,
)
import json
result = json.loads(outputs[0].outputs[0].text)
print(result)  # {"name": "Alice", "age": 30, "city": "Paris"}
```

For more, see the [Structured Outputs](../features/structured_outputs.md) guide.

---

## Working with Output Objects

Every `llm.generate()` and `llm.chat()` call returns a list of [`RequestOutput`][vllm.RequestOutput] objects:

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    # The original prompt
    print("Prompt:", output.prompt)

    # Token IDs of the prompt
    print("Prompt token IDs:", output.prompt_token_ids[:10], "...")

    # Number of tokens served from prefix cache
    print("Cached tokens:", output.num_cached_tokens)

    # Each completion (n=1 by default → one completion)
    for completion in output.outputs:
        print(f"  [{completion.index}] text: {completion.text!r}")
        print(f"  [{completion.index}] finish_reason: {completion.finish_reason}")
        print(f"  [{completion.index}] token_ids: {list(completion.token_ids)}")
```

See [Output Types](output_types.md) for the full reference.

---

## Performance Tips

| Tip | Details |
|---|---|
| **Batch all prompts together** | Submit all prompts in a single `generate()` call for maximum GPU utilization |
| **Use `use_tqdm=False`** | Disable the progress bar in production to reduce overhead |
| **Enable prefix caching** | Set `enable_prefix_caching=True` when prompts share a long common prefix |
| **Tune `gpu_memory_utilization`** | Increase toward `0.95` for larger KV cache; decrease if you hit OOM |
| **Use `enforce_eager=False`** | Keep CUDA graphs enabled (the default) for best throughput |
| **Quantize large models** | Use `quantization="awq"` or `"fp8"` to fit larger models in memory |

---

## Ray Data Integration

For very large datasets, use the Ray Data LLM API for distributed, fault-tolerant batch inference:

```python
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

config = vLLMEngineProcessorConfig(model_source="meta-llama/Llama-3.2-1B-Instruct")
processor = build_llm_processor(
    config,
    preprocess=lambda row: {
        "messages": [{"role": "user", "content": row["text"]}],
        "sampling_params": {"temperature": 0.7, "max_tokens": 256},
    },
    postprocess=lambda row: {"output": row["generated_text"]},
)

ds = ray.data.read_parquet("s3://my-bucket/prompts/")
ds = processor(ds)
ds.write_parquet("s3://my-bucket/outputs/")
```

See the [Ray Data documentation](https://docs.ray.io/en/latest/data/working-with-llms.html) for more.

---

## See Also

- [SamplingParams Reference](sampling_params.md) — all generation parameters
- [PoolingParams Reference](pooling_params.md) — embedding and pooling parameters
- [Output Types](output_types.md) — understanding output objects
- [Beam Search](beam_search.md) — beam search decoding
- [Log Probabilities](logprobs.md) — token-level log probabilities
