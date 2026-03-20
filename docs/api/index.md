# Python API Reference

vLLM exposes a clean, well-structured Python API for both offline batch inference and online serving. This reference covers every public class, method, and parameter you need to build applications on top of vLLM.

---

## Public API Surface

All symbols below are importable directly from the top-level `vllm` package:

```python
import vllm

# Core inference classes
from vllm import LLM, AsyncLLMEngine, LLMEngine

# Configuration
from vllm import EngineArgs, AsyncEngineArgs

# Sampling & pooling control
from vllm import SamplingParams, PoolingParams

# Prompt types
from vllm import PromptType, TextPrompt, TokensPrompt

# Output types
from vllm import (
    RequestOutput,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    EmbeddingOutput,
    EmbeddingRequestOutput,
    ClassificationOutput,
    ClassificationRequestOutput,
    ScoringOutput,
    ScoringRequestOutput,
)

# Utilities
from vllm import ModelRegistry, initialize_ray_cluster
```

---

## API Modules

<div class="grid cards" markdown>

-   :material-robot: **[LLM](llm.md)**

    ---

    The primary offline-inference class. Wraps a language model (possibly distributed across multiple GPUs) and exposes a synchronous, batch-oriented API for text generation, chat, embeddings, classification, scoring, and rewards.

    [:octicons-arrow-right-24: LLM reference](llm.md)

-   :material-lightning-bolt: **[AsyncLLMEngine](async_llm_engine.md)**

    ---

    The asynchronous engine used by the OpenAI-compatible HTTP server. Accepts requests concurrently and streams outputs back via `AsyncGenerator`. Designed for production serving workloads.

    [:octicons-arrow-right-24: AsyncLLMEngine reference](async_llm_engine.md)

-   :material-engine: **[LLMEngine](llm_engine.md)**

    ---

    The synchronous, lower-level engine that powers `LLM`. Useful when you need fine-grained control over request scheduling, step-by-step execution, or integration with custom serving loops.

    [:octicons-arrow-right-24: LLMEngine reference](llm_engine.md)

-   :material-cog: **[EngineArgs / AsyncEngineArgs](engine_args.md)**

    ---

    Dataclasses that collect every configuration knob for the engine: model path, parallelism, quantization, memory, scheduling, LoRA, speculative decoding, and more.

    [:octicons-arrow-right-24: EngineArgs reference](engine_args.md)

-   :material-tune: **[SamplingParams](sampling_params.md)**

    ---

    Controls how tokens are sampled during text generation: temperature, top-p, top-k, stop strings, logprobs, structured outputs, and more.

    [:octicons-arrow-right-24: SamplingParams reference](sampling_params.md)

-   :material-vector-combine: **[PoolingParams](pooling_params.md)**

    ---

    Controls how hidden states are pooled for embedding, classification, scoring, and reward models.

    [:octicons-arrow-right-24: PoolingParams reference](pooling_params.md)

-   :material-file-document-outline: **[Output Types](outputs.md)**

    ---

    All output dataclasses returned by the inference APIs: `RequestOutput`, `CompletionOutput`, `EmbeddingRequestOutput`, `ClassificationRequestOutput`, `ScoringRequestOutput`, and more.

    [:octicons-arrow-right-24: Output types reference](outputs.md)

-   :material-api: **[OpenAI Protocol Types](openai_protocol.md)**

    ---

    Pydantic models that define the OpenAI-compatible HTTP API: `ChatCompletionRequest`, `CompletionRequest`, `ChatCompletionResponse`, and all related types.

    [:octicons-arrow-right-24: OpenAI protocol reference](openai_protocol.md)

</div>

---

## Quick-Start Examples

### Offline text generation

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Hello, my name is"], sampling_params=params)
for out in outputs:
    print(out.outputs[0].text)
```

### Chat interface

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

outputs = llm.chat([
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the capital of France?"},
    ]
])
print(outputs[0].outputs[0].text)
```

### Embeddings

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-base-en-v1.5")
results = llm.embed(["Hello world", "vLLM is fast"])
print(results[0].outputs.embedding[:5])
```

### Async generation (server-side)

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

args = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
engine = AsyncLLMEngine.from_engine_args(args)

async def generate():
    async for output in engine.generate(
        "Hello, world",
        SamplingParams(max_tokens=64),
        request_id="req-001",
    ):
        print(output.outputs[0].text)
```

---

## Version

The installed vLLM version is available as:

```python
import vllm
print(vllm.__version__)
```
