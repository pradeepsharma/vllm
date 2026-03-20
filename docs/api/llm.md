# LLM

```python
from vllm import LLM
```

`LLM` is the primary entry point for **offline batch inference**. It wraps a language model (possibly distributed across multiple GPUs) together with its tokenizer and KV-cache memory, and exposes a synchronous, high-level API for text generation, chat, embeddings, classification, scoring, and reward modelling.

!!! note "Offline vs. online"
    `LLM` is designed for offline (batch) workloads. For online serving with concurrent requests, use [`AsyncLLMEngine`](async_llm_engine.md) or the built-in OpenAI-compatible HTTP server (`vllm serve`).

---

## Constructor

```python
LLM(
    model: str,
    *,
    # Runner / conversion
    runner: RunnerOption = "auto",
    convert: ConvertOption = "auto",

    # Tokenizer
    tokenizer: str | None = None,
    tokenizer_mode: str = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,

    # Media / multimodal
    allowed_local_media_path: str = "",
    allowed_media_domains: list[str] | None = None,

    # Parallelism
    tensor_parallel_size: int = 1,

    # Precision & quantization
    dtype: str = "auto",
    quantization: str | None = None,

    # Model version
    revision: str | None = None,
    tokenizer_revision: str | None = None,

    # Chat template
    chat_template: str | Path | None = None,

    # Sampling seed
    seed: int = 0,

    # Memory
    gpu_memory_utilization: float = 0.9,
    kv_cache_memory_bytes: int | None = None,
    cpu_offload_gb: float = 0,
    offload_group_size: int = 0,
    offload_num_in_group: int = 1,
    offload_prefetch_step: int = 1,
    offload_params: set[str] | None = None,

    # Execution
    enforce_eager: bool = False,
    enable_return_routed_experts: bool = False,
    disable_custom_all_reduce: bool = False,

    # HuggingFace
    hf_token: bool | str | None = None,
    hf_overrides: dict | Callable | None = None,

    # Multimodal processor
    mm_processor_kwargs: dict[str, Any] | None = None,

    # Pooling
    pooler_config: PoolerConfig | None = None,

    # Advanced configs
    structured_outputs_config: dict | StructuredOutputsConfig | None = None,
    profiler_config: dict | ProfilerConfig | None = None,
    attention_config: dict | AttentionConfig | None = None,
    compilation_config: int | dict | CompilationConfig | None = None,
    logits_processors: list[str | type] | None = None,

    # Additional EngineArgs kwargs
    **kwargs: Any,
)
```

### Parameters

#### Model & tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *(required)* | HuggingFace model name or local path (e.g. `"meta-llama/Llama-3.1-8B-Instruct"`). |
| `tokenizer` | `str \| None` | `None` | Override the tokenizer. Defaults to the same path as `model`. |
| `tokenizer_mode` | `str` | `"auto"` | `"auto"` uses the fast tokenizer when available; `"slow"` always uses the slow tokenizer. |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer and detokenizer initialisation. Requires callers to supply `prompt_token_ids` directly. |
| `trust_remote_code` | `bool` | `False` | Allow execution of remote code from HuggingFace Hub. |
| `revision` | `str \| None` | `None` | Model revision (branch, tag, or commit SHA). |
| `tokenizer_revision` | `str \| None` | `None` | Tokenizer revision (branch, tag, or commit SHA). |
| `hf_token` | `bool \| str \| None` | `None` | HuggingFace authentication token. `True` reads the token from `~/.cache/huggingface/token`. |
| `hf_overrides` | `dict \| Callable \| None` | `None` | Dict of config overrides forwarded to the HuggingFace config, or a callable that mutates it. |

#### Runner & conversion

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runner` | `str` | `"auto"` | Execution runner. `"auto"` selects the best runner for the model. Use `"generate"` for text generation or `"pooling"` for embedding/classification models. |
| `convert` | `str` | `"auto"` | Model conversion mode. `"auto"` detects the appropriate conversion. Use `"embed"`, `"classify"`, or `"score"` to force a specific pooling task. |

#### Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | `int` | `1` | Number of GPUs for tensor parallelism. |

#### Precision & quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | `str` | `"auto"` | Model weight and activation dtype. One of `"auto"`, `"float32"`, `"float16"`, `"bfloat16"`. `"auto"` reads from the model config (falling back to `float16` if the config specifies `float32`). |
| `quantization` | `str \| None` | `None` | Quantization method. Supported values include `"awq"`, `"gptq"`, `"fp8"`. `None` auto-detects from the model config. |

#### Memory management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory reserved for model weights, activations, and KV cache. Higher values increase KV cache size and throughput but risk OOM errors. |
| `kv_cache_memory_bytes` | `int \| None` | `None` | Explicit KV cache size per GPU in bytes. When set, overrides `gpu_memory_utilization` for KV cache sizing. |
| `cpu_offload_gb` | `float` | `0` | GiB of CPU memory used to offload model weights, effectively increasing usable GPU memory at the cost of CPU↔GPU transfer overhead. |
| `offload_group_size` | `int` | `0` | Prefetch offloading: group every N layers together. `0` disables prefetch offloading. |
| `offload_num_in_group` | `int` | `1` | Prefetch offloading: number of layers to offload per group. |
| `offload_prefetch_step` | `int` | `1` | Prefetch offloading: number of layers to prefetch ahead. Higher values hide more latency but use more GPU memory. |
| `offload_params` | `set[str] \| None` | `None` | Prefetch offloading: set of parameter name segments to selectively offload. `None` offloads all parameters. |

#### Execution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enforce_eager` | `bool` | `False` | Disable CUDA graph capture and always run in eager mode. Useful for debugging. |
| `enable_return_routed_experts` | `bool` | `False` | Return the routed expert indices for MoE models. |
| `disable_custom_all_reduce` | `bool` | `False` | Disable the custom all-reduce kernel and fall back to NCCL. |

#### Chat & multimodal

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_template` | `str \| Path \| None` | `None` | Custom Jinja2 chat template string or path to a template file. Overrides the model's built-in template. |
| `allowed_local_media_path` | `str` | `""` | Directory from which local image/video files may be read. **Security risk** — only enable in trusted environments. |
| `allowed_media_domains` | `list[str] \| None` | `None` | Allowlist of domains from which remote media URLs may be fetched. |
| `mm_processor_kwargs` | `dict \| None` | `None` | Keyword arguments forwarded to the model's multimodal processor (e.g. `{"num_crops": 4}` for Phi-3-Vision). |

#### Pooling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pooler_config` | `PoolerConfig \| None` | `None` | Non-default pooling configuration (e.g. `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`). |

#### Advanced configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | Global random seed for sampling. |
| `compilation_config` | `int \| dict \| CompilationConfig \| None` | `None` | Compilation optimisation level (integer) or full `CompilationConfig`. |
| `attention_config` | `dict \| AttentionConfig \| None` | `None` | Attention backend configuration. |
| `structured_outputs_config` | `dict \| StructuredOutputsConfig \| None` | `None` | Structured output (guided decoding) configuration. |
| `profiler_config` | `dict \| ProfilerConfig \| None` | `None` | Profiler configuration. |
| `logits_processors` | `list[str \| type] \| None` | `None` | Custom logits processors to apply during generation. |
| `**kwargs` | `Any` | — | Additional keyword arguments forwarded to [`EngineArgs`](engine_args.md). |

---

## Methods

### Text Generation

#### `generate`

```python
def generate(
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable = True,
    lora_request: LoRARequest | Sequence[LoRARequest] | None = None,
    priority: list[int] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]
```

Generate text completions for one or more prompts.

vLLM automatically batches all prompts together for maximum throughput. For best performance, pass all prompts in a single call rather than calling `generate` in a loop.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `PromptType \| Sequence[PromptType]` | *(required)* | One or more prompts. Each prompt can be a plain string, a `TextPrompt`, a `TokensPrompt`, or a multimodal dict. |
| `sampling_params` | `SamplingParams \| Sequence[SamplingParams] \| None` | `None` | Sampling parameters. A single value is applied to all prompts; a list must match the number of prompts. `None` uses model defaults. |
| `use_tqdm` | `bool \| Callable` | `True` | Show a tqdm progress bar. Pass a callable (e.g. `functools.partial(tqdm, leave=False)`) to customise the bar. |
| `lora_request` | `LoRARequest \| Sequence[LoRARequest] \| None` | `None` | LoRA adapter(s) to apply. |
| `priority` | `list[int] \| None` | `None` | Per-prompt scheduling priorities (lower = higher priority). Only valid when priority scheduling is enabled. |
| `tokenization_kwargs` | `dict \| None` | `None` | Extra keyword arguments forwarded to `tokenizer.encode`. |

**Returns** `list[RequestOutput]` — one output per prompt, in the same order as the input.

**Raises** `ValueError` — if the model's runner type is not `"generate"`.

**Example**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.8, max_tokens=128)

outputs = llm.generate(
    ["Tell me a joke.", "What is 2 + 2?"],
    sampling_params=params,
)
for out in outputs:
    print(out.outputs[0].text)
```

---

#### `chat`

```python
def chat(
    messages: list[ChatCompletionMessageParam]
           | Sequence[list[ChatCompletionMessageParam]],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    use_tqdm: bool | Callable = True,
    lora_request: LoRARequest | Sequence[LoRARequest] | None = None,
    chat_template: str | None = None,
    chat_template_content_format: str = "auto",
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]
```

Generate responses for one or more chat conversations.

Each conversation is a list of message dicts with `"role"` and `"content"` keys, following the OpenAI chat format. Multi-modal inputs (images, audio, etc.) can be embedded in message content using the same format as the OpenAI API.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict] \| Sequence[list[dict]]` | *(required)* | A single conversation or a list of conversations. |
| `sampling_params` | `SamplingParams \| Sequence[SamplingParams] \| None` | `None` | Sampling parameters. |
| `use_tqdm` | `bool \| Callable` | `True` | Progress bar control. |
| `lora_request` | `LoRARequest \| Sequence[LoRARequest] \| None` | `None` | LoRA adapter(s). |
| `chat_template` | `str \| None` | `None` | Override the model's built-in Jinja2 chat template. |
| `chat_template_content_format` | `str` | `"auto"` | `"string"` renders content as a plain string; `"openai"` renders it as a list of typed dicts. |
| `add_generation_prompt` | `bool` | `True` | Append the generation-prompt token(s) after the last message. |
| `continue_final_message` | `bool` | `False` | Continue the final assistant message instead of starting a new turn. Mutually exclusive with `add_generation_prompt=True`. |
| `tools` | `list[dict] \| None` | `None` | Tool definitions to include in the chat template context. |
| `chat_template_kwargs` | `dict \| None` | `None` | Extra keyword arguments forwarded to the Jinja2 template. |
| `tokenization_kwargs` | `dict \| None` | `None` | Extra keyword arguments forwarded to `tokenizer.encode`. |
| `mm_processor_kwargs` | `dict \| None` | `None` | Extra keyword arguments forwarded to the multimodal processor. |

**Returns** `list[RequestOutput]`

**Example**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

outputs = llm.chat(
    messages=[
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Explain quantum entanglement."},
        ]
    ],
    sampling_params=SamplingParams(max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

---

#### `beam_search`

```python
def beam_search(
    prompts: list[TokensPrompt | TextPrompt],
    params: BeamSearchParams,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    use_tqdm: bool = False,
    concurrency_limit: int | None = None,
) -> list[BeamSearchOutput]
```

Generate sequences using beam search.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `list[TokensPrompt \| TextPrompt]` | *(required)* | Input prompts. |
| `params` | `BeamSearchParams` | *(required)* | Beam search configuration (beam width, max tokens, temperature, length penalty, etc.). |
| `lora_request` | `LoRARequest \| list[LoRARequest] \| None` | `None` | LoRA adapter(s). |
| `use_tqdm` | `bool` | `False` | Show a progress bar over token steps. |
| `concurrency_limit` | `int \| None` | `None` | Maximum number of concurrent beam search instances. `None` runs all prompts concurrently. |

**Returns** `list[BeamSearchOutput]`

---

### Asynchronous / Queued Generation

#### `enqueue`

```python
def enqueue(
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    lora_request: LoRARequest | Sequence[LoRARequest] | None = None,
    priority: list[int] | None = None,
    use_tqdm: bool | Callable = True,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[str]
```

Add requests to the engine queue **without** waiting for results.

Use `enqueue` followed by `wait_for_completion` to decouple request submission from result collection — useful when you want to interleave enqueueing and processing.

**Returns** `list[str]` — request IDs for the enqueued requests.

---

#### `wait_for_completion`

```python
def wait_for_completion(
    output_type: type | tuple[type, ...] | None = None,
    *,
    use_tqdm: bool | Callable = True,
) -> list[RequestOutput | PoolingRequestOutput]
```

Process all queued requests and return their outputs.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_type` | `type \| tuple[type, ...] \| None` | `None` | Expected output type(s). Defaults to `(RequestOutput, PoolingRequestOutput)`. |
| `use_tqdm` | `bool \| Callable` | `True` | Progress bar control. |

**Returns** A list of output objects for all completed requests.

**Example**

```python
ids = llm.enqueue(["Hello", "World"], sampling_params=params)
outputs = llm.wait_for_completion()
```

---

### Pooling / Embedding Methods

#### `embed`

```python
def embed(
    prompts: PromptType | Sequence[PromptType],
    *,
    use_tqdm: bool | Callable = True,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[EmbeddingRequestOutput]
```

Generate dense embedding vectors for each prompt.

**Requires** a model with `runner="pooling"` and `convert="embed"` (or `"auto"` for models that natively support embeddings).

**Returns** `list[EmbeddingRequestOutput]` — each output contains an `EmbeddingOutput` with a `embedding: list[float]` field.

**Example**

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-base-en-v1.5")
results = llm.embed(["Hello world", "vLLM is fast"])
print(results[0].outputs.embedding[:5])
```

---

#### `classify`

```python
def classify(
    prompts: PromptType | Sequence[PromptType],
    *,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    use_tqdm: bool | Callable = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[ClassificationRequestOutput]
```

Generate class probability vectors for each prompt.

**Requires** a model with `convert="classify"`.

**Returns** `list[ClassificationRequestOutput]` — each output contains a `ClassificationOutput` with a `probs: list[float]` field.

---

#### `score`

```python
def score(
    data_1: SingletonPrompt | Sequence[SingletonPrompt] | ...,
    data_2: SingletonPrompt | Sequence[SingletonPrompt] | ...,
    /,
    *,
    use_tqdm: bool | Callable = True,
    pooling_params: PoolingParams | None = None,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
) -> list[ScoringRequestOutput]
```

Compute similarity scores for all `(data_1, data_2)` pairs.

Supports three scoring modes, selected automatically based on the model:

- **Cross-encoder** — concatenates the pair and runs a single forward pass.
- **Late interaction** (ColBERT-style) — encodes each side independently, then computes MaxSim.
- **Embedding cosine similarity** — encodes each side independently, then computes cosine similarity.

Input shapes:

- **1 → 1**: single query, single document.
- **1 → N**: single query replicated against N documents.
- **N → N**: N queries paired with N documents.

**Returns** `list[ScoringRequestOutput]` — each output contains a `ScoringOutput` with a `score: float` field.

---

#### `reward`

```python
def reward(
    prompts: PromptType | Sequence[PromptType],
    /,
    *,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    use_tqdm: bool | Callable = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[PoolingRequestOutput]
```

Generate reward scores for each prompt (token-level classification pooling).

**Returns** `list[PoolingRequestOutput]`

---

#### `encode`

```python
def encode(
    prompts: PromptType | Sequence[PromptType] | DataPrompt,
    pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
    *,
    use_tqdm: bool | Callable = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    pooling_task: PoolingTask | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> list[PoolingRequestOutput]
```

Low-level pooling method. Requires `pooling_task` to be set explicitly.

!!! tip
    Prefer the task-specific helpers (`embed`, `classify`, `score`, `reward`) over `encode` for clearer code.

**Raises** `ValueError` — if `pooling_task` is `None`.

---

### Engine Control

#### `sleep`

```python
def sleep(level: int = 1, mode: PauseMode = "abort") -> None
```

Put the engine into a low-power sleep state.

| Level | Effect |
|-------|--------|
| `0` | Pause scheduling; continue accepting requests (queued but not processed). |
| `1` | Offload model weights to CPU and discard KV cache. Good for sleeping and waking with the same model. Requires sufficient CPU memory. |
| `2` | Discard all GPU memory (weights + KV cache). Good for switching models or updating weights. Reduces CPU memory pressure. |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `int` | `1` | Sleep level (0, 1, or 2). |
| `mode` | `str` | `"abort"` | How to handle in-flight requests: `"abort"` cancels them, `"wait"` waits for completion, `"keep"` freezes them for resumption. |

---

#### `wake_up`

```python
def wake_up(tags: list[str] | None = None) -> None
```

Wake the engine from sleep mode.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tags` | `list[str] \| None` | `None` | Memory allocation tags to restore. Valid values: `"weights"`, `"kv_cache"`, `"scheduling"`. `None` restores all. Use `["scheduling"]` to resume from level-0 sleep. |

---

#### `collective_rpc`

```python
def collective_rpc(
    method: str | Callable,
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[Any]
```

Execute a method on all workers and return their results.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str \| Callable` | *(required)* | Worker method name or a callable serialised and sent to all workers. Callables receive `self` (the worker object) as their first argument. |
| `timeout` | `float \| None` | `None` | Maximum wait time in seconds. `None` waits indefinitely. |
| `args` | `tuple` | `()` | Positional arguments for the method. |
| `kwargs` | `dict \| None` | `None` | Keyword arguments for the method. |

**Returns** `list[Any]` — one result per worker.

!!! tip
    Use `collective_rpc` for control messages only. For data transfer, set up a dedicated data-plane channel.

---

#### `apply_model`

```python
def apply_model(func: Callable[[nn.Module], Any]) -> list[Any]
```

Run a function directly on the model inside each worker.

!!! warning
    Avoid returning large tensors or arrays. Move them to CPU first to prevent extra VRAM usage.

**Returns** `list[Any]` — one result per worker.

---

### Weight Management

#### `update_weights`

```python
def update_weights(request: WeightTransferUpdateRequest | dict) -> None
```

Update model weights at runtime (for RL training workflows).

---

#### `init_weight_transfer_engine`

```python
def init_weight_transfer_engine(
    request: WeightTransferInitRequest | dict
) -> None
```

Initialise the weight transfer backend for RL training.

---

### Profiling

#### `start_profile`

```python
def start_profile(profile_prefix: str | None = None) -> None
```

Start profiling. Trace files are named `<prefix>_dp<X>_pp<Y>_tp<Z>` when `profile_prefix` is provided.

---

#### `stop_profile`

```python
def stop_profile() -> None
```

Stop profiling and flush trace files.

---

### Utilities

#### `get_tokenizer`

```python
def get_tokenizer() -> TokenizerLike
```

Return the tokenizer used by the engine.

---

#### `get_world_size`

```python
def get_world_size(include_dp: bool = True) -> int
```

Return the total number of worker processes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_dp` | `bool` | `True` | Include data-parallel replicas in the count (`TP × PP × DP`). Set to `False` for `TP × PP` only. |

---

#### `get_default_sampling_params`

```python
def get_default_sampling_params() -> SamplingParams
```

Return the model's default sampling parameters (read from the model config).

---

#### `reset_mm_cache`

```python
def reset_mm_cache() -> None
```

Clear the multimodal processor cache.

---

#### `reset_prefix_cache`

```python
def reset_prefix_cache(
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool
```

Invalidate the prefix (KV) cache.

---

#### `get_metrics`

```python
def get_metrics() -> list[Metric]
```

Return a snapshot of all Prometheus metrics aggregated by the engine.

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm_engine` | `LLMEngine` | The underlying synchronous engine. |
| `model_config` | `ModelConfig` | Resolved model configuration. |
| `renderer` | `Renderer` | The tokenizer/renderer used for prompt processing. |
| `supported_tasks` | `tuple[SupportedTask, ...]` | Tasks supported by the loaded model. |

---

## Notes

### Prompt types

The `prompts` argument accepts several formats:

```python
# Plain string
llm.generate("Hello, world")

# TextPrompt dict
llm.generate({"prompt": "Hello, world"})

# Pre-tokenised
llm.generate({"prompt_token_ids": [1, 2, 3]})

# Multimodal (image + text)
llm.generate({
    "prompt": "Describe this image: <image>",
    "multi_modal_data": {"image": pil_image},
})
```

### Batching

Always pass all prompts in a single `generate` / `chat` / `embed` call. vLLM's continuous batching engine maximises GPU utilisation across the entire batch. Calling these methods in a loop defeats the purpose of batching.

### LoRA adapters

```python
from vllm.lora.request import LoRARequest

lora = LoRARequest("my-adapter", 1, "/path/to/adapter")
outputs = llm.generate(prompts, lora_request=lora)
```
