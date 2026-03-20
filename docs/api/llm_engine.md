# LLMEngine

```python
from vllm import LLMEngine
```

`LLMEngine` is the **synchronous, lower-level engine** that powers the [`LLM`](llm.md) class. It provides fine-grained control over request scheduling, step-by-step execution, and integration with custom serving loops.

!!! note "When to use LLMEngine directly"
    Most users should use [`LLM`](llm.md) for offline inference or [`AsyncLLMEngine`](async_llm_engine.md) for online serving. Use `LLMEngine` directly only when you need a custom synchronous serving loop or are building framework integrations.

---

## Construction

### `from_engine_args` *(classmethod)*

```python
@classmethod
def from_engine_args(
    engine_args: EngineArgs,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    enable_multiprocessing: bool = False,
) -> "LLMEngine"
```

The recommended way to create an `LLMEngine`. Accepts an [`EngineArgs`](engine_args.md) instance and returns a fully initialised engine.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_args` | `EngineArgs` | *(required)* | Engine configuration. |
| `usage_context` | `UsageContext` | `ENGINE_CONTEXT` | Usage context for telemetry. |
| `stat_loggers` | `list[StatLoggerFactory] \| None` | `None` | Custom stat logger factories. |
| `enable_multiprocessing` | `bool` | `False` | Run the engine core in a separate process. |

**Example**

```python
from vllm import LLMEngine, EngineArgs

args = EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
engine = LLMEngine.from_engine_args(args)
```

---

### `from_vllm_config` *(classmethod)*

```python
@classmethod
def from_vllm_config(
    vllm_config: VllmConfig,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    disable_log_stats: bool = False,
) -> "LLMEngine"
```

Create an `LLMEngine` from a pre-built `VllmConfig`.

---

### `__init__`

```python
LLMEngine(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    aggregate_engine_logging: bool = False,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    use_cached_outputs: bool = False,
    multiprocess_mode: bool = False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_config` | `VllmConfig` | *(required)* | Global engine configuration. |
| `executor_class` | `type[Executor]` | *(required)* | Executor implementation. |
| `log_stats` | `bool` | *(required)* | Enable Prometheus stat logging. |
| `aggregate_engine_logging` | `bool` | `False` | Aggregate logging across multiple engine instances. |
| `multiprocess_mode` | `bool` | `False` | Run the engine core in a separate process. |

---

## Request Management

### `add_request`

```python
def add_request(
    request_id: str,
    prompt: EngineCoreRequest | PromptType | ProcessorInputs,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    prompt_text: str | None = None,
) -> str
```

Add a single request to the engine queue.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_id` | `str` | *(required)* | Unique string identifier for this request. |
| `prompt` | `PromptType \| ProcessorInputs \| EngineCoreRequest` | *(required)* | The input prompt. |
| `params` | `SamplingParams \| PoolingParams` | *(required)* | Sampling or pooling parameters. |
| `arrival_time` | `float \| None` | `None` | Request arrival timestamp (Unix time). Defaults to `time.monotonic()`. |
| `lora_request` | `LoRARequest \| None` | `None` | LoRA adapter to apply. |
| `tokenization_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to `tokenizer.encode`. |
| `trace_headers` | `Mapping[str, str] \| None` | `None` | OpenTelemetry trace headers. |
| `priority` | `int` | `0` | Scheduling priority (lower = higher priority). |

**Returns** `str` — the `request_id` (for convenience).

**Raises** `TypeError` — if `request_id` is not a string.

---

### `abort_request`

```python
def abort_request(request_ids: list[str], internal: bool = False) -> None
```

Remove one or more requests from the engine and output processor.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_ids` | `list[str]` | *(required)* | IDs of requests to abort. |
| `internal` | `bool` | `False` | Whether this is an internal abort (e.g. due to stop strings). |

---

## Step Execution

### `step`

```python
def step() -> list[RequestOutput | PoolingRequestOutput]
```

Execute a single scheduling step and return all outputs produced in that step.

This is the core of the synchronous serving loop. Call `step` in a loop until all requests are finished:

```python
engine.add_request("req-1", "Hello", SamplingParams(max_tokens=64))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            print(output.outputs[0].text)
```

**Returns** `list[RequestOutput | PoolingRequestOutput]` — outputs produced in this step. A request appears in the list only when it has new output to report.

---

## Queue Inspection

### `has_unfinished_requests`

```python
def has_unfinished_requests() -> bool
```

Return `True` if there are any requests still being processed.

---

### `get_num_unfinished_requests`

```python
def get_num_unfinished_requests() -> int
```

Return the number of requests currently in the engine queue.

---

### `get_supported_tasks`

```python
def get_supported_tasks() -> tuple[SupportedTask, ...]
```

Return the tasks supported by the loaded model (e.g. `("generate",)` or `("embed", "classify")`).

---

## LoRA Management

### `add_lora`

```python
def add_lora(lora_request: LoRARequest) -> bool
```

Load a LoRA adapter into the engine for future requests. Returns `True` on success.

---

### `remove_lora`

```python
def remove_lora(lora_id: int) -> bool
```

Remove a loaded LoRA adapter by ID. Returns `True` on success.

---

### `list_loras`

```python
def list_loras() -> set[int]
```

Return the set of currently loaded LoRA adapter IDs.

---

### `pin_lora`

```python
def pin_lora(lora_id: int) -> bool
```

Prevent a LoRA adapter from being evicted from the cache. Returns `True` on success.

---

## Cache Management

### `reset_prefix_cache`

```python
def reset_prefix_cache(
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool
```

Invalidate the prefix (KV) cache.

---

### `reset_mm_cache`

```python
def reset_mm_cache() -> None
```

Clear the multimodal processor cache.

---

## Engine Control

### `sleep`

```python
def sleep(level: int = 1, mode: PauseMode = "abort") -> None
```

Put the engine into a low-power sleep state.

| Level | Effect |
|-------|--------|
| `0` | Pause scheduling; continue accepting requests. |
| `1` | Offload model weights to CPU and discard KV cache. |
| `2` | Discard all GPU memory (weights + KV cache). |

---

### `wake_up`

```python
def wake_up(tags: list[str] | None = None) -> None
```

Wake the engine from sleep mode.

---

### `is_sleeping`

```python
def is_sleeping() -> bool
```

Return `True` if the engine is currently sleeping.

---

## Profiling

### `start_profile`

```python
def start_profile(profile_prefix: str | None = None) -> None
```

Start profiling.

---

### `stop_profile`

```python
def stop_profile() -> None
```

Stop profiling and flush trace files.

---

## Distributed / RPC

### `collective_rpc`

```python
def collective_rpc(
    method: str | Callable,
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[Any]
```

Execute a method on all workers synchronously.

---

### `apply_model`

```python
def apply_model(func: Callable[[nn.Module], Any]) -> list[Any]
```

Run a function directly on the model inside each worker.

---

## Statistics & Observability

### `do_log_stats`

```python
def do_log_stats() -> None
```

Flush pending statistics to the configured stat loggers.

---

### `do_log_stats_with_interval`

```python
def do_log_stats_with_interval() -> None
```

Flush statistics only if the configured logging interval has elapsed.

---

### `get_metrics`

```python
def get_metrics() -> list[Metric]
```

Return a snapshot of all Prometheus metrics.

!!! note
    Requires `log_stats=True` (the default when using `from_engine_args`).

---

## Utilities

### `get_tokenizer`

```python
def get_tokenizer() -> TokenizerLike
```

Return the tokenizer used by the engine.

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `tokenizer` | `TokenizerLike \| None` | The tokenizer used by the engine. |
| `vllm_config` | `VllmConfig` | The resolved engine configuration. |
| `model_config` | `ModelConfig` | The resolved model configuration. |
| `renderer` | `Renderer` | The tokenizer/renderer used for prompt processing. |

---

## Custom Serving Loop Example

```python
from vllm import LLMEngine, EngineArgs, SamplingParams

engine = LLMEngine.from_engine_args(
    EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
)

prompts = [
    ("req-0", "Tell me a joke."),
    ("req-1", "What is the capital of France?"),
]

for req_id, prompt in prompts:
    engine.add_request(req_id, prompt, SamplingParams(max_tokens=64))

results = {}
while engine.has_unfinished_requests():
    for output in engine.step():
        if output.finished:
            results[output.request_id] = output.outputs[0].text

for req_id, text in results.items():
    print(f"[{req_id}] {text}")
```

---

## Data-Parallel Support

When `data_parallel_size > 1` with `distributed_executor_backend="external_launcher"`, `LLMEngine` coordinates across DP replicas using a process group:

```python
args = EngineArgs(
    model="...",
    data_parallel_size=4,
    distributed_executor_backend="external_launcher",
)
engine = LLMEngine.from_engine_args(args)
```

The `has_unfinished_requests` method automatically aggregates state across all DP replicas.
