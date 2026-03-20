# AsyncLLMEngine

```python
from vllm import AsyncLLMEngine
```

`AsyncLLMEngine` (backed by the internal `AsyncLLM` class) is the **asynchronous engine** used by vLLM's OpenAI-compatible HTTP server and any other production serving stack. It accepts requests concurrently, processes them in a background event loop, and streams outputs back to callers via `AsyncGenerator`.

!!! note "Online vs. offline"
    `AsyncLLMEngine` is designed for **online serving** with many concurrent clients. For offline batch inference, use the simpler synchronous [`LLM`](llm.md) class instead.

---

## Class Hierarchy

```
AsyncLLMEngine  (vllm.engine.async_llm_engine)
    └── AsyncLLM  (vllm.v1.engine.async_llm)   ← actual implementation
```

The public `AsyncLLMEngine` symbol re-exports `AsyncLLM` for backwards compatibility.

---

## Construction

### `from_engine_args` *(classmethod)*

```python
@classmethod
def from_engine_args(
    engine_args: AsyncEngineArgs,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
) -> "AsyncLLM"
```

The recommended way to create an `AsyncLLMEngine`. Accepts an [`AsyncEngineArgs`](engine_args.md) instance and returns a fully initialised engine.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_args` | `AsyncEngineArgs` | *(required)* | Engine configuration. |
| `start_engine_loop` | `bool` | `True` | Start the background output-handler loop immediately. |
| `usage_context` | `UsageContext` | `ENGINE_CONTEXT` | Usage context for telemetry. |
| `stat_loggers` | `list[StatLoggerFactory] \| None` | `None` | Custom stat logger factories. |

**Example**

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs

args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
)
engine = AsyncLLMEngine.from_engine_args(args)
```

---

### `from_vllm_config` *(classmethod)*

```python
@classmethod
def from_vllm_config(
    vllm_config: VllmConfig,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    enable_log_requests: bool = False,
    aggregate_engine_logging: bool = False,
    disable_log_stats: bool = False,
    client_addresses: dict[str, str] | None = None,
    client_count: int = 1,
    client_index: int = 0,
) -> "AsyncLLM"
```

Create an `AsyncLLM` from a pre-built `VllmConfig`. Useful for advanced use cases where you construct the config object yourself.

---

### `__init__`

```python
AsyncLLM(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    use_cached_outputs: bool = False,
    log_requests: bool = True,
    start_engine_loop: bool = True,
    stat_loggers: list[StatLoggerFactory] | None = None,
    aggregate_engine_logging: bool = False,
    client_addresses: dict[str, str] | None = None,
    client_count: int = 1,
    client_index: int = 0,
)
```

Direct constructor. Prefer `from_engine_args` or `from_vllm_config` in most cases.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_config` | `VllmConfig` | *(required)* | Global engine configuration. |
| `executor_class` | `type[Executor]` | *(required)* | Executor implementation (e.g. `MultiprocExecutor`). |
| `log_stats` | `bool` | *(required)* | Enable default Prometheus stat logging. |
| `log_requests` | `bool` | `True` | Log request IDs and parameters at INFO level. |
| `start_engine_loop` | `bool` | `True` | Start the background output-handler task immediately. |
| `stat_loggers` | `list[StatLoggerFactory] \| None` | `None` | Custom stat logger factories. |
| `aggregate_engine_logging` | `bool` | `False` | Aggregate logging across multiple engine instances. |
| `client_count` | `int` | `1` | Total number of API clients sharing this engine. |
| `client_index` | `int` | `0` | Index of this client (for multi-client setups). |

---

## Core Methods

### `generate`

```python
async def generate(
    prompt: EngineCoreRequest
          | PromptType
          | ProcessorInputs
          | AsyncGenerator[StreamingInput, None],
    sampling_params: SamplingParams,
    request_id: str,
    *,
    prompt_text: str | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
    reasoning_ended: bool | None = None,
) -> AsyncGenerator[RequestOutput, None]
```

The primary method for text generation. Called by the OpenAI-compatible server for each `/v1/completions` and `/v1/chat/completions` request.

Internally, this method:

1. Converts the prompt into an `EngineCoreRequest`.
2. Registers the request with the `OutputProcessor`.
3. Sends the request to the `EngineCore` (a separate process).
4. Yields `RequestOutput` objects as they are produced by the background output-handler loop.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `PromptType \| ProcessorInputs \| AsyncGenerator \| EngineCoreRequest` | *(required)* | The input prompt. Accepts a string, token IDs, multimodal dict, pre-processed `ProcessorInputs`, or an async generator for streaming inputs. |
| `sampling_params` | `SamplingParams` | *(required)* | Sampling configuration. |
| `request_id` | `str` | *(required)* | Unique identifier for this request. |
| `lora_request` | `LoRARequest \| None` | `None` | LoRA adapter to apply. |
| `tokenization_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to `tokenizer.encode`. |
| `trace_headers` | `Mapping[str, str] \| None` | `None` | OpenTelemetry trace headers for distributed tracing. |
| `priority` | `int` | `0` | Request scheduling priority (lower = higher priority). |
| `data_parallel_rank` | `int \| None` | `None` | Target data-parallel rank for this request. |
| `reasoning_ended` | `bool \| None` | `None` | Signal that reasoning has ended (for reasoning models). |

**Yields** `RequestOutput` — streaming output objects. Each object contains the cumulative or delta output depending on `sampling_params.output_kind`.

**Raises**

- `EngineDeadError` — if the engine has crashed.
- `ValueError` — for invalid request parameters.
- `asyncio.CancelledError` — if the client disconnects (request is automatically aborted).

**Example**

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
)

async def main():
    async for output in engine.generate(
        "Once upon a time",
        SamplingParams(max_tokens=128, temperature=0.8),
        request_id="req-001",
    ):
        if output.finished:
            print(output.outputs[0].text)

asyncio.run(main())
```

---

### `encode`

```python
async def encode(
    prompt: PromptType | ProcessorInputs,
    pooling_params: PoolingParams,
    request_id: str,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    tokenization_kwargs: dict[str, Any] | None = None,
    reasoning_ended: bool | None = None,
) -> AsyncGenerator[PoolingRequestOutput, None]
```

Asynchronous pooling / embedding. Called by the server for `/v1/embeddings` requests.

**Yields** `PoolingRequestOutput`

---

### `add_request`

```python
async def add_request(
    request_id: str,
    prompt: EngineCoreRequest
          | PromptType
          | ProcessorInputs
          | AsyncGenerator[StreamingInput, None],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
    prompt_text: str | None = None,
    reasoning_ended: bool | None = None,
) -> RequestOutputCollector
```

Add a request to the engine without immediately iterating its output. Returns a `RequestOutputCollector` queue that the caller can pull from.

This is the lower-level primitive used by `generate` and `encode`. Use it when you need to manage the output queue yourself.

---

### `abort`

```python
async def abort(
    request_id: str | Iterable[str],
    internal: bool = False,
) -> None
```

Abort one or more in-flight requests.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_id` | `str \| Iterable[str]` | *(required)* | Request ID(s) to abort. |
| `internal` | `bool` | `False` | Whether this is an internal abort (e.g. due to stop strings). |

---

## Lifecycle Management

### `shutdown`

```python
def shutdown(timeout: float | None = None) -> None
```

Shut down the engine, cleaning up background processes and IPC resources.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float \| None` | `None` | Maximum time to wait for clean shutdown. |

---

### `sleep`

```python
async def sleep(level: int = 1, mode: PauseMode = "abort") -> None
```

Put the engine into a low-power sleep state. See [`LLM.sleep`](llm.md#sleep) for level descriptions.

---

### `wake_up`

```python
async def wake_up(tags: list[str] | None = None) -> None
```

Wake the engine from sleep mode.

---

### `is_sleeping`

```python
async def is_sleeping() -> bool
```

Return `True` if the engine is currently sleeping.

---

## Generation Control

### `pause_generation`

```python
async def pause_generation(
    *,
    mode: PauseMode = "abort",
    wait_for_inflight_requests: bool | None = None,
    clear_cache: bool = True,
) -> None
```

Pause generation to allow model weight updates (e.g. during RL training).

New generation and encoding requests will not be scheduled until `resume_generation` is called.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"abort"` | How to handle in-flight requests: `"abort"` cancels them immediately, `"wait"` waits for completion, `"keep"` freezes them for resumption. |
| `wait_for_inflight_requests` | `bool \| None` | `None` | **Deprecated.** Use `mode="wait"` instead. |
| `clear_cache` | `bool` | `True` | Clear KV cache and prefix cache after draining. Set to `False` to preserve cache for faster resume. |

---

### `resume_generation`

```python
async def resume_generation() -> None
```

Resume generation after `pause_generation`.

---

### `is_paused`

```python
async def is_paused() -> bool
```

Return `True` if the engine is currently paused.

---

## LoRA Management

### `add_lora`

```python
async def add_lora(lora_request: LoRARequest) -> bool
```

Load a LoRA adapter into the engine. Returns `True` on success.

---

### `remove_lora`

```python
async def remove_lora(lora_id: int) -> bool
```

Remove a loaded LoRA adapter by ID. Returns `True` on success.

---

### `list_loras`

```python
async def list_loras() -> set[int]
```

Return the set of currently loaded LoRA adapter IDs.

---

### `pin_lora`

```python
async def pin_lora(lora_id: int) -> bool
```

Prevent a LoRA adapter from being evicted from the cache. Returns `True` on success.

---

## Cache Management

### `reset_prefix_cache`

```python
async def reset_prefix_cache(
    reset_running_requests: bool = False,
    reset_connector: bool = False,
) -> bool
```

Invalidate the prefix (KV) cache.

---

### `reset_mm_cache`

```python
async def reset_mm_cache() -> None
```

Clear the multimodal processor cache.

---

### `reset_encoder_cache`

```python
async def reset_encoder_cache() -> None
```

Clear the encoder cache (for encoder-decoder models).

---

## Profiling

### `start_profile`

```python
async def start_profile(profile_prefix: str | None = None) -> None
```

Start profiling.

---

### `stop_profile`

```python
async def stop_profile() -> None
```

Stop profiling and flush trace files.

---

## Distributed / RPC

### `collective_rpc`

```python
async def collective_rpc(
    method: str | Callable,
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[Any]
```

Execute a method on all workers asynchronously.

---

## Weight Management

### `init_weight_transfer_engine`

```python
async def init_weight_transfer_engine(
    request: WeightTransferInitRequest,
) -> None
```

Initialise the weight transfer backend for RL training.

---

### `update_weights`

```python
async def update_weights(request: WeightTransferUpdateRequest) -> None
```

Update model weights at runtime (for RL training).

---

## Health & Observability

### `check_health`

```python
async def check_health() -> None
```

Check engine health. Raises an exception if the engine is unhealthy.

---

### `do_log_stats`

```python
async def do_log_stats() -> None
```

Flush pending statistics to the configured stat loggers.

---

### `is_tracing_enabled`

```python
async def is_tracing_enabled() -> bool
```

Return `True` if OpenTelemetry tracing is enabled.

---

### `get_supported_tasks`

```python
async def get_supported_tasks() -> tuple[SupportedTask, ...]
```

Return the tasks supported by the loaded model (e.g. `("generate",)` or `("embed", "classify")`).

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `tokenizer` | `TokenizerLike \| None` | The tokenizer used by the engine. |
| `is_running` | `bool` | `True` if the output-handler loop is active. |
| `is_stopped` | `bool` | `True` if the engine has stopped (equivalent to `errored`). |
| `errored` | `bool` | `True` if the engine has encountered a fatal error. |
| `dead_error` | `BaseException` | The exception that caused the engine to die. |

---

## Streaming Inputs

`AsyncLLMEngine.generate` accepts an `AsyncGenerator[StreamingInput, None]` as the `prompt` argument. This enables **incremental prompt delivery** — useful for real-time transcription or other streaming input sources.

```python
from vllm.engine.protocol import StreamingInput

async def my_input_stream():
    for chunk in ["Hello", ", ", "world"]:
        yield StreamingInput(prompt=chunk)

async for output in engine.generate(
    my_input_stream(),
    SamplingParams(max_tokens=64),
    request_id="stream-001",
):
    print(output.outputs[0].text)
```

!!! warning "Streaming input limitations"
    Streaming inputs do not support `n > 1`, `output_kind=FINAL_ONLY`, or stop strings.

---

## Usage in the OpenAI Server

The OpenAI-compatible server creates a single `AsyncLLMEngine` instance at startup and routes all requests through it:

```python
# Simplified server startup
from vllm import AsyncLLMEngine, AsyncEngineArgs

engine_args = AsyncEngineArgs(model="...", ...)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Each request handler calls:
async for output in engine.generate(prompt, params, request_id):
    yield output
```

For full server usage, see `vllm serve --help` or the [OpenAI-compatible server guide](../serving/openai_compatible_server.md).
