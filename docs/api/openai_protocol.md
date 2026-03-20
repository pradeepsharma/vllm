# OpenAI Protocol Types

vLLM implements an OpenAI-compatible HTTP API. The protocol types defined here are the Pydantic models that power the request/response serialisation for every endpoint.

All models extend `OpenAIBaseModel`, which is a Pydantic `BaseModel` with `extra="allow"` — unknown fields are accepted and logged at DEBUG level rather than causing validation errors.

---

## Base Types

### `OpenAIBaseModel`

```python
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
```

Base class for all OpenAI protocol models. Accepts extra fields silently (logs them at DEBUG level).

---

### `UsageInfo`

```python
class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None
```

Token usage statistics included in all non-streaming responses.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Number of tokens in the input prompt. |
| `total_tokens` | `int` | Total tokens (prompt + completion). |
| `completion_tokens` | `int \| None` | Number of generated tokens. |
| `prompt_tokens_details` | `PromptTokenUsageInfo \| None` | Breakdown of prompt tokens (e.g. cached tokens). |

---

### `PromptTokenUsageInfo`

```python
class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None
```

| Field | Type | Description |
|-------|------|-------------|
| `cached_tokens` | `int \| None` | Number of prompt tokens served from the prefix cache. |

---

### `ErrorResponse`

```python
class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo
```

Returned by all endpoints on error.

```python
class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int
```

---

### `StreamOptions`

```python
class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = True
    continuous_usage_stats: bool | None = False
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_usage` | `bool \| None` | `True` | Include `usage` in the final streaming chunk. |
| `continuous_usage_stats` | `bool \| None` | `False` | Include `usage` in every streaming chunk. |

---

## Response Format Types

### `ResponseFormat`

```python
class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None
```

Controls the output format for completions.

| `type` value | Description |
|---|---|
| `"text"` | Plain text output (default). |
| `"json_object"` | Constrain output to any valid JSON object. |
| `"json_schema"` | Constrain output to a specific JSON schema. |

---

### `JsonSchemaResponseFormat`

```python
class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None = None
    json_schema: dict[str, Any] | None = None  # field alias: "schema"
    strict: bool | None = None
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the schema. |
| `description` | `str \| None` | Human-readable description. |
| `json_schema` | `dict \| None` | The JSON schema definition (sent as `"schema"` in the API). |
| `strict` | `bool \| None` | Whether to enforce strict schema validation. |

---

### `StructuralTagResponseFormat`

```python
class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any
```

vLLM-specific structural tag format for constrained generation.

---

## Tool Types

### `FunctionDefinition`

```python
class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
```

Defines a callable function for tool use.

---

### `FunctionCall`

```python
class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str  # JSON-encoded arguments
```

A function call produced by the model.

---

### `ToolCall`

```python
class ToolCall(OpenAIBaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall
```

A tool call produced by the model.

---

### `DeltaMessage`

```python
class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = []
```

Incremental message content in streaming chat responses.

---

### `DeltaToolCall`

```python
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None
```

Incremental tool call in streaming responses.

---

## Chat Completion

### `ChatCompletionRequest`

```python
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
```

Request body for `POST /v1/chat/completions`.

#### Standard OpenAI fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[ChatCompletionMessageParam]` | *(required)* | Conversation history. Each message has `"role"` and `"content"` keys. |
| `model` | `str \| None` | `None` | Model name (optional when only one model is served). |
| `frequency_penalty` | `float \| None` | `0.0` | Frequency penalty `[-2, 2]`. |
| `logit_bias` | `dict[str, float] \| None` | `None` | Per-token logit biases. |
| `logprobs` | `bool \| None` | `False` | Return log probabilities for output tokens. |
| `top_logprobs` | `int \| None` | `0` | Number of top log probabilities to return per token. |
| `max_tokens` | `int \| None` | `None` | **Deprecated.** Use `max_completion_tokens`. |
| `max_completion_tokens` | `int \| None` | `None` | Maximum number of tokens to generate. |
| `n` | `int \| None` | `1` | Number of completions to generate. |
| `presence_penalty` | `float \| None` | `0.0` | Presence penalty `[-2, 2]`. |
| `response_format` | `ResponseFormat \| StructuralTagResponseFormat \| None` | `None` | Output format constraint. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |
| `stop` | `str \| list[str] \| None` | `[]` | Stop strings. |
| `stream` | `bool \| None` | `False` | Enable streaming. |
| `stream_options` | `StreamOptions \| None` | `None` | Streaming options. |
| `temperature` | `float \| None` | `None` | Sampling temperature. |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability. |
| `tools` | `list[ChatCompletionToolsParam] \| None` | `None` | Available tools. |
| `tool_choice` | `str \| ChatCompletionNamedToolChoiceParam \| None` | `"none"` | Tool selection mode: `"none"`, `"auto"`, `"required"`, or a specific tool. |
| `reasoning_effort` | `Literal["low", "medium", "high"] \| None` | `None` | Reasoning effort level for reasoning models. |
| `parallel_tool_calls` | `bool \| None` | `True` | Allow parallel tool calls. |
| `user` | `str \| None` | `None` | User identifier (ignored by vLLM). |

#### vLLM sampling extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_beam_search` | `bool` | `False` | Use beam search instead of sampling. |
| `top_k` | `int \| None` | `None` | Top-k sampling. |
| `min_p` | `float \| None` | `None` | Minimum probability threshold. |
| `repetition_penalty` | `float \| None` | `None` | Repetition penalty. |
| `length_penalty` | `float` | `1.0` | Length penalty for beam search. |
| `stop_token_ids` | `list[int] \| None` | `[]` | Stop token IDs. |
| `include_stop_str_in_output` | `bool` | `False` | Include stop string in output. |
| `ignore_eos` | `bool` | `False` | Ignore EOS token. |
| `min_tokens` | `int` | `0` | Minimum tokens to generate. |
| `skip_special_tokens` | `bool` | `True` | Skip special tokens in output. |
| `spaces_between_special_tokens` | `bool` | `True` | Add spaces between special tokens. |
| `truncate_prompt_tokens` | `int \| None` | `None` | Truncate prompt to this many tokens. |
| `prompt_logprobs` | `int \| None` | `None` | Number of prompt token log probs to return. |
| `allowed_token_ids` | `list[int] \| None` | `None` | Restrict generation to these token IDs. |
| `bad_words` | `list[str]` | `[]` | Forbidden words. |
| `repetition_detection` | `RepetitionDetectionParams \| None` | `None` | Repetition detection configuration. |

#### vLLM extra extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `echo` | `bool` | `False` | Prepend the last message if it has the same role. |
| `add_generation_prompt` | `bool` | `True` | Add generation prompt to chat template. |
| `continue_final_message` | `bool` | `False` | Continue the final message instead of starting a new one. |
| `add_special_tokens` | `bool` | `False` | Add special tokens on top of chat template tokens. |
| `documents` | `list[dict[str, str]] \| None` | `None` | RAG documents for models that support it. |
| `chat_template` | `str \| None` | `None` | Override the model's chat template. |
| `chat_template_kwargs` | `dict \| None` | `None` | Extra kwargs for the chat template. |
| `mm_processor_kwargs` | `dict \| None` | `None` | Extra kwargs for the multimodal processor. |
| `structured_outputs` | `StructuredOutputsParams \| None` | `None` | Structured output configuration. |
| `priority` | `int` | `0` | Request scheduling priority. |
| `request_id` | `str` | *(auto)* | Unique request identifier. |
| `return_tokens_as_token_ids` | `bool \| None` | `None` | Return tokens as `"token_id:{id}"` strings. |
| `return_token_ids` | `bool \| None` | `None` | Include token IDs in the response. |
| `cache_salt` | `str \| None` | `None` | Salt for prefix cache isolation in multi-user environments. |
| `kv_transfer_params` | `dict \| None` | `None` | KV transfer parameters for disaggregated serving. |
| `vllm_xargs` | `dict \| None` | `None` | Custom extension parameters. |
| `include_reasoning` | `bool` | `True` | Include reasoning tokens in the response. |

---

### `ChatCompletionResponse`

```python
class ChatCompletionResponse(OpenAIBaseModel):
    id: str                                    # "chatcmpl-{uuid}"
    object: Literal["chat.completion"]
    created: int                               # Unix timestamp
    model: str
    choices: list[ChatCompletionResponseChoice]
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo

    # vLLM extensions
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    prompt_token_ids: list[int] | None = None
    kv_transfer_params: dict | None = None
```

---

### `ChatCompletionResponseChoice`

```python
class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = "stop"
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None
```

---

### `ChatMessage`

```python
class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] = []

    # vLLM extension
    reasoning: str | None = None
```

---

### `ChatCompletionStreamResponse`

```python
class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None
    prompt_token_ids: list[int] | None = None
```

---

### `ChatCompletionResponseStreamChoice`

```python
class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None
```

---

## Text Completion

### `CompletionRequest`

```python
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
```

Request body for `POST /v1/completions`.

#### Standard OpenAI fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model name. |
| `prompt` | `str \| list[str] \| list[int] \| list[list[int]] \| None` | `None` | Input prompt(s). |
| `echo` | `bool \| None` | `False` | Echo the prompt in the response. |
| `frequency_penalty` | `float \| None` | `0.0` | Frequency penalty. |
| `logit_bias` | `dict[str, float] \| None` | `None` | Per-token logit biases. |
| `logprobs` | `int \| None` | `None` | Number of log probabilities to return. |
| `max_tokens` | `int \| None` | `16` | Maximum tokens to generate. |
| `n` | `int` | `1` | Number of completions. |
| `presence_penalty` | `float \| None` | `0.0` | Presence penalty. |
| `seed` | `int \| None` | `None` | Random seed. |
| `stop` | `str \| list[str] \| None` | `[]` | Stop strings. |
| `stream` | `bool \| None` | `False` | Enable streaming. |
| `stream_options` | `StreamOptions \| None` | `None` | Streaming options. |
| `suffix` | `str \| None` | `None` | Suffix to append (not supported by all models). |
| `temperature` | `float \| None` | `None` | Sampling temperature. |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability. |
| `user` | `str \| None` | `None` | User identifier (ignored). |

#### vLLM sampling extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_beam_search` | `bool` | `False` | Use beam search. |
| `top_k` | `int \| None` | `None` | Top-k sampling. |
| `min_p` | `float \| None` | `None` | Minimum probability threshold. |
| `repetition_penalty` | `float \| None` | `None` | Repetition penalty. |
| `length_penalty` | `float` | `1.0` | Length penalty for beam search. |
| `stop_token_ids` | `list[int] \| None` | `[]` | Stop token IDs. |
| `include_stop_str_in_output` | `bool` | `False` | Include stop string in output. |
| `ignore_eos` | `bool` | `False` | Ignore EOS token. |
| `min_tokens` | `int` | `0` | Minimum tokens to generate. |
| `skip_special_tokens` | `bool` | `True` | Skip special tokens. |
| `spaces_between_special_tokens` | `bool` | `True` | Add spaces between special tokens. |
| `truncate_prompt_tokens` | `int \| None` | `None` | Truncate prompt to this many tokens. |
| `allowed_token_ids` | `list[int] \| None` | `None` | Restrict generation to these token IDs. |
| `prompt_logprobs` | `int \| None` | `None` | Number of prompt token log probs. |
| `repetition_detection` | `RepetitionDetectionParams \| None` | `None` | Repetition detection. |

#### vLLM extra extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_embeds` | `bytes \| list[bytes] \| None` | `None` | Pre-computed prompt embeddings. |
| `add_special_tokens` | `bool` | `True` | Add special tokens (e.g. BOS) to the prompt. |
| `response_format` | `ResponseFormat \| None` | `None` | Output format constraint. |
| `structured_outputs` | `StructuredOutputsParams \| None` | `None` | Structured output configuration. |
| `priority` | `int` | `0` | Request scheduling priority. |
| `request_id` | `str` | *(auto)* | Unique request identifier. |
| `return_tokens_as_token_ids` | `bool \| None` | `None` | Return tokens as `"token_id:{id}"` strings. |
| `return_token_ids` | `bool \| None` | `None` | Include token IDs in the response. |
| `cache_salt` | `str \| None` | `None` | Prefix cache salt. |
| `kv_transfer_params` | `dict \| None` | `None` | KV transfer parameters. |
| `vllm_xargs` | `dict \| None` | `None` | Custom extension parameters. |

---

### `CompletionResponse`

```python
class CompletionResponse(OpenAIBaseModel):
    id: str                                  # "cmpl-{uuid}"
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    service_tier: str | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo
    kv_transfer_params: dict | None = None
```

---

### `CompletionResponseChoice`

```python
class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    prompt_token_ids: list[int] | None = None
```

---

### `CompletionLogProbs`

```python
class CompletionLogProbs(OpenAIBaseModel):
    text_offset: list[int] = []
    token_logprobs: list[float | None] = []
    tokens: list[str] = []
    top_logprobs: list[dict[str, float] | None] = []
```

---

### `CompletionStreamResponse`

```python
class CompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = None
```

---

## Embeddings

### `EmbeddingCompletionRequest` / `EmbeddingChatRequest`

```python
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest,
    EmbeddingChatRequest,
    EmbeddingRequest,  # TypeAlias for either
)
```

Request body for `POST /v1/embeddings`.

`EmbeddingCompletionRequest` accepts a plain text or token-ID prompt; `EmbeddingChatRequest` accepts a chat-format prompt.

Common fields (from mixins):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model name. |
| `encoding_format` | `str` | `"float"` | Output format: `"float"` or `"base64"`. |
| `dimensions` | `int \| None` | `None` | Reduce embedding dimensions (Matryoshka models). |
| `use_activation` | `bool \| None` | `None` | Apply activation function to pooler output. |
| `truncate_prompt_tokens` | `int \| None` | `None` | Truncate prompt to this many tokens. |
| `add_special_tokens` | `bool` | `True` | Add special tokens to the prompt. |
| `priority` | `int` | `0` | Request scheduling priority. |
| `request_id` | `str` | *(auto)* | Unique request identifier. |

---

### `EmbeddingResponse`

```python
class EmbeddingResponse(OpenAIBaseModel):
    id: str                    # "embd-{uuid}"
    object: str = "list"
    created: int
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo
```

---

### `EmbeddingResponseData`

```python
class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str  # str when encoding_format="base64"
```

---

## Scoring / Reranking

### `ScoreRequest`

```python
from vllm.entrypoints.pooling.score.protocol import (
    ScoreRequest,               # TypeAlias
    ScoreDataRequest,           # data_1 / data_2
    ScoreQueriesDocumentsRequest,  # queries / documents
    ScoreQueriesItemsRequest,   # queries / items
    ScoreTextRequest,           # text_1 / text_2
)
```

Request body for `POST /v1/score`. Multiple request formats are supported:

**`ScoreDataRequest`**

```python
class ScoreDataRequest(ScoreRequestMixin):
    data_1: ScoreInputs   # query/queries
    data_2: ScoreInputs   # document/documents
```

**`ScoreQueriesDocumentsRequest`**

```python
class ScoreQueriesDocumentsRequest(ScoreRequestMixin):
    queries: ScoreInputs
    documents: ScoreInputs
```

**`ScoreTextRequest`**

```python
class ScoreTextRequest(ScoreRequestMixin):
    text_1: ScoreInputs
    text_2: ScoreInputs
```

Common fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str \| None` | `None` | Model name. |
| `use_activation` | `bool \| None` | `None` | Apply activation function. |
| `truncate_prompt_tokens` | `int \| None` | `None` | Truncate prompt. |
| `priority` | `int` | `0` | Scheduling priority. |
| `request_id` | `str` | *(auto)* | Unique request identifier. |

---

### `ScoreResponse`

```python
class ScoreResponse(OpenAIBaseModel):
    id: str
    object: str = "list"
    created: int
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo
```

---

### `ScoreResponseData`

```python
class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float
```

---

### `RerankRequest`

```python
class RerankRequest(PoolingBasicRequestMixin, ClassifyRequestMixin):
    query: ScoreInput
    documents: ScoreInputs
    top_n: int = 0
```

Request body for `POST /v1/rerank`. Returns the top-N most relevant documents.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str \| list[...]` | *(required)* | The query to rank documents against. |
| `documents` | `list[str \| ...]` | *(required)* | Documents to rank. |
| `top_n` | `int` | `0` | Number of top results to return. `0` returns all. |

---

### `RerankResponse`

```python
class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]
```

---

### `RerankResult`

```python
class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float
```

---

## Models

### `ModelCard`

```python
class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = []
```

---

### `ModelList`

```python
class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = []
```

Response for `GET /v1/models`.

---

## Logits Processors

### `LogitsProcessorConstructor`

```python
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
```

Specifies a custom logits processor by its fully-qualified class name.

```python
# In a ChatCompletionRequest or CompletionRequest:
{
    "logits_processors": [
        {
            "qualname": "mypackage.processors.MyProcessor",
            "kwargs": {"threshold": 0.5}
        }
    ]
}
```

!!! note
    The server must be started with `--logits-processor-pattern` to allow custom logits processors.

---

## HTTP Endpoints Summary

| Method | Path | Request Type | Response Type |
|--------|------|-------------|---------------|
| `POST` | `/v1/chat/completions` | `ChatCompletionRequest` | `ChatCompletionResponse` / `ChatCompletionStreamResponse` |
| `POST` | `/v1/completions` | `CompletionRequest` | `CompletionResponse` / `CompletionStreamResponse` |
| `POST` | `/v1/embeddings` | `EmbeddingRequest` | `EmbeddingResponse` |
| `POST` | `/v1/score` | `ScoreRequest` | `ScoreResponse` |
| `POST` | `/v1/rerank` | `RerankRequest` | `RerankResponse` |
| `GET` | `/v1/models` | — | `ModelList` |
| `GET` | `/health` | — | `200 OK` |

---

## Example: Chat Completion Request

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "What is the capital of France?"},
        ],
        "temperature": 0.7,
        "max_completion_tokens": 128,
        "stream": False,
    },
)
data = response.json()
print(data["choices"][0]["message"]["content"])
```

## Example: Embeddings Request

```python
response = httpx.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "BAAI/bge-base-en-v1.5",
        "input": ["Hello world", "vLLM is fast"],
    },
)
data = response.json()
for item in data["data"]:
    print(f"Index {item['index']}: {item['embedding'][:5]}...")
```

## Example: Streaming Chat Completion

```python
import httpx
import json

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "stream": True,
        "stream_options": {"include_usage": True},
    },
) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                print(delta["content"], end="", flush=True)
```
