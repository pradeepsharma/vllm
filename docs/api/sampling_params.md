# SamplingParams

```python
from vllm import SamplingParams
```

`SamplingParams` controls how tokens are sampled during text generation. It follows the OpenAI text completion API conventions and adds several vLLM-specific extensions.

---

## Constructor

```python
SamplingParams(
    n: int = 1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    bad_words: list[str] | None = None,
    ignore_eos: bool = False,
    max_tokens: int | None = 16,
    min_tokens: int = 0,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    flat_logprobs: bool = False,
    detokenize: bool = True,
    skip_special_tokens: bool = True,
    spaces_between_special_tokens: bool = True,
    include_stop_str_in_output: bool = False,
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
    structured_outputs: StructuredOutputsParams | None = None,
    logit_bias: dict[int, float] | None = None,
    allowed_token_ids: list[int] | None = None,
    extra_args: dict[str, Any] | None = None,
    repetition_detection: RepetitionDetectionParams | None = None,
)
```

---

## Parameters

### Sampling Strategy

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `temperature` | `float` | `1.0` | `≥ 0.0` | Controls randomness. `0.0` = greedy (deterministic), higher = more random. Values below `1e-2` are clamped to `1e-2` to avoid numerical issues. |
| `top_p` | `float` | `1.0` | `(0.0, 1.0]` | Nucleus sampling: only consider tokens whose cumulative probability exceeds `top_p`. `1.0` considers all tokens. |
| `top_k` | `int` | `0` | `≥ 0` | Top-k sampling: only consider the `k` most likely tokens. `0` (or `-1`) considers all tokens. |
| `min_p` | `float` | `0.0` | `[0.0, 1.0]` | Minimum probability threshold relative to the most likely token. `0.0` disables. |
| `seed` | `int \| None` | `None` | — | Random seed for reproducible sampling. `None` = non-deterministic. `-1` is treated as `None`. |

!!! tip "Greedy decoding"
    Set `temperature=0.0` for greedy (argmax) decoding. This automatically sets `top_p=1.0`, `top_k=0`, and `min_p=0.0`.

### Output Count

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1` | Number of output sequences to generate per prompt. When `n > 1` with `AsyncLLMEngine`, all outputs are streamed cumulatively. Use `output_kind=RequestOutputKind.FINAL_ONLY` to receive all outputs only at completion. |

### Penalties

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `presence_penalty` | `float` | `0.0` | `[-2.0, 2.0]` | Penalises tokens that have appeared at all in the generated text. Positive values encourage new tokens; negative values encourage repetition. |
| `frequency_penalty` | `float` | `0.0` | `[-2.0, 2.0]` | Penalises tokens proportional to how often they have appeared. Positive values encourage new tokens; negative values encourage repetition. |
| `repetition_penalty` | `float` | `1.0` | `> 0.0` | Multiplicative penalty applied to tokens that appear in the prompt or generated text. Values `> 1.0` discourage repetition; values `< 1.0` encourage it. |

### Length Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int \| None` | `16` | Maximum number of tokens to generate. `None` generates until EOS or model length limit. |
| `min_tokens` | `int` | `0` | Minimum number of tokens to generate before EOS or stop tokens are allowed. |
| `ignore_eos` | `bool` | `False` | Continue generating after the EOS token is produced. |

### Stop Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stop` | `str \| list[str] \| None` | `None` | Stop string(s). Generation halts when any stop string is produced. The stop string is **not** included in the output by default. |
| `stop_token_ids` | `list[int] \| None` | `None` | Stop token IDs. Generation halts when any of these token IDs is produced. Stop tokens **are** included in the output unless they are special tokens. |
| `bad_words` | `list[str] \| None` | `None` | Words that are forbidden in the output. Only the last token of a matching token sequence is blocked. |
| `include_stop_str_in_output` | `bool` | `False` | Include the stop string in the output text. |

### Log Probabilities

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logprobs` | `int \| None` | `None` | Number of top-token log probabilities to return per output token. `None` = no log probs. `-1` = all vocabulary log probs. The sampled token's log prob is always included, so the response may contain up to `logprobs + 1` entries. |
| `prompt_logprobs` | `int \| None` | `None` | Number of top-token log probabilities to return per prompt token. `-1` = all vocabulary log probs. |
| `flat_logprobs` | `bool` | `False` | Return log probs in a flat `FlatLogprob` format instead of `list[dict[int, Logprob]]`. Significantly reduces GC overhead for large vocabularies. |

### Detokenisation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detokenize` | `bool` | `True` | Detokenise the output token IDs into text. Set to `False` to receive only token IDs. |
| `skip_special_tokens` | `bool` | `True` | Omit special tokens (e.g. `<eos>`) from the output text. |
| `spaces_between_special_tokens` | `bool` | `True` | Add spaces between special tokens in the output text. |

### Streaming

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_kind` | `RequestOutputKind` | `CUMULATIVE` | Controls how outputs are streamed. See [`RequestOutputKind`](#requestoutputkind). |

### Structured Outputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structured_outputs` | `StructuredOutputsParams \| None` | `None` | Constrain generation to a specific format (JSON schema, regex, grammar, etc.). See [`StructuredOutputsParams`](#structuredoutputsparams). |

### Logit Manipulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logit_bias` | `dict[int, float] \| None` | `None` | Per-token logit biases. Keys are token IDs; values are bias amounts clamped to `[-100, 100]`. |
| `allowed_token_ids` | `list[int] \| None` | `None` | Restrict generation to only these token IDs. |

### Repetition Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repetition_detection` | `RepetitionDetectionParams \| None` | `None` | Detect and terminate repetitive N-gram patterns early. See [`RepetitionDetectionParams`](#repetitiondetectionparams). |

### Miscellaneous

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extra_args` | `dict[str, Any] \| None` | `None` | Arbitrary extra arguments for custom sampling implementations or plugins. Not used by any built-in sampling code. |

---

## Validation Rules

`SamplingParams` validates its fields on construction and raises `ValueError` or `VLLMValidationError` for invalid combinations:

- `n ≥ 1`
- `presence_penalty ∈ [-2, 2]`
- `frequency_penalty ∈ [-2, 2]`
- `repetition_penalty > 0`
- `temperature ≥ 0`
- `top_p ∈ (0, 1]`
- `top_k ≥ 0` (or `-1`)
- `min_p ∈ [0, 1]`
- `max_tokens ≥ 1` (when not `None`)
- `min_tokens ≥ 0`
- `min_tokens ≤ max_tokens` (when `max_tokens` is not `None`)
- `logprobs ≥ 0` or `logprobs == -1` (when not `None`)
- `prompt_logprobs ≥ 0` or `prompt_logprobs == -1` (when not `None`)
- `stop` strings must be non-empty

---

## Related Types

### `RequestOutputKind`

```python
from vllm.sampling_params import RequestOutputKind
```

Controls how outputs are delivered in streaming mode:

| Value | Description |
|-------|-------------|
| `CUMULATIVE` | *(default)* Each `RequestOutput` contains the full generated text so far. |
| `DELTA` | Each `RequestOutput` contains only the newly generated tokens since the last output. |
| `FINAL_ONLY` | No intermediate outputs; only the final `RequestOutput` is delivered. |

```python
from vllm.sampling_params import RequestOutputKind, SamplingParams

# Stream only deltas (efficient for large outputs)
params = SamplingParams(
    max_tokens=512,
    output_kind=RequestOutputKind.DELTA,
)

# Receive only the final result
params = SamplingParams(
    max_tokens=512,
    output_kind=RequestOutputKind.FINAL_ONLY,
)
```

---

### `StructuredOutputsParams`

```python
from vllm.sampling_params import StructuredOutputsParams
```

Constrain generation to a specific output format. Exactly one constraint field must be set.

| Field | Type | Description |
|-------|------|-------------|
| `json` | `str \| dict \| None` | JSON schema (as a string or dict) that the output must conform to. |
| `regex` | `str \| None` | Regular expression that the output must match. |
| `choice` | `list[str] \| None` | List of allowed output strings. |
| `grammar` | `str \| None` | EBNF grammar string that the output must conform to. |
| `json_object` | `bool \| None` | Constrain output to any valid JSON object. |
| `structural_tag` | `str \| None` | Structural tag format string. |
| `disable_fallback` | `bool` | Disable fallback to unconstrained generation on failure. |
| `disable_any_whitespace` | `bool` | Disable whitespace flexibility in JSON schema matching. |
| `disable_additional_properties` | `bool` | Disallow additional properties in JSON schema matching. |
| `whitespace_pattern` | `str \| None` | Custom whitespace pattern for JSON schema matching. |

**Example — JSON schema**

```python
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age":  {"type": "integer"},
    },
    "required": ["name", "age"],
}

params = SamplingParams(
    max_tokens=128,
    structured_outputs=StructuredOutputsParams(json=schema),
)
```

**Example — regex**

```python
params = SamplingParams(
    max_tokens=32,
    structured_outputs=StructuredOutputsParams(regex=r"\d{3}-\d{2}-\d{4}"),
)
```

**Example — choice**

```python
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(choice=["yes", "no", "maybe"]),
)
```

---

### `RepetitionDetectionParams`

```python
from vllm.sampling_params import RepetitionDetectionParams
```

Detect and terminate repetitive N-gram patterns early.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_pattern_size` | `int` | `0` | Maximum N-gram size to check. `0` disables detection. |
| `min_pattern_size` | `int` | `0` | Minimum N-gram size to check. Defaults to `1` when `0`. Must be `≤ max_pattern_size`. |
| `min_count` | `int` | `0` | Minimum number of repetitions to trigger termination. Must be `≥ 2`. |

**Example**

```python
from vllm import SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

params = SamplingParams(
    max_tokens=512,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=10,
        min_pattern_size=3,
        min_count=3,
    ),
)
```

---

## `from_optional` Factory

```python
@staticmethod
def from_optional(
    n: int | None = 1,
    presence_penalty: float | None = 0.0,
    frequency_penalty: float | None = 0.0,
    repetition_penalty: float | None = 1.0,
    temperature: float | None = 1.0,
    top_p: float | None = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    bad_words: list[str] | None = None,
    include_stop_str_in_output: bool = False,
    ignore_eos: bool = False,
    max_tokens: int | None = 16,
    min_tokens: int = 0,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    detokenize: bool = True,
    skip_special_tokens: bool = True,
    spaces_between_special_tokens: bool = True,
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
    structured_outputs: StructuredOutputsParams | None = None,
    logit_bias: dict[int, float] | dict[str, float] | None = None,
    allowed_token_ids: list[int] | None = None,
    extra_args: dict[str, Any] | None = None,
    skip_clone: bool = False,
    repetition_detection: RepetitionDetectionParams | None = None,
) -> "SamplingParams"
```

A factory method that accepts `None` for most numeric parameters and substitutes sensible defaults. Useful when constructing `SamplingParams` from API request objects where fields may be `None`.

`logit_bias` values are automatically clamped to `[-100, 100]` and string keys are converted to integers.

---

## Examples

### Greedy decoding

```python
from vllm import SamplingParams

params = SamplingParams(temperature=0.0, max_tokens=256)
```

### Creative generation

```python
params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    max_tokens=512,
    presence_penalty=0.3,
)
```

### Reproducible output

```python
params = SamplingParams(
    temperature=0.7,
    seed=42,
    max_tokens=128,
)
```

### Multiple outputs

```python
params = SamplingParams(
    n=5,
    temperature=1.0,
    max_tokens=64,
)
outputs = llm.generate("Tell me a joke.", sampling_params=params)
for completion in outputs[0].outputs:
    print(f"[{completion.index}] {completion.text}")
```

### With log probabilities

```python
params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
    logprobs=5,          # top-5 log probs per output token
    prompt_logprobs=1,   # top-1 log prob per prompt token
)
```

### Constrained to a stop string

```python
params = SamplingParams(
    max_tokens=256,
    stop=["###", "\n\n"],
    include_stop_str_in_output=False,
)
```
