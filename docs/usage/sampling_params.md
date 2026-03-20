# SamplingParams Reference

[`SamplingParams`][vllm.SamplingParams] controls every aspect of how vLLM generates text: randomness, length, stopping conditions, penalties, log probabilities, and structured output constraints.

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)
```

!!! tip "Default Sampling Parameters"
    By default, vLLM applies the model creator's recommended sampling parameters from `generation_config.json` (if present in the HuggingFace repository). To use vLLM's own defaults instead, set `generation_config="vllm"` when constructing `LLM`, or pass `--generation-config vllm` to `vllm serve`.

---

## Quick Reference

| Parameter | Type | Default | Category |
|---|---|---|---|
| [`n`](#n) | `int` | `1` | Output count |
| [`temperature`](#temperature) | `float` | `1.0` | Randomness |
| [`top_p`](#top_p) | `float` | `1.0` | Sampling |
| [`top_k`](#top_k) | `int` | `0` | Sampling |
| [`min_p`](#min_p) | `float` | `0.0` | Sampling |
| [`seed`](#seed) | `int \| None` | `None` | Reproducibility |
| [`max_tokens`](#max_tokens) | `int \| None` | `16` | Length |
| [`min_tokens`](#min_tokens) | `int` | `0` | Length |
| [`stop`](#stop) | `str \| list[str] \| None` | `None` | Stopping |
| [`stop_token_ids`](#stop_token_ids) | `list[int] \| None` | `None` | Stopping |
| [`ignore_eos`](#ignore_eos) | `bool` | `False` | Stopping |
| [`include_stop_str_in_output`](#include_stop_str_in_output) | `bool` | `False` | Stopping |
| [`presence_penalty`](#presence_penalty) | `float` | `0.0` | Penalties |
| [`frequency_penalty`](#frequency_penalty) | `float` | `0.0` | Penalties |
| [`repetition_penalty`](#repetition_penalty) | `float` | `1.0` | Penalties |
| [`bad_words`](#bad_words) | `list[str] \| None` | `None` | Penalties |
| [`repetition_detection`](#repetition_detection) | `RepetitionDetectionParams \| None` | `None` | Penalties |
| [`logprobs`](#logprobs) | `int \| None` | `None` | Log probs |
| [`prompt_logprobs`](#prompt_logprobs) | `int \| None` | `None` | Log probs |
| [`flat_logprobs`](#flat_logprobs) | `bool` | `False` | Log probs |
| [`logit_bias`](#logit_bias) | `dict[int, float] \| None` | `None` | Logit control |
| [`allowed_token_ids`](#allowed_token_ids) | `list[int] \| None` | `None` | Logit control |
| [`structured_outputs`](#structured_outputs) | `StructuredOutputsParams \| None` | `None` | Structured output |
| [`detokenize`](#detokenize) | `bool` | `True` | Output format |
| [`skip_special_tokens`](#skip_special_tokens) | `bool` | `True` | Output format |
| [`spaces_between_special_tokens`](#spaces_between_special_tokens) | `bool` | `True` | Output format |
| [`output_kind`](#output_kind) | `RequestOutputKind` | `CUMULATIVE` | Streaming |
| [`extra_args`](#extra_args) | `dict[str, Any] \| None` | `None` | Advanced |

---

## Output Count

### `n`

**Type:** `int` · **Default:** `1`

Number of independent output sequences to generate for each prompt. All `n` outputs are generated in parallel.

```python
# Generate 4 different completions for the same prompt
params = SamplingParams(n=4, temperature=0.9, max_tokens=100)
outputs = llm.generate("Once upon a time", params)

for i, completion in enumerate(outputs[0].outputs):
    print(f"[{i}] {completion.text!r}")
```

!!! note "Streaming with n > 1"
    When using `AsyncLLM` with `n > 1`, all outputs are streamed cumulatively. To receive all `n` outputs only upon completion, set `output_kind=RequestOutputKind.FINAL_ONLY`.

---

## Randomness & Sampling

### `temperature`

**Type:** `float` · **Default:** `1.0` · **Range:** `[0.0, ∞)`

Controls the randomness of token sampling by scaling the logits before applying softmax.

- `temperature=0.0` → **greedy decoding** (always picks the highest-probability token)
- `temperature=1.0` → **unmodified** model distribution
- `temperature > 1.0` → **more random**, flatter distribution
- `temperature < 1.0` → **more deterministic**, sharper distribution

```python
# Greedy (deterministic)
params = SamplingParams(temperature=0.0)

# Creative writing
params = SamplingParams(temperature=1.2, top_p=0.95)

# Balanced
params = SamplingParams(temperature=0.7)
```

!!! warning "Very low temperatures"
    Temperatures below `0.01` are clamped to `0.01` to avoid numerical instability (NaN/Inf in tensors). Use `temperature=0.0` for true greedy decoding.

### `top_p`

**Type:** `float` · **Default:** `1.0` · **Range:** `(0.0, 1.0]`

Nucleus sampling: only sample from the smallest set of tokens whose cumulative probability exceeds `top_p`. This filters out low-probability "tail" tokens.

- `top_p=1.0` → consider all tokens (disabled)
- `top_p=0.95` → sample from the top 95% of probability mass
- `top_p=0.5` → very conservative, only high-probability tokens

```python
# Standard nucleus sampling
params = SamplingParams(temperature=0.8, top_p=0.95)

# Very conservative
params = SamplingParams(temperature=0.5, top_p=0.5)
```

!!! tip "Combining temperature and top_p"
    `temperature` and `top_p` are applied together. A common recipe is `temperature=0.8, top_p=0.95` for general-purpose generation.

### `top_k`

**Type:** `int` · **Default:** `0`

Top-k sampling: only sample from the `k` highest-probability tokens. Set to `0` (or `-1`) to disable.

```python
# Only consider the top 50 tokens at each step
params = SamplingParams(temperature=0.8, top_k=50)

# Combine with top_p (both filters are applied)
params = SamplingParams(temperature=0.8, top_k=50, top_p=0.95)
```

### `min_p`

**Type:** `float` · **Default:** `0.0` · **Range:** `[0.0, 1.0]`

Minimum probability threshold relative to the most likely token. A token is only considered if its probability is at least `min_p × P(most_likely_token)`.

- `min_p=0.0` → disabled
- `min_p=0.05` → exclude tokens with probability less than 5% of the top token's probability

```python
params = SamplingParams(temperature=1.0, min_p=0.05)
```

---

## Reproducibility

### `seed`

**Type:** `int | None` · **Default:** `None`

Random seed for reproducible generation. When set, the same prompt + seed combination will always produce the same output.

```python
params = SamplingParams(temperature=0.8, seed=42)

# These two calls produce identical output
out1 = llm.generate("Hello", params)
out2 = llm.generate("Hello", params)
assert out1[0].outputs[0].text == out2[0].outputs[0].text
```

!!! note
    Setting `seed=-1` is treated as `None` (no seed).

---

## Length Control

### `max_tokens`

**Type:** `int | None` · **Default:** `16`

Maximum number of tokens to generate per output sequence. Generation stops when this limit is reached (with `finish_reason="length"`).

Set to `None` to generate until the model's maximum context length or a stop condition is reached.

```python
# Short responses
params = SamplingParams(max_tokens=50)

# Long-form generation
params = SamplingParams(max_tokens=2048)

# No explicit limit (use with caution)
params = SamplingParams(max_tokens=None)
```

### `min_tokens`

**Type:** `int` · **Default:** `0`

Minimum number of tokens to generate before EOS or stop tokens are allowed. Useful to prevent the model from immediately outputting an end-of-sequence token.

```python
# Ensure at least 20 tokens are generated
params = SamplingParams(min_tokens=20, max_tokens=200)
```

---

## Stopping Conditions

### `stop`

**Type:** `str | list[str] | None` · **Default:** `None`

One or more strings that stop generation when they appear in the output. The stop string itself is **not** included in the output (unless `include_stop_str_in_output=True`).

```python
# Stop at a newline
params = SamplingParams(stop="\n")

# Stop at any of several strings
params = SamplingParams(stop=["###", "END", "\n\n"])
```

### `stop_token_ids`

**Type:** `list[int] | None` · **Default:** `None`

Token IDs that stop generation when produced. Unlike `stop` strings, the stop token **is** included in the output unless it is a special token.

```python
# Stop at token ID 50256 (GPT-2's EOS token)
params = SamplingParams(stop_token_ids=[50256])
```

### `ignore_eos`

**Type:** `bool` · **Default:** `False`

When `True`, the model continues generating even after producing the EOS token. Useful for models that tend to stop too early, or for generating fixed-length outputs.

```python
params = SamplingParams(ignore_eos=True, max_tokens=500)
```

### `include_stop_str_in_output`

**Type:** `bool` · **Default:** `False`

When `True`, the stop string that triggered the end of generation is included in the output text.

```python
params = SamplingParams(stop="###", include_stop_str_in_output=True)
# Output will end with "###"
```

---

## Repetition Penalties

### `presence_penalty`

**Type:** `float` · **Default:** `0.0` · **Range:** `[-2.0, 2.0]`

Penalizes tokens based on whether they have **appeared at all** in the generated text so far. Applied as a flat additive penalty to the logit.

- `> 0` → discourages repeating any token that has appeared
- `< 0` → encourages repeating tokens that have appeared
- `= 0` → no effect

```python
# Encourage topic diversity
params = SamplingParams(presence_penalty=0.6)
```

### `frequency_penalty`

**Type:** `float` · **Default:** `0.0` · **Range:** `[-2.0, 2.0]`

Penalizes tokens based on their **frequency** in the generated text so far. The penalty scales with how many times the token has been used.

- `> 0` → discourages frequently repeated tokens
- `< 0` → encourages repeating frequently used tokens
- `= 0` → no effect

```python
# Reduce word repetition
params = SamplingParams(frequency_penalty=0.5)
```

### `repetition_penalty`

**Type:** `float` · **Default:** `1.0` · **Range:** `(0.0, ∞)`

Multiplicative penalty applied to tokens that appear in both the **prompt** and the **generated text**. Follows the Hugging Face Transformers convention.

- `> 1.0` → discourages repeating tokens from the prompt/output
- `< 1.0` → encourages repeating tokens from the prompt/output
- `= 1.0` → no effect

```python
# Penalize repetition of prompt tokens
params = SamplingParams(repetition_penalty=1.2)
```

!!! note "Difference from presence/frequency penalty"
    `repetition_penalty` is a **multiplicative** factor applied to logits, while `presence_penalty` and `frequency_penalty` are **additive** adjustments. `repetition_penalty` also considers the prompt tokens, not just the generated tokens.

### `bad_words`

**Type:** `list[str] | None` · **Default:** `None`

A list of words (or phrases) that must not appear in the generated output. More precisely, the last token of each bad word's token sequence is blocked when it would complete that sequence.

```python
params = SamplingParams(
    bad_words=["violence", "explicit content", "harmful"],
    max_tokens=200,
)
```

### `repetition_detection`

**Type:** `RepetitionDetectionParams | None` · **Default:** `None`

Detects and terminates generation when the model enters a repetitive loop (e.g., `"abcdabcdabcd..."` or repeated emoji sequences). This saves time and tokens when the model gets stuck.

```python
from vllm.sampling_params import RepetitionDetectionParams

params = SamplingParams(
    max_tokens=1000,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=10,  # detect patterns up to 10 tokens long
        min_pattern_size=1,   # check patterns as short as 1 token
        min_count=3,          # stop if a pattern repeats 3+ times
    ),
)
```

**`RepetitionDetectionParams` fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `max_pattern_size` | `int` | `0` | Maximum N-gram size to detect. Set to `0` to disable. |
| `min_pattern_size` | `int` | `0` | Minimum N-gram size to check (defaults to `1` if `0`). Must be ≤ `max_pattern_size`. |
| `min_count` | `int` | `0` | Minimum repetitions to trigger early stopping. Must be ≥ `2`. |

---

## Log Probabilities

### `logprobs`

**Type:** `int | None` · **Default:** `None`

Number of top-token log probabilities to return **per generated token**. When set, the result includes the log probabilities of the top `logprobs` tokens at each position, plus the sampled token itself (so up to `logprobs + 1` entries per position).

- `None` → no log probabilities returned
- `1` → return the top-1 log prob (plus the sampled token)
- `5` → return the top-5 log probs
- `-1` → return log probs for the entire vocabulary

```python
params = SamplingParams(temperature=0.8, logprobs=5)
outputs = llm.generate("The capital of France is", params)

for token_logprobs in outputs[0].outputs[0].logprobs:
    for token_id, logprob in token_logprobs.items():
        print(f"  token_id={token_id}, logprob={logprob.logprob:.4f}, "
              f"token={logprob.decoded_token!r}")
```

See the [Log Probabilities guide](logprobs.md) for detailed usage.

### `prompt_logprobs`

**Type:** `int | None` · **Default:** `None`

Number of top-token log probabilities to return **per prompt token**. Useful for analyzing how the model processes the input.

- `None` → no prompt log probabilities
- `-1` → return log probs for the entire vocabulary at each prompt position

```python
params = SamplingParams(prompt_logprobs=3, max_tokens=50)
outputs = llm.generate("Hello world", params)

# outputs[0].prompt_logprobs is a list of dicts, one per prompt token
for i, token_logprobs in enumerate(outputs[0].prompt_logprobs or []):
    if token_logprobs:
        top = max(token_logprobs.items(), key=lambda x: x[1].logprob)
        print(f"Prompt token {i}: top logprob token = {top[1].decoded_token!r}")
```

!!! note "Prefix caching interaction"
    When `prompt_logprobs` is set, prefix cache reading is automatically disabled for that request to ensure complete prompt log probability coverage.

### `flat_logprobs`

**Type:** `bool` · **Default:** `False`

When `True`, log probabilities are returned in a flattened `FlatLogprobs` format instead of `list[dict[int, Logprob]]`. This significantly reduces Python garbage collection overhead for high-throughput workloads.

```python
params = SamplingParams(logprobs=5, flat_logprobs=True)
```

---

## Logit Control

### `logit_bias`

**Type:** `dict[int, float] | None` · **Default:** `None`

Directly adjust the logit (pre-softmax score) for specific token IDs. Positive values increase the probability of a token; negative values decrease it. Values are clamped to `[-100, 100]`.

```python
# Get token IDs from the tokenizer
tokenizer = llm.get_tokenizer()
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

# Strongly bias toward "Yes"
params = SamplingParams(
    temperature=1.0,
    logit_bias={yes_id: 10.0, no_id: -10.0},
)
```

### `allowed_token_ids`

**Type:** `list[int] | None` · **Default:** `None`

Restrict generation to only the specified token IDs. All other tokens receive a logit of `-inf` (effectively zero probability).

```python
# Only allow digits 0-9
tokenizer = llm.get_tokenizer()
digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

params = SamplingParams(allowed_token_ids=digit_ids, max_tokens=10)
```

---

## Structured Outputs

### `structured_outputs`

**Type:** `StructuredOutputsParams | None` · **Default:** `None`

Constrain generation to follow a specific structure: JSON schema, regex pattern, choice list, or EBNF grammar.

```python
from vllm.sampling_params import StructuredOutputsParams
```

**`StructuredOutputsParams` fields** (exactly one constraint must be set):

| Field | Type | Description |
|---|---|---|
| `json` | `str \| dict \| None` | JSON schema (as dict or JSON string) |
| `regex` | `str \| None` | Regular expression pattern |
| `choice` | `list[str] \| None` | Restrict output to one of these strings |
| `grammar` | `str \| None` | EBNF grammar string |
| `json_object` | `bool \| None` | Require valid JSON object (no schema) |
| `structural_tag` | `str \| None` | Structural tag constraint |

**Additional options:**

| Field | Type | Default | Description |
|---|---|---|---|
| `disable_fallback` | `bool` | `False` | Disable fallback to unconstrained generation on error |
| `disable_any_whitespace` | `bool` | `False` | Disallow arbitrary whitespace in JSON |
| `disable_additional_properties` | `bool` | `False` | Reject extra JSON properties |
| `whitespace_pattern` | `str \| None` | `None` | Custom whitespace regex for JSON |

#### JSON Schema Example

```python
schema = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "tags":  {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "score"],
}

params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    structured_outputs=StructuredOutputsParams(json=schema),
)
```

#### Regex Example

```python
# Force output to be a US phone number
params = SamplingParams(
    temperature=0.0,
    structured_outputs=StructuredOutputsParams(regex=r"\(\d{3}\) \d{3}-\d{4}"),
)
```

#### Choice Example

```python
# Sentiment classification
params = SamplingParams(
    temperature=0.0,
    structured_outputs=StructuredOutputsParams(choice=["positive", "negative", "neutral"]),
)
```

#### Grammar Example

```python
# Simple arithmetic expression grammar
grammar = """
    expr   ::= term (('+' | '-') term)*
    term   ::= factor (('*' | '/') factor)*
    factor ::= NUMBER | '(' expr ')'
    NUMBER ::= [0-9]+
"""
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(grammar=grammar),
)
```

---

## Output Format

### `detokenize`

**Type:** `bool` · **Default:** `True`

When `False`, the output `text` field is empty and only `token_ids` are populated. Useful when you only need token IDs and want to avoid detokenization overhead.

```python
params = SamplingParams(detokenize=False, max_tokens=50)
outputs = llm.generate("Hello", params)
print(outputs[0].outputs[0].token_ids)  # list of ints
print(outputs[0].outputs[0].text)       # "" (empty)
```

### `skip_special_tokens`

**Type:** `bool` · **Default:** `True`

When `True`, special tokens (e.g., `<|endoftext|>`, `<s>`, `</s>`) are excluded from the decoded output text.

```python
# Include special tokens in output
params = SamplingParams(skip_special_tokens=False)
```

### `spaces_between_special_tokens`

**Type:** `bool` · **Default:** `True`

When `True`, spaces are inserted between special tokens during detokenization. Only relevant when `skip_special_tokens=False`.

---

## Streaming Control

### `output_kind`

**Type:** `RequestOutputKind` · **Default:** `RequestOutputKind.CUMULATIVE`

Controls how intermediate results are delivered during streaming (relevant for `AsyncLLM`):

| Value | Behavior |
|---|---|
| `CUMULATIVE` | Each update contains the full output so far |
| `DELTA` | Each update contains only the new tokens since the last update |
| `FINAL_ONLY` | No intermediate updates; only the final result is returned |

```python
from vllm.sampling_params import RequestOutputKind

# For streaming: receive only deltas
params = SamplingParams(output_kind=RequestOutputKind.DELTA)

# For n > 1: wait for all completions before returning
params = SamplingParams(n=4, output_kind=RequestOutputKind.FINAL_ONLY)
```

---

## Advanced

### `extra_args`

**Type:** `dict[str, Any] | None` · **Default:** `None`

Arbitrary key-value pairs passed through to custom sampling implementations, plugins, or experimental features. Not used by any built-in vLLM sampling logic.

```python
params = SamplingParams(
    extra_args={"my_plugin_option": True, "threshold": 0.5},
)
```

---

## Common Recipes

### Greedy decoding

```python
SamplingParams(temperature=0.0)
```

### Creative writing

```python
SamplingParams(temperature=1.1, top_p=0.95, frequency_penalty=0.3, max_tokens=500)
```

### Balanced generation

```python
SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
```

### Reproducible output

```python
SamplingParams(temperature=0.8, seed=42, max_tokens=100)
```

### Multiple diverse outputs

```python
SamplingParams(n=5, temperature=1.0, top_p=0.95, max_tokens=100)
```

### Constrained length

```python
SamplingParams(min_tokens=50, max_tokens=100, temperature=0.8)
```

### Stop at a delimiter

```python
SamplingParams(stop=["###", "\n\n"], max_tokens=500)
```

### JSON output

```python
from vllm.sampling_params import StructuredOutputsParams

SamplingParams(
    temperature=0.0,
    max_tokens=500,
    structured_outputs=StructuredOutputsParams(json={"type": "object"}),
)
```

### With log probabilities

```python
SamplingParams(temperature=0.8, logprobs=5, max_tokens=100)
```

---

## Validation Rules

vLLM validates `SamplingParams` on construction and raises `ValueError` or `VLLMValidationError` for invalid combinations:

| Constraint | Rule |
|---|---|
| `n` | Must be ≥ 1 |
| `presence_penalty` | Must be in `[-2.0, 2.0]` |
| `frequency_penalty` | Must be in `[-2.0, 2.0]` |
| `repetition_penalty` | Must be > 0 |
| `temperature` | Must be ≥ 0 |
| `top_p` | Must be in `(0.0, 1.0]` |
| `top_k` | Must be ≥ -1 |
| `min_p` | Must be in `[0.0, 1.0]` |
| `max_tokens` | Must be ≥ 1 (if not `None`) |
| `min_tokens` | Must be ≥ 0 |
| `logprobs` | Must be ≥ 0 or `-1` (if not `None`) |
| `prompt_logprobs` | Must be ≥ 0 or `-1` (if not `None`) |
| Greedy + `n > 1` | Not allowed (greedy is deterministic) |

---

## See Also

- [Log Probabilities Guide](logprobs.md) — detailed logprobs usage
- [Beam Search Guide](beam_search.md) — `BeamSearchParams` reference
- [Structured Outputs](../features/structured_outputs.md) — full structured output guide
- [Output Types](output_types.md) — understanding `RequestOutput` and `CompletionOutput`
