# Log Probabilities Guide

Log probabilities (logprobs) give you insight into the model's confidence at each token position — both for the tokens it generates and for the tokens in your input prompt. This is useful for:

- **Scoring and ranking** generated sequences by likelihood
- **Uncertainty estimation** — identifying where the model is unsure
- **Debugging** — understanding why the model chose a particular token
- **Perplexity computation** — measuring how well the model fits a text
- **Calibration** — comparing model confidence to actual accuracy

---

## Enabling Log Probabilities

Log probabilities are controlled by two fields in [`SamplingParams`][vllm.SamplingParams]:

| Parameter | Controls | Default |
|---|---|---|
| `logprobs` | Log probs for **generated** tokens | `None` (disabled) |
| `prompt_logprobs` | Log probs for **prompt** tokens | `None` (disabled) |

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

params = SamplingParams(
    temperature=0.0,
    max_tokens=10,
    logprobs=5,          # top-5 logprobs per generated token
    prompt_logprobs=3,   # top-3 logprobs per prompt token
)

outputs = llm.generate("The capital of France is", params)
```

---

## The `Logprob` Object

Each log probability entry is a [`Logprob`][vllm.logprobs.Logprob] object:

```python
from vllm.logprobs import Logprob
```

| Attribute | Type | Description |
|---|---|---|
| `logprob` | `float` | Natural log probability: `log(P(token))` |
| `rank` | `int \| None` | Vocabulary rank (1 = most likely token at this position) |
| `decoded_token` | `str \| None` | The decoded string for this token |

**Converting to probability:**

```python
import math
prob = math.exp(logprob.logprob)  # P(token) = e^logprob
```

---

## Generated Token Log Probabilities

### Setting `logprobs`

Set `logprobs=N` to receive the top-N log probabilities at each generated token position. The sampled token is always included, so you may receive up to `N+1` entries per position.

- `logprobs=None` → no log probabilities (default)
- `logprobs=1` → top-1 logprob + sampled token
- `logprobs=5` → top-5 logprobs + sampled token
- `logprobs=-1` → **all** vocabulary tokens (use with caution — very large output)

### Accessing Generated Logprobs

`output.outputs[0].logprobs` is a `list[dict[int, Logprob]]` — one dictionary per generated token. Each dictionary maps `token_id → Logprob`.

```python
params = SamplingParams(temperature=0.0, logprobs=5, max_tokens=10)
outputs = llm.generate("The capital of France is", params)

completion = outputs[0].outputs[0]
print(f"Generated: {completion.text!r}")
print()

for step, token_logprobs in enumerate(completion.logprobs or []):
    # Sort by logprob (highest first)
    ranked = sorted(token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
    print(f"Step {step}:")
    for token_id, lp in ranked:
        marker = " ← sampled" if lp.rank == 1 or lp.logprob == max(
            v.logprob for v in token_logprobs.values()
        ) else ""
        print(f"  rank={lp.rank:3d}  logprob={lp.logprob:8.4f}  "
              f"prob={math.exp(lp.logprob):.4f}  "
              f"token={lp.decoded_token!r}{marker}")
    print()
```

### Identifying the Sampled Token

The sampled token is the one with `rank=1` (when using greedy decoding) or the token that was actually chosen (when sampling). You can identify it by checking which token ID matches the generated `token_ids`:

```python
for step, (token_id, token_logprobs) in enumerate(
    zip(completion.token_ids, completion.logprobs or [])
):
    sampled_lp = token_logprobs.get(token_id)
    if sampled_lp:
        print(f"Step {step}: sampled {sampled_lp.decoded_token!r} "
              f"(logprob={sampled_lp.logprob:.4f}, rank={sampled_lp.rank})")
```

### Cumulative Log Probability

`completion.cumulative_logprob` is the sum of log probabilities of all sampled tokens — a measure of the overall sequence likelihood:

```python
print(f"Cumulative logprob: {completion.cumulative_logprob:.4f}")

# Equivalent to:
manual_sum = sum(
    token_logprobs[token_id].logprob
    for token_id, token_logprobs in zip(completion.token_ids, completion.logprobs or [])
    if token_id in token_logprobs
)
```

---

## Prompt Token Log Probabilities

### Setting `prompt_logprobs`

Set `prompt_logprobs=N` to receive the top-N log probabilities at each prompt token position.

- `prompt_logprobs=None` → no prompt log probabilities (default)
- `prompt_logprobs=1` → top-1 logprob per prompt token
- `prompt_logprobs=-1` → all vocabulary tokens per prompt token

### Accessing Prompt Logprobs

`output.prompt_logprobs` is a `list[dict[int, Logprob] | None]` — one entry per prompt token. The **first entry is always `None`** because there is no preceding context to condition on for the first token.

```python
params = SamplingParams(prompt_logprobs=3, max_tokens=5)
outputs = llm.generate("Hello world", params)

print("Prompt logprobs:")
for i, pos_lp in enumerate(outputs[0].prompt_logprobs or []):
    if pos_lp is None:
        print(f"  Token {i}: (first token — no logprob)")
    else:
        top3 = sorted(pos_lp.items(), key=lambda x: x[1].logprob, reverse=True)[:3]
        print(f"  Token {i}:")
        for token_id, lp in top3:
            print(f"    rank={lp.rank}  logprob={lp.logprob:.4f}  "
                  f"token={lp.decoded_token!r}")
```

!!! note "Prefix caching interaction"
    When `prompt_logprobs` is set, prefix cache reading is automatically disabled for that request to ensure complete coverage of all prompt token positions.

---

## Computing Perplexity

Perplexity measures how well the model predicts a text. Lower perplexity = better fit.

```python
import math
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

def compute_perplexity(text: str) -> float:
    """Compute perplexity of a text under the model."""
    params = SamplingParams(
        prompt_logprobs=0,  # just need the sampled token logprob
        max_tokens=1,       # generate one token to get prompt logprobs
        temperature=0.0,
    )
    output = llm.generate(text, params)[0]

    # Collect log probs for all prompt tokens (skip first None)
    log_probs = []
    for i, pos_lp in enumerate(output.prompt_logprobs or []):
        if pos_lp is None:
            continue  # first token
        # The actual token's logprob is the one with rank=1 (or the token itself)
        if pos_lp:
            # Get the logprob of the actual token at this position
            # (rank=1 is the most likely, which is what the model "expected")
            best = max(pos_lp.values(), key=lambda x: x.logprob)
            log_probs.append(best.logprob)

    if not log_probs:
        return float("inf")

    avg_neg_logprob = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_logprob)


texts = [
    "The capital of France is Paris.",
    "The capital of France is banana.",
    "xkcd 1234 zorp florp bleep bloop.",
]

for text in texts:
    ppl = compute_perplexity(text)
    print(f"Perplexity: {ppl:8.2f} | {text!r}")
```

---

## Scoring Sequences

Use cumulative log probabilities to rank multiple candidate completions:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

prompt = "The capital of France is"
candidates = [" Paris.", " London.", " Berlin.", " Madrid."]

# Score each candidate by computing its log probability given the prompt
all_prompts = [prompt + c for c in candidates]
params = SamplingParams(
    temperature=0.0,
    prompt_logprobs=0,
    max_tokens=1,
)

outputs = llm.generate(all_prompts, params)

scores = []
for candidate, output in zip(candidates, outputs):
    # Sum log probs of the candidate tokens (skip the shared prompt prefix)
    prompt_len = len(llm.get_tokenizer().encode(prompt))
    candidate_logprobs = (output.prompt_logprobs or [])[prompt_len:]

    total_logprob = sum(
        max(pos_lp.values(), key=lambda x: x.logprob).logprob
        for pos_lp in candidate_logprobs
        if pos_lp is not None
    )
    scores.append((candidate, total_logprob))

# Rank by score
scores.sort(key=lambda x: x[1], reverse=True)
print("Ranked candidates:")
for candidate, score in scores:
    print(f"  {score:8.4f} | {prompt!r}{candidate!r}")
```

---

## High-Performance Logprobs with `flat_logprobs`

For high-throughput workloads where you're collecting logprobs for many requests, the default `list[dict[int, Logprob]]` format can create significant Python garbage collection pressure. Use `flat_logprobs=True` to switch to the `FlatLogprobs` format:

```python
params = SamplingParams(
    logprobs=5,
    flat_logprobs=True,  # use memory-efficient flat format
    max_tokens=100,
)

outputs = llm.generate(prompts, params)

# FlatLogprobs supports the same list interface
completion = outputs[0].outputs[0]
for pos_lp in completion.logprobs or []:
    if pos_lp:
        for token_id, lp in pos_lp.items():
            print(token_id, lp.logprob)
```

The `FlatLogprobs` object stores all data in flat primitive lists (`list[int]`, `list[float]`, etc.) instead of nested Python objects, reducing GC overhead by orders of magnitude for large batches.

---

## Logprobs in the OpenAI API

When using vLLM's OpenAI-compatible server, logprobs are requested via the standard OpenAI API parameters:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "logprobs": 5,
        "echo": true
    }'
```

The response includes a `logprobs` field with `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset` arrays.

For chat completions:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "logprobs": true,
        "top_logprobs": 5
    }'
```

---

## Common Patterns

### Get the top-1 token at each position

```python
for step, token_logprobs in enumerate(completion.logprobs or []):
    top1 = max(token_logprobs.items(), key=lambda x: x[1].logprob)
    token_id, lp = top1
    print(f"Step {step}: {lp.decoded_token!r} (logprob={lp.logprob:.4f})")
```

### Find positions where the model was uncertain

```python
import math

UNCERTAINTY_THRESHOLD = 0.5  # probability of top token < 50%

for step, token_logprobs in enumerate(completion.logprobs or []):
    top1 = max(token_logprobs.values(), key=lambda x: x.logprob)
    top_prob = math.exp(top1.logprob)
    if top_prob < UNCERTAINTY_THRESHOLD:
        print(f"Step {step}: uncertain (top prob={top_prob:.3f}, "
              f"token={top1.decoded_token!r})")
```

### Compute sequence probability

```python
import math

def sequence_probability(completion) -> float:
    """Compute P(sequence) = product of P(token_i) for all generated tokens."""
    if not completion.logprobs:
        return float("nan")
    total_logprob = sum(
        max(pos_lp.values(), key=lambda x: x.logprob).logprob
        for pos_lp in completion.logprobs
        if pos_lp
    )
    return math.exp(total_logprob)

prob = sequence_probability(outputs[0].outputs[0])
print(f"Sequence probability: {prob:.6f}")
```

### Compare two models on the same text

```python
llm1 = LLM(model="facebook/opt-125m")
llm2 = LLM(model="facebook/opt-350m")

params = SamplingParams(temperature=0.0, logprobs=1, max_tokens=20)
text = "The capital of France is"

out1 = llm1.generate(text, params)[0]
out2 = llm2.generate(text, params)[0]

print(f"opt-125m: {out1.outputs[0].text!r} "
      f"(cumlogp={out1.outputs[0].cumulative_logprob:.4f})")
print(f"opt-350m: {out2.outputs[0].text!r} "
      f"(cumlogp={out2.outputs[0].cumulative_logprob:.4f})")
```

---

## Reference Summary

| Field | Location | Type | Description |
|---|---|---|---|
| `logprobs` | `SamplingParams` | `int \| None` | Number of top logprobs per generated token |
| `prompt_logprobs` | `SamplingParams` | `int \| None` | Number of top logprobs per prompt token |
| `flat_logprobs` | `SamplingParams` | `bool` | Use memory-efficient flat format |
| `completion.logprobs` | `CompletionOutput` | `SampleLogprobs \| None` | Per-token logprobs for generated tokens |
| `completion.cumulative_logprob` | `CompletionOutput` | `float \| None` | Sum of sampled token logprobs |
| `output.prompt_logprobs` | `RequestOutput` | `PromptLogprobs \| None` | Per-token logprobs for prompt tokens |
| `Logprob.logprob` | `Logprob` | `float` | `log(P(token))` |
| `Logprob.rank` | `Logprob` | `int \| None` | Vocabulary rank (1 = most likely) |
| `Logprob.decoded_token` | `Logprob` | `str \| None` | Decoded token string |

---

## See Also

- [SamplingParams Reference](sampling_params.md) — `logprobs` and `prompt_logprobs` parameters
- [Output Types](output_types.md) — `CompletionOutput`, `RequestOutput`, `Logprob`
- [Offline Inference Guide](offline_inference.md) — running inference with the `LLM` class
