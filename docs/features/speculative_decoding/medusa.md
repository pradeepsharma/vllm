---
description: >
  How to use Medusa speculative decoding in vLLM — configuration, examples,
  and available pre-trained Medusa heads.
---

# Medusa

[Medusa](https://arxiv.org/abs/2401.10774) is a speculative decoding method
that adds multiple independent MLP prediction heads to the target model. Each
head predicts a different future token position in parallel during the target
model's forward pass. Because the heads run alongside the target model rather
than in a separate draft step, draft generation adds minimal latency overhead.

---

## How Medusa works

During the target model's forward pass, the final hidden states are passed to
`N` Medusa heads simultaneously. Each head `i` predicts the token at position
`current + i + 1`. The predictions from all heads form a draft sequence that
the target model then verifies using the standard rejection-sampling scheme.

```
Target model forward pass
        │
        ▼
  Final hidden states
        │
   ┌────┴────┬────────┬────────┐
   ▼         ▼        ▼        ▼
 Head 1    Head 2   Head 3   Head N
 (pos+1)  (pos+2)  (pos+3)  (pos+N)
   │         │        │        │
   └────┬────┴────────┴────────┘
        ▼
  Draft tokens [d1, d2, d3, …, dN]
        │
        ▼
  Rejection sampling
```

Because all heads run in parallel within the same forward pass, Medusa has
essentially zero additional latency for the draft phase compared to standard
decoding.

---

## Quick start

### Offline inference

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="lmsys/vicuna-7b-v1.3",
    speculative_config={
        "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
        "num_speculative_tokens": 3,
        "method": "medusa",
    },
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online serving

```bash
vllm serve lmsys/vicuna-7b-v1.3 \
    --speculative_config '{
        "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
        "num_speculative_tokens": 3,
        "method": "medusa"
    }'
```

---

## Method auto-detection

vLLM automatically detects the Medusa method when the draft model's
`config.json` contains `"model_type": "medusa"`. You can omit the `"method"`
key in that case:

```python
llm = LLM(
    model="lmsys/vicuna-7b-v1.3",
    speculative_config={
        "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
        "num_speculative_tokens": 3,
        # "method" is auto-detected from the draft model's config.json
    },
)
```

---

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Hugging Face model ID or local path of the Medusa heads. |
| `method` | `str` | auto-detected | Set to `"medusa"` or omit to auto-detect. |
| `num_speculative_tokens` | `int` | required | Number of Medusa heads to use. Must not exceed the number of heads in the model. |
| `draft_tensor_parallel_size` | `int` | `1` | Tensor parallel size for the Medusa heads. MLP speculator and Medusa heads currently run at TP=1. |
| `quantization` | `str` | `null` | Quantization method for the Medusa head weights. |

!!! note
    `num_speculative_tokens` must be less than or equal to the number of
    Medusa heads in the draft model. Each head predicts one future position,
    so using more heads than available will raise an error.

---

## How Medusa differs from EAGLE

| Aspect | Medusa | EAGLE |
|---|---|---|
| Draft architecture | Multiple parallel MLP heads | Single auto-regressive transformer layer |
| Draft latency | Near-zero (runs inside target forward pass) | One extra forward pass per draft step |
| Acceptance rate | Moderate | High |
| Training complexity | Moderate | Moderate |
| Hidden state conditioning | Final hidden state only | Final hidden state + token IDs |
| Sequential drafting | No — all heads run in parallel | Yes — tokens generated one at a time |

---

## Measuring acceptance rate

vLLM exposes Prometheus metrics for speculative decoding. The same metrics
apply to Medusa as to other methods:

| Metric | Description |
|---|---|
| `vllm:spec_decode_num_drafts` | Total number of draft attempts. |
| `vllm:spec_decode_num_draft_tokens` | Total draft tokens generated. |
| `vllm:spec_decode_num_accepted_tokens` | Total draft tokens accepted. |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Accepted tokens by draft position. |

The per-position acceptance rate is particularly useful for Medusa because it
shows how quickly acceptance degrades across the heads. If head 3 has a very
low acceptance rate, reducing `num_speculative_tokens` to 2 may improve
overall throughput.

---

## Pre-trained Medusa models

| Target model | Medusa heads |
|---|---|
| `lmsys/vicuna-7b-v1.3` | [FasterDecoding/medusa-vicuna-7b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3) |
| `lmsys/vicuna-13b-v1.3` | [FasterDecoding/medusa-vicuna-13b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-13b-v1.3) |
| `lmsys/vicuna-33b-v1.3` | [FasterDecoding/medusa-vicuna-33b-v1.3](https://huggingface.co/FasterDecoding/medusa-vicuna-33b-v1.3) |

Search [Hugging Face](https://huggingface.co/models?search=medusa) for
additional community-trained Medusa heads.

---

## Training your own Medusa heads

The original Medusa training code is available at
[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa). The
[Speculators](speculators.md) library also supports training Medusa-style
heads in a Hugging Face-compatible format for direct deployment into vLLM.

---

## See also

- [Speculative decoding overview](README.md)
- [EAGLE guide](eagle.md)
- [Speculators library](speculators.md)
- [Internal design document](../../design/speculative_decoding_design.md)
- [Medusa paper](https://arxiv.org/abs/2401.10774)
