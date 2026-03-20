---
description: >
  How to use EAGLE and EAGLE3 speculative decoding in vLLM — configuration,
  examples, tree attention, parallel drafting, and available pre-trained heads.
---

# EAGLE draft models

[EAGLE](https://arxiv.org/pdf/2401.15077) (Extrapolation Algorithm for Greater
Language-model Efficiency) is a speculative decoding method that trains a
lightweight auto-regressive draft model conditioned on the target model's
hidden states. Because the draft model sees the target model's internal
representations, it achieves high acceptance rates with only one or two
transformer layers.

[EAGLE3](https://arxiv.org/abs/2503.01840) extends EAGLE by feeding auxiliary
hidden states from multiple intermediate layers of the target model into the
draft head, further improving acceptance rates for large models.

---

## How EAGLE works

At each decoding step, vLLM:

1. Runs the target model's forward pass and captures the final hidden states.
2. Passes those hidden states — together with the shifted input token IDs — to
   the EAGLE draft model.
3. The draft model autoregressively generates `num_speculative_tokens` draft
   tokens.
4. The target model verifies all draft tokens in a single batched forward pass.
5. Accepted tokens are emitted; rejected tokens are discarded and regenerated.

The key insight is that the draft model is conditioned on the target model's
hidden states, so it can predict the target model's next token much more
accurately than a standalone small model.

### EAGLE3 auxiliary hidden states

EAGLE3 additionally extracts hidden states from selected intermediate layers
of the target model (configured via `eagle_aux_hidden_state_layer_ids` in the
draft model's `config.json`). These auxiliary states are concatenated with the
final hidden state before being passed to the draft head, giving the draft
model richer context.

!!! note
    EAGLE3 auxiliary hidden states are currently supported for Llama, Qwen,
    MiniCPM, GPT-OSS, HunyuanVL, Nemotron-H, and a few other model families.
    Check `vllm/config/speculative.py` for the current list.

---

## Quick start

### Offline inference — EAGLE

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Offline inference — EAGLE3

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 3,
        "method": "eagle3",
    },
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online serving — EAGLE

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 4 \
    --speculative_config '{
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle"
    }'
```

---

## Tree attention

When `num_speculative_tokens > 1`, EAGLE can use **tree attention** to explore
multiple draft paths simultaneously. Instead of a single chain of draft tokens,
the draft model generates a tree of candidates. The target model verifies all
tree nodes in one forward pass and selects the longest accepted path.

Tree attention is configured via the `speculative_token_tree` parameter, which
specifies the tree structure as a list of tuples. Each tuple represents a node
in the tree, where the length of the tuple is the depth of the node.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 5,
        "method": "eagle",
        # Custom tree: 3 candidates at depth 1, 2 at depth 2, 1 at depth 3
        "speculative_token_tree": "[(0,), (1,), (2,), (0, 0), (0, 1), (0, 0, 0)]",
    },
)
```

!!! tip
    If you do not specify `speculative_token_tree`, vLLM generates a simple
    linear chain of `num_speculative_tokens` draft tokens, which is equivalent
    to standard sequential drafting.

---

## Parallel drafting with EAGLE

Setting `parallel_drafting: true` generates all speculative tokens in a single
parallel forward pass of the EAGLE draft model rather than autoregressively.
This reduces draft latency at the cost of slightly lower acceptance rates.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "your-parallel-eagle-model",
        "num_speculative_tokens": 5,
        "method": "eagle",
        "parallel_drafting": True,
    },
)
```

!!! important
    Parallel drafting requires a draft model that has been trained to support
    it. The draft model's `config.json` must contain a `pard_token` or
    `ptd_token_id` field.

---

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Hugging Face model ID or local path of the EAGLE head. |
| `method` | `str` | auto-detected | `"eagle"` or `"eagle3"`. |
| `num_speculative_tokens` | `int` | required | Number of draft tokens to generate per step. |
| `draft_tensor_parallel_size` | `int` | same as target | Tensor parallel size for the draft model. Must be `1` or equal to the target TP size. |
| `speculative_token_tree` | `str` | linear chain | Tree structure for multi-path drafting, as a Python list-of-tuples literal. |
| `parallel_drafting` | `bool` | `false` | Generate all draft tokens in one parallel pass. |
| `use_local_argmax_reduction` | `bool` | `false` | Use vocab-parallel local argmax instead of all-gathering full logits. Reduces communication from O(vocab_size) to O(2 × TP size). |
| `disable_padded_drafter_batch` | `bool` | `false` | Disable input padding for the draft batch. Only supported by certain attention backends. |
| `max_model_len` | `int` | same as target | Maximum sequence length for the draft model. |
| `quantization` | `str` | `null` | Quantization method for the draft model weights. |

---

## Measuring acceptance rate

You can measure the per-request acceptance rate in offline mode using the
`spec_decode_worker_metrics` field of the output:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 3,
        "method": "eagle",
    },
)

outputs = llm.generate(["Explain quantum computing in simple terms."])
for output in outputs:
    print(output.outputs[0].text)
```

In online serving, vLLM exposes Prometheus metrics for speculative decoding:

| Metric | Description |
|---|---|
| `vllm:spec_decode_num_drafts` | Total number of draft attempts. |
| `vllm:spec_decode_num_draft_tokens` | Total draft tokens generated. |
| `vllm:spec_decode_num_accepted_tokens` | Total draft tokens accepted. |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Accepted tokens broken down by draft position. |

The acceptance rate is:

```
rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
rate(vllm:spec_decode_num_draft_tokens_total[$interval])
```

The mean acceptance length (including the bonus token) is:

```
1 + (
  rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
  rate(vllm:spec_decode_num_drafts[$interval])
)
```

---

## Pre-trained EAGLE draft models

A variety of EAGLE draft models are available on Hugging Face:

- [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)
  — EAGLE3 heads for Llama 3.x and other models.
- [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)
  — Original EAGLE heads for Llama 2, Llama 3, Vicuna, and others.
- [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)
  — EAGLE3 head for Qwen3-8B.

!!! warning
    If you are using `vllm<0.7.0`, use
    [this conversion script](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)
    to convert the speculative model and specify
    `"model": "path/to/modified/eagle/model"` in `speculative_config`.

---

## Training your own EAGLE head

Use the [Speculators](speculators.md) library to train EAGLE heads for your
own target models. Speculators provides:

- Offline hidden-state extraction using vLLM.
- End-to-end EAGLE and EAGLE3 training.
- A Hugging Face-compatible format for direct deployment into vLLM.

See the [Speculators documentation](https://docs.vllm.ai/projects/speculators/en/latest/)
for a full training guide.

---

## See also

- [Speculative decoding overview](README.md)
- [Speculators library](speculators.md)
- [Internal design document](../../design/speculative_decoding_design.md)
- [EAGLE paper](https://arxiv.org/pdf/2401.15077)
- [EAGLE3 paper](https://arxiv.org/abs/2503.01840)
