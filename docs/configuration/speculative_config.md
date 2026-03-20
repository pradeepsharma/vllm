# Speculative Decoding Configuration

`SpeculativeConfig` controls speculative decoding — a technique that uses a fast "draft" model to propose multiple tokens, which are then verified in parallel by the target model. This can significantly improve throughput and reduce latency for generation tasks.

**Source:** `vllm/config/speculative.py`  
**CLI flag:** `--speculative-config` (JSON dict)

---

## Overview

Speculative decoding works by:

1. **Draft phase:** A small, fast model proposes `k` candidate tokens
2. **Verify phase:** The target model verifies all `k` tokens in a single forward pass
3. **Accept/reject:** Accepted tokens are kept; the first rejected token triggers a correction

When the draft model's predictions are accurate, this achieves near-`k`× speedup with identical output quality (mathematically equivalent to standard sampling).

```
Draft model:  [T1] [T2] [T3] [T4] [T5]  ← 5 speculative tokens
Target model: [✓]  [✓]  [✓]  [✗]        ← verify all at once
Result:       [T1] [T2] [T3] [T4']       ← 3 accepted + 1 corrected
```

---

## Configuration

Speculative decoding is configured via `--speculative-config` as a JSON dictionary:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_speculative_tokens": 5
  }'
```

---

## Core Parameters

### `method`

```
Type:    str | None
Default: None (auto-detect from model)
```

The speculative decoding method to use:

| Method | Description |
|---|---|
| `draft_model` | Use a separate smaller model as the draft |
| `ngram` | N-gram based speculation from the prompt |
| `ngram_gpu` | GPU-accelerated n-gram speculation |
| `medusa` | Medusa multi-head speculation |
| `mlp_speculator` | MLP-based speculator |
| `eagle` | EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) |
| `eagle3` | EAGLE3 with auxiliary hidden states |
| `suffix` | Suffix tree based speculation |
| `deepseek_mtp` | DeepSeek Multi-Token Prediction |
| `mimo_mtp` | MIMO Multi-Token Prediction |
| `extract_hidden_states` | Extract hidden states for speculation |

If `model` is provided, the method is auto-detected when possible.

### `model`

```
Type:    str | None
Default: None
```

The draft model name or path. Required for `draft_model`, `eagle`, `eagle3`, and MTP methods.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "num_speculative_tokens": 5
}
```

### `num_speculative_tokens`

```
Type:    int (> 0)
Default: None (read from draft model config if available)
```

Number of tokens to speculate per step. Higher values can improve throughput when the draft model is accurate, but increase the cost of rejected tokens.

**Typical values:** 3–8 tokens. Start with 5 and tune based on acceptance rate.

---

## Draft Model Configuration

### `draft_tensor_parallel_size`

```
Type:    int | None (≥ 1)
Default: None (same as target model)
```

Tensor parallel size for the draft model. Can only be `1` or the same as the target model's TP size.

### `quantization`

```
Type:    str | None
Default: None
```

Quantization method for the draft model weights. Independent of the target model's quantization.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "num_speculative_tokens": 5,
  "quantization": "fp8"
}
```

### `max_model_len`

```
Type:    int | None (≥ 1)
Default: None
```

Maximum context length for the draft model. Used when testing the ability to skip speculation for some sequences.

### `revision`

```
Type:    str | None
Default: None
```

Specific version of the draft model (branch, tag, or commit ID).

### `code_revision`

```
Type:    str | None
Default: None
```

Specific code revision for the draft model on HF Hub.

### `draft_load_config`

```
Type:    LoadConfig | None
Default: None (uses target model's load config)
```

Load configuration for the draft model. If not specified, uses the target model's load config.

---

## N-gram Speculation

N-gram speculation uses the prompt itself to predict future tokens — no separate model required.

```bash
vllm serve mymodel \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 1
  }'
```

### `prompt_lookup_max`

```
Type:    int | None (≥ 1)
Default: None (required for ngram method)
```

Maximum n-gram window size for token lookup. Required when `method="ngram"`.

### `prompt_lookup_min`

```
Type:    int | None (≥ 1)
Default: None (defaults to 1)
```

Minimum n-gram window size. Defaults to 1 if not specified.

---

## EAGLE Speculation

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) uses a lightweight draft head trained on the target model's hidden states.

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "lmzheng/sglang-EAGLE-LLaMA3.1-Instruct-70B",
    "method": "eagle",
    "num_speculative_tokens": 5
  }'
```

EAGLE3 additionally uses auxiliary hidden states from intermediate layers:

```bash
vllm serve mymodel \
  --speculative-config '{
    "model": "my-eagle3-head",
    "method": "eagle3",
    "num_speculative_tokens": 6
  }'
```

---

## Medusa Speculation

Medusa adds multiple prediction heads to the target model, each predicting tokens at different positions ahead.

```bash
vllm serve mymodel \
  --speculative-config '{
    "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
    "method": "medusa",
    "num_speculative_tokens": 5
  }'
```

---

## Suffix Decoding

Suffix decoding uses a global suffix tree built from past responses to predict future tokens.

```bash
vllm serve mymodel \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 5,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000
  }'
```

### `suffix_decoding_max_tree_depth`

```
Type:    int
Default: 24
```

Maximum depth of the suffix tree. Limits the sum of prefix match length and speculation length.

### `suffix_decoding_max_cached_requests`

```
Type:    int
Default: 10000
```

Maximum number of requests to cache in the global suffix tree. When exceeded, eviction happens in FIFO order. Set to `0` to disable the global tree (only prompt trees are used).

### `suffix_decoding_max_spec_factor`

```
Type:    float
Default: 1.0
```

Controls speculation length based on prefix match: `max_spec_tokens = max_spec_factor × prefix_match_length`.

### `suffix_decoding_min_token_prob`

```
Type:    float
Default: 0.1
```

Minimum estimated token probability (based on frequency counts) for a token to be speculated.

---

## DeepSeek Multi-Token Prediction (MTP)

DeepSeek V3 and related models support native multi-token prediction:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --speculative-config '{
    "method": "deepseek_mtp",
    "num_speculative_tokens": 1
  }'
```

Supported MTP model types:
- `deepseek_mtp`
- `mimo_mtp`
- `glm4_moe_mtp` / `glm4_moe_lite_mtp`
- `glm_ocr_mtp`
- `ernie_mtp`
- `nemotron_h_mtp`
- `exaone_moe_mtp`
- `qwen3_next_mtp` / `qwen3_5_mtp`
- `longcat_flash_mtp`
- `pangu_ultra_moe_mtp`
- `step3p5_mtp`

---

## Advanced Options

### `parallel_drafting`

```
Type:    bool
Default: False
```

Generate all speculative tokens in parallel rather than sequentially. Requires the draft model to be trained for parallel drafting. Compatible with EAGLE and draft model methods.

### `disable_padded_drafter_batch`

```
Type:    bool
Default: False
```

Disable input padding for speculative decoding. When `True`, speculative batches can contain sequences of different lengths. Currently only affects the EAGLE method.

### `use_local_argmax_reduction`

```
Type:    bool
Default: False
```

Use vocab-parallel local argmax instead of all-gathering full logits for draft token generation. Reduces communication from O(vocab_size) to O(2 × tp_size) per token. Only applies to greedy draft selection in non-tree speculation.

### `speculative_token_tree`

```
Type:    str | None
Default: None
```

Specifies the tree structure for speculative token generation (tree-based speculation).

### `enforce_eager`

```
Type:    bool | None
Default: None (uses target model's setting)
```

Override the `enforce_eager` setting for the draft model.

---

## Performance Tuning

### Acceptance rate

The key metric for speculative decoding is the **acceptance rate** — the fraction of speculated tokens that are accepted. Monitor this in vLLM's logs or metrics.

- **High acceptance rate (>0.8):** Increase `num_speculative_tokens` for more speedup
- **Low acceptance rate (<0.5):** Decrease `num_speculative_tokens` or try a different method

### Method selection guide

| Scenario | Recommended Method |
|---|---|
| Have a smaller version of the target model | `draft_model` |
| Repetitive/structured outputs | `ngram` or `suffix` |
| Model has EAGLE head available | `eagle` or `eagle3` |
| Model has Medusa heads | `medusa` |
| DeepSeek V3/V3.5 | `deepseek_mtp` |
| No draft model available | `ngram` |

---

## Complete Examples

### Draft model speculation

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1
  }'
```

### N-gram speculation (no draft model)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 5
  }'
```

### EAGLE with quantized draft

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "my-eagle-head",
    "method": "eagle",
    "num_speculative_tokens": 6,
    "quantization": "fp8"
  }'
```
