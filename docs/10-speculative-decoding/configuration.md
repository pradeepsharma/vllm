# Speculative Decoding Configuration

This page documents the `SpeculativeConfig` class and all configuration parameters for speculative decoding in vLLM.

## SpeculativeConfig

`SpeculativeConfig` is defined in `vllm/config/speculative.py` and is passed to `LLM` or the OpenAI-compatible server via the `speculative_config` parameter.

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)
```

## Core Parameters

### `method`

**Type**: `str | None`  
**Default**: Auto-detected

The speculative decoding method to use. If not specified, vLLM attempts to auto-detect the method from the draft model's architecture.

| Value | Description |
|-------|-------------|
| `"eagle"` | EAGLE / EAGLE-2 speculative decoding |
| `"eagle3"` | EAGLE-3 with auxiliary hidden states |
| `"medusa"` | Medusa multiple decoding heads |
| `"ngram"` | CPU n-gram prompt lookup |
| `"ngram_gpu"` | GPU-accelerated n-gram prompt lookup |
| `"suffix"` | Suffix tree-based speculation |
| `"mtp"` | Multi-token prediction (DeepSeek, Ernie, etc.) |
| `"draft_model"` | Separate smaller LLM as draft model |
| `"mlp_speculator"` | MLP-based speculator |

**Auto-detection rules**:
- Model name contains `"eagle-"` → `"eagle"`
- Model name contains `"eagle3"` → `"eagle3"`
- Model type is `"medusa"` → `"medusa"`
- Model type is `"mlp_speculator"` → `"mlp_speculator"`
- Model type is in MTP types → `"mtp"`
- Otherwise → `"draft_model"`

### `model`

**Type**: `str | None`  
**Default**: `None`

The name or path of the draft model, EAGLE head, or additional weights. For model-free methods (`ngram`, `suffix`), this is not required.

```python
# EAGLE draft model
speculative_config={"model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 5}

# MTP: uses the same model as target
speculative_config={"method": "mtp", "num_speculative_tokens": 1}
# (model is automatically set to target model path)
```

### `num_speculative_tokens`

**Type**: `int`  
**Default**: From draft model config (`n_predict`) if available, otherwise required

The number of draft tokens to generate per target model forward pass. This is the `k` in the draft-verify paradigm.

- For MTP models, defaults to `num_nextn_predict_layers` from the model config
- For suffix decoding, defaults to `suffix_decoding_max_tree_depth` if not specified
- Must be divisible by `num_nextn_predict_layers` if using MTP with `num_speculative_tokens > 1`

```python
speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,  # Generate 5 draft tokens per step
}
```

## Draft Model Parameters

### `quantization`

**Type**: `str | None`  
**Default**: `None` (inherits from target model for MTP)

Quantization method for the draft model weights. Supported values are the same as the main `quantization` parameter (e.g., `"fp8"`, `"awq"`, `"gptq"`).

```python
speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
    "quantization": "fp8",
}
```

### `max_model_len`

**Type**: `int | None`  
**Default**: `min(draft_max_model_len, target_max_model_len)`

Override the maximum sequence length for the draft model. Useful for testing speculation skip behavior at long contexts.

### `revision`

**Type**: `str | None`  
**Default**: `None`

Specific model version (branch, tag, or commit ID) for the draft model on Hugging Face Hub.

### `code_revision`

**Type**: `str | None`  
**Default**: `None`

Specific revision for the draft model code on Hugging Face Hub.

### `draft_load_config`

**Type**: `LoadConfig | None`  
**Default**: Inherits from target model

Load configuration for the draft model (e.g., custom weight loading format).

## Parallelism Parameters

### `draft_tensor_parallel_size`

**Type**: `int | None`  
**Default**: Same as target model's TP size

Tensor parallel size for the draft model. Must be either `1` or equal to the target model's tensor parallel size.

```python
# Run draft model with TP=1 while target uses TP=4
speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1,
}
```

> **Note**: `mlp_speculator` models always use `draft_tensor_parallel_size=1`.

## Advanced Control Parameters

### `disable_padded_drafter_batch`

**Type**: `bool`  
**Default**: `False`

Disable input padding for the draft model batch. When `True`, draft input batches can contain sequences of different lengths. Only affects EAGLE-based methods. Requires an attention backend that supports variable-length queries.

### `use_local_argmax_reduction`

**Type**: `bool`  
**Default**: `False`

Use vocab-parallel local argmax instead of all-gathering full logits for draft token generation. Reduces communication from O(vocab_size) to O(2 × tp_size) per token. Only applies to greedy draft selection in non-tree speculation.

### `parallel_drafting`

**Type**: `bool`  
**Default**: `False`

Enable parallel drafting, where all speculative tokens are generated in a single draft model forward pass rather than sequentially. Requires a specially trained draft model (e.g., PARD models) with `pard_token` or `ptd_token_id` in its config.

Compatible with EAGLE and draft model methods only.

### `speculative_token_tree`

**Type**: `str | None`  
**Default**: Linear chain `"[(0,), (0,0), ...]"`

Specifies the tree structure for speculative token generation as a Python list of tuples. Each tuple represents a path from the root of the draft tree.

```python
# Default: linear chain of 3 tokens
speculative_token_tree = "[(0,), (0, 0), (0, 0, 0)]"

# Tree with branching: top-2 at root, top-1 thereafter
speculative_token_tree = "[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]"
```

The tree is automatically sorted breadth-first. The depth of the tree determines the number of draft steps.

### `enforce_eager`

**Type**: `bool | None`  
**Default**: `None` (inherits from model config)

Override the `enforce_eager` setting for speculative decoding. Some MTP configurations (e.g., DeepSeek-V3-0324) automatically set this to `True` due to CUDA graph limitations.

## N-gram Parameters

These parameters apply only when `method` is `"ngram"` or `"ngram_gpu"`.

### `prompt_lookup_max`

**Type**: `int | None`  
**Default**: `5` (if neither min nor max is specified)

Maximum size of the n-gram token window for matching. Larger values find more specific matches but may miss more opportunities.

### `prompt_lookup_min`

**Type**: `int | None`  
**Default**: Equal to `prompt_lookup_max` if not specified

Minimum size of the n-gram token window. Must be ≤ `prompt_lookup_max`.

```python
# Search for n-grams of length 3 to 7
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_min": 3,
    "prompt_lookup_max": 7,
}
```

## Suffix Decoding Parameters

These parameters apply only when `method` is `"suffix"`.

### `suffix_decoding_max_tree_depth`

**Type**: `int`  
**Default**: `24`

Maximum depth of the suffix decoding global and prompt trees. Limits the sum of prefix match length and speculation length.

### `suffix_decoding_max_cached_requests`

**Type**: `int`  
**Default**: `10000`

Maximum number of past requests to cache in the global suffix tree. Eviction is FIFO. Set to `0` to disable global caching (only per-prompt trees are used).

### `suffix_decoding_max_spec_factor`

**Type**: `float`  
**Default**: `1.0`

Controls speculation aggressiveness based on prefix match length:
```
max_spec_tokens = max_spec_factor × prefix_match_length
```

### `suffix_decoding_min_token_prob`

**Type**: `float`  
**Default**: `0.1`

Minimum estimated token probability (based on frequency counts) for a token to be included in the draft. Must be in `[0, 1]`.

## Complete Configuration Examples

### EAGLE with Tree Attention

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "method": "eagle",
        "num_speculative_tokens": 5,
        "speculative_token_tree": "[(0,), (1,), (0,0), (1,0), (0,0,0)]",
        "draft_tensor_parallel_size": 1,
    },
)
```

### EAGLE-3

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "method": "eagle3",
        "num_speculative_tokens": 5,
    },
)
```

### DeepSeek-V3 with MTP

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
```

### N-gram (GPU)

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram_gpu",
        "num_speculative_tokens": 5,
        "prompt_lookup_min": 3,
        "prompt_lookup_max": 7,
    },
)
```

### Suffix Decoding

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 8,
        "suffix_decoding_max_tree_depth": 32,
        "suffix_decoding_max_cached_requests": 20000,
        "suffix_decoding_max_spec_factor": 1.5,
        "suffix_decoding_min_token_prob": 0.1,
    },
)
```

### Medusa

```python
llm = LLM(
    model="FasterDecoding/medusa-vicuna-7b-v1.3",
    speculative_config={
        "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
        "num_speculative_tokens": 3,
    },
)
```

## OpenAI-Compatible Server

When using the OpenAI-compatible server, pass `speculative_config` as a JSON argument:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 5}'
```

Or via environment variable / config file:

```yaml
# config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
speculative_config:
  model: yuhuili/EAGLE-LLaMA3.1-Instruct-8B
  num_speculative_tokens: 5
```

## Validation and Error Handling

`SpeculativeConfig` performs extensive validation in `__post_init__`:

- `prompt_lookup_min` must be ≤ `prompt_lookup_max`
- `draft_tensor_parallel_size` must be `1` or equal to target TP size
- `num_speculative_tokens` must be divisible by `num_nextn_predict_layers` for MTP
- Target and draft models must have the same vocabulary size for `draft_model` method
- `suffix_decoding_min_token_prob` must be in `[0, 1]`
- `suffix` method requires `arctic_inference` package

## Related Pages

- [Overview](README.md) — Speculative decoding overview
- [EAGLE](eagle.md) — EAGLE speculative decoding
- [Medusa](medusa.md) — Medusa multiple decoding heads
- [N-gram Proposer](ngram.md) — N-gram prompt lookup
- [Suffix Decoding](suffix-decoding.md) — Suffix tree-based speculation
- [MTP](mtp.md) — Multi-token prediction
- [Metrics](metrics.md) — Acceptance rate and performance metrics
