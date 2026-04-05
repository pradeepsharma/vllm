# N-gram Speculative Decoding

N-gram speculative decoding (also called **prompt lookup decoding**) is a lightweight, model-free approach that proposes draft tokens by finding matching n-gram patterns in the existing token context. It requires no additional model weights and adds minimal overhead, making it ideal for tasks with repetitive text patterns such as code generation, document summarization, and retrieval-augmented generation (RAG).

## Overview

The core idea is simple: if the last few tokens of the current context (the "suffix") appear earlier in the same context, the tokens that followed that earlier occurrence are likely to follow again. The proposer searches for the longest suffix match within `[min_n, max_n]` tokens and proposes the `k` tokens that came after the match.

```mermaid
graph LR
    CTX["Context: ... the quick brown fox jumps over the lazy dog the quick ..."]
    SUFFIX["Suffix: 'the quick'"]
    MATCH["Match found at position 4"]
    DRAFT["Draft: 'brown fox jumps'"]

    CTX --> SUFFIX
    SUFFIX --> MATCH
    MATCH --> DRAFT
```

vLLM provides two implementations:

| Implementation | Class | Execution | Best For |
|---------------|-------|-----------|----------|
| CPU N-gram | `NgramProposer` | Numba JIT, parallel CPU | General use, low GPU overhead |
| GPU N-gram | `NgramProposerGPU` | PyTorch tensor ops, CUDA | High-throughput batches |

## CPU N-gram Proposer

### Implementation

The CPU proposer (`vllm/v1/spec_decode/ngram_proposer.py`) uses Numba JIT-compiled functions for fast parallel processing across the batch.

#### Algorithm

The core algorithm (`_find_longest_matched_ngram_and_propose_tokens`) uses a KMP-style (Knuth-Morris-Pratt) failure function on the **reversed** token sequence to efficiently find the longest suffix match:

```python
@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, extract k tokens right after the matched ngram.
    """
    # Flip tokens: goal becomes finding longest prefix matching suffix
    tokens = origin_tokens[::-1]

    # LPS array (Longest Proper Prefix which is also Suffix)
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1

    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    # Extract k tokens after the match
    start_position = total_token - 1 - position + longest_ngram
    return origin_tokens[start_position : start_position + k]
```

#### Batch Processing

The `batch_propose_numba` function processes all requests in parallel using Numba's `prange`:

```python
@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        # ... find and store draft tokens for request idx
```

#### Thread Management

The proposer dynamically adjusts the number of Numba threads based on batch size:

```python
# From vllm/v1/spec_decode/ngram_proposer.py
if total_tokens >= self.num_tokens_threshold:  # threshold = 8192
    final_num_threads = max(
        1, min(self.num_numba_thread_available, num_ngram_requests)
    )
    set_num_threads(final_num_threads)
else:
    set_num_threads(1)
```

The maximum thread count is capped at 1 per TP rank (with a cap of 8) to avoid contention with other components like the tokenizer and structured outputs.

### Initialization

The proposer triggers Numba JIT compilation during initialization to avoid first-call latency:

```python
# From vllm/v1/spec_decode/ngram_proposer.py
# Trigger Numba JIT compilation for N-gram proposer.
# This usually takes less than 1 second.
self.propose(
    [[]] * 1024,
    np.zeros(1024, dtype=np.int32),
    np.zeros((1024, self.max_model_len), dtype=np.int32),
)
```

## GPU N-gram Proposer

### Implementation

The GPU proposer (`vllm/v1/spec_decode/ngram_proposer_gpu.py`) uses fully vectorized PyTorch tensor operations to find n-gram matches across all sequences in parallel on the GPU.

#### Algorithm

The GPU implementation uses `torch.unfold` to create sliding windows and compares them against the trailing suffix of each sequence:

```python
# From vllm/v1/spec_decode/ngram_proposer_gpu.py
def _find_first_and_extract_all_n_parallel(
    self,
    token_ids: torch.Tensor,      # [batch_size, max_seq_len]
    seq_lengths: torch.Tensor,    # [batch_size]
    min_ngram_len: int,
    max_ngram_len: int,
    num_draft_tokens: int,
) -> torch.Tensor:
    for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
        # Sliding windows of size ngram_len; unfold is O(1) view
        search_windows = token_ids.unfold(1, ngram_len, 1)

        # Trailing suffix (last ngram_len tokens) for each sequence
        suffix = torch.gather(token_ids, 1, suffix_indices.clamp(min=0))

        # Window matches for each sequence
        matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

        # Find earliest match
        first_match_idx = torch.argmax(final_matches.int(), dim=1)
```

The algorithm tries all n-gram sizes from `min_n` to `max_n` and selects the **longest** matching n-gram. This is the opposite strategy from the CPU version (which uses KMP to find the longest match efficiently), but is more amenable to GPU parallelism.

#### NgramGPUKernel

The `NgramGPUKernel` class is decorated with `@support_torch_compile()` for torch.compile optimization:

```python
@support_torch_compile()
class NgramGPUKernel(nn.Module):
    def forward(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,        # [batch_size, max_len]
        combined_mask: torch.Tensor,        # [batch_size] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns:
        #   draft_tokens: [batch_size, k]
        #   num_valid_draft_tokens: [batch_size] int32
```

The `combined_mask` filters out requests that should not receive speculative tokens (e.g., requests at max model length).

#### Compilation Configuration

The GPU proposer uses aggressive torch.compile settings for maximum performance:

```python
# From vllm/v1/spec_decode/ngram_proposer_gpu.py
compilation_config = CompilationConfig(
    mode=CompilationMode.VLLM_COMPILE,
    inductor_compile_config={
        "enable_auto_functionalized_v2": False,
        "max_autotune": True,
        "aggressive_fusion": True,
        "triton.autotune_pointwise": True,
        "coordinate_descent_tuning": True,
    },
    cudagraph_mode=CUDAGraphMode.NONE,
)
```

## Configuration

### Basic Setup

```python
from vllm import LLM

# CPU N-gram proposer
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 1,
    },
)

# GPU N-gram proposer
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram_gpu",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 1,
    },
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | `"ngram"` (CPU) or `"ngram_gpu"` (GPU) |
| `num_speculative_tokens` | int | — | Number of draft tokens to propose (`k`) |
| `prompt_lookup_max` | int | 5 | Maximum n-gram size to search for |
| `prompt_lookup_min` | int | 5 | Minimum n-gram size to search for |

> **Default values**: If neither `prompt_lookup_min` nor `prompt_lookup_max` is specified, both default to 5.

### Choosing `prompt_lookup_min` and `prompt_lookup_max`

- **Larger `prompt_lookup_max`**: More specific matches, higher precision but fewer matches found
- **Smaller `prompt_lookup_min`**: More matches found, but potentially lower precision
- **Equal values**: Exact n-gram size matching (most common setting)

For code generation tasks, `prompt_lookup_max=5` to `10` works well. For document summarization, `prompt_lookup_max=3` to `5` is typical.

## When to Use N-gram Decoding

N-gram decoding is most effective when:

1. **Repetitive patterns**: The output frequently repeats phrases from the input (RAG, summarization)
2. **Code generation**: Variable names, function signatures, and boilerplate repeat
3. **Low-resource environments**: No GPU memory available for a draft model
4. **Latency-sensitive applications**: No draft model warmup or memory allocation needed

It is less effective for:
- Creative writing with diverse vocabulary
- Tasks where the output rarely repeats input tokens
- Very short sequences (< `min_n` tokens)

## CPU vs GPU Comparison

| Aspect | CPU N-gram | GPU N-gram |
|--------|-----------|-----------|
| Implementation | Numba JIT + KMP | PyTorch tensor ops |
| Parallelism | Multi-threaded CPU | GPU SIMD |
| Compilation | JIT on first call | torch.compile |
| Memory | CPU RAM | GPU VRAM |
| Best batch size | Small-medium | Large |
| Async with GPU | Yes (CPU runs while GPU computes) | No (blocks GPU) |

## Related Pages

- [Overview](README.md) — Speculative decoding overview
- [Suffix Decoding](suffix-decoding.md) — Suffix tree-based speculation
- [Configuration](configuration.md) — Full `SpeculativeConfig` reference
- [Metrics](metrics.md) — Acceptance rate and performance metrics
