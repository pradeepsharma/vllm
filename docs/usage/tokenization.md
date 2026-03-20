# Tokenization

vLLM's tokenization layer converts raw text into token IDs (and back) for every model it serves.
This guide explains how tokenizers are selected, configured, and extended — including the
model-specific tokenizer implementations that ship with vLLM.

## Overview

Tokenization in vLLM is handled by the `vllm.tokenizers` package. The package exposes a
**registry-based** system that maps a *tokenizer mode* string to a concrete tokenizer class.
At startup, vLLM resolves the correct tokenizer automatically based on the model repository
contents, then caches the result so subsequent calls are free.

```
Model name / path
       │
       ▼
resolve_tokenizer_args()   ← detects mode from repo files
       │
       ▼
TokenizerRegistry.load_tokenizer()   ← instantiates the right class
       │
       ▼
cached_get_tokenizer()   ← LRU-cached for reuse across workers
```

## Tokenizer Modes

The `--tokenizer-mode` CLI flag (or `tokenizer_mode` in `LLM(...)`) controls which tokenizer
implementation is used.

| Mode | Class | When used |
|---|---|---|
| `auto` | Resolved at runtime | Default — vLLM inspects the repo and picks the best match |
| `hf` | `CachedHfTokenizer` | HuggingFace `AutoTokenizer` with property caching |
| `slow` | `CachedHfTokenizer` (slow) | Forces the Python-based slow tokenizer |
| `mistral` | `MistralTokenizer` | Mistral/Mixtral models with `tekken.json` or `tokenizer.model.v*` |
| `deepseek_v32` | `DeepseekV32Tokenizer` | DeepSeek V3.2 with custom chat template encoding |
| `grok2` | `Grok2Tokenizer` | Grok-2 `.tok.json` tiktoken format |
| `qwen_vl` | `QwenVLTokenizer` | Qwen-VL (patches out image-pad token logic) |

### Auto-Detection Logic

When `tokenizer_mode="auto"` (the default), vLLM inspects the model repository in this order:

1. **Mistral** — if the repo contains `tekken.json` or `tokenizer.model.v*` files and is a
   recognised Mistral repository, `mistral` mode is selected.
2. **Grok-2** — if the repo contains `tokenizer.tok.json`, `grok2` mode is selected.
3. **Qwen-VL** — if the model path contains `/Qwen-VL`, `qwen_vl` mode is selected.
4. **HuggingFace** — fallback for all other models.

## Tokenizer Implementations

### HuggingFace Tokenizer (`hf` / `auto`)

The default tokenizer wraps `transformers.AutoTokenizer` with a performance-optimised caching
layer. By default, `transformers` recomputes properties such as `all_special_ids`,
`all_special_tokens`, and `get_vocab()` on every access. vLLM's `CachedHfTokenizer` computes
these once at construction time and stores them on the instance.

**Key properties cached at construction:**

- `all_special_ids` — list of special token IDs
- `all_special_tokens` — list of special token strings
- `get_vocab()` — full vocabulary dictionary
- `max_token_id` — highest token ID in the vocabulary
- `max_chars_per_token` — longest token string length

**Slow tokenizer mode:**

```bash
vllm serve <model> --tokenizer-mode slow
```

Forces `use_fast=False` on the HuggingFace tokenizer. This is slower but may be required for
models whose fast tokenizer has known bugs.

**Trust remote code:**

```bash
vllm serve <model> --trust-remote-code
```

Required for models that ship a custom tokenizer class in their repository (e.g., some Qwen
variants). Without this flag, vLLM raises a `RuntimeError` with a clear message.

### Mistral Tokenizer (`mistral`)

Mistral models use the `mistral-common` library rather than HuggingFace tokenizers. vLLM's
`MistralTokenizer` wraps `MistralCommonBackend` (compatible with both `transformers` v4 and v5).

**Supported backends:**

- **Tekkenizer** — the newer BPE tokenizer used by Mistral v3+ models (`tekken.json`)
- **SentencePiece** — the original tokenizer used by Mistral v1/v2 (`tokenizer.model.v*`)

**Limitations:**

- `chat_template` and `chat_template_kwargs` per-request overrides are **not supported** for
  Mistral tokenizers. The template is always the one bundled with `mistral-common`.
- Tool call IDs are automatically truncated to the last 9 characters to satisfy Mistral's ID
  length requirement.

**Usage:**

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tokenizer-mode mistral
```

### DeepSeek V3.2 Tokenizer (`deepseek_v32`)

DeepSeek V3.2 uses a custom chat template encoding that differs from the standard HuggingFace
`apply_chat_template` path. The `DeepseekV32Tokenizer` wraps a standard HuggingFace tokenizer
and overrides `apply_chat_template` to call the internal `encode_messages` function.

**Thinking mode support:**

The DeepSeek V3.2 tokenizer supports a `thinking` (or `enable_thinking`) keyword argument to
`apply_chat_template`. When `True`, the template uses `thinking_mode="thinking"` encoding;
otherwise `thinking_mode="chat"`.

```python
from vllm import LLM

llm = LLM(model="deepseek-ai/DeepSeek-V3-2")
outputs = llm.chat(
    messages=[{"role": "user", "content": "Solve: 2+2"}],
    chat_template_kwargs={"thinking": True},
)
```

### Grok-2 Tokenizer (`grok2`)

Grok-2 uses a tiktoken-style tokenizer stored in `tokenizer.tok.json`. vLLM's `Grok2Tokenizer`
implements the full `TokenizerLike` protocol using this format, including BPE encoding with a
custom regex pattern and special token handling.

**Special tokens:**

| Token | Purpose |
|---|---|
| `<\|pad\|>` | Padding |
| `<\|eos\|>` | End of sequence |
| `<\|separator\|>` | Turn separator |

### Qwen-VL Tokenizer (`qwen_vl`)

The original Qwen-VL tokenizer adds image-pad tokens during tokenization, which conflicts with
vLLM's multimodal processing pipeline. `QwenVLTokenizer` patches out this behaviour so that
image tokens are handled exclusively by the `QwenVLProcessor`.

## Configuration Reference

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--tokenizer` | *(same as model)* | Path or HuggingFace repo ID for the tokenizer |
| `--tokenizer-mode` | `auto` | Tokenizer mode: `auto`, `slow`, `mistral`, `deepseek_v32`, `grok2`, `qwen_vl` |
| `--tokenizer-revision` | `None` | Git revision for the tokenizer (branch, tag, or commit SHA) |
| `--trust-remote-code` | `False` | Allow executing custom tokenizer code from the model repo |
| `--skip-tokenizer-init` | `False` | Skip tokenizer initialisation (for embedding-only workflows) |

### Python API

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",  # defaults to model
    tokenizer_mode="auto",
    tokenizer_revision=None,
    trust_remote_code=False,
    skip_tokenizer_init=False,
)
```

### Using a Separate Tokenizer

You can point vLLM at a different tokenizer than the model weights. This is useful when:

- The model repo does not include a tokenizer.
- You want to use a patched or updated tokenizer.

```bash
vllm serve /path/to/model \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct
```

### ModelScope Support

When the environment variable `VLLM_USE_MODELSCOPE=1` is set, vLLM downloads the tokenizer
from [ModelScope](https://modelscope.cn/) instead of HuggingFace Hub:

```bash
VLLM_USE_MODELSCOPE=1 vllm serve Qwen/Qwen2.5-7B-Instruct
```

### GGUF Tokenizers

For GGUF model files, vLLM automatically extracts the tokenizer from the GGUF file:

```bash
# Local GGUF file
vllm serve ./model.gguf

# Remote GGUF from HuggingFace Hub
vllm serve bartowski/Llama-3.2-3B-Instruct-GGUF \
    --tokenizer-mode auto
```

## Truncation Side

vLLM sets the tokenizer's `truncation_side` based on the runner type:

| Runner type | Truncation side | Reason |
|---|---|---|
| `generate` (default) | `left` | Preserves the most recent context when truncating |
| `draft` (speculative) | `left` | Same as generate |
| `pooling` (embeddings) | `right` | Preserves the beginning of the input |

## Incremental Detokenization

vLLM uses **incremental detokenization** during streaming to avoid re-decoding the entire
sequence on every new token. The algorithm is adapted from HuggingFace TGI:

1. Only the last few tokens of the prompt are converted to strings at the start
   (`INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5`).
2. On each new token, a small prefix window is decoded alongside the new token to defeat
   cleanup algorithms that add/remove spaces based on surrounding context.
3. If the new text ends with the Unicode replacement character (`\ufffd`), the token is part
   of an incomplete UTF-8 sequence and the output is held back until the sequence is complete.

### Spaces Between Special Tokens

The `spaces_between_special_tokens` parameter (default `True`) controls whether a space is
inserted between consecutive special tokens during detokenization. Set to `False` for models
that use special tokens as delimiters without spaces.

## Programmatic Tokenizer Access

### `get_tokenizer`

```python
from vllm.tokenizers import get_tokenizer

tokenizer = get_tokenizer(
    "meta-llama/Llama-3.2-3B-Instruct",
    tokenizer_mode="auto",
    trust_remote_code=False,
)

# Encode text
ids = tokenizer.encode("Hello, world!")
print(ids)  # [128000, 9906, 11, 1917, 0]

# Decode token IDs
text = tokenizer.decode(ids, skip_special_tokens=True)
print(text)  # Hello, world!
```

### `cached_get_tokenizer`

An LRU-cached version of `get_tokenizer`. Repeated calls with the same arguments return the
same tokenizer instance without re-loading from disk:

```python
from vllm.tokenizers import cached_get_tokenizer

# First call loads from disk
tok1 = cached_get_tokenizer("meta-llama/Llama-3.2-3B-Instruct")

# Second call returns the cached instance
tok2 = cached_get_tokenizer("meta-llama/Llama-3.2-3B-Instruct")

assert tok1 is tok2  # True
```

### `TokenizerRegistry`

The registry maps tokenizer mode strings to `(module, class_name)` tuples. You can register
a custom tokenizer class at runtime:

```python
from vllm.tokenizers import TokenizerRegistry

# Register a custom tokenizer
TokenizerRegistry.register(
    tokenizer_mode="my_custom",
    module="my_package.my_tokenizer",
    class_name="MyCustomTokenizer",
)
```

Your class must implement the `TokenizerLike` protocol (see below).

## The `TokenizerLike` Protocol

All vLLM tokenizers implement the `vllm.tokenizers.TokenizerLike` protocol. This is a
structural protocol (duck typing), so any class that implements the required methods and
properties is compatible — no inheritance required.

### Required Properties

| Property | Type | Description |
|---|---|---|
| `is_fast` | `bool` | Whether this is a fast (Rust-backed) tokenizer |
| `vocab_size` | `int` | Number of tokens in the base vocabulary |
| `max_token_id` | `int` | Highest token ID (may exceed `vocab_size` for added tokens) |
| `max_chars_per_token` | `int` | Length of the longest token string |
| `bos_token_id` | `int` | Beginning-of-sequence token ID |
| `eos_token_id` | `int` | End-of-sequence token ID |
| `pad_token_id` | `int` | Padding token ID |
| `all_special_tokens` | `list[str]` | All special token strings |
| `all_special_ids` | `list[int]` | All special token IDs |
| `truncation_side` | `str` | `"left"` or `"right"` |

### Required Methods

| Method | Signature | Description |
|---|---|---|
| `from_pretrained` | `(path_or_repo_id, *, trust_remote_code, revision, download_dir, **kwargs) → TokenizerLike` | Load from disk or Hub |
| `encode` | `(text, truncation, max_length, add_special_tokens) → list[int]` | Encode text to token IDs |
| `decode` | `(ids, skip_special_tokens) → str` | Decode token IDs to text |
| `apply_chat_template` | `(messages, tools, **kwargs) → str \| list[int]` | Apply a chat template |
| `convert_tokens_to_ids` | `(tokens) → int \| list[int]` | Token strings to IDs |
| `convert_ids_to_tokens` | `(ids, skip_special_tokens) → list[str]` | IDs to token strings |
| `convert_tokens_to_string` | `(tokens) → str` | Token strings to text |
| `get_vocab` | `() → dict[str, int]` | Full vocabulary |
| `get_added_vocab` | `() → dict[str, int]` | Added (non-base) tokens |
| `num_special_tokens_to_add` | `() → int` | Number of special tokens added by default |

### Implementing a Custom Tokenizer

```python
from pathlib import Path
from typing import Any
from vllm.tokenizers import TokenizerLike, TokenizerRegistry


class MyTokenizer:
    """Minimal custom tokenizer implementing TokenizerLike."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "MyTokenizer":
        instance = cls()
        # Load your tokenizer data here
        return instance

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return 32000

    # ... implement remaining required properties and methods ...

    def encode(self, text: str, **kwargs) -> list[int]:
        # Your encoding logic
        return []

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        # Your decoding logic
        return ""

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        # Your chat template logic
        return ""


# Register and use
TokenizerRegistry.register("my_custom", "my_package.tokenizer", "MyTokenizer")
```

## Performance Tips

### Use Fast Tokenizers

vLLM logs a warning if a slow tokenizer is detected:

```
Using a slow tokenizer. This might cause a significant slowdown.
Consider using a fast tokenizer instead.
```

Most modern models on HuggingFace Hub ship with a fast (Rust-backed) tokenizer. If yours does
not, consider converting it with `tokenizers` library's `Tokenizer.from_pretrained()`.

### Tokenizer Caching

`cached_get_tokenizer` uses Python's `functools.lru_cache`. The cache is process-local and
unbounded by default. In long-running servers, this is desirable since the same tokenizer is
reused for every request.

If you need to load multiple different tokenizers in the same process (e.g., in tests), be
aware that the cache will hold references to all of them.

### Offline / Air-Gapped Environments

Set `HF_HUB_OFFLINE=1` to prevent vLLM from making any network requests:

```bash
HF_HUB_OFFLINE=1 vllm serve /local/path/to/model
```

Pre-download the tokenizer with:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --include "tokenizer*" "special_tokens_map.json"
```

## Troubleshooting

**`RuntimeError: Failed to load the tokenizer`**
: The model uses a custom tokenizer class. Add `--trust-remote-code` to allow executing the
  tokenizer code from the model repository.

**`ValueError: No tokenizer registered for tokenizer_mode='...'`**
: An unknown `--tokenizer-mode` was specified. Check the list of supported modes above or
  register a custom tokenizer with `TokenizerRegistry.register()`.

**`ValueError: Cannot use the fast tokenizer in slow tokenizer mode`**
: You passed `--tokenizer-mode slow` together with `use_fast=True`. Remove the conflicting
  argument.

**Slow tokenization throughput**
: Ensure you are using a fast tokenizer (`is_fast=True`). If the model only ships a slow
  tokenizer, consider using a community-converted fast tokenizer from HuggingFace Hub.

**Incorrect output with special tokens**
: Check `skip_special_tokens` in your generation config. By default, vLLM includes special
  tokens in the output. Pass `skip_special_tokens=True` to strip them.

**Truncation cutting off the wrong end**
: For generation models, vLLM truncates from the **left** (oldest tokens) to preserve recent
  context. For pooling/embedding models, truncation is from the **right**. Override with
  `tokenizer.truncation_side = "right"` if needed.

## Further Reading

- [Chat Templates](chat_templates.md) — how vLLM applies Jinja2 templates to message lists
- [Tool Calling Templates](tool_templates.md) — templates that add function/tool calling support
- [Sampling Parameters](sampling_params.md) — `skip_special_tokens` and related options
- [Engine Arguments](../configuration/engine_args.md) — full list of tokenizer-related engine args
