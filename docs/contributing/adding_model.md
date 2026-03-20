# Contributing a New Model

This guide walks you through the complete process of contributing a new model architecture to vLLM. It covers the implementation, testing, and PR submission process.

> **See also:** The [Model Implementation Reference](../models/adding_model.md) provides detailed API documentation and code templates. This guide focuses on the contribution workflow and best practices.

---

## Before You Start

### Check Existing Support

Before implementing a new model, verify it isn't already supported:

1. Check [Supported Models](../models/supported_models.md)
2. Search [GitHub Issues](https://github.com/vllm-project/vllm/issues?q=label%3A%22new+model%22) for existing requests
3. Search the model registry: `grep -r "MyModelForCausalLM" vllm/model_executor/models/registry.py`

### Open a GitHub Issue First

For new models, it's good practice to open a [New Model issue](https://github.com/vllm-project/vllm/issues/new?template=600-new-model.yml) first. This:
- Lets maintainers confirm the model is a good fit
- Avoids duplicate work if someone else is already implementing it
- Gets early feedback on your approach

Include in the issue:
- The HuggingFace model URL
- The closest model vLLM already supports
- Any novel operators or architectural differences

### Understand the Model Architecture

Before writing code, study:
- The model's HuggingFace `config.json` — especially the `architectures` field
- The HuggingFace implementation in `transformers`
- The model's weight layout (parameter names in the checkpoint)
- Whether it's decoder-only, encoder-decoder, multimodal, or an embedding model

---

## Implementation Overview

Adding a model involves these steps:

1. **Create the model file** in `vllm/model_executor/models/`
2. **Implement weight loading** to map HuggingFace weights to vLLM parameters
3. **Implement required interfaces** (generation, embedding, multimodal, etc.)
4. **Register the architecture** in `vllm/model_executor/models/registry.py`
5. **Write tests** to verify correctness
6. **Update documentation** (`docs/models/supported_models.md`)

---

## Step 1: Create the Model File

Create `vllm/model_executor/models/mymodel.py`. Use an existing similar model as a reference — for example, `llama.py` for a standard decoder-only transformer.

### File Header

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

### Key Design Principles

**Use vLLM's parallel layers**, not standard PyTorch layers:

| Standard PyTorch | vLLM Equivalent | Purpose |
|---|---|---|
| `nn.Linear` | `ColumnParallelLinear` | Column-sharded linear |
| `nn.Linear` | `RowParallelLinear` | Row-sharded linear |
| `nn.Embedding` | `VocabParallelEmbedding` | Vocabulary-sharded embedding |
| `nn.Linear` (QKV) | `QKVParallelLinear` | Fused QKV projection |
| `nn.Linear` (LM head) | `ParallelLMHead` | Parallel language model head |

**Accept `VllmConfig`** as the primary configuration object:

```python
def __init__(
    self,
    *,
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
```

**Use `prefix` for parameter naming** — this enables pipeline parallelism and proper weight loading:

```python
self.self_attn = MyAttention(
    vllm_config=vllm_config,
    prefix=maybe_prefix(prefix, "self_attn"),
)
```

### Minimal Decoder-Only Template

```python
# vllm/model_executor/models/mymodel.py
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
from torch import nn
from transformers import MyModelConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class MyModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """MyModel for causal language modeling."""

    supports_lora: ClassVar[Literal[True]] = True
    supports_pp: ClassVar[Literal[True]] = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = MyModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

---

## Step 2: Implement Weight Loading

### Use `AutoWeightsLoader` When Possible

For models where HuggingFace parameter names match vLLM's internal names:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

### Handle Stacked Parameters

Most transformer models fuse QKV and gate/up projections. Use `stacked_params_mapping`:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        # (vllm_param_name, hf_weight_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue  # Skip positional embeddings

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            break
        else:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    return loaded_params
```

### Handle Name Remapping

If the HuggingFace checkpoint uses different names than vLLM's internal structure:

```python
from .utils import WeightsMapper

hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "transformer.": "model.",
        "transformer.wte": "model.embed_tokens",
    },
)

def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    weights = self.hf_to_vllm_mapper.apply(weights)
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

---

## Step 3: Register the Architecture

Add your model to `vllm/model_executor/models/registry.py`:

```python
_TEXT_GENERATION_MODELS = {
    # ... existing entries ...
    "MyModelForCausalLM": ("mymodel", "MyModelForCausalLM"),
}
```

The tuple is `(module_name, class_name)`:
- `module_name`: Python module relative to `vllm.model_executor.models` (filename without `.py`)
- `class_name`: The class name within that module

For different model types:
- Text generation → `_TEXT_GENERATION_MODELS`
- Embedding/pooling → `_EMBEDDING_MODELS`
- Multimodal → `_MULTIMODAL_MODELS`
- Encoder-decoder → `_ENCODER_DECODER_MODELS`

---

## Step 4: Update the Test Registry

Add an example model to `tests/models/registry.py`:

```python
from tests.models.registry import ModelInfo

# In the appropriate section:
"MyModelForCausalLM": ModelInfo(
    model="your-org/your-model-name",
    tokenizer="your-org/your-model-name",
),
```

---

## Step 5: Write Tests

### Correctness Test

Create `tests/models/language/generation/test_mymodel.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.mark.parametrize("model", ["your-org/your-model-name"])
@pytest.mark.core_model
def test_mymodel_generation(model: str, vllm_runner, hf_runner):
    """Test that MyModel output matches HuggingFace reference."""
    prompts = [
        "The capital of France is",
        "Once upon a time",
    ]
    max_tokens = 20

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompts, max_tokens=max_tokens)

    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens=max_tokens)

    for hf_out, vllm_out in zip(hf_outputs, vllm_outputs):
        assert hf_out[1] == vllm_out[1], (
            f"Output mismatch:\n  HF:   {hf_out[1]!r}\n  vLLM: {vllm_out[1]!r}"
        )
```

### Logprob Comparison Test

For more rigorous correctness checking:

```python
from tests.models.utils import check_logprobs_close

def test_mymodel_logprobs(vllm_runner, hf_runner):
    model = "your-org/your-model-name"
    prompts = ["The quick brown fox"]

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(prompts, max_tokens=10)

    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(prompts, max_tokens=10)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
```

### Running Your Tests

```bash
cd tests
pytest -v -s models/language/generation/test_mymodel.py
```

---

## Step 6: Update Documentation

Update `docs/models/supported_models.md` to include your model:

```markdown
| MyModel | `MyModelForCausalLM` | Text Generation | ✅ | ✅ | ✅ |
```

Include:
- Model family name
- HuggingFace architecture string
- Task type
- Supported features (LoRA, PP, etc.)

---

## PR Checklist

Before submitting your PR:

- [ ] Model file created in `vllm/model_executor/models/`
- [ ] Architecture registered in `registry.py`
- [ ] Test registry updated in `tests/models/registry.py`
- [ ] Tests written and passing
- [ ] `supported_models.md` updated
- [ ] Pre-commit hooks pass (`pre-commit run --all-files --hook-stage manual`)
- [ ] No mypy errors introduced
- [ ] PR description includes:
  - Link to the HuggingFace model
  - Test command and results
  - Comparison with HuggingFace reference output

---

## Common Pitfalls

### Weight Loading Failures

**Symptom:** `RuntimeError: Missing keys in state dict`

**Fix:** Check that your `stacked_params_mapping` covers all fused parameters, and that `WeightsMapper` correctly remaps any renamed parameters.

```python
# Debug: print all weight names from the checkpoint
for name, _ in weights:
    print(name)
```

### Numerical Differences

**Symptom:** Outputs differ significantly from HuggingFace

**Causes and fixes:**
- **Attention implementation**: Ensure you're using the correct attention backend
- **RoPE configuration**: Verify `rope_theta`, `rope_scaling`, and `max_position_embeddings`
- **Normalization**: Check if the model uses RMSNorm vs LayerNorm, and the epsilon value
- **Dtype**: Test with `dtype="float32"` to rule out precision issues

### Tensor Parallel Errors

**Symptom:** Errors when using `tensor_parallel_size > 1`

**Fix:** Ensure all linear layers use the appropriate parallel variants (`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`). The `num_heads` must be divisible by `tensor_parallel_size`.

### Pipeline Parallel Errors

**Symptom:** Errors when using `pipeline_parallel_size > 1`

**Fix:** Use `make_layers()` for the decoder layers and implement `make_empty_intermediate_tensors`. Ensure `SupportsPP` is declared.

---

## Getting Help

If you're stuck:
- Look at a similar model implementation (e.g., `llama.py`, `mistral.py`, `qwen2.py`)
- Ask in the `#contributing` channel on [vLLM Slack](https://slack.vllm.ai)
- Tag a maintainer in your GitHub issue or PR

See also:
- [Model Implementation Reference](../models/adding_model.md) — detailed API docs
- [Model Registry](../models/model_registry.md) — registry structure
- [Multimodal Models](../models/multimodal_models.md) — multimodal implementation guide
