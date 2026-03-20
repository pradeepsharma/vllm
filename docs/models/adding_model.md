# Adding a New Model

This guide walks you through the complete process of implementing and registering a new model architecture in vLLM. Whether you're adding a brand-new architecture or adapting an existing HuggingFace model, this guide covers every step.

---

## Overview

Adding a model to vLLM involves four main steps:

1. **Implement the model class** — write the `nn.Module` that defines the model's forward pass
2. **Implement weight loading** — map HuggingFace checkpoint weights to vLLM's internal parameter names
3. **Register the architecture** — add the architecture string to the `ModelRegistry`
4. **Test the model** — verify correctness against the HuggingFace reference implementation

---

## Prerequisites

Before starting, make sure you understand:

- The model's HuggingFace `config.json` structure and the `architectures` field
- The model's weight layout (parameter names in the checkpoint)
- Whether the model is decoder-only, encoder-decoder, multimodal, or an embedding model

---

## Step 1: Create the Model File

Create a new Python file in `vllm/model_executor/models/`. The filename should match the model family (e.g., `mymodel.py`).

### Minimal Decoder-Only Model Template

```python
# vllm/model_executor/models/mymodel.py
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
import torch
from torch import nn
from transformers import MyModelConfig  # or use AutoConfig

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


class MyModelAttention(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # ... add Attention layer, rotary embeddings, etc.

    def forward(self, positions, hidden_states):
        # Implement attention forward pass
        ...


class MyModelDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MyModelAttention(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        # Add MLP, layer norms, etc.

    def forward(self, positions, hidden_states, residual):
        # Implement decoder layer forward pass
        ...


class MyModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MyModelDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # See Step 2 for weight loading implementation
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class MyModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # Declare LoRA support
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
        self.model = MyModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

---

## Step 2: Implement Weight Loading

Weight loading maps checkpoint parameter names to vLLM's internal parameter names. vLLM provides several utilities to simplify this.

### Using `AutoWeightsLoader` (Recommended)

`AutoWeightsLoader` automatically handles most weight loading scenarios:

```python
from .utils import AutoWeightsLoader

def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

### Manual Weight Loading with Stacked Parameters

For models that stack QKV or gate/up projections, use `stacked_params_mapping`:

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
        # Skip positional embeddings that aren't needed
        if "rotary_emb.inv_freq" in name:
            continue

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

### Weight Name Remapping

If the HuggingFace checkpoint uses different parameter names than vLLM's internal names, use a `WeightsMapper`:

```python
from .utils import WeightsMapper

# In your model class:
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "transformer.": "model.",
        "transformer.wte": "model.embed_tokens",
    },
    orig_to_new_suffix={
        ".weight": "",  # strip .weight suffix if needed
    },
)

def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    weights = self.hf_to_vllm_mapper.apply(weights)
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

---

## Step 3: Implement Required Interfaces

### For Text Generation Models

Implement `VllmModelForTextGeneration` by providing `compute_logits`:

```python
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration

class MyModelForCausalLM(nn.Module, VllmModelForTextGeneration, SupportsPP):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
```

### For Embedding Models

Implement `VllmModelForPooling` by providing a pooler:

```python
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

class MyModelForEmbedding(nn.Module, VllmModelForPooling):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # ... model layers ...
        self._pooler = Pooler.from_config_with_defaults(
            vllm_config.model_config.pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=True,
            softmax=False,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata,
    ) -> torch.Tensor:
        return self._pooler(hidden_states, pooling_metadata)
```

### For Multimodal Models

Implement `SupportsMultiModal` and register a multimodal processor:

```python
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_processor(MyModelMultiModalProcessor)
class MyModelForConditionalGeneration(nn.Module, SupportsMultiModal):
    supports_multimodal: ClassVar[Literal[True]] = True

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        # Process and return multimodal embeddings
        ...
```

See [Multimodal Models](multimodal_models.md) for the full multimodal implementation guide.

### Declaring Pipeline Parallelism Support

To support pipeline parallelism, implement `SupportsPP` and use `make_layers`:

```python
from .interfaces import SupportsPP
from .utils import make_layers, PPMissingLayer

class MyModelForCausalLM(nn.Module, SupportsPP):
    supports_pp: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # make_layers automatically handles PP rank boundaries
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MyDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
```

### Declaring LoRA Support

To support LoRA fine-tuning, implement `SupportsLoRA`:

```python
from .interfaces import SupportsLoRA

class MyModelForCausalLM(nn.Module, SupportsLoRA):
    supports_lora: ClassVar[Literal[True]] = True

    # Map fused parameter names to their constituent weight names
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # List all modules that can have LoRA adapters
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
        "embed_tokens", "lm_head",
    ]

    # Map embedding module names to their role
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Modules that need padding to vocab size
    embedding_padding_modules = ["lm_head"]
```

---

## Step 4: Register the Architecture

Add your model to the appropriate dictionary in `vllm/model_executor/models/registry.py`:

```python
# In registry.py

_TEXT_GENERATION_MODELS = {
    # ... existing entries ...

    # Add your model:
    "MyModelForCausalLM": ("mymodel", "MyModelForCausalLM"),
    #  ^                    ^           ^
    #  HF architecture      module      class name
    #  string from          filename    in mymodel.py
    #  config.json          (without .py)
}
```

The tuple format is `(module_name, class_name)` where:
- `module_name` is the Python module name relative to `vllm.model_executor.models` (the filename without `.py`)
- `class_name` is the Python class name within that module

For embedding models, add to `_EMBEDDING_MODELS`. For multimodal models, add to `_MULTIMODAL_MODELS`.

!!! important "Update the test registry too"
    After adding to `registry.py`, also add an example HuggingFace model to `tests/models/registry.py`:
    ```python
    "MyModelForCausalLM": ModelInfo(
        model="your-org/your-model-name",
        tokenizer="your-org/your-model-name",
    ),
    ```

---

## Step 5: Add Architecture-Specific Config (Optional)

If your model requires special configuration validation or overrides, add a config class to `vllm/model_executor/models/config.py`:

```python
# In config.py

class MyModelForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        # Example: set a default reasoning parser
        if vllm_config.structured_outputs_config.reasoning_parser == "":
            vllm_config.structured_outputs_config.reasoning_parser = "my_parser"

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        # Example: validate HF config fields
        hf_config = model_config.hf_config
        assert hasattr(hf_config, "my_required_field"), (
            "MyModel requires 'my_required_field' in config.json"
        )
```

---

## Step 6: Write Tests

Add tests to verify your model works correctly:

### Basic Correctness Test

```python
# tests/models/test_mymodel.py
import pytest
from vllm import LLM, SamplingParams

@pytest.mark.parametrize("model", ["your-org/your-model-name"])
def test_mymodel_generation(model):
    llm = LLM(model=model, max_model_len=512)
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(max_tokens=20, temperature=0.0),
    )
    assert len(outputs[0].outputs[0].text) > 0
```

### Comparing Against HuggingFace Reference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def test_mymodel_matches_hf(model_name="your-org/your-model-name"):
    # HuggingFace reference
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits[0, -1]

    # vLLM
    llm = LLM(model=model_name, dtype="float16")
    # Compare logits or generated tokens
    ...
```

---

## Common Patterns and Tips

### Using `@support_torch_compile`

For models that benefit from `torch.compile`, add the decorator to the core model class:

```python
from vllm.compilation.decorators import support_torch_compile

@support_torch_compile
class MyModel(nn.Module):
    ...
```

### Handling Tied Embeddings

Many models share weights between the input embedding and the output LM head:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    params_dict = dict(self.named_parameters())
    loaded_params = set()

    for name, loaded_weight in weights:
        if "lm_head.weight" in name and self.config.tie_word_embeddings:
            # Skip — lm_head shares weights with embed_tokens
            continue
        # ... rest of loading logic
```

### Handling MoE (Mixture of Experts) Models

For MoE models, use `FusedMoE` from vLLM's layers:

```python
from vllm.model_executor.layers.fused_moe import FusedMoE

class MyMoELayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.experts",
        )
```

### Handling Quantization

vLLM's quantization is applied automatically through `QuantizationConfig`. Pass it to all linear layers:

```python
from vllm.model_executor.layers.quantization import QuantizationConfig

class MyLinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.linear = ColumnParallelLinear(
            in_features, out_features,
            quant_config=quant_config,
            prefix=prefix,
        )
```

### Prefix Handling

Always use `maybe_prefix` when constructing sub-module prefixes to correctly handle nested models:

```python
from .utils import maybe_prefix

class MyModelForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.model = MyModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
```

---

## Checklist

Before submitting a new model, verify:

- [ ] Model file created in `vllm/model_executor/models/`
- [ ] Architecture registered in `vllm/model_executor/models/registry.py`
- [ ] Example model added to `tests/models/registry.py`
- [ ] `load_weights` correctly maps all checkpoint parameters
- [ ] `embed_input_ids` method implemented
- [ ] `forward` method accepts `input_ids`, `positions`, `intermediate_tensors`
- [ ] `compute_logits` implemented (for text generation models)
- [ ] `SupportsLoRA` implemented if LoRA is desired
- [ ] `SupportsPP` implemented if pipeline parallelism is desired
- [ ] Tests pass: `pytest tests/models/test_mymodel.py`
- [ ] Docstring added to the model class
- [ ] Apache 2.0 license header added

---

## Using the Transformers Backend (Alternative)

If you don't want to write a full vLLM implementation, you can use the **Transformers backend** for any HuggingFace-compatible model:

```bash
vllm serve your-org/your-model --model-impl transformers
```

Or in Python:

```python
from vllm import LLM
llm = LLM(model="your-org/your-model", model_impl="transformers")
```

The Transformers backend provides full compatibility but may have lower throughput than a native vLLM implementation. It's a good starting point before writing a native implementation.

---

## Getting Help

- Browse existing model implementations in `vllm/model_executor/models/` for reference
- The `llama.py` implementation is a good reference for decoder-only models
- The `qwen2_vl.py` implementation is a good reference for vision-language models
- Open a GitHub issue or discussion if you need guidance on a specific architecture
