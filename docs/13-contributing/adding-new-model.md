# Adding a New Model

This page provides a contributor-focused guide to adding a new model architecture to vLLM. For the full technical reference on model interfaces, weight loading, and the model registry internals, see [Adding a New Model Architecture](../04-models/adding-new-model.md) in the Models section.

## When to Add a Model Here vs. Upstream

Before implementing a new model, consider:

1. **Is the architecture already supported?** Check `vllm/model_executor/models/` — many HuggingFace model families share an architecture (e.g., all Llama variants use `LlamaForCausalLM`).
2. **Is there an open issue or PR?** Search [GitHub Issues](https://github.com/vllm-project/vllm/issues) for the model name.
3. **Does it require new kernels?** New attention patterns or quantization schemes may need kernel work beyond the model file itself.

## Contributor Checklist

When submitting a PR to add a new model, ensure you have completed all of the following:

```
[ ] Created model file in vllm/model_executor/models/<model_name>.py
[ ] Added SPDX license header to the new file
[ ] Registered the model in vllm/model_executor/models/registry.py
[ ] Added test entry in tests/models/registry.py
[ ] Added the model to the supported models table in docs/04-models/supported-models.md
[ ] Verified correctness against HuggingFace reference outputs
[ ] Tested with tensor parallelism (if applicable)
[ ] Added multimodal processor registration (if multimodal)
[ ] Ran pre-commit hooks: pre-commit run --all-files
```

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/<model>.py` | Model implementation |
| `vllm/model_executor/models/registry.py` | Model registry (maps config class name → vLLM class) |
| `tests/models/registry.py` | Test registry (which models to test and how) |
| `vllm/model_executor/models/__init__.py` | Package exports |

## Step-by-Step Summary

### 1. Implement the Model

Create `vllm/model_executor/models/mymodel.py`. The model class must:

- Inherit from `nn.Module`
- Implement `forward()` accepting `input_ids`, `positions`, `intermediate_tensors`, and optionally `inputs_embeds`
- Implement `load_weights()` to map HuggingFace checkpoint keys to vLLM layer names
- Implement the appropriate interface (`SupportsLoRA`, `SupportsPP`, `SupportsV0Only`, etc.)

For a complete implementation template, see [Adding a New Model Architecture](../04-models/adding-new-model.md#step-1-create-the-model-file).

### 2. Register in the Model Registry

In `vllm/model_executor/models/registry.py`, add an entry to `_TEXT_GENERATION_MODELS` (or the appropriate dict):

```python
_TEXT_GENERATION_MODELS = {
    # ...existing entries...
    "MyModelForCausalLM": ("mymodel", "MyModelForCausalLM"),
}
```

The tuple is `(module_name, class_name)` where `module_name` is the filename without `.py`.

For multimodal models, add to `_MULTIMODAL_MODELS` instead:

```python
_MULTIMODAL_MODELS = {
    "MyVisionModelForConditionalGeneration": (
        "mymodel", "MyVisionModelForConditionalGeneration"
    ),
}
```

### 3. Add a Test Entry

In `tests/models/registry.py`, add the model to the test registry so it gets picked up by the CI model tests:

```python
# For a core model (tested on every PR):
@pytest.mark.core_model
def test_mymodel():
    ...

# Or add to the model list with appropriate markers
```

### 4. Verify Correctness

Run the correctness check against HuggingFace:

```bash
# Compare vLLM output to HuggingFace reference
pytest tests/models/test_mymodel.py -x -v

# Or use the generic correctness test
pytest tests/basic_correctness/ -k mymodel -x -v
```

### 5. Test Parallelism

If the model supports tensor parallelism, test it:

```bash
# TP=2
pytest tests/distributed/ -k mymodel -x -v
```

## Common Pitfalls

### Weight Key Mismatches

The most common issue when adding a model is weight key mismatches between HuggingFace checkpoints and vLLM's layer naming. Use `AutoWeightsLoader` to handle common patterns automatically:

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights)
```

For custom key mappings, implement `load_weights()` manually with explicit renaming.

### Missing `intermediate_tensors` Support

For pipeline parallelism, the model must handle `intermediate_tensors` correctly. Use `make_empty_intermediate_tensors_factory()` and `PPMissingLayer` for layers that don't exist on a given pipeline stage.

### Multimodal Input Processing

Multimodal models require registering a processor in the multimodal registry. See [Multimodal Models](../04-models/multimodal-models.md) and [Multimodal Registry](../10-multimodal/registry.md) for details.

### LoRA Compatibility

If the model should support LoRA, implement the `SupportsLoRA` interface and declare `supported_lora_modules` and `embedding_modules`. See [LoRA Adapters](../10-lora/README.md) for details.

## Cross-References

- **Full implementation guide**: [Adding a New Model Architecture](../04-models/adding-new-model.md)
- **Model interfaces**: [Model Interfaces](../04-models/model-interfaces.md)
- **Model registry internals**: [Registry Internals](../04-models/registry-internals.md)
- **Multimodal models**: [Multimodal Models](../04-models/multimodal-models.md)
- **LoRA support**: [LoRA Adapters](../10-lora/README.md)
- **Weight loading**: [Model Loading](../04-models/model-loading.md)
