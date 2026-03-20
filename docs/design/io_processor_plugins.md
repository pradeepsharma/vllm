# IO Processor Plugins

IO Processor plugins allow custom pre- and post-processing of model inputs and outputs for **pooling models**. The idea is that users can pass a custom input to vLLM that is converted into one or more model prompts and fed to the model's `encode` method. One potential use-case is using vLLM for generating multimodal data — for example, feeding an image to vLLM and receiving a processed image as output.

When performing inference with IO Processor plugins, the prompt type is defined by the plugin and the same is valid for the final request output. vLLM does not perform any validation of input/output data; it is up to the plugin to ensure the correct data is being fed to the model and returned to the user.

!!! note "Pooling models only"
    IO Processor plugins currently support **pooling models only**. They can be triggered via the `encode` method in `LLM` and `AsyncLLM`, or in online serving mode via the `/pooling` endpoint.

## Architecture Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  IOProcessor.parse_data()  ← validates & converts input │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  IOProcessor.pre_process()  ← converts to PromptType(s) │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  vLLM Engine  ← runs model.encode() on each prompt      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  IOProcessor.post_process()  ← converts to plugin output│
└─────────────────────────────────────────────────────────┘
    │
    ▼
Custom Output (returned to user)
```

## The `IOProcessor` Interface

IO Processor plugins implement the [`IOProcessor`][vllm.plugins.io_processors.interface.IOProcessor] abstract base class:

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Generic, TypeVar

from vllm.config import VllmConfig
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Abstract interface for pre/post-processing of engine I/O."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__()
        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        """Validate and convert raw user input into the plugin's input type."""
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        """Merge user-supplied SamplingParams with plugin defaults."""
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        """Merge user-supplied PoolingParams with plugin defaults."""
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        """Convert plugin input into one or more vLLM PromptType objects."""
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        """Async version of pre_process (defaults to calling pre_process)."""
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        """Convert model outputs into the plugin's output type."""
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        """Async version of post_process (sorts outputs by ID before processing)."""
        sorted_output = sorted(
            [(i, item) async for i, item in model_output],
            key=lambda output: output[0],
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
```

### Method Reference

| Method | Required | Description |
|---|---|---|
| `__init__(vllm_config, renderer)` | Yes | Initialise the processor; store `vllm_config` for later use |
| `parse_data(data)` | Yes | Validate raw user input and convert to `IOProcessorInput` |
| `merge_sampling_params(params)` | No | Merge user `SamplingParams` with plugin defaults |
| `merge_pooling_params(params)` | No | Merge user `PoolingParams` with plugin defaults |
| `pre_process(prompt, request_id, **kwargs)` | **Yes** | Convert `IOProcessorInput` to one or more `PromptType` objects |
| `pre_process_async(...)` | No | Async variant of `pre_process`; defaults to calling `pre_process` |
| `post_process(model_output, request_id, **kwargs)` | **Yes** | Convert `PoolingRequestOutput` list to `IOProcessorOutput` |
| `post_process_async(...)` | No | Async variant of `post_process`; sorts outputs by ID before calling `post_process` |

## Writing an IO Processor Plugin

### Step 1 — Implement the `IOProcessor` class

```python
# my_io_plugin/processor.py
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer


@dataclass
class MyInput:
    text: str
    extra_context: str = ""


@dataclass
class MyOutput:
    embedding: list[float]
    metadata: dict


class MyIOProcessor(IOProcessor[MyInput, MyOutput]):

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        # Initialise any resources your processor needs
        self.max_length = vllm_config.model_config.max_model_len

    def parse_data(self, data: object) -> MyInput:
        """Validate and convert raw user input."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        if "text" not in data:
            raise ValueError("Input must contain 'text' key")
        return MyInput(
            text=data["text"],
            extra_context=data.get("extra_context", ""),
        )

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        """Always use the 'plugin' pooling task."""
        return PoolingParams(task="plugin")

    def pre_process(
        self,
        prompt: MyInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        """Convert MyInput into a vLLM prompt."""
        # Combine text with extra context
        full_text = f"{prompt.extra_context}\n{prompt.text}".strip()
        return {"prompt": full_text}

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> MyOutput:
        """Convert pooling outputs into MyOutput."""
        # Aggregate embeddings from all outputs
        all_embeddings = []
        for output in model_output:
            if output.outputs.data is not None:
                all_embeddings.extend(output.outputs.data.tolist())

        return MyOutput(
            embedding=all_embeddings,
            metadata={"request_id": request_id, "num_outputs": len(model_output)},
        )
```

### Step 2 — Create the registration function

```python
# my_io_plugin/__init__.py

def register():
    """Return the fully qualified class name of the IOProcessor."""
    return "my_io_plugin.processor:MyIOProcessor"
```

### Step 3 — Declare the entry point

=== "pyproject.toml"

    ```toml
    [project.entry-points."vllm.io_processor_plugins"]
    my_io_processor = "my_io_plugin:register"
    ```

=== "setup.py"

    ```python
    from setuptools import setup

    setup(
        name="my_io_plugin",
        version="0.1",
        packages=["my_io_plugin"],
        entry_points={
            "vllm.io_processor_plugins": [
                "my_io_processor = my_io_plugin:register"
            ]
        },
    )
    ```

### Step 4 — Install the plugin

```bash
pip install -e .
```

After installation, verify the plugin is discoverable:

```bash
python -c "
from importlib.metadata import entry_points
eps = entry_points(group='vllm.io_processor_plugins')
for ep in eps:
    print(ep.name, '->', ep.value)
"
```

## Using an IO Processor Plugin

IO Processor plugins are loaded at engine startup. There are two methods for specifying the plugin to load:

### Method 1 — Via `EngineArgs` / CLI

**Offline inference (`LLM`):**

```python
from vllm import LLM

llm = LLM(
    model="my-org/my-pooling-model",
    io_processor_plugin="my_io_processor",  # plugin name from entry_points
    skip_tokenizer_init=True,
)

result = llm.encode({"text": "Hello, world!", "extra_context": "Greeting"})
output = result[0].outputs
print(output)
```

**Online serving:**

```bash
vllm serve my-org/my-pooling-model \
    --io-processor-plugin my_io_processor \
    --skip-tokenizer-init
```

Then call the `/pooling` endpoint:

```bash
curl http://localhost:8000/pooling \
    -H "Content-Type: application/json" \
    -d '{
        "model": "my-org/my-pooling-model",
        "data": {
            "text": "Hello, world!",
            "extra_context": "Greeting"
        }
    }'
```

### Method 2 — Via Model HF Configuration

Add an `io_processor_plugin` field to the model's `config.json`:

```json
{
  "model_type": "my_model",
  "io_processor_plugin": "my_io_processor",
  ...
}
```

When this field is present, vLLM automatically loads the named plugin at startup without any additional CLI arguments.

!!! note "Priority"
    Setting `io_processor_plugin` via `EngineArgs` (or `--io-processor-plugin` CLI flag) **overrides** any plugin name specified in the model's `config.json`.

## Async IO Processing

For I/O-bound pre-processing (e.g., downloading images, calling external APIs), override `pre_process_async` instead of `pre_process`:

```python
import asyncio
import aiohttp

class MyAsyncIOProcessor(IOProcessor[MyInput, MyOutput]):

    async def pre_process_async(
        self,
        prompt: MyInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        """Fetch data asynchronously before processing."""
        async with aiohttp.ClientSession() as session:
            async with session.get(prompt.url) as resp:
                data = await resp.read()

        # Process the fetched data into a prompt
        return {"prompt_token_ids": self._encode(data)}

    def pre_process(self, prompt, request_id=None, **kwargs):
        # Synchronous fallback (required by abstract base)
        return asyncio.run(self.pre_process_async(prompt, request_id, **kwargs))
```

## Real-World Example

An example implementation of a plugin that enables generating GeoTIFF images with the PrithviGeospatialMAE model is available in the [TerraTorch repository](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation).

**Offline inference example:**

```python
# examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py
import base64
import os
import torch
from vllm import LLM

torch.set_default_dtype(torch.float16)

img_data = dict(
    data="https://example.com/image.tiff",
    data_format="url",
    image_format="tiff",
    out_data_format="b64_json",
)

llm = LLM(
    model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    skip_tokenizer_init=True,
    trust_remote_code=True,
    enforce_eager=True,
    max_num_seqs=32,
    io_processor_plugin="terratorch_segmentation",
    model_impl="terratorch",
    enable_mm_embeds=True,
)

pooler_output = llm.encode({"data": img_data}, pooling_task="plugin")
output = pooler_output[0].outputs

decoded_data = base64.b64decode(output.data)
with open("offline_prediction.tiff", "wb") as f:
    f.write(decoded_data)
```

**Online serving example:**

```bash
# Start the server
vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
    --skip-tokenizer-init \
    --enforce-eager \
    --io-processor-plugin terratorch_segmentation \
    --enable-mm-embeds

# Send a request
curl http://localhost:8000/pooling \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "data": {
            "data": "https://example.com/image.tiff",
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json"
        }
    }'
```

## Deprecation Notices

!!! warning "Deprecated API"
    - `parse_request` has been renamed to `parse_data`. The old name will be removed in **v0.19**.
    - `validate_or_generate_params` has been split into `merge_sampling_params` and `merge_pooling_params`. The old name will be removed in **v0.19**.
    - The `renderer` argument in `IOProcessor.__init__` will be **required** in **v0.18**. Update your plugin to accept it even if you don't use it.

## See Also

- [Plugin System Overview](plugin_system.md)
- [LoRA Resolver Plugins](lora_resolver_plugins.md)
- [Pooling Parameters API](../api/pooling_params.md)
- [Offline Inference Usage](../usage/offline_inference.md)
