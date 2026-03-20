# Multimodal Models

vLLM supports a wide range of multimodal models that can process images, video, audio, and combinations thereof alongside text. This guide covers how to use multimodal models for inference, configure multimodal inputs, and implement new multimodal architectures.

---

## Supported Modalities

| Modality | Input Types | Example Models |
|---|---|---|
| **Image** | PIL Image, NumPy array, PyTorch tensor, URL, base64 | LLaVA, Qwen2-VL, Gemma3, PaliGemma |
| **Video** | List of frames, NumPy array, PyTorch tensor | LLaVA-NeXT-Video, Qwen2.5-VL |
| **Audio** | NumPy array, `(audio, sample_rate)` tuple, PyTorch tensor | Qwen2-Audio, Ultravox, Whisper |
| **Image + Audio** | Combined image and audio inputs | MiniCPM-o, Phi-4 Multimodal |
| **Image + Video + Audio** | All modalities | Qwen2.5-Omni, Qwen3-Omni |

---

## Quick Start

### Image Input

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")

# Using a PIL Image
image = Image.open("photo.jpg")
outputs = llm.generate([{
    "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n",
    "multi_modal_data": {"image": image},
}])
print(outputs[0].outputs[0].text)
```

### Image from URL

```python
outputs = llm.generate([{
    "prompt": "...",
    "multi_modal_data": {
        "image": "https://example.com/photo.jpg"
    },
}])
```

### Multiple Images

```python
from PIL import Image

images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
outputs = llm.generate([{
    "prompt": "Compare these two images: ...",
    "multi_modal_data": {"image": images},
}])
```

### Audio Input

```python
import numpy as np

llm = LLM(model="Qwen/Qwen2-Audio-7B-Instruct")

# Audio as (array, sample_rate) tuple
audio_array = np.zeros(16000, dtype=np.float32)  # 1 second of silence
outputs = llm.generate([{
    "prompt": "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this audio.",
    "multi_modal_data": {"audio": (audio_array, 16000)},
}])
```

### Video Input

```python
from PIL import Image

llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf")

# Video as list of PIL frames
frames = [Image.open(f"frame_{i:04d}.jpg") for i in range(16)]
outputs = llm.generate([{
    "prompt": "Describe what happens in this video.",
    "multi_modal_data": {"video": frames},
}])
```

---

## Using the OpenAI-Compatible API

vLLM's OpenAI-compatible server supports multimodal inputs via the Chat Completions API.

### Image via URL

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "text", "text": "What is in this image?"},
        ],
    }],
)
print(response.choices[0].message.content)
```

### Image via Base64

```python
import base64

with open("photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }],
)
```

### Audio Input

```python
import base64

with open("audio.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            },
            {"type": "text", "text": "Transcribe this audio."},
        ],
    }],
)
```

---

## Configuration

### Limiting Multimodal Inputs

Control the maximum number of multimodal items per request using `--limit-mm-per-prompt`:

```bash
# Allow up to 4 images per request
vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt image=4

# Allow images and audio
vllm serve Qwen/Qwen2.5-Omni-7B --limit-mm-per-prompt image=4,audio=2
```

In Python:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    limit_mm_per_prompt={"image": 4},
)
```

### Disabling Multimodal Processing

To run a multimodal model in text-only mode (ignoring all multimodal inputs), set all limits to 0:

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct --limit-mm-per-prompt image=0
```

### Pre-computed Embeddings

You can pass pre-computed multimodal embeddings directly, bypassing the model's encoder:

```python
import torch

# Pre-computed image embeddings (e.g., from a separate vision encoder)
image_embeds = torch.randn(1, 256, 4096)  # [batch, tokens, hidden_dim]

outputs = llm.generate([{
    "prompt": "...",
    "multi_modal_data": {"image": image_embeds},
}])
```

---

## Multimodal Input Types

### `MultiModalDataDict`

The `multi_modal_data` field accepts a `MultiModalDataDict` — a dictionary mapping modality names to input data:

```python
from vllm.multimodal import MultiModalDataDict

multi_modal_data: MultiModalDataDict = {
    "image": image_or_list_of_images,
    "video": video_or_list_of_videos,
    "audio": audio_or_list_of_audio,
}
```

### Supported Input Formats

**Images (`ImageItem`):**
- `PIL.Image.Image` — standard PIL image
- `np.ndarray` — NumPy array (H, W, C) or (H, W)
- `torch.Tensor` — PyTorch tensor
- `str` — URL or file path
- `bytes` — raw image bytes
- `torch.Tensor` (3D) — pre-computed image embeddings

**Videos (`VideoItem`):**
- `list[PIL.Image.Image]` — list of frames
- `np.ndarray` — (T, H, W, C) array
- `torch.Tensor` — (T, H, W, C) tensor
- `tuple[frames, metadata]` — frames with `VideoMetadata`

**Audio (`AudioItem`):**
- `np.ndarray` — 1D float array (mono audio)
- `tuple[np.ndarray, float]` — `(audio_array, sample_rate)` — resampled if needed
- `torch.Tensor` — 1D float tensor
- `torch.Tensor` (3D) — pre-computed audio embeddings

---

## Model-Specific Notes

### LLaVA Family

LLaVA models use `<image>` tokens as placeholders in the prompt:

```python
llm = LLM(model="llava-hf/llava-1.5-7b-hf")
outputs = llm.generate([{
    "prompt": "USER: <image>\nWhat is in this image? ASSISTANT:",
    "multi_modal_data": {"image": image},
}])
```

### Qwen2-VL / Qwen2.5-VL

Qwen2-VL uses vision tokens with special markers:

```python
llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
# The processor automatically handles token placement
outputs = llm.generate([{
    "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n",
    "multi_modal_data": {"image": image},
}])
```

### Whisper (ASR)

Whisper is an encoder-decoder model for automatic speech recognition:

```python
llm = LLM(model="openai/whisper-large-v3")
outputs = llm.generate([{
    "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    "multi_modal_data": {"audio": (audio_array, 16000)},
}])
```

### PaliGemma

PaliGemma uses a prefix prompt format:

```python
llm = LLM(model="google/paligemma-3b-pt-224")
outputs = llm.generate([{
    "prompt": "caption en\n",
    "multi_modal_data": {"image": image},
}])
```

---

## Implementing a New Multimodal Model

### 1. Implement `SupportsMultiModal`

Your model class must implement the `SupportsMultiModal` protocol:

```python
from typing import ClassVar, Literal
from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_processor(
    MyModelProcessor,
    info=MyModelProcessingInfo,
    dummy_inputs=MyModelDummyInputsBuilder,
)
class MyModelForConditionalGeneration(nn.Module, SupportsMultiModal):
    supports_multimodal: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # Initialize language model backbone
        self.language_model = MyLanguageModel(vllm_config=vllm_config)
        # Initialize vision encoder
        self.vision_encoder = MyVisionEncoder(vllm_config=vllm_config)
        # Initialize vision-language projector
        self.projector = nn.Linear(vision_dim, language_dim)

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> MultiModalEmbeddings:
        """Process visual inputs and return embeddings."""
        vision_features = self.vision_encoder(pixel_values)
        projected = self.projector(vision_features)
        return projected  # shape: [batch, num_tokens, hidden_dim]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)
```

### 2. Implement `BaseProcessingInfo`

Define what modalities your model supports and their limits:

```python
from vllm.multimodal.processing import BaseProcessingInfo

class MyModelProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> dict[str, int]:
        return {"image": 1}  # Support 1 image per request

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: dict[str, int],
    ) -> dict[str, int]:
        return {"image": 256}  # Each image uses 256 tokens
```

### 3. Implement `BaseMultiModalProcessor`

Define how raw multimodal data is converted to model inputs:

```python
from vllm.multimodal.processing import BaseMultiModalProcessor

class MyModelProcessor(BaseMultiModalProcessor[MyModelProcessingInfo]):
    def _get_mm_fields_config(self, hf_inputs, mm_processor_kwargs):
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        # Return token replacements for multimodal placeholders
        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=self._get_image_tokens(),
            )
        ]
```

### 4. Implement `BaseDummyInputsBuilder`

Provide dummy inputs for memory profiling:

```python
from vllm.multimodal.processing import BaseDummyInputsBuilder

class MyModelDummyInputsBuilder(BaseDummyInputsBuilder[MyModelProcessingInfo]):
    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options=None):
        from PIL import Image
        num_images = mm_counts.get("image", 0)
        mm_data = {"image": [Image.new("RGB", (224, 224)) for _ in range(num_images)]}
        return ProcessorInputs(
            prompt_text="<image>" * num_images,
            mm_data=mm_data,
        )
```

---

## Multimodal Caching

vLLM supports caching of processed multimodal inputs to avoid redundant processing when the same image/audio appears in multiple requests.

### Enabling Multimodal Caching

```python
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    mm_processor_kwargs={"cache_size": 256},  # Cache up to 256 processed items
)
```

### How Caching Works

1. Each multimodal item is hashed using its content.
2. If the hash is found in the cache, the pre-processed tensors are reused.
3. Cache entries are stored in shared memory for cross-process access.

---

## Tensor Parallelism for Multimodal Encoders

For large vision encoders, vLLM supports tensor parallelism across the encoder:

```bash
vllm serve Qwen/Qwen2-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data
```

The `--mm-encoder-tp-mode data` flag enables data-parallel processing of the vision encoder across TP ranks.

---

## Performance Tips

1. **Batch multimodal requests** — group requests with similar image sizes to minimize padding overhead.
2. **Use `limit-mm-per-prompt`** — set appropriate limits to avoid over-allocating KV cache for multimodal tokens.
3. **Pre-resize images** — resize images to the model's expected resolution before passing them to vLLM.
4. **Enable multimodal caching** — if the same images appear in multiple requests, enable the processor cache.
5. **Use FP16/BF16** — multimodal models benefit significantly from reduced precision.

---

## Related

- [Supported Models](supported_models.md) — full list of multimodal architectures
- [Adding a New Model](adding_model.md) — implementing a new multimodal model
- [Multimodal Processing Design](../design/mm_processing.md) — internals of multimodal input handling
- [Engine Arguments](../configuration/engine_args.md) — `--limit-mm-per-prompt` and related flags
