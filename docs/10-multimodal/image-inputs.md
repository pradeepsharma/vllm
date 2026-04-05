# Image Inputs

vLLM supports a rich variety of image input formats for vision-language models. Images can
be provided as PIL objects, HTTP/HTTPS URLs, local file URLs, or base64-encoded data — both
in offline inference via the `LLM` class and in online serving via the OpenAI-compatible API.

## Image Type Aliases

The core type definitions live in `vllm/multimodal/inputs.py`:

```python
HfImageItem: TypeAlias = Union[Image, np.ndarray, torch.Tensor]
"""A single image item compatible with HuggingFace ImageProcessor."""

ImageItem: TypeAlias = Union[HfImageItem, torch.Tensor, MediaWithBytes[HfImageItem]]
"""
A single image item. Can be:
- A PIL Image, numpy array, or torch.Tensor → passed to HF ImageProcessor
- A 3-D tensor or batch of 2-D tensors → treated as pre-computed embeddings,
  passed directly to the model without HF processing
"""
```

The `ModalityData[ImageItem]` type allows either a single item or a list:

```python
ModalityData: TypeAlias = _T | list[_T | None] | None
```

## Offline Inference

### PIL Image

The simplest way to pass an image is as a `PIL.Image.Image` object:

```python
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

image = Image.open("photo.jpg")

outputs = llm.generate(
    {
        "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
        "multi_modal_data": {"image": image},
    },
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

### Multiple Images

Pass a list of images for models that support multi-image input:

```python
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 2},
)

image1 = Image.open("duck.jpg")
image2 = Image.open("lion.jpg")

outputs = llm.generate(
    {
        "prompt": "USER: <image><image>\nWhat animals are in these images?\nASSISTANT:",
        "multi_modal_data": {"image": [image1, image2]},
    },
    sampling_params=SamplingParams(max_tokens=256),
)
```

### Fetching Images from URLs

Use the `fetch_image` utility to download an image before passing it to the model:

```python
from vllm.multimodal.utils import fetch_image

image = fetch_image("https://example.com/photo.jpg")
# Returns a PIL.Image.Image
```

> **Note**: `fetch_image` is intended for user code only. Never call it from the online
> server — use the `image_url` content type in chat messages instead.

### Pre-computed Image Embeddings

If you have pre-computed image embeddings (e.g., from a separate encoder), pass them as
a `torch.Tensor`. The tensor is passed directly to the model, bypassing the HF image
processor:

```python
import torch

# Shape: (num_patches, hidden_size) or (batch, hidden_size)
image_embeds = torch.load("image_embeddings.pt")

outputs = llm.generate(
    {
        "prompt": "USER: <image>\nDescribe this.\nASSISTANT:",
        "multi_modal_data": {"image": image_embeds},
    },
    sampling_params=sampling_params,
)
```

> **Warning**: Enable `--enable-mm-embeds` when using pre-computed embeddings. The engine
> may crash if embeddings with incorrect shapes are passed. Only enable this for trusted users.

## Online Serving (OpenAI-Compatible API)

The online API follows the OpenAI vision format using `image_url` content blocks.

### Image from HTTP URL

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
                           "vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                },
            },
        ],
    }],
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

### Image from Local File

Start the server with `--allowed-local-media-path` to enable local file access:

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
    --allowed-local-media-path /path/to/images
```

```python
response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "file:///path/to/images/photo.jpg"},
            },
        ],
    }],
    max_completion_tokens=256,
)
```

### Image from Base64

Encode the image as a base64 data URL:

```python
import base64

with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ],
    }],
    max_completion_tokens=256,
)
```

You can also use the `encode_image_url` utility to generate the data URL:

```python
from PIL import Image
from vllm.multimodal.utils import encode_image_url

image = Image.open("photo.jpg")
data_url = encode_image_url(image, image_mode="RGB", format="PNG")
# Returns: "data:image/png;base64,..."
```

### Multiple Images

```python
response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What animals are in these images?"},
            {"type": "image_url", "image_url": {"url": duck_url}},
            {"type": "image_url", "image_url": {"url": lion_url}},
        ],
    }],
    max_completion_tokens=256,
)
```

## Image Processing Internals

### ImageMediaIO

The `ImageMediaIO` class (in `vllm/multimodal/media/image.py`) handles loading images
from various sources:

```python
class ImageMediaIO(MediaIO[Image.Image]):
    def __init__(self, image_mode: str = "RGB", **kwargs) -> None:
        ...

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]: ...
    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]: ...
    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]: ...
    def encode_base64(self, media: Image.Image, *, image_format: str = "PNG") -> str: ...
```

Key behaviors:
- Images are loaded and converted to the target `image_mode` (default: `"RGB"`).
- RGBA images are composited onto a white background by default. Override with
  `rgba_background_color` in `--media-io-kwargs`.
- The original bytes are preserved in `MediaWithBytes` for efficient hashing.

### Image Utilities

`vllm/multimodal/image.py` provides helper functions:

```python
def rescale_image_size(image: Image.Image, size_factor: float, transpose: int = -1) -> Image.Image:
    """Rescale image dimensions by a constant factor."""

def rgba_to_rgb(image: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
    """Convert RGBA to RGB with filled background color."""

def convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    """Convert image to target color mode."""
```

## Controlling Image Limits

Use `--limit-mm-per-prompt` to set the maximum number of images per prompt:

```bash
# Allow up to 4 images per prompt
vllm serve microsoft/Phi-3.5-vision-instruct \
    --limit-mm-per-prompt '{"image": 4}'
```

Or with `EngineArgs`:

```python
from vllm import LLM, EngineArgs

llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    limit_mm_per_prompt={"image": 4},
)
```

You can also specify image dimensions for dummy profiling:

```bash
vllm serve my-vision-model \
    --limit-mm-per-prompt '{"image": {"count": 4, "width": 512, "height": 512}}'
```

## Image Hashing and Caching

vLLM uses `MultiModalHasher` to compute a content hash for each image. This hash is used
for:
- **Processor cache**: Avoid re-running the HF image processor for identical images.
- **Prefix caching**: Reuse KV cache entries for repeated images.

You can provide explicit UUIDs to override the hash:

```python
outputs = llm.generate(
    {
        "prompt": "USER: <image>\nDescribe this.\nASSISTANT:",
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": "my-unique-image-id"},
    },
    sampling_params=sampling_params,
)
```

See [Multimodal Cache](cache.md) for details on the caching architecture.

## Related Pages

- [Multimodal Registry](registry.md) — how models register image processors
- [Multimodal Cache](cache.md) — caching encoder outputs
- [MultiModalConfig](config.md) — full configuration reference
- [Code Examples](examples.md) — complete working examples
