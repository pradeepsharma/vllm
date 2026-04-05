# Video Inputs

vLLM supports video inputs for vision-language models that can process temporal sequences
of frames. Videos can be provided as frame arrays, HTTP/HTTPS URLs, local file URLs, or
base64-encoded data.

## Video Type Aliases

Defined in `vllm/multimodal/inputs.py`:

```python
HfVideoItem: TypeAlias = Union[
    list[Image],          # List of PIL frames
    np.ndarray,           # (T, H, W, C) array
    torch.Tensor,         # (T, H, W, C) tensor
    list[np.ndarray],     # List of frame arrays
    list[torch.Tensor],   # List of frame tensors
]

VideoItem: TypeAlias = Union[
    HfVideoItem,
    torch.Tensor,                        # Pre-computed video embeddings
    tuple[HfVideoItem, dict[str, Any]],  # Frames + HF VideoMetadata
]
```

The `tuple[HfVideoItem, dict[str, Any]]` form passes both frames and metadata (e.g.,
`total_num_frames`, `fps`, `duration`) to the HuggingFace `VideoProcessor`.

## Dependencies

Video loading requires `opencv-python-headless`:

```bash
pip install opencv-python-headless
```

> **Note**: Use `opencv-python-headless` (not `opencv-python`) in server environments to
> avoid GUI dependencies that can cause import errors.

## Video Backends

vLLM supports multiple video loading backends, controlled by the `VLLM_VIDEO_LOADER_BACKEND`
environment variable (default: `"opencv"`):

| Backend | Key | Notes |
|---------|-----|-------|
| OpenCV (uniform) | `opencv` | Uniform frame sampling, default |
| OpenCV (dynamic) | `opencv_dynamic` | FPS-based sampling with max duration |
| Decord | `decord` | Fast GPU-accelerated decoding (requires `decord`) |
| TorchCodec | `torchcodec` | PyTorch-native video decoding |

```bash
# Use dynamic FPS-based sampling
export VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic

# Or per-request via media_io_kwargs
vllm serve my-video-model \
    --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
```

## Offline Inference

### Video from URL

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_video

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    max_model_len=8192,
    limit_mm_per_prompt={"video": 1},
)

frames, metadata = fetch_video(
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
    video_io_kwargs={"num_frames": 32},
)

outputs = llm.generate(
    {
        "prompt": "USER: <video>\nDescribe what happens in this video.\nASSISTANT:",
        "multi_modal_data": {"video": (frames, metadata)},
    },
    sampling_params=SamplingParams(max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

### Video from Local File

```python
from vllm.multimodal.utils import fetch_video

frames, metadata = fetch_video(
    "file:///path/to/video.mp4",
    video_io_kwargs={"num_frames": 16},
)
```

> **Note**: `fetch_video` is for user code only. Use `video_url` content blocks in the
> online API instead.

### Controlling Frame Count

The number of frames to extract is controlled by `num_frames` in `media_io_kwargs`:

```python
frames, metadata = fetch_video(
    video_url,
    video_io_kwargs={"num_frames": 32},  # Extract 32 uniformly-sampled frames
)
```

Or globally via `--media-io-kwargs`:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --media-io-kwargs '{"video": {"num_frames": 32}}'
```

### FPS-Based Sampling

With the `opencv_dynamic` backend, use `fps` instead of `num_frames`:

```bash
export VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic

vllm serve my-video-model \
    --media-io-kwargs '{"video": {"fps": 2, "max_duration": 60}}'
```

## Online Serving (OpenAI-Compatible API)

### Video from HTTP URL

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {
                "type": "video_url",
                "video_url": {"url": video_url},
            },
        ],
    }],
    max_completion_tokens=512,
)
print(response.choices[0].message.content)
```

### Video from Base64

```python
import base64

with open("video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video."},
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
        ],
    }],
    max_completion_tokens=512,
)
```

## Video Processing Internals

### VideoMediaIO

`vllm/multimodal/media/video.py` provides `VideoMediaIO`:

```python
class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None: ...

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def load_base64(self, media_type: str, data: str) -> tuple[npt.NDArray, dict[str, Any]]: ...
    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]: ...
```

The returned tuple contains:
- `frames`: `np.ndarray` of shape `(T, H, W, 3)` in RGB format
- `metadata`: dict with `total_num_frames`, `fps`, `duration`, `video_backend`, `frames_indices`

### OpenCV Video Backend

The `OpenCVVideoBackend` (registered as `"opencv"`) uses OpenCV for frame extraction:

```python
@VIDEO_LOADER_REGISTRY.register("opencv")
class OpenCVVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]: ...
```

Frame sampling strategy:
- If `num_frames > 0`: uniformly sample `num_frames` frames from the video.
- If `fps > 0`: sample at the target FPS rate.
- The minimum of the two constraints is used.

### Frame Recovery

When frames fail to load (corrupted video), the `frame_recovery=True` option enables
forward-scan recovery: the next successfully grabbed frame is used to substitute for
the failed one:

```python
frames, metadata = fetch_video(
    video_url,
    video_io_kwargs={"frame_recovery": True},
)
```

### Video Utilities

`vllm/multimodal/video.py` provides utility functions:

```python
def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    """Resize all frames to (height, width)."""

def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    """Scale all frames by a constant factor."""

def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray:
    """Uniformly sample num_frames from the frame array."""
```

### Encoding Video for API Calls

```python
import numpy as np
from vllm.multimodal.utils import encode_video_url

# frames: np.ndarray of shape (T, H, W, 3)
frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
url = encode_video_url(frames, format="JPEG")
# Returns: "data:video/jpeg;base64,..."
```

## Controlling Video Limits

```bash
# Allow up to 2 videos per prompt, with 32 frames each
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --limit-mm-per-prompt '{"video": {"count": 2, "num_frames": 32, "width": 512, "height": 512}}'
```

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    limit_mm_per_prompt={"video": 2},
)
```

## Video Token Pruning (EVS)

For long videos, vLLM supports Efficient Video Sampling (EVS) to prune redundant tokens
based on inter-frame similarity. See [EVS Documentation](evs.md) for details.

```bash
# Prune 30% of video tokens
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --video-pruning-rate 0.3
```

## Supported Video Models

| Model | Notes |
|-------|-------|
| `Qwen/Qwen2.5-VL-*` | Vision-language with video support |
| `Qwen/Qwen2-VL-*` | Vision-language with video support |
| `Qwen/Qwen2.5-Omni-*` | Omni model (audio + video) |
| `microsoft/phi-4-multimodal-*` | Video + audio + image |
| `google/gemma-3n-*` | Video understanding |
| `nvidia/NVILA-*` | Video-language model |

## Related Pages

- [EVS — Efficient Video Sampling](evs.md) — video token pruning
- [Multimodal Registry](registry.md) — how models register video processors
- [Multimodal Cache](cache.md) — caching encoder outputs
- [MultiModalConfig](config.md) — full configuration reference
