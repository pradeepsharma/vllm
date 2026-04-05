# Audio Inputs

vLLM supports audio inputs for speech-to-text models and audio-language models. Audio can
be provided as raw waveform arrays, `(waveform, sample_rate)` tuples, HTTP/HTTPS URLs, or
base64-encoded data.

## Audio Type Aliases

Defined in `vllm/multimodal/inputs.py`:

```python
HfAudioItem: TypeAlias = Union[list[float], np.ndarray, torch.Tensor]
"""A single audio item compatible with HuggingFace AudioProcessor."""

AudioItem: TypeAlias = Union[
    HfAudioItem,
    tuple[np.ndarray, float],   # (waveform, sampling_rate) — resampled automatically
    torch.Tensor,               # Pre-computed audio embeddings
]
```

The `tuple[np.ndarray, float]` form is the most common: the waveform is a 1-D float32
array and the float is the original sample rate. vLLM resamples to the model's expected
rate before processing.

## Offline Inference

### Waveform + Sample Rate

```python
import numpy as np
from vllm import LLM, SamplingParams

llm = LLM(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    max_model_len=4096,
    limit_mm_per_prompt={"audio": 1},
)

# Load audio (e.g., with librosa or soundfile)
import librosa
waveform, sr = librosa.load("speech.wav", sr=None)  # sr=None preserves original rate

outputs = llm.generate(
    {
        "prompt": "<|audio|>\nTranscribe the audio.",
        "multi_modal_data": {"audio": (waveform, sr)},
    },
    sampling_params=SamplingParams(max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

### Multiple Audio Items

```python
import librosa

audio1, sr1 = librosa.load("speech1.wav", sr=None)
audio2, sr2 = librosa.load("speech2.wav", sr=None)

outputs = llm.generate(
    {
        "prompt": "<|audio|><|audio|>\nAre these two recordings the same?",
        "multi_modal_data": {"audio": [(audio1, sr1), (audio2, sr2)]},
    },
    sampling_params=SamplingParams(max_tokens=128),
)
```

### Fetching Audio from a URL

```python
from vllm.multimodal.utils import fetch_audio

waveform, sr = fetch_audio("https://example.com/speech.wav")
# Returns (np.ndarray, float) — the waveform and sample rate
```

> **Note**: `fetch_audio` is for user code only. Use `audio_url` content blocks in the
> online API instead.

### Whisper (Speech-to-Text)

Whisper is an encoder-decoder model. Use the `<|startoftranscript|>` prompt:

```python
llm = LLM(
    model="openai/whisper-large-v3-turbo",
    max_model_len=448,
    max_num_seqs=5,
    limit_mm_per_prompt={"audio": 1},
)

waveform, sr = fetch_audio("https://example.com/speech.wav")

outputs = llm.generate(
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {"audio": (waveform, sr)},
    },
    sampling_params=SamplingParams(max_tokens=200),
)
print(outputs[0].outputs[0].text)
```

## Online Serving (OpenAI-Compatible API)

### `input_audio` Content Block (OpenAI-Compatible)

The `input_audio` format matches the OpenAI Realtime API schema:

```python
from openai import OpenAI
import base64

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

with open("speech.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is being said in this audio?"},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav",   # Any format supported by librosa
                },
            },
        ],
    }],
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

### `audio_url` Content Block

```python
from vllm.assets.audio import AudioAsset

audio_url = AudioAsset("winning_call").url

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this audio?"},
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            },
        ],
    }],
    max_completion_tokens=256,
)
```

### Base64 Audio URL

```python
import base64

with open("speech.ogg", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this."},
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/ogg;base64,{audio_b64}"},
            },
        ],
    }],
    max_completion_tokens=256,
)
```

## Audio Processing Internals

### AudioMediaIO

`vllm/multimodal/media/audio.py` provides the `AudioMediaIO` class:

```python
class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return librosa.load(BytesIO(data), sr=None)

    def load_base64(self, media_type: str, data: str) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return librosa.load(filepath, sr=None)
```

`librosa` is used as the default audio loader. It supports WAV, MP3, OGG, FLAC, and any
other format supported by `soundfile` / `libsndfile`.

### AudioSpec and Channel Normalization

`vllm/multimodal/audio.py` defines `AudioSpec` for normalizing multi-channel audio:

```python
@dataclass
class AudioSpec:
    target_channels: int | None = 1
    channel_reduction: ChannelReduction = ChannelReduction.MEAN
```

`ChannelReduction` options:

| Value | Behavior |
|-------|----------|
| `MEAN` | Average across channels (default) |
| `FIRST` | Take the first channel only |
| `MAX` | Take the maximum value across channels |
| `SUM` | Sum across channels |

Pre-defined specs:

```python
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)  # No normalization
```

### AudioResampler

When the input audio has a different sample rate than the model expects, `AudioResampler`
handles the conversion:

```python
class AudioResampler:
    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["librosa", "scipy"] = "librosa",
    ): ...

    def resample(
        self,
        audio: npt.NDArray[np.floating],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]: ...
```

The `(waveform, sample_rate)` tuple form of `AudioItem` triggers automatic resampling
inside the model's processor.

### Audio Chunking

For long audio files, `split_audio` intelligently splits at low-energy regions to minimize
cutting through speech:

```python
from vllm.multimodal.audio import split_audio

chunks = split_audio(
    audio_data=waveform,
    sample_rate=16000,
    max_clip_duration_s=30.0,
    overlap_duration_s=1.0,
    min_energy_window_size=1600,
)
# Returns a list of numpy arrays, each at most 30 seconds
```

The algorithm:
1. Divides audio into chunks of at most `max_clip_duration_s` seconds.
2. For each split point, searches a `overlap_duration_s`-wide window for the quietest
   region (lowest RMS energy).
3. Splits at the quietest point to minimize artifacts.

### Extracting Audio from Video

When a model needs audio from a video file, `extract_audio_from_video_bytes` uses PyAV
(FFmpeg bindings) to extract the audio track without spawning a subprocess:

```python
from vllm.multimodal.media.audio import extract_audio_from_video_bytes

with open("video.mp4", "rb") as f:
    video_bytes = f.read()

waveform, native_sr = extract_audio_from_video_bytes(video_bytes)
```

> **Note**: PyAV wraps FFmpeg's C libraries in-process, which is critical to avoid
> crashing CUDA-active vLLM worker processes.

## Encoding Audio for API Calls

```python
import numpy as np
from vllm.multimodal.utils import encode_audio_base64, encode_audio_url

waveform = np.random.randn(16000).astype(np.float32)
sr = 16000

# Base64 string
b64 = encode_audio_base64(waveform, sr, format="WAV")

# Data URL
url = encode_audio_url(waveform, sr, format="WAV")
# Returns: "data:audio/wav;base64,..."
```

## Controlling Audio Limits

```bash
# Allow up to 2 audio items per prompt
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
    --limit-mm-per-prompt '{"audio": 2}'
```

```python
llm = LLM(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    limit_mm_per_prompt={"audio": 2},
)
```

Dummy audio options for profiling:

```bash
vllm serve my-audio-model \
    --limit-mm-per-prompt '{"audio": {"count": 2, "length": 16000}}'
```

## Supported Audio Models

| Model | Notes |
|-------|-------|
| `openai/whisper-*` | Speech-to-text, encoder-decoder |
| `fixie-ai/ultravox-*` | Audio-language model |
| `Qwen/Qwen2-Audio-*` | Multi-turn audio chat |
| `Qwen/Qwen2.5-Omni-*` | Omni model (audio + vision) |
| `nvidia/audio-flamingo-*` | Audio understanding |
| `microsoft/phi-4-multimodal-*` | Audio + vision |

## Related Pages

- [Multimodal Registry](registry.md) — how models register audio processors
- [Multimodal Cache](cache.md) — caching encoder outputs
- [MultiModalConfig](config.md) — full configuration reference
- [Code Examples](examples.md) — complete working examples
