# Multimodal Code Examples

This page provides complete, working code examples for common multimodal use cases in vLLM.

## Image in Chat Completion (Online API)

### Single Image from URL

Start the server:

```bash
vllm serve llava-hf/llava-1.5-7b-hf
```

Send a request:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

image_url = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

### Single Image from Base64

```python
import base64
import requests
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Fetch and encode image
image_url = "https://example.com/photo.jpg"
response = requests.get(image_url)
image_b64 = base64.b64encode(response.content).decode("utf-8")

chat_response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ],
    }],
    max_completion_tokens=512,
)
print(chat_response.choices[0].message.content)
```

### Multiple Images

```bash
# Start server with multi-image support
vllm serve microsoft/Phi-3.5-vision-instruct \
    --trust-remote-code \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 2}'
```

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

duck_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
lion_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg"

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
print(response.choices[0].message.content)
```

### Local File Image

```bash
# Start server with local file access
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

---

## Image in Offline Inference

### PIL Image

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

### Image from URL (Offline)

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

image = fetch_image(
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)

outputs = llm.generate(
    {
        "prompt": "USER: <image>\nDescribe the scene.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    },
    sampling_params=SamplingParams(max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

### Batch Image Inference

```python
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

images = [Image.open(f"image_{i}.jpg") for i in range(4)]
prompts = [
    {
        "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
        "multi_modal_data": {"image": img},
    }
    for img in images
]

outputs = llm.generate(prompts, sampling_params=sampling_params)
for i, output in enumerate(outputs):
    print(f"Image {i}: {output.outputs[0].text}")
```

---

## Audio Transcription (Online API)

### Whisper Speech-to-Text

```bash
vllm serve openai/whisper-large-v3-turbo \
    --max-model-len 448 \
    --limit-mm-per-prompt '{"audio": 1}'
```

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

with open("speech.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="openai/whisper-large-v3-turbo",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav",
                },
            },
        ],
    }],
    max_completion_tokens=200,
)
print(response.choices[0].message.content)
```

### Audio from URL

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

audio_url = "https://example.com/speech.wav"

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is being said?"},
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            },
        ],
    }],
    max_completion_tokens=256,
)
print(response.choices[0].message.content)
```

---

## Audio Transcription (Offline Inference)

### Whisper Offline

```python
import librosa
from vllm import LLM, SamplingParams

llm = LLM(
    model="openai/whisper-large-v3-turbo",
    max_model_len=448,
    max_num_seqs=5,
    limit_mm_per_prompt={"audio": 1},
)

# Load audio
waveform, sr = librosa.load("speech.wav", sr=None)

outputs = llm.generate(
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {"audio": (waveform, sr)},
    },
    sampling_params=SamplingParams(max_tokens=200),
)
print(outputs[0].outputs[0].text)
```

### Ultravox Audio-Language Model

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"audio": 1}'
```

```python
import librosa
from vllm import LLM, SamplingParams

llm = LLM(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    max_model_len=4096,
    limit_mm_per_prompt={"audio": 1},
)

waveform, sr = librosa.load("speech.wav", sr=None)

outputs = llm.generate(
    {
        "prompt": "<|audio|>\nTranscribe the audio.",
        "multi_modal_data": {"audio": (waveform, sr)},
    },
    sampling_params=SamplingParams(max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

---

## Video Understanding (Online API)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"video": 1}'
```

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
            {"type": "video_url", "video_url": {"url": video_url}},
        ],
    }],
    max_completion_tokens=512,
)
print(response.choices[0].message.content)
```

---

## Video Understanding (Offline Inference)

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

---

## Mixed Modalities (Omni Models)

### Qwen2.5-Omni (Audio + Video)

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_audio, fetch_video

llm = LLM(
    model="Qwen/Qwen2.5-Omni-7B",
    max_model_len=8192,
    limit_mm_per_prompt={"audio": 1, "video": 1},
)

audio, sr = fetch_audio("https://example.com/speech.wav")
frames, metadata = fetch_video("https://example.com/video.mp4",
                               video_io_kwargs={"num_frames": 16})

outputs = llm.generate(
    {
        "prompt": "Describe both the audio and video content.",
        "multi_modal_data": {
            "audio": (audio, sr),
            "video": (frames, metadata),
        },
    },
    sampling_params=SamplingParams(max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

---

## Pre-computed Embeddings

When you have pre-computed embeddings from a separate encoder:

```bash
vllm serve my-vision-model --enable-mm-embeds
```

```python
import torch
from vllm import LLM, SamplingParams

llm = LLM(
    model="my-vision-model",
    enable_mm_embeds=True,
    limit_mm_per_prompt={"image": 0},  # Disable raw image input
)

# Pre-computed image embeddings
# Shape: (num_patches, hidden_size)
image_embeds = torch.load("image_embeddings.pt")

outputs = llm.generate(
    {
        "prompt": "USER: <image>\nDescribe this.\nASSISTANT:",
        "multi_modal_data": {"image": image_embeds},
    },
    sampling_params=SamplingParams(max_tokens=256),
)
```

---

## Processor Cache Warm-Up

Use explicit UUIDs to maximize cache hit rates when the same images appear repeatedly:

```python
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    mm_processor_cache_gb=8.0,
)

image = Image.open("logo.jpg")
image_uuid = "company-logo-v1"  # Stable identifier

# First request — cache miss, processes image
outputs1 = llm.generate(
    {
        "prompt": "USER: <image>\nWhat is this logo?\nASSISTANT:",
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": image_uuid},
    },
    sampling_params=SamplingParams(max_tokens=128),
)

# Second request — cache hit, skips image processing
outputs2 = llm.generate(
    {
        "prompt": "USER: <image>\nDescribe the colors in this logo.\nASSISTANT:",
        "multi_modal_data": {"image": image},
        "multi_modal_uuids": {"image": image_uuid},
    },
    sampling_params=SamplingParams(max_tokens=128),
)
```

---

## Video with EVS Token Pruning

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --video-pruning-rate 0.3 \
    --limit-mm-per-prompt '{"video": 1}'
```

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_video

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    video_pruning_rate=0.3,  # Prune 30% of video tokens
    limit_mm_per_prompt={"video": 1},
)

frames, metadata = fetch_video(
    "https://example.com/long_video.mp4",
    video_io_kwargs={"num_frames": 64},
)

outputs = llm.generate(
    {
        "prompt": "USER: <video>\nSummarize this video.\nASSISTANT:",
        "multi_modal_data": {"video": (frames, metadata)},
    },
    sampling_params=SamplingParams(max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

---

## Encoding Utilities

### Encode Image for API

```python
from PIL import Image
from vllm.multimodal.utils import encode_image_url, encode_image_base64

image = Image.open("photo.jpg")

# As data URL
data_url = encode_image_url(image, image_mode="RGB", format="JPEG")
# "data:image/jpeg;base64,..."

# As base64 string
b64 = encode_image_base64(image, image_mode="RGB", format="PNG")
```

### Encode Audio for API

```python
import numpy as np
from vllm.multimodal.utils import encode_audio_url, encode_audio_base64

waveform = np.random.randn(16000).astype(np.float32)
sr = 16000

# As data URL
data_url = encode_audio_url(waveform, sr, format="WAV")
# "data:audio/wav;base64,..."

# As base64 string
b64 = encode_audio_base64(waveform, sr, format="WAV")
```

### Encode Video for API

```python
import numpy as np
from vllm.multimodal.utils import encode_video_url

# frames: (T, H, W, 3) uint8 array
frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)

data_url = encode_video_url(frames, format="JPEG")
# "data:video/jpeg;base64,..."
```

---

## Related Pages

- [Image Inputs](image-inputs.md) — detailed image input documentation
- [Audio Inputs](audio-inputs.md) — detailed audio input documentation
- [Video Inputs](video-inputs.md) — detailed video input documentation
- [Multimodal Config](config.md) — configuration reference
- [Multimodal Cache](cache.md) — caching architecture
