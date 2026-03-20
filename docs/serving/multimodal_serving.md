# Serving Multimodal Models

This guide covers how to deploy and serve multimodal models (vision, audio, and video) with vLLM's OpenAI-compatible server. For information on how to pass multimodal data in offline inference, see [Multimodal Inputs](../features/multimodal_inputs.md).

## Overview

vLLM supports serving multimodal language models that can process one or more of the following modalities alongside text:

| Modality | API Content Type | Offline Field |
|----------|-----------------|---------------|
| Images   | `image_url`, `image_pil`, `image_embeds` | `multi_modal_data["image"]` |
| Video    | `video_url` | `multi_modal_data["video"]` |
| Audio    | `input_audio`, `audio_url` | `multi_modal_data["audio"]` |

All multimodal inputs are served through the standard [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) (`/v1/chat/completions`). Audio transcription models additionally expose the [Transcriptions API](#transcriptions-api) (`/v1/audio/transcriptions`).

## Quick Start

### Vision Model (Images)

```bash
# Serve a vision-language model
vllm serve llava-hf/llava-1.5-7b-hf \
  --runner generate \
  --max-model-len 4096
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
                    },
                },
            ],
        }
    ],
    max_completion_tokens=256,
)

print(response.choices[0].message.content)
```

### Video Model

```bash
# Serve a video-language model
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --runner generate \
  --max-model-len 8192
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video."},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
                    },
                },
            ],
        }
    ],
    max_completion_tokens=256,
)

print(response.choices[0].message.content)
```

### Audio Model

```bash
# Serve an audio-language model
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

with open("speech.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is being said?"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
            ],
        }
    ],
    max_completion_tokens=256,
)

print(response.choices[0].message.content)
```

## Server Configuration

### Limiting Media Per Prompt

Use `--limit-mm-per-prompt` to control how many media items of each modality a single request may contain. This prevents runaway memory usage from requests with many images or long videos.

```bash
# Allow up to 4 images per prompt
vllm serve microsoft/Phi-3.5-vision-instruct \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 4}'
```

You can also use the dot-notation shorthand for individual modalities:

```bash
vllm serve llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --limit-mm-per-prompt.image 4 \
  --limit-mm-per-prompt.video 2
```

For models that support multiple modalities simultaneously:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --limit-mm-per-prompt '{"image": 4, "audio": 2}'
```

!!! note
    The default limit is 1 per modality. Always set this explicitly for models that support multiple media items per prompt.

### Media I/O Options

Use `--media-io-kwargs` to pass backend-specific options for media loading and preprocessing. The value is a JSON object keyed by modality name.

```bash
# Custom RGBA background color for images
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'

# Enable video frame recovery for corrupted files
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --media-io-kwargs '{"video": {"frame_recovery": true}}'

# Set video sampling rate and max duration
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --media-io-kwargs '{"video": {"fps": 2, "max_duration": 120}}'
```

**Available image options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rgba_background_color` | `[R, G, B]` | `[255, 255, 255]` | Background color for RGBA→RGB conversion |

**Available video options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fps` | `int` | `2` | Target frames per second for sampling |
| `max_duration` | `int` | `300` | Maximum video duration to process (seconds) |
| `frame_recovery` | `bool` | `false` | Enable forward-scan recovery for failed frames |

### Allowed Media Domains

For security, restrict which external domains vLLM can fetch media from using `--allowed-media-domains`. This is especially important in containerized environments with access to internal networks.

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --allowed-media-domains upload.wikimedia.org github.com cdn.example.com
```

To also prevent HTTP redirects from bypassing domain restrictions:

```bash
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
```

### Allowing Local File Paths

To allow clients to reference local files by path (e.g., `file:///data/images/photo.jpg`), use `--allowed-local-media-path`:

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --allowed-local-media-path /data/images
```

!!! warning
    Only enable `--allowed-local-media-path` in trusted environments. Clients can read any file under the specified path.

### Fetch Timeouts

Control how long vLLM waits when fetching media from URLs:

```bash
# Image fetch timeout (default: 5 seconds)
export VLLM_IMAGE_FETCH_TIMEOUT=10

# Video fetch timeout (default: 30 seconds)
export VLLM_VIDEO_FETCH_TIMEOUT=60

# Audio fetch timeout (default: 10 seconds)
export VLLM_AUDIO_FETCH_TIMEOUT=30
```

## Model-Specific Configurations

### Vision-Language Models

#### LLaVA Family

```bash
# LLaVA 1.5
vllm serve llava-hf/llava-1.5-7b-hf \
  --runner generate \
  --max-model-len 4096

# LLaVA-OneVision (supports video)
vllm serve llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --runner generate \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 8, "video": 2}'
```

#### Phi-3.5-Vision

```bash
vllm serve microsoft/Phi-3.5-vision-instruct \
  --runner generate \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 4}'
```

#### Qwen2-VL / Qwen2.5-VL / Qwen3-VL

Qwen VL models support both images and videos with dynamic resolution:

```bash
# Qwen2.5-VL
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --runner generate \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 8, "video": 2}'

# Qwen3-VL with larger context
vllm serve Qwen/Qwen3-VL-7B-Instruct \
  --runner generate \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"image": 16, "video": 4}'
```

#### InternVL

```bash
vllm serve OpenGVLab/InternVL2-8B \
  --runner generate \
  --trust-remote-code \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4}'
```

### Audio-Language Models

#### Ultravox

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
  --max-model-len 4096
```

#### Qwen2-Audio

```bash
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"audio": 4}'
```

#### Whisper (Transcription)

Whisper models are served through the [Transcriptions API](#transcriptions-api):

```bash
vllm serve openai/whisper-large-v3-turbo \
  --runner generate \
  --max-model-len 448
```

### Omni Models (Image + Audio)

Omni models process both images and audio simultaneously:

```bash
# Qwen2.5-Omni
vllm serve Qwen/Qwen2.5-Omni-7B \
  --runner generate \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4, "audio": 2}'

# Qwen3-Omni
vllm serve Qwen/Qwen3-Omni-7B \
  --runner generate \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4, "audio": 2}'
```

## Chat Templates

A chat template is required for the Chat Completions API. Most models include a default template in their `tokenizer_config.json`. For models that don't, or when you need a custom template, use `--chat-template`:

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --chat-template examples/template_llava.jinja
```

!!! tip
    vLLM includes built-in fallback templates for many popular multimodal models. If no template is found, an error is raised and you must provide one manually.

## Transcriptions API

For automatic speech recognition (ASR) models like Whisper, vLLM exposes the `/v1/audio/transcriptions` endpoint compatible with the OpenAI Transcriptions API.

### Starting the Server

```bash
vllm serve openai/whisper-large-v3-turbo \
  --runner generate \
  --max-model-len 448
```

### Making Transcription Requests

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Transcribe from a local file
with open("speech.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=audio_file,
        language="en",
    )

print(transcription.text)
```

```python
# Transcribe with timestamp granularity
with open("speech.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

for word in transcription.words:
    print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
```

### Supported Response Formats

| Format | Description |
|--------|-------------|
| `json` | JSON object with `text` field (default) |
| `text` | Plain text transcription |
| `verbose_json` | JSON with segments, words, and timestamps |
| `srt` | SubRip subtitle format |
| `vtt` | WebVTT subtitle format |

## Performance Tuning

### GPU Memory Utilization

Multimodal models often require more GPU memory than text-only models due to the vision/audio encoder. Adjust `--gpu-memory-utilization` accordingly:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384
```

### Tensor Parallelism for Large Models

For large multimodal models, use tensor parallelism across multiple GPUs:

```bash
# 4-GPU tensor parallel for a 72B vision model
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 8, "video": 2}'
```

### Encoder Budget

vLLM automatically computes an encoder budget based on the model configuration and `--limit-mm-per-prompt`. The encoder budget determines how many multimodal tokens can be processed concurrently. For models with large vision encoders, you may need to reduce `--max-num-seqs` to avoid OOM errors:

```bash
vllm serve llava-hf/llava-onevision-qwen2-72b-ov-hf \
  --tensor-parallel-size 4 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 4}'
```

### Prefix Caching for Repeated Media

Enable prefix caching to avoid re-encoding the same images or audio across requests. This is especially useful when many requests share a common system image (e.g., a logo or reference image):

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --enable-prefix-caching
```

When using prefix caching, provide stable UUIDs for media items to maximize cache hits:

```python
response = client.chat.completions.create(
    model="llava-hf/llava-1.5-7b-hf",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/logo.png"},
                    "uuid": "company-logo-v1",  # Stable UUID for caching
                },
            ],
        }
    ],
)
```

## Security Considerations

### SSRF Protection

When serving multimodal models in production, always restrict which domains vLLM can fetch media from:

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --allowed-media-domains cdn.mycompany.com storage.googleapis.com
```

Additionally, prevent redirect-based bypasses:

```bash
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
```

### Embedding Input Safety

Pre-computed embedding inputs (`image_embeds`) bypass the vision encoder entirely and are injected directly into the language model. Malformed embeddings can cause crashes or unexpected behavior.

!!! warning
    Only enable `enable_mm_embeds=True` (offline) or accept `image_embeds` content type (online) for **trusted clients**. Validate embedding shapes before passing them to the model.

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors when serving multimodal models:

1. **Reduce `--limit-mm-per-prompt`** — fewer media items per request means less peak memory.
2. **Reduce `--max-num-seqs`** — fewer concurrent requests reduces the encoder budget.
3. **Reduce `--max-model-len`** — shorter context windows use less KV cache memory.
4. **Increase `--gpu-memory-utilization`** — allow vLLM to use more of the available GPU memory (up to `0.95`).
5. **Use tensor parallelism** — distribute the model across multiple GPUs.

### Chat Template Errors

If you see `No chat template found` errors:

1. Check if the model has a `chat_template` in its `tokenizer_config.json`.
2. Provide a custom template with `--chat-template path/to/template.jinja`.
3. Check `vllm/transformers_utils/chat_templates/` for built-in fallback templates.

### Media Fetch Timeouts

If media fetches are timing out:

```bash
# Increase timeouts for slow networks
export VLLM_IMAGE_FETCH_TIMEOUT=30
export VLLM_VIDEO_FETCH_TIMEOUT=120
export VLLM_AUDIO_FETCH_TIMEOUT=60
```

### Video Frame Decoding Failures

For corrupted or truncated video files, enable frame recovery:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --media-io-kwargs '{"video": {"frame_recovery": true}}'
```

## See Also

- [Multimodal Inputs](../features/multimodal_inputs.md) — detailed API reference for all input types
- [Supported Models](../models/supported_models.md) — list of supported multimodal models
- [OpenAI-Compatible Server](openai_compatible_server.md) — full server configuration reference
- [Offline Inference](offline_inference.md) — using multimodal models without a server
