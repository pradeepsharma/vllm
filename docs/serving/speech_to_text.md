# Speech-to-Text

vLLM supports audio transcription and translation through two HTTP endpoints:

- `POST /v1/audio/transcriptions` — Transcribe audio to text in the original language
- `POST /v1/audio/translations` — Translate audio to English

Both endpoints are compatible with the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio).

!!! note "Audio Dependencies"
    Install audio dependencies before using these endpoints:
    ```bash
    pip install "vllm[audio]"
    ```

For real-time streaming transcription, see the [Realtime API](realtime_api.md).

---

## Quick Start

### Start the Server

```bash
vllm serve openai/whisper-large-v3-turbo \
  --host 0.0.0.0 \
  --port 8000
```

### Transcribe Audio

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
        )

    print(transcription.text)
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/audio/transcriptions \
      -F "file=@audio.mp3" \
      -F "model=openai/whisper-large-v3-turbo"
    ```

---

## Transcription Endpoint

### `POST /v1/audio/transcriptions`

**Request** — `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `file` | **required** | Audio file to transcribe |
| `model` | `string` | **required** | Model identifier |
| `language` | `string \| null` | `null` | ISO-639-1 language code (e.g., `"en"`, `"zh"`, `"fr"`) |
| `prompt` | `string` | `""` | Optional text to guide transcription style |
| `response_format` | `string` | `"json"` | Output format (see below) |
| `temperature` | `float` | `0.0` | Sampling temperature (0–1) |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `timestamp_granularities[]` | `array` | `[]` | `"word"` and/or `"segment"` timestamps |

### Supported Audio Formats

| Format | Extension |
|--------|-----------|
| FLAC | `.flac` |
| MP3 | `.mp3` |
| MP4 | `.mp4` |
| MPEG | `.mpeg` |
| MPGA | `.mpga` |
| M4A | `.m4a` |
| OGG | `.ogg` |
| WAV | `.wav` |
| WEBM | `.webm` |

!!! note "File Size Limit"
    The default maximum audio file size is 25 MB. Override with the `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` environment variable.

### Response Formats

| Format | Description |
|--------|-------------|
| `json` | Simple JSON with `text` field |
| `text` | Plain text |
| `srt` | SubRip subtitle format |
| `vtt` | WebVTT subtitle format |
| `verbose_json` | JSON with segments, words, and metadata |

---

## Response Examples

### `json` Format

```json
{
  "text": "Hello, this is a transcription of the audio file.",
  "usage": {"type": "duration", "seconds": 5}
}
```

### `verbose_json` Format

```json
{
  "text": "Hello, this is a transcription.",
  "language": "en",
  "duration": "5.42",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a transcription.",
      "tokens": [50364, 938, 428, 307, 275, 28347],
      "temperature": 0.0,
      "avg_logprob": -0.245,
      "compression_ratio": 1.235
    }
  ],
  "words": [
    {"word": "Hello,", "start": 0.0, "end": 0.4},
    {"word": "this", "start": 0.5, "end": 0.7}
  ]
}
```

### `srt` Format

```
1
00:00:00,000 --> 00:00:02,500
Hello, this is a transcription.

2
00:00:02,500 --> 00:00:05,420
Of the audio file.
```

### `vtt` Format

```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, this is a transcription.

00:00:02.500 --> 00:00:05.420
Of the audio file.
```

---

## Detailed Examples

### Transcribe with Language Hint

Providing the language improves accuracy and reduces latency:

```python
with open("french_audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        language="fr",
    )
print(transcription.text)
```

### Verbose JSON with Timestamps

```python
with open("audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

print(f"Duration: {transcription.duration}s")
for segment in transcription.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
```

### Streaming Transcription

```python
with open("audio.mp3", "rb") as f:
    stream = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        stream=True,
    )

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Style Prompt

Use a prompt to guide the transcription style or continue from previous context:

```python
with open("technical_audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        prompt="This is a technical discussion about machine learning and neural networks.",
    )
```

---

## Translation Endpoint

### `POST /v1/audio/translations`

Translate audio from any of 55 supported non-English languages into English.

!!! warning "Model Support"
    `openai/whisper-large-v3-turbo` does **not** support translation. Use `openai/whisper-large-v3` or `openai/whisper-large-v2` instead.

**Request** — `multipart/form-data`

Same fields as `/v1/audio/transcriptions`. The `language` field specifies the source language.

### Translate Audio

```python
with open("german_audio.mp3", "rb") as f:
    translation = client.audio.translations.create(
        model="openai/whisper-large-v3",
        file=f,
    )
print(translation.text)
# → English translation of the German audio
```

### Translation Response

```json
{
  "text": "Hello, this is the translated text in English."
}
```

---

## Sampling Parameters (vLLM Extensions)

Pass these via the form fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | `0.0` | Sampling temperature (0–1) |
| `top_p` | `float \| null` | `null` | Nucleus sampling |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `min_p` | `float \| null` | `null` | Minimum probability threshold |
| `seed` | `int \| null` | `null` | Random seed |
| `frequency_penalty` | `float \| null` | `0.0` | Frequency penalty |
| `presence_penalty` | `float \| null` | `0.0` | Presence penalty |
| `repetition_penalty` | `float \| null` | `null` | Repetition penalty |
| `max_completion_tokens` | `int \| null` | `null` | Maximum tokens to generate |

### Streaming Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream` | `bool` | `false` | Enable SSE streaming |
| `stream_include_usage` | `bool` | `false` | Include usage in stream |
| `stream_continuous_usage_stats` | `bool` | `false` | Continuous usage stats in stream |

---

## Batch Transcription

For processing many audio files offline, use the [Batch Inference](batch_inference.md) runner:

```jsonl
{"custom_id": "audio-001", "method": "POST", "url": "/v1/audio/transcriptions", "body": {"model": "openai/whisper-large-v3-turbo", "file_url": "https://example.com/audio1.mp3", "language": "en"}}
{"custom_id": "audio-002", "method": "POST", "url": "/v1/audio/transcriptions", "body": {"model": "openai/whisper-large-v3-turbo", "file_url": "https://example.com/audio2.mp3", "language": "fr"}}
```

```bash
python -m vllm.entrypoints.openai.run_batch \
  --model openai/whisper-large-v3-turbo \
  --input-file batch_input.jsonl \
  --output-file batch_output.jsonl
```

Note: Batch requests use `file_url` (a URL or base64 data URL) instead of `file`.

---

## Supported Models

| Model | Transcription | Translation | Notes |
|-------|--------------|-------------|-------|
| `openai/whisper-large-v3-turbo` | ✅ | ❌ | Fast, recommended for transcription |
| `openai/whisper-large-v3` | ✅ | ✅ | Best accuracy |
| `openai/whisper-large-v2` | ✅ | ✅ | Previous generation |
| `openai/whisper-medium` | ✅ | ✅ | Balanced speed/accuracy |
| `openai/whisper-small` | ✅ | ✅ | Fast, lower accuracy |
| `openai/whisper-base` | ✅ | ✅ | Very fast, basic accuracy |
| `openai/whisper-tiny` | ✅ | ✅ | Fastest, lowest accuracy |

---

## Comparison: Transcriptions vs. Realtime API

| Feature | Transcriptions API | Realtime API |
|---------|-------------------|-------------|
| Protocol | HTTP POST | WebSocket |
| Input | Complete audio file | Streaming chunks |
| Output | Complete transcription | Streaming deltas |
| Latency | Higher (batch) | Very low |
| Formats | FLAC, MP3, WAV, etc. | PCM16 @ 16kHz only |
| Timestamps | Word/segment level | Not supported |
| Streaming | Optional SSE | Always streaming |
| Use case | File processing | Live transcription |

---

## See Also

- [Realtime API](realtime_api.md) — WebSocket streaming transcription
- [Batch Inference](batch_inference.md) — Offline batch processing
- [API Reference](api_reference.md) — Full endpoint reference
