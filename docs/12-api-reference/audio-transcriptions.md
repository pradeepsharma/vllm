# POST /v1/audio/transcriptions

The Audio Transcriptions endpoint converts speech audio files to text. It is compatible with the OpenAI Audio Transcriptions API and supports multiple audio formats, output formats, streaming, and timestamp granularities.

**Source:** `vllm/entrypoints/openai/speech_to_text/`

> **Note:** This endpoint requires a model that supports the `transcription` task (e.g., Whisper-based models). The server must be started with a transcription-capable model.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text |
| `POST` | `/v1/audio/translations` | Translate audio to English text |

---

## POST /v1/audio/transcriptions

Transcribes audio into the input language.

### Request

Content-Type: `multipart/form-data`

The request uses form data (not JSON) because it includes a binary audio file upload.

#### Request Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `file` | **required** | The audio file to transcribe. Supported formats: `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`. |
| `model` | `string \| null` | `null` | Model ID to use. If `null`, uses the server's default model. |
| `language` | `string \| null` | `null` | Language of the input audio in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format (e.g., `"en"`, `"fr"`, `"de"`). Providing this improves accuracy and latency. |
| `prompt` | `string` | `""` | Optional text to guide the model's style or continue a previous audio segment. Should match the audio language. |
| `response_format` | `string` | `"json"` | Output format: `"json"`, `"text"`, `"srt"`, `"verbose_json"`, or `"vtt"`. |
| `timestamp_granularities[]` | `array` | `[]` | Timestamp granularities to populate. Requires `response_format="verbose_json"`. Options: `"word"`, `"segment"`. |
| `stream` | `bool \| null` | `false` | Stream partial results as Server-Sent Events. |
| `temperature` | `float` | `0.0` | Sampling temperature (0–1). `0` uses log-probability-based auto-temperature. |

#### vLLM Sampling Extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_p` | `float \| null` | `null` | Nucleus sampling probability. |
| `top_k` | `int \| null` | `null` | Top-K sampling. |
| `min_p` | `float \| null` | `null` | Minimum probability threshold. |
| `seed` | `int \| null` | `null` | Random seed for reproducibility. |
| `frequency_penalty` | `float \| null` | `0.0` | Frequency penalty. |
| `repetition_penalty` | `float \| null` | `null` | Repetition penalty. |
| `presence_penalty` | `float \| null` | `0.0` | Presence penalty. |
| `max_completion_tokens` | `int \| null` | `null` | Maximum tokens to generate. |

#### vLLM Extra Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stream_include_usage` | `bool \| null` | `false` | Include usage stats in streaming response. Requires `stream=true`. |
| `stream_continuous_usage_stats` | `bool \| null` | `false` | Include usage stats in every streaming chunk. Requires `stream=true`. |
| `vllm_xargs` | `object \| null` | `null` | Custom extension parameters. |

---

### Response Formats

#### `json` (default)

```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "usage": {
    "type": "duration",
    "seconds": 3
  }
}
```

#### `text`

Plain text response:

```
The quick brown fox jumps over the lazy dog.
```

#### `verbose_json`

Includes detailed segment and word-level information:

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": "3.5",
  "text": "The quick brown fox jumps over the lazy dog.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "The quick brown fox jumps over the lazy dog.",
      "tokens": [50364, 440, 2068, 6292, 21939, 18045, 670, 264, 14847, 3000, 13, 50539],
      "temperature": 0.0,
      "avg_logprob": -0.25,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ],
  "words": [
    {"word": "The", "start": 0.0, "end": 0.2},
    {"word": "quick", "start": 0.2, "end": 0.5},
    ...
  ]
}
```

#### `srt`

SubRip subtitle format:

```
1
00:00:00,000 --> 00:00:03,500
The quick brown fox jumps over the lazy dog.
```

#### `vtt`

WebVTT format:

```
WEBVTT

00:00:00.000 --> 00:00:03.500
The quick brown fox jumps over the lazy dog.
```

---

### Streaming Response

When `stream=true`, the server returns Server-Sent Events with incremental transcription chunks:

```
data: {"id":"trsc-abc123","object":"transcription.chunk","created":1714000000,"model":"openai/whisper-large-v3","choices":[{"delta":{"content":"The quick"},"finish_reason":null}]}

data: {"id":"trsc-abc123","object":"transcription.chunk","created":1714000000,"model":"openai/whisper-large-v3","choices":[{"delta":{"content":" brown fox"},"finish_reason":null}]}

data: {"id":"trsc-abc123","object":"transcription.chunk","created":1714000000,"model":"openai/whisper-large-v3","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":10,"total_tokens":10}}

data: [DONE]
```

---

## Examples

### Basic Transcription

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

with open("audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        response_format="json"
    )
print(transcription.text)
```

### Transcription with Language Hint

```python
with open("french_audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        language="fr",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

print(transcription.text)
for word in transcription.words:
    print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
```

### Streaming Transcription

```python
with open("audio.wav", "rb") as f:
    stream = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### Raw HTTP Request (curl)

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer token-abc123" \
  -F "file=@audio.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=json" \
  -F "language=en"
```

### Verbose JSON with Timestamps

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer token-abc123" \
  -F "file=@audio.wav" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment"
```

---

## POST /v1/audio/translations

Translates audio from any language into English text. The request and response schemas are identical to `/v1/audio/transcriptions`, except:

- The output is always in English regardless of the input language.
- The `to_language` field is available as a placeholder for future translation targets (currently unused).

```python
with open("german_audio.mp3", "rb") as f:
    translation = client.audio.translations.create(
        model="openai/whisper-large-v3",
        file=f
    )
print(translation.text)  # English translation
```

---

## Starting the Server for Transcription

To enable the transcription endpoint, serve a Whisper-compatible model:

```bash
vllm serve openai/whisper-large-v3 \
  --task transcription \
  --max-model-len 448
```

---

## Related Pages

- [Batch Inference](batch-inference.md) — Batch transcription via `run_batch.py`
- [CLI Arguments](cli-args.md) — Server configuration flags
- [Error Handling](error-handling.md) — HTTP status codes and error formats
