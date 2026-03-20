# Realtime API (WebSocket)

The `/v1/realtime` endpoint provides WebSocket-based streaming audio transcription, enabling real-time speech-to-text as audio is being recorded or streamed. It is designed for low-latency, interactive transcription applications.

!!! note "Audio Dependencies"
    Install audio dependencies before using this endpoint:
    ```bash
    pip install "vllm[audio]"
    ```

---

## Overview

Unlike the HTTP-based [Speech-to-Text](speech_to_text.md) endpoint which processes complete audio files, the Realtime API accepts audio in small chunks over a persistent WebSocket connection and returns transcription deltas as they are generated.

**Use cases:**
- Live microphone transcription
- Real-time meeting notes
- Voice-controlled applications
- Streaming audio file transcription

---

## Quick Start

### Start the Server

```bash
vllm serve openai/whisper-large-v3-turbo \
  --host 0.0.0.0 \
  --port 8000
```

### Connect and Transcribe

```python
import asyncio
import base64
import json
import wave
import websockets

async def transcribe_audio(audio_file: str):
    uri = "ws://localhost:8000/v1/realtime"

    async with websockets.connect(uri) as ws:
        # 1. Wait for session.created
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created"
        print(f"Session ID: {msg['id']}")

        # 2. Optionally configure the session
        await ws.send(json.dumps({
            "type": "session.update",
            "model": "openai/whisper-large-v3-turbo",
        }))

        # 3. Signal we're about to send audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

        # 4. Send audio chunks
        with wave.open(audio_file, "rb") as wf:
            chunk_size = 4096
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                audio_b64 = base64.b64encode(data).decode()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }))

        # 5. Signal end of audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
            "final": True,
        }))

        # 6. Collect transcription events
        full_text = ""
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "transcription.delta":
                print(event["delta"], end="", flush=True)
                full_text += event["delta"]
            elif event["type"] == "transcription.done":
                print()
                print(f"\nFinal: {event['text']}")
                if event.get("usage"):
                    print(f"Tokens: {event['usage']}")
                break
            elif event["type"] == "error":
                print(f"Error: {event['error']}")
                break

asyncio.run(transcribe_audio("audio.wav"))
```

---

## Audio Format Requirements

| Property | Value |
|----------|-------|
| Encoding | PCM16 (16-bit signed integer) |
| Sample rate | 16,000 Hz (16 kHz) |
| Channels | Mono (1 channel) |
| Byte order | Little-endian |
| Transfer format | Base64-encoded |

### Converting Audio to PCM16

```python
import subprocess
import base64

def convert_to_pcm16(input_file: str) -> bytes:
    """Convert any audio file to PCM16 @ 16kHz mono using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", input_file,
            "-ar", "16000",   # 16kHz sample rate
            "-ac", "1",       # Mono
            "-f", "s16le",    # PCM16 little-endian
            "-",              # Output to stdout
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout

# Convert and encode
pcm_data = convert_to_pcm16("audio.mp3")
audio_b64 = base64.b64encode(pcm_data).decode()
```

---

## Protocol Reference

### Connection

```
WebSocket: ws://localhost:8000/v1/realtime
```

With authentication:
```python
import websockets

async with websockets.connect(
    "ws://localhost:8000/v1/realtime",
    extra_headers={"Authorization": "Bearer my-api-key"},
) as ws:
    ...
```

---

### Client → Server Events

#### `input_audio_buffer.append`

Send a chunk of base64-encoded PCM16 audio:

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64-encoded-pcm16-data>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"input_audio_buffer.append"` |
| `audio` | `string` | Base64-encoded PCM16 audio chunk |

#### `input_audio_buffer.commit`

Trigger transcription processing or signal end of audio:

```json
{
  "type": "input_audio_buffer.commit",
  "final": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | — | Always `"input_audio_buffer.commit"` |
| `final` | `bool` | `false` | If `true`, signals that no more audio will be sent |

**When to commit:**
- Send `commit` (with `final: false`) before sending audio chunks to signal the server to start processing
- Send `commit` (with `final: true`) when you have finished sending all audio

#### `session.update`

Configure session parameters:

```json
{
  "type": "session.update",
  "model": "openai/whisper-large-v3-turbo"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"session.update"` |
| `model` | `string \| null` | Model to use for transcription |

---

### Server → Client Events

#### `session.created`

Sent immediately after connection is established:

```json
{
  "type": "session.created",
  "id": "sess-abc123",
  "created": 1710000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"session.created"` |
| `id` | `string` | Unique session identifier |
| `created` | `int` | Unix timestamp |

#### `transcription.delta`

Incremental transcription text as it is generated:

```json
{
  "type": "transcription.delta",
  "delta": "Hello, "
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"transcription.delta"` |
| `delta` | `string` | Incremental text chunk |

#### `transcription.done`

Final transcription with complete text and usage statistics:

```json
{
  "type": "transcription.done",
  "text": "Hello, world! This is a test.",
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 8,
    "total_tokens": 8
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"transcription.done"` |
| `text` | `string` | Complete transcription |
| `usage` | `object \| null` | Token usage statistics |

#### `error`

Error notification:

```json
{
  "type": "error",
  "error": "Audio format not supported",
  "code": "invalid_audio_format"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | Always `"error"` |
| `error` | `string` | Human-readable error message |
| `code` | `string \| null` | Machine-readable error code |

---

## Protocol Flow

```
Client                          Server
  |                               |
  |--- WebSocket connect -------->|
  |<-- session.created -----------|
  |                               |
  |--- session.update ----------->|  (optional)
  |                               |
  |--- input_audio_buffer.commit ->|  (signal ready)
  |                               |
  |--- input_audio_buffer.append ->|  (chunk 1)
  |--- input_audio_buffer.append ->|  (chunk 2)
  |--- input_audio_buffer.append ->|  (chunk N)
  |                               |
  |<-- transcription.delta -------|  (incremental text)
  |<-- transcription.delta -------|
  |<-- transcription.done --------|  (final text + usage)
  |                               |
  |--- input_audio_buffer.append ->|  (next utterance)
  |--- input_audio_buffer.commit ->|  (final=True to end)
  |                               |
  |<-- transcription.done --------|
  |                               |
  |--- WebSocket close ---------->|
```

---

## Live Microphone Transcription

```python
import asyncio
import base64
import json
import pyaudio
import websockets

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

async def live_transcription():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    async with websockets.connect("ws://localhost:8000/v1/realtime") as ws:
        # Wait for session
        await ws.recv()

        # Signal start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        print("Recording... Press Ctrl+C to stop.")

        async def send_audio():
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_b64 = base64.b64encode(data).decode()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }))
                await asyncio.sleep(0)

        async def receive_transcription():
            async for message in ws:
                event = json.loads(message)
                if event["type"] == "transcription.delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "transcription.done":
                    print()

        await asyncio.gather(send_audio(), receive_transcription())

asyncio.run(live_transcription())
```

---

## Authentication

When the server is started with `--api-key`, include the token in the WebSocket handshake:

```python
import websockets

async with websockets.connect(
    "ws://localhost:8000/v1/realtime",
    additional_headers={"Authorization": "Bearer my-secret-key"},
) as ws:
    ...
```

---

## Error Handling

```python
async with websockets.connect("ws://localhost:8000/v1/realtime") as ws:
    await ws.recv()  # session.created

    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    # Send audio...
    await ws.send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": audio_b64,
    }))

    async for message in ws:
        event = json.loads(message)
        match event["type"]:
            case "transcription.delta":
                print(event["delta"], end="", flush=True)
            case "transcription.done":
                print(f"\nDone: {event['text']}")
                break
            case "error":
                print(f"Error [{event.get('code')}]: {event['error']}")
                break
```

---

## Example Clients

The vLLM repository includes ready-to-use example clients:

- **File transcription**: `examples/online_serving/openai_realtime_client.py`
- **Live microphone**: `examples/online_serving/openai_realtime_microphone_client.py` (Gradio demo)

---

## Comparison: Realtime vs. Transcriptions API

| Feature | Realtime API | Transcriptions API |
|---------|-------------|-------------------|
| Protocol | WebSocket | HTTP POST |
| Input | Streaming audio chunks | Complete audio file |
| Output | Streaming deltas | Complete transcription |
| Latency | Very low (streaming) | Higher (batch) |
| Use case | Live transcription | File processing |
| Connection | Persistent | Per-request |
| Formats | PCM16 @ 16kHz | FLAC, MP3, WAV, etc. |

---

## See Also

- [Speech-to-Text](speech_to_text.md) — HTTP-based audio transcription
- [Streaming Guide](streaming.md) — General streaming patterns
- [API Reference](api_reference.md) — Full endpoint reference
