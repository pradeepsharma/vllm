# Encoder-Decoder Models

vLLM supports encoder-decoder architectures for sequence-to-sequence tasks. Unlike decoder-only models that generate text autoregressively from a prompt, encoder-decoder models first encode an input sequence (e.g., audio, an image, or text) and then decode a target sequence conditioned on the encoder's output.

---

## Supported Encoder-Decoder Architectures

| Architecture | Model Family | Task | Example Models |
|---|---|---|---|
| `WhisperForConditionalGeneration` | Whisper | Automatic Speech Recognition | `openai/whisper-large-v3` |
| `NemotronParseForConditionalGeneration` | Nemotron Parse | Document Parsing / OCR | `nvidia/Nemotron-Parse` |
| `FireRedASR2ForConditionalGeneration` | FireRedASR2 | Automatic Speech Recognition | — |
| `Qwen3ASRForConditionalGeneration` | Qwen3 ASR | Automatic Speech Recognition | `Qwen/Qwen3-ASR` |
| `Qwen3ASRRealtimeGeneration` | Qwen3 ASR Realtime | Real-time ASR | `Qwen/Qwen3-ASR-Realtime` |

!!! note "Previously supported encoder-decoder models"
    BART, mBART, and Florence2 were supported in vLLM until v0.10.2 but have since been removed as part of the V0 deprecation. Use an older version of vLLM if you need these architectures.

---

## Automatic Speech Recognition (ASR)

### Whisper

[Whisper](https://github.com/openai/whisper) is OpenAI's encoder-decoder model for multilingual speech recognition, translation, and language identification.

#### Basic Usage

```python
import numpy as np
from vllm import LLM, SamplingParams

llm = LLM(model="openai/whisper-large-v3")

# Load audio (16kHz mono float32)
audio_array = np.zeros(16000, dtype=np.float32)  # 1 second of silence

outputs = llm.generate(
    [{
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=200, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

#### Transcription with Timestamps

```python
outputs = llm.generate(
    [{
        "prompt": "<|startoftranscript|><|en|><|transcribe|>",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=448, temperature=0.0),
)
```

#### Language Detection

```python
# Use <|notimestamps|> without a language token to let Whisper detect the language
outputs = llm.generate(
    [{
        "prompt": "<|startoftranscript|><|notimestamps|>",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=5, temperature=0.0),
)
# The first token will be the detected language (e.g., <|en|>)
```

#### Translation to English

```python
# Translate from any language to English
outputs = llm.generate(
    [{
        "prompt": "<|startoftranscript|><|fr|><|translate|><|notimestamps|>",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=200, temperature=0.0),
)
```

#### Whisper Prompt Tokens Reference

| Token | Meaning |
|---|---|
| `<\|startoftranscript\|>` | Start of transcription (required) |
| `<\|en\|>`, `<\|fr\|>`, `<\|zh\|>`, ... | Language token (ISO 639-1 code) |
| `<\|transcribe\|>` | Transcription task |
| `<\|translate\|>` | Translation to English task |
| `<\|notimestamps\|>` | Disable timestamp tokens |
| `<\|endoftext\|>` | End of transcription |

#### Supported Whisper Models

| Model | Parameters | Languages | Notes |
|---|---|---|---|
| `openai/whisper-tiny` | 39M | 99 | Fastest, lowest accuracy |
| `openai/whisper-base` | 74M | 99 | — |
| `openai/whisper-small` | 244M | 99 | — |
| `openai/whisper-medium` | 769M | 99 | — |
| `openai/whisper-large-v2` | 1.5B | 99 | — |
| `openai/whisper-large-v3` | 1.5B | 99 | Best accuracy |
| `openai/whisper-large-v3-turbo` | 809M | 99 | Fast + accurate |
| `distil-whisper/distil-large-v3` | 756M | English | Distilled, fast |

#### Serving Whisper

```bash
vllm serve openai/whisper-large-v3 \
    --max-model-len 448 \
    --limit-mm-per-prompt audio=1
```

#### Using the OpenAI Audio API

vLLM's OpenAI-compatible server supports the `/v1/audio/transcriptions` endpoint for Whisper:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        language="en",
        response_format="text",
    )
print(transcript)
```

---

### Qwen3 ASR

Qwen3 ASR is a high-accuracy ASR model supporting multiple languages.

```python
from vllm import LLM, SamplingParams
import numpy as np

llm = LLM(model="Qwen/Qwen3-ASR")

audio_array = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds

outputs = llm.generate(
    [{
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=500, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

---

### FireRedASR2

FireRedASR2 is a high-performance ASR model optimized for Chinese and multilingual speech.

```python
from vllm import LLM, SamplingParams
import numpy as np

llm = LLM(model="FireRedTeam/FireRedASR2")

audio_array = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds

outputs = llm.generate(
    [{
        "prompt": "",
        "multi_modal_data": {"audio": (audio_array, 16000)},
    }],
    SamplingParams(max_tokens=200, temperature=0.0),
)
```

---

## Document Parsing

### Nemotron Parse

Nemotron Parse is an encoder-decoder model for structured document parsing and OCR.

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="nvidia/Nemotron-Parse")

image = Image.open("document.png")
outputs = llm.generate(
    [{
        "prompt": "Parse this document:",
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=1000, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

---

## Architecture Details

### How Encoder-Decoder Models Work in vLLM

Encoder-decoder models in vLLM use a two-phase processing approach:

1. **Encoding phase**: The encoder processes the input (audio, image, or text) and produces a sequence of hidden states (encoder outputs).
2. **Decoding phase**: The decoder generates tokens autoregressively, attending to both the encoder outputs (via cross-attention) and previously generated tokens (via self-attention).

The KV cache in vLLM stores both:
- **Self-attention KV cache** — for the decoder's autoregressive generation
- **Cross-attention KV cache** — for the encoder-decoder attention (computed once per input)

### Cross-Attention KV Cache

Unlike decoder-only models where the KV cache grows with each generated token, the cross-attention KV cache is fixed after encoding. This means:

- The encoder runs once per input
- Cross-attention keys and values are cached and reused across all decoding steps
- Memory usage is dominated by the encoder output size, not the generation length

---

## Audio Input Formats

All ASR models in vLLM accept audio in the following formats:

| Format | Description |
|---|---|
| `np.ndarray` | 1D float32 array (mono, any sample rate) |
| `tuple[np.ndarray, float]` | `(audio_array, sample_rate)` — resampled to model's expected rate |
| `torch.Tensor` | 1D float32 tensor |

!!! tip "Sample rate handling"
    Most ASR models expect 16kHz audio. If your audio has a different sample rate, pass it as `(audio_array, sample_rate)` and vLLM will automatically resample it.

```python
import librosa
import numpy as np

# Load audio at original sample rate
audio, sr = librosa.load("audio.mp3", sr=None)

# vLLM will resample from sr to 16000 Hz automatically
outputs = llm.generate([{
    "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    "multi_modal_data": {"audio": (audio, sr)},
}])
```

---

## Performance Considerations

### Batching ASR Requests

ASR models benefit from batching multiple audio inputs together:

```python
audio_inputs = [
    {"prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
     "multi_modal_data": {"audio": (audio1, 16000)}},
    {"prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
     "multi_modal_data": {"audio": (audio2, 16000)}},
    {"prompt": "<|startoftranscript|><|fr|><|transcribe|><|notimestamps|>",
     "multi_modal_data": {"audio": (audio3, 16000)}},
]

outputs = llm.generate(audio_inputs, SamplingParams(max_tokens=200))
```

### Memory Optimization

For ASR workloads with many short audio clips:

```bash
vllm serve openai/whisper-large-v3 \
    --max-model-len 448 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9
```

---

## Related

- [Supported Models](supported_models.md) — full list of encoder-decoder architectures
- [Multimodal Models](multimodal_models.md) — audio-language models that use decoder-only architectures
- [Engine Arguments](../configuration/engine_args.md) — configuration options for encoder-decoder models
