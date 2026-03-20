# Generative Models

Generative (text generation) models are the most common model type in vLLM. These are decoder-only transformer models that generate text autoregressively — predicting one token at a time conditioned on all previous tokens.

---

## Overview

vLLM supports **100+ decoder-only architectures** including:

- **Standard transformers** — Llama, Mistral, Qwen, Gemma, Phi, GPT-2, OPT, BLOOM
- **Mixture-of-Experts (MoE)** — Mixtral, DeepSeek-V3, Qwen3-MoE, Llama 4
- **State Space Models (SSM)** — Mamba, Mamba2, Falcon Mamba
- **Hybrid models** — Jamba (attention + Mamba), Falcon H1, Bamba, OLMo Hybrid
- **Long-context models** — models with extended context windows via RoPE scaling

---

## Quick Start

### Basic Text Generation

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

outputs = llm.generate(
    ["Tell me about the history of artificial intelligence."],
    SamplingParams(max_tokens=256, temperature=0.7),
)
print(outputs[0].outputs[0].text)
```

### Chat Completion

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Using chat template
outputs = llm.chat([
    {"role": "user", "content": "What is the capital of France?"},
])
print(outputs[0].outputs[0].text)
```

### Serving via OpenAI-Compatible API

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

---

## Model Families

### Llama Family

The Llama family is the most widely used open-source model family. vLLM supports all generations:

| Model | Architecture | Context | Notes |
|---|---|---|---|
| Llama 1 | `LlamaForCausalLM` | 2K | Original Llama |
| Llama 2 | `LlamaForCausalLM` | 4K | Improved Llama |
| Llama 3 / 3.1 / 3.2 / 3.3 | `LlamaForCausalLM` | 128K | Current generation |
| Llama 4 Scout/Maverick | `Llama4ForCausalLM` | 10M | MoE, multimodal |

```python
# Llama 3.1 8B
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Llama 3.3 70B with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
)
```

### Qwen Family

Alibaba's Qwen models are high-performance multilingual models:

| Model | Architecture | Notes |
|---|---|---|
| Qwen 1.x | `QWenLMHeadModel` | Original Qwen |
| Qwen2 / Qwen2.5 | `Qwen2ForCausalLM` | Current generation |
| Qwen2 MoE | `Qwen2MoeForCausalLM` | Mixture of Experts |
| Qwen3 | `Qwen3ForCausalLM` | Latest generation |
| Qwen3 MoE | `Qwen3MoeForCausalLM` | MoE variant |

```python
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
```

### Mistral / Mixtral Family

Mistral AI's models are known for efficiency and strong performance:

| Model | Architecture | Notes |
|---|---|---|
| Mistral 7B | `MistralForCausalLM` | Sliding window attention |
| Mixtral 8x7B / 8x22B | `MixtralForCausalLM` | Sparse MoE |
| Mistral Large 3 | `MistralLarge3ForCausalLM` | 123B parameters |
| Mistral Small 3.1 | `Mistral3ForConditionalGeneration` | Multimodal |

```python
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3")
```

### DeepSeek Family

DeepSeek models are known for strong reasoning and coding capabilities:

| Model | Architecture | Notes |
|---|---|---|
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | MLA attention |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | 671B MoE |
| DeepSeek-R1 | `DeepseekV3ForCausalLM` | Reasoning model |

```python
# DeepSeek-V3 requires multi-GPU
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
)
```

### Gemma Family

Google's Gemma models are efficient open models:

| Model | Architecture | Notes |
|---|---|---|
| Gemma 1 | `GemmaForCausalLM` | 2B, 7B |
| Gemma 2 | `Gemma2ForCausalLM` | 2B, 9B, 27B |
| Gemma 3 | `Gemma3ForCausalLM` | Text-only |
| Gemma 3n | `Gemma3nForCausalLM` | Nano variant |

```python
llm = LLM(model="google/gemma-2-9b-it")
```

### Phi Family

Microsoft's Phi models are small but capable:

| Model | Architecture | Notes |
|---|---|---|
| Phi-2 | `PhiForCausalLM` | 2.7B |
| Phi-3 Mini/Small/Medium | `Phi3ForCausalLM` | 3.8B–14B |
| Phi-3.5 MoE | `PhiMoEForCausalLM` | 16x3.8B |

```python
llm = LLM(model="microsoft/Phi-3-mini-4k-instruct")
```

---

## Mixture-of-Experts (MoE) Models

MoE models activate only a subset of their parameters for each token, enabling large parameter counts with manageable compute costs.

### Key MoE Models

| Model | Architecture | Total Params | Active Params |
|---|---|---|---|
| Mixtral 8x7B | `MixtralForCausalLM` | 46.7B | 12.9B |
| Mixtral 8x22B | `MixtralForCausalLM` | 141B | 39B |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | 671B | 37B |
| Qwen3-30B-A3B | `Qwen3MoeForCausalLM` | 30B | 3B |
| Llama 4 Scout | `Llama4ForCausalLM` | 109B | 17B |
| Qwen2-57B-A14B | `Qwen2MoeForCausalLM` | 57B | 14B |

### Running MoE Models

```python
# Mixtral 8x7B on 2 GPUs
llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,
)

# DeepSeek-V3 on 8 GPUs
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    max_model_len=32768,
)
```

### Expert Parallelism

For very large MoE models, use expert parallelism to distribute experts across GPUs:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8
```

---

## State Space Models (SSM)

SSMs like Mamba process sequences with linear complexity, making them efficient for long sequences.

### Mamba Models

```python
# Mamba 2.8B
llm = LLM(model="state-spaces/mamba-2.8b-hf")

# Mamba 2
llm = LLM(model="state-spaces/mamba2-2.7b")

# Falcon Mamba
llm = LLM(model="tiiuae/falcon-mamba-7b")
```

!!! note "Pipeline parallelism"
    Pure SSM models (Mamba, Mamba2) do not support pipeline parallelism. Use tensor parallelism instead.

### Hybrid Models (Attention + SSM)

Hybrid models combine attention layers with SSM layers for the best of both worlds:

| Model | Architecture | Notes |
|---|---|---|
| Jamba | `JambaForCausalLM` | Attention + Mamba |
| Falcon H1 | `FalconH1ForCausalLM` | Attention + Mamba2 |
| Bamba | `BambaForCausalLM` | IBM Bamba |
| OLMo Hybrid | `OlmoHybridForCausalLM` | OLMo + Mamba |
| Zamba2 | `Zamba2ForCausalLM` | Zyphra Zamba2 |
| Nemotron-H | `NemotronHForCausalLM` | NVIDIA Nemotron-H |

```python
llm = LLM(model="ai21labs/Jamba-v0.1")
```

---

## Long-Context Models

Many modern models support very long context windows through techniques like RoPE scaling:

| Model | Context Window |
|---|---|
| Llama 3.1 / 3.3 | 128K tokens |
| Qwen2.5-7B-Instruct | 128K tokens |
| Mistral Large 3 | 128K tokens |
| Llama 4 Scout | 10M tokens |
| Kimi Linear | 1M tokens |

```python
# Enable long context (may require more GPU memory)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=131072,  # 128K tokens
)
```

---

## Sampling Parameters

Control text generation with `SamplingParams`:

```python
from vllm import SamplingParams

params = SamplingParams(
    temperature=0.8,       # Randomness (0.0 = greedy, 1.0 = full sampling)
    top_p=0.95,            # Nucleus sampling threshold
    top_k=50,              # Top-k sampling
    max_tokens=512,        # Maximum tokens to generate
    stop=["</s>", "\n\n"], # Stop sequences
    presence_penalty=0.1,  # Penalize repeated topics
    frequency_penalty=0.1, # Penalize repeated tokens
    repetition_penalty=1.1,# Multiplicative repetition penalty
    n=3,                   # Number of output sequences
    best_of=5,             # Generate 5, return best 3
    seed=42,               # Random seed for reproducibility
)
```

### Greedy Decoding

```python
params = SamplingParams(temperature=0.0)  # Deterministic
```

### Beam Search

```python
params = SamplingParams(
    use_beam_search=True,
    best_of=5,
    temperature=0.0,
)
```

---

## Quantization

Most generative models support quantization to reduce memory usage and increase throughput:

```python
# AWQ quantization
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
)

# GPTQ quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
)

# BitsAndBytes INT4
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
)

# FP8 (requires Hopper or Ada GPU)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",
)

# GGUF
llm = LLM(
    model="TheBloke/Llama-2-7B-GGUF",
    tokenizer="meta-llama/Llama-2-7b-hf",
)
```

---

## LoRA Fine-Tuned Models

vLLM supports serving LoRA adapters on top of base models:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    enable_lora=True,
    max_lora_rank=64,
)

# Use a specific LoRA adapter
outputs = llm.generate(
    ["Tell me a story."],
    lora_request=LoRARequest(
        lora_name="my-adapter",
        lora_int_id=1,
        lora_path="/path/to/lora/adapter",
    ),
)
```

---

## Speculative Decoding

Speculative decoding uses a small draft model to propose tokens that are then verified by the main model, increasing throughput:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
)
```

vLLM also supports EAGLE and Medusa draft models:

```python
# EAGLE speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    speculative_model_uses_eagle=True,
)
```

---

## Parallelism

### Tensor Parallelism

Distribute model weights across multiple GPUs:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # Use 4 GPUs
)
```

### Pipeline Parallelism

Distribute model layers across GPUs (useful for very large models):

```python
llm = LLM(
    model="meta-llama/Llama-3.1-405B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,  # 8 GPUs total
)
```

---

## Structured Output

vLLM supports constrained decoding for structured outputs:

```python
from vllm.sampling_params import GuidedDecodingParams

# JSON schema
params = SamplingParams(
    guided_decoding=GuidedDecodingParams(
        json={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
    )
)

# Regex
params = SamplingParams(
    guided_decoding=GuidedDecodingParams(regex=r"\d{3}-\d{3}-\d{4}")
)

# Grammar (EBNF)
params = SamplingParams(
    guided_decoding=GuidedDecodingParams(grammar="root ::= 'yes' | 'no'")
)
```

---

## Related

- [Supported Models](supported_models.md) — complete list of supported architectures
- [Adding a New Model](adding_model.md) — implementing a new generative model
- [Multimodal Models](multimodal_models.md) — vision-language and audio-language models
- [Engine Arguments](../configuration/engine_args.md) — full configuration reference
- [Parallelism & Scaling](../serving/parallelism_scaling.md) — multi-GPU deployment
