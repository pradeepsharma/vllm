# Supported Models

vLLM supports a large and growing collection of model architectures across text generation, multimodal, pooling/embedding, transcription, and speculative decoding. All supported architectures are registered in `vllm/model_executor/models/registry.py`.

> **Note:** Whenever a new architecture is added to the registry, a corresponding entry with an example HuggingFace model must also be added to `tests/models/registry.py`.

## Text Generation Models (Decoder-only)

These models implement the `VllmModelForTextGeneration` interface and produce token-by-token output via autoregressive decoding.

| Architecture Class | Module | vLLM Class | Notes |
|---|---|---|---|
| `AfmoeForCausalLM` | `afmoe` | `AfmoeForCausalLM` | |
| `ApertusForCausalLM` | `apertus` | `ApertusForCausalLM` | |
| `AquilaModel` / `AquilaForCausalLM` | `llama` | `LlamaForCausalLM` | AquilaChat2 |
| `ArceeForCausalLM` | `arcee` | `ArceeForCausalLM` | |
| `ArcticForCausalLM` | `arctic` | `ArcticForCausalLM` | |
| `BaiChuanForCausalLM` | `baichuan` | `BaiChuanForCausalLM` | Baichuan-7B |
| `BaichuanForCausalLM` | `baichuan` | `BaichuanForCausalLM` | Baichuan-13B |
| `BailingMoeForCausalLM` | `bailing_moe` | `BailingMoeForCausalLM` | |
| `BambaForCausalLM` | `bamba` | `BambaForCausalLM` | Hybrid SSM |
| `BloomForCausalLM` | `bloom` | `BloomForCausalLM` | |
| `ChatGLMModel` / `ChatGLMForConditionalGeneration` | `chatglm` | `ChatGLMForCausalLM` | |
| `CohereForCausalLM` / `Cohere2ForCausalLM` | `commandr` | `CohereForCausalLM` | Command R |
| `DbrxForCausalLM` | `dbrx` | `DbrxForCausalLM` | |
| `DeepseekForCausalLM` | `deepseek_v2` | `DeepseekForCausalLM` | |
| `DeepseekV2ForCausalLM` | `deepseek_v2` | `DeepseekV2ForCausalLM` | |
| `DeepseekV3ForCausalLM` / `DeepseekV32ForCausalLM` | `deepseek_v2` | `DeepseekV3ForCausalLM` | |
| `Dots1ForCausalLM` | `dots1` | `Dots1ForCausalLM` | |
| `Ernie4_5ForCausalLM` | `ernie45` | `Ernie4_5ForCausalLM` | |
| `Ernie4_5_MoeForCausalLM` | `ernie45_moe` | `Ernie4_5_MoeForCausalLM` | |
| `ExaoneForCausalLM` | `exaone` | `ExaoneForCausalLM` | |
| `Exaone4ForCausalLM` | `exaone4` | `Exaone4ForCausalLM` | |
| `ExaoneMoEForCausalLM` | `exaone_moe` | `ExaoneMoeForCausalLM` | |
| `FalconForCausalLM` / `RWForCausalLM` | `falcon` | `FalconForCausalLM` | |
| `FalconMambaForCausalLM` | `mamba` | `MambaForCausalLM` | |
| `FalconH1ForCausalLM` | `falcon_h1` | `FalconH1ForCausalLM` | |
| `GemmaForCausalLM` | `gemma` | `GemmaForCausalLM` | Gemma 1 |
| `Gemma2ForCausalLM` | `gemma2` | `Gemma2ForCausalLM` | Gemma 2 |
| `Gemma3ForCausalLM` | `gemma3` | `Gemma3ForCausalLM` | Gemma 3 |
| `Gemma3nForCausalLM` | `gemma3n` | `Gemma3nForCausalLM` | Gemma 3n |
| `GlmForCausalLM` | `glm` | `GlmForCausalLM` | |
| `Glm4ForCausalLM` | `glm4` | `Glm4ForCausalLM` | GLM-4 |
| `Glm4MoeForCausalLM` | `glm4_moe` | `Glm4MoeForCausalLM` | |
| `GPT2LMHeadModel` | `gpt2` | `GPT2LMHeadModel` | |
| `GPTBigCodeForCausalLM` | `gpt_bigcode` | `GPTBigCodeForCausalLM` | StarCoder |
| `GPTJForCausalLM` | `gpt_j` | `GPTJForCausalLM` | |
| `GPTNeoXForCausalLM` | `gpt_neox` | `GPTNeoXForCausalLM` | |
| `GraniteForCausalLM` | `granite` | `GraniteForCausalLM` | IBM Granite |
| `GraniteMoeForCausalLM` | `granitemoe` | `GraniteMoeForCausalLM` | |
| `GritLM` | `gritlm` | `GritLM` | |
| `Grok1ForCausalLM` / `Grok1ModelForCausalLM` | `grok1` | `GrokForCausalLM` | |
| `HunYuanMoEV1ForCausalLM` | `hunyuan_v1` | `HunYuanMoEV1ForCausalLM` | |
| `InternLMForCausalLM` | `llama` | `LlamaForCausalLM` | InternLM 1 |
| `InternLM2ForCausalLM` | `internlm2` | `InternLM2ForCausalLM` | InternLM 2 |
| `InternLM3ForCausalLM` | `llama` | `LlamaForCausalLM` | InternLM 3 |
| `JAISLMHeadModel` | `jais` | `JAISLMHeadModel` | |
| `JambaForCausalLM` | `jamba` | `JambaForCausalLM` | Hybrid SSM |
| `KimiLinearForCausalLM` | `kimi_linear` | `KimiLinearForCausalLM` | |
| `Lfm2ForCausalLM` | `lfm2` | `Lfm2ForCausalLM` | |
| `LlamaForCausalLM` / `LLaMAForCausalLM` | `llama` | `LlamaForCausalLM` | Llama 1/2/3 |
| `Llama4ForCausalLM` | `llama4` | `Llama4ForCausalLM` | Llama 4 |
| `MambaForCausalLM` | `mamba` | `MambaForCausalLM` | Attention-free SSM |
| `Mamba2ForCausalLM` | `mamba2` | `Mamba2ForCausalLM` | |
| `MiniCPMForCausalLM` | `minicpm` | `MiniCPMForCausalLM` | |
| `MiniCPM3ForCausalLM` | `minicpm3` | `MiniCPM3ForCausalLM` | |
| `MiniMaxText01ForCausalLM` | `minimax_text_01` | `MiniMaxText01ForCausalLM` | |
| `MistralForCausalLM` | `mistral` | `MistralForCausalLM` | |
| `MistralLarge3ForCausalLM` | `mistral_large_3` | `MistralLarge3ForCausalLM` | |
| `MixtralForCausalLM` | `mixtral` | `MixtralForCausalLM` | MoE |
| `MPTForCausalLM` / `MptForCausalLM` | `mpt` | `MPTForCausalLM` | |
| `NemotronForCausalLM` | `nemotron` | `NemotronForCausalLM` | |
| `NemotronHForCausalLM` | `nemotron_h` | `NemotronHForCausalLM` | |
| `OlmoForCausalLM` | `olmo` | `OlmoForCausalLM` | |
| `Olmo2ForCausalLM` / `Olmo3ForCausalLM` | `olmo2` | `Olmo2ForCausalLM` | |
| `OlmoeForCausalLM` | `olmoe` | `OlmoeForCausalLM` | MoE |
| `OPTForCausalLM` | `opt` | `OPTForCausalLM` | |
| `OrionForCausalLM` | `orion` | `OrionForCausalLM` | |
| `PersimmonForCausalLM` | `persimmon` | `PersimmonForCausalLM` | |
| `PhiForCausalLM` | `phi` | `PhiForCausalLM` | Phi-1/2 |
| `Phi3ForCausalLM` | `phi3` | `Phi3ForCausalLM` | Phi-3 |
| `PhiMoEForCausalLM` | `phimoe` | `PhiMoEForCausalLM` | |
| `QWenLMHeadModel` | `qwen` | `QWenLMHeadModel` | Qwen 1 |
| `Qwen2ForCausalLM` | `qwen2` | `Qwen2ForCausalLM` | Qwen 2 |
| `Qwen2MoeForCausalLM` | `qwen2_moe` | `Qwen2MoeForCausalLM` | |
| `Qwen3ForCausalLM` | `qwen3` | `Qwen3ForCausalLM` | Qwen 3 |
| `Qwen3MoeForCausalLM` | `qwen3_moe` | `Qwen3MoeForCausalLM` | |
| `SarvamMoEForCausalLM` | `sarvam` | `SarvamMoEForCausalLM` | |
| `StableLMEpochForCausalLM` / `StableLmForCausalLM` | `stablelm` | `StablelmForCausalLM` | |
| `Starcoder2ForCausalLM` | `starcoder2` | `Starcoder2ForCausalLM` | |
| `TeleChat2ForCausalLM` | `telechat2` | `TeleChat2ForCausalLM` | |
| `XverseForCausalLM` | `llama` | `LlamaForCausalLM` | |
| `Zamba2ForCausalLM` | `zamba2` | `Zamba2ForCausalLM` | Hybrid SSM |

## Multimodal Models

These models implement `SupportsMultiModal` and can process images, video, and/or audio alongside text.

| Architecture Class | Module | Modalities |
|---|---|---|
| `AriaForConditionalGeneration` | `aria` | Image |
| `AudioFlamingo3ForConditionalGeneration` | `audioflamingo3` | Audio |
| `Blip2ForConditionalGeneration` | `blip2` | Image |
| `ChameleonForConditionalGeneration` | `chameleon` | Image |
| `DeepseekVLV2ForCausalLM` | `deepseek_vl2` | Image |
| `FuyuForCausalLM` | `fuyu` | Image |
| `Gemma3ForConditionalGeneration` | `gemma3_mm` | Image |
| `Gemma3nForConditionalGeneration` | `gemma3n_mm` | Image/Audio |
| `GLM4VForCausalLM` | `glm4v` | Image |
| `Glm4vForConditionalGeneration` | `glm4_1v` | Image |
| `GraniteSpeechForConditionalGeneration` | `granite_speech` | Audio |
| `H2OVLChatModel` | `h2ovl` | Image |
| `HunYuanVLForConditionalGeneration` | `hunyuan_vision` | Image |
| `Idefics3ForConditionalGeneration` | `idefics3` | Image |
| `InternVLChatModel` | `internvl` | Image |
| `KimiVLForConditionalGeneration` | `kimi_vl` | Image |
| `Llama4ForConditionalGeneration` | `mllama4` | Image |
| `LlavaForConditionalGeneration` | `llava` | Image |
| `LlavaNextForConditionalGeneration` | `llava_next` | Image |
| `LlavaNextVideoForConditionalGeneration` | `llava_next_video` | Image/Video |
| `LlavaOnevisionForConditionalGeneration` | `llava_onevision` | Image/Video |
| `MiniCPMO` | `minicpmo` | Image/Audio |
| `MiniCPMV` | `minicpmv` | Image |
| `Mistral3ForConditionalGeneration` | `mistral3` | Image |
| `MolmoForCausalLM` | `molmo` | Image |
| `NVLM_D` | `nvlm_d` | Image |
| `PaliGemmaForConditionalGeneration` | `paligemma` | Image |
| `Phi3VForCausalLM` | `phi3v` | Image |
| `Phi4MMForCausalLM` | `phi4mm` | Image/Audio |
| `PixtralForConditionalGeneration` | `pixtral` | Image |
| `Qwen2VLForConditionalGeneration` | `qwen2_vl` | Image/Video |
| `Qwen2_5_VLForConditionalGeneration` | `qwen2_5_vl` | Image/Video |
| `Qwen2AudioForConditionalGeneration` | `qwen2_audio` | Audio |
| `Qwen2_5OmniModel` | `qwen2_5_omni_thinker` | Image/Audio/Video |
| `Qwen3VLForConditionalGeneration` | `qwen3_vl` | Image/Video |
| `Qwen3ASRForConditionalGeneration` | `qwen3_asr` | Audio |
| `SmolVLMForConditionalGeneration` | `smolvlm` | Image |
| `UltravoxModel` | `ultravox` | Audio |
| `VoxtralForConditionalGeneration` | `voxtral` | Audio |
| `WhisperForConditionalGeneration` | `whisper` | Audio (Transcription) |

## Embedding / Pooling Models

These models implement `VllmModelForPooling` and produce dense or sparse vector representations.

| Architecture Class | Module | Type |
|---|---|---|
| `BertModel` | `bert` | Text embedding |
| `BertSpladeSparseEmbeddingModel` | `bert` | Sparse embedding |
| `HF_ColBERT` | `colbert` | Late interaction |
| `ColBERTModernBertModel` | `colbert` | Late interaction |
| `Gemma2Model` | `gemma2` | Text embedding |
| `GteModel` / `GteNewModel` | `bert_with_rope` | Text embedding |
| `InternLM2ForRewardModel` | `internlm2` | Reward model |
| `LlamaModel` | `llama` | Text embedding |
| `ModernBertModel` | `modernbert` | Text embedding |
| `NomicBertModel` | `bert_with_rope` | Text embedding |
| `Qwen2ForRewardModel` | `qwen2_rm` | Reward model |
| `Qwen2ForProcessRewardModel` | `qwen2_rm` | Process reward |
| `RobertaModel` / `RobertaForMaskedLM` | `roberta` | Text embedding |
| `XLMRobertaModel` | `roberta` | Multilingual embedding |
| `CLIPModel` | `clip` | Vision-text embedding |
| `SiglipModel` | `siglip` | Vision-text embedding |
| `ColQwen3` | `colqwen3` | Multimodal late interaction |
| `Qwen2VLForConditionalGeneration` | `qwen2_vl` | Multimodal embedding |

## Cross-Encoder Models

These models implement `SupportsCrossEncoding` for reranking tasks.

| Architecture Class | Module | Notes |
|---|---|---|
| `BertForSequenceClassification` | `bert` | |
| `BertForTokenClassification` | `bert` | |
| `ModernBertForSequenceClassification` | `modernbert` | |
| `RobertaForSequenceClassification` | `roberta` | |
| `XLMRobertaForSequenceClassification` | `roberta` | |
| `JinaVLForRanking` | `jina_vl` | Multimodal reranking |

## Speculative Decoding Models

These models are used as draft models for speculative decoding.

| Architecture Class | Module | Notes |
|---|---|---|
| `EagleLlamaForCausalLM` | `llama_eagle` | EAGLE draft for Llama |
| `Eagle3LlamaForCausalLM` | `llama_eagle3` | EAGLE3 draft for Llama |
| `EagleMistralLarge3ForCausalLM` | `mistral_large_3_eagle` | |
| `EagleDeepSeekMTPModel` | `deepseek_eagle` | |
| `DeepSeekMTPModel` | `deepseek_mtp` | Multi-token prediction |
| `MedusaModel` | `medusa` | Medusa heads |
| `ExtractHiddenStatesModel` | `extract_hidden_states` | |

## Transformers Backend Models

These architectures use the HuggingFace Transformers backend directly.

| Architecture Class | Notes |
|---|---|
| `TransformersForCausalLM` | Generic text generation via Transformers |
| `TransformersMoEForCausalLM` | MoE models via Transformers |
| `TransformersMultiModalForCausalLM` | Multimodal via Transformers |
| `TransformersEmbeddingModel` | Embedding via Transformers |
| `SmolLM3ForCausalLM` | Mapped to `TransformersForCausalLM` |
| `Emu3ForConditionalGeneration` | Mapped to `TransformersMultiModalForCausalLM` |

## Models by Family

### Llama Family

The Llama family is the most widely supported, with many architectures mapping to `LlamaForCausalLM`:

- **Llama 1/2/3**: `LlamaForCausalLM`, `LLaMAForCausalLM`
- **Llama 4**: `Llama4ForCausalLM` (with multimodal variant `Llama4ForConditionalGeneration`)
- **InternLM 1/3**: `InternLMForCausalLM`, `InternLM3ForCausalLM` → `LlamaForCausalLM`
- **Aquila**: `AquilaModel`, `AquilaForCausalLM` → `LlamaForCausalLM`
- **Xverse**: `XverseForCausalLM` → `LlamaForCausalLM`
- **CWM**: `CwmForCausalLM` → `LlamaForCausalLM`

### Mistral / Mixtral Family

- **Mistral**: `MistralForCausalLM`
- **Mixtral** (MoE): `MixtralForCausalLM`
- **Mistral Large 3**: `MistralLarge3ForCausalLM`
- **Mistral 3 VL**: `Mistral3ForConditionalGeneration`
- **Pixtral**: `PixtralForConditionalGeneration`

### Qwen Family

- **Qwen 1**: `QWenLMHeadModel`
- **Qwen 2**: `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`
- **Qwen 2 VL**: `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`
- **Qwen 2 Audio**: `Qwen2AudioForConditionalGeneration`
- **Qwen 2.5 Omni**: `Qwen2_5OmniModel`
- **Qwen 3**: `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`
- **Qwen 3 VL**: `Qwen3VLForConditionalGeneration`
- **Qwen 3 ASR**: `Qwen3ASRForConditionalGeneration`

### DeepSeek Family

- **DeepSeek V1**: `DeepseekForCausalLM`
- **DeepSeek V2**: `DeepseekV2ForCausalLM`
- **DeepSeek V3**: `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`
- **DeepSeek VL2**: `DeepseekVLV2ForCausalLM`
- **DeepSeek OCR**: `DeepseekOCRForCausalLM`, `DeepseekOCR2ForCausalLM`

### Gemma Family

- **Gemma 1**: `GemmaForCausalLM`
- **Gemma 2**: `Gemma2ForCausalLM`, `Gemma2Model` (embedding)
- **Gemma 3**: `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration` (multimodal)
- **Gemma 3n**: `Gemma3nForCausalLM`, `Gemma3nForConditionalGeneration`
- **PaliGemma**: `PaliGemmaForConditionalGeneration`

### GPT Family

- **GPT-2**: `GPT2LMHeadModel`
- **GPT-J**: `GPTJForCausalLM`
- **GPT-NeoX**: `GPTNeoXForCausalLM`
- **GPT-BigCode** (StarCoder): `GPTBigCodeForCausalLM`
- **Starcoder2**: `Starcoder2ForCausalLM`

### BERT / Encoder Family

- **BERT**: `BertModel`, `BertForSequenceClassification`
- **RoBERTa**: `RobertaModel`, `RobertaForMaskedLM`
- **ModernBERT**: `ModernBertModel`
- **ColBERT**: `HF_ColBERT`, `ColBERTModernBertModel`
- **GTE**: `GteModel`, `GteNewModel`
- **Nomic BERT**: `NomicBertModel`

### Falcon Family

- **Falcon**: `FalconForCausalLM`, `RWForCausalLM`
- **Falcon Mamba**: `FalconMambaForCausalLM`
- **Falcon H1**: `FalconH1ForCausalLM`

### Phi Family

- **Phi 1/2**: `PhiForCausalLM`
- **Phi 3**: `Phi3ForCausalLM`, `Phi3VForCausalLM` (vision)
- **Phi MoE**: `PhiMoEForCausalLM`
- **Phi 4 Multimodal**: `Phi4MMForCausalLM`

### Mamba / SSM Family

- **Mamba**: `MambaForCausalLM` (attention-free)
- **Mamba 2**: `Mamba2ForCausalLM`
- **Jamba**: `JambaForCausalLM` (hybrid attention+Mamba)
- **Bamba**: `BambaForCausalLM` (hybrid)
- **Zamba 2**: `Zamba2ForCausalLM` (hybrid)

### LLaVA Family

- **LLaVA**: `LlavaForConditionalGeneration`
- **LLaVA-NeXT**: `LlavaNextForConditionalGeneration`
- **LLaVA-NeXT-Video**: `LlavaNextVideoForConditionalGeneration`
- **LLaVA-OneVision**: `LlavaOnevisionForConditionalGeneration`
- **Mantis**: `MantisForConditionalGeneration`

### InternVL Family

- **InternVL**: `InternVLChatModel`
- **InternS1**: `InternS1ForConditionalGeneration`
- **InternS1 Pro**: `InternS1ProForConditionalGeneration`

### Granite Family (IBM)

- **Granite**: `GraniteForCausalLM`
- **Granite MoE**: `GraniteMoeForCausalLM`
- **Granite MoE Hybrid**: `GraniteMoeHybridForCausalLM`
- **Granite Speech**: `GraniteSpeechForConditionalGeneration`

### OLMo Family (AI2)

- **OLMo**: `OlmoForCausalLM`
- **OLMo 2/3**: `Olmo2ForCausalLM`
- **OLMoE**: `OlmoeForCausalLM`
- **OLMo Hybrid**: `OlmoHybridForCausalLM`

### Nemotron Family (NVIDIA)

- **Nemotron**: `NemotronForCausalLM`
- **Nemotron-H**: `NemotronHForCausalLM`
- **Nemotron NAS**: `DeciLMForCausalLM`
- **Nemotron VL**: `Llama_Nemotron_Nano_VL`

## Previously Supported Models

Some architectures were supported in earlier versions of vLLM but have since been removed:

| Architecture | Last Supported Version |
|---|---|
| `MotifForCausalLM` | v0.10.2 |
| `Phi3SmallForCausalLM` | v0.9.2 |
| `Phi4FlashForCausalLM` | v0.10.2 |
| `Phi4MultimodalForCausalLM` | v0.12.0 |
| `BartModel` / `BartForConditionalGeneration` | v0.10.2 |
| `MllamaForConditionalGeneration` | v0.10.2 |

> If you attempt to load one of these architectures, vLLM will raise a descriptive error indicating the last supported version.

## See Also

- [Model Registry Internals](registry-internals.md)
- [Model Interfaces](model-interfaces.md)
- [Adding a New Model](adding-new-model.md)
- [Model Loading](model-loading.md)
- [Multimodal Models](multimodal-models.md)
