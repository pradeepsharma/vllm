# Supported Models

vLLM supports **150+ model architectures** across text generation, embedding, multimodal, encoder-decoder, and speculative decoding categories. Models are identified by the `architectures` field in their HuggingFace `config.json`.

!!! tip "Finding your model's architecture"
    Run `python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('your/model').architectures)"` to find the architecture string for any HuggingFace model.

---

## Text Generation Models (Decoder-Only)

These models perform autoregressive text generation (causal language modeling). They are the most common model type in vLLM.

| Architecture String | Model Family | Example Models | LoRA | PP |
|---|---|---|---|---|
| `AfmoeForCausalLM` | AFMoE | AFMoE | ✓ | ✓ |
| `ApertusForCausalLM` | Apertus | Apertus | ✓ | ✓ |
| `AquilaModel` / `AquilaForCausalLM` | Aquila | BAAI/Aquila-7B, AquilaChat2 | ✓ | ✓ |
| `ArceeForCausalLM` | Arcee | Arcee models | ✓ | ✓ |
| `ArcticForCausalLM` | Arctic | Snowflake Arctic | ✓ | ✓ |
| `BaiChuanForCausalLM` / `BaichuanForCausalLM` | Baichuan | baichuan-inc/Baichuan-7B, Baichuan2-13B | ✓ | ✓ |
| `BailingMoeForCausalLM` | Bailing MoE | Bailing MoE | ✓ | ✓ |
| `BailingMoeV2ForCausalLM` | Bailing MoE v2 | Bailing MoE v2 | ✓ | ✓ |
| `BailingMoeV2_5ForCausalLM` | Bailing MoE v2.5 | Bailing MoE v2.5 | ✓ | ✓ |
| `BambaForCausalLM` | Bamba | IBM Bamba | ✓ | ✓ |
| `BloomForCausalLM` | BLOOM | bigscience/bloom | ✓ | ✓ |
| `ChatGLMModel` / `ChatGLMForConditionalGeneration` | ChatGLM | THUDM/chatglm3-6b | ✓ | ✓ |
| `CohereForCausalLM` / `Cohere2ForCausalLM` | Command R | CohereForAI/c4ai-command-r-v01 | ✓ | ✓ |
| `DbrxForCausalLM` | DBRX | databricks/dbrx-base | ✓ | ✓ |
| `DeciLMForCausalLM` | DeciLM / Nemotron-NAS | nvidia/Llama-3_1-Nemotron-51B-Instruct | ✓ | ✓ |
| `DeepseekForCausalLM` | DeepSeek | deepseek-ai/deepseek-llm-7b-base | ✓ | ✓ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | deepseek-ai/DeepSeek-V2 | ✓ | ✓ |
| `DeepseekV3ForCausalLM` / `DeepseekV32ForCausalLM` | DeepSeek-V3 | deepseek-ai/DeepSeek-V3 | ✓ | ✓ |
| `Dots1ForCausalLM` | Dots1 | rednote-hilab/dots.llm1 | ✓ | ✓ |
| `Ernie4_5ForCausalLM` | ERNIE 4.5 | baidu/ERNIE-4.5 | ✓ | ✓ |
| `Ernie4_5_MoeForCausalLM` | ERNIE 4.5 MoE | baidu/ERNIE-4.5-MoE | ✓ | ✓ |
| `ExaoneForCausalLM` | EXAONE | lgai-exaone/EXAONE-3.5-7.8B-Instruct | ✓ | ✓ |
| `Exaone4ForCausalLM` | EXAONE 4 | lgai-exaone/EXAONE-4.0 | ✓ | ✓ |
| `ExaoneMoEForCausalLM` | EXAONE MoE | lgai-exaone/EXAONE-MoE | ✓ | ✓ |
| `FalconForCausalLM` / `RWForCausalLM` | Falcon | tiiuae/falcon-7b | ✓ | ✓ |
| `FalconMambaForCausalLM` | Falcon Mamba | tiiuae/falcon-mamba-7b | ✓ | ✓ |
| `FalconH1ForCausalLM` | Falcon H1 | tiiuae/Falcon-H1 | ✓ | ✓ |
| `GemmaForCausalLM` | Gemma | google/gemma-7b | ✓ | ✓ |
| `Gemma2ForCausalLM` | Gemma 2 | google/gemma-2-9b | ✓ | ✓ |
| `Gemma3ForCausalLM` | Gemma 3 | google/gemma-3-27b-it | ✓ | ✓ |
| `Gemma3nForCausalLM` | Gemma 3n | google/gemma-3n | ✓ | ✓ |
| `GlmForCausalLM` | GLM | THUDM/glm-4-9b | ✓ | ✓ |
| `Glm4ForCausalLM` | GLM-4 | THUDM/glm-4-9b-chat | ✓ | ✓ |
| `Glm4MoeForCausalLM` | GLM-4 MoE | THUDM/GLM-4-MoE | ✓ | ✓ |
| `Glm4MoeLiteForCausalLM` | GLM-4 MoE Lite | THUDM/GLM-4-MoE-Lite | ✓ | ✓ |
| `GPT2LMHeadModel` | GPT-2 | openai-community/gpt2 | ✓ | ✓ |
| `GPTBigCodeForCausalLM` | StarCoder / SantaCoder | bigcode/starcoder | ✓ | ✓ |
| `GPTJForCausalLM` | GPT-J | EleutherAI/gpt-j-6b | ✓ | ✓ |
| `GPTNeoXForCausalLM` | GPT-NeoX | EleutherAI/gpt-neox-20b | ✓ | ✓ |
| `GraniteForCausalLM` | Granite | ibm-granite/granite-3.0-8b-instruct | ✓ | ✓ |
| `GraniteMoeForCausalLM` | Granite MoE | ibm-granite/granite-3.0-3b-a800m | ✓ | ✓ |
| `GraniteMoeHybridForCausalLM` | Granite MoE Hybrid | ibm-granite/granite-4.0-tiny-preview | ✓ | ✓ |
| `GraniteMoeSharedForCausalLM` | Granite MoE Shared | ibm-granite/granite-moe-shared | ✓ | ✓ |
| `Grok1ModelForCausalLM` / `Grok1ForCausalLM` | Grok-1 | xai-org/grok-1 | ✓ | ✓ |
| `HunYuanMoEV1ForCausalLM` | HunYuan MoE | tencent/Hunyuan-A13B-Instruct | ✓ | ✓ |
| `HunYuanDenseV1ForCausalLM` | HunYuan Dense | tencent/Hunyuan-Dense | ✓ | ✓ |
| `InternLMForCausalLM` | InternLM | internlm/internlm-7b | ✓ | ✓ |
| `InternLM2ForCausalLM` | InternLM2 | internlm/internlm2-7b | ✓ | ✓ |
| `InternLM3ForCausalLM` | InternLM3 | internlm/internlm3-8b-instruct | ✓ | ✓ |
| `JAISLMHeadModel` | JAIS | core42/jais-13b | ✓ | ✓ |
| `Jais2ForCausalLM` | JAIS 2 | core42/jais2 | ✓ | ✓ |
| `JambaForCausalLM` | Jamba | ai21labs/Jamba-v0.1 | ✓ | ✓ |
| `KimiLinearForCausalLM` | Kimi Linear | moonshot-ai/Kimi-Linear | ✓ | ✓ |
| `Lfm2ForCausalLM` | LFM-2 | LiquidAI/LFM-2 | ✓ | ✓ |
| `Lfm2MoeForCausalLM` | LFM-2 MoE | LiquidAI/LFM-2-MoE | ✓ | ✓ |
| `LlamaForCausalLM` / `LLaMAForCausalLM` | Llama / Llama 2 / Llama 3 | meta-llama/Llama-3.1-8B-Instruct | ✓ | ✓ |
| `Llama4ForCausalLM` | Llama 4 | meta-llama/Llama-4-Scout-17B-16E | ✓ | ✓ |
| `MambaForCausalLM` / `FalconMambaForCausalLM` | Mamba | state-spaces/mamba-2.8b | ✓ | ✗ |
| `Mamba2ForCausalLM` | Mamba 2 | state-spaces/mamba2-2.7b | ✓ | ✗ |
| `MiniCPMForCausalLM` | MiniCPM | openbmb/MiniCPM-2B-sft-bf16 | ✓ | ✓ |
| `MiniCPM3ForCausalLM` | MiniCPM3 | openbmb/MiniCPM3-4B | ✓ | ✓ |
| `MiniMaxText01ForCausalLM` | MiniMax Text-01 | MiniMaxAI/MiniMax-Text-01 | ✓ | ✓ |
| `MiniMaxM2ForCausalLM` | MiniMax M2 | MiniMaxAI/MiniMax-M2 | ✓ | ✓ |
| `MistralForCausalLM` | Mistral / Mistral v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | ✓ | ✓ |
| `MistralLarge3ForCausalLM` | Mistral Large 3 | mistralai/Mistral-Large-Instruct-2411 | ✓ | ✓ |
| `MixtralForCausalLM` | Mixtral | mistralai/Mixtral-8x7B-Instruct-v0.1 | ✓ | ✓ |
| `MptForCausalLM` / `MPTForCausalLM` | MPT | mosaicml/mpt-7b | ✓ | ✓ |
| `MiMoForCausalLM` | MiMo | xiaomi/MiMo-7B | ✓ | ✓ |
| `NemotronForCausalLM` | Nemotron | nvidia/Nemotron-4-340B-Instruct | ✓ | ✓ |
| `NemotronHForCausalLM` | Nemotron-H | nvidia/Nemotron-H-8B-Base | ✓ | ✓ |
| `OlmoForCausalLM` | OLMo | allenai/OLMo-7B | ✓ | ✓ |
| `Olmo2ForCausalLM` | OLMo 2 | allenai/OLMo-2-1124-7B | ✓ | ✓ |
| `Olmo3ForCausalLM` | OLMo 3 | allenai/OLMo-3 | ✓ | ✓ |
| `OlmoeForCausalLM` | OLMoE | allenai/OLMoE-1B-7B-0924 | ✓ | ✓ |
| `OPTForCausalLM` | OPT | facebook/opt-66b | ✓ | ✓ |
| `OrionForCausalLM` | Orion | OrionStarAI/Orion-14B-Base | ✓ | ✓ |
| `PersimmonForCausalLM` | Persimmon | adept/persimmon-8b-base | ✓ | ✓ |
| `PhiForCausalLM` | Phi-1.5 / Phi-2 | microsoft/phi-2 | ✓ | ✓ |
| `Phi3ForCausalLM` | Phi-3 / Phi-3.5 | microsoft/Phi-3-mini-4k-instruct | ✓ | ✓ |
| `PhiMoEForCausalLM` | Phi-3.5 MoE | microsoft/Phi-3.5-MoE-instruct | ✓ | ✓ |
| `Plamo2ForCausalLM` | PLaMo-2 | pfnet/plamo-2 | ✓ | ✓ |
| `Plamo3ForCausalLM` | PLaMo-3 | pfnet/plamo-3 | ✓ | ✓ |
| `QWenLMHeadModel` | Qwen 1.x | Qwen/Qwen-7B | ✓ | ✓ |
| `Qwen2ForCausalLM` | Qwen2 / Qwen2.5 | Qwen/Qwen2.5-7B-Instruct | ✓ | ✓ |
| `Qwen2MoeForCausalLM` | Qwen2 MoE | Qwen/Qwen2-57B-A14B-Instruct | ✓ | ✓ |
| `Qwen3ForCausalLM` | Qwen3 | Qwen/Qwen3-8B | ✓ | ✓ |
| `Qwen3MoeForCausalLM` | Qwen3 MoE | Qwen/Qwen3-30B-A3B | ✓ | ✓ |
| `SarvamMoEForCausalLM` | Sarvam MoE | sarvamai/sarvam-m | ✓ | ✓ |
| `StableLMEpochForCausalLM` / `StableLmForCausalLM` | StableLM | stabilityai/stablelm-3b-4e1t | ✓ | ✓ |
| `Starcoder2ForCausalLM` | StarCoder2 | bigcode/starcoder2-15b | ✓ | ✓ |
| `SolarForCausalLM` | SOLAR | upstage/SOLAR-10.7B-v1.0 | ✓ | ✓ |
| `TeleChat2ForCausalLM` | TeleChat2 | Tele-AI/TeleChat2 | ✓ | ✓ |
| `XverseForCausalLM` | XVERSE | xverse/XVERSE-13B | ✓ | ✓ |
| `Zamba2ForCausalLM` | Zamba2 | Zyphra/Zamba2-7B | ✓ | ✓ |

> **PP** = Pipeline Parallelism supported. **LoRA** = LoRA fine-tuning supported.

---

## Embedding & Pooling Models

These models produce vector representations for semantic search, retrieval, and ranking tasks.

| Architecture String | Model Family | Example Models | Modality |
|---|---|---|---|
| `BertModel` | BERT | bert-base-uncased | Text |
| `BertSpladeSparseEmbeddingModel` | BERT SPLADE | naver/splade-v3 | Text |
| `HF_ColBERT` | ColBERT | colbert-ir/colbertv2.0 | Text |
| `ColBERTModernBertModel` | ColBERT ModernBERT | — | Text |
| `ColBERTJinaRobertaModel` | ColBERT Jina | jinaai/jina-colbert-v2 | Text |
| `Gemma2Model` | Gemma 2 Embedding | google/gemma-2-9b | Text |
| `GritLM` | GritLM | GritLM/GritLM-7B | Text |
| `GteModel` / `GteNewModel` | GTE / Snowflake Arctic | Snowflake/snowflake-arctic-embed-l-v2.0 | Text |
| `InternLM2ForRewardModel` | InternLM2 Reward | internlm/internlm2-7b-reward | Text |
| `JambaForSequenceClassification` | Jamba Classifier | ai21labs/Jamba-v0.1 | Text |
| `LlamaModel` / `LlamaBidirectionalModel` | Llama Embedding | meta-llama/Llama-3.1-8B | Text |
| `MistralModel` | Mistral Embedding | mistralai/Mistral-7B-v0.1 | Text |
| `ModernBertModel` | ModernBERT | answerdotai/ModernBERT-base | Text |
| `NomicBertModel` | Nomic Embed | nomic-ai/nomic-embed-text-v1.5 | Text |
| `Phi3ForCausalLM` | Phi-3 Embedding | microsoft/Phi-3-mini-4k-instruct | Text |
| `Qwen2Model` / `Qwen2ForCausalLM` | Qwen2 Embedding | Qwen/Qwen2.5-7B | Text |
| `Qwen2ForRewardModel` | Qwen2 Reward | Qwen/Qwen2.5-Math-RM-72B | Text |
| `Qwen2ForProcessRewardModel` | Qwen2 PRM | Qwen/Qwen2.5-Math-PRM-7B | Text |
| `RobertaModel` / `RobertaForMaskedLM` | RoBERTa | FacebookAI/roberta-large | Text |
| `XLMRobertaModel` | XLM-RoBERTa | FacebookAI/xlm-roberta-large | Text |
| `BgeM3EmbeddingModel` | BGE-M3 | BAAI/bge-m3 | Text |
| `VoyageQwen3BidirectionalEmbedModel` | Voyage Qwen3 | voyageai/voyage-qwen3 | Text |
| `CLIPModel` | CLIP | openai/clip-vit-large-patch14 | Image |
| `SiglipModel` | SigLIP | google/siglip-so400m-patch14-384 | Image |
| `ColQwen3` / `OpsColQwen3Model` | ColQwen3 | Qwen/Qwen3-VL | Image+Text |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT Embedding | llava-hf/llava-v1.6-mistral-7b-hf | Image+Text |
| `Phi3VForCausalLM` | Phi-3 Vision Embedding | microsoft/Phi-3-vision-128k-instruct | Image+Text |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL Embedding | Qwen/Qwen2-VL-7B-Instruct | Image+Text |
| `LlamaNemotronVLModel` | Llama Nemotron VL | nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 | Image+Text |
| `ColModernVBertForRetrieval` | ColModernVBert | — | Image+Text |
| `PrithviGeoSpatialMAE` / `Terratorch` | Terratorch | ibm-nasa-geospatial/Prithvi-EO-2.0 | Image |

---

## Cross-Encoder / Reranker Models

These models score query-document pairs for reranking in retrieval pipelines.

| Architecture String | Model Family | Example Models |
|---|---|---|
| `BertForSequenceClassification` | BERT Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `BertForTokenClassification` | BERT Token Classifier | — |
| `GteNewForSequenceClassification` | GTE Reranker | Alibaba-NLP/gte-reranker-modernbert-8b |
| `JinaVLForRanking` | Jina VL Reranker | jinaai/jina-reranker-m0 |
| `LlamaBidirectionalForSequenceClassification` | Llama Bidirectional Reranker | — |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron VL Reranker | nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 |
| `ModernBertForSequenceClassification` | ModernBERT Reranker | answerdotai/ModernBERT-large |
| `ModernBertForTokenClassification` | ModernBERT Token Classifier | — |
| `RobertaForSequenceClassification` | RoBERTa Reranker | cross-encoder/stsb-roberta-large |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa Reranker | — |

---

## Multimodal Models

These models process one or more non-text modalities (images, video, audio) alongside text.

### Vision-Language Models (Image + Text)

| Architecture String | Model Family | Example Models |
|---|---|---|
| `AriaForConditionalGeneration` | Aria | rhymes-ai/Aria |
| `AyaVisionForConditionalGeneration` | Aya Vision | CohereForAI/aya-vision-8b |
| `BagelForConditionalGeneration` | Bagel | ByteDance-Seed/BAGEL-34B-V0.1 |
| `Blip2ForConditionalGeneration` | BLIP-2 | Salesforce/blip2-opt-2.7b |
| `ChameleonForConditionalGeneration` | Chameleon | facebook/chameleon-7b |
| `Cohere2VisionForConditionalGeneration` | Cohere2 Vision | CohereForAI/c4ai-aya-vision |
| `DeepseekVLV2ForCausalLM` | DeepSeek-VL2 | deepseek-ai/deepseek-vl2 |
| `Eagle2_5_VLForConditionalGeneration` | Eagle2.5-VL | NovaSky-Berkeley/Eagle2.5-VL |
| `Ernie4_5_VLMoeForConditionalGeneration` | ERNIE 4.5 VL MoE | baidu/ERNIE-4.5-VL-MoE |
| `FuyuForCausalLM` | Fuyu | adept/fuyu-8b |
| `Gemma3ForConditionalGeneration` | Gemma 3 Multimodal | google/gemma-3-27b-it |
| `Gemma3nForConditionalGeneration` | Gemma 3n Multimodal | google/gemma-3n |
| `GLM4VForCausalLM` | GLM-4V | THUDM/glm-4v-9b |
| `Glm4vForConditionalGeneration` | GLM-4.1V | THUDM/GLM-4.1V-9B-Thinking |
| `H2OVLChatModel` | H2OVL | h2oai/h2ovl-mississippi-800m |
| `HunYuanVLForConditionalGeneration` | HunYuan VL | tencent/HunyuanVideo |
| `Idefics3ForConditionalGeneration` | IDEFICS3 / SmolVLM | HuggingFaceM4/Idefics3-8B-Llama3 |
| `InternVLChatModel` | InternVL | OpenGVLab/InternVL2-8B |
| `InternS1ForConditionalGeneration` | InternS1 | OpenGVLab/InternS1 |
| `KananaVForConditionalGeneration` | Kanana-V | kakaoai/kanana-1.5-8b-vision |
| `KeyeForConditionalGeneration` | Keye | Kwai-Keye/Keye-VL-8B-Instruct |
| `KimiVLForConditionalGeneration` | Kimi VL | moonshot-ai/Kimi-VL-A3B-Instruct |
| `KimiK25ForConditionalGeneration` | Kimi K2.5 | moonshot-ai/Kimi-K2.5 |
| `Lfm2VlForConditionalGeneration` | LFM-2 VL | LiquidAI/LFM-2-VL |
| `Llama4ForConditionalGeneration` | Llama 4 Multimodal | meta-llama/Llama-4-Scout-17B-16E |
| `LlavaForConditionalGeneration` | LLaVA-1.5 | llava-hf/llava-1.5-7b-hf |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | llava-hf/llava-v1.6-mistral-7b-hf |
| `LlavaNextVideoForConditionalGeneration` | LLaVA-NeXT-Video | llava-hf/LLaVA-NeXT-Video-7B-hf |
| `LlavaOnevisionForConditionalGeneration` | LLaVA-OneVision | llava-hf/llava-onevision-qwen2-7b-ov-hf |
| `MantisForConditionalGeneration` | Mantis | TIGER-Lab/Mantis-8B-siglip-llama3 |
| `MiniCPMV` | MiniCPM-V | openbmb/MiniCPM-V-2_6 |
| `MiniCPMO` | MiniCPM-o | openbmb/MiniCPM-o-2_6 |
| `MiniMaxVL01ForConditionalGeneration` | MiniMax VL-01 | MiniMaxAI/MiniMax-VL-01 |
| `Mistral3ForConditionalGeneration` | Mistral 3 Multimodal | mistralai/Mistral-Small-3.1-24B-Instruct-2503 |
| `MolmoForCausalLM` | Molmo | allenai/Molmo-7B-D-0924 |
| `Molmo2ForConditionalGeneration` | Molmo 2 | allenai/Molmo-2 |
| `NVLM_D` | NVLM-D | nvidia/NVLM-D-72B |
| `OpenCUAForConditionalGeneration` | OpenCUA | — |
| `Ovis` | Ovis | AIDC-AI/Ovis1.6-Gemma2-9B |
| `Ovis2_5` | Ovis 2.5 | AIDC-AI/Ovis2.5 |
| `PaddleOCRVLForConditionalGeneration` | PaddleOCR VL | PaddlePaddle/PaddleOCR-VL |
| `PaliGemmaForConditionalGeneration` | PaliGemma | google/paligemma-3b-pt-224 |
| `Phi3VForCausalLM` | Phi-3 Vision | microsoft/Phi-3-vision-128k-instruct |
| `Phi4MMForCausalLM` | Phi-4 Multimodal | microsoft/phi-4-multimodal-instruct |
| `PixtralForConditionalGeneration` | Pixtral | mistralai/Pixtral-12B-2409 |
| `QwenVLForConditionalGeneration` | Qwen-VL | Qwen/Qwen-VL-Chat |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL | Qwen/Qwen2-VL-7B-Instruct |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | Qwen/Qwen2.5-VL-7B-Instruct |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL | Qwen/Qwen3-VL |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL MoE | Qwen/Qwen3-VL-MoE |
| `SkyworkR1VChatModel` | Skywork R1-V | Skywork/Skywork-R1V |
| `SmolVLMForConditionalGeneration` | SmolVLM | HuggingFaceTB/SmolVLM-Instruct |
| `Step3VLForConditionalGeneration` | Step3-VL | stepfun-ai/Step3-VL |
| `StepVLForConditionalGeneration` | Step-VL | stepfun-ai/Step-VL |
| `TarsierForConditionalGeneration` | Tarsier | omni-research/Tarsier2-Recap-7B |
| `Tarsier2ForConditionalGeneration` | Tarsier2 | omni-research/Tarsier2 |

### Audio-Language Models

| Architecture String | Model Family | Example Models |
|---|---|---|
| `AudioFlamingo3ForConditionalGeneration` | AudioFlamingo3 | — |
| `MusicFlamingoForConditionalGeneration` | MusicFlamingo | — |
| `FunASRForConditionalGeneration` | FunASR | — |
| `FunAudioChatForConditionalGeneration` | FunAudioChat | — |
| `GlmAsrForConditionalGeneration` | GLM ASR | THUDM/glm-asr |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | ibm-granite/granite-speech-3.3-8b |
| `MiDashengLMModel` | MiDasheng | — |
| `Qwen2AudioForConditionalGeneration` | Qwen2-Audio | Qwen/Qwen2-Audio-7B-Instruct |
| `UltravoxModel` | Ultravox | fixie-ai/ultravox-v0_5 |
| `VoxtralForConditionalGeneration` | Voxtral | mistralai/Voxtral-Mini-3B-2507 |
| `VoxtralRealtimeGeneration` | Voxtral Realtime | mistralai/Voxtral-Realtime |

### Omni Models (Audio + Vision + Text)

| Architecture String | Model Family | Example Models |
|---|---|---|
| `Qwen2_5OmniModel` / `Qwen2_5OmniForConditionalGeneration` | Qwen2.5-Omni | Qwen/Qwen2.5-Omni-7B |
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni MoE | Qwen/Qwen3-Omni-MoE |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5 | Qwen/Qwen3.5 |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5 MoE | Qwen/Qwen3.5-MoE |

---

## Encoder-Decoder Models

These models use a separate encoder and decoder, typically for sequence-to-sequence tasks.

| Architecture String | Model Family | Task | Example Models |
|---|---|---|---|
| `WhisperForConditionalGeneration` | Whisper | Automatic Speech Recognition | openai/whisper-large-v3 |
| `NemotronParseForConditionalGeneration` | Nemotron Parse | Document Parsing | nvidia/Nemotron-Parse |
| `FireRedASR2ForConditionalGeneration` | FireRedASR2 | Automatic Speech Recognition | — |
| `Qwen3ASRForConditionalGeneration` | Qwen3 ASR | Automatic Speech Recognition | Qwen/Qwen3-ASR |
| `Qwen3ASRRealtimeGeneration` | Qwen3 ASR Realtime | Real-time ASR | Qwen/Qwen3-ASR-Realtime |

---

## Speculative Decoding Models

These models serve as draft models or multi-token prediction (MTP) heads to accelerate inference via speculative decoding.

| Architecture String | Draft Type | Base Model |
|---|---|---|
| `EagleLlamaForCausalLM` | EAGLE | Llama |
| `Eagle3LlamaForCausalLM` | EAGLE-3 | Llama |
| `EagleLlama4ForCausalLM` | EAGLE | Llama 4 |
| `EagleMiniCPMForCausalLM` | EAGLE | MiniCPM |
| `EagleMistralLarge3ForCausalLM` | EAGLE | Mistral Large 3 |
| `EagleDeepSeekMTPModel` | EAGLE | DeepSeek-V3 |
| `DeepSeekMTPModel` | MTP | DeepSeek-V3 |
| `MedusaModel` | Medusa | Any |
| `MiMoMTPModel` | MTP | MiMo |
| `ExtractHiddenStatesModel` | Hidden State Extraction | Any |

---

## Transformers Backend Models

For models not natively implemented in vLLM, the Transformers backend provides a compatibility layer:

| Architecture String | Task |
|---|---|
| `TransformersForCausalLM` | Text Generation |
| `TransformersMoEForCausalLM` | MoE Text Generation |
| `TransformersMultiModalForCausalLM` | Multimodal Generation |
| `TransformersEmbeddingModel` | Embedding |
| `TransformersForSequenceClassification` | Classification |

Additionally, these HuggingFace architectures are automatically routed to the Transformers backend:

| Architecture String | Model |
|---|---|
| `SmolLM3ForCausalLM` | SmolLM3 |
| `Emu3ForConditionalGeneration` | Emu3 |

---

## Quantization Compatibility

Most text generation and multimodal models support the following quantization formats:

| Format | INT4 | INT8 | FP8 | GGUF |
|---|---|---|---|---|
| AWQ | ✓ | — | — | — |
| GPTQ | ✓ | ✓ | — | — |
| BitsAndBytes | ✓ | ✓ | — | — |
| FP8 (W8A8) | — | — | ✓ | — |
| GGUF | — | — | — | ✓ |
| Marlin | ✓ | — | — | — |
| AQLM | ✓ | — | — | — |
| SqueezeLLM | ✓ | — | — | — |

!!! note "Quantization availability"
    Not all quantization formats are available for all models. Embedding models and cross-encoders have more limited quantization support. Check the model card on HuggingFace for pre-quantized variants.

---

## Previously Supported Models

The following architectures were supported in earlier versions of vLLM but have since been removed:

| Architecture | Last Supported Version | Reason |
|---|---|---|
| `MotifForCausalLM` | 0.10.2 | Deprecated |
| `Phi3SmallForCausalLM` | 0.9.2 | Deprecated |
| `Phi4FlashForCausalLM` | 0.10.2 | Deprecated |
| `Phi4MultimodalForCausalLM` | 0.12.0 | Replaced by `Phi4MMForCausalLM` |
| `BartModel` / `BartForConditionalGeneration` | 0.10.2 | V0 deprecation |
| `MBartForConditionalGeneration` | 0.10.2 | V0 deprecation |
| `MllamaForConditionalGeneration` | 0.10.2 | V0 deprecation |
| `Florence2ForConditionalGeneration` | 0.10.2 | V0 deprecation |

---

## Adding a New Model

Don't see your model? See the [Adding a New Model](adding_model.md) guide for step-by-step instructions on implementing and registering a new architecture in vLLM.
