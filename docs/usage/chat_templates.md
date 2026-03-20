# Chat Templates

Chat templates tell vLLM how to format a list of messages into a single text prompt that the
model understands. They are [Jinja2](https://jinja.palletsprojects.com/) templates stored either
inside the model's tokenizer configuration on HuggingFace Hub or as standalone `.jinja` files.

## Why Chat Templates Matter

Language models are trained on raw text, not on structured message objects. A chat template
bridges the gap: it takes a list of `{"role": ..., "content": ...}` dicts and renders them into
the exact token sequence the model was fine-tuned on. Using the wrong template — or no template
at all — produces garbled or off-topic responses even from a well-trained model.

## How vLLM Selects a Chat Template

When you start a vLLM server or create an `LLM` object, vLLM selects a chat template in this
order:

1. **Explicit `--chat-template` flag** (highest priority) — a path to a `.jinja` file or an
   inline template string.
2. **Tokenizer's built-in template** — loaded automatically from `tokenizer_config.json` on
   HuggingFace Hub.
3. **No template** — the server cannot process chat requests and returns an error for every
   `/v1/chat/completions` call.

## Using a Built-In Template

Most modern instruction-tuned models ship with a chat template in their tokenizer config. vLLM
loads it automatically — no extra configuration needed:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct
```

You can verify which template is active by checking the startup logs for a line like:

```
Using chat template from tokenizer: ...
```

## Specifying a Custom Template

### Via CLI (server)

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

You can also pass the template as an inline string (useful for quick tests):

```bash
vllm serve <model> --chat-template \
  "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
```

### Via Python (offline inference)

```python
from vllm import LLM

llm = LLM(
    model="<model>",
    tokenizer_mode="auto",
)

# Apply a custom template at generation time
outputs = llm.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    chat_template=open("./my-template.jinja").read(),
)
```

## Built-In Chat Templates

vLLM ships a collection of ready-to-use chat templates in the
[`examples/`](https://github.com/vllm-project/vllm/tree/main/examples) directory. These cover
models that either lack a built-in template or whose built-in template does not support tool
calling.

### Plain Chat Templates

These templates handle standard `system` / `user` / `assistant` conversations without tool
calling.

#### `template_chatml.jinja`

**Model family**: ChatML — Qwen, InternLM, and many fine-tunes  
**Format**: `<|im_start|>role\ncontent<|im_end|>`

The most widely used open-source chat format. Each turn is wrapped in `<|im_start|>` and
`<|im_end|>` special tokens with the role on the first line.

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>assistant
```

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --chat-template examples/template_chatml.jinja
```

#### `template_alpaca.jinja`

**Model family**: Alpaca-style instruction-tuned models  
**Format**: `### Instruction:` / `### Response:`

Renders user messages under `### Instruction:` and assistant messages under `### Response:`.
Supports an optional `user_context` role rendered under `### Input:`. System messages are
extracted and placed at the top.

```
Below is an instruction that describes a task.

### Instruction:
What is the capital of France?

### Response:
Paris.

### Instruction:
```

```bash
vllm serve tatsu-lab/alpaca-7b \
    --chat-template examples/template_alpaca.jinja
```

#### `template_baichuan.jinja`

**Model family**: Baichuan (1 and 2)  
**Format**: `<reserved_106>` / `<reserved_107>` special tokens

Uses Baichuan's reserved special tokens as turn delimiters. System messages are placed at the
top without a delimiter.

```bash
vllm serve baichuan-inc/Baichuan2-13B-Chat \
    --chat-template examples/template_baichuan.jinja
```

#### `template_chatglm.jinja`

**Model family**: ChatGLM (v1)  
**Format**: `[Round N]\n问：content\n答：content`

Chinese-first round-based format. Each user turn is numbered starting from 0 and prefixed with
`问：` (Question:); assistant turns are prefixed with `答：` (Answer:).

```bash
vllm serve THUDM/chatglm-6b \
    --chat-template examples/template_chatglm.jinja
```

#### `template_chatglm2.jinja`

**Model family**: ChatGLM2 and ChatGLM3  
**Format**: `[Round N]\n\n问：content\n\n答：content`

Updated ChatGLM format with double newlines between sections. Round numbering starts at 1
instead of 0.

```bash
vllm serve THUDM/chatglm2-6b \
    --chat-template examples/template_chatglm2.jinja
```

#### `template_falcon.jinja`

**Model family**: Falcon 7B, 40B  
**Format**: `User: content\nAssistant: content`

Simple prefix-based format. No special tokens — just plain text prefixes.

```bash
vllm serve tiiuae/falcon-7b-instruct \
    --chat-template examples/template_falcon.jinja
```

#### `template_falcon_180b.jinja`

**Model family**: Falcon 180B  
**Format**: `User: content\nFalcon: content`

Variant of the Falcon template for the 180B model. The assistant prefix is `Falcon:` instead
of `Assistant:`. Also supports a `System:` prefix for system messages.

```bash
vllm serve tiiuae/falcon-180B-chat \
    --chat-template examples/template_falcon_180b.jinja
```

#### `template_inkbot.jinja`

**Model family**: InkBot  
**Format**: `<#meta#>` / `<#system#>` / `<#chat#>` / `<#user#>` / `<#bot#>` tags

Custom tag-based format with metadata support. Supports special `meta-current_date` and
`meta-task_name` roles for injecting metadata into the prompt header.

```bash
vllm serve <inkbot-model> \
    --chat-template examples/template_inkbot.jinja
```

#### `template_teleflm.jinja`

**Model family**: TeleFLM  
**Format**: `<_system>` / `<_user>` / `<_bot>` tags

Uses angle-bracket prefixed tags as turn delimiters. Supports system, user, and assistant roles.

```bash
vllm serve <teleflm-model> \
    --chat-template examples/template_teleflm.jinja
```

### Template Summary Table

| File | Model family | Turn format |
|---|---|---|
| `template_chatml.jinja` | Qwen, InternLM, many fine-tunes | `<\|im_start\|>role\n...<\|im_end\|>` |
| `template_alpaca.jinja` | Alpaca-style | `### Instruction:` / `### Response:` |
| `template_baichuan.jinja` | Baichuan 1 & 2 | `<reserved_106>` / `<reserved_107>` |
| `template_chatglm.jinja` | ChatGLM v1 | `[Round N]\n问：` / `答：` |
| `template_chatglm2.jinja` | ChatGLM2 / ChatGLM3 | `[Round N]\n\n问：` / `答：` |
| `template_falcon.jinja` | Falcon 7B / 40B | `User:` / `Assistant:` |
| `template_falcon_180b.jinja` | Falcon 180B | `User:` / `Falcon:` |
| `template_inkbot.jinja` | InkBot | `<#user#>` / `<#bot#>` |
| `template_teleflm.jinja` | TeleFLM | `<_user>` / `<_bot>` |

## Content Format Detection

Most chat templates expect the `content` field of each message to be a plain string. Newer
multimodal models (e.g., `meta-llama/Llama-Guard-3-1B`) expect content in the OpenAI
multi-part format:

```python
{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
```

vLLM auto-detects which format the template expects and logs:

```
Detected the chat template content format to be: string
```

If the auto-detection is wrong, override it explicitly:

```bash
# Force string format
vllm serve <model> --chat-template-content-format string

# Force OpenAI multi-part format
vllm serve <model> --chat-template-content-format openai
```

The three valid values for `--chat-template-content-format` are:

| Value | Behaviour |
|---|---|
| `auto` | Detect from the template (default) |
| `string` | Always pass content as a plain string |
| `openai` | Always pass content as a list of `{"type": ..., ...}` dicts |

## Chat Template Variables

Jinja2 templates receive the following variables from vLLM:

| Variable | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | The conversation history. Each dict has `role` and `content` keys. |
| `add_generation_prompt` | `bool` | When `True`, append the assistant turn prefix so the model continues generating. |
| `tools` | `list[dict] \| None` | Tool definitions in OpenAI schema format (only present when tools are provided). |
| `bos_token` | `str` | The tokenizer's beginning-of-sequence token string. |
| `eos_token` | `str` | The tokenizer's end-of-sequence token string. |
| `strftime_now` | `callable` | A function that formats the current date/time (e.g., `strftime_now("%d %b %Y")`). |

### Message Object Fields

Each message dict in `messages` may contain:

| Field | Type | Description |
|---|---|---|
| `role` | `str` | `"system"`, `"user"`, `"assistant"`, `"tool"`, or a custom role |
| `content` | `str \| list[dict] \| None` | Message content (string or multi-part) |
| `tool_calls` | `list[dict] \| None` | Tool calls made by the assistant |
| `tool_call_id` | `str \| None` | ID of the tool call this message responds to |
| `name` | `str \| None` | Optional participant name |
| `reasoning` | `str \| None` | Reasoning/thinking content for interleaved thinking models |

### Per-Request Template Variables (`chat_template_kwargs`)

You can pass additional variables to the template on a per-request basis using
`chat_template_kwargs`. This is commonly used to toggle reasoning / thinking modes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "Solve: 2 + 2"}],
    extra_body={"chat_template_kwargs": {"thinking": True}},
)
```

### Server-Level Default Template Variables

Set default `chat_template_kwargs` for all requests at server startup:

```bash
# Enable thinking mode by default for all requests
vllm serve deepseek-ai/DeepSeek-R1 \
    --default-chat-template-kwargs '{"thinking": true}'

# Disable thinking mode by default (clients can still override per-request)
vllm serve Qwen/Qwen3-8B \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

Request-level `chat_template_kwargs` always override server defaults.

## Writing a Custom Chat Template

A minimal template for a model that uses `[INST]` / `[/INST]` markers (Llama 2 style):

```jinja
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' -}}
    {%- elif message['role'] == 'user' -%}
        {{- '[INST] ' + message['content'] + ' [/INST]' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- ' ' + message['content'] + ' </s>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
    {{- ' ' -}}
{%- endif -%}
```

### Tips for Writing Templates

- **Always handle `add_generation_prompt`**: When `True`, append the prefix that signals the
  model to start generating (e.g., `<|im_start|>assistant\n`).
- **Strip whitespace carefully**: Use `|trim` on content fields to avoid accidental
  leading/trailing spaces that can confuse the model.
- **Test with the tokenizer**: Use `tokenizer.apply_chat_template(messages, tokenize=False)`
  to preview the rendered string before deploying.
- **Handle missing system messages gracefully**: Not every conversation includes a system
  message; use `selectattr` or conditional checks.
- **Use `raise_exception`** for invalid inputs: Jinja2 templates can call
  `raise_exception("message")` to surface errors clearly.
- **Use `strftime_now`** for date injection: Many models benefit from knowing the current date.
  Use `strftime_now("%d %b %Y")` rather than hardcoding a date string.
- **Handle both string and list content**: Multimodal models pass content as a list of dicts.
  Check `message['content'] is string` before accessing it directly.

### Jinja2 Filters Available in vLLM

| Filter | Description |
|---|---|
| `\|tojson` | Serialize a value to a JSON string |
| `\|tojson(indent=4)` | Pretty-print JSON with indentation |
| `\|trim` | Strip leading and trailing whitespace |
| `\|items` | Iterate over dict key-value pairs |
| `selectattr(attr, test, value)` | Filter a list by attribute value |
| `rejectattr(attr, test, value)` | Exclude items by attribute value |
| `map(attribute=attr)` | Extract an attribute from each item |

## Multimodal Content in Templates

For multimodal models, message content may be a list of typed parts:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
```

vLLM replaces multimodal items with placeholder strings before passing `messages` to the
template:

| Modality | Placeholder |
|---|---|
| Image | `<##IMAGE##>` |
| Audio | `<##AUDIO##>` |
| Video | `<##VIDEO##>` |

The template sees these placeholders as plain text and can position them anywhere in the prompt.

## Embedding Model Templates

Embedding and reranking models can also use custom chat templates to format inputs. vLLM ships
several templates for pooling models in `examples/pooling/`:

| File | Model | Task |
|---|---|---|
| `pooling/embed/template/vlm2vec_phi3v.jinja` | VLM2Vec (Phi-3V) | Multimodal embedding |
| `pooling/embed/template/vlm2vec_qwen2vl.jinja` | VLM2Vec (Qwen2-VL) | Multimodal embedding |
| `pooling/embed/template/dse_qwen2_vl.jinja` | DSE (Qwen2-VL) | Document screenshot embedding |
| `pooling/embed/template/nemotron_embed_vl.jinja` | Nemotron Embed VL | Multimodal embedding |
| `pooling/score/template/bge-reranker-v2-gemma.jinja` | BGE Reranker v2 (Gemma) | Reranking |
| `pooling/score/template/nemotron-rerank.jinja` | Nemotron Rerank | Reranking |
| `pooling/score/template/nemotron-vl-rerank.jinja` | Nemotron VL Rerank | Multimodal reranking |
| `pooling/score/template/qwen3_reranker.jinja` | Qwen3 Reranker | Reranking |
| `pooling/score/template/qwen3_vl_reranker.jinja` | Qwen3 VL Reranker | Multimodal reranking |
| `pooling/score/template/mxbai_rerank_v2.jinja` | MxBai Rerank v2 | Reranking |

```bash
vllm serve TIGER-Lab/VLM2Vec-Full \
    --task embed \
    --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
```

## Troubleshooting

**`ValueError: No chat template found`**
: The model has no built-in template and none was specified. Pass `--chat-template` with a path
  to a `.jinja` file.

**Garbled or repetitive output**
: The template is likely wrong for the model. Check the model card on HuggingFace for the
  correct prompt format.

**`raise_exception` triggered**
: The template detected an invalid message sequence (e.g., two consecutive user messages).
  Check that your `messages` list alternates roles correctly.

**Wrong content format**
: If you see unexpected `[{'type': 'text', 'text': ...}]` in the prompt, set
  `--chat-template-content-format string`. If you see raw strings where dicts are expected,
  set `--chat-template-content-format openai`.

**`ValueError: chat_template is not supported for Mistral tokenizers`**
: Mistral models use `mistral-common` for templating and do not support per-request
  `chat_template` or `chat_template_kwargs` overrides.

**Template renders but model ignores system message**
: Some models (e.g., Gemma) do not have a native system role. The template typically prepends
  the system message to the first user turn. Check the template source to confirm.

## Further Reading

- [Tool Calling Templates](tool_templates.md) — Jinja2 templates that add function/tool calling
  support.
- [Tokenization](tokenization.md) — how vLLM loads and caches tokenizers.
- [OpenAI-Compatible Server](../serving/chat_completions.md) — full server configuration
  reference for chat completions.
- [Multimodal Inputs](../features/multimodal_inputs.md) — how multimodal content is processed
  before templating.
