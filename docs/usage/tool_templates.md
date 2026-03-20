# Tool Calling Jinja Templates

vLLM supports OpenAI-compatible tool calling (function calling) through model-specific Jinja2
templates. These templates extend the standard chat template format to handle tool definitions,
tool call outputs from the model, and tool results returned by the client.

The `examples/` directory ships **25 ready-to-use tool chat templates** covering the most
popular open-source model families. This guide explains how to use them, how they differ, and
how to write your own.

## Quick Start

```bash
# Serve Llama 3.1 with JSON-format tool calling
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja

# Make a tool-calling request
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }]
    }'
```

## How Tool Calling Works in vLLM

When a request includes a `tools` list, vLLM:

1. **Renders the prompt** using the tool chat template, which injects tool definitions into the
   system prompt or user message.
2. **Generates a response** that may contain a tool call in the model's native format (JSON
   object, Python-style call, XML tags, etc.).
3. **Parses the tool call** using the `--tool-call-parser` specified at server startup,
   converting the raw text into a structured `tool_calls` object.
4. **Returns the response** to the client. The client executes the tool, then sends back a
   `tool` role message with the result.
5. **Continues the conversation** — the template renders the tool result into the prompt and
   the model generates a final answer.

The template and the parser must be matched: each template produces output in a specific format,
and the parser knows how to decode that format.

## Tool Call Output Formats

Different model families use different formats for expressing tool calls:

| Format | Example | Parser |
|---|---|---|
| JSON object | `{"name": "func", "parameters": {...}}` | `llama3_json`, `mistral` |
| JSON array | `[{"name": "func", "arguments": {...}}]` | `xlam`, `mistral` |
| Pythonic | `[func_name(param=value)]` | `pythonic` |
| XML tags | `<tool_call>{"name": ..., "arguments": ...}</tool_call>` | `hermes` |
| Unicode tokens | `｟tool_calls_begin｠｟tool_call_begin｠...` | `deepseekv3` |
| Custom XML | `<tool_call>func_name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` | `glm4` |
| functools | `functools[{"name": ..., "arguments": {...}}]` | `phi4_mini` |

## Available Tool Chat Templates

### Llama Family

#### `tool_chat_template_llama3.1_json.jinja`

**Models**: Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-405B-Instruct  
**Parser**: `--tool-call-parser llama3_json`  
**Format**: JSON object `{"name": "func_name", "parameters": {...}}`

Tool definitions are injected into the first user message (not the system prompt) by default.
The template supports a `tools_in_user_message` variable to control placement. Tool results are
wrapped in an `ipython` role header. Supports a `date_string` variable for injecting the current
date (defaults to `strftime_now("%d %b %Y")` if available).

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

**Template variables:**

| Variable | Default | Description |
|---|---|---|
| `tools_in_user_message` | `true` | Inject tools into first user message instead of system prompt |
| `date_string` | Today's date | Date string injected into the system prompt |
| `custom_tools` | — | Override the `tools` variable |

#### `tool_chat_template_llama3.2_json.jinja`

**Models**: Meta-Llama-3.2-1B-Instruct, Meta-Llama-3.2-3B-Instruct  
**Parser**: `--tool-call-parser llama3_json`  
**Format**: JSON object (same as 3.1)

Variant of the 3.1 template tuned for the smaller 3.2 models.

#### `tool_chat_template_llama3.2_pythonic.jinja`

**Models**: Meta-Llama-3.2-1B-Instruct, Meta-Llama-3.2-3B-Instruct  
**Parser**: `--tool-call-parser pythonic`  
**Format**: Python-style list `[func_name(param=value, ...)]`

The model outputs tool calls as Python function call syntax rather than JSON. Tool definitions
are placed in the system prompt by default (`tools_in_user_message = false`). Tool results are
wrapped in an `ipython` role header.

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --tool-call-parser pythonic \
    --chat-template examples/tool_chat_template_llama3.2_pythonic.jinja
```

#### `tool_chat_template_llama4_json.jinja`

**Models**: Meta-Llama-4 series  
**Parser**: `--tool-call-parser llama4_json`  
**Format**: JSON object `{"name": "func_name", "parameters": {...}}`

Uses Llama 4's `<|header_start|>` / `<|header_end|>` / `<|eot|>` token format. Adds support
for multimodal content (image tokens via `<|image|>`) in messages alongside tool calling. Tool
definitions are injected into the first user message by default.

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --tool-call-parser llama4_json \
    --chat-template examples/tool_chat_template_llama4_json.jinja
```

#### `tool_chat_template_llama4_pythonic.jinja`

**Models**: Meta-Llama-4 series  
**Parser**: `--tool-call-parser pythonic`  
**Format**: Python-style list `[func_name(param="value", ...)]`

Includes a detailed system prompt that enforces strict function composition rules (from the
official Llama 4 model card). Supports parallel tool calls in a single list. Supports
multimodal content (image tokens).

```bash
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    --tool-call-parser pythonic \
    --chat-template examples/tool_chat_template_llama4_pythonic.jinja
```

#### `tool_chat_template_toolace.jinja`

**Models**: ToolACE (Llama-3.1-based tool-calling fine-tune)  
**Parser**: `--tool-call-parser llama3_json`  
**Format**: Python-style list `[func_name(param=value, ...)]`

Based on the Llama 3.1 template with a modified system prompt emphasising expert function
composition. Tool definitions are placed in the system prompt.

```bash
vllm serve Team-ACE/ToolACE-8B \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_toolace.jinja
```

### Mistral Family

#### `tool_chat_template_mistral.jinja`

**Models**: Mistral-7B-Instruct-v0.3, Mixtral-8x7B-Instruct-v0.1  
**Parser**: `--tool-call-parser mistral`  
**Format**: `[TOOL_CALLS] [{"name": ..., "arguments": ..., "id": ...}]`

Tool definitions are injected as `[AVAILABLE_TOOLS] [...] [/AVAILABLE_TOOLS]` before the last
user message. Tool results use `[TOOL_RESULTS] {"content": ..., "call_id": ...} [/TOOL_RESULTS]`.
Tool call IDs must be alphanumeric strings of length ≥ 9 (the template raises an exception
otherwise).

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tool-call-parser mistral \
    --chat-template examples/tool_chat_template_mistral.jinja
```

#### `tool_chat_template_mistral3.jinja`

**Models**: Mistral Small 3 (Mistral-Small-3.1-24B-Instruct-2503)  
**Parser**: `--tool-call-parser mistral`  
**Format**: Same as `mistral.jinja` with `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` wrapper

Adds a default system message with the model's knowledge cutoff date and today's date via
`strftime_now`. Injects a parallel tool call instruction into the system prompt when tools are
present. Supports multimodal content (image tokens via `[IMG]`).

```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --tool-call-parser mistral \
    --chat-template examples/tool_chat_template_mistral3.jinja
```

#### `tool_chat_template_mistral_parallel.jinja`

**Models**: Mistral models with parallel tool call support  
**Parser**: `--tool-call-parser mistral`  
**Format**: JSON array of tool calls

Identical to `mistral.jinja` but injects a system prompt instructing the model to return all
tool calls in a single JSON array, enabling parallel (simultaneous) tool execution.

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tool-call-parser mistral \
    --chat-template examples/tool_chat_template_mistral_parallel.jinja
```

### Hermes / NousResearch

#### `tool_chat_template_hermes.jinja`

**Models**: NousResearch Hermes-2-Pro, Hermes-3, and compatible models  
**Parser**: `--tool-call-parser hermes`  
**Format**: `<tool_call>{"name": ..., "arguments": ...}</tool_call>` XML tags

Tool definitions are rendered in a Python-docstring style inside `<tools>...</tools>` XML tags
in the system prompt. The template includes a `json_to_python_type` macro that converts JSON
Schema types to Python type annotations (e.g., `"string"` → `str`, `"array"` → `list[...]`).
Tool results are wrapped in `<tool_response>...</tool_response>` tags. Supports multiple tool
calls per turn.

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B \
    --tool-call-parser hermes \
    --chat-template examples/tool_chat_template_hermes.jinja
```

**Tool definition format rendered by this template:**

```
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "get_weather(city: str) - Get weather for a city

    Args:
        city(str): The city name", "parameters": {...}}}
</tools>
```

### DeepSeek Family

#### `tool_chat_template_deepseekv3.jinja`

**Models**: DeepSeek-V3  
**Parser**: `--tool-call-parser deepseekv3`  
**Format**: Custom Unicode special tokens

Tool definitions are listed in a `# Tools` section in the system prompt. Tool calls use a
distinctive multi-token format with JSON code blocks:

```
｟tool_calls_begin｠｟tool_call_begin｠function｟tool_sep｠function_name
```json
{"param": "value"}
```｟tool_call_end｠｟tool_calls_end｠｟end_of_sentence｠
```

Tool results use `｟tool_outputs_begin｠` / `｟tool_output_begin｠` / `｟tool_output_end｠` /
`｟tool_outputs_end｠` wrapper tokens.

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tool-call-parser deepseekv3 \
    --chat-template examples/tool_chat_template_deepseekv3.jinja
```

#### `tool_chat_template_deepseekv31.jinja`

**Models**: DeepSeek-V3.1  
**Parser**: `--tool-call-parser deepseekv3`  
**Format**: Same Unicode token format as V3

Adds support for a `thinking` variable (passed via `chat_template_kwargs`) to enable/disable
chain-of-thought reasoning alongside tool calling. When `thinking=True`, the template emits
`<think>` tokens; when `False`, it emits `</think>` to skip the thinking phase.

```bash
vllm serve deepseek-ai/DeepSeek-V3-1 \
    --tool-call-parser deepseekv3 \
    --chat-template examples/tool_chat_template_deepseekv31.jinja \
    --default-chat-template-kwargs '{"thinking": false}'
```

#### `tool_chat_template_deepseekr1.jinja`

**Models**: DeepSeek-R1  
**Parser**: `--tool-call-parser deepseekv3`  
**Format**: Same Unicode token format as V3

Adapted from the SGLang community template. Combines DeepSeek's reasoning (thinking) tokens
with tool calling capability. The template strips `</think>` and everything before it from
assistant message content when rendering history.

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tool-call-parser deepseekv3 \
    --chat-template examples/tool_chat_template_deepseekr1.jinja
```

### Gemma Family

#### `tool_chat_template_gemma3_pythonic.jinja`

**Models**: Gemma 3 (google/gemma-3-*)  
**Parser**: `--tool-call-parser pythonic`  
**Format**: Python-style list `[func_name(param=value, ...)]`

Gemma does not have a native system role; the template prepends system message content to the
first user message. Tool definitions are injected as a JSON list in the first user turn with
instructions. Supports multimodal content (image tokens via `<start_of_image>`). Validates
alternating user/model roles and raises an exception on violations.

Tool results are wrapped in `<tool_response>...</tool_response>` tags and rendered as `user`
role turns (since Gemma maps `tool` → `user`).

```bash
vllm serve google/gemma-3-27b-it \
    --tool-call-parser pythonic \
    --chat-template examples/tool_chat_template_gemma3_pythonic.jinja
```

#### `tool_chat_template_functiongemma.jinja`

**Models**: Google FunctionGemma  
**Parser**: `--tool-call-parser pythonic`  
**Format**: `<start_function_call>call:func_name{key:<escape>value<escape>}</start_function_call>`

Uses `developer` / `system` roles for tool definitions. Tool schemas are rendered in a
simplified `Function: name / Description: ... / Parameters: ...` format rather than full JSON.
Tool results are rendered as `user` role messages with a `Function result for name:` prefix.

```bash
vllm serve google/gemma-function-calling \
    --tool-call-parser pythonic \
    --chat-template examples/tool_chat_template_functiongemma.jinja
```

### IBM Granite Family

#### `tool_chat_template_granite.jinja`

**Models**: IBM Granite 3.x (granite-3.0-8b-instruct, granite-3.1-8b-instruct, etc.)  
**Parser**: `--tool-call-parser granite`  
**Format**: `<|tool_call|>[{"name": ..., "arguments": {...}}]`

Tool definitions are placed in an `available_tools` role block before the conversation using
`<|start_of_role|>available_tools<|end_of_role|>`. Tool results use a `tool_response` role.

```bash
vllm serve ibm-granite/granite-3.1-8b-instruct \
    --tool-call-parser granite \
    --chat-template examples/tool_chat_template_granite.jinja
```

#### `tool_chat_template_granite_20b_fc.jinja`

**Models**: IBM Granite 20B Function Calling  
**Parser**: `--tool-call-parser granite`  
**Format**: `<function_call> {"name": ..., "arguments": {...}}`

Uses the same `json_to_python_type` macro as the Hermes template to render tool signatures in
a Python-docstring format inside a `<|function_call_library|>` block. Supports a
`full_function_description` variable to toggle between full Python-docstring style and simple
description-only style.

```bash
vllm serve ibm-granite/granite-20b-functioncalling \
    --tool-call-parser granite \
    --chat-template examples/tool_chat_template_granite_20b_fc.jinja
```

### Qwen / Alibaba Family

#### `tool_chat_template_qwen3coder.jinja`

**Models**: Qwen3-Coder series  
**Parser**: `--tool-call-parser hermes`  
**Format**: `<tool_call>...</tool_call>` XML tags (Hermes-compatible)

Includes a `render_extra_keys` macro for rendering non-standard tool schema fields as XML
elements. Tool definitions are rendered with `<function>`, `<name>`, `<description>`, and
`<parameters>` XML tags inside `<tools>...</tools>`. Supports agentic computer-use scenarios
with a default system prompt for tool-enabled assistants.

```bash
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct \
    --tool-call-parser hermes \
    --chat-template examples/tool_chat_template_qwen3coder.jinja
```

### GLM Family

#### `tool_chat_template_glm4.jinja`

**Models**: THUDM GLM-4 series  
**Parser**: `--tool-call-parser glm4`  
**Format**: `<tool_call>function_name\n<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`

Uses a custom XML tag format with separate `<arg_key>` and `<arg_value>` elements for each
argument. Tool definitions are rendered as JSON objects with a preamble explaining the format.
The conversation format is the same as `template_chatglm.jinja` (round-based Chinese format).

```bash
vllm serve THUDM/glm-4-9b-chat \
    --tool-call-parser glm4 \
    --chat-template examples/tool_chat_template_glm4.jinja
```

### InternLM Family

#### `tool_chat_template_internlm2_tool.jinja`

**Models**: InternLM2 series  
**Parser**: `--tool-call-parser internlm`  
**Format**: JSON array in a `<|plugin|>` system block

Tool definitions are placed in a special `system name=<|plugin|>` block as a JSON array of
function objects. Tool calls are returned as `<|action_start|><|plugin|>` blocks with JSON.
Tool results use an `environment name=<|plugin|>` role.

```bash
vllm serve internlm/internlm2_5-7b-chat \
    --tool-call-parser internlm \
    --chat-template examples/tool_chat_template_internlm2_tool.jinja
```

### Salesforce xLAM Family

#### `tool_chat_template_xlam_qwen.jinja`

**Models**: Salesforce xLAM (Qwen-based, e.g., xLAM-2-3b-fc-r)  
**Parser**: `--tool-call-parser xlam`  
**Format**: JSON array `[{"name": ..., "arguments": {...}}, ...]`

Uses ChatML (`<|im_start|>` / `<|im_end|>`) format. Tool definitions and format instructions
are injected into the system prompt. Supports parallel tool calls in a single JSON array.
Tool results are rendered in a `tool` role block.

```bash
vllm serve Salesforce/xLAM-2-3b-fc-r \
    --tool-call-parser xlam \
    --chat-template examples/tool_chat_template_xlam_qwen.jinja
```

#### `tool_chat_template_xlam_llama.jinja`

**Models**: Salesforce xLAM (Llama-based)  
**Parser**: `--tool-call-parser xlam`  
**Format**: JSON array (same as xLAM Qwen)

Uses Llama 3 header format (`<|start_header_id|>` / `<|eot_id|>`) instead of ChatML. Tool
definitions are placed in the system prompt with the same format instructions as the Qwen
variant.

```bash
vllm serve Salesforce/xLAM-2-8b-fc-r \
    --tool-call-parser xlam \
    --chat-template examples/tool_chat_template_xlam_llama.jinja
```

### Microsoft Phi Family

#### `tool_chat_template_phi4_mini.jinja`

**Models**: Microsoft Phi-4-mini-instruct  
**Parser**: `--tool-call-parser phi4_mini`  
**Format**: `functools[{"name": ..., "arguments": {...}}, ...]`

Uses `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tools|>` role tags. Tool definitions are
listed in the system block with instructions. Tool calls are prefixed with `functools` marker.

```bash
vllm serve microsoft/Phi-4-mini-instruct \
    --tool-call-parser phi4_mini \
    --chat-template examples/tool_chat_template_phi4_mini.jinja
```

### MiniMax Family

#### `tool_chat_template_minimax_m1.jinja`

**Models**: MiniMax-M1  
**Parser**: `--tool-call-parser minimax`  
**Format**: `<tool_calls>{"name": ..., "arguments": {...}}</tool_calls>` XML tags

Uses `<beginning_of_sentence>` / `<end_of_sentence>` delimiters with role-specific prefixes
(`system ai_setting=assistant`, `user name=user`, `ai name=assistant`, `tool name=tools`).
Tool definitions are placed in a `system tool_setting=tools` block.

```bash
vllm serve MiniMaxAI/MiniMax-M1-40k \
    --tool-call-parser minimax \
    --chat-template examples/tool_chat_template_minimax_m1.jinja
```

### Tencent Hunyuan Family

#### `tool_chat_template_hunyuan_a13b.jinja`

**Models**: Tencent Hunyuan-A13B  
**Parser**: `--tool-call-parser hunyuan_a13b`  
**Format**: `<tool_calls>[{"name": ..., "arguments": {...}}]</tool_calls>` XML tags

Uses `<|startoftext|>` / `<|extra_4|>` / `<|extra_0|>` / `<|eos|>` special tokens. Injects
the current date and time (including Chinese weekday name) into the system prompt. Supports
both tool-calling and plain conversation modes.

```bash
vllm serve tencent/Hunyuan-A13B-Instruct \
    --tool-call-parser hunyuan_a13b \
    --chat-template examples/tool_chat_template_hunyuan_a13b.jinja
```

## Template Summary Table

| Template file | Model family | Parser | Format |
|---|---|---|---|
| `tool_chat_template_llama3.1_json.jinja` | Llama 3.1 | `llama3_json` | JSON object |
| `tool_chat_template_llama3.2_json.jinja` | Llama 3.2 | `llama3_json` | JSON object |
| `tool_chat_template_llama3.2_pythonic.jinja` | Llama 3.2 | `pythonic` | Python list |
| `tool_chat_template_llama4_json.jinja` | Llama 4 | `llama4_json` | JSON object |
| `tool_chat_template_llama4_pythonic.jinja` | Llama 4 | `pythonic` | Python list |
| `tool_chat_template_toolace.jinja` | ToolACE (Llama 3.1) | `llama3_json` | Python list |
| `tool_chat_template_mistral.jinja` | Mistral v0.3 / Mixtral | `mistral` | `[TOOL_CALLS]` |
| `tool_chat_template_mistral3.jinja` | Mistral Small 3 | `mistral` | `[TOOL_CALLS]` |
| `tool_chat_template_mistral_parallel.jinja` | Mistral (parallel) | `mistral` | JSON array |
| `tool_chat_template_hermes.jinja` | Hermes 2/3 | `hermes` | `<tool_call>` XML |
| `tool_chat_template_deepseekv3.jinja` | DeepSeek-V3 | `deepseekv3` | Unicode tokens |
| `tool_chat_template_deepseekv31.jinja` | DeepSeek-V3.1 | `deepseekv3` | Unicode tokens |
| `tool_chat_template_deepseekr1.jinja` | DeepSeek-R1 | `deepseekv3` | Unicode tokens |
| `tool_chat_template_gemma3_pythonic.jinja` | Gemma 3 | `pythonic` | Python list |
| `tool_chat_template_functiongemma.jinja` | FunctionGemma | `pythonic` | Custom XML |
| `tool_chat_template_granite.jinja` | IBM Granite 3.x | `granite` | `<\|tool_call\|>` |
| `tool_chat_template_granite_20b_fc.jinja` | IBM Granite 20B FC | `granite` | `<function_call>` |
| `tool_chat_template_qwen3coder.jinja` | Qwen3-Coder | `hermes` | `<tool_call>` XML |
| `tool_chat_template_glm4.jinja` | GLM-4 | `glm4` | Custom XML |
| `tool_chat_template_internlm2_tool.jinja` | InternLM2 | `internlm` | JSON in plugin block |
| `tool_chat_template_xlam_qwen.jinja` | xLAM (Qwen) | `xlam` | JSON array |
| `tool_chat_template_xlam_llama.jinja` | xLAM (Llama) | `xlam` | JSON array |
| `tool_chat_template_phi4_mini.jinja` | Phi-4-mini | `phi4_mini` | `functools[...]` |
| `tool_chat_template_minimax_m1.jinja` | MiniMax-M1 | `minimax` | `<tool_calls>` XML |
| `tool_chat_template_hunyuan_a13b.jinja` | Hunyuan-A13B | `hunyuan_a13b` | `<tool_calls>` XML |

## Tool Definition Schema

All templates accept tools in the OpenAI function-calling schema:

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}
```

### Mistral-Specific Requirements

Mistral templates require:

- `parameters` must be present (use `{}` for no parameters).
- `description` must be present (use `""` for no description).
- Tool call IDs must be alphanumeric strings of **length ≥ 9**. vLLM automatically truncates
  longer IDs to the last 9 characters.

## Multi-Turn Tool Calling Example

A complete multi-turn conversation with tool calling:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

# Turn 1: User asks a question
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
)

# The model responds with a tool call
tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # get_weather
print(tool_call.function.arguments)  # {"city": "Tokyo"}

# Turn 2: Execute the tool and return the result
messages.append(response.choices[0].message)  # assistant message with tool_calls
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": '{"temperature": 22, "condition": "Sunny"}',
})

# Turn 3: Model generates a final answer
final_response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
)
print(final_response.choices[0].message.content)
# "The weather in Tokyo is currently 22°C and sunny."
```

## Parallel Tool Calls

Some templates support parallel tool calls — the model returns multiple tool calls in a single
response. Templates that support this include:

- `tool_chat_template_mistral_parallel.jinja`
- `tool_chat_template_mistral3.jinja`
- `tool_chat_template_xlam_qwen.jinja`
- `tool_chat_template_xlam_llama.jinja`
- `tool_chat_template_llama4_pythonic.jinja`
- `tool_chat_template_minimax_m1.jinja`

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[{"role": "user", "content": "Get weather for Paris and Tokyo"}],
    tools=tools,
    parallel_tool_calls=True,
)

# Multiple tool calls in one response
for tc in response.choices[0].message.tool_calls:
    print(tc.function.name, tc.function.arguments)
```

## Writing a Custom Tool Chat Template

A minimal tool chat template that injects tools into the system prompt and uses JSON format:

```jinja
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "You are a helpful assistant." %}
{%- endif %}

{{- '<|system|>\n' + system_message }}

{%- if tools %}
    {{- '\n\nYou have access to these tools:\n' }}
    {%- for tool in tools %}
        {{- tool | tojson(indent=2) + '\n' }}
    {%- endfor %}
    {{- '\nRespond with JSON: {"name": "func_name", "arguments": {...}}' }}
{%- endif %}
{{- '<|end|>\n' }}

{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<|user|>\n' + message['content'] + '<|end|>\n' }}
    {%- elif message['role'] == 'assistant' and message.tool_calls is defined %}
        {{- '<|assistant|>\n' }}
        {%- for tc in message.tool_calls %}
            {{- tc.function | tojson }}
        {%- endfor %}
        {{- '<|end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|assistant|>\n' + message['content'] + '<|end|>\n' }}
    {%- elif message['role'] == 'tool' %}
        {{- '<|tool|>\n' + message['content'] + '<|end|>\n' }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

### Template Design Checklist

- [ ] **Inject tool definitions** into the system prompt or first user message.
- [ ] **Handle `tool_calls` in assistant messages** — render each tool call in your chosen format.
- [ ] **Handle `tool` role messages** — render tool results so the model can see them.
- [ ] **Handle `add_generation_prompt`** — append the assistant prefix at the end.
- [ ] **Handle missing tools** — the template must work when `tools` is `None` or empty.
- [ ] **Handle missing system messages** — not every conversation starts with a system message.
- [ ] **Match your parser** — the format you render must be parseable by the `--tool-call-parser`
  you specify.

## Tool Call Parsers

The `--tool-call-parser` flag specifies which parser vLLM uses to extract structured tool calls
from the model's raw text output. Each parser is matched to one or more template formats:

| Parser | Compatible templates | Output format expected |
|---|---|---|
| `llama3_json` | `llama3.1_json`, `llama3.2_json`, `toolace` | JSON object |
| `llama4_json` | `llama4_json` | JSON object |
| `pythonic` | `llama3.2_pythonic`, `llama4_pythonic`, `gemma3_pythonic`, `functiongemma`, `toolace` | Python list |
| `mistral` | `mistral`, `mistral3`, `mistral_parallel` | `[TOOL_CALLS]` format |
| `hermes` | `hermes`, `qwen3coder` | `<tool_call>` XML |
| `deepseekv3` | `deepseekv3`, `deepseekv31`, `deepseekr1` | Unicode token format |
| `granite` | `granite`, `granite_20b_fc` | Granite format |
| `glm4` | `glm4` | GLM-4 XML format |
| `internlm` | `internlm2_tool` | InternLM plugin format |
| `xlam` | `xlam_qwen`, `xlam_llama` | JSON array |
| `phi4_mini` | `phi4_mini` | `functools[...]` |
| `minimax` | `minimax_m1` | `<tool_calls>` XML |
| `hunyuan_a13b` | `hunyuan_a13b` | `<tool_calls>` XML |

## Troubleshooting

**Tool calls not parsed (empty `tool_calls` in response)**
: The `--tool-call-parser` does not match the template. Verify that the parser name matches the
  format produced by the template (see the table above).

**`raise_exception: Tool call IDs should be alphanumeric strings with length >= 9`**
: Mistral templates require tool call IDs of at least 9 characters. vLLM auto-truncates IDs
  longer than 9 characters. If IDs are shorter, generate longer IDs on the client side.

**`ValueError: mistral-common only supports function tools`**
: The Mistral tokenizer (used with `--tokenizer-mode mistral`) only supports `type: "function"`
  tools. Remove any non-function tools from your request.

**Model ignores tools and responds in plain text**
: The model may not have been fine-tuned for tool calling, or the template is injecting tools
  in a format the model was not trained on. Try a different template or a model specifically
  fine-tuned for tool calling.

**`Cannot set add_generation_prompt to True when the last message is from the assistant`**
: The Mistral tokenizer raises this error. Use `continue_final_message=True` instead of
  `add_generation_prompt=True` when the last message is an assistant message.

**Parallel tool calls not working**
: Ensure you are using a template that supports parallel calls (e.g., `mistral_parallel`,
  `xlam_qwen`) and that the model was fine-tuned for parallel tool calling.

## Further Reading

- [Chat Templates](chat_templates.md) — standard chat templates without tool calling.
- [Tool Calling](../features/tool_calling.md) — end-to-end guide to tool calling in vLLM.
- [OpenAI-Compatible Server](../serving/chat_completions.md) — server configuration for tool
  calling.
- [Tokenization](tokenization.md) — how vLLM loads and caches tokenizers.
