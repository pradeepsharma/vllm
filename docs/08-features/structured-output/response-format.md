# `response_format` Parameter

The `response_format` parameter in vLLM's OpenAI-compatible API controls the format
of the model's output. It is the standard OpenAI API mechanism for requesting
structured output and maps directly to vLLM's internal `StructuredOutputsParams`.

This parameter is available in both the **chat completions** endpoint
(`POST /v1/chat/completions`) and the **completions** endpoint (`POST /v1/completions`).

---

## Supported Format Types

| `type` value | Description |
|---|---|
| `"text"` | Default; no constraints on output format |
| `"json_object"` | Constrain output to any valid JSON object |
| `"json_schema"` | Constrain output to a JSON object matching a specific schema |
| `"structural_tag"` | Constrain tagged regions to a JSON schema (tool-calling) |

---

## `"text"` Format

The default. No structured output constraints are applied.

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "Hello!"}],
  "response_format": {"type": "text"}
}
```

---

## `"json_object"` Format

Constrains the output to any valid JSON object. No schema is required.

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [
    {"role": "user", "content": "Return a JSON object with your name and version."}
  ],
  "response_format": {"type": "json_object"}
}
```

**Python (OpenAI client):**

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Return info about yourself as JSON."}],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)
```

---

## `"json_schema"` Format

Constrains the output to a JSON object matching a specific JSON Schema. The schema
is provided in the `json_schema` field.

### Request Structure

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "Describe the Toyota Supra as JSON."
    }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "car-description",
      "description": "A description of a car",
      "schema": {
        "type": "object",
        "properties": {
          "brand": {"type": "string"},
          "model": {"type": "string"},
          "year": {"type": "integer"},
          "car_type": {
            "type": "string",
            "enum": ["sedan", "SUV", "Truck", "Coupe"]
          }
        },
        "required": ["brand", "model", "year", "car_type"]
      },
      "strict": true
    }
  }
}
```

### `json_schema` Object Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | A name for the schema (used for identification) |
| `description` | string | ❌ | Human-readable description of the schema |
| `schema` | object | ✅ | The JSON Schema definition |
| `strict` | boolean | ❌ | Reserved for future use (currently ignored by vLLM) |

> **Note:** The `schema` field uses the alias `schema` in the API, but is stored
> internally as `json_schema` to avoid conflicts with Pydantic's reserved `schema`
> attribute.

### Python Example with Pydantic

```python
import openai
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    SEDAN = "SEDAN"
    SUV = "SUV"
    TRUCK = "TRUCK"
    COUPE = "COUPE"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Describe the most iconic car from the 90s as JSON.",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema(),
        },
    },
    max_tokens=200,
)
import json
car = json.loads(response.choices[0].message.content)
print(car)
# {"brand": "Toyota", "model": "Supra", "car_type": "COUPE"}
```

---

## `"structural_tag"` Format

Constrains specific tagged regions of the output to a JSON schema, while allowing
free-form text outside those regions. This is useful for tool-calling formats where
the model emits structured function calls embedded in natural language.

### Request Structure

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather in New York?"
    }
  ],
  "response_format": {
    "type": "structural_tag",
    "structures": [
      {
        "begin": "<function=get_weather>",
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        },
        "end": "</function>"
      }
    ],
    "triggers": ["<function="]
  }
}
```

### `structural_tag` Object Fields

| Field | Type | Description |
|---|---|---|
| `type` | `"structural_tag"` | Identifies this as a structural tag format |
| `structures` | array | List of tag structures to constrain |
| `triggers` | array of strings | Prefixes that trigger structured output mode |

Each structure in `structures` has:

| Field | Type | Description |
|---|---|---|
| `begin` | string | The opening tag (must start with one of the `triggers`) |
| `schema` | object | JSON Schema for the content between begin and end |
| `end` | string | The closing tag |

### Python Example

```python
import openai
import asyncio

client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def call_with_tool():
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": """
You have access to get_weather(city: str).
Call it using: <function=get_weather>{"city": "..."}</function>

What is the weather in Paris?
""",
            }
        ],
        response_format={
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_weather>",
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "end": "</function>",
                }
            ],
            "triggers": ["<function="],
        },
        max_tokens=100,
    )
    print(response.choices[0].message.content)

asyncio.run(call_with_tool())
```

---

## Internal Mapping

When `response_format` is set, vLLM's protocol layer converts it to a
`StructuredOutputsParams` object before passing it to the engine:

```python
# From vllm/entrypoints/openai/chat_completion/protocol.py
response_format = self.response_format
if response_format is not None:
    structured_outputs_kwargs = {}

    if response_format.type == "json_object":
        structured_outputs_kwargs["json_object"] = True
    elif response_format.type == "json_schema":
        json_schema = response_format.json_schema
        structured_outputs_kwargs["json"] = json_schema.json_schema
    elif response_format.type == "structural_tag":
        s_tag_obj = structural_tag.model_dump(by_alias=True)
        structured_outputs_kwargs["structural_tag"] = json.dumps(s_tag_obj)

    self.structured_outputs = StructuredOutputsParams(**structured_outputs_kwargs)
```

---

## `structured_outputs` Extension Parameter

In addition to the standard `response_format`, vLLM supports a vLLM-specific
`structured_outputs` field in the request body. This provides access to additional
constraint types not available through `response_format`:

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "Classify this: vLLM is great!"}],
  "structured_outputs": {
    "choice": ["positive", "negative", "neutral"]
  }
}
```

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "Generate an SQL query"}],
  "structured_outputs": {
    "grammar": "root ::= select_statement\nselect_statement ::= \"SELECT \" column \" FROM \" table\ncolumn ::= \"id\" | \"name\"\ntable ::= \"users\" | \"orders\"\n"
  }
}
```

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "Generate an email address"}],
  "structured_outputs": {
    "regex": "[a-z0-9.]{1,20}@\\w{6,10}\\.com"
  }
}
```

### `structured_outputs` Fields

| Field | Type | Description |
|---|---|---|
| `json` | string or object | JSON Schema (string or dict) |
| `regex` | string | Regular expression pattern |
| `choice` | array of strings | List of allowed output strings |
| `grammar` | string | EBNF grammar specification |
| `json_object` | boolean | Constrain to any valid JSON object |
| `structural_tag` | string | JSON-encoded structural tag specification |
| `disable_fallback` | boolean | Disable backend fallback on error |
| `disable_any_whitespace` | boolean | Force compact JSON output |
| `disable_additional_properties` | boolean | Inject `additionalProperties: false` (guidance only) |

---

## Streaming Support

Both `response_format` and `structured_outputs` work with streaming responses.
The constraint is applied at each token, so the stream will only emit tokens that
keep the output valid:

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Describe a car as JSON"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car",
            "schema": {"type": "object", "properties": {"brand": {"type": "string"}}},
        },
    },
    stream=True,
    max_tokens=100,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Validation Errors

If the `response_format` is invalid, vLLM returns a 400 error:

| Error | Cause |
|---|---|
| `"json_schema" field must be provided` | `type="json_schema"` without `json_schema` field |
| `Invalid JSON grammar specification` | Malformed JSON schema |
| `Failed to transform json schema into a grammar` | Schema has unsupported features |

---

## See Also

- [Structured Output Overview](overview.md)
- [guided_decoding_backend configuration](guided-decoding-backend.md)
- [xgrammar backend](backend-xgrammar.md)
- [outlines backend](backend-outlines.md)
