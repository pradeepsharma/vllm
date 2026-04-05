# Structured Output Code Examples

This page provides complete, runnable code examples for all structured output
constraint types in vLLM, covering both offline (Python API) and online (OpenAI API)
usage patterns.

---

## Offline Inference Examples

These examples use the `LLM` class directly for batch inference.

### JSON Schema with Pydantic

```python
# examples/offline_inference/structured_outputs.py
from enum import Enum
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

# Generate the JSON schema from the Pydantic model
json_schema = CarDescription.model_json_schema()

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", max_model_len=100)

params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json=json_schema),
    max_tokens=50,
)

outputs = llm.generate(
    "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
# {"brand": "Toyota", "model": "Supra", "car_type": "Coupe"}
```

### JSON Schema from a Dict

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string"},
        "is_active": {"type": "boolean"},
    },
    "required": ["name", "age"],
}

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json=schema),
    max_tokens=100,
)
outputs = llm.generate("Create a user profile for Alice, age 30", sampling_params=params)
import json
user = json.loads(outputs[0].outputs[0].text)
print(user)
```

### Regex Constraint

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# Email address pattern
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(regex=r"\w+@\w+\.com"),
    max_tokens=50,
)
outputs = llm.generate(
    "Generate an email address for Alan Turing, who works in Enigma. End in .com.",
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
# alan.turing@enigma.com

# Phone number pattern
params_phone = SamplingParams(
    structured_outputs=StructuredOutputsParams(regex=r"\(\d{3}\) \d{3}-\d{4}"),
    max_tokens=20,
)
outputs_phone = llm.generate("Generate a US phone number:", sampling_params=params_phone)
print(outputs_phone[0].outputs[0].text)
# (555) 867-5309

# ISO date pattern
params_date = SamplingParams(
    structured_outputs=StructuredOutputsParams(regex=r"\d{4}-\d{2}-\d{2}"),
    max_tokens=10,
)
outputs_date = llm.generate("What is today's date?", sampling_params=params_date)
print(outputs_date[0].outputs[0].text)
# 2024-01-15
```

### Choice Constraint

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# Sentiment classification
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        choice=["Positive", "Negative", "Neutral"]
    ),
)
outputs = llm.generate(
    "Classify this sentiment: vLLM is wonderful!",
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
# Positive

# Yes/No question
params_yn = SamplingParams(
    structured_outputs=StructuredOutputsParams(choice=["Yes", "No"]),
)
outputs_yn = llm.generate(
    "Is Python a programming language?",
    sampling_params=params_yn,
)
print(outputs_yn[0].outputs[0].text)
# Yes
```

### EBNF Grammar Constraint

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# SQL query grammar
sql_grammar = """
root ::= select_statement
select_statement ::= "SELECT " column " from " table " where " condition
column ::= "col_1 " | "col_2 "
table ::= "table_1 " | "table_2 "
condition ::= column "= " number
number ::= "1 " | "2 "
"""

params = SamplingParams(
    structured_outputs=StructuredOutputsParams(grammar=sql_grammar),
    max_tokens=50,
)
outputs = llm.generate(
    "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
    sampling_params=params,
)
print(outputs[0].outputs[0].text)
# SELECT col_1  from table_1  where col_1 = 1

# Arithmetic expression grammar
arithmetic_grammar = """
root ::= expr
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= number | "(" expr ")"
number ::= [0-9]+
"""

params_arith = SamplingParams(
    structured_outputs=StructuredOutputsParams(grammar=arithmetic_grammar),
    max_tokens=30,
)
outputs_arith = llm.generate(
    "Write a simple arithmetic expression:",
    sampling_params=params_arith,
)
print(outputs_arith[0].outputs[0].text)
```

### JSON Object (No Schema)

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json_object=True),
    max_tokens=200,
)
outputs = llm.generate(
    "Return information about the Python programming language as a JSON object.",
    sampling_params=params,
)
import json
data = json.loads(outputs[0].outputs[0].text)
print(data)
```

### Batch Processing with Multiple Constraints

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

prompts = [
    "Classify: vLLM is great!",
    "Classify: This product is terrible.",
    "Classify: The weather is okay.",
]

# All requests in a batch must use the same backend
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        choice=["positive", "negative", "neutral"]
    ),
)

outputs = llm.generate(prompts, sampling_params=params)
for prompt, output in zip(prompts, outputs):
    print(f"{prompt!r} → {output.outputs[0].text!r}")
```

---

## Online Serving Examples (OpenAI API)

These examples use the OpenAI Python client against a running vLLM server.

### Start the Server

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --structured-outputs-config '{"backend": "xgrammar"}'
```

### JSON Schema via `response_format`

```python
import openai
import json
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

car = json.loads(response.choices[0].message.content)
print(f"Brand: {car['brand']}, Model: {car['model']}, Type: {car['car_type']}")
```

### Regex via `structured_outputs` Extension

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an email address for Alan Turing at Enigma. End in .com.",
        }
    ],
    extra_body={
        "structured_outputs": {
            "regex": r"[a-z0-9.]{1,20}@\w{6,10}\.com"
        }
    },
    max_tokens=50,
)
print(response.choices[0].message.content)
```

### Choice via `structured_outputs` Extension

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Classify this sentiment: vLLM is wonderful!",
        }
    ],
    extra_body={
        "structured_outputs": {
            "choice": ["positive", "negative", "neutral"]
        }
    },
)
print(response.choices[0].message.content)
# positive
```

### Grammar via `structured_outputs` Extension

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an SQL query for the users table.",
        }
    ],
    extra_body={
        "structured_outputs": {
            "grammar": """
root ::= select_statement
select_statement ::= "SELECT " column " from " table " where " condition
column ::= "col_1 " | "col_2 "
table ::= "table_1 " | "table_2 "
condition ::= column "= " number
number ::= "1 " | "2 "
"""
        }
    },
    max_tokens=50,
)
print(response.choices[0].message.content)
```

### Structural Tag for Tool Calling

```python
import openai
import json

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

system_prompt = """
You have access to the following function:
get_weather(city: str) - Get the weather for a city

Call it using: <function=get_weather>{"city": "..."}</function>
"""

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the weather in New York?"},
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
# <function=get_weather>{"city": "New York"}</function>
```

### Streaming with JSON Schema

```python
import openai
import json

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "rating": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["title", "summary", "rating"],
}

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Review the movie Inception as JSON."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "movie-review", "schema": schema},
    },
    stream=True,
    max_tokens=200,
)

full_response = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        full_response += delta
        print(delta, end="", flush=True)

print()
review = json.loads(full_response)
print(f"\nTitle: {review['title']}, Rating: {review['rating']}/10")
```

---

## Async Examples

```python
import asyncio
import openai
import json

async def structured_output_async():
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "population": {"type": "integer"},
            "continent": {"type": "string"},
        },
        "required": ["capital", "population", "continent"],
    }

    # Run multiple requests concurrently
    tasks = [
        client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=[{"role": "user", "content": f"Give me info about {country} as JSON."}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "country-info", "schema": schema},
            },
            max_tokens=100,
        )
        for country in ["France", "Japan", "Brazil"]
    ]

    responses = await asyncio.gather(*tasks)
    for country, response in zip(["France", "Japan", "Brazil"], responses):
        data = json.loads(response.choices[0].message.content)
        print(f"{country}: capital={data['capital']}, continent={data['continent']}")

asyncio.run(structured_output_async())
```

---

## Backend-Specific Examples

### Force xgrammar Backend

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    structured_outputs_config={"backend": "xgrammar"},
)

# xgrammar supports all constraint types
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json={"type": "object", "properties": {"answer": {"type": "string"}}},
        disable_any_whitespace=True,  # Compact JSON
    ),
    max_tokens=50,
)
outputs = llm.generate("What is 2+2?", sampling_params=params)
print(outputs[0].outputs[0].text)
# {"answer":"4"}
```

### Force guidance Backend

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    structured_outputs_config={
        "backend": "guidance",
        "disable_additional_properties": True,
    },
)

# guidance supports complex schemas with additionalProperties control
schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["name", "quantity"],
            },
        }
    },
    "required": ["items"],
}

params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json=schema),
    max_tokens=200,
)
outputs = llm.generate("List 3 grocery items as JSON", sampling_params=params)
print(outputs[0].outputs[0].text)
```

### Force outlines Backend

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    structured_outputs_config={"backend": "outlines"},
)

# outlines is ideal for regex and simple JSON schemas
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        regex=r"[A-Z][a-z]+ [A-Z][a-z]+"  # First Last name format
    ),
    max_tokens=20,
)
outputs = llm.generate("Generate a full name:", sampling_params=params)
print(outputs[0].outputs[0].text)
# Alan Turing
```

---

## Error Handling

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")

# This will raise ValueError: schema has unsupported features for xgrammar
# but will fall back to guidance in auto mode
schema_with_multiple_of = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "multipleOf": 5},  # Not supported by xgrammar
    },
}

try:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=schema_with_multiple_of),
    )
    # In auto mode, this falls back to guidance automatically
    outputs = llm.generate("Give me a count", sampling_params=params)
    print(outputs[0].outputs[0].text)
except ValueError as e:
    print(f"Validation error: {e}")

# Disable fallback to get explicit errors
try:
    params_no_fallback = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json=schema_with_multiple_of,
            disable_fallback=True,
        ),
    )
except ValueError as e:
    print(f"Error with disable_fallback=True: {e}")
```

---

## See Also

- [Structured Output Overview](overview.md)
- [response_format parameter](response-format.md)
- [guided_decoding_backend configuration](guided-decoding-backend.md)
- [xgrammar backend](backend-xgrammar.md)
- [outlines backend](backend-outlines.md)
- [llguidance backend](backend-guidance.md)
- [lm-format-enforcer backend](backend-lm-format-enforcer.md)
