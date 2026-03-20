# Structured Outputs

vLLM supports the generation of structured outputs using
[xgrammar](https://github.com/mlc-ai/xgrammar),
[outlines](https://github.com/dottxt-ai/outlines-core),
[guidance / llguidance](https://github.com/guidance-ai/llguidance), or
[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) as backends.
This document shows you some examples of the different options that are
available to generate structured outputs.

!!! warning
    If you are still using the following deprecated API fields which were removed in v0.12.0, please update your code to use `structured_outputs` as demonstrated in the rest of this document:

    - `guided_json` -> `{"structured_outputs": {"json": ...}}` or `StructuredOutputsParams(json=...)`
    - `guided_regex` -> `{"structured_outputs": {"regex": ...}}` or `StructuredOutputsParams(regex=...)`
    - `guided_choice` -> `{"structured_outputs": {"choice": ...}}` or `StructuredOutputsParams(choice=...)`
    - `guided_grammar` -> `{"structured_outputs": {"grammar": ...}}` or `StructuredOutputsParams(grammar=...)`
    - `guided_whitespace_pattern` -> `{"structured_outputs": {"whitespace_pattern": ...}}` or `StructuredOutputsParams(whitespace_pattern=...)`
    - `structural_tag` -> `{"structured_outputs": {"structural_tag": ...}}` or `StructuredOutputsParams(structural_tag=...)`
    - `guided_decoding_backend` -> Remove this field from your request

## Backends

vLLM supports four structured output backends. The default is `auto`, which selects the most appropriate backend based on the request type and available packages.

### Choosing a Backend

Set the backend globally when starting the server:

```bash
# Use xgrammar (default for most requests)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend xgrammar

# Use outlines
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend outlines

# Use guidance (llguidance)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend guidance

# Use lm-format-enforcer
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend lm-format-enforcer

# Let vLLM choose automatically (default)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend auto
```

!!! note
    vLLM uses a **single backend per engine instance**. You cannot mix backends across requests in the same server process.

### Backend Comparison

| Feature | xgrammar | outlines | guidance (llguidance) | lm-format-enforcer |
|---------|----------|----------|----------------------|-------------------|
| JSON schema | ✅ | ✅ | ✅ | ✅ |
| Regex | ✅ | ✅ | ✅ | ✅ |
| Choice | ✅ | ✅ | ✅ | ✅ |
| EBNF grammar | ✅ | ❌ | ✅ | ❌ |
| Structural tag | ✅ | ❌ | ✅ | ❌ |
| Speculative decoding | ✅ | ✅ | ✅ | ❌ |
| Async compilation | ✅ | ✅ | ✅ | ✅ |
| Regex syntax | Rust-style | Rust-style | Rust-style | Python `re` |
| Install | `pip install xgrammar` | `pip install outlines-core` | `pip install llguidance` | `pip install lm-format-enforcer` |

### xgrammar

[xgrammar](https://github.com/mlc-ai/xgrammar) is the **default backend** for most requests. It uses a compiled grammar approach with a finite-state machine (FSM) that is cached across requests. xgrammar supports the full range of structured output types including JSON schema, regex, EBNF grammar, and structural tags.

**Key features:**
- Fast grammar compilation with an LRU cache (configurable via `VLLM_XGRAMMAR_CACHE_MB`)
- Supports speculative decoding with rollback
- Handles Mistral tokenizers with special byte-fallback vocabulary
- Supports `any_whitespace` mode for flexible JSON formatting

**Configuration options:**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend xgrammar \
  --structured-outputs-config.disable_any_whitespace true
```

| Option | Default | Description |
|--------|---------|-------------|
| `disable_any_whitespace` | `false` | Restrict whitespace in JSON to exact schema-specified positions |
| `disable_additional_properties` | `false` | Disallow extra JSON keys not in the schema |

**Cache size:**

```bash
# Set xgrammar cache to 512 MB (default: 128 MB)
export VLLM_XGRAMMAR_CACHE_MB=512
```

### outlines

[outlines-core](https://github.com/dottxt-ai/outlines-core) uses a deterministic finite automaton (DFA) approach. It converts JSON schemas and regex patterns into a DFA index over the model vocabulary, enabling efficient token-level masking.

**Key features:**
- DFA-based approach with vocabulary-level index caching
- Supports JSON schema, regex, and choice
- Does **not** support EBNF grammar or structural tags
- Regex syntax follows Rust-style regex

**Supported request types:**

| Type | Supported |
|------|-----------|
| `json` | ✅ |
| `regex` | ✅ |
| `choice` | ✅ |
| `grammar` (EBNF) | ❌ |
| `structural_tag` | ❌ |

**Installation:**

```bash
pip install outlines-core
```

### guidance (llguidance)

[llguidance](https://github.com/guidance-ai/llguidance) is the backend library from the [guidance](https://github.com/guidance-ai/guidance) project. It uses a parser-based approach that supports the full range of structured output types.

**Key features:**
- Supports JSON schema, regex, choice, EBNF grammar, and structural tags
- Automatically adds `additionalProperties: false` to JSON schemas for stricter validation
- Detects and rejects JSON schemas with unsupported features (e.g., `patternProperties`)
- Supports speculative decoding with rollback

**Configuration options:**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend guidance \
  --structured-outputs-config.disable_any_whitespace true \
  --structured-outputs-config.disable_additional_properties true
```

**Log level:**

```bash
# Set llguidance log level (0=silent, 1=warnings, 2=info, 3=debug)
export LLGUIDANCE_LOG_LEVEL=1
```

**Installation:**

```bash
pip install llguidance
```

### lm-format-enforcer

[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) is a token-level enforcement library that uses Python's `re` module for regex patterns.

**Key features:**
- Uses Python `re` syntax for regex (not Rust-style)
- Supports JSON schema, regex, and choice
- Does **not** support EBNF grammar, structural tags, or speculative decoding

**Supported request types:**

| Type | Supported |
|------|-----------|
| `json` | ✅ |
| `regex` | ✅ |
| `choice` | ✅ |
| `grammar` (EBNF) | ❌ |
| `structural_tag` | ❌ |

**Installation:**

```bash
pip install lm-format-enforcer
```

## Online Serving (OpenAI API)

You can generate structured outputs using the OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) API.

The following parameters are supported, which must be added as extra parameters:

- `choice`: the output will be exactly one of the choices.
- `regex`: the output will follow the regex pattern.
- `json`: the output will follow the JSON schema.
- `grammar`: the output will follow the context free grammar.
- `structural_tag`: Follow a JSON schema within a set of specified tags within the generated text.

You can see the complete list of supported parameters on the [OpenAI-Compatible Server](../serving/openai_compatible_server.md) page.

Structured outputs are supported by default in the OpenAI-Compatible Server. You
may choose to specify the backend to use by setting the
`--structured-outputs-config.backend` flag to `vllm serve`. The default backend is `auto`,
which will try to choose an appropriate backend based on the details of the
request. You may also choose a specific backend, along with
some options. A full set of options is available in the `vllm serve --help`
text.

Now let's see an example for each of the cases, starting with the `choice`, as it's the easiest one:

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )
    model = client.models.list().data[0].id

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
    )
    print(completion.choices[0].message.content)
    ```

The next example shows how to use the `regex`. The supported regex syntax depends on the structured output backend. For example, `xgrammar`, `guidance`, and `outlines` use Rust-style regex, while `lm-format-enforcer` uses Python's `re` module. The idea is to generate an email address, given a simple regex template:

??? code

    ```python
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
    )
    print(completion.choices[0].message.content)
    ```

One of the most relevant features in structured text generation is the option to generate a valid JSON with pre-defined fields and formats.
For this we can use the `json` parameter in two different ways:

- Using directly a [JSON Schema](https://json-schema.org/)
- Defining a [Pydantic model](https://docs.pydantic.dev/latest/) and then extracting the JSON Schema from it (which is normally an easier option).

The next example shows how to use the `response_format` parameter with a Pydantic model:

??? code

    ```python
    from pydantic import BaseModel
    from enum import Enum

    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    json_schema = CarDescription.model_json_schema()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "car-description",
                "schema": CarDescription.model_json_schema()
            },
        },
    )
    print(completion.choices[0].message.content)
    ```

!!! tip
    While not strictly necessary, normally it's better to indicate in the prompt the
    JSON schema and how the fields should be populated. This can improve the
    results notably in most cases.

Finally we have the `grammar` option, which is probably the most
difficult to use, but it's really powerful. It allows us to define complete
languages like SQL queries. It works by using a context free EBNF grammar.
As an example, we can use to define a specific format of simplified SQL queries:

!!! note
    EBNF grammar is supported by `xgrammar` and `guidance` backends only. It is **not** supported by `outlines` or `lm-format-enforcer`.

??? code

    ```python
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
    )
    print(completion.choices[0].message.content)
    ```

See also: [full example](../examples/online_serving/structured_outputs.md)

## Reasoning Outputs

You can also use structured outputs with <project:#reasoning-outputs> for reasoning models.

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

Note that you can use reasoning with any provided structured outputs feature. The following uses one with JSON schema:

??? code

    ```python
    from pydantic import BaseModel


    class People(BaseModel):
        name: str
        age: int


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the name and age of one random person.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people",
                "schema": People.model_json_schema()
            }
        },
    )
    print("reasoning: ", completion.choices[0].message.reasoning)
    print("content: ", completion.choices[0].message.content)
    ```

See also: [full example](../examples/online_serving/structured_outputs.md)

!!! note
    When using Qwen3 Coder models with reasoning enabled, structured outputs might become disabled if the reasoning content does not get parsed into the `reasoning` field separately (v0.11.2+).
    To use both features together, you must explicitly enable structured outputs in reasoning mode.
    To do so, add the following flag when starting the vLLM server: `--structured-outputs-config.enable_in_reasoning=True`.
    See also: [Reasoning Outputs](reasoning_outputs.md) documentation.

## Experimental Automatic Parsing (OpenAI API)

This section covers the OpenAI beta wrapper over the `client.chat.completions.create()` method that provides richer integrations with Python specific types.

At the time of writing (`openai==1.54.4`), this is a "beta" feature in the OpenAI client library. Code reference can be found [here](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104).

For the following examples, vLLM was set up using `vllm serve meta-llama/Llama-3.1-8B-Instruct`

Here is a simple example demonstrating how to get structured output using Pydantic models:

??? code

    ```python
    from pydantic import BaseModel
    from openai import OpenAI

    class Info(BaseModel):
        name: str
        age: int

    client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
    model = client.models.list().data[0].id
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
        ],
        response_format=Info,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    print("Name:", message.parsed.name)
    print("Age:", message.parsed.age)
    ```

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

Here is a more complex example using nested Pydantic models to handle a step-by-step math solution:

??? code

    ```python
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful expert math tutor."},
            {"role": "user", "content": "Solve 8x + 31 = 2."},
        ],
        response_format=MathResponse,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    for i, step in enumerate(message.parsed.steps):
        print(f"Step #{i}:", step)
    print("Answer:", message.parsed.final_answer)
    ```

Output:

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
Step #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
Step #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
Step #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
Answer: x = -29/8
```

An example of using `structural_tag` can be found here: [examples/online_serving/structured_outputs](../../examples/online_serving/structured_outputs)

## Offline Inference

Offline inference allows for the same types of structured outputs.
To use it, we'll need to configure the structured outputs using the class `StructuredOutputsParams` inside `SamplingParams`.
The main available options inside `StructuredOutputsParams` are:

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

These parameters can be used in the same way as the parameters from the Online
Serving examples above. One example for the usage of the `choice` parameter is
shown below:

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

### Specifying a Backend in Offline Inference

You can specify the backend per-request in offline inference by setting the `_backend` field on `StructuredOutputsParams`:

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    # Use xgrammar backend explicitly
    structured_outputs_params = StructuredOutputsParams(
        choice=["Positive", "Negative"],
        _backend="xgrammar",
    )
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

!!! note
    In the V1 engine, all requests in a single engine instance share the same backend. The backend is initialized on the first structured output request and cannot be changed afterward.

### JSON Schema Example

??? code

    ```python
    import json
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int
        occupation: str

    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    structured_outputs_params = StructuredOutputsParams(
        json=Person.model_json_schema()
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        structured_outputs=structured_outputs_params,
    )

    outputs = llm.generate(
        prompts="Generate a JSON object for a fictional software engineer.",
        sampling_params=sampling_params,
    )

    result = json.loads(outputs[0].outputs[0].text)
    person = Person(**result)
    print(f"Name: {person.name}, Age: {person.age}, Occupation: {person.occupation}")
    ```

### Regex Example

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Generate a date in YYYY-MM-DD format
    structured_outputs_params = StructuredOutputsParams(
        regex=r"\d{4}-\d{2}-\d{2}"
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        structured_outputs=structured_outputs_params,
    )

    outputs = llm.generate(
        prompts="What is today's date?",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

### EBNF Grammar Example

EBNF grammar is supported by `xgrammar` and `guidance` backends:

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Define a grammar for arithmetic expressions
    arithmetic_grammar = """
        root   ::= expr
        expr   ::= term (("+" | "-") term)*
        term   ::= factor (("*" | "/") factor)*
        factor ::= number | "(" expr ")"
        number ::= [0-9]+
    """

    structured_outputs_params = StructuredOutputsParams(grammar=arithmetic_grammar)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        structured_outputs=structured_outputs_params,
    )

    outputs = llm.generate(
        prompts="Write an arithmetic expression for the sum of 3 and 4 multiplied by 2:",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

See also: [full example](../examples/online_serving/structured_outputs.md)

## Advanced Configuration

### Disabling Whitespace Flexibility

By default, xgrammar and guidance allow flexible whitespace in JSON output (e.g., extra spaces or newlines between fields). To enforce strict whitespace matching:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.disable_any_whitespace true
```

### Disabling Additional Properties

By default, guidance automatically adds `additionalProperties: false` to JSON schemas. To disable this behavior:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --structured-outputs-config.backend guidance \
  --structured-outputs-config.disable_additional_properties true
```

### Async Grammar Compilation

Grammar compilation happens asynchronously by default, allowing the engine to continue processing other requests while a new grammar is being compiled. This is disabled automatically in `external_launcher` mode to preserve determinism across tensor-parallel ranks.

### xgrammar Cache Size

The xgrammar backend caches compiled grammars in an LRU cache. Increase the cache size for workloads with many distinct schemas:

```bash
# Set cache to 512 MB (default: 128 MB)
export VLLM_XGRAMMAR_CACHE_MB=512
```

## Troubleshooting

### Grammar Compilation Failures

If grammar compilation fails, check:

1. **JSON schema validity** — ensure the schema is valid JSON Schema (Draft 7 or later).
2. **Backend compatibility** — EBNF grammar requires `xgrammar` or `guidance`; `outlines` and `lm-format-enforcer` do not support it.
3. **Unsupported JSON features** — `guidance` does not support `patternProperties`. Use `xgrammar` for schemas with `patternProperties`.

### Slow First Request

The first request with a new grammar incurs compilation overhead. Subsequent requests with the same grammar are served from the cache. To pre-warm the cache, send a dummy request at startup.

### Speculative Decoding Compatibility

All backends except `lm-format-enforcer` support speculative decoding. If you use speculative decoding with structured outputs, ensure you are not using `lm-format-enforcer`.
