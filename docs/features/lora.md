# LoRA Adapters

This document shows you how to use [LoRA adapters](https://arxiv.org/abs/2106.09685) with vLLM on top of a base model.

LoRA adapters can be used with any vLLM model that implements [SupportsLoRA][vllm.model_executor.models.interfaces.SupportsLoRA].

Adapters can be efficiently served on a per-request basis with minimal overhead. First we download the adapter(s) and save
them locally with

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
```

Then we instantiate the base model and pass in the `enable_lora=True` flag:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)
```

We can now submit the prompts and call `llm.generate` with the `lora_request` parameter. The first parameter
of `LoRARequest` is a human identifiable name, the second parameter is a globally unique ID for the adapter and
the third parameter is the path to the LoRA adapter.

??? code

    ```python
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )
    ```

Check out [examples/offline_inference/multilora_inference.py](../../examples/offline_inference/multilora_inference.py) for an example of how to use LoRA adapters with the async engine and how to use more advanced configuration options.

## Serving LoRA Adapters

LoRA adapted models can also be served with the Open-AI compatible vLLM server. To do so, we use
`--lora-modules {name}={path} {name}={path}` to specify each LoRA module when we kick off the server:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

The server entrypoint accepts all other LoRA configuration parameters (`max_loras`, `max_lora_rank`, `max_cpu_loras`,
etc.), which will apply to all forthcoming requests. Upon querying the `/models` endpoint, we should see our LoRA along
with its base model (if `jq` is not installed, you can follow [this guide](https://jqlang.org/download/) to install it.):

??? console "Command"

    ```bash
    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }
    ```

Requests can specify the LoRA adapter as if it were any other model via the `model` request parameter. The requests will be
processed according to the server-wide LoRA configuration (i.e. in parallel with base model requests, and potentially other
LoRA adapter requests if they were provided and `max_loras` is set high enough).

The following is an example request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## Multi-LoRA Serving

vLLM is designed to serve multiple LoRA adapters simultaneously in a single deployment. This is one of its key
differentiators: rather than running a separate server process per adapter, vLLM batches requests across different
adapters in the same forward pass, sharing the base model weights.

### How It Works

Internally, vLLM maintains two pools for LoRA adapters:

- **GPU slots** (`max_loras`): The number of adapters that can be *active* (loaded into GPU memory) at the same time. Each active adapter occupies a pre-allocated slot in the LoRA weight buffers.
- **CPU cache** (`max_cpu_loras`): The total number of adapters that can be *registered* (held in CPU memory) at once. Must be `>= max_loras`. When the GPU slots are full, the least-recently-used adapter is evicted from GPU memory but remains in CPU memory for fast re-activation.

When a batch arrives containing requests for different adapters, vLLM:

1. Checks which adapters are already active in GPU slots.
2. Loads any missing adapters from CPU cache (or disk) into available GPU slots, evicting the LRU adapter if needed.
3. Constructs a **LoRA mapping** that tells each LoRA-enabled layer which slot to use for each token in the batch.
4. Runs the forward pass, applying the correct adapter weights per token using the [Punica](https://arxiv.org/abs/2310.18547) batched GEMM kernels.

This design means that serving 8 different adapters concurrently costs almost no extra memory beyond the `max_loras` GPU slots, and the base model weights are shared across all of them.

### Configuring Multi-LoRA

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-cpu-loras 16 \
    --max-lora-rank 64 \
    --lora-modules \
        sql-lora=jeeejeee/llama32-3b-text2sql-spider \
        code-lora=/path/to/code-adapter \
        chat-lora=/path/to/chat-adapter
```

| Parameter | Default | Description |
|---|---|---|
| `--max-loras` | `1` | Maximum adapters active in GPU simultaneously |
| `--max-cpu-loras` | same as `--max-loras` | Maximum adapters held in CPU memory |
| `--max-lora-rank` | `16` | Maximum LoRA rank across all adapters |
| `--fully-sharded-loras` | `False` | Enable fully sharded LoRA computation (faster at high rank/TP) |

### Pinning Adapters

If you have adapters that are used very frequently and should never be evicted from GPU memory, you can pin them
programmatically using the `pin_lora` method on the engine:

```python
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_cpu_loras=16,
)

# Pin a frequently-used adapter so it is never evicted from GPU
llm.llm_engine.pin_lora(lora_id=1)
```

Pinned adapters count against `max_loras` but are excluded from LRU eviction.

## Dynamically Serving LoRA Adapters

In addition to serving LoRA adapters at server startup, the vLLM server supports dynamically configuring LoRA adapters at runtime through dedicated API endpoints and plugins. This feature can be particularly useful when the flexibility to change models on-the-fly is needed.

!!! warning
    This feature comes with security risks. It should not be used in production unless it is an isolated, fully trusted environment.

To enable dynamic LoRA configuration, ensure that the environment variable `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
is set to `True`.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### Using API Endpoints

Loading a LoRA Adapter:

To dynamically load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter` endpoint with the necessary
details of the adapter to be loaded. The request payload should include the name and path to the LoRA adapter.

Example request to load a LoRA adapter:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

Upon a successful request, the API will respond with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' added successfully`. If an error occurs, such as if the adapter
cannot be found or loaded, an appropriate error message will be returned.

Unloading a LoRA Adapter:

To unload a LoRA adapter that has been previously loaded, send a POST request to the `/v1/unload_lora_adapter` endpoint
with the name or ID of the adapter to be unloaded.

Upon a successful request, the API responds with a `200 OK` status code from `vllm serve`, and `curl` returns the response body: `Success: LoRA adapter 'sql_adapter' removed successfully`.

Example request to unload a LoRA adapter:

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### Using Resolver Plugins

Alternatively, you can use the LoRAResolver plugin system to dynamically load LoRA adapters. LoRAResolver plugins enable you to load LoRA adapters from both local and remote sources such as local file system and S3. On every request, when there's a new model name that hasn't been loaded yet, the LoRAResolver will try to resolve and load the corresponding LoRA adapter.

You can set up multiple LoRAResolver plugins if you want to load LoRA adapters from different sources. For example, you might have one resolver for local files and another for S3 storage. vLLM will load the first LoRA adapter that it finds.

You can either install existing plugins or implement your own. By default, vLLM comes with a [resolver plugin to load LoRA adapters from a local directory, as well as a resolver plugin to load LoRA adapters from repositories on Hugging Face Hub](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers).
To enable either of these resolvers, you must set `VLLM_ALLOW_RUNTIME_LORA_UPDATING` to `True`.

#### Filesystem Resolver

The filesystem resolver scans a local directory for LoRA adapters. When vLLM receives a request for an unknown model name, it looks for a subdirectory with that name inside `VLLM_LORA_RESOLVER_CACHE_DIR` and validates the `adapter_config.json` inside it.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
```

Expected directory layout:

```text
/path/to/lora/adapters/
├── sql-adapter/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (if applicable)
├── code-adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── ...
```

Each `adapter_config.json` must contain `"peft_type": "LORA"` and a `base_model_name_or_path` that matches the running base model. When vLLM receives a request for `sql-adapter`, it automatically loads the adapter from the directory and makes it available for future requests — no server restart required.

#### Hugging Face Hub Resolver

The HF Hub resolver downloads adapters on demand from specified Hugging Face repositories.

!!! warning
    Enabling remote downloads is a security risk. This resolver is **not** intended for production environments.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_PLUGINS=lora_hf_hub_resolver
export VLLM_LORA_RESOLVER_HF_REPO_LIST=my-org/my-lora-repo,another-org/another-repo
```

When vLLM receives a request for the LoRA adapter `my-org/my-lora-repo/subpath`, it:

1. Matches `my-org/my-lora-repo` against the allowed repository list.
2. Lists files in the repository to find subdirectories containing `adapter_config.json`.
3. Downloads the matching subdirectory via `snapshot_download`.
4. Validates and loads the adapter from the local cache.

Adapter names follow the pattern `<org>/<repo>/<subpath>` (or just `<org>/<repo>` if the adapter is at the repo root).

#### Multiple Resolvers

You can enable multiple resolvers simultaneously. vLLM tries each resolver in the order they are listed until one succeeds:

```bash
export VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver
```

#### Custom Resolver Implementation

Follow these steps to implement your own resolver plugin:

1. **Implement the `LoRAResolver` interface.**

    ??? code "Example: S3 LoRAResolver"

        ```python
        import os
        import s3fs
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver

        class S3LoRAResolver(LoRAResolver):
            def __init__(self):
                self.s3 = s3fs.S3FileSystem()
                self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
                self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

            async def resolve_lora(self, base_model_name, lora_name):
                s3_path = self.s3_path_format.format(
                    base_model_name=base_model_name, lora_name=lora_name
                )
                local_path = self.local_path_format.format(
                    base_model_name=base_model_name, lora_name=lora_name
                )

                # Download the LoRA from S3 to the local path
                await self.s3._get(
                    s3_path, local_path, recursive=True, maxdepth=1
                )

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=local_path,
                    lora_int_id=abs(hash(lora_name)),
                )
                return lora_request
        ```

2. **Register the resolver** with the global registry.

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    For more details, refer to [vLLM's Plugins System](../design/plugin_system.md).

### In-Place LoRA Reloading

When dynamically loading LoRA adapters, you may need to replace an existing adapter with updated weights while keeping the same name. The `load_inplace` parameter enables this functionality. This commonly occurs in asynchronous reinforcement learning setups, where adapters are continuously updated and swapped in without interrupting ongoing inference.

When `load_inplace=True`, vLLM will replace the existing adapter with the new one.

Example request to load or replace a LoRA adapter with the same name:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter/v2",
    "load_inplace": true
}'
```

## New format for `--lora-modules`

In the previous version, users would provide LoRA modules via the following format, either as a key-value pair or in JSON format. For example:

```bash
--lora-modules  sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

This would only include the `name` and `path` for each LoRA module, but did not provide a way to specify a `base_model_name`.
Now, you can specify a base_model_name alongside the name and path using JSON format. For example:

```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

To provide the backward compatibility support, you can still use the old key-value format (name=path), but the `base_model_name` will remain unspecified in that case.

## LoRA model lineage in model card

The new format of `--lora-modules` is mainly to support the display of parent model information in the model card. Here's an explanation of how your current response supports this:

- The `parent` field of LoRA model `sql-lora` now links to its base model `meta-llama/Llama-3.2-3B-Instruct`. This correctly reflects the hierarchical relationship between the base model and the LoRA adapter.
- The `root` field points to the artifact location of the lora adapter.

??? console "Command output"

    ```bash
    $ curl http://localhost:8000/v1/models

    {
        "object": "list",
        "data": [
            {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "meta-llama/Llama-3.2-3B-Instruct",
            "parent": null,
            "permission": [
                {
                .....
                }
            ]
            },
            {
            "id": "sql-lora",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "jeeejeee/llama32-3b-text2sql-spider",
            "parent": "meta-llama/Llama-3.2-3B-Instruct",
            "permission": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```

## LoRA Support for Tower and Connector of Multi-Modal Model

Currently, vLLM experimentally supports LoRA for the Tower and Connector components of multi-modal models. To enable this feature, you need to implement the corresponding token helper functions for the tower and connector. For more details on the rationale behind this approach, please refer to [PR 26674](https://github.com/vllm-project/vllm/pull/26674). We welcome contributions to extend LoRA support to additional models' tower and connector. Please refer to [Issue 31479](https://github.com/vllm-project/vllm/issues/31479) to check the current model support status.

## Default LoRA Models For Multimodal Models

Some models, e.g., [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) and [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) multimodal, contain LoRA adapter(s) that are expected to always be applied when a given modality is present. This can be a bit tedious to manage with the above approaches, as it requires the user to send the `LoRARequest` (offline) or to filter requests between the base model and LoRA model (server) depending on the content of the request's multimodal data.

To this end, we allow registration of default multimodal LoRAs to handle this automatically, where users can map each modality to a LoRA adapter to automatically apply it when the corresponding inputs are present. Note that currently, we only allow one LoRA per prompt; if several modalities are provided, each of which are registered to a given modality, none of them will be applied.

??? code "Example usage for offline inference"

    ```python
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    model_id = "ibm-granite/granite-speech-3.3-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_prompt(question: str, has_audio: bool):
        """Build the input prompt to send to vLLM."""
        if has_audio:
            question = f"<|audio|>{question}"
        chat = [
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False)


    llm = LLM(
        model=model_id,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
        # Will always pass a `LoRARequest` with the `model_id`
        # whenever audio is contained in the request data.
        default_mm_loras = {"audio": model_id},
        enforce_eager=True,
    )

    question = "can you transcribe the speech into a written format?"
    prompt_with_audio = get_prompt(
        question=question,
        has_audio=True,
    )
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

    inputs = {
        "prompt": prompt_with_audio,
        "multi_modal_data": {
            "audio": audio,
        }
    }


    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=0.2,
            max_tokens=64,
        ),
    )
    ```

You can also pass a json dictionary of `--default-mm-loras` mapping modalities to LoRA model IDs. For example, when starting the server:

```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --max-model-len 2048 \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

Note: Default multimodal LoRAs are currently only available for `.generate` and chat completions.

## Configuration Reference

The table below summarises every `LoRAConfig` field and its corresponding CLI flag.

| Python field | CLI flag | Default | Description |
|---|---|---|---|
| `max_lora_rank` | `--max-lora-rank` | `16` | Maximum rank allowed for any loaded adapter. Must be one of: 1, 8, 16, 32, 64, 128, 256, 320, 512. |
| `max_loras` | `--max-loras` | `1` | Maximum number of adapters active in GPU at the same time. |
| `max_cpu_loras` | `--max-cpu-loras` | same as `max_loras` | Maximum adapters held in CPU memory. Must be `>= max_loras`. |
| `lora_dtype` | `--lora-dtype` | `"auto"` | Data type for LoRA weights. `"auto"` inherits the base model dtype. |
| `fully_sharded_loras` | `--fully-sharded-loras` | `False` | Use fully sharded LoRA layers. Faster at high sequence length, rank, or tensor-parallel size. |
| `specialize_active_lora` | `--specialize-active-lora` | `False` | Capture separate CUDA graphs for each power-of-2 count of active LoRAs. Improves throughput at the cost of longer startup. |
| `enable_tower_connector_lora` | `--enable-tower-connector-lora` | `False` | Enable experimental LoRA for vision tower and connector in multimodal models. |
| `default_mm_loras` | `--default-mm-loras` | `None` | JSON dict mapping modality names to LoRA model paths for automatic multimodal LoRA application. |

## Using Tips

### Configuring `max_lora_rank`

The `--max-lora-rank` parameter controls the maximum rank allowed for LoRA adapters. This setting affects memory allocation and performance:

- **Set it to the maximum rank** among all LoRA adapters you plan to use
- **Avoid setting it too high** - using a value much larger than needed wastes memory and can cause performance issues

For example, if your LoRA adapters have ranks [16, 32, 64], use `--max-lora-rank 64` rather than 256

```bash
# Good: matches actual maximum rank
vllm serve model --enable-lora --max-lora-rank 64

# Bad: unnecessarily high, wastes memory
vllm serve model --enable-lora --max-lora-rank 256
```

### Supported LoRA Features

vLLM supports the following PEFT LoRA variants:

- **Standard LoRA** — the default low-rank decomposition.
- **rsLoRA** (`use_rslora: true` in `adapter_config.json`) — Rank-Stabilized LoRA, which scales the adapter by `alpha / sqrt(r)` instead of `alpha / r`. vLLM detects this automatically from the adapter config.

The following features are **not yet supported**:

- **DoRA** (`use_dora: true`) — Weight-Decomposed Low-Rank Adaptation.
- `modules_to_save` — saving non-LoRA modules alongside the adapter.

### Supported Target Modules

vLLM can apply LoRA to any linear layer that is registered as a supported module in the model's `SupportsLoRA` implementation. Common target modules include:

- `q_proj`, `k_proj`, `v_proj`, `o_proj` — attention projections
- `gate_proj`, `up_proj`, `down_proj` — MLP / feed-forward projections
- `embed_tokens`, `lm_head` — embedding and output layers

The exact set of supported modules depends on the model architecture. Refer to the model's source file for the `supported_lora_modules` list.

### Adapter Identity and Caching

`LoRARequest` uses `lora_name` (not `lora_int_id`) for equality and hashing. This means:

- Two `LoRARequest` objects with the same `lora_name` but different `lora_int_id` values are considered equal.
- `lora_int_id` must be globally unique and greater than 0. It is used internally to index into the GPU slot array.
- When using the dynamic loading API, vLLM derives `lora_int_id` from `abs(hash(lora_name))`.

### Performance Considerations

- **Tensor parallelism**: By default, only half of the LoRA computation is sharded across tensor-parallel ranks. Enable `--fully-sharded-loras` to shard all LoRA layers, which is faster at high rank or large TP degree.
- **CUDA graphs**: When `--enforce-eager` is not set, vLLM captures CUDA graphs for LoRA inference. Use `--specialize-active-lora` to capture separate graphs for each power-of-2 count of active adapters, reducing overhead when the number of active adapters varies.
- **CPU offloading**: Adapters not currently needed in GPU are kept in CPU memory (up to `max_cpu_loras`). Reactivating a CPU-cached adapter is much faster than reloading from disk.

## Further Reading

- [LoRA Resolver Plugins Design](../design/lora_resolver_plugins.md) — deep dive into the resolver plugin architecture.
- [Multi-LoRA inference example](../../examples/offline_inference/multilora_inference.py) — offline inference with multiple adapters.
- [LoRA paper](https://arxiv.org/abs/2106.09685) — the original LoRA paper by Hu et al.
- [rsLoRA paper](https://arxiv.org/abs/2312.03732) — Rank-Stabilized LoRA.
- [Punica paper](https://arxiv.org/abs/2310.18547) — the batched GEMM kernel used for multi-LoRA inference.
