# Plugin System

The community frequently requests the ability to extend vLLM with custom features. To facilitate this, vLLM includes a plugin system that allows users to add custom features without modifying the vLLM codebase. This document explains how plugins work in vLLM and how to create a plugin for vLLM.

## How Plugins Work in vLLM

Plugins are user-registered code that vLLM executes. Given vLLM's architecture (see [Arch Overview](arch_overview.md)), multiple processes may be involved, especially when using distributed inference with various parallelism techniques. To enable plugins successfully, every process created by vLLM needs to load the plugin. This is done by the [load_plugins_by_group][vllm.plugins.load_plugins_by_group] function in the `vllm.plugins` module.

### Plugin Loading Lifecycle

Different plugin groups are loaded at different points in the vLLM process lifecycle:

| Plugin Group | Loaded In | When |
|---|---|---|
| `vllm.general_plugins` | All processes (process0, engine core, workers) | At engine startup |
| `vllm.platform_plugins` | All processes | When `current_platform` is first accessed |
| `vllm.io_processor_plugins` | Process0 only | At engine startup, when an IO processor is requested |
| `vllm.stat_logger_plugins` | Process0 only | At engine startup in async serving mode |

## How vLLM Discovers Plugins

vLLM's plugin system uses the standard Python `entry_points` mechanism. This mechanism allows developers to register functions in their Python packages for use by other packages.

### Entry Points Reference

vLLM ships with the following built-in entry points defined in `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
lora_filesystem_resolver = "vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver"
lora_hf_hub_resolver     = "vllm.plugins.lora_resolvers.hf_hub_resolver:register_hf_hub_resolver"
```

These are the LoRA resolver plugins that ship with vLLM itself. Third-party packages declare their own entry points in the same way.

### Minimal Plugin Example

??? code "General plugin — registering a custom model"

    ```python
    # inside `setup.py`
    from setuptools import setup

    setup(
        name="vllm_add_dummy_model",
        version="0.1",
        packages=["vllm_add_dummy_model"],
        entry_points={
            "vllm.general_plugins": [
                "register_dummy_model = vllm_add_dummy_model:register"
            ]
        },
    )

    # inside `vllm_add_dummy_model/__init__.py`
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

??? code "Same example using pyproject.toml"

    ```toml
    # pyproject.toml
    [project.entry-points."vllm.general_plugins"]
    register_dummy_model = "vllm_add_dummy_model:register"
    ```

For more information on adding entry points to your package, please check the [official setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).

### Anatomy of a Plugin Entry Point

Every plugin has three parts:

1. **Plugin group**: The name of the entry point group. This is the key of `entry_points` in `setup.py` (or `[project.entry-points."<group>"]` in `pyproject.toml`). Always use `vllm.general_plugins` for general-purpose vLLM plugins.

2. **Plugin name**: The name of the plugin. This is the key in the entry point mapping (e.g., `register_dummy_model`). Plugins can be filtered by their names using the `VLLM_PLUGINS` environment variable. To load only a specific plugin, set `VLLM_PLUGINS` to the plugin name.

3. **Plugin value**: The fully qualified name of the function or class to register. In the example above, `vllm_add_dummy_model:register` refers to a function named `register` in the `vllm_add_dummy_model` module.

## Controlling Which Plugins Load

Use the `VLLM_PLUGINS` environment variable to control which plugins are loaded:

```bash
# Load all available plugins (default when VLLM_PLUGINS is unset)
# (no action needed)

# Load only a specific plugin
export VLLM_PLUGINS=register_dummy_model

# Load multiple specific plugins
export VLLM_PLUGINS=register_dummy_model,lora_filesystem_resolver

# Disable all plugins
export VLLM_PLUGINS=
```

!!! note
    When `VLLM_PLUGINS` is unset (`None`), **all** discovered plugins are loaded. When set to an empty string, **no** plugins are loaded. When set to a comma-separated list, only the named plugins are loaded.

## Types of Supported Plugins

### General Plugins (`vllm.general_plugins`)

The primary use case for these plugins is to register custom, out-of-the-tree models into vLLM. This is done by calling `ModelRegistry.register_model` inside the plugin function.

**Example — registering a custom model:**

```python
# vllm_my_model/__init__.py
def register():
    from vllm import ModelRegistry

    if "MyCustomModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyCustomModel",
            "vllm_my_model.model:MyCustomModel",
        )
```

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
my_custom_model = "vllm_my_model:register"
```

For an official example, see the [bart-plugin](https://github.com/vllm-project/bart-plugin) which adds support for `BartForConditionalGeneration`.

---

### Platform Plugins (`vllm.platform_plugins`)

The primary use case for these plugins is to register custom, out-of-the-tree hardware platforms into vLLM. The plugin function should return `None` when the platform is not supported in the current environment, or the platform class's fully qualified name when the platform is supported.

**Example — registering a custom platform:**

```python
# vllm_my_platform/__init__.py
def register():
    # Return None if the hardware is not present
    try:
        import my_hardware_sdk
    except ImportError:
        return None
    return "vllm_my_platform.platform:MyPlatform"
```

```toml
# pyproject.toml
[project.entry-points."vllm.platform_plugins"]
my_platform = "vllm_my_platform:register"
```

See [Adding a Platform Plugin](../contributing/adding_platform.md) for a complete step-by-step guide.

---

### IO Processor Plugins (`vllm.io_processor_plugins`)

The primary use case for these plugins is to register custom pre-/post-processing of the model prompt and model output for pooling models. The plugin function returns the `IOProcessor` class's fully qualified name.

**Example — registering an IO processor:**

```python
# vllm_my_io_processor/__init__.py
def register():
    return "vllm_my_io_processor.processor:MyIOProcessor"
```

```toml
# pyproject.toml
[project.entry-points."vllm.io_processor_plugins"]
my_io_processor = "vllm_my_io_processor:register"
```

See [IO Processor Plugins](io_processor_plugins.md) for a complete guide.

---

### Stat Logger Plugins (`vllm.stat_logger_plugins`)

The primary use case for these plugins is to register custom, out-of-the-tree metric loggers into vLLM. The entry point should be a **class** (not a function) that subclasses `StatLoggerBase`.

**Example — registering a custom stat logger:**

```python
# vllm_my_logger/logger.py
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.config import VllmConfig
from vllm.v1.metrics.stats import SchedulerStats, IterationStats, MultiModalCacheStats

class MyCustomLogger(StatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        # Send metrics to your custom backend
        if iteration_stats:
            print(f"[Engine {engine_idx}] Tokens generated: {iteration_stats.num_generation_tokens}")

    def log_engine_initialized(self):
        print(f"[Engine {self.engine_index}] Initialized!")
```

```toml
# pyproject.toml
[project.entry-points."vllm.stat_logger_plugins"]
my_custom_logger = "vllm_my_logger.logger:MyCustomLogger"
```

!!! note
    Unlike other plugin types, the stat logger entry point value is the **class itself**, not a factory function that returns a class name string.

---

### LoRA Resolver Plugins (via `vllm.general_plugins`)

LoRA resolver plugins are a specialised form of general plugin that enable dynamic, on-demand loading of LoRA adapters. They are registered under `vllm.general_plugins` and use the `LoRAResolverRegistry`.

vLLM ships two built-in LoRA resolver plugins:

| Plugin Name | Entry Point | Description |
|---|---|---|
| `lora_filesystem_resolver` | `vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver` | Loads LoRA adapters from a local directory |
| `lora_hf_hub_resolver` | `vllm.plugins.lora_resolvers.hf_hub_resolver:register_hf_hub_resolver` | Downloads LoRA adapters from Hugging Face Hub |

See [LoRA Resolver Plugins](lora_resolver_plugins.md) for full documentation.

## Guidelines for Writing Plugins

### Re-entrancy

The function specified in the entry point should be **re-entrant**, meaning it can be called multiple times without causing issues. This is necessary because the function might be called multiple times in some processes.

```python
# Good — idempotent registration
def register():
    from vllm import ModelRegistry
    if "MyModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyModel", "my_pkg.model:MyModel")

# Bad — will raise an error on the second call
def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model("MyModel", "my_pkg.model:MyModel")  # raises if already registered
```

### Error Handling

Plugins that fail to load are logged as exceptions but do not crash vLLM. However, if a required plugin is missing (e.g., an IO processor plugin that a model depends on), vLLM will raise a `ValueError` at runtime.

### Versioning

Always pin the vLLM version range your plugin supports. The internal interfaces (model runner, worker, attention backend) may change between vLLM releases.

## Platform Plugin Guidelines

### Project Structure

Create a platform plugin project, for example, `vllm_add_dummy_platform`. The project structure should look like this:

```shell
vllm_add_dummy_platform/
├── vllm_add_dummy_platform/
│   ├── __init__.py
│   ├── my_dummy_platform.py
│   ├── my_dummy_worker.py
│   ├── my_dummy_attention.py
│   ├── my_dummy_device_communicator.py
│   └── my_dummy_custom_ops.py
└── setup.py
```

### Entry Point Registration

In the `setup.py` file, add the following entry point:

```python
setup(
    name="vllm_add_dummy_platform",
    ...
    entry_points={
        "vllm.platform_plugins": [
            "my_dummy_platform = vllm_add_dummy_platform:register"
        ]
    },
    ...
)
```

Make sure `vllm_add_dummy_platform:register` is a callable function and returns the platform class's fully qualified name:

```python
def register():
    return "vllm_add_dummy_platform.my_dummy_platform.MyDummyPlatform"
```

### Platform Class Implementation

Implement the platform class `MyDummyPlatform` in `my_dummy_platform.py`. The platform class should inherit from `vllm.platforms.interface.Platform`. There are some important functions and properties that should be implemented at least:

- `_enum`: This property is the device enumeration from [PlatformEnum][vllm.platforms.interface.PlatformEnum]. Usually, it should be `PlatformEnum.OOT`, which means the platform is out-of-tree.
- `device_type`: This property should return the type of the device which pytorch uses. For example, `"cpu"`, `"cuda"`, etc.
- `device_name`: This property is set the same as `device_type` usually. It's mainly used for logging purposes.
- `check_and_update_config`: This function is called very early in the vLLM's initialization process. It's used for plugins to update the vllm configuration. For example, the block size, graph mode config, etc., can be updated in this function. The most important thing is that the **worker_cls** should be set in this function to let vLLM know which worker class to use for the worker process.
- `get_attn_backend_cls`: This function should return the attention backend class's fully qualified name.
- `get_device_communicator_cls`: This function should return the device communicator class's fully qualified name.

### Worker Class Implementation

Implement the worker class `MyDummyWorker` in `my_dummy_worker.py`. The worker class should inherit from [WorkerBase][vllm.v1.worker.worker_base.WorkerBase]. All interfaces in the base class should be implemented. To make sure a model can be executed, the basic functions that should be implemented are:

- `init_device`: Set up the device for the worker.
- `initialize_cache`: Set cache config for the worker.
- `load_model`: Load the model weights to device.
- `get_kv_cache_spec`: Generate the KV cache spec for the model.
- `determine_available_memory`: Profile the peak memory usage of the model to determine how much memory can be used for KV cache without OOMs.
- `initialize_from_config`: Allocate device KV cache with the specified `kv_cache_config`.
- `execute_model`: Called every step to run inference.

Additional optional functions:

- `sleep` and `wakeup`: Support sleep mode feature.
- `compile_or_warm_up_model`: Support graph mode feature.
- `take_draft_token_ids`: Support speculative decoding feature.
- `add_lora`, `remove_lora`, `list_loras`, `pin_lora`: Support LoRA feature.
- `execute_dummy_batch`: Support data parallelism feature.

### Attention Backend Implementation

Implement the attention backend class `MyDummyAttention` in `my_dummy_attention.py`. The attention backend class should inherit from [AttentionBackend][vllm.v1.attention.backend.AttentionBackend]. It's used to calculate attentions with your device. Take `vllm.v1.attention.backends` as examples, it contains many attention backend implementations.

### Custom Ops

Most ops can be run by PyTorch native implementation, while the performance may not be good. In this case, you can implement specific custom ops for your plugins. Currently, there are kinds of custom ops vLLM supports:

- **PyTorch ops**:
    - `communicator ops`: Device communicator ops such as all-reduce, all-gather, etc. Implement the device communicator class `MyDummyDeviceCommunicator` in `my_dummy_device_communicator.py`. The device communicator class should inherit from [DeviceCommunicatorBase][vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase].
    - `common ops`: Common ops such as matmul, softmax, etc. Implement these by registering them via the OOT way. See more detail in [CustomOp][vllm.model_executor.custom_op.CustomOp] class.
    - `csrc ops`: C++ ops implemented in C++ and registered as torch custom ops. Follow the `csrc` module and `vllm._custom_ops` to implement your ops.

- **Triton ops**: Custom way doesn't work for triton ops now.

### Optional Pluggable Modules

You may also implement other pluggable modules, such as LoRA, graph backend, quantization, Mamba attention backend, etc.

## Compatibility Guarantee

vLLM guarantees the interface of documented plugins, such as `ModelRegistry.register_model`, will always be available for plugins to register models. However, it is the responsibility of plugin developers to ensure their plugins are compatible with the version of vLLM they are targeting. For example, `"vllm_add_dummy_model.my_llava:MyLlava"` should be compatible with the version of vLLM that the plugin targets.

The interface for the model/module may change during vLLM's development. If you see any deprecation log info, please upgrade your plugin to the latest version.

## Deprecation Announcements

!!! warning "Deprecations"
    - `use_v1` parameter in `Platform.get_attn_backend_cls` is deprecated. It has been removed in v0.13.0.
    - `_Backend` in `vllm.attention` is deprecated. It has been removed in v0.13.0. Please use `vllm.v1.attention.backends.registry.register_backend` to add new attention backend to `AttentionBackendEnum` instead.
    - `seed_everything` platform interface is deprecated. It has been removed in v0.16.0. Please use `vllm.utils.torch_utils.set_random_seed` instead.
    - `prompt` in `Platform.validate_request` is deprecated and will be removed in v0.18.0.
    - `parse_request` in `IOProcessor` has been renamed to `parse_data`. The old name will be removed in v0.19.
    - `validate_or_generate_params` in `IOProcessor` has been split into `merge_sampling_params` and `merge_pooling_params`. The old name will be removed in v0.19.
    - The `renderer` argument in `IOProcessor.__init__` will be required in v0.18.
