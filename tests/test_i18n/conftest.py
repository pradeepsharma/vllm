# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conftest for i18n tests.

This conftest stubs out the vllm package to avoid importing torch and other
heavy dependencies that are not needed for i18n unit tests.

It runs BEFORE any test collection so that `import vllm.i18n` works without
triggering vllm/__init__.py (which requires torch, cuda, etc.).
"""

import sys
import types
import importlib
import importlib.util
import pathlib


def _stub_and_load_i18n():
    """Stub vllm package and load i18n submodules directly."""
    root = pathlib.Path(__file__).parent.parent.parent

    # Only stub if the real vllm package hasn't been loaded yet
    if 'vllm' not in sys.modules:
        vllm_stub = types.ModuleType('vllm')
        vllm_stub.__path__ = [str(root / 'vllm')]
        vllm_stub.__package__ = 'vllm'
        sys.modules['vllm'] = vllm_stub

    # Create vllm.i18n package stub first (needed for relative imports)
    if 'vllm.i18n' not in sys.modules:
        i18n_pkg_stub = types.ModuleType('vllm.i18n')
        i18n_pkg_stub.__path__ = [str(root / 'vllm' / 'i18n')]
        i18n_pkg_stub.__package__ = 'vllm.i18n'
        sys.modules['vllm.i18n'] = i18n_pkg_stub

    # Load vllm.i18n._config
    if 'vllm.i18n._config' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'vllm.i18n._config',
            root / 'vllm' / 'i18n' / '_config.py'
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'vllm.i18n'
        sys.modules['vllm.i18n._config'] = mod
        spec.loader.exec_module(mod)

    # Load vllm.i18n._lazy
    if 'vllm.i18n._lazy' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'vllm.i18n._lazy',
            root / 'vllm' / 'i18n' / '_lazy.py'
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'vllm.i18n'
        sys.modules['vllm.i18n._lazy'] = mod
        spec.loader.exec_module(mod)

    # Now load the real vllm.i18n __init__ (replacing the stub)
    spec = importlib.util.spec_from_file_location(
        'vllm.i18n',
        root / 'vllm' / 'i18n' / '__init__.py',
        submodule_search_locations=[str(root / 'vllm' / 'i18n')]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'vllm.i18n'
    sys.modules['vllm.i18n'] = mod
    spec.loader.exec_module(mod)


# Run immediately when conftest is imported (before test collection)
_stub_and_load_i18n()
