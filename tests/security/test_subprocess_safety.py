# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Security tests: verify that subprocess calls in vllm/platforms/cpu.py
do NOT use shell=True (shell-injection risk).

These tests use AST analysis of the source file to confirm that every
subprocess.check_output / subprocess.run / subprocess.Popen call:
  1. Passes a *list* as the first argument (not a bare string).
  2. Does NOT include the keyword argument shell=True.

This approach is independent of the runtime environment (no GPU, no lscpu,
no sysctl, no heavy vllm deps required) and gives a deterministic, fast result.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CPU_PY = REPO_ROOT / "vllm" / "platforms" / "cpu.py"


def _load_source() -> str:
    return CPU_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AST-based static analysis helpers
# ---------------------------------------------------------------------------

SUBPROCESS_FUNCS = {
    "check_output",
    "run",
    "Popen",
    "call",
    "check_call",
}


class SubprocessCallVisitor(ast.NodeVisitor):
    """Collect every subprocess.<func>(...) call in the AST."""

    def __init__(self):
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call):  # noqa: N802
        func = node.func
        # Match `subprocess.check_output(...)` style
        if (
            isinstance(func, ast.Attribute)
            and func.attr in SUBPROCESS_FUNCS
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            self.calls.append(node)
        self.generic_visit(node)


def _get_subprocess_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    visitor = SubprocessCallVisitor()
    visitor.visit(tree)
    return visitor.calls


def _has_shell_true(call: ast.Call) -> bool:
    """Return True if the call contains the keyword argument shell=True."""
    for kw in call.keywords:
        if kw.arg == "shell":
            # shell=True  →  Constant(value=True)
            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
    return False


def _first_arg_is_list(call: ast.Call) -> bool:
    """Return True if the first positional argument is a list literal."""
    if not call.args:
        return False
    return isinstance(call.args[0], ast.List)


def _first_arg_is_string(call: ast.Call) -> bool:
    """Return True if the first positional argument is a string literal."""
    if not call.args:
        return False
    return isinstance(call.args[0], ast.Constant) and isinstance(
        call.args[0].value, str
    )


def _extract_list_elements(call: ast.Call) -> list:
    """Extract string elements from a list literal first argument."""
    if not call.args or not isinstance(call.args[0], ast.List):
        return []
    return [
        elt.value
        for elt in call.args[0].elts
        if isinstance(elt, ast.Constant)
    ]


# ---------------------------------------------------------------------------
# Phase 1 — Subprocess Safety (AST-level)
# ---------------------------------------------------------------------------


class TestSubprocessSafety:
    """Verify shell=True has been removed from all subprocess calls."""

    def test_cpu_py_exists(self):
        """The source file must exist."""
        assert CPU_PY.exists(), f"Expected source file not found: {CPU_PY}"

    def test_no_shell_true_in_vllm_python_files(self):
        """
        Grep-level check: no shell=True should appear anywhere in vllm/*.py.
        This mirrors the plan-defined verification criterion exactly.
        """
        vllm_dir = REPO_ROOT / "vllm"
        violations = []
        for py_file in vllm_dir.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "shell=True" in line:
                    violations.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                    )
        assert violations == [], (
            "shell=True found in vllm/ Python files:\n" + "\n".join(violations)
        )

    def test_cpu_py_syntax_valid(self):
        """cpu.py must parse without SyntaxError."""
        source = _load_source()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"SyntaxError in {CPU_PY}: {exc}")

    def test_subprocess_calls_use_list_not_string(self):
        """
        Every subprocess call in cpu.py must pass a list as the first
        argument, not a bare string (which would require shell=True to work).
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)
        assert calls, "Expected at least one subprocess call in cpu.py"

        string_arg_calls = [c for c in calls if _first_arg_is_string(c)]
        assert string_arg_calls == [], (
            f"{len(string_arg_calls)} subprocess call(s) still use a bare string "
            f"as the first argument (shell-injection risk). "
            f"Lines: {[c.lineno for c in string_arg_calls]}"
        )

    def test_subprocess_calls_no_shell_true(self):
        """
        Every subprocess call in cpu.py must NOT include shell=True.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)
        assert calls, "Expected at least one subprocess call in cpu.py"

        shell_true_calls = [c for c in calls if _has_shell_true(c)]
        assert shell_true_calls == [], (
            f"{len(shell_true_calls)} subprocess call(s) still use shell=True. "
            f"Lines: {[c.lineno for c in shell_true_calls]}"
        )

    def test_sysctl_call_is_list_form(self):
        """
        The sysctl call (macOS ARM BF16 detection) must use list form:
        ['sysctl', '-n', 'hw.optional.arm.FEAT_BF16']
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        sysctl_calls = []
        for c in calls:
            elts = _extract_list_elements(c)
            if elts and elts[0] == "sysctl":
                sysctl_calls.append(c)

        assert len(sysctl_calls) == 1, (
            f"Expected exactly 1 sysctl subprocess call, found {len(sysctl_calls)}"
        )
        c = sysctl_calls[0]
        assert not _has_shell_true(c), (
            f"sysctl call at line {c.lineno} still uses shell=True"
        )
        arg_values = _extract_list_elements(c)
        assert arg_values == ["sysctl", "-n", "hw.optional.arm.FEAT_BF16"], (
            f"sysctl call args mismatch: {arg_values}"
        )

    def test_lscpu_call_is_list_form(self):
        """
        The lscpu call (Linux CPU topology) must use list form:
        ['lscpu', '-J', '-e=CPU,CORE,NODE']
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        lscpu_calls = []
        for c in calls:
            elts = _extract_list_elements(c)
            if elts and elts[0] == "lscpu":
                lscpu_calls.append(c)

        assert len(lscpu_calls) == 1, (
            f"Expected exactly 1 lscpu subprocess call, found {len(lscpu_calls)}"
        )
        c = lscpu_calls[0]
        assert not _has_shell_true(c), (
            f"lscpu call at line {c.lineno} still uses shell=True"
        )
        arg_values = _extract_list_elements(c)
        assert arg_values == ["lscpu", "-J", "-e=CPU,CORE,NODE"], (
            f"lscpu call args mismatch: {arg_values}"
        )

    def test_total_subprocess_calls_count(self):
        """
        cpu.py should have exactly 2 subprocess calls (sysctl + lscpu).
        If this count changes, the test suite needs updating.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)
        assert len(calls) == 2, (
            f"Expected 2 subprocess calls in cpu.py, found {len(calls)}. "
            f"Lines: {[c.lineno for c in calls]}"
        )

    def test_all_subprocess_calls_use_list_form(self):
        """
        Comprehensive check: every subprocess call in cpu.py uses a list
        as the first argument (not a string, not a variable that could be
        a string).
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)
        assert calls, "Expected at least one subprocess call in cpu.py"

        for c in calls:
            assert _first_arg_is_list(c), (
                f"subprocess call at line {c.lineno} does not use a list literal "
                f"as the first argument. This may be a shell-injection risk."
            )


# ---------------------------------------------------------------------------
# AST-based LogicalCPUInfo structure tests
# ---------------------------------------------------------------------------


class TestLogicalCPUInfoStructure:
    """
    Verify the LogicalCPUInfo dataclass structure via AST analysis.
    This avoids importing cpu.py (which requires Python 3.10+ and heavy deps).
    """

    def _find_logical_cpu_info_class(self) -> ast.ClassDef:
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                return node
        pytest.fail("LogicalCPUInfo class not found in cpu.py")

    def test_logical_cpu_info_class_exists(self):
        """LogicalCPUInfo class must exist in cpu.py."""
        cls = self._find_logical_cpu_info_class()
        assert cls is not None
        assert cls.name == "LogicalCPUInfo"

    def test_logical_cpu_info_is_dataclass(self):
        """LogicalCPUInfo must be decorated with @dataclass."""
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                decorator_names = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorator_names.append(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorator_names.append(dec.attr)
                assert "dataclass" in decorator_names, (
                    f"LogicalCPUInfo is not decorated with @dataclass. "
                    f"Found decorators: {decorator_names}"
                )
                return
        pytest.fail("LogicalCPUInfo class not found")

    def test_logical_cpu_info_has_required_fields(self):
        """LogicalCPUInfo must have id, physical_core, and numa_node fields."""
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                field_names = set()
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name):
                            field_names.add(item.target.id)
                required = {"id", "physical_core", "numa_node"}
                assert required.issubset(field_names), (
                    f"LogicalCPUInfo missing fields: {required - field_names}"
                )
                return
        pytest.fail("LogicalCPUInfo class not found")

    def test_logical_cpu_info_has_json_decoder(self):
        """LogicalCPUInfo must have a json_decoder static method."""
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                method_names = {
                    item.name
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                }
                assert "json_decoder" in method_names, (
                    f"LogicalCPUInfo missing json_decoder method. "
                    f"Found: {method_names}"
                )
                return
        pytest.fail("LogicalCPUInfo class not found")

    def test_logical_cpu_info_has_int_helper(self):
        """LogicalCPUInfo must have a _int class method."""
        source = _load_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                method_names = {
                    item.name
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                }
                assert "_int" in method_names, (
                    f"LogicalCPUInfo missing _int method. Found: {method_names}"
                )
                return
        pytest.fail("LogicalCPUInfo class not found")


# ---------------------------------------------------------------------------
# Behavioural tests using subprocess mock (no module import needed)
# ---------------------------------------------------------------------------


class TestSubprocessListFormBehaviour:
    """
    Verify that the subprocess calls in cpu.py use list form by inspecting
    the AST and confirming the exact argument structure at the call sites.
    These tests complement the static analysis with precise argument verification.
    """

    def test_sysctl_args_are_safe_list(self):
        """
        The sysctl call must use exactly ['sysctl', '-n', 'hw.optional.arm.FEAT_BF16'].
        A string form like 'sysctl -n hw.optional.arm.FEAT_BF16' would be injectable.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        sysctl_calls = [
            c for c in calls
            if _extract_list_elements(c) and _extract_list_elements(c)[0] == "sysctl"
        ]
        assert len(sysctl_calls) == 1

        args = _extract_list_elements(sysctl_calls[0])
        # Verify no shell metacharacters could be injected via the list form
        assert " " not in args[0], "Command element contains space (shell-injection risk)"
        assert len(args) == 3, f"Expected 3 args, got {len(args)}: {args}"
        assert args[0] == "sysctl"
        assert args[1] == "-n"
        assert args[2] == "hw.optional.arm.FEAT_BF16"

    def test_lscpu_args_are_safe_list(self):
        """
        The lscpu call must use exactly ['lscpu', '-J', '-e=CPU,CORE,NODE'].
        A string form like 'lscpu -J -e=CPU,CORE,NODE' would be injectable.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        lscpu_calls = [
            c for c in calls
            if _extract_list_elements(c) and _extract_list_elements(c)[0] == "lscpu"
        ]
        assert len(lscpu_calls) == 1

        args = _extract_list_elements(lscpu_calls[0])
        assert " " not in args[0], "Command element contains space (shell-injection risk)"
        assert len(args) == 3, f"Expected 3 args, got {len(args)}: {args}"
        assert args[0] == "lscpu"
        assert args[1] == "-J"
        assert args[2] == "-e=CPU,CORE,NODE"

    def test_no_subprocess_call_uses_string_concatenation(self):
        """
        No subprocess call should use string concatenation (f-string, +, %)
        as the first argument — that would be a shell-injection vector.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        for c in calls:
            if not c.args:
                continue
            first_arg = c.args[0]
            # Check for f-string (JoinedStr)
            assert not isinstance(first_arg, ast.JoinedStr), (
                f"subprocess call at line {c.lineno} uses an f-string as the "
                f"first argument — potential shell-injection risk"
            )
            # Check for string concatenation (BinOp with Add)
            assert not (
                isinstance(first_arg, ast.BinOp)
                and isinstance(first_arg.op, ast.Add)
            ), (
                f"subprocess call at line {c.lineno} uses string concatenation "
                f"as the first argument — potential shell-injection risk"
            )

    def test_lscpu_call_has_text_kwarg(self):
        """
        The lscpu call should include text=True to get string output
        (not bytes), which is the correct form after the fix.
        """
        source = _load_source()
        calls = _get_subprocess_calls(source)

        lscpu_calls = [
            c for c in calls
            if _extract_list_elements(c) and _extract_list_elements(c)[0] == "lscpu"
        ]
        assert len(lscpu_calls) == 1
        c = lscpu_calls[0]

        kwarg_names = {kw.arg for kw in c.keywords}
        assert "text" in kwarg_names, (
            f"lscpu call at line {c.lineno} is missing text=True keyword argument"
        )

        # Verify text=True (not text=False)
        for kw in c.keywords:
            if kw.arg == "text":
                assert isinstance(kw.value, ast.Constant) and kw.value.value is True, (
                    f"lscpu call has text= but it is not True: {ast.dump(kw.value)}"
                )


# ---------------------------------------------------------------------------
# Integration: LogicalCPUInfo json_decoder logic (pure Python, no imports)
# ---------------------------------------------------------------------------


class TestLogicalCPUInfoLogic:
    """
    Test the LogicalCPUInfo json_decoder and _int logic by extracting and
    executing just the dataclass definition from cpu.py via exec().
    This avoids the Python 3.10+ union syntax issue in the CpuPlatform class.
    """

    @pytest.fixture(scope="class")
    def logical_cpu_info_cls(self):
        """
        Extract and exec just the LogicalCPUInfo dataclass from cpu.py.
        This is safe because we only exec the dataclass portion, not the
        full module with its Python 3.10+ type annotations.
        """
        source = _load_source()
        tree = ast.parse(source)

        # Find the LogicalCPUInfo class node
        cls_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "LogicalCPUInfo":
                cls_node = node
                break

        if cls_node is None:
            pytest.fail("LogicalCPUInfo not found in cpu.py")

        # Find the @dataclass decorator import and the class definition
        # We need to reconstruct a minimal executable snippet
        lines = source.splitlines()
        # Get the line range of the class (include decorators)
        if cls_node.decorator_list:
            start_line = cls_node.decorator_list[0].lineno - 1  # 0-indexed
        else:
            start_line = cls_node.lineno - 1  # 0-indexed

        # Find end of class (next top-level definition or end of file)
        end_line = len(lines)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name != "LogicalCPUInfo"
                and node.lineno > cls_node.lineno
            ):
                # Check it's at top level (col_offset == 0)
                if node.col_offset == 0:
                    end_line = min(end_line, node.lineno - 1)

        class_source = "\n".join(lines[start_line:end_line])

        # Build a minimal executable snippet
        snippet = textwrap.dedent(f"""
from dataclasses import dataclass

{class_source}
""")
        namespace = {}
        exec(compile(snippet, "<test_snippet>", "exec"), namespace)
        return namespace["LogicalCPUInfo"]

    def test_defaults(self, logical_cpu_info_cls):
        """Default LogicalCPUInfo has id=-1, physical_core=-1, numa_node=-1."""
        info = logical_cpu_info_cls()
        assert info.id == -1
        assert info.physical_core == -1
        assert info.numa_node == -1

    def test_int_valid_values(self, logical_cpu_info_cls):
        """_int() converts valid string integers correctly."""
        assert logical_cpu_info_cls._int("0") == 0
        assert logical_cpu_info_cls._int("42") == 42
        assert logical_cpu_info_cls._int("255") == 255

    def test_int_invalid_values(self, logical_cpu_info_cls):
        """_int() returns -1 for non-integer strings."""
        assert logical_cpu_info_cls._int("") == -1
        assert logical_cpu_info_cls._int("abc") == -1
        assert logical_cpu_info_cls._int("1.5") == -1
        assert logical_cpu_info_cls._int(None) == -1

    def test_json_decoder_valid_dict(self, logical_cpu_info_cls):
        """json_decoder creates a LogicalCPUInfo from a valid dict."""
        obj = {"cpu": "2", "core": "1", "node": "0"}
        result = logical_cpu_info_cls.json_decoder(obj)
        assert isinstance(result, logical_cpu_info_cls)
        assert result.id == 2
        assert result.physical_core == 1
        assert result.numa_node == 0

    def test_json_decoder_missing_key_returns_dict(self, logical_cpu_info_cls):
        """json_decoder returns the original dict when a key is missing."""
        obj = {"cpu": "0", "core": "0"}  # Missing 'node'
        result = logical_cpu_info_cls.json_decoder(obj)
        assert result is obj  # unchanged dict returned

    def test_json_decoder_invalid_int_values(self, logical_cpu_info_cls):
        """json_decoder handles non-integer string values gracefully."""
        obj = {"cpu": "bad", "core": "0", "node": "0"}
        result = logical_cpu_info_cls.json_decoder(obj)
        assert isinstance(result, logical_cpu_info_cls)
        assert result.id == -1  # "bad" → -1
        assert result.physical_core == 0
        assert result.numa_node == 0

    def test_json_decoder_all_none_returns_dict(self, logical_cpu_info_cls):
        """json_decoder returns the original dict when all keys are None."""
        obj = {}
        result = logical_cpu_info_cls.json_decoder(obj)
        assert result is obj

    def test_json_decoder_roundtrip(self, logical_cpu_info_cls):
        """
        Full roundtrip: JSON string → json.loads with object_hook → LogicalCPUInfo.
        This mirrors how the lscpu output is parsed in get_allowed_cpu_core_node_list.
        """
        lscpu_json = json.dumps({
            "cpus": [
                {"cpu": "0", "core": "0", "node": "0"},
                {"cpu": "1", "core": "0", "node": "0"},
                {"cpu": "2", "core": "1", "node": "1"},
            ]
        })
        result = json.loads(
            lscpu_json,
            object_hook=logical_cpu_info_cls.json_decoder,
        )
        # The top-level dict is returned as-is (no cpu/core/node keys)
        assert "cpus" in result
        cpus = result["cpus"]
        assert len(cpus) == 3
        assert all(isinstance(c, logical_cpu_info_cls) for c in cpus)
        assert cpus[0].id == 0
        assert cpus[0].physical_core == 0
        assert cpus[0].numa_node == 0
        assert cpus[2].id == 2
        assert cpus[2].physical_core == 1
        assert cpus[2].numa_node == 1
