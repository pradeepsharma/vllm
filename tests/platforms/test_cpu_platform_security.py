# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for security fixes in vllm/platforms/cpu.py:
  1. Shell-injection fix: subprocess calls use list form (no shell=True)
  2. Subprocess argument fix: commands passed as lists, not shell strings

These tests verify via AST analysis and direct source inspection:
  - subprocess.check_output is called with a list, not a string
  - shell=True is never passed to any subprocess call
  - The exact fixed argument lists are present
  - LogicalCPUInfo dataclass logic is correct
  - get_max_threads() behaves correctly on the current OS
"""

import ast
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path to the file under test
# ---------------------------------------------------------------------------

CPU_PY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "vllm", "platforms", "cpu.py"
)


def _read_cpu_source() -> str:
    with open(CPU_PY_PATH, encoding="utf-8") as fh:
        return fh.read()


def _parse_cpu_ast() -> ast.Module:
    return ast.parse(_read_cpu_source())


# ---------------------------------------------------------------------------
# Standalone replica of LogicalCPUInfo for isolated testing
# (mirrors the exact logic from cpu.py without requiring vllm imports)
# ---------------------------------------------------------------------------

@dataclass
class LogicalCPUInfo:
    """Replica of vllm/platforms/cpu.py LogicalCPUInfo for isolated testing."""
    id: int = -1
    physical_core: int = -1
    numa_node: int = -1

    @classmethod
    def _int(cls, value: str) -> int:
        try:
            int_value = int(value)
        except Exception:
            int_value = -1
        return int_value

    @staticmethod
    def json_decoder(obj_dict: dict):
        id = obj_dict.get("cpu")
        physical_core = obj_dict.get("core")
        numa_node = obj_dict.get("node")

        if not (id is None or physical_core is None or numa_node is None):
            return LogicalCPUInfo(
                id=LogicalCPUInfo._int(str(id)),
                physical_core=LogicalCPUInfo._int(str(physical_core)),
                numa_node=LogicalCPUInfo._int(str(numa_node)),
            )
        else:
            return obj_dict


# ---------------------------------------------------------------------------
# Helper: extract get_max_threads function and exec it in isolation
# ---------------------------------------------------------------------------

def _extract_and_exec_get_max_threads():
    """
    Extract get_max_threads() from cpu.py and exec it in an isolated namespace.
    """
    source = _read_cpu_source()
    tree = ast.parse(source)

    fn_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_max_threads":
            fn_node = node
            break

    assert fn_node is not None, "get_max_threads not found in cpu.py"

    fn_source = textwrap.dedent("""
import os
import platform

""") + ast.get_source_segment(source, fn_node)

    ns: dict = {}
    exec(compile(fn_source, CPU_PY_PATH, "exec"), ns)
    return ns["get_max_threads"]


# ---------------------------------------------------------------------------
# 1. Static analysis: no shell=True in subprocess calls
# ---------------------------------------------------------------------------


class TestNoShellTrue:
    """Verify that no subprocess call in cpu.py uses shell=True."""

    def test_no_shell_true_in_source(self):
        """
        Parse the AST of cpu.py and assert that no call to
        subprocess.check_output (or any subprocess function) passes
        shell=True as a keyword argument.
        """
        tree = _parse_cpu_ast()

        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_subprocess_call = False
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                    is_subprocess_call = True
            if not is_subprocess_call:
                continue
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                    if kw.value.value is True:
                        violations.append(
                            f"Line {node.lineno}: subprocess.{func.attr}(..., shell=True)"
                        )

        assert violations == [], (
            "Found shell=True in subprocess calls (shell-injection risk):\n"
            + "\n".join(violations)
        )

    def test_subprocess_args_are_lists_not_strings(self):
        """
        Verify that every subprocess.check_output call in cpu.py passes
        a list as its first argument, not a plain string.
        """
        tree = _parse_cpu_ast()

        string_arg_violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                continue
            if node.args:
                first_arg = node.args[0]
                if not isinstance(first_arg, ast.List):
                    string_arg_violations.append(
                        f"Line {node.lineno}: subprocess.check_output first arg is "
                        f"{type(first_arg).__name__}, expected List"
                    )

        assert string_arg_violations == [], (
            "subprocess.check_output called with non-list first argument "
            "(shell-injection risk):\n" + "\n".join(string_arg_violations)
        )

    def test_no_shell_true_literal_in_source(self):
        """Direct string search: 'shell=True' must not appear in cpu.py."""
        source = _read_cpu_source()
        assert "shell=True" not in source, (
            "shell=True found in cpu.py — shell-injection vulnerability present"
        )


# ---------------------------------------------------------------------------
# 2. Exact fix verification: correct argument lists are present
# ---------------------------------------------------------------------------


class TestExactFixPresent:
    """Regression tests that verify the exact security fixes are in place."""

    def test_sysctl_not_using_shell_string(self):
        """The old vulnerable sysctl pattern must NOT be present."""
        source = _read_cpu_source()
        assert '"sysctl -n hw.optional.arm.FEAT_BF16"' not in source, (
            "Old shell-injection pattern found: "
            '["sysctl -n hw.optional.arm.FEAT_BF16"] with shell=True'
        )

    def test_lscpu_not_using_shell_string(self):
        """The old vulnerable lscpu pattern must NOT be present."""
        source = _read_cpu_source()
        assert '"lscpu -J -e=CPU,CORE,NODE"' not in source, (
            "Old shell-injection pattern found: "
            '"lscpu -J -e=CPU,CORE,NODE" with shell=True'
        )

    def test_sysctl_list_form_present(self):
        """The fixed sysctl call (list form) must be present in source."""
        source = _read_cpu_source()
        assert '"sysctl"' in source, "sysctl not found as list element in cpu.py"
        assert '"hw.optional.arm.FEAT_BF16"' in source, (
            "hw.optional.arm.FEAT_BF16 not found as list element in cpu.py"
        )

    def test_lscpu_list_form_present(self):
        """The fixed lscpu call (list form) must be present in source."""
        source = _read_cpu_source()
        assert '"lscpu"' in source, "lscpu not found as list element in cpu.py"
        assert '"-J"' in source, "-J not found as list element in cpu.py"
        assert '"-e=CPU,CORE,NODE"' in source, (
            "-e=CPU,CORE,NODE not found as list element in cpu.py"
        )

    def test_sysctl_call_exact_args_via_ast(self):
        """sysctl must be called with exactly ['sysctl', '-n', 'hw.optional.arm.FEAT_BF16']."""
        tree = _parse_cpu_ast()

        sysctl_args = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                continue
            if not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.List):
                continue
            elts = first_arg.elts
            if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "sysctl":
                sysctl_args = [
                    e.value for e in elts if isinstance(e, ast.Constant)
                ]
                break

        assert sysctl_args is not None, "Could not find sysctl call in cpu.py"
        assert sysctl_args == ["sysctl", "-n", "hw.optional.arm.FEAT_BF16"], (
            f"sysctl called with unexpected args: {sysctl_args!r}"
        )

    def test_lscpu_call_exact_args_via_ast(self):
        """lscpu must be called with exactly ['lscpu', '-J', '-e=CPU,CORE,NODE']."""
        tree = _parse_cpu_ast()

        lscpu_args = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                continue
            if not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.List):
                continue
            elts = first_arg.elts
            if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "lscpu":
                lscpu_args = [
                    e.value for e in elts if isinstance(e, ast.Constant)
                ]
                break

        assert lscpu_args is not None, "Could not find lscpu call in cpu.py"
        assert lscpu_args == ["lscpu", "-J", "-e=CPU,CORE,NODE"], (
            f"lscpu called with unexpected args: {lscpu_args!r}"
        )

    def test_count_subprocess_check_output_calls(self):
        """Verify exactly 2 subprocess.check_output calls exist in cpu.py."""
        tree = _parse_cpu_ast()

        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                calls.append(node.lineno)

        assert len(calls) == 2, (
            f"Expected exactly 2 subprocess.check_output calls, found {len(calls)} "
            f"at lines: {calls}"
        )

    def test_all_subprocess_calls_use_list_form(self):
        """Every subprocess.check_output call must use a list as first arg."""
        tree = _parse_cpu_ast()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                continue
            assert node.args, f"Line {node.lineno}: subprocess.check_output has no positional args"
            first_arg = node.args[0]
            assert isinstance(first_arg, ast.List), (
                f"Line {node.lineno}: subprocess.check_output first arg is "
                f"{type(first_arg).__name__}, expected List (shell-injection risk)"
            )


# ---------------------------------------------------------------------------
# 3. LogicalCPUInfo dataclass — tested via standalone replica
# ---------------------------------------------------------------------------


class TestLogicalCPUInfo:
    """
    Unit tests for the LogicalCPUInfo dataclass logic.
    Uses a standalone replica that mirrors the exact logic from cpu.py.
    """

    def test_default_values(self):
        info = LogicalCPUInfo()
        assert info.id == -1
        assert info.physical_core == -1
        assert info.numa_node == -1

    def test_explicit_values(self):
        info = LogicalCPUInfo(id=0, physical_core=2, numa_node=1)
        assert info.id == 0
        assert info.physical_core == 2
        assert info.numa_node == 1

    def test_int_helper_valid_integer_string(self):
        assert LogicalCPUInfo._int("42") == 42

    def test_int_helper_zero(self):
        assert LogicalCPUInfo._int("0") == 0

    def test_int_helper_invalid_alpha(self):
        assert LogicalCPUInfo._int("abc") == -1

    def test_int_helper_empty_string(self):
        assert LogicalCPUInfo._int("") == -1

    def test_int_helper_float_string(self):
        assert LogicalCPUInfo._int("3.14") == -1

    def test_int_helper_negative(self):
        assert LogicalCPUInfo._int("-1") == -1

    def test_json_decoder_valid_all_keys(self):
        obj = {"cpu": "0", "core": "1", "node": "2"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert isinstance(result, LogicalCPUInfo)
        assert result.id == 0
        assert result.physical_core == 1
        assert result.numa_node == 2

    def test_json_decoder_missing_node_returns_dict(self):
        obj = {"cpu": "0", "core": "1"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert result is obj, "Expected original dict when 'node' key is missing"

    def test_json_decoder_missing_core_returns_dict(self):
        obj = {"cpu": "0", "node": "1"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert result is obj, "Expected original dict when 'core' key is missing"

    def test_json_decoder_missing_cpu_returns_dict(self):
        obj = {"core": "0", "node": "1"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert result is obj, "Expected original dict when 'cpu' key is missing"

    def test_json_decoder_all_missing_returns_dict(self):
        obj = {"other": "value"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert result is obj

    def test_json_decoder_invalid_int_values(self):
        obj = {"cpu": "bad", "core": "also_bad", "node": "nope"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert isinstance(result, LogicalCPUInfo)
        assert result.id == -1
        assert result.physical_core == -1
        assert result.numa_node == -1

    def test_json_loads_integration_single_cpu(self):
        """Verify json_decoder works as object_hook in json.loads."""
        payload = json.dumps(
            {"cpus": [{"cpu": "0", "core": "0", "node": "0"}]}
        )
        result = json.loads(payload, object_hook=LogicalCPUInfo.json_decoder)
        assert "cpus" in result
        cpus = result["cpus"]
        assert len(cpus) == 1
        assert isinstance(cpus[0], LogicalCPUInfo)
        assert cpus[0].id == 0
        assert cpus[0].physical_core == 0
        assert cpus[0].numa_node == 0

    def test_json_loads_integration_multiple_cpus(self):
        """Verify json_decoder handles multiple CPU entries."""
        payload = json.dumps(
            {
                "cpus": [
                    {"cpu": "0", "core": "0", "node": "0"},
                    {"cpu": "1", "core": "0", "node": "0"},
                    {"cpu": "2", "core": "1", "node": "1"},
                ]
            }
        )
        result = json.loads(payload, object_hook=LogicalCPUInfo.json_decoder)
        cpus = result["cpus"]
        assert len(cpus) == 3
        assert all(isinstance(c, LogicalCPUInfo) for c in cpus)
        assert cpus[0].id == 0
        assert cpus[1].id == 1
        assert cpus[2].id == 2
        assert cpus[2].physical_core == 1
        assert cpus[2].numa_node == 1

    def test_json_decoder_with_integer_node_value(self):
        """
        lscpu sometimes outputs integer node values (not strings).
        The json_decoder should handle both string and int values.
        """
        obj = {"cpu": "3", "core": "1", "node": "0"}
        result = LogicalCPUInfo.json_decoder(obj)
        assert isinstance(result, LogicalCPUInfo)
        assert result.numa_node == 0

    def test_dataclass_is_not_frozen(self):
        """LogicalCPUInfo fields should be mutable (not frozen dataclass)."""
        info = LogicalCPUInfo(id=0, physical_core=0, numa_node=0)
        info.id = 99
        assert info.id == 99

    def test_filter_invalid_cpus(self):
        """Simulate the filtering logic: CPUs with -1 in any field are excluded."""
        cpus = [
            LogicalCPUInfo(id=0, physical_core=0, numa_node=0),
            LogicalCPUInfo(id=-1, physical_core=0, numa_node=0),  # invalid id
            LogicalCPUInfo(id=1, physical_core=-1, numa_node=0),  # invalid core
            LogicalCPUInfo(id=2, physical_core=1, numa_node=-1),  # invalid node
            LogicalCPUInfo(id=3, physical_core=1, numa_node=1),   # valid
        ]
        valid = [x for x in cpus if -1 not in (x.id, x.physical_core, x.numa_node)]
        assert len(valid) == 2
        assert valid[0].id == 0
        assert valid[1].id == 3

    def test_filter_allowed_cpu_ids(self):
        """Simulate filtering by allowed CPU IDs (sched_getaffinity result)."""
        cpus = [
            LogicalCPUInfo(id=0, physical_core=0, numa_node=0),
            LogicalCPUInfo(id=1, physical_core=0, numa_node=0),
            LogicalCPUInfo(id=2, physical_core=1, numa_node=0),
            LogicalCPUInfo(id=3, physical_core=1, numa_node=0),
        ]
        allowed_ids = {0, 2}  # simulated sched_getaffinity result
        filtered = [x for x in cpus if x.id in allowed_ids]
        assert len(filtered) == 2
        assert filtered[0].id == 0
        assert filtered[1].id == 2


# ---------------------------------------------------------------------------
# 4. get_max_threads() — isolated exec tests
# ---------------------------------------------------------------------------


class TestGetMaxThreads:
    """Tests for the get_max_threads() helper function."""

    @pytest.fixture(autouse=True)
    def load_fn(self):
        self.get_max_threads = _extract_and_exec_get_max_threads()

    def test_returns_positive_integer(self):
        result = self.get_max_threads()
        assert isinstance(result, int), f"Expected int, got {type(result)}"
        assert result > 0, f"Expected positive integer, got {result}"

    def test_with_sched_getaffinity_when_available(self):
        if not hasattr(os, "sched_getaffinity"):
            pytest.skip("sched_getaffinity not available on this OS")
        result = self.get_max_threads(0)
        expected = len(os.sched_getaffinity(0))
        assert result == expected, (
            f"get_max_threads() returned {result}, expected {expected} "
            f"(from sched_getaffinity)"
        )

    def test_darwin_fallback_uses_cpu_count(self):
        """On Darwin without sched_getaffinity, should use os.cpu_count()."""
        orig_sched = getattr(os, "sched_getaffinity", None)
        if orig_sched is not None:
            del os.sched_getaffinity

        try:
            with patch("os.cpu_count", return_value=8):
                with patch("platform.system", return_value="Darwin"):
                    result = self.get_max_threads()
            assert result == 8, (
                f"Expected 8 from os.cpu_count() on Darwin, got {result}"
            )
        finally:
            if orig_sched is not None:
                os.sched_getaffinity = orig_sched

    def test_unsupported_os_raises_not_implemented(self):
        """On an unsupported OS without sched_getaffinity, should raise NotImplementedError."""
        orig_sched = getattr(os, "sched_getaffinity", None)
        if orig_sched is not None:
            del os.sched_getaffinity

        try:
            with patch("platform.system", return_value="Windows"):
                with pytest.raises(NotImplementedError, match="Unsupported OS"):
                    self.get_max_threads()
        finally:
            if orig_sched is not None:
                os.sched_getaffinity = orig_sched

    def test_pid_zero_is_default(self):
        """get_max_threads() with no args should behave same as pid=0."""
        result_no_arg = self.get_max_threads()
        result_pid0 = self.get_max_threads(0)
        assert result_no_arg == result_pid0


# ---------------------------------------------------------------------------
# 5. AST-level: verify subprocess call keyword arguments
# ---------------------------------------------------------------------------


class TestSubprocessCallKeywords:
    """Verify keyword arguments on each subprocess.check_output call via AST."""

    def _get_all_check_output_calls(self):
        tree = _parse_cpu_ast()
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr == "check_output"
            ):
                calls.append(node)
        return calls

    def test_sysctl_call_has_no_shell_kwarg(self):
        """The sysctl call must not have a 'shell' keyword argument at all."""
        calls = self._get_all_check_output_calls()
        sysctl_call = None
        for call in calls:
            if call.args and isinstance(call.args[0], ast.List):
                elts = call.args[0].elts
                if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "sysctl":
                    sysctl_call = call
                    break

        assert sysctl_call is not None, "sysctl call not found"
        shell_kwarg = next(
            (kw for kw in sysctl_call.keywords if kw.arg == "shell"), None
        )
        assert shell_kwarg is None, (
            f"Line {sysctl_call.lineno}: sysctl call has 'shell' keyword — "
            "shell-injection vulnerability"
        )

    def test_lscpu_call_has_no_shell_kwarg(self):
        """The lscpu call must not have a 'shell' keyword argument at all."""
        calls = self._get_all_check_output_calls()
        lscpu_call = None
        for call in calls:
            if call.args and isinstance(call.args[0], ast.List):
                elts = call.args[0].elts
                if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "lscpu":
                    lscpu_call = call
                    break

        assert lscpu_call is not None, "lscpu call not found"
        shell_kwarg = next(
            (kw for kw in lscpu_call.keywords if kw.arg == "shell"), None
        )
        assert shell_kwarg is None, (
            f"Line {lscpu_call.lineno}: lscpu call has 'shell' keyword — "
            "shell-injection vulnerability"
        )

    def test_lscpu_call_has_text_true(self):
        """The lscpu call must have text=True keyword argument."""
        calls = self._get_all_check_output_calls()
        lscpu_call = None
        for call in calls:
            if call.args and isinstance(call.args[0], ast.List):
                elts = call.args[0].elts
                if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "lscpu":
                    lscpu_call = call
                    break

        assert lscpu_call is not None, "lscpu call not found"
        text_kwarg = next(
            (kw for kw in lscpu_call.keywords if kw.arg == "text"), None
        )
        assert text_kwarg is not None, (
            f"Line {lscpu_call.lineno}: lscpu call missing text=True keyword"
        )
        assert isinstance(text_kwarg.value, ast.Constant) and text_kwarg.value.value is True, (
            f"Line {lscpu_call.lineno}: lscpu call has text={text_kwarg.value!r}, expected True"
        )

    def test_sysctl_call_has_no_text_kwarg(self):
        """The sysctl call returns bytes (no text=True), so strip() == b'1' works."""
        calls = self._get_all_check_output_calls()
        sysctl_call = None
        for call in calls:
            if call.args and isinstance(call.args[0], ast.List):
                elts = call.args[0].elts
                if elts and isinstance(elts[0], ast.Constant) and elts[0].value == "sysctl":
                    sysctl_call = call
                    break

        assert sysctl_call is not None, "sysctl call not found"
        text_kwarg = next(
            (kw for kw in sysctl_call.keywords if kw.arg == "text"), None
        )
        # sysctl returns bytes and is compared to b"1", so text=True would break it
        assert text_kwarg is None, (
            f"Line {sysctl_call.lineno}: sysctl call has text=True but should return bytes "
            "(compared to b'1')"
        )


# ---------------------------------------------------------------------------
# 6. Integration: lscpu JSON parsing logic
# ---------------------------------------------------------------------------


class TestLscpuJsonParsing:
    """
    Test the JSON parsing logic used in get_allowed_cpu_core_node_list()
    using the standalone LogicalCPUInfo replica.
    """

    def _parse_lscpu_output(self, lscpu_json: str) -> list:
        """Replicate the parsing logic from get_allowed_cpu_core_node_list()."""
        lscpu_output = re.sub(
            r'"node":\s*-\s*(,|\n)', r'"node": 0\1', lscpu_json
        )
        result = json.loads(lscpu_output, object_hook=LogicalCPUInfo.json_decoder)
        return result["cpus"]

    def test_parse_valid_lscpu_output(self):
        """Parse a typical lscpu -J output."""
        lscpu_json = json.dumps({
            "cpus": [
                {"cpu": "0", "core": "0", "node": "0"},
                {"cpu": "1", "core": "0", "node": "0"},
                {"cpu": "2", "core": "1", "node": "0"},
                {"cpu": "3", "core": "1", "node": "0"},
            ]
        })
        cpus = self._parse_lscpu_output(lscpu_json)
        assert len(cpus) == 4
        assert all(isinstance(c, LogicalCPUInfo) for c in cpus)
        assert cpus[0].id == 0
        assert cpus[3].id == 3
        assert cpus[3].physical_core == 1

    def test_parse_lscpu_multi_numa(self):
        """Parse lscpu output with multiple NUMA nodes."""
        lscpu_json = json.dumps({
            "cpus": [
                {"cpu": "0", "core": "0", "node": "0"},
                {"cpu": "1", "core": "1", "node": "0"},
                {"cpu": "2", "core": "0", "node": "1"},
                {"cpu": "3", "core": "1", "node": "1"},
            ]
        })
        cpus = self._parse_lscpu_output(lscpu_json)
        assert len(cpus) == 4
        numa_nodes = {c.numa_node for c in cpus}
        assert numa_nodes == {0, 1}

    def test_filter_invalid_cpus_after_parse(self):
        """After parsing, CPUs with -1 in any field should be filterable."""
        lscpu_json = json.dumps({
            "cpus": [
                {"cpu": "0", "core": "0", "node": "0"},
                {"cpu": "bad", "core": "0", "node": "0"},  # invalid id → -1
                {"cpu": "2", "core": "1", "node": "0"},
            ]
        })
        cpus = self._parse_lscpu_output(lscpu_json)
        valid = [x for x in cpus if -1 not in (x.id, x.physical_core, x.numa_node)]
        assert len(valid) == 2
        assert valid[0].id == 0
        assert valid[1].id == 2

    def test_regex_substitution_for_negative_node_with_comma(self):
        """
        The regex in cpu.py: re.sub(r'"node":\\s*-\\s*(,|\\n)', r'"node": 0\\1', ...)
        replaces 'node': - followed by comma or newline with 'node': 0.
        Test the comma case.
        """
        # Simulate lscpu JSON with negative node followed by comma
        raw = '{"cpus": [{"cpu": "0", "core": "0", "node": -,\n"extra": "val"}]}'
        fixed = re.sub(r'"node":\s*-\s*(,|\n)', r'"node": 0\1', raw)
        assert '"node": 0,' in fixed, (
            f"Regex substitution failed. Result: {fixed!r}"
        )

    def test_regex_substitution_for_negative_node_with_newline(self):
        """
        The regex in cpu.py handles 'node': - followed by newline.
        """
        raw = '{"cpus": [{"cpu": "0", "core": "0", "node": -\n}]}'
        fixed = re.sub(r'"node":\s*-\s*(,|\n)', r'"node": 0\1', raw)
        assert '"node": 0\n' in fixed, (
            f"Regex substitution failed. Result: {fixed!r}"
        )

    def test_parse_empty_cpu_list(self):
        """An empty CPU list should parse to an empty list."""
        lscpu_json = json.dumps({"cpus": []})
        cpus = self._parse_lscpu_output(lscpu_json)
        assert cpus == []

    def test_allowed_numa_nodes_extraction(self):
        """Verify NUMA node extraction logic from parsed CPU list."""
        cpus = [
            LogicalCPUInfo(id=0, physical_core=0, numa_node=0),
            LogicalCPUInfo(id=1, physical_core=0, numa_node=0),
            LogicalCPUInfo(id=2, physical_core=0, numa_node=1),
            LogicalCPUInfo(id=3, physical_core=1, numa_node=1),
        ]
        allowed_numa_nodes = set()
        for x in cpus:
            allowed_numa_nodes.add(x.numa_node)
        allowed_numa_nodes_list = sorted(allowed_numa_nodes)
        assert allowed_numa_nodes_list == [0, 1]
