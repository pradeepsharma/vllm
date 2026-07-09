#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive test runner for security tests.
This script tests security features without requiring full vllm imports.
"""

import sys
import traceback
from pathlib import Path

# Add the repo root to the path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))


def run_test(test_class, test_name):
    """Run a single test and return True if it passes."""
    try:
        test_instance = test_class()
        getattr(test_instance, test_name)()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main():
    """Run all security tests."""
    from tests.security.test_subprocess_safety import (
        TestSubprocessSafety,
        TestLogicalCPUInfoStructure,
    )

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    # Test subprocess safety
    print("\n" + "=" * 70)
    print("PHASE 1: Subprocess Safety Tests")
    print("=" * 70)

    subprocess_tests = [
        "test_cpu_py_exists",
        "test_no_shell_true_in_vllm_python_files",
        "test_cpu_py_syntax_valid",
        "test_subprocess_calls_use_list_not_string",
        "test_subprocess_calls_no_shell_true",
        "test_sysctl_call_is_list_form",
        "test_lscpu_call_is_list_form",
        "test_total_subprocess_calls_count",
        "test_all_subprocess_calls_use_list_form",
    ]

    for test_name in subprocess_tests:
        total_tests += 1
        print(f"\n{test_name}...", end=" ")
        if run_test(TestSubprocessSafety, test_name):
            print("✓ PASSED")
            passed_tests += 1
        else:
            print("✗ FAILED")
            failed_tests.append(f"TestSubprocessSafety::{test_name}")

    # Test LogicalCPUInfo structure
    print("\n" + "=" * 70)
    print("PHASE 1: LogicalCPUInfo Structure Tests")
    print("=" * 70)

    logical_cpu_tests = [
        "test_logical_cpu_info_class_exists",
        "test_logical_cpu_info_is_dataclass",
        "test_logical_cpu_info_has_required_fields",
        "test_logical_cpu_info_has_json_decoder",
        "test_logical_cpu_info_has_int_helper",
    ]

    for test_name in logical_cpu_tests:
        total_tests += 1
        print(f"\n{test_name}...", end=" ")
        if run_test(TestLogicalCPUInfoStructure, test_name):
            print("✓ PASSED")
            passed_tests += 1
        else:
            print("✗ FAILED")
            failed_tests.append(f"TestLogicalCPUInfoStructure::{test_name}")

    # Test HTTP client validation
    print("\n" + "=" * 70)
    print("PHASE 4: HTTP Client Security Tests")
    print("=" * 70)
    print("\nNote: HTTP client tests require importing vllm modules.")
    print("These tests are skipped due to Python 3.10+ syntax requirements.")
    print("The tests are available in tests/security/test_http_client.py")

    # Test CORS defaults
    print("\n" + "=" * 70)
    print("PHASE 3: CORS Defaults Tests")
    print("=" * 70)
    print("\nNote: CORS tests require importing vllm modules.")
    print("These tests are skipped due to Python 3.10+ syntax requirements.")
    print("The tests are available in tests/security/test_cors_defaults.py")

    # Test authentication middleware
    print("\n" + "=" * 70)
    print("PHASE 2: Authentication Middleware Tests")
    print("=" * 70)
    print("\nNote: Auth middleware tests require importing vllm modules.")
    print("These tests are skipped due to Python 3.10+ syntax requirements.")
    print("The tests are available in tests/security/test_auth_middleware.py")

    # Test serialization guards
    print("\n" + "=" * 70)
    print("PHASE 5: Serialization Guard Tests")
    print("=" * 70)
    print("\nNote: Serialization tests require importing vllm modules.")
    print("These tests are skipped due to Python 3.10+ syntax requirements.")
    print("The tests are available in tests/security/test_serialization_guard.py")

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        print("\nNote: Some test files require Python 3.10+ to run.")
        print("The following test files are available but require Python 3.10+:")
        print("  - tests/security/test_auth_middleware.py")
        print("  - tests/security/test_cors_defaults.py")
        print("  - tests/security/test_serialization_guard.py")
        print("  - tests/security/test_http_client.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
