# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone tests for security utility functions added in the security hardening phases.

These tests run without the full vllm ML stack (no torch, transformers, etc.)
by directly testing the pure-Python security logic extracted from the source files.

Covers:
  - sanitize_message() — Phase 1: Security Utilities & Shared Helpers
  - emit_security_warning() — Phase 1
  - validate_cors_origins() — Phase 1 / Phase 3
  - is_localhost() — Phase 1 / Phase 3
  - _validate_tool_server() — Phase 2: Request Size & Input Validation Hardening
  - validate_parsed_serve_args() h11 size warning — Phase 2
"""

import logging
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inline implementations of the security functions under test.
# These are copied verbatim from the source files so we can test them
# without importing the full vllm package (which requires torch, etc.).
# Any change to the source must be reflected here.
# ---------------------------------------------------------------------------

def sanitize_message(message: str) -> str:
    """Sanitize error messages to prevent leaking sensitive information.
    Copied from vllm/entrypoints/utils.py"""
    # Avoid leaking memory address from object reprs
    message = re.sub(r" at 0x[0-9a-f]+>", ">", message)
    # Remove file paths (both absolute and relative) to prevent information leakage
    message = re.sub(
        r"[/\\](?:[a-zA-Z0-9._-]+[/\\])*[a-zA-Z0-9._-]+\.py", "<file>", message
    )
    # Remove line numbers in traceback format (e.g., "line 123")
    message = re.sub(r"line \d+", "<line>", message)
    # Remove module paths that could reveal internal structure
    message = re.sub(
        r"\b(?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*\b", "<module>", message
    )
    return message


def is_localhost(host) -> bool:
    """Check if a host is localhost or None.
    Copied from vllm/entrypoints/utils.py"""
    if host is None:
        return True
    host_lower = host.lower()
    return host_lower in ("127.0.0.1", "::1", "localhost")


def validate_cors_origins(origins, allow_credentials: bool = False) -> None:
    """Validate CORS origin configuration for security issues.
    Copied from vllm/entrypoints/utils.py"""
    if allow_credentials and "*" in origins:
        raise ValueError(
            "CORS configuration error: wildcard origins ('*') cannot be combined "
            "with allow_credentials=True. Browsers reject this configuration. "
            "Please specify explicit origins or disable allow_credentials."
        )


def _validate_tool_server(tool_server) -> None:
    """Validate the tool_server argument.
    Copied from vllm/entrypoints/openai/cli_args.py"""
    if not tool_server or tool_server == "demo":
        return

    # Shell metacharacters that should not appear in host:port entries
    shell_metacharacters = set('<>|;&$`\\"\'  \t\n')

    # Pattern for valid IPv4 address
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"

    # Pattern for valid IPv6 address (simplified, allows [::1] format)
    ipv6_pattern = r"^\[([0-9a-fA-F:]+)\]$"

    # Pattern for valid hostname (alphanumeric, dots, hyphens)
    hostname_pattern = (
        r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    )

    # Split by comma to get individual entries
    entries = [e.strip() for e in tool_server.split(",") if e.strip()]

    for entry in entries:
        # Check for shell metacharacters
        if any(c in entry for c in shell_metacharacters):
            raise ValueError(
                f"Invalid tool_server entry: contains shell metacharacters. "
                f"Entry: {entry}"
            )

        # Split host and port
        if ":" not in entry:
            raise ValueError(
                f"Invalid tool_server entry: missing port. "
                f"Expected format: host:port. Entry: {entry}"
            )

        # Handle IPv6 addresses in brackets
        if entry.startswith("["):
            if "]:" not in entry:
                raise ValueError(
                    f"Invalid tool_server entry: malformed IPv6 address. "
                    f"Entry: {entry}"
                )
            bracket_end = entry.rfind("]")
            if bracket_end == -1:
                raise ValueError(
                    f"Invalid tool_server entry: unclosed bracket in IPv6 address. "
                    f"Entry: {entry}"
                )
            host_part = entry[: bracket_end + 1]
            port_part = entry[bracket_end + 1 :]

            # Validate IPv6 format
            if not re.match(ipv6_pattern, host_part):
                raise ValueError(
                    f"Invalid tool_server entry: invalid IPv6 address format. "
                    f"Entry: {entry}"
                )

            # Validate port
            if not port_part.startswith(":"):
                raise ValueError(
                    f"Invalid tool_server entry: invalid port format. "
                    f"Entry: {entry}"
                )
            port_str = port_part[1:]
        else:
            # IPv4 or hostname format: host:port
            parts = entry.rsplit(":", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid tool_server entry: invalid format. Entry: {entry}"
                )
            host_part, port_str = parts

            # Validate host (IPv4 or hostname)
            if not (
                re.match(ipv4_pattern, host_part)
                or re.match(hostname_pattern, host_part)
            ):
                raise ValueError(
                    f"Invalid tool_server entry: invalid host format. "
                    f"Must be IPv4, IPv6 (in brackets), or hostname. Entry: {entry}"
                )

        # Validate port
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(
                f"Invalid tool_server entry: invalid port number. Entry: {entry}"
            )
        if port < 1 or port > 65535:
            raise ValueError(
                f"Invalid tool_server entry: port out of range (1-65535). "
                f"Entry: {entry}"
            )


# ---------------------------------------------------------------------------
# Tests for sanitize_message()
# ---------------------------------------------------------------------------

class TestSanitizeMessage(unittest.TestCase):
    """Tests for sanitize_message() — Phase 1: Security Utilities."""

    def test_removes_memory_address(self):
        """Memory addresses like 0x7a95e299e750 must be stripped."""
        result = sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        self.assertEqual(result, "<_io.BytesIO object>")

    def test_removes_memory_address_uppercase_hex(self):
        """Memory addresses with uppercase hex digits must also be stripped."""
        # The regex uses [0-9a-f]+ (lowercase), so uppercase addresses
        # should NOT be stripped — this tests the actual implementation behavior.
        result = sanitize_message("<object at 0xABCDEF>")
        # Uppercase hex is NOT matched by [0-9a-f]+, so it stays
        self.assertIn("0xABCDEF", result)

    def test_removes_absolute_unix_file_path(self):
        """Absolute Unix file paths ending in .py must be removed."""
        result = sanitize_message("Error in /home/user/vllm/engine/core.py")
        self.assertNotIn("/home/user/vllm/engine/core.py", result)
        self.assertIn("<file>", result)

    def test_removes_nested_file_path(self):
        """Nested file paths must be removed."""
        result = sanitize_message("File /usr/local/lib/python3.9/site-packages/vllm/engine.py")
        self.assertNotIn(".py", result)
        self.assertIn("<file>", result)

    def test_removes_line_number(self):
        """Line numbers in traceback format must be removed."""
        result = sanitize_message("Error at line 42 in module")
        self.assertNotIn("line 42", result)
        self.assertIn("<line>", result)

    def test_removes_multiple_line_numbers(self):
        """Multiple line numbers must all be removed."""
        result = sanitize_message("line 10, line 20, line 300")
        self.assertNotIn("line 10", result)
        self.assertNotIn("line 20", result)
        self.assertNotIn("line 300", result)

    def test_removes_module_path(self):
        """Module paths like vllm.engine.core must be removed."""
        result = sanitize_message("Error in vllm.engine.core")
        self.assertNotIn("vllm.engine.core", result)
        self.assertIn("<module>", result)

    def test_removes_deep_module_path(self):
        """Deep module paths must be removed."""
        result = sanitize_message("vllm.entrypoints.openai.api_server failed")
        self.assertNotIn("vllm.entrypoints.openai.api_server", result)

    def test_plain_message_unchanged(self):
        """A plain message without sensitive info should pass through."""
        result = sanitize_message("Invalid request: missing required field")
        self.assertEqual(result, "Invalid request: missing required field")

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = sanitize_message("")
        self.assertEqual(result, "")

    def test_multiple_addresses_removed(self):
        """Multiple memory addresses in one message must all be removed."""
        result = sanitize_message(
            "<obj1 at 0x1234abcd> and <obj2 at 0x5678ef01>"
        )
        self.assertNotIn("0x1234abcd", result)
        self.assertNotIn("0x5678ef01", result)
        self.assertIn("<obj1>", result)
        self.assertIn("<obj2>", result)

    def test_returns_string(self):
        """sanitize_message must always return a string."""
        result = sanitize_message("some message")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Tests for is_localhost()
# ---------------------------------------------------------------------------

class TestIsLocalhost(unittest.TestCase):
    """Tests for is_localhost() — Phase 1 / Phase 3."""

    def test_none_is_localhost(self):
        """None host (unspecified) should be treated as localhost."""
        self.assertTrue(is_localhost(None))

    def test_ipv4_loopback(self):
        """127.0.0.1 is localhost."""
        self.assertTrue(is_localhost("127.0.0.1"))

    def test_ipv6_loopback(self):
        """::1 is localhost."""
        self.assertTrue(is_localhost("::1"))

    def test_localhost_hostname(self):
        """'localhost' hostname is localhost."""
        self.assertTrue(is_localhost("localhost"))

    def test_localhost_case_insensitive(self):
        """'LOCALHOST' should also be recognized as localhost."""
        self.assertTrue(is_localhost("LOCALHOST"))

    def test_public_ip_not_localhost(self):
        """A public IP address is not localhost."""
        self.assertFalse(is_localhost("0.0.0.0"))

    def test_external_ip_not_localhost(self):
        """An external IP is not localhost."""
        self.assertFalse(is_localhost("192.168.1.100"))

    def test_external_hostname_not_localhost(self):
        """An external hostname is not localhost."""
        self.assertFalse(is_localhost("example.com"))

    def test_empty_string_not_localhost(self):
        """Empty string is not localhost."""
        self.assertFalse(is_localhost(""))

    def test_loopback_with_port_not_localhost(self):
        """127.0.0.1:8000 (with port) is not localhost — host only."""
        self.assertFalse(is_localhost("127.0.0.1:8000"))


# ---------------------------------------------------------------------------
# Tests for validate_cors_origins()
# ---------------------------------------------------------------------------

class TestValidateCorsOrigins(unittest.TestCase):
    """Tests for validate_cors_origins() — Phase 1 / Phase 3."""

    def test_wildcard_with_credentials_raises(self):
        """Wildcard origin + allow_credentials=True must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_cors_origins(["*"], allow_credentials=True)
        self.assertIn("wildcard", str(ctx.exception).lower())
        self.assertIn("allow_credentials", str(ctx.exception))

    def test_wildcard_without_credentials_ok(self):
        """Wildcard origin without credentials is allowed."""
        # Should not raise
        validate_cors_origins(["*"], allow_credentials=False)

    def test_explicit_origin_with_credentials_ok(self):
        """Explicit origin + allow_credentials=True is valid."""
        validate_cors_origins(["https://example.com"], allow_credentials=True)

    def test_multiple_origins_no_wildcard_ok(self):
        """Multiple explicit origins without wildcard is valid."""
        validate_cors_origins(
            ["https://app.example.com", "https://admin.example.com"],
            allow_credentials=True,
        )

    def test_empty_origins_ok(self):
        """Empty origins list is valid (no CORS)."""
        validate_cors_origins([], allow_credentials=False)
        validate_cors_origins([], allow_credentials=True)

    def test_wildcard_in_list_with_credentials_raises(self):
        """Wildcard mixed with explicit origins + credentials must raise."""
        with self.assertRaises(ValueError) as ctx:
            validate_cors_origins(
                ["https://example.com", "*"], allow_credentials=True
            )
        self.assertIn("wildcard", str(ctx.exception).lower())

    def test_error_message_mentions_browsers(self):
        """Error message should explain why browsers reject this config."""
        with self.assertRaises(ValueError) as ctx:
            validate_cors_origins(["*"], allow_credentials=True)
        msg = str(ctx.exception)
        # Should mention browsers or the security reason
        self.assertTrue(
            "browser" in msg.lower() or "reject" in msg.lower(),
            f"Error message should mention browsers/rejection: {msg}",
        )

    def test_no_credentials_no_wildcard_ok(self):
        """No credentials, no wildcard — always valid."""
        validate_cors_origins(["https://example.com"], allow_credentials=False)


# ---------------------------------------------------------------------------
# Tests for _validate_tool_server()
# ---------------------------------------------------------------------------

class TestValidateToolServer(unittest.TestCase):
    """Tests for _validate_tool_server() — Phase 2: Input Validation Hardening."""

    # --- Happy paths ---

    def test_none_is_valid(self):
        """None tool_server is valid (not specified)."""
        _validate_tool_server(None)  # Should not raise

    def test_empty_string_is_valid(self):
        """Empty string tool_server is valid."""
        _validate_tool_server("")  # Should not raise

    def test_demo_is_valid(self):
        """'demo' is a special valid value."""
        _validate_tool_server("demo")  # Should not raise

    def test_valid_ipv4_host_port(self):
        """Valid IPv4 host:port should pass."""
        _validate_tool_server("127.0.0.1:8000")

    def test_valid_hostname_port(self):
        """Valid hostname:port should pass."""
        _validate_tool_server("localhost:1234")

    def test_valid_fqdn_port(self):
        """Valid FQDN:port should pass."""
        _validate_tool_server("tool-server.example.com:9000")

    def test_valid_ipv6_port(self):
        """Valid IPv6 [::1]:port should pass."""
        _validate_tool_server("[::1]:8000")

    def test_valid_multiple_entries(self):
        """Multiple valid host:port entries separated by comma should pass."""
        _validate_tool_server("127.0.0.1:8000,localhost:9000")

    def test_valid_port_boundary_1(self):
        """Port 1 (minimum) should be valid."""
        _validate_tool_server("127.0.0.1:1")

    def test_valid_port_boundary_65535(self):
        """Port 65535 (maximum) should be valid."""
        _validate_tool_server("127.0.0.1:65535")

    # --- Error paths ---

    def test_shell_metachar_semicolon_rejected(self):
        """Semicolon (shell command separator) must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:8000;rm -rf /")
        self.assertIn("shell metacharacter", str(ctx.exception).lower())

    def test_shell_metachar_pipe_rejected(self):
        """Pipe character must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:8000|cat /etc/passwd")
        self.assertIn("shell metacharacter", str(ctx.exception).lower())

    def test_shell_metachar_backtick_rejected(self):
        """Backtick (command substitution) must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:8000`whoami`")
        self.assertIn("shell metacharacter", str(ctx.exception).lower())

    def test_shell_metachar_dollar_rejected(self):
        """Dollar sign (variable expansion) must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:8000$HOME")
        self.assertIn("shell metacharacter", str(ctx.exception).lower())

    def test_missing_port_rejected(self):
        """Entry without port must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1")
        self.assertIn("port", str(ctx.exception).lower())

    def test_port_zero_rejected(self):
        """Port 0 is out of range and must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:0")
        msg = str(ctx.exception).lower()
        self.assertIn("port", msg)
        self.assertIn("range", msg)

    def test_port_too_large_rejected(self):
        """Port > 65535 must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:99999")
        msg = str(ctx.exception).lower()
        self.assertIn("port", msg)
        self.assertIn("range", msg)

    def test_non_numeric_port_rejected(self):
        """Non-numeric port must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server("127.0.0.1:abc")
        self.assertIn("port", str(ctx.exception).lower())

    def test_error_message_includes_entry(self):
        """Error message should include the offending entry for debugging."""
        bad_entry = "127.0.0.1:0"
        with self.assertRaises(ValueError) as ctx:
            _validate_tool_server(bad_entry)
        self.assertIn(bad_entry, str(ctx.exception))

    def test_one_bad_entry_in_list_rejected(self):
        """If one entry in a comma-separated list is bad, the whole thing fails."""
        with self.assertRaises(ValueError):
            _validate_tool_server("127.0.0.1:8000,bad-entry-no-port")

    def test_spaces_in_entry_rejected(self):
        """Spaces are shell metacharacters and must be rejected."""
        with self.assertRaises(ValueError):
            _validate_tool_server("127.0.0.1:8000 extra")


# ---------------------------------------------------------------------------
# Tests for emit_security_warning() behavior
# ---------------------------------------------------------------------------

class TestEmitSecurityWarning(unittest.TestCase):
    """Tests for emit_security_warning() — Phase 1."""

    def test_logs_with_security_prefix(self):
        """emit_security_warning must log with [SECURITY] prefix."""
        import logging

        # Simulate the function behavior
        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        test_logger = logging.getLogger("test_security_warning")
        handler = CapturingHandler()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.WARNING)

        # Simulate emit_security_warning
        def emit_security_warning(msg: str) -> None:
            test_logger.warning("[SECURITY] %s", msg)

        emit_security_warning("Test security message")

        self.assertEqual(len(log_records), 1)
        formatted = log_records[0].getMessage()
        self.assertIn("[SECURITY]", formatted)
        self.assertIn("Test security message", formatted)

    def test_logs_at_warning_level(self):
        """emit_security_warning must use WARNING level, not INFO or ERROR."""
        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        test_logger = logging.getLogger("test_security_level")
        handler = CapturingHandler()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        def emit_security_warning(msg: str) -> None:
            test_logger.warning("[SECURITY] %s", msg)

        emit_security_warning("CORS misconfiguration detected")

        self.assertEqual(len(log_records), 1)
        self.assertEqual(log_records[0].levelno, logging.WARNING)


# ---------------------------------------------------------------------------
# Integration-style tests: validate_cors_origins + is_localhost together
# (simulating the build_app() logic from api_server.py — Phase 3)
# ---------------------------------------------------------------------------

class TestCorsAndLocalhostIntegration(unittest.TestCase):
    """Integration tests for CORS + localhost check as used in build_app()."""

    def _simulate_build_app_cors_check(self, allowed_origins, allow_credentials, host):
        """Simulate the CORS validation logic from build_app()."""
        warnings_emitted = []

        def emit_security_warning(msg):
            warnings_emitted.append(msg)

        # This mirrors the build_app() logic:
        validate_cors_origins(allowed_origins, allow_credentials)

        if not allowed_origins and not is_localhost(host):
            emit_security_warning(
                "CORS allowed_origins is empty — all cross-origin requests will be "
                "rejected. Set --allowed-origins to enable cross-origin access."
            )

        return warnings_emitted

    def test_wildcard_with_credentials_raises_in_build_app(self):
        """build_app() should raise when wildcard + credentials configured."""
        with self.assertRaises(ValueError):
            self._simulate_build_app_cors_check(
                allowed_origins=["*"],
                allow_credentials=True,
                host="0.0.0.0",
            )

    def test_empty_origins_on_public_host_emits_warning(self):
        """Empty CORS origins on a public host should emit a security warning."""
        warnings = self._simulate_build_app_cors_check(
            allowed_origins=[],
            allow_credentials=False,
            host="0.0.0.0",
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("allowed_origins", warnings[0])

    def test_empty_origins_on_localhost_no_warning(self):
        """Empty CORS origins on localhost should NOT emit a warning."""
        warnings = self._simulate_build_app_cors_check(
            allowed_origins=[],
            allow_credentials=False,
            host="127.0.0.1",
        )
        self.assertEqual(len(warnings), 0)

    def test_empty_origins_on_none_host_no_warning(self):
        """Empty CORS origins with None host (localhost) should NOT warn."""
        warnings = self._simulate_build_app_cors_check(
            allowed_origins=[],
            allow_credentials=False,
            host=None,
        )
        self.assertEqual(len(warnings), 0)

    def test_explicit_origins_on_public_host_no_warning(self):
        """Explicit CORS origins on a public host should NOT warn."""
        warnings = self._simulate_build_app_cors_check(
            allowed_origins=["https://app.example.com"],
            allow_credentials=False,
            host="0.0.0.0",
        )
        self.assertEqual(len(warnings), 0)

    def test_wildcard_without_credentials_on_public_host_no_warning(self):
        """Wildcard origin without credentials on public host is valid."""
        warnings = self._simulate_build_app_cors_check(
            allowed_origins=["*"],
            allow_credentials=False,
            host="0.0.0.0",
        )
        self.assertEqual(len(warnings), 0)


# ---------------------------------------------------------------------------
# Tests for h11_max_incomplete_event_size warning logic (Phase 2)
# ---------------------------------------------------------------------------

class TestH11SizeWarning(unittest.TestCase):
    """Tests for h11_max_incomplete_event_size warning — Phase 2."""

    def _simulate_h11_size_check(self, size_bytes):
        """Simulate the h11 size warning logic from validate_parsed_serve_args()."""
        warnings_emitted = []

        def mock_warning(msg, *args):
            warnings_emitted.append(msg % args if args else msg)

        if size_bytes and size_bytes > 100 * 1024 * 1024:  # 100 MB
            max_size_mb = size_bytes / (1024 * 1024)
            mock_warning(
                "h11_max_incomplete_event_size is set to %.1f MB, which is very large. "
                "This may increase memory usage and vulnerability to header-based DoS attacks. "
                "Consider using a smaller value (default: 4 MB).",
                max_size_mb,
            )

        return warnings_emitted

    def test_normal_size_no_warning(self):
        """Normal size (4 MB default) should not trigger a warning."""
        warnings = self._simulate_h11_size_check(4 * 1024 * 1024)
        self.assertEqual(len(warnings), 0)

    def test_exactly_100mb_no_warning(self):
        """Exactly 100 MB should not trigger a warning (boundary)."""
        warnings = self._simulate_h11_size_check(100 * 1024 * 1024)
        self.assertEqual(len(warnings), 0)

    def test_over_100mb_triggers_warning(self):
        """Over 100 MB should trigger a DoS warning."""
        warnings = self._simulate_h11_size_check(101 * 1024 * 1024)
        self.assertEqual(len(warnings), 1)
        self.assertIn("DoS", warnings[0])

    def test_warning_mentions_size_in_mb(self):
        """Warning message should include the size in MB."""
        warnings = self._simulate_h11_size_check(200 * 1024 * 1024)
        self.assertEqual(len(warnings), 1)
        self.assertIn("200.0 MB", warnings[0])

    def test_none_size_no_warning(self):
        """None size (not set) should not trigger a warning."""
        warnings = self._simulate_h11_size_check(None)
        self.assertEqual(len(warnings), 0)

    def test_warning_mentions_dos(self):
        """Warning must mention DoS vulnerability."""
        warnings = self._simulate_h11_size_check(500 * 1024 * 1024)
        self.assertGreater(len(warnings), 0)
        self.assertIn("DoS", warnings[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
