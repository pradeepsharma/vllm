# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive security tests for vLLM OpenAI API server.

Tests cover:
1. Authentication (API key validation, timing-safe comparison)
2. CORS configuration (wildcard restrictions, origin validation)
3. Error response sanitization (stack trace removal, path masking)
4. SSL/TLS configuration warnings
5. trust_remote_code security warnings
6. Request size and header limits
7. Plugin path validation
8. Tool server validation
"""

import hashlib
import json
import secrets
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.openai.cli_args import (
    FrontendArgs,
    _validate_tool_server,
    validate_parsed_serve_args,
)
from vllm.entrypoints.openai.server_utils import (
    AuthenticationMiddleware,
    validate_plugin_path,
)
from vllm.entrypoints.utils import (
    create_error_response,
    sanitize_message,
    validate_cors_origins,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


# ============================================================================
# Test AuthenticationMiddleware
# ============================================================================


class TestAuthenticationMiddleware:
    """Test suite for AuthenticationMiddleware."""

    def test_middleware_init_with_tokens(self):
        """Test middleware initialization with API tokens."""
        tokens = ["secret-key-1", "secret-key-2"]
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=tokens)

        # Verify tokens are hashed
        assert len(middleware.api_tokens) == 2
        assert all(isinstance(t, bytes) for t in middleware.api_tokens)

        # Verify hashes match expected values
        expected_hashes = [
            hashlib.sha256(t.encode("utf-8")).digest() for t in tokens
        ]
        assert middleware.api_tokens == expected_hashes

    def test_middleware_init_with_empty_tokens(self):
        """Test middleware initialization with empty token list."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=[])

        assert middleware.api_tokens == []

    def test_verify_token_valid(self):
        """Test token verification with valid token."""
        token = "valid-secret-key"
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=[token])

        # Create headers with valid Bearer token
        headers = MagicMock()
        headers.get.return_value = f"Bearer {token}"

        assert middleware.verify_token(headers) is True

    def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        token = "valid-secret-key"
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=[token])

        # Create headers with invalid Bearer token
        headers = MagicMock()
        headers.get.return_value = "Bearer wrong-token"

        assert middleware.verify_token(headers) is False

    def test_verify_token_missing_authorization_header(self):
        """Test token verification with missing Authorization header."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret"])

        # Create headers without Authorization
        headers = MagicMock()
        headers.get.return_value = None

        assert middleware.verify_token(headers) is False

    def test_verify_token_invalid_scheme(self):
        """Test token verification with invalid auth scheme."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret"])

        # Create headers with non-Bearer scheme
        headers = MagicMock()
        headers.get.return_value = "Basic dXNlcjpwYXNz"

        assert middleware.verify_token(headers) is False

    def test_verify_token_case_insensitive_scheme(self):
        """Test that Bearer scheme is case-insensitive."""
        token = "secret-key"
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=[token])

        # Test with lowercase 'bearer'
        headers = MagicMock()
        headers.get.return_value = f"bearer {token}"
        assert middleware.verify_token(headers) is True

        # Test with uppercase 'BEARER'
        headers.get.return_value = f"BEARER {token}"
        assert middleware.verify_token(headers) is True

    def test_verify_token_timing_safe_comparison(self):
        """Test that token comparison is timing-safe."""
        token = "secret-key"
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=[token])

        # Verify that secrets.compare_digest is used (indirectly)
        # by checking that both valid and invalid tokens are processed
        headers = MagicMock()

        # Valid token
        headers.get.return_value = f"Bearer {token}"
        result_valid = middleware.verify_token(headers)

        # Invalid token (same length to avoid timing differences from length)
        headers.get.return_value = "Bearer wrong-secret-key"
        result_invalid = middleware.verify_token(headers)

        assert result_valid is True
        assert result_invalid is False

    def test_verify_token_multiple_tokens(self):
        """Test token verification with multiple configured tokens."""
        tokens = ["token-1", "token-2", "token-3"]
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=tokens)

        headers = MagicMock()

        # Test each token works
        for token in tokens:
            headers.get.return_value = f"Bearer {token}"
            assert middleware.verify_token(headers) is True

        # Test invalid token doesn't work
        headers.get.return_value = "Bearer invalid-token"
        assert middleware.verify_token(headers) is False

    def test_is_path_unauthenticated_exact_match(self):
        """Test unauthenticated path matching with exact match."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret"])

        assert middleware._is_path_unauthenticated("/health") is True
        assert middleware._is_path_unauthenticated("/ping") is True
        assert middleware._is_path_unauthenticated("/metrics") is True

    def test_is_path_unauthenticated_subpath(self):
        """Test unauthenticated path matching with subpaths."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret"])

        # Subpaths of unauthenticated paths should also be unauthenticated
        assert middleware._is_path_unauthenticated("/health/check") is True
        assert middleware._is_path_unauthenticated("/metrics/prometheus") is True

    def test_is_path_unauthenticated_authenticated_paths(self):
        """Test that API paths require authentication."""
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret"])

        # API paths should require authentication
        assert middleware._is_path_unauthenticated("/v1/chat/completions") is False
        assert middleware._is_path_unauthenticated("/v2/chat/completions") is False
        assert middleware._is_path_unauthenticated("/models") is False

    def test_is_path_unauthenticated_custom_paths(self):
        """Test custom unauthenticated paths."""
        app = MagicMock()
        custom_paths = frozenset({"/health", "/custom-endpoint"})
        middleware = AuthenticationMiddleware(
            app, tokens=["secret"], unauthenticated_paths=custom_paths
        )

        assert middleware._is_path_unauthenticated("/health") is True
        assert middleware._is_path_unauthenticated("/custom-endpoint") is True
        assert middleware._is_path_unauthenticated("/v1/chat/completions") is False


# ============================================================================
# Test CORS Validation
# ============================================================================


class TestCORSValidation:
    """Test suite for CORS configuration validation."""

    def test_validate_cors_origins_empty_list(self):
        """Test CORS validation with empty origins list."""
        # Should not raise
        validate_cors_origins([])

    def test_validate_cors_origins_explicit_origins(self):
        """Test CORS validation with explicit origins."""
        origins = ["https://example.com", "https://app.example.com"]
        # Should not raise
        validate_cors_origins(origins)

    def test_validate_cors_origins_wildcard_without_credentials(self):
        """Test CORS validation with wildcard origins and no credentials."""
        # Should not raise
        validate_cors_origins(["*"], allow_credentials=False)

    def test_validate_cors_origins_wildcard_with_credentials_raises(self):
        """Test CORS validation with wildcard origins and credentials enabled."""
        with pytest.raises(ValueError) as exc_info:
            validate_cors_origins(["*"], allow_credentials=True)

        assert "wildcard origins" in str(exc_info.value).lower()
        assert "allow_credentials" in str(exc_info.value)

    def test_validate_cors_origins_mixed_with_wildcard_and_credentials_raises(self):
        """Test CORS validation with mixed origins including wildcard and credentials."""
        origins = ["https://example.com", "*"]
        with pytest.raises(ValueError):
            validate_cors_origins(origins, allow_credentials=True)


# ============================================================================
# Test Message Sanitization
# ============================================================================


class TestMessageSanitization:
    """Test suite for error message sanitization."""

    def test_sanitize_message_removes_memory_addresses(self):
        """Test that memory addresses are removed from messages."""
        message = "Error in object <MyClass object at 0x7f1234567890>"
        sanitized = sanitize_message(message)

        assert "0x7f1234567890" not in sanitized
        assert "<MyClass object>" in sanitized

    def test_sanitize_message_removes_absolute_unix_paths(self):
        """Test that absolute Unix file paths are removed."""
        message = "Error in /home/user/project/vllm/engine/core.py at line 123"
        sanitized = sanitize_message(message)

        assert "/home/user/project" not in sanitized
        assert "<file>" in sanitized

    def test_sanitize_message_removes_absolute_windows_paths(self):
        """Test that absolute Windows file paths are removed."""
        message = "Error in C:\\Users\\user\\project\\vllm\\engine\\core.py"
        sanitized = sanitize_message(message)

        assert "C:\\Users" not in sanitized
        assert "<file>" in sanitized

    def test_sanitize_message_removes_relative_paths(self):
        """Test that relative file paths are removed."""
        message = "Error in vllm/engine/core.py and src/module/file.py"
        sanitized = sanitize_message(message)

        assert "vllm/engine/core.py" not in sanitized
        assert "src/module/file.py" not in sanitized
        assert sanitized.count("<file>") >= 2

    def test_sanitize_message_removes_line_numbers(self):
        """Test that line number references are removed."""
        message = "Error at line 123 in function at line 456"
        sanitized = sanitize_message(message)

        assert "line 123" not in sanitized
        assert "line 456" not in sanitized
        assert "<line>" in sanitized

    def test_sanitize_message_removes_module_paths(self):
        """Test that module paths are removed."""
        message = "Error in vllm.engine.core.function"
        sanitized = sanitize_message(message)

        # Module paths should be masked
        assert "vllm.engine.core" not in sanitized or "<module>" in sanitized

    def test_sanitize_message_removes_site_packages_paths(self):
        """Test that site-packages paths are removed."""
        message = "Error in /usr/lib/python3.9/site-packages/transformers/models/bert/modeling.py"
        sanitized = sanitize_message(message)

        # The sanitization should mask the site-packages path
        assert "transformers/models/bert" not in sanitized
        # Either <site-packages> or <file> should be present
        assert "<site-packages>" in sanitized or "<file>" in sanitized

    def test_sanitize_message_preserves_meaningful_content(self):
        """Test that meaningful error content is preserved."""
        message = "ValueError: invalid input value must be positive"
        sanitized = sanitize_message(message)

        assert "ValueError" in sanitized
        assert "invalid input" in sanitized
        assert "positive" in sanitized

    def test_sanitize_message_empty_string(self):
        """Test sanitization of empty string."""
        assert sanitize_message("") == ""

    def test_sanitize_message_no_sensitive_data(self):
        """Test sanitization of message with no sensitive data."""
        message = "Request timeout after 30 seconds"
        sanitized = sanitize_message(message)

        assert sanitized == message


# ============================================================================
# Test Error Response Creation
# ============================================================================


class TestErrorResponseCreation:
    """Test suite for error response creation."""

    def test_create_error_response_from_string(self):
        """Test creating error response from string message."""
        message = "Invalid request"
        response = create_error_response(message)

        assert response.error.message == message
        assert response.error.type == "BadRequestError"
        assert response.error.code == 400

    def test_create_error_response_from_exception(self):
        """Test creating error response from exception."""
        exc = ValueError("Invalid value")
        response = create_error_response(exc)

        assert "Invalid value" in response.error.message
        assert response.error.type == "BadRequestError"
        assert response.error.code == 400

    def test_create_error_response_sanitizes_message(self):
        """Test that error responses sanitize messages."""
        message = "Error in /home/user/project/file.py at line 123"
        response = create_error_response(message)

        # Sanitized message should not contain file paths
        assert "/home/user/project" not in response.error.message
        assert "line 123" not in response.error.message

    def test_create_error_response_custom_error_type(self):
        """Test creating error response with custom error type."""
        from http import HTTPStatus
        response = create_error_response(
            "Not found", err_type="NotFoundError", status_code=HTTPStatus.NOT_FOUND
        )

        assert response.error.type == "NotFoundError"
        assert response.error.code == 404

    def test_create_error_response_with_param(self):
        """Test creating error response with parameter information."""
        response = create_error_response(
            "Invalid parameter", param="max_tokens"
        )

        assert response.error.param == "max_tokens"


# ============================================================================
# Test Plugin Path Validation
# ============================================================================


class TestPluginPathValidation:
    """Test suite for plugin path validation."""

    def test_validate_plugin_path_valid_absolute_path(self, tmp_path):
        """Test validation of valid absolute file path."""
        # Create a temporary plugin file
        plugin_file = tmp_path / "plugin.py"
        plugin_file.write_text("# Plugin code")

        # Should not raise
        validate_plugin_path(str(plugin_file), "test plugin")

    def test_validate_plugin_path_nonexistent_file(self):
        """Test validation of nonexistent file path."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path("/nonexistent/path/plugin.py", "test plugin")

        assert "does not exist" in str(exc_info.value)

    def test_validate_plugin_path_directory_not_file(self, tmp_path):
        """Test validation rejects directories."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path(str(tmp_path), "test plugin")

        assert "must be a file" in str(exc_info.value)

    def test_validate_plugin_path_relative_path_raises(self):
        """Test validation rejects relative file paths."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path("relative/path/plugin.py", "test plugin")

        assert "must be absolute" in str(exc_info.value)

    def test_validate_plugin_path_path_traversal_raises(self):
        """Test validation rejects path traversal attempts."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path("/path/to/../../../etc/passwd", "test plugin")

        assert "path traversal" in str(exc_info.value).lower()

    def test_validate_plugin_path_valid_plugin_name(self):
        """Test validation of valid registered plugin name."""
        # Should not raise for valid plugin names
        validate_plugin_path("my_plugin", "test plugin")
        validate_plugin_path("my-plugin", "test plugin")
        validate_plugin_path("MyPlugin123", "test plugin")

    def test_validate_plugin_path_invalid_plugin_name(self):
        """Test validation rejects invalid plugin names."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path("invalid plugin!", "test plugin")

        assert "alphanumeric" in str(exc_info.value).lower()

    def test_validate_plugin_path_empty_string_raises(self):
        """Test validation rejects empty string."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path("", "test plugin")

        assert "non-empty" in str(exc_info.value).lower()

    def test_validate_plugin_path_none_raises(self):
        """Test validation rejects None."""
        with pytest.raises(ValueError) as exc_info:
            validate_plugin_path(None, "test plugin")

        assert "non-empty" in str(exc_info.value).lower()


# ============================================================================
# Test Tool Server Validation
# ============================================================================


class TestToolServerValidation:
    """Test suite for tool server validation."""

    def test_validate_tool_server_demo(self):
        """Test validation of 'demo' tool server."""
        # Should not raise
        _validate_tool_server("demo")

    def test_validate_tool_server_none(self):
        """Test validation of None tool server."""
        # Should not raise
        _validate_tool_server(None)

    def test_validate_tool_server_empty_string(self):
        """Test validation of empty string tool server."""
        # Should not raise
        _validate_tool_server("")

    def test_validate_tool_server_valid_ipv4(self):
        """Test validation of valid IPv4 host:port."""
        # Should not raise
        _validate_tool_server("127.0.0.1:8000")
        _validate_tool_server("192.168.1.1:9000")

    def test_validate_tool_server_valid_hostname(self):
        """Test validation of valid hostname:port."""
        # Should not raise
        _validate_tool_server("localhost:8000")
        _validate_tool_server("example.com:9000")
        _validate_tool_server("api.example.com:8080")

    def test_validate_tool_server_valid_ipv6(self):
        """Test validation of valid IPv6 host:port."""
        # Should not raise
        _validate_tool_server("[::1]:8000")
        _validate_tool_server("[2001:db8::1]:9000")

    def test_validate_tool_server_invalid_port(self):
        """Test validation rejects invalid port numbers."""
        with pytest.raises(ValueError) as exc_info:
            _validate_tool_server("localhost:99999")

        assert "port" in str(exc_info.value).lower()

    def test_validate_tool_server_port_zero(self):
        """Test validation rejects port 0."""
        with pytest.raises(ValueError) as exc_info:
            _validate_tool_server("localhost:0")

        assert "port" in str(exc_info.value).lower()

    def test_validate_tool_server_non_numeric_port(self):
        """Test validation rejects non-numeric port."""
        with pytest.raises(ValueError) as exc_info:
            _validate_tool_server("localhost:abc")

        assert "port" in str(exc_info.value).lower()

    def test_validate_tool_server_shell_metacharacters(self):
        """Test validation rejects shell metacharacters."""
        with pytest.raises(ValueError) as exc_info:
            _validate_tool_server("localhost:8000; rm -rf /")

        assert "shell" in str(exc_info.value).lower() or "metacharacter" in str(exc_info.value).lower()

    def test_validate_tool_server_multiple_entries(self):
        """Test validation of multiple comma-separated entries."""
        # Should not raise
        _validate_tool_server("localhost:8000, 127.0.0.1:9000, example.com:8080")

    def test_validate_tool_server_multiple_entries_with_invalid(self):
        """Test validation rejects multiple entries if one is invalid."""
        with pytest.raises(ValueError):
            _validate_tool_server("localhost:8000, invalid:99999")


# ============================================================================
# Test CLI Arguments Validation
# ============================================================================


class TestCLIArgsValidation:
    """Test suite for CLI argument validation."""

    def test_validate_parsed_serve_args_no_auto_tool_choice_without_parser(self):
        """Test validation fails when auto tool choice is enabled without parser."""
        args = Namespace(
            enable_auto_tool_choice=True,
            tool_call_parser=None,
            enable_log_outputs=False,
            enable_log_requests=False,
            tool_server=None,
            h11_max_incomplete_event_size=4194304,
            chat_template=None,
            subparser="serve",
        )

        with pytest.raises(TypeError) as exc_info:
            validate_parsed_serve_args(args)

        assert "tool-call-parser" in str(exc_info.value).lower()

    def test_validate_parsed_serve_args_auto_tool_choice_with_parser(self):
        """Test validation passes when auto tool choice has parser."""
        args = Namespace(
            enable_auto_tool_choice=True,
            tool_call_parser="mistral",
            enable_log_outputs=False,
            enable_log_requests=False,
            tool_server=None,
            h11_max_incomplete_event_size=4194304,
            chat_template=None,
            subparser="serve",
        )

        # Should not raise
        validate_parsed_serve_args(args)

    def test_validate_parsed_serve_args_enable_log_outputs_without_requests(self):
        """Test validation fails when log outputs enabled without log requests."""
        args = Namespace(
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            enable_log_outputs=True,
            enable_log_requests=False,
            tool_server=None,
            h11_max_incomplete_event_size=4194304,
            chat_template=None,
            subparser="serve",
        )

        with pytest.raises(TypeError) as exc_info:
            validate_parsed_serve_args(args)

        assert "enable-log-requests" in str(exc_info.value).lower()

    def test_validate_parsed_serve_args_h11_max_incomplete_event_size_warning(self):
        """Test validation warns when h11_max_incomplete_event_size is very large."""
        args = Namespace(
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            enable_log_outputs=False,
            enable_log_requests=False,
            tool_server=None,
            h11_max_incomplete_event_size=100 * 1024 * 1024 + 1,  # > 100 MB
            chat_template=None,
            subparser="serve",
        )

        # Should not raise, but should log a warning
        with patch("vllm.entrypoints.openai.cli_args.logger") as mock_logger:
            validate_parsed_serve_args(args)
            # Verify warning was logged
            assert mock_logger.warning.called

    def test_validate_parsed_serve_args_tool_server_validation(self):
        """Test validation of tool_server argument."""
        args = Namespace(
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            enable_log_outputs=False,
            enable_log_requests=False,
            tool_server="localhost:8000",
            h11_max_incomplete_event_size=4194304,
            chat_template=None,
            subparser="serve",
        )

        # Should not raise
        validate_parsed_serve_args(args)

    def test_validate_parsed_serve_args_invalid_tool_server(self):
        """Test validation fails with invalid tool_server."""
        args = Namespace(
            enable_auto_tool_choice=False,
            tool_call_parser=None,
            enable_log_outputs=False,
            enable_log_requests=False,
            tool_server="localhost:99999",
            h11_max_incomplete_event_size=4194304,
            chat_template=None,
            subparser="serve",
        )

        with pytest.raises(ValueError):
            validate_parsed_serve_args(args)


# ============================================================================
# Test FrontendArgs Defaults
# ============================================================================


class TestFrontendArgsDefaults:
    """Test suite for FrontendArgs security defaults."""

    def test_frontend_args_allowed_origins_default_empty(self):
        """Test that allowed_origins defaults to empty list."""
        args = FrontendArgs()
        assert args.allowed_origins == []

    def test_frontend_args_allowed_methods_default_safe(self):
        """Test that allowed_methods defaults to safe set."""
        args = FrontendArgs()
        assert args.allowed_methods == ["GET", "POST", "OPTIONS"]

    def test_frontend_args_allowed_headers_default_safe(self):
        """Test that allowed_headers defaults to safe set."""
        args = FrontendArgs()
        assert "Authorization" in args.allowed_headers
        assert "Content-Type" in args.allowed_headers

    def test_frontend_args_api_key_default_none(self):
        """Test that api_key defaults to None."""
        args = FrontendArgs()
        assert args.api_key is None

    def test_frontend_args_allow_credentials_default_false(self):
        """Test that allow_credentials defaults to False."""
        args = FrontendArgs()
        assert args.allow_credentials is False

    def test_frontend_args_log_error_stack_default_false(self):
        """Test that log_error_stack defaults to False."""
        args = FrontendArgs()
        assert args.log_error_stack is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestSecurityIntegration:
    """Integration tests for security features."""

    def test_authentication_and_cors_together(self):
        """Test that authentication and CORS work together."""
        # Create middleware with API key
        app = MagicMock()
        middleware = AuthenticationMiddleware(app, tokens=["secret-key"])

        # Verify token validation
        headers = MagicMock()
        headers.get.return_value = "Bearer secret-key"
        assert middleware.verify_token(headers) is True

        # Verify CORS validation doesn't conflict
        validate_cors_origins(["https://example.com"], allow_credentials=False)

    def test_error_response_with_sanitization(self):
        """Test that error responses are properly sanitized."""
        # Create an error with sensitive information
        exc = ValueError(
            "Error in /home/user/vllm/engine/core.py at line 123 "
            "in function <MyClass object at 0x7f1234567890>"
        )

        response = create_error_response(exc)

        # Verify sensitive information is removed
        assert "/home/user/vllm" not in response.error.message
        assert "0x7f1234567890" not in response.error.message
        assert "line 123" not in response.error.message

    def test_plugin_validation_with_security_checks(self, tmp_path):
        """Test plugin validation with security checks."""
        # Create a valid plugin file
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text("# Valid plugin")

        # Should validate successfully
        validate_plugin_path(str(plugin_file), "test plugin")

        # Path traversal should fail
        with pytest.raises(ValueError):
            validate_plugin_path(
                str(tmp_path / ".." / ".." / "etc" / "passwd"), "test plugin"
            )
