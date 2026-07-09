# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Security tests: verify that CORS defaults are secure and that
ssl_cert_reqs defaults to CERT_REQUIRED.
"""

import argparse
import ssl

import pytest

from vllm.entrypoints.openai.cli_args import (
    FrontendArgs,
    validate_parsed_serve_args,
)


class TestCORSDefaults:
    """Test CORS security defaults."""

    def test_allowed_origins_default_is_empty_list(self):
        """Test that allowed_origins defaults to [] (not ['*'])."""
        args = FrontendArgs()
        assert args.allowed_origins == []
        assert isinstance(args.allowed_origins, list)

    def test_allowed_methods_default_is_safe(self):
        """Test that allowed_methods defaults to safe methods."""
        args = FrontendArgs()
        assert args.allowed_methods == ["GET", "POST", "OPTIONS"]
        assert "DELETE" not in args.allowed_methods
        assert "PUT" not in args.allowed_methods

    def test_allowed_headers_default_is_safe(self):
        """Test that allowed_headers defaults to safe headers."""
        args = FrontendArgs()
        assert args.allowed_headers == ["Authorization", "Content-Type", "X-Request-Id"]
        # Should not include sensitive headers by default
        assert "Cookie" not in args.allowed_headers

    def test_allow_credentials_default_is_false(self):
        """Test that allow_credentials defaults to False."""
        args = FrontendArgs()
        assert args.allow_credentials is False

    def test_ssl_cert_reqs_default_is_cert_required(self):
        """Test that ssl_cert_reqs defaults to CERT_REQUIRED."""
        args = FrontendArgs()
        assert args.ssl_cert_reqs == int(ssl.CERT_REQUIRED)
        assert args.ssl_cert_reqs != int(ssl.CERT_NONE)
        assert args.ssl_cert_reqs != int(ssl.CERT_OPTIONAL)

    def test_validate_rejects_wildcard_origins_with_credentials(self):
        """
        Test that validate_parsed_serve_args rejects the insecure combination
        of allow_credentials=True and allowed_origins=['*'].
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=True,
            allowed_origins=["*"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            validate_parsed_serve_args(args)
        
        assert "CORS configuration is insecure" in str(exc_info.value)
        assert "allow_credentials=True" in str(exc_info.value)
        assert "allowed_origins=['*']" in str(exc_info.value)

    def test_validate_allows_wildcard_origins_without_credentials(self):
        """
        Test that wildcard origins are allowed when allow_credentials=False.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=False,
            allowed_origins=["*"],
        )
        
        # Should not raise
        validate_parsed_serve_args(args)

    def test_validate_allows_credentials_with_explicit_origins(self):
        """
        Test that allow_credentials=True is allowed with explicit origins.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=True,
            allowed_origins=["https://example.com", "https://app.example.com"],
        )
        
        # Should not raise
        validate_parsed_serve_args(args)

    def test_validate_allows_credentials_with_empty_origins(self):
        """
        Test that allow_credentials=True is allowed with empty origins list.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=True,
            allowed_origins=[],
        )
        
        # Should not raise
        validate_parsed_serve_args(args)

    def test_validate_skips_non_serve_subparser(self):
        """
        Test that validation is skipped for non-serve subparsers.
        """
        args = argparse.Namespace(
            subparser="other",
            allow_credentials=True,
            allowed_origins=["*"],
        )
        
        # Should not raise (validation skipped for non-serve)
        validate_parsed_serve_args(args)

    def test_validate_handles_missing_attributes(self):
        """
        Test that validation handles missing attributes gracefully.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            # Missing allow_credentials and allowed_origins
        )
        
        # Should not raise (hasattr checks prevent AttributeError)
        validate_parsed_serve_args(args)

    def test_allowed_origins_can_be_set_to_specific_domains(self):
        """
        Test that allowed_origins can be set to specific domains.
        """
        args = FrontendArgs(
            allowed_origins=["https://example.com", "https://app.example.com"]
        )
        assert args.allowed_origins == ["https://example.com", "https://app.example.com"]

    def test_allowed_origins_can_be_set_to_localhost(self):
        """
        Test that allowed_origins can be set to localhost for development.
        """
        args = FrontendArgs(
            allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"]
        )
        assert args.allowed_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]

    def test_ssl_cert_reqs_can_be_overridden(self):
        """
        Test that ssl_cert_reqs can be overridden if needed.
        """
        args = FrontendArgs(ssl_cert_reqs=int(ssl.CERT_NONE))
        assert args.ssl_cert_reqs == int(ssl.CERT_NONE)

    def test_validate_error_message_is_helpful(self):
        """
        Test that the error message for insecure CORS is helpful.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=True,
            allowed_origins=["*"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            validate_parsed_serve_args(args)
        
        error_msg = str(exc_info.value)
        # Check that the error message includes helpful guidance
        assert "set allow_credentials=False" in error_msg or "specify explicit allowed_origins" in error_msg

    def test_multiple_wildcard_origins_rejected(self):
        """
        Test that even if wildcard is mixed with other origins, it's still rejected
        when allow_credentials=True.
        """
        args = argparse.Namespace(
            subparser="serve",
            chat_template=None,
            enable_auto_tool_choice=False,
            enable_log_outputs=False,
            allow_credentials=True,
            allowed_origins=["https://example.com", "*"],
        )
        
        with pytest.raises(ValueError) as exc_info:
            validate_parsed_serve_args(args)
        
        assert "CORS configuration is insecure" in str(exc_info.value)

    def test_cors_defaults_are_immutable_per_instance(self):
        """
        Test that modifying CORS defaults on one instance doesn't affect others.
        """
        args1 = FrontendArgs()
        args2 = FrontendArgs()
        
        args1.allowed_origins.append("https://example.com")
        
        # args2 should still have empty list
        assert args2.allowed_origins == []
        assert args1.allowed_origins == ["https://example.com"]
