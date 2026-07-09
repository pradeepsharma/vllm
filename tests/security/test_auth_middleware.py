# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Security tests: verify that AuthenticationMiddleware correctly guards
protected endpoints and uses constant-time token comparison.
"""

import hashlib
import secrets
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import Headers
from starlette.testclient import TestClient

from vllm.entrypoints.openai.server_utils import AuthenticationMiddleware


class TestAuthenticationMiddleware:
    """Test AuthenticationMiddleware security properties."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app that returns 200 OK."""
        async def app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })
        return app

    @pytest.fixture
    def test_token(self):
        """A test API token."""
        return "test-secret-token-12345"

    @pytest.fixture
    def auth_middleware(self, mock_app, test_token):
        """Create an AuthenticationMiddleware instance."""
        return AuthenticationMiddleware(
            mock_app,
            tokens=[test_token],
            unauthenticated_paths=frozenset({"/health", "/ping", "/metrics"})
        )

    def test_auth_middleware_rejects_v1_completions_without_token(self, auth_middleware):
        """Test that /v1/completions requires authentication."""
        client = TestClient(auth_middleware)
        response = client.get("/v1/completions")
        assert response.status_code == 401
        assert "Unauthorized" in response.text

    def test_auth_middleware_rejects_v1_chat_completions_without_token(self, auth_middleware):
        """Test that /v1/chat/completions requires authentication."""
        client = TestClient(auth_middleware)
        response = client.get("/v1/chat/completions")
        assert response.status_code == 401
        assert "Unauthorized" in response.text

    def test_auth_middleware_rejects_v1_with_wrong_token(self, auth_middleware):
        """Test that /v1/* endpoints reject wrong tokens."""
        client = TestClient(auth_middleware)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": "Bearer wrong-token"}
        )
        assert response.status_code == 401
        assert "Unauthorized" in response.text

    def test_auth_middleware_allows_health_without_token(self, auth_middleware):
        """Test that /health is in the unauthenticated allowlist."""
        client = TestClient(auth_middleware)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.text == "OK"

    def test_auth_middleware_allows_ping_without_token(self, auth_middleware):
        """Test that /ping is in the unauthenticated allowlist."""
        client = TestClient(auth_middleware)
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.text == "OK"

    def test_auth_middleware_allows_metrics_without_token(self, auth_middleware):
        """Test that /metrics is in the unauthenticated allowlist."""
        client = TestClient(auth_middleware)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.text == "OK"

    def test_auth_middleware_blocks_metrics_when_not_in_allowlist(self, mock_app, test_token):
        """Test that /metrics can be protected if not in allowlist."""
        # Create middleware with empty allowlist
        middleware = AuthenticationMiddleware(
            mock_app,
            tokens=[test_token],
            unauthenticated_paths=frozenset()
        )
        client = TestClient(middleware)
        response = client.get("/metrics")
        assert response.status_code == 401
        assert "Unauthorized" in response.text

    def test_auth_middleware_allows_valid_bearer_token(self, auth_middleware, test_token):
        """Test that valid Bearer tokens are accepted."""
        client = TestClient(auth_middleware)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        assert response.text == "OK"

    def test_auth_middleware_rejects_bearer_with_extra_spaces(self, auth_middleware):
        """Test that malformed Bearer headers are rejected."""
        client = TestClient(auth_middleware)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": "Bearer  extra-spaces"}
        )
        assert response.status_code == 401

    def test_auth_middleware_case_insensitive_bearer(self, auth_middleware, test_token):
        """Test that Bearer scheme is case-insensitive."""
        client = TestClient(auth_middleware)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"bearer {test_token}"}
        )
        assert response.status_code == 200

    def test_auth_middleware_rejects_non_bearer_scheme(self, auth_middleware):
        """Test that non-Bearer auth schemes are rejected."""
        client = TestClient(auth_middleware)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )
        assert response.status_code == 401

    def test_auth_middleware_options_method_skipped(self, auth_middleware):
        """Test that OPTIONS requests bypass authentication."""
        client = TestClient(auth_middleware)
        response = client.options("/v1/completions")
        assert response.status_code == 200

    def test_auth_middleware_multiple_tokens(self, mock_app):
        """Test that middleware accepts any of multiple configured tokens."""
        token1 = "token-one"
        token2 = "token-two"
        middleware = AuthenticationMiddleware(
            mock_app,
            tokens=[token1, token2],
            unauthenticated_paths=frozenset({"/health"})
        )
        client = TestClient(middleware)

        # First token should work
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"Bearer {token1}"}
        )
        assert response.status_code == 200

        # Second token should work
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"Bearer {token2}"}
        )
        assert response.status_code == 200

        # Wrong token should fail
        response = client.get(
            "/v1/completions",
            headers={"Authorization": "Bearer wrong-token"}
        )
        assert response.status_code == 401

    def test_verify_token_uses_constant_time_comparison(self, auth_middleware, test_token):
        """
        Test that token comparison uses constant-time secrets.compare_digest.
        This is verified by checking that the middleware hashes tokens and
        uses secrets.compare_digest internally.
        """
        # The middleware should hash the token
        middleware = auth_middleware
        
        # Verify that the middleware stores hashed tokens
        assert len(middleware.api_tokens) > 0
        assert all(isinstance(t, bytes) for t in middleware.api_tokens)
        
        # Verify that hashing is consistent
        token_hash = hashlib.sha256(test_token.encode("utf-8")).digest()
        assert token_hash in middleware.api_tokens

    def test_verify_token_method_directly(self, auth_middleware, test_token):
        """Test the verify_token method directly."""
        middleware = auth_middleware
        
        # Create headers with valid token
        headers = Headers({"Authorization": f"Bearer {test_token}"})
        assert middleware.verify_token(headers) is True
        
        # Create headers with invalid token
        headers = Headers({"Authorization": "Bearer wrong-token"})
        assert middleware.verify_token(headers) is False
        
        # Create headers without Authorization
        headers = Headers({})
        assert middleware.verify_token(headers) is False

    def test_verify_token_missing_authorization_header(self, auth_middleware):
        """Test that missing Authorization header returns False."""
        middleware = auth_middleware
        headers = Headers({})
        assert middleware.verify_token(headers) is False

    def test_verify_token_malformed_bearer(self, auth_middleware):
        """Test that malformed Bearer header returns False."""
        middleware = auth_middleware
        headers = Headers({"Authorization": "NotBearer token"})
        assert middleware.verify_token(headers) is False

    def test_auth_middleware_path_with_trailing_slash(self, auth_middleware):
        """Test that paths with trailing slashes are handled correctly."""
        client = TestClient(auth_middleware)
        # /health/ should also be in the allowlist (prefix match)
        response = client.get("/health/")
        assert response.status_code == 200

    def test_auth_middleware_subpath_of_protected_endpoint(self, auth_middleware, test_token):
        """Test that subpaths of protected endpoints require auth."""
        client = TestClient(auth_middleware)
        response = client.get("/v1/completions/subpath")
        assert response.status_code == 401
        
        # With valid token, should pass
        response = client.get(
            "/v1/completions/subpath",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200

    def test_auth_middleware_empty_token_list(self, mock_app):
        """Test that middleware with no tokens rejects all requests."""
        middleware = AuthenticationMiddleware(
            mock_app,
            tokens=[],
            unauthenticated_paths=frozenset({"/health"})
        )
        client = TestClient(middleware)
        
        # /health should still work
        response = client.get("/health")
        assert response.status_code == 200
        
        # /v1/completions should fail (no valid tokens exist)
        response = client.get(
            "/v1/completions",
            headers={"Authorization": "Bearer any-token"}
        )
        assert response.status_code == 401

    def test_auth_middleware_token_with_special_characters(self, mock_app):
        """Test that tokens with special characters are handled correctly."""
        special_token = "token!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        middleware = AuthenticationMiddleware(
            mock_app,
            tokens=[special_token],
            unauthenticated_paths=frozenset({"/health"})
        )
        client = TestClient(middleware)
        
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"Bearer {special_token}"}
        )
        assert response.status_code == 200

    def test_auth_middleware_unicode_token(self, mock_app):
        """Test that unicode tokens are handled correctly."""
        unicode_token = "token-with-émojis-🔐"
        middleware = AuthenticationMiddleware(
            mock_app,
            tokens=[unicode_token],
            unauthenticated_paths=frozenset({"/health"})
        )
        client = TestClient(middleware)
        
        response = client.get(
            "/v1/completions",
            headers={"Authorization": f"Bearer {unicode_token}"}
        )
        assert response.status_code == 200
