# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-Factor Authentication (MFA) ASGI middleware for vLLM.

This module provides a pure-ASGI middleware that enforces TOTP-based MFA
on authenticated API requests. It is designed to be stacked after the
AuthenticationMiddleware in the middleware chain.
"""

from collections.abc import Awaitable
from typing import Optional

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.auth.mfa import MFAManager
from vllm.logger import init_logger

logger = init_logger(__name__)


class MFAMiddleware:
    """Pure ASGI middleware that enforces TOTP-based MFA on API requests.

    This middleware checks for a valid TOTP code in the X-MFA-Code header
    for requests to protected endpoints when MFA is enabled. It should be
    inserted after AuthenticationMiddleware in the middleware stack.

    The middleware exempts certain paths from MFA checks (health checks,
    metrics, and the MFA provisioning endpoint) to allow monitoring and
    initial setup without requiring an MFA code.

    Attributes:
        app: The wrapped ASGI application.
        mfa_manager: The MFAManager instance for TOTP verification.
            If None, MFA is effectively disabled.
        mfa_header: The name of the header containing the TOTP code.
            Defaults to "X-MFA-Code".
        unauthenticated_paths: A frozenset of URL paths that are exempt
            from MFA checks. Defaults to health/metrics endpoints and
            the MFA provisioning endpoint.
    """

    def __init__(
        self,
        app: ASGIApp,
        mfa_manager: Optional[MFAManager] = None,
        mfa_header: str = "X-MFA-Code",
        unauthenticated_paths: Optional[frozenset[str]] = None,
    ) -> None:
        """Initialize the MFA middleware.

        Args:
            app: The ASGI application to wrap.
            mfa_manager: The MFAManager instance for TOTP verification.
                If None, MFA is disabled and all requests pass through.
            mfa_header: The name of the header containing the TOTP code.
                Defaults to "X-MFA-Code".
            unauthenticated_paths: A frozenset of URL paths exempt from MFA.
                If None, defaults to {"/health", "/ping", "/metrics",
                "/v1/mfa/provision"}.
        """
        self.app = app
        self.mfa_manager = mfa_manager
        self.mfa_header = mfa_header

        if unauthenticated_paths is None:
            self.unauthenticated_paths = frozenset(
                {"/health", "/ping", "/metrics", "/v1/mfa/provision"}
            )
        else:
            self.unauthenticated_paths = unauthenticated_paths

    def _is_mfa_exempt(self, url_path: str) -> bool:
        """Check if a URL path is exempt from MFA checks.

        Uses prefix matching to allow for path parameters and query strings.
        For example, "/health" matches "/health", "/health/", and
        "/health?status=ok".

        Args:
            url_path: The URL path to check (without query string).

        Returns:
            True if the path is exempt from MFA checks, False otherwise.
        """
        for exempt_path in self.unauthenticated_paths:
            if url_path.startswith(exempt_path):
                return True
        return False

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        """ASGI middleware entry point.

        Processes HTTP and WebSocket requests. For HTTP requests to protected
        endpoints with MFA enabled, verifies the TOTP code in the X-MFA-Code
        header before forwarding to the next middleware/application.

        Args:
            scope: The ASGI scope dictionary.
            receive: The ASGI receive callable.
            send: The ASGI send callable.

        Returns:
            An awaitable that processes the request.
        """
        # Only process HTTP and WebSocket requests
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        # Skip MFA check for OPTIONS requests (CORS preflight)
        if scope["method"] == "OPTIONS":
            return self.app(scope, receive, send)

        # If MFA is not enabled, pass through
        if self.mfa_manager is None:
            return self.app(scope, receive, send)

        # Extract the URL path
        from starlette.datastructures import URL

        root_path = scope.get("root_path", "")
        url_path = URL(scope=scope).path.removeprefix(root_path)

        # Check if this path is exempt from MFA
        if self._is_mfa_exempt(url_path):
            logger.debug("MFA check skipped for exempt path: %s", url_path)
            return self.app(scope, receive, send)

        # Extract headers
        headers = Headers(scope=scope)

        # Get the MFA code from the header
        mfa_code = headers.get(self.mfa_header)

        if not mfa_code:
            logger.debug(
                "MFA code missing in header '%s' for path: %s",
                self.mfa_header,
                url_path,
            )
            response = JSONResponse(
                content={"error": "MFA code required"},
                status_code=401,
            )
            return response(scope, receive, send)

        # Verify the TOTP code
        logger.debug("Verifying MFA code for path: %s", url_path)
        if not self.mfa_manager.verify_totp(mfa_code):
            logger.debug("MFA code verification failed for path: %s", url_path)
            response = JSONResponse(
                content={"error": "Invalid or expired MFA code"},
                status_code=401,
            )
            return response(scope, receive, send)

        logger.debug("MFA code verified successfully for path: %s", url_path)
        return self.app(scope, receive, send)
