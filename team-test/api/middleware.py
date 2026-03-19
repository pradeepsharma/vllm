"""
Authentication middleware for the team-test REST API.

Design mirrors ``vllm/entrypoints/openai/server_utils.py::AuthenticationMiddleware``:

- Pure ASGI middleware (no ``starlette.middleware.base.BaseHTTPMiddleware`` overhead).
- Validates ``Authorization: Bearer <token>`` on every protected request.
- Skips authentication for:
    - HTTP ``OPTIONS`` requests (CORS pre-flight).
    - Paths listed in ``_PUBLIC_PATHS``: ``/health``, ``/docs``,
      ``/openapi.json``, ``/redoc``.
- Token(s) are read from the ``API_TOKEN`` environment variable.
  Multiple tokens may be provided as a comma-separated list
  (e.g. ``API_TOKEN=token1,token2``).
  Falls back to ``"dev-token"`` when the variable is not set.
- Token comparison uses SHA-256 digests + ``secrets.compare_digest`` to
  prevent timing-based side-channel attacks.
- Returns HTTP 401 JSON ``{"detail": "Unauthorized — valid Bearer token required"}``
  when authentication fails.

Usage
-----
::

    from fastapi import FastAPI
    from team_test.api.middleware import BearerAuthMiddleware

    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware)

    # Or, to supply tokens programmatically instead of via env var:
    app.add_middleware(BearerAuthMiddleware, tokens=["my-secret-token"])
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

# Paths that do not require a Bearer token
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)

# Fallback token used when API_TOKEN env var is not set
_DEFAULT_TOKEN: str = "dev-token"


class BearerAuthMiddleware:
    """Pure ASGI middleware that validates a Bearer token on every protected request.

    Parameters
    ----------
    app:
        The next ASGI application in the middleware stack.
    tokens:
        Optional explicit list of accepted tokens.  When omitted (the normal
        case), tokens are read from the ``API_TOKEN`` environment variable.
        Comma-separated values in ``API_TOKEN`` are each treated as a
        separate valid token.

    Authentication flow
    -------------------
    1. Non-HTTP/WebSocket scopes (e.g. ``lifespan``) pass through unchanged.
    2. ``OPTIONS`` requests pass through (CORS pre-flight).
    3. Requests whose ``path`` is in ``_PUBLIC_PATHS`` pass through.
    4. All other requests must supply ``Authorization: Bearer <token>`` where
       ``<token>`` matches one of the configured tokens.  A mismatch returns
       HTTP 401.

    Security notes
    --------------
    - Tokens are stored as SHA-256 digests; the incoming token is also hashed
      before comparison so the plain-text value is never held in memory longer
      than necessary.
    - ``secrets.compare_digest`` is used for constant-time comparison to
      prevent timing attacks.
    """

    def __init__(self, app: ASGIApp, tokens: list[str] | None = None) -> None:
        self.app = app

        if tokens is not None:
            resolved_tokens = [t.strip() for t in tokens if t.strip()]
        else:
            raw = os.getenv("API_TOKEN", _DEFAULT_TOKEN)
            resolved_tokens = [t.strip() for t in raw.split(",") if t.strip()]

        # Fall back to the default token if the list is empty after filtering
        if not resolved_tokens:
            resolved_tokens = [_DEFAULT_TOKEN]

        # Store SHA-256 digests so plain-text tokens are not kept in memory
        self._token_hashes: list[bytes] = [
            hashlib.sha256(t.encode("utf-8")).digest() for t in resolved_tokens
        ]

    def verify_token(self, headers: Headers) -> bool:
        """Return ``True`` if the request carries a valid Bearer token.

        Parameters
        ----------
        headers:
            The request headers from the ASGI scope.

        Returns
        -------
        bool
            ``True`` when a valid token is present, ``False`` otherwise.
        """
        authorization = headers.get("Authorization", "")
        if not authorization:
            return False

        scheme, _, param = authorization.partition(" ")
        if scheme.lower() != "bearer" or not param:
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for stored_hash in self._token_hashes:
            token_match |= secrets.compare_digest(param_hash, stored_hash)

        return token_match

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        # Pass non-HTTP scopes (e.g. lifespan) straight through
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = scope.get("method", "")
        path: str = scope.get("path", "")

        # Allow CORS pre-flight and public paths without authentication
        if method == "OPTIONS" or path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        if not self.verify_token(headers):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized — valid Bearer token required"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
