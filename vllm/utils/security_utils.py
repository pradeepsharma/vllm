# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Security utilities for vLLM.

This module provides shared security-related helper functions that can be
imported from both config and entrypoints modules without circular dependencies.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def emit_security_warning(msg: str) -> None:
    """Emit a security-related warning with a [SECURITY] prefix.
    
    This helper function logs security-relevant warnings at WARNING level
    with a consistent [SECURITY] prefix for easy identification in logs.
    Used for startup checks and configuration validation.
    
    Args:
        msg: The security warning message to log.
    """
    logger.warning("[SECURITY] %s", msg)


def is_localhost(host: str | None) -> bool:
    """Check if a host is localhost or None.
    
    Returns True for:
    - None (unspecified host)
    - "127.0.0.1" (IPv4 loopback)
    - "::1" (IPv6 loopback)
    - "localhost" (hostname)
    
    Used to determine if SSL/TLS and authentication checks should be enforced.
    
    Args:
        host: The hostname or IP address to check.
        
    Returns:
        True if the host is localhost or None, False otherwise.
    """
    if host is None:
        return True
    
    host_lower = host.lower()
    return host_lower in ("127.0.0.1", "::1", "localhost")
