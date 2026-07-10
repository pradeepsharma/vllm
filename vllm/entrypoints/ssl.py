# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import ssl
from collections.abc import Callable
from ssl import SSLContext

from watchfiles import Change, awatch

from vllm.logger import init_logger

logger = init_logger(__name__)


def validate_ssl_config(
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    ssl_cert_reqs: int,
    ssl_ciphers: str | None = None,
) -> None:
    """Validate SSL/TLS configuration and emit security warnings.
    
    This function checks for common SSL/TLS misconfigurations and emits
    warnings for security issues such as:
    - Client certificate verification disabled (CERT_NONE)
    - Weak or unrestricted cipher suites
    
    Args:
        ssl_keyfile: Path to SSL key file, or None if not configured
        ssl_certfile: Path to SSL certificate file, or None if not configured
        ssl_cert_reqs: SSL certificate requirement level (ssl.CERT_NONE, etc.)
        ssl_ciphers: Custom cipher suite string, or None for defaults
        
    Raises:
        ValueError: If SSL configuration is invalid
    """
    from vllm.entrypoints.utils import emit_security_warning
    
    # Check if client certificate verification is disabled
    if ssl_cert_reqs == ssl.CERT_NONE:
        emit_security_warning(
            "Client certificate verification is disabled (ssl_cert_reqs=CERT_NONE). "
            "This means the server will not verify client certificates. "
            "For production deployments with mutual TLS, set --ssl-cert-reqs to "
            "ssl.CERT_REQUIRED (2) or ssl.CERT_OPTIONAL (1)."
        )
    
    # Check for weak cipher suites
    if ssl_ciphers:
        weak_patterns = ["ALL", "DEFAULT", "RC4", "DES", "3DES", "EXPORT", "NULL", "aNULL", "eNULL"]
        ciphers_upper = ssl_ciphers.upper()
        
        # Check if any weak patterns are used
        for pattern in weak_patterns:
            if pattern in ciphers_upper:
                emit_security_warning(
                    f"Weak cipher suite pattern '{pattern}' detected in ssl_ciphers. "
                    f"This may expose the connection to cryptographic attacks. "
                    f"Recommended cipher string: "
                    f"'ECDH+AESGCM:ECDH+CHACHA20:!aNULL:!MD5:!DSS'. "
                    f"Current value: {ssl_ciphers}"
                )
                break


def create_ssl_context(
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    ssl_ca_certs: str | None = None,
    ssl_cert_reqs: int = ssl.CERT_NONE,
    ssl_ciphers: str | None = None,
) -> SSLContext | None:
    """Create a secure SSL context for HTTPS server.
    
    Creates an SSL context using modern TLS protocols and secure defaults.
    Uses ssl.PROTOCOL_TLS_SERVER which automatically selects the highest
    protocol version supported by both client and server.
    
    Args:
        ssl_keyfile: Path to SSL private key file
        ssl_certfile: Path to SSL certificate file
        ssl_ca_certs: Path to CA certificates file for client verification
        ssl_cert_reqs: Certificate requirement level (ssl.CERT_NONE, etc.)
        ssl_ciphers: Custom cipher suite string, or None for secure defaults
        
    Returns:
        SSLContext configured for secure HTTPS, or None if SSL is not configured
        
    Raises:
        FileNotFoundError: If certificate files do not exist
        ssl.SSLError: If SSL context creation fails
    """
    if not ssl_keyfile or not ssl_certfile:
        return None
    
    # Validate SSL configuration for security issues
    validate_ssl_config(ssl_keyfile, ssl_certfile, ssl_cert_reqs, ssl_ciphers)
    
    # Create SSL context using PROTOCOL_TLS_SERVER (modern, auto-negotiates best protocol)
    # This is preferred over deprecated PROTOCOL_TLSv1_2 or PROTOCOL_TLS
    context = SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Load certificate chain
    context.load_cert_chain(ssl_certfile, ssl_keyfile)
    
    # Load CA certificates if provided
    if ssl_ca_certs:
        context.load_verify_locations(ssl_ca_certs)
    
    # Set certificate requirement level
    context.verify_mode = ssl_cert_reqs
    
    # Set cipher suites
    if ssl_ciphers:
        try:
            context.set_ciphers(ssl_ciphers)
        except ssl.SSLError as e:
            raise ssl.SSLError(
                f"Failed to set SSL ciphers '{ssl_ciphers}': {e}"
            ) from e
    else:
        # Use secure default ciphers that exclude weak algorithms
        # This excludes: RC4, DES, 3DES, EXPORT, NULL, aNULL, eNULL, MD5, DSS
        default_ciphers = "ECDH+AESGCM:ECDH+CHACHA20:!aNULL:!MD5:!DSS"
        try:
            context.set_ciphers(default_ciphers)
        except ssl.SSLError:
            # If default ciphers fail, fall back to system defaults
            # (this should rarely happen on modern systems)
            logger.warning(
                "Failed to set secure default ciphers, using system defaults. "
                "Consider explicitly setting --ssl-ciphers to a secure cipher suite."
            )
    
    return context


class SSLCertRefresher:
    """A class that monitors SSL certificate files and
    reloads them when they change.
    """

    def __init__(
        self,
        ssl_context: SSLContext,
        key_path: str | None = None,
        cert_path: str | None = None,
        ca_path: str | None = None,
    ) -> None:
        self.ssl = ssl_context
        self.key_path = key_path
        self.cert_path = cert_path
        self.ca_path = ca_path

        # Setup certification chain watcher
        def update_ssl_cert_chain(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL certificate chain")
            assert self.key_path and self.cert_path
            self.ssl.load_cert_chain(self.cert_path, self.key_path)

        self.watch_ssl_cert_task = None
        if self.key_path and self.cert_path:
            self.watch_ssl_cert_task = asyncio.create_task(
                self._watch_files(
                    [self.key_path, self.cert_path], update_ssl_cert_chain
                )
            )

        # Setup CA files watcher
        def update_ssl_ca(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL CA certificates")
            assert self.ca_path
            self.ssl.load_verify_locations(self.ca_path)

        self.watch_ssl_ca_task = None
        if self.ca_path:
            self.watch_ssl_ca_task = asyncio.create_task(
                self._watch_files([self.ca_path], update_ssl_ca)
            )

    async def _watch_files(self, paths, fun: Callable[[Change, str], None]) -> None:
        """Watch multiple file paths asynchronously."""
        logger.info("SSLCertRefresher monitors files: %s", paths)
        async for changes in awatch(*paths):
            try:
                for change, file_path in changes:
                    logger.info("File change detected: %s - %s", change.name, file_path)
                    fun(change, file_path)
            except Exception as e:
                logger.error(
                    "SSLCertRefresher failed taking action on file change. Error: %s", e
                )

    def stop(self) -> None:
        """Stop watching files."""
        if self.watch_ssl_cert_task:
            self.watch_ssl_cert_task.cancel()
            self.watch_ssl_cert_task = None
        if self.watch_ssl_ca_task:
            self.watch_ssl_ca_task.cancel()
            self.watch_ssl_ca_task = None
