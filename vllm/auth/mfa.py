# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-Factor Authentication (MFA) support using TOTP (Time-based One-Time Password).

This module provides TOTP-based MFA functionality using the pyotp library.
It handles secret generation, code verification, and provisioning URI generation
for authenticator apps.
"""

import base64
import io
from typing import Optional

import pyotp

from vllm.logger import init_logger

logger = init_logger(__name__)


class MFAManager:
    """Manages TOTP-based Multi-Factor Authentication.

    This class encapsulates all TOTP logic including secret generation,
    code verification, and provisioning URI construction for authenticator apps.

    Attributes:
        secret: The base32-encoded TOTP secret.
        issuer: The issuer name displayed in authenticator apps.
        account_name: The account name displayed in authenticator apps.
        totp: The pyotp.TOTP instance for verification.
    """

    def __init__(
        self,
        secret: Optional[str] = None,
        issuer: str = "vLLM",
        account_name: str = "vllm-api",
    ) -> None:
        """Initialize the MFA manager.

        Args:
            secret: The base32-encoded TOTP secret. If None, a new secret
                is generated using generate_secret().
            issuer: The issuer name to display in authenticator apps.
                Defaults to "vLLM".
            account_name: The account name to display in authenticator apps.
                Defaults to "vllm-api".
        """
        self.issuer = issuer
        self.account_name = account_name

        if secret is None:
            self.secret = self.generate_secret()
        else:
            self.secret = secret

        self.totp = pyotp.TOTP(self.secret)

    @staticmethod
    def generate_secret() -> str:
        """Generate a new random TOTP secret.

        Returns:
            A 32-character base32-encoded string suitable for use as a TOTP secret.
        """
        return pyotp.random_base32()

    def verify_totp(self, code: str, valid_window: int = 1) -> bool:
        """Verify a TOTP code.

        Verifies the provided code against the current TOTP secret using
        constant-time comparison to prevent timing attacks. Returns False
        for non-digit or wrong-length codes without raising exceptions.

        Args:
            code: The TOTP code to verify (typically 6 digits).
            valid_window: The number of 30-second windows to check before
                and after the current time. Defaults to 1 (±30 seconds).

        Returns:
            True if the code is valid, False otherwise.
        """
        # Validate input: code should be digits only
        if not isinstance(code, str):
            return False

        if not code.isdigit():
            return False

        # Standard TOTP codes are 6 digits, but we allow some flexibility
        # Most authenticators produce 6-digit codes
        if len(code) < 6 or len(code) > 8:
            return False

        try:
            return self.totp.verify(code, valid_window=valid_window)
        except Exception:
            # Catch any unexpected exceptions and return False
            return False

    def get_provisioning_uri(self) -> str:
        """Get the provisioning URI for authenticator app enrollment.

        Returns the otpauth:// URI that can be converted to a QR code
        for scanning into authenticator apps like Google Authenticator,
        Authy, Microsoft Authenticator, etc.

        Returns:
            The provisioning URI string.
        """
        return self.totp.provisioning_uri(
            name=self.account_name, issuer_name=self.issuer
        )

    def get_qr_data(self) -> Optional[str]:
        """Generate a base64-encoded PNG QR code for the provisioning URI.

        Generates a QR code image from the provisioning URI. This is useful
        for displaying in web interfaces or logs for easy scanning by
        authenticator apps.

        Returns:
            A base64-encoded PNG image string if qrcode is installed,
            None otherwise. A warning is logged if qrcode is not available.
        """
        try:
            import qrcode
        except ImportError:
            logger.warning(
                "qrcode library not installed. QR code generation disabled. "
                "Install with: pip install 'qrcode[pil]>=7.4'"
            )
            return None

        try:
            # Generate QR code from the provisioning URI
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(self.get_provisioning_uri())
            qr.make(fit=True)

            # Create PIL image
            img = qr.make_image(fill_color="black", back_color="white")

            # Convert to PNG bytes
            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
            png_bytes = png_buffer.getvalue()

            # Encode as base64
            qr_data = base64.b64encode(png_bytes).decode("utf-8")
            return qr_data
        except Exception as e:
            logger.warning(
                "Failed to generate QR code: %s. "
                "Provisioning URI will still be available.",
                str(e),
            )
            return None
