# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MFA provisioning endpoint for vLLM.

This module provides a FastAPI router for MFA provisioning, allowing clients
to retrieve the provisioning URI and QR code for setting up TOTP-based
multi-factor authentication.
"""

from typing import Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter(prefix="/v1/mfa", tags=["mfa"])


class MFAProvisionResponse(BaseModel):
    """Response model for MFA provisioning endpoint.

    Attributes:
        provisioning_uri: The otpauth:// URI for enrolling in an authenticator app.
            This URI can be converted to a QR code for easy scanning.
        qr_code_data: Optional base64-encoded PNG image of the QR code.
            If qrcode library is not installed or QR generation fails, this will be None.
        secret: The base32-encoded TOTP secret. Provided for manual entry into
            authenticator apps if QR code scanning is not available.
        issuer: The issuer name displayed in authenticator apps.
        account_name: The account name displayed in authenticator apps.
    """

    provisioning_uri: str = Field(
        ..., description="otpauth:// URI for authenticator app enrollment"
    )
    qr_code_data: Optional[str] = Field(
        None, description="Base64-encoded PNG QR code image (optional)"
    )
    secret: str = Field(..., description="Base32-encoded TOTP secret")
    issuer: str = Field(..., description="Issuer name displayed in authenticator apps")
    account_name: str = Field(
        ..., description="Account name displayed in authenticator apps"
    )


@router.get("/provision")
async def get_mfa_provision(request: Request) -> MFAProvisionResponse:
    """Get MFA provisioning information for authenticator app enrollment.

    This endpoint returns the provisioning URI and optional QR code for setting up
    TOTP-based multi-factor authentication. The provisioning URI can be scanned by
    authenticator apps like Google Authenticator, Authy, Microsoft Authenticator, etc.

    The endpoint is protected by API key authentication (via AuthenticationMiddleware).
    MFA code verification is not required for this endpoint.

    Args:
        request: The FastAPI request object.

    Returns:
        MFAProvisionResponse: Contains the provisioning URI, optional QR code,
            secret, issuer, and account name.

    Raises:
        HTTPException: If MFA is not enabled on the server (500 Internal Server Error).
    """
    # Get the MFA manager from app state
    mfa_manager = getattr(request.app.state, "mfa_manager", None)

    if mfa_manager is None:
        logger.error("MFA manager not found in app state")
        return JSONResponse(
            content={"error": "MFA is not enabled on this server"},
            status_code=500,
        )

    # Construct the response with provisioning information
    response = MFAProvisionResponse(
        provisioning_uri=mfa_manager.get_provisioning_uri(),
        qr_code_data=mfa_manager.get_qr_data(),
        secret=mfa_manager.secret,
        issuer=mfa_manager.issuer,
        account_name=mfa_manager.account_name,
    )

    return response


def attach_router(app: FastAPI) -> None:
    """Attach the MFA router to the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    app.include_router(router)
