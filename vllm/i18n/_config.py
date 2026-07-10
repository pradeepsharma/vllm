# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration and locale resolution for vLLM's internationalization (I18N) system."""

import gettext
import locale
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class I18NConfig:
    """Configuration for the vLLM I18N system.
    
    Attributes:
        locale: The locale code (e.g., 'en', 'zh_CN', 'de', 'fr', 'ja').
        domain: The translation domain name (default: 'vllm').
        localedir: Path to the directory containing locale catalogs.
    """

    locale: str
    domain: str = "vllm"
    localedir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Initialize localedir if not provided."""
        if self.localedir is None:
            # Default to the locales directory within the i18n package
            self.localedir = Path(__file__).parent / "locales"


def _resolve_locale() -> str:
    """Resolve the locale to use for translation.
    
    Locale resolution order:
    1. VLLM_LOCALE environment variable (if set)
    2. LC_ALL environment variable (if set)
    3. LANG environment variable (if set)
    4. System locale (via locale.getdefaultlocale())
    5. Fallback to 'en' (English)
    
    Returns:
        A locale code string (e.g., 'en', 'zh_CN', 'de', 'fr', 'ja').
    """
    # Check VLLM_LOCALE first (highest priority for operator override)
    vllm_locale = os.environ.get("VLLM_LOCALE")
    if vllm_locale:
        return vllm_locale

    # Check LC_ALL
    lc_all = os.environ.get("LC_ALL")
    if lc_all:
        # Extract language code from locale string (e.g., 'en_US.UTF-8' -> 'en_US')
        locale_code = lc_all.split(".")[0]
        if locale_code:
            return locale_code

    # Check LANG
    lang = os.environ.get("LANG")
    if lang:
        # Extract language code from locale string (e.g., 'en_US.UTF-8' -> 'en_US')
        locale_code = lang.split(".")[0]
        if locale_code:
            return locale_code

    # Try system locale
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            return system_locale
    except Exception:
        # If getdefaultlocale() fails, continue to fallback
        pass

    # Fallback to English
    return "en"


def get_translator(
    locale_code: Optional[str] = None,
    domain: str = "vllm",
    localedir: Optional[Path] = None,
) -> gettext.GNUTranslations:
    """Get a GNUTranslations object for the specified locale.
    
    Args:
        locale_code: The locale code to use. If None, uses _resolve_locale().
        domain: The translation domain name (default: 'vllm').
        localedir: Path to the directory containing locale catalogs.
                  If None, uses the default locales directory.
    
    Returns:
        A gettext.GNUTranslations object for the specified locale.
        If the locale is 'en' or the catalog is not found, returns
        a NullTranslations object (identity translation).
    """
    if locale_code is None:
        locale_code = _resolve_locale()

    if localedir is None:
        localedir = Path(__file__).parent / "locales"

    # For English, return a NullTranslations (no-op translation)
    if locale_code == "en" or locale_code.startswith("en_"):
        return gettext.NullTranslations()

    try:
        # Try to load the .mo file for the specified locale
        translator = gettext.translation(
            domain=domain,
            localedir=str(localedir),
            languages=[locale_code],
            fallback=True,
        )
        return translator
    except Exception:
        # If loading fails, return NullTranslations (identity translation)
        return gettext.NullTranslations()
