# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM Internationalization (I18N) Module.

This module provides translation support for vLLM's user-facing strings,
including log messages, error responses, and CLI help text.

The module uses Python's standard `gettext` library for runtime translation
and provides a lazy translation mechanism for strings that need to be marked
for translation at module import time (e.g., in class docstrings and argparse
help text).

Locale Resolution:
    The locale is resolved in the following order:
    1. VLLM_LOCALE environment variable (if set)
    2. LC_ALL environment variable (if set)
    3. LANG environment variable (if set)
    4. System locale (via locale.getdefaultlocale())
    5. Fallback to 'en' (English)

Usage:
    >>> from vllm.i18n import _
    >>> message = _("Hello, World!")
    >>> print(message)
    Hello, World!
    
    >>> from vllm.i18n import lazy_
    >>> help_text = lazy_("This is help text")
    >>> # Translation is deferred until the string is used
    >>> print(help_text)
    This is help text
    
    >>> from vllm.i18n import ngettext
    >>> message = ngettext("1 file", "{} files", 5).format(5)
    >>> print(message)
    5 files
"""

import gettext as _gettext_module
from pathlib import Path
from typing import Optional

from vllm.i18n._config import I18NConfig, _resolve_locale, get_translator
from vllm.i18n._lazy import LazyString

__all__ = [
    "gettext",
    "ngettext",
    "lazy_gettext",
    "setup_i18n",
    "LazyString",
    "I18NConfig",
    "_",
    "lazy_",
]

# Global translator instance
_translator: Optional[_gettext_module.GNUTranslations] = None
_current_locale: Optional[str] = None


def _initialize_translator(locale_code: Optional[str] = None) -> None:
    """Initialize the global translator instance.
    
    Args:
        locale_code: The locale code to use. If None, uses _resolve_locale().
    """
    global _translator, _current_locale

    if locale_code is None:
        locale_code = _resolve_locale()

    _current_locale = locale_code
    localedir = Path(__file__).parent / "locales"
    _translator = get_translator(locale_code, domain="vllm", localedir=localedir)


def gettext(message: str) -> str:
    """Translate a message string.
    
    This is the main translation function. It looks up the message in the
    current locale's translation catalog and returns the translated string.
    If the message is not found or the translator is not initialized, the
    original message is returned.
    
    Args:
        message: The message string to translate.
    
    Returns:
        The translated message string, or the original message if no
        translation is found.
    
    Example:
        >>> from vllm.i18n import gettext as _
        >>> message = _("Hello, World!")
        >>> print(message)
        Hello, World!
    """
    global _translator

    if _translator is None:
        _initialize_translator()

    if _translator is None:
        return message

    return _translator.gettext(message)


def ngettext(singular: str, plural: str, n: int) -> str:
    """Translate a message with plural forms.
    
    This function handles plural forms in translations. It selects the
    appropriate form based on the count `n` and returns the translated string.
    
    Args:
        singular: The singular form of the message.
        plural: The plural form of the message.
        n: The count to determine which form to use.
    
    Returns:
        The translated message string in the appropriate form.
    
    Example:
        >>> from vllm.i18n import ngettext
        >>> message = ngettext("1 file", "{} files", 5).format(5)
        >>> print(message)
        5 files
    """
    global _translator

    if _translator is None:
        _initialize_translator()

    if _translator is None:
        return singular if n == 1 else plural

    return _translator.ngettext(singular, plural, n)


def lazy_gettext(message: str) -> LazyString:
    """Create a lazy-evaluated translation string.
    
    This function creates a LazyString object that defers translation until
    the string is actually used. This is useful for marking strings for
    translation at module import time (e.g., in class docstrings, argparse
    help text) without requiring the translator to be initialized yet.
    
    Args:
        message: The message string to translate (lazily).
    
    Returns:
        A LazyString object that will be translated when converted to a string.
    
    Example:
        >>> from vllm.i18n import lazy_gettext as lazy_
        >>> help_text = lazy_("This is help text")
        >>> # Translation is deferred until the string is used
        >>> print(help_text)
        This is help text
    """
    return LazyString(message)


def setup_i18n(locale_code: Optional[str] = None) -> None:
    """Set up the I18N system with a specific locale.
    
    This function initializes the translator with the specified locale.
    If no locale is provided, it uses the default locale resolution logic.
    
    This function is typically called once at application startup, but can
    be called multiple times to switch locales (e.g., for multi-tenant
    deployments).
    
    Args:
        locale_code: The locale code to use (e.g., 'en', 'zh_CN', 'de', 'fr', 'ja').
                    If None, uses _resolve_locale().
    
    Example:
        >>> from vllm.i18n import setup_i18n
        >>> setup_i18n('zh_CN')
    """
    _initialize_translator(locale_code)


def get_current_locale() -> str:
    """Get the currently active locale.
    
    Returns:
        The locale code of the currently active translator.
    """
    global _current_locale

    if _current_locale is None:
        _initialize_translator()

    return _current_locale or "en"


# Convenience aliases for common usage patterns
_ = gettext
lazy_ = lazy_gettext

# Initialize the translator on module import
_initialize_translator()
