# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lazy string translation for vLLM's I18N system.

This module provides the LazyString class, which allows strings to be marked
for translation at module import time without requiring the translator to be
initialized yet. The actual translation is deferred until the string is used.
"""

from typing import Any, Optional


class LazyString:
    """A lazy-evaluated translation string.
    
    This class allows strings to be wrapped with _() at module import time
    (e.g., in class docstrings, argparse help text) without requiring the
    translator to be initialized yet. The actual translation is deferred
    until the string is converted to a string or formatted.
    
    Attributes:
        _msgid: The message ID (untranslated string).
        _args: Optional positional arguments for string formatting.
        _kwargs: Optional keyword arguments for string formatting.
    """

    __slots__ = ("_msgid", "_args", "_kwargs")

    def __init__(
        self,
        msgid: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a LazyString.
        
        Args:
            msgid: The message ID (untranslated string).
            *args: Positional arguments for string formatting (optional).
            **kwargs: Keyword arguments for string formatting (optional).
        """
        self._msgid = msgid
        self._args = args
        self._kwargs = kwargs

    def __str__(self) -> str:
        """Convert the lazy string to a regular string.
        
        This method triggers the actual translation lookup and formatting.
        
        Returns:
            The translated and formatted string.
        """
        # Import here to avoid circular imports
        # We import the module directly to avoid triggering the full vllm package init
        import sys
        if 'vllm.i18n' in sys.modules:
            from vllm.i18n import gettext as _gettext
            translated = _gettext(self._msgid)
        else:
            # If vllm.i18n is not yet initialized, return the msgid as-is
            translated = self._msgid

        # Apply formatting if arguments were provided
        if self._args or self._kwargs:
            try:
                return translated.format(*self._args, **self._kwargs)
            except (IndexError, KeyError):
                # If formatting fails, return the translated string as-is
                return translated
        return translated

    def __repr__(self) -> str:
        """Return a representation of the lazy string.
        
        Returns:
            A string representation showing the message ID and any arguments.
        """
        if self._args or self._kwargs:
            args_str = ", ".join(repr(arg) for arg in self._args)
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            return f"LazyString({self._msgid!r}, {all_args})"
        return f"LazyString({self._msgid!r})"

    def __format__(self, format_spec: str) -> str:
        """Format the lazy string using a format specification.
        
        Args:
            format_spec: The format specification string.
        
        Returns:
            The formatted string.
        """
        return format(str(self), format_spec)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another object.
        
        Args:
            other: The object to compare with.
        
        Returns:
            True if the other object is a LazyString with the same msgid,
            args, and kwargs, or if it's a string equal to the translated
            string.
        """
        if isinstance(other, LazyString):
            return (
                self._msgid == other._msgid
                and self._args == other._args
                and self._kwargs == other._kwargs
            )
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        """Return the hash of the lazy string.
        
        Returns:
            The hash of the message ID (args and kwargs are not included
            in the hash to allow for consistent hashing).
        """
        return hash(self._msgid)

    def __bool__(self) -> bool:
        """Check if the lazy string is truthy.
        
        Returns:
            True if the translated string is non-empty.
        """
        return bool(str(self))

    def __len__(self) -> int:
        """Return the length of the translated string.
        
        Returns:
            The length of the translated and formatted string.
        """
        return len(str(self))

    def __add__(self, other: Any) -> str:
        """Concatenate with another string.
        
        Args:
            other: The string to concatenate with.
        
        Returns:
            The concatenated string.
        """
        return str(self) + str(other)

    def __radd__(self, other: Any) -> str:
        """Right-hand concatenation with another string.
        
        Args:
            other: The string to concatenate with.
        
        Returns:
            The concatenated string.
        """
        return str(other) + str(self)

    def __mul__(self, other: int) -> str:
        """Repeat the lazy string.
        
        Args:
            other: The number of times to repeat.
        
        Returns:
            The repeated string.
        """
        return str(self) * other

    def __rmul__(self, other: int) -> str:
        """Right-hand repeat of the lazy string.
        
        Args:
            other: The number of times to repeat.
        
        Returns:
            The repeated string.
        """
        return other * str(self)

    def __contains__(self, item: Any) -> bool:
        """Check if a substring is contained in the lazy string.
        
        Args:
            item: The substring to check for.
        
        Returns:
            True if the substring is in the translated string.
        """
        return str(item) in str(self)

    def __getitem__(self, key: Any) -> str:
        """Get a character or slice from the lazy string.
        
        Args:
            key: The index or slice.
        
        Returns:
            The character or substring.
        """
        return str(self)[key]
