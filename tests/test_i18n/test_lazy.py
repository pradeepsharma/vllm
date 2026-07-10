# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.i18n._lazy module (LazyString class)."""

import sys
import pytest


def get_lazy_module():
    """Return the vllm.i18n._lazy module."""
    return sys.modules['vllm.i18n._lazy']


def get_i18n_module():
    """Return the vllm.i18n module."""
    return sys.modules['vllm.i18n']


def LazyString(msgid, *args, **kwargs):
    """Convenience factory for LazyString."""
    return get_lazy_module().LazyString(msgid, *args, **kwargs)


# ---------------------------------------------------------------------------
# Basic construction and __str__
# ---------------------------------------------------------------------------

class TestLazyStringBasic:
    """Tests for basic LazyString construction and string conversion."""

    def test_str_returns_msgid_for_english(self):
        """For English locale, str(LazyString) returns the original msgid."""
        ls = LazyString('unimplemented endpoint')
        assert str(ls) == 'unimplemented endpoint'

    def test_str_returns_msgid_for_unknown_string(self):
        """For a string not in any catalog, str() returns the msgid."""
        ls = LazyString('this string is not translated anywhere')
        assert str(ls) == 'this string is not translated anywhere'

    def test_msgid_stored_correctly(self):
        """The _msgid attribute stores the original message ID."""
        ls = LazyString('hello world')
        assert ls._msgid == 'hello world'

    def test_no_args_stored_as_empty_tuple(self):
        """When no args are passed, _args is an empty tuple."""
        ls = LazyString('hello')
        assert ls._args == ()

    def test_no_kwargs_stored_as_empty_dict(self):
        """When no kwargs are passed, _kwargs is an empty dict."""
        ls = LazyString('hello')
        assert ls._kwargs == {}

    def test_args_stored_correctly(self):
        """Positional args are stored in _args."""
        ls = LazyString('hello {}', 'world')
        assert ls._args == ('world',)

    def test_kwargs_stored_correctly(self):
        """Keyword args are stored in _kwargs."""
        ls = LazyString('hello {name}', name='world')
        assert ls._kwargs == {'name': 'world'}

    def test_str_with_positional_args_formats(self):
        """str() applies positional format args."""
        ls = LazyString('hello {}', 'world')
        assert str(ls) == 'hello world'

    def test_str_with_keyword_args_formats(self):
        """str() applies keyword format args."""
        ls = LazyString('hello {name}', name='Alice')
        assert str(ls) == 'hello Alice'

    def test_str_with_bad_format_returns_translated(self):
        """If formatting fails, str() returns the translated string without formatting."""
        ls = LazyString('hello {}')  # no args provided for {}
        # Should not raise; returns the translated string as-is
        result = str(ls)
        assert result == 'hello {}'

    def test_empty_string(self):
        """Empty string msgid works correctly."""
        ls = LazyString('')
        assert str(ls) == ''


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestLazyStringRepr:
    """Tests for LazyString.__repr__."""

    def test_repr_no_args(self):
        """repr() shows just the msgid."""
        ls = LazyString('hello')
        assert repr(ls) == "LazyString('hello')"

    def test_repr_with_positional_args(self):
        """repr() shows msgid and positional args."""
        ls = LazyString('hello {}', 'world')
        assert repr(ls) == "LazyString('hello {}', 'world')"

    def test_repr_with_keyword_args(self):
        """repr() shows msgid and keyword args."""
        ls = LazyString('hello {name}', name='Alice')
        assert repr(ls) == "LazyString('hello {name}', name='Alice')"


# ---------------------------------------------------------------------------
# __format__
# ---------------------------------------------------------------------------

class TestLazyStringFormat:
    """Tests for LazyString.__format__."""

    def test_format_no_spec(self):
        """format(ls, '') returns str(ls)."""
        ls = LazyString('hello')
        assert format(ls, '') == 'hello'

    def test_format_with_spec(self):
        """format(ls, '>10') right-aligns the string."""
        ls = LazyString('hi')
        result = format(ls, '>10')
        assert result == '        hi'
        assert len(result) == 10

    def test_fstring_works(self):
        """f-string interpolation works correctly."""
        ls = LazyString('world')
        assert f'hello {ls}' == 'hello world'


# ---------------------------------------------------------------------------
# __eq__ and __hash__
# ---------------------------------------------------------------------------

class TestLazyStringEquality:
    """Tests for LazyString equality and hashing."""

    def test_equal_to_same_msgid(self):
        """Two LazyStrings with the same msgid are equal."""
        ls1 = LazyString('hello')
        ls2 = LazyString('hello')
        assert ls1 == ls2

    def test_not_equal_to_different_msgid(self):
        """Two LazyStrings with different msgids are not equal."""
        ls1 = LazyString('hello')
        ls2 = LazyString('world')
        assert ls1 != ls2

    def test_equal_to_equivalent_string(self):
        """LazyString equals the string it translates to."""
        ls = LazyString('hello')
        assert ls == 'hello'

    def test_not_equal_to_different_string(self):
        """LazyString is not equal to a different string."""
        ls = LazyString('hello')
        assert ls != 'world'

    def test_not_equal_to_non_string(self):
        """LazyString is not equal to a non-string, non-LazyString."""
        ls = LazyString('hello')
        assert ls != 42
        assert ls != None
        assert ls != []

    def test_hash_is_hash_of_msgid(self):
        """Hash of LazyString equals hash of its msgid."""
        ls = LazyString('hello')
        assert hash(ls) == hash('hello')

    def test_can_be_used_as_dict_key(self):
        """LazyString can be used as a dict key."""
        ls = LazyString('hello')
        d = {ls: 'value'}
        assert d[ls] == 'value'

    def test_equal_lazy_strings_have_same_hash(self):
        """Equal LazyStrings have the same hash."""
        ls1 = LazyString('hello')
        ls2 = LazyString('hello')
        assert hash(ls1) == hash(ls2)


# ---------------------------------------------------------------------------
# __bool__ and __len__
# ---------------------------------------------------------------------------

class TestLazyStringBoolLen:
    """Tests for LazyString truthiness and length."""

    def test_non_empty_is_truthy(self):
        """Non-empty LazyString is truthy."""
        ls = LazyString('hello')
        assert bool(ls) is True

    def test_empty_is_falsy(self):
        """Empty LazyString is falsy."""
        ls = LazyString('')
        assert bool(ls) is False

    def test_len_returns_string_length(self):
        """len() returns the length of the translated string."""
        ls = LazyString('hello')
        assert len(ls) == 5

    def test_len_empty(self):
        """len() of empty LazyString is 0."""
        ls = LazyString('')
        assert len(ls) == 0


# ---------------------------------------------------------------------------
# String operations: __add__, __radd__, __mul__, __rmul__
# ---------------------------------------------------------------------------

class TestLazyStringOperations:
    """Tests for LazyString string operations."""

    def test_add_string(self):
        """LazyString + str concatenates correctly."""
        ls = LazyString('hello')
        result = ls + ' world'
        assert result == 'hello world'
        assert isinstance(result, str)

    def test_radd_string(self):
        """str + LazyString concatenates correctly."""
        ls = LazyString('world')
        result = 'hello ' + ls
        assert result == 'hello world'
        assert isinstance(result, str)

    def test_mul_int(self):
        """LazyString * n repeats the string."""
        ls = LazyString('ab')
        result = ls * 3
        assert result == 'ababab'
        assert isinstance(result, str)

    def test_rmul_int(self):
        """n * LazyString repeats the string."""
        ls = LazyString('ab')
        result = 3 * ls
        assert result == 'ababab'
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# __contains__ and __getitem__
# ---------------------------------------------------------------------------

class TestLazyStringContainsGetitem:
    """Tests for LazyString containment and indexing."""

    def test_contains_substring(self):
        """'sub' in LazyString works correctly."""
        ls = LazyString('hello world')
        assert 'hello' in ls
        assert 'world' in ls
        assert 'xyz' not in ls

    def test_getitem_index(self):
        """ls[0] returns the first character."""
        ls = LazyString('hello')
        assert ls[0] == 'h'
        assert ls[-1] == 'o'

    def test_getitem_slice(self):
        """ls[1:3] returns a slice."""
        ls = LazyString('hello')
        assert ls[1:3] == 'el'


# ---------------------------------------------------------------------------
# Translation behavior (zh_CN locale)
# ---------------------------------------------------------------------------

class TestLazyStringTranslation:
    """Tests for LazyString translation with non-English locales."""

    def test_zh_cn_translates_on_str(self, monkeypatch):
        """LazyString translates to Chinese when zh_CN locale is active."""
        i18n = get_i18n_module()
        # Re-initialize with zh_CN
        i18n.setup_i18n('zh_CN')
        try:
            ls = LazyString('unimplemented endpoint')
            result = str(ls)
            assert result == '未实现的端点'
        finally:
            # Reset to English to avoid affecting other tests
            i18n.setup_i18n('en')

    def test_de_translates_on_str(self, monkeypatch):
        """LazyString translates to German when de locale is active."""
        i18n = get_i18n_module()
        i18n.setup_i18n('de')
        try:
            ls = LazyString('unimplemented endpoint')
            result = str(ls)
            assert result == 'Nicht implementierter Endpunkt'
        finally:
            i18n.setup_i18n('en')

    def test_translation_deferred_until_str_called(self):
        """Translation is deferred: the LazyString object itself is not a str."""
        ls = LazyString('hello')
        assert not isinstance(ls, str)
        # Only when str() is called does translation happen
        result = str(ls)
        assert isinstance(result, str)

    def test_lazy_string_from_i18n_lazy_(self):
        """lazy_() from vllm.i18n returns a LazyString."""
        i18n = get_i18n_module()
        ls = i18n.lazy_('hello world')
        assert isinstance(ls, get_lazy_module().LazyString)
        assert str(ls) == 'hello world'


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

class TestLazyStringJSON:
    """Tests for LazyString JSON serialization."""

    def test_json_dumps_with_str_conversion(self):
        """LazyString can be serialized to JSON via str()."""
        import json
        ls = LazyString('unimplemented endpoint')
        result = json.dumps({'error': str(ls)})
        assert result == '{"error": "unimplemented endpoint"}'

    def test_json_dumps_zh_cn(self):
        """LazyString serializes Chinese translation to JSON."""
        import json
        i18n = get_i18n_module()
        i18n.setup_i18n('zh_CN')
        try:
            ls = LazyString('unimplemented endpoint')
            result = json.dumps({'error': str(ls)}, ensure_ascii=False)
            assert '未实现的端点' in result
        finally:
            i18n.setup_i18n('en')
