# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.i18n.__init__ module (public API)."""

import json
import sys
import pytest


def get_i18n():
    """Return the vllm.i18n module."""
    return sys.modules['vllm.i18n']


def get_lazy_cls():
    """Return the LazyString class."""
    return sys.modules['vllm.i18n._lazy'].LazyString


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Tests that the public API is correctly exported."""

    def test_underscore_is_callable(self):
        """_ is callable."""
        i18n = get_i18n()
        assert callable(i18n._)

    def test_lazy_underscore_is_callable(self):
        """lazy_ is callable."""
        i18n = get_i18n()
        assert callable(i18n.lazy_)

    def test_gettext_is_callable(self):
        """gettext is callable."""
        i18n = get_i18n()
        assert callable(i18n.gettext)

    def test_ngettext_is_callable(self):
        """ngettext is callable."""
        i18n = get_i18n()
        assert callable(i18n.ngettext)

    def test_lazy_gettext_is_callable(self):
        """lazy_gettext is callable."""
        i18n = get_i18n()
        assert callable(i18n.lazy_gettext)

    def test_setup_i18n_is_callable(self):
        """setup_i18n is callable."""
        i18n = get_i18n()
        assert callable(i18n.setup_i18n)

    def test_get_current_locale_is_callable(self):
        """get_current_locale is callable."""
        i18n = get_i18n()
        assert callable(i18n.get_current_locale)

    def test_lazy_string_class_exported(self):
        """LazyString class is exported."""
        i18n = get_i18n()
        assert hasattr(i18n, 'LazyString')
        assert i18n.LazyString is get_lazy_cls()

    def test_i18n_config_class_exported(self):
        """I18NConfig class is exported."""
        i18n = get_i18n()
        assert hasattr(i18n, 'I18NConfig')

    def test_all_contains_expected_names(self):
        """__all__ contains the expected public names."""
        i18n = get_i18n()
        expected = {'gettext', 'ngettext', 'lazy_gettext', 'setup_i18n',
                    'LazyString', 'I18NConfig', '_', 'lazy_'}
        assert expected.issubset(set(i18n.__all__))


# ---------------------------------------------------------------------------
# gettext() / _() function
# ---------------------------------------------------------------------------

class TestGettext:
    """Tests for the gettext() / _() function."""

    def setup_method(self):
        """Reset to English before each test."""
        get_i18n().setup_i18n('en')

    def test_english_identity(self):
        """_() returns the original string for English."""
        i18n = get_i18n()
        assert i18n._('hello world') == 'hello world'

    def test_unknown_string_identity(self):
        """_() returns the original string when no translation exists."""
        i18n = get_i18n()
        assert i18n._('no translation for this') == 'no translation for this'

    def test_zh_cn_translates(self):
        """_() translates to Chinese when zh_CN locale is active."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            result = i18n._('unimplemented endpoint')
            assert result == '未实现的端点'
        finally:
            i18n.setup_i18n('en')

    def test_de_translates(self):
        """_() translates to German when de locale is active."""
        i18n = get_i18n()
        i18n.setup_i18n('de')
        try:
            result = i18n._('Internal server error')
            assert result == 'Interner Serverfehler'
        finally:
            i18n.setup_i18n('en')

    def test_fr_translates(self):
        """_() translates to French when fr locale is active."""
        i18n = get_i18n()
        i18n.setup_i18n('fr')
        try:
            result = i18n._('Model not found')
            assert result == 'Modèle introuvable'
        finally:
            i18n.setup_i18n('en')

    def test_ja_translates(self):
        """_() translates to Japanese when ja locale is active."""
        i18n = get_i18n()
        i18n.setup_i18n('ja')
        try:
            result = i18n._('Invalid request')
            assert result == '無効なリクエスト'
        finally:
            i18n.setup_i18n('en')

    def test_returns_string_type(self):
        """_() always returns a str."""
        i18n = get_i18n()
        result = i18n._('hello')
        assert isinstance(result, str)

    def test_underscore_alias_same_as_gettext(self):
        """_ is the same function as gettext."""
        i18n = get_i18n()
        assert i18n._ is i18n.gettext


# ---------------------------------------------------------------------------
# ngettext() function
# ---------------------------------------------------------------------------

class TestNgettext:
    """Tests for the ngettext() function."""

    def setup_method(self):
        """Reset to English before each test."""
        get_i18n().setup_i18n('en')

    def test_singular_for_n_equals_1(self):
        """ngettext returns singular form when n=1."""
        i18n = get_i18n()
        result = i18n.ngettext('1 file', '{} files', 1)
        assert result == '1 file'

    def test_plural_for_n_greater_than_1(self):
        """ngettext returns plural form when n>1."""
        i18n = get_i18n()
        result = i18n.ngettext('1 file', '{} files', 5)
        assert result == '{} files'

    def test_plural_for_n_equals_0(self):
        """ngettext returns plural form when n=0 (English)."""
        i18n = get_i18n()
        result = i18n.ngettext('1 file', '{} files', 0)
        assert result == '{} files'

    def test_zh_cn_plural(self):
        """ngettext returns Chinese translation for plural."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            result = i18n.ngettext('1 file', '{} files', 5)
            assert result == '{} 个文件'
        finally:
            i18n.setup_i18n('en')

    def test_de_singular(self):
        """ngettext returns German singular translation."""
        i18n = get_i18n()
        i18n.setup_i18n('de')
        try:
            result = i18n.ngettext('1 file', '{} files', 1)
            assert result == '1 Datei'
        finally:
            i18n.setup_i18n('en')

    def test_de_plural(self):
        """ngettext returns German plural translation."""
        i18n = get_i18n()
        i18n.setup_i18n('de')
        try:
            result = i18n.ngettext('1 file', '{} files', 3)
            assert result == '{} Dateien'
        finally:
            i18n.setup_i18n('en')

    def test_format_with_count(self):
        """ngettext result can be formatted with the count."""
        i18n = get_i18n()
        result = i18n.ngettext('1 file', '{} files', 7).format(7)
        assert result == '7 files'


# ---------------------------------------------------------------------------
# lazy_gettext() / lazy_() function
# ---------------------------------------------------------------------------

class TestLazyGettext:
    """Tests for the lazy_gettext() / lazy_() function."""

    def setup_method(self):
        """Reset to English before each test."""
        get_i18n().setup_i18n('en')

    def test_returns_lazy_string(self):
        """lazy_() returns a LazyString instance."""
        i18n = get_i18n()
        result = i18n.lazy_('hello')
        assert isinstance(result, get_lazy_cls())

    def test_lazy_string_translates_on_str(self):
        """LazyString from lazy_() translates when str() is called."""
        i18n = get_i18n()
        ls = i18n.lazy_('unimplemented endpoint')
        # English: identity
        assert str(ls) == 'unimplemented endpoint'

    def test_lazy_string_translates_zh_cn(self):
        """LazyString translates to Chinese when locale is set to zh_CN."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            ls = i18n.lazy_('unimplemented endpoint')
            assert str(ls) == '未实现的端点'
        finally:
            i18n.setup_i18n('en')

    def test_lazy_alias_same_as_lazy_gettext(self):
        """lazy_ is the same function as lazy_gettext."""
        i18n = get_i18n()
        assert i18n.lazy_ is i18n.lazy_gettext


# ---------------------------------------------------------------------------
# setup_i18n() and get_current_locale()
# ---------------------------------------------------------------------------

class TestSetupAndLocale:
    """Tests for setup_i18n() and get_current_locale()."""

    def setup_method(self):
        """Reset to English before each test."""
        get_i18n().setup_i18n('en')

    def test_get_current_locale_returns_string(self):
        """get_current_locale() returns a string."""
        i18n = get_i18n()
        result = i18n.get_current_locale()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_setup_i18n_changes_locale(self):
        """setup_i18n() changes the active locale."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            assert i18n.get_current_locale() == 'zh_CN'
        finally:
            i18n.setup_i18n('en')

    def test_setup_i18n_de(self):
        """setup_i18n('de') sets locale to de."""
        i18n = get_i18n()
        i18n.setup_i18n('de')
        try:
            assert i18n.get_current_locale() == 'de'
        finally:
            i18n.setup_i18n('en')

    def test_setup_i18n_none_uses_env(self, monkeypatch):
        """setup_i18n(None) uses environment variable for locale."""
        i18n = get_i18n()
        monkeypatch.setenv('VLLM_LOCALE', 'fr')
        i18n.setup_i18n(None)
        try:
            assert i18n.get_current_locale() == 'fr'
        finally:
            i18n.setup_i18n('en')

    def test_setup_i18n_can_switch_locales(self):
        """setup_i18n() can switch locales multiple times."""
        i18n = get_i18n()
        for locale in ['zh_CN', 'de', 'fr', 'ja', 'en']:
            i18n.setup_i18n(locale)
            assert i18n.get_current_locale() == locale
        # Ensure we end on English
        assert i18n.get_current_locale() == 'en'

    def test_vllm_locale_env_respected(self, monkeypatch):
        """VLLM_LOCALE env var is respected by setup_i18n(None)."""
        i18n = get_i18n()
        monkeypatch.setenv('VLLM_LOCALE', 'ja')
        i18n.setup_i18n(None)
        try:
            result = i18n._('unimplemented endpoint')
            assert result == '未実装のエンドポイント'
        finally:
            i18n.setup_i18n('en')


# ---------------------------------------------------------------------------
# Catalog compilation check (verification criteria #5)
# ---------------------------------------------------------------------------

class TestCatalogCompilation:
    """Tests that all locale catalogs load correctly."""

    def test_all_locales_load(self):
        """All 5 locales (en, zh_CN, de, fr, ja) load without error."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        for locale in ['zh_CN', 'de', 'fr', 'ja']:
            t = gettext.translation('vllm', localedir=str(localedir),
                                    languages=[locale], fallback=True)
            assert t is not None, f"Failed to load locale: {locale}"

    def test_en_fallback_works(self):
        """English locale uses NullTranslations (identity)."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        t = gettext.translation('vllm', localedir=str(localedir),
                                languages=['en'], fallback=True)
        assert t.gettext('hello') == 'hello'

    def test_zh_cn_catalog_has_translations(self):
        """zh_CN catalog has actual translations (not identity)."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        t = gettext.translation('vllm', localedir=str(localedir),
                                languages=['zh_CN'], fallback=True)
        result = t.gettext('unimplemented endpoint')
        assert result != 'unimplemented endpoint'
        assert result == '未实现的端点'

    def test_de_catalog_has_translations(self):
        """de catalog has actual translations."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        t = gettext.translation('vllm', localedir=str(localedir),
                                languages=['de'], fallback=True)
        result = t.gettext('unimplemented endpoint')
        assert result == 'Nicht implementierter Endpunkt'

    def test_fr_catalog_has_translations(self):
        """fr catalog has actual translations."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        t = gettext.translation('vllm', localedir=str(localedir),
                                languages=['fr'], fallback=True)
        result = t.gettext('unimplemented endpoint')
        assert result == 'Point de terminaison non implémenté'

    def test_ja_catalog_has_translations(self):
        """ja catalog has actual translations."""
        import gettext
        import pathlib
        localedir = pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'
        t = gettext.translation('vllm', localedir=str(localedir),
                                languages=['ja'], fallback=True)
        result = t.gettext('unimplemented endpoint')
        assert result == '未実装のエンドポイント'


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end integration tests for the i18n system."""

    def setup_method(self):
        """Reset to English before each test."""
        get_i18n().setup_i18n('en')

    def test_full_flow_english(self):
        """Full flow: English locale, identity translation."""
        i18n = get_i18n()
        i18n.setup_i18n('en')
        assert i18n.get_current_locale() == 'en'
        assert i18n._('unimplemented endpoint') == 'unimplemented endpoint'

    def test_full_flow_zh_cn(self):
        """Full flow: zh_CN locale, Chinese translation."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            assert i18n.get_current_locale() == 'zh_CN'
            assert i18n._('unimplemented endpoint') == '未实现的端点'
        finally:
            i18n.setup_i18n('en')

    def test_full_flow_lazy_string_json(self):
        """Full flow: lazy string serializes to JSON correctly."""
        i18n = get_i18n()
        i18n.setup_i18n('zh_CN')
        try:
            msg = i18n.lazy_('unimplemented endpoint')
            result = json.dumps({'error': str(msg)}, ensure_ascii=False)
            data = json.loads(result)
            assert data['error'] == '未实现的端点'
        finally:
            i18n.setup_i18n('en')

    def test_full_flow_ngettext_with_format(self):
        """Full flow: ngettext with format works end-to-end."""
        i18n = get_i18n()
        i18n.setup_i18n('de')
        try:
            result = i18n.ngettext('1 file', '{} files', 5).format(5)
            assert result == '5 Dateien'
        finally:
            i18n.setup_i18n('en')

    def test_locale_switch_affects_translation(self):
        """Switching locale changes translation output."""
        i18n = get_i18n()
        i18n.setup_i18n('en')
        en_result = i18n._('unimplemented endpoint')

        i18n.setup_i18n('zh_CN')
        zh_result = i18n._('unimplemented endpoint')

        i18n.setup_i18n('en')
        en_result2 = i18n._('unimplemented endpoint')

        assert en_result == 'unimplemented endpoint'
        assert zh_result == '未实现的端点'
        assert en_result2 == 'unimplemented endpoint'
        assert en_result != zh_result
