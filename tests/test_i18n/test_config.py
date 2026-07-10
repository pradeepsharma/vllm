# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.i18n._config module."""

import gettext
import os
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_config_module():
    """Return the vllm.i18n._config module (loaded via conftest stub)."""
    import sys
    return sys.modules['vllm.i18n._config']


# ---------------------------------------------------------------------------
# I18NConfig tests
# ---------------------------------------------------------------------------

class TestI18NConfig:
    """Tests for the I18NConfig dataclass."""

    def test_basic_construction(self):
        """I18NConfig can be constructed with just a locale."""
        cfg = get_config_module().I18NConfig(locale='en')
        assert cfg.locale == 'en'
        assert cfg.domain == 'vllm'
        assert cfg.localedir is not None

    def test_default_domain_is_vllm(self):
        """Default domain is 'vllm'."""
        cfg = get_config_module().I18NConfig(locale='zh_CN')
        assert cfg.domain == 'vllm'

    def test_custom_domain(self):
        """Custom domain is stored correctly."""
        cfg = get_config_module().I18NConfig(locale='de', domain='myapp')
        assert cfg.domain == 'myapp'

    def test_default_localedir_is_set(self):
        """Default localedir points to the i18n/locales directory."""
        cfg = get_config_module().I18NConfig(locale='en')
        assert cfg.localedir is not None
        assert isinstance(cfg.localedir, pathlib.Path)
        # Should end with 'locales'
        assert cfg.localedir.name == 'locales'

    def test_custom_localedir(self):
        """Custom localedir is stored correctly."""
        custom_dir = pathlib.Path('/tmp/custom_locales')
        cfg = get_config_module().I18NConfig(locale='fr', localedir=custom_dir)
        assert cfg.localedir == custom_dir

    def test_locale_stored_as_given(self):
        """Locale is stored exactly as provided."""
        for locale in ['en', 'zh_CN', 'de', 'fr', 'ja', 'en_US']:
            cfg = get_config_module().I18NConfig(locale=locale)
            assert cfg.locale == locale


# ---------------------------------------------------------------------------
# _resolve_locale tests
# ---------------------------------------------------------------------------

class TestResolveLocale:
    """Tests for the _resolve_locale() function."""

    def test_vllm_locale_env_takes_priority(self, monkeypatch):
        """VLLM_LOCALE env var takes highest priority."""
        monkeypatch.setenv('VLLM_LOCALE', 'zh_CN')
        monkeypatch.setenv('LC_ALL', 'de_DE.UTF-8')
        monkeypatch.setenv('LANG', 'fr_FR.UTF-8')
        result = get_config_module()._resolve_locale()
        assert result == 'zh_CN'

    def test_lc_all_used_when_no_vllm_locale(self, monkeypatch):
        """LC_ALL is used when VLLM_LOCALE is not set."""
        monkeypatch.delenv('VLLM_LOCALE', raising=False)
        monkeypatch.setenv('LC_ALL', 'de_DE.UTF-8')
        monkeypatch.setenv('LANG', 'fr_FR.UTF-8')
        result = get_config_module()._resolve_locale()
        assert result == 'de_DE'

    def test_lang_used_when_no_lc_all(self, monkeypatch):
        """LANG is used when VLLM_LOCALE and LC_ALL are not set."""
        monkeypatch.delenv('VLLM_LOCALE', raising=False)
        monkeypatch.delenv('LC_ALL', raising=False)
        monkeypatch.setenv('LANG', 'fr_FR.UTF-8')
        result = get_config_module()._resolve_locale()
        assert result == 'fr_FR'

    def test_lc_all_strips_encoding(self, monkeypatch):
        """LC_ALL encoding suffix (e.g., .UTF-8) is stripped."""
        monkeypatch.delenv('VLLM_LOCALE', raising=False)
        monkeypatch.setenv('LC_ALL', 'ja_JP.UTF-8')
        result = get_config_module()._resolve_locale()
        assert result == 'ja_JP'

    def test_lang_strips_encoding(self, monkeypatch):
        """LANG encoding suffix is stripped."""
        monkeypatch.delenv('VLLM_LOCALE', raising=False)
        monkeypatch.delenv('LC_ALL', raising=False)
        monkeypatch.setenv('LANG', 'zh_CN.UTF-8')
        result = get_config_module()._resolve_locale()
        assert result == 'zh_CN'

    def test_fallback_to_en_when_nothing_set(self, monkeypatch):
        """Falls back to 'en' when no locale env vars are set and system locale fails."""
        monkeypatch.delenv('VLLM_LOCALE', raising=False)
        monkeypatch.delenv('LC_ALL', raising=False)
        monkeypatch.delenv('LANG', raising=False)
        # Patch locale.getdefaultlocale to return None
        import locale as locale_mod
        monkeypatch.setattr(locale_mod, 'getdefaultlocale', lambda: (None, None))
        result = get_config_module()._resolve_locale()
        assert result == 'en'

    def test_vllm_locale_no_encoding_suffix(self, monkeypatch):
        """VLLM_LOCALE is returned as-is (no encoding stripping needed)."""
        monkeypatch.setenv('VLLM_LOCALE', 'zh_CN')
        result = get_config_module()._resolve_locale()
        assert result == 'zh_CN'


# ---------------------------------------------------------------------------
# get_translator tests
# ---------------------------------------------------------------------------

class TestGetTranslator:
    """Tests for the get_translator() function."""

    def _localedir(self):
        return pathlib.Path(__file__).parent.parent.parent / 'vllm' / 'i18n' / 'locales'

    def test_english_returns_null_translations(self):
        """English locale returns NullTranslations (identity)."""
        t = get_config_module().get_translator('en', localedir=self._localedir())
        assert isinstance(t, gettext.NullTranslations)
        assert t.gettext('hello') == 'hello'

    def test_en_us_returns_null_translations(self):
        """en_US locale returns NullTranslations (identity)."""
        t = get_config_module().get_translator('en_US', localedir=self._localedir())
        assert isinstance(t, gettext.NullTranslations)

    def test_zh_cn_translates_known_string(self):
        """zh_CN locale translates 'unimplemented endpoint' to Chinese."""
        t = get_config_module().get_translator('zh_CN', localedir=self._localedir())
        result = t.gettext('unimplemented endpoint')
        assert result == '未实现的端点'

    def test_de_translates_known_string(self):
        """de locale translates 'unimplemented endpoint' to German."""
        t = get_config_module().get_translator('de', localedir=self._localedir())
        result = t.gettext('unimplemented endpoint')
        assert result == 'Nicht implementierter Endpunkt'

    def test_fr_translates_known_string(self):
        """fr locale translates 'unimplemented endpoint' to French."""
        t = get_config_module().get_translator('fr', localedir=self._localedir())
        result = t.gettext('unimplemented endpoint')
        assert result == 'Point de terminaison non implémenté'

    def test_ja_translates_known_string(self):
        """ja locale translates 'unimplemented endpoint' to Japanese."""
        t = get_config_module().get_translator('ja', localedir=self._localedir())
        result = t.gettext('unimplemented endpoint')
        assert result == '未実装のエンドポイント'

    def test_unknown_locale_falls_back_gracefully(self):
        """Unknown locale returns NullTranslations (fallback=True)."""
        t = get_config_module().get_translator('xx_XX', localedir=self._localedir())
        assert isinstance(t, gettext.NullTranslations)
        # Identity translation for unknown locale
        assert t.gettext('hello') == 'hello'

    def test_none_locale_uses_resolve(self, monkeypatch):
        """None locale triggers _resolve_locale()."""
        monkeypatch.setenv('VLLM_LOCALE', 'zh_CN')
        t = get_config_module().get_translator(None, localedir=self._localedir())
        result = t.gettext('unimplemented endpoint')
        assert result == '未实现的端点'

    def test_none_localedir_uses_default(self):
        """None localedir uses the default locales directory."""
        # Should not raise; uses the built-in locales dir
        t = get_config_module().get_translator('en', localedir=None)
        assert isinstance(t, gettext.NullTranslations)

    def test_translator_returns_gnu_translations_for_zh_cn(self):
        """zh_CN returns a GNUTranslations instance (not just NullTranslations)."""
        t = get_config_module().get_translator('zh_CN', localedir=self._localedir())
        # GNUTranslations is a subclass of NullTranslations
        assert isinstance(t, gettext.NullTranslations)
        # But it should actually translate (not identity)
        assert t.gettext('unimplemented endpoint') != 'unimplemented endpoint'
