# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Security tests: verify that HTTPConnection uses secure defaults
and validates URLs properly.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.connections import HTTPConnection


class TestHTTPConnectionSecurity:
    """Test HTTPConnection security properties."""

    @pytest.fixture
    def http_connection(self):
        """Create an HTTPConnection instance."""
        return HTTPConnection()

    def test_validate_http_url_accepts_http(self, http_connection):
        """Test that _validate_http_url accepts http:// URLs."""
        # Should not raise
        http_connection._validate_http_url("http://example.com")

    def test_validate_http_url_accepts_https(self, http_connection):
        """Test that _validate_http_url accepts https:// URLs."""
        # Should not raise
        http_connection._validate_http_url("https://example.com")

    def test_validate_http_url_rejects_file_scheme(self, http_connection):
        """Test that _validate_http_url rejects file:// URLs."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("file:///etc/passwd")
        
        assert "Invalid HTTP URL" in str(exc_info.value)
        assert "http" in str(exc_info.value).lower()

    def test_validate_http_url_rejects_ftp_scheme(self, http_connection):
        """Test that _validate_http_url rejects ftp:// URLs."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("ftp://example.com/file")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_validate_http_url_rejects_gopher_scheme(self, http_connection):
        """Test that _validate_http_url rejects gopher:// URLs."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("gopher://example.com")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_validate_http_url_rejects_data_scheme(self, http_connection):
        """Test that _validate_http_url rejects data: URLs."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("data:text/html,<script>alert('xss')</script>")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_validate_http_url_rejects_javascript_scheme(self, http_connection):
        """Test that _validate_http_url rejects javascript: URLs."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("javascript:alert('xss')")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_validate_http_url_rejects_no_scheme(self, http_connection):
        """Test that _validate_http_url rejects URLs without a scheme."""
        with pytest.raises(ValueError) as exc_info:
            http_connection._validate_http_url("example.com/path")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_validate_http_url_accepts_http_with_port(self, http_connection):
        """Test that _validate_http_url accepts http:// URLs with ports."""
        # Should not raise
        http_connection._validate_http_url("http://example.com:8080")

    def test_validate_http_url_accepts_https_with_port(self, http_connection):
        """Test that _validate_http_url accepts https:// URLs with ports."""
        # Should not raise
        http_connection._validate_http_url("https://example.com:443")

    def test_validate_http_url_accepts_http_with_path(self, http_connection):
        """Test that _validate_http_url accepts http:// URLs with paths."""
        # Should not raise
        http_connection._validate_http_url("http://example.com/api/v1/models")

    def test_validate_http_url_accepts_http_with_query(self, http_connection):
        """Test that _validate_http_url accepts http:// URLs with query strings."""
        # Should not raise
        http_connection._validate_http_url("http://example.com/api?key=value&foo=bar")

    def test_validate_http_url_accepts_http_with_fragment(self, http_connection):
        """Test that _validate_http_url accepts http:// URLs with fragments."""
        # Should not raise
        http_connection._validate_http_url("http://example.com/page#section")

    def test_validate_http_url_accepts_localhost(self, http_connection):
        """Test that _validate_http_url accepts localhost URLs."""
        # Should not raise
        http_connection._validate_http_url("http://localhost:8000")
        http_connection._validate_http_url("http://127.0.0.1:8000")

    def test_validate_http_url_accepts_ipv4(self, http_connection):
        """Test that _validate_http_url accepts IPv4 URLs."""
        # Should not raise
        http_connection._validate_http_url("http://192.168.1.1:8000")
        http_connection._validate_http_url("https://10.0.0.1")

    def test_validate_http_url_accepts_ipv6(self, http_connection):
        """Test that _validate_http_url accepts IPv6 URLs."""
        # Should not raise
        http_connection._validate_http_url("http://[::1]:8000")
        http_connection._validate_http_url("https://[2001:db8::1]")

    def test_get_response_validates_url(self, http_connection):
        """Test that get_response validates the URL."""
        with pytest.raises(ValueError) as exc_info:
            http_connection.get_response("file:///etc/passwd")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_get_async_response_validates_url(self, http_connection):
        """Test that get_async_response validates the URL."""
        async def test():
            with pytest.raises(ValueError) as exc_info:
                await http_connection.get_async_response("file:///etc/passwd")
            
            assert "Invalid HTTP URL" in str(exc_info.value)
        
        asyncio.run(test())

    def test_get_bytes_validates_url(self, http_connection):
        """Test that get_bytes validates the URL."""
        with pytest.raises(ValueError) as exc_info:
            http_connection.get_bytes("ftp://example.com/file")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_async_get_bytes_validates_url(self, http_connection):
        """Test that async_get_bytes validates the URL."""
        async def test():
            with pytest.raises(ValueError) as exc_info:
                await http_connection.async_get_bytes("ftp://example.com/file")
            
            assert "Invalid HTTP URL" in str(exc_info.value)
        
        asyncio.run(test())

    def test_get_text_validates_url(self, http_connection):
        """Test that get_text validates the URL."""
        with pytest.raises(ValueError) as exc_info:
            http_connection.get_text("javascript:alert('xss')")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_async_get_text_validates_url(self, http_connection):
        """Test that async_get_text validates the URL."""
        async def test():
            with pytest.raises(ValueError) as exc_info:
                await http_connection.async_get_text("javascript:alert('xss')")
            
            assert "Invalid HTTP URL" in str(exc_info.value)
        
        asyncio.run(test())

    def test_get_json_validates_url(self, http_connection):
        """Test that get_json validates the URL."""
        with pytest.raises(ValueError) as exc_info:
            http_connection.get_json("data:application/json,{}")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_async_get_json_validates_url(self, http_connection):
        """Test that async_get_json validates the URL."""
        async def test():
            with pytest.raises(ValueError) as exc_info:
                await http_connection.async_get_json("data:application/json,{}")
            
            assert "Invalid HTTP URL" in str(exc_info.value)
        
        asyncio.run(test())

    def test_download_file_validates_url(self, http_connection, tmp_path):
        """Test that download_file validates the URL."""
        with pytest.raises(ValueError) as exc_info:
            http_connection.download_file("file:///etc/passwd", tmp_path / "output")
        
        assert "Invalid HTTP URL" in str(exc_info.value)

    def test_async_download_file_validates_url(self, http_connection, tmp_path):
        """Test that async_download_file validates the URL."""
        async def test():
            with pytest.raises(ValueError) as exc_info:
                await http_connection.async_download_file("file:///etc/passwd", tmp_path / "output")
            
            assert "Invalid HTTP URL" in str(exc_info.value)
        
        asyncio.run(test())

    def test_get_sync_client_creates_session(self, http_connection):
        """Test that get_sync_client creates a requests.Session."""
        client = http_connection.get_sync_client()
        assert client is not None
        assert hasattr(client, 'get')
        assert hasattr(client, 'post')

    def test_get_sync_client_reuses_session(self, http_connection):
        """Test that get_sync_client reuses the same session."""
        client1 = http_connection.get_sync_client()
        client2 = http_connection.get_sync_client()
        assert client1 is client2

    def test_get_sync_client_no_reuse(self):
        """Test that get_sync_client creates new sessions when reuse_client=False."""
        http_connection = HTTPConnection(reuse_client=False)
        client1 = http_connection.get_sync_client()
        client2 = http_connection.get_sync_client()
        assert client1 is not client2

    @pytest.mark.asyncio
    async def test_get_async_client_creates_session(self, http_connection):
        """Test that get_async_client creates an aiohttp.ClientSession."""
        client = await http_connection.get_async_client()
        assert client is not None
        assert hasattr(client, 'get')
        assert hasattr(client, 'post')
        await client.close()

    @pytest.mark.asyncio
    async def test_get_async_client_reuses_session(self, http_connection):
        """Test that get_async_client reuses the same session."""
        client1 = await http_connection.get_async_client()
        client2 = await http_connection.get_async_client()
        assert client1 is client2
        await client1.close()

    @pytest.mark.asyncio
    async def test_get_async_client_no_reuse(self):
        """Test that get_async_client creates new sessions when reuse_client=False."""
        http_connection = HTTPConnection(reuse_client=False)
        client1 = await http_connection.get_async_client()
        client2 = await http_connection.get_async_client()
        assert client1 is not client2
        await client1.close()
        await client2.close()

    def test_headers_includes_user_agent(self, http_connection):
        """Test that _headers includes a User-Agent."""
        headers = http_connection._headers()
        assert "User-Agent" in headers
        assert "vLLM" in headers["User-Agent"]

    def test_headers_includes_extra_headers(self, http_connection):
        """Test that _headers includes extra headers."""
        headers = http_connection._headers(Authorization="Bearer token", Custom="value")
        assert headers["Authorization"] == "Bearer token"
        assert headers["Custom"] == "value"
        assert "User-Agent" in headers

    def test_validate_http_url_case_insensitive_scheme(self, http_connection):
        """Test that scheme validation is case-insensitive."""
        # Should not raise
        http_connection._validate_http_url("HTTP://example.com")
        http_connection._validate_http_url("HTTPS://example.com")
        http_connection._validate_http_url("Http://example.com")
        http_connection._validate_http_url("HtTpS://example.com")

    def test_validate_http_url_rejects_empty_string(self, http_connection):
        """Test that _validate_http_url rejects empty strings."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("")

    def test_validate_http_url_rejects_whitespace_only(self, http_connection):
        """Test that _validate_http_url rejects whitespace-only strings."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("   ")

    def test_validate_http_url_rejects_url_with_embedded_newline(self, http_connection):
        """Test that _validate_http_url rejects URLs with embedded newlines."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("http://example.com\nhttp://evil.com")

    def test_validate_http_url_accepts_url_with_userinfo(self, http_connection):
        """Test that _validate_http_url accepts URLs with userinfo."""
        # Should not raise (though userinfo in URLs is generally not recommended)
        http_connection._validate_http_url("http://user:pass@example.com")

    def test_validate_http_url_accepts_url_with_encoded_chars(self, http_connection):
        """Test that _validate_http_url accepts URLs with percent-encoded characters."""
        # Should not raise
        http_connection._validate_http_url("http://example.com/path%20with%20spaces")

    def test_validate_http_url_rejects_relative_url(self, http_connection):
        """Test that _validate_http_url rejects relative URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("/path/to/resource")

    def test_validate_http_url_rejects_protocol_relative_url(self, http_connection):
        """Test that _validate_http_url rejects protocol-relative URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("//example.com/path")

    def test_validate_http_url_rejects_ldap_scheme(self, http_connection):
        """Test that _validate_http_url rejects ldap:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("ldap://example.com/cn=admin")

    def test_validate_http_url_rejects_dict_scheme(self, http_connection):
        """Test that _validate_http_url rejects dict:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("dict://example.com/word")

    def test_validate_http_url_rejects_telnet_scheme(self, http_connection):
        """Test that _validate_http_url rejects telnet:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("telnet://example.com:23")

    def test_validate_http_url_rejects_ssh_scheme(self, http_connection):
        """Test that _validate_http_url rejects ssh:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("ssh://user@example.com")

    def test_validate_http_url_rejects_sftp_scheme(self, http_connection):
        """Test that _validate_http_url rejects sftp:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("sftp://user@example.com")

    def test_validate_http_url_rejects_smb_scheme(self, http_connection):
        """Test that _validate_http_url rejects smb:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("smb://server/share")

    def test_validate_http_url_rejects_nfs_scheme(self, http_connection):
        """Test that _validate_http_url rejects nfs:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("nfs://server/export")

    def test_validate_http_url_rejects_rtmp_scheme(self, http_connection):
        """Test that _validate_http_url rejects rtmp:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("rtmp://example.com/stream")

    def test_validate_http_url_rejects_rtsp_scheme(self, http_connection):
        """Test that _validate_http_url rejects rtsp:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("rtsp://example.com/stream")

    def test_validate_http_url_rejects_mms_scheme(self, http_connection):
        """Test that _validate_http_url rejects mms:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("mms://example.com/stream")

    def test_validate_http_url_rejects_news_scheme(self, http_connection):
        """Test that _validate_http_url rejects news:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("news://example.com/group")

    def test_validate_http_url_rejects_nntp_scheme(self, http_connection):
        """Test that _validate_http_url rejects nntp:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("nntp://example.com/group")

    def test_validate_http_url_rejects_irc_scheme(self, http_connection):
        """Test that _validate_http_url rejects irc:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("irc://example.com/channel")

    def test_validate_http_url_rejects_ircs_scheme(self, http_connection):
        """Test that _validate_http_url rejects ircs:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("ircs://example.com/channel")

    def test_validate_http_url_rejects_git_scheme(self, http_connection):
        """Test that _validate_http_url rejects git:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("git://example.com/repo.git")

    def test_validate_http_url_rejects_svn_scheme(self, http_connection):
        """Test that _validate_http_url rejects svn:// URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("svn://example.com/repo")

    def test_validate_http_url_rejects_ws_scheme(self, http_connection):
        """Test that _validate_http_url rejects ws:// URLs (WebSocket)."""
        # Note: WebSocket URLs might be supported in the future, but for now
        # they should be rejected as they're not HTTP/HTTPS
        with pytest.raises(ValueError):
            http_connection._validate_http_url("ws://example.com/socket")

    def test_validate_http_url_rejects_wss_scheme(self, http_connection):
        """Test that _validate_http_url rejects wss:// URLs (Secure WebSocket)."""
        # Note: WebSocket URLs might be supported in the future, but for now
        # they should be rejected as they're not HTTP/HTTPS
        with pytest.raises(ValueError):
            http_connection._validate_http_url("wss://example.com/socket")

    def test_validate_http_url_rejects_mailto_scheme(self, http_connection):
        """Test that _validate_http_url rejects mailto: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("mailto:user@example.com")

    def test_validate_http_url_rejects_news_scheme_with_article(self, http_connection):
        """Test that _validate_http_url rejects news: URLs with article IDs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("news:comp.lang.python")

    def test_validate_http_url_rejects_urn_scheme(self, http_connection):
        """Test that _validate_http_url rejects urn: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("urn:isbn:0451450523")

    def test_validate_http_url_rejects_magnet_scheme(self, http_connection):
        """Test that _validate_http_url rejects magnet: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("magnet:?xt=urn:btih:123456")

    def test_validate_http_url_rejects_bitcoin_scheme(self, http_connection):
        """Test that _validate_http_url rejects bitcoin: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("bitcoin:1A1z7agoat")

    def test_validate_http_url_rejects_geo_scheme(self, http_connection):
        """Test that _validate_http_url rejects geo: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("geo:37.786971,-122.399677")

    def test_validate_http_url_rejects_tel_scheme(self, http_connection):
        """Test that _validate_http_url rejects tel: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("tel:+1-201-555-0123")

    def test_validate_http_url_rejects_sms_scheme(self, http_connection):
        """Test that _validate_http_url rejects sms: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("sms:+1-201-555-0123")

    def test_validate_http_url_rejects_sip_scheme(self, http_connection):
        """Test that _validate_http_url rejects sip: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("sip:user@example.com")

    def test_validate_http_url_rejects_sips_scheme(self, http_connection):
        """Test that _validate_http_url rejects sips: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("sips:user@example.com")

    def test_validate_http_url_rejects_xmpp_scheme(self, http_connection):
        """Test that _validate_http_url rejects xmpp: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("xmpp:user@example.com")

    def test_validate_http_url_rejects_view_source_scheme(self, http_connection):
        """Test that _validate_http_url rejects view-source: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("view-source:http://example.com")

    def test_validate_http_url_rejects_about_scheme(self, http_connection):
        """Test that _validate_http_url rejects about: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("about:blank")

    def test_validate_http_url_rejects_blob_scheme(self, http_connection):
        """Test that _validate_http_url rejects blob: URLs."""
        with pytest.raises(ValueError):
            http_connection._validate_http_url("blob:http://example.com/123e4567")
