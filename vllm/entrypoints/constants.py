# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared constants for vLLM entrypoints.
"""

# HTTP header limits for h11 parser
# These constants help mitigate header abuse attacks
# H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT: Maximum size (in bytes) of an incomplete
# HTTP event (request line + headers) before the h11 parser rejects it.
# Set to 4 MB (4194304 bytes) to balance between supporting legitimate large headers
# (e.g., long Authorization headers, large cookies) and preventing memory exhaustion
# attacks via oversized headers. This limit applies to the raw HTTP event data before
# parsing, protecting against slowloris and header-based DoS attacks.
H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT = 4194304  # 4 MB

# H11_MAX_HEADER_COUNT_DEFAULT: Maximum number of HTTP headers allowed in a single
# request before the h11 parser rejects it. Set to 256 to prevent header-based DoS
# attacks where attackers send thousands of headers to exhaust server memory and CPU.
# This is a reasonable limit for legitimate HTTP clients, as typical requests have
# 10-50 headers. Exceeding 256 headers is a strong indicator of malicious intent.
H11_MAX_HEADER_COUNT_DEFAULT = 256

MCP_PREFIX = "mcp_"
