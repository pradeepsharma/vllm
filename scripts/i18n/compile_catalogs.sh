#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Compile translation catalogs (.po files) to binary .mo files
# This script uses pybabel to compile .po files into .mo files that can be
# used by Python's gettext module at runtime.
#
# Usage:
#   ./scripts/i18n/compile_catalogs.sh
#
# Requirements:
#   - pybabel (from babel package)
#   - Python 3.10+
#   - .po files in vllm/i18n/locales/*/LC_MESSAGES/

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCALES_DIR="$PROJECT_ROOT/vllm/i18n/locales"

echo "vLLM Catalog Compilation Tool"
echo "=============================="
echo "Project root: $PROJECT_ROOT"
echo "Locales directory: $LOCALES_DIR"
echo ""

# Check if pybabel is available
if ! command -v pybabel &> /dev/null; then
    echo "ERROR: pybabel not found. Please install it with:"
    echo "  pip install babel"
    exit 1
fi

# Check if locales directory exists
if [ ! -d "$LOCALES_DIR" ]; then
    echo "ERROR: Locales directory not found: $LOCALES_DIR"
    exit 1
fi

echo "Compiling translation catalogs..."
echo ""

# Use pybabel compile to compile all .po files to .mo files
pybabel compile \
    -d "$LOCALES_DIR" \
    -D vllm \
    --statistics

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully compiled all translation catalogs"
    echo ""
    echo "Compiled .mo files:"
    find "$LOCALES_DIR" -name "*.mo" -type f | sort | while read mo_file; do
        locale=$(echo "$mo_file" | sed -E 's|.*/([^/]+)/LC_MESSAGES.*|\1|')
        echo "  - $locale: $(basename "$mo_file")"
    done
else
    echo ""
    echo "✗ Failed to compile translation catalogs"
    exit 1
fi

echo ""
echo "Translation catalogs are ready for use!"
echo "The .mo files will be loaded automatically by vLLM at runtime."
