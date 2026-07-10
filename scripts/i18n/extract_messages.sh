#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Extract translatable messages from vLLM source code and generate/update the .pot file
# This script uses pybabel to extract strings marked with _(), lazy_(), ngettext(), etc.
#
# Usage:
#   ./scripts/i18n/extract_messages.sh
#
# Requirements:
#   - pybabel (from babel package)
#   - Python 3.10+

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCALES_DIR="$PROJECT_ROOT/vllm/i18n/locales"
BABEL_CFG="$SCRIPT_DIR/babel.cfg"

echo "vLLM Message Extraction Tool"
echo "============================="
echo "Project root: $PROJECT_ROOT"
echo "Locales directory: $LOCALES_DIR"
echo "Babel config: $BABEL_CFG"
echo ""

# Check if pybabel is available
if ! command -v pybabel &> /dev/null; then
    echo "ERROR: pybabel not found. Please install it with:"
    echo "  pip install babel"
    exit 1
fi

# Create locales directory if it doesn't exist
mkdir -p "$LOCALES_DIR"

echo "Extracting messages from source code..."
echo "Command: pybabel extract -F $BABEL_CFG -o $LOCALES_DIR/vllm.pot $PROJECT_ROOT/vllm"
echo ""

# Extract messages from the vllm package
pybabel extract \
    -F "$BABEL_CFG" \
    -o "$LOCALES_DIR/vllm.pot" \
    "$PROJECT_ROOT/vllm"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully extracted messages to: $LOCALES_DIR/vllm.pot"
    
    # Count the number of messages extracted
    if command -v grep &> /dev/null; then
        MSG_COUNT=$(grep -c "^msgid" "$LOCALES_DIR/vllm.pot" || echo "unknown")
        echo "  Total messages: $MSG_COUNT"
    fi
else
    echo ""
    echo "✗ Failed to extract messages"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. To create a new locale catalog:"
echo "     ./scripts/i18n/update_catalogs.sh"
echo "  2. To compile .po files to .mo files:"
echo "     ./scripts/i18n/compile_catalogs.sh"
