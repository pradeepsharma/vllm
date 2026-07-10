#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to compile .po files to .mo files for vLLM i18n support.

This script compiles all .po files in the locales directory to binary .mo
files that Python's gettext module can use at runtime.

The .mo file format is documented at:
https://www.gnu.org/software/gettext/manual/html_node/MO-Files.html

Usage:
    python3 vllm/i18n/compile_locales.py
"""

import struct
import pathlib
import sys


def unescape_po_string(s: str) -> str:
    """Unescape a PO file string (handle \\n, \\t, \\", \\\\)."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            c = s[i + 1]
            if c == 'n':
                result.append('\n')
            elif c == 't':
                result.append('\t')
            elif c == '"':
                result.append('"')
            elif c == '\\':
                result.append('\\')
            else:
                result.append('\\')
                result.append(c)
            i += 2
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def parse_po_file(po_path: pathlib.Path) -> list:
    """Parse a .po file and return a list of (msgid_bytes, msgstr_bytes) pairs.
    
    Handles both regular entries and plural form entries.
    For plural entries: msgid_bytes = b"singular\\x00plural"
                        msgstr_bytes = b"form0\\x00form1\\x00..."
    
    Args:
        po_path: Path to the .po file.
        
    Returns:
        A list of (msgid_bytes, msgstr_bytes) pairs, including the header
        entry (empty msgid).
    """
    entries = []

    # State machine
    current_msgid = None        # str or None
    current_msgid_plural = None # str or None (for plural forms)
    current_msgstr = None       # str or None (for simple entries)
    current_msgstr_plural = {}  # dict {int: str} for plural forms
    in_field = None             # 'msgid' | 'msgid_plural' | 'msgstr' | 'msgstr_N'

    def flush_entry():
        nonlocal current_msgid, current_msgid_plural, current_msgstr, current_msgstr_plural
        if current_msgid is None:
            return

        msgid_unesc = unescape_po_string(current_msgid)

        if current_msgid_plural is not None:
            # Plural entry: msgid = "singular\x00plural"
            msgid_plural_unesc = unescape_po_string(current_msgid_plural)
            msgid_bytes = (msgid_unesc + '\x00' + msgid_plural_unesc).encode('utf-8')
            # msgstr = "form0\x00form1\x00..."
            forms = []
            for i in sorted(current_msgstr_plural.keys()):
                forms.append(unescape_po_string(current_msgstr_plural[i]))
            msgstr_bytes = '\x00'.join(forms).encode('utf-8')
        else:
            # Simple entry
            msgid_bytes = msgid_unesc.encode('utf-8')
            msgstr_unesc = unescape_po_string(current_msgstr or '')
            msgstr_bytes = msgstr_unesc.encode('utf-8')

        entries.append((msgid_bytes, msgstr_bytes))

        # Reset state
        current_msgid = None
        current_msgid_plural = None
        current_msgstr = None
        current_msgstr_plural = {}
        in_field = None

    with open(po_path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')

            # Skip comments
            if line.startswith('#'):
                continue

            # msgid "..."
            if line.startswith('msgid "') and line.endswith('"'):
                flush_entry()
                current_msgid = line[7:-1]
                in_field = 'msgid'

            # msgid_plural "..."
            elif line.startswith('msgid_plural "') and line.endswith('"'):
                current_msgid_plural = line[14:-1]
                in_field = 'msgid_plural'

            # msgstr "..."  (simple)
            elif line.startswith('msgstr "') and line.endswith('"'):
                current_msgstr = line[8:-1]
                in_field = 'msgstr'

            # msgstr[N] "..."  (plural form)
            elif line.startswith('msgstr[') and ']' in line:
                bracket_end = line.index(']')
                n = int(line[7:bracket_end])
                rest = line[bracket_end + 1:].strip()
                if rest.startswith('"') and rest.endswith('"'):
                    current_msgstr_plural[n] = rest[1:-1]
                in_field = f'msgstr_{n}'

            # Continuation line "..."
            elif line.startswith('"') and line.endswith('"'):
                content = line[1:-1]
                if in_field == 'msgid' and current_msgid is not None:
                    current_msgid += content
                elif in_field == 'msgid_plural' and current_msgid_plural is not None:
                    current_msgid_plural += content
                elif in_field == 'msgstr' and current_msgstr is not None:
                    current_msgstr += content
                elif in_field and in_field.startswith('msgstr_'):
                    n = int(in_field[7:])
                    if n in current_msgstr_plural:
                        current_msgstr_plural[n] += content

            # Empty line: flush current entry
            elif line.strip() == '':
                flush_entry()
                in_field = None

    # Flush last entry
    flush_entry()

    return entries


def compile_entries_to_mo(entries: list) -> bytes:
    """Compile a list of (msgid_bytes, msgstr_bytes) pairs to binary .mo format.
    
    The .mo file format:
    - Magic number (4 bytes, little-endian): 0x950412de
    - File format revision (4 bytes): 0
    - Number of strings N (4 bytes)
    - Offset of original string table (4 bytes)
    - Offset of translation string table (4 bytes)
    - Size of hash table (4 bytes): 0 (unused)
    - Offset of hash table (4 bytes)
    - Original string table: N * (length, offset) pairs
    - Translation string table: N * (length, offset) pairs
    - String data (NUL-terminated)
    
    Args:
        entries: List of (msgid_bytes, msgstr_bytes) pairs.
        
    Returns:
        Binary content of the .mo file.
    """
    # Sort by msgid for binary search (required by gettext)
    entries_sorted = sorted(entries, key=lambda x: x[0])

    n = len(entries_sorted)

    MAGIC = 0x950412de
    REVISION = 0

    # Offsets:
    # Header: 7 * 4 = 28 bytes
    # Original string table: n * 8 bytes
    # Translation string table: n * 8 bytes
    # String data starts after that
    header_size = 28
    orig_table_offset = header_size
    trans_table_offset = orig_table_offset + n * 8
    strings_start = trans_table_offset + n * 8

    # Build string data and offset tables
    orig_offsets = []
    trans_offsets = []
    string_data = b''
    current_offset = strings_start

    for msgid_bytes, _ in entries_sorted:
        orig_offsets.append((len(msgid_bytes), current_offset))
        string_data += msgid_bytes + b'\x00'
        current_offset += len(msgid_bytes) + 1

    for _, msgstr_bytes in entries_sorted:
        trans_offsets.append((len(msgstr_bytes), current_offset))
        string_data += msgstr_bytes + b'\x00'
        current_offset += len(msgstr_bytes) + 1

    # Build the binary file
    result = struct.pack('<IIIIIII',
                        MAGIC,
                        REVISION,
                        n,
                        orig_table_offset,
                        trans_table_offset,
                        0,              # hash table size (unused)
                        strings_start)  # hash table offset (unused)

    # Original string table
    for length, offset in orig_offsets:
        result += struct.pack('<II', length, offset)

    # Translation string table
    for length, offset in trans_offsets:
        result += struct.pack('<II', length, offset)

    # String data
    result += string_data

    return result


def main():
    """Compile all .po files in the locales directory."""
    locales_dir = pathlib.Path(__file__).parent / 'locales'

    if not locales_dir.exists():
        print(f'ERROR: Locales directory not found: {locales_dir}')
        sys.exit(1)

    compiled = 0
    errors = 0

    for po_file in sorted(locales_dir.rglob('*.po')):
        mo_file = po_file.with_suffix('.mo')
        try:
            entries = parse_po_file(po_file)
            mo_data = compile_entries_to_mo(entries)
            mo_file.write_bytes(mo_data)
            locale = po_file.parent.parent.name
            print(f'  Compiled: {locale}/LC_MESSAGES/{po_file.name} -> {mo_file.name} ({len(entries)} entries)')
            compiled += 1
        except Exception as e:
            print(f'  ERROR compiling {po_file}: {e}')
            import traceback
            traceback.print_exc()
            errors += 1

    print(f'\nDone: {compiled} compiled, {errors} errors')
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
