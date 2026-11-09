"""Lexically extract notebook cell SOURCE strings; never decode outputs/numbers.

The original downloaded bytes are retained. No notebook execution/import/data
opening occurs. This parser skips output subtrees as lexical text without JSON
numeric conversion or returned output content. Only source text lines mentioning
method/unit/state terms are saved. Literal long numeric arrays are not exported.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
SOURCE = HERE / '10_machine_learning_analysis_code.ipynb.ipynb'
EXPECTED_SHA = '571ae2995901938fc1630de5303d670fda949daaeab93bdec8aa559825297443'


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
        raise ValueError('Downloaded notebook bytes changed')
    text = raw.decode('utf-8-sig')
    decoder = json.JSONDecoder()
    sources, cell_types, skipped_outputs = {}, {}, []
    def white(i):
        while i < len(text) and text[i] in ' \t\r\n':
            i += 1
        return i
    def string_end(i):
        if text[i] != '"':
            raise ValueError('Expected string token')
        i += 1
        while i < len(text):
            if text[i] == '\\':
                i += 2
            elif text[i] == '"':
                return i + 1
            else:
                i += 1
        raise ValueError('Unterminated JSON string')
    def skip(i):
        i = white(i)
        if text[i] == '"':
            return string_end(i)
        if text[i] in '[{':
            stack = [text[i]]
            i += 1
            while stack:
                if text[i] == '"':
                    i = string_end(i)
                    continue
                if text[i] in '[{':
                    stack.append(text[i])
                elif text[i] in ']}':
                    expected = '[' if text[i] == ']' else '{'
                    if stack.pop() != expected:
                        raise ValueError('Unbalanced JSON')
                i += 1
            return i
        # Scalar tokens including any output numbers are only skipped, not
        # converted to Python numbers, inspected, aggregated or exported.
        while i < len(text) and text[i] not in ',]} \t\r\n':
            i += 1
        return i
    def visit(i, path):
        i = white(i)
        if len(path) == 3 and path[0] == 'cells' and path[2] == 'outputs':
            end = skip(i)
            skipped_outputs.append({'cell': path[1], 'characters_skipped': end - i})
            return end
        if len(path) == 3 and path[0] == 'cells' and path[2] in ('source', 'cell_type'):
            # Only SOURCE strings or a cell-type string enter JSON decoding.
            end = skip(i)
            value = decoder.decode(text[i:end])
            if path[2] == 'source':
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                    raise ValueError('Notebook SOURCE must contain only strings')
                sources[path[1]] = ''.join(value)
            else:
                if not isinstance(value, str):
                    raise ValueError('cell_type must be a string')
                cell_types[path[1]] = value
            return end
        if text[i] == '{':
            i = white(i + 1)
            while text[i] != '}':
                end = string_end(i)
                key = decoder.decode(text[i:end])
                i = white(end)
                if text[i] != ':':
                    raise ValueError('Missing JSON colon')
                i = white(visit(i + 1, path + [key]))
                if text[i] == ',':
                    i = white(i + 1)
                elif text[i] != '}':
                    raise ValueError('Bad JSON object')
            return i + 1
        if text[i] == '[':
            i, at = white(i + 1), 0
            while text[i] != ']':
                i = white(visit(i, path + [at])); at += 1
                if text[i] == ',':
                    i = white(i + 1)
                elif text[i] != ']':
                    raise ValueError('Bad JSON array')
            return i + 1
        return skip(i)
    end = white(visit(0, []))
    if end != len(text):
        raise ValueError('Trailing JSON content')
    pattern = re.compile(r'viscos|modul|shear|frequency|frequenc|temp(?:erature)?|strain|gelma|'
                         r'concentr|methacryl|\bwt\b|w/v|w/w|\breplic|\brep\b|'
                         r'crosslink|photocur|uncur|\bPa\b|Pa[ .*/-]*s|rad.?s|deg.?C|°C|℃', re.I)
    rows, omitted_numeric_arrays = [], 0
    for cell, content in sorted(sources.items()):
        for line_number, line in enumerate(content.splitlines(), 1):
            if not pattern.search(line):
                continue
            if len(re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?', line)) > 10 and line.count(',') > 3:
                omitted_numeric_arrays += 1
                continue
            if len(line) > 800:
                continue
            rows.append({'cell_index_0based': cell, 'cell_type': cell_types.get(cell),
                         'source_line_1based': line_number, 'method_source_text': line})
    result = {'actual_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'source_notebook_sha256': EXPECTED_SHA,
              'source_license': 'CC-BY4.0 per Zenodo21759156 rev6',
              'read_scope': 'cell SOURCE strings and cell_type only; filtered method/unit text',
              'notebook_execution': False, 'datafile_opened': False,
              'numeric_output_tokens_decoded': 0, 'skipped_outputs': skipped_outputs,
              'numeric_array_lines_not_exported': omitted_numeric_arrays,
              'source_cells': len(sources), 'selected_source_lines': rows}
    with (HERE / 'NOTEBOOK_METHOD_SOURCE_ONLY.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'source_cells': len(sources), 'selected_lines': len(rows),
                      'outputs_skipped': len(skipped_outputs), 'numeric_outputs_decoded': 0}))


if __name__ == '__main__':
    main()
