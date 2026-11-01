"""Rebuild fixed source vectors from a non-numeric cell map; no network or models."""
import argparse
import hashlib
import json
import math
import posixpath
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

VERSION = '1.0.0'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL_ID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
CELL = re.compile(r'^([A-Z]+)([1-9][0-9]*)$')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def vector_bytes(gamma, eta):
    return json.dumps({'gamma': [float(x) for x in gamma],
                       'eta': [float(x) for x in eta]},
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def xlsx_cells(path, requested):
    values = {}
    with zipfile.ZipFile(path) as z:
        rels = {e.attrib['Id']: e.attrib['Target']
                for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets = ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS)
        parts = {}
        for sheet in sheets:
            target = rels[sheet.attrib[REL_ID]]
            parts[sheet.attrib['name']] = (target.lstrip('/') if target.startswith('/')
                else posixpath.normpath(posixpath.join('xl', target)))
        for name, cells in requested.items():
            if name not in parts:
                raise ValueError('Missing mapped worksheet: ' + repr(name))
            root = ET.fromstring(z.read(parts[name]))
            for c in root.findall('.//s:c', NS):
                address = c.attrib['r']
                if address not in cells:
                    continue
                if c.find('s:f', NS) is not None or c.attrib.get('t') not in (None, 'n'):
                    raise ValueError('Mapped XLSX cell is not a native numeric cell: ' + name + '!' + address)
                v = c.find('s:v', NS)
                if v is None or v.text is None:
                    raise ValueError('Mapped numeric cell is blank: ' + name + '!' + address)
                values[(name, address)] = float(v.text)
            for address in cells:
                if (name, address) not in values:
                    raise ValueError('Mapped cell not found: ' + name + '!' + address)
    return values


def xls_cells(path, requested):
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError('XLS input requires task-local xlrd 2.0.2 on PYTHONPATH; no installation is performed.') from exc
    if xlrd.__version__ != '2.0.2':
        raise RuntimeError('Expected xlrd 2.0.2, found ' + xlrd.__version__)
    values = {}
    book = xlrd.open_workbook(str(path), on_demand=True)
    try:
        for name, cells in requested.items():
            sheet = book.sheet_by_name(name)
            for address in cells:
                m = CELL.fullmatch(address)
                if m is None:
                    raise ValueError('Invalid cell address: ' + address)
                column = 0
                for letter in m[1]:
                    column = column * 26 + ord(letter) - ord('A') + 1
                cell = sheet.cell(int(m[2]) - 1, column - 1)
                if cell.ctype != xlrd.XL_CELL_NUMBER:
                    raise ValueError('Mapped XLS cell is not numeric: ' + name + '!' + address)
                values[(name, address)] = float(cell.value)
    finally:
        book.release_resources()
    return values


def rebuild(source_map, raw_dir):
    files = source_map['source_files']
    required = {c['source_file_key'] for c in source_map['curves']}
    if len({c['curve_id'] for c in source_map['curves']}) != len(source_map['curves']):
        raise ValueError('Duplicate curve_id in source map')
    paths = {}
    for key, spec in files.items():
        if Path(key).name != key or '/' in key or '\\' in key or key in ('.', '..'):
            raise ValueError('Source file keys must be plain basenames')
        path = (raw_dir / key).resolve()
        if path.parent != raw_dir.resolve():
            raise ValueError('Source file escapes raw directory')
        if not path.exists() and key not in required:
            continue
        if not path.is_file():
            raise ValueError('Required source file is absent: ' + key)
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Source bytes/hash mismatch: ' + key)
        paths[key] = path
    # Verify all supplied/required source bytes before reading any measurements.
    needed = {}
    for c in source_map['curves']:
        key = c['source_file_key']
        if key not in files or c['file_sha256'] != files[key]['sha256']:
            raise ValueError('Curve/source hash binding mismatch')
        if len(c['gamma_cells']) != len(c['eta_cells']) or len(c['gamma_cells']) < 12:
            raise ValueError('Invalid mapped pair count')
        for address in c['gamma_cells'] + c['eta_cells']:
            if CELL.fullmatch(address) is None:
                raise ValueError('Invalid cell address')
        needed.setdefault(key, {}).setdefault(c['sheet'], set()).update(c['gamma_cells'] + c['eta_cells'])
    cache = {}
    for key, requested in needed.items():
        kind = files[key]['format']
        if kind == 'xlsx':
            cache[key] = xlsx_cells(paths[key], requested)
        elif kind == 'xls':
            cache[key] = xls_cells(paths[key], requested)
        else:
            raise ValueError('Unsupported mapped workbook type: ' + kind)
    curves = []
    for c in source_map['curves']:
        values = cache[c['source_file_key']]
        gamma = [float(values[(c['sheet'], a)]) for a in c['gamma_cells']]
        divisor = float(c['eta_divisor'])
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError('Invalid viscosity divisor')
        eta = [float(values[(c['sheet'], a)]) / divisor for a in c['eta_cells']]
        if not all(math.isfinite(x) and x > 0 for x in gamma + eta):
            raise ValueError('Rebuilt pair contains non-positive or non-finite value')
        if len(set(gamma)) != len(gamma) or gamma != sorted(gamma):
            raise ValueError('Mapped rates are not unique and ascending')
        if hashlib.sha256(vector_bytes(gamma, eta)).hexdigest() != c['expected_vector_sha256']:
            raise ValueError('Vector hash mismatch: ' + c['curve_id'])
        curves.append({k: c[k] for k in ('curve_id', 'group_id', 'source_doi')} |
                      {'gamma': gamma, 'eta': eta})
    return curves, len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-map', required=True, type=Path)
    parser.add_argument('--raw-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; refusing overwrite')
    source_map = json.loads(args.source_map.read_text(encoding='utf-8'))
    if source_map['schema_version'] != '1.0':
        raise ValueError('Unsupported source-map schema')
    curves, checked_files = rebuild(source_map, args.raw_dir)
    payload = {'schema_version': '1.0', 'rebuild_version': VERSION,
               'source_map_sha256': sha256(args.source_map), 'curves': curves}
    output = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(output)
    pairs = sum(len(c['gamma']) for c in curves)
    print(json.dumps({'version': VERSION, 'curves': len(curves), 'sources': len({c['source_doi'] for c in curves}),
                      'complete_native_pairs': pairs, 'six_anchor_pairs': 6 * len(curves),
                      'higher_rate_target_pairs': pairs - 6 * len(curves), 'verified_source_files': checked_files,
                      'vector_hashes_verified': len(curves), 'output_sha256': sha256(args.output)}))


if __name__ == '__main__':
    main()
