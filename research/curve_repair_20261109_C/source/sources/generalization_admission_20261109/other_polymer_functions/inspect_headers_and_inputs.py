"""Whitelisted coordinate/identity admission; never parse numeric G/stress Y."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
INPUTS = {'Result No.', 'Interval No.', 'Point No.', 'Angular Frequency', 'Frequency',
          'Temperature', 'Relative Humidity', 'Gap', 'Test Time', 'Shear Strain'}
RESPONSES = {'Complex Viscosity', 'Storage Modulus', 'Loss Modulus', 'Complex Shear Modulus',
             'Loss Factor', 'Shear Stress', 'Torque'}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def non_numeric_text(cell):
    # No numeric conversion: redact any field starting as a numeric token.
    return '<numeric-redacted>' if re.match(r'^\s*[\"\s]*[+\-]?(?:\d|\.\d)', cell) else cell


def darus(path):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        line = stream.readline()
        sep = max((',', ';', '\t'), key=line.count)
        header = next(csv.reader([line], delimiter=sep))
        if not INPUTS.intersection(header) or not RESPONSES.issubset(header):
            raise ValueError('Unexpected registered frequency-sweep schema')
        indices = {name: header.index(name) for name in INPUTS if name in header}
        reader = csv.reader(stream, delimiter=sep)
        first = next(reader)
        if any('[' in cell or ']' in cell for cell in first):
            units = dict(zip(header, first))
            units = {name: token for name, token in units.items() if re.fullmatch(r'\s*\[.*\]\s*', token)}
            rows = reader
        else:
            units = {}
            import itertools
            rows = itertools.chain([first], reader)
        groups = {}
        rejected = 0
        for row in rows:
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                # ONLY identity/coordinate fields; response cells are not
                # indexed, converted, summarized, printed or stored.
                values = {name: float(row[index]) for name, index in indices.items()}
            except (ValueError, IndexError):
                rejected += 1
                continue
            group = (values['Result No.'], values['Interval No.'])
            record = groups.setdefault(str(group), {'input_rows': 0, 'frequencies': [], 'angular_frequencies': [],
                'temperatures': [], 'relative_humidities': [], 'shear_strain_inputs': [], 'point_numbers': []})
            record['input_rows'] += 1
            for label, name in [('frequencies', 'Frequency'), ('angular_frequencies', 'Angular Frequency'),
                                ('temperatures', 'Temperature'), ('relative_humidities', 'Relative Humidity'),
                                ('shear_strain_inputs', 'Shear Strain'), ('point_numbers', 'Point No.')]:
                if name in values:
                    record[label].append(values[name])
        return {'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'header': header, 'unit_labels': units,
                'input_fields_read': sorted(indices), 'response_fields_never_numeric_decoded': sorted(RESPONSES),
                'non_numeric_or_irregular_input_rows': rejected, 'groups': groups}


def pcl_header(path):
    # Instruments prepend metadata. Seek a row with several conventional field
    # labels and stop BEFORE its numeric measurement body.
    found, safe_metadata_names = None, []
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as stream:
        for line_no in range(1, 121):
            line = stream.readline()
            if not line:
                break
            sep = max((',', ';', '\t'), key=line.count)
            row = next(csv.reader([line], delimiter=sep))
            joined = ' '.join(row).lower()
            indicators = sum(word in joined for word in ['time', 'strain', 'stress', 'force', 'temperature', 'modulus', 'storage', 'loss'])
            if indicators >= 3:
                found = {'source_line_1based': line_no, 'delimiter': repr(sep),
                         'field_tokens': [non_numeric_text(cell) for cell in row]}
                break
            # Store label cells only; metadata values can include response
            # summaries, so no right-hand cells or numeric values are exposed.
            if row and re.match(r'^[A-Za-z][A-Za-z /()_.-]{1,50}$', row[0]):
                safe_metadata_names.append(row[0])
    return {'file': str(path.relative_to(ROOT)), 'sha256': sha(path), 'header_found': found,
            'metadata_labels_only': safe_metadata_names, 'numeric_response_cells_decoded': 0,
            'measurement_body_read': False}


def main():
    plan = {'actual_utc': utc(), 'scope': 'DaRUS names/units plus input-only Result/Interval/Frequency/Temperature; PCL header labels only',
            'darus_numeric_whitelist': sorted(INPUTS), 'numeric_response_cells_decoded': 0, 'fits': 0,
            'curve_groups_provisional': 'Result/Interval groups are condition curves, not certified independent specimens'}
    with (ROOT / 'INPUT_ONLY_PLAN.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    results = [darus(path) for path in sorted((ROOT / 'darus_smp_originals').glob('*.csv'))]
    pcl = [pcl_header(path) for path in sorted((ROOT / 'pcl_originals').glob('*.csv'))]
    output = {'actual_utc': utc(), 'darus': results, 'pcl_headers': pcl,
              'numeric_response_cells_decoded': 0, 'fits': 0}
    with (ROOT / 'INPUT_ONLY_RESULTS.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    for row in results:
        print(json.dumps({'file': row['file'], 'units': row['unit_labels'], 'condition_groups': len(row['groups']),
            'points_per_group': sorted(set(g['input_rows'] for g in row['groups'].values())),
            'non_numeric_input_rows': row['non_numeric_or_irregular_input_rows']}, ensure_ascii=False))
    for row in pcl:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
