"""TA-export metadata only; stop exactly at StartOfData before any body."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    utc = dt.datetime.now(dt.timezone.utc).isoformat()
    plan = {'actual_utc': utc, 'scope': 'TA signal labels/units and selected specimen/program metadata before StartOfData',
            'numeric_response_cells_decoded': 0, 'fits': 0, 'stop_at': 'StartOfData'}
    with (ROOT / 'PCL_METADATA_ONLY_PLAN.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    prior = ROOT / 'INPUT_ONLY_RESULTS.json'
    correction = {'actual_utc': utc, 'prior_results_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
        'scope_correction': 'Earlier generic PCL locator inspected up to120 lines as text without stopping at StartOfData; measurement_body_read=False was inaccurate. No property numeric conversion or numeric body value output occurred. Its no-response-float claim remains; do not repeat its body-unread claim.',
        'action': 'Separate TA metadata-only reader now stops before measurement body; original prior result retained unchanged'}
    with (ROOT / 'PCL_HEADER_SCAN_LIMITATION.json').open('x', encoding='utf-8') as stream:
        json.dump(correction, stream, ensure_ascii=False, indent=2)
    result = []
    for path in sorted((ROOT / 'pcl_originals').glob('*.csv')):
        labels, inputs, stop = [], {}, None
        with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as stream:
            for line_no, line in enumerate(stream, 1):
                sep = max((',', ';', '\t'), key=line.count)
                row = next(csv.reader([line], delimiter=sep))
                if row and row[0] == 'StartOfData':
                    stop = line_no
                    break
                if row and re.fullmatch(r'Sig\d+', row[0]):
                    labels.append(row)
                if row and row[0] in ('Sample', 'Geometry', 'Size', 'Method', 'FreqTab', 'Module', 'Clamp'):
                    inputs.setdefault(row[0], []).append(row[1:])
        result.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'signal_labels_units': labels, 'specimen_program_inputs': inputs,
                       'stopped_before_body_at_line': stop, 'measurement_body_read': False,
                       'numeric_response_cells_decoded': 0})
    payload = {'actual_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'records': result,
               'prior_scope_correction': correction, 'numeric_response_cells_decoded': 0, 'fits': 0}
    with (ROOT / 'PCL_METADATA_ONLY_RESULTS.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    for row in result:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
