"""Independent DaRUS coordinate/header gate. Numeric Gprime/Gdoubleprime0."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

EXPECTED_COUNTS = {'30': 33, '50': 25, '70': 30, 'dry': 27, 'redry': 27, 'wet': 25}
WHITELIST = ['Result No.', 'Interval No.', 'Point No.', 'Frequency', 'Angular Frequency',
             'Temperature', 'Gap', 'Test Time', 'Shear Strain', 'Relative Humidity']
RESPONSE_NAMES = ['Storage Modulus', 'Loss Modulus', 'Complex Viscosity',
                  'Complex Shear Modulus', 'Loss Factor', 'Shear Stress', 'Torque']
GRID = [.01, .0178, .0316, .0562, .1, .178, .316, .562, 1., 1.78, 3.16, 5.62, 10.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--previous-input-results', required=True)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.source_directory)
    auth = json.loads(Path(args.authorization).read_text(encoding='utf-8'))
    assert auth['scope'] == 'darus_independent_header_input_only'
    assert auth['source_sha256'] == sha(__file__)
    assert auth['metadata_sha256'] == sha(args.metadata)
    assert auth['previous_input_results_sha256'] == sha(args.previous_input_results)
    metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8'))['data']['latestVersion']
    assert (metadata['versionNumber'], metadata['versionMinorNumber']) == (1, 1)
    assert metadata['license']['uri'] == 'http://creativecommons.org/licenses/by/4.0'
    previous = json.loads(Path(args.previous_input_results).read_text(encoding='utf-8'))
    old = {Path(r['file']).name: r for r in previous['darus']}
    results = []; comparisons = []; maxima = []
    for state, expected_count in EXPECTED_COUNTS.items():
        file = root/('frequency_sweep_SMP_'+state+'.csv')
        assert sha(file) == auth['original_sha256'][file.name]
        with file.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream); header = next(reader); units = next(reader)
            mapping = dict(zip(header, units))
            assert mapping['Storage Modulus'] == mapping['Loss Modulus'] == '[MPa]'
            assert mapping['Frequency'] == '[Hz]' and mapping['Angular Frequency'] == '[rad/s]'
            assert mapping['Temperature'] == '[C]'
            indices = {k: header.index(k) for k in WHITELIST if k in header}
            groups = {}
            for row in reader:
                if not row or not any(v.strip() for v in row):
                    raise ValueError('Unexpected empty source row')
                values = {k: float(row[i]) for k, i in indices.items()}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite input')
                # G/stress/torque/viscosity columns are not indexed or decoded.
                result, interval = values['Result No.'], values['Interval No.']
                assert result.is_integer() and interval.is_integer() and values['Point No.'].is_integer()
                key = (int(result), int(interval))
                groups.setdefault(key, []).append({'source_line_1based': reader.line_num,
                    'point_no': int(values['Point No.']),
                    'frequency_Hz': values['Frequency'], 'reported_angular_frequency_rad_per_s': values['Angular Frequency'],
                    'computed_angular_frequency_rad_per_s': 2*math.pi*values['Frequency'],
                    'temperature_C': values['Temperature'], 'shear_strain_percent': values['Shear Strain'],
                    'gap_mm': values['Gap'], 'test_time_h': values['Test Time'],
                    'relative_humidity_percent': values.get('Relative Humidity')})
        assert len(groups) == expected_count
        for (result, interval), rows in groups.items():
            assert len(rows) == 13 and [r['frequency_Hz'] for r in rows] == GRID
            assert len({r['source_line_1based'] for r in rows}) == 13
            assert len({r['point_no'] for r in rows}) == 13
            previous_key = str((float(result), float(interval)))
            prior = old[file.name]['groups'][previous_key]
            assert prior['frequencies'] == GRID and prior['temperatures'] == [r['temperature_C'] for r in rows]
            temperatures = [r['temperature_C'] for r in rows]
            omega_error = max(abs(r['reported_angular_frequency_rad_per_s']/r['computed_angular_frequency_rad_per_s']-1) for r in rows)
            maxima.append(omega_error)
            results.append({'condition_id': state+'|result'+str(result)+'|interval'+str(interval),
                'state_block': state, 'result_no': result, 'interval_no': interval,
                'file': file.name, 'source_sha256': sha(file), 'native_rows': rows,
                'native_frequency_points': 13, 'temperature_range_C': [min(temperatures), max(temperatures)],
                'temperature_drift_span_C': max(temperatures)-min(temperatures),
                'reported_omega_relative_rounding_difference_max': omega_error,
                'numeric_response_cells_decoded': 0})
        comparisons.append({'state_block': state, 'curves': len(groups),
                            'source_sha256': sha(file), 'independent_input_values_match_prior': True})
    assert len(results) == 167 and len({r['condition_id'] for r in results}) == 167
    now = dt.datetime.now(dt.timezone.utc)
    output = {'schema': 'darus-functional-response-independent-input-gate-v1',
        'status': 'PASS_SOURCE_UNITS_INPUT_ONLY_NOT_INDEPENDENT_SPECIMEN_CERTIFICATION',
        'actual_utc': now.isoformat(),
        'asia_shanghai': now.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        'source_sha256': sha(__file__), 'authorization_sha256': sha(args.authorization),
        'source_doi': '10.18419/DARUS-2021', 'source_version': '1.1', 'license': 'CC BY 4.0',
        'source_response_units': 'Storage Modulus and Loss Modulus MPa',
        'exact_response_conversion_when_authorized': 'G_Pa = 1000000 * G_MPa, both channels; no conversion executed',
        'input_frequency_conversion': 'omega_rad_per_s = 2*pi*frequency_Hz; not shear rate gamma',
        'reported_omega_kept_as_original_input': True,
        'reported_omega_relative_rounding_difference_max': max(maxima),
        'commercial_material_count': 1, 'humidity_or_history_file_blocks': 6,
        'temperature_condition_curves': 167, 'native_frequency_coordinates': 2171,
        'independent_specimen_or_batch_count': None,
        'conservative_material_group': 'one commercial polyurethane SMP; all state blocks potentially related',
        'wet_redry_history_relationship_not_independence': True,
        'numeric_Gprime_or_Gdoubleprime_cells_decoded': 0, 'numeric_other_response_cells_decoded': 0,
        'fits': 0, 'production_imports': 0, 'new_material_effect': False,
        'source_method_temperature_overview_C': [5, 85],
        'actual_input_temperature_range_C': [min(r['temperature_range_C'][0] for r in results), max(r['temperature_range_C'][1] for r in results)],
        'source_overview_vs_recorded_input_discrepancy_preserved': True,
        'counts_by_file': comparisons, 'condition_curves': results,
        'physical_state_caveats': ['Humidity blocks and temperature sweeps are not new chemical formulations',
            'Unknown common specimen/lot across files; retain all within one material group',
            'Temperature is actually recorded at each frequency; no silent nominal rounding',
            'Mastercurves and alpha_T are derived, excluded from independent native labels',
            'Complex modulus/loss factor/complex viscosity are derived or coupled, not extra independent property labels']}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': output['status'], 'curves': 167, 'source_numeric_moduli': 0,
                      'output': args.output, 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
