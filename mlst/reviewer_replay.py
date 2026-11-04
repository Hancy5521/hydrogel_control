"""One-command, standard-library replay of saved hydrogel score arithmetic.

Never imports model code or reconstructs/reads response arrays. Existing
output directories are refused. A success receipt is written only after both
child replays and fixed-input integrity checks succeed.
"""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

EXPECTED_BINDINGS_SHA256 = 'f8ec43507ec6ebb41d42440e18e264c899af5ad5da31e965125c7914f4a7e6b5'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def verify_bindings(repo_root, bindings_path, expected_digest=EXPECTED_BINDINGS_SHA256):
    """Verify a pinned subset without importing any bound program."""
    if file_sha256(bindings_path) != expected_digest:
        raise ValueError('Replay binding document changed')
    data = json.loads(Path(bindings_path).read_text(encoding='utf-8'))
    seen = set()
    root = Path(repo_root).resolve(strict=True)
    for item in data['bindings']:
        name = item['path']
        posix = PurePosixPath(name)
        if posix.is_absolute() or '..' in posix.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe replay binding path')
        if name.casefold() in seen:
            raise ValueError('Duplicate replay binding path')
        seen.add(name.casefold())
        p = root.joinpath(*posix.parts)
        if not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError('Missing or outside-checkout replay input: ' + name)
        if p.stat().st_size != item['bytes'] or file_sha256(p) != item['sha256']:
            raise ValueError('Replay input changed: ' + name)
    if not seen:
        raise ValueError('Empty replay bindings')
    return data


def reject_existing_output(output_dir):
    p = Path(output_dir)
    if p.exists() or p.is_symlink():
        raise ValueError('Output path already exists; choose a new directory')
    return p.resolve()


def invoke(script, arguments, transcript_path):
    # -B avoids bytecode writes to the checkout; -E/-S remove environment and
    # site-package contributions. All child programs use only stdlib.
    command = [sys.executable, '-B', '-E', '-S', str(script), *map(str, arguments)]
    started = datetime.now(timezone.utc)
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    finished = datetime.now(timezone.utc)
    # Store portable command descriptions without machine-specific prefixes.
    write_json(transcript_path, {
        'started_utc': started.isoformat(), 'finished_utc': finished.isoformat(),
        'entry': script.name, 'exit_code': run.returncode,
        'stdout': run.stdout, 'stderr': run.stderr,
    })
    if run.returncode:
        raise RuntimeError(script.name + ' failed; retained transcript, no success receipt')


def remap_outputs(output, uq):
    tables = output / 'mlst_tables'
    tables.mkdir()
    # Numbers changed during journal-focused editing; underlying scores did not.
    copies = {
        'main_table2.csv': 'main_table1.csv',
        'main_table3.csv': 'main_table3.csv',
        'SI_tableS11.csv': 'main_table2.csv',
    }
    for new_name, old_name in copies.items():
        shutil.copyfile(output / 'development' / old_name, tables / new_name)
    shutil.copyfile(output / 'external_uq' / 'SAVED_UQ_SOURCE_METHODS.csv', tables / 'SI_tableS12.csv')
    rows = uq['source_method_results']
    sources = list(dict.fromkeys(r['source_doi'] for r in rows))
    external = []
    for source in sources:
        own = {r['method']: r for r in rows if r['source_doi'] == source}
        cand = own['curvature_joint_gaussian']
        external.append({'source': cand['source'], 'native_curves': cand['native_curves'],
                         'analysis_groups': cand['analysis_groups'],
                         'candidate_CRPS': cand['crps_log'],
                         'GP_CRPS': own['reference_gp']['crps_log'],
                         'conditional_G_CRPS': own['conditional_gaussian']['crps_log'],
                         'candidate_coverage_percent': 100 * cand['coverage90']})
    with (tables / 'main_table4.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(external[0]))
        writer.writeheader(); writer.writerows(external)
    crosswalk = [
        {'item': 'Main Table 1', 'prior_item': 'New domain-role table',
         'replay': 'Author-reviewed data-role description; no numerical replay'},
        {'item': 'Main Table 2', 'prior_item': 'Original Main Table 1', 'replay': 'mlst_tables/main_table2.csv'},
        {'item': 'Main Table 3', 'prior_item': 'Original Main Table 3', 'replay': 'mlst_tables/main_table3.csv'},
        {'item': 'Main Table 4', 'prior_item': 'Original external source table', 'replay': 'mlst_tables/main_table4.csv'},
        {'item': 'SI Table S11', 'prior_item': 'Original Main Table 2', 'replay': 'mlst_tables/SI_tableS11.csv'},
        {'item': 'SI Table S12', 'prior_item': 'New saved-score UQ organization', 'replay': 'mlst_tables/SI_tableS12.csv'},
        {'item': 'Main Figure 3', 'prior_item': 'Original Main Figure 5',
         'replay': 'Original archived figure; scalar data in development/SI_tableS3.csv and development/SI_tableS5.csv'},
        {'item': 'Main Figure 4', 'prior_item': 'Original Main Figure 3',
         'replay': 'Original archived P0 coverage-failure figure; no measured-array access or numerical regeneration in this scalar-only entry'},
        {'item': 'SI Figure S5', 'prior_item': 'Original Main Figure 4',
         'replay': 'Original archived model-information figure; no numerical regeneration in this scalar-only entry'},
    ]
    write_json(output / 'TABLE_AND_FIGURE_CROSSWALK.json', crosswalk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = None
    created = False
    try:
        output = reject_existing_output(args.output_dir)
        root = args.repo_root.resolve(strict=True)
        bindings_path = Path(__file__).resolve().parent / 'REPLAY_INPUT_BINDINGS.json'
        bindings = verify_bindings(root, bindings_path)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        invoke(root / 'reproducibility/replay_tables.py',
               ['--input-dir', root / 'reproducibility', '--output-dir', output / 'development'],
               output / 'DEVELOPMENT_REPLAY_TRANSCRIPT.json')
        development = json.loads((output / 'development/validation.json').read_text(encoding='utf-8'))
        if development.get('status') != 'PASS' or any(development.get(k) != 0 for k in
                ('new_fits', 'new_forecasts', 'raw_target_scoring_calls')):
            raise RuntimeError('Development replay did not report successful scalar-only validation')
        invoke(root / 'mlst/saved_uq/replay_saved_uq.py',
               ['--repo-root', root, '--output-dir', output / 'external_uq'],
               output / 'EXTERNAL_UQ_REPLAY_TRANSCRIPT.json')
        uq = json.loads((output / 'external_uq/SAVED_UQ_RESULTS.json').read_text(encoding='utf-8'))
        if len(uq['source_method_results']) != 12 or uq['cohort'] != {
            'score_rows': 276, 'native_curves': 46, 'analysis_groups': 23,
            'sources': 4, 'native_targets': 781, 'probabilistic_methods_reported': 3}:
            raise RuntimeError('External saved-score replay cohort mismatch')
        verify_bindings(root, bindings_path)  # Detect input changes during replay.
        remap_outputs(output, uq)
        now = datetime.now(timezone.utc)
        result = {'status': 'PASS_SAVED_SCALAR_REPLAY', 'actual_utc': now.isoformat(),
                  'actual_asia_shanghai': now.astimezone(timezone(timedelta(hours=8))).isoformat(),
                  'script_sha256': file_sha256(__file__),
                  'base_commit': bindings['base_commit'],
                  'bindings_sha256': EXPECTED_BINDINGS_SHA256,
                  'verified_input_files': len(bindings['bindings']),
                  'development_validation': development,
                  'external_source_method_rows': 12,
                  'new_fits': 0, 'new_forecasts': 0, 'raw_target_scoring_calls': 0,
                  'scope': 'Only recorded scalar arithmetic and table remapping; no raw-data reconstruction, inference, calibration or new material evidence',
                  'limits': ['Deployment weights are not original out-of-fold models',
                             'Some original processed inputs and complete scientific-study replay remain separately scoped',
                             'This entry is not a complete raw-data reproduction of the material experiments',
                             'Manuscript figure rendering and author domain-role descriptions are not regenerated here'],
                  'output_files': [{'path': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size,
                                    'sha256': file_sha256(p)}
                                   for p in sorted(output.rglob('*')) if p.is_file()]}
        write_json(output / 'REVIEWER_REPLAY_RECEIPT.json', result)
        print(json.dumps({'status': result['status'], 'verified_input_files': result['verified_input_files'],
                          'new_fits': 0, 'new_forecasts': 0, 'raw_target_scoring_calls': 0}))
        return 0
    except Exception as exc:
        if created:
            write_json(output / 'REVIEWER_REPLAY_FAILURE.json', {
                'status': 'FAILED_NO_SUCCESS_RECEIPT', 'actual_utc': datetime.now(timezone.utc).isoformat(),
                'error_type': type(exc).__name__, 'error': str(exc),
                'scope': 'Partial scalar-replay outputs retained for inspection, not successful reproduction'})
        print(json.dumps({'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
