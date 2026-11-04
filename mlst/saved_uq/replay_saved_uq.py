"""Organize archived score scalars; never imports models or reads response arrays.

Run from a checkout with --repo-root .; the pinned source is required. Existing
output directories are refused. The result is retrospective description of
already consumed curves, not calibration or independent validation.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path

RELATIVE_SCORES = Path('external_validation/20261101T1239423181800Z/SCORES.json')
EXPECTED_SHA256 = '555f39ca076351b063511940f7b2a2a011e47ec9a55b0ce968c905e5f9b8c8d0'
METHODS = ('curvature_joint_gaussian', 'reference_gp', 'conditional_gaussian')
ALL_METHODS = frozenset((*METHODS, 'pchip_log', 'powerlaw_6anchors', 'powerlaw_last2'))
SOURCES = {
    '10.17638/datacat.liverpool.ac.uk/1296': ('Liverpool inks', 15, 5),
    '10.1371/journal.pone.0162849': ('HA and CMHA', 10, 8),
    '10.1016/j.matt.2023.01.011': ('Granular hydrogels', 16, 5),
    '10.34894/HBKOLJ': ('Dynamic alginate', 5, 5),
}
METRICS = ('crps_log', 'mse_log', 'coverage90', 'width90_log',
           'interval_score90_log', 'marginal_nll_log')


def reject(message):
    raise ValueError(message)


def validate_rows(rows, expected_cohort=True):
    """Validate recorded identities and scalar schema before aggregation."""
    if not isinstance(rows, list) or not rows:
        reject('Expected a nonempty score_rows list')
    by_curve = defaultdict(dict)
    source_groups = defaultdict(set)
    source_curves = defaultdict(set)
    group_source = {}
    for r in rows:
        if not isinstance(r, dict):
            reject('Score row must be an object')
        for k in ('curve_id', 'group_id', 'source_doi', 'method'):
            if not isinstance(r.get(k), str) or not r[k]:
                reject('Missing or invalid identity: ' + k)
        curve, group, source, method = (r[k] for k in ('curve_id', 'group_id', 'source_doi', 'method'))
        if source not in SOURCES or method not in ALL_METHODS:
            reject('Unregistered source or method')
        if method in by_curve[curve]:
            reject('Duplicate curve-method row: ' + curve + '/' + method)
        if group in group_source and group_source[group] != source:
            reject('Analysis group spans different sources')
        group_source[group] = source
        score = r.get('scores')
        if not isinstance(score, dict) or not score:
            reject('Missing scalar score object')
        # Disallow nested containers explicitly: this program has no response-
        # array or forecast path, including in its supplied fixtures.
        for key, value in score.items():
            if isinstance(value, (dict, list, tuple)):
                reject('Nonscalar score field: ' + key)
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                reject('Nonfinite or nonnumeric score field: ' + key)
        n = score.get('n_targets')
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            reject('Invalid target count')
        if not isinstance(score.get('mse_log'), (float, int)) or score['mse_log'] < 0:
            reject('Invalid mean squared error')
        if method in METHODS:
            if not all(k in score for k in (*METRICS, 'whole_curve_covered90')):
                reject('Missing probability score fields')
            cov = score['coverage90']
            if isinstance(cov, bool) or not 0 <= cov <= 1:
                reject('Coverage outside [0,1]')
            whole = score['whole_curve_covered90']
            if type(whole) is not bool or whole != (cov == 1.0):
                reject('Whole-band indicator disagrees with saved marginal coverage')
            if any(score[k] < 0 for k in ('crps_log', 'width90_log', 'interval_score90_log')):
                reject('Negative loss or width')
            if score['interval_score90_log'] + 1e-12 < score['width90_log']:
                reject('Interval score is smaller than interval width')
        by_curve[curve][method] = r
        source_curves[source].add(curve)
        source_groups[source].add(group)
    for curve, methods in by_curve.items():
        if set(methods) != ALL_METHODS:
            reject('Incomplete method set for curve: ' + curve)
        identities = {(r['source_doi'], r['group_id'], r['scores']['n_targets']) for r in methods.values()}
        if len(identities) != 1:
            reject('Methods disagree on source, group or target count: ' + curve)
    if expected_cohort:
        if len(rows) != 276 or len(by_curve) != 46:
            reject('Expected original 276 score rows on 46 native curves')
        if set(source_curves) != set(SOURCES):
            reject('Expected all four original sources')
        for source, (_, curves, groups) in SOURCES.items():
            if len(source_curves[source]) != curves or len(source_groups[source]) != groups:
                reject('Original source cohort identity/count mismatch: ' + source)
        if sum(next(iter(methods.values()))['scores']['n_targets'] for methods in by_curve.values()) != 781:
            reject('Original 781 target positions not retained')
    return by_curve


def summarize(rows, expected_cohort=True):
    by_curve = validate_rows(rows, expected_cohort=expected_cohort)
    output = []
    for source, (label, _, _) in SOURCES.items():
        present = [r for methods in by_curve.values() for method, r in methods.items()
                   if method in METHODS and r['source_doi'] == source]
        if not present:
            continue
        for method in METHODS:
            rs = [r for r in present if r['method'] == method]
            groups = defaultdict(list)
            for r in rs:
                groups[r['group_id']].append(r['scores'])
            group_means = [{key: math.fsum(s[key] for s in ss) / len(ss) for key in METRICS}
                           for _, ss in sorted(groups.items())]
            macro = {key: math.fsum(g[key] for g in group_means) / len(group_means) for key in METRICS}
            macro['root_group_macro_mse_log'] = math.sqrt(macro.pop('mse_log'))
            coverages = [r['scores']['coverage90'] for r in rs]
            output.append({
                'source': label, 'source_doi': source, 'method': method,
                'native_curves': len(rs), 'analysis_groups': len(groups),
                'native_targets': sum(r['scores']['n_targets'] for r in rs),
                **macro,
                'zero_coverage_native_curves': sum(c == 0.0 for c in coverages),
                'all_targets_in_marginal_bands_native_curves': sum(r['scores']['whole_curve_covered90'] for r in rs),
                'native_curve_coverage_min': min(coverages),
                'native_curve_coverage_max': max(coverages),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    path = args.repo_root / RELATIVE_SCORES
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        reject('Saved score bytes do not match the frozen source SHA256')
    source = json.loads(raw)
    values = summarize(source['score_rows'])
    now = datetime.now(timezone.utc)
    result = {
        'actual_utc': now.isoformat(),
        'actual_asia_shanghai': now.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'classification': 'Descriptive reorganization of previously consumed curve-level score scalars',
        'input_relative_path': RELATIVE_SCORES.as_posix(),
        'input_sha256': digest,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_protocol_sha256': source['protocol_sha256'],
        'original_prediction_sha256': source['prediction_sha256'],
        'cohort': {'score_rows': 276, 'native_curves': 46, 'analysis_groups': 23,
                   'sources': 4, 'native_targets': 781, 'probabilistic_methods_reported': 3},
        'weighting': 'Within each source, average curves within analysis group, then groups equally; no pooling across sources',
        'counting': 'Zero/all-target band counts are native-curve descriptions, not group-weighted rates or independent trial counts',
        'limits': ['No new prediction, target scoring, fitting, calibration, threshold selection or independent validation',
                   'Nominal 90 percent marginal-band inclusion of all targets is not a simultaneous 90 percent coverage guarantee',
                   'Curves and analysis groups are not certified independent preparation batches',
                   'Reported means and related runs retain their original roles',
                   'Small widths describe sharpness but are useful only together with calibration'],
        'source_method_results': values,
    }
    # Exclusive new directory protects every old result and this replay output.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'SAVED_UQ_RESULTS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.output_dir / 'SAVED_UQ_SOURCE_METHODS.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    print(json.dumps({'status': 'SAVED_SCALAR_REPLAY_COMPLETE', 'result_rows': len(values),
                      'input_sha256': digest, 'material_fit_forecast_targetscore_calls': 0,
                      'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
