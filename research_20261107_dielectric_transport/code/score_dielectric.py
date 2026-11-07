"""One first scoring pass over fully frozen new-source predictive laws.

All scores remain native dimensionless permittivity. Same-source sequential
batches do not become three independent campaigns. No fit, retuning or split.
"""
import argparse
from bisect import bisect_right
from fractions import Fraction
import math
from pathlib import Path
from scipy.special import ndtri
import task_io as io
from empirical_native import canonical, crps

LEVELS = (('0.8', Fraction(1, 10), Fraction(9, 10)),
          ('0.9', Fraction(1, 20), Fraction(19, 20)),
          ('0.95', Fraction(1, 40), Fraction(39, 40)))


def score_law(y, law):
    y = float(y)
    if not math.isfinite(y) or law.get('unit') != io.UNIT:
        raise ValueError('finite true target and source-correct native unit required')
    if law['family'] == 'weighted_empirical':
        value = canonical(law)
        x, w = value['support'], value['mass']
        def quantile(p):
            total = Fraction(0)
            for support, mass in zip(x, w):
                total += mass
                if total >= p: return support
            raise AssertionError('unit exact mass must reach quantile')
        score = crps(law, y)
        left = sum((mass for support, mass in zip(x, w) if support < y), Fraction(0))
        right = sum((mass for support, mass in zip(x, w) if support <= y), Fraction(0))
        pit_left, pit_right = float(left), float(right)
        pit_kind = 'discrete_CDF_interval_not_randomized_uniform_sample'
    elif law['family'] == 'normal':
        mu, sd = float(law['location']), float(law['scale'])
        if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 0:
            raise ValueError('finite normal with positive scale required')
        z = (y - mu) / sd
        pit_left = pit_right = .5 * (1 + math.erf(z / math.sqrt(2)))
        density = math.exp(-.5 * z * z) / math.sqrt(2 * math.pi)
        score = sd * (z * (2 * pit_right - 1) + 2 * density - 1 / math.sqrt(math.pi))
        quantile = lambda p: mu + sd * float(ndtri(float(p)))
        pit_kind = 'continuous_normal_CDF'
    else:
        raise ValueError('only registered empirical or normal predictive families allowed')
    median = quantile(Fraction(1, 2))
    intervals = {}
    for level, lo, hi in LEVELS:
        a, b = quantile(lo), quantile(hi)
        intervals[level] = {'lower': a, 'upper': b, 'width': b - a, 'covered': a <= y <= b}
    if not math.isfinite(score) or score < -1e-10:
        raise ArithmeticError('nonfinite/negative CRPS')
    return {'CRPS_native': score, 'squared_error_native': (median - y) ** 2,
            'predictive_median_native': median, 'intervals': intervals,
            'PIT_left': pit_left, 'PIT_right': pit_right, 'PIT_midpoint': .5 * (pit_left + pit_right),
            'PIT_kind': pit_kind, 'log_density_native': None,
            'density_reason': 'atomic and continuous laws compared by proper CRPS; NLL omitted'}


def aggregate(rows):
    n = len(rows)
    return {'groups': n, 'CRPS_native_mean': math.fsum(r['CRPS_native'] for r in rows) / n,
            'RMSE_median_native': math.sqrt(math.fsum(r['squared_error_native'] for r in rows) / n),
            'coverage': {level: sum(r['intervals'][level]['covered'] for r in rows) / n for level, _, _ in LEVELS},
            'width_native_mean': {level: math.fsum(r['intervals'][level]['width'] for r in rows) / n
                                  for level, _, _ in LEVELS}}


def exact_envelope_at(envelope, y):
    knots = [Fraction.from_float(float(v)) for v in envelope['knots_native']]
    values = [Fraction(v) for v in envelope['knot_gaps_exact_fraction']]
    y = Fraction.from_float(float(y))
    index = bisect_right(knots, y) - 1
    if index < 0: result = values[0]
    elif index >= len(knots) - 1: result = values[-1]
    else:
        t = (y - knots[index]) / (knots[index + 1] - knots[index])
        result = (1 - t) * values[index] + t * values[index + 1]
    if not Fraction(envelope['lower_exact_fraction']) <= result <= Fraction(envelope['upper_exact_fraction']):
        raise AssertionError('observed exact gap outside pretarget envelope')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = io.validate(args.plan, args.approval, 'score_saved_batches')
    io.load_roles(plan)
    frozen = io.read(args.forecast_dir / 'ALL_BATCHES_FROZEN.json')
    if (frozen['status'] != 'ALL1344_PREQUENTIAL_LAWS_AND192_TRUE_OUTCOMES_COMPLETE'
            or frozen['source_sha256'] != io.sources() or frozen['plan_sha256'] != io.sha(args.plan)):
        raise ValueError('complete exact registered predictions before first score required')
    io.verify_payloads(args.forecast_dir, frozen['files_sha256'])
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    io.write(folder / 'START.json', {'actual_utc': io.utc(), 'source_sha256': io.sources(),
             'plan_sha256': io.sha(args.plan), 'all_batch_freeze_sha256': io.sha(args.forecast_dir / 'ALL_BATCHES_FROZEN.json')})
    try:
        records, checks, anomalies = [], [], []
        for batch_number, role in enumerate(io.BATCHES, 1):
            batch = args.forecast_dir / role
            truth = io.read(batch / 'TRUE_BATCH_REVEAL.json')['rows']
            lookup = {r['identity']['component_id']: r['native_target'] for r in truth}
            forecasts = io.read(batch / 'FORECASTS.json')['records']
            expected_ids = set(lookup)
            if len(truth) != 64 or len(expected_ids) != 64 or len(forecasts) != 448:
                raise ValueError('complete64 truths and448 forecasts required')
            scores = {}
            for row in forecasts:
                method, component = row['method'], row['identity']['component_id']
                if method not in io.METHODS or component not in expected_ids or (method, component) in scores:
                    raise ValueError('unique registered method/component prediction required')
                y = lookup[component]
                result = score_law(y, row['law'])
                scores[(method, component)] = result['CRPS_native']
                records.append({'batch': batch_number, 'method': method, 'component_id': component,
                                'native_target': y, 'score': result})
                if method == 'RF_identity':
                    electronic = row['known_native_proxies'][1]
                    anomalies.append({'component_id': component, 'batch': batch_number,
                                      'total_below_electronic': y < electronic,
                                      'total_minus_electronic_native': y - electronic,
                                      'not_removed_or_used_for_eligibility': True})
            for row in io.read(batch / 'PRETARGET_LOSS_GAP_ENVELOPES.json')['records']:
                component, method = row['component_id'], row['method']
                exact = exact_envelope_at(row['loss_gap_envelope'], lookup[component])
                floating = scores[(method, component)] - scores[('RF_identity', component)]
                error = abs(floating - float(exact))
                if error > 1e-9 + 1e-11 * max(abs(floating), abs(float(exact))):
                    raise ArithmeticError('CRPS subtraction disagrees with exact saved-law envelope')
                checks.append({'batch': batch_number, 'method': method, 'component_id': component,
                               'exact_observed_loss_gap_fraction': str(exact), 'floating_CRPS_subtraction_error': error})
        if len(records) != 1344 or len(checks) != 768 or len(anomalies) != 192:
            raise ValueError('all registered score/gap/anomaly positions required')
        summaries = {method: {str(batch): aggregate([r['score'] for r in records
                     if r['method'] == method and (batch == 0 or r['batch'] == batch)])
                     for batch in (0, 1, 2, 3)} for method in io.METHODS}
        candidate = {r['component_id']: r['score']['CRPS_native'] for r in records if r['method'] == 'RF_auto4'}
        paired = {}
        for method in io.METHODS:
            if method == 'RF_auto4': continue
            other = {r['component_id']: r['score']['CRPS_native'] for r in records if r['method'] == method}
            differences = [candidate[key] - other[key] for key in candidate]
            mean = math.fsum(differences) / 192
            paired[method] = {'groups': 192, 'candidate_minus_other_CRPS_mean': mean,
                             'standard_error_descriptive_only': math.sqrt(math.fsum((v - mean) ** 2 for v in differences) / 191 / 192),
                             'better': sum(v < 0 for v in differences), 'worse': sum(v > 0 for v in differences),
                             'exact_ties': sum(v == 0 for v in differences), 'all_differences': differences,
                             'not_formal_independent_campaign_confidence_interval': True}
        selected = io.read(args.forecast_dir / 'DESCRIPTOR_SELECTION_FROZEN.json')
        io.write(folder / 'RESULTS.json', {'status': 'COMPLETE_DIELECTRIC_SOURCE_REUSE_FRESH192', 'actual_utc': io.utc(),
                 'source_sha256': io.sources(), 'plan_sha256': io.sha(args.plan),
                 'all_batch_freeze_sha256': io.sha(args.forecast_dir / 'ALL_BATCHES_FROZEN.json'),
                 'summaries': summaries, 'paired_RF_auto4_minus_controls': paired, 'scores': records,
                 'observed_loss_gap_envelope_checks': checks, 'decomposition_diagnostics': anomalies,
                 'selected_descriptor_index': selected['auto_index'], 'anchor_descriptor_index': selected['anchor_index'],
                 'initial_training_groups': 256, 'prequential_validation_groups': 192, 'score_coordinates': 1344,
                 'initial_training_report_cells': 768, 'future_report_cells': 576, 'actual_sklearn_fits': 11,
                 'source_task_refit_not_old_model_zero_shot': True, 'same_source_three_batches_not_independent_campaigns': True,
                 'all_report_values_computed_not_new_wet_or_DFT_measurements': True,
                 'logical_disclosure_schedule_not_actual_experiment_saving_proof': True,
                 'support_preserving_scalar_repair_not_arbitrary_mask_joint_generator': True,
                 'electronic_delta_is_naive_no_ionic_sanity_control_not_primary_strong_baseline': True})
        print({m: summaries[m]['0']['CRPS_native_mean'] for m in io.METHODS})
    except Exception as problem:
        io.write(folder / 'FAILED_STOP.json', {'actual_utc': io.utc(), 'type': type(problem).__name__,
                 'message': str(problem), 'source_sha256': io.sources(), 'no_refit_or_retune': True})
        raise


if __name__ == '__main__': main()
