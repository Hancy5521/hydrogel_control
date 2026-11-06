"""First terminalY revelation only after every policy, reference and process law freezes."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import data_io as task


def ecdf(values):
    points = sorted(values); n = len(points)
    return max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(points))


def aggregate(records):
    result = {'formula_groups': len(records),
        'mean_CRPS_native': float(np.mean([r['CRPS_native'] for r in records])),
        'RMSE_predictive_median_native': float(math.sqrt(np.mean([r['squared_error_native'] for r in records]))),
        'coverage': {}, 'mean_interval_width_native': {},
        'PIT_ECDF_deviation_descriptive': None if any('discrete' in r['PIT_kind'] for r in records)
            else ecdf([r['PIT_midpoint'] for r in records]),
        'mean_PIT_left': float(np.mean([r['PIT_left'] for r in records])),
        'mean_PIT_right': float(np.mean([r['PIT_right'] for r in records]))}
    for level in ('0.8', '0.9', '0.95'):
        result['coverage'][level] = float(np.mean([r['intervals'][level]['covered'] for r in records]))
        result['mean_interval_width_native'][level] = float(np.mean([r['intervals'][level]['width'] for r in records]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(); plan, roles = task.validate(args.plan, args.approval, 'score_after_freeze')
    primary, references, prequential, freeze = task.verify_freeze(args.forecast_dir, args.plan, roles)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    task.write(output / 'TERMINAL_Y_REVEAL_START.json', {'actual_utc': task.utc(), 'source_sha256': task.sources(),
        'plan_sha256': task.sha(args.plan), 'approval_sha256': task.sha(args.approval),
        'forecast_freeze_sha256': task.sha(Path(args.forecast_dir) / 'FORECAST_FREEZE.json'),
        'all2112terminal_and48prequential_payloads_verified_before_terminalY': True})
    try:
        truth = task.reveal_terminal_truth(plan, roles, args.forecast_dir, args.plan)
        task.write(output / 'TERMINAL_TRUTH.json', {'scope': 'new64terminaldevelopment', 'truth': truth})
        targets = {r['identity']['component_id']: r['native_target'] for r in truth}
        frozen_score = task.score_module(plan); cache = {}; rows = []; reference_rows = []
        def evaluate(y, law):
            key = (y, json.dumps(law, sort_keys=True, separators=(',', ':'), allow_nan=False))
            if key not in cache: cache[key] = frozen_score.score_law(y, law)
            return cache[key]
        for item in primary:
            rows.append({key: item[key] for key in ('component_id', 'jid', 'policy', 'budget')} |
                        {'native_target': targets[item['component_id']], 'score': evaluate(targets[item['component_id']], item['law'])})
        for item in references:
            reference_rows.append({key: item[key] for key in ('component_id', 'jid', 'policy', 'budget', 'method')} |
                        {'native_target': targets[item['component_id']], 'score': evaluate(targets[item['component_id']], item['law'])})
        summary = {}; reference_summary = {}
        for policy in task.POLICIES:
            summary[policy] = {}
            for budget in task.BUDGETS:
                subset = [r['score'] for r in rows if r['policy'] == policy and r['budget'] == budget]
                if len(subset) != 64: raise AssertionError('all64terminalrequired')
                summary[policy][str(budget)] = aggregate(subset)
        for policy in task.REF_POLICIES + ('gp_variance',):
            reference_summary[policy] = {}
            for method in (('GP', 'RF') if policy in task.REF_POLICIES else ('GP',)):
                reference_summary[policy][method] = {}
                for budget in task.BUDGETS:
                    subset = [r['score'] for r in reference_rows if r['policy'] == policy and r['method'] == method and r['budget'] == budget]
                    if len(subset) != 64: raise AssertionError('all64referenceforecastsrequired')
                    reference_summary[policy][method][str(budget)] = aggregate(subset)
        lookup = {(r['component_id'], r['policy'], r['budget']): r['score']['CRPS_native'] for r in rows}
        ref_lookup = {(r['component_id'], r['policy'], r['method'], r['budget']): r['score']['CRPS_native'] for r in reference_rows}
        paired = [{'component_id': group,
            'native_student_minus_other_NIG_policies': {policy: {str(n): lookup[(group, 'native_student', n)] - lookup[(group, policy, n)]
                                                              for n in task.BUDGETS} for policy in task.POLICIES if policy != 'native_student'},
            'native_student_final_minus_GP_active_GP_prediction': lookup[(group, 'native_student', 8)] - ref_lookup[(group, 'gp_variance', 'GP', 8)],
            'native_student_own_final_minus_warm': lookup[(group, 'native_student', 8)] - lookup[(group, 'native_student', 0)]}
                  for group in targets]
        directory = Path(args.forecast_dir); secondary = []
        for item in prequential:
            before = task.read(directory / item['forecast_file']); after = task.read(directory / item['outcome_file'])
            secondary.append({'policy': item['policy'], 'step': item['step'], 'pool_index': item['pool_index'],
                'component_id': before['identity']['component_id'], 'forecast_sha256': item['forecast_sha256'],
                'outcome_sha256': item['outcome_sha256'], 'prediction_utc': before['actual_utc'],
                'reveal_utc': after['true_Y_reveal']['actual_utc'], 'native_target': after['true_Y_reveal']['native_target'],
                'score': evaluate(after['true_Y_reveal']['native_target'], before['law']),
                'process_only_not_terminal_confirmation': True})
        states = task.read(directory / 'MODEL_STATES.json')
        sequences = {policy: next(row['selected_pool_indices'] for row in states['checkpoint_states']
                                  if row['policy'] == policy and row['budget'] == 8) for policy in task.POLICIES}
        same_paths = [{'left': a, 'right': b, 'exact_sequence_equal': sequences[a] == sequences[b],
                       'same_final_set': set(sequences[a]) == set(sequences[b])} for a, b in itertools.combinations(task.POLICIES, 2)]
        candidate = summary['native_student']['8']['mean_CRPS_native']
        reference_losses = {'NIGpolicy_' + policy: summary[policy]['8']['mean_CRPS_native'] for policy in task.POLICIES if policy != 'native_student'}
        reference_losses.update({policy + '_' + method: reference_summary[policy][method]['8']['mean_CRPS_native']
                                for policy in reference_summary for method in reference_summary[policy]})
        best = min(reference_losses.values())
        result = {'status': 'COMPLETE_ACTIVE_ONLYY_FROZEN_DEVELOPMENT_COMPARISON', 'actual_end_utc': task.utc(),
            'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
            'forecast_freeze_sha256': task.sha(directory / 'FORECAST_FREEZE.json'),
            'roles': {'warm': 64, 'shared_query_pool': 24, 'terminal_development': 64},
            'summary_NIG_prediction_by_policy': summary, 'summary_strong_native_prediction_references': reference_summary,
            'primary_comparison': {'candidate_native_student_mean_CRPS_budget8': candidate,
                'other_final_losses': reference_losses, 'candidate_to_best_reference_ratio': candidate / best if best else None,
                'within2percent_descriptive': candidate <= 1.02 * best,
                'no_formal_noninferiority_or_monotonic_gain_claim': True},
            'per_terminal_paired_differences': paired, 'primary_NIG_forecast_scores': rows,
            'strong_reference_forecast_scores': reference_rows, 'secondary_prequential_scores': secondary,
            'actual_selected_pool_sequences': sequences, 'policy_path_comparisons': same_paths,
            'unique_law_outcome_score_cache_entries': len(cache),
            'fixed_perpolicy_onlyY_cost': 8, 'common_setup_native_cells': 368,
            'actual_sklearn_fits': 18, 'materialN_not_counterfactual_policy_count': True,
            'scope': 'all152groupsdevelopment;reportedDFTsource;queryefficiencynotactualwet/DFTcostsaving',
            'current64terminal_must_not_be_resplit_or_tuned': True}
        if task.sources() != plan['source_sha256']: raise ValueError('sourcechanged duringscore')
        task.write(output / 'RESULTS.json', result)
        print(json.dumps(result['primary_comparison'], ensure_ascii=True))
    except Exception as problem:
        task.write(output / 'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': task.sources(), 'retain_failure_no_recipe_rescue': True})
        raise


if __name__ == '__main__': main()
