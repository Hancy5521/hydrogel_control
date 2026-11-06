"""Evaluate the first terminal outcomes only after all six trajectories freeze."""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import data_io as task
import mixture_scoring


def aggregate(records):
    if len(records) != 96: raise ValueError('all96 registered terminal groups required')
    pit = sorted(r['PIT_midpoint'] for r in records)
    discrete = any('discrete' in r['PIT_kind'] for r in records)
    return {'formula_groups': 96, 'mean_CRPS_native': float(np.mean([r['CRPS_native'] for r in records])),
        'RMSE_predictive_median_native': float(math.sqrt(np.mean([r['squared_error_native'] for r in records]))),
        'coverage': {level: float(np.mean([r['intervals'][level]['covered'] for r in records]))
                     for level in ('0.8', '0.9', '0.95')},
        'mean_interval_width_native': {level: float(np.mean([r['intervals'][level]['width'] for r in records]))
                     for level in ('0.8', '0.9', '0.95')},
        'PIT_ECDF_deviation_descriptive': None if discrete else max(
            max((i+1)/96-x, x-i/96) for i, x in enumerate(pit)),
        'mean_PIT_left': float(np.mean([r['PIT_left'] for r in records])),
        'mean_PIT_right': float(np.mean([r['PIT_right'] for r in records]))}


def paired_summary(differences):
    a = np.asarray(differences, dtype=float)
    return {'groups': len(a), 'mean_signed_CRPS_difference': float(a.mean()),
            'standard_error_descriptive': float(a.std(ddof=1)/math.sqrt(len(a))),
            'candidate_better_groups': int((a < 0).sum()), 'candidate_worse_groups': int((a > 0).sum()),
            'exact_score_ties': int((a == 0).sum()),
            'not_between_campaign_uncertainty_or_formal_noninferiority': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--forecast-dir', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(); plan, roles = task.validate(args.plan, args.approval, 'score_after_freeze')
    primary, references, prequential, freeze = task.verify_freeze(args.forecast_dir, args.plan, roles)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    task.write(output/'TERMINAL_Y_REVEAL_START.json', {'actual_utc': task.utc(),
        'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
        'approval_sha256': task.sha(args.approval),
        'forecast_freeze_sha256': task.sha(Path(args.forecast_dir)/'FORECAST_FREEZE.json'),
        'all4032terminal_and48prequery_laws_verified_before_terminalY': True})
    try:
        truth = task.reveal_terminal_truth(plan, roles, args.forecast_dir, args.plan)
        task.write(output/'TERMINAL_TRUTH.json', {'scope': 'new96terminaldevelopment', 'truth': truth})
        targets = {r['identity']['component_id']: r['native_target'] for r in truth}
        legacy = task.score_module(plan); scalar_cache, cache = {}, {}
        def scalar_evaluate(y, law):
            key = (y, json.dumps(law, sort_keys=True, separators=(',', ':'), allow_nan=False))
            if key not in scalar_cache: scalar_cache[key] = legacy.score_law(y, law)
            return scalar_cache[key]
        def evaluate(y, law):
            key = (y, json.dumps(law, sort_keys=True, separators=(',', ':'), allow_nan=False))
            if key not in cache:
                cache[key] = (mixture_scoring.score_law(y, law, scalar_evaluate)
                              if law['family'] == 'mixture_student_t' else scalar_evaluate(y, law))
            return cache[key]
        rows, reference_rows = [], []
        for item in primary:
            y = targets[item['component_id']]
            rows.append({k: item[k] for k in ('component_id', 'jid', 'policy', 'budget')} |
                        {'native_target': y, 'score': evaluate(y, item['law'])})
        for item in references:
            y = targets[item['component_id']]
            reference_rows.append({k: item[k] for k in ('component_id', 'jid', 'policy', 'budget', 'method')} |
                                  {'native_target': y, 'score': evaluate(y, item['law'])})
        summary = {policy: {str(b): aggregate([r['score'] for r in rows if r['policy'] == policy and r['budget'] == b])
                           for b in task.BUDGETS} for policy in task.POLICIES}
        ref_summary = {}
        for policy in task.REF_POLICIES + ('gp_variance',):
            methods = ('GP', 'RF', 'affine_NIG') if policy in task.REF_POLICIES else ('GP',)
            if policy == 'full_native': methods += ('frozen_weights_BMA',)
            ref_summary[policy] = {method: {str(b): aggregate([r['score'] for r in reference_rows
                if r['policy'] == policy and r['method'] == method and r['budget'] == b])
                for b in task.BUDGETS} for method in methods}
        lookup = {(r['component_id'], r['policy'], r['budget']): r['score']['CRPS_native'] for r in rows}
        refl = {(r['component_id'], r['policy'], r['method'], r['budget']): r['score']['CRPS_native'] for r in reference_rows}
        paired, paired_summaries = [], {}
        comparators = [(p, None) for p in task.POLICIES if p != 'full_native']
        comparators += [(p, m) for p in ref_summary for m in ref_summary[p]]
        for policy, method in comparators:
            key = policy + ('_'+method if method else '_BMA')
            diffs = []
            for group in targets:
                other = refl[(group, policy, method, 8)] if method else lookup[(group, policy, 8)]
                difference = lookup[(group, 'full_native', 8)] - other
                paired.append({'component_id': group, 'comparator': key, 'budget': 8,
                               'full_native_minus_reference_CRPS': difference})
                diffs.append(difference)
            paired_summaries[key] = paired_summary(diffs)
        directory = Path(args.forecast_dir); secondary = []
        for item in prequential:
            before = task.read(directory/item['forecast_file']); after = task.read(directory/item['outcome_file'])
            y = after['true_Y_reveal']['native_target']
            secondary.append({'policy': item['policy'], 'step': item['step'], 'pool_index': item['pool_index'],
                'component_id': before['identity']['component_id'], 'native_target': y,
                'forecast_sha256': item['forecast_sha256'], 'outcome_sha256': item['outcome_sha256'],
                'prediction_utc': before['actual_utc'], 'reveal_utc': after['true_Y_reveal']['actual_utc'],
                'score': evaluate(y, before['law']), 'process_only_not_terminal_confirmation': True})
        states = task.read(directory/'MODEL_STATES.json')
        sequences = {p: next(r['selected_pool_indices'] for r in states['checkpoint_states']
                             if r['policy'] == p and r['budget'] == 8) for p in task.POLICIES}
        paths = [{'left': a, 'right': b, 'exact_sequence_equal': sequences[a] == sequences[b],
                  'same_final_set': set(sequences[a]) == set(sequences[b])}
                 for a, b in itertools.combinations(task.POLICIES, 2)]
        final = summary['full_native']['8']['mean_CRPS_native']
        losses = {p+'_BMA': summary[p]['8']['mean_CRPS_native'] for p in task.POLICIES if p != 'full_native'}
        losses.update({p+'_'+m: ref_summary[p][m]['8']['mean_CRPS_native'] for p in ref_summary for m in ref_summary[p]})
        best = min(losses.values())
        result = {'status': 'COMPLETE_FRESH160_RELATIONAL_FEEDBACK_DEVELOPMENT', 'actual_end_utc': task.utc(),
            'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
            'forecast_freeze_sha256': task.sha(directory/'FORECAST_FREEZE.json'),
            'roles': {'warm': 32, 'shared_query_pool': 32, 'terminal_development': 96},
            'summary_BMA_prediction_by_policy': summary, 'summary_sameinformation_references': ref_summary,
            'primary_comparison': {'full_native_BMA_CRPS_budget8': final, 'other_final_losses': losses,
                'candidate_to_best_reference_ratio': final/best if best else None,
                'within2percent_descriptive_only': final <= 1.02*best,
                'no_formal_noninferiority_or_material_universality_claim': True},
            'paired_final_differences': paired, 'paired_final_summaries_descriptive': paired_summaries,
            'main_forecast_scores': rows, 'reference_forecast_scores': reference_rows,
            'secondary_prequery_scores': secondary, 'actual_selected_pool_sequences': sequences,
            'policy_path_comparisons': paths, 'unique_outer_score_cache_entries': len(cache),
            'unique_scalar_score_cache_entries_including_components': len(scalar_cache),
            'actual_sklearn_fits': 18, 'terminal_score_coordinates': 4032, 'prequery_score_coordinates': 48,
            'common_learning_setup_cells': 352, 'perpolicy_acquiredY_cells': 8,
            'terminal_validation_Y_cells_separate': 96, 'singlepolicy_including_validation_cells': 456,
            'all160groups_development_no_new_wet_or_DFT_experiments': True,
            'same_model_policies_and_repeated_queries_not_independent_materialN': True,
            'same96terminal_must_not_be_retuned_or_resplit': True}
        if task.sources() != plan['source_sha256']: raise ValueError('scientific source changed during evaluation')
        task.write(output/'RESULTS.json', result)
        print(json.dumps(result['primary_comparison']))
    except Exception as problem:
        task.write(output/'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': task.sources(), 'retain_failure_no_samecohort_recipe_rescue': True})
        raise


if __name__ == '__main__': main()
