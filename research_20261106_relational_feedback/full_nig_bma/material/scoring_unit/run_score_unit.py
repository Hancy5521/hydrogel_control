"""Four scalar scores from two already saved artificial NIG-BMA laws, once."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    if (approval.get('approved') is not True or approval.get('scope') != 'FIXED_SAVED_LAWS_MIXTURE_SCORING_UNIT'
            or approval['plan_sha256'] != sha(args.plan) or sha(__file__) != plan['script_sha256']):
        raise ValueError('exact actual-UTC registered scoring-unit approval required')
    for pin in plan['inputs'].values():
        if sha(pin['file']) != pin['sha256']: raise ValueError('fixed scoring source/input changed')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        inputs = plan['inputs']
        sys.path.insert(0, str(Path(inputs['frozen_core']['file']).parent))
        sys.path.insert(0, str(Path(inputs['mixture_scoring']['file']).parent))
        sys.path.append(str(Path(inputs['scalar_score']['file']).parent))
        import material_model
        import mixture_scoring
        spec = importlib.util.spec_from_file_location('_fixed_parent_single_score', inputs['scalar_score']['file'])
        scalar = importlib.util.module_from_spec(spec); spec.loader.exec_module(scalar)
        fixture = read(inputs['saved_full_NIG_fixture']['file'])
        initial = fixture['initial_ensemble']; response = fixture['fixed_fixture']['response_transform']
        context = fixture['fixed_fixture']['target_context']
        positive = next(r for r in fixture['synthetic_fixed_label_probes'] if r['artificial_modeled_query_Y'] == 1)
        updated = copy.deepcopy(initial)
        updated['models'] = [{'id': r['id'], 'state': r['updated_NIG']} for r in positive['updated_models']]
        updated['log_weights'] = positive['updated_log_weights']; updated['true_label_updates'] = 1
        records = []
        for name, ensemble in [('initial', initial), ('after_positive', updated)]:
            law = material_model.law(ensemble, context, response)
            records.append({'id': name, 'law': law,
                'scores': [{'native_target': y, 'score': mixture_scoring.score_law(y, law, scalar.score_law)}
                           for y in (-1., 1.)],
                'quantiles': [{'probability': p, 'native_quantile': mixture_scoring.quantile_native(law, p)}
                              for p in (.025, .05, .1, .5, .9, .95, .975)]})
        result = {'status': 'COMPLETE_FIXED_FOUR_MIXTURE_SCORES', 'law_records': records,
            'new_material_measurements_fits_or_joint_VOI_reintegrals': 0,
            'scope': 'only original fixed full-NIG fixture initial/after+1 laws; native hypothetical y=-1/+1'}
    except Exception as problem:
        result = {'status': 'FAIL', 'type': type(problem).__name__, 'message': str(problem),
                  'traceback': traceback.format_exc(), 'new_material_measurements_or_fits': 0}
    result.update({'actual_start_utc': started, 'actual_end_utc': datetime.now(timezone.utc).isoformat(),
                   'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval)})
    target = output/'RESULTS.json'
    with target.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'file': str(target), 'sha256': sha(target)}))
    return 0 if result['status'].startswith('COMPLETE') else 1


if __name__ == '__main__': raise SystemExit(main())
