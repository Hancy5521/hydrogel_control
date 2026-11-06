"""Run the single pre-registered eight-value three-hypothesis mechanism gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import hypothesis_model
import risk_message


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    source = {name: sha(Path(__file__).parent / name) for name in
              ('hypothesis_model.py', 'risk_message.py', 'run_bounded_gate.py')}
    if (plan['status'] != 'REGISTERED' or plan['source_sha256'] != source
            or approval.get('approved') is not True
            or approval.get('scope') != 'RELATIONAL_THREE_HYPOTHESIS_FIXED8_MECHANISM'
            or approval['plan_sha256'] != sha(args.plan)):
        raise ValueError('registered actual source and root exact approval required')
    for pin in plan['inputs'].values():
        if sha(pin['file']) != pin['sha256']: raise ValueError('frozen input changed')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    try:
        config = read(plan['inputs']['fixed_proposal']['file'])
        prior = config['prior']; states = [('prior', prior)]
        for name, observation in [('after_negative', -1.), ('after_positive', 1.)]:
            posterior, log_density = hypothesis_model.observation_message(
                prior, config['probe_means'], config['probe_standard_deviations'], observation)
            states.append((name, posterior.tolist()))
        records = []
        for name, weights in states:
            for query in ('qL', 'qR'):
                q = config['remaining_queries'][query]
                records.append({'id': name + '_' + query, 'weights': weights,
                    'query_means': q['means'], 'query_SD': q['standard_deviations'],
                    'target_means': config['target_means'], 'target_SD': config['target_standard_deviations']} |
                    risk_message.risk_value(weights, q['means'], q['standard_deviations'],
                                            config['target_means'], config['target_standard_deviations']))
        weights = states[1][1]; query = config['remaining_queries']['qL']
        for name, qm, tm in [('null_query_after_negative', [0.] * 3, config['target_means']),
                             ('null_target_after_negative', query['means'], [0.] * 3)]:
            records.append({'id': name, 'weights': weights, 'query_means': qm, 'query_SD': [1.] * 3,
                            'target_means': tm, 'target_SD': [1.] * 3} |
                           risk_message.risk_value(weights, qm, [1.] * 3, tm, [1.] * 3))
        if len(records) != 8: raise AssertionError('exactly eight fixed values required')
        result = {'status': 'COMPLETE_FIXED8_RELATIONAL_FEEDBACK_MECHANISM', 'evaluations': records,
            'posterior_states': [{'id': name, 'weights': weights} for name, weights in states],
            'source_sha256': source, 'material_N_added': 0, 'numeric_hypotheticals_not_real_material_readings': True,
            'identity': 'Delta=-.5 sum(Cov[posterior H weights]_ij*Eabs(Yi-Yj)); classicBayes specialization',
            'withinH_product_fixed_Gaussian_assumption': True,
            'scope': 'bounded mechanism only; no claim of material performance or new BMA theory'}
    except Exception as problem:
        result = {'status': 'FAIL', 'type': type(problem).__name__, 'message': str(problem),
                  'traceback': traceback.format_exc(), 'material_N_added': 0}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                   'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval)})
    target = output / 'RESULTS.json'
    with target.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'file': str(target), 'sha256': sha(target)}))
    return 0 if result['status'].startswith('COMPLETE') else 1


if __name__ == '__main__': raise SystemExit(main())
