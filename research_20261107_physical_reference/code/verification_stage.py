"""Verify a fixed current pool, freeze adoption, then disclose its census and future pool.

The finite-pool interval is deterministic for the saved laws. It is not a
confidence interval for a future population. All128 targets are eventually
read for research/training; early certification is not total label savings.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import json
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent


def verify(planfile, approvalfile, rolesfile, number, forecasts, output):
    io = load('physical_task_io_verification', HERE/'task_io.py')
    frame = load('physical_affine_verification', HERE.parent/'frame_principle/affine_reference.py')
    plan = io.validate(planfile, approvalfile, 'verify_saved_round')
    roles = io.load_roles(plan,rolesfile)
    if roles['plan_sha256'] != io.sha(planfile):
        raise ValueError('roles must bind the actual execution plan')
    role = roles['round'+str(number)]
    freeze_file = forecasts/'ALL1280_FORECASTS_FROZEN.json'
    freeze = io.read(freeze_file)
    if (freeze['status'] != 'ALL1280_FORECASTS_FROZEN_NO_FUTURE_TARGET_DISCLOSURE'
            or freeze['round'] != number
            or freeze['source_sha256'] != io.sources()
            or freeze['plan_sha256'] != io.sha(planfile)
            or freeze['roles_sha256'] != io.sha(rolesfile)):
        raise ValueError('complete matching current-round forecasts required')
    for name, pin in freeze['files_sha256'].items():
        path = (forecasts/name).resolve(); path.relative_to(forecasts.resolve())
        if io.sha(path) != pin: raise ValueError('changed frozen forecast payload')
    rows = io.read(forecasts/'FORECASTS1280.json')['records']
    if len(rows) != 1280: raise ValueError('all10 methods times128 positions required')
    reference = {r['component_id']:r for r in rows if r['method']=='ionicRF'}
    ids = {kind:[r['component_id'] for r in role[kind]] for kind in ('certificate','confirmation')}
    if any(len(x)!=64 or len(set(x))!=64 for x in ids.values()) or set(ids['certificate']) & set(ids['confirmation']):
        raise ValueError('two disjoint whole-component pools required')
    if set(reference) != set(ids['certificate'])|set(ids['confirmation']):
        raise ValueError('reference positions do not match roles')
    bound_rows = io.read(forecasts/'PRIMARY_GAP_BOUNDS128.json')['records']
    bounds = {r['component_id']:r['envelope'] for r in bound_rows}
    if len(bound_rows)!=128 or set(bounds)!=set(reference): raise ValueError('complete exact bounds required')
    orders = io.read(forecasts/'QUERY_ORDERS.json')['certificate']
    if set(orders) != {'width','metadata_hash','amplitude','reference_expected_abs_gap'}:
        raise ValueError('four registered label-free orderings required')
    for order in orders.values():
        if len(order)!=64 or set(order)!=set(ids['certificate']): raise ValueError('each order must cover the same pool')
    lower = {k:Fraction(bounds[k]['lower_exact_fraction']) for k in ids['certificate']}
    upper = {k:Fraction(bounds[k]['upper_exact_fraction']) for k in ids['certificate']}
    if any(lower[k] > upper[k] for k in lower): raise ValueError('invalid bounds')
    output.mkdir(parents=True, exist_ok=False)
    io.write(output/'START.json', {'actual_utc':datetime.now(timezone.utc).isoformat(),
             'plan_sha256':io.sha(planfile),'round':number,'forecast_freeze_sha256':io.sha(freeze_file),
             'primary_policy':'width','no_full10method_scoring':True})
    reader = io.TrustedReader(plan, roles, round_index=number)
    revealed, requests, prefixes, gap_values = {}, [], [], {}
    total_l, total_u = sum(lower.values(), Fraction()), sum(upper.values(), Fraction())
    decision_file = output/'ADOPTION_FREEZE.json'

    def disclose(key, phase):
        if key in revealed: raise ValueError('no repeated target requests')
        permit = {'phase':phase,'plan_sha256':io.sha(planfile),'round':number,
                  'forecast_freeze':{'file':str(freeze_file),'sha256':io.sha(freeze_file)},
                  'authorized_component_id':key,'request_index':len(requests)}
        if phase != 'certificate_primary':
            permit['decision_freeze']={'file':str(decision_file),'sha256':io.sha(decision_file)}
        before = datetime.now(timezone.utc).isoformat()
        y = reader.reveal_one(key, permit)
        row = reference[key]
        item = {'role':row['role'],'component_id':key,'identity':row['identity'],
                'native_target':y,'known_electronic':row['known_electronic'],
                'raw_known_proxies2':row['raw_known_proxies2'],
                'bounded_descriptors8':row['bounded_descriptors8']}
        revealed[key] = item
        request = {'request_index':len(requests),'actual_request_utc':before,
                   'actual_return_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,
                   'component_id':key,'native_target':y,'permit':permit}
        requests.append(request)
        with (output/'REQUESTS.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(request,ensure_ascii=False,allow_nan=False)+'\n')
        return y

    count = 0
    while True:
        prefixes.append({'queried':count,'lower_mean_exact':str(total_l/64),'upper_mean_exact':str(total_u/64)})
        if total_u < 0:
            choice, reason = 'ionicAuto', 'strict_negative_finite_pool_upper_bound'; break
        if total_l > 0:
            choice, reason = 'ionicRF', 'strict_positive_finite_pool_lower_bound'; break
        if total_l == total_u == 0:
            choice, reason = 'ionicRF', 'exact_tie_keep_reference'; break
        if count == 64:
            raise ArithmeticError('all revealed outcomes must produce a point interval')
        key = orders['width'][count]
        y = disclose(key, 'certificate_primary')
        gap = frame.gap_at(bounds[key], frame.rational(y))
        if not lower[key] <= gap <= upper[key]: raise ArithmeticError('observed gap outside prior envelope')
        gap_values[key] = gap
        total_l += gap-lower[key]; total_u += gap-upper[key]
        count += 1
    io.write(output/'PRIMARY_CERTIFICATE.json', {'actual_utc':datetime.now(timezone.utc).isoformat(),
             'round':number,'primary_target_queries':count,'choice':choice,'reason':reason,
             'prefixes':prefixes,'scope':'saved current64 fixed predictions only; no future-population guarantee'})
    io.write(decision_file, {'status':'ADOPTION_FROZEN_BEFORE_CENSUS_AND_CONFIRMATION',
             'actual_utc':datetime.now(timezone.utc).isoformat(),'round':number,
             'plan_sha256':io.sha(planfile),'forecast_freeze_sha256':io.sha(freeze_file),
             'primary_certificate_sha256':io.sha(output/'PRIMARY_CERTIFICATE.json'),
             'choice':choice,'reason':reason,'primary_target_queries':count,
             'lower_mean_exact':str(total_l/64),'upper_mean_exact':str(total_u/64)})
    # Research census and the forward confirmation are distinct, post-adoption accesses.
    for key in ids['certificate']:
        if key not in revealed: disclose(key, 'post_decision_census')
    for key in ids['confirmation']: disclose(key, 'confirmation')
    ordered = [revealed[k] for kind in ('certificate','confirmation') for k in ids[kind]]
    io.write(output/'ALL128_TARGETS_DISCLOSED.json', {'actual_utc':datetime.now(timezone.utc).isoformat(),
             'round':number,'plan_sha256':io.sha(planfile),'forecast_freeze_sha256':io.sha(freeze_file),
             'decision_freeze_sha256':io.sha(decision_file),'records':ordered})
    # All alternative policy costs below are retrospective lookups of this same pool.
    for key in ids['certificate']:
        if key not in gap_values:
            gap_values[key] = frame.gap_at(bounds[key],frame.rational(revealed[key]['native_target']))
    replay = {}
    for name, order in orders.items():
        lo,hi = sum(lower.values(),Fraction()),sum(upper.values(),Fraction())
        trace=[]
        for k in range(65):
            trace.append({'queried':k,'lower_mean_exact':str(lo/64),'upper_mean_exact':str(hi/64)})
            if hi<0 or lo>0 or lo==hi==0: break
            key=order[k]; lo+=gap_values[key]-lower[key]; hi+=gap_values[key]-upper[key]
        replay[name]={'first_certificate_queries':k,'prefixes':trace,
                      'scope':'counterfactual saved-label lookup after primary decision; not additional measurements'}
    io.write(output/'POSTDECISION_POLICY_REPLAY.json', {'round':number,'orders':replay,
             'primary_policy':'width','new_target_queries_for_replay':0})
    files = {p.name:io.sha(p) for p in output.iterdir() if p.is_file()}
    io.write(output/'CLOSED_ROUND.json', {'status':'ROUND_CLOSED_WITH_ALL128_TRUE_TARGETS',
             'actual_utc':datetime.now(timezone.utc).isoformat(),'round':number,
             'plan_sha256':io.sha(planfile),'forecast_freeze_sha256':io.sha(freeze_file),
             'files_sha256':files,'primary_target_queries':count,'research_census_queries':64-count,
             'forward_confirmation_queries':64,'total_new_target_queries':128,
             'operational_candidate_reference_gap_evaluations':count,
             'postdecision_counterfactual_gap_evaluations':64-count,
             'full10method_score_calls':0,'new_physical_measurements':0,'choice':choice})
    return {'round':number,'choice':choice,'primary_queries':count,'total_queries':128}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True); ap.add_argument('--approval',required=True)
    ap.add_argument('--roles',required=True); ap.add_argument('--round',type=int,choices=(1,2,3),required=True)
    ap.add_argument('--forecast-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args()
    print(verify(a.plan,a.approval,a.roles,a.round,a.forecast_dir,a.output_dir))


if __name__=='__main__': main()
