"""Registered bounded synthetic implementation gate; no material data reader.

The sole curve, observation prefixes and target coordinates are fixed below.
Four optimizer calls are permitted: n=4,5,6 and legacy n=6 parity. Boundary
checks must fail before any optimizer call. A separate saved-file auditor
reconstructs the Gaussian law without importing this module or either model.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True)
    parser.add_argument('--output', default=str(HERE/'CURVE_REFERENCE_NUMERIC_GATE.json'))
    args = parser.parse_args()
    registration_path = Path(args.registration)
    reg = json.loads(registration_path.read_text(encoding='utf-8'))
    if (reg['schema'] != 'curve-reference-synthetic-registration-v1' or
            reg['scope'] != 'SYNTHETIC_ONLY_NO_MATERIAL_Y' or
            reg['maximum_optimizer_calls'] != 4):
        raise ValueError('Wrong registered synthetic scope')
    for entry in reg['source_pins']:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Source pin mismatch: '+entry['path'])
    new_path = Path(reg['runtime_path'])
    old_path = Path(reg['legacy_path'])
    new = load_source('generic_curve_reference_for_synthetic_gate', new_path)
    old = load_source('legacy_six_reference_for_synthetic_gate', old_path)

    # This is one explicit deterministic synthetic curve, not measured data.
    x = np.array([-2.3, -1.1, -0.2, 0.7, 1.9, 3.2])
    y = 0.3*x + np.sin(1.2*x) + 0.08*x*x
    xt = np.array([-2., -1.5, -0.6, 0.15, 1.2, 2.4, 3.5, 4.2])
    checks, cases, boundary, calls = [], [], [], []
    current_case = ['not_started']
    original_fit = GaussianProcessRegressor.fit

    def counted_fit(model, train_x, train_y):
        calls.append({'case':current_case[0], 'observations':len(train_x)})
        if len(calls) > 4:
            raise RuntimeError('Registered optimizer-call budget exceeded')
        return original_fit(model, train_x, train_y)

    checks.append({'check':'signature_has_only_xO_yO_xT',
                   'pass':list(inspect.signature(new.fit_predict).parameters)==['xO','yO','xT']})
    with patch.object(GaussianProcessRegressor, 'fit', counted_fit):
        for n in (4,5,6):
            current_case[0] = 'generic_n'+str(n)
            law = new.fit_predict(x[:n], y[:n], xt)
            cases.append({'case':current_case[0], 'xO':x[:n].tolist(),
                          'yO':y[:n].tolist(), 'xT':xt.tolist(), 'law':law})
            checks.append({'check':current_case[0]+'_finite_spd_shape',
                           'pass':bool(np.asarray(law['mean']).shape==(8,) and
                                       np.linalg.eigvalsh(law['covariance']).min()>0)})
        current_case[0] = 'legacy_n6_parity'
        legacy = old.fit_predict(x, y, xt)
        latest = cases[-1]['law']
        comparisons = {}
        for key in ('mean','covariance','OLS_beta','SSE','residual_scale',
                    'standardized_anchor_residual','kernel_theta_log',
                    'log_marginal_likelihood_standardized'):
            comparisons[key] = float(np.max(np.abs(np.asarray(latest[key])-np.asarray(legacy[key]))))
        comparisons['log_likelihood_y_units'] = abs(
            latest['log_marginal_likelihood_y_units']-legacy['log_marginal_likelihood_log_eta'])
        checks.append({'check':'legacy_n6_numerical_parity',
                       'pass':max(comparisons.values())<=2e-11,
                       'absolute_errors':comparisons})
        fixtures = [
            ('constant', np.arange(6.), np.ones(6)),
            ('rank_deficient_distinct_x', np.arange(6.)*1e-18,
             np.array([0.,1.,-0.1,2.,0.4,1.5])),
            ('exact_line', np.arange(6.), 1.+2.*np.arange(6.)),
            ('near_exact_line', np.arange(6.),
             1.+2.*np.arange(6.)+4*np.finfo(float).eps*np.array([1.,-1.,1.,-1.,1.,-1.]))]
        for name, bx, by in fixtures:
            current_case[0] = name
            before = len(calls)
            row = {'case':name, 'xO':bx.tolist(), 'yO':by.tolist(), 'xT':[1.23456789]}
            try:
                new.fit_predict(bx, by, row['xT'])
                row.update(rejected=False, error='unexpected success')
            except new.CurveGPFailure as error:
                row.update(rejected=True, error=str(error), details=error.details)
            row['optimizer_calls'] = len(calls)-before
            boundary.append(row)
            checks.append({'check':name+'_rejects_before_optimizer',
                           'pass':row['rejected'] and row['optimizer_calls']==0})
    checks.append({'check':'exactly_four_registered_optimizer_calls',
                   'pass':calls==[{'case':'generic_n4','observations':4},
                                 {'case':'generic_n5','observations':5},
                                 {'case':'generic_n6','observations':6},
                                 {'case':'legacy_n6_parity','observations':6}]})
    result = {'schema':'curve-reference-numeric-gate-v1',
              'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'registration_path':str(registration_path),
              'registration_sha256':digest(registration_path),
              'source_pins':reg['source_pins'],
              'scope':'SYNTHETIC_ONLY_NO_MATERIAL_Y',
              'synthetic_curve':'y=0.3*x+sin(1.2*x)+0.08*x*x, fixed nonuniform x',
              'cases':cases, 'legacy_n6_law':legacy, 'boundary_cases':boundary,
              'optimizer_calls':calls, 'checks':checks,
              'status':'PASS' if all(q['pass'] for q in checks) else 'FAIL',
              'material_response_reads':0, 'material_fits':0,
              'effectiveness_or_calibration_claim':False}
    with Path(args.output).open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'checks':len(checks),
                      'optimizer_calls':len(calls),'output':args.output}))


if __name__ == '__main__':
    main()
