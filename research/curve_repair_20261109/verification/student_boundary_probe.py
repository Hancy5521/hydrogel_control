"""Registered synthetic math gate, not a material fit or benchmark score."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.stats import t


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.authorization.read_text(encoding='utf-8-sig'))
    if reg['scope'] != 'student_synthetic_boundary_and_cdf_gate':
        raise PermissionError('Explicit math-only source registration required')
    for name, expected in reg['source_pins'].items():
        if sha(name) != expected:
            raise ValueError('Source hash changed')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'baselines'))
    import pure_student as sut
    x = np.arange(6, dtype=float)
    results = []
    for name, y, kind in [('nonzero_constant', np.full(6, 7.), 'constant_log_viscosity'),
                          ('nonzero_constant_powerlaw', np.full(6, 7.), 'power_law'),
                          ('exact_linear', 1. + 2.*x, 'power_law')]:
        try:
            sut.fit_predict(x, y, np.array([6., 7.]), kind)
        except sut.UndefinedStudentPrediction as e:
            results.append({'case': name, 'status': 'correct_rejection', 'reason': str(e)})
        else:
            raise AssertionError(name + ' should not yield a proper scale posterior')
    comparisons = []
    for nu in [4, 5]:
        for y in [-2., 0., 3.]:
            integral = (quad(lambda v: t.cdf(v, nu)**2, -np.inf, y,
                             epsabs=1e-10, epsrel=1e-10)[0] +
                        quad(lambda v: t.sf(v, nu)**2, y, np.inf,
                             epsabs=1e-10, epsrel=1e-10)[0])
            got = float(sut.marginal_crps(np.array([0.]), np.array([1.]), nu, np.array([y]))[0])
            if not np.isclose(got, integral, atol=1e-9, rtol=1e-9):
                raise AssertionError('Student CRPS integral mismatch')
            comparisons.append({'df': nu, 'standardized_target': y, 'closed_form': got,
                                'independent_cdf_integral': integral, 'absolute_error': abs(got-integral)})
    payload = {'status': 'PASS', 'actual_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
         'authorization_sha256': sha(args.authorization), 'boundary_cases': results,
         'cdf_comparisons': comparisons, 'source_labels': 'fixed synthetic mathematical inputs',
         'material_labels_accessed': 0, 'benchmark_fits': 0,
         'independent_definition': 'integral F squared below y plus survival squared above y'}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    print('PASS: three improper-scale boundary rejections and six independent CRPS integrals')


if __name__ == '__main__':
    main()
