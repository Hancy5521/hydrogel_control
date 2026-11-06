"""Fixed YJ-half adapter. Parent full-NIW mathematics remains byte-pinned."""
import importlib
import hashlib
from pathlib import Path
import sys

import numpy as np

PARENT_SHA = {
    'admission_io.py': '90011d05e1071b02422622249006cf669bd88e2f7792f8da87f87e0589188466',
    'niw_predictive.py': 'a8d74e5398f2bb606efb195d3395324f014ba3f9b4f801a67b022c1cc6687818',
    'score_admission.py': 'ef7e29f8dab89a0a644011b8c188fcbeaa178095bbd823d082be8887e0aaccc0'}
MODELS = ('raw_joint_t', 'raw_momentmatched_Gaussian', 'yj_joint_t',
          'yj_latent_momentmatched_Gaussian', 'yj_target_marginal')
MASKS = ((), (0,), (1,), (0, 1))
MASK_NAMES = ('none', 'e_form', 'gap_opt', 'both')
_PARENTS = None


def parent_directory():
    return Path(__file__).resolve().parents[2] / 'admission' / 'code'


def parent_hashes():
    return {name: hashlib.sha256((parent_directory() / name).read_bytes()).hexdigest() for name in PARENT_SHA}


def parents():
    global _PARENTS
    if _PARENTS is None:
        if parent_hashes() != PARENT_SHA:
            raise ValueError('Original admission mathematics/IO/scorer source changed')
        folder = parent_directory()
        for name in ('niw_predictive', 'admission_io', 'score_admission'):
            if name in sys.modules and Path(sys.modules[name].__file__).resolve() != (folder / (name + '.py')).resolve():
                raise ValueError('Conflicting parent module already loaded')
        sys.path.insert(0, str(folder))
        _PARENTS = tuple(importlib.import_module(name) for name in ('niw_predictive', 'admission_io', 'score_admission'))
    return _PARENTS


def transform(value):
    v = np.asarray(value, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError('Finite real YJ input required')
    result = np.empty_like(v)
    positive = v >= 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[positive] = v[positive] / (.5 * (np.sqrt(1 + v[positive]) + 1))
        result[~positive] = -np.expm1(1.5 * np.log1p(-v[~positive])) / 1.5
    if not np.isfinite(result).all():
        raise ArithmeticError('YJ transform outside supported finite range')
    return result


def inverse(value):
    u = np.asarray(value, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError('Finite inverse-YJ input required')
    result = np.empty_like(u)
    positive = u >= 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[positive] = u[positive] * (1 + u[positive] / 4)
        result[~positive] = -np.expm1((2 / 3) * np.log1p(-1.5 * u[~positive]))
    if not np.isfinite(result).all():
        raise ArithmeticError('Inverse YJ outside supported finite range')
    return result


def inverse_derivative(value):
    u = np.asarray(value, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError('Finite inverse-YJ derivative input required')
    result = np.empty_like(u)
    positive = u >= 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[positive] = 1 + .5 * u[positive]
        result[~positive] = np.exp(-np.log1p(-1.5 * u[~positive]) / 3)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ArithmeticError('Invalid inverse YJ derivative')
    return result


def log_native_to_standardized_jacobian(native_values, first_scale, second_scale):
    x, s0, b = (np.asarray(v, dtype=float) for v in (native_values, first_scale, second_scale))
    if not np.isfinite(x).all() or not np.isfinite(s0).all() or not np.isfinite(b).all() or (s0 <= 0).any() or (b <= 0).any():
        raise ValueError('Finite native inputs and positive fit-only scales required')
    v = x / s0
    result = np.empty_like(v)
    positive = v >= 0
    result[positive] = -.5 * np.log1p(v[positive])
    result[~positive] = .5 * np.log1p(-v[~positive])
    result = result - np.log(s0) - np.log(b)
    if not np.isfinite(result).all():
        raise ArithmeticError('Nonfinite native-to-standardized log Jacobian')
    return result


def fit_reference(values):
    base, _, _ = parents()
    raw = base.fit_niw(values)
    s0 = np.asarray(raw['affine_scale_sample_SD'])
    u = transform(np.asarray(values, dtype=float) / s0)
    transported = base.fit_niw(u)
    return {'raw': raw, 'yj': transported, 'native_first_scale_s0': s0.tolist(),
            'lambda': .5, 'all_three_coordinates_transformed': True,
            'actual_NIW_fits': 2, 'fit_rows_per_reference': 64,
            'new_fit_parameters_selected_or_optimized': 0,
            'scope': 'fit-dependent empirical reference; not a native-scale pre-data pure Bayes prior'}


def predict_law(fitted, kind, mask, native_proxies):
    if kind not in MODELS or tuple(mask) not in MASKS:
        raise ValueError('One of fixed five law roles and four masks required')
    base, _, _ = parents()
    values = np.asarray(native_proxies, dtype=float)
    if values.shape != (len(mask),) or not np.isfinite(values).all():
        raise ValueError('Only the declared native proxy cells may be passed')
    if kind.startswith('raw_'):
        parent_kind = 'niw_joint_t' if kind == 'raw_joint_t' else 'gaussian_momentmatched'
        law = base.predict_law(fitted['raw'], parent_kind, mask, values)
        return {'coordinate': 'native', 'family': law['family'], 'location': law['location'],
                'scale': law['scale'], 'df': law['df'], 'unit': 'source bulk field GPa attribution'}
    s0 = np.asarray(fitted['native_first_scale_s0'])
    proxy_u = transform(values / s0[list(mask)])
    parent_kind = {'yj_joint_t': 'niw_joint_t', 'yj_latent_momentmatched_Gaussian': 'gaussian_momentmatched',
                   'yj_target_marginal': 'independent_target_marginal'}[kind]
    law = base.predict_law(fitted['yj'], parent_kind, mask, proxy_u)
    return {'coordinate': 'u=YJ_half(native/s0)', 'family': law['family'],
            'location': law['location'], 'scale': law['scale'], 'df': law['df'],
            'native_first_scale_s0': float(s0[2]), 'lambda': .5,
            'unit': 'source bulk field GPa attribution',
            'latent_moment_matching_not_native_moment_matching': True}
