"""Use the frozen estimator with the separately audited stable Newton core.

No model/IO imports at module load. Inputs are already disclosed calibration
pairs. Mathematical objective, 200 iterations and KKT 1e-9 remain unchanged.
"""
from copy import deepcopy


def fit_repair(modules, records, descriptor_index=None):
    old, stable = modules['simplex_transport'], modules['stable_simplex']
    laws = [r['law'] for r in records]
    targets = [r['native_target'] for r in records]
    g = None if descriptor_index is None else [r['bounded_descriptors8'][descriptor_index] for r in records]
    stats = old.make_statistics(laws, targets, g)
    if stats['all_reference_laws_singleton']:
        result = old.fit_simplex(stats)
    else:
        import numpy as np
        fitted = stable.fit(stats)
        result = old._output(stats, np.asarray(fitted['q']), fitted['status'],
                             fitted['iterations'], fitted['history'], fitted.get('reason'))
        result['diagnostics']['stable_projected_KKT'] = fitted['projected_KKT']
        result['diagnostics']['line_search_arithmetic'] = fitted['arithmetic']
    result['descriptor_index'] = descriptor_index
    result['numerical_adapter'] = 'same quadratic+reverseKL objective; stable analytic objective difference'
    return result


def fit_fixed_views(modules, records):
    return {'global': fit_repair(modules, records),
            'anchor': fit_repair(modules, records, 0),
            'auto': fit_repair(modules, records, 5),
            'anchor_index': 0, 'auto_index': 5,
            'descriptor_reselection': False}


def corrected(modules, law, fitted, descriptors):
    if fitted['kind'] == 'identity' or fitted['status'].startswith('FAILED'):
        return deepcopy(law)
    apply = modules['empirical_native'].apply_spline
    if fitted['kind'] == 'global':
        return apply(law, fitted['theta0'])
    return apply(law, fitted['theta0'], fitted['theta1'], descriptors[fitted['descriptor_index']])
