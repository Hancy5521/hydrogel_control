"""Two fixed learning references; importing this file does not import sklearn or fit."""
from fractions import Fraction
import time
import warnings

import numpy as np

METHODS = ('gpr_fixed_recipe', 'forest_all_original_leaf_cdf')
MASKS = ((), (0,), (1,), (0, 1))
MASK_NAMES = ('none', 'e_form', 'gap_opt', 'both')
SEED = 20261106


def feature_transform(fit_values, diagnostic_proxies):
    fit, proxy = np.asarray(fit_values, dtype=float), np.asarray(diagnostic_proxies, dtype=float)
    if fit.shape != (64, 3) or proxy.shape != (192, 2) or not np.isfinite(fit).all() or not np.isfinite(proxy).all():
        raise ValueError('Exactly64x3fit and192x2native proxy inputs required')
    center, scale = fit[:, :2].mean(axis=0), fit[:, :2].std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Fit-only proxy sample SD must be positive')
    y = fit[:, 2]
    if not np.isfinite(y.std(ddof=1)) or y.std(ddof=1) <= 0:
        raise ValueError('Fit target sample SD must be positive')
    return (fit[:, :2] - center) / scale, (proxy - center) / scale, y, {
        'native_feature_center_fit64': center.tolist(), 'native_feature_sample_SD_fit64': scale.tolist(),
        'fit_feature_values_native': fit[:, :2].tolist(), 'diagnostic_feature_values_native': proxy.tolist(),
        'fit_targets_native': y.tolist(), 'standardization_calculation_dtype': 'float64',
        'diagnostic_targets_visible_to_fit': 0}


def warning_records(items):
    return [{'category': x.category.__name__, 'message': str(x.message),
             'filename': str(x.filename), 'line': int(x.lineno)} for x in items]


def fit_gpr(xfit, yfit, xdiagnostic):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    kernel = ConstantKernel(1., (.01, 100.)) * RBF(np.ones(xfit.shape[1]), (.01, 100.)) + WhiteKernel(.01, (1e-6, 10.))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
            optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, random_state=SEED)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_fit:
        warnings.simplefilter('always')
        gp.fit(np.asarray(xfit, dtype=np.float64), yfit)
    fit_seconds = time.perf_counter() - started
    with warnings.catch_warnings(record=True) as captured_predict:
        warnings.simplefilter('always')
        mean, std = gp.predict(np.asarray(xdiagnostic, dtype=np.float64), return_std=True)
    if mean.shape != (192,) or std.shape != (192,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ArithmeticError('GPR must yield192finite positive-SD predictive laws')
    state = {'kind': 'fitted_GPR_replay_state', 'fit_seconds_descriptive_only': fit_seconds,
        'feature_dtype': 'float64', 'learned_kernel_repr': str(gp.kernel_),
        'kernel_theta_log': gp.kernel_.theta.tolist(), 'kernel_bounds_log': gp.kernel_.bounds.tolist(),
        'constant_value': float(gp.kernel_.k1.k1.constant_value),
        'RBF_length_scale': np.atleast_1d(gp.kernel_.k1.k2.length_scale).tolist(),
        'white_noise_level': float(gp.kernel_.k2.noise_level), 'training_alpha_nugget': 1e-8,
        'alpha_dual_coefficients': gp.alpha_.tolist(), 'L_training_cholesky': gp.L_.tolist(),
        'X_train_standardized_float64': gp.X_train_.tolist(), 'y_train_normalized': gp.y_train_.tolist(),
        'y_train_mean': float(np.asarray(gp._y_train_mean)), 'y_train_std_population': float(np.asarray(gp._y_train_std)),
        'log_marginal_likelihood_value': float(gp.log_marginal_likelihood_value_),
        'fit_warnings_all_categories': warning_records(captured_fit),
        'predict_warnings_all_categories': warning_records(captured_predict),
        'test_noise_rule': 'WhiteKernel included in test diagonal; WhiteKernel cross(X,Y)=0; alpha only training diagonal'}
    laws = [{'family': 'normal', 'location': float(m), 'scale': float(s), 'df': None,
             'unit': 'source bulk GPa attribution', 'predictive_WhiteKernel_noise_included': True} for m, s in zip(mean, std)]
    return laws, state


def exact_leaf_weights(train_leaves, diagnostic_leaves, trees=256):
    train, diagnostic = np.asarray(train_leaves), np.asarray(diagnostic_leaves)
    if train.shape != (64, trees) or diagnostic.shape != (192, trees):
        raise ValueError('All64original train rows and192diagnostic leaves per tree required')
    result = []
    for i in range(192):
        weights = [Fraction(0)] * 64
        for t in range(trees):
            members = np.flatnonzero(train[:, t] == diagnostic[i, t])
            if not len(members):
                raise ArithmeticError('A fitted tree leaf has no original training member')
            mass = Fraction(1, trees * len(members))
            for j in members:
                weights[int(j)] += mass
        if sum(weights, Fraction(0)) != 1:
            raise AssertionError('Forest empirical weights must sum exactly to1')
        result.append(weights)
    return result


def empirical_law(support, weights):
    if len(support) != 64 or len(weights) != 64 or sum(weights, Fraction(0)) != 1 or any(w < 0 for w in weights):
        raise ValueError('Exactly64source support positions with exact simplex masses required')
    return {'family': 'weighted_empirical', 'support_native': [float(x) for x in support],
            'weight_fractions': [str(w) for w in weights], 'weights_float_display': [float(w) for w in weights],
            'unit': 'source bulk GPa attribution', 'not_samples_from_a_generated_distribution': True}


def fit_forest(xfit, yfit, xdiagnostic):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=256, min_samples_leaf=3, max_features=1,
            bootstrap=True, criterion='squared_error', random_state=SEED, n_jobs=2, max_depth=None)
    # sklearn RF internally uses float32; export the exact values used for replay.
    train32, diagnostic32 = np.asarray(xfit, dtype=np.float32), np.asarray(xdiagnostic, dtype=np.float32)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        rf.fit(train32, yfit)
    fit_seconds = time.perf_counter() - started
    train_leaves, diagnostic_leaves = rf.apply(train32), rf.apply(diagnostic32)
    masses = exact_leaf_weights(train_leaves, diagnostic_leaves)
    trees = []
    for t, estimator in enumerate(rf.estimators_):
        tree = estimator.tree_
        trees.append({'tree_id': t, 'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(), 'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(), 'n_node_samples': tree.n_node_samples.tolist(),
            'weighted_n_node_samples': tree.weighted_n_node_samples.tolist(),
            'training_tree_value_native': tree.value.reshape(-1).tolist(),
            'original_train_leaf_ids': train_leaves[:, t].tolist(),
            'diagnostic_leaf_ids': diagnostic_leaves[:, t].tolist()})
    state = {'kind': 'all_original_leaf_CDF_forest_replay_state',
        'variant': 'QRF-inspired all-original-row leaf CDF; not a claim of executing an official QRF package',
        'tree_input_dtype': 'float32', 'fit_seconds_descriptive_only': fit_seconds,
        'X_train_standardized_then_float32': train32.astype(float).tolist(),
        'X_diagnostic_standardized_then_float32': diagnostic32.astype(float).tolist(),
        'leaf_weights_include_OOB_original_rows': True, 'bootstrap_multiplicity_in_CDF_weights': False,
        'weight_rule': 'sum_t I(train_leaf_jt=query_leaf_t)/(256*n_leaf_all_original_t)',
        'fit_warnings_all_categories': warning_records(captured), 'trees': trees}
    return [empirical_law(yfit, w) for w in masses], state


def fit_and_forecast(fit_values, diagnostic_proxies):
    xfit, xdiag, yfit, feature_state = feature_transform(fit_values, diagnostic_proxies)
    gpr_none = {'family': 'normal', 'location': float(yfit.mean()), 'scale': float(yfit.std(ddof=1)),
                'df': None, 'unit': 'source bulk GPa attribution', 'no_fit_fixed_target_marginal': True}
    forest_none = empirical_law(yfit, [Fraction(1, 64)] * 64)
    laws = {method: {'none': [gpr_none if method == METHODS[0] else forest_none] * 192} for method in METHODS}
    states = []
    for method in METHODS:
        for name, mask in zip(MASK_NAMES[1:], MASKS[1:]):
            a, b = xfit[:, list(mask)], xdiag[:, list(mask)]
            current, state = fit_gpr(a, yfit, b) if method == METHODS[0] else fit_forest(a, yfit, b)
            laws[method][name] = current
            states.append({'method': method, 'mask': name, 'feature_indices': list(mask), 'state': state})
    return laws, {'feature_transform': feature_state, 'fits': states, 'actual_fits': 6,
                  'none_mask_fits': 0, 'target_values_from_fit_only': True}
