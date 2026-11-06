"""Original fixed GP and all-original-leaf RF recipes with dynamic row counts.

These are strong prediction/control references, not new models. All feature
scales are warm-only and fixed; GP normalize_y follows currently observed labels.
"""
from datetime import datetime, timezone
from fractions import Fraction
import time
import warnings
import numpy as np


def _warnings(records):
    return [{'category': item.category.__name__, 'message': str(item.message),
             'filename': str(item.filename), 'line': int(item.lineno)} for item in records]


def standardized(train_native, query_native, transform):
    train, query = np.asarray(train_native, dtype=float), np.asarray(query_native, dtype=float)
    center, scale = np.asarray(transform['center']), np.asarray(transform['sd'])
    if (train.ndim != 2 or train.shape[1] != 3 or query.ndim != 2 or query.shape[1] != 2
            or center.shape != (2,) or scale.shape != (2,) or (scale <= 0).any()
            or not np.isfinite(train).all() or not np.isfinite(query).all()):
        raise ValueError('finite observed complete train and known proxy rows required')
    return (train[:, :2] - center) / scale, train[:, 2], (query - center) / scale


def fit_gp(train_native, prediction_native, transform):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    x, y, query = standardized(train_native, prediction_native, transform)
    kernel = ConstantKernel(1., (.01, 100.)) * RBF(np.ones(2), (.01, 100.)) + WhiteKernel(.01, (1e-6, 10.))
    estimator = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
        optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, random_state=20261106)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as fitted:
        warnings.simplefilter('always'); estimator.fit(np.asarray(x, dtype=np.float64), y)
    seconds = time.perf_counter() - start
    with warnings.catch_warnings(record=True) as predicted:
        warnings.simplefilter('always'); mean, sd = estimator.predict(np.asarray(query, dtype=np.float64), return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ArithmeticError('finite GP laws with positive SD required')
    state = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'fixed_recipe_GPR_saved_state',
        'fit_seconds_descriptive_only': seconds, 'train_rows': len(y), 'prediction_rows': len(query),
        'constant_value': float(estimator.kernel_.k1.k1.constant_value),
        'RBF_length_scale': np.atleast_1d(estimator.kernel_.k1.k2.length_scale).tolist(),
        'white_noise_level': float(estimator.kernel_.k2.noise_level), 'training_alpha_nugget': 1e-8,
        'alpha_dual_coefficients': estimator.alpha_.tolist(), 'L_training_cholesky': estimator.L_.tolist(),
        'X_train_standardized_float64': estimator.X_train_.tolist(), 'X_prediction_standardized_float64': query.tolist(),
        'y_train_native': y.tolist(), 'y_train_normalized': estimator.y_train_.tolist(),
        'y_train_mean': float(np.asarray(estimator._y_train_mean)),
        'y_train_std_population': float(np.asarray(estimator._y_train_std)),
        'kernel_theta_log': estimator.kernel_.theta.tolist(), 'kernel_bounds_log': estimator.kernel_.bounds.tolist(),
        'log_marginal_likelihood_value': float(estimator.log_marginal_likelihood_value_),
        'fit_warnings_all_categories': _warnings(fitted), 'predict_warnings_all_categories': _warnings(predicted),
        'predictive_noise': 'learnedWhiteincludedtestdiagonal; alpha trainingonly'}
    laws = [{'coordinate': 'native', 'family': 'normal', 'location': float(a), 'scale': float(b),
             'df': None, 'unit': 'source bulk GPa attribution, aggregation qualified'} for a, b in zip(mean, sd)]
    return laws, state


def fit_forest(train_native, prediction_native, transform):
    from sklearn.ensemble import RandomForestRegressor
    x, y, query = standardized(train_native, prediction_native, transform)
    x, query = np.asarray(x, dtype=np.float32), np.asarray(query, dtype=np.float32)
    estimator = RandomForestRegressor(n_estimators=256, min_samples_leaf=3, max_features=1,
        bootstrap=True, criterion='squared_error', random_state=20261106, n_jobs=2, max_depth=None)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as fitted:
        warnings.simplefilter('always'); estimator.fit(x, y)
    seconds = time.perf_counter() - start
    train_leaves, query_leaves = estimator.apply(x), estimator.apply(query)
    masses = []
    for i in range(len(query)):
        weights = [Fraction(0)] * len(y)
        for tree in range(256):
            members = np.flatnonzero(train_leaves[:, tree] == query_leaves[i, tree])
            if not len(members):
                raise ArithmeticError('no original member in fitted leaf')
            increment = Fraction(1, 256 * len(members))
            for j in members:
                weights[int(j)] += increment
        if sum(weights, Fraction(0)) != 1:
            raise AssertionError('exact original-row empirical simplex required')
        masses.append(weights)
    trees = []
    for index, child in enumerate(estimator.estimators_):
        tree = child.tree_
        trees.append({'tree_id': index, 'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(), 'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
            'n_node_samples': tree.n_node_samples.tolist(), 'weighted_n_node_samples': tree.weighted_n_node_samples.tolist(),
            'training_tree_value_native': tree.value.reshape(-1).tolist(),
            'original_train_leaf_ids': train_leaves[:, index].tolist(), 'prediction_leaf_ids': query_leaves[:, index].tolist()})
    state = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'kind': 'fixed_all_original_leaf_RF_state',
        'fit_seconds_descriptive_only': seconds, 'train_rows': len(y), 'prediction_rows': len(query),
        'X_train_standardized_then_float32': x.astype(float).tolist(),
        'X_prediction_standardized_then_float32': query.astype(float).tolist(), 'y_train_native': y.tolist(),
        'leaf_weights_include_OOB_original_rows': True, 'bootstrap_multiplicity_in_CDF_weights': False,
        'fit_warnings_all_categories': _warnings(fitted), 'trees': trees,
        'scope': 'QRF-inspired all-original-leaf variant, not official QRF software'}
    laws = [{'coordinate': 'native', 'family': 'weighted_empirical', 'support_native': y.tolist(),
             'weight_fractions': [str(value) for value in weights],
             'unit': 'source bulk GPa attribution, aggregation qualified'} for weights in masses]
    return laws, state
