"""Independent inference from JSON states; no SUT, sklearn or NGBoost imports.

Only saved/legally disclosed arrays enter these functions. No fit, optimizer,
raw-source decoding, role selection, outcome score or external IO is performed.
RF/CRPS finite masses are exact. GP/NGB use independent numerical arithmetic.
Training optimality and unsaved RF bootstrap draws are not certified here.
"""
from collections import defaultdict
from fractions import Fraction as F
import math
import random
import struct
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

UNIT = 'source bulk GPa attribution, aggregation qualified'
NORMAL_ATOL = {'fixed_recipe_GPR_saved_state': 2e-6,
               'official_NGBoost_Normal_CRPS_0.5.11': 2e-9}
NORMAL_RTOL = {'fixed_recipe_GPR_saved_state': 2e-8,
               'official_NGBoost_Normal_CRPS_0.5.11': 2e-11}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def finite(value):
    require(not isinstance(value, bool), 'Boolean scientific scalar')
    value = float(value)
    require(math.isfinite(value), 'Nonfinite scientific scalar')
    return value


def matrix(rows, columns=10):
    result = [[finite(v) for v in row] for row in rows]
    require(result and all(len(row) == columns for row in result), 'Finite rectangular matrix required')
    return result


def exact_array(a, b, label):
    require(a == b, label+': exact saved array mismatch')


def numerical_array(a, b, label, atol=2e-9, rtol=2e-11):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label+': shape/finite')
    error = np.abs(a-b)
    require(np.all(error <= atol+rtol*np.abs(b)), label+': numerical mismatch')
    return float(error.max()) if error.size else 0.0


def standardized(rows, transform):
    rows = np.asarray(matrix(rows), dtype=np.float64)
    center = np.asarray([finite(v) for v in transform['center']])
    sd = np.asarray([finite(v) for v in transform['sd']])
    require(center.shape == sd.shape == (10,) and (sd > 0).all(), 'Fixed ten-feature transform')
    out = (rows-center)/sd
    require(np.isfinite(out).all(), 'Standardization not finite')
    return out.tolist()


def float32(value):
    result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    require(math.isfinite(result), 'Tree float32 routing input overflow')
    return result


def _tree_arrays(tree, features):
    names = ('children_left', 'children_right', 'feature', 'threshold')
    arrays = tuple(tree[name] for name in names)
    size = len(arrays[0])
    require(size > 0 and all(len(a) == size for a in arrays), 'Tree array sizes')
    for i, (left, right, feature, threshold) in enumerate(zip(*arrays)):
        require(type(left) is int and type(right) is int, 'Integer tree child')
        require(math.isfinite(float(threshold)), 'Finite tree threshold')
        if left == -1:
            require(right == -1, 'Tree leaf sentinels disagree')
        else:
            require(0 <= left < size and 0 <= right < size and left != right,
                    'Tree child out of range')
            require(type(feature) is int and 0 <= feature < features, 'Tree feature out of range')
    # Audit the entire graph, including saved unreachable/cyclic nodes.
    visited, active = set(), set()
    def visit(node, depth):
        require(node not in active and node not in visited, 'Cyclic/shared tree child')
        active.add(node); visited.add(node)
        height = depth
        if arrays[0][node] != -1:
            height = max(visit(arrays[0][node], depth+1), visit(arrays[1][node], depth+1))
        active.remove(node)
        return height
    height = visit(0, 0)
    require(len(visited) == size, 'Saved tree contains unreachable node')
    return arrays, height


def _leaf(arrays, row):
    left, right, feature, threshold = arrays
    node, steps = 0, 0
    while left[node] != -1:
        node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        steps += 1
        require(steps <= len(left), 'Cyclic tree route')
    return node


def audit_rf(state, training_X, training_y, prediction_X):
    """Rebuild all-original-leaf CDFs, including original OOB rows exactly."""
    train = [[float32(v) for v in row] for row in standardized(training_X, state['fixed_feature_transform'])]
    query = [[float32(v) for v in row] for row in standardized(prediction_X, state['fixed_feature_transform'])]
    y = [finite(v) for v in training_y]
    require(len(train) == len(y) == state['train_rows'] and len(query) == state['prediction_rows'], 'RF row counts')
    require(state['feature_count'] == 10 and len(state['trees']) == 256, 'RF fixed ten features/256 trees')
    require(state['leaf_weights_include_OOB_original_rows'] is True
            and state['bootstrap_multiplicity_in_CDF_weights'] is False, 'RF original-row CDF recipe')
    exact_array(state['X_train_standardized_then_float32'], train, 'RF training float32')
    exact_array(state['X_prediction_standardized_then_float32'], query, 'RF query float32')
    exact_array(state['y_train_native'], y, 'RF native targets')
    weights = [[F() for _ in y] for _ in query]
    nodes = 0
    for number, tree in enumerate(state['trees']):
        require(tree['tree_id'] == number, 'RF tree index order')
        arrays, _ = _tree_arrays(tree, 10)
        nodes += len(arrays[0])
        training_leaves = [_leaf(arrays, row) for row in train]
        query_leaves = [_leaf(arrays, row) for row in query]
        exact_array(tree['original_train_leaf_ids'], training_leaves, 'RF own training routes')
        exact_array(tree['prediction_leaf_ids'], query_leaves, 'RF own query routes')
        members = defaultdict(list)
        for index, leaf in enumerate(training_leaves):
            members[leaf].append(index)
        for row, leaf in zip(weights, query_leaves):
            require(leaf in members, 'RF query leaf has no original training member')
            mass = F(1, len(state['trees'])*len(members[leaf]))
            for index in members[leaf]:
                row[index] += mass
    require(all(sum(w, F()) == 1 for w in weights), 'RF exact finite mass')
    laws = [{'coordinate': 'native', 'family': 'weighted_empirical', 'support_native': y,
             'weight_fractions': list(map(str, w)), 'unit': UNIT} for w in weights]
    return laws, {'kind': state['kind'], 'train_rows': len(train), 'prediction_rows': len(query),
        'trees': 256, 'saved_nodes_validated': nodes, 'routing_dtype': 'float32',
        'exact_routes_checked': 256*(len(train)+len(query)),
        'exact_weight_positions': len(query)*len(train), 'new_fits': 0,
        'bootstrap_draws_or_split_optimality_certified': False}


def _rbf(a, b, constant, lengths):
    # Pairwise differences and squared Euclidean metric, not sklearn.kernel_.
    aa, bb = np.asarray(a)/lengths, np.asarray(b)/lengths
    return constant*np.exp(-0.5*cdist(aa, bb, metric='sqeuclidean'))


def _infinity_norm(a):
    return float(np.max(np.sum(np.abs(a), axis=1)))


def audit_gp(state, training_X, training_y, prediction_X):
    train = standardized(training_X, state['fixed_feature_transform'])
    query = standardized(prediction_X, state['fixed_feature_transform'])
    y = np.asarray([finite(v) for v in training_y])
    n = len(y)
    require(n == len(train) == state['train_rows'] and len(query) == state['prediction_rows'], 'GP row counts')
    require(state['feature_count'] == 10, 'GP ten features')
    exact_array(state['y_train_native'], y.tolist(), 'GP native targets')
    numerical_array(state['X_train_standardized_float64'], train, 'GP standardized training', 0, 0)
    numerical_array(state['X_prediction_standardized_float64'], query, 'GP standardized query', 0, 0)
    center = math.fsum(map(float, y))/n
    sd = math.sqrt(math.fsum((float(v)-center)**2 for v in y)/n) or 1.0
    numerical_array([state['y_train_mean'], state['y_train_std_population']], [center, sd], 'GP y mean/population SD')
    response = (y-center)/sd
    numerical_array(state['y_train_normalized'], response, 'GP normalized response')
    constant, white, nugget = map(finite, (state['constant_value'], state['white_noise_level'], state['training_alpha_nugget']))
    lengths = np.asarray([finite(v) for v in state['RBF_length_scale']])
    require(lengths.shape == (10,) and np.all((lengths >= .01*(1-1e-12)) & (lengths <= 100*(1+1e-12))), 'GP registered RBF bounds')
    require(.01*(1-1e-12) <= constant <= 100*(1+1e-12)
            and 1e-6*(1-1e-12) <= white <= 10*(1+1e-12) and nugget == 1e-8, 'GP kernel bounds/nugget')
    numerical_array(state['kernel_theta_log'], [math.log(constant)]+list(map(math.log, lengths))+[math.log(white)], 'GP log hyperparameters')
    L, alpha = np.asarray(state['L_training_cholesky'], dtype=float), np.asarray(state['alpha_dual_coefficients'], dtype=float)
    require(L.shape == (n, n) and alpha.shape == (n,) and np.isfinite(L).all() and np.isfinite(alpha).all(), 'GP factor/dual shapes')
    require(np.all(np.diag(L) > 0) and np.all(L[np.triu_indices(n, 1)] == 0), 'GP positive lower Cholesky')
    kernel = _rbf(train, train, constant, lengths)
    kernel[np.diag_indices(n)] += white+nugget
    factor_error = _infinity_norm(kernel-L@L.T)
    factor_backward = factor_error/(_infinity_norm(kernel)+_infinity_norm(L)*_infinity_norm(L.T))
    residual = kernel@alpha-response
    dual_error = float(np.max(np.abs(residual)))
    denominator = _infinity_norm(kernel)*float(np.max(np.abs(alpha)))+float(np.max(np.abs(response)))
    dual_backward = dual_error/denominator if denominator else dual_error
    require(factor_backward <= 1e-11 and dual_backward <= 1e-11, 'GP normwise backward error')
    cross = _rbf(query, train, constant, lengths)
    mean = center+sd*(cross@alpha)
    solved = solve_triangular(L, cross.T, lower=True, check_finite=True)
    variance = constant+white-np.sum(solved*solved, axis=0)
    require(np.isfinite(variance).all() and (variance > 0).all(), 'GP positive predictive variance; no clipping/floor')
    scale = sd*np.sqrt(variance)
    laws = [{'coordinate': 'native', 'family': 'normal', 'location': float(m),
             'scale': float(s), 'df': None, 'unit': UNIT} for m, s in zip(mean, scale)]
    return laws, {'kind': state['kind'], 'train_rows': n, 'prediction_rows': len(query),
        'factor_backward_error': factor_backward, 'dual_backward_error': dual_backward,
        'factor_residual_infinity_norm': factor_error, 'dual_residual_max_abs': dual_error,
        'test_variance_includes_white_not_training_nugget': True,
        'condition_upper_bound': (n*constant+white+nugget)/(white+nugget),
        'hyperparameter_optimality_certified': False, 'new_fits': 0}


def _entropy(y):
    # Exact sorted energy identity, independent of producer's Fenwick prefixes.
    z = sorted(F.from_float(float(v)) for v in y)
    n = len(z)
    require(n > 0, 'Empty empirical leaf')
    return sum(((2*i-n+1)*v for i, v in enumerate(z)), F())/(n*n)


def audit_crps(state, training_X, training_y, prediction_X):
    train, query = matrix(training_X), matrix(prediction_X)
    y = [finite(v) for v in training_y]
    exact_array(state['training_X'], train, 'CRPS RAW native training features')
    exact_array(state['training_y'], y, 'CRPS native targets')
    require(len(train) == len(y), 'CRPS training dimensions')
    require(state['n_trees'] == len(state['trees']) == len(state['subsample_row_indices']) == 256,
            'CRPS fixed256 trees')
    require(state['min_leaf'] == 3 and state['max_depth'] is None
            and state['subsample_fraction'] == .6 and state['seed'] == 20261107,
            'CRPS registered recipe')
    require(state['no_bootstrap_multiplicity_or_OOB_reweighting'] is True, 'CRPS leaf-CDF weighting recipe')
    rng, nodes = random.Random(20261107), 0
    masses, leaf_ids = [defaultdict(F) for _ in query], [[] for _ in query]
    for number, tree in enumerate(state['trees']):
        sampled = rng.sample(range(len(y)), int(.6*len(y)))
        exact_array(state['subsample_row_indices'][number], sampled, 'CRPS deterministic subsample')
        exact_array(tree['source_row_indices'], sampled, 'CRPS original sampled row IDs')
        exact_array(tree['training_X'], [train[i] for i in sampled], 'CRPS raw subsample features')
        exact_array(tree['training_y'], [y[i] for i in sampled], 'CRPS subsample targets')
        require(tree['feature_count'] == 10 and tree['all_features_searched'] is True,
                'CRPS tree ten-feature contract')
        def visit(node, expected, path):
            nonlocal nodes
            nodes += 1
            exact_array(node['rows'], expected, 'CRPS saved node partition')
            require(node['node_id'] == path and node['n'] == len(expected), 'CRPS node identity/count')
            entropy = _entropy([y[i] for i in expected])
            numerical_array(node['entropy_native'], float(entropy), 'CRPS exact node entropy')
            if node['leaf']:
                exact_array(node['target_values'], [y[i] for i in expected], 'CRPS leaf target identities')
                return
            j, t = node['feature'], finite(node['threshold'])
            require(type(j) is int and 0 <= j < 10, 'CRPS split feature')
            left = [i for i in expected if train[i][j] <= t]
            right = [i for i in expected if train[i][j] > t]
            require(min(len(left), len(right)) >= 3, 'CRPS split min_leaf')
            risk = (len(left)*_entropy([y[i] for i in left])+len(right)*_entropy([y[i] for i in right]))/len(expected)
            numerical_array(node['child_weighted_entropy_native'], float(risk), 'CRPS selected split entropy')
            require(risk < entropy, 'CRPS saved selected split lacks exact improvement')
            visit(node['left'], left, path+'L'); visit(node['right'], right, path+'R')
        visit(tree['root'], sampled, 'root')
        for i, row in enumerate(query):
            node = tree['root']
            while not node['leaf']:
                node = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
            leaf_ids[i].append(node['node_id'])
            mass = F(1, 256*len(node['rows']))
            for index in node['rows']:
                masses[i][y[index]] += mass
    laws = []
    for weights, leaves in zip(masses, leaf_ids):
        require(sum(weights.values(), F()) == 1 and all(v > 0 for v in weights.values()), 'CRPS exact leaf mixture')
        x = sorted(weights)
        laws.append({'coordinate': 'native', 'family': 'weighted_empirical', 'support_native': x,
                     'weight_fractions': [str(weights[v]) for v in x], 'unit': UNIT,
                     'leaf_ids_in_tree_order': leaves})
    return laws, {'kind': state['kind'], 'train_rows': len(train), 'prediction_rows': len(query),
        'trees': 256, 'saved_nodes_validated': nodes, 'routing_dtype': 'native binary64',
        'all_feature_split_optimality_certified': False, 'new_fits': 0}


def audit_ngboost(state, training_X, training_y, prediction_X):
    train = standardized(training_X, state['fixed_feature_transform'])
    query = standardized(prediction_X, state['fixed_feature_transform'])
    y = [finite(v) for v in training_y]
    require(len(train) == len(y) == state['train_rows'], 'NGB training row count')
    exact_array(state['training_y_native'], y, 'NGB native targets')
    numerical_array(state['training_X_standardized_float64'], train, 'NGB standardized training', 0, 0)
    numerical_array(state['prediction_X_standardized_float64'], query, 'NGB standardized query', 0, 0)
    require(state['score'] == 'CRPScore' and state['natural_gradient'] is True
            and state['minibatch_frac'] == state['col_sample'] == 1.0
            and state['learning_rate'] == .01 and state['tol'] == 1e-4
            and state['random_state'] == 20261107 and state['early_stopping_rounds'] is None,
            'NGB fixed official stable Normal CRPS recipe')
    count = state['n_stages_actual']
    require(1 <= count <= state['n_estimators_requested'] == 500
            and count == len(state['base_models']) == len(state['scalings']) == len(state['col_idxs']),
            'NGB stage lengths')
    mean = math.fsum(y)/len(y)
    sd = math.sqrt(math.fsum((v-mean)**2 for v in y)/len(y))
    require(sd > 0, 'NGB training Normal scale positive')
    numerical_array(state['init_params'], [mean, math.log(sd)], 'NGB fitted marginal mean/log-SD')
    parameters = np.tile(np.asarray(state['init_params'], dtype=float), (len(query), 1))
    require(parameters.shape == (len(query), 2), 'NGB two Normal parameter columns')
    nodes, max_depth = 0, 0
    for models, scaling, columns in zip(state['base_models'], state['scalings'], state['col_idxs']):
        require(len(models) == 2 and columns == list(range(10)), 'NGB two trees/all ten columns')
        scaling = finite(scaling)
        require(scaling > 0, 'NGB positive line-search multiplier')
        routed = [[float32(row[j]) for j in columns] for row in query]
        for parameter, tree in enumerate(models):
            arrays, depth = _tree_arrays(tree, len(columns))
            require(depth <= 3, 'NGB base tree fixed max_depth3')
            max_depth = max(max_depth, depth); nodes += len(arrays[0])
            require(len(tree['value']) == len(arrays[0]), 'NGB saved node-value length')
            values = [finite(v) for v in tree['value']]
            predictions = np.asarray([values[_leaf(arrays, row)] for row in routed])
            # Preserve official stage ordering and multiplication order. No
            # package .predict, .pred_param, .pred_dist, or model reconstruction.
            parameters[:, parameter] -= (state['learning_rate']*predictions)*scaling
    require(np.isfinite(parameters).all(), 'NGB finite independently rebuilt parameters')
    scales = np.exp(parameters[:, 1])
    require(np.isfinite(scales).all() and (scales > 0).all(), 'NGB exp(log-scale) positive, no floor')
    laws = [{'coordinate': 'native', 'family': 'normal', 'location': float(p[0]),
             'scale': float(s), 'df': None, 'unit': UNIT} for p, s in zip(parameters, scales)]
    return laws, {'kind': state['kind'], 'train_rows': len(train), 'prediction_rows': len(query),
        'stages': count, 'saved_nodes_validated': nodes, 'maximum_base_tree_depth': max_depth,
        'routing_dtype': 'float32 after fixed binary64 standardization',
        'initial_parameters_checked_against_observed_y': True,
        'boosting_gradient_fit_or_line_search_optimality_certified': False,
        'new_fits': 0, 'package_model_predict_calls': 0}


def audit_saved_model(state, training_X, training_y, prediction_X):
    """Full raw-feature/target consistency plus independent saved inference."""
    routes = {'fixed_all_original_leaf_RF_state': audit_rf,
              'fixed_recipe_GPR_saved_state': audit_gp,
              'independent_classical_CRPS_subsample_forest': audit_crps,
              'official_NGBoost_Normal_CRPS_0.5.11': audit_ngboost}
    require(state.get('kind') in routes, 'Unsupported saved model kind')
    return routes[state['kind']](state, training_X, training_y, prediction_X)


def _mass_map(law):
    require(law['family'] == 'weighted_empirical' and law['coordinate'] == 'native', 'Native empirical law')
    require(len(law['support_native']) == len(law['weight_fractions']) > 0, 'Matched empirical law arrays')
    masses = defaultdict(F)
    for x, value in zip(law['support_native'], law['weight_fractions']):
        x, mass = finite(x), F(value)
        require(mass >= 0, 'Negative empirical mass')
        if mass:
            masses[x] += mass
    require(sum(masses.values(), F()) == 1, 'Exact total probability one')
    return dict(masses)


def compare_laws(saved, rebuilt, model_kind=None):
    require(len(saved) == len(rebuilt), 'Saved/rebuilt prediction counts')
    max_location_error, max_scale_error, exact_mass_positions = 0.0, 0.0, 0
    for left, right in zip(saved, rebuilt):
        require(left['family'] == right['family'] and left['coordinate'] == right['coordinate'] == 'native', 'Law family/coordinate')
        require(left.get('unit') == right.get('unit') == UNIT, 'Native law unit')
        if right['family'] == 'weighted_empirical':
            a, b = _mass_map(left), _mass_map(right)
            exact_array(a, b, 'Rebuilt exact CDF')
            exact_mass_positions += len(b)
            if 'leaf_ids_in_tree_order' in left:
                exact_array(left['leaf_ids_in_tree_order'], right['leaf_ids_in_tree_order'], 'CRPS query leaf IDs')
        else:
            require(right['family'] == 'normal' and left.get('df') is None, 'Native Normal law')
            require(model_kind in NORMAL_ATOL, 'Normal comparison kind must be explicit')
            e = numerical_array([left['location'], left['scale']], [right['location'], right['scale']],
                'Independent Normal forecast', NORMAL_ATOL[model_kind], NORMAL_RTOL[model_kind])
            require(finite(left['scale']) > 0 and finite(right['scale']) > 0, 'Positive forecast SD')
            max_location_error = max(max_location_error, abs(left['location']-right['location']))
            max_scale_error = max(max_scale_error, abs(left['scale']-right['scale']))
    return {'predictions': len(saved), 'exact_positive_mass_positions': exact_mass_positions,
            'maximum_native_location_error': max_location_error,
            'maximum_native_scale_error': max_scale_error, 'new_fits': 0}
