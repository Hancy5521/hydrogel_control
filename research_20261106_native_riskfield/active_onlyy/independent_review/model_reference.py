"""Independent dense batch/NIG and stored baseline-state arithmetic; no SUT imports."""
import math
from collections import defaultdict
from fractions import Fraction
import numpy as np


def require(value, message):
    if not value:
        raise AssertionError(message)


class Checks:
    def __init__(self):
        self.counts, self.errors = {}, {}

    def exact(self, name, a, b):
        require(type(a) is type(b) and a == b, name + ': exact mismatch')
        self.counts[name] = self.counts.get(name, 0) + 1

    def number(self, name, a, b, atol=2e-9, rtol=2e-11):
        require(math.isfinite(a) and math.isfinite(b), name + ': nonfinite')
        error = abs(a - b)
        require(error <= atol + rtol * abs(b), '%s: %r vs %r, error %r' % (name, a, b, error))
        self.counts[name] = self.counts.get(name, 0) + 1
        self.errors[name] = max(self.errors.get(name, 0.), error)

    def array(self, name, a, b):
        require(len(a) == len(b), name + ': length')
        for x, y in zip(a, b):
            if isinstance(y, list):
                self.array(name, x, y)
            else:
                self.number(name, x, y)


def induced_nig(niw, label_count):
    """Block inverse closed form, independently of the producer precision inverse."""
    mu, psi = np.asarray(niw['mean']), np.asarray(niw['Psi'])
    inverse_x = np.linalg.inv(psi[:-1, :-1])
    beta = inverse_x @ psi[:-1, -1]
    mx = mu[:-1]
    covariance = np.empty_like(psi)
    covariance[0, 0] = 1 / niw['kappa'] + mx @ inverse_x @ mx
    covariance[0, 1:] = -mx @ inverse_x
    covariance[1:, 0] = -inverse_x @ mx
    covariance[1:, 1:] = inverse_x
    return {'mean': [float(mu[-1] - beta @ mx)] + beta.tolist(),
            'V': covariance.tolist(), 'alpha': float(niw['nu'] / 2),
            'b': float((psi[-1, -1] - psi[-1, :-1] @ beta) / 2),
            'labeled_updates': label_count}


def nig_check(saved, expected, c, prefix='NIG'):
    for key in ('mean', 'V'):
        c.array(prefix + '_' + key, saved[key], expected[key])
    for key in ('alpha', 'b'):
        c.number(prefix + '_' + key, saved[key], expected[key])
    c.exact(prefix + '_label_count', saved['labeled_updates'], expected['labeled_updates'])


def t_check(saved, expected, c):
    c.exact('t_family', saved['family'], 'student_t')
    c.exact('t_coordinate', saved['coordinate'], 'u=YJ_half(native/s0)')
    c.number('t_lambda', saved['lambda'], .5, 0, 0)
    for name in ('location', 'scale', 'df', 'native_first_scale_s0'):
        c.number('t_' + name, saved[name], expected[name])


def gp_check(state, xfit, xquery, y, c, h):
    """Rebuild native laws from saved RBF, Cholesky and dual; never optimize/refit."""
    n = len(y)
    c.array('GP_Xfit', state['X_train_standardized_float64'], xfit)
    c.array('GP_Xquery', state['X_prediction_standardized_float64'], xquery)
    c.array('GP_y_native', state['y_train_native'], y)
    c.exact('GP_train_rows', state['train_rows'], n)
    c.exact('GP_prediction_rows', state['prediction_rows'], len(xquery))
    ym = math.fsum(y) / n
    ys = math.sqrt(math.fsum((v - ym) ** 2 for v in y) / n)
    if ys == 0:
        ys = 1.
    c.number('GP_ymean', state['y_train_mean'], ym)
    c.number('GP_ySD', state['y_train_std_population'], ys)
    normalized = [(v - ym) / ys for v in y]
    c.array('GP_normalized_y', state['y_train_normalized'], normalized)
    const, ell, white = state['constant_value'], state['RBF_length_scale'], state['white_noise_level']
    nugget = state['training_alpha_nugget']
    require(len(ell) == 2 and .01 * (1-1e-12) <= const <= 100*(1+1e-12)
            and all(.01*(1-1e-12) <= v <= 100*(1+1e-12) for v in ell)
            and 1e-6*(1-1e-12) <= white <= 10*(1+1e-12) and nugget == 1e-8,
            'GP fixed recipe bounds')
    for a, b in zip(state['kernel_theta_log'], [math.log(const)] + [math.log(v) for v in ell] + [math.log(white)]):
        c.number('GP_logtheta', a, b)
    lower, dual = state['L_training_cholesky'], state['alpha_dual_coefficients']
    require(len(lower) == len(dual) == n and all(len(r) == n for r in lower), 'GP factor shape')
    require(all(lower[i][j] == 0 for i in range(n) for j in range(i+1,n))
            and all(lower[i][i] > 0 for i in range(n)), 'GP triangular factor')
    matrix = [[h.scalar_rbf(a, b, const, ell) + (white+nugget if i == j else 0.)
               for j,b in enumerate(xfit)] for i,a in enumerate(xfit)]
    error = [[matrix[i][j] - math.fsum(lower[i][k]*lower[j][k] for k in range(min(i,j)+1))
              for j in range(n)] for i in range(n)]
    transposed = [list(r) for r in zip(*lower)]
    factor_abs = h.norm_inf(error)
    factor_backward = factor_abs / (h.norm_inf(matrix) + h.norm_inf(lower)*h.norm_inf(transposed))
    dual_abs = max(abs(math.fsum(matrix[i][j]*dual[j] for j in range(n))-normalized[i]) for i in range(n))
    denominator = h.norm_inf(matrix)*max(map(abs,dual)) + max(map(abs,normalized))
    dual_backward = dual_abs / denominator if denominator else dual_abs
    require(factor_backward <= 1e-11 and dual_backward <= 1e-11, 'GP normwise backward threshold')
    laws = []
    for point in xquery:
        cross = [h.scalar_rbf(point,row,const,ell) for row in xfit]
        solved = h.forward_solve(lower,cross)
        variance = const + white - math.fsum(v*v for v in solved)
        require(variance > 0, 'positive GP native variance')
        laws.append({'family':'normal','coordinate':'native',
                     'location':ym + ys*math.fsum(a*b for a,b in zip(cross,dual)),
                     'scale':ys*math.sqrt(variance)})
    return laws, {'train_rows':n,'prediction_rows':len(xquery),
                  'factor_backward':factor_backward,'dual_backward':dual_backward,
                  'factor_residual_abs':factor_abs,'dual_residual_abs':dual_abs,
                  'conservative_condition_upper_bound':(n*const+white+nugget)/(white+nugget),
                  'fit_warnings':state['fit_warnings_all_categories'],
                  'predict_warnings':state['predict_warnings_all_categories']}


def forest_check(state, xfit, xquery, y, c, h):
    a = [[h.float32(v) for v in row] for row in xfit]
    q = [[h.float32(v) for v in row] for row in xquery]
    c.exact('RF_Xfit32',state['X_train_standardized_then_float32'],a)
    c.exact('RF_Xquery32',state['X_prediction_standardized_then_float32'],q)
    c.array('RF_y',state['y_train_native'],y)
    c.exact('RF_train_rows',state['train_rows'],len(y))
    c.exact('RF_prediction_rows',state['prediction_rows'],len(q))
    c.exact('RF_OOB_original',state['leaf_weights_include_OOB_original_rows'],True)
    c.exact('RF_no_bootstrap_CDF_multiplicity',state['bootstrap_multiplicity_in_CDF_weights'],False)
    require(len(state['trees']) == 256,'256 RF trees')
    masses = [[Fraction(0) for _ in y] for _ in q]
    for i,tree in enumerate(state['trees']):
        c.exact('RF_tree_id',tree['tree_id'],i)
        tl = [h.tree_leaf(tree,row) for row in a]
        ql = [h.tree_leaf(tree,row) for row in q]
        c.exact('RF_train_paths',tree['original_train_leaf_ids'],tl)
        c.exact('RF_query_paths',tree['prediction_leaf_ids'],ql)
        members = defaultdict(list)
        for j,leaf in enumerate(tl):
            members[leaf].append(j)
        for j,leaf in enumerate(ql):
            require(leaf in members,'RF unsupported query leaf')
            increment = Fraction(1,256*len(members[leaf]))
            for k in members[leaf]:
                masses[j][k] += increment
    for weights in masses:
        require(sum(weights,Fraction(0)) == 1,'RF exact simplex')
    laws = [{'family':'weighted_empirical','coordinate':'native','support_native':y,
             'weight_fractions':[str(w) for w in weights]} for weights in masses]
    return laws, {'train_rows':len(y),'prediction_rows':len(q),
                  'exact_tree_paths':256*(len(a)+len(q)),'exact_mass_positions':len(q)*len(y),
                  'fit_warnings':state['fit_warnings_all_categories']}


def baseline_law_check(saved, expected, c):
    for key in ('family','coordinate'):
        c.exact('reference_'+key,saved[key],expected[key])
    if expected['family'] == 'normal':
        for key in ('location','scale'):
            c.number('GP_native_'+key,saved[key],expected[key],2e-6,2e-8)
    else:
        c.exact('RF_support',saved['support_native'],expected['support_native'])
        c.exact('RF_exact_masses',[Fraction(x) for x in saved['weight_fractions']],
                [Fraction(x) for x in expected['weight_fractions']])
