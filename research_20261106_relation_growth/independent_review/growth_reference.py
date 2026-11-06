"""Independent batch conjugate statistics and composition/proxy features.

No import of growth SUT, sklearn, model fit/predict, or material data loader.
GP stored-factor reconstruction is an explicitly ten-dimensional adaptation of
the previously audited scalar path. All evidence checks are saved-state audits.
"""
import math
from fractions import Fraction
import numpy as np
from scipy import special

IDS = ('affine', 'interaction', 'fullquad')
DESCRIPTORS = ('O_present', 'f_O', 'f_H', 'f_C_plus_N', 'f_F_Cl_Br_I',
               'f_S_Se_Te', 'element_count', 'stoichiometric_entropy')


def composition(identity):
    amounts = {key: Fraction(str(value)) for key, value in identity['composition_amounts'].items()
               if not key.startswith('@') and Fraction(str(value)) > 0}
    total = sum(amounts.values(), Fraction())
    if total <= 0:
        raise ValueError('positive composition required')
    fractions = {key: float(value / total) for key, value in amounts.items()}
    def group(names):
        return math.fsum(fractions.get(key, 0.) for key in names)
    return [float('O' in amounts), group(('O',)), group(('H',)), group(('C', 'N')),
            group(('F', 'Cl', 'Br', 'I')), group(('S', 'Se', 'Te')),
            float(len(amounts)), -math.fsum(v * math.log(v) for v in fractions.values())]


def column_scale(rows):
    n = len(rows)
    dimension = len(rows[0])
    means = [math.fsum(row[j] for row in rows) / n for j in range(dimension)]
    sd = [math.sqrt(math.fsum((row[j] - means[j]) ** 2 for row in rows) / (n - 1))
          for j in range(dimension)]
    return means, sd


def descriptor_scale(rows):
    center, sample = column_scale(rows)
    flags = [s <= 1e-12 * max(1., max(abs(row[j]) for row in rows))
             for j, s in enumerate(sample)]
    return {'center': center, 'sd': [1. if flag else s for flag, s in zip(flags, sample)],
            'raw_sample_SD': sample, 'constant_or_numerically_constant': flags}


def transform(rows, coordinate, forward):
    return [[(forward(value / coordinate['native_first_scale_s0'][j])
              - coordinate['u_center_warm'][j]) / coordinate['u_sample_SD_warm'][j]
             for j, value in enumerate(row)] for row in rows]


def standardized(rows, coordinate):
    return [[(value - coordinate['center'][j]) / coordinate['sd'][j]
             for j, value in enumerate(row)] for row in rows]


def expansion(rows):
    terms = [[row[0] * row[1], row[0] ** 2, row[1] ** 2] for row in rows]
    center, sd = column_scale(terms)
    return {'center': center, 'sd': sd}


def design(kind, row, expanded, descriptor=None, index=None):
    a, b = row[:2]
    if kind == 'descriptor':
        return [1., a, b, descriptor[index]]
    out = [1., a, b]
    if kind != 'affine':
        values = [a*b, a*a, b*b]
        out += [(values[j] - expanded['center'][j]) / expanded['sd'][j]
                for j in range(1 if kind == 'interaction' else 3)]
    return out


def dense(designs, targets):
    n, d = len(designs), len(designs[0])
    precision = np.array([[(1. if i == j else 0.)
                           + math.fsum(row[i] * row[j] for row in designs)
                           for j in range(d)] for i in range(d)])
    cross = np.array([math.fsum(row[i] * y for row, y in zip(designs, targets))
                      for i in range(d)])
    mean = np.linalg.solve(precision, cross)
    inverse = np.linalg.inv(precision)
    alpha = 3. + n / 2
    b = 1.5 + .5 * (math.fsum(y*y for y in targets) - float(mean @ precision @ mean))
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0 or b <= 0:
        raise ArithmeticError('positive batch conjugate precision and scale required')
    evidence = (math.lgamma(alpha) - math.lgamma(3.) + 3.*math.log(1.5)
                - alpha*math.log(b) - .5*logdet - n/2*math.log(2*math.pi))
    return {'mean': mean.tolist(), 'V': inverse.tolist(), 'alpha': alpha,
            'b': b, 'labeled_updates': n}, evidence


def normalized(logs):
    constant = float(special.logsumexp(logs))
    return [float(value - constant) for value in logs]


def predictive(state, phi, coordinate):
    v = np.array(phi)
    mu = float(v @ np.array(state['mean']))
    scale = math.sqrt(state['b']/state['alpha'] * (1 + float(v @ np.array(state['V']) @ v)))
    return {'location': coordinate['u_center_warm'][2] + coordinate['u_sample_SD_warm'][2]*mu,
            'scale': coordinate['u_sample_SD_warm'][2]*scale, 'df': 2*state['alpha']}


def state_check(saved, expected, checks, prefix):
    for key in ('mean', 'V'):
        checks.array(prefix + '_' + key, saved[key], expected[key])
    for key in ('alpha', 'b'):
        checks.number(prefix + '_' + key, saved[key], expected[key])
    checks.exact(prefix + '_labels', saved['labeled_updates'], expected['labeled_updates'])


def law_check(actual, expected, checks, prefix):
    for key in ('family', 'coordinate'):
        checks.exact(prefix + '_' + key, actual[key], expected[key])
    checks.number(prefix + '_lambda', actual['lambda'], .5, 0, 0)
    checks.number(prefix + '_s0', actual['native_first_scale_s0'], expected['native_first_scale_s0'])
    if expected['family'] == 'mixture_student_t':
        checks.exact(prefix + '_components', len(actual['components']), len(expected['components']))
        checks.array(prefix + '_log_weights', actual['log_weights'], expected['log_weights'])
        for row, wanted in zip(actual['components'], expected['components']):
            checks.exact(prefix + '_component_id', row['id'], wanted['id'])
            for key in ('location', 'scale', 'df'):
                checks.number(prefix + '_' + key, row[key], wanted[key])
    else:
        for key in ('location', 'scale', 'df'):
            checks.number(prefix + '_' + key, actual[key], expected[key])


def gp10_check(state, xfit, xquery, y, checks, scalar):
    """Audit saved10D factor and dual. No fit/optimizer/predict invocation."""
    n = len(y)
    if any(len(row) != 10 for row in xfit+xquery):
        raise ValueError('exactly ten predictors required')
    checks.exact('GP_feature_count', state['feature_count'], 10)
    checks.array('GP_Xtrain', state['X_train_standardized_float64'], xfit)
    checks.array('GP_Xquery', state['X_prediction_standardized_float64'], xquery)
    checks.array('GP_y', state['y_train_native'], y)
    checks.exact('GP_train_count', state['train_rows'], n)
    checks.exact('GP_prediction_count', state['prediction_rows'], len(xquery))
    center = math.fsum(y)/n
    sd = math.sqrt(math.fsum((value-center)**2 for value in y)/n) or 1.
    checks.number('GP_normalize_center', state['y_train_mean'], center)
    checks.number('GP_normalize_SD', state['y_train_std_population'], sd)
    response = [(value-center)/sd for value in y]
    checks.array('GP_normalized_y', state['y_train_normalized'], response)
    constant, ell, white = state['constant_value'], state['RBF_length_scale'], state['white_noise_level']
    nugget = state['training_alpha_nugget']
    if not (len(ell) == 10 and .01*(1-1e-12) <= constant <= 100*(1+1e-12)
            and all(.01*(1-1e-12) <= e <= 100*(1+1e-12) for e in ell)
            and 1e-6*(1-1e-12) <= white <= 10*(1+1e-12) and nugget == 1e-8):
        raise AssertionError('registered10D kernel/bounds')
    checks.array('GP_logtheta', state['kernel_theta_log'],
                 [math.log(constant)] + [math.log(e) for e in ell] + [math.log(white)])
    lower, dual = state['L_training_cholesky'], state['alpha_dual_coefficients']
    if len(lower) != n or len(dual) != n or any(len(row) != n for row in lower):
        raise AssertionError('saved factor dimensions')
    if any(lower[i][j] != 0 for i in range(n) for j in range(i+1,n)):
        raise AssertionError('saved factor triangularity')
    matrix = [[scalar.scalar_rbf(a,b,constant,ell) + (white+nugget if i == j else 0.)
               for j,b in enumerate(xfit)] for i,a in enumerate(xfit)]
    residual = [[matrix[i][j] - math.fsum(lower[i][k]*lower[j][k]
                 for k in range(min(i,j)+1)) for j in range(n)] for i in range(n)]
    factor_abs = scalar.norm_inf(residual)
    transpose = [list(row) for row in zip(*lower)]
    factor_backward = factor_abs/(scalar.norm_inf(matrix)+scalar.norm_inf(lower)*scalar.norm_inf(transpose))
    dual_abs = max(abs(math.fsum(matrix[i][j]*dual[j] for j in range(n))-response[i]) for i in range(n))
    denominator = scalar.norm_inf(matrix)*max(map(abs,dual)) + max(map(abs,response))
    dual_backward = dual_abs/denominator if denominator else dual_abs
    if factor_backward > 1e-11 or dual_backward > 1e-11:
        raise AssertionError('GP normwise backward threshold')
    laws = []
    for point in xquery:
        cross = [scalar.scalar_rbf(point,row,constant,ell) for row in xfit]
        solved = scalar.forward_solve(lower,cross)
        variance = constant+white-math.fsum(value*value for value in solved)
        if variance <= 0:
            raise AssertionError('positive native predictive variance')
        laws.append({'coordinate':'native','family':'normal',
                     'location':center+sd*math.fsum(a*b for a,b in zip(cross,dual)),
                     'scale':sd*math.sqrt(variance)})
    return laws, {'factor_backward':factor_backward,'dual_backward':dual_backward,
                  'factor_residual_abs':factor_abs,'dual_residual_abs':dual_abs,
                  'condition_upper_bound':(n*constant+white+nugget)/(white+nugget),
                  'fit_warnings':state['fit_warnings_all_categories'],
                  'predict_warnings':state['predict_warnings_all_categories']}
