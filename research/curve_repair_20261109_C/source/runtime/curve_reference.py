"""Generic curve reference: fixed OLS-mean, observation-fitted Matérn GP.

Prepared as source only; no material fit was run while creating this module.
This is a standard Gaussian reference, not a new repair mechanism. It has no
file/data reader and accepts no target Y. An external actual-UTC/source protocol
must precede any material fit. All optimizer/length/noise bounds are fixed here;
there are no restarts, hidden jitter, variance floors or failure fallbacks.
"""
from __future__ import annotations
import json
import math
import warnings
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


METHOD_ID = 'OLS_mean_observationfitted_Matern'
LEGACY_SIX_OBSERVATION_METHOD = 'OLS_mean_anchorfitted_Matern'
SIGNAL_BOUNDS = (1e-4, 1e4)
LENGTH_BOUNDS = (1e-2, 1e2)
WHITE_BOUNDS = (1e-6, 1.)


class CurveGPFailure(RuntimeError):
    def __init__(self, reason, details):
        self.details = details
        super().__init__(reason + '; details=' + json.dumps(details, allow_nan=False, sort_keys=True))


def _vector(value, name):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in 'iuf' or
            any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat)):
        raise ValueError(name + ' must be a finite one-dimensional real vector')
    result = np.array(raw, dtype=float, copy=True)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(name + ' requires nonempty finite values')
    return result


def _warning_rows(caught):
    return [{'category': row.category.__name__, 'message': str(row.message),
             'module_file': row.filename.replace('\\', '/').split('/')[-1],
             'line': int(row.lineno)} for row in caught]


def _boundary_status(value, bounds):
    log_value, lower, upper = math.log(value), math.log(bounds[0]), math.log(bounds[1])
    # This is reporting only, matching sklearn's warning-scale closeness. No
    # optimized value is modified, clipped or restarted when it hits a bound.
    return {'value': float(value), 'bounds': list(bounds),
            'near_lower_bound': bool(np.isclose(log_value, lower)),
            'near_upper_bound': bool(np.isclose(log_value, upper)),
            'log_distance_to_lower': float(log_value-lower),
            'log_distance_to_upper': float(upper-log_value)}


def fit_predict(xO, yO, xT):
    """Fit n >= 4 own observations and predict a joint Gaussian at xT.

xO and xT are finite, declared one-dimensional coordinates. yO is the declared
real-valued response. This module applies no logarithm or unit conversion and
uses no population statistics. The caller must freeze the coordinate/response
representation before observing held-out targets; fixed length bounds are in
those declared x units and do not imply invariance to arbitrary unit changes.

The OLS mean is beta[0] + beta[1] * x. The residual scale is
sqrt(SSE / (n - 2)), using only these n observations. GPR acts on residual/s
with normalize_y=False and the fixed Matérn3/2 plus WhiteKernel design.
Returned covariance includes observational white variance and is rescaled by
s². OLS and kernel parameters are point estimates: their uncertainty is not
integrated. Exact/numerically unresolved linear response is rejected without
variance floor or fallback. This function accepts no target response values.

At n=6 the numerical design, optimizer, scale rule and predictive law match
the frozen legacy six-anchor reference, before any differences in metadata.
    """
    xo, yo, xt = _vector(xO, 'xO'), _vector(yO, 'yO'), _vector(xT, 'xT')
    n = len(xo)
    if n < 4 or yo.shape != xo.shape:
        raise ValueError('At least four own observed coordinates and matching values are required')
    if (len(np.unique(xo)) != n or len(np.unique(xt)) != len(xt) or
            len(np.intersect1d(xo, xt))):
        raise ValueError('Distinct observed/target coordinates and disjoint masks are required')
    design = np.column_stack([np.ones(n), xo])
    target_design = np.column_stack([np.ones(len(xt)), xt])
    beta, _, rank, _ = np.linalg.lstsq(design, yo, rcond=None)
    residual = yo-design@beta
    sse = float(residual@residual)
    resolution = float(64*np.finfo(float).eps*(np.linalg.norm(design,2)*np.linalg.norm(beta)+np.linalg.norm(yo)))
    threshold = resolution**2
    details = {'method_id': METHOD_ID, 'n': n, 'p_OLS': 2, 'residual_degrees_of_freedom': n-2,
               'OLS_beta': beta.tolist(), 'rank_OLS': int(rank),
               'SSE': sse if math.isfinite(sse) else None,
               'logSSE': math.log(sse) if math.isfinite(sse) and sse>0 else None,
               'residual_norm_resolution': resolution if math.isfinite(resolution) else None,
               'SSE_resolution_threshold': threshold if math.isfinite(threshold) else None,
               'resolution_multiplier': 64, 'optimizer_warnings': [],
               'sklearn_version': sklearn.__version__, 'target_response_argument_accepted': False}
    if (rank!=2 or np.all(yo==yo[0]) or not math.isfinite(sse) or sse<=0 or
            not math.isfinite(threshold) or sse<=threshold):
        raise CurveGPFailure('STOP_NO_FALLBACK zero/rank-deficient/numerically unresolved anchor residual', details)
    residual_scale=math.sqrt(sse/(n-2))
    standardized=residual/residual_scale
    kernel=(ConstantKernel(1.,SIGNAL_BOUNDS)*Matern(length_scale=1.,length_scale_bounds=LENGTH_BOUNDS,nu=1.5)
            +WhiteKernel(noise_level=1e-2,noise_level_bounds=WHITE_BOUNDS))
    gp=GaussianProcessRegressor(kernel=kernel,alpha=0.,optimizer='fmin_l_bfgs_b',
                               n_restarts_optimizer=0,normalize_y=False,copy_X_train=True,random_state=None)
    caught=[]
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            caught=captured
            gp.fit(xo[:,None],standardized)
            standardized_mean,standardized_covariance=gp.predict(xt[:,None],return_cov=True)
    except Exception as error:
        details['optimizer_warnings']=_warning_rows(caught)
        details['fit_exception_type']=type(error).__name__
        details['fit_exception_message']=str(error)
        raise CurveGPFailure('STOP_NO_FALLBACK own-anchor GP fit/predict failed', details) from error
    details['optimizer_warnings']=_warning_rows(caught)
    mean=target_design@beta+residual_scale*standardized_mean
    covariance=(residual_scale**2)*standardized_covariance
    if (mean.shape!=(len(xt),) or covariance.shape!=(len(xt),len(xt)) or
            not np.isfinite(mean).all() or not np.isfinite(covariance).all() or
            not np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-12)):
        raise CurveGPFailure('STOP_NO_FALLBACK invalid predictive Gaussian law', details)
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise CurveGPFailure('STOP_NO_FALLBACK non-SPD covariance, no jitter/repair',details) from error
    signal=float(gp.kernel_.k1.k1.constant_value)
    length=float(gp.kernel_.k1.k2.length_scale)
    noise=float(gp.kernel_.k2.noise_level)
    return {**details, 'distribution':'multivariate_gaussian','mean':mean.tolist(),
            'covariance':covariance.tolist(), 'residual_scale':residual_scale,
            'residual_scale_rule':'sqrt(own-observation SSE/(n-2))',
            'standardized_anchor_residual':standardized.tolist(),
            'kernel_initial':'Constant(1,(1e-4,1e4))*Matern(length1,bounds(1e-2,1e2),nu1.5)+White(1e-2,(1e-6,1))',
            'kernel_fitted_repr':str(gp.kernel_),
            'kernel_theta_log':gp.kernel_.theta.tolist(),
            'hyperparameters':{'signal_variance_standardized':signal,'length_scale_x_units':length,
                               'white_variance_standardized':noise,
                               'white_variance_y_units':residual_scale**2*noise},
            'boundary_status':{'signal':_boundary_status(signal,SIGNAL_BOUNDS),
                               'length_scale':_boundary_status(length,LENGTH_BOUNDS),
                               'white_noise':_boundary_status(noise,WHITE_BOUNDS)},
            'optimizer':'fmin_l_bfgs_b','n_restarts_optimizer':0,'alpha':0.,'normalize_y':False,
            'optimizer_termination_status':'not separately exposed by sklearn default; all warnings retained',
            'log_marginal_likelihood_standardized':float(gp.log_marginal_likelihood_value_),
            'log_marginal_likelihood_y_units':float(gp.log_marginal_likelihood_value_-n*math.log(residual_scale)),
            'covariance_includes_observational_white_variance':True,
            'hyperparameter_posterior_uncertainty_integrated':False,
            'OLS_mean_uncertainty_integrated':False,
            'synthetic_samples_created':0,'observed_response_values_used':n,
            'legacy_six_observation_method':LEGACY_SIX_OBSERVATION_METHOD,
            'coordinate_transform':'none; declared by external protocol',
            'response_transform':'none; declared by external protocol',
            'target_response_values_accepted':False,'no_fallback_no_jitter_no_variance_floor':True}
