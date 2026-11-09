"""SOURCE ONLY: classical generalized-Maxwell same-budget point comparator.

No file reader, automatic experiment, stochastic prediction or uncertainty
score. Calling fit_predict on material responses requires a separate actualUTC
protocol. This source has not been fitted to DaRUS material values.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import nnls


def _vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'iuf' or any(
            isinstance(v, (bool, np.bool_)) for v in np.asarray(values, dtype=object).flat):
        raise ValueError(name + ' requires a finite real vector')
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(name + ' must be finite')
    return out


def fixed_relaxation_times(full_native_frequencies_Hz):
    """Six fixed log-spaced tau from input support only, never response Y."""
    f = _vector(full_native_frequencies_Hz, 'full_native_frequencies_Hz')
    if len(f) < 6 or np.any(f <= 0) or len(np.unique(f)) != len(f):
        raise ValueError('Positive distinct original input frequencies required')
    return np.geomspace(1/(2*math.pi*float(f.max())),
                        1/(2*math.pi*float(f.min())), 6)


def design(frequencies_Hz, relaxation_times_s):
    """Columns G_infinity[Pa], six G_k[Pa], eta_s[Pa.s]; rows Gprime then Gdoubleprime.

    Gprime(w)=Ginf+sum Gk*(w*tauk)^2/(1+(w*tauk)^2)
    Gdoubleprime(w)=eta_s*w+sum Gk*(w*tauk)/(1+(w*tauk)^2).
    Frequency is converted Hz to angular frequency rad/s exactly with2pi.
    It is never interpreted as a steady shear rate.
    """
    f = _vector(frequencies_Hz, 'frequencies_Hz')
    tau = _vector(relaxation_times_s, 'relaxation_times_s')
    if not len(f) or np.any(f <= 0) or tau.shape != (6,) or np.any(tau <= 0):
        raise ValueError('Positive frequencies and exactly six positive fixed tau required')
    omega = 2*math.pi*f
    product = omega[:, None]*tau[None, :]
    denominator = 1+product*product
    storage = np.column_stack((np.ones(len(f)), product*product/denominator, np.zeros(len(f))))
    loss = np.column_stack((np.zeros(len(f)), product/denominator, omega))
    out = np.vstack((storage, loss))
    if not np.isfinite(out).all():
        raise ValueError('Unrepresentable Maxwell design; no clipping')
    return out


def fit_predict(observed_frequencies_Hz, observed_Gprime_Pa, observed_Gdoubleprime_Pa,
                target_frequencies_Hz, full_native_frequencies_Hz):
    """Own6frequency/12scalar observations ->7frequency/14scalar point predictions.

    All measured moduli must be explicitly converted MPa->Pa by the external
    source reader. Unweighted least squares in common Pa units is fixed here;
    no target-dependent channel weights, tau optimization or curve resampling.
    Negative measured readings are retained in the declared LS objective,
    never floored/deleted. Predictions are constrained by the physical cone.
    """
    fo = _vector(observed_frequencies_Hz, 'observed_frequencies_Hz')
    ft = _vector(target_frequencies_Hz, 'target_frequencies_Hz')
    gp = _vector(observed_Gprime_Pa, 'observed_Gprime_Pa')
    gl = _vector(observed_Gdoubleprime_Pa, 'observed_Gdoubleprime_Pa')
    native = _vector(full_native_frequencies_Hz, 'full_native_frequencies_Hz')
    if fo.shape != (6,) or gp.shape != (6,) or gl.shape != (6,) or ft.shape != (7,):
        raise ValueError('Exactly six paired observed and seven paired target frequencies required')
    if (len(np.unique(fo)) != 6 or len(np.unique(ft)) != 7 or np.intersect1d(fo, ft).size or
            len(native) != 13 or set(np.concatenate((fo, ft))) != set(native)):
        raise ValueError('Disjoint native6/7 frequency partition required; no interpolation')
    tau = fixed_relaxation_times(native)
    a = design(fo, tau); target = design(ft, tau)
    if np.linalg.matrix_rank(a) != 8:
        raise ValueError('Unresolved Maxwell design rank; stop without fallback')
    y = np.concatenate((gp, gl))
    column_norm = np.linalg.norm(a, axis=0)
    if np.any(column_norm <= 0) or not np.isfinite(column_norm).all():
        raise ValueError('Invalid design columns')
    # Positive column rescaling is solely an exact reparameterization of the
    # same raw-Pa least-squares objective. It is not response normalization.
    scaled_coefficients, scaled_residual_norm = nnls(a/column_norm, y, maxiter=600)
    coefficients = scaled_coefficients/column_norm
    fitted = a@coefficients; prediction = target@coefficients
    if not np.isfinite(prediction).all() or np.any(coefficients < 0):
        raise ArithmeticError('Invalid NNLS solution; no repair')
    return {'model': 'classical_generalized_Maxwell_fixed6tau_NNLS',
            'prediction_kind': 'point_only_no_stochastic_law',
            'observed_frequency_budget': 6, 'observed_scalar_response_budget': 12,
            'target_frequency_count': 7, 'target_scalar_response_dimension': 14,
            'fixed_relaxation_times_s': tau.tolist(),
            'coefficients': {'G_infinity_Pa': float(coefficients[0]),
                             'G_modes_Pa': coefficients[1:7].tolist(),
                             'eta_solvent_Pa_s': float(coefficients[7])},
            'predicted_Gprime_Pa': prediction[:7].tolist(),
            'predicted_Gdoubleprime_Pa': prediction[7:].tolist(),
            'own_observed_Gprime_fit_Pa': fitted[:6].tolist(),
            'own_observed_Gdoubleprime_fit_Pa': fitted[6:].tolist(),
            'own_observed_residual_sum_squares_Pa2': float(np.sum((y-fitted)**2)),
            'nnls_reported_residual_norm_Pa': float(scaled_residual_norm),
            'parameter_count': 8, 'design_rank': 8, 'nnls_maxiter': 600,
            'loss': 'unweighted least squares over12 paired scalar readings in Pa',
            'frequency_coordinate': 'omega_rad_per_s=2*pi*frequency_Hz',
            'target_responses_received': 0, 'probabilistic_scores_supported': False,
            'relaxation_times_fitted': False, 'material_performance_established': False}
