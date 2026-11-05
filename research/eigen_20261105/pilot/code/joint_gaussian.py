"""A fixed warm ICM reference and exact noiseless conditioning of source labels.

The .05 diagonal is a preregistered independent response residual in the joint
law, not jitter. Observing a cell determines it exactly; the resulting full
posterior is PSD, with positive conditional covariance on unobserved cells.
"""
from dataclasses import dataclass
import numpy as np


def warm_reference(features, warm_values):
    x, y = np.asarray(features, dtype=float), np.asarray(warm_values, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 3 or len(y) < 2:
        raise ValueError('Finite metadata features and at least two complete warm rows required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite features/warm responses required')
    location, scale = y.mean(axis=0), y.std(axis=0, ddof=1)
    constant = scale == 0
    scale[constant] = 1.
    standardized = (y - location) / scale
    b = .8 * np.cov(standardized, rowvar=False, ddof=1) + .2 * np.eye(3)
    squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    kx = np.exp(-.5 * squared)
    covariance = np.kron(kx, b) + .05 * np.eye(3 * len(x))
    np.linalg.cholesky(covariance)
    details = {'warm_location': location.tolist(), 'warm_scale': scale.tolist(),
               'constant_warm_properties': np.flatnonzero(constant).tolist(),
               'task_covariance_B': b.tolist(), 'kernel': 'RBF lengthscale1 in metadata RMS-feature-distance units',
               'independent_response_residual_variance': .05,
               'task_relation_frozen_after_warm_fit': True}
    return GaussianState(np.zeros(3 * len(x)), covariance), location, scale, details


@dataclass
class GaussianState:
    mean: np.ndarray
    covariance: np.ndarray
    observed: np.ndarray | None = None

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float).copy()
        self.covariance = np.asarray(self.covariance, dtype=float).copy()
        n = len(self.mean)
        if self.mean.shape != (n,) or self.covariance.shape != (n, n) or n == 0:
            raise ValueError('Aligned nonempty Gaussian law required')
        if (not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all()
                or not np.allclose(self.covariance, self.covariance.T, rtol=1e-12, atol=1e-12)):
            raise ValueError('Finite symmetric law required')
        self.observed = np.zeros(n, dtype=bool) if self.observed is None else np.asarray(self.observed, dtype=bool).copy()
        if self.observed.shape != (n,):
            raise ValueError('Aligned observed mask required')
        self._check_unobserved()

    def _indices(self, indices):
        if not isinstance(indices, (list, tuple)) or not indices or any(type(i) is not int for i in indices):
            raise ValueError('Nonempty integer index list required')
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(self.mean):
            raise ValueError('Distinct indices inside the joint law required')
        return np.asarray(indices, dtype=int)

    def _check_unobserved(self):
        unknown = np.flatnonzero(~self.observed)
        if len(unknown):
            diag = self.covariance[unknown, unknown]
            if not np.isfinite(diag).all() or (diag <= 0).any():
                raise ValueError('Unobserved conditional variances must remain strictly positive; no clipping')

    def condition(self, indices, values):
        indices = self._indices(indices)
        y = np.asarray(values, dtype=float)
        if y.shape != indices.shape or not np.isfinite(y).all() or self.observed[indices].any():
            raise ValueError('Finite aligned newly observed responses required')
        cross = self.covariance[:, indices].copy()
        block = self.covariance[np.ix_(indices, indices)]
        chol = np.linalg.cholesky(block)
        whitened = np.linalg.solve(chol, cross.T)
        delta = np.linalg.solve(chol, y - self.mean[indices])
        self.mean = self.mean + whitened.T @ delta
        self.covariance = self.covariance - whitened.T @ whitened
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.observed[indices] = True
        # Exact identities for noiselessly revealed cells, not variance clipping.
        self.mean[indices] = y
        known = np.flatnonzero(self.observed)
        self.covariance[known, :] = 0.
        self.covariance[:, known] = 0.
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise OverflowError('Posterior update overflow')
        self._check_unobserved()

    def acquisition_gains(self, indices, test_indices, frontier_target_indices,
                          initial_test_trace, initial_frontier_sd, utility_sign=1):
        indices = self._indices(indices)
        test = self._indices(test_indices)
        frontier = self._indices(frontier_target_indices)
        if self.observed[indices].any():
            raise ValueError('Acquisition must be a fully unknown bundle')
        if type(utility_sign) is not int or utility_sign not in (-1, 1):
            raise ValueError('Predeclared scalar property utility sign required')
        if not np.isfinite(initial_test_trace) or initial_test_trace <= 0 or not np.isfinite(initial_frontier_sd) or initial_frontier_sd <= 0:
            raise ValueError('Fixed finite positive initial normalizers required')
        chol = np.linalg.cholesky(self.covariance[np.ix_(indices, indices)])
        test_effect = np.linalg.solve(chol, self.covariance[np.ix_(indices, test)])
        ivr = float(np.sum(test_effect ** 2) / initial_test_trace)
        effect = np.linalg.solve(chol, self.covariance[np.ix_(indices, frontier)]).T
        nodes, weights = np.polynomial.hermite.hermgauss(5)
        nodes, weights = np.sqrt(2.) * nodes, weights / np.sqrt(np.pi)
        if len(indices) == 1:
            normals, probability = nodes[:, None], weights
        elif len(indices) == 2:
            a, b = np.meshgrid(nodes, nodes, indexing='ij')
            normals = np.column_stack((a.ravel(), b.ravel()))
            probability = np.outer(weights, weights).ravel()
        else:
            raise ValueError('This fixed pilot only supports one/two-cell bundles')
        utility_mean = utility_sign * self.mean[frontier]
        fantasy = utility_mean[None, :] + utility_sign * (normals @ effect.T)
        kg = float((probability @ np.max(fantasy, axis=1) - np.max(utility_mean)) / initial_frontier_sd)
        if not np.isfinite(ivr) or not np.isfinite(kg) or ivr < -1e-12 or kg < -1e-12:
            raise ArithmeticError('IVR/KG nonnegativity or finiteness failed')
        # Preserve any tiny signed quadrature residual; do not silently tune it.
        return {'IVR': ivr, 'KG': kg, 'joint': .5 * (ivr + kg)}
