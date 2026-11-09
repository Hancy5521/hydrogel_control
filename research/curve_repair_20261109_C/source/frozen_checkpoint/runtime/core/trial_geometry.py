"""Fixed reference projections and covariance-only channel diagnostics."""
import numpy as np
from scipy.linalg import solve_triangular
from latent_reference import GaussianReference, LatentProjection
from local_shape import local_curvature_functionals, whiten_functionals, conditional_channel_statistics


def projection(model, report_coordinates, q, basis, generic):
    law = model.report_latent_joint(report_coordinates, q)
    qi = np.arange(len(report_coordinates), len(report_coordinates) + len(q))
    k = law.cov[np.ix_(qi, qi)]
    if basis == 'generic':
        u = generic
    elif basis == 'local':
        u = whiten_functionals(k, local_curvature_functionals(q))['directions']
    else:
        raise ValueError('Only the two registered bases are allowed')
    return LatentProjection(GaussianReference(law.mean, law.cov), qi, u)


def diagnostics(p, observed, values, hidden):
    _, v = p.reference.posterior(observed, values)
    q = p.q
    l = np.linalg.cholesky(p.reference.covariance[np.ix_(q, q)])
    vx = solve_triangular(l, v[np.ix_(q, q)], lower=True)
    vx = solve_triangular(l, vx.T, lower=True).T
    cross = solve_triangular(l, v[np.ix_(q, hidden)], lower=True).T
    return conditional_channel_statistics((vx + vx.T) / 2, cross,
        v[np.ix_(hidden, hidden)], p.directions)
