"""Prior-only basis constructors; no measurement labels or fitted parameters."""
import numpy as np
from scipy.linalg import solve_triangular

from assets import METHODS, SEEDS
from joint_code_gaussian import CodeReference
from local_shape import local_curvature_functionals, whiten_functionals

ROTATION45 = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)


def directions(method, covariance, support):
    if method not in METHODS:
        raise ValueError('Unregistered representation')
    k = np.asarray(covariance)
    q = np.asarray(support)
    if k.shape != (17, 17) or q.shape != (17,):
        raise ValueError('The original17-point support is required')
    if method.startswith('random_'):
        seed = int(method.split('_')[1])
        if seed not in SEEDS:
            raise ValueError('Unregistered seed')
        # Exact same distribution/construction as original seed20261030.
        u = np.linalg.qr(np.random.default_rng(seed).normal(size=(17, 2)))[0]
    elif method == 'prior_kl':
        eigenvalues, eigenvectors = np.linalg.eigh((k + k.T) / 2.)
        order = np.argsort(eigenvalues)[::-1][:2]
        values, vectors = eigenvalues[order], eigenvectors[:, order].copy()
        if values[-1] <= 0:
            raise ValueError('Positive prior-KL modes required; no jitter')
        for column in range(2):
            pivot = np.argmax(np.abs(vectors[:, column]))
            if vectors[pivot, column] < 0:
                vectors[:, column] *= -1
        # z_j=v_j.T(fQ-mQ)/sqrt(lambda_j)=u_j.T L^-1(fQ-mQ).
        u = np.linalg.cholesky(k).T @ (vectors / np.sqrt(values))
    else:
        original = whiten_functionals(k, local_curvature_functionals(q))['directions']
        u = original @ ROTATION45
    if not np.allclose(u.T @ u, np.eye(2), atol=1e-10, rtol=0):
        raise ValueError('Whitened basis failed orthonormality')
    return u


def reference(model, coordinate, support, basis):
    n = len(coordinate)
    law = model.report_latent_joint(coordinate, support)
    lower = np.linalg.cholesky(law.cov[n:, n:])
    cross = law.cov[:n, n:] @ solve_triangular(lower.T, basis, lower=False)
    return CodeReference(law.mean[:n], law.cov[:n, :n], cross)
