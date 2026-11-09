"""Prospective numerical revision: the original GP with a per-recipe objective.

Only optimizer objective/gradient units change. The registered V1 implementation,
model, inputs, covariance, and total-unit likelihood evaluation remain intact.
"""
import numpy as np
from scipy.optimize import minimize

from coordinate_reference import CoordinateGP, FitConfig, Normalization, _curves, observed_nll


class ScaledCoordinateGP(CoordinateGP):
    @classmethod
    def fit(cls, curves, config=FitConfig(gtol=1e-4)):
        curves = _curves(curves)
        normalization = Normalization.from_curves(curves)
        count = len(curves)

        def objective(theta):
            value, gradient, _ = observed_nll(theta, curves, normalization)
            return value / count, gradient / count

        result = minimize(objective, np.log(config.initial), jac=True, method="L-BFGS-B",
                          bounds=np.log(config.bounds), options={"maxiter": config.maxiter,
                          "maxfun": config.maxfun, "ftol": config.ftol, "gtol": config.gtol})
        value, gradient, beta = observed_nll(result.x, curves, normalization)
        receipt = {"optimization_performed": True, "runs": 1, "method": "L-BFGS-B",
                   "success": bool(result.success), "status": int(result.status),
                   "message": str(result.message), "nit": int(result.nit),
                   "nfev": int(result.nfev), "njev": int(result.njev),
                   "observed_nll": value, "gradient_inf": float(np.max(np.abs(gradient))),
                   "optimizer_objective": value / count,
                   "optimizer_gradient_per_recipe": (gradient / count).tolist(),
                   "optimizer_gradient_inf_per_recipe": float(np.max(np.abs(gradient)) / count),
                   "optimizer_normalization": {"divisor": count,
                       "objective": "total observed NLL / n_recipes",
                       "gradient": "total log-hyperparameter gradient / n_recipes",
                       "gtol_units": "projected log-hyperparameter gradient per recipe",
                       "statistical_model_changed": False},
                   "n_recipes": count, "n_visible": sum(len(c.y) for c in curves),
                   "config": {key: getattr(config, key) for key in config.__dataclass_fields__}}
        return cls(normalization, result.x, beta, receipt)
