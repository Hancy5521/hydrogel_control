# Arbitrary-mask curve repair API — implementation boundary

Prepared 2026-11-08. These new modules are separate from the frozen runtime.
They do not modify `deployment/generate_curve.py`, its exactly-six-positive-
anchors contract, the frozen model, old sources, archives, or any GitHub object.
Source preparation is not a material experiment or a performance result.

## Modules

| File | Responsibility |
| --- | --- |
| `frozen_loader.py` | Verify the frozen core SHA before importing its exact bytes. No model JSON or material labels are loaded. |
| `arbitrary_mask.py` | Role manifest, observed-only payload, arbitrary finite-grid conditioning, full covariance, correlated generation. |
| `early_influence.py` | Authorized train/development conditional NLL derivatives and local candidate-update diagnostics. No fitting or step selection. |
| `synthetic_gates.py` | Prepared fixed 7×7 matrix checks; run only with root's actual-UTC/source-hash authorization. |

The fixed-grid reference must already be built from allowed inputs by an
externally audited pipeline. This API does **not** implement continuous
shear-rate/frequency projection, a new GP builder, extrapolation, or refitting.
Grid indices may arrive in any order and may form several separated internal
holes. Arbitrary grid masks are distinct from arbitrary new physical positions.

## Minimal use

Run examples from the parent directory of `core/` so namespace imports resolve.
The snippet is an API example, not a report that material values were used.

```python
import numpy as np
from core.frozen_loader import load_frozen_core
from core.arbitrary_mask import (
    CurveIdentity, MaskRoles, ObservedReadings, ArbitraryMaskConditioner,
)

frozen = load_frozen_core()  # only verified mathematical source is imported
# m[n], K[n,n], B[n,2] must come from an admitted, frozen reference builder.
reference = frozen.CodeReference(m, K, B)
identity = CurveIdentity("registered_source", "alias_closed_group", "curve_id", "unseen")
roles = MaskRoles(identity, n, observed=[5, 0, 2], pseudo_hole_train=[],
                  sentinel_dev=[], unseen=[i for i in range(n) if i not in [5, 0, 2]])
observed = ObservedReadings(identity, [5, 0, 2], values_log_eta,
                            additional_noise_variance=[0.02, 0.0, 0.01])
repair = ArbitraryMaskConditioner(reference, roles, observed)
posterior = repair.condition(theta=np.zeros(5), target_indices=[6, 1, 4, 3])
target_mean, target_covariance = posterior.marginal()
draws = posterior.sample(100, np.random.default_rng(123))  # [100, 4], same target order
```

`posterior.mean` and `.covariance` always cover the **entire frozen grid** in its
original order, not just the two-dimensional code or requested target block.
`observed_indices` preserve reading order; `target_indices` preserve requested
order. A `None` target selects all non-observed grid indices in ascending order.
Targets cannot include observed indices. Call `.marginal(other_indices)` to
inspect any full-law block, including uncertain noisy observed variables.

Values, means, covariance and all extra noise variances use the frozen
reference's log-viscosity coordinates. Passing raw positive viscosity values or
standard deviations as variances is a caller error: neither can be inferred
reliably from a numeric array. To generate viscosity samples, explicitly
exponentiate generated log samples; `exp(mean_log)` is a median, not an arithmetic
posterior mean in viscosity units.

## Complete residual and observation noise

The frozen core defines

```text
m_theta = m_ref + B a
K_theta = K_ref + B (S - I) B^T
         = (K_ref - B B^T) + B S B^T
```

The wrapper conditions this **complete** Gaussian. At zero additional noise it
delegates the non-observed block to the original `CodeReference.condition` and
fixes exactly observed rows. With known extra independent observation-noise
variances `r`, it uses

```text
A = K_theta[O,O] + diag(r)
m_post = m_theta + K_theta[:,O] solve(A, y_O - m_theta[O])
K_post = K_theta - K_theta[:,O] solve(A, K_theta[O,:])
```

There is no code-only posterior with an unconditional high-dimensional residual.
The existing reference discrepancy/nugget stays in `K_ref`. `r` represents
additional independent measurement variance **beyond** that reference; do not
add an existing nugget a second time. A positive `r` uses an observation
likelihood and leaves uncertainty at that observed coordinate. Only `r=0`
anchors are exactly fixed. "Latent" in `PosteriorGaussian` means the reference
variable before this *additional* noise; it does not assert an identified smooth
physical curve with the frozen discrepancy removed. Correlated extra noise is
not implemented in this first interface.

No jitter, diagonal variance replacement, or fitting fallback is used. Singular
posterior sampling uses an eigendecomposition; only tiny nonpositive
eigenvalues within an explicit floating-point tolerance are rounded to zero.
Materially negative eigenvalues raise an error. Complete samples keep zero-noise
observed values exact. Empty observed and empty target sets are supported.

## Identity and reference strength

`repair.condition(theta, v=...)` uses the explicit parameter path
`theta_effective = v * theta`, with finite `0 <= v <= 1`. The original five
parameters retain their frozen meaning. At `v=0`, effective parameters are
exact zeros and the result is the unchanged **full reference conditional**.
This is a Gaussian parameter path; it is not a learned transport, nonlinear
flow, diffusion process, or a new claim of Gaussian probability theory.

The old strong reference remains directly loadable. `ConditionalObjective`,
`CurveTask`, the original optimizer and six-anchor deployment interface remain
unchanged. `GaussianConditionalScore` reuses `nll_gradient` when all extra noise
is zero, and supplies its analytic five-parameter extension for known extra
noise. The original `ConditionalObjective` requires complete authorized
training curves; it is not called with unseen/development labels by this API.

## Roles and information boundaries

Every curve carries `(source_id, group_id, curve_id, split)`. A `MaskRoles`
manifest partitions the whole fixed grid, disjointly and exhaustively:

| Role | Can enter conditioning? | Can enter this module's scoring API? |
| --- | --- | --- |
| `observed` | Yes: explicit currently available readings only | No held-out score role |
| `pseudo_hole_train` | No | Yes, on `split="train"` curves only |
| `sentinel_dev` | No | Yes, on `split="dev"` curves only |
| `unseen` | No | Never |

`ObservedReadings` contains only the manifest's observed values. No complete
hidden curve is accepted by the wrapper. Held-out label payloads are separate
`ScoringReadings` and must be a subset of the matching permitted role and
identity. On `split="unseen"`, pseudo-hole and sentinel roles are forbidden.
Prediction itself can cover any non-observed role without reading its truth.

After a genuinely acquired new measurement, a caller may create a new frozen
manifest with that coordinate promoted to observed and record its real query
ticket/cost externally. A score label is not automatically promoted. This API
does not certify whether supplied labels were truly measured, whether group
aliases were closed, or whether the source reference used impermissible data.
Those require the predeclared split/source/query ledger. Direct train/dev group
identity overlap is rejected by `inspect_parameter_update`; renamed or
cross-source duplicates still need external alias closure. Final evaluation
must freeze predictions before separately reading final truth; this source has
no final-truth evaluator.

## Early influence interface

```python
from core.early_influence import (
    ScoringReadings, GaussianConditionalScore, inspect_parameter_update,
)

# train_repair and dev_repair have distinct permitted group identities.
train_score = GaussianConditionalScore(train_repair, ScoringReadings(
    train_identity, "pseudo_hole_train", train_indices, train_withheld_log_values))
dev_score = GaussianConditionalScore(dev_repair, ScoringReadings(
    dev_identity, "sentinel_dev", dev_indices, dev_withheld_log_values,
    additional_noise_variance=dev_label_variances))
value, gradient = train_score(theta)  # NLL(Y_T | observed Y_O), five derivatives
diagnostic = inspect_parameter_update(
    theta, [train_score], [dev_score],
    step_sizes=[0.0, 0.001, 0.01],
    evaluate_development_steps=False,
    development_use_id="registered_dev_use_0001",
)
```

The default candidate direction is the negative mean training score gradient;
an explicit five-vector `direction=` is also accepted. For development score
gradient `g_dev`, predicted risk change is
`step * (g_dev @ direction)`. The routine records baseline gradients, the local
warning, step sensitivity, label-using call counts, and that it selected no
step and ran no optimizer. Optional `evaluate_development_steps=True` evaluates
actual development NLL at every nonzero requested step and counts those calls.
That option is development tuning, including when used repeatedly.
Candidate endpoints follow an **unconstrained diagnostic** parameter path. This
routine does not project the frozen optimizer's `BOUNDS` or admit a fitted
update; a later comparison must declare its parameter constraints and budgets.

These are local supervised diagnostics using permitted labels, not estimates
of all unseen truths or guarantees that a finite update is beneficial. The
method does not differentiate through RF split construction. Task averaging is
equal per score task; repeated masks change curve weighting, so masks/weights
must be predeclared by the caller. This interface does not expand the original
five-parameter Gaussian correction family's expressive capacity by itself.

The root's separate query-effect component may consume this full posterior to
predict a prospective measurement's response/expected uncertainty reduction.
Fixed-Gaussian covariance rankings are outcome independent; they do not become
value-adaptive merely because observed means change.

## Prepared gates and exact execution boundary

No `run_gates` execution was performed while preparing these sources. The root
must complete all edits, obtain actual UTC, hash all `core/*.py`, and write a
JSON authorization containing:

```json
{
  "scope": "fixed_matrix_synthetic_only",
  "authorized_by": "root",
  "root_actual_utc": "<actual UTC with Z or +00:00>",
  "frozen_core_sha256": "0c8a57c1bdf96e1c880ffd1ba2258e1b379952b844e960a7c67d6cdcdc65e299",
  "source_sha256": {"<every core Python filename>": "<actual SHA256>"}
}
```

From the extension root, the prepared command is:

```powershell
& 'D:/Anaconda/python.exe' -B -m core.synthetic_gates `
  --root-authorization '<absolute root authorization JSON>' `
  --output '<new absolute synthetic result JSON>'
```

The gate runner rechecks all source hashes and refuses a future timestamp or
changed source bytes. A successful receipt covers only the fixed synthetic
matrix. It records zero material loads, fits, old-25 scores and model-JSON
loads. It cannot establish material accuracy, publication novelty, calibrated
coverage, data efficiency, or acceptance by a venue. Real experiments remain a
separate predeclared task with untouched unseen data.
