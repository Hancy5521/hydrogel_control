# Frozen inference verification

This small addition checks the **current frozen hydrogel continuation inference**
against an independently assembled Gaussian calculation. It adds no model,
training run, measured material, efficacy score or calibration result.

Use Python 3.10 or newer with the versions in the repository's `requirements.txt`.
Place this directory at `verification/` in a checkout, then run from that checkout:

```sh
python verify_checkpoint.py
python verification/verify_frozen_inference.py --repo-root . --output new_software_result.json
```

The verification directory can also remain elsewhere: invoke its script and pass
the checkout path through `--repo-root`. An existing result file is never
overwritten. No optimizer is run. The script reads the public saved model and
runtime; it does not open original material curves or hidden S outcomes. Its CLI
smoke check writes only temporary synthetic inputs and outputs.

## What is independent

`independent_gaussian.py` imports NumPy and SciPy only. It explicitly constructs
the normalized level, slope and Matern-3/2 covariance and the report nugget in
physical log-response units. It builds finite curvature by differencing adjacent
slopes, averages the two fixed physical-coordinate windows, symmetrically
whitens their prior covariance, retains `K0 - C C.T`, inserts the frozen Gaussian
code mean/covariance and performs dense block conditioning.

These independently assembled moments determine the expected analytic quantiles
and the expected full-covariance seeded draws. The production generator is the
implementation under test. The oracle does not call its kernel, feature,
whitening or conditioning helpers. This is independence of the numerical
implementation, not independent authorship or new material validation.

## Ten software checks

1. Independent full-joint quantiles and correlated draws.
2. Anchor and target ordering invariance of analytic predictions.
3. Agreement of target-batch **marginals** with the joint query.
4. Analytic summaries unaffected by seed or draw count.
5. Exact stored-anchor identity, explicitly distinct from a fresh measurement.
6. Sequential and simultaneous assimilation of the same six anchors, using the
   lower-level Gaussian interface and independently assembled full joint law.
7. Affine-function annihilation of the contrasts and noncollapsed residual on
   the synthetic query.
8. Rejection of invalid values, dimensions, hidden-target fields and support.
9. Strict JSON parsing, no-clobber CLI output and model/input hash binding.
10. Closed-support endpoint agreement and rejection of distinct physical rates
    that alias in binary floating-point log coordinates.

The public `generate()` interface still requires exactly six anchors. The
sequential check does not extend that interface. Separate target batches do not
retain random dependence between batches, and fixed-seed Cholesky draws need
not be samplewise invariant under target permutations. The tests assert the
appropriate distributions or marginal summaries, rather than those false
samplewise requirements.

## Synthetic fixture and result

Every anchor in `SYNTHETIC_INPUT.json` is created by the explicit numerical rule

`eta = exp(0.3 - 0.35*ln(gamma) + 0.06*sin(ln(gamma)))`.

These values are not measured hydrogel records or independent test labels. They
provide a nontrivial input to qualify the numerical interface. The saved
`SOFTWARE_VERIFICATION_RESULT.json` records the actual execution UTC, software
versions, source/model/input hashes, errors and test status. Its result is
software evidence only. `VERIFICATION_PROTOCOL.json` records the test design
and the addition of the support/alias boundary check before the final run.

Earlier local scientific audits already reconstructed the saved outer-fold
objectives, gradients and predictive laws. They are retained rather than rerun
here. This portable addition closes a different gap: an independent check that
can run from public inference parameters without the original material cache.
It does not reproduce the paper's leave-one-formulation-out CRPS from the
full-data deployment weights.

## Limits and rights

This checks the current `same_local_code_jointGaussian_v1` inference entry point,
not every training or legacy baseline routine retained in the runtime bundle.
It does not test arbitrary memory budgets, hostile service workloads or every
possible floating-point environment. A caller deploying a service must set
resource limits; the frozen CLI is a local numerical tool. Passing checks does
not establish new-batch transfer, calibrated intervals or experimental savings.

No new reuse license is assigned by this addition. Source and reuse permissions
remain subject to the repository owner's completed licensing decision.
