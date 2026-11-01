# Synthetic training-primitive demonstration

This module exposes the existing five-parameter training primitive in the
frozen `runtime/core/joint_code_gaussian.py`. It is separate from the inference
verification directory and leaves the frozen code and model unchanged.

It uses **only synthetic variables**: eight complete 39-dimensional Gaussian
vectors, a known reference mean, a two-column cross covariance and a strictly
positive-definite residual. It does not read P/S curves, fit a reference GP,
learn a material representation or compute a material prediction score.

## Run

With the repository's NumPy/SciPy dependencies, place this directory at
`training_demo/` in the checkout and run from the checkout root:

```sh
python training_demo/training_primitive_demo.py --repo-root . --output new_synthetic_training_receipt.json
```

Alternatively invoke the script from its actual location and pass the checkout
through `--repo-root`. The source, fixture and frozen core must match the
pre-run protocol hashes. Existing output receipts are never overwritten. Each
fresh execution makes at most **one original zero cold-start fit call**. The
script does not restart, change the seed/sample, widen parameter bounds or
relax admission criteria after seeing an outcome.

## Synthetic construction

For 39 coordinates `t = linspace(-2, 2, 39)`, the declared reference is

```text
m0(t) = 0.12 - 0.2*t
C[:,0] = 0.30*cos(0.75*t)
C[:,1] = 0.24*sin(1.25*t)
R[i,j] = 0.05*I[i,j] + 0.04*exp(-abs(t[i]-t[j])/0.6)
K0 = R + C @ C.T
```

The positive diagonal term makes the residual strictly positive definite.
Eight vectors are generated once with seed `20261101`, code mean `[0.2,-0.25]`
and code covariance `[[0.75,0.15],[0.15,1.2]]`. These are arbitrary numerical
demonstration parameters, not measured rheological quantities. The complete
generated arrays and their roles are saved in `SYNTHETIC_TRAINING_INPUT.json`.
The three six-anchor masks use the same index layouts as the paper; reusing a
mask does not turn the synthetic vectors into materials.

The demonstration supplies `CodeReference(m0, K0, C)` directly. Thus it exposes
the correction-training primitive without pretending to reconstruct the
paper's reference fitting or physical-coordinate feature preparation.

## Independent mathematical check before fitting

At four fixed parameter vectors, including zero, the checker builds the full
39-dimensional covariance and its five parameter derivatives independently of
the production numerical helper. It evaluates

```text
NLL(A) = 0.5*(|A|*log(2*pi) + logdet(K_A) + r_A.T @ inv(K_A) @ r_A)
dNLL = -dm.T @ inv(K_A) @ r_A
       + 0.5*trace((inv(K_A)-w_A@w_A.T) @ dK_A), w_A=inv(K_A)@r_A
```

For each vector it averages `NLL(all)-mean_mask NLL(anchors)` across the eight
curves and adds `0.02*sum(theta**2)`. Subtracting the independently computed
zero-parameter objective gives the centered value returned by
`ConditionalObjective`. The analytic gradient remains unchanged by this
constant. Full matrices are used by the oracle; the production objective uses
its cached two-dimensional determinant/Woodbury expression.

If the independent value/gradient or removed constant fails its fixed gate,
the script stops **before fitting**. Otherwise it calls the unchanged `fit`
once with the original bounds, penalty, optimizer options and admission gate.
A rejected endpoint or exception is saved without retry. All statuses and
actual UTC values are retained in the receipt.

## Recorded outcome and interpretation

The supplied receipt records agreement at all four fixed parameter points and
one admitted synthetic optimizer call. Admission establishes that this toy
input passes the original numerical rules; it is not a material performance
result, proof of parameter recovery or a global-optimality guarantee. With
eight arbitrary vectors and a conditional training objective, the fitted
parameters need not equal the parameters used to generate the toy sample.

This is a runnable explanation of the frozen training primitive. It is **not**
complete replay of the original leave-one-formulation-out experiment: it does
not include reference-GP fitting, raw material curves, provenance admission or
the original outer-fold scientific scores. The separate inference-verification
module checks a different computational boundary.

The protocol records real pre-fit UTC and input/source hashes. The receipt
records actual evaluations, fit settings, status, projected gradient and
unchanged core identity. No new reuse license is assigned here; the repository
owner's licensing decision remains separate.
