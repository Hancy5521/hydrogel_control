Bounded representation robustness study (new retrospective development)

This code does not replace or tune the frozen hydrogel candidate.  It adds:
  - eight fixed random17x2 bases, seeds20261105 through20261112;
  - one prior-only top-two KL basis;
  - a fixed45degree rotation of the original curvature span.
Each uses the original five Gaussian-code coefficients, optimizer, bounds,
regularizer, three masks and original per-held-P GP reference receipts.

Equal coefficient count is not rotationally equal capacity.  The bounds on
the mean/Cholesky coordinates and the coefficient penalty depend on basis
orientation.  The rotated span is a diagnostic of that dependence, not a
novel representation or new physical information.

The existing25P formulations have already been used in development.  These
new fits are not independent confirmation and add no measured materials.
The closed TRAIN file contains only six lawful values per S formulation;
every other S response must remain NaN.  No hidden S cache is accepted.

Modules:
  assets.py: explicit input allowlist, hashes, role admission, originalscores.
  representations.py: prior-only basis mathematics; no outcome input.
  experiment.py: syntheticqualification/register/fit/globalfreeze/score CLI.
  frozen/: four byte-exact original mathematical source modules.

Execution uses Python3.13/NumPy2.1.3/SciPy1.15.3 plus threadpoolctl as observed
in the existing scientific environment.  It can run elsewhere with those
dependencies; the actual versions are saved at qualification/registration/fit.

Choose RUN_ROOT as a new stage folder with this code retained alongside it.
Choose ASSET_ROOT as the lawful original missing_controls stage.  Its exact
required relative files and known hashes are explicit in assets.py.  The
local original stage is the one cited in FEASIBILITY.json.

  python code/experiment.py qualify --run-root RUN_ROOT
  python code/experiment.py register --run-root RUN_ROOT --asset-root ASSET_ROOT
  python code/experiment.py fit --run-root RUN_ROOT
  python code/experiment.py score --run-root RUN_ROOT

Review the qualification and actual-UTC protocol before fitting.  All750
new forecast laws must be frozen, with hashes, before any new target score.
Any optimizer/admission/hash/role failure stops the whole new analysis.
There is no retry, fallback, best-seed choice, reference GP refit or model
deployment step.  Existing candidate/random/GP/conditionalGaussian scores
are imported unchanged.  Follow independent audit before using conclusions.

Protected local resources:
  experiment/inputs/data/TRAIN.npz holds restricted original measurements.
  experiment/forecasts holds measured contexts and fitted/reference laws.
The producer does not publish these files or assign new data licenses.
Public archival should include reviewed source and derived summaries only,
unless the original asset rights/access obligations have been resolved.

No material fit, score or qualification was executed by the implementation
agent.  The root research run owns authorization, preregistration and execution.
