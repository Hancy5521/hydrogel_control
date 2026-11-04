Saved uncertainty diagnostic for MLST author review

Purpose
This program organizes the original saved external curve-level score scalars
into accuracy, calibration diagnostics and sharpness summaries. It does not
fit a model, generate a prediction, read measured response arrays or score
targets again. The four sources remain separate and all original curves and
methods remain in the input. The analysis was registered in
../SAVED_UQ_ANALYSIS_PLAN.json before execution.

Replay from a hydrogel_control checkout
Python 3.10 or later; standard library only.

python PATH_TO/replay_saved_uq.py --repo-root . --output-dir NEW_OUTPUT_DIRECTORY

The exact original SCORES.json SHA256 is required. Outputs must go to a new
directory. The source must retain 276 curve-method rows, 46 native curves,
23 analysis groups, four sources and 781 higher-rate target positions.
The script reports the three probabilistic methods; the source also retains
all three point controls. No covariance or probabilistic score is invented
for the point controls.

Weighting and interpretation
For every source-method metric, native curves are averaged within their
original analysis groups and groups are then weighted equally. Log-RMSE is
the square root of the group-macro MSE, not mean curve RMSE. The sources are
never pooled. Native-curve zero-coverage and all-target-band counts are
unweighted descriptions of native records and are intentionally distinct
from group-weighted coverage. Related runs and reported means are not
certified independent preparations. Counting a curve whose targets all lie
inside nominal marginal 90 percent bands does not supply a simultaneous
90 percent confidence or predictive-coverage guarantee.

What the new organization shows
The candidate's original Liverpool CRPS advantage coexists with four of
15 native curves having zero covered target positions. The conditional
Gaussian has zero such curves. Candidate average interval width is
0.730671 in log viscosity, versus 1.488886 for the conditional Gaussian;
their interval scores are 3.377353 and 1.640179, respectively. Thus narrower
intervals and better CRPS do not establish better interval reliability.
The candidate has zero-coverage counts of 0/10, 2/16 and 4/5 in the HA/CMHA,
granular and dynamic-alginate sources. These are descriptive reorganizations
of already evaluated curves, not newly independent findings or calibration.

Nine synthetic checks
Run from this directory:
python -m unittest -v test_replay_saved_uq.py

The hand-checkable weighting example has two records in one group with
losses 1 and 3, and one record in another group with loss 8. Its correct
group-macro loss is 5, whereas a naive mean of the three records is 4.
Other fixtures check duplicate rows, missing methods, disagreeing identities,
unregistered sources, nonfinite values, array-valued fields, inconsistent
coverage indicators and an undersized cohort. Fixtures contain invented
score scalars only and are not material data or predictive validation.

No adoption rule
This result does not define or select an applicability-domain threshold,
fallback, risk gate, scale correction, model winner or new primary endpoint.
Parameter uncertainty, independent-batch generalization and calibration on
an unconsumed compatible cohort remain outside the evidence.
