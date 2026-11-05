PROSPECTIVE FINITE-GRID CALIBRATION AND CONFIRMATION HELPER
========================================================

Status: isolated new source; no real measurements, calibration or confirmation
have been executed here. This is confirmation preparation, not a new algorithm
or completed material validation. The old predictor and historical results stay
unchanged. The null real templates intentionally cannot pass registration.

1. What is reused

The existing Oct31 confirmation preparation supplies six-anchor conditional
Gaussian forecasts and their full target covariance. Its coordinate-only plan
and frozen model/runtime bindings remain the source of those forecasts.
CONTRACT_REUSE.json in the parent directory records the reviewed files/hashes.
This helper does not run or rewrite that predictor, retrain a network, choose a
new correction, rescale its density, or read historical measurement tables.

The current shared model SHA256 is prefilled only because the original public
author-handoff already freezes it. Baseline and runtime-manifest hashes must be
copied from the actual frozen artifact receipts; they have not been invented.
The future six-anchor policy is NOT selected by this helper. The author fills
anchor_rule/anchor_policy_reference, freezes six native indices in every record,
and checks compatibility with the existing runtime before seeing targets. The
development one-sided indices [0,3,6,9,12,15], first-six-native external policy,
and Oct31 physical-fraction proposal are distinct policies, not synonyms.

2. Four source modules

contract.py: strict JSON, input/code SHA bindings, declared domain, native-grid
and source/physical-unit role checks. No target measurements in cohort metadata.
calibration.py: pure unit-max scores and exact split-conformal order statistic;
distinct prediction-set object; explicit rejection of band CRPS/log density.
evaluation.py: original Gaussian CRPS/NLL/joint log density and separate set
coverage/width/interval-score diagnostics, with native log-rate quadrature.
cli.py: exclusive local freeze/calibrate/confirm receipts. It calls no predictor.

3. Scientific unit and role ledger

Choose exactly one goal before labels:
* new_formulation: every curve/mask of a canonical formulation belongs to one
  unit; consumed formulations are blocked from the reserved cohorts.
* new_preparation_known_formula: a previously studied formulation may appear,
  but a preparation batch/source record/physical alias cannot cross units or
  roles. All curves and masks of the same independent preparation stay together.

Canonical formulation IDs must already reconcile true identity aliases across
source names, publications and family naming variants. identity_aliases lists
physical record/preparation aliases; they are not arbitrary broad chemical
family labels. The reviewed consumed_registry must include every development,
previously evaluated and otherwise inspected original record/preparation and its
aliases. This helper cannot discover false metadata, unknown aliases, real
independence, rights, exchangeability, true material state or actual blinding.
review_complete=true and a reviewer name are declarations, never certification.

Every query/mask has a unique record_id, but repeats of an original curve retain
the same source_record_key. These repeats can increase the finite-grid maximum;
they cannot increase calibration sample count. All planned records, all methods
and all native targets must be present; no successful-subset fallback occurs.

4. Mathematical meaning

For method j, curve/mask r in calibration physical unit g, and hidden coordinate t:
  R_g,j = max_{r in g,t} |y_rt - mu_j,rt| / sigma_j,rt.
For exact rational alpha, n independent calibration units, use
  k = ceil((n+1)*(1-alpha)); q_j = kth sorted R_g,j if k<=n, else +infinity.
No interpolated quantile or clamped index is used. At alpha='0.1', fewer than
9 calibration units yield the vacuous unbounded set. Nine is only the algebraic
minimum for a potentially finite threshold, not a justified material sample size
or evidence of stable width/conditional calibration. Ties give conservative
coverage; the guarantee requires exchangeability of complete units under the
same preregistered target/mask policy. It is marginal over units, not a separate
guarantee for every formulation, subgroup, rate, new domain, or continuous curve.

The log-viscosity prediction set is mu_j,rt +/- q_j*sigma_j,rt simultaneously on
the declared finite target grids of the new unit. It does NOT define a probability
density. Gaussian CRPS and log density continue to score the unchanged original
frozen Gaussian. Each fixed comparator gets its own q using exactly the same
calibration units, masks, alpha and grids. Do not compare calibrated candidate
coverage against only raw comparator intervals as an information-matched claim,
or select the candidate/comparator after inspecting q/width/calibration labels.

Unbounded q is strict JSON q:null plus unbounded:true; set endpoints are null,
never nonstandard JSON Infinity/NaN. Such sets have vacuous finite-grid coverage,
unbounded width, and no finite interval score. A finite-q arithmetic overflow
stops the phase instead of clipping or silently widening the band.

5. Primary score and scope

All native rates must be positive and already strictly increasing; this helper
does not sort, deduplicate, clip, interpolate, synthesize labels, or use an anchor
as a hidden target. Targets are every declared native point strictly above the
highest of six anchors, with at least two targets. With x=ln(gamma), trapezoidal
weights are endpoint half-gaps and interior half-neighbor-gaps, normalized by
x_last_hidden-x_first_hidden. The integral domain is exactly [first hidden native,
last hidden native]. The omitted highest-anchor-to-first-hidden gap is explicitly
reported, not supplied by extrapolating a missing endpoint. One hidden point is
not a valid integrated primary endpoint and is rejected.

Within each original curve, registered masks are averaged equally. Original
curves are averaged equally within independent unit, then units equally. This
prevents more masks/replicate curves from becoming more independent material
samples. Paired primary CRPS differences use the exact same complete cohort;
negative candidate-minus-comparator favors the preregistered primary method.
The helper reports no invented confidence interval, significance or superiority.
Raw point coverage, raw whole-unit finite-grid coverage, conformal point coverage
and conformal whole-unit coverage are separate quantities. The same-alpha interval
score is a diagnostic of the set, not proof the original Gaussian is calibrated
or that this width is optimal. No between-grid coverage is claimed.

6. Local future use (do not execute on old consumed data)

Copy the null templates and fill actual author-reviewed source, state, temperature,
unit counts, sample-size rationale and practical viscosity tolerance. Resolve
the existing author-handoff source-admission requirements first. Source names
being different do not establish independence. Obtain a fresh eligible cohort;
do not use old25P/46 or already inspected targets as reserved confirmation.

From this directory, with Python and NumPy installed:

  python cli.py check-plan --config DESIGN.json --cohort COHORT.json
  python cli.py register --config DESIGN.json --cohort COHORT.json --run NEW_RUN
  python cli.py freeze-calibration --run NEW_RUN --predictions CAL_PRED.json
  python cli.py calibrate --run NEW_RUN --truth CAL_TRUTH.json
  python cli.py freeze-confirmation --run NEW_RUN --predictions CONF_PRED.json
  python cli.py evaluate --run NEW_RUN --truth CONF_TRUTH.json

After registration, put the printed PROTOCOL.json byte SHA256 into each prediction
and truth envelope. FORMAT_EXAMPLES.json specifies fields but contains no data.
Obtain actual model forecasts with the existing frozen runtime; map its three
law names and coordinate vectors into the prediction envelope. Copy actual
model/runtime/source/anchor-input hashes from that runtime's receipt. This is an
export/transport conversion, not refitting. Unknown prediction fields (including
truth) are rejected. The helper verifies declared byte-identity consistency, not
whether a model truly produced an arbitrary caller-supplied array.

Calibration forecasts must be frozen before calibration truth is loaded here.
The thresholds are then frozen before confirmation forecasts/sets are frozen,
and confirmation truth is revealed only afterward. All outputs are exclusive;
code changes invalidate a registered run. Retain any FAILURE receipt. A failure
in a label-consuming phase creates a stop marker, and cannot be silently retried
as fresh evidence. Interrupted ACTIVE_PHASE_LOCK files require process inspection
before human resolution; never launch a duplicate live phase. Do not edit a
frozen receipt to bypass a stop. Hashes/phase order aid an audit; they cannot prove
that the author or another program had not already viewed the measurements.

7. Qualification and present limits

No material run is supplied. Synthetic mathematical smoke checks, if performed
under a separately registered audit, must label mode='synthetic' and every record
and unit ID with SYNTHETIC_. They test formulas/workflow only and never establish
material sample size, new-batch generalization, experiment saving, uncertainty
calibration, or paper readiness. The unfinished real plan remains waiting for
actual independent material source/enrollment and author-approved tolerances.

The methodology is standard split conformal applied to frozen finite-grid curve
predictions; this helper is confirmation infrastructure, not an originality claim.
