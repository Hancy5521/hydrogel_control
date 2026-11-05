EIGEN-INSPIRED PREDICTIVE ALLOCATION: BOUNDED OFFLINE FEASIBILITY PILOT
=================================================================

Current status
--------------
This source is a preregistration candidate, not an executed material experiment,
independently qualified implementation, new algorithm or top-conference result.
Only static reading and AST syntax checks precede the source receipt. Root must
pin the source and approve independent synthetic checks before real-source runs.
No original MLST source, model, result, protected S target or manuscript is used.

This pilot asks whether a small predictive cross-gain allocation heuristic is
worth further development against a same-information joint lookahead policy.
It does not yet implement a learned, independently verified catalytic-rate
matrix or a generative crystal/chemistry model. The "agent" is a transparent
sequential program choosing from a fixed, preexisting public candidate pool.
No LLM calls, external model APIs, wet experiments or new DFT calculations occur.

Source evidence and identity
----------------------------
Two pinned public gzip/JSON tables are used, separately:
* UCSB thermoelectrics: experimental literature aggregation. Eligible rows have
  finite rho, S, kappa, known temperature and source. Native sigma, PF and zT are
  excluded entirely from predictive features and oracle returns; these derived
  columns could algebraically reveal a hidden target and are not independent
  validation evidence.
* JARVIS: preexisting computed DFT fields e_form, gap opt and source bulk modulus.
  Negative bulk values remain unmodified. No log(abs(value)), deletion of hard
  cases or inference that these are wet measurements is permitted. The precise
  modulus aggregation is qualified in the source addendum; not asserted VRH.

Original source hashes, versions and licenses are in ../data/SOURCE_GATE.json,
DATA_MANIFEST.json and SOURCE_GATE_PILOT_ADDENDUM.json. Eligible metadata contains
no response values. Native missing labels remain unknown: the pilot selects the
complete-case eligible subset and imposes masks. It does not validate missing
native values or remove the complete-case measurement/computation selection bias.

Role/row selection is metadata-only and agrees with DATASET_ROLE_PLAN.json.
Retain identity components formed on the FULL source before eligibility filtering:
UCSB joins normalized composition and source citation; JARVIS joins available IDs
and normalized composition. These are conservative metadata groups, not certified
independent preparation batches. Do not reconstruct identities on a small subset.

For each dataset, sort groups by SHA256('hypercycle_pilot_v1|' + component_id),
with component_id tie breaker. The first40 groups become8 warm,8 repair,16
frontier and8 test groups. Within each group choose up to2 rows by the pinned
hash of dataset and original raw-row index. All remaining groups stay unused.
Selections are not changed after seeing performance. The two datasets are
separate portability tasks with separately fitted warm statistics, not zero-shot
transfer of one material model or a same-domain validation cohort.

Allowed predictive metadata features are118 element fractions and additionally
UCSB known T/1000. Source, role, IDs and outputs are not predictive descriptors.
Each metadata feature is centered/scaled using the common known, unlabeled X;
constant columns use unit scale. Divide the resulting vector by sqrt(number of
nonconstant columns), or1 if all are constant. Thus fixed RBF lengthscale1 uses
RMS-feature-distance units, without a label-driven bandwidth search. Crystal
structure is not encoded: this is deliberately a small mechanism pilot, not
state-of-the-art material property regression.

Joint reference and observability
---------------------------------
Obtain all three original warm responses, and the first two responses of each
repair row, through the same bundle oracle. These common initial observations
have their own recorded bundle/cell count, equal for every policy. Warm means
and sample standard deviations define response units; a zero warm SD explicitly
uses1 and is recorded. No test response enters this scaling.

The fixed three-output intrinsic-coregionalization reference is
  B = 0.8 * sample_cov(standardized warm Y) + 0.2 * I_3
  K = RBF(X,X; lengthscale=1) kron B + 0.05 * I.
Mean is zero in warm-standardized units. The0.05 diagonal is a declared
independent response residual in the joint law, not numerical jitter. B is
fitted once from warm values in each campaign and then remains fixed. The
repeated identical warm-statistic calculations across matched policies do not
constitute independent learned models or learning new task relations.

Observing source values conditions this joint law exactly. The whole posterior
is positive SEMI-definite: observed coordinates have mean=observed value and
zero covariance rows/columns by analytical identity. Fresh queried blocks and
the never-observed test block must be SPD. Negative unknown variances, nonfinite
arithmetic or failed factorization stop the run; no clipping/jitter/retuning
rescues a result. Rank-one or rank-two conditioning retains every revealed
value, even when a prediction was poor. There is no acceptance-by-agreement.

The trusted IO layer parses a public source container which itself includes
all source values. It selects only the declared three responses for non-test
rows into the campaign oracle; test responses in that view are NaN and queries
on test rows are rejected. The Gaussian learner/policy receives only explicitly
revealed cells. This is a role-level information-flow contract, not process
isolation or cryptographic proof of blinding against malicious Python callers.
The original source is parsed once/dataset. Each simulated campaign forks
independent source-array/mask/receipt/cost state; no observation or budget state
is shared between policies. The evaluator constructs true test vectors only
after the complete forecast freeze and a separate root approval.

Actions, costs and scalar goal
------------------------------
Every eligible repair/frontier row has two fixed action bundles:
  A={property0,property1}, B={property2}.
Each bundle costs1, regardless of1 or2 cells. This is an explicit offline cost
scenario, NOT measured instrument time, DFT walltime or evidence every original
experiment produced exactly this bundle. All policies share it. Initial warm
costs two bundles/row; initial repair costs one/row; the additional budget is24.
Known/partly known bundles, repeats, warm campaign actions and all test actions
are rejected. An action is R if its row already has any observed response, E
otherwise. After expansion, the remaining unknown bundle on that row is R.

The primary endpoint is held-out standardized Gaussian CRPS, equal weighting
of the8 test metadata groups and rows/properties within each group. Additional
RMSE and verified finite-pool property-extreme retrieval are reported separately.
The retrieval direction was chosen BEFORE fitting: low kappa for UCSB (sign-1),
high source bulk for JARVIS (sign+1). This is not zT optimization or a claim of
newly synthesized materials. No derived zT score or extra response is used.

For any available action, compute two model-based quantities:
1. IVR = exact sum test-coordinate variance reduction / initial test trace.
2. Scalar finite-pool KG = expected increase in the maximum posterior signed
   property2 mean over ALL fixed frontier rows / initial maximum frontier SD.
   Approximate this expectation by5-node Gauss-Hermite per queried dimension
   (5 or25 tensor-product fantasies). This is not exact integration, HV-KG,
   hypervolume optimization, or a reproduction of published full HV-KG.
The comparator joint acquisition is0.5*IVR+0.5*KG, with frozen normalizers.
Test X may enter variance reduction; test responses do not enter the policy.

Eight policies and classical nature
----------------------------------
All policies use identical initial observations, model, query pool and budgets.
  random: fixed-seed uniform choice among available bundles.
  repair_only_IVR: maximumIVR withinR, fallback all actions ifR is empty.
  expansion_only_KG: maximumKG withinE, fallback all actions ifE is empty.
  fixed_cycle: R/E alternate, within-role maximum joint acquisition.
  fixed_random_modules: EXACTLY the same best-joint R/E proposals and uniform
    stream as hypercycle, but fixed[.5,.5] module weights and no C update.
  joint_lookahead_all: maximum joint acquisition over every available action.
  hypercycle_predictive: propose best joint R/E actions, construct
      C = [[IVR_R, KG_R], [IVR_E, KG_E]], f=C*w,
      w_next=0.9*softmax(log(w)+0.5*f)+0.1/2,
    and use a pre-fixed uniform variate to choose R/E with w_next. Startw=[.5,.5].
  hypercycle_no_cross: identical code and uniform variates, but zero C off-diagonal
    entries before computing fitness.
A pre-fit amendment added fixed_random_modules to distinguish an adaptive
catalytic-weight effect from ordinary randomized module exploration. It changes
no source, role, query budget, response or model hyperparameter.
The update is a classical replicator/exponential-weight heuristic. C is
predictive, not an independently measured causal transfer or verified rate.
The noise/residual law, candidate proposals and true oracle feedback are fixed
components; giving the scheduler an Eigen name does not establish originality.

Random, fixed_random_modules and the two hypercycle policies use four episode seeds2026110501..04.
Four deterministic policies each execute once/dataset; their other logical
seed names alias the SAME frozen laws. Total40 actual campaigns,64 logical
episodes,160 actual checkpoint forecasts. These are not64 independent material
studies; the4 trajectories reuse the same groups and source labels. Do not use
aliases or trajectory seeds to manufacture a material sample size.

Outputs, strict freezing and separate evaluation
-----------------------------------------------
Five source modules, each with one responsibility:
  code/benchmark_io.py: pinned IO, metadata roles, features, restricted oracle.
  code/joint_gaussian.py: fixed warm reference, conditioning, IVR/scalarKG.
  code/policies.py: matched decisions and predictive resource-weight update.
  code/run_campaign.py: prepare metadata plan; approved campaigns; freeze laws.
  code/evaluate_campaign.py: post-freeze source truth/scoring, no fitting.
Each campaign stores INITIAL_REFERENCE.json, ACTIONS.json,
CAMPAIGN_COMPLETE.json and forecast_00/08/16/24.npz. The aggregate freeze binds
all281 files (7*40+EXECUTION_START), exact actual identities and all64 explicit
logical aliases. No missing, duplicate, extra or misbound mapping can pass the
evaluator preflight. The full joint Gaussian laws allow independent replay.

Metadata preparation (does not fit or score):
  python code/run_campaign.py prepare --data-dir ORIGINAL_DATA_DIRECTORY \
    --role-plan DATASET_ROLE_PLAN.json --output NEW_PROTOCOL.json
Root separately verifies exact inputs/source/independent synthetic audit and
creates APPROVAL.json with approved=true,
scope="BOUNDED_REAL_SOURCE_OFFLINE_PILOT", plan_sha256=<actual protocol SHA>.
Only then:
  python code/run_campaign.py run --plan NEW_PROTOCOL.json \
    --approval APPROVAL.json --output NEW_CAMPAIGN_DIRECTORY
The command first validates complete fixed metadata, source and input pins.
Any failure retains EXECUTION_FAILED and prevents subset scoring. No target
scores are computed by run_campaign. After all40 campaigns are complete,
ALL_FORECASTS_FROZEN.json binds every law and action receipt.

Root then creates a DIFFERENT scoring approval with approved=true,
scope="SCORE_FROZEN_OFFLINE_PILOT", plan_sha256, freeze_sha256. Only then:
  python code/evaluate_campaign.py --plan NEW_PROTOCOL.json \
    --campaign-dir NEW_CAMPAIGN_DIRECTORY --approval SCORE_APPROVAL.json \
    --output-dir NEW_EVALUATION_DIRECTORY
This verifies identities/artifacts before test-truth selection, records reveal
intent, scores every frozen checkpoint once and keeps all failures. Outputs
contain all per-group results and policy-minus-joint-lookahead comparisons.
Top-decile thresholds are computed only by the evaluator from fixed frontier
group maxima; only originally revealed target values count as verified hits.
Neither unqueried predictions nor duplicate states count as discoveries.

Required runtime: Python, NumPy2.1.3, SciPy1.15.3, threadpoolctl3.5.0 in the
existing scientific environment. Campaign execution limits BLAS to one thread;
it records actual walltime and zero LLM/GPU use. No packages are installed here.

Decision boundary
-----------------
Compare the identical-proposal fixed_random_modules control as well as the
no-cross ablation; random-over-all-actions alone does not isolate catalysis.
If this classical predictive coupling does not improve the matched joint
lookahead endpoint, retain the outcome and stop the extra-mechanism claim.
Do not change groups, weights, bandwidth, seeds, cost bundles or objective
after reviewing this test set. Further work needs a separately justified and
registered increment, particularly real uncertainty about cross-property
relations and empirical, budget-accounted catalytic evidence. This small pilot
does not certify field strong-baseline parity, general material transfer,
reliable calibration, new measurements, real resource savings, or top-conference
readiness. Original MLST science stays independently archived.
