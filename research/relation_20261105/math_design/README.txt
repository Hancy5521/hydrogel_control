FINITE RELATION POSTERIOR / CENTERED-CRPS SYNTHETIC CONTROL
========================================================

Status: two-module static prototype; no execution is authorized by this README.
Do not run it until root registers exact source/plan SHA and separately approves.
No materials are read/fitted/scored. The old failed fixed-B scheduler remains STOP.
Detailed derivations/limitations are in MATH_AND_MODEL_ZH.txt; independent prior
CDF/energy/branch proofs are reused by SHA, not reexecuted here.

Purpose and contribution gate
-----------------------------
This tests a classical low-capacity relation-posterior computation primitive.
It is not a new Bayesian algorithm, a complete self-improving agent, learned
catalytic-rate mechanism, material result or top-conference-readiness claim.
There is no complex screening implementation: it has not been shown to beat a
strong fully cached/blocked-vectorized baseline, especially at small H.

The supported subproblem chooses an old-unit pure relation repair, followed
by ONE new-unit pure proxy. The second proxy choice may depend on repair outcome.
The repair is not allowed to observe the current target. Proxy marginal must be
independent of the relation hypothesis; all hypotheses share the target marginal.
Conditional independence of units givenTheta is a declared model assumption,
not an empirical certification. Mixed panels and unsupported scopes are rejected.
The general full-joint planner is an external fallback, not implemented here.

In particular, pure E does not itself learnTheta. An actual closed loop needs
paid target V later to create a new genuine co-measurement and updateTheta.
This two-step primitive does not optimize that third action or count inferred
targets as verified discoveries. Same-action direct V/STOP alternatives need a
full scientific planner; the primitive itself never claims global optimality
over omitted actions. Proxy-alone values are exposed to detect useless repairs.

Files
-----
relation_core.py:
  Exact Fraction priors, finite outcome posterior branches, posterior covarianceF;
  stable centered Gaussian-copula CRPS kernel;
  precommitted fixed-pair values versus full retained-branch adaptive values;
  explicit loop correctness reference and strong cached/blocked vectorized controls.
run_synthetic.py:
  Fixed synthetic inputs, one source/plan-approved run, all controls/results/costs.

Key formulas
------------
w_h=P(Theta=h|D). Repair likelihood L[o,h] columns sum exactly1.
Only EXACT zero-probability branches are removed; every positive branch retains
its Fraction probability and posterior. Fa=E[(w_a-w)(w_a-w)^T], Fa*1=0.
No posterior mean-rho substitution is made.

For (U,Y)|h bivariate standardNormal with correlationrho_h, and target scaleσ,
the centered CRPS kernel is
  S_hk=σ*rho_h*rho_k / (sqrt(pi)*(1+sqrt(1-rho_h*rho_k))).
This stable form avoids cancellation for weak correlations. Correlations must
lie strictly inside(-1,1); no jitter/clipping treats degenerate components.
S may have negative off-diagonal entries; it is not entrywise clipped.
Material multi-target loss weights can multiply each S; this prototype's cases
use a single target and alternative single proxies unless otherwise labelled.

Pure relation gain alone=0. Proxy gain=w^T S_b w. Fixed(a,b) gain is
  w^T S_b w + trace(Fa*S_b).
Adaptive two-step gain after first actiona is
  sum_o P(o|a) max_b w_a(o)^T S_b w_a(o),
with b constrained by the SAME declared hard budget. The program retains
posterior branches; it does not replace E[max] by max[E].

Exact finite-state probabilities are different from exact real arithmetic:
Gaussian square roots and score arithmetic are float64. Positive exact entries
which would become zero/Inf in the numeric path cause NumericResolutionError;
they are not erased and renormalized. Matrix PSD tests allow only a recorded
scaled backward-error tolerance; no matrix is projected or changed. Ordinary
rounding remains; no universal interval-arithmetic certificate is claimed.

Controls and computation comparison
----------------------------------
Six principle cases include:
  symmetric ±rho (proxy CRPS can improve via shape even when mean rho=0);
  knownTheta; independent wrong old-calibration identity; rho=0;
  non-collinear single-proxy selector; sameF/different adaptive posterior branches.
The last is an EXACT binary Brier fixture, not falsely a Gaussian CRPS result:
  sameFa, fixed gains5/64, adaptive gains9/64 versus7/64, risks7/64 versus9/64.

Three declared benchmark families, each at16x32 and64x128 repair/proxy counts:
1. homogeneous_scaling: same rho profile, different known target scales.
   An analytic feasible-scale ranking is checked; no pair-search need is claimed.
2. known_state_noncollinear: rho_h(c)=.8*tanh(beta_h0+beta_h1*c), with frozen
   beta=(-1,-.5),(-1,.5),(1,-.5),(1,.5) and known synthetic state c∈[-2,2].
   Repair probes are finite noisy reports of one component of this hypothetical
   relation state, with known rational likelihoods. They are abstract synthetic
   observations, NOT equivalent to measured continuous material co-measurements.
   No rho table is fitted to target results; real physical applicability is unknown.
3. dense_uninformative: all repair probes uninformative and proxy profiles equal.
   No repair complementarity exists; analytic ranking is again adequate.

Every algorithm uses the same prior/likelihoods/kernels/costs/budget. Costs are
positive integer synthetic ticks, not measured lab/DFT savings. The explicit
tree loop is a correctness reference ONLY. Winning against that intentionally
uncached implementation is not an algorithmic efficiency contribution.
Fully cached/vectorized and blocked(16-row) vectorized implementations are the
strong computing controls. Both include posterior/F/S preprocessing in timings.
Separable closed-form ranking is retained whenever applicable. No approximate
top-k filter or unsupported safety guarantee is introduced.

Timing is one ordered wallclock measurement per case/algorithm, with BLAS1thread.
tracemalloc reports managed allocations, NOT whole-process peakRSS. NumPy array
workspace counts and kernel-build counts are separately supplied. These are
bounded implementation observations, not robust universal speed rankings;
repeated timing or repeated same-world trajectories are not new material units.

Execution
---------
PLAN_TEMPLATE.json has statusTEMPLATE and no registrationUTC. Root copies it to
a NEW plan, fills statusREGISTERED/actualUTC, pins these two source hashes and
the prior independent proof receipt. A separate approval must contain:
  approved:true
  scope:"RELATION_POSTERIOR_SYNTHETIC_CONTROL"
  plan_sha256:<actual registered plan SHA>
Then, once:
  python run_synthetic.py --plan NEW_REGISTERED_PLAN.json \
    --approval ROOT_APPROVAL.json --output-dir NEW_OUTPUT_DIRECTORY
No output directory/file is overwritten. EXECUTION_START is written first.
Failures are retained inFAILED_STOP; configuration/source changes require a
new root-reviewed plan, not silent retries. Do not repeat the prior CDF integrals.

The runtime uses existing Python/NumPy/threadpoolctl only; no package installed.
All dataset/file IO beyond source/plan/approval/prior-proof receipt/output is
absent. It never calls the old pilot or reaches a material oracle.

Main API for independent reviewer
--------------------------------
prior_weights(values)->tuple[Fraction]
posterior_branches(prior,likelihood_outcome_by_h)->tuple[Branch]
relation_covariance(prior,branches)->exact Fraction matrix
centered_crps_kernel(rhos,scale=1.,weight=1.)->S
prepare(prior,probes,proxies,budget,scope)->cached exact posterior/numeric kernels
fixed_pair_values(prepared)->values/complementarity/proxy-alone values
adaptive_loop_reference(prior,probes,proxies,budget,scope)->branch adaptive values
adaptive_cached(prepared,block_rows=None)->same adaptive values/choices/workspace

Stopping rule
-------------
If the result is simply that classical caching/vectorization evaluates the
restricted problem correctly, state that result. Do not invent a separate new
algorithm or top-conference contribution. A future claim requires an additional
task/computation benefit against these strong controls, independently justified
real-material model assumptions, paid verification and genuinely new data.
