MOMENT DECISION READOUT: A BOUNDED SYNTHETIC COMPUTATIONAL CONTROL

Scope and present status
------------------------
These two new source files prepare one synthetic computational control. They do
not train, predict or score any material dataset, and they do not modify the prior
frozen relation_core.py or run_synthetic.py. No producer import or execution is
authorized by this README. Root must register an actual-UTC exact-source plan and
approve its SHA256 separately before running the command below.

The original SYNTHETIC_PLAN_TEMPLATE.json and DESIGN_RECEIPT.json are preserved
as design history. EXECUTION_PLAN_TEMPLATE.json is the later, root-authorized
12-case implementation scope; its TEMPLATE status is deliberately non-executable.
Do not retroactively relabel any old design receipt as execution registration.

This is a classical positive dot-product-kernel expansion adapted to an exact
Gaussian-copula CRPS decision calculation, not a new Taylor expansion, new kernel,
new Bayesian planning algorithm, or demonstrated material advantage. The question
is narrower: can decision-specific moments approximate this finite relation
posterior's information value accurately while using less computation or
workspace than competent exact implementations? A negative answer is retained.

Mathematical readout
-------------------
For relation hypotheses h, posterior branch weights w_h, target scale sigma,
and one known proxy correlation profile rho_h, define

  S_hk = sigma/sqrt(pi) * [1 - sqrt(1-rho_h*rho_k)].
  a_n = choose(2n,n)/(4^n*(2n-1)), n >= 1.
  m_n(w) = sum_h w_h*rho_h^n.
  g_D(w) = sigma/sqrt(pi) * sum_{n=1}^D a_n*m_n(w)^2.

The exact value is g(w)=w^T S w. The finite positive expansion gives a PSD
truncation and an ideal-arithmetic lower value. For r=max_h|rho_h|<1,

  0 <= g-g_D <= sigma/sqrt(pi)*r^(2D+2)*
                 min(beta_D, a_(D+1)/(1-r^2)),
  beta_D=choose(2D,D)/4^D.

The full posterior weights and every first-action outcome branch remain stored.
This readout does not compress the posterior for future Bayesian updates, does
not force the material distribution to be low rank, and does not certify a
density approximation. Heterogeneous proxies require proxy-specific moments.

For two repair outcomes o, the adaptive value is E_o max_b g_b(w_a,o). It must not
be replaced by max_b E_o g_b(w_a,o). Lower and upper branch scores yield bounds
on this restricted two-stage value. Selecting both first and second actions by
the lower scores has model regret at most max_b epsilon_b in ideal arithmetic.
That claim concerns the declared posterior and action set only; it does not cover
an entire repeated agent, unmodeled actions, material calibration, or inference
from predicted labels. A real loop still needs a paid, genuine target observation
to update an uncertain relation from new co-measurements.

Independent mathematical receipts
--------------------------------
EXECUTION_PLAN_TEMPLATE.json pins two existing independent receipts: the centered
CRPS/CDF identity audit and the Fraction/isqrt moment proof audit. Their quadrature,
exact coefficient, branch-policy and rational-interval proofs are reused, not
executed again by this runner. The runner only checks the new float implementation
and its bounded cost panel. A short D1 implementation smoke retains the known
ranking reversal; its exact order proof is read from the pinned prior receipt.

The ideal-arithmetic tail is not an IEEE directed interval. Float readouts and
score comparisons have a separately declared absolute tolerance 2e-12 and
relative tolerance 2e-11. Arithmetic error, posterior error, quadrature error and
material model error are not silently included in the truncation certificate.
The two prior exact-audit files must retain their actual bytes and hashes.

Single fixed test panel
----------------------
H in {8,64,128}; max|rho| in {.25,.6,.9,.99}; eight repair probes; sixteen known
proxy states; two outcomes per repair; CRPS epsilon 1e-6; degree cap 64. There are
twelve main cases, no random search, fitted parameter, material response, or
outcome-dependent fixture choice. Probe likelihoods have exact rational finite
definitions; posterior branches are computed from them and converted to float.

The relation grid z is equally spaced on [-1,1], prior uniform. Repair a uses
threshold 2(a+1)/9-1 and strength (a+1)/9 in a two-outcome likelihood. Each known
proxy state c on a sixteen-point [-1,1] grid has

  raw_h = tanh(1.1*z_h + .2*c*z_h^2),
  rho_h = rmax*raw_h/max_h|raw_h|, sigma=1+.25*c^2.

This is a declared synthetic statistical family, not an observed physical law.
Its monotone relation grid avoids duplicate hypotheses; nonlinearity means a
small finite global polynomial code is not assumed. It does not guarantee a
strict adaptive gain. The output reports actual adaptive-minus-best-fixed value
gaps and counts only gaps exceeding 2e-12. Proxy-index changes can be ties and
are never reported as scientific improvement.

Comparators and accounting
-------------------------
1. exact_streamed: build one HxH kernel per proxy, multiply all posterior branches
   by it, then release it. No bank of posterior outer products or eigendecomposition.
2. exact_cached: build a bank of all sixteen HxH kernels and reuse their products.
3. moment_streamed: compute the predeclared certified degree and moment readout
   per proxy; fallback to the same exact calculation if D>=H or the requested
   tail is not attained by degree64. r=0 is an exact zero relation. Near-one
   profiles cannot relax epsilon just to claim a speed advantage.

All methods receive identical posterior branches and proxy profiles. Common
input/posterior setup time is separately shown and added to each total. Per-method
timing includes validation, kernel or feature construction, truncation-degree
selection, moments and exact fallbacks. Each method has one ordered timing with
BLAS threads fixed to one; this is descriptive and does not support a general
speedup claim. No timing repetition, outcome-based tuning or old benchmark rerun
is part of this protocol.

Memory reports retain the full common H weights/profile arrays. Hand-counted
numeric workspace is explicitly named as an estimate of principal arrays; it is
not process RSS or an exact peak accounting of every temporary. Per-call managed
tracemalloc peak is reported alongside it, including algorithm setup. Python,
allocator and native-BLAS resident memory can differ. The strong streamed exact
baseline already avoids O(B*H^2) kernel-bank memory, so a claimed moment memory
advantage must be assessed against it as well as against the cached variant.

Necessary limited controls and numerical support
------------------------------------------------
- D1 can rank proxies incorrectly despite being PSD and a lower value. Prior
  exact-rational proof is pinned; the implementation reproduces its D1 values.
- Symmetric rho=+/-1e-8 has positive O(rho^4) shape value despite zero mean
  correlation. Positive D2 moments are compared with a stable O(1) analytic H2
  formula; the cancellation-prone full matrix number is shown, not treated as a
  mathematical oracle. The stable H2 formula is itself a required strong baseline.
- Two deliberate final-gain/posterior-feature underflow cases must raise an
  explicit numerical-range refusal. This limited conservative refusal does not
  establish universal float robustness, particularly for cancellation in a
  general full kernel product. The moderate main fixtures are the supported panel.
- Same-rho profiles with varying scale collapse to a simple scale ordering;
  the control acknowledges that no pair-search mechanism is needed there.

The matrix kernel may have negative off-diagonal entries. Entrywise clipping,
post-hoc jitter, changed epsilon and clipping large errors are prohibited. All
source and plan hashes are checked before execution and again at its end.

File/API boundary and approved execution
----------------------------------------
moment_core.py is source-only reusable numerical logic:
  coefficients(D), analytic_tail_bound(r,sigma,D), degree_for(...),
  exact_kernel(rho,sigma), exact_streamed(W,RHO,SIGMA),
  exact_cached(W,RHO,SIGMA), moment_streamed(W,RHO,SIGMA,epsilon,cap),
  adaptive_value(scores,branch_probabilities), h2_symmetric_stable(r,sigma).
Shapes: W=[branch,H], RHO=[proxy,H], SIGMA=[proxy]; scores=[branch,proxy].
Each repair has exactly two ordered branch rows. Domain violations raise
ValueError; selected nonrepresentable positive values raise
NumericalRangeUnsupported. No material IO, optimizer, agent, network or LLM calls.

run_moment_control.py owns fixed input creation, one-shot comparison, smoke
controls, exact plan/approval checks, and exclusive execution receipts. Example
after ROOT approval only (new, absent output directory required):

  D:\Anaconda\python.exe run_moment_control.py --plan <registered_plan.json>
      --approval <root_approval.json> --output-dir <new_execution_directory>

The plan must have status REGISTERED, actual UTC, exact source hashes, exact
configuration(), no-material flag, and two unchanged independent audit receipts.
Approval must bind its exact SHA256 with scope MOMENT_READOUT_SYNTHETIC_CONTROL.
EXECUTION_START.json precedes computation; RESULTS.json or FAILED_STOP.json is
exclusive. Preserve a failure and ask root to decide any actual correction;
do not retune or re-register secretly after observing results.

Decision boundary
-----------------
Passing algebra and this computational control would justify only a bounded
decision-readout feasibility result. If fallback dominates, memory is not improved
against streamed exact, or gains are already captured by simple analytic ranking,
retain that negative result. Independent material relations, source identity,
state/missingness assumptions, genuine co-measurements, baseline-level predictive
accuracy and an actual closed-loop task remain separate, unproven evidence gates.
