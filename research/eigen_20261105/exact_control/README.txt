EXACT LOGICAL COMPLEMENTARITY CONTROL
===================================

Status
------
This is a separate finite mathematical fixture after the material pilot. The
source may be parsed/pinned before root approval; do not run it, call its
functions or import it for numeric checks until the exact plan is registered
and approved. All results will be exact fractions. No hydrogel, UCSB, JARVIS,
model forecasts, measurements, DFT data or internet service is used here.

The purpose is a necessary logical control: can an individually uninformative
repair/calibration observation and an individually uninformative expansion
observation become useful together, and does that survive matched direct-label
actions, costs and a requirement to VERIFY the target? This is classical Bayes
inference and finite-budget lookahead, not an Eigen/new-agent algorithm claim.

Worlds and observations
----------------------
All unspecified bits are independent and uniform on{-1,+1}. Probability is
uniform over all explicitly enumerated worlds, not Monte Carlo sampling.

1. coupled,8worlds: theta,u1,u2; Y1=theta*u1, Y2=theta*u2.
   R reveals theta, Ei reveals ui, Vi reveals Yi.
2. theta_known,4worlds: theta is publicly known+1; u1,u2 vary independently.
   Same action definitions; a useless R call is still charged if a fixed policy
   chooses it. Public prior knowledge is not called a new oracle measurement.
3. fake_pair,16worlds: theta_target,u1,u2,theta_old independent.
   Yi=theta_target*ui, but R reveals theta_old, not theta_target. Ei/Vi unchanged.
   This tests wrong report/calibration identity, not noisy same-sample data.
4. no_relation,32worlds: theta,u1,u2,y1,y2 independent.
   R reveals theta, Ei reveals ui, Vi reveals independent yi.

The source uses immutable World(latents,targets,observations), all observations
listed in the fixed action orderV1,V2,R,E1,E2. A belief is a nonempty sorted
unique tuple of world IDs. Conditional beliefs remain uniform because the
observations are deterministic functions of the uniform finite world.

Loss and information
--------------------
Predict positive-class probabilities p_i=P(Yi=+1|observations). The Brier score
is(p_i-1{Yi=+1})^2. The exact Bayes risk is p_i*(1-p_i); the endpoint averages
the risks of the TWO targets. No-target information gives risk1/4. Knowing one
target exactly and leaving the other uniform gives mean risk1/8, not1/4.

In the base world, R alone and E1 alone have zero marginal-target information;
R+E1 determinesY1. Its per-Y1 risk reduction is1/4 and the two-target mean
reduction is1/8. The program reports exact expected risks and both marginal
and joint target mutual information, using exact entropy on the uniform
power-of-two images in these linear binary fixtures. This helper is deliberately
not a generic real-number entropy estimator.

Zero marginal Brier gain is not equivalent to zero JOINT information. For
example, E1+E2 can reveal the parity relation betweenY1/Y2 while neither
marginal becomes certain. The chosen endpoint is mean marginal Brier; no claim
about all multivariate probabilistic losses follows from it.

Cost settings and classical policies
-----------------------------------
One integer tick costs1/4. Two fully fixed settings:
  cheap_repair_expansion: R/E1/E2 each1tick, V1/V2 each4ticks, total4ticks(B=1).
  equal_cost: all five actions each4ticks(cost1), total8ticks(B=2).
An action may be used at most once. Unused budget is allowed. No rule silently
makes a direct verification free, and all oracle calls are explicitly charged.

Three policies use the SAME belief model, observations and costs:
* exact_budget_DP: dynamic program over belief-world subset, used-action mask
  and remaining integer ticks. Minimize final objective, then exact expected
  charged ticks; exact remaining action ties useV1,V2,R,E1,E2. STOP is available
  and wins when further spending cannot improve the objective.
* myopic_cost_normalized: maximize strictly positive immediate expected
  objective reduction per charged tick, ties in the sameV1,V2,R,E1,E2 order.
  Stop if no affordable positive-gain action exists. It does not count future
  complementary actions as immediate information.
* static_RE_cycle: fixed orderR,E1,E2, skipping already used/unaffordable
  actions. There is no target-value-dependent choice or later V fallback.

Two objective versions are separately enumerated:
* brier: final mean two-target Bayes risk.
* verification_shortfall: (2-number of DISTINCT DIRECT V actions)/2.
  Here derived/predicted target knowledge does not count as target verification.

Summaries keep Bayes risk, objective, charged cost, total oracle-call count,
directly verified targets, inferred-but-unverified targets and information bits
SEPARATE. Information bits refer only to the declared independent binary latent
fixture. They are not a count of independent material samples or experiments.

Hand-derived pre-execution checks (not computed-result receipts)
--------------------------------------------------------------
For the base cheap setting, the optimal risk routeR+E1+E2 costs3/4 and attains
Bayes risk0. Myopic spends1 onV1 and finishes at risk1/8. The former directly
verifies0 targets; the latter directly verifies1. Under mandatory verification,
the former is not superior: with B=1 at most one direct V can be afforded.

For equal costs and B=2, V1+V2 attains risk0, while fixedR+E1 costs2 and yields
risk1/8 with0 directly verified targets. Thus a universal advantage claim is
already refuted by its cost assumption and the matched direct-label comparator.

When theta is known, Ei has immediate target information and the cheap-setting
lookahead barrier vanishes. A wrong calibration identity(fake_pair) or absent
relation(no_relation) also invalidates the proposed R/E complementarity about
each target. These negative controls must be retained.

Independent pseudo-calibration control
-------------------------------------
The final, separate illustration enumerates16worlds(theta,u1,u2,h) with an
independent fair generated guessh. Treatingh as true theta after E1/E2 produces
confident pseudo Yi=h*ui and reported risk0, but expected actual mean Brier1/2.
The correctly specified posterior knowsh is independent, keeps each p_i=1/2,
and has mean Brier1/4. The two E calls cost1/2 in the cheap setting; h and its
derived pseudo targets are not measurements. Neither target has been directly
verified. This is a classical wrong-model overconfidence example, not new data.

Public API
----------
build_worlds(scenario)->tuple[World]
target_probabilities(worlds,ids)->(Fraction,Fraction)
bayes_risk(worlds,ids)->Fraction
transitions(worlds,ids,action)->tuple[(observed_bit,Fraction,next_ids)]
after_actions(worlds,action_tuple,ids=None)->tuple[(probability,belief_ids)]
expected_risk(worlds,action_tuple)->Fraction
target_information(worlds,action_tuple,target='joint')->Fraction bits
optimize/myopic/static_cycle(worlds,costs_ticks,budget_ticks,objective='brier')->Policy
Policy.choose(ids,used=(),remaining=None)->action orNone
Policy.value(ids=None,used=(),remaining=None)->(objective,expected_extra_ticks,action)
  value is defined only for the exact-budget DP policy.
summarize(policy)->exact Fraction-valued statistics and exhaustive leaves
complementarity(scenario), pseudo_overconfidence()
json_ready(value)->explicit fraction/numerator/denominator plus displayfloat
plan_configuration()->fixed metadata configuration only

The JSON floating fields are convenience displays. All inference, ranking,
expectations, costs and score calculations use integers/Fraction. There is no
floating-point optimization or outcome-dependent tolerance/parameter choice.

Registration and execution
--------------------------
PLAN_TEMPLATE.json is intentionally unexecutable: status isTEMPLATE and the
registration time is null. Root copies it to a NEW file, sets statusREGISTERED
and actualUTC, checks source SHA and exact configuration, then separately writes
approval{approved:true,scope:"EXACT_HYPERCYCLE_LOGICAL_CONTROL",plan_sha256:<SHA>}.
Only after root approval:
  python enumerate_control.py --plan NEW_REGISTERED_PLAN.json \
    --approval ROOT_APPROVAL.json --output-dir NEW_OUTPUT_DIRECTORY

The output directory must not already exist. EXECUTION_START precedes any
enumeration; EXACT_RESULTS holds48 scenario/cost/objective/policy configurations,
four complementarity summaries and the pseudo-control. Failures are preserved
inFAILED_CONTROL; nothing retries, mutates the material model or retunes policy.

The independently assigned mathematical reviewer should use a different
verification route, e.g. GF(2)linear spans and subset enumeration, rather than
copying the producer DP. These particular fixtures have outcome-independent
linear spans, so adaptation may not beat the best subset; this is not a general
theorem for nonlinear or unknown-relation scientific discovery problems.
