POST-HOC FIXED LEARNING BASELINES: SOURCE PREPARATION, NOT EXECUTION

Purpose and evidence boundary
The main YJ-NIW model, its3840 frozen forecasts, scores and original four gates
are unchanged. This adds two standard learning-reference recipes on the same
already-consumed64fit/192development cohort. It is a post-hoc baseline check,
not new independent confirmation or proof against every method in the field.
No baseline outcomes have been observed during source preparation; no source
function, sklearn import, fit or prediction is executed before root approval.

Three small modules
code/baseline_models.py: fit-only feature normalization, fixed GPR and forest
recipes, export of all numerical state and exact empirical probability weights.
sklearn is imported only inside its authorized fit functions.
code/run_baselines.py: exact source/input/recipe/library/approval checks, six
fits, all1536forecast laws and complete exclusive freeze. It does not even open
or hash the saved target or candidate-score files in the forecast process.
code/score_baselines.py: verifies all1536laws and state bindings first; then reads
the existing frozen target and candidate summary, computes only the new baseline
scores, and compares same-cohort both-proxy macroCRPS. It never reruns the main
model or recalculates its score.

Inputs and fixed number of fits
Use exactly phase2 forecast_once/VISIBLE_INPUTS.json, which is bound to the
unchanged completed main FORECAST_FREEZE.json. No new group, raw gzip, response
cleaning, teacher prediction or diagnostic-target fit is permitted. Models see
64x3native fit values and192x2native proxies; only e_form/gap are features and
bulk is the target. All identity metadata stays for source/group checks only.

For each of two methods, train once on each of the three nonempty masks:
e_form only, gap only, both. Thus6fits, not8. The none mask is a fixed target
marginal without any learning-method fit. Two methods x4masks x192groups=1536
laws all freeze before scoring. Training the same64labels six times does not
increase material N. Only fit64 feature means and sampleSD are used to normalize
features; none of the192proxy values estimates these transforms.

Gaussian-process recipe, without a search over recipes
GaussianProcessRegressor normalize_y=True, alpha=1e-8,
optimizer=fmin_l_bfgs_b, n_restarts_optimizer=0, random_state=20261106;
kernel Constant(1,[.01,100])*RBF(ones(k),[.01,100])+White(.01,[1e-6,10]).
The optimizer is allowed its one prescribed learning run; no additional seed,
kernel, bound or hyperparameter grid is selected after seeing scores. All fit
and prediction warnings, including every ConvergenceWarning, are retained.
One run need not find a globally optimal kernel and bounds may be reached.

Features and GPR calculations are float64. normalize_y uses sklearn's learned
target mean/populationSD, which is exported separately from feature sampleSD.
Export kernel shape/parameters/logtheta/logbounds, normalized Xtrain/ytrain,
alpha_ coefficients, L_ Cholesky, target mean/SD and LML. Independent replay can
reconstruct RBF cross-kernel, mean and predictive variance without rerunningfit:
cross WhiteKernel(X,Y)=0; test diagonal includes learned WhiteKernel noise;
alpha nugget belongs only to training diagonal and must not be added again to
test noise. Predictive std from sklearn includes WhiteKernel observation noise.
None mask: Gaussian with native fit-target mean and sampleSD; no GPRfit.

Forest recipe and precise distribution convention
RandomForestRegressor256trees,min_samples_leaf3,max_features=1(integer),
bootstrap=True,criterion=squared_error,random_state20261106,n_jobs2,
max_depth=None. Other choices are sklearn1.6.1 defaults, not tuned.

After float64 fit-only standardization, forest fit/apply inputs are explicitly
float32, matching sklearn RF internal casting. Export the exact float32 values
as numerical JSON together with tree_input_dtype='float32'. An independent
threshold traversal must compare these float32 inputs against stored thresholds;
using original float64 features can change boundary leaves and is not a replay.

For each tree terminal leaf, include every one of the original64fit rows that
lands in it, including OOB rows. Do not use bootstrap multiplicity as CDF mass.
For queryi and original supportj:
 w_j(i)=sum_t I(leaf_train_jt=leaf_query_it)/(256*n_leaf_all_original_t).
Weights sum exactly1 using Fraction; serialize each rational mass and a labeled
float display. Support is the original64native fit-target values, not model
means or synthetic samples. Export every tree's children/features/thresholds,
node counts/value and all64train/192diagnostic leaf IDs. Independent replay can
traverse exported trees, rebuild memberships and exact weights without refit.

This is explicitly an all-original-leaf empirical/QRF-inspired convention,
not a claim of executing an official Quantile Regression Forest package. Its
probability mean/median may differ from rf.predict(), because rf.predict averages
bootstrap-trained node means while this CDF uses all-original membership. Scores
must come from the declared CDF, not secretly from rf.predict(). None mask is
uniform1/64empirical fit-target CDF, with no forest fit.

Exact empirical scoring and atoms
Aggregate identical source target values by their rational masses. Quantiles use
Q(p)=inf{x:F(x)>=p}, retaining ties; predictive median is Q(.5). Central80/90/95
intervals may legitimately have zero width. CRPS is a finite, deterministic
piecewise-CDF integral, mathematically equal to the weighted energy expression:
sum_j w_j|y_j-y| - .5sum_jk w_jw_k|y_j-y_k|.
The producer integrates step-CDF intervals as nonnegative terms, avoiding random
samples and subtractive pair-sum cancellation. Native source positions remain
floating-point values; 'exact' refers to the empirical law/no sampling approximation,
not a claim of full directed-interval arithmetic.

Empirical PIT is recorded as F(y-),F(y),and their midpoint, including exact mass
fractions. Its midpoint is not a continuous-uniform PIT. No original continuous
PIT admission threshold or KS-uniformity claim is applied to the discrete forest
law. GPR continuous PIT is descriptive only. Both methods retain every mask,
interval coverage/width, median-RMSE and per-group score.

Comparison with the already-frozen main candidate
Only existing phase2RESULTS.summary.yj_joint_t.both is used, after its unchanged
file hash has been checked. The scorer's trusted JSON parser physically parses
that saved object, but it does not use its other scores/labels, refit its model
or calculate its CRPS again. Report candidate / min(new GPRdual, new forestdual)
and native difference, without changing the original main model's four-gate pass.
These are fixed standard references under this64-fit-row task, not all possible
field SOTA methods or a new confirmation set.

Pinned runtime, no upgrade
scikit-learn1.6.1,numpy2.1.3,scipy1.15.3,joblib1.4.2. Runtime package metadata is
checked before fits; no pip/conda/update/network is called. RF usesn_jobs2 as
specified. No new neural network, agent, calibration model or main algorithm.

Source/plan and separate-process order
The TEMPLATE remains non-executable until root fills actualUTC, a pre-recipe
scope and the independent empirical/Gaussian score audit receipt, and approves
its exact SHA256 with scope POSTHOC_FIXED_STRONG_BASELINES. All new sources and
the unchanged three parent scoring dependencies are pinned. Score-only truth
and candidate files are deliberately not opened/hashed by forecast; scorer first
validates complete1536freeze, then checks their exact bindings and reads them.

Only after exact root approval:
D:/Anaconda/python.exe code/run_baselines.py --plan <registered_plan.json>
 --approval <forecast_approval.json> --output-dir <new_absent_forecast_dir>
D:/Anaconda/python.exe code/score_baselines.py --plan <registered_plan.json>
 --approval <score_after_freeze_approval.json> --forecast-dir <frozen_forecast_dir>
 --output-dir <new_absent_score_dir>

All outputs exclusive. Retain convergence warnings, model states, bad results
and failed launches; no recipe sweep, no automatic rerun on failure, no candidate
retune or old-result overwrite. A baseline defeat changes the strength of the
paper's claims, not the contents of the frozen main-model experiment.
