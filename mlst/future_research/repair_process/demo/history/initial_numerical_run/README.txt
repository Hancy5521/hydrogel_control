Local repair as a computational process

This directory is a small synthetic process demonstrator. It does not simulate
a spider, biological tissue, a specific material or the source hydrogel data.
Nodes represent data locations. Graph edges are a fixed declared relation.
The central blank region is absent information, distinct from a numerical zero.

What happens
1. Known nodes stay fixed and retain their immutable observation mask.
2. A mask/topology-only activation front reaches the blank region from its edge.
3. Local tension-like conditional-mean updates coordinate adjacent free nodes.
4. Continued relaxation approaches the exact Gaussian conditional mean.
5. A separate Gaussian Gibbs run retains the corresponding nonzero uncertainty.

The first four steps make the healing analogy visible. They do not create new
evidence. The precision, energy, Gauss-Seidel solver and Gibbs sampler are
established mathematics and are not claimed as a new learning algorithm.
In particular, neither this small grid nor its numerical success establishes
accuracy on sparse material matrices or one-sided response continuation.

Core boundaries
repair_graph.py defines the synthetic grid, observation mask and proper Gaussian.
local_repair.py contains the local relaxation and exact full-conditional sampler.
run_process_demo.py runs the fixed configuration, stores results and raw plots.
render_process_snapshots.py only renders already recorded snapshots with more
readable panel spacing. It does not repeat sampling or modify recorded states.

Requirements
Python with NumPy and Matplotlib. Original run: Python 3.13.5, NumPy 2.1.3,
Matplotlib 3.10.0. Runtime/library binaries are not embedded.

Run from this directory or use absolute script paths
python run_process_demo.py --output-dir NEW_DIRECTORY
python render_process_snapshots.py --input-dir NEW_DIRECTORY

Both commands refuse to overwrite their existing output. PROTOCOL.json binds
the fixed numerical configuration and three numerical source files. No seed,
budget or precision was selected after seeing the diagnostics. The plotting
helper was added later to correct crowded labels in the first figure; its own
source/input hashes and exact saved-frame identity are recorded separately.

Selected figures from the actual run
output/REPAIR_PROCESS_R2.png shows six computed repair states at sweeps
0, 1, 2, 4, 20 and 300, with 0, 24, 40, 49, 49 and 49 active missing nodes.
The early front states are computation transients, not posterior samples.
output/REPAIR_PROBABILITY.png contrasts the deterministic point result with
the exact marginal uncertainty and the empirical Gibbs mean/quantiles.
The first crowded output/REPAIR_PROCESS.png is retained as a non-selected draft.

Recorded diagnostics
The maximum deterministic mean difference from the exact conditional result is
1.11e-16. Over the single fixed 12000-retained-sample Gibbs run, maximum mean
error divided by exact marginal SD is 0.0302, maximum relative marginal variance
error is 0.0337 and relative covariance Frobenius error is 0.0502. Thirty-two
deterministic initializations collapse to an approximately zero-variance point
result. These are synthetic implementation diagnostics, not calibration claims.

Generated entries stay latent algorithm states. They never turn into new observed
nodes. A proper ancestral or Gibbs sampler can condition on generated latent
values internally while still integrating their uncertainty. Permanently treating
a generated mean or one chosen draw as newly measured evidence would be different.

No material model, material response array, hidden S outcome, frozen paper or Git
asset is loaded or altered. The original MLST manuscript and inference model
remain separate. Independent numerical/code audit is performed outside this
directory after the first run and is not falsely prefilled as complete here.
