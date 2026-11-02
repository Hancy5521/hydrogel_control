# Saved score weighting sensitivity

This is a descriptive reaggregation of the already frozen Liverpool external-study scores. It adds no material measurement, forecast, target score or fitted parameter. The original five-group primary endpoint remains unchanged. All four external sources and all six controls remain in the manuscript.

The primary population weights three gellan-gum groups and two peK groups equally. Its candidate/GP/conditional-Gaussian CRPS values are 0.273635/0.327109/0.293629. Giving GG and peK each half instead yields 0.312928/0.340916/0.295105: the advantage over conditional Gaussian reverses. Equal-family log-RMSE also no longer favours the candidate. Candidate equal-family coverage is 64.44%. These are different evaluation populations, not evidence that a new model was fitted.

For a GG weight w the risk is R(w) = w R_GG + (1-w) R_peK. The candidate's saved CRPS is lower than GP above w = 0.390181 and lower than conditional Gaussian above w = 0.547129. The crossing is not a chosen deployment parameter or a future-population guarantee. Arithmetic omission of a saved group is influence analysis, not cross-validation or leave-one-group-out training.

From the repository root, with Python standard library only:

```sh
python robustness/20261102_weighting/reaverage_saved_scores.py --repo-root . --output new_weighting_summary.json
```

The exact score file SHA is checked. Existing outputs are refused. The portable program matches all saved method metrics and 30 arithmetic group omissions exactly. An independently written, 50-digit Decimal audit retained from the original diagnostic compared 307 scalar quantities; maximum difference from float arithmetic was 1.31e-13. Replaying arithmetic is not another independent material test.

The bilingual report is in report/READ_FIRST.html. Original plan timing, source score identity, figure data, full scalar results and audit outcomes are preserved. No statistical significance, certified batch independence, independent calibration or laboratory saving is claimed.
