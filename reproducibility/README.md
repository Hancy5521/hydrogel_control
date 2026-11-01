# Replay the saved hydrogel development results

This small package closes a **result-transparency** gap. It publishes scalar scores already computed for the retained 25 P formulations and re-creates their tables with Python's standard library. It does not fit a model, evaluate predictive densities against target labels, open hidden S outcomes, or reproduce the original training and posterior calculations.

```sh
python replay_tables.py --output-dir replayed_tables
python -m unittest -v test_score_replay.py
```

Python 3.9 or newer is sufficient. No NumPy, SciPy, plotting library, original data, model checkpoint, internet connection or GPU is used. The generated JSON and CSV retain numeric precision; Markdown rounds values to six decimal places for reading. Output is deterministic. `replayed_tables/validation.json` checks 238 saved comparison scalars, including all nine methods, the inside/continuation split, VS1 and the point-only baseline.

## Files and roles

- `ledger/marginal_scores.json` and CSV: 675 saved method-query rows, nine methods × 25 formulations × three six-anchor layouts. Fields contain scalar CRPS, mean squared log error, joint log density, marginal coverage counts/fractions and log-interval width. Where originally recorded, interval score and pair-sampling MCSE are retained.
- `ledger/variogram_scores.json` and CSV: 375 saved VS1 rows, five laws × the same 25 formulations × the same three layouts. Each query uses 528 unordered pairs.
- `ledger/point_only_scores.json` and CSV: 75 original PCHIP endpoint-cubic point-baseline MSEs. It is not assigned a predictive covariance, CRPS, probability interval or joint density.
- `ledger/reported_facts.json`: original aggregate comparison witnesses and saved bootstrap intervals. Copying an interval is distinct from rerunning its bootstrap. Original interval-score aggregates are available as reported facts; missing per-query interval scores are not fabricated.
- `provenance/SOURCES.json`: source hashes, logical roles, exact grouping and export rules. The original source result files are not mirrored: some also contain forecast arrays. Their local machine paths remain outside this public package.
- `provenance/NUMERICAL_HISTORY.json`: relevant stopped numerical branches and their resolutions. Failed partial aggregates are not substituted for the completed evaluation; all final 75-query method collections, including unfavourable comparisons, are retained.
- `score_replay.py`: identity validation and pure aggregation. `replay_io.py`: deterministic JSON/CSV/Markdown formatting. `replay_tables.py`: the command-line entry point.

## What is reproduced

| Paper item | Coverage from this package |
| --- | --- |
| Main Table 1 | All five original methods and numeric columns re-aggregated from saved query scores. |
| Main Table 2 | Original five methods re-aggregated from sums of already saved target CRPS in the original ten-inside/23-continuation partition. The extended table includes all nine methods. |
| Main Table 3 | All five original additional-control rows and numeric columns re-aggregated. |
| SI Table S1 | Paired means and win counts re-aggregated by formulation; original 2000-draw interval endpoints copied with explicit source identity. No new resampling. |
| SI Tables S2 and S3 | Original layout CRPS and the completed representation/correction-family comparison re-aggregated. |
| SI Table S4 / Main Figure 4 | Not reconstructed: these require saved model-information/covariance diagnostics, not just score scalars. |
| SI Tables S5 and S6 | Original five-law VS1 table and layout-difference balance re-aggregated. |
| Main Figure 2a / 5a / 5b | Exact numeric plotting inputs are the generated S2 / S3 / S5 CSVs. Original rendering style is not recreated here. |
| Main Figure 2b | Generated `figure2b_paired_data.csv` supplies all 25 formulation-level candidate-minus-conditional-control differences. |
| SI coverage heatmap | Generated `figureS1_coverage_data.csv` supplies the retained candidate's counts out of 33 for every formulation/layout. |
| Main Figure 3 / fixed P0/P1 response-curve panels | Not reconstructed: raw measured curves, predicted mean curves and interval arrays are deliberately absent. |

## Grouping and metrics

All methods use the same 25 formulation identities. The 675, 375 and 75 rows are **not additional materials**. For each probabilistic method the 75 queries are repeated views of 25 formulations, with 2,475 scored positions but only 950 unique formulation-position pairs. Preparation-batch independence remains unresolved.

The response is `y = ln[viscosity/(1 Pa s)]`. CRPS, pooled RMSE and interval width are in dimensionless natural-log response units; score percentage improvements are not percentage viscosity errors. Joint log density is in nats per 33-target query. VS1 is the average over 528 squared discrepancies of absolute log-response differences, so it has squared log-response units. Lower losses and higher log density evaluate their respective declared targets; coverage is not maximized without regard to interval width.

Pooled RMSE is `sqrt(sum(n_targets * mean_mse) / sum(n_targets))`. Averaging query RMSEs gives a different and incorrect statistic for the reported pooled endpoint. The one-sided query still includes ten internal targets and 23 continuation targets. Its partition is descriptive arithmetic over frozen scores, not a new primary endpoint or an experiment performed with fewer measurements.

The Gaussian candidate is not uniformly best: the conditional Gaussian is stronger for interpolation and some uncertainty metrics; fixed Student-t has slightly lower macro CRPS while worsening continuation and overcoverage. These rows remain visible. Neither joint log density nor VS1 establishes calibrated whole-curve prediction. All comparisons remain retrospective development results.

## Data and rights

The reported rheology source is dataset version 0.0.2, DOI [10.5281/zenodo.19602891](https://doi.org/10.5281/zenodo.19602891). Inherited records identify CC BY 4.0 for the dataset. This package contains the authors' existing derived score scalars; it does not mirror original responses, recipe descriptions, fitted source parameters, forecast means/intervals/covariances or third-party full text. No new software/data/manuscript reuse license is granted here; the authors must finalize licensing and publication declarations.

Score transparency is improved; full raw-data, fitting, posterior and out-of-fold predictive-law reproducibility remains a separate obligation. No new-batch validation, calibration, experimental resource saving or journal acceptance is claimed.
