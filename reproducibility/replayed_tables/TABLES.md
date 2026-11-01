# Saved hydrogel score replay

Existing 25 development formulations, three repeated layouts per method. No fitting, raw-data scoring, calibration or new materials.

CRPS/RMSE/width use natural-log viscosity; joint log density is nats per 33-target query. VS1 uses squared log-response units. Original saved bootstrap intervals are copied, not recomputed.

## main_table1

| model | CRPS | RMSE | mean_joint_log_density | coverage90 | width90 |
| --- | --- | --- | --- | --- | --- |
| Coordinate GP | 0.055219 | 0.196569 | 63.122372 | 0.973737 | 0.405326 |
| Generic bounded tilt | 0.052243 | 0.185702 | 63.387907 | 0.973333 | 0.385359 |
| Curvature bounded tilt | 0.048253 | 0.156631 | 63.449706 | 0.972929 | 0.418607 |
| Conditional Gaussian | 0.050252 | 0.161703 | 82.925879 | 0.966061 | 0.364930 |
| Gaussian curvature correction | 0.043277 | 0.119524 | 63.658108 | 0.972121 | 0.403654 |

## main_table2

| model | inside_10_CRPS | continuation_23_CRPS |
| --- | --- | --- |
| Coordinate GP | 0.007155 | 0.179938 |
| Generic bounded tilt | 0.007114 | 0.168071 |
| Curvature bounded tilt | 0.007159 | 0.149808 |
| Conditional Gaussian | 0.005667 | 0.167131 |
| Gaussian curvature correction | 0.007143 | 0.127161 |

## main_table2_extended

| model | inside_10_CRPS | continuation_23_CRPS |
| --- | --- | --- |
| Coordinate GP | 0.007155 | 0.179938 |
| Generic bounded tilt | 0.007114 | 0.168071 |
| Curvature bounded tilt | 0.007159 | 0.149808 |
| Conditional Gaussian | 0.005667 | 0.167131 |
| Gaussian curvature correction | 0.007143 | 0.127161 |
| Matched random Gaussian | 0.006670 | 0.156660 |
| Bayesian Cross | 0.013144 | 0.324593 |
| Bayesian Carreau a2 | 0.014856 | 0.413504 |
| Fixed Student-t diagnostic | 0.006328 | 0.130609 |

## main_table3

| model | macro_CRPS | one_sided_CRPS | mean_joint_log_density | coverage90 |
| --- | --- | --- | --- | --- |
| Gaussian curvature correction | 0.043277 | 0.090792 | 63.658108 | 0.972121 |
| Matched random Gaussian | 0.049908 | 0.111209 | 63.918708 | 0.970101 |
| Bayesian Cross | 0.111590 | 0.230214 | 57.200038 | 0.936162 |
| Bayesian Carreau a2 | 0.153763 | 0.292701 | 23.542170 | 0.873535 |
| Fixed Student-t diagnostic | 0.042906 | 0.092948 | 106.984958 | 0.982626 |

## SI_tableS1

| comparator | mean_difference | saved_descriptive_interval_lower | saved_descriptive_interval_upper | candidate_lower_CRPS_formulations | formulations | interval_role | interval_source_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coordinate GP | -0.011942 | -0.014736 | -0.009183 | 22 | 25 | Copied original fixed-forecast bootstrap, not recomputed | curvature_gaussian |
| Generic bounded tilt | -0.008966 | -0.011612 | -0.006349 | 21 | 25 | Copied original fixed-forecast bootstrap, not recomputed | curvature_gaussian |
| Curvature bounded tilt | -0.004976 | -0.006614 | -0.003177 | 22 | 25 | Copied original fixed-forecast bootstrap, not recomputed | curvature_gaussian |
| Conditional Gaussian | -0.006975 | -0.014525 | -0.001754 | 16 | 25 | Copied original fixed-forecast bootstrap, not recomputed | curvature_gaussian |
| Matched random Gaussian | -0.006631 | -0.009569 | -0.003581 | 20 | 25 | Copied original fixed-forecast bootstrap, not recomputed | comparison |

## SI_tableS2

| layout | Coordinate GP | Generic bounded tilt | Curvature bounded tilt | Conditional Gaussian | Gaussian curvature correction |
| --- | --- | --- | --- | --- | --- |
| spread | 0.012883 | 0.012462 | 0.013048 | 0.012168 | 0.013203 |
| middle_gap | 0.025196 | 0.024970 | 0.025129 | 0.020387 | 0.025835 |
| one_sided | 0.127579 | 0.119296 | 0.106581 | 0.118202 | 0.090792 |

## SI_tableS3

| representation | bounded_CRPS | Gaussian_CRPS |
| --- | --- | --- |
| Fixed random QR | 0.052243 | 0.049908 |
| Fixed curvature contrasts | 0.048253 | 0.043277 |

## SI_tableS5

| model | spread | middle_gap | one_sided | macro_VS1 |
| --- | --- | --- | --- | --- |
| gp | 0.002232 | 0.003017 | 0.142191 | 0.049147 |
| conditional_gaussian | 0.002269 | 0.003601 | 0.240236 | 0.082035 |
| random_gaussian | 0.002223 | 0.003821 | 0.115087 | 0.040377 |
| curvature_gaussian | 0.002245 | 0.003292 | 0.042980 | 0.016172 |
| fixed_student_t | 0.002236 | 0.003349 | 0.084169 | 0.029918 |

## SI_tableS6

| layout | candidate_CRPS | conditional_Gaussian_CRPS | difference |
| --- | --- | --- | --- |
| spread | 0.013203 | 0.012168 | 0.001036 |
| middle_gap | 0.025835 | 0.020387 | 0.005448 |
| one_sided | 0.090792 | 0.118202 | -0.027410 |
| macro | 0.043277 | 0.050252 | -0.006975 |

## paired_all

| comparator | mean_difference | median_difference | wins | losses | ties | formulations |
| --- | --- | --- | --- | --- | --- | --- |
| Coordinate GP | -0.011942 | -0.013810 | 22 | 3 | 0 | 25 |
| Generic bounded tilt | -0.008966 | -0.010840 | 21 | 4 | 0 | 25 |
| Curvature bounded tilt | -0.004976 | -0.006213 | 22 | 3 | 0 | 25 |
| Conditional Gaussian | -0.006975 | -0.006906 | 16 | 9 | 0 | 25 |
| Matched random Gaussian | -0.006631 | -0.006706 | 20 | 5 | 0 | 25 |
| Bayesian Cross | -0.068313 | -0.054763 | 24 | 1 | 0 | 25 |
| Bayesian Carreau a2 | -0.110486 | -0.088643 | 24 | 1 | 0 | 25 |
| Fixed Student-t diagnostic | 0.000371 | 0.001725 | 6 | 19 | 0 | 25 |

## point_only_PCHIP

| layout | pooled_MSE | pooled_RMSE | queries | score_role |
| --- | --- | --- | --- | --- |
| spread | 0.000996 | 0.031561 | 25 | Point-only; no probability law assigned |
| middle_gap | 0.003420 | 0.058481 | 25 | Point-only; no probability law assigned |
| one_sided | 1.401779 | 1.183968 | 25 | Point-only; no probability law assigned |
| macro | 0.468732 | 0.684640 | 75 | Point-only; no probability law assigned |

## figure2b_paired_data

| P | formulation_id | candidate_CRPS | comparator_CRPS | candidate_minus_comparator |
| --- | --- | --- | --- | --- |
| 0 | P0 | 0.130743 | 0.210328 | -0.079584 |
| 1 | P1 | 0.034423 | 0.030956 | 0.003466 |
| 3 | P3 | 0.038692 | 0.032522 | 0.006169 |
| 4 | P4 | 0.032335 | 0.024066 | 0.008269 |
| 5 | P5 | 0.033895 | 0.031274 | 0.002621 |
| 6 | P6 | 0.034871 | 0.036030 | -0.001159 |
| 11 | P11 | 0.034131 | 0.030766 | 0.003364 |
| 15 | P15 | 0.032220 | 0.030349 | 0.001871 |
| 16 | P16 | 0.037756 | 0.044661 | -0.006906 |
| 20 | P20 | 0.050139 | 0.063169 | -0.013030 |
| 21 | P21 | 0.030243 | 0.023188 | 0.007055 |
| 24 | P24 | 0.042821 | 0.054820 | -0.011999 |
| 26 | P26 | 0.030430 | 0.025784 | 0.004646 |
| 27 | P27 | 0.034383 | 0.035568 | -0.001184 |
| 28 | P28 | 0.039810 | 0.047611 | -0.007801 |
| 29 | P29 | 0.044576 | 0.059061 | -0.014485 |
| 30 | P30 | 0.046269 | 0.057222 | -0.010953 |
| 31 | P31 | 0.038178 | 0.044597 | -0.006419 |
| 33 | P33 | 0.040696 | 0.049584 | -0.008889 |
| 35 | P35 | 0.046028 | 0.062297 | -0.016268 |
| 36 | P36 | 0.042486 | 0.053334 | -0.010848 |
| 37 | P37 | 0.035318 | 0.029817 | 0.005501 |
| 40 | P40 | 0.042107 | 0.050094 | -0.007987 |
| 41 | P41 | 0.051468 | 0.062121 | -0.010653 |
| 44 | P44 | 0.057910 | 0.067092 | -0.009182 |

## figureS1_coverage_data

| P | spread | middle_gap | one_sided |
| --- | --- | --- | --- |
| 0 | 20 | 8 | 22 |
| 1 | 31 | 31 | 32 |
| 3 | 33 | 33 | 33 |
| 4 | 33 | 33 | 33 |
| 5 | 33 | 33 | 33 |
| 6 | 33 | 33 | 33 |
| 11 | 33 | 33 | 33 |
| 15 | 33 | 33 | 33 |
| 16 | 33 | 33 | 33 |
| 20 | 33 | 33 | 33 |
| 21 | 33 | 33 | 33 |
| 24 | 33 | 33 | 33 |
| 26 | 33 | 33 | 33 |
| 27 | 33 | 33 | 33 |
| 28 | 33 | 33 | 33 |
| 29 | 33 | 33 | 33 |
| 30 | 33 | 33 | 33 |
| 31 | 33 | 33 | 33 |
| 33 | 33 | 33 | 33 |
| 35 | 27 | 31 | 33 |
| 36 | 33 | 33 | 33 |
| 37 | 29 | 32 | 31 |
| 40 | 33 | 33 | 33 |
| 41 | 33 | 33 | 33 |
| 44 | 33 | 33 | 33 |

## SI_tableS7

| model | spread_RMSE | middle_gap_RMSE | one_sided_RMSE | pooled_RMSE | prediction_role |
| --- | --- | --- | --- | --- | --- |
| Coordinate GP | 0.033434 | 0.061926 | 0.333115 | 0.196569 | Log-response point predictor from the declared probabilistic model |
| Generic bounded tilt | 0.033264 | 0.062887 | 0.313679 | 0.185702 | Log-response point predictor from the declared probabilistic model |
| Curvature bounded tilt | 0.033593 | 0.062477 | 0.261854 | 0.156631 | Log-response point predictor from the declared probabilistic model |
| Conditional Gaussian | 0.033876 | 0.054387 | 0.272651 | 0.161703 | Log-response point predictor from the declared probabilistic model |
| Gaussian curvature correction | 0.033548 | 0.065566 | 0.193478 | 0.119524 | Log-response point predictor from the declared probabilistic model |
| Matched random Gaussian | 0.033414 | 0.071444 | 0.294642 | 0.176101 | Log-response point predictor from the declared probabilistic model |
| Bayesian Cross | 0.073283 | 0.143975 | 0.528100 | 0.318846 | Log-response point predictor from the declared probabilistic model |
| Bayesian Carreau a2 | 0.102374 | 0.172827 | 0.635368 | 0.384726 | Log-response point predictor from the declared probabilistic model |
| Fixed Student-t diagnostic | 0.033548 | 0.065566 | 0.193478 | 0.119524 | Log-response point predictor from the declared probabilistic model |
| PCHIP endpoint cubic | 0.031561 | 0.058481 | 1.183968 | 0.684640 | Point-only; no probability law assigned |
