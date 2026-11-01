# Hydrogel curve continuation

Additional checkpoint: [four-source external evaluation](external_validation/20261101T1239423181800Z/README.md). The original manuscript, model and development results remain unchanged; new external gains and failures are reported separately.

This repository archives the **2026-11-01 expanded author-review checkpoint** of a focused hydrogel precursor-ink study: two fixed local curvature contrasts receive a shared Gaussian correction, while the complete high-dimensional reference residual is retained. The main intended use is continuation of a flow curve from six low-rate anchors.

论文与程序存档，非已发表或保证录用的论文。本次固定保存扩充版21页英文主文、11页补充材料及其推断程序和冻结权重。基础Gaussian方法有既有先例；研究贡献是特定水凝胶续接任务的表示与训练适配，以及对应的受控证据。

## Start here

- [English manuscript draft](paper/Hydrogel_Continuation_English_Revised.docx) and [supplement](paper/Supplement_Revised.docx)
- [Revision details](paper/REVISION_NOTES.md)
- [Saved score replay and tables](reproducibility/README.md)
- [Independent frozen inference checks](verification/README.md)
- [Synthetic training primitive](training_demo/README.md)
- [Current gaps and author tasks](docs/evidence/GAP_REVIEW_ZH.txt)
- [Author decisions and optional new evidence](docs/AUTHOR_ACTIONS.md)
- [Version history](docs/VERSION_HISTORY.json)
- [Model and scope](MODEL_CARD.md)
- [Preserved result summary](results/SUMMARY.json)
- [Remaining evidence and blank resource templates](docs/evidence/FEASIBILITY_ZH.txt)
- [Source and rights](RIGHTS_AND_PROVENANCE.md)
- `CHECKPOINT.json`, `MANIFEST.json` and `models/RUNTIME_BINDINGS.json` identify exact file versions.

## Run frozen inference

Requires Python, NumPy and SciPy. The archived dependency versions are in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python verify_checkpoint.py
python runtime/deployment/generate_curve.py --model models/SHARED_MODEL.json --input examples/SYNTHETIC_ANCHORS.json --output demo_generated.json --draws 64 --seed 20261031
```

The example anchors are **synthetic values from the frozen reference mean**, not measurements or independent validation. Generated outputs explicitly say `is_measured_data: false`. The command does not fit a model. Existing output files are not overwritten. Use exactly six positive, distinct `observed_gamma` values in s^-1 and six positive `observed_eta` values in Pa s. `target_gamma` must stay inside the frozen support; see the model card. `runtime/program/complete_curve.py` is an inherited dependency/helper, not the current candidate's main entry point.

## What the current evidence supports

On the already-inspected 25 formulation records and three layouts, the frozen curvature Gaussian has macro CRPS **0.043277**, compared with **0.050252** for the conditional Gaussian and **0.049908** for the matched fixed-random Gaussian. Gains concentrate in one-sided continuation. The fixed secondary one-sided variogram-1 score is **0.042980**, versus **0.142191** for the GP, **0.240236** for the conditional Gaussian and **0.115087** for matched random Gaussian.

These are retrospective development comparisons. P0 is retained and materially influences some means. The variogram score is not strictly proper, is blind to common shifts and common-mode variance, and does not certify a correct copula or whole-curve calibration. Interpolation and some uncertainty metrics favour the conditional Gaussian. The fixed Student-t diagnostic alleviates one local coverage failure but worsens continuation and overall overcoverage; it is not substituted for the frozen candidate.

**The deployment model uses all 25 P curves plus the lawful six-anchor portions of 20 S records. Do not reevaluate old P with these deployment weights and call it the leave-one-formulation-out paper result.** This public checkpoint is runnable inference plus paper/results archival. It does not contain all original out-of-fold predictive laws or raw measurements required for complete experiment replay.

## Remaining evidence

No new compatible independent preparation batch, reserved calibration cohort or real comparison of instrument/lab resources is available in this checkpoint. Existing data cannot supply those stronger claims by new masks, bootstrap, generated curves or the ratio 6/39. The two CSV files under `docs/evidence` contain **headers only, zero observations**. They prepare future data collection; they are not registered experiments or savings results.

## Archive policy

Current manuscripts, source and model files are copied byte-for-byte and verified by SHA256. `.gitattributes` disables automatic line-ending conversion for this exact-byte checkpoint. Preserve prior commits/checkpoint branches; do not force-push them when updating the main branch. Later archival should explicitly select reviewed files, verify source/model/data roles and create a new checkpoint.

Large local ZIPs, private/raw data, hidden S responses, local paths, credentials and internal logs are deliberately excluded. `LOCAL_ARCHIVE_INDEX.json` records retained local ZIP checksums; it does **not** imply those ZIPs have been uploaded. No scheduled auto-upload or journal submission is configured.

Repository: <https://github.com/Hancy5521/hydrogel_control>.

## Computational evidence commands

These commands perform saved-score arithmetic or synthetic software checks, not new material experiments. Run in the repository root with the declared dependencies:

```sh
python reproducibility/replay_tables.py --output-dir replayed_local
python -m unittest discover -s reproducibility -p test_score_replay.py -v
python verification/verify_frozen_inference.py --repo-root . --output software_check_new.json
python training_demo/training_primitive_demo.py --repo-root . --output synthetic_training_check_new.json
```

The last command makes one original optimizer call on the explicitly synthetic fixture; it never refits the material checkpoint. Read each module README before interpreting its receipt. Model/source/result bytes and original scientific scores are unchanged. Full original raw-data/outer-fold replay remains separately scoped.

This release still does not include raw measured curves, the original training runs or all outer-fold predictive laws. Full experiment replay remains incomplete until the authors approve and provide the remaining processed-input access and reuse terms.

The current 21-page main manuscript and 11-page supplement include the complete point-error comparison, including favourable PCHIP interpolation and unfavourable continuation. The S7 companion adds a ninth replay test and direct S7 CSV export without changing old score values. See paper/REVISION_NOTES.md and docs/AUTHOR_ACTIONS.md.
