# Frozen-model external source assessment

This is an additional source-inspected, zero-retraining evaluation. The original
21-page hydrogel manuscript,11-page supplement, frozen model and development
results remain unchanged. It is not a new universal-material model.

Start with [the Chinese report](report/READ_FIRST.html), [all source scores](report/ALL_SOURCE_RESULTS.csv),
[composition detail](report/LIVERPOOL_GROUP_RESULTS.csv) and [English addendum](report/ENGLISH_EVIDENCE_ADDENDUM.txt).
Four sources,46 curves,23 conservative material analysis groups and781 native
targets were retained. A group is not a certified preparation batch. Native
curves/conditions within a group can be concentration variants, not replicates.

Liverpool group-macro log-CRPS improves from0.327109(GP) and0.293629(conditional
Gaussian) to0.273635. Marginal90% coverage remains71.56%, with zero coverage in
the peK10% group. Candidate CRPS is worse than the reference GP on the other
three sources. Dynamic crosslinked alginate is a severe failure. All results
and point-only power-law/PCHIP controls remain visible; point methods receive
no invented CRPS or coverage.

The public projection contains own code, protocols, derived scores/moments and
source identity. Raw workbooks, admitted numeric curves, full HTTP responses,
signed redirects and local locators are not uploaded. In particular the granular
source's other-open metadata is not treated as a verified CC license. Its public
original URL/hash permits readers to inspect upstream access and terms themselves.

The original execution protocol's hash and content bindings are preserved. A
later portable reconstruction is a reproducibility replay, not a new independent
confirmation. Source inspectors and the assessor saw the new responses; no
blinding claim is made. This does not certify same-formulation new batches,
independent calibration or measured laboratory savings. No original P/S outcomes
were reopened and no candidate/reference weights were retrained.

## Reconstruct the external inputs on H drive

The non-numeric cell map in `reproduction/` identifies the exact eight numeric
source files and all 46 vectors by SHA256. It contains cell addresses, not the
source viscosity/rate arrays. The source reader rejects changed files, blank or
formula cells, unit conversion mismatches and altered vector hashes. See each
upstream source's own terms; no new redistribution license is assigned.

With Python 3.10+ and the versions in `requirements-external.txt`, run from the
repository root (use a fresh receipt/output name on subsequent replays):

```sh
python external_validation/20261101T1239423181800Z/reproduction/fetch_sources.py --raw-dir H:/hydrogel_external_raw --receipt H:/hydrogel_external_fetch_receipt.json
python external_validation/20261101T1239423181800Z/reproduction/run_replay.py --repo . --raw-dir H:/hydrogel_external_raw --output-dir H:/hydrogel_external_replay_01 --prepare-only
```

The first command reuses exact cached files or attempts each public download
once, stopping on a failure without replacement or retries. The second only
rebuilds inputs and records a fresh local replay protocol. Omit `--prepare-only`
and use a different output directory to execute the frozen comparison as a
reproducibility replay; it is not a new independent study. No model is trained.
The original execution protocol and scores remain in their original archive.

The portable preparation path was actually tested against all eight cached
numeric sources: 46 vector hashes and all 1057 native pairs match exactly.
This test did not repeat network downloads, predictions or material scoring.
The downloader's existing-file path was exercised; original network acquisition
has separate source receipts, not a claim that every future endpoint will work.
