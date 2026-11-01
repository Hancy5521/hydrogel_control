# Fixed external input reconstruction, version 1.0.0

This small reader reconstructs the existing four-source, 46-curve external input from original native workbook cells. It performs no downloading, installation, fitting, prediction, calibration or scoring. It does not change the frozen experiment or make the source-inspected experiment blind.

`SOURCE_CELL_MAP.json` contains public source URLs, original byte hashes, cell addresses, viscosity divisors and expected vector hashes. It contains no measured shear-rate or viscosity arrays and no internal original-file paths. `REBUILD_VALIDATION.json` records the actual comparison against the frozen input; it is a builder self-check, not an independent review.

## Supply the original files

Obtain the source files yourself from the `retrieval_url` entries and save them in one raw directory using each `source_file_key` as the basename. The table has eight required numeric workbooks and two optional supporting text files (Liverpool README and DataverseNL LabBookReference). Missing optional text files do not prevent reconstruction; if supplied, their bytes and hashes are checked. No excluded workbook adds a curve to this map.

Every required file must match both the recorded byte length and SHA256 before any measurements are read. A changed publisher attachment must be treated as a different source version. The reader will stop rather than silently accept it.

The granular-hydrogel source has an official `other-open` metadata label, with no named raw-data redistribution license verified. Fetch it from the original author URL for local reading. This package does not include or authorize redistribution of that workbook or its numerical vectors. Follow the original terms and attribution for every source; the other source records identify CC BY3.0 or CC BY4.0 as recorded in the map.

## Run

Use Python 3.9 or newer. XLSX reading uses only the standard library (`zipfile` and XML). The five XLS workbooks require task-local `xlrd` version 2.0.2. Make that existing package available through `PYTHONPATH` or the selected Python environment; the script never installs it.

For example, with the map, script and a locally populated `raw` directory in the current directory:

```powershell
python .\rebuild_inputs.py --source-map .\SOURCE_CELL_MAP.json --raw-dir .\raw --output .\rebuilt_inputs.json
```

If using a separate local directory containing the `xlrd` package, set `PYTHONPATH` to that directory before running the command. No system spreadsheet application is used. Choose a new output path: an existing output is always refused.

The output is minimal curve JSON containing `curve_id`, `group_id`, `source_doi`, `gamma` and `eta`. Retain that local numerical output outside the public Git archive, particularly for the granular source. The printed receipt contains only counts and output hash.

## Exact reconstruction rules

- Preserve exact worksheet names, including Liverpool trailing spaces, and read only the listed cells in their listed order.
- Liverpool uses native per-run locations and viscosity divided by 1000, converting mPa.s to Pa.s. It does not use combined or summary curves.
- PLOS uses the retained `metadata.source_cells`; two duplicate figure records and the incomplete Figure7 row remain excluded exactly as in the frozen input.
- DataverseNL uses the retained native rows, worksheet `Flow sweep - 2`, shear rate column B and viscosity column C.
- Granular hydrogels uses retained native rows of worksheet `Flow`, shear rate AF and viscosity AG.

No support filtering, rate selection, interpolation, sorting, digitization or fitted transformation is repeated. The map already identifies the previously admitted in-support rows. The reader checks that the mapped vectors remain finite, positive, distinct and ascending, then verifies each expected vector hash. Mapped XLSX formula/string/boolean cells are rejected. For XLS, `xlrd` supplies numeric BIFF cell values; the reader does not claim independent detection of BIFF formula identities.

The vector hash is SHA256 over UTF-8 encoding of `json.dumps({'gamma': [float(x), ...], 'eta': [float(x), ...]}, separators=(',', ':'), allow_nan=False)`, with key order gamma then eta. Numeric values are not rounded and no epsilon is used. The original frozen input file has SHA256 `b4fdec609115627df63fed1758bf8ee9471c501b00071f31b2575cfe9b43532a`; the protocol has SHA256 `348d507321a41b60edf97f1da757f62f0a68c0d926b89696bfa4eeba0e50f4bd`.

## Actual validation

The existing original assets were copied byte-for-byte to a private raw directory and read with this CLI. All 10 supplied file hashes, 46 expected vector hashes and 2,114 scalar `float.hex()` comparisons passed with zero tolerance. The result contains 1,057 native pairs: 276 six-per-curve anchors and 781 higher-rate targets. Re-running to the same output correctly refused overwrite. Frozen input and protocol hashes were unchanged. No forecasts or scores were rerun.

The core reconstruction package consists of this README, `SOURCE_CELL_MAP.json`, `rebuild_inputs.py` and `REBUILD_VALIDATION.json`. The parent project also adds public `fetch_sources.py` and `run_replay.py` orchestration entry points plus a model-byte guard check receipt. The private directory contains original-file locators, raw copies, the local reconstructed numerical output and construction/verification receipts; it remains outside Git. The replay entry point verifies the original frozen runtime, model and scoring-program hashes before constructing a new replay protocol.
