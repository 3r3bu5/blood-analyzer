# WBC Blood Film Analyzer

Research proof-of-concept scaffold for the M0/M1 data gates and the M2 DinoBloom-B MLL23 classifier baseline in `WBC_Blood_Film_Analyzer_Technical_Design.md`.

Implemented now:

- repository skeleton under `src/bloodfilm`
- typed stdlib configuration objects loaded from JSON-compatible YAML files
- environment report command
- asset registry and DinoBloom-B smoke checks that fail clearly when weights are absent
- PNG image intake and quality measurements without silently substituting models
- MLL23 mapping, manifest, audit, checksum, and deterministic split workflow
- frozen DinoBloom-B embedding extraction for MLL23 crops
- linear, MLP, and cosine head training from cached embeddings
- research-only classifier bundle packaging and inspection
- single-crop and folder classification commands that fail clearly when weights or ML dependencies are absent

Not implemented yet:

- detector training or end-to-end field analysis
- clinical validation or diagnostic use

Configuration files:

- `configs/classifier_mll23.yaml` (M2 classifier baseline settings)
- `configs/classifier_uncertainty.yaml` (research-only classifier review thresholds)
- `configs/detector_txl_pbc.yaml` (M3 detector settings)
- `configs/inference.yaml` (M4 analyzer settings)
- `configs/mappings/wbcbench.yaml`, `matek19.yaml`, `bodzas.yaml` (M7 external-dataset mappings)

## Commands

```bash
PYTHONPATH=src python3 -m bloodfilm.cli environment --output outputs/reports/environment.json
PYTHONPATH=src python3 -m bloodfilm.cli smoke --config configs/base.yaml
PYTHONPATH=src python3 -m bloodfilm.cli assets list --output outputs/reports/assets.json
PYTHONPATH=src python3 -m bloodfilm.cli assets download mll23 --report-output outputs/reports/mll23_download.json
PYTHONPATH=src python3 -m bloodfilm.cli assets verify mll23 --report-output outputs/reports/mll23_verify.json
PYTHONPATH=src python3 -m bloodfilm.cli dataset audit mll23 --report-output outputs/reports/mll23_audit.json
PYTHONPATH=src python3 -m bloodfilm.cli dataset build-manifest mll23 --output data/manifests/mll23_manifest.csv --invalid-output data/manifests/mll23_invalid.csv --checksum-output data/manifests/mll23_checksums.csv --report-output outputs/reports/mll23_audit.json
PYTHONPATH=src python3 -m bloodfilm.cli dataset split --manifest data/manifests/mll23_manifest.csv --output data/manifests/mll23_splits.csv --report-output outputs/reports/mll23_leakage_report.json
PYTHONPATH=src python3 -m bloodfilm.cli embeddings extract --config configs/classifier_mll23.yaml --splits data/manifests/mll23_splits.csv --device auto --output data/embeddings/mll23_embeddings.pt --report-output outputs/reports/mll23_embeddings.json
PYTHONPATH=src python3 -m bloodfilm.cli train classifier --config configs/classifier_mll23.yaml --head all --embeddings data/embeddings/mll23_embeddings.pt --checkpoint-dir outputs/checkpoints --report-output outputs/reports/mll23_head_comparison.json
PYTHONPATH=src python3 -m bloodfilm.cli classifier package-bundle --checkpoint outputs/checkpoints/mlp.pt --comparison-report outputs/reports/mll23_head_comparison.json --bundle models/mll23-dinobloom-b-mlp-v0.1
PYTHONPATH=src python3 -m bloodfilm.cli classifier inspect-bundle --bundle models/mll23-dinobloom-b-mlp-v0.1 --output outputs/reports/mll23_bundle_inspection.json
PYTHONPATH=src python3 -m bloodfilm.cli classifier classify-crop --bundle models/mll23-dinobloom-b-mlp-v0.1 --image path/to/cell.tif --output outputs/predictions/cell.json
PYTHONPATH=src python3 -m bloodfilm.cli classifier classify-folder --bundle models/mll23-dinobloom-b-mlp-v0.1 --input path/to/crops --output outputs/predictions/
```

Classifier inputs are single WBC crop images readable by Pillow. The production path converts
to RGB, directly resizes to 224x224, applies the DinoBloom-B ImageNet normalization recipe,
then runs the frozen DinoBloom-B backbone and packaged MLP head. Outputs are JSON with
`research_only`, `bundle`, `image`, `decision`, `top_predictions`, and full per-class
probabilities. `decision.status` is one of:

- `accepted`: high-confidence research prediction, still not diagnostic
- `review_required`: threshold, margin, entropy, or high-risk class requires review
- `unknown`: confidence is too low or an error/unsupported input prevents classification

Missing assets fail explicitly. For example, absent or checksum-mismatched DinoBloom-B weights
return a non-zero CLI exit with a `MISSING_ASSET`, `MODEL_LOAD_ERROR`, or `CONFIG_ERROR` message.
Folder inference writes one JSON per input and records per-file errors instead of silently
forcing a class.

Verification commands used for this milestone:

```bash
PYTHONPATH=src /tmp/opencode/ml-venv/bin/python -m pytest -q
/tmp/opencode/typecheck-venv/bin/python -m ruff check .
/tmp/opencode/typecheck-venv/bin/python -m mypy src
make docker-config
```

## Kaggle parity run

Live parity needs backbone weights plus raw crops, so it runs on a GPU Kaggle runner
via `notebooks/03_kaggle_parity.ipynb` (no retraining). Upload three files from this
checkout as a Kaggle dataset first:

- `data/embeddings/mll23_embeddings.pt` (~129 MB)
- `outputs/checkpoints/mlp.pt` (~1.6 MB)
- `outputs/reports/mll23_head_comparison.json`

The notebook then downloads MLL23 and DinoBloom-B, restores the three files,
regenerates evaluation artifacts with `classifier evaluate-cache`, proves the live
path with `classifier parity-live --device cuda` and one real `classifier classify-crop`,
and finishes with `classifier validation-reports --parity-evidence ... --verification-report ...`.
The verification JSON is written only if `pytest`, `ruff`, and `mypy` all exit 0
on the runner, so a passing acceptance can never rest on an unmeasured claim.
Bring the resulting `parity_evidence.json` (and `parity_crop.json` if you want it)
back to refresh the local acceptance report.

The `data` command group remains as an alias of `dataset`. The `scripts/` wrappers
(`audit_dataset.py`, `build_manifest.py`, `download_assets.py`) only parse arguments
and call the `bloodfilm.data` package.

## Docker

```bash
make docker-config   # validate compose.yaml
make docker-build    # build app and test images
make docker-test     # run the full suite in a container
docker compose run --rm app smoke --config configs/base.yaml
```

`configs/`, `data/`, `models/`, and `outputs/` are mounted from the host. The image
bakes in `src/`, `configs/`, `scripts/`, packaging metadata, and (test stage only)
`tests/` — never datasets or weights. The `test` service bind-mounts the working
tree over the baked copy, and `dev` sets `PYTHONPATH=/app/src`, so both always
exercise live code rather than the install snapshot. Build with
`INSTALL_ML=1` (e.g. `INSTALL_ML=1 make docker-build`) to include the heavy ML
extras. GPU access is intentionally not configured yet; add a `deploy.resources`
reservation when classifier/detector training needs it.

`configs/*.yaml` are JSON-compatible YAML so the scaffold runs before optional ML/YAML dependencies are installed. Install the `ml` extra before model training milestones.

## MLL23 Acquisition Blocker

MLL23 is not bundled. Before running the manifest command, download the official record and preserve the exact files and terms used:

```bash
mkdir -p data/raw/MLL23 data/manifests
python3 - <<'PY'
from pathlib import Path
from urllib.request import urlopen

record_url = "https://zenodo.org/api/records/14277609"
Path("data/manifests/mll23_zenodo_record.json").write_bytes(urlopen(record_url, timeout=60).read())
print("Saved", record_url, "to data/manifests/mll23_zenodo_record.json")
print("Inspect file entries and license, then download the class archives into data/raw/MLL23 without renaming class folders.")
PY
```

Do not claim an independent split until official patient/source grouping metadata has been inspected and recorded. Do not report test-set classifier metrics until the dedicated final test evaluation has been run in a Torch-capable environment and the resulting report is present.

The committed `configs/mappings/mll23.yaml` fixes the canonical project class order, but its folder-label mappings are marked provisional until the downloaded release labels are inspected.
