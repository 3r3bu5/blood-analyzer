# WBC Blood Film Analyzer

Research proof-of-concept scaffold for the first M0 and M1 milestones in `WBC_Blood_Film_Analyzer_Technical_Design.md`.

Implemented now:

- repository skeleton under `src/bloodfilm`
- typed stdlib configuration objects loaded from JSON-compatible YAML files
- environment report command
- asset registry and DinoBloom-B smoke checks that fail clearly when weights are absent
- PNG image intake and quality measurements without silently substituting models
- MLL23 mapping, manifest, audit, checksum, and deterministic split workflow

Not implemented yet:

- classifier training or detector training
- real DinoBloom-B model loading without explicit local weights and ML dependencies
- clinical validation or performance claims

## Commands

```bash
PYTHONPATH=src python3 -m bloodfilm.cli environment --output outputs/reports/environment.json
PYTHONPATH=src python3 -m bloodfilm.cli smoke --config configs/base.yaml
PYTHONPATH=src python3 -m bloodfilm.cli data build-manifest --dataset-root data/raw/MLL23 --mapping configs/mappings/mll23.yaml --output data/manifests/mll23_manifest.csv --invalid-output data/manifests/mll23_invalid.csv --checksum-output data/manifests/mll23_checksums.csv --report-output outputs/reports/mll23_audit.json
PYTHONPATH=src python3 -m bloodfilm.cli data split --manifest data/manifests/mll23_manifest.csv --output data/manifests/mll23_splits.csv --report-output outputs/reports/mll23_leakage_report.json
```

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

Do not claim an independent split until official patient/source grouping metadata has been inspected and recorded.

The committed `configs/mappings/mll23.yaml` fixes the canonical project class order, but its folder-label mappings are marked provisional until the downloaded release labels are inspected.
