# Product Requirements Document: WBC Blood Film Analyzer

**Version:** 1.2  
**Date:** 26 September 2026  
**Status:** Draft for product and laboratory review  
**Owner:** Project team  
**Initial deliverable:** Research notebook and reusable inference modules  
**Intended use at this stage:** AI-assisted experimental analysis of peripheral blood smear images, subject to expert review

## 1. Executive summary

Build a reproducible proof of concept (POC) that takes microscope field images of peripheral blood smears, finds white blood cells (WBCs), crops each detected cell, predicts one of five common mature WBC types when supported by the model, and presents confidence, review flags, annotated images, and a provisional differential. The immediate deliverable is a Jupyter notebook backed by small reusable Python modules, not a clinical product.

The decisive test is performance on images from **the actual microscope and camera intended for use**. Public dataset metrics alone cannot establish that the pipeline works on those images. A qualified reviewer must inspect detections and classifications. The system must surface uncertainty and technical failures instead of presenting every image as a valid result.

The implementation follows two independent stages:

`Microscope field → quality check → WBC detector → cell crops → WBC classifier → review status → provisional differential and exports`

### 1.1 Product decisions in this PRD

| Decision | Requirement |
| --- | --- |
| Primary classifier taxonomy | MLL23 18 classes: mature, immature, lymphoid, myeloid, smudge-cell and normoblast morphologies |
| Detection task | One WBC class; ignore RBC and platelet predictions at the product boundary |
| Selected classifier backbone | **DinoBloom-B** (86M parameters, 768-dimensional embedding) plus a trained task-specific head |
| Initial head | Configurable 18-class linear, MLP, and cosine heads; select using held-out macro F1 and per-class results |
| Supporting classifiers | Keep `class_names` and `num_classes` configurable; retain five-class Raabin and 13-class WBCBench models as separate benchmarks |
| Detector candidate | Ultralytics YOLO for a research POC, behind a replaceable interface |
| Primary dataset candidates | MLL23 for expanded cell classification; TXL-PBC for field detection |
| Licensing posture | Personal, non-commercial research POC; research-only tools and datasets may be used with recorded terms |
| Source of truth for feasibility | Expert-labeled images from the intended microscope/camera |
| Default differential | Accepted or reviewed cells only; show denominator and excluded count |
| Output status | Experimental; no diagnosis or clinical release claim |

These are starting choices, not claims that weights have been loaded, models trained, or accuracy established.

## 2. Problem and opportunity

Manual review of blood films involves locating cells across microscope fields and identifying their type. The proposed workflow could help a trained reviewer organize cells, inspect model suggestions, and record a traceable provisional differential. It must handle blurred fields, atypical cells, staining and camera differences, and false or missed detections.

The key product risk is a plausible but wrong answer: a detector can miss cells entirely, and a five-class classifier can assign a high softmax probability to a cell outside its training classes. The POC therefore needs separate detector and classifier evaluation, an explicit review queue, and a way to record expert corrections.

## 3. Goals and non-goals

### 3.1 Goals

1. Demonstrate field-image → WBC box → crop → class suggestion on real microscope images.
2. Make every model output inspectable, with original field, boxes, crops, probabilities, and reasons for review.
3. Quantify detector, classifier, and end-to-end errors using independent, labeled test sets.
4. Export a stable per-cell record suitable for later use by a desktop application.
5. Preserve enough metadata to reproduce and audit experiments.
6. Determine whether local imaging differences require additional annotation or fine-tuning.

### 3.2 Non-goals for the POC

No autonomous diagnosis; no official laboratory result; no replacement for manual differential or a CBC analyzer; no patient database or identifying data requirement; no desktop UI, LIS integration, live camera control, motorized stage, or slide scanning. RBC morphology, platelet analysis, malaria, leukemia, immature/abnormal cell diagnosis, and WBC nucleus/cytoplasm segmentation are later research topics, not first-release features.

## 4. Users and jobs to be done

| User | Job | Needed outcome |
| --- | --- | --- |
| ML engineer | Run and compare reproducible experiments | Configs, dataset lineage, model checkpoints, metrics, error examples |
| Laboratory reviewer / hematology expert | Inspect WBC candidates and correct model suggestions | Field and crop views, review flags, corrections, exclusion reasons |
| Product engineer, later phase | Integrate inference into an application | Stable analyzer API and versioned CSV/JSON schemas |
| Project owner | Decide whether to invest in the next phase | Real-camera results, failure modes, effort and licensing assessment |

## 5. Assumptions and dependencies

- The team can obtain peripheral blood smear field images from the target microscope/camera and permission to use them for research. No such images are assumed to be available yet.
- A qualified reviewer can provide ground-truth boxes and cell labels on a representative local sample. Labeling capacity and sample size remain to be agreed.
- MLL23 contains 41,906 expert-labelled peripheral-blood single-cell images across 18 morphology classes and provides dataset analysis/harmonization resources [S7]. Exact version, provenance, grouping metadata and terms must be captured before ingestion.
- Raabin-WBC contains images of the five normal WBC types; it remains a simpler benchmark and cross-camera evaluation source. Exact version, provenance, terms, and any restrictions must be captured before ingestion [S1].
- TXL-PBC publishes YOLO-format WBC/RBC/platelet boxes and a repository-level MIT license. The team must separately inspect the rights and lineage of its constituent images [S2].
- DinoBloom publishes pretrained feature-extractor variants, including S and B, and its repository displays Apache-2.0. The selected DinoBloom-B backbone has an 86M-parameter ViT-B/14 and produces 768-dimensional embeddings. It does **not** supply this project's trained MLL23 18-class head or any supporting benchmark head [S3]. Verify the selected weight artifact and terms independently.
- WBCBench 2026 provides a current 13-class research benchmark with mature, immature, and abnormal WBC morphologies, patient-separated evaluation, severe imbalance, and simulated acquisition degradation. Its public page describes a research/non-commercial license; exact dataset terms must be reviewed before use in a commercial project [S5].
- The detector framework and its weights have separate licensing implications. Ultralytics describes an AGPL-3.0 or Enterprise path and explicitly includes its code and trained models in its guidance [S4]. Commercial use requires a deliberate license decision.
- Hardware, camera resolution, image format, acquisition settings, sample counts, and target turnaround time are not specified. Baselines and release targets will be set after collecting real data.

## 6. Product scope and priority

**P0** means required for the first working POC; **P1** means needed for a defensible evaluation; **P2** means useful after the main risks are measured.

| ID | Priority | Capability | Requirement and observable behavior |
| --- | --- | --- | --- |
| FR-01 | P0 | Image input | Read JPG, JPEG, PNG, TIFF/TIF from a configured local folder; report missing, corrupt, and unsupported files individually. |
| FR-02 | P0 | Quality check | Calculate dimensions, focus score, mean brightness, and exposure flags; show raw values and configurable experimental thresholds. |
| FR-03 | P0 | WBC detection | Load explicitly selected detector weights; return pixel-space `xyxy` boxes, scores, and model version for WBCs. |
| FR-04 | P0 | Crop generation | Expand each box by a configured percentage, clip to image boundaries, save lossless crops, and retain source coordinates. |
| FR-05 | P0 | WBC classification | Load DinoBloom-B plus a trained MLL23 18-class head; return top class and the full ordered probability vector. Never treat raw embeddings as class probabilities. |
| FR-06 | P0 | Review state | Label uncertain cells for review, expose low-quality or failed fields, and allow `accepted`, `corrected`, `rejected`, `unreviewed` status in data records. |
| FR-07 | P0 | Visualization | Display field boxes with cell IDs, original and cropped images, top class, score, and review warning. |
| FR-08 | P0 | Differential | Compute counts and percentages with an explicit denominator and inclusion rule; mark provisional results. |
| FR-09 | P0 | Exports | Save per-cell CSV, structured JSON, annotated fields, and crops with stable IDs and schema version. |
| FR-10 | P0 | Execution modes | Expose explicit `smoke`, `train`, and `inference` modes; missing weights or data produce actionable errors, never a silent model swap. |
| FR-11 | P1 | Detector evaluation | Evaluate held-out WBC recall, precision, false positives per field, matching IoU, and missed-cell examples. |
| FR-12 | P1 | Classifier evaluation | Report accuracy, balanced accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrices, confidence and calibration. |
| FR-13 | P1 | Real-image evaluation | Compare public and local imaging distributions; measure expert-labeled local detector and classifier results separately and end to end. |
| FR-14 | P1 | Experiment tracking | Save config, seed, package versions, data manifest, git commit when available, training curves, metrics, checkpoint IDs and timings. |
| FR-15 | P1 | Correction capture | Keep a separate reviewed label and review note without overwriting the original prediction. |
| FR-16 | P2 | Model comparison | Compare DinoBloom-B linear, MLP, and cosine heads; use DinoBloom-S and ConvNeXt V2 as resource/performance baselines on identical splits and metrics. |
| FR-17 | P2 | Explainability | Optionally generate a suitable attention/attribution view labeled experimental and non-diagnostic. |
| FR-18 | P2 | Duplicate-cell analysis | Identify potential overlaps across adjacent fields if slide coordinates become available; do not claim reliable duplicate suppression in POC. |
| FR-19 | P2 | Supporting taxonomies | Support separate configurable five-class Raabin and 13-class WBCBench checkpoints; preserve each mapping and never combine incompatible labels silently. |

## 7. Primary workflows

### 7.1 Engineer: first run

1. Install pinned dependencies in a clean environment and open `notebooks/01_wbc_poc.ipynb`.
2. Configure dataset and weights paths and choose `smoke` mode.
3. Validate image loading and quality computations on one field.
4. Load the chosen detector and classifier weights. Show checkpoint IDs and class mapping.
5. Run on an actual microscope field; inspect box overlay, crop gallery, warnings, and exports.
6. If any required artifact is missing, stop the dependent step with a precise remedy.

The smoke mode may use fixture data to test plumbing, but it may not create synthetic performance claims or present an untrained head as a working classifier.

### 7.2 Reviewer: inspect a field

1. Open the field overlay and each numbered crop.
2. Inspect the predicted type and full probabilities alongside the field context.
3. Confirm, correct, or reject each cell; optionally add a note such as artifact, damaged cell, or out-of-scope morphology.
4. Inspect any cells missed by the detector during a labeled evaluation. The review UI for the POC can be a notebook table/manual annotation file; a production GUI is out of scope.
5. Recalculate the accepted/reviewed differential and retain the original provisional output.

### 7.3 Batch inference

Given `data/microscope_samples/`, process supported files in deterministic filename order, continue after a single bad file, record per-file status, and write one consolidated result set. If zero WBCs are detected, report `zero_detections` and do not infer a zero differential.

## 8. Functional details and business rules

### 8.1 Image intake and acquisition metadata

- Accept local paths and preserve a source ID and checksum. File names alone are not a patient identity or reliable unique key.
- Record optional `microscope_model`, `camera_model`, objective, magnification, pixel size in µm, stain, exposure, gain, white balance, slide ID, field position, acquisition session, and capture timestamp.
- Restrict identifiable patient metadata in the POC. If local images contain identifiers in labels or EXIF, remove or control access before sharing or export.
- Detect an unsupported color depth/encoding explicitly; retain original image for traceability and note any conversion used for model input.

### 8.2 Quality and preprocessing

- Compute focus via variance of the grayscale Laplacian and report brightness/exposure measures. Thresholds are configurable engineering heuristics, calibrated against local images, and are not clinical cutoffs.
- Quality outcome is `acceptable`, `review_quality`, or `unusable` with reasons. `unusable` suppresses automatic classification by default but retains an audit record; override requires an explicit configuration and visible warning.
- Default color handling is **none** beyond the normalization required by the selected model. Optional white balance/stain experiments have independent configuration and logged results.
- Use consistent RGB conversion, resizing, and model-specific input normalization. Padding around detector boxes defaults to a configurable percentage and is clipped safely.

### 8.3 Detection

- Detector interface: `detect(image_rgb) -> list[Detection]`. Each `Detection` has pixel coordinates, confidence, and optional diagnostic metadata.
- Reject NaN, reversed, zero-area, and out-of-image boxes. Record their count. Use deterministic NMS/postprocessing settings where the framework requires them.
- A model merely pretrained on generic objects does not satisfy WBC detection. The chosen weights must be trained or demonstrably suitable for WBC boxes; the notebook must identify the training source.
- A low detector score is not proof that no WBC exists. The field-level status must make zero detections and possible misses visible.

### 8.4 Classification and uncertainty

- The primary class order contains the 18 MLL23 labels and is stored in the checkpoint. It includes basophil, eosinophil, band and segmented neutrophil, monocyte, typical/reactive/large-granular/other-neoplastic lymphocyte, hairy cell, plasma cell, smudge cell, myeloblast, promyelocyte, atypical promyelocyte, myelocyte, metamyelocyte and normoblast.
- The classifier configuration owns `class_names` and derives `num_classes`; the analyzer must not hard-code the output dimension.
- DinoBloom-B is a pretrained feature extractor. A new 18-class head starts with untrained parameters and must be trained once on labelled cell crops. Routine analyzer inference loads the saved checkpoint and does not retrain it.
- The five probabilities are generated by the trained head and sum approximately to one. Record raw logits or probability calibration method if used.
- `confidence` means the highest model probability, not the probability the diagnosis is correct. Thresholds are workflow settings to tune on held-out data; example exploratory values are high ≥0.90, review 0.70–<0.90, mandatory review <0.70.
- A high 18-class probability does **not** prove that a cell belongs to the trained taxonomy. Mott cells, Sézary cells, macrophages, erythrophagocytes and other unsupported morphologies require a way to mark `rare_morphology`, `unclassified`, or `out_of_scope`. Thresholding alone does not solve open-set recognition.
- Classifier failures should preserve the detector box/crop and set `classification_failed` instead of dropping the cell.

### 8.5 Differential and denominator

- Compute a **provisional predicted differential** only for eligible predictions and a **reviewed differential** only from accepted/corrected expert labels. Do not use the same label for both.
- Default eligible set for provisional summaries: cells with a valid predicted label, not rejected, not marked out of scope, and not requiring mandatory review. The system produces a detailed morphology summary and a separately configured grouped research differential.
- Reviewed denominator: number of cells explicitly accepted or corrected into one of the five classes. Exclude unreviewed, rejected, artifact, and out-of-scope cells.
- Display `detected_count`, `classified_count`, `review_required_count`, `reviewed_count`, `included_count`, and `excluded_count`, plus denominator and percentages. If denominator is zero, percentages are `null` and status is `insufficient_data`.
- A differential from very few cells is descriptive only; the POC will not assert that a count of four, or any unvalidated count, is sufficient for clinical interpretation.
- When fields overlap, potential repeat cells are flagged as a limitation; do not combine fields into a slide-level differential as though duplicates were resolved.

### 8.6 Outputs and schema

Minimum CSV columns: `schema_version`, `run_id`, `source_image`, `cell_id`, `x1`, `y1`, `x2`, `y2`, `detector_confidence`, `predicted_class`, `classification_confidence`, one probability for each of the five classes, `review_required`, `review_status`, `reviewed_class`, `review_note`, `crop_path`, `quality_status`, `detector_version`, `classifier_version`.

JSON records add quality measurements, acquisition metadata, preprocessing configuration, per-field errors, summary denominator rule, and source/checkpoint hashes. Every detected candidate has a stable `cell_id`, derived from a run and source ID plus detection index; changing box coordinates alone must not silently overwrite previous review work.

Example directory layout:

```text
blood-film-poc/
  notebooks/01_wbc_poc.ipynb
  src/imaging/  src/detection/  src/classification/
  src/pipeline/ src/evaluation/ src/visualization/
  data/classification/ data/detection/ data/microscope_samples/
  models/detector/ models/classifier/ models/metadata/
  outputs/annotated/ outputs/crops/ outputs/reports/ outputs/experiments/
  requirements.txt  README.md  config.example.yaml
```

## 9. Data requirements and labeling

| Data | Role | Minimum required metadata | Key validation |
| --- | --- | --- | --- |
| MLL23 | Primary 18-class crop training/evaluation | Class, patient/source group, source, license, checksum | Exact 18-class mapping; class imbalance; duplicates and group leakage |
| Raabin-WBC | Five-class benchmark and cross-camera evaluation | Class, original image/slide/patient group if available, source, license, checksum | Label mapping; class counts; duplicates and group leakage |
| TXL-PBC or equivalent | Field detector training/evaluation | Image, YOLO boxes, source dataset, split, license, checksum | WBC class mapping, box bounds, source overlap across splits |
| Actual microscope fields | Feasibility and local holdout | Camera/session/slide grouping, capture settings when possible, WBC boxes and expert cell labels for evaluation subset | Representative focus/stain/field variation; independent expert review |
| WBCBench 2026 | Later 13-class morphology experiment | Patient group, original 13-class label, degradation level, source, license, checksum | Preserve patient split; do not merge incompatible labels or use test data for training |

The data loader must validate class names and file pairing, report class distribution and invalid examples, and write a data manifest. Split by patient when possible, otherwise slide, acquisition session, or original source field. Preserve any official split only after checking for source overlap. If no grouping identifiers exist, state that leakage risk remains; do not claim an independent generalization estimate from a random crop split.

### 9.1 Primary 18-class head data plan

Download the MLL23 class archives from the official Zenodo record and use its repository files to audit labels and harmonization. Build an immutable image manifest containing source folder, canonical class, checksum, dimensions and available patient/source grouping. Create group-aware train, validation and frozen test manifests. Report support for every class; reactive lymphocytes are particularly rare and must remain visible in selection metrics.

The primary checkpoint uses the exact ordered 18-class list recorded in `class_names.json`. Cache DinoBloom-B embeddings during head-only experiments so linear, MLP and cosine heads can be compared without repeatedly running the frozen backbone.

### 9.2 Five-class benchmark data plan

Download the Raabin-WBC `Train`, `TestA`, and `TestB` portions from the dataset publisher or its official Kaggle account. Use `Train` to fit the head and derive a validation subset without crossing any available patient, slide, or acquisition grouping. Keep both test portions untouched: `TestA` measures the closer acquisition domain, while `TestB` was designed to test generalization across a different microscope/camera. Record the exact downloaded release because prepared Raabin versions can contain different image counts.

The five-class directory mapping is `Basophil`, `Eosinophil`, `Lymphocyte`, `Monocyte`, and `Neutrophil`. Do not infer a finer neutrophil maturation label from the generic Raabin label.

### 9.3 WBCBench 13-class data plan

Treat the 13-class model as a separate checkpoint and experiment. The WBCBench taxonomy contains segmented and band neutrophils, eosinophils, basophils, lymphocytes, variant lymphocytes, monocytes, myelocytes, metamyelocytes, promyelocytes, prolymphocytes, plasma cells, and blasts. Preserve the official patient-separated protocol and use macro F1 because rare classes can disappear behind high overall accuracy.

Do not combine Raabin and WBCBench by folder name alone. Maintain a label-mapping manifest with `source`, `patient_or_group_id`, `original_label`, and `target_label`. Only map labels when their definitions are equivalent. In particular, a generic Raabin neutrophil label cannot safely be divided into band and segmented neutrophils.

Local data plan: collect fields spanning routine and difficult imaging conditions; keep a frozen local test set disjoint by slide/session from training and tuning; have an expert annotate all visible WBCs in the evaluation fields, including cells that the detector missed. Use an agreed adjudication process for ambiguous labels. Sample-size targets and label quality rules are pending consultation with the laboratory reviewer. Record how uncertain, abnormal, damaged, and artifact candidates are labeled even though the first classifier has only five classes.

## 10. Model training and evaluation

### 10.1 Classifier

Start with frozen DinoBloom-B embeddings and an 18-class linear head. Evaluate an 18-class MLP head and cosine head on the same split. Cache the 768-dimensional embeddings to make head experiments fast. Optionally unfreeze the final backbone blocks, then consider full fine-tuning at a lower learning rate only if head-only results and local domain shift justify the cost. Document input size, normalization, augmentations, loss, class weighting/sampling choice, optimizer, seed, early stopping, and checkpoint selection. Never oversample the test set.

Evaluate on the frozen MLL23 holdout plus compatible-class external datasets and an expert-labelled local crop holdout. Report per-class support, all metrics listed in FR-12, normalized and raw confusion matrices, high-confidence errors, and performance by camera/stain/session where sample size allows. Rare-class support must be explicit. Run Raabin TestA/TestB only through the separately trained five-class benchmark or a documented compatible mapping. Benchmark DinoBloom-S and ConvNeXt V2 only after the DinoBloom-B baseline, using the same protocol.

Recent WBCBench 2026 results show that the highest-scoring methods are ensembles rather than a universally superior single model. The first-ranked entry combined DinoBloom-B with two DINOv3 backbones and specialized rare-class logic; the second used a Swin-Small/MedSigLIP ensemble plus restoration and heuristics [S5]. A separate DinoBloom-B study found complementary behavior across linear, MLP, and cosine heads [S6]. These findings justify the selected head comparisons but do not establish performance on this project's five classes or microscope.

### 10.2 Detector

Train or fine-tune a WBC-only detector from authorized field images and evaluate on separate fields. Report recall and precision at documented score/IoU operating points, false positives per field, missed WBC gallery, and crop integrity. Metrics on TXL-PBC and on target-camera images are separate. Record detector latency per field and image resolution.

### 10.3 End-to-end performance

On local annotated fields, measure detected-and-correctly-classified cells relative to **all expert-annotated WBCs**, plus per-cell review coverage. A strong crop classifier cannot compensate for missed detections. Compare classifier accuracy on ground-truth crops with accuracy on detector-generated crops to expose crop and detection effects. Report errors by quality flag, stain, capture session, and magnification when possible.

### 10.4 Calibration and thresholds

Plot confidence distributions and reliability/calibration metrics on held-out data. Choose review thresholds based on the cost of confidently wrong predictions and the reviewer workload; record the chosen threshold version. If out-of-scope examples are available, evaluate how many receive high-confidence normal labels. Do not claim unknown-cell rejection has been solved unless that test supports it.

## 11. Nonfunctional requirements

| ID | Requirement |
| --- | --- |
| NFR-01 Reproducibility | Pin compatible dependency versions and checkpoint hashes; record seed, code revision, dataset manifest, and configuration for each experiment. |
| NFR-02 Portability | CPU execution must work for a small inference set; use CUDA when available; missing GPU is visible, not fatal. Mixed precision is optional on supported CUDA hardware. |
| NFR-03 Reliability | One corrupt image does not abort the batch; field errors and partial-cell failures appear in a run report. |
| NFR-04 Performance | Measure detector/field, classifier/crop, full-pipeline/field, peak memory, and device details. Agree acceptable targets only after target-hardware baselines. |
| NFR-05 Security | Default local processing; no upload of microscope images or identifiers to a cloud service without explicit configuration and authorization. Restrict raw image and output access. |
| NFR-06 Provenance | Record every dataset, code, weight, and dependency license; decide the distribution strategy before shipping a proprietary application. |
| NFR-07 Auditability | Keep raw images, original predictions, model versions, manual changes, and export timestamps separately; never overwrite an expert correction with a rerun. |
| NFR-08 Usability | Label every generated result as experimental, distinguish provisional from reviewed, and show why a field/cell needs review. |

## 12. Phased implementation and acceptance gates

| Milestone | Deliverable | Exit gate |
| --- | --- | --- |
| M0 — Inventory | Confirm hardware, sample formats, data rights, available weights, reviewer, and dataset manifests | Dependencies, gaps, and planned evaluation set recorded; no unverified model assumed to work |
| M1 — Intake and quality | Notebook skeleton, config, environment, image loading and quality metrics | One valid field renders; bad/missing image and low-quality paths are shown correctly |
| M2 — Detection | Detector adapter, weights loading, WBC boxes, crops and overlay | On target-camera fields, actual boxes and saved crops are inspectable; no claim of measured recall until expert boxes exist |
| M3 — Classification | DinoBloom-B embedding extraction, trained 18-class linear/MLP/cosine heads, checkpoint loading, crop inference and evaluation | Correct MLL23 class order; measured frozen-holdout and local metrics where labels exist; probabilities and failure paths shown |
| M4 — Integration | End-to-end analysis, uncertainty, provisional differential, batch mode and exports | Target-camera folder runs with real trained components; outputs reconcile to displayed cells; failure cases persist |
| M5 — Local validation | Frozen, expert-labeled local evaluation and domain-shift report | Detector, crop classifier, and end-to-end results measured separately; high-confidence errors reviewed; next-phase decision recorded |
| M6 — Refinement | Fine-tuning, thresholds, model benchmarks, optional attribution | Changes compared on the same frozen holdout; any regression and review workload recorded |

M2 and M3 may be developed independently, but M4 must not present classification results before the 18-class head has actually been trained and validated. Do not defer target-camera testing until a desktop application exists.

### 12.1 POC acceptance checklist

The POC passes its **engineering acceptance** when all of the following are demonstrably true:

1. The user puts supported microscope images in `data/microscope_samples/` and runs documented inference commands/cells.
2. The run identifies exactly which detector and classifier checkpoint was loaded and fails clearly when either is absent.
3. The notebook shows original fields, WBC boxes, saved crops, predicted types, full probabilities, and uncertain/review flags.
4. `outputs/results.csv`, `outputs/results.json`, `outputs/annotated/`, and `outputs/crops/` correspond to the same run; counts and cell IDs reconcile.
5. Zero-detection, bad image, invalid box, classifier error, and unusable-quality cases produce explicit statuses.
6. The differential shows its denominator and exclusions; zero eligible cells have no fabricated percentages.
7. A qualified reviewer can record corrections without erasing raw predictions.
8. The report states real measured detector/classifier/end-to-end metrics **only where labeled test data exist** and calls out absent data otherwise.
9. All outputs are labeled experimental and requiring expert review.

Engineering acceptance does not imply clinical suitability. A separate future decision would require intended-use definition, prospective validation, quality system, regulatory analysis, and clinical stakeholder sign-off.

## 13. Success measures and decision gates

No numerical accuracy, recall, latency, or clinical acceptance target is asserted before seeing the target hardware and a representative expert-labeled set. During M0, agree on a test protocol and decision thresholds with the reviewer. At M5, present:

- WBC detector recall, precision, and false positives per field on public and local holdouts, with counts and uncertainty intervals where appropriate.
- Five-class per-class recall/F1, macro F1, high-confidence errors, and calibration on ground-truth and detected crops.
- End-to-end fraction of expert-annotated WBCs both found and correctly classified; reviewer correction rate and proportion flagged.
- Number and reasons for unusable fields, missed cells, out-of-scope cells, and failed cases.
- Per-field processing time and memory on the intended CPU/GPU.
- Representative side-by-side public versus local performance and image characteristics.

**Go to the next phase** only if the team can trace every result, reproduce the run, review failures, and state a concrete data/model plan to address the observed local errors. If local evaluation is not available, the outcome is “pipeline demonstrated, feasibility unproven,” not “model validated.”

## 14. Risks and mitigations

| Risk | Impact | POC mitigation |
| --- | --- | --- |
| Detector misses WBCs | Differential is biased without visible warning | Expert-boxed local fields, recall evaluation, missed-cell gallery, explicit zero-detection status |
| Domain shift from public datasets | Optimistic benchmark and poor local predictions | Frozen local holdout, metadata stratification, image/metric comparisons, local fine-tuning plan |
| DinoBloom-B compute cost | Slower training/inference or GPU memory failure | Frozen cached embeddings first; mixed precision where supported; retain DinoBloom-S as the lightweight benchmark |
| Rare basophils and imbalance | Hidden failure behind overall accuracy | Per-class support/recall/F1, macro metrics, reviewed sampling and class-weight experiments |
| Open-set cells receive a normal class | Misleading high-confidence suggestion | Out-of-scope annotation, reviewer workflow, high-confidence error audit; no claim that confidence threshold alone solves it |
| Leakage and near-duplicates | Inflated test metrics | Group-aware splits, duplicate checks, source lineage, frozen holdouts |
| Overlapping fields | Double-counted cells | Preserve field IDs and positions; state limitation until coordinates or deduplication are evaluated |
| Licensing incompatibility | Blocks commercial deployment | Inventory code, weight, dataset, and derived-model terms before product choice |
| Sensitive image data | Unauthorized disclosure | Local default, de-identification, access controls, approved retention rules |
| False clinical interpretation | Harmful reliance on experimental result | Prominent scope label, reviewer sign-off, no patient report or diagnostic language |

## 15. Open decisions for the project owner and reviewer

These do not block writing the PRD, but affect implementation and validation:

1. What microscope/camera, magnification, objective, stain, resolution, and capture workflow will be used?
2. Can the team share a small, de-identified, representative set of full fields and obtain rights to use them for evaluation?
3. Who will annotate WBC boxes and cell types, adjudicate ambiguous cells, and set local sample targets?
4. Which machine(s) and GPU/CPU will be used for training and inference? Is offline operation required?
5. Is the near-term objective research only, or is a proprietary commercial device planned? This controls detector licensing and future validation work.
6. What are the laboratory's review policy, acceptable missed-cell risk, desired review workload, and required cell-count protocol? These determine operating thresholds and whether a differential is displayable.

## 16. Source register and verification notes

The source register describes candidate dependencies; it does not certify clinical use or legal compliance. Verified on 26 September 2026.

- **[S1] Raabin Health Data**, dataset publisher: https://raabindata.com/ — states availability of >40,000 images of five normal WBC types and describes free access for commercial and non-commercial projects. Check the actual download terms and version.
- **[S2] TXL-PBC repository**, dataset creators: https://github.com/lugan113/TXL-PBC_Dataset — README states 1,260 images, 18,143 boxes, YOLO-format WBC/RBC/platelet labels, and MIT repository license. The related paper/preprint describes a different historical split/count, so record the exact downloaded revision and validate the files rather than adopting a number from a paper.
- **[S3] DinoBloom repository**, model creators: https://github.com/marrlab/DinoBloom — lists DinoBloom-S/B/L/G variants, feature dimensions and parameter counts, download locations, and an Apache-2.0 repository license; inspect the selected DinoBloom-B weight artifact before use.
- **[S4] Ultralytics licensing**, vendor: https://www.ultralytics.com/license — describes AGPL-3.0 and Enterprise options, with specific guidance for code and trained models. Obtain professional review for a proposed distribution model.
- **[S5] WBCBench 2026 official challenge and overview paper:** https://xudong-ma.github.io/WBCBench2026-Robust-White-Blood-Cell-Classification/ and https://arxiv.org/abs/2604.10797 — document 55,012 crops from 493 patients, 13 classes, patient-separated evaluation, simulated domain shift, the final leaderboard, and research-oriented dataset terms.
- **[S6] Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling:** https://arxiv.org/abs/2603.20383 and https://github.com/Antony-gitau/msfit — publishes a DinoBloom-B training implementation and comparisons of linear, MLP, and cosine heads for the 13-class WBCBench task.
- **[S7] MLL23 dataset and resources:** https://zenodo.org/records/14277609 and https://github.com/marrlab/MLL23 — publish 41,906 expert-labelled peripheral-blood single-cell images across 18 morphology classes, label counts, an analysis notebook and multi-dataset harmonization files.

## 17. Handoff instructions for implementation

Keep this PRD as the scope and acceptance source. Implement one milestone at a time and include: files changed, model and dataset versions actually used, commands executed, real outputs/metrics, failure cases, unresolved risks, and the next gate. Keep the reusable analyzer independent of Jupyter and expose a conceptual interface such as `BloodFilmAnalyzer(detector, classifier).analyze(image)`.

**First implementation task:** M0–M1. Inspect the execution environment and available local data, build the project skeleton, make one microscope field load and render, compute configurable image-quality measurements, and stop with a clear request for actual microscope samples if none exist. Do not represent an untrained classifier or generic pretrained detector as the end-to-end POC.
