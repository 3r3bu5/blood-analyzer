# txl-pbc-yolo26n-v0.1

Research-only WBC candidate detector bundle. This artifact is not a diagnostic device.

## Metrics

- `dataset`: `TXL-PBC`
- `false_positives_per_image`: `0.001`
- `image_count`: `126`
- `inference_ms_per_image`: `6.7`
- `mAP50`: `0.995`
- `mAP50_95`: `0.886`
- `missed_wbc_count`: `0`
- `missed_wbc_rate`: `0.0`
- `model`: `YOLO26n`
- `postprocess_ms_per_image`: `3.0`
- `precision`: `0.999`
- `preprocess_ms_per_image`: `2.8`
- `recall`: `1.0`
- `split`: `test`
- `wbc_instance_count`: `131`
- `weights`: `runs/detect/outputs/detector/txl-pbc-yolo26n/weights/best.pt`

## Thresholds

- `candidates`: `[{'confidence_threshold': 0.25, 'false_positives_per_image': 0.001, 'precision': 0.999, 'recall': 1.0}]`
- `max_false_positives_per_image`: `2.0`
- `notes`: `['Threshold sweep not yet run; selected configurable default 0.25 after strong test recall.', 'Run prediction-level threshold sweep before clinical or operational use.']`
- `primary_objective`: `wbc_recall`
- `selected`: `{'confidence_threshold': 0.25, 'false_positives_per_image': 0.001, 'iou_threshold': 0.5, 'precision': 0.999, 'recall': 1.0, 'selection_reason': 'default_threshold_meets_high_recall_on_test_summary'}`
- `target_recall`: `0.95`

## Known limitations

- Research-only detector; not a diagnostic device.
- TXL-PBC detects broad WBC candidates and does not establish 18-class morphology.
- Local microscope-field validation is still required before operational use.
