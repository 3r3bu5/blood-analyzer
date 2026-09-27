from __future__ import annotations

from pathlib import Path
from typing import Any

from bloodfilm.classification.bundle import load_bundle_documents, verify_backbone_checksum
from bloodfilm.classification.dinobloom import (
    extract_backbone_embedding,
    load_dinobloom_b_backbone,
    preprocess_crop,
)
from bloodfilm.classification.heads import build_head
from bloodfilm.classification.uncertainty import UncertaintyPolicy, decide_prediction
from bloodfilm.errors import ConfigError, ModelLoadError
from bloodfilm.ml import require_torch


def classify_probabilities(
    *,
    probabilities: list[float],
    class_names: list[str],
    policy: UncertaintyPolicy,
    image_path: Path | str,
    bundle_name: str,
    top_k: int = 5,
) -> dict[str, object]:
    decision = decide_prediction(probabilities, class_names, policy)
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    top_predictions = [
        {
            "class_index": index,
            "class_code": class_names[index],
            "probability": probability,
        }
        for index, probability in ranked[:top_k]
    ]
    return {
        "research_only": True,
        "bundle": bundle_name,
        "image": str(image_path),
        "decision": decision.to_dict(),
        "top_predictions": top_predictions,
        "probabilities": [
            {"class_index": index, "class_code": name, "probability": probabilities[index]}
            for index, name in enumerate(class_names)
        ],
    }


def classify_crop(
    *,
    bundle_dir: Path | str,
    image_path: Path | str,
    backbone_weights: Path | str,
    device: str = "cpu",
) -> dict[str, object]:
    documents = load_bundle_documents(bundle_dir)
    bundle = documents["bundle"]
    taxonomy = documents["taxonomy"]
    class_names = _class_names(taxonomy)
    temperature = float(documents["calibration"].get("temperature", 1.0))
    policy = UncertaintyPolicy(**documents["thresholds"])

    torch = require_torch("Classifier inference", ModelLoadError)
    verify_backbone_checksum(bundle, backbone_weights)
    model = load_dinobloom_b_backbone(backbone_weights, device=device)
    head = build_head(str(bundle["head"]), len(class_names))
    checkpoint = torch.load(
        Path(bundle_dir) / str(bundle["head_file"]), map_location="cpu", weights_only=True
    )
    state = (
        checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )
    if isinstance(checkpoint, dict):
        checkpoint_classes = [str(name) for name in checkpoint.get("class_names", [])]
        if checkpoint.get("head") != bundle["head"]:
            raise ModelLoadError("Classifier checkpoint head does not match bundle metadata")
        if checkpoint_classes != class_names:
            raise ModelLoadError("Classifier checkpoint class order does not match bundle taxonomy")
    try:
        head.load_state_dict(state)
    except Exception as exc:
        raise ModelLoadError(f"Classifier head state is not loadable: {bundle_dir}") from exc
    head.eval()
    head = head.to(device)

    batch = preprocess_crop(image_path)
    if device != "cpu":
        batch = batch.to(device)
    embedding = extract_backbone_embedding(model, batch)
    with torch.no_grad():
        logits = head(embedding) / temperature
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    return classify_probabilities(
        probabilities=[float(value) for value in probabilities],
        class_names=class_names,
        policy=policy,
        image_path=image_path,
        bundle_name=str(bundle.get("name", Path(bundle_dir).name)),
    )


def _class_names(taxonomy: dict[str, Any]) -> list[str]:
    value = taxonomy.get("class_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError("Bundle taxonomy.json must contain class_names")
    if not value:
        raise ConfigError("Bundle taxonomy must contain at least one class")
    return list(value)
