from __future__ import annotations

from bloodfilm.data.manifests import ManifestBuildResult


def manifest_audit_report(result: ManifestBuildResult) -> dict[str, object]:
    invalid_by_reason: dict[str, int] = {}
    for row in result.invalid_rows:
        invalid_by_reason[row.reason] = invalid_by_reason.get(row.reason, 0) + 1
    return {
        "valid_image_count": len(result.valid_rows),
        "invalid_item_count": len(result.invalid_rows),
        "class_distribution": result.class_distribution,
        "invalid_by_reason": dict(sorted(invalid_by_reason.items())),
        "grouping_status": "unverified_image_level_surrogate",
        "independence_claim": False,
        "leakage_note": (
            "M0/M1 scaffold assigns one surrogate group per image until official MLL23 grouping "
            "metadata is available. Random image-level independence is not claimed."
        ),
    }
