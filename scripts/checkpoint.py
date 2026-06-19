from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scripts.data import LabelMaps


CHECKPOINT_VERSION = 1
REQUIRED_KEYS = {
    "format_version",
    "model_state",
    "model_config",
    "diffusion_config",
    "label_maps",
    "seed",
    "training",
}


def label_maps_to_dict(label_maps: LabelMaps) -> dict[str, object]:
    return {
        "animal_to_id": dict(label_maps.animal_to_id),
        "object_to_id": dict(label_maps.object_to_id),
        "id_to_animal": list(label_maps.id_to_animal),
        "id_to_object": list(label_maps.id_to_object),
    }


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing keys: {', '.join(missing)}")
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {payload['format_version']}")
    maps = payload["label_maps"]
    for key in ("animal_to_id", "object_to_id", "id_to_animal", "id_to_object"):
        if key not in maps:
            raise ValueError(f"checkpoint label_maps missing {key}")


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    validate_checkpoint_payload(payload)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    validate_checkpoint_payload(payload)
    return payload
