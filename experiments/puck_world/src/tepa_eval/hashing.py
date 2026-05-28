from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from tepa_eval.schemas import ConditionRendering, InterventionEvent, SceneSpec


def stable_hash(value: Any) -> str:
    payload = json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def scene_hash(scene: SceneSpec) -> str:
    data = scene.model_dump()
    for key in ("scene_id", "scene_hash", "seed"):
        data.pop(key, None)
    data["pucks"] = sorted(data["pucks"], key=lambda puck: puck["object_id"])
    data["obstacles"] = sorted(data["obstacles"], key=lambda obstacle: obstacle["obstacle_id"])
    data["goal_zones"] = sorted(data["goal_zones"], key=lambda goal: goal["goal_id"])
    return stable_hash({"kind": "scene", "value": data})


def event_hash(event: InterventionEvent, scene_identity: str) -> str:
    data = event.model_dump()
    for key in ("event_id", "event_hash", "scene_id", "target_object_aliases"):
        data.pop(key, None)
    data["scene_hash"] = scene_identity
    return stable_hash({"kind": "event", "value": data})


def outcome_hash(event_identity: str, simulator_version: str, target_config_hash: str) -> str:
    return stable_hash(
        {
            "kind": "outcome",
            "event_hash": event_identity,
            "simulator_version": simulator_version,
            "target_config_hash": target_config_hash,
        }
    )


def rendering_hash(rendering: ConditionRendering, event_identity: str) -> str:
    data = rendering.model_dump()
    for key in ("condition_id", "event_id", "rendering_hash", "metadata"):
        data.pop(key, None)
    data["event_hash"] = event_identity
    return stable_hash({"kind": "rendering", "value": data})


def config_hash(value: Any) -> str:
    return stable_hash({"kind": "config", "value": value})


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _canonicalize(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value
