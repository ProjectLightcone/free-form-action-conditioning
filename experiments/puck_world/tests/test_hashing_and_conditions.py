from __future__ import annotations

import json
from hashlib import sha256

import yaml

from tepa_eval.conditions import HOLDOUT_NATURAL_LANGUAGE_TEMPLATES, TRAIN_NATURAL_LANGUAGE_TEMPLATES, render_conditions
from tepa_eval.hashing import event_hash, scene_hash
from tepa_eval.schemas import InterventionEvent, PuckSpec, SceneSpec


def test_scene_hash_ignores_generated_ids_and_seed() -> None:
    pucks = [PuckSpec(object_id=0, color="blue", x=0.4, y=0.5, radius=0.06)]
    first = SceneSpec(scene_id=1, seed=10, image_size=64, horizon=40, pucks=pucks, friction=0.98, restitution=0.9)
    second = SceneSpec(scene_id=99, seed=999, image_size=64, horizon=40, pucks=pucks, friction=0.98, restitution=0.9)
    assert scene_hash(first) == scene_hash(second)


def test_event_hash_ignores_generated_ids_and_aliases() -> None:
    event = InterventionEvent(
        event_id=1,
        scene_id=1,
        target_object_id=0,
        target_object_aliases=["blue puck"],
        impulse=(1.0, -0.5),
        horizon=40,
        magnitude_bucket="small",
        direction_bucket="up-right",
    )
    same_semantics = event.model_copy(update={"event_id": 200, "target_object_aliases": ["puck zero"]})
    assert event_hash(event, "scene-a") == event_hash(same_semantics, "scene-a")


def test_condition_renderers_create_parseable_payloads() -> None:
    event = InterventionEvent(
        event_id=0,
        scene_id=0,
        target_object_id=0,
        target_object_aliases=["blue puck", "blue"],
        impulse=(-1.25, 0.75),
        horizon=40,
        magnitude_bucket="small",
        direction_bucket="down-left",
    )
    rows = render_conditions(event, "event-a", 0, ["natural_language", "json", "yaml", "key_value"], 4)
    assert len({row.rendering_hash for row in rows}) == 4
    json.loads(next(row.payload_inline for row in rows if row.family == "json"))
    yaml.safe_load(next(row.payload_inline for row in rows if row.family == "yaml"))
    key_value = next(row.payload_inline for row in rows if row.family == "key_value")
    assert "object:" in key_value
    assert "horizon_frames:" in key_value


def test_natural_language_templates_cover_train_and_holdout_sets() -> None:
    train_ids = set()
    holdout_ids = set()
    for index in range(512):
        event = _event(index)
        identity = sha256(f"event-{index}".encode("utf-8")).hexdigest()
        row = render_conditions(event, identity, 0, ["natural_language"], 1, allow_holdout_templates=True)[0]
        if index % 4 == 3:
            holdout_ids.add(row.template_id)
        else:
            train_ids.add(row.template_id)

    assert train_ids == {template_id for template_id, _ in TRAIN_NATURAL_LANGUAGE_TEMPLATES}
    assert holdout_ids == {template_id for template_id, _ in HOLDOUT_NATURAL_LANGUAGE_TEMPLATES}

    val_ids = set()
    for index in range(128):
        event = _event(index)
        identity = sha256(f"val-event-{index}".encode("utf-8")).hexdigest()
        row = render_conditions(event, identity, 0, ["natural_language"], 1, allow_holdout_templates=False)[0]
        val_ids.add(row.template_id)

    assert val_ids <= {template_id for template_id, _ in TRAIN_NATURAL_LANGUAGE_TEMPLATES}


def _event(event_id: int) -> InterventionEvent:
    return InterventionEvent(
        event_id=event_id,
        scene_id=0,
        target_object_id=event_id % 2,
        target_object_aliases=["blue puck", "object 0"],
        impulse=(-1.25, 0.75),
        horizon=40,
        magnitude_bucket="small",
        direction_bucket="down-left",
    )
