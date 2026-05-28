from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import Literal

import yaml

from tepa_eval.hashing import rendering_hash
from tepa_eval.schemas import ConditionRendering, InterventionEvent

DIRECTION_BUCKETS: tuple[str, ...] = (
    "center",
    "up",
    "down",
    "left",
    "right",
    "up-left",
    "up-right",
    "down-left",
    "down-right",
)
DIRECTION_TO_INDEX = {name: index for index, name in enumerate(DIRECTION_BUCKETS)}
MAGNITUDE_BUCKETS: tuple[str, ...] = ("tiny", "small", "medium", "large")
MAGNITUDE_TO_INDEX = {name: index for index, name in enumerate(MAGNITUDE_BUCKETS)}

TRAIN_NATURAL_LANGUAGE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("nl_train_001", "Push {object_alias} {direction} with {magnitude} force and predict {horizon} frames ahead."),
    ("nl_train_002", "Forecast the scene after {object_alias} receives a {magnitude} impulse toward {direction} for {horizon} steps."),
    ("nl_train_003", "{object_alias}: apply a {magnitude} {direction} nudge, then estimate the state at frame {horizon}."),
    ("nl_train_004", "After a {magnitude} push sends {object_alias} {direction}, predict the next {horizon} frames."),
    ("nl_train_005", "Move {object_alias} {direction} using a {magnitude} impulse; forecast {horizon} frames."),
    ("nl_train_006", "Simulate {horizon} frames after {object_alias} is nudged {direction} with {magnitude} strength."),
    ("nl_train_007", "Apply {magnitude} force to {object_alias} in the {direction} direction and roll forward {horizon} frames."),
    ("nl_train_008", "What trajectory follows if {object_alias} gets a {magnitude} {direction} shove for a {horizon}-frame rollout?"),
    ("nl_train_009", "Use screen-space impulse dx={dx}, dy={dy} on {object_alias}; forecast {horizon} frames."),
    ("nl_train_010", "Kick {object_alias} with vector ({dx}, {dy}) and predict the scene at {horizon} frames."),
    ("nl_train_011", "Object {object_id} receives impulse dx={dx}, dy={dy}; estimate the outcome after {horizon} steps."),
    ("nl_train_012", "For puck {object_id}, apply impulse ({dx}, {dy}) and simulate {horizon} frames."),
    ("nl_train_013", "The target is {object_alias}. Add velocity impulse dx {dx} and dy {dy}, then forecast {horizon} frames."),
    ("nl_train_014", "Predict what happens when {object_alias} is pushed {direction}; impulse components are dx={dx}, dy={dy}."),
    ("nl_train_015", "{object_alias} gets a {magnitude} shove toward {direction}. Where is the world after {horizon} frames?"),
    ("nl_train_016", "Roll out the world for {horizon} frames after pushing {object_alias} {direction} with {magnitude} force."),
    ("nl_train_017", "Send {object_alias} {direction} with {magnitude} strength and infer the future puck positions."),
    ("nl_train_018", "Starting from the image, give {object_alias} a {magnitude} impulse toward {direction} and predict the rollout."),
    ("nl_train_019", "Target {object_alias}; direction {direction}; magnitude {magnitude}; horizon {horizon} frames."),
    ("nl_train_020", "Target object {object_id}; impulse dx={dx}; impulse dy={dy}; horizon={horizon}."),
    ("nl_train_021", "Forecast after applying the vector dx={dx}, dy={dy} to {object_alias}."),
    ("nl_train_022", "{object_alias} should receive a screen-coordinate impulse of ({dx}, {dy}); predict {horizon} frames."),
    ("nl_train_023", "Run the simulator forward after object {object_id} is pushed {direction} with {magnitude} force."),
    ("nl_train_024", "Estimate final motion for {object_alias} after a {magnitude} push in direction {direction}."),
)
HOLDOUT_NATURAL_LANGUAGE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("nl_holdout_001", "Where does the world end up {horizon} ticks after {object_alias} is sent {direction} with {magnitude} strength?"),
    ("nl_holdout_002", "Project the future state once {object_alias} has been launched {direction} at {magnitude} intensity."),
    ("nl_holdout_003", "If object {object_id} is given impulse components {dx} and {dy}, what rollout follows?"),
    ("nl_holdout_004", "Advance the scene after a shove of ({dx}, {dy}) is applied to {object_alias}."),
    ("nl_holdout_005", "How does the puck world evolve when {object_alias} is driven {direction} with {magnitude} power?"),
    ("nl_holdout_006", "Infer the consequence of forcing puck {object_id} along {direction} for a {horizon}-frame prediction."),
)


def render_conditions(
    event: InterventionEvent,
    event_identity: str,
    start_condition_id: int,
    families: Sequence[str],
    renderings_per_event: int,
    allow_holdout_templates: bool = True,
) -> list[ConditionRendering]:
    renderings: list[ConditionRendering] = []
    family_index = 0
    family_counts: dict[str, int] = {}
    natural_index = _template_start_index(event_identity)
    natural_template_group = _natural_template_group(event, allow_holdout_templates)
    while len(renderings) < renderings_per_event:
        family = families[family_index % len(families)]
        family_index += 1
        family_occurrence = family_counts.get(family, 0)
        family_counts[family] = family_occurrence + 1
        if family == "natural_language":
            template_id, payload, exact_impulse = _natural_language(event, natural_index, natural_template_group)
            natural_index += 1
            payload_type = "text"
        elif family == "json":
            template_id, payload = _json(event, family_occurrence)
            payload_type = "structured_text"
            exact_impulse = True
        elif family == "yaml":
            template_id, payload = _yaml(event, family_occurrence)
            payload_type = "structured_text"
            exact_impulse = True
        elif family == "key_value":
            template_id, payload = _key_value(event, family_occurrence)
            payload_type = "structured_text"
            exact_impulse = True
        else:
            continue
        rendering = ConditionRendering(
            condition_id=start_condition_id + len(renderings),
            event_id=event.event_id,
            family=family,
            template_id=template_id,
            payload_type=payload_type,
            payload_inline=payload,
            metadata={
                "template_group": "holdout" if "holdout" in template_id else "train",
                "exact_impulse": exact_impulse,
            },
        )
        renderings.append(rendering.model_copy(update={"rendering_hash": rendering_hash(rendering, event_identity)}))
    return renderings


def family_order(weights: dict[str, float]) -> list[str]:
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return [family for family, weight in ordered if weight > 0]


def _base_payload(event: InterventionEvent) -> dict[str, object]:
    return {
        "object": event.target_object_aliases[0],
        "object_id": event.target_object_id,
        "impulse": {"dx": round(event.impulse[0], 4), "dy": round(event.impulse[1], 4)},
        "direction": event.direction_bucket,
        "magnitude": event.magnitude_bucket,
        "horizon_frames": event.horizon,
    }


def _natural_template_group(event: InterventionEvent, allow_holdout_templates: bool) -> Literal["train", "holdout"]:
    return "holdout" if allow_holdout_templates and event.event_id % 4 == 3 else "train"


def _template_start_index(event_identity: str) -> int:
    digest = sha256(event_identity.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _natural_language(
    event: InterventionEvent,
    index: int,
    template_group: Literal["train", "holdout"],
) -> tuple[str, str, bool]:
    templates = (
        HOLDOUT_NATURAL_LANGUAGE_TEMPLATES
        if template_group == "holdout"
        else TRAIN_NATURAL_LANGUAGE_TEMPLATES
    )
    template_id, template = templates[index % len(templates)]
    exact_impulse = "{dx}" in template or "{dy}" in template
    return (
        template_id,
        template.format(
            object_alias=event.target_object_aliases[index % len(event.target_object_aliases)],
            object_id=event.target_object_id,
            dx=round(event.impulse[0], 4),
            dy=round(event.impulse[1], 4),
            direction=event.direction_bucket,
            magnitude=event.magnitude_bucket,
            horizon=event.horizon,
        ),
        exact_impulse,
    )


def _json(event: InterventionEvent, occurrence: int = 0) -> tuple[str, str]:
    payload = _base_payload(event)
    impulse = payload["impulse"]
    assert isinstance(impulse, dict)
    variants = (
        ("json_train_001", payload),
        (
            "json_train_002",
            {
                "target": {"object_id": event.target_object_id, "alias": event.target_object_aliases[0]},
                "intervention": {"type": "impulse", "dx": impulse["dx"], "dy": impulse["dy"]},
                "prediction": {"horizon_frames": event.horizon},
            },
        ),
        (
            "json_train_003",
            {
                "object_id": event.target_object_id,
                "impulse_dx": impulse["dx"],
                "impulse_dy": impulse["dy"],
                "horizon": event.horizon,
                "direction_bucket": event.direction_bucket,
                "magnitude_bucket": event.magnitude_bucket,
            },
        ),
        (
            "json_train_004",
            {
                "commands": [
                    {
                        "object": event.target_object_aliases[0],
                        "object_id": event.target_object_id,
                        "impulse": [impulse["dx"], impulse["dy"]],
                    }
                ],
                "rollout_frames": event.horizon,
            },
        ),
    )
    template_id, selected = variants[occurrence % len(variants)]
    return template_id, json.dumps(selected, sort_keys=True)


def _yaml(event: InterventionEvent, occurrence: int = 0) -> tuple[str, str]:
    payload = _base_payload(event)
    impulse = payload["impulse"]
    assert isinstance(impulse, dict)
    variants = (
        ("yaml_train_001", payload),
        (
            "yaml_train_002",
            {
                "target": {"object_id": event.target_object_id, "alias": event.target_object_aliases[0]},
                "intervention": {"dx": impulse["dx"], "dy": impulse["dy"]},
                "horizon_frames": event.horizon,
            },
        ),
        (
            "yaml_train_003",
            {
                "object_id": event.target_object_id,
                "impulse_dx": impulse["dx"],
                "impulse_dy": impulse["dy"],
                "direction": event.direction_bucket,
                "magnitude": event.magnitude_bucket,
                "horizon": event.horizon,
            },
        ),
    )
    template_id, selected = variants[occurrence % len(variants)]
    return template_id, yaml.safe_dump(selected, sort_keys=True)


def _key_value(event: InterventionEvent, occurrence: int = 0) -> tuple[str, str]:
    payload = _base_payload(event)
    impulse = payload["impulse"]
    assert isinstance(impulse, dict)
    variants = (
        (
            "kv_train_001",
            [
                f"object: {payload['object']}",
                f"object_id: {payload['object_id']}",
                f"dx: {impulse['dx']}",
                f"dy: {impulse['dy']}",
                f"direction: {payload['direction']}",
                f"magnitude: {payload['magnitude']}",
                f"horizon_frames: {payload['horizon_frames']}",
            ],
        ),
        (
            "kv_train_002",
            [
                f"horizon_frames: {payload['horizon_frames']}",
                f"object_id: {payload['object_id']}",
                f"object: {payload['object']}",
                f"magnitude: {payload['magnitude']}",
                f"direction: {payload['direction']}",
                f"dx: {impulse['dx']}",
                f"dy: {impulse['dy']}",
            ],
        ),
        (
            "kv_train_003",
            [
                f"object: {payload['object']}",
                f"object_id: {payload['object_id']}",
                f"direction: {payload['direction']}",
                f"dx: {impulse['dx']}",
                f"dy: {impulse['dy']}",
                f"magnitude: {payload['magnitude']}",
                f"horizon_frames: {payload['horizon_frames']}",
                "coordinate_frame: screen_xy",
            ],
        ),
    )
    template_id, lines = variants[occurrence % len(variants)]
    return template_id, "\n".join(lines)
