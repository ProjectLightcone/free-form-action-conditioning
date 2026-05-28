from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tepa_eval.io import read_jsonl


def validate_dataset(dataset: str | Path) -> dict[str, int]:
    dataset_dir = Path(dataset)
    manifests = dataset_dir / "manifests"
    scenes = read_jsonl(manifests / "scenes.jsonl")
    events = read_jsonl(manifests / "events.jsonl")
    outcomes = read_jsonl(manifests / "outcomes.jsonl")
    condition_rows = []
    condition_counts: dict[str, int] = {}
    for path in sorted(manifests.glob("conditions_*.jsonl")):
        split = path.stem.removeprefix("conditions_")
        rows = read_jsonl(path)
        condition_counts[split] = len(rows)
        condition_rows.extend(rows)

    scene_hashes = {row["scene_hash"] for row in scenes}
    event_hashes = {row["event_hash"] for row in events}
    event_ids = {int(row["event_id"]) for row in events}
    outcome_event_ids = {int(row["event_id"]) for row in outcomes}

    _assert_unique([row["scene_hash"] for row in scenes], "scene_hash")
    _assert_unique([row["event_hash"] for row in events], "event_hash")
    _assert_unique([row["outcome_hash"] for row in outcomes], "outcome_hash")
    _assert_unique([row["rendering_hash"] for row in condition_rows], "rendering_hash")

    for event in events:
        if event["scene_hash"] not in scene_hashes:
            raise ValueError(f"Event points to missing scene_hash: {event['event_hash']}")
    for outcome in outcomes:
        if outcome["event_hash"] not in event_hashes:
            raise ValueError(f"Outcome points to missing event_hash: {outcome['outcome_hash']}")
    for condition in condition_rows:
        if int(condition["event_id"]) not in event_ids:
            raise ValueError(f"Condition points to missing event_id: {condition['condition_id']}")
        _validate_payload(condition)

    if event_ids != outcome_event_ids:
        raise ValueError("Every event must have exactly one outcome.")

    train_scene_hashes = {row["scene_hash"] for row in condition_rows if row["split"] == "train"}
    val_scene_hashes = {row["scene_hash"] for row in condition_rows if row["split"] == "val"}
    leakage = train_scene_hashes.intersection(val_scene_hashes)
    if leakage:
        raise ValueError(f"scene_hash leakage between train and val: {sorted(leakage)[:3]}")

    scene_arrays = np.load(dataset_dir / "arrays" / "scenes" / "shard_00000.npz")
    outcome_arrays = np.load(dataset_dir / "arrays" / "outcomes" / "shard_00000.npz")
    condition_arrays = np.load(dataset_dir / "arrays" / "conditions" / "shard_00000.npz")
    if len(scene_arrays["scene_id"]) != len(scenes):
        raise ValueError("Scene array count does not match scene manifest count.")
    if len(outcome_arrays["event_id"]) != len(events):
        raise ValueError("Outcome array count does not match event manifest count.")
    if len(condition_arrays["condition_id"]) != len(condition_rows):
        raise ValueError("Condition array count does not match condition manifest count.")

    summary = {
        "scenes": len(scenes),
        "events": len(events),
        "conditions": len(condition_rows),
        "train_conditions": sum(1 for row in condition_rows if row["split"] == "train"),
        "val_conditions": sum(1 for row in condition_rows if row["split"] == "val"),
        "test_template_conditions": sum(1 for row in condition_rows if row["split"] == "test_templates"),
    }
    for split, count in condition_counts.items():
        summary[f"{split}_conditions"] = count
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated TEPA puck-world dataset.")
    parser.add_argument("--dataset", required=True, help="Path to generated dataset directory.")
    args = parser.parse_args()
    summary = validate_dataset(args.dataset)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _assert_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} values detected.")


def _validate_payload(row: dict[str, object]) -> None:
    family = row["family"]
    payload = str(row["payload_inline"])
    if family == "json":
        json.loads(payload)
    elif family == "yaml":
        yaml.safe_load(payload)
    elif family == "key_value":
        keys = {line.split(":", 1)[0].strip() for line in payload.splitlines() if ":" in line}
        required = {"object", "object_id", "dx", "dy", "direction", "magnitude", "horizon_frames"}
        if not required.issubset(keys):
            raise ValueError(f"Key-value payload is missing keys: {required - keys}")


if __name__ == "__main__":
    main()
