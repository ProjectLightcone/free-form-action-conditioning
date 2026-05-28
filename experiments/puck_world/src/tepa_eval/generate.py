from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from tepa_eval.conditions import family_order, render_conditions
from tepa_eval.hashing import config_hash, event_hash, outcome_hash
from tepa_eval.io import write_jsonl
from tepa_eval.schemas import ExperimentConfig, InterventionEvent, OutcomeBundle, PuckSpec, load_config
from tepa_eval.simulator import (
    SIMULATOR_VERSION,
    condition_params,
    context_state,
    direction_bucket,
    magnitude_bucket,
    render_context,
    sample_event,
    sample_scene,
    simulate,
)
from tepa_eval.storage import DedupeIndex, split_from_hash


def generate_dataset(config: ExperimentConfig) -> Path:
    output = config.output_dir
    manifests = output / "manifests"
    arrays = output / "arrays"
    for folder in (manifests, arrays / "scenes", arrays / "outcomes", arrays / "conditions"):
        folder.mkdir(parents=True, exist_ok=True)

    index = DedupeIndex(output / "dedupe_index.sqlite")
    families = family_order(config.condition_families)
    target_config_hash = config_hash({"horizon": config.horizon, "heatmap_size": config.heatmap_size})

    scenes = []
    events = []
    outcomes = []
    condition_splits = (
        (config.counterfactual_split,)
        if config.generation_mode == "counterfactual_eval"
        else ("train", "val", "test_templates")
    )
    conditions_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in condition_splits}

    scene_ids = []
    scene_hashes = []
    scene_images = []
    scene_states = []

    outcome_event_ids = []
    outcome_event_hashes = []
    outcome_hashes = []
    target_final_pos = []
    target_traj = []
    target_heatmap = []
    target_wall_contact = []
    target_ttc = []

    condition_ids = []
    condition_event_ids = []
    condition_hashes = []
    condition_params_rows = []

    event_id = 0
    outcome_id = 0
    condition_id = 0
    scene_seed_base = config.seed * 1_000_003

    for scene_id in tqdm(range(config.num_scenes), desc="scenes"):
        scene_seed = scene_seed_base + scene_id
        scene = sample_scene(scene_id=scene_id, seed=scene_seed, config=config)
        scene_split = (
            config.counterfactual_split
            if config.generation_mode == "counterfactual_eval"
            else split_from_hash(scene.scene_hash, config.scene_val_fraction)
        )
        index.upsert_scene(scene.scene_hash, scene.scene_id, scene_split)

        scenes.append(scene.model_dump(mode="json") | {"split": scene_split})
        scene_ids.append(scene.scene_id)
        scene_hashes.append(scene.scene_hash)
        scene_images.append(render_context(scene))
        scene_states.append(context_state(scene, config.max_pucks))

        scene_events = _events_for_scene(config, scene, scene_id, event_id)
        for local_event, (event, event_metadata) in enumerate(scene_events):
            event_seed = config.seed * 10_000_019 + scene_id * 10_007 + local_event
            identity = event_hash(event, scene.scene_hash)
            event = event.model_copy(update={"event_hash": identity})
            index.upsert_event(identity, event.event_id, scene.scene_hash)
            events.append(
                event.model_dump(mode="json")
                | {"scene_hash": scene.scene_hash, "split": scene_split, "metadata": event_metadata}
            )

            rollout = simulate(scene, event)
            outcome_identity = outcome_hash(identity, SIMULATOR_VERSION, target_config_hash)
            outcome = OutcomeBundle(
                outcome_id=outcome_id,
                event_id=event.event_id,
                outcome_hash=outcome_identity,
                rollout_seed=event_seed,
                simulator_version=SIMULATOR_VERSION,
            )
            index.upsert_outcome(outcome_identity, outcome.outcome_id, identity)
            outcomes.append(outcome.model_dump(mode="json") | {"event_hash": identity})

            outcome_event_ids.append(event.event_id)
            outcome_event_hashes.append(identity)
            outcome_hashes.append(outcome_identity)
            target_final_pos.append(_pad_positions(rollout["final_positions"], config.max_pucks))
            target_traj.append(_pad_trajectory(rollout["trajectory"], config.max_pucks))
            target_heatmap.append(rollout["heatmap"])
            target_wall_contact.append(rollout["wall_contact"])
            target_ttc.append(rollout["time_to_contact"])

            renderings = render_conditions(
                event=event,
                event_identity=identity,
                start_condition_id=condition_id,
                families=families,
                renderings_per_event=config.renderings_per_event,
                allow_holdout_templates=scene_split != "val",
            )
            params = condition_params(scene, event)
            for rendering in renderings:
                persisted_condition_id = index.upsert_rendering(
                    rendering.rendering_hash,
                    rendering.condition_id,
                    identity,
                    rendering.family,
                    rendering.template_id,
                )
                rendering = rendering.model_copy(update={"condition_id": persisted_condition_id})
                split = _condition_split(config, scene_split, rendering.metadata)
                conditions_by_split[split].append(
                    rendering.model_dump(mode="json")
                    | {"event_hash": identity, "scene_hash": scene.scene_hash, "split": split}
                )
                condition_ids.append(rendering.condition_id)
                condition_event_ids.append(rendering.event_id)
                condition_hashes.append(rendering.rendering_hash)
                condition_params_rows.append(params)
            condition_id += len(renderings)
            event_id += 1
            outcome_id += 1

    np.savez_compressed(
        arrays / "scenes" / "shard_00000.npz",
        scene_id=np.array(scene_ids, dtype=np.int64),
        scene_hash=np.array(scene_hashes),
        context_image=np.stack(scene_images).astype(np.uint8),
        context_state=np.stack(scene_states).astype(np.float32),
    )
    np.savez_compressed(
        arrays / "outcomes" / "shard_00000.npz",
        event_id=np.array(outcome_event_ids, dtype=np.int64),
        event_hash=np.array(outcome_event_hashes),
        outcome_hash=np.array(outcome_hashes),
        target_final_pos=np.stack(target_final_pos).astype(np.float32),
        target_traj=np.stack(target_traj).astype(np.float32),
        target_heatmap=np.stack(target_heatmap).astype(np.float32),
        target_wall_contact=np.stack(target_wall_contact).astype(np.float32),
        target_ttc=np.stack(target_ttc).astype(np.float32),
    )
    np.savez_compressed(
        arrays / "conditions" / "shard_00000.npz",
        condition_id=np.array(condition_ids, dtype=np.int64),
        event_id=np.array(condition_event_ids, dtype=np.int64),
        rendering_hash=np.array(condition_hashes),
        condition_params=np.stack(condition_params_rows).astype(np.float32),
    )

    write_jsonl(manifests / "scenes.jsonl", scenes)
    write_jsonl(manifests / "events.jsonl", events)
    write_jsonl(manifests / "outcomes.jsonl", outcomes)
    for split, rows in conditions_by_split.items():
        write_jsonl(manifests / f"conditions_{split}.jsonl", rows)

    with (output / "dataset_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=True)

    index.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the TEPA puck-world dataset.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()
    output = generate_dataset(load_config(args.config))
    print(f"Generated dataset at {output}")


def _pad_positions(value: np.ndarray, max_pucks: int) -> np.ndarray:
    output = np.zeros((max_pucks, 2), dtype=np.float32)
    output[: value.shape[0]] = value
    return output


def _pad_trajectory(value: np.ndarray, max_pucks: int) -> np.ndarray:
    output = np.zeros((value.shape[0], max_pucks, 2), dtype=np.float32)
    output[:, : value.shape[1], :] = value
    return output


def _condition_split(config: ExperimentConfig, scene_split: str, metadata: dict[str, object]) -> str:
    if config.generation_mode == "counterfactual_eval":
        return config.counterfactual_split
    if scene_split == "val":
        return "val"
    return "test_templates" if metadata.get("template_group") == "holdout" else "train"


def _events_for_scene(
    config: ExperimentConfig,
    scene,
    scene_id: int,
    start_event_id: int,
) -> list[tuple[InterventionEvent, dict[str, object]]]:
    if config.generation_mode != "counterfactual_eval":
        rows = []
        for local_event in range(config.events_per_scene):
            event_seed = config.seed * 10_000_019 + scene_id * 10_007 + local_event
            rows.append((sample_event(event_id=start_event_id + local_event, scene=scene, seed=event_seed), {"family": "random"}))
        return rows

    seed = config.seed * 10_000_019 + scene_id * 10_007
    return _counterfactual_events_for_scene(
        scene=scene,
        start_event_id=start_event_id,
        count=config.events_per_scene,
        seed=seed,
    )


def _counterfactual_events_for_scene(
    scene,
    start_event_id: int,
    count: int,
    seed: int,
) -> list[tuple[InterventionEvent, dict[str, object]]]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[InterventionEvent, dict[str, object]]] = []
    seen: set[tuple[int, int, int]] = set()

    def add_event(puck: PuckSpec, impulse: tuple[float, float], family: str) -> None:
        if len(rows) >= count:
            return
        key = (puck.object_id, round(impulse[0], 5), round(impulse[1], 5))
        if key in seen:
            return
        seen.add(key)
        event = _make_event(
            event_id=start_event_id + len(rows),
            scene_id=scene.scene_id,
            horizon=scene.horizon,
            puck=puck,
            impulse=impulse,
        )
        rows.append((event, {"family": family}))

    focus_puck = scene.pucks[int(rng.integers(0, len(scene.pucks)))]
    base_angles = [index * math.tau / 8.0 + float(rng.uniform(-0.08, 0.08)) for index in range(8)]
    for angle in base_angles:
        for magnitude in (0.65, 1.05, 1.55, 2.15):
            add_event(
                focus_puck,
                (magnitude * math.cos(angle), magnitude * math.sin(angle)),
                "nearby_force_direction_grid",
            )

    binding_angles = [index * math.tau / 8.0 + float(rng.uniform(-0.04, 0.04)) for index in range(8)]
    for angle in binding_angles:
        impulse = (1.55 * math.cos(angle), 1.55 * math.sin(angle))
        for puck in scene.pucks:
            add_event(puck, impulse, "object_binding_swap")

    while len(rows) < count:
        puck = scene.pucks[int(rng.integers(0, len(scene.pucks)))]
        angle = float(rng.uniform(0.0, math.tau))
        magnitude = float(rng.uniform(0.45, 2.4))
        add_event(puck, (magnitude * math.cos(angle), magnitude * math.sin(angle)), "random_filler")

    return rows


def _make_event(
    event_id: int,
    scene_id: int,
    horizon: int,
    puck: PuckSpec,
    impulse: tuple[float, float],
) -> InterventionEvent:
    magnitude = math.hypot(*impulse)
    return InterventionEvent(
        event_id=event_id,
        scene_id=scene_id,
        target_object_id=puck.object_id,
        target_object_aliases=[f"{puck.color} puck", puck.color, f"puck {puck.object_id}"],
        impulse=impulse,
        horizon=horizon,
        magnitude_bucket=magnitude_bucket(magnitude),
        direction_bucket=direction_bucket(*impulse),
    )


if __name__ == "__main__":
    main()
