from __future__ import annotations

from pathlib import Path

from tepa_eval.schemas import ExperimentConfig, InterventionEvent, PuckSpec, SceneSpec
from tepa_eval.simulator import sample_scene, simulate
from tepa_eval.storage import DedupeIndex, split_from_hash


def test_simulator_is_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scene = sample_scene(0, 123, config)
    event = InterventionEvent(
        event_id=0,
        scene_id=0,
        target_object_id=scene.pucks[0].object_id,
        target_object_aliases=["blue puck"],
        impulse=(1.2, -0.8),
        horizon=config.horizon,
        magnitude_bucket="small",
        direction_bucket="up-right",
    )
    first = simulate(scene, event)
    second = simulate(scene, event)
    assert (first["trajectory"] == second["trajectory"]).all()
    assert (first["heatmap"] == second["heatmap"]).all()


def test_wall_collision_stays_inside_bounds() -> None:
    scene = SceneSpec(
        scene_id=0,
        seed=1,
        image_size=64,
        horizon=20,
        pucks=[PuckSpec(object_id=0, color="blue", x=0.9, y=0.5, radius=0.06)],
        friction=1.0,
        restitution=0.8,
    )
    event = InterventionEvent(
        event_id=0,
        scene_id=0,
        target_object_id=0,
        target_object_aliases=["blue puck"],
        impulse=(6.0, 0.0),
        horizon=20,
        magnitude_bucket="large",
        direction_bucket="right",
    )
    rollout = simulate(scene, event)
    assert rollout["wall_contact"][0] == 1.0
    assert rollout["trajectory"][:, 0, 0].max() <= 0.94


def test_dedupe_index_reuses_existing_identity(tmp_path: Path) -> None:
    index = DedupeIndex(tmp_path / "dedupe.sqlite")
    assert index.upsert_scene("abc", 1, "train") == 1
    assert index.upsert_scene("abc", 99, "val") == 1
    assert split_from_hash("0" * 24, 1.0) == "val"
    index.close()


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=tmp_path / "data",
        run_dir=tmp_path / "runs",
        checkpoint_dir=tmp_path / "checkpoints",
        num_scenes=2,
        events_per_scene=2,
        renderings_per_event=2,
        horizon=12,
    )
