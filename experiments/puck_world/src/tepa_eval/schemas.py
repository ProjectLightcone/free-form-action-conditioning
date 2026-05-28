from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PuckSpec(BaseModel):
    object_id: int
    color: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float
    mass: float = 1.0


class WallSpec(BaseModel):
    left: float = 0.0
    right: float = 1.0
    top: float = 0.0
    bottom: float = 1.0


class ObstacleSpec(BaseModel):
    obstacle_id: int
    kind: Literal["rect"] = "rect"
    x: float
    y: float
    width: float
    height: float


class GoalZoneSpec(BaseModel):
    goal_id: int
    x: float
    y: float
    radius: float


class SceneSpec(BaseModel):
    scene_id: int
    scene_hash: str = ""
    seed: int
    image_size: int
    horizon: int
    pucks: list[PuckSpec]
    walls: WallSpec = Field(default_factory=WallSpec)
    obstacles: list[ObstacleSpec] = Field(default_factory=list)
    goal_zones: list[GoalZoneSpec] = Field(default_factory=list)
    friction: float
    restitution: float


class InterventionEvent(BaseModel):
    event_id: int
    scene_id: int
    event_hash: str = ""
    target_object_id: int
    target_object_aliases: list[str]
    impulse: tuple[float, float]
    impulse_frame: int = 0
    horizon: int
    coordinate_frame: Literal["screen_xy", "world_xy"] = "screen_xy"
    magnitude_bucket: Literal["tiny", "small", "medium", "large"]
    direction_bucket: str


class OutcomeBundle(BaseModel):
    outcome_id: int
    event_id: int
    outcome_hash: str = ""
    rollout_seed: int
    simulator_version: str


class ConditionRendering(BaseModel):
    condition_id: int
    event_id: int
    rendering_hash: str = ""
    family: Literal["natural_language", "json", "yaml", "key_value"]
    template_id: str
    renderer_version: str = "condition-renderer-v1"
    payload_type: Literal["text", "structured_text"] = "text"
    payload_inline: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    dataset_version: str = "puck-v0.1"
    seed: int = 7
    output_dir: Path
    run_dir: Path
    checkpoint_dir: Path
    generation_mode: Literal["standard", "counterfactual_eval"] = "standard"
    counterfactual_split: str = "counterfactual"
    image_size: int = 64
    horizon: int = 40
    num_scenes: int = 32
    events_per_scene: int = 4
    renderings_per_event: int = 3
    min_pucks: int = 1
    max_pucks: int = 2
    friction: float = 0.985
    restitution: float = 0.86
    scene_val_fraction: float = 0.2
    heldout_template_fraction: float = 0.2
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-3
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = 0.01
    condition_encoder_lr_scale: float = 1.0
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    min_learning_rate: float = 1e-4
    lr_decay_start_epoch: int = 0
    early_stopping_enabled: bool = True
    early_stopping_monitor: str = "auto"
    early_stopping_min_delta: float = 0.001
    early_stopping_patience: int = 12
    early_stopping_min_epochs: int = 20
    latent_dim: int = 64
    max_text_len: int = 192
    heatmap_size: int = 32
    normalize_latents: bool = False
    sigreg_weight: float = 0.1
    sigreg_num_slices: int = 1024
    sigreg_num_points: int = 17
    sigreg_integration_bound: float = 5.0
    sigreg_on_predictions: bool = False
    prediction_loss_weight: float = 1.0
    latent_loss_weight: float = 0.25
    target_reconstruction_loss_weight: float = 0.5
    context_state_loss_weight: float = 0.0
    target_sigreg_weight: float | None = None
    target_sigreg_warmup_epochs: int = 0
    target_sigreg_ramp_epochs: int = 0
    condition_family_filter: list[str] | None = None
    condition_families: dict[str, float] = Field(
        default_factory=lambda: {
            "natural_language": 0.4,
            "json": 0.25,
            "yaml": 0.2,
            "key_value": 0.15,
        }
    )


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    config = ExperimentConfig.model_validate(data)
    base = config_path.parent.parent
    return config.model_copy(
        update={
            "output_dir": _resolve(base, config.output_dir),
            "run_dir": _resolve(base, config.run_dir),
            "checkpoint_dir": _resolve(base, config.checkpoint_dir),
        }
    )


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
