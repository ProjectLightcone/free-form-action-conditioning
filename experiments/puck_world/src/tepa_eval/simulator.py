from __future__ import annotations

import math

import numpy as np

from tepa_eval.hashing import scene_hash
from tepa_eval.schemas import ExperimentConfig, InterventionEvent, PuckSpec, SceneSpec, WallSpec

SIMULATOR_VERSION = "puck-sim-v1"
COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (43, 116, 255),
    "red": (224, 72, 72),
    "green": (40, 170, 105),
    "gold": (230, 177, 65),
}
COLOR_NAMES = tuple(COLORS.keys())


def sample_scene(scene_id: int, seed: int, config: ExperimentConfig) -> SceneSpec:
    rng = np.random.default_rng(seed)
    num_pucks = int(rng.integers(config.min_pucks, config.max_pucks + 1))
    pucks: list[PuckSpec] = []

    for object_id in range(num_pucks):
        radius = float(rng.uniform(0.055, 0.075))
        for _ in range(200):
            x = float(rng.uniform(radius + 0.04, 1.0 - radius - 0.04))
            y = float(rng.uniform(radius + 0.04, 1.0 - radius - 0.04))
            if all(math.hypot(x - puck.x, y - puck.y) > radius + puck.radius + 0.04 for puck in pucks):
                break
        else:
            x = 0.2 + object_id * 0.3
            y = 0.5
        pucks.append(
            PuckSpec(
                object_id=object_id,
                color=COLOR_NAMES[object_id % len(COLOR_NAMES)],
                x=x,
                y=y,
                radius=radius,
                mass=float(rng.uniform(0.9, 1.15)),
            )
        )

    scene = SceneSpec(
        scene_id=scene_id,
        seed=seed,
        image_size=config.image_size,
        horizon=config.horizon,
        pucks=pucks,
        walls=WallSpec(),
        friction=config.friction,
        restitution=config.restitution,
    )
    return scene.model_copy(update={"scene_hash": scene_hash(scene)})


def sample_event(event_id: int, scene: SceneSpec, seed: int) -> InterventionEvent:
    rng = np.random.default_rng(seed)
    puck = scene.pucks[int(rng.integers(0, len(scene.pucks)))]
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    magnitude = float(rng.uniform(0.45, 2.4))
    impulse = (magnitude * math.cos(angle), magnitude * math.sin(angle))
    return InterventionEvent(
        event_id=event_id,
        scene_id=scene.scene_id,
        target_object_id=puck.object_id,
        target_object_aliases=[f"{puck.color} puck", puck.color, f"puck {puck.object_id}"],
        impulse=impulse,
        horizon=scene.horizon,
        magnitude_bucket=magnitude_bucket(magnitude),
        direction_bucket=direction_bucket(*impulse),
    )


def simulate(scene: SceneSpec, event: InterventionEvent) -> dict[str, np.ndarray | float]:
    state = np.array([[puck.x, puck.y, puck.vx, puck.vy, puck.radius, puck.mass] for puck in scene.pucks], dtype=np.float32)
    target_index = next(index for index, puck in enumerate(scene.pucks) if puck.object_id == event.target_object_id)
    state[target_index, 2] += event.impulse[0] / state[target_index, 5]
    state[target_index, 3] += event.impulse[1] / state[target_index, 5]

    trajectory = np.zeros((event.horizon, len(scene.pucks), 2), dtype=np.float32)
    wall_contact = np.zeros((event.horizon,), dtype=np.float32)
    time_to_contact = float(event.horizon + 1)

    for frame in range(event.horizon):
        state[:, 2:4] *= scene.friction
        state[:, 0:2] += state[:, 2:4] * 0.025

        contacted = _resolve_walls(state, scene)
        if contacted and time_to_contact > event.horizon:
            time_to_contact = float(frame)
        wall_contact[frame] = 1.0 if contacted else 0.0
        trajectory[frame] = state[:, 0:2]

    return {
        "trajectory": trajectory,
        "final_positions": trajectory[-1],
        "heatmap": trajectory_heatmap(trajectory, scene.image_size // 2),
        "wall_contact": np.array([float(wall_contact.max())], dtype=np.float32),
        "time_to_contact": np.array([time_to_contact], dtype=np.float32),
    }


def render_context(scene: SceneSpec) -> np.ndarray:
    size = scene.image_size
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    image[[0, -1], :, :] = 40
    image[:, [0, -1], :] = 40
    yy, xx = np.mgrid[0:size, 0:size]
    for puck in scene.pucks:
        cx = puck.x * (size - 1)
        cy = puck.y * (size - 1)
        radius = puck.radius * size
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = COLORS[puck.color]
        rim = np.abs((xx - cx) ** 2 + (yy - cy) ** 2 - radius**2) <= max(2.0, radius)
        image[rim] = 32
    return image


def context_state(scene: SceneSpec, max_pucks: int) -> np.ndarray:
    features = np.zeros((max_pucks, 7), dtype=np.float32)
    for index, puck in enumerate(scene.pucks[:max_pucks]):
        features[index] = np.array(
            [puck.x, puck.y, puck.vx, puck.vy, puck.radius, puck.mass, float(index + 1)],
            dtype=np.float32,
        )
    return features.reshape(-1)


def condition_params(scene: SceneSpec, event: InterventionEvent) -> np.ndarray:
    puck = next(puck for puck in scene.pucks if puck.object_id == event.target_object_id)
    return np.array(
        [
            float(event.target_object_id),
            float(event.impulse[0]),
            float(event.impulse[1]),
            float(event.horizon),
            puck.x,
            puck.y,
        ],
        dtype=np.float32,
    )


def trajectory_heatmap(trajectory: np.ndarray, size: int) -> np.ndarray:
    heatmap = np.zeros((size, size), dtype=np.float32)
    for frame in trajectory:
        for x, y in frame:
            px = int(np.clip(round(float(x) * (size - 1)), 0, size - 1))
            py = int(np.clip(round(float(y) * (size - 1)), 0, size - 1))
            heatmap[py, px] = 1.0
    return heatmap


def _resolve_walls(state: np.ndarray, scene: SceneSpec) -> bool:
    contacted = False
    walls = scene.walls
    for puck in state:
        radius = puck[4]
        if puck[0] - radius < walls.left:
            puck[0] = walls.left + radius
            puck[2] = abs(puck[2]) * scene.restitution
            contacted = True
        if puck[0] + radius > walls.right:
            puck[0] = walls.right - radius
            puck[2] = -abs(puck[2]) * scene.restitution
            contacted = True
        if puck[1] - radius < walls.top:
            puck[1] = walls.top + radius
            puck[3] = abs(puck[3]) * scene.restitution
            contacted = True
        if puck[1] + radius > walls.bottom:
            puck[1] = walls.bottom - radius
            puck[3] = -abs(puck[3]) * scene.restitution
            contacted = True
    return contacted


def magnitude_bucket(magnitude: float) -> str:
    if magnitude < 0.8:
        return "tiny"
    if magnitude < 1.3:
        return "small"
    if magnitude < 1.9:
        return "medium"
    return "large"


def direction_bucket(dx: float, dy: float) -> str:
    horizontal = "right" if dx > 0.2 else "left" if dx < -0.2 else ""
    vertical = "down" if dy > 0.2 else "up" if dy < -0.2 else ""
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "center"
