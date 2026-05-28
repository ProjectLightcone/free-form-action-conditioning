from __future__ import annotations

import torch


def trajectory_to_soft_heatmap(
    trajectory: torch.Tensor,
    heatmap_size: int,
    max_pucks: int,
    sigma: float | None = None,
) -> torch.Tensor:
    """Render normalized puck trajectories into differentiable occupancy maps."""
    points = _reshape_trajectory(trajectory, max_pucks)
    sigma_value = sigma if sigma is not None else 1.0 / max(float(heatmap_size - 1), 1.0)
    grid = _grid(points.device, points.dtype, heatmap_size)

    deltas = points[:, :, :, None, None, :] - grid
    squared_distance = deltas.square().sum(dim=-1)
    point_heat = torch.exp(-0.5 * squared_distance / (sigma_value * sigma_value))
    point_heat = point_heat * _active_point_weights(points)[:, :, :, None, None]

    accumulated = point_heat.sum(dim=(1, 2))
    heatmap = 1.0 - torch.exp(-accumulated)
    return heatmap.flatten(start_dim=1).clamp(0.0, 1.0)


def trajectory_to_heatmap_logits(
    trajectory: torch.Tensor,
    heatmap_size: int,
    max_pucks: int,
    sigma: float | None = None,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    heatmap = trajectory_to_soft_heatmap(
        trajectory=trajectory,
        heatmap_size=heatmap_size,
        max_pucks=max_pucks,
        sigma=sigma,
    )
    return torch.logit(heatmap.clamp(epsilon, 1.0 - epsilon))


def _reshape_trajectory(trajectory: torch.Tensor, max_pucks: int) -> torch.Tensor:
    if trajectory.ndim == 4:
        return trajectory
    if trajectory.ndim != 2:
        raise ValueError("Trajectory must have shape [batch, flat] or [batch, horizon, pucks, xy].")
    if trajectory.shape[1] % (max_pucks * 2) != 0:
        raise ValueError("Flat trajectory length must be divisible by max_pucks * 2.")
    horizon = trajectory.shape[1] // (max_pucks * 2)
    return trajectory.reshape(trajectory.shape[0], horizon, max_pucks, 2)


def _grid(device: torch.device, dtype: torch.dtype, heatmap_size: int) -> torch.Tensor:
    coordinates = torch.linspace(0.0, 1.0, heatmap_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack([xx, yy], dim=-1)[None, None, None]


def _active_point_weights(points: torch.Tensor) -> torch.Tensor:
    # Padded absent pucks are represented as all-zero trajectories. Keep those
    # points from painting a false path in the top-left corner.
    max_distance_from_padding_origin = points.norm(dim=-1).amax(dim=1, keepdim=True)
    return torch.sigmoid((max_distance_from_padding_origin - 0.08) * 160.0).expand_as(points[..., 0])
