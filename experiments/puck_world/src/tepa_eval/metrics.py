from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from tepa_eval.trajectory import trajectory_to_soft_heatmap

TRAJECTORY_DELTA_LOSS_WEIGHT = 0.25
TRAJECTORY_FINAL_LOSS_WEIGHT = 1.0
FINAL_POSITION_CONSISTENCY_LOSS_WEIGHT = 0.5
CONDITION_IMPULSE_LOSS_WEIGHT = 10.0
CONDITION_HORIZON_LOSS_WEIGHT = 2.0
CONDITION_CONSISTENCY_LOSS_WEIGHT = 0.1


def loss_and_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    horizon: int,
    sigreg_weight: float = 0.0,
    sigreg_num_slices: int = 1024,
    sigreg_num_points: int = 17,
    sigreg_integration_bound: float = 5.0,
    sigreg_on_predictions: bool = False,
    prediction_loss_weight: float = 1.0,
    latent_loss_weight: float = 0.25,
    target_reconstruction_loss_weight: float = 0.5,
    context_state_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_loss, metrics = _supervised_loss_and_metrics(predictions, batch, horizon)
    loss = prediction_loss_weight * prediction_loss
    metrics["prediction_loss"] = float(prediction_loss.detach().cpu())
    metrics["weighted_prediction_loss"] = float((prediction_loss_weight * prediction_loss).detach().cpu())

    reconstruction = predictions.get("target_reconstruction")
    if isinstance(reconstruction, dict):
        reconstruction_loss, reconstruction_metrics = _supervised_loss_and_metrics(reconstruction, batch, horizon)
        loss = loss + target_reconstruction_loss_weight * reconstruction_loss
        metrics["target_reconstruction_loss"] = float(reconstruction_loss.detach().cpu())
        metrics["weighted_target_reconstruction_loss"] = float(
            (target_reconstruction_loss_weight * reconstruction_loss).detach().cpu()
        )
        for key, value in reconstruction_metrics.items():
            if key != "loss":
                metrics[f"target_reconstruction_{key}"] = value

    if "z_hat_target" in predictions and "z_target" in predictions:
        latent_mse = F.mse_loss(predictions["z_hat_target"], predictions["z_target"].detach())
        loss = loss + latent_loss_weight * latent_mse
        metrics["target_latent_mse"] = float(latent_mse.detach().cpu())
        metrics["weighted_target_latent_mse"] = float((latent_loss_weight * latent_mse).detach().cpu())

        if sigreg_weight > 0:
            target_sigreg = sigreg_loss(
                predictions["z_target"],
                num_slices=sigreg_num_slices,
                num_points=sigreg_num_points,
                integration_bound=sigreg_integration_bound,
            )
            sigreg_total = target_sigreg
            metrics["target_sigreg_loss"] = float(target_sigreg.detach().cpu())
            metrics.update(_latent_distribution_metrics("target", predictions["z_target"]))

            if sigreg_on_predictions:
                prediction_sigreg = sigreg_loss(
                    predictions["z_hat_target"],
                    num_slices=sigreg_num_slices,
                    num_points=sigreg_num_points,
                    integration_bound=sigreg_integration_bound,
                )
                sigreg_total = sigreg_total + prediction_sigreg
                metrics["prediction_sigreg_loss"] = float(prediction_sigreg.detach().cpu())
                metrics.update(_latent_distribution_metrics("prediction", predictions["z_hat_target"]))

            loss = loss + sigreg_weight * sigreg_total
            metrics["sigreg_loss"] = float(sigreg_total.detach().cpu())

    if "context_state" in predictions and "context_state" in batch:
        context_state_mse = F.mse_loss(predictions["context_state"], batch["context_state"])
        loss = loss + context_state_loss_weight * context_state_mse
        metrics["context_state_mse"] = float(context_state_mse.detach().cpu())
        metrics["weighted_context_state_mse"] = float((context_state_loss_weight * context_state_mse).detach().cpu())

    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


def target_autoencoder_loss_and_metrics(
    predictions: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    horizon: int,
    sigreg_weight: float = 0.0,
    sigreg_num_slices: int = 1024,
    sigreg_num_points: int = 17,
    sigreg_integration_bound: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = predictions.get("target_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("Target autoencoder predictions must include target_reconstruction.")

    reconstruction_loss, reconstruction_metrics = _supervised_loss_and_metrics(reconstruction, batch, horizon)
    loss = reconstruction_loss
    metrics = {
        "target_reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        **{
            f"target_reconstruction_{key}": value
            for key, value in reconstruction_metrics.items()
            if key != "loss"
        },
    }

    z_target = predictions.get("z_target")
    if sigreg_weight > 0 and isinstance(z_target, torch.Tensor):
        target_sigreg = sigreg_loss(
            z_target,
            num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            integration_bound=sigreg_integration_bound,
        )
        loss = loss + sigreg_weight * target_sigreg
        metrics["target_sigreg_loss"] = float(target_sigreg.detach().cpu())
        metrics["sigreg_loss"] = float(target_sigreg.detach().cpu())
        metrics.update(_latent_distribution_metrics("target", z_target))

    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics


def condition_semantics_loss_and_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    horizon: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    object_target = batch["condition_object_id"].long()
    direction_target = batch["condition_direction"].long()
    magnitude_target = batch["condition_magnitude"].long()
    impulse_target = batch["condition_impulse"].float()
    horizon_target = batch["condition_horizon"].float()
    exact_impulse_mask = batch["condition_has_exact_impulse"].float().reshape(-1) > 0.5

    object_loss = F.cross_entropy(predictions["object_logits"], object_target)
    direction_loss = F.cross_entropy(predictions["direction_logits"], direction_target)
    magnitude_loss = F.cross_entropy(predictions["magnitude_logits"], magnitude_target)
    if exact_impulse_mask.any():
        impulse_loss = F.smooth_l1_loss(
            predictions["impulse"][exact_impulse_mask],
            impulse_target[exact_impulse_mask],
            beta=0.05,
        )
        impulse_mae = F.l1_loss(
            predictions["impulse"][exact_impulse_mask],
            impulse_target[exact_impulse_mask],
        )
        impulse_abs_error = torch.abs(predictions["impulse"][exact_impulse_mask] - impulse_target[exact_impulse_mask])
        impulse_rmse = torch.sqrt(F.mse_loss(predictions["impulse"][exact_impulse_mask], impulse_target[exact_impulse_mask]))
        impulse_max_abs_error = impulse_abs_error.max()
        impulse_within_005 = (impulse_abs_error <= 0.05).all(dim=-1).float().mean()
        impulse_within_010 = (impulse_abs_error <= 0.10).all(dim=-1).float().mean()
        impulse_within_025 = (impulse_abs_error <= 0.25).all(dim=-1).float().mean()
    else:
        impulse_loss = predictions["impulse"].new_zeros(())
        impulse_mae = predictions["impulse"].new_zeros(())
        impulse_rmse = predictions["impulse"].new_zeros(())
        impulse_max_abs_error = predictions["impulse"].new_zeros(())
        impulse_within_005 = predictions["impulse"].new_zeros(())
        impulse_within_010 = predictions["impulse"].new_zeros(())
        impulse_within_025 = predictions["impulse"].new_zeros(())
    horizon_scale = float(max(horizon, 1))
    horizon_loss = F.smooth_l1_loss(
        predictions["horizon"] / horizon_scale,
        horizon_target / horizon_scale,
        beta=0.01,
    )
    horizon_mae = F.l1_loss(predictions["horizon"], horizon_target)
    horizon_within_05 = (torch.abs(predictions["horizon"] - horizon_target) <= 0.5).float().mean()
    consistency_loss = condition_embedding_consistency_loss(
        predictions["z_condition"],
        batch.get("event_id", []),
    )

    loss = (
        object_loss
        + direction_loss
        + magnitude_loss
        + CONDITION_IMPULSE_LOSS_WEIGHT * impulse_loss
        + CONDITION_HORIZON_LOSS_WEIGHT * horizon_loss
        + CONDITION_CONSISTENCY_LOSS_WEIGHT * consistency_loss
    )
    metrics = {
        "condition_loss": float(loss.detach().cpu()),
        "condition_object_loss": float(object_loss.detach().cpu()),
        "condition_direction_loss": float(direction_loss.detach().cpu()),
        "condition_magnitude_loss": float(magnitude_loss.detach().cpu()),
        "condition_impulse_loss": float(impulse_loss.detach().cpu()),
        "condition_horizon_loss": float(horizon_loss.detach().cpu()),
        "condition_consistency_loss": float(consistency_loss.detach().cpu()),
        "condition_object_accuracy": _accuracy(predictions["object_logits"], object_target),
        "condition_direction_accuracy": _accuracy(predictions["direction_logits"], direction_target),
        "condition_magnitude_accuracy": _accuracy(predictions["magnitude_logits"], magnitude_target),
        "condition_impulse_mae": float(impulse_mae.detach().cpu()),
        "condition_impulse_rmse": float(impulse_rmse.detach().cpu()),
        "condition_impulse_max_abs_error": float(impulse_max_abs_error.detach().cpu()),
        "condition_impulse_within_0_05": float(impulse_within_005.detach().cpu()),
        "condition_impulse_within_0_10": float(impulse_within_010.detach().cpu()),
        "condition_impulse_within_0_25": float(impulse_within_025.detach().cpu()),
        "condition_horizon_mae": float(horizon_mae.detach().cpu()),
        "condition_horizon_within_0_5": float(horizon_within_05.detach().cpu()),
        "condition_exact_impulse_fraction": float(exact_impulse_mask.float().mean().detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics


def condition_embedding_consistency_loss(z_condition: torch.Tensor, event_ids: Any) -> torch.Tensor:
    if not isinstance(event_ids, list):
        return z_condition.new_zeros(())

    groups: dict[int, list[int]] = {}
    for index, event_id in enumerate(event_ids):
        groups.setdefault(int(event_id), []).append(index)

    losses = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group = z_condition[indices]
        losses.append(F.mse_loss(group, group.mean(dim=0, keepdim=True).expand_as(group)))
    if not losses:
        return z_condition.new_zeros(())
    return torch.stack(losses).mean()


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())


def sigreg_loss(
    embeddings: torch.Tensor,
    num_slices: int = 1024,
    num_points: int = 17,
    integration_bound: float = 5.0,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("SIGReg expects embeddings with shape [batch, dim].")
    if num_slices < 1:
        raise ValueError("SIGReg requires at least one random projection.")
    if num_points < 2:
        raise ValueError("SIGReg requires at least two integration points.")
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    directions = torch.randn(
        embeddings.shape[1],
        num_slices,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    directions = F.normalize(directions, dim=0)
    knots = torch.linspace(
        -integration_bound,
        integration_bound,
        num_points,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    expected_cf = torch.exp(-0.5 * knots.square())

    projected = embeddings @ directions
    projected_knots = projected.unsqueeze(-1) * knots
    empirical_real = torch.cos(projected_knots).mean(dim=0)
    empirical_imag = torch.sin(projected_knots).mean(dim=0)
    squared_error = (empirical_real - expected_cf).square() + empirical_imag.square()
    weighted_error = squared_error * expected_cf
    per_slice = torch.trapz(weighted_error, knots, dim=-1) * embeddings.shape[0]
    return per_slice.mean()


def _latent_distribution_metrics(prefix: str, embeddings: torch.Tensor) -> dict[str, float]:
    detached = embeddings.detach()
    std = torch.std(detached, dim=0, unbiased=False)
    return {
        f"{prefix}_latent_abs_mean": float(detached.mean(dim=0).abs().mean().cpu()),
        f"{prefix}_latent_mean_std": float(std.mean().cpu()),
        f"{prefix}_latent_min_std": float(std.min().cpu()),
    }


def _supervised_loss_and_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    horizon: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_mse = F.mse_loss(predictions["final_pos"], batch["target_final_pos"])
    heatmap_target = _heatmap_target_for_predictions(predictions, batch)
    heatmap_bce = F.binary_cross_entropy_with_logits(predictions["heatmap"], heatmap_target)
    heatmap_dice = _soft_dice_loss(predictions["heatmap"], heatmap_target)
    heatmap_loss = heatmap_bce + heatmap_dice
    trajectory_losses = _optional_trajectory_losses(predictions, batch)
    wall_bce = F.binary_cross_entropy_with_logits(predictions["wall_contact"], batch["target_wall_contact"])
    ttc_pred = predictions["time_to_contact"] / float(horizon + 1)
    ttc_target = batch["target_ttc"] / float(horizon + 1)
    ttc_mae = F.l1_loss(predictions["time_to_contact"], batch["target_ttc"])
    ttc_mse = F.mse_loss(ttc_pred, ttc_target)
    loss = final_mse + 0.5 * heatmap_loss + wall_bce + 0.1 * ttc_mse
    if trajectory_losses is not None:
        loss = (
            loss
            + trajectory_losses["trajectory_mse"]
            + TRAJECTORY_DELTA_LOSS_WEIGHT * trajectory_losses["trajectory_delta_mse"]
            + TRAJECTORY_FINAL_LOSS_WEIGHT * trajectory_losses["trajectory_final_position_mse"]
            + FINAL_POSITION_CONSISTENCY_LOSS_WEIGHT
            * trajectory_losses["final_position_consistency_mse"]
        )

    wall_pred = (torch.sigmoid(predictions["wall_contact"]) > 0.5).float()
    wall_true = batch["target_wall_contact"]
    tp = ((wall_pred == 1) & (wall_true == 1)).sum().item()
    fp = ((wall_pred == 1) & (wall_true == 0)).sum().item()
    fn = ((wall_pred == 0) & (wall_true == 1)).sum().item()
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    return (
        loss,
        {
            "loss": float(loss.detach().cpu()),
            "final_position_mse": float(final_mse.detach().cpu()),
            "trajectory_heatmap_bce": float(heatmap_bce.detach().cpu()),
            "trajectory_heatmap_dice_loss": float(heatmap_dice.detach().cpu()),
            "trajectory_heatmap_loss": float(heatmap_loss.detach().cpu()),
            "wall_contact_f1": float(f1),
            "time_to_contact_mae": float(ttc_mae.detach().cpu()),
            **(
                {
                    key: float(value.detach().cpu())
                    for key, value in trajectory_losses.items()
                }
                if trajectory_losses is not None
                else {}
            ),
        },
    )


def _optional_trajectory_losses(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    if "trajectory" not in predictions or "target_traj" not in batch:
        return None
    predicted = predictions["trajectory"].reshape(
        batch["target_traj"].shape[0],
        -1,
        batch["target_final_pos"].shape[-1] // 2,
        2,
    )
    target = batch["target_traj"].reshape_as(predicted)
    trajectory_mse = F.mse_loss(predicted, target)

    if predicted.shape[1] > 1:
        delta_scale = float(predicted.shape[1] - 1)
        predicted_delta = (predicted[:, 1:] - predicted[:, :-1]) * delta_scale
        target_delta = (target[:, 1:] - target[:, :-1]) * delta_scale
        trajectory_delta_mse = F.mse_loss(predicted_delta, target_delta)
    else:
        trajectory_delta_mse = predicted.new_zeros(())

    predicted_final_from_trajectory = predicted[:, -1].flatten(start_dim=1)
    target_final = batch["target_final_pos"].reshape_as(predicted_final_from_trajectory)
    trajectory_final_position_mse = F.mse_loss(predicted_final_from_trajectory, target_final)
    final_position_consistency_mse = F.mse_loss(
        predictions["final_pos"].reshape_as(predicted_final_from_trajectory),
        predicted_final_from_trajectory,
    )
    return {
        "trajectory_mse": trajectory_mse,
        "trajectory_delta_mse": trajectory_delta_mse,
        "trajectory_final_position_mse": trajectory_final_position_mse,
        "final_position_consistency_mse": final_position_consistency_mse,
    }


def _heatmap_target_for_predictions(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "trajectory" not in predictions or "target_traj" not in batch:
        return batch["target_heatmap"]
    heatmap_dim = predictions["heatmap"].shape[-1]
    if "target_soft_heatmap" in batch and batch["target_soft_heatmap"].shape[-1] == heatmap_dim:
        return batch["target_soft_heatmap"]
    heatmap_size = math.isqrt(heatmap_dim)
    if heatmap_size * heatmap_size != heatmap_dim:
        return batch["target_heatmap"]
    max_pucks = batch["target_final_pos"].shape[-1] // 2
    return trajectory_to_soft_heatmap(
        batch["target_traj"],
        heatmap_size=heatmap_size,
        max_pucks=max_pucks,
    )


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
