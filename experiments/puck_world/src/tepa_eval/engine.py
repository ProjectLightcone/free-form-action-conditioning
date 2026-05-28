from __future__ import annotations

import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tepa_eval.dataset import PuckDataset, TargetOutcomeDataset, byte_tokenize, collate_metadata
from tepa_eval.io import read_jsonl, write_text
from tepa_eval.metrics import (
    average_metrics,
    condition_semantics_loss_and_metrics,
    loss_and_metrics,
    target_autoencoder_loss_and_metrics,
)
from tepa_eval.models import build_model
from tepa_eval.schemas import ExperimentConfig
from tepa_eval.trajectory import trajectory_to_soft_heatmap


def default_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}. Expected one of: auto, cpu, mps, cuda.")


def device_report(device: torch.device, requested: str = "auto") -> dict[str, Any]:
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "requested_device": requested,
        "selected_device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(mps_backend and torch.backends.mps.is_built()),
        "mps_available": bool(mps_backend and torch.backends.mps.is_available()),
    }


def print_device_report(device: torch.device, requested: str = "auto") -> None:
    report = device_report(device, requested)
    parts = ", ".join(f"{key}={value}" for key, value in report.items())
    print(f"Device: {parts}", flush=True)


def make_loader(
    config: ExperimentConfig,
    split: str,
    shuffle: bool,
    include_condition_image: bool = False,
    target_latents_by_event: dict[int, np.ndarray] | None = None,
    condition_latents_by_condition: dict[int, np.ndarray] | None = None,
) -> DataLoader:
    dataset = PuckDataset(
        config.output_dir,
        split=split,
        max_text_len=config.max_text_len,
        include_condition_image=include_condition_image,
        target_latents_by_event=target_latents_by_event,
        condition_latents_by_condition=condition_latents_by_condition,
        condition_family_filter=set(config.condition_family_filter) if config.condition_family_filter else None,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_metadata,
        num_workers=0,
    )


def make_target_loader(config: ExperimentConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = TargetOutcomeDataset(
        config.output_dir,
        split=split,
        max_text_len=config.max_text_len,
        include_condition_image=False,
        condition_family_filter=set(config.condition_family_filter) if config.condition_family_filter else None,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_metadata,
        num_workers=0,
    )


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def model_forward(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    forward_batch = getattr(model, "forward_batch", None)
    if callable(forward_batch):
        return forward_batch(batch)
    return model(batch["context_image"], batch["text_tokens"], batch.get("condition_image"))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    parameters = _optimizer_parameter_groups(model, config)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _optimizer_parameter_groups(model: torch.nn.Module, config: ExperimentConfig) -> list[dict[str, Any]]:
    condition_lr_scale = float(config.condition_encoder_lr_scale)
    if condition_lr_scale == 1.0 or not hasattr(model, "condition_encoder"):
        return [{"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr_scale": 1.0}]

    condition_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("condition_encoder."):
            condition_params.append(parameter)
        else:
            other_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr_scale": 1.0})
    if condition_params:
        groups.append({"params": condition_params, "lr_scale": condition_lr_scale})
    return groups


def learning_rate_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    if config.lr_schedule == "constant":
        return config.learning_rate
    start_epoch = max(0, config.lr_decay_start_epoch)
    if epoch < start_epoch:
        return config.learning_rate
    decay_epochs = max(config.epochs - start_epoch, 1)
    progress = min(max((epoch - start_epoch + 1) / decay_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))


def metric_improved(current: float, best: float | None, min_delta: float) -> bool:
    return best is None or current < best - min_delta


def should_stop_early(config: ExperimentConfig, epoch_number: int, epochs_without_improvement: int) -> bool:
    if not config.early_stopping_enabled:
        return False
    if config.early_stopping_patience <= 0:
        return False
    if epoch_number < config.early_stopping_min_epochs:
        return False
    return epochs_without_improvement >= config.early_stopping_patience


def target_sigreg_base_weight(config: ExperimentConfig) -> float:
    return config.sigreg_weight if config.target_sigreg_weight is None else config.target_sigreg_weight


def target_sigreg_weight_for_epoch(config: ExperimentConfig, epoch: int) -> float:
    target_weight = target_sigreg_base_weight(config)
    if epoch < config.target_sigreg_warmup_epochs:
        return 0.0
    if config.target_sigreg_ramp_epochs <= 0:
        return target_weight
    ramp_step = epoch - config.target_sigreg_warmup_epochs + 1
    ramp_fraction = min(max(ramp_step / config.target_sigreg_ramp_epochs, 0.0), 1.0)
    return target_weight * ramp_fraction


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    shuffle_conditions: bool = False,
    sigreg_weight: float | None = None,
) -> dict[str, float]:
    effective_sigreg_weight = config.sigreg_weight if sigreg_weight is None else sigreg_weight
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            if shuffle_conditions:
                condition_batch_size = _condition_batch_size(batch)
                if condition_batch_size > 1:
                    permutation = torch.randperm(condition_batch_size, device=device)
                    if "text_tokens" in batch:
                        batch["text_tokens"] = batch["text_tokens"][permutation]
                    if "z_condition" in batch:
                        batch["z_condition"] = batch["z_condition"][permutation]
                    if "condition_params" in batch:
                        batch["condition_params"] = batch["condition_params"][permutation]
                    if "condition_image" in batch:
                        batch["condition_image"] = batch["condition_image"][permutation]
            predictions = model_forward(model, batch)
            _, metrics = loss_and_metrics(
                predictions,
                batch,
                config.horizon,
                sigreg_weight=effective_sigreg_weight,
                sigreg_num_slices=config.sigreg_num_slices,
                sigreg_num_points=config.sigreg_num_points,
                sigreg_integration_bound=config.sigreg_integration_bound,
                sigreg_on_predictions=config.sigreg_on_predictions,
                prediction_loss_weight=config.prediction_loss_weight,
                latent_loss_weight=config.latent_loss_weight,
                target_reconstruction_loss_weight=config.target_reconstruction_loss_weight,
                context_state_loss_weight=config.context_state_loss_weight,
            )
            rows.append(metrics)
    return average_metrics(rows)


def _condition_batch_size(batch: dict[str, Any]) -> int:
    for key in ("text_tokens", "z_condition", "condition_image", "condition_params"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return 0


def evaluate_target_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    sigreg_weight: float | None = None,
) -> dict[str, float]:
    effective_sigreg_weight = target_sigreg_base_weight(config) if sigreg_weight is None else sigreg_weight
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            predictions = model_forward(model, batch)
            _, metrics = target_autoencoder_loss_and_metrics(
                predictions,
                batch,
                config.horizon,
                sigreg_weight=effective_sigreg_weight,
                sigreg_num_slices=config.sigreg_num_slices,
                sigreg_num_points=config.sigreg_num_points,
                sigreg_integration_bound=config.sigreg_integration_bound,
            )
            rows.append(metrics)
    return average_metrics(rows)


def evaluate_condition_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            predictions = model_forward(model, batch)
            _, metrics = condition_semantics_loss_and_metrics(predictions, batch, config.horizon)
            rows.append(metrics)
    return average_metrics(rows)


def equivalent_condition_consistency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
) -> float:
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            event_ids = batch["event_id"]
            batch = batch_to_device(batch, device)
            predictions = model_forward(model, batch)["final_pos"].detach().cpu()
            for row_index, event_id in enumerate(event_ids):
                grouped[int(event_id)].append(predictions[row_index])

    distances = []
    for predictions in grouped.values():
        if len(predictions) < 2:
            continue
        stack = torch.stack(predictions)
        distances.append(float(torch.var(stack, dim=0).mean()))
    return sum(distances) / len(distances) if distances else 0.0


def build_from_config(model_name: str, config: ExperimentConfig) -> torch.nn.Module:
    return build_model(
        model_name=model_name,
        latent_dim=config.latent_dim,
        max_pucks=config.max_pucks,
        heatmap_size=config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
        normalize_latents=config.normalize_latents,
        context_state_dim=config.max_pucks * 7 if config.context_state_loss_weight > 0 else None,
    )


def precompute_target_latents(
    model: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[int, np.ndarray]:
    target_encoder = getattr(model, "target_encoder", None)
    if target_encoder is None:
        raise ValueError("Target latent caching requires a model with target_encoder.")

    outcomes = np.load(config.output_dir / "arrays" / "outcomes" / "shard_00000.npz")
    event_ids = outcomes["event_id"]
    final_pos = outcomes["target_final_pos"].reshape(event_ids.shape[0], -1).astype(np.float32)
    trajectory = outcomes["target_traj"].reshape(event_ids.shape[0], -1).astype(np.float32)
    heatmap = outcomes["target_heatmap"].reshape(event_ids.shape[0], -1).astype(np.float32)
    wall_contact = outcomes["target_wall_contact"].astype(np.float32)
    time_to_contact = outcomes["target_ttc"].astype(np.float32)
    chunk_size = batch_size or max(config.batch_size * 4, 256)

    was_training = target_encoder.training
    target_encoder.eval()
    cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, event_ids.shape[0], chunk_size):
            end = start + chunk_size
            z_target = target_encoder(
                torch.from_numpy(final_pos[start:end]).to(device),
                torch.from_numpy(trajectory[start:end]).to(device),
                torch.from_numpy(heatmap[start:end]).to(device),
                torch.from_numpy(wall_contact[start:end]).to(device),
                torch.from_numpy(time_to_contact[start:end]).to(device),
            )
            latent_rows = z_target.detach().cpu().numpy().astype(np.float32)
            for event_id, latent in zip(event_ids[start:end], latent_rows, strict=True):
                cache[int(event_id)] = latent
    if was_training:
        target_encoder.train()
    return cache


def precompute_condition_latents(
    model: torch.nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    splits: tuple[str, ...] = ("train", "val", "test_templates"),
    batch_size: int | None = None,
) -> dict[int, np.ndarray]:
    condition_encoder = getattr(model, "condition_encoder", None)
    if condition_encoder is None:
        raise ValueError("Condition latent caching requires a model with condition_encoder.")

    condition_rows: dict[int, str] = {}
    for split in splits:
        manifest_path = config.output_dir / "manifests" / f"conditions_{split}.jsonl"
        if not manifest_path.exists() or manifest_path.stat().st_size == 0:
            continue
        for row in read_jsonl(manifest_path):
            condition_rows[int(row["condition_id"])] = str(row["payload_inline"])

    ids = np.array(sorted(condition_rows), dtype=np.int64)
    if ids.size == 0:
        return {}
    token_rows = np.array(
        [byte_tokenize(condition_rows[int(condition_id)], config.max_text_len) for condition_id in ids],
        dtype=np.int64,
    )
    chunk_size = batch_size or max(config.batch_size * 8, 512)

    was_training = condition_encoder.training
    condition_encoder.eval()
    cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, ids.shape[0], chunk_size):
            end = start + chunk_size
            z_condition = condition_encoder(torch.from_numpy(token_rows[start:end]).to(device))
            latent_rows = z_condition.detach().cpu().numpy().astype(np.float32)
            for condition_id, latent in zip(ids[start:end], latent_rows, strict=True):
                cache[int(condition_id)] = latent
    if was_training:
        condition_encoder.train()
    return cache


def write_run_report(
    run_dir: Path,
    model_name: str,
    config: ExperimentConfig,
    history: list[dict[str, float]],
    final_metrics: dict[str, dict[str, float] | float],
    training_summary: dict[str, Any] | None = None,
) -> None:
    lines = [
        f"# TEPA Puck-World Run: {model_name}",
        "",
        f"- dataset: `{config.output_dir}`",
        f"- requested_epochs: `{config.epochs}`",
        f"- completed_epochs: `{len(history)}`",
        f"- latent_dim: `{config.latent_dim}`",
        f"- batch_size: `{config.batch_size}`",
        f"- trainable_params: `{final_metrics.get('trainable_params', 'unknown')}`",
        f"- checkpoint: `model.pt` (best validation checkpoint)",
        f"- best_checkpoint: `best_model.pt`",
        f"- last_checkpoint: `last_model.pt`",
        "",
    ]
    if training_summary:
        lines.extend(["## Training Summary", ""])
        for key, value in training_summary.items():
            if isinstance(value, float):
                lines.append(f"- {key}: `{value:.6f}`")
            else:
                lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(["## Final Metrics", ""])
    for split, metrics in final_metrics.items():
        lines.append(f"### {split}")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- value: {metrics:.6f}")
        lines.append("")
    lines.extend(["## Training History", "", "```json"])
    lines.extend(json.dumps(row, sort_keys=True) for row in history)
    lines.extend(["```", ""])
    write_text(run_dir / "report.md", "\n".join(lines))


def write_sample_panel(
    run_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    include_decoded_target: bool = False,
    max_samples: int = 1,
    include_condition_text: bool = False,
    output_stem: str = "sample_panel",
    sample_batch: dict[str, Any] | None = None,
) -> None:
    try:
        batch = sample_batch if sample_batch is not None else panel_sample_batch(loader, max_samples)
    except StopIteration:
        return
    sample_count = min(max(int(max_samples), 1), int(batch["context_image"].shape[0]))
    model.eval()
    with torch.no_grad():
        device_batch = batch_to_device(batch, device)
        predictions = model_forward(model, device_batch)
        diagnostics = [
            sample_diagnostics(predictions, batch, config, sample_index=sample_index)
            for sample_index in range(sample_count)
        ]
        decoded_target_heatmap = None
        decoded_target = None
        if include_decoded_target:
            decoded_target = decoded_target_for_panel(model, predictions, sample_count=sample_count)
            if isinstance(decoded_target, dict):
                decoded_target_heatmap = [
                    prediction_heatmap(
                        decoded_target,
                        config,
                        batch["target_traj"][sample_index],
                        sample_index=sample_index,
                    )
                    for sample_index in range(sample_count)
                ]
                for sample_index, row in enumerate(diagnostics):
                    row.update(
                        decoded_target_diagnostics(
                            decoded_target,
                            batch,
                            config,
                            sample_index=sample_index,
                        )
                    )

    has_decoded_target = decoded_target_heatmap is not None
    column_titles = ["context", "target"]
    if has_decoded_target:
        column_titles.append("decoded target")
    column_titles.append("prediction")
    if include_condition_text:
        column_titles = ["condition / metrics"] + column_titles

    image_column_count = len(column_titles) - (1 if include_condition_text else 0)
    figure_width = image_column_count * 3.0 + (4.4 if include_condition_text else 0.0)
    figure_height = max(3.0 * sample_count, 3.0)
    width_ratios = ([1.55] if include_condition_text else []) + [1.0] * image_column_count
    fig, axes = plt.subplots(
        sample_count,
        len(column_titles),
        figsize=(figure_width, figure_height),
        dpi=140,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title)

    for sample_index in range(sample_count):
        axis_offset = 0
        if include_condition_text:
            condition_axis = axes[sample_index, 0]
            condition_axis.text(
                0.02,
                0.98,
                condition_label_for_panel(batch, sample_index, diagnostics[sample_index]),
                transform=condition_axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                linespacing=1.15,
            )
            axis_offset = 1

        context = batch["context_image"][sample_index].permute(1, 2, 0).numpy()
        target_heatmap = target_heatmap_for_panel(batch, config, sample_index=sample_index)
        predicted_heatmap = prediction_heatmap(
            predictions,
            config,
            batch["target_traj"][sample_index],
            sample_index=sample_index,
        )
        active_mask = active_puck_mask_for_panel(batch["target_traj"][sample_index], config)
        target_final = final_positions_for_panel(batch["target_final_pos"], config, sample_index)
        predicted_final = final_positions_for_panel(predictions["final_pos"], config, sample_index)
        axes[sample_index, axis_offset].imshow(context)
        axes[sample_index, axis_offset + 1].imshow(target_heatmap, cmap="magma", vmin=0, vmax=1)
        overlay_final_positions(
            axes[sample_index, axis_offset + 1],
            target_final,
            config,
            active_mask,
            color="#32d5ff",
            marker="o",
        )
        prediction_axis_index = axis_offset + 2
        if has_decoded_target:
            axes[sample_index, axis_offset + 2].imshow(
                decoded_target_heatmap[sample_index],
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            overlay_final_positions(
                axes[sample_index, axis_offset + 2],
                target_final,
                config,
                active_mask,
                color="#32d5ff",
                marker="o",
            )
            if decoded_target is not None and "final_pos" in decoded_target:
                decoded_final = final_positions_for_panel(decoded_target["final_pos"], config, sample_index)
                overlay_final_positions(
                    axes[sample_index, axis_offset + 2],
                    decoded_final,
                    config,
                    active_mask,
                    color="#7CFF6B",
                    marker="x",
                )
            prediction_axis_index = axis_offset + 3
        axes[sample_index, prediction_axis_index].imshow(predicted_heatmap, cmap="magma", vmin=0, vmax=1)
        overlay_final_positions(
            axes[sample_index, prediction_axis_index],
            target_final,
            config,
            active_mask,
            color="#32d5ff",
            marker="o",
        )
        overlay_final_positions(
            axes[sample_index, prediction_axis_index],
            predicted_final,
            config,
            active_mask,
            color="#7CFF6B",
            marker="x",
        )

    for axis in axes.reshape(-1):
        axis.axis("off")
    fig.tight_layout(pad=0.8)
    fig.savefig(run_dir / f"{output_stem}.png", dpi=140)
    plt.close(fig)
    if include_condition_text:
        write_sample_panel_manifest(run_dir, batch, sample_count, diagnostics, output_stem=output_stem)


def write_worst_sample_panel(
    run_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    include_decoded_target: bool = False,
    max_samples: int = 6,
    include_condition_text: bool = True,
    output_stem: str = "sample_panel_worst",
) -> None:
    try:
        batch = panel_worst_sample_batch(model, loader, config, device, max_samples=max_samples)
    except StopIteration:
        return
    write_sample_panel(
        run_dir,
        model,
        loader,
        device,
        config,
        include_decoded_target=include_decoded_target,
        max_samples=max_samples,
        include_condition_text=include_condition_text,
        output_stem=output_stem,
        sample_batch=batch,
    )


def panel_sample_batch(loader: DataLoader, max_samples: int) -> dict[str, Any]:
    if max_samples <= 1:
        return next(iter(loader))

    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        return next(iter(loader))

    dataset_length = len(dataset)
    if dataset_length == 0:
        raise StopIteration

    indices = panel_sample_indices(dataset, min(max_samples, dataset_length))
    return collate_metadata([dataset[index] for index in indices])


def panel_sample_indices(dataset: Any, max_samples: int) -> list[int]:
    total = len(dataset)
    if total <= max_samples:
        return list(range(total))

    conditions = getattr(dataset, "conditions", None)
    if not conditions:
        return sorted(set(np.linspace(0, total - 1, max_samples, dtype=int).tolist()))

    candidate_count = min(total, max_samples * 12)
    candidates = sorted(set(np.linspace(0, total - 1, candidate_count, dtype=int).tolist()))
    selected: list[int] = []
    seen_scene_hashes: set[str] = set()
    seen_event_ids: set[int] = set()

    for index in candidates:
        row = conditions[index]
        scene_hash = str(row.get("scene_hash", ""))
        event_id = int(row.get("event_id", index))
        if scene_hash in seen_scene_hashes or event_id in seen_event_ids:
            continue
        selected.append(index)
        seen_scene_hashes.add(scene_hash)
        seen_event_ids.add(event_id)
        if len(selected) == max_samples:
            return selected

    for index in candidates:
        row = conditions[index]
        event_id = int(row.get("event_id", index))
        if index in selected or event_id in seen_event_ids:
            continue
        selected.append(index)
        seen_event_ids.add(event_id)
        if len(selected) == max_samples:
            return selected

    for index in candidates:
        if index not in selected:
            selected.append(index)
        if len(selected) == max_samples:
            break
    return selected


def panel_worst_sample_batch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    max_samples: int,
) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        return panel_sample_batch(loader, max_samples)

    total = len(dataset)
    if total == 0:
        raise StopIteration

    batch_size = int(getattr(loader, "batch_size", None) or config.batch_size or 64)
    batch_size = max(batch_size, 1)
    top_rows: list[tuple[float, int]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = collate_metadata([dataset[index] for index in range(start, end)])
            device_batch = batch_to_device(batch, device)
            predictions = model_forward(model, device_batch)
            scores = batch_sample_scores(predictions, device_batch, config)
            for offset, score in enumerate(scores):
                top_rows.append((float(score), start + offset))
            top_rows.sort(key=lambda row: row[0], reverse=True)
            del top_rows[max_samples:]

    if not top_rows:
        raise StopIteration
    selected = [index for _, index in sorted(top_rows, key=lambda row: row[0], reverse=True)]
    return collate_metadata([dataset[index] for index in selected])


def batch_sample_scores(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: ExperimentConfig,
) -> list[float]:
    batch_size = int(batch["context_image"].shape[0])
    device = next(
        (value.device for value in predictions.values() if isinstance(value, torch.Tensor)),
        batch["context_image"].device,
    )
    score = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    if "final_pos" in predictions:
        target_final_pos = batch["target_final_pos"].to(device)
        final_mse = (predictions["final_pos"] - target_final_pos).square().mean(dim=1)
        score = score + final_mse

    if "heatmap" in predictions:
        target_heatmap = batch.get("target_soft_heatmap", batch["target_heatmap"]).to(device)
        heatmap_mse = (torch.sigmoid(predictions["heatmap"]) - target_heatmap).square().mean(dim=1)
        score = score + heatmap_mse

    if "trajectory" in predictions:
        target_traj = batch["target_traj"].to(device)
        trajectory_mse = (predictions["trajectory"] - target_traj).square().mean(dim=1)
        score = score + 0.5 * trajectory_mse

    if "z_hat_target" in predictions and "z_target" in predictions:
        latent_mse = (predictions["z_hat_target"] - predictions["z_target"]).square().mean(dim=1)
        score = score + 0.25 * latent_mse

    if "wall_contact" in predictions:
        target_wall_contact = batch["target_wall_contact"].to(device)
        wall_error = (torch.sigmoid(predictions["wall_contact"]) - target_wall_contact).abs().flatten()
        score = score + 0.05 * wall_error

    if "time_to_contact" in predictions:
        target_ttc = batch["target_ttc"].to(device)
        ttc_error = (predictions["time_to_contact"] - target_ttc).abs().flatten()
        score = score + 0.05 * (ttc_error / float(config.horizon + 1))

    return score.detach().cpu().numpy().astype(float).tolist()


def decoded_target_for_panel(
    model: torch.nn.Module,
    predictions: dict[str, Any],
    sample_count: int | None = None,
) -> dict[str, torch.Tensor] | None:
    reconstruction = predictions.get("target_reconstruction")
    if isinstance(reconstruction, dict):
        return reconstruction

    z_target = predictions.get("z_target")
    target_decoder = getattr(model, "target_decoder", None)
    if isinstance(z_target, torch.Tensor) and callable(target_decoder):
        if sample_count is not None:
            z_target = z_target[:sample_count]
        return target_decoder(z_target)
    return None


def sample_diagnostics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: ExperimentConfig,
    sample_index: int,
) -> dict[str, float]:
    target_final = final_positions_for_panel(batch["target_final_pos"], config, sample_index)
    predicted_final = final_positions_for_panel(predictions["final_pos"], config, sample_index)
    active_mask = active_puck_mask_for_panel(batch["target_traj"][sample_index], config)
    if not active_mask.any():
        active_mask = np.ones((config.max_pucks,), dtype=np.bool_)

    object_id = int(batch_value(batch, "condition_object_id", sample_index, default=0))
    object_id = int(np.clip(object_id, 0, config.max_pucks - 1))
    final_diff = predicted_final[active_mask] - target_final[active_mask]
    object_diff = predicted_final[object_id] - target_final[object_id]
    target_heatmap = target_heatmap_for_panel(batch, config, sample_index=sample_index)
    predicted_heatmap = prediction_heatmap(
        predictions,
        config,
        batch["target_traj"][sample_index],
        sample_index=sample_index,
    )
    diagnostics = {
        "sample_score": sample_score_for_panel(predictions, batch, config, sample_index),
        "final_position_mse": float(np.mean(np.square(final_diff))),
        "object_final_error": float(np.linalg.norm(object_diff)),
        "heatmap_mse": float(np.mean(np.square(predicted_heatmap - target_heatmap))),
        "object_id": float(object_id),
        "target_object_x": float(target_final[object_id, 0]),
        "target_object_y": float(target_final[object_id, 1]),
        "predicted_object_x": float(predicted_final[object_id, 0]),
        "predicted_object_y": float(predicted_final[object_id, 1]),
    }

    if "trajectory" in predictions:
        target_trajectory = (
            batch["target_traj"][sample_index]
            .detach()
            .cpu()
            .reshape(config.horizon, config.max_pucks, 2)
            .numpy()
        )
        predicted_trajectory = (
            predictions["trajectory"][sample_index]
            .detach()
            .cpu()
            .reshape(config.horizon, config.max_pucks, 2)
            .numpy()
        )
        trajectory_diff = predicted_trajectory[:, active_mask, :] - target_trajectory[:, active_mask, :]
        diagnostics["trajectory_mse"] = float(np.mean(np.square(trajectory_diff)))

    if "z_hat_target" in predictions and "z_target" in predictions:
        latent_diff = predictions["z_hat_target"][sample_index] - predictions["z_target"][sample_index]
        diagnostics["target_latent_mse"] = float(latent_diff.square().mean().detach().cpu())

    if "wall_contact" in predictions:
        wall_probability = float(torch.sigmoid(predictions["wall_contact"][sample_index]).detach().cpu().reshape(-1)[0])
        diagnostics["wall_contact_true"] = float(batch["target_wall_contact"][sample_index].detach().cpu().reshape(-1)[0])
        diagnostics["wall_contact_probability"] = wall_probability

    if "time_to_contact" in predictions:
        diagnostics["time_to_contact_true"] = float(batch["target_ttc"][sample_index].detach().cpu().reshape(-1)[0])
        diagnostics["time_to_contact_predicted"] = float(
            predictions["time_to_contact"][sample_index].detach().cpu().reshape(-1)[0]
        )
        diagnostics["time_to_contact_abs_error"] = abs(
            diagnostics["time_to_contact_predicted"] - diagnostics["time_to_contact_true"]
        )

    return diagnostics


def sample_score_for_panel(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: ExperimentConfig,
    sample_index: int,
) -> float:
    single_batch = {
        key: value[sample_index : sample_index + 1] if isinstance(value, torch.Tensor) else [value[sample_index]]
        for key, value in batch.items()
    }
    single_predictions = {
        key: value[sample_index : sample_index + 1] if isinstance(value, torch.Tensor) else value
        for key, value in predictions.items()
    }
    return batch_sample_scores(single_predictions, single_batch, config)[0]


def decoded_target_diagnostics(
    decoded_target: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: ExperimentConfig,
    sample_index: int,
) -> dict[str, float]:
    target_final = final_positions_for_panel(batch["target_final_pos"], config, sample_index)
    decoded_final = final_positions_for_panel(decoded_target["final_pos"], config, sample_index)
    active_mask = active_puck_mask_for_panel(batch["target_traj"][sample_index], config)
    if not active_mask.any():
        active_mask = np.ones((config.max_pucks,), dtype=np.bool_)
    decoded_heatmap = prediction_heatmap(
        decoded_target,
        config,
        batch["target_traj"][sample_index],
        sample_index=sample_index,
    )
    target_heatmap = target_heatmap_for_panel(batch, config, sample_index=sample_index)
    return {
        "decoded_final_position_mse": float(np.mean(np.square(decoded_final[active_mask] - target_final[active_mask]))),
        "decoded_heatmap_mse": float(np.mean(np.square(decoded_heatmap - target_heatmap))),
    }


def final_positions_for_panel(values: torch.Tensor, config: ExperimentConfig, sample_index: int) -> np.ndarray:
    if values.ndim == 1:
        row = values
    else:
        row = values[sample_index]
    return row.detach().cpu().reshape(config.max_pucks, 2).numpy().astype(np.float32)


def active_puck_mask_for_panel(target_trajectory: torch.Tensor, config: ExperimentConfig) -> np.ndarray:
    trajectory = target_trajectory.detach().cpu().reshape(config.horizon, config.max_pucks, 2).numpy()
    return np.abs(trajectory).sum(axis=(0, 2)) > 1.0e-5


def overlay_final_positions(
    axis: Any,
    positions: np.ndarray,
    config: ExperimentConfig,
    active_mask: np.ndarray,
    color: str,
    marker: str,
) -> None:
    for puck_index, (x_position, y_position) in enumerate(positions):
        if puck_index >= active_mask.shape[0] or not bool(active_mask[puck_index]):
            continue
        x_pixel = float(np.clip(x_position, 0.0, 1.0) * (config.heatmap_size - 1))
        y_pixel = float(np.clip(y_position, 0.0, 1.0) * (config.heatmap_size - 1))
        if marker == "x":
            axis.scatter([x_pixel], [y_pixel], marker=marker, c=color, s=42, linewidths=1.4)
        else:
            axis.scatter(
                [x_pixel],
                [y_pixel],
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=42,
                linewidths=1.4,
            )
        axis.text(
            x_pixel + 0.7,
            y_pixel + 0.7,
            str(puck_index),
            color=color,
            fontsize=5.5,
            weight="bold",
        )


def condition_label_for_panel(
    batch: dict[str, Any],
    sample_index: int,
    diagnostics: dict[str, float] | None = None,
) -> str:
    family = batch_value(batch, "family", sample_index, default="unknown")
    event_id = batch_value(batch, "event_id", sample_index, default="?")
    condition_id = batch_value(batch, "condition_id", sample_index, default="?")
    condition_text = " ".join(str(batch_value(batch, "condition_text", sample_index, default="")).split())
    wrapped_condition = "\n".join(
        textwrap.wrap(condition_text, width=38, max_lines=7, placeholder="...")
    )
    lines = [
        f"sample {sample_index + 1}",
        f"{family}",
        f"event {event_id} / condition {condition_id}",
        "",
        wrapped_condition,
    ]
    if diagnostics:
        lines.extend(
            [
                "",
                f"score {diagnostics['sample_score']:.4f}",
                f"final mse {diagnostics['final_position_mse']:.4f}",
                f"obj {int(diagnostics['object_id'])} err {diagnostics['object_final_error']:.3f}",
                (
                    f"obj true ({diagnostics['target_object_x']:.2f}, {diagnostics['target_object_y']:.2f}) "
                    f"pred ({diagnostics['predicted_object_x']:.2f}, {diagnostics['predicted_object_y']:.2f})"
                ),
            ]
        )
        if "target_latent_mse" in diagnostics:
            lines.append(f"latent mse {diagnostics['target_latent_mse']:.4f}")
        if "decoded_final_position_mse" in diagnostics:
            lines.append(
                f"decoded final mse {diagnostics['decoded_final_position_mse']:.4f}"
            )
        if "wall_contact_probability" in diagnostics:
            lines.append(
                f"wall true {diagnostics['wall_contact_true']:.0f} pred {diagnostics['wall_contact_probability']:.2f}"
            )
        if "time_to_contact_abs_error" in diagnostics:
            lines.append(
                f"ttc true {diagnostics['time_to_contact_true']:.1f} "
                f"pred {diagnostics['time_to_contact_predicted']:.1f} "
                f"err {diagnostics['time_to_contact_abs_error']:.1f}"
            )
    return "\n".join(lines)



def batch_value(batch: dict[str, Any], key: str, sample_index: int, default: Any = None) -> Any:
    values = batch.get(key, default)
    if isinstance(values, torch.Tensor):
        value = values[sample_index]
        return value.item() if value.ndim == 0 else value
    if isinstance(values, list):
        return values[sample_index]
    return values


def write_sample_panel_manifest(
    run_dir: Path,
    batch: dict[str, Any],
    sample_count: int,
    diagnostics: list[dict[str, float]] | None = None,
    output_stem: str = "sample_panel",
) -> None:
    lines = ["# Sample Panel Rows", ""]
    for sample_index in range(sample_count):
        row_diagnostics = diagnostics[sample_index] if diagnostics is not None else {}
        lines.extend(
            [
                f"## Sample {sample_index + 1}",
                "",
                f"- family: `{batch_value(batch, 'family', sample_index, default='unknown')}`",
                f"- event_id: `{batch_value(batch, 'event_id', sample_index, default='?')}`",
                f"- condition_id: `{batch_value(batch, 'condition_id', sample_index, default='?')}`",
                f"- condition: {batch_value(batch, 'condition_text', sample_index, default='')}",
            ]
        )
        for key in (
            "sample_score",
            "final_position_mse",
            "object_final_error",
            "heatmap_mse",
            "trajectory_mse",
            "target_latent_mse",
            "decoded_final_position_mse",
            "decoded_heatmap_mse",
            "wall_contact_true",
            "wall_contact_probability",
            "time_to_contact_true",
            "time_to_contact_predicted",
            "time_to_contact_abs_error",
        ):
            if key in row_diagnostics:
                lines.append(f"- {key}: `{row_diagnostics[key]:.6f}`")
        if row_diagnostics:
            lines.extend(
                [
                    (
                        "- target_object_final: "
                        f"`({row_diagnostics['target_object_x']:.6f}, {row_diagnostics['target_object_y']:.6f})`"
                    ),
                    (
                        "- predicted_object_final: "
                        f"`({row_diagnostics['predicted_object_x']:.6f}, {row_diagnostics['predicted_object_y']:.6f})`"
                    ),
                ]
            )
        lines.append("")
    write_text(run_dir / f"{output_stem}_rows.md", "\n".join(lines))


def write_target_sample_panel(
    run_dir: Path,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    output_stem: str = "sample_panel",
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    model.eval()
    with torch.no_grad():
        device_batch = batch_to_device(batch, device)
        predictions = model_forward(model, device_batch)
        reconstruction = predictions.get("target_reconstruction")
        if not isinstance(reconstruction, dict):
            return
        reconstructed_heatmap = prediction_heatmap(reconstruction, config, batch["target_traj"][0])

    context = batch["context_image"][0].permute(1, 2, 0).numpy()
    target_heatmap = target_heatmap_for_panel(batch, config)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(context)
    axes[0].set_title("context")
    axes[1].imshow(target_heatmap, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("target")
    axes[2].imshow(reconstructed_heatmap, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("decoded target")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / f"{output_stem}.png", dpi=140)
    plt.close(fig)


def prediction_heatmap(
    predictions: dict[str, torch.Tensor],
    config: ExperimentConfig,
    target_trajectory: torch.Tensor | None = None,
    sample_index: int = 0,
) -> np.ndarray:
    if "heatmap" in predictions:
        return (
            torch.sigmoid(predictions["heatmap"][sample_index])
            .detach()
            .cpu()
            .reshape(config.heatmap_size, config.heatmap_size)
            .numpy()
        )
    trajectory = predictions.get("trajectory")
    if isinstance(trajectory, torch.Tensor):
        return trajectory_to_heatmap(trajectory[sample_index], config, target_trajectory)
    raise ValueError("Predictions must include either heatmap or trajectory.")


def target_heatmap_for_panel(batch: dict[str, Any], config: ExperimentConfig, sample_index: int = 0) -> np.ndarray:
    return (
        trajectory_to_soft_heatmap(
            batch["target_traj"][sample_index : sample_index + 1],
            heatmap_size=config.heatmap_size,
            max_pucks=config.max_pucks,
        )[0]
        .reshape(config.heatmap_size, config.heatmap_size)
        .numpy()
    )


def trajectory_to_heatmap(
    trajectory: torch.Tensor,
    config: ExperimentConfig,
    target_trajectory: torch.Tensor | None = None,
) -> np.ndarray:
    points = trajectory.detach().cpu().reshape(config.horizon, config.max_pucks, 2).numpy()
    active_pucks = np.ones((config.max_pucks,), dtype=np.bool_)
    if target_trajectory is not None:
        target_points = target_trajectory.detach().cpu().reshape(config.horizon, config.max_pucks, 2).numpy()
        active_pucks = np.abs(target_points).sum(axis=(0, 2)) > 1e-5
    heatmap = np.zeros((config.heatmap_size, config.heatmap_size), dtype=np.float32)
    for frame in points:
        for puck_index, (x, y) in enumerate(frame):
            if not active_pucks[puck_index]:
                continue
            if abs(float(x)) < 1e-5 and abs(float(y)) < 1e-5:
                continue
            px = int(np.clip(round(float(x) * (config.heatmap_size - 1)), 0, config.heatmap_size - 1))
            py = int(np.clip(round(float(y) * (config.heatmap_size - 1)), 0, config.heatmap_size - 1))
            heatmap[py, px] = 1.0
    return heatmap
