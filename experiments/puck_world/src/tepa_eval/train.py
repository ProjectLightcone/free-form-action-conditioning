from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from tepa_eval.engine import (
    build_from_config,
    default_device,
    device_report,
    equivalent_condition_consistency,
    evaluate_loader,
    learning_rate_for_epoch,
    make_loader,
    make_optimizer,
    metric_improved,
    model_forward,
    parameter_count,
    print_device_report,
    precompute_condition_latents,
    precompute_target_latents,
    set_optimizer_learning_rate,
    should_stop_early,
    write_run_report,
    write_sample_panel,
)
from tepa_eval.generate import generate_dataset
from tepa_eval.io import write_jsonl
from tepa_eval.metrics import average_metrics, loss_and_metrics
from tepa_eval.models import CONDITION_MODEL_NAME, PREDICTION_MODEL_NAMES, TARGET_MODEL_NAME
from tepa_eval.schemas import load_config
from tepa_eval.validate_data import validate_dataset


def train_model(
    config_path: str | Path,
    model_name: str,
    target_checkpoint: str | Path | None = None,
    freeze_target: bool = False,
    condition_checkpoint: str | Path | None = None,
    freeze_condition_encoder: bool = False,
    device_name: str = "auto",
    seed_override: int | None = None,
) -> Path:
    config = load_config(config_path)
    if seed_override is not None:
        config = config.model_copy(update={"seed": int(seed_override)})
    if not (config.output_dir / "manifests" / "conditions_train.jsonl").exists():
        generate_dataset(config)
    validate_dataset(config.output_dir)

    torch.manual_seed(config.seed)
    device = default_device(device_name)
    print_device_report(device, device_name)
    run_device_report = device_report(device, device_name)
    model = build_from_config(model_name, config).to(device)
    target_checkpoint_path = Path(target_checkpoint).expanduser().resolve() if target_checkpoint else None
    if target_checkpoint_path is not None:
        load_target_side(model, target_checkpoint_path, freeze_target=freeze_target)
    condition_checkpoint_path = Path(condition_checkpoint).expanduser().resolve() if condition_checkpoint else None
    if condition_checkpoint_path is not None:
        load_condition_encoder(model, condition_checkpoint_path, freeze_condition_encoder=freeze_condition_encoder)
    effective_sigreg_weight = 0.0 if freeze_target else config.sigreg_weight
    include_condition_image = model_name == "stuffed_image"
    target_latents_by_event = (
        precompute_target_latents(model, config, device)
        if freeze_target and hasattr(model, "target_encoder")
        else None
    )
    condition_latents_by_condition = (
        precompute_condition_latents(model, config, device)
        if freeze_condition_encoder and hasattr(model, "condition_encoder")
        else None
    )
    optimizer = make_optimizer(model, config)
    train_loader = make_loader(
        config,
        "train",
        shuffle=True,
        include_condition_image=include_condition_image,
        target_latents_by_event=target_latents_by_event,
        condition_latents_by_condition=condition_latents_by_condition,
    )
    val_loader = make_loader(
        config,
        "val",
        shuffle=False,
        include_condition_image=include_condition_image,
        target_latents_by_event=target_latents_by_event,
        condition_latents_by_condition=condition_latents_by_condition,
    )

    run_dir = config.run_dir / f"{int(time.time())}_{model_name}_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "device.json").write_text(
        json.dumps(run_device_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    history: list[dict[str, float]] = []
    best_value: float | None = None
    best_epoch = 0
    best_row: dict[str, float] | None = None
    best_monitor_name = config.early_stopping_monitor
    epochs_without_improvement = 0
    early_stopped = False
    stop_reason = "completed"

    for epoch in range(config.epochs):
        learning_rate = learning_rate_for_epoch(config, epoch)
        set_optimizer_learning_rate(optimizer, learning_rate)
        model.train()
        train_rows = []
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch + 1}/{config.epochs}"):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            predictions = model_forward(model, batch)
            loss, metrics = loss_and_metrics(
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
            loss.backward()
            optimizer.step()
            train_rows.append(metrics)

        train_metrics = average_metrics(train_rows)
        val_metrics = evaluate_loader(model, val_loader, config, device, sigreg_weight=effective_sigreg_weight)
        row = {
            "epoch": float(epoch + 1),
            "learning_rate": float(learning_rate),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        monitor_name = _resolve_monitor(config.early_stopping_monitor, row, ("val_prediction_loss", "val_loss"))
        monitor_value = _monitor_value(row, monitor_name)
        is_best = metric_improved(monitor_value, best_value, config.early_stopping_min_delta)
        if is_best:
            best_value = monitor_value
            best_epoch = epoch + 1
            best_monitor_name = monitor_name
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row.update(
            {
                "monitor_value": monitor_value,
                "best_epoch_so_far": float(best_epoch),
                "epochs_without_improvement": float(epochs_without_improvement),
                "is_best": float(is_best),
            }
        )
        if is_best:
            best_row = dict(row)
        history.append(row)
        write_jsonl(run_dir / "metrics.jsonl", history)
        epoch_checkpoint = _prediction_checkpoint(
            model_name=model_name,
            model=model,
            config=config,
            epoch=epoch + 1,
            history=history,
            device_info=run_device_report,
            target_checkpoint_path=target_checkpoint_path,
            freeze_target=freeze_target,
            condition_checkpoint_path=condition_checkpoint_path,
            freeze_condition_encoder=freeze_condition_encoder,
            effective_sigreg_weight=effective_sigreg_weight,
        )
        torch.save(epoch_checkpoint, run_dir / "last_model.pt")
        if is_best:
            torch.save(epoch_checkpoint, run_dir / "best_model.pt")

        if should_stop_early(config, epoch + 1, epochs_without_improvement):
            early_stopped = True
            stop_reason = "early_stopping"
            break

    if best_epoch > 0:
        best_checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state"])

    final_metrics: dict[str, dict[str, float] | float] = {
        "val": evaluate_loader(model, val_loader, config, device, sigreg_weight=effective_sigreg_weight),
        "val_condition_shuffle": evaluate_loader(
            model,
            val_loader,
            config,
            device,
            shuffle_conditions=True,
            sigreg_weight=effective_sigreg_weight,
        ),
        "train_equivalent_condition_consistency": equivalent_condition_consistency(model, train_loader, device),
        "trainable_params": float(parameter_count(model)),
    }
    test_template_path = config.output_dir / "manifests" / "conditions_test_templates.jsonl"
    if test_template_path.exists() and test_template_path.stat().st_size > 0:
        test_loader = make_loader(
            config,
            "test_templates",
            shuffle=False,
            include_condition_image=include_condition_image,
            target_latents_by_event=target_latents_by_event,
            condition_latents_by_condition=condition_latents_by_condition,
        )
        final_metrics["heldout_template"] = evaluate_loader(
            model,
            test_loader,
            config,
            device,
            sigreg_weight=effective_sigreg_weight,
        )
    training_summary = _training_summary(
        config=config,
        history=history,
        best_epoch=best_epoch,
        monitor=best_monitor_name,
        best_value=best_value,
        best_row=best_row,
        early_stopped=early_stopped,
        stop_reason=stop_reason,
    )
    final_metrics["run"] = _numeric_training_summary(training_summary)

    checkpoint = _prediction_checkpoint(
        model_name=model_name,
        model=model,
        config=config,
        epoch=best_epoch,
        history=history,
        device_info=run_device_report,
        final_metrics=final_metrics,
        training_summary=training_summary,
        target_checkpoint_path=target_checkpoint_path,
        freeze_target=freeze_target,
        condition_checkpoint_path=condition_checkpoint_path,
        freeze_condition_encoder=freeze_condition_encoder,
        effective_sigreg_weight=effective_sigreg_weight,
    )
    torch.save(checkpoint, run_dir / "model.pt")
    torch.save(checkpoint, run_dir / "best_model.pt")
    if condition_checkpoint_path is not None and not freeze_condition_encoder and hasattr(model, "condition_encoder"):
        torch.save(
            {
                "model_name": CONDITION_MODEL_NAME,
                "checkpoint_kind": "fine_tuned_condition_encoder",
                "source_condition_checkpoint": str(condition_checkpoint_path),
                "parent_run": str(run_dir),
                "condition_encoder_state": getattr(model, "condition_encoder").state_dict(),
                "config": config.model_dump(mode="json"),
                "epoch": best_epoch,
                "history": history,
            },
            run_dir / "condition_encoder_finetuned.pt",
        )
    (run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_run_report(run_dir, model_name, config, history, final_metrics, training_summary)
    write_sample_panel(run_dir, model, val_loader, device, config)
    print(f"Saved run to {run_dir} (best epoch {best_epoch})")
    return run_dir


def _monitor_value(row: dict[str, float], monitor: str) -> float:
    try:
        return float(row[monitor])
    except KeyError as exc:
        available = ", ".join(sorted(row))
        raise KeyError(f"Early stopping monitor {monitor!r} was not found. Available metrics: {available}") from exc


def _resolve_monitor(requested: str, row: dict[str, float], preferred: tuple[str, ...]) -> str:
    if requested != "auto":
        return requested
    for candidate in preferred:
        if candidate in row:
            return candidate
    available = ", ".join(sorted(row))
    raise KeyError(f"Could not resolve automatic early stopping monitor. Available metrics: {available}")


def _training_summary(
    *,
    config: Any,
    history: list[dict[str, float]],
    best_epoch: int,
    monitor: str,
    best_value: float | None,
    best_row: dict[str, float] | None,
    early_stopped: bool,
    stop_reason: str,
) -> dict[str, Any]:
    return {
        "requested_epochs": float(config.epochs),
        "completed_epochs": float(len(history)),
        "best_epoch": float(best_epoch),
        "monitor": monitor,
        "configured_monitor": config.early_stopping_monitor,
        "best_monitor_value": best_value,
        "early_stopping_enabled": config.early_stopping_enabled,
        "early_stopping_min_delta": float(config.early_stopping_min_delta),
        "early_stopping_patience": float(config.early_stopping_patience),
        "early_stopping_min_epochs": float(config.early_stopping_min_epochs),
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "best_row": best_row,
    }


def _numeric_training_summary(summary: dict[str, Any]) -> dict[str, float]:
    best_monitor_value = summary["best_monitor_value"]
    return {
        "requested_epochs": float(summary["requested_epochs"]),
        "completed_epochs": float(summary["completed_epochs"]),
        "best_epoch": float(summary["best_epoch"]),
        "best_monitor_value": float(best_monitor_value) if best_monitor_value is not None else float("nan"),
        "early_stopped": 1.0 if summary["early_stopped"] else 0.0,
    }


def _prediction_checkpoint(
    *,
    model_name: str,
    model: torch.nn.Module,
    config: Any,
    epoch: int,
    history: list[dict[str, float]],
    device_info: dict[str, Any],
    target_checkpoint_path: Path | None,
    freeze_target: bool,
    condition_checkpoint_path: Path | None,
    freeze_condition_encoder: bool,
    effective_sigreg_weight: float,
    final_metrics: dict[str, dict[str, float] | float] | None = None,
    training_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "model_name": model_name,
        "model_state": model.state_dict(),
        "config": config.model_dump(mode="json"),
        "epoch": epoch,
        "history": history,
        "device": device_info,
        "target_checkpoint": str(target_checkpoint_path) if target_checkpoint_path is not None else None,
        "target_frozen": freeze_target,
        "condition_checkpoint": str(condition_checkpoint_path) if condition_checkpoint_path is not None else None,
        "condition_encoder_frozen": freeze_condition_encoder,
        "sigreg_weight_used": effective_sigreg_weight,
    }
    if final_metrics is not None:
        checkpoint["final_metrics"] = final_metrics
    if training_summary is not None:
        checkpoint["training_summary"] = training_summary
    return checkpoint


def load_target_side(model: torch.nn.Module, checkpoint_path: Path, freeze_target: bool) -> None:
    if not hasattr(model, "target_encoder") or not hasattr(model, "target_decoder"):
        raise ValueError("Target checkpoint loading is only supported for models with a target side.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") != TARGET_MODEL_NAME:
        raise ValueError(f"Expected a {TARGET_MODEL_NAME} checkpoint, got {checkpoint.get('model_name')!r}.")

    target_state = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith("target_encoder.") or key.startswith("target_decoder.")
    }
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    relevant_missing = [
        key
        for key in missing
        if key.startswith("target_encoder.") or key.startswith("target_decoder.")
    ]
    if relevant_missing or unexpected:
        raise ValueError(
            "Target checkpoint did not match the model target side: "
            f"missing={relevant_missing}, unexpected={unexpected}"
        )

    if freeze_target:
        if hasattr(model, "include_target_reconstruction"):
            model.include_target_reconstruction = False
        for module_name in ("target_encoder", "target_decoder"):
            module = getattr(model, module_name)
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def load_condition_encoder(
    model: torch.nn.Module,
    checkpoint_path: Path,
    freeze_condition_encoder: bool,
) -> None:
    if not hasattr(model, "condition_encoder"):
        raise ValueError("Condition checkpoint loading is only supported for models with a condition_encoder.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") != CONDITION_MODEL_NAME:
        raise ValueError(f"Expected a {CONDITION_MODEL_NAME} checkpoint, got {checkpoint.get('model_name')!r}.")

    condition_state = checkpoint.get("condition_encoder_state")
    if condition_state is None:
        condition_state = {
            key.removeprefix("condition_encoder."): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("condition_encoder.")
        }
    condition_encoder = getattr(model, "condition_encoder")
    condition_encoder.load_state_dict(condition_state)

    if freeze_condition_encoder:
        condition_encoder.eval()
        for parameter in condition_encoder.parameters():
            parameter.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TEPA puck-world model.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--model", required=True, choices=PREDICTION_MODEL_NAMES)
    parser.add_argument(
        "--target-checkpoint",
        help="Optional target_autoencoder model.pt used to initialize a latent TEPA target side.",
    )
    parser.add_argument(
        "--freeze-target",
        action="store_true",
        help="Freeze the loaded target encoder/decoder and train only the context-condition predictor side.",
    )
    parser.add_argument(
        "--condition-checkpoint",
        help="Optional condition_semantics model.pt used to initialize the TEPA condition encoder.",
    )
    parser.add_argument(
        "--freeze-condition-encoder",
        action="store_true",
        help="Freeze the loaded condition encoder during prediction training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device. Use --device mps to force Apple Silicon GPU and fail if unavailable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional training seed override. The frozen dataset is reused when it already exists.",
    )
    args = parser.parse_args()
    train_model(
        args.config,
        args.model,
        args.target_checkpoint,
        freeze_target=args.freeze_target,
        condition_checkpoint=args.condition_checkpoint,
        freeze_condition_encoder=args.freeze_condition_encoder,
        device_name=args.device,
        seed_override=args.seed,
    )


if __name__ == "__main__":
    main()
