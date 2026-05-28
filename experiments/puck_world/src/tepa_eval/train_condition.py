from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from tepa_eval.conditions import DIRECTION_BUCKETS, MAGNITUDE_BUCKETS
from tepa_eval.engine import (
    batch_to_device,
    build_from_config,
    default_device,
    device_report,
    evaluate_condition_loader,
    learning_rate_for_epoch,
    make_loader,
    make_optimizer,
    metric_improved,
    model_forward,
    parameter_count,
    print_device_report,
    set_optimizer_learning_rate,
    should_stop_early,
    write_run_report,
)
from tepa_eval.generate import generate_dataset
from tepa_eval.io import write_jsonl
from tepa_eval.metrics import average_metrics, condition_semantics_loss_and_metrics
from tepa_eval.models import CONDITION_MODEL_NAME
from tepa_eval.schemas import load_config
from tepa_eval.validate_data import validate_dataset


def pretrain_condition(config_path: str | Path, device_name: str = "auto") -> Path:
    config = load_config(config_path)
    if not (config.output_dir / "manifests" / "conditions_train.jsonl").exists():
        generate_dataset(config)
    validate_dataset(config.output_dir)

    torch.manual_seed(config.seed)
    device = default_device(device_name)
    print_device_report(device, device_name)
    run_device_report = device_report(device, device_name)
    model = build_from_config(CONDITION_MODEL_NAME, config).to(device)
    optimizer = make_optimizer(model, config)
    # Keep event fan-outs adjacent so equivalent-condition consistency has useful positive pairs.
    train_loader = make_loader(config, "train", shuffle=False, include_condition_image=False)
    val_loader = make_loader(config, "val", shuffle=False, include_condition_image=False)

    run_dir = config.run_dir / f"{int(time.time())}_{CONDITION_MODEL_NAME}"
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
        for batch in tqdm(train_loader, desc=f"{CONDITION_MODEL_NAME} epoch {epoch + 1}/{config.epochs}"):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model_forward(model, batch)
            loss, metrics = condition_semantics_loss_and_metrics(predictions, batch, config.horizon)
            loss.backward()
            optimizer.step()
            train_rows.append(metrics)

        train_metrics = average_metrics(train_rows)
        val_metrics = evaluate_condition_loader(model, val_loader, config, device)
        row = {
            "epoch": float(epoch + 1),
            "learning_rate": float(learning_rate),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        monitor_name = _resolve_monitor(config.early_stopping_monitor, row, ("val_condition_loss", "val_loss"))
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
        epoch_checkpoint = _condition_checkpoint(
            model=model,
            config=config,
            epoch=epoch + 1,
            history=history,
            device_info=run_device_report,
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
        "train": evaluate_condition_loader(model, train_loader, config, device),
        "val": evaluate_condition_loader(model, val_loader, config, device),
        "train_conditions": float(len(train_loader.dataset)),
        "val_conditions": float(len(val_loader.dataset)),
        "trainable_params": float(parameter_count(model)),
    }
    test_template_path = config.output_dir / "manifests" / "conditions_test_templates.jsonl"
    if test_template_path.exists() and test_template_path.stat().st_size > 0:
        test_loader = make_loader(config, "test_templates", shuffle=False, include_condition_image=False)
        final_metrics["heldout_template"] = evaluate_condition_loader(model, test_loader, config, device)

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
    checkpoint = _condition_checkpoint(
        model=model,
        config=config,
        epoch=best_epoch,
        history=history,
        device_info=run_device_report,
        final_metrics=final_metrics,
        training_summary=training_summary,
    )
    torch.save(checkpoint, run_dir / "model.pt")
    torch.save(checkpoint, run_dir / "best_model.pt")
    (run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_condition_samples(run_dir, model, val_loader, device, max_samples=16)
    write_run_report(run_dir, CONDITION_MODEL_NAME, config, history, final_metrics, training_summary)
    print(f"Saved condition pretraining run to {run_dir} (best epoch {best_epoch})")
    return run_dir


def write_condition_samples(
    run_dir: Path,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    max_samples: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    rows = []
    model.eval()
    with torch.no_grad():
        device_batch = batch_to_device(batch, device)
        predictions = model_forward(model, device_batch)
        count = min(max_samples, int(predictions["object_logits"].shape[0]))
        for index in range(count):
            rows.append(
                {
                    "condition_text": batch["condition_text"][index],
                    "family": batch["family"][index],
                    "event_id": int(batch["event_id"][index]),
                    "target": {
                        "object_id": int(batch["condition_object_id"][index]),
                        "direction": DIRECTION_BUCKETS[int(batch["condition_direction"][index])],
                        "magnitude": MAGNITUDE_BUCKETS[int(batch["condition_magnitude"][index])],
                        "impulse": [
                            float(batch["condition_impulse"][index][0]),
                            float(batch["condition_impulse"][index][1]),
                        ],
                        "horizon": float(batch["condition_horizon"][index][0]),
                    },
                    "prediction": {
                        "object_id": int(predictions["object_logits"][index].argmax().detach().cpu()),
                        "direction": DIRECTION_BUCKETS[
                            int(predictions["direction_logits"][index].argmax().detach().cpu())
                        ],
                        "magnitude": MAGNITUDE_BUCKETS[
                            int(predictions["magnitude_logits"][index].argmax().detach().cpu())
                        ],
                        "impulse": [
                            float(predictions["impulse"][index][0].detach().cpu()),
                            float(predictions["impulse"][index][1].detach().cpu()),
                        ],
                        "horizon": float(predictions["horizon"][index][0].detach().cpu()),
                    },
                    "absolute_error": {
                        "impulse_dx": abs(
                            float(predictions["impulse"][index][0].detach().cpu())
                            - float(batch["condition_impulse"][index][0])
                        ),
                        "impulse_dy": abs(
                            float(predictions["impulse"][index][1].detach().cpu())
                            - float(batch["condition_impulse"][index][1])
                        ),
                        "horizon": abs(
                            float(predictions["horizon"][index][0].detach().cpu())
                            - float(batch["condition_horizon"][index][0])
                        ),
                    },
                }
            )
    write_jsonl(run_dir / "condition_samples.jsonl", rows)


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


def _condition_checkpoint(
    *,
    model: torch.nn.Module,
    config: Any,
    epoch: int,
    history: list[dict[str, float]],
    device_info: dict[str, Any],
    final_metrics: dict[str, dict[str, float] | float] | None = None,
    training_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "model_name": CONDITION_MODEL_NAME,
        "model_state": model.state_dict(),
        "condition_encoder_state": model.condition_encoder.state_dict(),
        "config": config.model_dump(mode="json"),
        "epoch": epoch,
        "history": history,
        "device": device_info,
    }
    if final_metrics is not None:
        checkpoint["final_metrics"] = final_metrics
    if training_summary is not None:
        checkpoint["training_summary"] = training_summary
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the puck-world condition encoder.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device. Use --device mps to force Apple Silicon GPU and fail if unavailable.",
    )
    args = parser.parse_args()
    pretrain_condition(args.config, device_name=args.device)


if __name__ == "__main__":
    main()
