from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tepa_eval.engine import (
    build_from_config,
    default_device,
    print_device_report,
    evaluate_loader,
    evaluate_target_loader,
    make_loader,
    make_target_loader,
    parameter_count,
    precompute_condition_latents,
    precompute_target_latents,
    write_sample_panel,
    write_target_sample_panel,
    write_worst_sample_panel,
)
from tepa_eval.io import write_text
from tepa_eval.models import TARGET_MODEL_NAME
from tepa_eval.schemas import ExperimentConfig


def evaluate_run(
    run_dir: str | Path,
    split: str = "val",
    panel_samples: int = 6,
    device_name: str = "auto",
    dataset: str | Path | None = None,
) -> dict[str, float]:
    run_path = Path(run_dir)
    checkpoint_path = _checkpoint_path(run_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.model_validate(checkpoint["config"])
    if dataset is not None:
        config = config.model_copy(update={"output_dir": Path(dataset).expanduser().resolve()})
    device = default_device(device_name)
    print_device_report(device, device_name)
    model = build_from_config(checkpoint["model_name"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _apply_checkpoint_freezes(model, checkpoint)
    include_condition_image = checkpoint["model_name"] == "stuffed_image"
    target_latents_by_event = (
        precompute_target_latents(model, config, device)
        if checkpoint.get("target_frozen") and hasattr(model, "target_encoder")
        else None
    )
    condition_latents_by_condition = (
        precompute_condition_latents(model, config, device, splits=(split,))
        if checkpoint.get("condition_encoder_frozen") and hasattr(model, "condition_encoder")
        else None
    )
    if checkpoint["model_name"] == TARGET_MODEL_NAME:
        loader = make_target_loader(config, split, shuffle=False)
        metrics = evaluate_target_loader(model, loader, config, device)
    else:
        loader = make_loader(
            config,
            split,
            shuffle=False,
            include_condition_image=include_condition_image,
            target_latents_by_event=target_latents_by_event,
            condition_latents_by_condition=condition_latents_by_condition,
        )
        metrics = evaluate_loader(
            model,
            loader,
            config,
            device,
            sigreg_weight=checkpoint.get("sigreg_weight_used"),
        )
    metrics["trainable_params"] = float(parameter_count(model))
    metrics["checkpoint_epoch"] = float(checkpoint.get("epoch", 0))
    (run_path / f"evaluation_{split}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_text(run_path / f"evaluation_{split}.md", _evaluation_markdown(checkpoint["model_name"], split, metrics))
    panel_stem = f"sample_panel_{split}"
    if checkpoint["model_name"] == TARGET_MODEL_NAME:
        write_target_sample_panel(run_path, model, loader, device, config, output_stem=panel_stem)
    else:
        write_sample_panel(
            run_path,
            model,
            loader,
            device,
            config,
            include_decoded_target=True,
            max_samples=panel_samples,
            include_condition_text=True,
            output_stem=panel_stem,
        )
        write_worst_sample_panel(
            run_path,
            model,
            loader,
            device,
            config,
            include_decoded_target=True,
            max_samples=panel_samples,
            include_condition_text=True,
            output_stem=f"{panel_stem}_worst",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved TEPA puck-world run.")
    parser.add_argument("--run", required=True, help="Path to a run directory containing model.pt, best_model.pt, or last_model.pt.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test_templates", "counterfactual"])
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset directory override. Useful for eval-only counterfactual datasets.",
    )
    parser.add_argument("--panel-samples", type=int, default=6, help="Number of rows to render in sample_panel.png.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Evaluation device. Use --device mps to force Apple Silicon GPU and fail if unavailable.",
    )
    args = parser.parse_args()
    metrics = evaluate_run(
        args.run,
        args.split,
        panel_samples=args.panel_samples,
        device_name=args.device,
        dataset=args.dataset,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _evaluation_markdown(model_name: str, split: str, metrics: dict[str, float]) -> str:
    lines = [f"# Evaluation: {model_name} on {split}", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")
    return "\n".join(lines)


def _checkpoint_path(run_path: Path) -> Path:
    for name in ("model.pt", "best_model.pt", "last_model.pt"):
        candidate = run_path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_path}. Expected model.pt, best_model.pt, or last_model.pt.")


def _apply_checkpoint_freezes(model: torch.nn.Module, checkpoint: dict[str, object]) -> None:
    if checkpoint.get("target_frozen"):
        if hasattr(model, "include_target_reconstruction"):
            model.include_target_reconstruction = False
        for module_name in ("target_encoder", "target_decoder"):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    if checkpoint.get("condition_encoder_frozen"):
        condition_encoder = getattr(model, "condition_encoder", None)
        if condition_encoder is not None:
            condition_encoder.eval()
            for parameter in condition_encoder.parameters():
                parameter.requires_grad_(False)


if __name__ == "__main__":
    main()
