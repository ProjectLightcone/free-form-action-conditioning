from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tepa_eval.dataset import PuckDataset, collate_metadata
from tepa_eval.engine import (
    build_from_config,
    default_device,
    device_report,
    parameter_count,
    precompute_target_latents,
    print_device_report,
)
from tepa_eval.io import write_text
from tepa_eval.metrics import average_metrics, loss_and_metrics
from tepa_eval.schemas import ExperimentConfig


def benchmark_counterfactuals(
    tepa_run: str | Path,
    fused_run: str | Path,
    dataset: str | Path,
    split: str = "counterfactual",
    conditions_per_scene: tuple[int, ...] = (1, 4, 16, 64, 256),
    max_scenes: int | None = None,
    output_dir: str | Path = "reports/counterfactual_benchmarks",
    device_name: str = "auto",
    include_shuffle: bool = True,
) -> Path:
    device = default_device(device_name)
    print_device_report(device, device_name)
    dataset_path = Path(dataset).expanduser().resolve()
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    run_output = output_path / f"{int(time.time())}_counterfactual_benchmark"
    run_output.mkdir(parents=True, exist_ok=True)

    tepa = _load_benchmark_model(tepa_run, dataset_path, split, device)
    fused = _load_benchmark_model(fused_run, dataset_path, split, device)

    results = {
        "dataset": str(dataset_path),
        "split": split,
        "conditions_per_scene": list(conditions_per_scene),
        "max_scenes": max_scenes,
        "device": device_report(device, device_name),
        "models": {
            "tepa": _benchmark_model(
                label="tepa",
                bundle=tepa,
                split=split,
                conditions_per_scene=conditions_per_scene,
                max_scenes=max_scenes,
                device=device,
                include_shuffle=include_shuffle,
            ),
            "fused": _benchmark_model(
                label="fused",
                bundle=fused,
                split=split,
                conditions_per_scene=conditions_per_scene,
                max_scenes=max_scenes,
                device=device,
                include_shuffle=include_shuffle,
            ),
        },
    }
    results["comparison"] = _comparison_rows(results["models"]["tepa"]["rows"], results["models"]["fused"]["rows"])

    (run_output / "benchmark_counterfactuals.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_text(run_output / "benchmark_counterfactuals.md", _markdown_report(results))
    print(json.dumps(_compact_summary(results), indent=2, sort_keys=True))
    return run_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark amortized counterfactual inference.")
    parser.add_argument("--tepa-run", required=True, help="Run directory for a TEPA latent model.")
    parser.add_argument("--fused-run", required=True, help="Run directory for the fused latent transformer baseline.")
    parser.add_argument("--dataset", required=True, help="Eval-only counterfactual dataset directory.")
    parser.add_argument("--split", default="counterfactual", help="Dataset split to benchmark.")
    parser.add_argument(
        "--conditions-per-scene",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 256],
        help="K values: number of condition rows evaluated for each scene.",
    )
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap for quicker benchmark runs.")
    parser.add_argument("--output-dir", default="reports/counterfactual_benchmarks")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Benchmark device. Use --device mps to force Apple Silicon GPU.",
    )
    parser.add_argument("--skip-shuffle", action="store_true", help="Skip condition-shuffle degradation metrics.")
    args = parser.parse_args()
    output = benchmark_counterfactuals(
        tepa_run=args.tepa_run,
        fused_run=args.fused_run,
        dataset=args.dataset,
        split=args.split,
        conditions_per_scene=tuple(args.conditions_per_scene),
        max_scenes=args.max_scenes,
        output_dir=args.output_dir,
        device_name=args.device,
        include_shuffle=not args.skip_shuffle,
    )
    print(f"Saved benchmark to {output}")


class _BenchmarkBundle:
    def __init__(
        self,
        run_dir: Path,
        checkpoint: dict[str, Any],
        config: ExperimentConfig,
        model: torch.nn.Module,
        dataset: PuckDataset,
    ) -> None:
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.config = config
        self.model = model
        self.dataset = dataset


def _load_benchmark_model(run_dir: str | Path, dataset_path: Path, split: str, device: torch.device) -> _BenchmarkBundle:
    run_path = Path(run_dir).expanduser().resolve()
    checkpoint = torch.load(_checkpoint_path(run_path), map_location="cpu")
    config = ExperimentConfig.model_validate(checkpoint["config"]).model_copy(update={"output_dir": dataset_path})
    model = build_from_config(checkpoint["model_name"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _apply_checkpoint_freezes(model, checkpoint)
    model.eval()

    if not hasattr(model, "target_encoder") or not hasattr(model, "target_decoder"):
        raise ValueError(f"{checkpoint['model_name']} is not a latent model with a target side.")

    started = time.perf_counter()
    target_latents = precompute_target_latents(model, config, device)
    _synchronize(device)
    checkpoint["benchmark_target_latent_precompute_seconds"] = time.perf_counter() - started
    dataset = PuckDataset(
        config.output_dir,
        split=split,
        max_text_len=config.max_text_len,
        include_condition_image=False,
        target_latents_by_event=target_latents,
        condition_family_filter=set(config.condition_family_filter) if config.condition_family_filter else None,
    )
    return _BenchmarkBundle(run_path, checkpoint, config, model, dataset)


def _benchmark_model(
    label: str,
    bundle: _BenchmarkBundle,
    split: str,
    conditions_per_scene: tuple[int, ...],
    max_scenes: int | None,
    device: torch.device,
    include_shuffle: bool,
) -> dict[str, Any]:
    scene_groups = _scene_index_groups(bundle.dataset)
    scene_hashes = sorted(scene_groups)
    if max_scenes is not None:
        scene_hashes = scene_hashes[:max_scenes]
    if not scene_hashes:
        raise ValueError(f"No scenes found for split {split!r}.")

    max_k = max(conditions_per_scene)
    available = min(len(scene_groups[scene_hash]) for scene_hash in scene_hashes)
    if max_k > available:
        raise ValueError(f"Requested K={max_k}, but the smallest selected scene has only {available} rows.")

    rows = []
    for k in conditions_per_scene:
        _warmup(bundle.model, bundle.dataset, scene_groups[scene_hashes[0]], k, bundle.config, device)
        rows.append(
            _benchmark_k(
                model=bundle.model,
                dataset=bundle.dataset,
                scene_groups=scene_groups,
                scene_hashes=scene_hashes,
                k=k,
                config=bundle.config,
                device=device,
                include_shuffle=include_shuffle,
            )
        )

    return {
        "label": label,
        "run_dir": str(bundle.run_dir),
        "model_name": str(bundle.checkpoint["model_name"]),
        "checkpoint_epoch": float(bundle.checkpoint.get("epoch", 0)),
        "target_latent_precompute_seconds": float(bundle.checkpoint["benchmark_target_latent_precompute_seconds"]),
        "trainable_params": float(parameter_count(bundle.model)),
        "scene_count": len(scene_hashes),
        "rows": rows,
    }


def _benchmark_k(
    model: torch.nn.Module,
    dataset: PuckDataset,
    scene_groups: dict[str, list[int]],
    scene_hashes: list[str],
    k: int,
    config: ExperimentConfig,
    device: torch.device,
    include_shuffle: bool,
) -> dict[str, float]:
    metric_rows: list[dict[str, float]] = []
    shuffle_rows: list[dict[str, float]] = []
    elapsed_seconds = 0.0
    condition_rows = 0
    memory_before = _allocated_memory_bytes(device)

    with torch.inference_mode():
        for scene_hash in scene_hashes:
            selected = _select_evenly_spaced_indices(scene_groups[scene_hash], k)
            batch = collate_metadata([dataset[index] for index in selected])
            batch = _batch_to_device(batch, device)
            _synchronize(device)
            started = time.perf_counter()
            predictions = _forward_prediction_only(model, batch)
            _synchronize(device)
            elapsed_seconds += time.perf_counter() - started
            condition_rows += k

            _, metrics = loss_and_metrics(
                predictions,
                batch,
                config.horizon,
                prediction_loss_weight=config.prediction_loss_weight,
                latent_loss_weight=config.latent_loss_weight,
                target_reconstruction_loss_weight=0.0,
                context_state_loss_weight=0.0,
            )
            metric_rows.append(metrics)

            if include_shuffle and k > 1:
                shuffled = _shuffle_conditions(batch)
                shuffled_predictions = _forward_prediction_only(model, shuffled)
                _, shuffled_metrics = loss_and_metrics(
                    shuffled_predictions,
                    shuffled,
                    config.horizon,
                    prediction_loss_weight=config.prediction_loss_weight,
                    latent_loss_weight=config.latent_loss_weight,
                    target_reconstruction_loss_weight=0.0,
                    context_state_loss_weight=0.0,
                )
                shuffle_rows.append(shuffled_metrics)

    memory_after = _allocated_memory_bytes(device)
    metrics = average_metrics(metric_rows)
    row: dict[str, float] = {
        "conditions_per_scene": float(k),
        "scene_count": float(len(scene_hashes)),
        "condition_rows": float(condition_rows),
        "inference_seconds": float(elapsed_seconds),
        "latency_per_scene_ms": float(1000.0 * elapsed_seconds / max(len(scene_hashes), 1)),
        "latency_per_condition_ms": float(1000.0 * elapsed_seconds / max(condition_rows, 1)),
        "conditions_per_second": float(condition_rows / max(elapsed_seconds, 1.0e-9)),
        "allocated_memory_before_mb": float(memory_before / 1_000_000.0) if memory_before is not None else 0.0,
        "allocated_memory_after_mb": float(memory_after / 1_000_000.0) if memory_after is not None else 0.0,
        **metrics,
    }
    if shuffle_rows:
        shuffle_metrics = average_metrics(shuffle_rows)
        row["condition_shuffle_prediction_loss"] = shuffle_metrics.get("prediction_loss", 0.0)
        row["condition_shuffle_target_latent_mse"] = shuffle_metrics.get("target_latent_mse", 0.0)
        row["condition_shuffle_prediction_loss_degradation"] = (
            row["condition_shuffle_prediction_loss"] - row.get("prediction_loss", 0.0)
        )
        row["condition_shuffle_target_latent_mse_degradation"] = (
            row["condition_shuffle_target_latent_mse"] - row.get("target_latent_mse", 0.0)
        )
    return row


def _forward_prediction_only(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if hasattr(model, "encode_context_memory") and hasattr(model, "predict_from_memory"):
        context_memory = model.encode_context_memory(batch["context_image"][:1])
        z_hat_target = model.predict_from_memory(
            context_memory,
            text_tokens=batch.get("text_tokens"),
            z_condition=batch.get("z_condition"),
        )
    elif hasattr(model, "encode_fused"):
        z_context_condition = model.encode_fused(batch["context_image"], batch["text_tokens"])
        z_hat_target = model.predictor(z_context_condition)
    elif hasattr(model, "context_encoder") and hasattr(model, "condition_encoder"):
        z_context = model.context_encoder(batch["context_image"][:1])
        if "z_condition" in batch:
            z_condition = batch["z_condition"]
        else:
            z_condition = model.condition_encoder(batch["text_tokens"])
        z_context = z_context.expand(z_condition.shape[0], -1)
        z_hat_target = model.predictor(torch.cat([z_context, z_condition], dim=-1))
    else:
        raise ValueError(f"Unsupported benchmark model: {type(model).__name__}")

    if bool(getattr(model, "normalize_latents", False)):
        z_hat_target = F.normalize(z_hat_target, dim=-1)
    predictions = model.target_decoder(z_hat_target)
    predictions["z_hat_target"] = z_hat_target
    if "z_target" in batch:
        predictions["z_target"] = batch["z_target"]
    return predictions


def _shuffle_conditions(batch: dict[str, Any]) -> dict[str, Any]:
    shuffled = dict(batch)
    row_count = batch["target_final_pos"].shape[0]
    order = torch.roll(torch.arange(row_count, device=batch["target_final_pos"].device), shifts=1)
    for key in ("text_tokens", "z_condition", "condition_params"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == row_count:
            shuffled[key] = value[order]
    return shuffled


def _warmup(
    model: torch.nn.Module,
    dataset: PuckDataset,
    scene_indices: list[int],
    k: int,
    config: ExperimentConfig,
    device: torch.device,
) -> None:
    selected = _select_evenly_spaced_indices(scene_indices, k)
    batch = collate_metadata([dataset[index] for index in selected])
    batch = _batch_to_device(batch, device)
    with torch.inference_mode():
        for _ in range(2):
            _forward_prediction_only(model, batch)
            _synchronize(device)


def _scene_index_groups(dataset: PuckDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.conditions):
        groups[str(row["scene_hash"])].append(index)
    return dict(groups)


def _select_evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if count > len(indices):
        raise ValueError(f"Requested {count} rows, but only {len(indices)} are available.")
    if count == len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return [indices[int(position)] for position in positions]


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


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


def _comparison_rows(tepa_rows: list[dict[str, float]], fused_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    fused_by_k = {int(row["conditions_per_scene"]): row for row in fused_rows}
    rows = []
    for tepa in tepa_rows:
        k = int(tepa["conditions_per_scene"])
        fused = fused_by_k[k]
        rows.append(
            {
                "conditions_per_scene": float(k),
                "tepa_speedup_over_fused": fused["latency_per_condition_ms"]
                / max(tepa["latency_per_condition_ms"], 1.0e-9),
                "fused_to_tepa_latency_ratio": fused["latency_per_condition_ms"]
                / max(tepa["latency_per_condition_ms"], 1.0e-9),
                "tepa_latency_per_condition_ms": tepa["latency_per_condition_ms"],
                "fused_latency_per_condition_ms": fused["latency_per_condition_ms"],
                "tepa_target_latent_mse": tepa.get("target_latent_mse", 0.0),
                "fused_target_latent_mse": fused.get("target_latent_mse", 0.0),
                "tepa_final_position_mse": tepa.get("final_position_mse", 0.0),
                "fused_final_position_mse": fused.get("final_position_mse", 0.0),
                "tepa_prediction_loss": tepa.get("prediction_loss", 0.0),
                "fused_prediction_loss": fused.get("prediction_loss", 0.0),
            }
        )
    return rows


def _compact_summary(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": "benchmark_counterfactuals.json",
        "comparison": results["comparison"],
    }


def _markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Counterfactual Benchmark",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- Split: `{results['split']}`",
        f"- Device: `{results['device']['selected_device']}`",
        "",
        "## Latency And Quality",
        "",
        "| K | TEPA ms/condition | Fused ms/condition | TEPA speedup over fused | TEPA latent MSE | Fused latent MSE | TEPA final MSE | Fused final MSE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["comparison"]:
        lines.append(
            "| {k:.0f} | {tepa_ms:.4f} | {fused_ms:.4f} | {speedup:.3f} | {tepa_latent:.4f} | {fused_latent:.4f} | {tepa_final:.4f} | {fused_final:.4f} |".format(
                k=row["conditions_per_scene"],
                tepa_ms=row["tepa_latency_per_condition_ms"],
                fused_ms=row["fused_latency_per_condition_ms"],
                speedup=row["tepa_speedup_over_fused"],
                tepa_latent=row["tepa_target_latent_mse"],
                fused_latent=row["fused_target_latent_mse"],
                tepa_final=row["tepa_final_position_mse"],
                fused_final=row["fused_final_position_mse"],
            )
        )
    lines.append("")
    for label, model_results in results["models"].items():
        lines.extend(
            [
                f"## {label.title()}",
                "",
                f"- Model: `{model_results['model_name']}`",
                f"- Run: `{model_results['run_dir']}`",
                f"- Target latent precompute seconds: {model_results['target_latent_precompute_seconds']:.3f}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
