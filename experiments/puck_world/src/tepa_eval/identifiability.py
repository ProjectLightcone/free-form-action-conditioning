from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tepa_eval.dataset import PuckDataset, collate_metadata
from tepa_eval.engine import (
    batch_to_device,
    build_from_config,
    default_device,
    device_report,
    model_forward,
    precompute_condition_latents,
    precompute_target_latents,
    print_device_report,
)
from tepa_eval.io import write_text
from tepa_eval.schemas import ExperimentConfig


DEFAULT_PROBE_TARGETS = (
    "context_state",
    "condition_params",
    "outcome_final_pos",
    "outcome_event",
    "trajectory_summary",
)


def write_identifiability_report(
    run_dir: str | Path,
    split: str = "val",
    dataset: str | Path | None = None,
    device_name: str = "auto",
    max_samples: int = 4096,
    output_dir: str | Path | None = None,
    ridge_alpha: float = 1.0e-3,
) -> Path:
    run_path = Path(run_dir).expanduser().resolve()
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

    result = identifiability_report(
        model=model,
        config=config,
        checkpoint=checkpoint,
        run_dir=run_path,
        split=split,
        device=device,
        device_name=device_name,
        max_samples=max_samples,
        ridge_alpha=ridge_alpha,
    )

    output_path = Path(output_dir).expanduser().resolve() if output_dir else run_path
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"identifiability_{split}"
    json_path = output_path / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_text(output_path / f"{stem}.md", _markdown_report(result))
    print(json.dumps(_compact_summary(result), indent=2, sort_keys=True))
    return json_path


def identifiability_report(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    checkpoint: dict[str, Any],
    run_dir: Path,
    split: str,
    device: torch.device,
    device_name: str,
    max_samples: int,
    ridge_alpha: float,
) -> dict[str, Any]:
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
    dataset = PuckDataset(
        config.output_dir,
        split=split,
        max_text_len=config.max_text_len,
        include_condition_image=include_condition_image,
        target_latents_by_event=target_latents_by_event,
        condition_latents_by_condition=condition_latents_by_condition,
        condition_family_filter=set(config.condition_family_filter) if config.condition_family_filter else None,
    )
    rows = _collect_rows(model, dataset, config, device, max_samples=max_samples)
    if rows["z_target"].shape[0] < 8:
        raise ValueError("Identifiability report requires at least 8 samples.")

    sample_count = int(rows["z_target"].shape[0])
    train_indices, test_indices = _probe_split_indices(sample_count)
    latents = {
        "target": rows["z_target"],
        "prediction": rows["z_hat_target"],
    }
    probes: dict[str, Any] = {}
    for latent_name, latent_values in latents.items():
        probes[latent_name] = {}
        for target_name in DEFAULT_PROBE_TARGETS:
            probes[latent_name][target_name] = _ridge_probe(
                latent_values,
                rows[target_name],
                train_indices=train_indices,
                test_indices=test_indices,
                alpha=ridge_alpha,
            )

    alignment = {
        "direct_mse": float(np.mean(np.square(rows["z_hat_target"] - rows["z_target"]))),
        "mean_cosine": _mean_cosine(rows["z_hat_target"], rows["z_target"]),
        "linear_r2_prediction_to_target": _ridge_probe(
            rows["z_hat_target"],
            rows["z_target"],
            train_indices=train_indices,
            test_indices=test_indices,
            alpha=ridge_alpha,
        )["r2"],
        "orthogonal_procrustes_r2_prediction_to_target": _orthogonal_procrustes(
            rows["z_hat_target"],
            rows["z_target"],
            train_indices=train_indices,
            test_indices=test_indices,
        )["r2"],
    }

    return {
        "run_dir": str(run_dir),
        "model_name": str(checkpoint["model_name"]),
        "dataset": str(config.output_dir),
        "split": split,
        "sample_count": sample_count,
        "max_samples": int(max_samples),
        "ridge_alpha": float(ridge_alpha),
        "device": device_report(device, device_name),
        "latent_distribution": {
            "target": _latent_distribution(rows["z_target"]),
            "prediction": _latent_distribution(rows["z_hat_target"]),
        },
        "alignment": alignment,
        "linear_probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write latent identifiability diagnostics for a saved run.")
    parser.add_argument("--run", required=True, help="Run directory containing model.pt, best_model.pt, or last_model.pt.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test_templates", "counterfactual"])
    parser.add_argument("--dataset", default=None, help="Optional dataset directory override.")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to the run directory.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Evaluation device. Use --device mps to force Apple Silicon GPU.",
    )
    args = parser.parse_args()
    output = write_identifiability_report(
        run_dir=args.run,
        split=args.split,
        dataset=args.dataset,
        device_name=args.device,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        ridge_alpha=args.ridge_alpha,
    )
    print(f"Saved identifiability report to {output}")


def _collect_rows(
    model: torch.nn.Module,
    dataset: PuckDataset,
    config: ExperimentConfig,
    device: torch.device,
    max_samples: int,
) -> dict[str, np.ndarray]:
    sample_count = min(max(int(max_samples), 1), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64).tolist()
    batch_size = max(1, min(config.batch_size, 512))
    chunks: dict[str, list[np.ndarray]] = {
        "z_target": [],
        "z_hat_target": [],
        "context_state": [],
        "condition_params": [],
        "outcome_final_pos": [],
        "outcome_event": [],
        "trajectory_summary": [],
    }

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = collate_metadata([dataset[index] for index in batch_indices])
            device_batch = batch_to_device(batch, device)
            predictions = model_forward(model, device_batch)
            chunks["z_target"].append(_cpu_array(predictions["z_target"]))
            chunks["z_hat_target"].append(_cpu_array(predictions["z_hat_target"]))
            chunks["context_state"].append(_cpu_array(device_batch["context_state"]))
            chunks["condition_params"].append(_cpu_array(device_batch["condition_params"]))
            chunks["outcome_final_pos"].append(_cpu_array(device_batch["target_final_pos"]))
            chunks["outcome_event"].append(
                np.concatenate(
                    [
                        _cpu_array(device_batch["target_wall_contact"]),
                        _cpu_array(device_batch["target_ttc"]) / float(max(config.horizon, 1)),
                    ],
                    axis=1,
                )
            )
            chunks["trajectory_summary"].append(_trajectory_summary(device_batch["target_traj"], config))

    return {key: np.concatenate(value, axis=0).astype(np.float64) for key, value in chunks.items()}


def _trajectory_summary(target_traj: torch.Tensor, config: ExperimentConfig) -> np.ndarray:
    trajectory = target_traj.detach().cpu().reshape(target_traj.shape[0], config.horizon, config.max_pucks, 2).numpy()
    start = trajectory[:, 0].reshape(trajectory.shape[0], -1)
    final = trajectory[:, -1].reshape(trajectory.shape[0], -1)
    displacement = final - start
    mean_position = trajectory.mean(axis=1).reshape(trajectory.shape[0], -1)
    return np.concatenate([start, final, displacement, mean_position], axis=1)


def _ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1.0e-8, 1.0, x_std)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_centered = y_train - y_mean

    gram = x_train.T @ x_train
    regularizer = float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(gram + regularizer, x_train.T @ y_train_centered)
    prediction = x_test @ weights + y_mean
    residual = y_test - prediction
    mse = float(np.mean(np.square(residual)))
    variance = float(np.mean(np.square(y_test - y_test.mean(axis=0, keepdims=True))))
    r2 = 1.0 - mse / max(variance, 1.0e-12)
    return {
        "r2": float(r2),
        "mse": mse,
        "target_variance": variance,
        "target_dim": float(y.shape[1]),
    }


def _orthogonal_procrustes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    u, _, vt = np.linalg.svd(x_centered.T @ y_centered, full_matrices=False)
    rotation = u @ vt
    prediction = (x[test_indices] - x_mean) @ rotation + y_mean
    y_test = y[test_indices]
    mse = float(np.mean(np.square(y_test - prediction)))
    variance = float(np.mean(np.square(y_test - y_test.mean(axis=0, keepdims=True))))
    return {
        "r2": float(1.0 - mse / max(variance, 1.0e-12)),
        "mse": mse,
    }


def _latent_distribution(values: np.ndarray) -> dict[str, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0)
    safe_std = np.where(std < 1.0e-8, 1.0, std)
    normalized = centered / safe_std
    covariance = np.cov(values, rowvar=False)
    diagonal = np.diag(covariance)
    off_diagonal = covariance - np.diag(diagonal)
    eigenvalues = np.linalg.eigvalsh(covariance + 1.0e-6 * np.eye(covariance.shape[0], dtype=np.float64))
    return {
        "dim": float(values.shape[1]),
        "mean_abs_mean": float(np.abs(values.mean(axis=0)).mean()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "covariance_diag_mean": float(diagonal.mean()),
        "covariance_offdiag_abs_mean": float(np.abs(off_diagonal).mean()),
        "covariance_condition_number": float(eigenvalues.max() / max(eigenvalues.min(), 1.0e-12)),
        "skew_abs_mean": float(np.abs(np.mean(normalized**3, axis=0)).mean()),
        "excess_kurtosis_abs_mean": float(np.abs(np.mean(normalized**4, axis=0) - 3.0).mean()),
    }


def _mean_cosine(x: np.ndarray, y: np.ndarray) -> float:
    numerator = np.sum(x * y, axis=1)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.mean(numerator / np.maximum(denominator, 1.0e-12)))


def _probe_split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(sample_count, dtype=np.int64)
    train_count = max(1, int(math.floor(sample_count * 0.8)))
    if train_count >= sample_count:
        train_count = sample_count - 1
    return order[:train_count], order[train_count:]


def _cpu_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


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


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": result["model_name"],
        "split": result["split"],
        "sample_count": result["sample_count"],
        "prediction_to_target_direct_mse": result["alignment"]["direct_mse"],
        "prediction_to_target_linear_r2": result["alignment"]["linear_r2_prediction_to_target"],
        "prediction_final_pos_probe_r2": result["linear_probes"]["prediction"]["outcome_final_pos"]["r2"],
        "target_final_pos_probe_r2": result["linear_probes"]["target"]["outcome_final_pos"]["r2"],
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Identifiability Report",
        "",
        f"- Model: `{result['model_name']}`",
        f"- Run: `{result['run_dir']}`",
        f"- Dataset: `{result['dataset']}`",
        f"- Split: `{result['split']}`",
        f"- Samples: `{result['sample_count']}`",
        "",
        "## Latent Shape",
        "",
        "| Latent | mean | std mean | std min | offdiag abs mean | excess kurtosis abs mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in result["latent_distribution"].items():
        lines.append(
            "| {label} | {mean:.5f} | {std:.5f} | {std_min:.5f} | {offdiag:.5f} | {kurtosis:.5f} |".format(
                label=label,
                mean=stats["mean_abs_mean"],
                std=stats["std_mean"],
                std_min=stats["std_min"],
                offdiag=stats["covariance_offdiag_abs_mean"],
                kurtosis=stats["excess_kurtosis_abs_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Prediction To Target Alignment",
            "",
            f"- Direct latent MSE: `{result['alignment']['direct_mse']:.6f}`",
            f"- Mean cosine: `{result['alignment']['mean_cosine']:.6f}`",
            f"- Linear R2 from prediction to target: `{result['alignment']['linear_r2_prediction_to_target']:.6f}`",
            f"- Orthogonal Procrustes R2 from prediction to target: `{result['alignment']['orthogonal_procrustes_r2_prediction_to_target']:.6f}`",
            "",
            "## Linear Probes",
            "",
            "| Probe Target | z_target R2 | z_hat_target R2 | Gap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    target_probes = result["linear_probes"]["target"]
    prediction_probes = result["linear_probes"]["prediction"]
    for name in DEFAULT_PROBE_TARGETS:
        target_r2 = target_probes[name]["r2"]
        prediction_r2 = prediction_probes[name]["r2"]
        lines.append(f"| {name} | {target_r2:.6f} | {prediction_r2:.6f} | {target_r2 - prediction_r2:.6f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
