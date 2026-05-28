from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from tepa_eval.evaluate import evaluate_run
from tepa_eval.identifiability import write_identifiability_report
from tepa_eval.io import write_text
from tepa_eval.models import PREDICTION_MODEL_NAMES
from tepa_eval.train import train_model

DEFAULT_MODELS = ("tepa_latent_vit", "context_memory_tepa", "fused_latent_transformer")
DEFAULT_SEEDS = (17, 43, 101)
SUMMARY_METRICS = (
    "val_target_latent_mse",
    "val_final_position_mse",
    "val_wall_contact_f1",
    "counterfactual_target_latent_mse",
    "counterfactual_final_position_mse",
    "counterfactual_wall_contact_f1",
    "identifiability_prediction_to_target_linear_r2",
    "identifiability_prediction_final_pos_probe_r2",
    "identifiability_prediction_condition_probe_r2",
    "identifiability_prediction_std_mean",
)


def compare_seeds(
    *,
    config: str | Path,
    target_checkpoint: str | Path,
    condition_checkpoint: str | Path | None,
    models: tuple[str, ...] = DEFAULT_MODELS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device_name: str = "auto",
    counterfactual_dataset: str | Path | None = None,
    output_dir: str | Path = "reports/seed_comparisons",
    identifiability_max_samples: int = 4096,
    panel_samples: int = 6,
    fine_tune_condition_encoder: bool = False,
) -> Path:
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    comparison_dir = output_path / f"{int(time.time())}_three_seed_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for model_name in models:
            print(f"Starting {model_name} seed={seed}", flush=True)
            run_dir = train_model(
                config,
                model_name,
                target_checkpoint=target_checkpoint,
                freeze_target=True,
                condition_checkpoint=condition_checkpoint if model_name != "fused_latent_transformer" else None,
                freeze_condition_encoder=(
                    condition_checkpoint is not None
                    and model_name != "fused_latent_transformer"
                    and not fine_tune_condition_encoder
                ),
                device_name=device_name,
                seed_override=seed,
            )
            record = _record_from_run(run_dir, seed=seed, model_name=model_name)
            ident_path = write_identifiability_report(
                run_dir,
                split="val",
                device_name=device_name,
                max_samples=identifiability_max_samples,
            )
            record.update(_identifiability_metrics(ident_path, prefix="identifiability"))

            if counterfactual_dataset is not None:
                counterfactual_metrics = evaluate_run(
                    run_dir,
                    split="counterfactual",
                    panel_samples=panel_samples,
                    device_name=device_name,
                    dataset=counterfactual_dataset,
                )
                record.update(_flat_metrics(counterfactual_metrics, prefix="counterfactual"))
                counterfactual_ident_path = write_identifiability_report(
                    run_dir,
                    split="counterfactual",
                    dataset=counterfactual_dataset,
                    device_name=device_name,
                    max_samples=identifiability_max_samples,
                )
                record.update(_identifiability_metrics(counterfactual_ident_path, prefix="counterfactual_identifiability"))

            records.append(record)
            _write_summary(
                comparison_dir,
                records,
                seeds=seeds,
                models=models,
                fine_tune_condition_encoder=fine_tune_condition_encoder,
            )

    _write_summary(
        comparison_dir,
        records,
        seeds=seeds,
        models=models,
        fine_tune_condition_encoder=fine_tune_condition_encoder,
    )
    print(f"Saved seed comparison to {comparison_dir}")
    return comparison_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and aggregate a multi-seed puck-world comparison.")
    parser.add_argument("--config", required=True, help="Base config YAML.")
    parser.add_argument("--target-checkpoint", required=True, help="Frozen target_autoencoder checkpoint.")
    parser.add_argument("--condition-checkpoint", default=None, help="Frozen condition_semantics checkpoint for TEPA-style models.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=PREDICTION_MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counterfactual-dataset", default=None)
    parser.add_argument("--output-dir", default="reports/seed_comparisons")
    parser.add_argument("--identifiability-max-samples", type=int, default=4096)
    parser.add_argument("--panel-samples", type=int, default=6)
    parser.add_argument(
        "--fine-tune-condition-encoder",
        action="store_true",
        help="Load the condition checkpoint but keep TEPA-style condition encoders trainable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training/evaluation device. Use --device mps to force Apple Silicon GPU.",
    )
    args = parser.parse_args()
    compare_seeds(
        config=args.config,
        target_checkpoint=args.target_checkpoint,
        condition_checkpoint=args.condition_checkpoint,
        models=tuple(args.models),
        seeds=tuple(args.seeds),
        device_name=args.device,
        counterfactual_dataset=args.counterfactual_dataset,
        output_dir=args.output_dir,
        identifiability_max_samples=args.identifiability_max_samples,
        panel_samples=args.panel_samples,
        fine_tune_condition_encoder=args.fine_tune_condition_encoder,
    )


def _record_from_run(run_dir: Path, *, seed: int, model_name: str) -> dict[str, Any]:
    final_metrics_path = run_dir / "final_metrics.json"
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "seed": int(seed),
        "model_name": model_name,
        "run_dir": str(run_dir),
    }
    if isinstance(final_metrics.get("val"), dict):
        record.update(_flat_metrics(final_metrics["val"], prefix="val"))
    if isinstance(final_metrics.get("run"), dict):
        record.update(_flat_metrics(final_metrics["run"], prefix="run"))
    if "trainable_params" in final_metrics:
        record["trainable_params"] = float(final_metrics["trainable_params"])
    return record


def _flat_metrics(metrics: dict[str, Any], *, prefix: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, int | float):
            rows[f"{prefix}_{key}"] = float(value)
    return rows


def _identifiability_metrics(path: Path, *, prefix: str) -> dict[str, float]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        f"{prefix}_prediction_to_target_direct_mse": float(report["alignment"]["direct_mse"]),
        f"{prefix}_prediction_to_target_linear_r2": float(report["alignment"]["linear_r2_prediction_to_target"]),
        f"{prefix}_prediction_to_target_procrustes_r2": float(
            report["alignment"]["orthogonal_procrustes_r2_prediction_to_target"]
        ),
        f"{prefix}_prediction_final_pos_probe_r2": float(
            report["linear_probes"]["prediction"]["outcome_final_pos"]["r2"]
        ),
        f"{prefix}_prediction_condition_probe_r2": float(
            report["linear_probes"]["prediction"]["condition_params"]["r2"]
        ),
        f"{prefix}_target_final_pos_probe_r2": float(report["linear_probes"]["target"]["outcome_final_pos"]["r2"]),
        f"{prefix}_target_condition_probe_r2": float(report["linear_probes"]["target"]["condition_params"]["r2"]),
        f"{prefix}_prediction_std_mean": float(report["latent_distribution"]["prediction"]["std_mean"]),
        f"{prefix}_prediction_kurtosis_abs_mean": float(
            report["latent_distribution"]["prediction"]["excess_kurtosis_abs_mean"]
        ),
    }


def _write_summary(
    comparison_dir: Path,
    records: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    fine_tune_condition_encoder: bool,
) -> None:
    aggregates = _aggregate(records)
    summary = {
        "seeds": list(seeds),
        "models": list(models),
        "fine_tune_condition_encoder": fine_tune_condition_encoder,
        "records": records,
        "aggregates": aggregates,
    }
    (comparison_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_text(comparison_dir / "summary.md", _markdown_summary(summary))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(str(record["model_name"]), []).append(record)

    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_records in by_model.items():
        aggregates[model_name] = {}
        metric_names = sorted(
            {
                key
                for record in model_records
                for key, value in record.items()
                if isinstance(value, int | float) and key != "seed"
            }
        )
        for metric_name in metric_names:
            values = [float(record[metric_name]) for record in model_records if metric_name in record]
            if not values:
                continue
            aggregates[model_name][metric_name] = {
                "count": float(len(values)),
                "mean": float(statistics.fmean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            }
    return aggregates


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Three-Seed Puck-World Comparison",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        f"- Models: `{', '.join(summary['models'])}`",
        f"- Condition encoder mode: `{'fine-tuned' if summary.get('fine_tune_condition_encoder') else 'frozen for TEPA-style models'}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | " + " | ".join(summary["models"]) + " |",
        "| --- | " + " | ".join("---:" for _ in summary["models"]) + " |",
    ]
    for metric_name in SUMMARY_METRICS:
        cells = [metric_name]
        best_model = _best_model_for_metric(summary, metric_name)
        for model_name in summary["models"]:
            metric = summary["aggregates"].get(model_name, {}).get(metric_name)
            if metric is None:
                cells.append("")
            else:
                value = f"{metric['mean']:.6f} +/- {metric['std']:.6f}"
                if model_name == best_model:
                    value = f"**{value}**"
                cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Runs", "", "| Seed | Model | Run |", "| ---: | --- | --- |"])
    for record in summary["records"]:
        lines.append(f"| {record['seed']} | {record['model_name']} | `{record['run_dir']}` |")
    lines.append("")
    return "\n".join(lines)


def _best_model_for_metric(summary: dict[str, Any], metric_name: str) -> str | None:
    scored: list[tuple[float, str]] = []
    for model_name in summary["models"]:
        metric = summary["aggregates"].get(model_name, {}).get(metric_name)
        if metric is None:
            continue
        mean = float(metric["mean"])
        if metric_name.endswith("_std_mean"):
            score = -abs(mean - 1.0)
        elif any(token in metric_name for token in ("mse", "mae", "loss")):
            score = -mean
        else:
            score = mean
        scored.append((score, model_name))
    if not scored:
        return None
    return max(scored)[1]


if __name__ == "__main__":
    main()
