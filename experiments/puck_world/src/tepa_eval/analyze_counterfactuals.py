from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tepa_eval.benchmark_counterfactuals import (
    _BenchmarkBundle,
    _batch_to_device,
    _forward_prediction_only,
    _load_benchmark_model,
    _synchronize,
)
from tepa_eval.dataset import collate_metadata
from tepa_eval.engine import default_device, device_report, print_device_report
from tepa_eval.io import write_text
from tepa_eval.metrics import average_metrics, loss_and_metrics


def analyze_counterfactuals(
    tepa_run: str | Path,
    fused_run: str | Path,
    dataset: str | Path,
    split: str = "counterfactual",
    max_scenes: int | None = None,
    output_dir: str | Path = "reports/counterfactual_analyses",
    device_name: str = "auto",
) -> Path:
    device = default_device(device_name)
    print_device_report(device, device_name)
    dataset_path = Path(dataset).expanduser().resolve()
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    run_output = output_path / f"{int(time.time())}_counterfactual_analysis"
    run_output.mkdir(parents=True, exist_ok=True)

    tepa = _load_benchmark_model(tepa_run, dataset_path, split, device)
    fused = _load_benchmark_model(fused_run, dataset_path, split, device)

    results = {
        "dataset": str(dataset_path),
        "split": split,
        "max_scenes": max_scenes,
        "device": device_report(device, device_name),
        "models": {
            "tepa": _analyze_model(tepa, max_scenes=max_scenes, device=device),
            "fused": _analyze_model(fused, max_scenes=max_scenes, device=device),
        },
    }
    results["comparison"] = _comparison(results["models"]["tepa"], results["models"]["fused"])

    (run_output / "counterfactual_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_text(run_output / "counterfactual_analysis.md", _markdown_report(results))
    print(json.dumps(results["comparison"], indent=2, sort_keys=True))
    print(f"Saved counterfactual analysis to {run_output}")
    return run_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze counterfactual condition sensitivity.")
    parser.add_argument("--tepa-run", required=True, help="Run directory for a TEPA latent model.")
    parser.add_argument("--fused-run", required=True, help="Run directory for the fused latent transformer baseline.")
    parser.add_argument("--dataset", required=True, help="Eval-only counterfactual dataset directory.")
    parser.add_argument("--split", default="counterfactual", help="Dataset split to analyze.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap for quicker analysis runs.")
    parser.add_argument("--output-dir", default="reports/counterfactual_analyses")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Analysis device. Use --device mps to force Apple Silicon GPU.",
    )
    args = parser.parse_args()
    analyze_counterfactuals(
        tepa_run=args.tepa_run,
        fused_run=args.fused_run,
        dataset=args.dataset,
        split=args.split,
        max_scenes=args.max_scenes,
        output_dir=args.output_dir,
        device_name=args.device,
    )


def _analyze_model(bundle: _BenchmarkBundle, max_scenes: int | None, device: torch.device) -> dict[str, Any]:
    scene_groups = _scene_index_groups(bundle.dataset)
    scene_hashes = sorted(scene_groups)
    if max_scenes is not None:
        scene_hashes = scene_hashes[:max_scenes]
    if not scene_hashes:
        raise ValueError("No scenes found for counterfactual analysis.")

    records = _collect_prediction_records(bundle, scene_groups, scene_hashes, device)
    return {
        "run_dir": str(bundle.run_dir),
        "model_name": str(bundle.checkpoint["model_name"]),
        "scene_count": len(scene_hashes),
        "condition_count": len(records),
        "equivalent_rendering": _equivalent_rendering_metrics(records),
        "nearby_semantic": _nearby_semantic_metrics(records),
        "object_binding": _object_binding_metrics(records),
        "corrected_shuffle": _corrected_shuffle_metrics(bundle, scene_groups, scene_hashes, device),
    }


def _collect_prediction_records(
    bundle: _BenchmarkBundle,
    scene_groups: dict[str, list[int]],
    scene_hashes: list[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for scene_hash in scene_hashes:
            indices = scene_groups[scene_hash]
            batch = collate_metadata([bundle.dataset[index] for index in indices])
            batch = _batch_to_device(batch, device)
            predictions = _forward_prediction_only(bundle.model, batch)
            _synchronize(device)

            final_pos = predictions["final_pos"].detach().cpu().numpy().astype(np.float32)
            z_hat = predictions["z_hat_target"].detach().cpu().numpy().astype(np.float32)
            target_final = batch["target_final_pos"].detach().cpu().numpy().astype(np.float32)
            z_target = batch["z_target"].detach().cpu().numpy().astype(np.float32)
            context_state = batch["context_state"].detach().cpu().numpy().astype(np.float32)
            condition_ids = [int(value) for value in batch["condition_id"]]
            event_ids = [int(value) for value in batch["event_id"]]
            event_hashes = [str(value) for value in batch["event_hash"]]
            template_ids = [str(row["template_id"]) for row in (bundle.dataset.conditions[index] for index in indices)]

            for row_index, event_id in enumerate(event_ids):
                event = bundle.dataset.events_by_id[event_id]
                records.append(
                    {
                        "condition_id": condition_ids[row_index],
                        "event_id": event_id,
                        "event_hash": event_hashes[row_index],
                        "scene_hash": scene_hash,
                        "template_id": template_ids[row_index],
                        "event_family": str(event.get("metadata", {}).get("family", "")),
                        "target_object_id": int(event["target_object_id"]),
                        "impulse": np.array(event["impulse"], dtype=np.float32),
                        "pred_final_pos": final_pos[row_index],
                        "target_final_pos": target_final[row_index],
                        "z_hat_target": z_hat[row_index],
                        "z_target": z_target[row_index],
                        "context_state": context_state[row_index],
                    }
                )
    return records


def _equivalent_rendering_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["event_id"])].append(record)

    z_spreads = []
    final_spreads = []
    event_sizes = []
    template_counts: Counter[str] = Counter()
    for group in groups.values():
        if len(group) < 2:
            continue
        z_hat = np.stack([row["z_hat_target"] for row in group])
        final_pos = np.stack([row["pred_final_pos"] for row in group])
        z_spreads.append(float(np.mean((z_hat - z_hat.mean(axis=0, keepdims=True)) ** 2)))
        final_spreads.append(float(np.mean((final_pos - final_pos.mean(axis=0, keepdims=True)) ** 2)))
        event_sizes.append(float(len(group)))
        template_counts.update(str(row["template_id"]) for row in group)

    return {
        "event_count": float(len(z_spreads)),
        "mean_renderings_per_event": float(np.mean(event_sizes)) if event_sizes else 0.0,
        "z_hat_mse_to_event_mean": float(np.mean(z_spreads)) if z_spreads else 0.0,
        "final_pos_mse_to_event_mean": float(np.mean(final_spreads)) if final_spreads else 0.0,
        **{f"template_{key}_count": float(value) for key, value in sorted(template_counts.items())},
    }


def _nearby_semantic_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _one_record_per_event(records):
        if record["event_family"] == "nearby_force_direction_grid":
            by_scene[str(record["scene_hash"])].append(record)

    true_distances = []
    pred_distances = []
    latent_distances = []
    impulse_distances = []
    cosines = []
    for scene_records in by_scene.values():
        for left_index in range(len(scene_records)):
            for right_index in range(left_index + 1, len(scene_records)):
                left = scene_records[left_index]
                right = scene_records[right_index]
                true_delta = left["target_final_pos"] - right["target_final_pos"]
                pred_delta = left["pred_final_pos"] - right["pred_final_pos"]
                latent_delta = left["z_hat_target"] - right["z_hat_target"]
                impulse_delta = left["impulse"] - right["impulse"]
                true_distance = _norm(true_delta)
                pred_distance = _norm(pred_delta)
                true_distances.append(true_distance)
                pred_distances.append(pred_distance)
                latent_distances.append(_norm(latent_delta))
                impulse_distances.append(_norm(impulse_delta))
                cosine = _cosine(pred_delta, true_delta)
                if cosine is not None:
                    cosines.append(cosine)

    true_arr = np.array(true_distances, dtype=np.float64)
    pred_arr = np.array(pred_distances, dtype=np.float64)
    latent_arr = np.array(latent_distances, dtype=np.float64)
    impulse_arr = np.array(impulse_distances, dtype=np.float64)
    return {
        "scene_count": float(len(by_scene)),
        "pair_count": float(len(true_distances)),
        "true_final_delta_mean": float(true_arr.mean()) if true_arr.size else 0.0,
        "pred_final_delta_mean": float(pred_arr.mean()) if pred_arr.size else 0.0,
        "pred_to_true_delta_ratio": _safe_ratio(float(pred_arr.mean()) if pred_arr.size else 0.0, float(true_arr.mean()) if true_arr.size else 0.0),
        "pred_true_final_distance_corr": _pearson(pred_arr, true_arr),
        "latent_true_final_distance_corr": _pearson(latent_arr, true_arr),
        "impulse_pred_final_distance_corr": _pearson(impulse_arr, pred_arr),
        "pred_true_delta_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
    }


def _object_binding_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    one_per_event = [record for record in _one_record_per_event(records) if record["event_family"] == "object_binding_swap"]
    motion_correct = []
    target_motion = []
    other_motion = []
    by_scene_impulse: dict[tuple[str, tuple[float, float]], list[dict[str, Any]]] = defaultdict(list)
    for record in one_per_event:
        initial_positions = _initial_positions(record)
        pred_positions = record["pred_final_pos"].reshape(initial_positions.shape)
        target_object_id = int(record["target_object_id"])
        displacement = np.linalg.norm(pred_positions - initial_positions, axis=1)
        target_value = float(displacement[target_object_id])
        other_value = float(np.max(np.delete(displacement, target_object_id))) if len(displacement) > 1 else 0.0
        target_motion.append(target_value)
        other_motion.append(other_value)
        motion_correct.append(1.0 if target_value > other_value else 0.0)
        by_scene_impulse[(str(record["scene_hash"]), _rounded_impulse(record["impulse"]))].append(record)

    true_pair_distances = []
    pred_pair_distances = []
    pair_cosines = []
    for group in by_scene_impulse.values():
        if len(group) < 2:
            continue
        by_object = {int(record["target_object_id"]): record for record in group}
        objects = sorted(by_object)
        for left_index in range(len(objects)):
            for right_index in range(left_index + 1, len(objects)):
                left = by_object[objects[left_index]]
                right = by_object[objects[right_index]]
                true_delta = left["target_final_pos"] - right["target_final_pos"]
                pred_delta = left["pred_final_pos"] - right["pred_final_pos"]
                true_pair_distances.append(_norm(true_delta))
                pred_pair_distances.append(_norm(pred_delta))
                cosine = _cosine(pred_delta, true_delta)
                if cosine is not None:
                    pair_cosines.append(cosine)

    true_arr = np.array(true_pair_distances, dtype=np.float64)
    pred_arr = np.array(pred_pair_distances, dtype=np.float64)
    return {
        "event_count": float(len(one_per_event)),
        "paired_impulse_group_count": float(len(true_pair_distances)),
        "target_motion_accuracy": float(np.mean(motion_correct)) if motion_correct else 0.0,
        "target_motion_mean": float(np.mean(target_motion)) if target_motion else 0.0,
        "other_motion_mean": float(np.mean(other_motion)) if other_motion else 0.0,
        "target_to_other_motion_ratio": _safe_ratio(float(np.mean(target_motion)) if target_motion else 0.0, float(np.mean(other_motion)) if other_motion else 0.0),
        "pred_pair_delta_mean": float(pred_arr.mean()) if pred_arr.size else 0.0,
        "true_pair_delta_mean": float(true_arr.mean()) if true_arr.size else 0.0,
        "pred_to_true_pair_delta_ratio": _safe_ratio(float(pred_arr.mean()) if pred_arr.size else 0.0, float(true_arr.mean()) if true_arr.size else 0.0),
        "pred_true_pair_delta_corr": _pearson(pred_arr, true_arr),
        "pred_true_pair_delta_cosine_mean": float(np.mean(pair_cosines)) if pair_cosines else 0.0,
    }


def _corrected_shuffle_metrics(
    bundle: _BenchmarkBundle,
    scene_groups: dict[str, list[int]],
    scene_hashes: list[str],
    device: torch.device,
) -> dict[str, float]:
    baseline_rows = []
    shuffled_rows = []
    with torch.inference_mode():
        for scene_hash in scene_hashes:
            batch = collate_metadata([bundle.dataset[index] for index in scene_groups[scene_hash]])
            batch = _batch_to_device(batch, device)
            predictions = _forward_prediction_only(bundle.model, batch)
            _, metrics = loss_and_metrics(
                predictions,
                batch,
                bundle.config.horizon,
                prediction_loss_weight=bundle.config.prediction_loss_weight,
                latent_loss_weight=bundle.config.latent_loss_weight,
                target_reconstruction_loss_weight=0.0,
                context_state_loss_weight=0.0,
            )
            baseline_rows.append(metrics)

            shuffled = _shuffle_conditions_across_events(batch)
            shuffled_predictions = _forward_prediction_only(bundle.model, shuffled)
            _, shuffled_metrics = loss_and_metrics(
                shuffled_predictions,
                shuffled,
                bundle.config.horizon,
                prediction_loss_weight=bundle.config.prediction_loss_weight,
                latent_loss_weight=bundle.config.latent_loss_weight,
                target_reconstruction_loss_weight=0.0,
                context_state_loss_weight=0.0,
            )
            shuffled_rows.append(shuffled_metrics)

    baseline = average_metrics(baseline_rows)
    shuffled = average_metrics(shuffled_rows)
    return {
        "baseline_prediction_loss": baseline.get("prediction_loss", 0.0),
        "shuffled_prediction_loss": shuffled.get("prediction_loss", 0.0),
        "prediction_loss_degradation": shuffled.get("prediction_loss", 0.0) - baseline.get("prediction_loss", 0.0),
        "baseline_target_latent_mse": baseline.get("target_latent_mse", 0.0),
        "shuffled_target_latent_mse": shuffled.get("target_latent_mse", 0.0),
        "target_latent_mse_degradation": shuffled.get("target_latent_mse", 0.0) - baseline.get("target_latent_mse", 0.0),
        "baseline_final_position_mse": baseline.get("final_position_mse", 0.0),
        "shuffled_final_position_mse": shuffled.get("final_position_mse", 0.0),
        "final_position_mse_degradation": shuffled.get("final_position_mse", 0.0) - baseline.get("final_position_mse", 0.0),
    }


def _shuffle_conditions_across_events(batch: dict[str, Any]) -> dict[str, Any]:
    shuffled = dict(batch)
    event_ids = [int(value) for value in batch["event_id"]]
    order = _cross_event_permutation(event_ids)
    index = torch.tensor(order, dtype=torch.long, device=batch["target_final_pos"].device)
    for key in ("text_tokens", "z_condition", "condition_params"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == len(event_ids):
            shuffled[key] = value[index]
    return shuffled


def _cross_event_permutation(event_ids: list[int]) -> list[int]:
    if len(set(event_ids)) < 2:
        raise ValueError("Cross-event shuffle requires at least two event ids.")
    for offset in range(1, len(event_ids)):
        order = [(index + offset) % len(event_ids) for index in range(len(event_ids))]
        if all(event_ids[index] != event_ids[shuffled_index] for index, shuffled_index in enumerate(order)):
            return order
    raise ValueError("Could not construct a cross-event permutation.")


def _one_record_per_event(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        event_id = int(record["event_id"])
        if event_id not in selected or str(record["template_id"]) == "json_train_001":
            selected[event_id] = record
    return list(selected.values())


def _scene_index_groups(dataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.conditions):
        groups[str(row["scene_hash"])].append(index)
    return dict(groups)


def _initial_positions(record: dict[str, Any]) -> np.ndarray:
    context_state = record["context_state"].reshape(-1, 7)
    active = context_state[:, 6] > 0.0
    return context_state[active, :2]


def _rounded_impulse(impulse: np.ndarray) -> tuple[float, float]:
    return (round(float(impulse[0]), 3), round(float(impulse[1]), 3))


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.reshape(-1)))


def _cosine(left: np.ndarray, right: np.ndarray, epsilon: float = 1.0e-8) -> float | None:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator <= epsilon:
        return None
    return float(np.dot(left_flat, right_flat) / denominator)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) <= 1.0e-12 or float(np.std(right)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(denominator, 0.0, abs_tol=1.0e-12):
        return 0.0
    return float(numerator / denominator)


def _comparison(tepa: dict[str, Any], fused: dict[str, Any]) -> dict[str, Any]:
    return {
        "equivalent_rendering": {
            "tepa_z_hat_mse_to_event_mean": tepa["equivalent_rendering"]["z_hat_mse_to_event_mean"],
            "fused_z_hat_mse_to_event_mean": fused["equivalent_rendering"]["z_hat_mse_to_event_mean"],
            "tepa_final_pos_mse_to_event_mean": tepa["equivalent_rendering"]["final_pos_mse_to_event_mean"],
            "fused_final_pos_mse_to_event_mean": fused["equivalent_rendering"]["final_pos_mse_to_event_mean"],
        },
        "nearby_semantic": {
            "tepa_pred_true_final_distance_corr": tepa["nearby_semantic"]["pred_true_final_distance_corr"],
            "fused_pred_true_final_distance_corr": fused["nearby_semantic"]["pred_true_final_distance_corr"],
            "tepa_pred_to_true_delta_ratio": tepa["nearby_semantic"]["pred_to_true_delta_ratio"],
            "fused_pred_to_true_delta_ratio": fused["nearby_semantic"]["pred_to_true_delta_ratio"],
            "tepa_pred_true_delta_cosine_mean": tepa["nearby_semantic"]["pred_true_delta_cosine_mean"],
            "fused_pred_true_delta_cosine_mean": fused["nearby_semantic"]["pred_true_delta_cosine_mean"],
        },
        "object_binding": {
            "tepa_target_motion_accuracy": tepa["object_binding"]["target_motion_accuracy"],
            "fused_target_motion_accuracy": fused["object_binding"]["target_motion_accuracy"],
            "tepa_target_to_other_motion_ratio": tepa["object_binding"]["target_to_other_motion_ratio"],
            "fused_target_to_other_motion_ratio": fused["object_binding"]["target_to_other_motion_ratio"],
            "tepa_pred_true_pair_delta_cosine_mean": tepa["object_binding"]["pred_true_pair_delta_cosine_mean"],
            "fused_pred_true_pair_delta_cosine_mean": fused["object_binding"]["pred_true_pair_delta_cosine_mean"],
        },
        "corrected_shuffle": {
            "tepa_prediction_loss_degradation": tepa["corrected_shuffle"]["prediction_loss_degradation"],
            "fused_prediction_loss_degradation": fused["corrected_shuffle"]["prediction_loss_degradation"],
            "tepa_target_latent_mse_degradation": tepa["corrected_shuffle"]["target_latent_mse_degradation"],
            "fused_target_latent_mse_degradation": fused["corrected_shuffle"]["target_latent_mse_degradation"],
        },
    }


def _markdown_report(results: dict[str, Any]) -> str:
    comparison = results["comparison"]
    lines = [
        "# Counterfactual Condition Analysis",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- Split: `{results['split']}`",
        f"- Device: `{results['device']['selected_device']}`",
        "",
        "## Equivalent Renderings",
        "",
        "| Metric | TEPA | Fused |",
        "| --- | ---: | ---: |",
        f"| z-hat MSE to event mean | {comparison['equivalent_rendering']['tepa_z_hat_mse_to_event_mean']:.6f} | {comparison['equivalent_rendering']['fused_z_hat_mse_to_event_mean']:.6f} |",
        f"| final-position MSE to event mean | {comparison['equivalent_rendering']['tepa_final_pos_mse_to_event_mean']:.6f} | {comparison['equivalent_rendering']['fused_final_pos_mse_to_event_mean']:.6f} |",
        "",
        "## Nearby Semantic Sensitivity",
        "",
        "| Metric | TEPA | Fused |",
        "| --- | ---: | ---: |",
        f"| predicted/true final-distance correlation | {comparison['nearby_semantic']['tepa_pred_true_final_distance_corr']:.6f} | {comparison['nearby_semantic']['fused_pred_true_final_distance_corr']:.6f} |",
        f"| predicted-to-true delta ratio | {comparison['nearby_semantic']['tepa_pred_to_true_delta_ratio']:.6f} | {comparison['nearby_semantic']['fused_pred_to_true_delta_ratio']:.6f} |",
        f"| predicted/true delta cosine | {comparison['nearby_semantic']['tepa_pred_true_delta_cosine_mean']:.6f} | {comparison['nearby_semantic']['fused_pred_true_delta_cosine_mean']:.6f} |",
        "",
        "## Object Binding",
        "",
        "| Metric | TEPA | Fused |",
        "| --- | ---: | ---: |",
        f"| target-motion accuracy | {comparison['object_binding']['tepa_target_motion_accuracy']:.6f} | {comparison['object_binding']['fused_target_motion_accuracy']:.6f} |",
        f"| target/other motion ratio | {comparison['object_binding']['tepa_target_to_other_motion_ratio']:.6f} | {comparison['object_binding']['fused_target_to_other_motion_ratio']:.6f} |",
        f"| predicted/true pair-delta cosine | {comparison['object_binding']['tepa_pred_true_pair_delta_cosine_mean']:.6f} | {comparison['object_binding']['fused_pred_true_pair_delta_cosine_mean']:.6f} |",
        "",
        "## Corrected Cross-Event Shuffle",
        "",
        "| Metric | TEPA | Fused |",
        "| --- | ---: | ---: |",
        f"| prediction-loss degradation | {comparison['corrected_shuffle']['tepa_prediction_loss_degradation']:.6f} | {comparison['corrected_shuffle']['fused_prediction_loss_degradation']:.6f} |",
        f"| target-latent-MSE degradation | {comparison['corrected_shuffle']['tepa_target_latent_mse_degradation']:.6f} | {comparison['corrected_shuffle']['fused_target_latent_mse_degradation']:.6f} |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
