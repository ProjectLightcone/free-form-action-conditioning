from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from tepa_eval.analyze_counterfactuals import _cross_event_permutation, _pearson
from tepa_eval.benchmark_counterfactuals import _select_evenly_spaced_indices
from tepa_eval.dataset import PuckDataset, TargetOutcomeDataset
from tepa_eval.dataset import byte_tokenize, collate_metadata
from tepa_eval.engine import (
    build_from_config,
    decoded_target_for_panel,
    metric_improved,
    model_forward,
    precompute_condition_latents,
    precompute_target_latents,
    should_stop_early,
    write_sample_panel,
    write_worst_sample_panel,
)
from tepa_eval.generate import generate_dataset
from tepa_eval.io import read_jsonl
from tepa_eval.metrics import condition_semantics_loss_and_metrics, sigreg_loss, target_autoencoder_loss_and_metrics
from tepa_eval.models import build_model
from tepa_eval.schemas import ExperimentConfig, load_config
from tepa_eval.train import load_condition_encoder
from tepa_eval.train_condition import pretrain_condition
from tepa_eval.trajectory import trajectory_to_heatmap_logits, trajectory_to_soft_heatmap
from tepa_eval.validate_data import validate_dataset


def test_generate_validate_and_load_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    summary = validate_dataset(output)
    assert summary["scenes"] == config.num_scenes
    assert summary["events"] == config.num_scenes * config.events_per_scene
    assert summary["conditions"] == summary["train_conditions"] + summary["val_conditions"] + summary["test_template_conditions"]

    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    sample = dataset[0]
    assert sample["context_image"].shape == (3, config.image_size, config.image_size)
    assert sample["context_state"].shape == (config.max_pucks * 7,)
    assert sample["condition_image"].shape == (3, config.image_size, config.image_size)
    assert sample["text_tokens"].shape == (config.max_text_len,)
    assert sample["condition_object_id"].shape == ()
    assert sample["condition_impulse"].shape == (2,)
    assert sample["condition_horizon"].shape == (1,)
    assert sample["condition_direction"].shape == ()
    assert sample["condition_magnitude"].shape == ()
    assert sample["condition_has_exact_impulse"].shape == ()
    assert sample["target_final_pos"].shape == (config.max_pucks * 2,)
    assert sample["target_traj"].shape == (config.horizon * config.max_pucks * 2,)
    assert sample["target_heatmap"].shape == (config.heatmap_size * config.heatmap_size,)
    assert sample["target_soft_heatmap"].shape == (config.heatmap_size * config.heatmap_size,)

    lean_dataset = PuckDataset(output, "train", max_text_len=config.max_text_len, include_condition_image=False)
    assert "condition_image" not in lean_dataset[0]
    json_dataset = PuckDataset(
        output,
        "train",
        max_text_len=config.max_text_len,
        include_condition_image=False,
        condition_family_filter={"json"},
    )
    assert len(json_dataset) > 0
    assert all(row["family"] == "json" for row in json_dataset.conditions)
    val_conditions = read_jsonl(output / "manifests" / "conditions_val.jsonl")
    test_template_conditions = read_jsonl(output / "manifests" / "conditions_test_templates.jsonl")
    assert all(row["metadata"]["template_group"] == "train" for row in val_conditions)
    assert all(row["metadata"]["template_group"] == "holdout" for row in test_template_conditions)


def test_counterfactual_eval_generation_uses_eval_split_and_json_fanout(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "generation_mode": "counterfactual_eval",
            "counterfactual_split": "counterfactual",
            "num_scenes": 2,
            "events_per_scene": 52,
            "renderings_per_event": 4,
            "min_pucks": 2,
            "max_pucks": 2,
            "condition_families": {"json": 1.0},
            "condition_family_filter": ["json"],
        }
    )
    output = generate_dataset(config)
    summary = validate_dataset(output)

    expected_conditions = config.num_scenes * config.events_per_scene * config.renderings_per_event
    assert summary["events"] == config.num_scenes * config.events_per_scene
    assert summary["conditions"] == expected_conditions
    assert summary["counterfactual_conditions"] == expected_conditions

    rows = read_jsonl(output / "manifests" / "conditions_counterfactual.jsonl")
    events = read_jsonl(output / "manifests" / "events.jsonl")
    assert all(row["split"] == "counterfactual" for row in rows)
    assert {row["template_id"] for row in rows}.issuperset(
        {"json_train_001", "json_train_002", "json_train_003", "json_train_004"}
    )
    assert {row["metadata"]["family"] for row in events}.issuperset(
        {"nearby_force_direction_grid", "object_binding_swap", "random_filler"}
    )

    dataset = PuckDataset(output, "counterfactual", max_text_len=config.max_text_len)
    assert len(dataset) == expected_conditions
    assert dataset[0]["text_tokens"].shape == (config.max_text_len,)


def test_counterfactual_benchmark_index_selection_is_deterministic() -> None:
    indices = list(range(256))

    assert _select_evenly_spaced_indices(indices, 1) == [0]
    assert _select_evenly_spaced_indices(indices, 4) == [0, 85, 170, 255]
    assert _select_evenly_spaced_indices(indices, 256) == indices


def test_counterfactual_analysis_cross_event_permutation_avoids_equivalent_rows() -> None:
    event_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    order = _cross_event_permutation(event_ids)

    assert sorted(order) == list(range(len(event_ids)))
    assert all(event_ids[index] != event_ids[shuffled_index] for index, shuffled_index in enumerate(order))


def test_counterfactual_analysis_pearson_handles_constant_inputs() -> None:
    assert _pearson(torch.ones(4).numpy(), torch.arange(4).numpy()) == 0.0
    assert _pearson(torch.arange(4).numpy(), torch.arange(4).numpy()) > 0.99


def test_models_have_expected_output_shapes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    image = torch.zeros((2, 3, config.image_size, config.image_size), dtype=torch.float32)
    condition_image = torch.zeros((2, 3, config.image_size, config.image_size), dtype=torch.float32)
    tokens = torch.zeros((2, config.max_text_len), dtype=torch.long)
    for model_name in (
        "tepa",
        "tepa_latent",
        "tepa_latent_vit",
        "context_memory_tepa",
        "monolithic",
        "fused_transformer",
        "fused_latent_transformer",
        "stuffed_image",
    ):
        model = build_model(
            model_name,
            config.latent_dim,
            config.max_pucks,
            config.heatmap_size,
            image_size=config.image_size,
            max_text_len=config.max_text_len,
            horizon=config.horizon,
        )
        output = model(image, tokens, condition_image)
        assert output["final_pos"].shape == (2, config.max_pucks * 2)
        assert output["heatmap"].shape == (2, config.heatmap_size * config.heatmap_size)
        assert output["wall_contact"].shape == (2, 1)
    assert output["time_to_contact"].shape == (2, 1)


def test_condition_encoder_is_sequence_sensitive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = build_model(
        "condition_semantics",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )
    model.eval()
    tokens = torch.tensor(
        [
            byte_tokenize('{"dx":1.25,"dy":-0.50,"object_id":1}', config.max_text_len),
            byte_tokenize('{"dy":-0.50,"dx":1.25,"object_id":1}', config.max_text_len),
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        embeddings = model.condition_encoder(tokens)

    assert not torch.allclose(embeddings[0], embeddings[1])


def test_condition_semantics_model_trains_against_canonical_event_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1], dataset[2], dataset[3]])
    model = build_model(
        "condition_semantics",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    predictions = model_forward(model, batch)
    loss, metrics = condition_semantics_loss_and_metrics(predictions, batch, config.horizon)

    assert predictions["z_condition"].shape == (4, config.latent_dim)
    assert predictions["object_logits"].shape == (4, config.max_pucks)
    assert predictions["impulse"].shape == (4, 2)
    assert predictions["direction_logits"].shape[0] == 4
    assert predictions["magnitude_logits"].shape[0] == 4
    assert torch.isfinite(loss)
    assert "condition_object_accuracy" in metrics
    assert "condition_impulse_within_0_10" in metrics
    assert "condition_horizon_mae" in metrics


def test_condition_pretraining_checkpoint_loads_into_tepa(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    run_dir = pretrain_condition(config_path)
    checkpoint_path = run_dir / "model.pt"
    config = load_config(config_path)
    model = build_from_config("tepa_latent", config)

    load_condition_encoder(model, checkpoint_path, freeze_condition_encoder=True)

    assert all(not parameter.requires_grad for parameter in model.condition_encoder.parameters())


def test_latent_tepa_encodes_targets_from_batch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    predictions = model_forward(model, batch)

    assert predictions["z_hat_target"].shape == (2, config.latent_dim)
    assert predictions["z_target"].shape == (2, config.latent_dim)
    assert predictions["target_reconstruction"]["final_pos"].shape == (2, config.max_pucks * 2)
    assert predictions["target_reconstruction"]["trajectory"].shape == (2, config.horizon * config.max_pucks * 2)
    assert predictions["target_reconstruction"]["heatmap"].shape == (2, config.heatmap_size * config.heatmap_size)


def test_latent_tepa_can_skip_target_reconstruction_for_frozen_predictor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )
    model.include_target_reconstruction = False
    cached_target = torch.randn((2, config.latent_dim), dtype=torch.float32)
    batch["z_target"] = cached_target

    predictions = model_forward(model, batch)

    assert predictions["z_hat_target"].shape == (2, config.latent_dim)
    assert torch.equal(predictions["z_target"], cached_target)
    assert "target_reconstruction" not in predictions


def test_fused_latent_transformer_encodes_context_and_condition_together(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "fused_latent_transformer",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
        context_state_dim=config.max_pucks * 7,
    )

    predictions = model_forward(model, batch)

    assert predictions["z_hat_target"].shape == (2, config.latent_dim)
    assert predictions["z_target"].shape == (2, config.latent_dim)
    assert predictions["trajectory"].shape == (2, config.horizon * config.max_pucks * 2)
    assert predictions["context_state"].shape == (2, config.max_pucks * 7)


def test_context_memory_tepa_uses_reusable_context_memory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "context_memory_tepa",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
        context_state_dim=config.max_pucks * 7,
    )

    context_memory = model.encode_context_memory(batch["context_image"][:1])
    z_hat_target = model.predict_from_memory(context_memory, text_tokens=batch["text_tokens"])
    predictions = model_forward(model, batch)

    assert context_memory.ndim == 3
    assert context_memory.shape[0] == 1
    assert z_hat_target.shape == (2, config.latent_dim)
    assert predictions["z_hat_target"].shape == (2, config.latent_dim)
    assert predictions["trajectory"].shape == (2, config.horizon * config.max_pucks * 2)
    assert predictions["context_state"].shape == (2, config.max_pucks * 7)


def test_sample_panel_can_decode_cached_target_latent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )
    model.include_target_reconstruction = False
    batch["z_target"] = torch.randn((2, config.latent_dim), dtype=torch.float32)

    predictions = model_forward(model, batch)
    decoded_target = decoded_target_for_panel(model, predictions)

    assert isinstance(decoded_target, dict)
    assert decoded_target["final_pos"].shape == (2, config.max_pucks * 2)
    assert decoded_target["trajectory"].shape == (2, config.horizon * config.max_pucks * 2)
    assert decoded_target["heatmap"].shape == (2, config.heatmap_size * config.heatmap_size)


def test_sample_panel_can_render_multiple_diagnostic_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_metadata)
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    write_sample_panel(
        tmp_path,
        model,
        loader,
        torch.device("cpu"),
        config,
        include_decoded_target=True,
        max_samples=3,
        include_condition_text=True,
    )

    assert (tmp_path / "sample_panel.png").exists()
    rows = (tmp_path / "sample_panel_rows.md").read_text(encoding="utf-8")
    assert "Sample 3" in rows
    assert "final_position_mse" in rows

    write_worst_sample_panel(
        tmp_path,
        model,
        loader,
        torch.device("cpu"),
        config,
        include_decoded_target=True,
        max_samples=2,
        include_condition_text=True,
    )
    assert (tmp_path / "sample_panel_worst.png").exists()
    assert "sample_score" in (tmp_path / "sample_panel_worst_rows.md").read_text(encoding="utf-8")


def test_precomputed_target_latents_are_joined_by_event(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    cache = precompute_target_latents(model, config, torch.device("cpu"), batch_size=2)
    dataset = PuckDataset(
        output,
        "train",
        max_text_len=config.max_text_len,
        include_condition_image=False,
        target_latents_by_event=cache,
    )
    sample = dataset[0]

    assert len(cache) == config.num_scenes * config.events_per_scene
    assert "condition_image" not in sample
    assert sample["z_target"].shape == (config.latent_dim,)


def test_precomputed_condition_latents_are_joined_by_condition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    model = build_model(
        "tepa_latent",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    cache = precompute_condition_latents(model, config, torch.device("cpu"), batch_size=2)
    uncached_dataset = PuckDataset(output, "train", max_text_len=config.max_text_len, include_condition_image=False)
    cached_dataset = PuckDataset(
        output,
        "train",
        max_text_len=config.max_text_len,
        include_condition_image=False,
        condition_latents_by_condition=cache,
    )
    uncached_sample = uncached_dataset[0]
    cached_sample = cached_dataset[0]

    model.eval()
    with torch.inference_mode():
        expected = model.condition_encoder(uncached_sample["text_tokens"].unsqueeze(0)).squeeze(0)

    assert len(cache) == validate_dataset(output)["conditions"]
    assert "condition_image" not in cached_sample
    assert "text_tokens" not in cached_sample
    assert cached_sample["z_condition"].shape == (config.latent_dim,)
    assert torch.allclose(cached_sample["z_condition"], expected)

    batch = collate_metadata([cached_dataset[0], cached_dataset[1]])
    predictions = model_forward(model, batch)

    assert predictions["z_hat_target"].shape == (2, config.latent_dim)


def test_target_outcome_dataset_dedupes_condition_fanout(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    condition_dataset = PuckDataset(output, "train", max_text_len=config.max_text_len)
    target_dataset = TargetOutcomeDataset(output, "train", max_text_len=config.max_text_len)

    unique_event_ids = {int(row["event_id"]) for row in condition_dataset.conditions}

    assert len(target_dataset) == len(unique_event_ids)
    assert len(target_dataset) < len(condition_dataset)


def test_target_autoencoder_reconstructs_from_target_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = TargetOutcomeDataset(output, "train", max_text_len=config.max_text_len)
    batch = collate_metadata([dataset[0], dataset[1]])
    model = build_model(
        "target_autoencoder",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
        horizon=config.horizon,
    )

    predictions = model_forward(model, batch)
    loss, metrics = target_autoencoder_loss_and_metrics(predictions, batch, config.horizon)

    assert predictions["z_target"].shape == (2, config.latent_dim)
    assert predictions["target_reconstruction"]["final_pos"].shape == (2, config.max_pucks * 2)
    assert predictions["target_reconstruction"]["trajectory"].shape == (2, config.horizon * config.max_pucks * 2)
    assert predictions["target_reconstruction"]["heatmap"].shape == (2, config.heatmap_size * config.heatmap_size)
    assert torch.isfinite(loss)
    assert "target_reconstruction_loss" in metrics
    assert "target_reconstruction_trajectory_mse" in metrics
    assert "target_reconstruction_trajectory_delta_mse" in metrics
    assert "target_reconstruction_trajectory_final_position_mse" in metrics
    assert "target_reconstruction_final_position_consistency_mse" in metrics


def test_dataset_caches_soft_heatmap_targets(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output = generate_dataset(config)
    dataset = TargetOutcomeDataset(output, "train", max_text_len=config.max_text_len)
    sample = dataset[0]

    rendered = trajectory_to_soft_heatmap(
        sample["target_traj"].reshape(1, config.horizon, config.max_pucks, 2),
        heatmap_size=config.heatmap_size,
        max_pucks=config.max_pucks,
    )[0]

    assert torch.allclose(sample["target_soft_heatmap"], rendered, atol=1e-6)


def test_trajectory_renderer_is_differentiable_and_ignores_padded_pucks() -> None:
    trajectory = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    trajectory[:, :, 0, :] = torch.tensor([0.5, 0.5])
    trajectory.requires_grad_()

    heatmap = trajectory_to_soft_heatmap(trajectory, heatmap_size=16, max_pucks=2)
    logits = trajectory_to_heatmap_logits(trajectory.reshape(1, -1), heatmap_size=16, max_pucks=2)
    center_value = heatmap.reshape(1, 16, 16)[0, 8, 8]
    corner_value = heatmap.reshape(1, 16, 16)[0, 0, 0]

    assert heatmap.shape == (1, 16 * 16)
    assert logits.shape == (1, 16 * 16)
    assert center_value > corner_value
    assert corner_value < 0.05
    heatmap.sum().backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()


def test_sigreg_penalizes_collapsed_embeddings_more_than_gaussian_embeddings() -> None:
    torch.manual_seed(11)
    gaussian = torch.randn((128, 32), dtype=torch.float32)
    collapsed = torch.zeros((128, 32), dtype=torch.float32)

    gaussian_loss = sigreg_loss(gaussian, num_slices=128)
    collapsed_loss = sigreg_loss(collapsed, num_slices=128)

    assert collapsed_loss > gaussian_loss


def test_monolithic_baseline_uses_one_fused_encoder(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = build_model(
        "monolithic",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
    )
    assert hasattr(model, "fused_encoder")
    assert not hasattr(model, "context_encoder")
    assert not hasattr(model, "text_encoder")
    assert not hasattr(model, "condition_encoder")


def test_stuffed_image_baseline_has_no_text_adapter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = build_model(
        "stuffed_image",
        config.latent_dim,
        config.max_pucks,
        config.heatmap_size,
        image_size=config.image_size,
        max_text_len=config.max_text_len,
    )
    assert hasattr(model, "stuffed_encoder")
    assert not hasattr(model, "text_embedding")
    assert not hasattr(model, "condition_encoder")


def test_early_stopping_helpers_respect_min_delta_and_patience(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "early_stopping_enabled": True,
            "early_stopping_min_epochs": 3,
            "early_stopping_patience": 2,
        }
    )

    assert metric_improved(0.9, None, 0.001)
    assert metric_improved(0.898, 0.9, 0.001)
    assert not metric_improved(0.8995, 0.9, 0.001)
    assert not should_stop_early(config, epoch_number=2, epochs_without_improvement=2)
    assert should_stop_early(config, epoch_number=3, epochs_without_improvement=2)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_version="test",
        seed=3,
        output_dir=tmp_path / "data",
        run_dir=tmp_path / "runs",
        checkpoint_dir=tmp_path / "checkpoints",
        image_size=32,
        horizon=12,
        num_scenes=8,
        events_per_scene=3,
        renderings_per_event=4,
        max_pucks=2,
        batch_size=4,
        epochs=1,
        latent_dim=16,
        max_text_len=64,
        heatmap_size=16,
    )


def _write_config(tmp_path: Path) -> Path:
    config = _config(tmp_path)
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    payload.update(
        {
            "output_dir": "data",
            "run_dir": "runs",
            "checkpoint_dir": "checkpoints",
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path
