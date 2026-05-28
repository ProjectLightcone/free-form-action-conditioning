# TEPA Puck-World Evaluation

This package implements the first TEPA evaluation vertical slice: deterministic puck-world data generation, structured/text condition fan-out, TEPA versus fused monolithic baseline training, and lightweight reporting.

Run from this directory:

```bash
uv run tepa-puck-generate --config configs/smoke.yaml
uv run tepa-puck-validate-data --dataset data/processed/puck-smoke
uv run tepa-puck-generate --config configs/counterfactual_eval_json.yaml
uv run tepa-puck-validate-data --dataset data/processed/puck-v0.1-counterfactual-eval-json
uv run tepa-puck-pretrain-target --config configs/smoke.yaml
uv run tepa-puck-train --config configs/smoke.yaml --model tepa
uv run tepa-puck-train --config configs/smoke.yaml --model tepa_latent
uv run tepa-puck-train --config configs/smoke.yaml --model tepa_latent --target-checkpoint reports/runs/<target_run>/model.pt --freeze-target
uv run tepa-puck-train --config configs/smoke_context_memory.yaml --model context_memory_tepa --target-checkpoint reports/runs/<target_run>/model.pt --freeze-target
uv run tepa-puck-train --config configs/smoke.yaml --model monolithic
uv run tepa-puck-train --config configs/smoke.yaml --model fused_latent_transformer --target-checkpoint reports/runs/<target_run>/model.pt --freeze-target
uv run tepa-puck-train --config configs/smoke.yaml --model stuffed_image
uv run tepa-puck-evaluate --run reports/runs/<run_id>
uv run tepa-puck-evaluate --run reports/runs/<run_id> --dataset data/processed/puck-v0.1-counterfactual-eval-json --split counterfactual
uv run tepa-puck-identifiability-report --run reports/runs/<run_id> --split val --device mps
uv run tepa-puck-compare-seeds --config configs/phase1_json_latent_context_aux.yaml --target-checkpoint reports/runs/1779776068_target_autoencoder/model.pt --condition-checkpoint reports/runs/1779812783_condition_semantics/model.pt --counterfactual-dataset data/processed/puck-v0.1-counterfactual-eval-json --device mps
uv run tepa-puck-benchmark-counterfactuals --tepa-run reports/runs/<tepa_run> --fused-run reports/runs/<fused_run> --dataset data/processed/puck-v0.1-counterfactual-eval-json --device mps
uv run tepa-puck-analyze-counterfactuals --tepa-run reports/runs/<tepa_run> --fused-run reports/runs/<fused_run> --dataset data/processed/puck-v0.1-counterfactual-eval-json --device mps
uv run pytest
```

`fused_latent_transformer` is the primary context-stuffing baseline for the frozen-target Phase 2 comparison: image patches and raw condition text tokens are concatenated into one sequence and passed through one shared transformer encoder, then decoded through the same frozen target decoder used by `tepa_latent`. `monolithic` is the earlier supervised fused-token baseline without the frozen target path. `stuffed_image` preserves the byte-grid image-stuffing stress test, but it should not be treated as the primary comparison.

`context_memory_tepa` is the third candidate architecture. It keeps TEPA's reusable context idea, but caches context as image-patch memory tokens rather than one vector. Each condition vector performs a cross-attention read from the cached context memory before predicting the target latent:

```text
C = E_context_tokens(context)
q = E_condition(condition)
z_hat_target = P(CrossAttention(q, C), q, mean(C))
```

This tests a middle path between late vector fusion and full context stuffing: reusable context memory with condition-specific readout.

`tepa_latent` is the first Phase 2 model. It adds a target encoder and spatial target decoder, trains the target side to reconstruct simulator outcomes, and trains the TEPA predictor to match the target latent as well as decode to the supervised heads. Phase 2.1 also applies SIGReg to `z_target` by default, using the LeWorldModel-style settings `sigreg_weight: 0.1`, `sigreg_num_slices: 1024`, 17 integration points, and an integration range of `[-5, 5]`.

After target pretraining, the cleanest predictor diagnostic is to initialize `tepa_latent` from the saved `target_autoencoder` checkpoint and freeze the target encoder/decoder. The frozen decoder still passes gradients back to the predicted latent, but the target space itself stays fixed while the context and condition encoders learn to land in it. When the target side is frozen, the trainer omits the SIGReg term from the optimization/reporting loss because it would only be a constant with respect to trainable parameters.

Frozen-target predictor training also skips the true-target reconstruction branch. The model still encodes each outcome into `z_target` for latent matching, and it still decodes `z_hat_target` for supervised prediction heads, but it no longer decodes `z_target` or adds the constant reconstruction loss. This keeps the run focused on the context-condition predictor and avoids expensive duplicate target-decoder work.

The frozen-target path also precomputes an in-memory `event_id -> z_target` cache once per run, so equivalent condition renderings for the same intervention reuse the same target latent instead of re-running the target encoder. Training loaders also skip the byte-grid `condition_image` unless the selected model is `stuffed_image`, which avoids per-sample condition image construction for the main TEPA and fused-token baselines.

`tepa-puck-pretrain-target` trains only the target encoder and decoder on one deduped outcome row per intervention event. It reconstructs scalar facets and a continuous trajectory, then renders the heatmap probe from decoded trajectory coordinates with a differentiable Gaussian renderer. The target decoder emits an initial position plus bounded per-frame deltas, so the decoded path has temporal structure instead of being an unrelated bag of coordinates. The target loss includes trajectory coordinate MSE, scaled delta MSE, decoded-final-position MSE, and final-position consistency between the scalar head and the last decoded trajectory point. Soft target heatmaps are cached by the dataset and reused during loss computation. Use this stage before interpreting `tepa_latent` results: if the target autoencoder cannot reconstruct crisp outcome bundles, then noisy TEPA predictions may be a target-space problem rather than a context-condition prediction problem. The target pretraining command supports a reconstruction-first SIGReg warmup via `target_sigreg_warmup_epochs`, a smooth ramp via `target_sigreg_ramp_epochs`, and a target-specific regularization strength via `target_sigreg_weight`.

Training uses configurable optimization defaults:

```yaml
optimizer: adamw
weight_decay: 0.01
lr_schedule: cosine
min_learning_rate: 0.00008
lr_decay_start_epoch: 10
early_stopping_enabled: true
early_stopping_monitor: auto
early_stopping_min_delta: 0.001
early_stopping_patience: 12
early_stopping_min_epochs: 20
```

With `early_stopping_monitor: auto`, prediction runs monitor `val_prediction_loss`, while target pretraining monitors `val_target_reconstruction_loss`. That keeps checkpoint selection comparable across epochs even when SIGReg is warming up or ramping.

Each completed epoch now writes `last_model.pt`, and each validation improvement writes `best_model.pt`. At normal completion, `model.pt` is rewritten from the best validation checkpoint so downstream evaluation uses the best model by default. Reports also include `training_summary.json` with the best epoch, monitored metric, and early-stopping status. `tepa-puck-evaluate` will fall back to `best_model.pt` or `last_model.pt` when `model.pt` is not present, which makes interrupted runs recoverable after at least one completed epoch.

## Current Phase 2 Result

The current JSON-only frozen-target comparison is close rather than decisive. Using `configs/phase1_json_latent_context_aux.yaml`, the target checkpoint in `reports/runs/1779776068_target_autoencoder/model.pt`, the condition checkpoint in `reports/runs/1779812783_condition_semantics/model.pt`, and the fixed counterfactual dataset in `data/processed/puck-v0.1-counterfactual-eval-json`, the latest results average seeds `17`, `43`, and `101`. Values are mean +/- standard deviation. Bold values mark the current winner for each metric.

| Metric | Standard TEPA `tepa_latent_vit` | Context-memory TEPA `context_memory_tepa` | Context-stuffed JEPA `fused_latent_transformer` |
| --- | ---: | ---: | ---: |
| Validation target latent MSE | 0.073806 +/- 0.000692 | **0.059853 +/- 0.003236** | 0.091188 +/- 0.043016 |
| Validation final-position MSE | 0.006069 +/- 0.000132 | **0.004966 +/- 0.000360** | 0.006445 +/- 0.003753 |
| Validation wall-contact F1 | 0.976738 +/- 0.002419 | **0.980101 +/- 0.001245** | 0.976719 +/- 0.004538 |
| Counterfactual target latent MSE | 0.762004 +/- 0.013295 | **0.661938 +/- 0.007201** | 0.852729 +/- 0.019713 |
| Counterfactual final-position MSE | 0.068043 +/- 0.001533 | **0.061505 +/- 0.000877** | 0.068403 +/- 0.001572 |
| Counterfactual wall-contact F1 | 0.899413 +/- 0.006541 | 0.901635 +/- 0.011977 | **0.919787 +/- 0.012805** |
| Prediction-to-target linear R2 | 0.913299 +/- 0.002923 | **0.932091 +/- 0.001762** | 0.906237 +/- 0.044096 |
| Prediction final-position probe R2 | 0.948676 +/- 0.001893 | **0.958487 +/- 0.001430** | 0.935375 +/- 0.033108 |
| Prediction condition-parameter probe R2 | 0.776008 +/- 0.008914 | **0.818038 +/- 0.017387** | 0.724755 +/- 0.050849 |

Report: `reports/seed_comparisons/1779918499_three_seed_comparison/`.

Interpretation: the current result does not prove that any architecture is categorically more accurate. Context-memory TEPA is strongest and most stable across the three-seed validation/counterfactual target-space snapshot, the context-stuffed baseline remains very strong and wins counterfactual wall-contact F1, and standard TEPA remains the simplest reusable-context architecture. A key fairness caveat is that TEPA uses a pretrained frozen condition encoder in these runs, while the fused baseline trains a joint context-condition encoder end to end for this prediction task. The next ablation should compare frozen, fine-tuned, and scratch TEPA condition encoders.

`configs/counterfactual_eval_json.yaml` creates an eval-only dataset for that benchmark. It uses 256 held-out two-puck scenes, 64 canonical intervention events per scene, and 4 semantically equivalent JSON renderings per event. The split is named `counterfactual`, and it is intentionally separate from the frozen training dataset.

`tepa-puck-benchmark-counterfactuals` measures the actual amortized inference premise. For each selected scene and each `K` in `1, 4, 16, 64, 256`, TEPA encodes the context image once and reuses `z_context` across all K condition queries. The fused baseline re-encodes the combined context-condition sequence for every condition row. The benchmark writes JSON and Markdown reports under `reports/counterfactual_benchmarks/`.

First full amortized benchmark:

| K conditions/scene | TEPA ms/condition | Fused ms/condition | TEPA speedup | TEPA latent MSE | Fused latent MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6107 | 2.2794 | 0.873x | 0.1312 | 0.1274 |
| 4 | 0.8454 | 1.0166 | 1.203x | 0.3935 | 0.7393 |
| 16 | 0.3519 | 0.3922 | 1.114x | 0.6387 | 0.7841 |
| 64 | 0.2373 | 0.2780 | 1.171x | 0.7682 | 0.7020 |
| 256 | 0.2079 | 0.2917 | 1.403x | 0.7598 | 0.8406 |

Report: `reports/counterfactual_benchmarks/1779836613_counterfactual_benchmark/`.

Context-memory TEPA benchmark against the same fused implementation:

| K conditions/scene | Context-memory TEPA ms/condition | Fused ms/condition | Context-memory speedup | Context-memory latent MSE | Fused latent MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.8063 | 2.2070 | 0.786x | 0.0871 | 0.1274 |
| 4 | 0.9029 | 0.7155 | 0.792x | 0.3634 | 0.7393 |
| 16 | 0.3605 | 0.3971 | 1.101x | 0.5540 | 0.7841 |
| 64 | 0.2423 | 0.2864 | 1.182x | 0.6475 | 0.7020 |
| 256 | 0.2113 | 0.2797 | 1.324x | 0.6617 | 0.8406 |

Report: `reports/counterfactual_benchmarks/1779841687_counterfactual_benchmark/`.

`tepa-puck-analyze-counterfactuals` adds condition-sensitivity diagnostics that are more targeted than aggregate loss: equivalent JSON rendering consistency, nearby force/direction sensitivity, object-binding sensitivity, and corrected cross-event shuffle degradation. The corrected shuffle intentionally swaps condition text across different event ids, avoiding the earlier adjacent-row shuffle artifact where most rows were exchanged with equivalent renderings of the same event.

First full condition-sensitivity analysis:

| Diagnostic | Standard TEPA | Context-memory TEPA | Context-stuffed JEPA |
| --- | ---: | ---: | ---: |
| Equivalent z-hat MSE to event mean | 0.364868 | 0.274644 | 0.375729 |
| Equivalent final-position MSE to event mean | 0.042620 | 0.030906 | 0.030049 |
| Nearby final-distance correlation | 0.783040 | 0.808127 | 0.799168 |
| Nearby predicted-to-true delta ratio | 1.155461 | 1.094278 | 1.033002 |
| Object-binding target-motion accuracy | 0.968750 | 0.978516 | 0.988525 |
| Object-binding pair-delta cosine | 0.951131 | 0.958176 | 0.940312 |
| Corrected shuffle prediction-loss degradation | 0.611587 | 0.664220 | 0.619840 |
| Corrected shuffle target-latent-MSE degradation | 0.408633 | 0.377488 | 0.361184 |

Report: `reports/counterfactual_analyses/1779837269_counterfactual_analysis/`.

Current interpretation: this is a parity-plus-amortization result with context-memory TEPA as the best current candidate, not an overall accuracy proof. All three models are close enough that metric-level differences should still be treated carefully, but the three-seed pass makes the context-memory signal more credible than the earlier n=1 snapshot. The defensible finding is that reusable-context architectures can preserve broadly comparable prediction quality while becoming faster for multi-query counterfactual use. The next ablation should test whether TEPA improves when the condition encoder is fine-tuned or trained from scratch rather than frozen after pretraining.

The identifiability report is inspired by Klindt, LeCun, and Balestriero's LeJEPA identifiability analysis. It checks whether `z_target` and `z_hat_target` behave like useful world coordinates by reporting latent Gaussianity, prediction-to-target alignment, orthogonal Procrustes alignment, and linear-probe R2 for simulator state, condition parameters, final position, event/contact variables, and trajectory summaries.

Generated data, reports, checkpoints, and SQLite indexes are ignored by Git.
