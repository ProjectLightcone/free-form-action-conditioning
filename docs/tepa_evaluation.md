# TEPA Evaluation Plan: Puck World

This document describes a concrete first evaluation for TEPA. The goal is to test the architectural claim that a separate condition path improves controllability, condition-form generalization, and multi-query reuse compared with monolithic context stuffing.

The first benchmark should be synthetic, reproducible, fast to generate, and hard to solve by shortcut. It should not try to prove broad generality. It should answer one question:

> Given the same physical scene, can a model use structured and unstructured conditions to predict the correct conditioned outcome, while treating semantically equivalent conditions as equivalent and small condition changes as physically meaningful?

## Summary

Use a top-down 2D puck-world simulator to generate supervised examples:

```text
context x          rendered scene or state grid
condition c        text, structured record, arrow mask, or demo trace
target y_c         conditioned outcome bundle after simulation
```

Train TEPA:

```text
z_context = E_context(x)
z_condition = E_condition(c)
z_hat_target = P(z_context, z_condition)
```

Phase 2 adds an explicit target side:

```text
z_target       = E_target(y_c)
decoded_target = D_target(z_target)
decoded_hat    = D_target(z_hat_target)
```

The target encoder is trained by reconstruction/probe losses on the simulator outcome bundle. The predictor is trained both through the decoded supervised heads and through a latent alignment loss against `stopgrad(z_target)`. This makes the sample visualization diagnostic sharper: if `D_target(z_target)` is poor, the target encoder/decoder is the bottleneck; if `D_target(z_target)` is good but `D_target(z_hat_target)` is poor, the context-condition predictor is the bottleneck.

Before training the full latent TEPA model, train the target side by itself:

```text
z_target       = E_target(y_c)
decoded_target = D_target(z_target)
```

This target-only stage uses one deduped outcome row per intervention event, not every condition rendering for the same event. The goal is to answer a narrow but essential question: can the chosen target bundle be compressed into the proposed latent and decoded back into a useful simulator outcome? If the answer is no, the architecture does not yet have a clear prediction target. If the answer is yes, the remaining TEPA problem is better isolated: learn a context-condition predictor that lands in an already interpretable target space.

Target pretraining should report reconstruction metrics for every target facet:

```text
final position MSE
trajectory coordinate MSE
trajectory delta MSE
trajectory final-position MSE
final-position consistency MSE
trajectory heatmap BCE / Dice
wall-contact F1
time-to-contact MAE
target latent distribution statistics
```

The trajectory coordinate target should be treated as the primary rollout target. The heatmap is useful as a visual probe and auxiliary loss, but it is a lossy view of the trajectory. The target decoder should not learn a separate heatmap head for target reconstruction; it should decode the trajectory and render the heatmap from that trajectory with a differentiable Gaussian renderer. This keeps the target bundle internally coherent: the displayed path is a consequence of the decoded motion, not a second independent output that can disagree with it. The target decoder should also preserve temporal structure by decoding an initial puck position plus per-frame motion deltas rather than predicting every trajectory coordinate as an independent flat vector. The loss should include scaled delta reconstruction, decoded-final-position reconstruction, and consistency between the scalar final-position head and the last decoded trajectory point. Soft target heatmaps can be cached in the dataset, while predicted heatmaps remain differentiably rendered from decoded trajectories. The target sample panel should show `context`, `true target`, and `target reconstruction`, with both target and reconstruction heatmaps rendered from trajectory coordinates. Only after this panel is crisp should the predicted TEPA panel be interpreted as evidence about context and condition understanding.

In practice, the first target-only runs should use a short reconstruction warmup before applying SIGReg. During warmup, `E_target` and `D_target` learn the outcome bundle without pressure to match an isotropic Gaussian. After warmup, SIGReg can shape the target latent while the decoder preserves semantic detail:

```text
target_sigreg_warmup_epochs: 10
target_sigreg_ramp_epochs: 20
target_sigreg_weight: 0.02
sigreg_weight: 0.1  # used by the full TEPA run unless overridden
```

The target trainer should avoid a hard transition from no SIGReg to full SIGReg. A linear ramp prevents the loss jump from being interpreted as model regression when the objective has simply changed. Longer target pretraining should use AdamW and cosine learning-rate decay:

```text
optimizer: adamw
weight_decay: 0.01
lr_schedule: cosine
learning_rate: 0.0008
min_learning_rate: 0.00008
lr_decay_start_epoch: 10
epochs: 80
```

Phase 2.1 adds SIGReg to prevent latent collapse in the target space:

```text
L = L_prediction
  + 0.5 * L_target_reconstruction
  + 0.25 * ||z_hat_target - stopgrad(z_target)||^2
  + lambda_sigreg * SIGReg(z_target)
```

SIGReg is only meaningful if the latent vectors are allowed to follow an isotropic Gaussian distribution, so the SIGReg configuration uses unconstrained target latents rather than unit-normalized latents. The default mirrors the LeWorldModel setting: `lambda_sigreg = 0.1`, `M = 1024` random projections, 17 Epps-Pulley integration points, and an integration interval of `[-5, 5]`. The implementation reports `target_sigreg_loss`, `target_latent_abs_mean`, `target_latent_mean_std`, and `target_latent_min_std` so collapse or under-dispersed embeddings are visible in each run report.

Compare against a monolithic baseline:

```text
z_context_condition = E_fused(x, c)
z_hat_target        = P(z_context_condition)
```

The baseline should use modality-appropriate input adapters. The architectural contrast is not "TEPA gets language and the baseline gets pixels." The contrast is separate reusable context and condition embeddings versus one fused context-condition embedding.

The current interpretation should be more careful than "TEPA must be more accurate on every scalar metric." TEPA earns its premise if it can preserve competitive prediction quality while improving modularity, condition diagnostics, and amortized multi-query inference. Direct endpoint regression is a simple low-dimensional objective and may favor a fused transformer that can learn shortcuts from the combined context-condition input. TEPA's advantage should become clearer when one context is queried under many different conditions, because `z_context` can be cached and reused.

## Current Results Snapshot

The first Phase 2 comparison uses the JSON-only puck-world benchmark in `experiments/puck_world/configs/phase1_json_latent_context_aux.yaml`, the frozen target autoencoder from `reports/runs/1779776068_target_autoencoder/model.pt`, and MPS training on Apple Silicon.

Target autoencoder validation reconstruction is strong enough to serve as the shared target path:

| Metric | Validation value |
| --- | ---: |
| Target reconstruction final-position MSE | 0.000208 |
| Target reconstruction trajectory MSE | 0.000288 |
| Target reconstruction trajectory heatmap loss | 0.333871 |
| Target reconstruction wall-contact F1 | 1.000000 |
| Target latent mean std | 0.951240 |
| Target latent min std | 0.475128 |

Frozen-target predictor comparison. `fused_latent_transformer` is the context-stuffed JEPA-style baseline: context image patches and condition text tokens are processed together by one fused transformer. The latest comparison keeps the dataset, target checkpoint, condition checkpoint, and counterfactual eval dataset fixed, then varies training seeds `17`, `43`, and `101`. Values are mean +/- standard deviation. Bold values mark the current winner for each metric.

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

Report:

```text
experiments/puck_world/reports/seed_comparisons/1779918499_three_seed_comparison/
```

Standalone write-up:

```text
docs/three_seed_model_comparison.md
```

The three-seed result is stronger than the earlier one-seed snapshot but still should not be treated as a final architecture ranking. Context-memory TEPA is currently the strongest and most stable model across target-space error, endpoint error, and identifiability probes. The context-stuffed baseline remains very strong and wins counterfactual wall-contact F1, but it also shows much higher seed-to-seed variance on target-space and probe metrics. Standard TEPA remains important because it is the simplest reusable-context factorization and a useful lower-complexity reference point.

One important fairness caveat: this comparison freezes the TEPA condition encoders after condition-semantics pretraining, while the fused baseline trains a joint context-condition encoder end to end for the prediction task. That gives the fused baseline a possible advantage: it can learn a purpose-built condition interface shaped by the target loss, including whatever context-conditioned shortcuts help in puck world. The next ablation should therefore compare:

- frozen-condition TEPA: current setup, best for modular reuse;
- fine-tuned-condition TEPA: initialize from condition pretraining, then allow predictor loss to update the condition encoder;
- scratch-condition TEPA: train the condition encoder only through the prediction task;
- fused context-condition transformer: current early-interaction baseline.

The earlier one-seed training-set comparison is still useful for diagnosing fit capacity:

| Metric | Standard TEPA | Context-memory TEPA | Context-stuffed JEPA |
| --- | ---: | ---: | ---: |
| Train target latent MSE | 0.028861 | 0.037627 | 0.011878 |
| Train final-position MSE | 0.002783 | 0.003377 | 0.001142 |
| Train trajectory MSE | 0.001679 | 0.001924 | 0.000739 |
| Train wall-contact F1 | 0.999551 | 0.994022 | 0.999919 |

The fused baseline fit the one-seed training set most aggressively, which reinforces the ablation above: early fusion and a trainable joint encoder may be especially good at shaping task-specific features.

The amortized counterfactual benchmark tests cached-context reuse:

```text
for one context scene:
  encode context once for TEPA
  evaluate K conditions against cached z_context

for the fused baseline:
  re-encode the combined context-condition sequence K times
```

Run this for `K = 1, 4, 16, 64, 256` and report latency, memory, final-position MSE, trajectory MSE, target-latent MSE, condition-shuffle degradation, and sample-panel quality. This remains the cleanest way to test TEPA's runtime premise without overstating the current accuracy evidence.

Before making stronger claims from the scalar table, run the condition-encoder ablation and either parameter-match the models or report the slight parameter-count difference explicitly. A fair next baseline pass should also make the context-state auxiliary loss exactly analogous across architectures or remove it from both.

The first eval-only counterfactual dataset is:

```text
configs/counterfactual_eval_json.yaml
data/processed/puck-v0.1-counterfactual-eval-json/
```

It contains 256 held-out two-puck scenes, 64 canonical intervention events per scene, and 4 semantically equivalent JSON renderings per event. This yields 16,384 unique simulated events and 65,536 condition rows in the `counterfactual` split, with exactly 256 condition rows per scene. Event families are stratified into nearby force/direction edits, object-binding swaps, and random filler events.

The first amortized benchmark report for standard TEPA versus context-stuffed JEPA is:

```text
reports/counterfactual_benchmarks/1779836613_counterfactual_benchmark/
```

It compares cached-context `tepa_latent_vit` against `fused_latent_transformer` on MPS:

| Conditions per scene | TEPA ms/condition | Fused ms/condition | TEPA speedup over fused | TEPA target-latent MSE | Fused target-latent MSE | TEPA final-position MSE | Fused final-position MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6107 | 2.2794 | 0.873x | 0.1312 | 0.1274 | 0.0156 | 0.0078 |
| 4 | 0.8454 | 1.0166 | 1.203x | 0.3935 | 0.7393 | 0.0414 | 0.0614 |
| 16 | 0.3519 | 0.3922 | 1.114x | 0.6387 | 0.7841 | 0.0679 | 0.0629 |
| 64 | 0.2373 | 0.2780 | 1.171x | 0.7682 | 0.7020 | 0.0715 | 0.0486 |
| 256 | 0.2079 | 0.2917 | 1.403x | 0.7598 | 0.8406 | 0.0716 | 0.0619 |

The corresponding context-memory TEPA benchmark is:

```text
reports/counterfactual_benchmarks/1779841687_counterfactual_benchmark/
```

The benchmark invocations were separate, so latency should be read as approximate. The ratios are still useful because each run compares against the same fused implementation on the same device.

| Conditions per scene | Standard TEPA ms/condition | Context-memory TEPA ms/condition | Context-stuffed JEPA ms/condition | Standard TEPA speedup | Context-memory TEPA speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6107 | 2.8063 | 2.2794 / 2.2070 | 0.873x | 0.786x |
| 4 | 0.8454 | 0.9029 | 1.0166 / 0.7155 | 1.203x | 0.792x |
| 16 | 0.3519 | 0.3605 | 0.3922 / 0.3971 | 1.114x | 1.101x |
| 64 | 0.2373 | 0.2423 | 0.2780 / 0.2864 | 1.171x | 1.182x |
| 256 | 0.2079 | 0.2113 | 0.2917 / 0.2797 | 1.403x | 1.324x |

This supports the amortized-runtime premise for multi-query use: both reusable-context architectures are slower for a single condition query, but faster once a scene is queried many times. Standard TEPA is slightly faster at high K, while context-memory TEPA keeps nearly the same amortized speed despite its cross-attention readout. The quality story remains mixed. Context-memory TEPA improves the three-seed validation and counterfactual target-space metrics, but the context-stuffed baseline still wins counterfactual wall-contact F1. This should be reported as a runtime/modularity result plus a promising third-candidate signal, not an accuracy proof.

The first condition-sensitivity analysis report is:

```text
reports/counterfactual_analyses/1779837269_counterfactual_analysis/
```

This report corrects the earlier shuffle artifact by swapping condition text across different event ids rather than adjacent rows. It also separates three condition-use questions:

| Diagnostic | Standard TEPA | Context-memory TEPA | Context-stuffed JEPA | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Equivalent rendering z-hat MSE to event mean | 0.364868 | 0.274644 | 0.375729 | Lower is more invariant across equivalent JSON renderings. |
| Equivalent rendering final-position MSE to event mean | 0.042620 | 0.030906 | 0.030049 | Lower is more decoded-output invariant. |
| Nearby predicted/true final-distance correlation | 0.783040 | 0.808127 | 0.799168 | Higher means predicted changes track true force/direction changes. |
| Nearby predicted-to-true delta ratio | 1.155461 | 1.094278 | 1.033002 | Closer to 1 means better calibrated sensitivity magnitude. |
| Nearby predicted/true delta cosine | 0.848749 | 0.877184 | 0.860598 | Higher means predicted semantic changes point in the right direction. |
| Object-binding target-motion accuracy | 0.968750 | 0.978516 | 0.988525 | Higher means the instructed puck moves more than the non-target puck. |
| Object-binding predicted/true pair-delta cosine | 0.951131 | 0.958176 | 0.940312 | Higher means object-swap differences point in the right direction. |
| Corrected shuffle prediction-loss degradation | 0.611587 | 0.664220 | 0.619840 | Higher means wrong conditions hurt more. |
| Corrected shuffle target-latent-MSE degradation | 0.408633 | 0.377488 | 0.361184 | Higher means wrong conditions hurt target-space alignment more. |

The result suggests all three models are using the condition signal. Context-memory TEPA looks promising on equivalent-rendering latent consistency, nearby semantic direction, object-swap direction, and prediction-loss shuffle degradation. The context-stuffed baseline remains strongest on direct decoded-output invariance, nearby magnitude calibration, and object-binding target-motion accuracy. Standard TEPA remains competitive while being the simplest reusable-context architecture. This is still not a clean accuracy win for any architecture, but the third candidate is now supported by both the condition-sensitivity analysis and the three-seed aggregate above.

## Current Interpretation

The current evidence should be described as parity plus amortization, with context-memory TEPA as the strongest current candidate rather than a settled winner. Across the fixed-data three-seed JSON-only puck-world comparison, reusable-context models and the fused context-stuffing transformer are broadly comparable. Metric-level differences are now more credible than the earlier n=1 snapshot, but the benchmark is still narrow, synthetic, and sensitive to training-design choices.

What the current result supports:

- TEPA is competitive with a strong fused context-condition baseline on this narrow synthetic benchmark.
- TEPA shows the expected amortized runtime advantage when many conditions are evaluated against the same context.
- Both architectures use condition information, as shown by corrected cross-event condition-shuffle degradation.
- Context-memory TEPA currently has the best mean validation target-space metrics and the strongest identifiability probes across three seeds.
- The separated context and condition path does not obviously break target-space alignment, but it also does not yet produce a decisive general accuracy advantage.

What the current result does not yet support:

- TEPA is more accurate than context stuffing.
- TEPA generalizes better across condition semantics.
- TEPA has a robust semantic advantage.
- A frozen condition encoder is the best possible version of the factorized architecture.

The paper should therefore frame this stage as an early architectural diagnostic:

> The initial experiment does not show a clear prediction-quality advantage for TEPA over early fusion. Instead, it shows approximate parity on a narrow synthetic benchmark, while TEPA provides a reusable context embedding that improves amortized inference when many counterfactual conditions are evaluated for the same scene.

The three-seed credibility pass has now been run:

```text
seeds: 17, 43, 101
dataset: keep puck-v0.1-json-context-aux fixed
target checkpoint: keep reports/runs/1779776068_target_autoencoder/model.pt fixed
counterfactual eval dataset: keep puck-v0.1-counterfactual-eval-json fixed
models: tepa_latent_vit, context_memory_tepa, fused_latent_transformer
report: reports/seed_comparisons/1779918499_three_seed_comparison/
```

The defensible claim after this pass is: TEPA trades early fusion for a more modular and reusable representation while preserving roughly comparable prediction quality on the first benchmark. Context-memory TEPA is currently the best-performing factorization, suggesting that cached context memory plus condition-specific readout is a promising middle path.

The next credibility step is a condition-encoder fairness ablation. The current TEPA variants use pretrained frozen condition encoders, while the fused baseline learns a joint context-condition representation directly from the prediction objective. That could explain part of the fused baseline's strength, because its condition interface is purpose-built for the task rather than preserved as a reusable module. The next fixed-data comparison should train frozen-condition TEPA, fine-tuned-condition TEPA, scratch-condition TEPA, and fused context stuffing under the same target path.

The new LeJEPA identifiability analysis by Klindt, LeCun, and Balestriero sharpens what should be measured here. It argues that Gaussian latent regularization is not merely an anti-collapse stabilizer; under stationary additive-noise latent dynamics, it is part of the condition under which a JEPA representation can recover the true world coordinates up to rotation. The puck-world experiment should therefore report whether the learned target and predicted latents are useful as world coordinates, not only whether the decoder's endpoint heads score well.

The implementation now includes:

```bash
uv run tepa-puck-identifiability-report \
  --run reports/runs/<run_id> \
  --split val \
  --max-samples 4096 \
  --device mps

uv run tepa-puck-compare-seeds \
  --config configs/phase1_json_latent_context_aux.yaml \
  --target-checkpoint reports/runs/1779776068_target_autoencoder/model.pt \
  --condition-checkpoint reports/runs/1779812783_condition_semantics/model.pt \
  --counterfactual-dataset data/processed/puck-v0.1-counterfactual-eval-json \
  --device mps
```

The identifiability report writes `identifiability_<split>.json` and `identifiability_<split>.md` next to the run. It measures:

- target and prediction latent Gaussianity: mean, standard deviation, covariance off-diagonal size, skew, and excess kurtosis;
- direct prediction-to-target latent alignment: MSE, cosine, linear R2, and orthogonal Procrustes R2;
- linear probe R2 from `z_target` and `z_hat_target` to context state, condition parameters, final position, event/contact variables, and trajectory summaries.

This does not prove the LeJEPA theorem applies to puck world. Our simulator has collisions, friction, deterministic impulses, and a conditioned target bundle rather than a pure stationary additive-noise latent process. But the diagnostics give the right failure modes to watch: a target latent that cannot linearly recover simulator state is probably a poor world-coordinate substrate, and a predicted latent that decodes well but has weak linear probes may be overfit to decoder-specific shortcuts.

## Broader Design-Space Takeaway

The close TEPA-vs-fused result is still encouraging because it points beyond the narrow architecture comparison. Both systems are learning a version of the same larger capability:

```text
world context + condition/query/intervention -> predicted latent outcome
```

That pattern is the seed of a general-purpose prediction model. The current experiment suggests that the important research direction may be conditioned latent prediction itself, while TEPA and fused context stuffing represent different architectural tradeoffs.

Fused context stuffing is attractive because it allows early interaction between the world state and the condition. A transformer can let condition tokens attend directly to image patches and let image patches attend back to the condition. This is likely to remain a strong baseline, especially when the condition determines which details of the context should be encoded in the first place.

TEPA is attractive because it factorizes the context and condition before prediction. That makes the context embedding reusable, supports modular condition encoders, exposes separate diagnostic surfaces, and naturally supports many counterfactual questions over the same world state. Its value may grow when the same scene or document is queried repeatedly, when condition encoders are swapped or specialized, or when target-space alignment matters more than single-query endpoint accuracy.

The right framing is therefore not:

```text
TEPA versus context stuffing, winner takes all.
```

It is:

```text
early fusion versus reusable factorization in conditioned latent prediction.
```

Future experiments should treat both as legitimate points in the design space and ask where each tradeoff becomes useful: single-query accuracy, many-query amortization, condition-surface generalization, cross-domain transfer, interpretability, and target-space reuse.

## Third Candidate: Context-Memory TEPA

The third candidate architecture is `context_memory_tepa`, a hybrid between single-vector TEPA and fused context stuffing:

```text
C = E_context_tokens(context)
q = E_condition(condition)
r = CrossAttention(q, C)
z_hat_target = P(r, q, mean(C))
```

The intent is to keep TEPA's reusable-context advantage while reducing the weakness of compressing the context into one condition-agnostic vector. The context encoder runs once and produces a cached memory of image-patch tokens. Each condition then performs a lightweight cross-attention read from that memory.

Expected tradeoff:

| Model | Reusable context | Condition/context interaction | Expected role |
| --- | --- | --- | --- |
| `tepa_latent_vit` | Yes, one vector | Late vector fusion | Fastest factorized baseline. |
| `fused_latent_transformer` | No | Full early fusion | Strong accuracy baseline. |
| `context_memory_tepa` | Yes, token memory | Cross-attention readout | Middle path: reusable context plus condition-specific selection. |

The implementation is wired into the same frozen-target path as the other Phase 2 models. A context-memory run uses the same frozen target and condition checkpoints as standard TEPA:

```bash
uv run tepa-puck-train \
  --config configs/phase1_json_latent_context_aux.yaml \
  --model context_memory_tepa \
  --target-checkpoint reports/runs/1779776068_target_autoencoder/model.pt \
  --freeze-target \
  --condition-checkpoint reports/runs/1779812783_condition_semantics/model.pt \
  --freeze-condition-encoder \
  --device mps
```

A tiny smoke config, `configs/smoke_context_memory.yaml`, exists only to verify the training/evaluation path quickly. The first full run completed as `reports/runs/1779840773_context_memory_tepa/`, and the three-seed aggregate now appears in the current results snapshot above.

## Library Choices

### Core Training

Use PyTorch.

Reasons:

- Straightforward custom `Dataset` and `DataLoader` support.
- Mature tensor, model, optimizer, checkpoint, and metric tooling.
- Can run on CPU, CUDA, or Apple Silicon MPS when supported.
- Easy to implement both TEPA and monolithic baselines with identical heads and comparable parameter counts.

Recommended Python packages:

```text
torch
numpy
pydantic
pillow
matplotlib
tqdm
```

Optional later:

```text
zarr          chunked storage for larger datasets
wandb         experiment tracking
tensorboard   local metric dashboards
polars        fast aggregate analysis
```

### Configuration and Schemas

Use Pydantic for boundary objects:

```text
experiment configs
simulator configs
dataset manifests
condition specs
target bundle schemas
run metadata
```

Pydantic is the right choice for validating serialized data and making configs explicit. It should not be used inside the hot training path. Validate configs and records at the boundary, then convert to NumPy arrays, PyTorch tensors, dataclasses, or plain dictionaries for batches.

Rule of thumb:

```text
Pydantic for configuration and serialized records.
Tensors, arrays, dataclasses, or plain dicts for in-memory training batches.
```

### Simulation

Start with pure NumPy, not a physics engine.

Reasons:

- The required dynamics are simple.
- Exact control over target construction matters.
- No dependency on game engines or rigid-body libraries.
- Easier to version, debug, seed, and vectorize.

The simulator should be deterministic first. Add stochasticity only after deterministic prediction works.

### Rendering

Use NumPy arrays plus Pillow or direct array rasterization.

Render small images first:

```text
64x64 for fastest iteration
96x96 or 128x128 for richer visual tests
```

Each context image should be generated from the canonical simulator state, not hand-authored assets.

## Proposed Repository Layout

The evaluation code can live beside the Astro whitepaper, but separate the research code from the website source:

```text
whitepaper/
  docs/
    tepa_evaluation.md
  experiments/
    puck_world/
      configs/
        smoke.yaml
        phase1_structured.yaml
        phase2_condition_surfaces.yaml
      data/
        raw/
        processed/
        manifests/
      reports/
        runs/
        figures/
      src/
        tepa_eval/
          __init__.py
          core/
            schemas.py
            batch.py
            losses.py
            metrics.py
            registry.py
          models/
            context_encoder.py
            condition_encoder.py
            predictor.py
            tepa.py
            monolithic.py
            heads.py
          experiments/
            puck_world/
              simulator.py
              render.py
              conditions.py
              targets.py
              dataset.py
              heads.py
              metrics.py
              visualize.py
          generate.py
          train.py
          evaluate.py
```

Keep generated data out of Git. Commit configs, source code, and small sample manifests only.

## Model and Experiment Boundaries

Separate the model from the experiment, but keep the boundary thin.

The shared TEPA model should not know about puck world specifically. It should know about generic inputs and representations:

```text
context
condition
z_context
z_condition
z_hat_target
target bundle
metadata
```

The experiment should own domain-specific choices:

```text
simulator
data generation
condition surfaces
target construction
splits
target heads
loss weighting
metrics
visualization
```

The likely coupling point is the target head. Puck world predicts final positions, heatmaps, collision events, and time-to-contact. A future spreadsheet or code-impact experiment would need different heads and losses. The TEPA trunk can remain shared:

```text
z_context = E_context(x)
z_condition = E_condition(c)
z_hat_target = P(z_context, z_condition)
```

Then the experiment attaches the appropriate head:

```text
predictions = puck_world_head(z_hat_target)
loss = puck_world_loss(predictions, target_bundle)
```

Expect the first experiment to shape the model API. Avoid trying to design a universal framework now. The goal is a small shared core plus experiment-specific adapters.

## Environment Design

### World State

Represent each scene as explicit state:

```text
pucks:
  x, y
  vx, vy
  radius
  mass
  color_id

world:
  walls
  rectangular or circular obstacles
  optional goal zones
  friction
  timestep dt
```

Start simple:

```text
1 to 3 pucks
axis-aligned walls
no obstacles or one rectangular obstacle
one impulse at t = 0
fixed rollout horizon
```

Then add complexity:

```text
more pucks
puck-puck collisions
obstacles
goal zones
noisy impulse magnitude
noisy friction
variable horizons
```

### Physics

Use a simple semi-implicit Euler step:

```text
v = v + impulse_at_t0 / mass
v = friction * v
x = x + v * dt
resolve wall collisions
resolve obstacle collisions
resolve puck-puck collisions
record events and trajectory
```

Wall collisions can be elastic with damping:

```text
if x - radius < left_wall:
  x = left_wall + radius
  vx = -restitution * vx
```

Puck-puck collision can start with a simple impulse-based resolver. It does not need to be physically perfect; it needs to be consistent, seeded, and rich enough that conditions matter.

## Conditions

Each training example should have a canonical intervention:

```text
object_id
impulse_dx
impulse_dy
force_bucket or continuous magnitude
horizon
```

Then generate one or more condition surfaces from the same canonical intervention.

### Condition Surface Types

Structured:

```json
{
  "object": "blue",
  "impulse": [-2.0, -2.0],
  "horizon": 40
}
```

Text:

```text
Push the blue puck up-left with medium force. Predict 40 frames ahead.
Nudge blue toward the upper-left and show where things are after 40 steps.
Apply a moderate northwest impulse to the blue puck for a 40-frame prediction.
```

Arrow mask:

```text
single-channel image with an arrow from the selected puck in the impulse direction
```

Demo trace:

```text
short prefix trajectory showing the selected puck beginning to move under the intended impulse
```

The same canonical intervention should be expressible through all condition surfaces. This enables the central invariance test:

```text
same context + equivalent conditions -> same target
```

And the sensitivity test:

```text
same context + small condition change -> different target
```

## Target Bundle

The target should be richer than a final coordinate. Store a conditioned outcome bundle:

```text
final_positions        float32 [num_pucks, 2]
trajectory             float32 [horizon, num_pucks, 2]
trajectory_heatmap     float32 [H, W] or [num_pucks, H, W]
collision_flags        bool or float32 [event_types]
contact_pairs          optional sparse event records
goal_success           bool
time_to_contact        float32, sentinel if no contact
uncertainty            optional, from noisy rollouts
```

Initial target heads:

```text
final position regression
trajectory heatmap prediction
collision event classification
goal success classification
time-to-contact regression
```

Later target heads:

```text
multiple sampled futures
mean and variance heads
energy score for candidate outcomes
retrieval embedding for nearest true rollout
```

## Dataset Generation

The detailed dataset generation plan lives in `docs/tepa_dataset_generation.md`. This section summarizes the training-data modes used by the evaluation.

### Fast Iteration Mode

Start with on-the-fly generation inside the PyTorch dataset.

Benefits:

- Fast iteration while simulator APIs are changing.
- No stale stored data.
- Easy smoke tests.

Use this for early debugging:

```text
num_examples = 10_000
image_size = 64
num_pucks = 1
condition_surface = structured
target = final position + trajectory heatmap
```

### Frozen Benchmark Mode

Once the simulator is stable, generate fixed dataset shards. Frozen data is required for real comparisons between TEPA and baselines.

Recommended format for the first benchmark: sharded `.npz`.

Example:

```text
experiments/puck_world/data/processed/v0.1/
  manifest.jsonl
  train/
    shard_00000.npz
    shard_00001.npz
  val/
    shard_00000.npz
  test_condition_forms/
    shard_00000.npz
  test_magnitudes/
    shard_00000.npz
  test_layouts/
    shard_00000.npz
```

Shard contents:

```text
context_image      uint8   [N, H, W, 3]
context_state      float32 [N, state_dim]
condition_params   float32 [N, param_dim]
condition_kind     int64   [N]
text_tokens        int64   [N, max_len]      optional for template-token conditions
arrow_image        uint8   [N, H, W, 1]      optional
demo_trace         float32 [N, T, trace_dim] optional
target_final_pos   float32 [N, num_pucks, 2]
target_traj        float32 [N, horizon, num_pucks, 2]
target_heatmap     float32 [N, H, W] or [N, num_pucks, H, W]
target_events      float32 [N, event_dim]
target_ttc         float32 [N]
seed               int64   [N]
scene_id           int64   [N]
intervention_id    int64   [N]
```

Use `numpy.savez_compressed` at first. Move to Zarr only when random partial reads or very large datasets become painful.

### Manifest

Use `manifest.jsonl`, one line per shard:

```json
{
  "dataset_version": "puck-v0.1",
  "split": "train",
  "path": "train/shard_00000.npz",
  "num_examples": 2048,
  "image_size": 64,
  "num_pucks": 2,
  "horizon": 40,
  "condition_surfaces": ["structured", "text", "arrow"],
  "sim_config_hash": "..."
}
```

Also store a top-level config with simulator and dataset parameters. YAML is readable, but the loaded object should be validated into a Pydantic model before generation or training:

```yaml
dataset_version: puck-v0.1
image_size: 64
horizon: 40
num_pucks: [1, 3]
friction: 0.98
restitution: 0.85
splits:
  train: 100000
  val: 10000
  test_condition_forms: 10000
  test_magnitudes: 10000
  test_layouts: 10000
```

## Data Splits

The split design matters more than raw dataset size.

Recommended splits:

```text
train
  seen templates, seen magnitudes, seen obstacle families

val
  same distribution as train

test_condition_forms
  held-out phrasings, held-out arrow styles, held-out demo lengths

test_magnitudes
  held-out force magnitudes and horizons

test_layouts
  held-out object counts, obstacle placements, or goal layouts

test_multi_query
  many conditions per same context
```

The `test_multi_query` split should group examples by `scene_id` so evaluation can measure reuse:

```text
one scene
many candidate interventions
many equivalent condition surfaces
many slightly different conditions
```

## Model Design

The model design has two layers:

```text
shared trunk
  context encoder
  condition encoder
  predictor
  generic latent output

experiment adapter
  target heads
  losses
  metrics
  visualization
```

This keeps TEPA portable without pretending all experiments share identical targets.

### Initial Representation Choice

The first implementation should use one vector per stream:

```text
z_context   = E_context(x)      float32 [d]
z_condition = E_condition(c)    float32 [d]
z_target    = E_target(y_c)     float32 [d]
z_hat_target = P(z_context, z_condition)
```

This is the simplest version of the claim. It gives the experiment a clear answer to measure: does a separately encoded condition vector improve conditioned prediction over a monolithic context-plus-condition model?

Do not start with token matrices, object slots, or variable-length latent sets. They may be useful later, but they add another design question before the core condition-path hypothesis has been tested. If the single-vector version fails, the failure analysis should determine whether the bottleneck is condition understanding, context representation, target construction, model capacity, or loss design before adding a richer latent format.

### TEPA Model

Context encoder:

```text
small CNN or tiny ViT
input: context_image
output: z_context [d]
```

Condition encoder:

```text
structured MLP for condition_params
small text encoder for template-token text
small CNN for arrow_image
GRU/Transformer/MLP for demo_trace
fusion MLP to common z_condition [d]
```

Predictor:

```text
MLP or small cross-attention module
input: z_context, z_condition
output: z_hat_target [d]
```

Target heads:

```text
final_pos_head
trajectory_head
trajectory_to_heatmap_renderer
collision_event_head
goal_success_head
time_to_contact_head
```

These heads are puck-world-specific. They should live under the puck-world experiment package or be registered as experiment adapters, not hardcoded into the generic TEPA trunk.

### Monolithic Baseline

The primary baseline should have a similar parameter count and receive context and condition together through one fused encoder:

```text
context image -> patch tokens
condition text -> byte/token embeddings
optional structured values -> learned numeric/field embeddings

[image tokens, condition tokens, optional structured tokens]
  -> shared transformer encoder
  -> pooled fused latent
  -> same target heads
```

This is still a monolithic baseline because it does not expose a reusable `z_condition` or a separate condition encoder. It has modality adapters at the edge, then one shared trunk that produces one fused context-condition latent.

Keep a literal image-stuffing baseline as a secondary diagnostic:

```text
context image + rendered condition byte grid
  -> single CNN encoder
  -> same target heads
```

That stress test is useful because it asks whether naive context stuffing can discover the condition at all, but it is not the fairest comparison. A reviewer would reasonably object if the paper used byte-grid stuffing as the main baseline.

Important: do not make the baseline weak on purpose. The test is meaningful only if the baseline is credible.

### Future Token or Slot Variant

A later experiment can replace each single vector with a small set of latent tokens or slots:

```text
Z_context    = E_context(x)      float32 [n_context, d]
Z_condition  = E_condition(c)    float32 [n_condition, d]
Z_target     = E_target(y_c)     float32 [n_target, d]
Z_hat_target = P(Z_context, Z_condition)
```

This variant is worth testing if the single-vector model struggles with object binding, multiple simultaneous outcomes, spatially localized queries, or compositional conditions. For example, separate context slots could represent pucks, walls, obstacles, and goal regions; condition slots could represent selected object, direction, magnitude, and horizon; target slots could represent future object states and event summaries.

Treat this as a future scaling path, not as part of the first benchmark. The first benchmark should deliberately keep the representation surface small so that any TEPA-vs-monolithic difference is easier to interpret.

### Pretraining Strategy

Use staged training first:

```text
1. Pretrain E_context and E_target with self-supervised, prediction-oriented objectives.
2. Freeze or lightly tune them.
3. Train E_condition and P on conditioned rollouts.
4. Add probes to verify position, velocity, identity, contact, and uncertainty are recoverable.
5. Unfreeze gradually and fine-tune end to end.
```

Pretraining should preserve dynamics-relevant detail. Avoid objectives that become invariant to exact position, velocity, identity, count, or timing.

Good pretraining tasks:

```text
masked scene/state modeling
future latent prediction
adjacent versus non-adjacent state contrast
trajectory reconstruction
state-delta prediction
object-centric prediction
```

## Metrics

Primary metrics:

```text
final position MSE
trajectory heatmap MSE or BCE
collision event accuracy / F1
goal success accuracy
time-to-contact MAE
retrieval rank of true target latent
```

Condition-specific metrics:

```text
equivalent-condition consistency
  distance between predictions for text/JSON/arrow/demo versions of same intervention

condition sensitivity
  target change should track small changes in force, direction, object, or horizon

condition ablation failure
  predictions should degrade when condition is shuffled or zeroed

held-out condition-form generalization
  performance on unseen text templates, arrows, and demo traces

multi-query efficiency
  runtime for many conditions over the same context
```

TEPA should not be declared successful merely because it has lower average error. The strongest claim requires:

```text
same or better target accuracy
better held-out condition-form generalization
better condition sensitivity
lower compute for repeated queries over one context
cleaner diagnostics under ablation
```

## Result Display

Results should be easy to inspect visually and quantitatively.

### Static Report

Generate a run report as Markdown or HTML:

```text
experiments/puck_world/reports/runs/<run_id>/index.html
```

Include:

```text
model config
dataset config
training curves
validation metrics
test split summary
baseline comparison table
example rollouts
failure cases
condition ablation plots
```

### Visual Panels

For each sampled example, show:

```text
context image
condition surface
ground-truth trajectory heatmap
predicted trajectory heatmap
final position overlay
collision/contact flags
top nearest retrieved outcomes
```

For equivalent condition groups:

```text
same scene
same canonical intervention
text condition prediction
structured condition prediction
arrow condition prediction
demo condition prediction
ground truth
pairwise prediction distances
```

For sensitivity:

```text
same scene
force magnitude sweep
horizon sweep
direction sweep
predicted versus true final positions
```

### Tables

Minimum comparison table:

```text
model
params
train split loss
val loss
test_condition_forms
test_magnitudes
test_layouts
equivalent-condition consistency
condition sensitivity
multi-query runtime
```

### Plots

Recommended plots:

```text
training and validation loss curves
per-head metric curves
scatter: predicted vs true final x/y
heatmap IoU or correlation distribution
bar chart by condition surface
bar chart by split
runtime vs number of queries per context
```

## Phased Implementation

### Phase 0: Smoke Test

Goal: verify pipeline end to end.

```text
1 puck
no obstacles
structured condition only
target: final position
dataset: generated on the fly
models: tiny TEPA and tiny fused monolithic baseline
```

Success:

```text
both models overfit 1,000 examples
simulator is deterministic
visualization overlays are correct
```

### Phase 1: Dynamics Target Bundle

Goal: predict richer targets.

```text
1 to 2 pucks
walls
target: final positions + trajectory heatmap + wall contacts
dataset: frozen .npz shards
```

Success:

```text
target heads train stably
decoded trajectories align with ground truth
condition ablation hurts performance
```

### Phase 2: Condition Surface Equivalence

Goal: test the core TEPA claim.

```text
condition surfaces: structured, text templates, arrow masks
same canonical intervention appears in multiple surfaces
held-out text templates and arrow styles
```

Success:

```text
TEPA beats baseline on held-out condition forms
equivalent condition predictions are closer for TEPA
small condition changes still change the target
```

### Phase 3: Demonstrations and Multi-Query Reuse

Goal: add richer unstructured condition surfaces and efficiency tests.

```text
demo trace condition
many conditions per same scene
multi-query runtime measurement
```

Success:

```text
TEPA reuses context embeddings efficiently
demo conditions align with equivalent text/structured conditions
```

### Phase 4: Uncertainty

Goal: handle multimodal or noisy outcomes.

```text
sample noisy rollouts for the same canonical intervention
target: mean trajectory, variance, event probability, or multiple samples
```

Success:

```text
model confidence tracks empirical rollout variance
uncertain cases are calibrated better than deterministic point estimates
```

## Practical Defaults

Initial defaults:

```yaml
image_size: 64
horizon: 40
num_pucks: 1
batch_size: 256
context_dim: 128
condition_dim: 128
target_dim: 256
optimizer: adamw
learning_rate: 0.0003
weight_decay: 0.01
train_examples: 50000
val_examples: 5000
test_examples_per_split: 5000
```

Scale only after the smoke test and visualization are correct.

## Open Questions

- Should `E_target` encode raw outcome bundles, rendered outcome images, or both?
- Should condition-equivalent examples receive an explicit contrastive alignment loss?
- Should TEPA predict one latent, a set of latent hypotheses, or a distribution over target latents?
- How much of the context encoder should be frozen after pretraining?
- Which target facets are directly supervised versus probed after training?
- Does the shared target space improve transfer, or does it blur important domain-specific detail?

## First Build Checklist

1. Implement deterministic simulator and renderer.
2. Generate and visualize 100 random rollouts.
3. Implement on-the-fly PyTorch dataset.
4. Train a tiny model to overfit final position on 1,000 examples.
5. Add frozen `.npz` shard generation and manifest.
6. Add TEPA and monolithic baseline with comparable parameter counts.
7. Add trajectory heatmap and collision heads.
8. Add text and arrow condition surfaces.
9. Add held-out condition-form split.
10. Generate the first HTML report comparing TEPA and baseline.
