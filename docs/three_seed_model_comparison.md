# Three-Seed Puck-World Model Comparison

This document summarizes the first three-seed comparison between three conditioned latent prediction architectures:

- standard TEPA with a single reusable context vector;
- context-memory TEPA with cached context tokens and a condition-specific readout;
- a context-stuffed JEPA-style fused transformer.

The goal of this comparison is not to prove broad generality. It is to test whether TEPA-style reusable context can remain competitive with a strong early-fusion baseline on the first puck-world benchmark, while preserving the runtime and modularity benefits that motivate the architecture.

## Executive Summary

Across seeds `17`, `43`, and `101`, context-memory TEPA is the strongest current candidate overall. It has the best mean validation target-latent MSE, validation final-position MSE, validation wall-contact F1, counterfactual target-latent MSE, counterfactual final-position MSE, and the strongest latent identifiability probe scores.

The fused context-stuffed model remains a serious baseline. It wins counterfactual wall-contact F1 and has the important advantage that its joint context-condition encoder is trained end to end for the prediction task. This may let it learn a more purpose-built representation than the TEPA models, whose condition encoders were pretrained and frozen.

The standard TEPA model remains useful as the simplest reusable-context reference point. It is stable across seeds, but the single-vector context bottleneck appears weaker than context-memory TEPA's cached token memory plus cross-attention readout.

## Models Compared

| Model | Code name | Trainable params | Context handling | Condition handling | Main tradeoff |
| --- | --- | ---: | --- | --- | --- |
| Standard TEPA | `tepa_latent_vit` | 811,790 | ViT image encoder compresses the scene to one vector, `z_context`. | Byte-text condition encoder produces one vector, `z_condition`; pretrained and frozen in this comparison. | Cleanest factorization and easiest context cache, but the context vector is condition-agnostic. |
| Context-memory TEPA | `context_memory_tepa` | 845,326 | ViT-like encoder produces reusable patch-token memory, `C`. | Pretrained frozen condition vector queries the cached context memory through cross-attention. | Middle path: reusable context plus condition-specific context selection. |
| Context-stuffed JEPA | `fused_latent_transformer` | 853,774 | Image patches are concatenated with condition text tokens and encoded jointly. | No separate condition checkpoint; condition tokens are learned inside the fused transformer. | Strong early interaction, but the context must be re-encoded for each condition. |

### Standard TEPA

Standard TEPA keeps the original factorization:

```text
z_context = E_context(context)
z_condition = E_condition(condition)
z_hat_target = P(z_context, z_condition)
```

In this run, `E_context` is a small ViT-style image encoder. `E_condition` is a byte-level text encoder initialized from condition-semantics pretraining and frozen. The predictor is an MLP over the concatenated context and condition vectors. The target encoder and decoder are loaded from a frozen target autoencoder checkpoint, so the predictor is trained to land in a fixed target latent space.

This is the cleanest test of TEPA's modular premise. It is also the harshest form of the context bottleneck: every future condition must use the same single scene vector.

### Context-Memory TEPA

Context-memory TEPA modifies the reusable context side:

```text
C = E_context_tokens(context)
q = E_condition(condition)
r = CrossAttention(q, C)
z_hat_target = P(r, q, mean(C))
```

The context encoder still runs once per scene, but it returns a patch-token memory instead of one pooled vector. The condition vector performs a lightweight cross-attention read from that memory. This lets the condition select different scene details without forcing the whole context and condition through a fully fused transformer for every query.

This model is the clearest current compromise between TEPA and context stuffing. It preserves cached context reuse, but avoids making the context representation entirely condition-agnostic.

### Context-Stuffed JEPA

The fused model is the primary early-fusion baseline:

```text
z_context_condition = E_fused(context, condition)
z_hat_target = P(z_context_condition)
```

Image patches and condition text tokens are concatenated into one transformer sequence. The model can learn direct cross-attention between condition tokens and image patches throughout the encoder. That makes it a strong accuracy baseline, especially when the condition determines which context details matter.

The cost is reuse. For `K` different conditions on the same scene, fused context stuffing reprocesses the image tokens `K` times. TEPA-style models can encode the context once and reuse the cached vector or token memory.

## Evaluation Setup

The comparison used a fixed JSON-only puck-world benchmark:

```text
config: experiments/puck_world/configs/phase1_json_latent_context_aux.yaml
target checkpoint: experiments/puck_world/reports/runs/1779776068_target_autoencoder/model.pt
condition checkpoint: experiments/puck_world/reports/runs/1779812783_condition_semantics/model.pt
counterfactual dataset: experiments/puck_world/data/processed/puck-v0.1-counterfactual-eval-json
comparison report: experiments/puck_world/reports/seed_comparisons/1779918499_three_seed_comparison/
```

The training dataset uses 64x64 rendered puck-world scenes, a 40-frame horizon, 1-2 pucks, 1,000 scenes, 16 intervention events per scene, and 4 condition renderings per event. The active comparison filters condition renderings to JSON so the first architecture comparison is not dominated by natural-language surface variation.

The held-out validation split is assigned by `scene_hash`, not row order, so validation scenes are held out as worlds rather than just alternate condition rows. The counterfactual dataset is eval-only and separate from the training dataset. It contains 256 held-out two-puck scenes, 64 canonical intervention events per scene, and 4 semantically equivalent JSON renderings per event.

The three seeds are training-randomness seeds. They do not create new dataset splits. The dataset, target checkpoint, condition checkpoint, and counterfactual dataset are fixed. The seeds affect model initialization, data-loader order, dropout, and PyTorch randomness.

## Results

Values are mean +/- standard deviation across seeds `17`, `43`, and `101`. Bold values mark the current winner for each metric.

| Metric | Standard TEPA | Context-memory TEPA | Context-stuffed JEPA |
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
| Prediction latent std mean | **0.957430 +/- 0.003435** | 0.952828 +/- 0.004593 | 0.948214 +/- 0.026605 |

## Analysis

### Context-memory TEPA is the strongest current model

Context-memory TEPA wins most of the target-space and endpoint metrics, and its standard deviations are small. That pattern matters more than a single scalar win. It suggests that cached context tokens plus condition-specific readout are a real improvement over compressing the scene to one condition-agnostic vector.

The identifiability probes point in the same direction. Context-memory TEPA has the strongest prediction-to-target linear R2, final-position probe R2, and condition-parameter probe R2. In plain terms, its predicted latents are not just decoding to decent puck positions; they also retain more linearly recoverable information about the simulated world and intervention.

### Fused context stuffing remains a strong and potentially advantaged baseline

The fused model wins counterfactual wall-contact F1 and remains close on some endpoint metrics. It also has a structural advantage in this comparison: its joint context-condition encoder is trained directly by the prediction objective. By contrast, the TEPA condition encoders are initialized from condition-semantics pretraining and then frozen.

That may be a meaningful confound. The fused model can learn a task-specific condition interface that is useful for puck-world prediction, even if it is less modular. TEPA is being asked to preserve a reusable condition representation. That is closer to the long-term architectural premise, but it may understate TEPA's best possible prediction accuracy on this benchmark.

The next fair ablation should compare:

- frozen-condition TEPA, which is the current setup;
- fine-tuned-condition TEPA, initialized from the condition checkpoint but updated by prediction loss;
- scratch-condition TEPA, trained only through the prediction task;
- fused context stuffing, unchanged as the early-fusion baseline.

### Standard TEPA is useful but likely underpowered by the single-vector context

Standard TEPA is stable and competitive, but it does not win the main metrics. The likely bottleneck is the single reusable context vector. A condition such as "push object 1 upward" may need different scene details than "push object 0 leftward," but standard TEPA gives both conditions the same compressed scene embedding.

That does not make the architecture moot. It clarifies where the pressure is. A reusable context representation may need to be richer than one vector if conditions must select different details from the same world.

### The result supports parity plus amortization, not an accuracy proof

The three models are still close enough that this should not be framed as a decisive prediction-quality victory. The strongest defensible claim is narrower:

> Reusable-context architectures can preserve broadly comparable prediction quality while enabling cached-context reuse for multi-query counterfactual evaluation.

That is still a useful result. If a scene is queried once, fused early interaction is attractive. If a scene is queried many times under many conditions, TEPA-style cached context becomes more attractive because the expensive context encoding can be reused.

### The identifiability report is an important addition

The new LeJEPA identifiability paper motivates looking beyond decoded endpoint metrics. A useful world-model latent should behave like a coordinate system for the simulated world, at least locally. The linear-probe results are therefore important: they test whether predicted latents preserve recoverable information about outcomes and conditions, not only whether a decoder head can map them to final positions.

Context-memory TEPA currently looks best under that lens. The fused model has higher variance on identifiability metrics, which suggests it may be more sensitive to initialization or more prone to task-specific shortcuts in this setup.

## Conclusion

The three-seed comparison strengthens the current working thesis: conditioned latent prediction is viable in this toy domain, and TEPA-style reusable context is competitive with a strong fused context-condition transformer.

The best current architecture is context-memory TEPA. It keeps the practical TEPA advantage of reusable context, but gives the condition a lightweight way to select relevant scene details. That appears to address the main weakness of standard single-vector TEPA without giving up the core amortization premise.

The fused context-stuffed model should remain the primary baseline. It is strong, conceptually simple, and may be better when single-query accuracy matters more than reusable context. Its current comparison may also be helped by a trainable joint context-condition encoder, so the next ablation should focus on the TEPA condition encoder: frozen versus fine-tuned versus trained from scratch.

The right framing is not "TEPA has defeated context stuffing." The better conclusion is:

> Context-memory TEPA is the strongest current point in this small design space, and the broader idea of conditioned latent prediction remains promising. The next question is how much modularity can be preserved while allowing the condition path to become task-useful.
