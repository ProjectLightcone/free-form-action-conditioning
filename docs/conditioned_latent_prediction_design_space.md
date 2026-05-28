# Conditioned Latent Prediction Design Space

The current puck-world results suggest a broader framing than a simple TEPA-versus-baseline contest. Both TEPA and fused context stuffing are plausible routes toward the same larger capability:

```text
world context + condition/query/intervention -> predicted latent outcome
```

That capability is the interesting part. It is a small step toward general-purpose prediction models: systems that can look at a state of the world, accept a flexible question or intervention, and predict a useful outcome representation.

## Two Useful Points In The Space

### Fused Context Stuffing

In the fused approach, the context and condition are placed into one joint model input:

```text
z_context_condition = E_fused(context, condition)
z_hat_target = P(z_context_condition)
```

This is powerful because the condition can influence context processing immediately. Condition tokens can attend to image patches, state tokens, document chunks, or other context elements from the first layers of the model.

Likely strengths:

- Strong single-query accuracy.
- Early cross-attention between condition and context.
- Simple interface: one model input, one fused representation.
- Natural fit for transformer-style multimodal prompting.

Likely weaknesses:

- The context usually has to be re-encoded for every new condition.
- Harder to cache reusable world representations.
- Harder to inspect whether failures come from context understanding, condition understanding, or prediction.
- Less modular if future systems need many condition encoders or specialized condition surfaces.

### TEPA-Style Factorization

In TEPA, context and condition are encoded separately before prediction:

```text
z_context = E_context(context)
z_condition = E_condition(condition)
z_hat_target = P(z_context, z_condition)
```

This is useful when the same world state is queried many times. The context can be encoded once, then reused across many conditions.

Likely strengths:

- Cached context embeddings for many-query counterfactual evaluation.
- Modular condition encoders.
- Cleaner diagnostic surfaces for context, condition, and target.
- Better fit for target-space reuse, retrieval, probes, and downstream decoders.
- Natural support for "ask many questions about the same world state."

Likely weaknesses:

- Later interaction between condition and context may reduce single-query accuracy.
- The context encoder may preserve the wrong information if it is not condition-aware.
- The predictor has to reconcile condition meaning with context after both have been compressed.
- Benefits may only appear when many queries share the same context or when modularity matters.

### Context-Memory TEPA

A third candidate keeps TEPA's reusable context premise but caches the context as a memory of tokens rather than one vector:

```text
C = E_context_tokens(context)
q = E_condition(condition)
r = CrossAttention(q, C)
z_hat_target = P(r, q, mean(C))
```

This is a middle path. The expensive context encoder can still run once per world state, but the condition gets a cheap cross-attention read from the cached context memory. That lets the condition select relevant context details without requiring the entire context and condition to be fused from scratch for every query.

Likely strengths:

- Reusable context memory for many-query evaluation.
- Condition-specific context selection.
- Better context grounding than a single condition-agnostic context vector.
- Potentially closer accuracy to fused context stuffing while retaining TEPA-style caching.

Likely weaknesses:

- More expensive than a single-vector TEPA readout.
- More complex to cache and benchmark correctly.
- The cached memory may still miss condition-relevant information if the context encoder is not trained broadly enough.
- Cross-attention readout may be too shallow for conditions that require deep re-interpretation of the context.

## Current Experimental Read

The current puck-world experiment shows rough parity between reusable-factorized models and a fused context-condition transformer. That should not be overclaimed. It does not yet show that any architecture is categorically more accurate. The latest three-seed pass does suggest:

- conditioned latent prediction is viable in this toy setting;
- all tested architectures use the condition signal;
- TEPA-style reusable context can be competitive;
- fused context stuffing remains a strong accuracy baseline;
- context-memory TEPA is currently the strongest and most stable candidate.

Three-seed frozen-target snapshot. Values are mean +/- standard deviation across seeds `17`, `43`, and `101`. Bold values mark the current winner for each metric:

| Metric | Standard TEPA | Context-memory TEPA | Context-stuffed JEPA |
| --- | ---: | ---: | ---: |
| Validation target latent MSE | 0.073806 +/- 0.000692 | **0.059853 +/- 0.003236** | 0.091188 +/- 0.043016 |
| Validation final-position MSE | 0.006069 +/- 0.000132 | **0.004966 +/- 0.000360** | 0.006445 +/- 0.003753 |
| Validation wall-contact F1 | 0.976738 +/- 0.002419 | **0.980101 +/- 0.001245** | 0.976719 +/- 0.004538 |
| Counterfactual target latent MSE | 0.762004 +/- 0.013295 | **0.661938 +/- 0.007201** | 0.852729 +/- 0.019713 |
| Counterfactual final-position MSE | 0.068043 +/- 0.001533 | **0.061505 +/- 0.000877** | 0.068403 +/- 0.001572 |
| Counterfactual wall-contact F1 | 0.899413 +/- 0.006541 | 0.901635 +/- 0.011977 | **0.919787 +/- 0.012805** |
| Prediction-to-target linear R2 | 0.913299 +/- 0.002923 | **0.932091 +/- 0.001762** | 0.906237 +/- 0.044096 |
| Condition-parameter probe R2 | 0.776008 +/- 0.008914 | **0.818038 +/- 0.017387** | 0.724755 +/- 0.050849 |

The fused baseline should still be read generously: it trains a joint context-condition encoder directly against the prediction task, while the TEPA variants currently use pretrained frozen condition encoders. That may make fused more purpose-built for puck world. The next clean ablation is to compare frozen-condition TEPA, fine-tuned-condition TEPA, scratch-condition TEPA, and fused early interaction with the same target path.

The cleanest current thesis is:

> Conditioned latent prediction is a promising direction for general-purpose prediction models. Standard TEPA, context-memory TEPA, and fused context stuffing are useful architectural points in that design space, with different tradeoffs between early interaction, condition-specific context selection, and reusable factorization.

## Future Questions

The next experiments should ask where each tradeoff matters:

- Single-query accuracy: does fused early interaction dominate?
- Many-query amortization: how much does cached TEPA context help?
- Condition-surface generalization: which architecture handles paraphrases and structured/unstructured variants better?
- Cross-domain transfer: does a separated condition interface help when moving beyond puck world?
- Target-space reuse: which architecture better supports probes, retrieval, decoders, and planning?
- Interpretability: does factorization make failure analysis easier?

The current result does not end the architecture question. It makes the question sharper.
