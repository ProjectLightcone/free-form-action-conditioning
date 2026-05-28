# AGENTS.md

## Project Purpose

This repository contains a concise Astro technical note and a local puck-world experiment package for exploring free-form action conditioning in JEPA-style world models.

The main public note is `src/pages/index.astro`. The longer `src/pages/tepa.astro` page is an early draft and is intentionally not linked from the homepage. Treat the homepage as the public-facing artifact unless the user explicitly asks to work on the long draft.

## Research Framing

- Keep the central question clear: can context and condition both be flexible encoders while still producing quality JEPA-style predictions?
- Frame TEPA as the proposed architectural answer being tested, not as a proven winner.
- Preserve the core TEPA thesis: context, condition, and target are distinct learned terms for latent consequence prediction.
- Describe context-memory TEPA as a richer reusable context cache with condition-specific cross-attention readout. Do not imply that context caching is unique to context-memory TEPA; standard TEPA can also cache a compact context vector.
- Describe context-stuffed JEPA as a strong baseline that jointly encodes context and condition, with early interaction but less clean context reuse.
- Treat current results as capability evidence and design-space evidence, not proof of architectural superiority.
- When discussing "world understanding," ground the claim in training pressure, data, ablations, and evaluation. The architecture alone is not enough.

## Architecture Content Rules

- Do not turn the base model into a mandatory typed-schema system. The condition interface should be able to support structured and unstructured condition surfaces, including prose, JSON-like records, tables, gestures, demonstrations, tool traces, and multimodal prompts.
- Describe the target space as a shared latent outcome substrate and make the coherence hypothesis explicit when discussing the larger TEPA draft.
- Treat broad cross-domain target sharing as something to earn through staged experiments, not as a settled assumption.
- When useful, describe `y_c` as a conditioned outcome bundle: the direct answer plus uncertainty, alternatives, support variables, evidence anchors, temporal/object/causal structure, constraints, and affordances. Make clear that these facets require training signal.
- In the core math, `z_target` is the conditioned true target. Do not pass the condition into a separate loss-side projection unless a future draft explicitly introduces and justifies that as an extension.
- If a dataset stores a rich full outcome `Y`, require an explicit target-construction mechanism `T(Y, c)` before encoding the target.
- SIGReg should be described accurately for the current frozen-target setup: target autoencoder pretraining uses SIGReg to shape `z_target`; frozen predictor training matches `stopgrad(z_target)` and does not directly apply SIGReg unless a future ablation adds a separate prediction-side regularizer.

## Implementation Rules

- Public note: `src/pages/index.astro`.
- Longer early draft: `src/pages/tepa.astro`.
- Reusable presentation pieces live in `src/components/`.
- Diagrams are generated as inline SVG from editable data in `src/data/diagrams.ts` using the `GeneratedDiagram` component. Prefer editing graph data over exporting bitmap images.
- Experiment code lives under `experiments/puck_world/` and is packaged with `uv`.
- Generated data, reports, checkpoints, virtual environments, SQLite indexes, and build output should remain ignored by Git.
- Keep pages semantic and print-friendly. Use native headings, paragraphs, figures, tables, lists, and captions.
- Use restrained research-document styling. Avoid generic landing-page sections, gradient text, nested cards, and decorative UI that competes with the paper.
- Prefer ASCII punctuation in source text. Avoid em dashes.

## Verification

- Run `npm run build` from the repository root before handing off site changes.
- Run `uv run pytest` from `experiments/puck_world/` before handing off experiment-code changes.
- For visual QA, run `npm run dev` and inspect the page in a browser at the reported localhost URL.
- The page includes print styles so a browser or Playwright can export PDF later.
