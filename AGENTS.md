# AGENTS.md

## Project Purpose

This directory contains the Astro source for the TEPA whitepaper. Treat the Astro site as the canonical draft. The legacy DOCX in the parent directory is source material, not the preferred editing surface.

## Architecture Content Rules

- Preserve the core TEPA thesis: context, condition, and target are distinct learned terms for latent consequence prediction.
- Do not turn the base model into a mandatory typed-schema system. TEPA should support structured and unstructured condition surfaces, including prose, JSON-like records, tables, gestures, demonstrations, tool traces, and multimodal prompts.
- Describe the target space as a shared latent outcome substrate and make the coherence hypothesis explicit: target latents should organize around outcome factors such as entities, relations, quantities, temporal structure, constraints, uncertainty, evidence, and affordances rather than output format alone.
- Treat broad cross-domain target sharing as something to earn through staged experiments, not as a settled assumption.
- When useful, describe `y_c` as a conditioned outcome bundle: the direct answer plus uncertainty, alternatives, support variables, evidence anchors, temporal/object/causal structure, constraints, and affordances. Make clear that these facets require training signal.
- In the core math, `z_target` is the conditioned true target. Do not pass the condition into a separate loss-side projection unless a future draft explicitly introduces and justifies that as an extension.
- If a dataset stores a rich full outcome `Y`, require an explicit target-construction mechanism `T(Y, c)` before encoding the target.
- When discussing "world understanding," ground the claim in training pressure, data, ablations, and evaluation. The architecture alone is not enough.

## Implementation Rules

- Main draft: `src/pages/index.astro`.
- Reusable presentation pieces live in `src/components/`.
- Diagrams are generated as inline SVG from editable data in `src/data/diagrams.ts` using the `GeneratedDiagram` component. Prefer editing graph data over exporting bitmap images.
- Keep the page semantic and print-friendly. Use native headings, paragraphs, figures, tables, lists, and captions.
- Use restrained research-document styling. Avoid generic landing-page sections, gradient text, nested cards, and decorative UI that competes with the paper.
- Prefer ASCII punctuation in source text. Avoid em dashes.

## Verification

- Run `npm run build` from this directory before handing off changes.
- For visual QA, run `npm run dev` and inspect the page in a browser at the reported localhost URL.
- The page includes print styles so a browser or Playwright can export PDF later.
