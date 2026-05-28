# Free-Form Action Conditioning for JEPA-Style World Models

This repository contains an exploratory technical note and local experiment code for testing whether a LeWorldModel-style action term can be generalized into a flexible condition input for JEPA-style latent prediction.

The core question is:

> Can context and condition both be treated as flexible encoders while still producing quality JEPA-style predictions?

The proposed architecture explored here is **TEPA**: Tri-Embedding Predictive Architecture. TEPA keeps context, condition, and target as distinct embeddings, then tests whether that factorization preserves prediction quality while improving flexibility, reuse, and diagnostics.

## Current Status

This is early research code and a concise technical note, not a finished paper or polished library. The current evidence comes from a deterministic 2D puck-world benchmark with structured condition inputs.

The current result is capability evidence, not proof of architectural superiority:

- Standard TEPA, context-memory TEPA, and a context-stuffed JEPA baseline all learn useful predictions against the same frozen target space.
- Context-memory TEPA is currently the strongest factorized variant across most three-seed validation, counterfactual, and latent-probe metrics.
- The context-stuffed JEPA baseline remains a strong comparator, especially when early context-condition interaction is valuable.

## Repository Layout

```text
src/pages/index.astro        Concise technical note
src/pages/tepa.astro         Longer early TEPA whitepaper draft, not linked from the homepage
src/components/              Astro components for callouts and generated diagrams
src/data/diagrams.ts         Editable graph data for generated SVG figures
docs/                        Experiment notes, design-space notes, and comparison summaries
experiments/puck_world/      uv-packaged Python puck-world evaluation
public/figures/              Small static figures used by the note
AGENTS.md                    Working rules for future agent edits
```

Generated datasets, reports, checkpoints, virtual environments, and local SQLite indexes are intentionally ignored by Git.

Published metrics in the note and docs are copied from local runs. The generated run directories, datasets, model weights, and SQLite indexes are not committed.

## Site Commands

Run from the repository root:

```sh
npm install
npm run dev
npm run build
```

The Astro site builds as a static site under `dist/`.

## Experiment Commands

Run from `experiments/puck_world/`:

```sh
uv run pytest
uv run tepa-puck-generate --config configs/smoke.yaml
uv run tepa-puck-validate-data --dataset data/processed/puck-smoke
uv run tepa-puck-pretrain-target --config configs/smoke.yaml
uv run tepa-puck-train --config configs/smoke.yaml --model tepa_latent --target-checkpoint reports/runs/<target_run>/model.pt --freeze-target
uv run tepa-puck-evaluate --run reports/runs/<run_id>
```

See [experiments/puck_world/README.md](experiments/puck_world/README.md) for the full experiment workflow.

## Reproduce

Verify the site and experiment package:

```sh
npm install
npm run build
cd experiments/puck_world
uv run pytest
```

Run a small smoke experiment:

```sh
uv run tepa-puck-generate --config configs/smoke.yaml
uv run tepa-puck-validate-data --dataset data/processed/puck-smoke
uv run tepa-puck-pretrain-target --config configs/smoke.yaml
uv run tepa-puck-train --config configs/smoke.yaml --model tepa_latent --target-checkpoint reports/runs/<target_run>/model.pt --freeze-target
uv run tepa-puck-evaluate --run reports/runs/<run_id>
```

Run the larger structured benchmark:

- Use the configs under `experiments/puck_world/configs/`.
- Start with `configs/phase1_json_latent_context_aux.yaml` for the frozen-target predictor comparison.
- Use `configs/counterfactual_eval_json.yaml` for the separate counterfactual evaluation set.
- See [experiments/puck_world/README.md](experiments/puck_world/README.md) for the multi-seed comparison, identifiability report, and counterfactual benchmark commands.

## Verification

Current local checks:

```sh
npm run build
cd experiments/puck_world && uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).

## References

The note is grounded in JEPA, LeJEPA, LeWorldModel, and recent LeJEPA identifiability work. The compact reference list is maintained in `src/pages/index.astro`.
