# TEPA Whitepaper

Astro source for the TEPA whitepaper.

## Commands

```sh
npm run dev
npm run build
npm run preview
```

## Source Layout

```text
src/pages/index.astro       Main whitepaper draft
src/components/             Reusable callout and generated diagram components
src/data/diagrams.ts        Editable graph data for generated SVG figures
AGENTS.md                   Working rules for future agent edits
```

The Astro page is the canonical draft. The DOCX in the parent directory is legacy source material and should not be edited directly unless a Word deliverable is explicitly requested.
