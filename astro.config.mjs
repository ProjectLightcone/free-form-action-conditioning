// @ts-check
import { defineConfig } from 'astro/config';

const site = process.env.ASTRO_SITE;
const base = process.env.ASTRO_BASE;

// https://astro.build/config
export default defineConfig({
	...(site ? { site } : {}),
	...(base ? { base } : {}),
});
