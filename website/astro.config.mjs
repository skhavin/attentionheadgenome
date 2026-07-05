import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://skhavin.github.io',
  base: '/attentionheadgenome',
  output: 'static',

  vite: {
    plugins: [tailwindcss()],
  },

  integrations: [mdx()],
});