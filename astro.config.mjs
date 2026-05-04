import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// Convert ```mermaid blocks into raw <pre class="mermaid"> HTML before
// Shiki touches them, so Base.astro's client-side mermaid loader can
// render them without fighting a syntax highlighter.
function remarkMermaid() {
  return (tree) => {
    const walk = (node) => {
      if (!node.children) return;
      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        if (child.type === "code" && child.lang === "mermaid") {
          const escaped = (child.value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
          node.children[i] = {
            type: "html",
            value: `<pre class="mermaid">${escaped}</pre>`,
          };
        } else {
          walk(child);
        }
      }
    };
    walk(tree);
  };
}

// https://astro.build/config
export default defineConfig({
  site: "https://bluethestyle.github.io",
  output: "static",
  trailingSlash: "always",
  integrations: [react(), mdx()],
  markdown: {
    remarkPlugins: [remarkMath, remarkMermaid],
    rehypePlugins: [rehypeKatex],
  },
  // MRM Thread Ep 4-6 overhaul (2026-05-04): old slugs → new slugs.
  // Old eps were regulation walkthroughs (FRIA / Human Oversight / Fairness);
  // new eps are philosophy-led (XAI foundation / RAG audit / Modular adaptability).
  // Activate redirect entries as each new ep file is created — leaving Ep 5/6
  // commented until the new files exist avoids broken-link state.
  redirects: {
    "/2026/04/28/mrm-ep4-fria/": "/2026/04/28/mrm-ep4-xai-foundation/",
    "/2026/04/28/mrm-ep4-fria-ko/": "/2026/04/28/mrm-ep4-xai-foundation-ko/",
    "/2026/05/01/mrm-ep5-human-oversight/": "/2026/05/01/mrm-ep5-rag-lancedb/",
    "/2026/05/01/mrm-ep5-human-oversight-ko/": "/2026/05/01/mrm-ep5-rag-lancedb-ko/",
    "/2026/05/05/mrm-ep6-fairness-production-path/": "/2026/05/05/mrm-ep6-modular-adaptability/",
    "/2026/05/05/mrm-ep6-fairness-production-path-ko/": "/2026/05/05/mrm-ep6-modular-adaptability-ko/",
  },
});
