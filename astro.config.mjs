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
});
