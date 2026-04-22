// RSS feed for bluethestyle / field notes.
//
// Produces /rss.xml at build time from the Astro content collection.
// Discovered by feed readers via the <link rel="alternate"> tag added
// in Base.astro. Useful as an automated ingestion surface for crawlers
// that prefer feeds over HTML (Anthropic-style content bots, Feedly,
// etc.).
//
// Posts are listed across languages; readers who want a single-language
// feed can filter locally. Each item's description is the excerpt when
// available, else a short fallback.
import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE } from "../data.js";

export async function GET(context) {
  const posts = await getCollection("posts");

  const items = posts
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime())
    .map((entry) => {
      const d = entry.data.date;
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      const base = entry.id
        .replace(/^\d{4}-\d{2}-\d{2}-/, "")
        .replace(/\.md$/, "");
      const url = `/${y}/${m}/${day}/${base}/`;
      return {
        title: entry.data.title,
        pubDate: d,
        link: url,
        description:
          entry.data.excerpt ??
          `${entry.data.categories?.[0] ?? "Post"} — ${entry.data.lang.toUpperCase()}`,
        categories: entry.data.tags ?? [],
        author: SITE.owner,
      };
    });

  return rss({
    title: "bluethestyle / field notes",
    description:
      "Independent research notes on financial AI, model risk " +
      "management, and agentic systems. Papers, decisions, failed " +
      "experiments — from a three-person team building in public.",
    site: context.site,
    items,
    customData: "<language>en</language>",
    stylesheet: false,
  });
}
