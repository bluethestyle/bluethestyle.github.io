/**
 * Posts loader — single source of truth.
 *
 * Titles, dates, tags, categories, lang come from the markdown
 * frontmatter of each post (via Astro's content collection). Excerpts
 * and planned-but-unpublished "draft" entries live in data.js as
 * lightweight auxiliary maps (EXCERPTS, DRAFTS).
 *
 * Returns the legacy POSTS shape so existing consumers don't need to
 * change field names:
 *   { date, title, cat, lang, url, ex, tags, draft, series, part }
 */

import { getCollection } from "astro:content";
import { EXCERPTS, DRAFTS } from "../data.js";

function entryUrl(entry) {
  const d = entry.data.date;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const base = entry.id.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.md$/, "");
  return `/${y}/${m}/${day}/${base}/`;
}

function toPost(entry) {
  const url = entryUrl(entry);
  const d = entry.data.date;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return {
    date: `${y}-${m}-${day}`,
    title: entry.data.title,
    cat: entry.data.categories?.[0] ?? "",
    lang: (entry.data.lang ?? "en").toUpperCase(),
    url,
    ex: entry.data.excerpt ?? EXCERPTS[url] ?? "",
    tags: entry.data.tags ?? [],
    draft: false,
    series: entry.data.series,
    part: entry.data.part,
  };
}

/**
 * Load every published post from the content collection plus the
 * DRAFTS list from data.js, sorted newest first. Draft entries are
 * marked with `draft: true` and have no URL.
 */
export async function loadPosts() {
  const entries = await getCollection("posts");
  const published = entries.map(toPost);
  const drafts = (DRAFTS ?? []).map((d) => ({
    date: d.date,
    title: d.title,
    cat: d.cat,
    lang: d.lang,
    url: undefined,
    ex: d.ex ?? "",
    tags: d.tags ?? [],
    draft: true,
  }));
  return [...published, ...drafts].sort((a, b) =>
    b.date.localeCompare(a.date),
  );
}

/** Top-N recent published posts (date-descending). */
export async function loadRecent(n = 4) {
  const all = await loadPosts();
  return all
    .filter((p) => !p.draft)
    .slice(0, n)
    .map(({ title, date, url }) => ({ title, date, url }));
}
