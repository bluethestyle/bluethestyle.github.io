# bluethestyle — Jekyll site (standalone)

A complete, ready-to-deploy Jekyll site for **bluethestyle**. Serves the design
from the prototype without depending on any upstream theme (Chirpy, Minima, etc.).
Deploys to GitHub Pages via the included Actions workflow.

---

## What you get

```
.
├── _config.yml                 ← site config (author, social, URL)
├── Gemfile                     ← ruby deps (jekyll 4.3 + 3 plugins)
├── .github/workflows/pages.yml ← CI — builds & deploys on push to main
│
├── _layouts/
│   ├── default.html            ← page shell (topbar + 3-col grid + footer)
│   ├── home.html               ← hero + current-work + co-authors + post index + series
│   ├── post.html               ← blog post
│   └── page.html               ← static page (e.g. /about/)
│
├── _includes/
│   ├── topbar.html             ← logo · tabs · search · lang · dark-mode
│   ├── sidebar.html            ← LEFT column — bio · nav · elsewhere · active series
│   ├── right-panel.html        ← RIGHT column — stats · recent · tags · ORCID · colophon
│   ├── footer.html             ← bottom bar
│   └── ple-diagram.svg         ← the 7→gate→13 diagram on the home page
│
├── _data/
│   ├── series.yml              ← "Series · active" sidebar + home grid
│   └── coauthors.yml           ← co-authors block on home
│
├── _posts/                     ← 4 sample posts (EN + 1 KO), dated 2026-03/04
│
├── assets/
│   ├── css/site.css            ← ALL styling (plain CSS, .bts-* scoped)
│   └── js/theme-toggle.js      ← dark-mode toggle, localStorage-backed
│
├── index.html                  ← uses layout: home
├── categories.html             ← /categories/
├── tags.html                   ← /tags/
├── archives.html               ← /archives/  (includes 12-week cadence heatmap)
└── about.md                    ← /about/
```

## Local preview

```bash
bundle install
bundle exec jekyll serve
# → http://127.0.0.1:4000
```

Ruby 3.0+ required. On macOS: `brew install ruby` then `gem install bundler`.

## Deploy on GitHub Pages

1. Push this repo to GitHub as `bluethestyle.github.io` (or any repo; adjust `url` in `_config.yml`).
2. Repo → Settings → **Pages** → Source: **GitHub Actions**.
3. Push to `main`. The included `.github/workflows/pages.yml` runs `jekyll build` and publishes `_site/`.

## Customise

- **Author + social** — `_config.yml` top of file. Every template reads from `site.author` and `site.social`.
- **Active series** — `_data/series.yml`. Controls the sidebar *Series · active* block and the home-page series grid.
- **Co-authors** — `_data/coauthors.yml`. Controls the co-authors block on home.
- **Current working paper** — edit the `<article class="bts-work-card">` block in `_layouts/home.html` directly. (It's hard-coded to one paper; revisit when there's a second.)
- **Colors + type** — `:root` block at the top of `assets/css/site.css`. Both light (Paper) and dark (Ink) themes defined there as CSS custom properties; nothing else touches colors directly.

## Post front matter

```yaml
---
title: "[FinAI Build] Ep 1 — The Premise"
subtitle: "One developer, a stack of agents, and a ninety-day clock."
date: 2026-04-15
last_modified_at: 2026-04-18
lang: en              # or "ko" — flips font stack and the lang badge
categories: [FinAI Build]
tags: [finai-build, claude-code, agents]
toc: true             # optional — kramdown TOC at top
read_time: 7          # optional — shows in meta line
---
```

## Bilingual (EN + KO)

No plugin needed. Just set `lang: ko` on a post. The layout switches to
**Pretendard** and adjusts sizes/letter-spacing. The language toggle in the
topbar currently points to `/` (EN) and `/ko/` (KO); create `ko/index.html`
with `layout: home` and a KO `_posts_ko/` collection when you're ready for a
parallel site — or leave it pointing at a filtered archive.

## Known edges

- The trending-tags block zero-pads counts before string-sorting. It handles up to **999 posts per tag**. If you ever hit that, widen the `"000"` pad in `right-panel.html`.
- The archive heatmap is generated in Liquid at build time — no JS. It's O(posts × 84). Fine up to a few thousand posts; beyond that, cache it.
- Search box in the topbar is visual only. Wire up lunr.js or Pagefind when you want real search.
- Dark mode inherits the OS preference on first load, then persists the user's explicit choice in `localStorage["mode"]`.
