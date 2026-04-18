# bluethestyle.github.io

Source for the personal blog of **Seonkyu Jeong** —
an independent researcher working on financial AI, model risk
management, and agentic systems.

🔗 **Live site**: [https://bluethestyle.github.io](https://bluethestyle.github.io)

Built with [Jekyll](https://jekyllrb.com) and hosted on
[GitHub Pages](https://pages.github.com).

---

## What lives here

- **`index.md`** — home page (bio, research summary, post list)
- **`about.md`** — longer author bio, co-authors, licensing
- **`_posts/`** — blog posts, one file per post (`YYYY-MM-DD-slug.md`)
- **`_config.yml`** — Jekyll site configuration

---

## Research artifacts

The research this blog discusses lives in a separate repository:

- **[bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)**
  — source code (MIT License) for the Heterogeneous Expert PLE project

Accompanying preprints on Zenodo (CC BY 4.0):

- Paper 1 — *Heterogeneous Expert PLE: Architecture & Ablation*
  [doi.org/10.5281/zenodo.19621884](https://doi.org/10.5281/zenodo.19621884)
- Paper 2 — *From Prediction to Persuasion: Agentic Reason Generation & Compliance*
  [doi.org/10.5281/zenodo.19622052](https://doi.org/10.5281/zenodo.19622052)

---

## Adding a post

Create a file in `_posts/` named `YYYY-MM-DD-slug.md`:

```markdown
---
layout: default
title: "Post title here"
date: 2026-05-01
---

# {% raw %}{{ page.title }}{% endraw %}

*{% raw %}{{ page.date | date: "%Y-%m-%d" }}{% endraw %}*

Post content...
```

Push to `main` — GitHub Pages rebuilds in 1-2 minutes.
Check the [Actions tab](https://github.com/bluethestyle/bluethestyle.github.io/actions)
for build status.

## Local preview (optional)

```bash
# one-time setup
gem install bundler jekyll

# create a Gemfile
cat > Gemfile <<'EOF'
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
EOF

# serve locally
bundle install
bundle exec jekyll serve
```

Open [http://localhost:4000](http://localhost:4000) — changes
auto-reload on file save.

---

## Co-authors

The research discussed on this blog is joint work with
**Euncheol Sim** and **Youngchan Kim**. Both have write access to
this repository to contribute corrections, co-authored posts, or
Korean translations.

---

## License

- **Text and posts**: Creative Commons Attribution 4.0 International
  (CC BY 4.0)
- **Jekyll template / layout files**: MIT License

Research papers and source code linked from this blog have their
own licenses; see their respective repositories for details.

---

## Acknowledgments

Posts on this blog may be drafted with
[Claude Code (Anthropic)](https://claude.com/claude-code)
assistance. Ideas, experience, and final review are by the author.
This practice is consistent with the authors' stated methodology
across the entire research project.
