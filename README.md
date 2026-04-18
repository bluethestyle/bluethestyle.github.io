# bluethestyle.github.io

Personal blog of Seonkyu Jeong. Built with Jekyll + GitHub Pages.

Live at: https://bluethestyle.github.io

## Local preview (optional)

If you want to preview locally before pushing:

```bash
# install bundler + jekyll (one-time)
gem install bundler jekyll

# create Gemfile
echo 'source "https://rubygems.org"' > Gemfile
echo 'gem "github-pages", group: :jekyll_plugins' >> Gemfile

# install and serve
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000

Most of the time you can just push to `main` and let GitHub Pages build.

## Adding a post

Create a file in `_posts/` named `YYYY-MM-DD-slug.md`:

```markdown
---
layout: default
title: "My post title"
date: 2026-04-20
---

# {{ page.title }}

Content here...
```

Push to `main` — GitHub Pages rebuilds in 1-2 minutes.
