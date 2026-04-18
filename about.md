---
layout: default
title: About
permalink: /about/
---

# About

## Who I am

I am an independent researcher based in Seoul, Republic of Korea,
working at the intersection of financial services, AI/ML systems,
and regulatory compliance.

**Credentials**:

- **GARP Financial Risk Manager (FRM)**
- Career in Korean financial sector spanning credit and market risk
  analysis, regulatory compliance, MyData licensing, big data
  platform operations, data science projects, and recommendation
  system management

**Online identity**:

- GitHub: [bluethestyle](https://github.com/bluethestyle)
- ORCID: [0009-0005-3291-9112](https://orcid.org/0009-0005-3291-9112)
- Email: jsk320098 \[at\] gmail \[dot\] com

---

## Current work

### Heterogeneous Expert PLE for Financial Product Recommendation

A three-person research project started in early 2026, built on
personal time with a single consumer GPU (RTX 4070, 12 GB VRAM)
and no institutional funding.

**Co-authors**:

- Seonkyu Jeong — architecture, regulatory framing, paper writing
- Euncheol Sim — engineering, experimentation
- Youngchan Kim — engineering, evaluation

**What it is**: a 13-task multi-task learning system with seven
structurally distinct expert networks, distilled to LightGBM for
AWS Lambda serving, with regulatory-grade audit infrastructure
designed in at the architecture level — not applied post-hoc.

**Public artifacts**:

- Paper 1 — [doi.org/10.5281/zenodo.19621884](https://doi.org/10.5281/zenodo.19621884)
- Paper 2 — [doi.org/10.5281/zenodo.19622052](https://doi.org/10.5281/zenodo.19622052)
- Source code — [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)

**Non-public artifact**: a separate on-premises production
codebase at a Korean financial institution (12 million real
customers, 734 production features) validates the same patterns at
operational scale. That codebase is not public for regulatory
reasons; a v2.0 preprint on Zenodo will summarize production
validation results.

---

## How this project worked

The system was built with [Claude Code (Anthropic)](https://claude.com/claude-code)
as the primary development partner. Roughly 90% of the codebase
and both Zenodo preprints went through human-AI collaboration —
Claude Opus for architecture and cross-disciplinary reasoning
(topology ↔ finance, chemical kinetics ↔ spending dynamics),
Claude Sonnet for parallel code implementation, bilingual
documentation, and the production agent pipelines.

Patterns that carried the project across 3.5 months and 240+ source
files are documented in the repository's
[Advanced Claude Code Usage Patterns](https://github.com/bluethestyle/aws_ple_for_financial#advanced-claude-code-usage-patterns)
section.

---

## Positioning statement

This blog, the GitHub repository, and the Zenodo preprints are
produced on personal time and reflect **personal research views**.
They are not official output of any affiliated institution and do
not represent the views of any employer, current or past.

Contributions, corrections, and constructive critique are welcome.
For collaboration inquiries — academic, industry, or regulatory —
please use the email above.

---

## License

- **Blog posts**: CC BY 4.0 unless noted otherwise
- **Research papers** (on Zenodo): CC BY 4.0
- **Source code** (on GitHub): MIT License

Attribution is always appreciated but not strictly required for
short quotations within fair-use / fair-dealing doctrine.
