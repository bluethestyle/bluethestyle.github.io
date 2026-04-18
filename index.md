---
layout: default
title: Home
---

# Seonkyu Jeong

**Independent researcher based in Seoul.**
Working at the intersection of financial services, AI/ML systems,
and regulatory compliance. GARP Financial Risk Manager (FRM).

This site collects notes, working papers, and long-form thinking
on topics that do not fit cleanly into a journal paper or a GitHub
README. Most posts are bilingual (English with Korean context where
relevant).

---

## What I am working on

### Heterogeneous Expert PLE for Financial Product Recommendation

A 13-task multi-task learning system with seven structurally
different expert networks (DeepFM, Temporal Ensemble, Hyperbolic
GCN, PersLay-TDA, LightGCN, Causal, Optimal Transport), distilled
to LightGBM for AWS Lambda serving, with regulatory-grade audit
infrastructure built into the architecture — not bolted on.

Built by three people on personal time, starting early 2026, with
Claude Code (Anthropic) as the primary development partner.
Roughly 90% of the codebase and both preprints went through
Opus + Sonnet collaboration.

**Preprints** (Zenodo, CC BY 4.0, permanent DOIs):

- **Paper 1** — *Heterogeneous Expert PLE: An Explainable Multi-Task Architecture for Financial Product Recommendation*
  DOI: [10.5281/zenodo.19621884](https://doi.org/10.5281/zenodo.19621884)
- **Paper 2** — *From Prediction to Persuasion: Agentic Recommendation Reason Generation for Regulatory-Compliant Financial AI*
  DOI: [10.5281/zenodo.19622052](https://doi.org/10.5281/zenodo.19622052)
- **Paper 3** — *Loss Dynamics* (work in progress)

**Source code** (MIT License):

- Repository: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial)

### How to cite

```bibtex
@misc{jeong2026heteroexpertple,
  author    = {Jeong, Seonkyu and Sim, Euncheol and Kim, Youngchan},
  title     = {{Heterogeneous Expert PLE: An Explainable Multi-Task
                Architecture for Financial Product Recommendation}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19621884},
  url       = {https://doi.org/10.5281/zenodo.19621884}
}

@misc{jeong2026agenticreason,
  author    = {Jeong, Seonkyu and Sim, Euncheol and Kim, Youngchan},
  title     = {{From Prediction to Persuasion: Agentic Recommendation
                Reason Generation for Regulatory-Compliant Financial AI}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19622052},
  url       = {https://doi.org/10.5281/zenodo.19622052}
}
```

**Production deployment** on a Korean financial institution's live
data is running in parallel; a v2.0 preprint with real-data results
will follow.

---

## Why this blog

A paper tells you what was built. A README tells you how to run it.
Neither tells you *why we made the decisions we did*, *what we tried
that did not work*, or *what it felt like to collaborate with an AI
system over three months*. That is what goes here.

Topics I expect to write about:

- Embedding model risk management into AI system architecture
  (SR 11-7, EU AI Act, Korean AI Basic Act)
- Three-person teams and consumer hardware as a legitimate
  research substrate
- Negative results — what adaTT, GradSurgery, and several
  other ideas looked like when they failed
- Claude Code workflows that actually work for multi-month projects
  (CLAUDE.md, auto-memory, parallel subagents)
- Financial-sector-specific AI problems that do not map cleanly
  onto e-commerce recommender conventions

---

## Posts

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
  </li>
{% endfor %}
</ul>

---

## Elsewhere

- [GitHub — bluethestyle](https://github.com/bluethestyle) — source code and open data
- Zenodo — [Paper 1](https://doi.org/10.5281/zenodo.19621884) · [Paper 2](https://doi.org/10.5281/zenodo.19622052)
- [ORCID: 0009-0005-3291-9112](https://orcid.org/0009-0005-3291-9112) — academic identity

---

## Contact

Email: jsk320098 \[at\] gmail \[dot\] com

Korean-language inquiries welcome.
Please include *blog* in the subject line so it does not land in spam.

---

<small>
All content on this blog is released under
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
unless otherwise noted.
Posts may be drafted with Claude Code assistance; ideas,
experience, and final review are by the author.
</small>
