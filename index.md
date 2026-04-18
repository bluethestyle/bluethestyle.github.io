---
layout: default
title: Home
---

# Seonkyu Jeong

Independent researcher based in Seoul.
Working on explainable financial AI, model risk management,
and agentic systems.
GARP Financial Risk Manager (FRM).

**ORCID**: [0009-0005-3291-9112](https://orcid.org/0009-0005-3291-9112)
**GitHub**: [bluethestyle](https://github.com/bluethestyle)

---

## Research

**Heterogeneous Expert PLE for Financial Product Recommendation** — a 13-task
multi-task learning system with seven structurally different expert networks,
distilled to LightGBM for Lambda serving, with regulatory-grade audit
infrastructure. Open-sourced with two Zenodo preprints.

- [Paper 1 — Architecture & Ablation](https://doi.org/10.5281/zenodo.19621884)
- [Paper 2 — Agentic Reason Generation & Compliance](https://doi.org/10.5281/zenodo.19622052)
- [GitHub repository](https://github.com/bluethestyle/aws_ple_for_financial)

---

## Posts

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small>— {{ post.date | date: "%Y-%m-%d" }}</small>
  </li>
{% endfor %}
</ul>

---

## Contact

Email: jsk320098 [at] gmail [dot] com
