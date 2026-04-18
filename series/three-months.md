---
layout: default
title: "Series: Building a Financial AI in Three Months"
permalink: /series/three-months/
---

# Building a Financial AI in Three Months

*A series on building a financial recommendation system with
Claude Code, on consumer hardware, as three people on personal
time. Tag: `[FinAI Build]`*

🇰🇷 [한국어 목차 →](/series/three-months-ko/)

---

Between January and April 2026, three of us — one PM / lead
architect and two engineers — built a 13-task multi-task learning
recommender with seven heterogeneous expert networks, distilled
to LightGBM for Lambda serving, with regulatory-grade audit
infrastructure. We did it on a single consumer GPU (RTX 4070,
12 GB VRAM), on personal time, with Claude Code as the primary
development partner.

This series tells that story.

The papers (on [Zenodo](https://doi.org/10.5281/zenodo.19621884))
explain what we built. The
[GitHub repo](https://github.com/bluethestyle/aws_ple_for_financial)
shows how it works. This series is for the parts that do not fit
in either: why we made the decisions we did, what broke, what
surprised us, what we would not do again, and what three months of
human-AI collaboration actually feels like.

---

## Episodes

1. **[The Premise](/2026/04/18/ep1-premise-en/)** — Team, infrastructure, constraints, and the architecture decision journey from ALS to PLE
2. *The Partnership* (coming soon) — How three people organized the AI collaboration
3. *Guardrails* (coming soon) — CLAUDE.md, Memory Bank, and interface contracts
4. *Technical Challenges* (coming soon) — Data integrity, numerical stability, pipeline engineering
5. *Design Philosophy* (coming soon) — Why seven experts
6. *The Data Integrity Audit* (coming soon) — v3 to v4, deterministic leakage, HGCN vs LightGCN
7. *The Bug That Mattered More Than Architecture* (coming soon) — Uncertainty weighting, softmax vs sigmoid, adaTT at scale, GradSurgery
8. *Results & Lessons* (coming soon) — What we built, what we learned

---

## A companion thread

A second, shorter series — [*The MRM Thread*](/series/mrm-thread/) —
runs in parallel. That one covers the
regulatory and model-risk-management angle: why MRM belongs in
architecture rather than validation, how Champion-Challenger works
as a gate rather than a report, and what SR 11-7, EU AI Act, and
Korean AI Basic Act mapping looks like in practice. Aimed at a
different audience — financial practitioners, risk managers,
regulatory staff — but built from the same project.

---

## Source material

This series adapts and expands
[development_story_en.pdf](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf)
(and its Korean counterpart) from the research repository.
The development story is the single most complete internal
write-up of the project's history. If you prefer reading it as one
document, the PDFs are there.

The series format sacrifices continuity for breathing room — each
episode is a 5–10 minute read, one topic at a time.

---

## Posts by date

<ul>
{% for post in site.posts %}
  {% if post.series == "three-months" and post.lang == "en" %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
  {% endif %}
{% endfor %}
</ul>
