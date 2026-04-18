---
layout: default
title: "Series: The MRM Thread"
permalink: /series/mrm-thread/
---

# The MRM Thread

*Model Risk Management for AI recommendation systems, written
from a GARP FRM practitioner perspective.*

🇰🇷 [한국어 목차 →](/series/mrm-thread-ko/)

---

A short parallel series to
[*Building a Financial AI in Three Months*](/series/three-months/),
aimed at a different audience — financial practitioners, risk
managers, regulatory staff, and MRM teams.

The core claim: as AI recommendation systems become agent
pipelines rather than single models, the *validation-first* model
of MRM starts to break. The alternative is to push MRM obligations
into the *architecture* itself, so that compliance properties are
structural invariants rather than post-hoc reports.

This series takes three cuts at that claim, using the production
system described in
[Paper 2](https://doi.org/10.5281/zenodo.19622052) as a
worked example.

---

## Episodes

1. **[Why MRM Belongs in the Architecture, Not Validation](/2026/04/18/mrm-ep1-architecture-en/)** — the frame: what breaks when the model is an LLM agent system, and what architectural MRM actually looks like
2. *Champion-Challenger as a Gate, Not a Report* (coming soon) — `ModelCompetition.evaluate()`, `--force-promote` override, HMAC audit entries on every promotion decision
3. *Agentic Systems and the Regulatory Chain of Custody* (coming soon) — seven audit tables, HMAC hash chain verification, EU AI Act 13-14 and KFCPA §17 as code paths

---

## Who this is for

- **GARP FRM-holders** working in financial AI or model risk
- **MRM / second-line** teams at banks, card issuers, insurers
- **Regulatory staff** at FSS, FSC, or international counterparts
- **Financial AI engineers** looking for an MRM-aligned deployment pattern
- Anyone wondering why SR 11-7 needs to evolve when the "model" is a pipeline

---

## What this is not

This series does not argue that MRM is obsolete or that validation
teams should be dissolved. It argues that the *surface of audit*
should shift from "model outputs after training" to "architectural
invariants, continuously."

The validation-first model works. It worked for twenty years on
credit and market risk models. It starts to break, not because it
is wrong, but because the thing it is validating has changed.

---

## Source material

The primary source for this series is
[Paper 2: *From Prediction to Persuasion*](https://doi.org/10.5281/zenodo.19622052)
— specifically §5 (Operational Agent Pipeline), §6 (Regulatory
Compliance), and §7 (Experiments, compliance audit section). The
paper contains the full mapping tables and code references; this
series narrativizes the design choices behind them.

---

## Posts by date

<ul>
{% for post in site.posts %}
  {% if post.series == "mrm-thread" and post.lang == "en" %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
  {% endif %}
{% endfor %}
</ul>
