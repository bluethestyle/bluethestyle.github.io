---
title: "[MRM Thread] Ep 6 — Modular Adaptability: When Regulations Evolve, Architecture Doesn't"
date: 2026-05-05 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, modularity, regulation, financial-ai, ai-basic-act, eu-ai-act]
lang: en
excerpt: "Korean AI Basic Act enforcement decree, EU AI Act amendments, future US framework — they will all arrive. The five regulatory generators are five modules above the same audit log substrate, not five documents that get rewritten. Why architectural modularity is the long-term bet that makes the previous five episodes pay off."
series: mrm-thread
part: 6
alt_lang: /2026/05/05/mrm-ep6-modular-adaptability-ko/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 6 of "The MRM Thread" — the final episode. Eps 1–5 built
the substrate: MRM in the architecture (Ep 1), the
champion-challenger gate (Ep 2), the audit log layer (Ep 3),
inherent XAI as the explanation column (Ep 4), and RAG +
LanceDB as the retrieval layer (Ep 5). The five regulatory
generators that sit above this substrate — `KoreanFRIAAssessor`,
`FRIAEvaluator`, `AnnexIVMapper`, `PIAEvaluator`,
`PublicDisclosureGenerator` — are described in Paper 2. This
episode is the longer-term thesis behind that whole stack:
regulations are going to keep changing, and the entire point of
the architecture is to make that change cheap. If the change
isn't cheap, the previous five episodes don't pay off.*

## A scenario, twelve months out

It's May 2027. The Korean AI Basic Act enforcement decree — the
시행령 — has just dropped after the year-long calibration period
ended on 2026-01-22. The decree adds two specific obligations
that weren't in the original act:

1. A new dimension in the §35 impact assessment for *cross-border
   data exposure*, which the original seven-dimension assessment
   doesn't cover.
2. A change in the retention requirement: instead of five years
   on the assessment record, the decree adds a *separate*
   three-year retention on the *raw inputs* that fed into the
   assessment, with a specific encryption-at-rest requirement.

Independently, on the EU side, the AI Office has issued a
clarification on Annex IV Section 5 (training data attributes)
that requires *demographic distribution disclosure* on the
training set, not just statistical summary statistics.

In a system where compliance is documents, this is six weeks of
work. Re-draft the FRIA template. Re-draft the EU FRIA. Re-draft
the Annex IV Section 5 narrative. Coordinate retention policy
changes with cloud ops. Update the data-handling SOP. Schedule
training for the compliance team. The risk team picks one of
these to skip because the bandwidth isn't there.

In a system where compliance is queries over an audit log, this
is one PR per generator and a config change. The substrate
underneath doesn't move.

This is what *modular adaptability* means in practice, and why
it's worth the upfront cost of building Eps 1–5 the way we did.

## Why we built five generators instead of one

The temptation, when designing the regulatory layer, was a
single `ComplianceReporter` class that takes a regulation
identifier and returns the appropriate report. One class, one
method, parametrise on jurisdiction. It looks DRY.

We rejected that early for the same reason Ep 4 kept
`KoreanFRIAAssessor` separate from `FRIAEvaluator` even though
their dimensions look similar: *different legal bases must not
share a coupling that makes one regulation's amendment ripple
into another's compliance posture*.

The five generators are deliberately five separate modules:

- `KoreanFRIAAssessor` — Korean AI Basic Act §35, seven-dimension
  impact assessment, five-year retention.
- `FRIAEvaluator` — EU AI Act Article 9, five-dimension risk
  management process record.
- `AnnexIVMapper` — EU AI Act Article 11 + Annex IV,
  twelve-section technical-documentation evidence mapping.
- `PIAEvaluator` — Korean PIPA + GDPR Article 35, six-domain
  privacy impact assessment.
- `PublicDisclosureGenerator` — Korean FSC AI guideline,
  five-section quarterly public disclosure.

Each module owns its own dimensions, retention rules, output
format, and update cadence. They share only the substrate
underneath — the same audit log, the same XAI explanation
column, the same RAG retrieval interface.

When the May 2027 scenario above lands, the changes are scoped:

- New §35 cross-border-data dimension → add a method to
  `KoreanFRIAAssessor`, register it in the YAML config, write
  the corresponding query against the audit log. The other four
  generators don't move.
- New §35 raw-input retention → add a retention policy to the
  audit log's `log_data_access` table specifically for inputs
  that feed `KoreanFRIAAssessor` runs. The other four generators
  don't move.
- Annex IV Section 5 demographic disclosure → add a query in
  `AnnexIVMapper` that pulls demographic distribution from the
  training-data snapshot table. `KoreanFRIAAssessor` doesn't
  move; `FRIAEvaluator` doesn't move.

The substrate (Eps 3–5) absorbs each change in one place. The
regulatory module above it changes in one place. Two PRs total,
neither touching the model or the inference path.

## What "module" means here

It's worth being concrete about what a generator module looks
like, because the word "module" gets thrown around loosely.

Each generator module has four parts:

**A scope declaration.** Which sections of the audit log it
queries (which `log_*` tables, which time windows, which slice
filters). This is YAML config, not code. It can be changed
without redeployment, and the change itself is logged.

**An aggregation specification.** What the queries compute. For
`KoreanFRIAAssessor`, this is seven scalar dimensions plus their
evidence pointers. For `AnnexIVMapper`, it's twelve evidence
bundles, each composed of pointers to specific audit log rows
or config snapshot files. The specification is versioned and
hash-stamped at runtime, so the assessment record carries the
exact specification version that produced it.

**A serialisation format.** The output shape — required by the
regulation. JSON for §35, structured PDF + JSON for Annex IV, an
Excel-compatible CSV for the FSC quarterly disclosure. The
formats differ because the receiving authorities differ. Putting
all five into one common shape would force the most demanding
format on all of them.

**A retention and access policy.** Five-year WORM for §35, no
fixed retention but immutable for Article 9 (so we keep
ten-year by default), three-year audit-controlled access for
PIA outputs, public posting for FSC disclosure. Each policy is
a configuration on the storage layer, not a hand-coded behaviour.

These four parts are what *makes the module a module*. None of
them depends on the others' implementation. Replace one without
touching the rest.

## The substrate doesn't move

The argument for this whole approach is that *the substrate
underneath the modules is regulation-agnostic*. Walking down the
stack:

**The audit log (Ep 3).** Seven tables, HMAC chain, multi-agent
consensus arbitration. None of these tables knows what
"compliance" means. They record what happened. The same log
serves §35, Article 9, Annex IV, PIPA, and the FSC disclosure
because those regulations are queries over the recorded events,
not separate event streams.

**The XAI explanation column (Ep 4).** Gate weights, CEH
attribution, Mahalanobis OOD flag. None of these knows about a
specific transparency obligation. They produce structured
per-prediction reasoning data because that's what the
architecture does, and the regulatory generators consume that
data because it happens to be the right substrate for their
queries.

**The retrieval layer (Ep 5).** RAG over LanceDB. Doesn't know
about regulations either. Provides the same vector-similarity
and time-travel queries to the human oversight queue, the
fairness monitor, the counterfactual evaluator, and the
quarterly aggregation generators.

This is the payoff of the previous five episodes. Each component
was built without baking in any specific regulation's
assumptions. When a new regulation arrives, it lands as a new
module above the same substrate, not as a re-architecture.

## Three regulations we're betting on

Three changes we expect within the next eighteen months, and how
the architecture will absorb them:

**Korean AI Basic Act 시행령.** The act took effect 2026-01-22
with a one-year-plus calibration period. The enforcement decree
will detail the §31 transparency obligations, the §34 high-impact
provider obligations, and the §35 impact assessment specifics.
Our position: each of these will translate into either a new
dimension in `KoreanFRIAAssessor`, a new field in the public
disclosure (`PublicDisclosureGenerator`), or both. New audit log
fields, if needed, get added without breaking existing queries
because the audit log is append-schema (Lance versioning, Ep 5).

**EU AI Act phase-in.** The AI Act is being phased in across
2025–2027. Each phase brings clarifications and, in some cases,
amendments. The AI Office has been particularly active on
General Purpose AI obligations and Annex IV technical-
documentation requirements. Our position: `FRIAEvaluator` and
`AnnexIVMapper` will absorb amendments as new query patterns and
new evidence-pointer mappings, both YAML-config changes.

**A US federal AI framework.** The Trump administration's March
2026 National Policy Framework laid out seven pillars and a
preemption strategy, but no comprehensive federal AI Act has
passed as of writing. If one passes, we expect it to specify
disclosure and risk-management requirements for high-impact AI
in financial services that will overlap substantially with the
EU and Korean requirements but with different jurisdictional
boundaries. Our position: a sixth generator,
`USComplianceGenerator`, gets registered. The substrate doesn't
move.

The point isn't that we predicted the specific changes
correctly. The point is that the architecture absorbs *any* of
these without re-architecting, because the substrate (Eps 3–5)
is built to be regulation-agnostic and the regulatory layer
(this episode) is built to be modular.

## What modularity isn't

A few things that this approach doesn't claim to do:

**It doesn't reduce the *content* work.** Adding a new dimension
to `KoreanFRIAAssessor` still requires understanding what the
dimension means, what evidence is appropriate, and how to compute
it. The architecture saves the *plumbing* effort, not the *legal
interpretation* effort. The compliance team and the FRM team are
still indispensable.

**It doesn't make compliance posture independent of model
quality.** If the model itself produces unfair predictions or
unstable explanations, no amount of modular reporting fixes that.
Eps 4 (XAI) and Ep 6 of the original Paper 2 series (fairness on
production stream) cover that side. Modular adaptability covers
the *reporting* layer; the *substantive* layer is a different
discipline.

**It doesn't shield from major architectural shifts.** If
regulators decide that all financial AI must use a specific
algorithmic framework (unlikely but not impossible), or that
explainability must come from a specific certified explainer
(equally unlikely), the modular substrate doesn't help. We
explicitly bet on inherent XAI in Ep 4; that bet has a downside.

**It doesn't predict every regulatory direction.** Privacy law
in particular is moving in directions that may require structural
redesign of how features are processed (homomorphic encryption,
federated learning, on-device inference for protected attributes).
The current substrate doesn't natively support those. We will
have to extend the substrate when those directions become
binding, not just add modules above it.

## Six episodes, one idea

The MRM Thread series was, at root, an attempt to articulate one
idea: *MRM that lives in the architecture is cheaper, more
defensible, and longer-lived than MRM that lives in periodic
documents.*

Ep 1 made the case for putting MRM into the architecture in the
first place — when the model is an LLM agent pipeline, the
periodic-validation model breaks down. Ep 2 showed what that
looks like at the promotion gate — every promotion decision is
already an audit entry. Ep 3 went into the audit log layer
itself: seven tables, an HMAC chain, multi-agent consensus
because *who watches the watchers* is a real question. Ep 4
went a layer down to the explanation column — inherent XAI is
what makes the audit log capture *reasoning*, not just events,
and FD-TVS rides on top as the operational scoring philosophy.
Ep 5 went a layer up to the retrieval interface — RAG over
LanceDB makes the audit log into a queryable knowledge base,
which is what the live oversight, fairness monitoring, and
quarterly aggregation workflows all need.

This episode (Ep 6) closes the loop: because the substrate is
regulation-agnostic and the regulatory layer is modular, when
regulations change — and they will — the work is scoped to a
PR per generator and a config change, not a re-architecture. The
five upfront episodes pay off only if this final claim holds.

## A note on what comes next

There are at least two natural extensions of this work that we
haven't covered yet:

- *Counterfactual explainability* (CCP, mentioned briefly in
  Paper 2) — Pearl Rung 3 reasoning on the causal teacher's
  amplified DAG. The architecture supports it, but the regulatory
  status of counterfactual explanations is still being formed.
- *Cross-jurisdictional reconciliation reporting* — when the same
  prediction needs to be reported to both Korean and EU
  authorities under different framings, the aggregation pattern
  above the five generators becomes its own subject.

Both of these are likely future MRM Thread episodes once the
underlying work matures. For now, the six-episode series is the
core argument: the architecture is the MRM, and the MRM is the
architecture.

Source: Paper 2 [Zenodo DOI](https://doi.org/10.5281/zenodo.19622052)
§5–§6 cover the modular generator design and the substrate
guarantees. Implementation lives in
[`core/compliance/`](https://github.com/bluethestyle/aws_ple_for_financial)
in the open-source repository.
