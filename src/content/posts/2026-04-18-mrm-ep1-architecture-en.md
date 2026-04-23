---
title: "[MRM Thread] Ep 1 — Why MRM Belongs in the Architecture"
date: 2026-04-18 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, sr-11-7, regulation, financial-ai, audit]
lang: en
excerpt: "The validation-first view of Model Risk Management breaks once the model becomes an LLM agent pipeline — multi-step attack surface, non-traditional failure modes, drift between validation cycles. Push MRM into the architecture itself, not the review calendar."
series: mrm-thread
part: 1
alt_lang: /2026/04/18/mrm-ep1-architecture-ko/
next_title: "Ep 2 — Champion-Challenger as a Gate"
next_desc: "Champion-Challenger as a synchronous gate — the `--force-promote` override pattern and how every promotion decision (approval, rejection, bootstrap, manual override) becomes an HMAC-signed audit entry under SR 11-7."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 1 of "The MRM Thread" — a short parallel series on regulatory
compliance and model risk management for AI recommendation systems,
written from a GARP FRM practitioner perspective.*

## The standard picture

In most financial institutions, Model Risk Management (MRM) looks
like this:

1. A development team builds a model.
2. The model is handed to a second line — an independent MRM or
   validation team.
3. The validation team runs tests: performance, stability,
   fairness, compliance with SR 11-7.
4. If it passes, deployment. If not, back to development.
5. Post-deployment, the validation team reviews monitoring outputs
   on a schedule.

This is the *validation-first* picture. MRM is a gate that sits
*after* the model is built. It is the standard approach in
Basel-style risk management, and it has served credit and market
risk modeling well for twenty years.

What this looks like at a mid-size Korean institution, concretely:
every quarter, the model-risk team receives a packet — performance
charts, stability plots, sample predictions, and a narrative
summary. They review it over two weeks, flag questions, schedule a
committee meeting, and sign off. For a logistic regression trained
on last year's loan book, this cadence is roughly right. The
model hasn't changed since the last packet; the regulatory
environment hasn't moved; customer distributions shift
gradually. Quarterly is fast enough.

It starts to break when the "model" stops being a logistic
regression and starts being an *agent pipeline*.

## What breaks when the model is an LLM agent system

Our deployed system for financial recommendation is not a single
model. It is a pipeline of five agents on AWS Bedrock:

- **Feature Selector** (Sonnet) — chooses which feature
  attributions are worth explaining to the customer
- **Reason Generator** (Sonnet) — rewrites the selected features
  into natural-language, honorific Korean that a bank branch
  employee can read to a customer
- **Safety Gate** (Sonnet) — validates the generated reason
  against regulatory, suitability, hallucination, tone, and
  factuality criteria before it leaves the Lambda handler
- **OpsAgent** (Sonnet) — interprets monitoring and drift reports
- **AuditAgent** (Sonnet) — chain-of-custody verification over
  audit logs

If an MRM team tries to "validate" this as if it were one model,
several things happen at once:

1. **The attack surface is multi-step.** A hallucination can
   originate in the Reason Generator and be missed by a Safety
   Gate that was trained on the wrong distribution of failure
   modes. "Validation" has to decompose into per-agent validation
   *plus* pipeline-level interaction validation.

2. **The failure modes have changed.** AUC drift is still there,
   but the real risks are different — prompt injection,
   instruction-following failure under distribution shift,
   refusal-to-respond on valid queries, confidence-signal
   miscalibration in customer-facing text. Most MRM frameworks
   have no vocabulary for these.

3. **The stability assumption breaks down.** Traditional MRM assumes the
   model is stable between validation runs. An agent pipeline
   calling a hosted LLM inherits model-provider updates whose
   timing is not controlled by the financial institution. "The
   model we validated" and "the model we deployed" drift apart
   *between* validation runs, not just across them.

You can address all of this with a larger, faster, more technical
MRM team — if you have one. Most Korean financial institutions,
including ours, do not.

## The architectural alternative

The alternative is to push MRM obligations into the *architecture*
itself, so that compliance properties are built into the system
rather than checked after the fact. This is not a new idea
in software engineering — it is the same principle as "parse,
don't validate" or "make illegal states unrepresentable" — but it
is relatively new in financial AI.

What this looks like, concretely, in our system:

**Explainability as architecture, not post-hoc.** The seven
expert networks produce gate weights that are directly readable
as business explanations. "35% spending-trend expert + 28%
product-fit expert" is not a SHAP approximation — it is what the
model actually computed. The explanation is a structural output,
not a reconstruction. SR 11-7's "effective challenge" requirement
becomes trivially satisfiable because the contribution
decomposition is already in the computation graph.

**Champion-Challenger as a gate, not a report.** More on this in
Episode 2. The short version: `ModelCompetition.evaluate()` is a
code path that either returns `promotion_approved=True` or
blocks deployment. There is no "MRM committee reviews the report
next month" — the gate is called every time a candidate model is
registered, synchronously, with an HMAC-signed audit entry.

**Audit trail as immutable chain.** Every prediction, every
agent decision, every promotion decision is logged as one entry
in an HMAC hash-chained log. Tamper-evidence is structural — you
cannot modify entry *n* without invalidating every entry from *n*
onward. The AuditAgent's job is not to trust the log; it is to
verify the hash chain.

**Fairness monitoring as a production path.** The fairness
monitor (Disparate Impact, Statistical Parity Difference, Equal
Opportunity Difference on five protected attributes) runs on the
production prediction stream, not on a validation sample. If a
threshold is breached, the prediction is blocked and escalated —
not noted in a weekly report.

**Kill switch as a first-class operator.** The human-in-the-loop
override is an API endpoint, not an organizational process.
Disabling the entire pipeline takes one call; disabling a
specific task takes another. The operator's ability to stop the
system is not dependent on a ticket queue.

## How we arrived at this approach

None of the five properties were in the original plan. The plan
was simple — build the model, write the validation report
quarterly, submit it for review, deploy. As the pipeline took
shape, failure cases surfaced one by one and changed the plan
each time. The first Reason Generator we tested produced a
plausible-but-factually-wrong recommendation, and we asked "how
would the reviewers have caught this if we'd shipped it?" — the
honest answer was "they wouldn't, because they see outputs, not
the process that generated them". That's where the
"explainability as structural output" property came from.

Each of the other four properties followed a similar route — a
concrete failure case or regulatory question that the
validation-first approach couldn't answer, and a structural
change that made the question *not require* an out-of-band
process to answer. This episode frames the outcome, but the
subsequent episodes walk through the specific scenes that forced
each property into existence.

## Why the architectural approach is not an argument against MRM

There is a lazy reading of this argument: "if the architecture
takes care of it, we don't need MRM." That is wrong and regulators
will catch it.

What the architectural approach *does*:

- Converts compliance from a post-hoc report into a structural
  property
- Lets a small team (three people, in our case) build and
  operate a compliant system without a dedicated MRM department
- Surfaces violations in real time instead of at review cadence
- Produces audit artifacts as a byproduct of normal operation

What MRM oversight *still has to do*:

- Verify that the architectural properties are *actually*
  holding — the gate weights really are the explanation, the
  hash chain really is verified, the kill switch really works
- Review the architecture itself — is the Safety Gate
  calibrated for the failure cases we actually care about?
- Review incidents and investigate root causes, not healthy operation
- Take responsibility for model inventory, governance reporting,
  and regulatory correspondence
- Second-guess the development team on design decisions that
  affect risk

The architectural approach does not remove MRM oversight. It
*changes what MRM oversight is auditing*. Instead of validating
model outputs after training, MRM is continuously checking
whether the architectural conditions still hold. The work shifts
from "did the model pass the tests" to "are the compliance
conditions still met."

## Why this matters more in 2026

Three regulatory developments converge this year:

- **Korean AI Basic Act** (effective 2026-01-22). Classifies
  financial product recommendation as high-risk AI. Requires
  impact assessment, transparency obligations, and human
  oversight. The domestic FSS guidelines are expected to issue
  concrete implementation rules in Q2 2026.

- **EU AI Act Title III, Chapter 2**. Article 13 (transparency),
  Article 14 (human oversight), Article 15 (accuracy &
  robustness), Article 9 (risk management). Applicable to
  Korean-origin AI if deployed to EU customers.

- **Korean Financial Consumer Protection Act (KFCPA)** Article 17
  (suitability principle) — unchanged but increasingly tested
  against AI recommendation systems in customer dispute cases.

The common thread: regulators expect compliance to be
*verifiable from the system itself*, not merely *documented
after the fact*. "We run validation quarterly" is not sufficient when the
model is making real-time decisions at customer-facing points.

The architectural MRM approach matches this expectation
naturally. The validation-first approach matches it only if the
validation team is large enough to run continuously and fast
enough to catch interaction failures across an agent pipeline —
which almost no Korean financial institution can afford.

## What this series will cover

This episode established the frame. Episode 2 will go into
Champion-Challenger as a synchronous gate, including the
`--force-promote` override pattern and how every promotion
decision (approved, rejected, bootstrapped, manually overridden)
becomes an HMAC-signed audit entry under SR 11-7.

Episode 3 will cover what "chain of custody" looks like for an
agent pipeline: the seven audit tables, the HMAC hash chain, the
EU AI Act Article 13-14 and KFCPA §17 mappings as code paths
rather than checklists.

The source material for all three is
[Paper 2 on Zenodo](https://doi.org/10.5281/zenodo.19622052);
this series adapts and contextualizes rather than replaces the
paper. If you want the complete regulatory mapping tables, they
are in §6 of the paper.
