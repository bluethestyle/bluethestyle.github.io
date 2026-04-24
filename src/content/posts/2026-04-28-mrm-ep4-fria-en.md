---
title: "[MRM Thread] Ep 4 — FRIA: How the Korean AI Basic Act §35 Lives in Code"
date: 2026-04-28 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, fria, regulation, financial-ai, ai-basic-act]
lang: en
excerpt: "Seven-dimension impact assessment and five-year retention under Korea AI Basic Act §35. Why the KoreanFRIAAssessor is kept as a separate class from the EU AI Act Article 9 FRIAEvaluator even when the outputs are reported jointly."
series: mrm-thread
part: 4
alt_lang: /2026/04/28/mrm-ep4-fria-ko/
next_title: "Ep 5 — Human Oversight as an API, Not a Ticket Queue"
next_desc: "EU AI Act Article 14, kill-switch API, HumanReviewQueue tier 2/3, and why auto_promote=false is forced as a production posture."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 4 of "The MRM Thread". Ep 3 covered the audit log's
structure. This episode goes one layer up — a *regulatory artifact*
that lives on top of that log: the FRIA (Fundamental Rights Impact
Assessment).*

## What AI Basic Act §35 can ask for

Korea AI Basic Act took effect 2026-01-22. §35 defines "high-impact
AI" (고영향 AI) as systems that may materially affect life, bodily
safety, or fundamental rights. Classification is determined by
usage context and the scope/severity/frequency of risk, and is
self-assessed by the operator. Given that financial product
recommendation directly affects customer financial decisions and
suitability rights, we assessed the system as likely to fall under
§35 high-impact AI and built the FRIA (Fundamental Rights Impact
Assessment) architecture preemptively on that premise. Consider an
illustrative scenario of what a supervisor might ask under this
classification:

"Has your recommendation AI performed the seven-dimension impact
assessment required by §35? Submit the evaluation result and
mitigation measures in a form that can be retained for five years."

Ep 1 argued that compliance should sit in the architecture, not in
post-hoc documents. FRIA is the sharpest test of that argument —
because FRIA isn't one document. It has to be re-evaluated on
every model change, and the trail has to remain auditable for
five years.

## Why two classes

The codebase has two FRIA-related classes:

- `KoreanFRIAAssessor` — AI Basic Act §35, seven dimensions,
  five-year retention.
- `FRIAEvaluator` — EU AI Act
  Article 9, five dimensions, continuous monitoring.

On the surface this looks redundant. Both are "impact assessment";
both evaluate similar-sounding dimensions. There was a temptation
early on to merge them.

The reason not to merge — *different legal bases*. AI Basic Act §35
and EU AI Act Article 9 are independent obligations under their
respective jurisdictions, and satisfying one does not satisfy the
other. The dimensional composition differs subtly too:

- §35 seven dimensions: life and safety, fundamental rights,
  discrimination, transparency, human oversight, personal
  information, accountability.
- Article 9 five dimensions: risk identification, risk estimation,
  risk evaluation, risk management measures, residual risk.

§35 enumerates *rights that may be violated*; Article 9 enumerates
*risk management procedures*. Different vantage points.

A combined external report is possible — the common components are
substantial. But *internal storage is separate*. If one class
served both laws, one law's amendment would break the other's
compliance posture. Two separate classes, each owning its own
pipeline, is the stable structure.

## And a third axis — EU AI Act Article 11 (Annex IV)

That's the "two-class story". But EU compliance doesn't end at
Article 9. High-risk AI-system providers must also prepare a
**technical documentation package** under **Article 11 + Annex IV**
*before* placing the system on the market. That's a *different*
obligation from Article 9's risk assessment — one is ongoing
evidence that "the risk management process is operating", the
other is a *pre-market technical file* submitted to the
conformity-assessment authority.

Annex IV enumerates 12 sections — general system description,
design details, monitoring and control techniques, training-data
provenance and attributes, test and validation results, human
oversight measures, post-market monitoring plans, and so on.
Each section demands *verifiable evidence* that can be produced
on request. Not a prose "we comply", but a concrete artefact —
a code path, a config key, a document path.

In our implementation this obligation is owned by `AnnexIVMapper`,
a class independent from the two FRIA classes. Its job is to map
each of the 12 sections to the **actual evidence source** in the
repository — Ep 3's seven audit tables, Ep 2's promotion-gate
config, `KoreanFRIAAssessor`'s seven-dimension output,
`FRIAEvaluator`'s five-dimension risk record, training-data
snapshot hashes, and so on, all registered as candidate evidence
for the relevant section. When a conformity assessor asks for
"Annex IV Section 5 (training data attributes)", the mapper
pulls the evidence bundle immediately.

So on the compliance side we actually manage **three independent
artefacts**, not two.

- `KoreanFRIAAssessor` — Korea AI Basic Act §35, seven-dimension
  impact assessment, five-year WORM retention.
- `FRIAEvaluator` — EU AI Act Art. 9, five-dimension risk-
  management process record.
- `AnnexIVMapper` — EU AI Act Art. 11 + Annex IV, twelve-section
  technical-documentation evidence mapping.

The reason we don't merge them is the same as before. Different
legal bases, different submission formats and timings, and a
change in one jurisdiction must not ripple into another. Each
class owns its own obligation; any combined report sits *above*
them as a separate aggregator layer.

## The seven-dimension assessment

`KoreanFRIAAssessor` is invoked automatically on new-model
registration. For each dimension it records (score, evidence_ref,
mitigation):

- **Life and safety** — financial recommendation is low on direct
  threat to life/safety, but cumulative financial harm from
  inappropriate recommendations is medium.
- **Fundamental rights** — discrimination (see next dimension),
  impact on financial access.
- **Discrimination** — Disparate Impact, Statistical Parity
  Difference, Equal Opportunity Difference across five protected
  attributes (gender, age band, region, income tier, disability).
  This score is pulled directly from the fairness monitor covered
  in Ep 6.
- **Transparency** — explanation-generation success rate (from
  expert gate weights), reading-level of customer-facing text.
- **Human oversight** — kill-switch firing history,
  HumanReviewQueue tier 2/3 volume, escalation response time.
- **Personal information** — PIPA §37의2 profiling opt-out
  handling rate, Credit Information Act §36의2 compliance records.
- **Accountability** — traceability of model version, training
  data version, config version — answered by joining Ep 3's audit
  tables.

Each dimension's evidence field carries a pointer to rows in Ep 3's
audit tables — *the assessment sits on top of the audit log*. This
is what "FRIA lives in code" means in practice.

## Five-year retention — WORM plus hash chain

§35 requires *five-year retention* of the assessment. Not just
storage but tamper-evidence. Implementation:

- FRIA results are persisted as a the FRIA results table Parquet
  table in S3.
- The bucket is in Object Lock (WORM) mode.
- Each entry gets signed into Ep 3's HMAC chain once.
- Deletion blocked for five years (bucket policy).

"Deletion blocked" matters. Even when a past assessment looks
embarrassing after a later model iteration, from an audit
perspective it must remain. The regulatory answer isn't that
*every assessment was perfect*; it is that *assessments happened
and their traces are preserved*.

## KoreanFRIAAssessor vs FRIAEvaluator — the actual call flow

When a new model registers:

1. Ep 2's promotion gate runs.
2. If promotion is approved, `KoreanFRIAAssessor` runs its
   seven-dimension evaluation and persists to the Korean results
   table.
3. In parallel, `FRIAEvaluator` runs its five-dimension evaluation
   and persists to its own separate table.
4. `AnnexIVMapper` runs evidence checks against each of the 12
   Annex IV sections — verifying that required artefacts (audit
   logs, config values, FRIA outputs, training-data hashes) exist
   in the repository, are accessible, and are current, and writes
   the results to an Annex IV evidence table. When an Art. 11
   submission is requested, this table becomes the evidence
   pointer set directly.
5. A FRIA-complete event lands in the audit log, and separately
   an Annex IV evidence-check-complete event is written. A
   combined submission, if needed, is assembled above these by a
   separate aggregator — the three stores remain independent.

Failures at any step do not fail the pipeline itself — this is a
best-effort invocation (same principle as Ep 2's
`_audit_promotion`, preventing audit-infra failure from halting
production). Failures are recorded as separate `log_operation`
entries for the AuditAgent to surface overnight.

## The FSS query's answer

"Submit the FRIA results retained for five years." One query.
Filter the the FRIA results table table by the model version at the
moment of interest. The seven-dimension scores come back with
per-dimension evidence pointers (rows in the audit tables) and
mitigation records, as a set.

One year, two years, five years later, the same query returns the
same answer. Even if the responsible staff has rotated or the
model has been replaced many times in between. This is what §35's
*continuous reconstruction property* requires.

## Still human work

What the architecture doesn't buy:

- **The actual dimension scores are set by people.** "Is a
  discrimination-dimension score of 0.72 appropriate?" is answered
  by the fairness committee. What's automated is score computation,
  not score judgment.
- **Mitigation design.** Which measures follow from a given
  evaluation — threshold adjustment, training-data rebalancing,
  stricter human review — is FRM-credential territory.
- **Responding to law changes.** If §35 gets amended, the
  dimensions of `KoreanFRIAAssessor` have to change. The reason
  for having two separate classes — an amendment on one side must
  not touch the other — becomes concrete here.

## Next

Ep 5 goes deep on the "human oversight" dimension — how EU AI Act
Article 14's human oversight requirement is implemented as an API
endpoint rather than a ticket queue. Kill switch, HumanReviewQueue
tier 2/3, and why `auto_promote=false` is enforced as a production
posture.

Source: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
§6 "Regulatory mapping"; implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial).
