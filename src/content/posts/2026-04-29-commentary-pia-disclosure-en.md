---
title: "[Commentary] Privacy Impact Assessment and AI Disclosure as Byproducts of the Audit Log"
date: 2026-04-29 12:00:00 +0900
categories: [Commentary]
tags: [commentary, pia, pipa, disclosure, mrm, financial-ai]
lang: en
excerpt: "PIPA-grounded Privacy Impact Assessment and the Financial Services Commission's quarterly public-disclosure report, both delivered as automatic byproducts of the audit log rather than documents written from scratch. How they quietly ride on top of Ep 3's `log_*` tables, and what remains on the MRM committee's desk."
alt_lang: /2026/04/29/commentary-pia-disclosure-ko/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*MRM Thread Ep 4 used FRIA as the canonical example of a
regulatory artefact that "lives on top of code". This Commentary
covers two places the same pattern repeats — Privacy Impact
Assessment (PIA) and the Financial Services Commission's
quarterly public-disclosure report. The shared property is one
sentence: **neither starts as a document to be written. Both
start as aggregations over the audit log.***

## The second and third auto-generated artefacts

If FRIA (Ep 4) is the *§35 · Art. 9 · Art. 11* triad of impact-
assessment artefacts, two other regulatory artefacts live next to
them, separate obligations on their own.

- **PIA (Privacy Impact Assessment)** — Personal Information
  Protection Act (PIPA) Article 33 governs it. For *public-sector*
  institutions §33(1) makes it a legal requirement; for
  *private-sector* financial institutions it sits at a
  recommended / internal-policy level. Where the data flow
  involves EU data subjects, GDPR Article 35 (DPIA) creates a
  separate, binding obligation. We conservatively stand up a
  DPIA-equivalent structure in advance, so the same artefact
  answers requests from the public-sector, private-sector, or EU
  path.
- **Public Disclosure Report** — an internally institutionalized
  quarterly disclosure aligned with the Financial Services
  Commission (FSC, 금융위원회)'s AI-in-financial-services
  guideline direction. It periodically aggregates AI system
  transparency, model performance, fairness indicators, incident
  summary, and customer-impact assessment. At present this is
  closer to *anticipatory conformance to supervisor expectations*
  than to a hard legal mandate, and the five-section structure
  is kept modular so it can extend as regulations crystallize.

Both were traditionally *quarterly documents, hand-written by the
owner*. In our implementation both live under `core/monitoring/`
as auto-generators (`pia_evaluator.py`,
`public_disclosure_generator.py`). When a quarter closes, no one
opens a Word file — an aggregation query runs against the audit
log.

## PIA Evaluator — the six-domain structure

The impact assessment required by PIPA and GDPR Art. 35 has to
cover the *full personal-data lifecycle*. `PIAEvaluator`
structures that into six domains:

- **Data collection** — scope and legal basis for collection
- **Data processing** — purpose limitation and processing
  safeguards
- **Data storage** — retention policy and encryption state
- **Data sharing** — third-party transfers and access controls
- **Data minimization** — necessity and proportionality analysis
- **Cross-border transfer** — multi-region data flows (AWS-
  specific)

For each domain the evaluator reads from audit logs and config
files and produces a score and a risk level. The *data collection*
domain, for example, scans Ep 3's `log_data_access` across the
previous quarter to compute "did the collection scope stay within
pre-disclosed use purposes?" and "is there a gap between the
accessor's declared permissions and their actual access?". The
*data minimization* domain cross-references the feature-
engineering pipeline's input column list against the trained
model's attribution distribution, auto-detecting "columns
collected but effectively unused".

Output is a structured `PIAReport` dataclass, with partial
results per domain, optionally persisted to
`ComplianceAuditStore`. The owner reviews that report rather than
drafting a document, and adds *mitigation* items — the parts that
still need human judgment.

Two things differ from hand-written drafting. First, because the
aggregation runs from the audit log, *omission is structurally
impossible* — the scenario of "the owner forgot to include this
quarter's data-sharing activity" doesn't exist. Second, the
assessment is *reproducible* — if a supervisor asks a year later
to "rerun last quarter's PIA", the same code returns the same
answer.

## Public Disclosure Generator — the quarterly 5-section report

A separate artefact from PIA. In line with the FSC's
AI-in-financial-services guideline direction, we institutionalize
the following five-section aggregation and publish it periodically:

1. **AI system transparency** — summary of the recommender's
   purpose, data used, decision flow
2. **Model performance** — per-task AUC, F1-macro, MAE, NDCG
3. **Fairness indicators** — DI, SPD, EOD per protected attribute
   (Ep 6's 15 metrics + intersectional)
4. **Incident summary** — kill-switch firings, fairness breaches,
   system failures during the quarter
5. **Customer-impact assessment** — HumanReviewQueue tier 2/3
   handling counts, opt-out exercise counts, rights-related
   inquiries

`PublicDisclosureGenerator` runs an aggregation query for each of
these five sections on the quarterly cycle. Inputs are Ep 3's
`log_operation`, `log_model_inference`, `log_attribution`,
`log_guardrail`, plus Ep 6's fairness archive. Output lands in
two formats — **JSON** (machine-readable, for the supervisor's
automated intake path) and **Markdown** (human-readable, for the
disclosure web page and PDF conversion). Versioned storage in S3.

When a supervisor or external stakeholder requests the end-of-
quarter disclosure, a human verifies the JSON generated correctly
and presses submit. The step of typing the numbers is
structurally removed.

## Why "byproduct of the audit log"

The shared pattern of these two generators is one sentence —
**regulatory artefacts should be natural byproducts of running
the system**. Ep 3 stated the same principle first: "wire audit
infrastructure separately and it drops in maintenance priority;
wire it into the hot path and it can only stop when the system
itself stops, forcing recovery."

PIA and disclosure sit on the same principle. Starting from a
document — 1) the writing cadence is tied to the owner's
calendar, 2) there's an *unverifiable summarization step* between
audit log and report, 3) when a supervisor tries to re-audit an
old version, the source data has already faded. Starting from a
generator — 1) the report follows the audit log's accumulation
rate, 2) tracing a report value back reaches individual audit
entries (Ep 3's reconstruction guarantee), 3) a re-calculation
request a year later returns the *same answer*.

Auto-generation doesn't replace *judgment*. The six domain scores
in PIA are computed mechanically, but "is this score acceptable?"
and "which mitigation should we take?" still belong to the CPO or
DPO. The numbers in the disclosure report are filled
automatically, but "what narrative do we attach to this quarter's
incident summary?" is still written by a person. What the
machine removes is *repetition* and *inconsistency risk*, not
*judgment*.

## What remains on the MRM committee

Like FRIA, PIA and disclosure don't reduce the committee's work —
they sharpen it. The review target gets clearer.

- Is the **rule set** that produces the PIA's per-domain scores
  appropriate? For example, does "unused-feature detection via
  attribution distribution" in *data minimization* hold for every
  model type?
- Does the disclosure report's five-section structure still match
  the supervisor's current expectations? New items (e.g., carbon
  footprint, supply-chain risk) require the report structure to
  extend.
- Is the *human edit delta* between the auto-generated report and
  the final version sent to the supervisor kept reasonably small?
  A large delta means the generator isn't reflecting reality
  well.

The committee moves one step back from reviewing the *contents*
of the reports to reviewing the *generation logic*. The same
pattern surfaces from Ep 2, Ep 3, Ep 6 repeats with PIA and
disclosure.

## In summary

Including FRIA, the regulatory artefacts our system produces as
*automatic byproducts* are now the following set:

- `KoreanFRIAAssessor` — AI Basic Act §35 impact assessment
  (7 dimensions)
- `FRIAEvaluator` — EU AI Act Art. 9 risk record (5 dimensions)
- `AnnexIVMapper` — EU AI Act Art. 11 + Annex IV technical
  documentation evidence (12 sections)
- `PIAEvaluator` — PIPA + GDPR Art. 35 privacy impact assessment
  (6 domains)
- `PublicDisclosureGenerator` — FSC quarterly disclosure
  (5 sections)

Five generators running on the same audit log, each changing
*what the MRM committee reviews* in the same way. The essential
shift is that the regulatory-response *starting point* moved
from a blank document to the audit log itself.

---

Source:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §6
"Regulatory mapping"; implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial)
(`core/monitoring/pia_evaluator.py`,
`core/monitoring/public_disclosure_generator.py`).
