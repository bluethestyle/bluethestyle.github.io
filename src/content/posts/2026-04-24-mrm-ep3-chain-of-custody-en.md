---
title: "[MRM Thread] Ep 3 — Chain of Custody for an Agent Pipeline"
date: 2026-04-24 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, audit, sr-11-7, regulation, financial-ai]
lang: en
excerpt: "Fifteen months after a recommendation, a customer disputes it. Seven audit tables and one HMAC hash chain determine the shape of the answer — making EU AI Act 13-14 and KFCPA §17 reconstructable code paths, not checklists."
series: mrm-thread
part: 3
alt_lang: /2026/04/24/mrm-ep3-chain-of-custody-ko/
next_title: "Ep 4 — FRIA: How the Korean AI Basic Act §35 Lives in Code"
next_desc: "Korea AI Basic Act §35 seven-dimension impact assessment, five-year retention, and why it's kept separate from the EU AI Act Article 9 FRIAEvaluator."
next_status: draft
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 3 of "The MRM Thread". Ep 2 covered the record of promotion
decisions. This episode covers everything around them — each
prediction, each agent verdict, each guardrail block — and how the
record survives verifiable fifteen months later.*

## A question from June 2027

Scenario. A customer who received a recommendation in April 2026
files a dispute with the financial-consumer ombudsman in June 2027.
"Fifteen months ago you recommended this product to me. I believe
the recommendation discriminated against me on income tier. What
was the basis?" Under Korean Financial Consumer Protection Act §17
suitability principle, the institution must answer.

Conventional answers take a few shapes. "We cannot reproduce the
feature scores from that moment exactly, but the model generally…"
— qualitative narrative. Or "the customer belonged to segment A
with preference for product family B" — a SHAP-like post-hoc
approximation. Neither satisfies regulatory reconstruction; both
lack point-in-time specificity.

Our pipeline's answer takes a different shape. The moment that
recommendation went out fifteen months ago, *what was written to
which audit table* is known with precision. This episode is about
the structure that produces that precision.

## Seven tables, divided by concern

`core/monitoring/audit_logger.py` exposes seven `log_*` methods,
each writing to a separate audit table:

- `log_operation` — system state transitions (retrain
  start/finish, promotion, serving-manifest swap)
- `log_model_inference` — every prediction, timestamp, subject,
  model version
- `log_data_access` — who read which data
- `log_dimension_change` — feature schema evolution (349D → 403D,
  for instance)
- `log_attribution` — per-prediction feature contribution ranking
- `log_guardrail` — which outputs the Safety Gate blocked or
  modified
- `log_model_promotion` — champion-challenger decisions (Ep 2)

The seven-way split is separation of concerns, not table sprawl.
Mixing all events into one table entangles retention, access, and
audit policies. `log_data_access` is governed by PIPA and the
Credit Information Act with one retention schedule; `log_model_promotion`
is governed by SR 11-7 Pillar 2 with a different access boundary.
A unified table collapses to the strictest policy across the set,
which inflates storage cost without improving any individual
audit.

## HMAC hash chain — why this isn't "just a log"

An entry is not simply written. When `log_operation()` is called,
three things happen at once:

1. **Entry serialization** — timestamp, event type, and payload
   normalized to canonical JSON.
2. **HMAC signing** — the previous entry's hash combined with the
   current entry's content, signed with `AUDIT_HMAC_SECRET_KEY`.
3. **Chain linking** — the current entry's `prev_hash` field
   carries the previous entry's hash.

One property this buys — *modifying entry n requires re-signing
n+1, n+2, …, up to the latest entry*. Impossible without the HMAC
key. Tamper-evidence becomes structural, not a matter of trust.

On the AWS side, `AUDIT_HMAC_SECRET_KEY` lives in SSM Parameter
Store (SecureString) and is only reachable by Lambda roles at
runtime. Key exposure is catastrophic — it breaks the entire chain
— so this is one of MRM oversight's *individual verification
points*. The answer to "do you have audit logs" is not enough; it
has to be "do you have audit logs whose HMAC key lives under a
separate rotation policy".

## AuditAgent — verifies, does not trust

Having a log doesn't make the log correct. Who confirms the hash
chain has not been broken this week? That's the role of the
AuditAgent (Haiku-based).

The AuditAgent runs nightly. For every entry written in the last
24 hours it checks:

1. That the HMAC signature recomputes to the stored signature.
2. That the `prev_hash` field matches the previous entry's actual
   hash.
3. That the chain is continuous — no gaps between entry n and
   entry n+1.

Any failure opens an incident; the OpsAgent pages the on-call
engineer. This is the difference between *reading the log* and
*verifying the log*. Reading it trusts whatever is on disk. The
AuditAgent doesn't read; it recomputes.

## Turning regulations into code paths

EU AI Act Article 13 (transparency) — "high-risk AI systems shall
be designed so users can interpret the output." The conventional
answer is "we publish a transparency report." In this structure:

- `log_attribution` entries → feature-contribution ranking for a
  given prediction, directly queryable.
- `log_guardrail` entries → how the explanation passed the Safety
  Gate criteria.
- Article 13's answer is *a query*, not a document.

Article 14 (human oversight) — "users shall have means for
effective oversight." The conventional answer is "monthly MRM
committee review." In this structure:

- `HumanReviewQueue` API → real-time queueing of tier 2/3
  suspicious cases.
- Kill-switch API → one call to disable a specific task.
- `log_operation` entry records that the kill switch fired,
  permanently.
- Article 14's answer is *an API spec*.

Article 15 (accuracy, robustness, cybersecurity) — join
`log_model_inference` with `log_dimension_change` and you get
"model v143 served which schema with what accuracy on what date"
on demand. The conventional quarterly report becomes a SQL query.

KFCPA §17 (financial-consumer dispute handling) — the scenario
question from the opening. The answer comes from joining
`log_model_inference`, `log_attribution`, and `log_data_access`.
"At 2026-04-15 14:37, what feature combination caused model v143
to recommend what product for this customer ID" — reconstructable
fifteen months later, with the AuditAgent's continuous hash-chain
verification underneath certifying *that the reconstruction has
not been tampered with*.

## Structure, not checklist

The standard failure mode of MRM compliance is *documentation
dependency*. "We maintain a compliance document for Article 13."
But regulators ask not for documentation; they ask for
*reconstruction*. A document is a document; whether it holds for a
specific customer case is a separate verification question.

Code paths answer this differently. The `log_*` methods are
invoked as a *byproduct* of normal operation, so Article 13
compliance is equivalent to "the system is running". Compliance
breaks only when the system itself breaks, and that break
generates its own audit entry. It's a closed loop.

This is the reason the audit infrastructure is wired into the hot
path rather than built as a "separate" subsystem. A separate
subsystem drops to second-priority maintenance and a few months
later its cron job is silently failing. A wired-in one can't stop
without the main system also stopping, so it recovers by the same
recovery path.

## Still the MRM committee's job

Same pattern as Ep 2. Some things the architecture buys, some it
doesn't.

What it buys:
- *Reconstruction* of any prediction, decision, or configuration
  change at any prior moment.
- *Structural certainty* of answers to regulator queries 15 months
  later.
- *Bounded tamper-detection latency* via the AuditAgent's daily
  verification.

Still human work:
- Whether the HMAC key rotation policy is adequate.
- Whether the seven-way table split remains correct (new
  regulation might require a new table or a schema change).
- Meta-audit — is the AuditAgent's nightly job actually running.
- Whether `AUDIT_HMAC_SECRET_KEY` management satisfies SR 11-7
  Pillar 4 (governance) standards.

The misreading "audit is automated so the MRM committee is
redundant" was already treated in Ep 1, and the same argument
recurs here from a different angle. What automation replaced is
*entry writing*, not *institutional design*. The latter is still
done by someone with an FRM credential.

## Next

Ep 4 covers how Korean AI Basic Act §35 FRIA (Fundamental Rights
Impact Assessment) lives in the codebase. Seven-dimension impact
assessment, five-year retention, and why it is kept as a *separate
class* from the EU AI Act Article 9 FRIAEvaluator — the temptation
to unify two frameworks with different legal bases, and the risk
of doing so.

Source material:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §4
"Audit infrastructure". Implementation lives in
`core/monitoring/audit_logger.py` and the `core/compliance/`
submodules.
