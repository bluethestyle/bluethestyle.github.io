---
title: "[MRM Thread] Ep 5 — RAG + LanceDB: Why Audit Infrastructure Is a Retrieval Problem"
date: 2026-05-01 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, rag, lancedb, audit, retrieval, financial-ai]
lang: en
excerpt: "Audit logs are not write-only. They are queryable knowledge bases. Why we built ops/audit retrieval as RAG over LanceDB — columnar, version-aware, time-travel-capable — and what that unlocks for human oversight, fairness monitoring on the production stream, and quarterly aggregation."
series: mrm-thread
part: 5
alt_lang: /2026/05/01/mrm-ep5-rag-lancedb-ko/
next_title: "Ep 6 — Modular Adaptability: When Regulations Evolve, Architecture Doesn't"
next_desc: "Korean AI Basic Act enforcement decree, EU AI Act amendments, future US framework — they will all arrive. The five regulatory generators are five modules, not five documents. Why architectural modularity is the long-term bet."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 5 of "The MRM Thread". Ep 4 established that the audit log
captures structured per-prediction explanation — gate weights, CEH
attribution, Mahalanobis OOD flags — because the architecture
produces them as byproducts. Ep 5 is about the next layer up: how
that log gets queried. The mistake we wanted to avoid was treating
the audit log as a write-only sink that someone reads in batch
once a quarter. It has to be a live, queryable knowledge base, or
none of the things we built it for actually work.*

## A scenario from the human oversight queue

A risk officer is on call at 22:30. The HumanReviewQueue surfaces
a tier-2 alert: a recommendation just escalated because the
Causal Guardrail (the Mahalanobis OOD flag from Ep 4) fired on a
specific customer's prediction. The officer needs to decide in
under five minutes whether to override the model, hold for
overnight review, or let it proceed.

What the officer needs in that moment:

- The full per-prediction record (input features, gate weights,
  CEH attribution, OOD distance score)
- *Similar past predictions* — same customer, similar feature
  pattern, what did the model say then?
- *Recent OOD-flagged predictions in the same product category* —
  is this an isolated event or a drift signal?
- The current model version's training-data snapshot reference,
  so they can sanity-check whether the input is in fact outside
  what the model should have seen

Three of these four are *retrieval queries*, not single-record
lookups. They don't have a primary key. They have a *similarity
condition* and a *time window*.

If the audit log is a flat append-only Parquet table, these
queries take minutes (full scan) and the officer's five-minute
decision window collapses. If the audit log is exposed through
RAG over a vector-aware columnar store, they take seconds.

## Why audit logs are not write-only

The default mental model for an audit log in financial systems is
*the regulator's archive*. Write events as they happen, hash-chain
the entries (Ep 3), retain for the required period (Ep 4 covered
the five-year case), respond when asked. Read access is rare and
batch.

This model breaks for three reasons in an AI MRM context.

**Live oversight needs live retrieval.** The example above — a
risk officer needing similar-case context inside a five-minute
window — is not a quarterly batch query. It is an interactive
decision-support workload, with strict latency requirements.
Treating the audit log as cold storage means oversight teams
build their own *separate* working store, with all the divergence
problems that come with two sources of truth.

**Fairness monitoring runs on the production stream.** Disparate
Impact, Statistical Parity Difference, Equal Opportunity
Difference — these aren't computed on a validation sample once a
quarter. In a production AI system that cares about fairness,
they are computed *continuously* on the actual prediction stream,
across protected attribute slices. That stream lives in the audit
log. If the log isn't queryable in near-real-time, the fairness
monitor either runs against a staler proxy or gets its own
duplicate stream.

**Regulatory artifact generation is a query.** Ep 4 already
argued this for the five generators (FRIA / EU FRIA / Annex IV /
PIA / public disclosure). Each generator runs as an aggregation
query over the audit log. If that query takes hours, the
quarterly artifact pipeline becomes a batch job that nobody can
debug. If it takes seconds, it can be re-run on demand when the
generator's logic changes.

The audit log isn't the *final* destination for the data. It is
the *source of truth* that everything else queries against.

## The two-store anti-pattern

The path many teams take, when they realise the audit log is too
slow for live queries, is to build a second store. Audit log =
the immutable Parquet archive. Operational queries = a separate
Postgres or Elasticsearch instance. They sync via change-data
capture or batch jobs.

The problem is divergence. The two stores will, at any given
moment, contain slightly different views of the same events.
Sync delays, schema drift, replication failures, retention
mismatches — every one of these is a source of *audit-vs-ops
disagreement* that an external supervisor can flag. Worse, when
that disagreement is found six months later, neither store can
explain why.

The single-store rule we follow: *the audit log is the only
source of record, and operational queries run against it
directly*. That moves the burden onto the storage layer — it has
to support both immutable append-only writes (for compliance)
and fast indexed retrieval (for ops). LanceDB is the choice that
made that work.

## Why LanceDB

A few non-obvious properties matter for this combined workload.

**Columnar storage with vector-native indexing.** LanceDB stores
data in Apache Arrow columnar format, which is the right shape
for analytical queries over the audit log (filter by
time range, aggregate by protected attribute, scan a single
column without touching the rest). On top of that, it natively
supports IVF-PQ vector indexes, so the *explanation column* —
the per-prediction gate weights and CEH attribution — can be
queried by similarity, not just by exact match.

**Version-aware "time travel".** Every write produces a new
versioned snapshot. You can query *the audit log as it was on
2026-04-15 14:00:00 UTC*, not just *the audit log right now*.
This is what lets the supervisor's fifteen-month reconstruction
query work — the model registry from that point in time, joined
to the inference log from that point in time, joined to the
attribution log from that point in time, all consistent.

**Append-only by design.** New writes are new versions; old
versions are not overwritten. Combined with HMAC chaining (Ep 3),
this gives the immutability property the audit log needs without
fighting the storage layer.

**Cheap to embed.** No separate cluster, no operator. Runs
in-process or as a sidecar. For a small team where the
ops/audit infrastructure is one of many things one person owns
overnight, this matters more than it looks.

The cost is real. LanceDB is younger than Postgres or Elastic;
the operator-facing tooling is less mature; the community
ecosystem is smaller. We took that cost as the price of having
a single store instead of two.

## RAG over the explanation column

The retrieval-augmented part is what makes the live oversight
workflow viable.

When the risk officer looks at the OOD-flagged prediction, the
RAG layer takes the explanation vector for *that specific
prediction* (gate weights as a 7-element vector, plus CEH
attribution as a sparse feature-contribution vector, plus the
OOD distance score as a scalar) and runs a vector similarity
search over the past 90 days of recommendations. The result is a
ranked list of *predictions whose reasoning was structurally
similar*, regardless of customer ID, regardless of product, scoped
to a configurable time window.

This isn't a SQL `WHERE customer_id = X` query. It is a "find me
predictions that *thought the same way*". For the officer
deciding whether the current OOD flag is an anomaly or a pattern,
this is exactly the question that needs answering.

The same retrieval pattern serves three other workloads:

- **Drift detection.** Are this week's predictions drawing on the
  same expert mix as last quarter's? A drift in the distribution
  of gate-weight vectors over time is a leading indicator of
  feature distribution drift, weeks before traditional drift
  metrics catch up.

- **Counterfactual review.** Given a specific predicted
  recommendation, what would similar customers with one feature
  perturbed have been recommended? RAG retrieves the comparison
  set, and the counterfactual layer (covered briefly in Paper 2)
  runs against it.

- **Explanation consistency check.** Does the model give similar
  explanations for similar inputs? RAG over the explanation
  column lets us verify that the model's *reasoning is stable*,
  not just that its outputs are. This is a different and stricter
  property than output-consistency monitoring.

## The fairness path

The continuous fairness monitor runs as a streaming query over
the audit log, scoped by protected attribute slice. *Disparate
Impact* across the five protected attributes (gender, age band,
region, income tier, disability) is computed on a rolling
24-hour window of actual production predictions, not a curated
validation sample.

Two design choices that fall out of having LanceDB underneath:

**Counterfactual Champion-Challenger.** The fairness layer
doesn't only ask *"is the current production model fair?"* It
asks *"would the challenger model, on the same production
stream, have been more or less fair?"* The challenger model's
predictions are computed offline on the same retrieved input
batches and compared. RAG retrieves the matched batches; the
champion-challenger comparison runs above. The Parquet archive
of the comparison results is itself a LanceDB table, queryable
the same way as the prediction log.

**Real-time threshold breach.** When a protected-attribute slice
crosses the fairness threshold in the rolling window, the
HumanReviewQueue immediately gets a tier-3 alert. This is the
same queue the OOD-flagged predictions go to, with a different
severity tier. Both flow from the same retrieval substrate.

The point is not that we built a fancy fairness monitor. The
point is that *fairness monitoring became cheap because the
retrieval substrate was already in place for other reasons*.
This is what good infrastructure choices look like — they keep
producing dividends in places you didn't initially design for.

## The human oversight path

EU AI Act Article 14's human oversight requirement, in our
implementation, is a set of API endpoints rather than a ticket
queue. Three flows:

**Kill switch.** A single API call, requiring two-factor
operator authentication, that halts new predictions across the
entire system. The kill-switch event itself is a `log_operation`
write to the audit log, which means the *reason for the halt*
is recoverable later.

**Tier 2 / Tier 3 escalation.** The HumanReviewQueue has tiered
severity. Tier 2 = OOD flag fired, fairness slice neared
threshold, or the consensus arbiter dissented. Tier 3 = fairness
threshold breached, multiple OOD flags in correlated predictions,
or kill switch was tripped. Each tier has its own retrieval
template — the officer sees pre-fetched similar-case context
appropriate to the tier.

**`auto_promote=false` as default posture.** Model promotions
require explicit operator approval (covered in Ep 2's
Champion-Challenger). The reason this lives here, in the
oversight section, is that *the operator's approval decision is
itself a query against the audit log*: was the challenger model's
fairness monitor green? Were there OOD flags in the test window?
What did the consensus arbiter say? RAG fetches the relevant
context bundle as part of the approval interface, so the
operator's decision is informed without being onerous.

All three flows write back to the audit log. Oversight is not
something that happens *outside* the system; it is recorded
*inside* the system, queryable like everything else.

## What this doesn't solve

A few honest limits of the RAG-over-LanceDB approach:

**Embedding drift.** The vector representation of the
explanation column depends on the embedding choices we made.
If the model architecture changes substantially (new expert added,
gate dimension changed), the existing explanation vectors are no
longer comparable to new ones. We handle this with versioned
embedding stores, but the fact remains: long-term retrieval
across major architectural revisions is harder than the within-
version case. Ep 6 (modularity) is partly about not making this
harder than it has to be.

**Cold-start workloads.** New deployments without historical
prediction data don't get the similarity-search benefit. The
oversight workflow degrades to single-record lookups for the
first weeks. This is a known limitation; the workaround is to
seed the explanation store with synthetic-benchmark predictions
during the staging phase so retrieval has something to work
against from day one.

**Query expertise.** Vector similarity queries are easier to get
*subtly* wrong than SQL queries. *"Find similar predictions"* with
the wrong distance metric or the wrong time window can return a
pile of false neighbours. We mitigate by exposing only a small
set of pre-defined retrieval templates to the operator interface,
not a free-text query box. Ad-hoc queries go through a notebook
interface restricted to the data science team.

## Next

Ep 6 closes the series with the longer-term thesis: regulations
will change. Korean AI Basic Act enforcement decree details will
land. EU AI Act will get amended. The US framework, when it
arrives, will require its own generator. The five regulatory
artifacts (Paper 2's KoreanFRIAAssessor, FRIAEvaluator,
AnnexIVMapper, PIAEvaluator, PublicDisclosureGenerator) are
*modules*, not documents — and the architecture is set up so a
new regulation becomes a new module above the same audit log
substrate, not a re-architecture of the system.

Source: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
on the operational architecture; LanceDB choice and retrieval
templates live in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial)
under `core/audit/` and `core/retrieval/`.
