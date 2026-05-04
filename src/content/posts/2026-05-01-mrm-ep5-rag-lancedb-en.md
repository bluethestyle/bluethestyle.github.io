---
title: "[MRM Thread] Ep 5 — RAG + LanceDB: Why Audit Infrastructure Is a Retrieval Problem"
date: 2026-05-01 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, rag, lancedb, audit, retrieval, financial-ai]
lang: en
excerpt: "We started with the idea that the audit log was a write-only archive. The first time a risk officer needed similar-case context inside a five-minute decision window, that idea broke. RAG over LanceDB is what came out of refusing to maintain two copies of the same source of truth — and what unlocked human oversight, fairness monitoring, and quarterly aggregation as queries on the same store."
series: mrm-thread
part: 5
alt_lang: /2026/05/01/mrm-ep5-rag-lancedb-ko/
next_title: "Ep 6 — Modular Adaptability: When Regulations Evolve, Architecture Doesn't"
next_desc: "Korean AI Basic Act enforcement decree, EU AI Act amendments, future US framework — they will all arrive. The five regulatory generators are five modules, not five documents. Why architectural modularity is the long-term bet."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 5 of "The MRM Thread". Ep 4 left us with a structured
per-prediction explanation log — gate weights, CEH attribution,
Mahalanobis OOD flags, all produced as byproducts of the
forward pass. The next question wasn't whether the log existed.
It was whether the log was usable. The first design we tried
treated the audit log as a write-only archive. That design
broke the first time it met a real oversight workflow, and the
second design — RAG over LanceDB — is the architecture that
came out of that break.*

## A scenario that broke the original design

A risk officer is on call at 22:30. The HumanReviewQueue
surfaces a tier-2 alert: a recommendation just escalated
because the Causal Guardrail (the Mahalanobis OOD flag from
Ep 4) fired on a specific customer's prediction. The officer
has under five minutes to decide — override the model, hold for
overnight review, or let it proceed.

What the officer needs in those five minutes:

- The full per-prediction record (input features, gate weights,
  CEH attribution, OOD distance score)
- *Similar past predictions* — same customer, similar feature
  pattern, what did the model say then?
- *Recent OOD-flagged predictions in the same product category*
  — is this an isolated event or a drift signal?
- The current model version's training-data snapshot reference,
  to sanity-check whether the input is genuinely outside what
  the model should have seen

Three of those four are *retrieval queries*, not single-record
lookups. They have no primary key. They have a *similarity
condition* and a *time window*.

The first time this scenario actually played out, we were
running the audit log as a flat append-only Parquet table on
S3. The retrieval queries took minutes — full scan, no vector
index, no time-window pruning. By the time the officer had any
context, the five-minute decision window was gone, and the
default fallback (hold for overnight review) had kicked in.
Operationally that was fine. But it meant the oversight team
could never make decisions in real time, which made the
"oversight as API, not ticket queue" position from the
internal design memo (and what later became the Ep 5 framing in
Paper 2) operational fiction.

That was the first design break. The second came two weeks
later when the fairness monitoring team asked for the same
thing at a different cadence.

## Why audit logs aren't write-only — three workloads

Once we started accepting that audit logs need to be queried in
near-real-time, three workloads became visible. Each had a
different reason for needing the same retrieval substrate.

The first was the live oversight case above. Risk officers,
fairness committee members, and the on-call ops team all need
to look at *similar past situations* when deciding what to do
about a current alert. Without a queryable log, they build
their own working store on the side — usually a spreadsheet, or
in the worse case a separate Postgres instance with copy-paste
from screenshots. We saw both. Both diverge from the audit log
within a week.

The second was fairness monitoring. Disparate Impact, Statistical
Parity Difference, and Equal Opportunity Difference across the
five protected attributes (gender, age band, region, income tier,
disability) aren't computed once a quarter on a curated
validation sample. In a system that takes fairness seriously,
they're computed continuously on the actual production
prediction stream, sliced by protected attribute. That stream
lives in the audit log. If the log isn't queryable in
near-real-time, the fairness monitor either runs against a
staler proxy (which means the fairness numbers reported to the
board are an approximation) or maintains its own duplicate
stream (back to the divergence problem).

The third was regulatory artifact generation. Ep 4 already
made the case for the five generators (`KoreanFRIAAssessor`,
`FRIAEvaluator`, `AnnexIVMapper`, `PIAEvaluator`,
`PublicDisclosureGenerator`) running as aggregation queries
over the audit log. If the queries take hours, the quarterly
artifact pipeline becomes a slow batch nobody can debug. If
they take seconds, the same query can be re-run on demand
when the generator's logic changes.

Three different teams, three different cadences, all needing
the same thing: *the audit log as a queryable knowledge base,
not a write-only archive*.

## Why we didn't pick the obvious second store

The path most teams take, when they realize the audit log is
too slow for live queries, is to build a second store. Audit
log = the immutable Parquet archive. Operational queries = a
separate Postgres or Elasticsearch instance. They sync via
change-data capture or batch jobs.

We almost went there. The Postgres-with-pgvector option was on
the whiteboard for a week. What stopped it was thinking
through what the divergence looked like.

The two stores would, at any given moment, contain slightly
different views of the same events. Sync delays of a few
seconds during peak load. Schema drift when the audit-log table
got a new column and the operational store lagged behind.
Replication failures that nobody noticed for a week because the
operational store still answered queries — just with stale
data. Retention-policy mismatches when the operational store
purged six-month-old rows that the audit store still had to
retain for SR 11-7. Each one was a source of *audit-vs-ops
disagreement* that an external supervisor could flag. Worse,
when that disagreement was discovered six months later, neither
store could explain why.

The single-store rule we ended up with: *the audit log is the
only source of record, and operational queries run against it
directly*. That moves the burden onto the storage layer — it
has to support both immutable append-only writes (for
compliance) and fast indexed retrieval (for ops). LanceDB is
the choice that made that combined workload feasible.

## Why LanceDB and not the alternatives

The shortlist when we were picking was LanceDB, Chroma, and
Postgres + pgvector. Each had attractive properties. Each was
rejected for a specific reason.

**Postgres + pgvector** was the most operationally familiar —
mature ops, well-understood backup story, large hiring pool.
What it didn't give us was the columnar storage shape that
analytical queries over the audit log needed. The same row-store
that made transactional updates fast was the wrong shape for
"aggregate by protected attribute over the last 90 days." We
would have ended up with a separate analytical store anyway,
which put us back in the two-store problem we were trying to
avoid. We also didn't want to fight Postgres on append-only
versus update semantics for the WORM-style retention the
compliance side needed.

**Chroma** had the easier vector-search developer experience
and was popular in the LLM application stack we were already
running. What we couldn't get past was the lack of
version-aware time travel. Audit reconstruction queries —
*"what did the audit log look like as of 2026-04-15 14:00 UTC"*
— need the storage layer to remember its own past versions. A
vector store that only answers *"what's there right now"*
forces you to track snapshots externally, which is another
divergence source waiting to happen.

**LanceDB** got picked for four properties that matter
specifically for combined audit + retrieval workloads.

First, columnar storage with vector-native indexing. LanceDB
stores in Apache Arrow columnar format, which is the right
shape for analytical queries (filter by time range, aggregate
by a single column, scan one column without touching the rest).
On top of that, IVF-PQ vector indexes are native, so the
explanation column from Ep 4 — the per-prediction gate weights
plus CEH attribution — can be queried by similarity, not just
by exact match.

Second, version-aware time travel. Every write produces a new
versioned snapshot. The audit log as it existed on
2026-04-15 14:00 UTC is queryable directly, not just *the
audit log right now*. This is what makes the supervisor's
fifteen-month reconstruction query work — the model registry
from that point in time, joined to the inference log from that
point in time, joined to the attribution log from that point
in time, all consistent.

Third, append-only by design. New writes are new versions; old
versions don't get overwritten. Combined with HMAC chaining
(Ep 3), this gives the immutability property the audit log
needs without fighting the storage layer.

Fourth, cheap to embed. No separate cluster, no operator. Runs
in-process or as a sidecar. For a small team where the
ops/audit infrastructure is one of many things one person owns
overnight, this matters more than it should. We measured the
on-call burden of Postgres-with-pgvector at "non-zero per week"
and LanceDB at "essentially zero." That difference compounded
over a year of operation.

The cost was real. LanceDB is younger than Postgres or Elastic;
the operator-facing tooling is less mature; the community
ecosystem is smaller. We took that cost as the price of running
a single store instead of two.

## RAG over the explanation column — what it actually looks like

The retrieval-augmented part of the design is what makes the
oversight workflow viable in the five-minute window the
original design failed.

When the risk officer looks at the OOD-flagged prediction, the
RAG layer takes the explanation vector for *that specific
prediction* — gate weights as a 7-element vector, plus CEH
attribution as a sparse feature-contribution vector, plus the
OOD distance score as a scalar — and runs a vector similarity
search over the past 90 days of recommendations. The result is
a ranked list of *predictions whose reasoning was structurally
similar*, regardless of customer ID, regardless of product,
scoped to the configurable time window.

This isn't a SQL `WHERE customer_id = X` query. It's a "find me
predictions that *thought the same way*" query. For the officer
deciding whether the current OOD flag is an anomaly or a
pattern, that's exactly the question that needs answering.

The same retrieval pattern serves three other workloads, all
of which we initially thought would need their own
infrastructure and turned out not to:

- **Drift detection.** Are this week's predictions drawing on
  the same expert mix as last quarter's? A drift in the
  distribution of gate-weight vectors over time is a leading
  indicator of feature distribution drift, weeks before
  traditional drift metrics catch up. The detection runs as a
  scheduled vector-distribution comparison on LanceDB.

- **Counterfactual review.** Given a specific predicted
  recommendation, what would similar customers with one
  feature perturbed have been recommended? RAG retrieves the
  comparison set, and the counterfactual layer (covered briefly
  in Paper 2) runs against it. We expected to need a separate
  counterfactual cache. We didn't.

- **Explanation consistency check.** Does the model give
  similar explanations for similar inputs? RAG over the
  explanation column lets us verify that the model's
  *reasoning is stable*, not just that its outputs are. This is
  a stricter property than output-consistency monitoring, and
  it's a property regulators have been asking about more often.

## How fairness monitoring stopped needing its own pipeline

The continuous fairness monitor runs as a streaming query over
the audit log, scoped by protected-attribute slice. *Disparate
Impact* across the five protected attributes is computed on a
rolling 24-hour window of actual production predictions, not a
curated validation sample.

Two design choices fell out of having LanceDB underneath, both
of which we initially scoped as separate sub-projects.

**Counterfactual Champion-Challenger.** The fairness layer
doesn't only ask *"is the current production model fair?"* It
asks *"would the challenger model, on the same production
stream, have been more or less fair?"* The challenger model's
predictions are computed offline on the same retrieved input
batches and compared. RAG retrieves the matched batches; the
champion-challenger comparison runs on top. The Parquet archive
of the comparison results is itself a LanceDB table, queryable
the same way as the prediction log. We had budgeted two weeks
for the comparison-cache infrastructure. The actual work was a
SQL query.

**Real-time threshold breach.** When a protected-attribute
slice crosses the fairness threshold in the rolling window,
the HumanReviewQueue immediately gets a tier-3 alert. This is
the same queue the OOD-flagged predictions go to, with a
different severity tier. Both flow from the same retrieval
substrate. We had originally planned a separate fairness alert
pipeline. Once the retrieval substrate was in place, that
pipeline became a configuration on the existing queue.

The point isn't that we built a sophisticated fairness monitor.
The point is that *fairness monitoring became cheap because the
retrieval substrate was already in place for other reasons*.
This is what good infrastructure choices look like — they keep
producing dividends in places you didn't initially design for.

## Human oversight as API, not ticket queue — what made it possible

EU AI Act Article 14's human oversight requirement, in our
implementation, is a set of API endpoints rather than a ticket
queue. The architectural reason this works is that the audit
log is queryable in near-real-time. The three flows:

**Kill switch.** A single API call, requiring two-factor
operator authentication, halts new predictions across the
entire system. The kill-switch event itself is a `log_operation`
write to the audit log, which means the *reason for the halt*
is recoverable later. The retrieval substrate matters here for
the post-mortem: when a kill switch fires, the on-call needs
to know *what was happening in the prediction stream just
before the halt*, and that's a retrieval query.

**Tier 2 / Tier 3 escalation.** The HumanReviewQueue has
tiered severity. Tier 2 = OOD flag fired, fairness slice neared
threshold, or the consensus arbiter dissented (Ep 3). Tier 3 =
fairness threshold breached, multiple correlated OOD flags,
or kill switch was tripped. Each tier has its own retrieval
template — the officer sees pre-fetched similar-case context
appropriate to the tier. The templates themselves were the part
that took the longest to design. The retrieval engine was the
easy part once LanceDB was in place.

**`auto_promote=false` as default posture.** Model promotions
require explicit operator approval (covered in Ep 2's
Champion-Challenger). The reason this lives here, in the
oversight section, is that the operator's approval decision is
itself a query against the audit log: was the challenger
model's fairness monitor green? Were there OOD flags in the
test window? What did the consensus arbiter say? RAG fetches
the relevant context bundle as part of the approval interface,
so the operator's decision is informed without being onerous.

All three flows write back to the audit log. Oversight isn't
something that happens *outside* the system. It's recorded
*inside* the system, queryable like everything else. The
architectural shift from "ticket queue" to "API" only worked
because the underlying log was queryable.

## What this approach doesn't solve

A few honest limits of RAG-over-LanceDB.

**Embedding drift.** The vector representation of the
explanation column depends on the embedding choices we made.
If the model architecture changes substantially (new expert
added, gate dimension changed), existing explanation vectors
are no longer comparable to new ones. We handle this with
versioned embedding stores, but the fact remains: long-term
retrieval across major architectural revisions is harder than
within-version retrieval. Ep 6 (modularity) is partly about
not making this harder than it has to be.

**Cold-start workloads.** New deployments without historical
prediction data don't get the similarity-search benefit. The
oversight workflow degrades to single-record lookups for the
first weeks. The workaround is to seed the explanation store
with synthetic-benchmark predictions during the staging phase
so retrieval has something to work against from day one. We
learned this the hard way during the first soft launch.

**Query expertise.** Vector similarity queries are easier to
get *subtly* wrong than SQL queries. *"Find similar
predictions"* with the wrong distance metric or the wrong time
window can return a pile of false neighbors. We mitigate by
exposing only a small set of pre-defined retrieval templates to
the operator interface, not a free-text query box. Ad-hoc
queries go through a notebook interface restricted to the data
science team. The operator-facing surface is intentionally
small.

## Next

Ep 6 closes the series with the longer-term thesis: regulations
will change. Korean AI Basic Act enforcement decree details
will land. EU AI Act will get amended. The US framework, when
it arrives, will require its own generator. The five regulatory
artifacts (Paper 2's `KoreanFRIAAssessor`, `FRIAEvaluator`,
`AnnexIVMapper`, `PIAEvaluator`, `PublicDisclosureGenerator`)
are *modules*, not documents — and the architecture is set up
so a new regulation becomes a new module above the same audit
log substrate, not a re-architecture of the system.

Source: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
on the operational architecture; LanceDB choice and retrieval
templates live in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial)
under `core/audit/` and `core/retrieval/`.
