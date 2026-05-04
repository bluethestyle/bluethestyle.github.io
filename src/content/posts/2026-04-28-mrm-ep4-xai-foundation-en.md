---
title: "[MRM Thread] Ep 4 — When Explanation Is Architecture: Inherent XAI and FD-TVS Scoring"
date: 2026-04-28 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, xai, explainability, ple, fd-tvs, financial-ai]
lang: en
excerpt: "Post-hoc XAI (SHAP, LIME) is unstable at production scale and decoupled from the model. We chose architectural XAI instead — gate weights, CEH attribution, Mahalanobis OOD — so explanation is a byproduct of the prediction path itself, not a sidecar. FD-TVS is the operational scoring philosophy that rides on top."
series: mrm-thread
part: 4
alt_lang: /2026/04/28/mrm-ep4-xai-foundation-ko/
next_title: "Ep 5 — RAG + LanceDB: Why Audit Infrastructure Is a Retrieval Problem"
next_desc: "Audit logs aren't write-only. They are queryable knowledge bases. Why we built ops/audit retrieval as RAG over LanceDB, and what that unlocks for human oversight, fairness monitoring, and quarterly aggregation."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19621884
source_label: "Paper 1 + Paper 3 (Zenodo DOIs)"
---

*Part 4 of "The MRM Thread". Ep 3 covered the audit log layer —
seven tables, an HMAC chain, and consensus arbitration on top.
But who decides what gets logged? Where do the per-prediction
explanation, attribution, and reliability flags come from? They
come from XAI choices made in the architecture, not from a
post-hoc layer bolted on at serving time. Ep 4 is the case for
why explanation has to be architectural, and what FD-TVS adds
on top.*

## The post-hoc XAI problem

The default reflex when a regulator asks "why did the model
predict this?" is to reach for SHAP, LIME, or Integrated
Gradients. Add a post-hoc attribution module to the inference
path, surface the top-k contributing features per prediction,
and call it explainability. This is what most production
financial AI systems do, and it has three structural problems
that get worse at scale.

**Instability.** Salih et al. (2023) and several follow-up
studies have documented that SHAP and LIME attributions are
sensitive to background-distribution choice, sample size, and
even random-seed variation. The same prediction can produce
materially different "top contributing features" depending on
how you call the explainer. For a one-off research artifact this
is tolerable. For a regulator who is trying to reconstruct a
specific decision fifteen months after the fact, instability is
not a research-paper caveat — it is a compliance liability.

**Computational cost at serving time.** SHAP-class methods are
expensive. On a CPU Lambda inference path serving thousands of
recommendations per minute, per-prediction SHAP is a budget
killer. Most production deployments end up either sampling
(lose per-prediction guarantees) or pre-computing attributions
on a small subset (lose universality). Either way the
"explainability for every prediction" promise quietly breaks.

**Decoupling from the model.** SHAP and LIME treat the model as
a black box. The attribution is whatever a *separate* approximator
infers about the model's behaviour around the input. When the
model and the explainer disagree — and they sometimes do — the
explainer's answer is the one shown to regulators, customers,
and oversight committees. The model's actual reasoning, if there
is such a thing for an opaque MLP, is never visible.

These three problems compound. For a regulated AI system that
has to answer *the same question* about a specific prediction
years later, post-hoc explainability is a moving floor.

## Architectural XAI as the alternative

The choice we made early was to push the explanation work into
the architecture itself, so explanation isn't computed *after*
the prediction — it *is* part of the prediction.

The Heterogeneous Expert PLE (Paper 1) is built around seven
architecturally distinct shared experts: DeepFM, Temporal
Ensemble, Hyperbolic GCN, PersLay, Causal, LightGCN, and
Optimal Transport. Each expert is a *named mathematical
operation* — not a generic MLP with random initialisation.
Customised Gate Control (CGC) routes each task's prediction
through the expert basket with explicit per-expert weights.

What this buys is that the gate weights themselves *are* the
explanation. When the system predicts "customer X has a 0.78
probability of cross-sell on product P," the gate weights
attached to that prediction tell us — at the moment of the
prediction, in the same forward pass — that the prediction was
driven 35% by Temporal (spending trend), 28% by HGCN (product
hierarchy fit), 15% by Causal (intervention reasoning), and so
on. There is no separate explainer call. The explanation is
recorded because the routing decision is recorded.

This is what *inherent XAI* means in practice. Explanation is
not a UI layer. It is an architectural decision.

## Three layers of per-prediction explanation

The single forward pass produces three layers of explanation
that are logged automatically alongside the prediction:

**Gate weights** — the per-expert contribution. Because each
expert encodes a named inductive bias, the gate weight maps
directly to a business-readable narrative. *"Temporal 35%"*
means *"recent spending pattern"*, not *"hidden unit 47
activated"*. This is what we use as the customer-facing reason
in the recommendation generation layer (Paper 2).

**CEH attribution** — Causal Explainability Head, the per-feature
contribution within the Causal expert. When the Causal expert is
the dominant route for a given prediction, CEH exposes which
features inside the causal DAG drove the conclusion. This is the
fine-grained attribution layer underneath the gate weights — for
when "Causal 38%" isn't enough and the supervisor wants to know
*which specific causal pathway*.

**Mahalanobis OOD on the Causal latent** — a per-prediction
reliability flag. We compute Mahalanobis distance on the Causal
expert's latent space against an in-distribution reference, and
emit a binary trust flag per prediction. On a synthetic OOD probe
this achieves 100% TPR at 5% FPR. The interpretation: when this
flag fires, the prediction is in a region of feature space the
model wasn't trained for, and the customer-facing reason should
be downgraded or withheld.

All three are computed at prediction time, all three land in the
audit log. None of them require a separate post-hoc explainer
call. The prediction path produces them as byproducts.

## Why this is the regulatory foundation

This is the part that connects upward to the regulatory layer.

The five regulatory artifacts described in Paper 2 — the Korean
AI Basic Act §35 impact assessment, the EU AI Act Article 9
risk record, the Annex IV technical-documentation evidence
mapping, the PIPA + GDPR Art. 35 privacy impact assessment, the
FSC AI guideline quarterly disclosure — all consume the same
structured per-prediction log. They are aggregation queries, not
authored documents.

But that pattern only works because the per-prediction log
contains *structured explanation data*, not just inputs and
outputs. If the prediction record were `(input_vector, output_score,
timestamp)`, no aggregation query could answer "why did the
model decide X for customer Y?" — the answer wouldn't be in the
data. The five generators are queries because the log captures
*reasoning*, and the log captures reasoning because the
architecture produces reasoning as output.

Inherent XAI is the foundation. Audit log is the second floor.
The five regulatory generators are the roof. Replace the
foundation with post-hoc SHAP and the second and third floors
collapse — the per-prediction log loses its structured
explanation column, which means the aggregation queries lose
their substrate, which means the regulatory artifacts revert to
hand-written documents.

EU AI Act Article 13 (transparency obligations) and Korean AI
Basic Act §31 (transparency) are not satisfied by *having an
explainer*. They are satisfied by *being able to produce a
specific, stable, reconstructible explanation for any prediction
on demand*. Inherent XAI is the only architecture we know of
that lets you keep that promise across model retraining cycles.

## FD-TVS — scoring philosophy on top of XAI

The on-prem precedent for this system used per-product weights
in the scoring layer. Each financial product had its own static
weight, configured manually. When a new product launched, the
config was updated by hand. The scoring layer was a flat lookup.

This was fragile in three ways. New product launches required
manual reconfiguration. Customer segment differences (a 25-year-old
first-time depositor vs. a 60-year-old high-net-worth client) were
not reflected in the scoring weights — segment behaviour was
hoped to be captured by the model's prediction, with the scoring
layer agnostic. And behavioural shifts (a sudden spike in a
specific feature signalling, say, life-event-triggered demand)
had no mechanism to influence the score.

FD-TVS — Financial DNA Targeted Value Scoring — is the
re-architecture of that scoring layer. Three philosophical shifts:

**Task-level instead of product-level.** Weights are attached to
the *task* (cross-sell intent, churn risk, suitability fit, etc.),
not the *product*. New products inherit the existing task
structure and don't require reconfiguration. The XAI gate weights
described above feed directly into this — task selection is
informed by the per-expert routing of the prediction.

**Segment-aware (`segment_task_weights`).** Each customer segment
gets its own multiplier on the task weights, clipped to the range
1.0–1.5. The clipping is deliberate. Allowing weights to drop
below 1.0 would let segment heuristics suppress task signals,
breaking the model's role as the primary signal source. Allowing
weights above 1.5 would let segment overrides dominate the model.
The 1.0–1.5 range says: *segment matters as a multiplier, not as
an override*.

**Behaviour-aware (`dynamic_weight_rules`).** Specific feature
thresholds can boost specific task weights at scoring time. A
spike in a feature correlated with churn raises the churn-task
weight; a sequence of small deposits in a previously inactive
account raises the deposit-product task weight. This is reactive
scoring — *behaviour itself is the signal that triggers weight
adjustment*, not a periodic re-tuning.

All three live in `pipeline.yaml`. Operations can adjust the
segment table or add a behaviour rule without code change. This
matters: scoring policy adjustments can ship in hours, not weeks,
and every adjustment is config-version-stamped in the audit log
for the same fifteen-month reconstruction window the rest of the
MRM stack obeys.

The connection to XAI is direct. The XAI layer tells the system
*why* a prediction was made (gate weights × CEH × OOD). FD-TVS
tells the system *how much that prediction should weigh in the
final score, given who this customer is and how they are
currently behaving*. Both layers log their inputs. The customer-
facing reason becomes "we recommended product P because your
recent spending pattern (Temporal 35%) and product hierarchy
fit (HGCN 28%), weighted up by your segment's historical
preference for this category" — a single string that is
defensible because every component of it is independently
recoverable from the audit log.

## What the XAI foundation enables

Looking forward to Eps 5 and 6:

**Ep 5 (RAG + LanceDB)** describes how the per-prediction
explanation log is queried at scale. Vector retrieval over the
explanation column is what makes "find me predictions where
Temporal dominated and OOD fired in the last quarter" answerable
in seconds. The explanation column has to exist before retrieval
can do anything useful.

**Ep 6 (Modular adaptability)** describes why this architecture
holds up when regulations change. New regulation = new
aggregation query over the same explanation log. The XAI
foundation is regulation-agnostic; the regulatory layer is
swappable.

## What this doesn't replace

A few things the architectural XAI choice does not buy:

**Score validation by humans.** "Is a Temporal contribution of
35% the *right* explanation for this customer's recommendation?"
is still a judgement call for the recommendation review committee
or the customer-facing relationship manager. What's automated is
that the contribution is recorded, stable, and reconstructible.
What's not automated is whether the explanation makes business
sense.

**Edge-case interpretability.** When all seven experts contribute
roughly equally (gate entropy near maximum), the gate-weight
explanation is "everything contributed a little." This is honest
but not satisfying. We treat high-entropy predictions as a
distinct interpretive category — *low-confidence, high-entropy*
predictions are flagged for human review under the oversight
layer, regardless of whether the OOD flag fires.

**Architecture lock-in.** The whole argument depends on the
heterogeneous expert basket being kept stable. If a future
iteration replaces the seven experts with a single transformer,
the gate-weight explanation disappears and we are back to
post-hoc XAI. This is a long-term architectural commitment, not
a short-term implementation choice. The five regulatory
generators (Ep 6) are designed assuming this commitment holds.

## Next

Ep 5 goes one floor up — into the retrieval layer that turns the
per-prediction explanation log into a queryable system. Why we
chose RAG + LanceDB for ops/audit infrastructure, what
columnar version-aware retrieval gives us for fairness monitoring,
human oversight escalation, and quarterly aggregation, and why
the audit log isn't write-only.

Source: [Paper 1 (Zenodo)](https://doi.org/10.5281/zenodo.19621884)
on the heterogeneous expert architecture and gate-weight
explainability, [Paper 3 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
on CEH and Causal Guardrail (Mahalanobis OOD); FD-TVS scoring
config lives in
[`configs/pipeline.yaml`](https://github.com/bluethestyle/aws_ple_for_financial)
under `scoring.segment_task_weights` and
`scoring.dynamic_weight_rules`.
