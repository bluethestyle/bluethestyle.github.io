---
title: "[MRM Thread] Ep 4 — When Explanation Is Architecture: Inherent XAI and FD-TVS Scoring"
date: 2026-04-28 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, xai, explainability, ple, fd-tvs, financial-ai]
lang: en
excerpt: "Post-hoc XAI (SHAP, LIME) wobbled on us when we ran the cost and stability numbers. We chose architectural XAI instead — the gate weights of seven heterogeneous PLE experts as the explanation, with CEH attribution and Mahalanobis OOD as the second and third layers. FD-TVS is the operational scoring philosophy that grew out of fixing the on-prem per-product weights model."
series: mrm-thread
part: 4
alt_lang: /2026/04/28/mrm-ep4-xai-foundation-ko/
next_title: "Ep 5 — RAG + LanceDB: Why Audit Infrastructure Is a Retrieval Problem"
next_desc: "Audit logs aren't write-only. They are queryable knowledge bases. Why we built ops/audit retrieval as RAG over LanceDB, and what that unlocks for human oversight, fairness monitoring, and quarterly aggregation."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19621884
source_label: "Paper 1 (Zenodo DOI); Paper 3 forthcoming"
---

*Part 4 of "The MRM Thread". Eps 1–3 covered the audit log layer
— seven tables, an HMAC chain, multi-agent consensus on top.
That layer captures *what* happened. Ep 4 is one floor down: who
decides what gets logged in the explanation column? Where do
gate weights, per-feature attribution, and the per-prediction
reliability flag come from? They don't come from a sidecar
explainer bolted onto the inference path. They come from
architectural choices made during model design — and this
episode is about how we got to those choices, not just what
they look like. Interpretability has always been an MRM concern;
on statistical models it took the form of conceptual soundness.
On deep learning, that requirement reappears as XAI, and the
architectural choice in this episode is what makes it tractable.*

## Why we walked away from SHAP

When we moved off the on-prem ALS-based recommender and started
on the PLE prototype, the first instinct was to bolt SHAP onto
the inference path. The on-prem system had no explainability
layer at all — it returned a ranked product list and that was
the entire output. For the new model, slotting in a SHAP-style
post-hoc attribution module looked like the cheapest path to
"now we have explanations." Two things shifted us off that path.

The first was the stability issue documented by Salih et al.
(2023) and several follow-up studies: SHAP and LIME
attributions are sensitive to background-distribution choice,
sample size, and even random-seed variation. Same prediction,
different "top contributing features" depending on how you
called the explainer. For a one-off research artifact this is
tolerable. For a regulated financial AI system that has to
return a *consistent* answer to "why did the model say X for
this customer" fifteen months after the prediction, it was
disqualifying. A supervisor coming back to that prediction in
2027 wouldn't accept "well, our explainer happened to pick
different features that quarter."

The second was the cost projection. We profiled SHAP-class
methods on the actual Lambda inference path and the per-call
overhead would have multiplied serving cost several-fold.
The standard fallbacks — sample only some predictions, or
pre-compute attributions on a subset — both broke the property
we needed. Either we no longer had per-prediction
explainability, or we no longer had universal coverage across
the customer base. Either way the compliance story got fuzzy
in exactly the place it needed to be sharp.

There was a third issue underneath those two, more conceptual.
SHAP and LIME treat the model as a black box. The attribution
is what a *separate* approximator infers about model behavior
near the input. When the model and the explainer give different
answers — and they do — the explainer's answer is the one the
regulator, the customer, and the oversight committee see. The
model's actual reasoning, if such a thing even exists for a
generic MLP, stays hidden. We didn't want to defend a system
where the explanation was provably decoupled from what the
model was doing.

Stack those three issues and the conclusion was hard to avoid:
post-hoc XAI is a moving floor when what you need is a
reconstructible record.

## The other path — gate weights as the explanation

So we went looking for a different path. The question we put to
ourselves: could we move the explanation work *into* the
architecture, so explanation isn't computed *after* the
prediction — it *is* part of the prediction?

The Heterogeneous Expert PLE that ended up in Paper 1 grew out
of that question. Seven shared experts with structurally
distinct designs — DeepFM, Temporal Ensemble, Hyperbolic GCN,
PersLay, Causal, LightGCN, Optimal Transport — sit behind a
CGC (Customised Gate Control) layer that routes each task's
prediction through the basket with explicit per-expert weights.
The non-obvious property: because each expert is a *named
mathematical operation* rather than a generic MLP with random
init, the gate weights themselves carry meaning. *"Temporal
35%"* maps to *"recent spending pattern,"* not to *"hidden unit
47 fired."* The routing decision *is* the business-readable
explanation, and the routing decision is recorded as a
byproduct of the forward pass.

That's what *inherent XAI* came to mean for us in practice.
Explanation isn't a UI layer or a sidecar. It's an architectural
decision, made at the point where you choose what kind of
experts to compose.

## Three layers that arrived at different times

The current per-prediction explanation has three layers. None
of them arrived together — each got added because the previous
layer wasn't enough on its own.

**Gate weights** were the first layer and the original argument
for the architecture. Each expert encodes a named inductive
bias, so the gate weight maps directly to a business narrative.
The recommendation generation layer (Paper 2) uses these
weights as the primary input to the customer-facing reason
string. This worked well as long as the explanation question
was *"which lens did the model look through."*

**CEH (Causal Explainability Head)** was the second layer, and
it got added because supervisors kept asking a follow-up
question we couldn't answer with gate weights alone. After
*"Causal 38%"*, the next question was *"so which feature
*inside* the Causal expert drove the conclusion?"* We needed a
per-feature attribution layer underneath the per-expert one.
The first version of CEH — v1 with a task-logit target — was a
negative result we now keep around as Finding 13: the head
collapsed to a global importance pattern rather than producing
per-sample variance. v2 with a demeaned target restored the
per-sample signal, and that's what's running now. The
existence of the v1→v2 transition is itself part of the story
— an attribution head's target design is more sensitive than
the v1 framing led us to believe.

**Mahalanobis OOD on the Causal latent** was the third layer,
and it got added because a stable explanation isn't useful if
it comes from outside the model's training distribution. We
compute Mahalanobis distance on the Causal expert's latent
space against an in-distribution reference, and emit a binary
trust flag per prediction. On the synthetic OOD probe in Paper
3, this hits 100% TPR at 5% FPR. When the flag fires, the
recommendation reason string gets downgraded or withheld — the
gate weights are still there, but the system warns that the
explanation is being asked to interpolate outside what it
should be asked to.

All three layers compute at prediction time and land in the
audit log automatically. None requires a separate post-hoc
explainer call. The forward pass produces all of them.

## Why this turned out to be the regulatory foundation

This is the place where the architectural choice connects
upward to the compliance layer.

The five regulatory generators in Paper 2 — `KoreanFRIAAssessor`
(AI Basic Act §35), `FRIAEvaluator` (EU AI Act Art. 9),
`AnnexIVMapper` (EU AI Act Art. 11 + Annex IV), `PIAEvaluator`
(PIPA + GDPR Art. 35), and `PublicDisclosureGenerator` (FSC AI
guidelines) — all run as aggregation queries over the same
per-prediction audit log. They're queries, not authored
documents.

That pattern only works because the per-prediction record
contains *structured reasoning data*, not just inputs and
outputs. If the log entry were `(input_vector, output_score,
timestamp)`, no aggregation query could answer "why did the
model decide X for customer Y." The answer wouldn't be in the
data. The five generators can be queries because the log
captures reasoning, and the log captures reasoning because the
architecture produces reasoning as output.

Inherent XAI is the foundation. The audit log is the second
floor. The five regulatory generators are the roof. Swap the
foundation for post-hoc SHAP and the second and third floors
collapse — the per-prediction log loses its structured
explanation column, the aggregation queries lose their
substrate, and the regulatory artifacts revert to hand-written
documents.

EU AI Act Art. 13 (transparency obligations) and Korean AI
Basic Act §31 (transparency) aren't satisfied by *having an
explainer*. They're satisfied by *being able to produce a
specific, stable, reconstructible explanation for any
prediction on demand*. Inherent XAI is the only architecture
we know of that lets you keep that promise across model
retraining cycles.

## FD-TVS — what we learned from per-product weights breaking

The on-prem precedent for the scoring layer used per-product
weights. Each financial product had its own static weight,
configured manually. Whenever a new product launched, the
config was updated by hand. The scoring layer was, effectively,
a flat lookup table.

Three observations from running that for half a year pushed us
toward redesigning it.

The manual-config problem caused incidents. Each new product
required a config update, and a couple of misses produced
"new product never gets recommended" outages. Every single one
was visible only after the fact, in the form of zero
recommendations for that product over a week or more.

The segment-mismatch problem was quieter but more pervasive.
A 25-year-old first-time depositor and a 60-year-old high-net-
worth client were running through the same product weight
table. The original assumption was that segment effects would
be captured inside the model itself, with the scoring layer
agnostic. In practice, the model's predictions still benefited
from a layer of segment correction on top, and we didn't have
anywhere to put it.

The behavioral-shift problem was the third. Life-event
triggers — a sudden burst of small deposits in a previously
inactive account, a spike in a feature correlated with churn —
had no mechanism to influence the score. The behavioral signal
was visible in the features, but the scoring layer was
indifferent to it. Behavior was a signal with nowhere to land.

FD-TVS — Financial DNA Targeted Value Scoring — is the
redesign that came out of those three problems. Three decisions
shape it.

First, *task-level instead of product-level*. Weights attach to
the *task* (cross-sell intent, churn risk, suitability fit, etc.),
not the product. New products inherit the existing task
structure and don't require reconfiguration. The XAI gate
weights from the model feed into this directly — task selection
is informed by the per-expert routing of the prediction.

Second, *segment-aware*. `segment_task_weights` applies a
per-segment multiplier on the task weights, clipped to the
range 1.0–1.5. The clipping isn't arbitrary. We considered
allowing weights below 1.0, which would let segment heuristics
suppress task signals, but that broke the model's role as the
primary signal source — a segment override that turned off a
task entirely defeated the point of the model. We considered
allowing weights above 1.5, which would let segment overrides
dominate the model. Both were rejected. The 1.0–1.5 range
encodes the position: *segment matters as a multiplier, not as
an override*.

Third, *behavior-aware*. `dynamic_weight_rules` lets specific
feature thresholds boost specific task weights at scoring time.
A spike in a feature correlated with churn raises the
churn-task weight. A sequence of small deposits in a previously
inactive account raises the deposit-product task weight. This
is reactive scoring — *behavior itself is the signal that
triggers weight adjustment*, rather than a periodic re-tuning
that lags behind by days or weeks.

All three live in `pipeline.yaml`. Operations can adjust the
segment table or add a behavior rule without a code change.
That mattered specifically because scoring policy adjustments
need to ship in hours, not weeks, and every adjustment gets a
config-version stamp in the audit log so it falls inside the
same fifteen-month reconstruction window the rest of the MRM
stack obeys.

The connection to XAI is direct. The XAI layer tells the
system *why* a prediction was made (gate weights × CEH × OOD).
FD-TVS tells the system *how much that prediction should
weigh in the final score, given who this customer is and how
they're currently behaving*. Both layers log their inputs.
The customer-facing reason becomes "we recommended product P
because your recent spending pattern (Temporal 35%) and
product hierarchy fit (HGCN 28%), weighted up by your
segment's historical preference for this category" — a single
sentence whose every component is independently recoverable
from the audit log.

## What this foundation enables

Looking ahead to Eps 5 and 6:

**Ep 5 (RAG + LanceDB)** describes how the per-prediction
explanation log gets queried at scale. Vector retrieval over
the explanation column is what makes "find me predictions
where Temporal dominated and OOD fired in the last quarter"
answerable in seconds. The explanation column has to exist
before retrieval can do anything useful with it.

**Ep 6 (Modular adaptability)** describes why this architecture
holds up when regulations change. New regulation = new
aggregation query over the same explanation log. The XAI
foundation is regulation-agnostic; the regulatory layer is
swappable.

## What inherent XAI doesn't buy

Three honest limits of the architectural-XAI choice.

**Score validation by humans is still human work.** *"Is a
Temporal contribution of 35% the right explanation for this
customer's recommendation?"* is a judgment call that sits with
the recommendation review committee or the customer-facing
relationship manager. What's automated is that the contribution
is recorded, stable, and reconstructible. What's not automated
is whether the explanation makes business sense.

**Edge-case interpretability degrades to honest noise.** When
all seven experts contribute roughly equally (gate entropy
near maximum), the gate-weight explanation is "everything
contributed a little." That's an accurate answer but not a
useful one. We treat high-entropy predictions as a distinct
interpretive category — *low-confidence, high-entropy*
predictions get flagged for human review under the oversight
layer regardless of whether the OOD flag fires.

**The architectural commitment has a downside.** This whole
argument depends on the heterogeneous expert basket staying
stable. If a future iteration replaces the seven experts with
a single transformer, the gate-weight explanation disappears
and we're back to post-hoc XAI. This is a long-term commitment,
not a short-term implementation choice. The five regulatory
generators in Ep 6 are designed assuming this commitment holds.

## Next

Ep 5 goes one floor up — into the retrieval layer that turns
the per-prediction explanation log into a queryable system.
Why we chose RAG + LanceDB for ops/audit infrastructure, what
columnar version-aware retrieval gives us for fairness
monitoring, human oversight escalation, and quarterly
aggregation, and why the audit log isn't write-only.

Source: [Paper 1 (Zenodo)](https://doi.org/10.5281/zenodo.19621884)
on the heterogeneous expert architecture and gate-weight
explainability, and Paper 3 (forthcoming on Zenodo)
on CEH and Causal Guardrail (Mahalanobis OOD); FD-TVS scoring
config lives in
the [open-source repo](https://github.com/bluethestyle/aws_ple_for_financial)
in `configs/datasets/santander.yaml`, under the
`scoring.segment_task_weights` and `scoring.dynamic_weight_rules`
keys (the merged form is `configs/merged_config.json`). The design
rule itself is stated in the repo's root `AGENTS.md` and `CLAUDE.md`.
