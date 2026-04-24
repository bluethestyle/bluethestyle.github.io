---
title: "[MRM Thread] Ep 6 — Fairness as a Production Path"
date: 2026-05-05 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, fairness, monitoring, regulation, financial-ai]
lang: en
excerpt: "Disparate Impact, Statistical Parity, and Equal Opportunity across five protected attributes, computed on the production stream rather than a validation sample. The role of Counterfactual Champion-Challenger, and the Parquet archive where the evidence accumulates."
series: mrm-thread
part: 6
alt_lang: /2026/05/05/mrm-ep6-fairness-production-path-ko/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 6 of "The MRM Thread" (final). Eps 1-5 covered audit,
promotion, recording, FRIA, and human oversight. This episode is
the remaining big piece — fairness — and how it was implemented as
*real-time computation on the production path*, and how
Counterfactual Champion-Challenger fits into fairness judgment.*

## Not a quarterly report — every request

The conventional fairness audit pattern: quarterly, compute
Disparate Impact, Statistical Parity, and Equal Opportunity on a
validation sample, and submit a report. The FSS and other Korean
supervisors *ordinarily expect* this level.

The problem is that quarterly samples *aren't guaranteed to reflect
the production distribution*. Validation samples are usually random
hold-outs; production traffic shifts with seasonality, marketing
campaigns, new-customer-group inflows. A model with DI 0.88
(healthy) on validation showing DI 0.76 (threshold breach) in
production is a common scenario.

Our structure is *real-time computation on the production prediction
stream*. As each prediction accumulates into Ep 3's
`log_model_inference`, a fairness monitor recomputes DI/SPD/EOD
over a configurable sliding window (default 10,000 predictions or
24 hours). A threshold breach triggers the automatic kill switch
(Ep 5).

## Five protected attributes

The fairness computation targets five protected customer
attributes:

1. **Gender** (male / female / undisclosed)
2. **Age band** (≤20s, 30s, 40s, 50s, 60+)
3. **Region** (Seoul / metro cities / provinces)
4. **Income tier** (5-quantile)
5. **Disability** (self-reported, optional field)

Three metrics per attribute, simultaneously:

- **Disparate Impact (DI)** — positive rate in the protected group
  divided by the unprotected group. 4/5 rule (0.8 threshold)
  applied, alternative threshold in config.
- **Statistical Parity Difference (SPD)** — positive-rate
  difference between the groups. Alert at |SPD| ≥ 0.1.
- **Equal Opportunity Difference (EOD)** — True Positive Rate
  difference between the groups. Alert at |EOD| ≥ 0.1.

5 × 3 = 15 metrics under simultaneous monitoring per window. Any
single threshold breach triggers the automatic kill switch on the
affected task.

## Where 15 metrics still aren't enough — intersectional fairness

15 metrics look exhaustive, but there's a class of cases where
*every single-attribute metric passes* and the violation only
surfaces at the intersection. A synthetic-data example from our
testing, just in numbers:

- Gender alone DI = 0.85 (pass)
- Age band alone DI = 0.88 (pass)
- Income tier alone DI = 0.91 (pass)
- But **female × age 65+ × low-income** intersection DI = 0.62
  (clear violation)

Looking only at the marginal distributions, all three metrics are
above the threshold. The bias only appears when the protected
group clusters at the *intersection* of three axes. Marginal
checks are structurally blind to this. A fairness report that
reads "DI 0.85, 0.88, 0.91 — all passing" can hide a specific
intersectional cohort that is being consistently discriminated
against in the serving path.

Given AI Basic Act §34's requirements on high-impact AI providers
around explainability and user-protection, covering only
single-attribute metrics without looking at the intersectional
layer makes it hard to credibly claim "we monitored protected
groups adequately".

In our implementation this layer lives in
`core/agent/audit/intersectional_fairness.py`. It auto-scans
2-way and 3-way combinations of the five protected attributes.
The full space is combinatorially large (26 non-trivial
intersections beyond the singletons), but in practice we
evaluate **only the intersections whose sample size supports
statistical significance**. When an intersectional cohort is too
small, DI itself gets buried in noise. For each intersection we
compute DI / SPD / EOD and write a `log_operation` entry with
`event="intersectional_fairness"`. Threshold breaches flow into
the same alert path as the original 15 metrics.

This layer doesn't fire often. On synthetic data it's mostly
quiet. Once real production traffic accumulates, meaningful
intersectional biases may start to surface. A new line gets added
to the MRM committee's quarterly review: "intersectional
protected-cohort violations this quarter". When the single-
attribute DI sits at a comfortable-looking 0.85, this layer is
the thing that quietly catches 0.62 underneath.

## Sliding window, why that way

Recomputing on every prediction is computationally heavy and
over-reacts to minute-scale noise. Daily computation is too slow
to react. Between those extremes, a 10,000-prediction or 24-hour
sliding window is the pragmatic sweet spot.

The 10,000 shuffling runs on a dedicated Lambda worker, off the
main serving path — no latency impact. When the computation
finishes, the metrics push to CloudWatch; threshold breaches page
PagerDuty and write a `log_operation` entry.

## Evidence accumulating as Parquet archive

Every window's 15 metrics accumulate in an S3 Parquet archive,
partitioned by date, protected attribute, and metric name,
Snappy-compressed, append-only.

Why a Parquet archive — three uses:

1. **Regulatory audit evidence.** "Fairness trend from July through
   September 2026" answered in one query. DuckDB's httpfs extension
   queries S3 directly.
2. **FRIA evidence_ref.** When Ep 4's Korean FRIA assessor computes
   the "discrimination" dimension score on new-model promotion, it
   reads from this archive. The assessment rests not on a static
   document but on *observations from the recent production stream*.
3. **Input to Counterfactual Champion-Challenger.** Next section.

## Counterfactual Champion-Challenger

Ep 2's `_decide_promotion()` uses training-metric comparison (AUC
etc.). Fairness can be included, but training-set fairness does
not guarantee production-distribution fairness.

Counterfactual C-C answers a different question on the production
archive — "if the Challenger had been served, what would the
fairness metrics look like?" This is computed without actually
serving the Challenger, via Importance Sampling (IPS) /
Self-Normalized IPS (SNIPS).

The requirement: *logged propensities* from production (Champion's
probability for each alternative product recommendation). When
those are logged, the Challenger's hypothetical fairness can be
reconstructed. The counterfactual evaluator reads its estimator
choice (IPS / SNIPS), minimum lift threshold, and bootstrap CI
settings from config and executes.

This answers "is the Challenger better on fairness than the
Champion?" *on real production traffic*, without a live A/B split.
Particularly useful for a three-person team where a large serving
split is operationally burdensome.

## Still human: who sets the thresholds

The 4/5 rule (DI 0.8 threshold) is the US EEOC standard, but
Korean Financial Consumer Protection Act and AI Basic Act §35 may
demand stricter thresholds. Threshold decisions remain the fairness
committee's job, managed config-driven rather than baked into code.

Threshold changes must flow through meeting minutes → PR → review →
merge. Silent config edits are caught because every change lands
as a `threshold_change` entry in the audit chain — who changed
what, when, and for which reason is traceable.

The core of this structure — *the threshold decision itself is an
audit subject*. The quarterly committee review covers not just
"were there threshold breaches?" but also "is this threshold
reasonable?".

## How the full chain is designed to run

Real-traffic collection started 2026-04-30, so the chain below
has been exercised on synthetic fairness-breach scenarios rather
than actual production incidents. What follows is the design
target, not a case study — how the sequence is supposed to unfold
once production traffic accumulates at partner institutions.

Take a 주담대 추천 task where Disparate Impact on the 65+ age
segment drops below the 금소법 §17 적합성 threshold in a
10,000-prediction rolling window. The designed response chain:

*Minute 0* — fairness monitor closes the window, detects the
threshold breach, writes `log_operation` entry
`event="fairness_breach"`.

*Seconds later* — automatic kill switch fires on the affected
task. Serving Lambda env var updated; new inference requests for
that task return the Layer-3 financial-DNA rule recommendation
(a conservative fallback) instead of the distilled model output.
PagerDuty pages the on-call engineer; the `#mrm-incidents` Slack
channel notifies.

*Around minute 10* — the engineer, now at the dashboard, verifies
SPD and EOD confirm the DI drop isn't measurement noise. They
query the audit-log archive for the affected window to see which
features pushed predictions on the protected segment.

*Around minute 25* — Counterfactual Champion-Challenger analysis
runs on the archived window, testing whether the previous
Champion would have produced a better DI on the same requests.
If it would, that candidate surfaces as a rollback option.

*Around minute 35* — the engineer runs the force-promote command
targeting the previous known-good Champion, documenting the
rollback reason. `log_model_promotion` entry:
`decision=force_promote`, `trigger=manual`, reason = "DI breach
on 65+ segment for 주담대 task; previous Champion counterfactual
DI restores threshold", operator ID.

*Around minute 37* — the previous Champion is serving. The
affected task is re-enabled.

*Around minute 48* — OpsAgent, having watched the chain, drafts
an incident ticket summarizing the sequence for the MRM
committee's weekly review, linking to the relevant audit entries.

Well inside 금감원's 72-hour written-report window, the committee
receives a full narrative grounded in reconstructable audit
evidence rather than reconstructed-after-the-fact testimony.

Whether 45 minutes is achievable will be tested when actual
incidents arrive in production. The design goal — one human
decision, everything else structural — is what this episode's
architecture aims to enable. The minute count matters because
Korean mid-size institutions without 24/7 ML-ops rotations cannot
afford multi-day resolution times; this architecture is what makes
single-digit-hour resolution plausible for a small team.

## Recovery paths after a fairness failure

Fairness violation detected → automatic kill switch → recovery is
human judgment. Possible paths:

1. **Immediate rollback to a previous known-good model** (Ep 2's
   force-promote scenario)
2. **Fallback-rule application to specific segments only** (Layer 3)
3. **Disable only the affected task** (other tasks keep serving)
4. **Evaluate alternative models via Counterfactual C-C, then
   manually promote**

Which path depends on violation type and severity; the on-call
engineer decides. Every recovery trajectory logs to `log_operation`
for post-hoc pattern analysis.

## Closing the thread

Across six episodes we covered "MRM from post-hoc reports to
structural invariants" in concrete form — architecture (Ep 1),
promotion gate (Ep 2), audit chain (Ep 3), FRIA (Ep 4), human
oversight (Ep 5), fairness (Ep 6).

The pattern that runs through them — *regulatory compliance is
strongest when it is a byproduct of the code path*. Trying to
prove compliance via documents causes documents to drift from code
and lose effectiveness. Papering it over with committee meetings
causes meeting cadences to fall out of sync with incident
cadences. Wiring compliance into the hot path makes it maintained
as long as the system runs, and any break in compliance leaves its
own audit record.

In the constraints of a three-person team, one RTX 4070, and
limited resources, this structure became feasible precisely because
*regulatory compliance was not built as a separate deliverable*.
Audit logs are a byproduct of architectural decisions; promotion
records are a byproduct of the competition gate; regulatory
evidence accumulates as a byproduct of the fairness monitor.
Rather than adding a new moving part, the structure is arranged so
*what is already happening answers regulatory requirements*.

Still human work — threshold decisions, design review, incident
recovery judgment, law-change response. The MRM committee's
*subject of review* moved from reports to code paths, but the role
itself did not vanish. In an era when supervisors expect compliance
to be *architecturally legible*, this alignment is a natural
response.

## Where the thread rests

The MRM Thread closed its skeleton in six episodes. Subsequent
pieces will appear irregularly as Commentary in response to
specific incidents or issues (the Commentary category).

Source material: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052);
implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial).
