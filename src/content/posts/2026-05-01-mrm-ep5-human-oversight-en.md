---
title: "[MRM Thread] Ep 5 — Human Oversight as an API, Not a Ticket Queue"
date: 2026-05-01 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, human-oversight, kill-switch, regulation, financial-ai]
lang: en
excerpt: "EU AI Act Article 14's human-oversight requirement implemented as API endpoints rather than a ticket queue — kill switch, HumanReviewQueue tiers 2/3, and why auto_promote=false is enforced as a production posture."
series: mrm-thread
part: 5
alt_lang: /2026/05/01/mrm-ep5-human-oversight-ko/
next_title: "Ep 6 — Fairness as a Production Path"
next_desc: "Disparate Impact, Statistical Parity, and Equal Opportunity across five protected attributes, computed in real time on the production stream — not a validation sample — and the role of Counterfactual Champion-Challenger."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 5 of "The MRM Thread". Ep 4 mentioned the "human oversight"
dimension of FRIA only in passing. This episode unfolds how that
dimension is actually implemented — EU AI Act Article 14 human
oversight living as API endpoints rather than a ticket queue.*

## What a fairness incident costs in response time

Real-traffic collection started 2026-04-30 with partner
institutions, so the system hasn't yet processed a genuine
fairness incident in production. But the design intent is
explicit — *the on-call engineer's decision window matters more
than the detection latency*, and the architecture is oriented
around shortening that window.

Consider the shape of an incident the system is built for. A
protected-group breach surfaces in the fairness monitor: say,
Disparate Impact on the 65+ age segment drops below the 금소법
§17 적합성 threshold for an 적금 상품 추천 task. At a mid-size
Korean institution with a small ML team, three response paths
exist — wait for the next scheduled retrain (typically four to
seven days away), route through the human review backlog (twelve
to twenty-four hours on a healthy Korean IT operations calendar),
or intervene now.

The first two are unacceptable because 금감원 incident reporting
requires phone notification within 24 hours of a significant
failure and written report within 72 hours. Waiting five days is
not a response; it's a second incident.

So the question is what "intervene now" looks like in practice.
In most Korean financial institutions, it's an incident ticket
filed into a queue that on-call staff triage in business hours;
mean time to resolution is measured in days. In our architecture,
it's one API call by the on-call engineer — design choice, not
inevitability.

## Organizational process vs. API endpoint

EU AI Act Article 14 — "high-risk AI systems shall be designed so
users can effectively oversee them." What counts as "effective
oversight" is not dictated in concrete form by the regulation.
Most financial institutions answer with *organizational process*:
"monthly MRM committee reviews monitoring outputs", "anomaly
detection alerts go to the duty officer by email", "urgent
situations are managed via JIRA tickets".

All of the above technically satisfy the *letter* of Article 14.
Whether they work during an actual incident is a different
question. Does the duty officer read email in time? Does the JIRA
queue have a five-day backlog? Is the monthly committee meeting
now discussing an incident from three weeks ago?

Our approach is to deliver the oversight means as *API endpoints*
rather than processes. A call, not a process. Calls are real-time
and idempotent; they do not get buried in a queue.

## Kill switch — one call to disable a specific task

The simplest API is the kill switch. Disable the whole pipeline,
disable a specific task (one of the 13), roll back to a specific
model version — one API call each.

```
POST /admin/kill-switch
{
  "scope": "task",
  "target": "churn_signal",
  "reason": "DI breach in age 60+ segment",
  "operator": "engineer_id"
}
```

What this single call accomplishes:
- Immediately blocks the task's serving path (Lambda env var update)
- Writes `event="kill_switch_fired"` to the `log_operation` audit
  table
- Dispatches to Slack and PagerDuty simultaneously
- Triggers OpsAgent to open an incident ticket automatically

*What organizational process cannot do* — at 3 a.m. when the
fairness monitor fires, there's no JIRA ticket to wait for. The
on-call engineer calls this API. Decision time equals action
time. The "we knew but it sat in the backlog" scenario is
structurally impossible.

## HumanReviewQueue — automatic escalation at tiers 2 and 3

The kill switch is for "something is already wrong" moments. The
step before that — *automatically handing suspicious cases to a
human* — is the HumanReviewQueue.

Three tiers:

- **Tier 1** — auto-approved. Model confidence high (low gate
  entropy) + fairness within margin + suitability filter passed.
  The vast majority.
- **Tier 2** — human review required. Fairness margin near the
  threshold, or low confidence, or protected-attribute customer
  (elderly, low-income). Queued for processing within 24 business
  hours.
- **Tier 3** — blocked immediately and escalated for human
  decision. Clear fairness-violation signal, hallucination
  detection, regulatory-keyword triggers (insider information,
  capital-products restricted-phrase list, etc.). Not delivered to
  the customer until a human adjudicates.

The key property: tier assignment is *automatic*. A human does not
classify "this is tier 2"; if one of the three conditions holds,
the tier is raised. The tier taxonomy is a rule set, and those
rules themselves become the MRM committee's review target.

## auto_promote=false — a production posture

Ep 2 described the promotion gate as auto-promoting when all
conditions pass. *Technically*, yes. But in production, auto-promote
is disabled by config, so no Challenger auto-promotes even if it
clears every metric. The current Champion stays unless an
operator-issued force-promote override fires.

Why this matters — *the condition for auto-promotion* sits on the
assumption that every gate is a structural invariant. In reality,
that assumption itself is a matter of human judgment. "Is this
Challenger actually ready for production?" is not answered by a
single metric. In the first half of 2026, our team gates every
promotion with a *manual sign-off*.

This connects to Article 14's *meaningful human oversight* clause.
Automated promotion leans toward "the system decides"; manual
promotion leans toward "a human decides". Until the real-data
metrics stabilize, the latter posture is judged appropriate.

## Layer 4 — opt-in human fallback

The serving path has a three-layer fallback router:

- **Layer 1** — PLE → LGBM distilled model (the normal path,
  99%+ of traffic).
- **Layer 2** — when distillation fails, the LGBM trained
  directly on labels.
- **Layer 3** — financial-DNA rules (when the full model pipeline
  fails).

In the first half of 2026, a Layer 4 was added — **human
fallback**. It activates only when the tier-3 human-fallback flag
is enabled in config. When Layers 1-3 all have low confidence, no
recommendation is generated at all; instead the customer sees
"connecting you to a representative".

Layer 4 is opt-in because not every institution can put the
human resources behind it. It's available as an option, default
off. An institution that turns it on has to operate a parallel
human-review queue.

## Answering Article 14 with actions, not documents

When an FSS auditor requests Article 14 compliance evidence:

- Kill-switch API call history → `log_operation` table.
- HumanReviewQueue tier 2/3 processing history → its own table.
- Promotion manual-signoff history → `log_model_promotion`
  entries with `trigger=manual`.
- If Layer 4 is active, human-fallback call counts → its own
  aggregation.

Each record is the byproduct of an actual API call or event. The
answer is not "we *have* these processes" but "we have *performed*
these calls N times in the last twelve months". The latter answers
audit questions far more robustly.

## Still human work

Providing an API and *actually calling it* are different. A kill
switch API doesn't help if the on-call engineer at 3 a.m. doesn't
read the fairness alert and make a decision.

So "API-based human oversight" is not an automation argument; it
is an argument for *lowering the barrier to human intervention*.
Designed so that an API call replaces the ticket-queue wait, the
on-call engineer's judgment translates immediately into action
instead of aging in a backlog. But the *judgment itself* remains a
human decision.

The MRM committee's job shifts too. Instead of reviewing a report
at a monthly meeting, they look at call logs — "was this kill
switch firing appropriate?", "is the tier 2/3 ratio healthy?",
"are the manual-promotion rejections excessive?". The *subject of
the review* moved from the report to the call log.

## Next

Ep 6 covers Article 14's cousin — the fairness architecture that
computes Disparate Impact, Statistical Parity, and Equal
Opportunity *in real time on the production stream* across five
protected attributes, and the role of Counterfactual
Champion-Challenger in fairness judgments.

Source:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §7
"Human-in-the-loop design"; implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial).
