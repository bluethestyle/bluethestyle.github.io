---
title: "[MRM Thread] Ep 2 — Champion-Challenger as a Gate"
date: 2026-04-21 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, sr-11-7, regulation, financial-ai, audit]
lang: en
series: mrm-thread
part: 2
alt_lang: /2026/04/21/mrm-ep2-champion-challenger-ko/
next_title: "Ep 3 — Chain of Custody for an Agent Pipeline"
next_desc: "Seven audit tables, the HMAC hash chain, and how EU AI Act Article 13-14 / KFCPA §17 mappings become code paths rather than checklists."
next_status: draft
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 2 of "The MRM Thread". Ep 1 framed MRM as structural
invariants; this episode walks one Challenger through the flow —
from training completion to production, where rejection halts it,
how emergencies intrude, and how the post-deployment loop closes.*

## 3 a.m., Monday

The weekly scheduled retraining job finishes on SageMaker. A
Challenger — call it v147 — sits in the registry, waiting. What
happens between this moment and a customer getting a new
recommendation is where the old process and ours diverge
completely.

In the conventional flow: a duty officer reads the email Tuesday
morning. A comparison report is requested, drafted on Wednesday,
submitted, and reviewed at the MRM committee's monthly meeting
next week. If approved, an engineer files a deployment ticket for
the next release window. Training-end to production: *two to four
weeks*; the rationale lives scattered across spreadsheets, slides,
and meeting minutes.

In our flow: the registration event fires `_decide_promotion()`
synchronously. Within seconds of the training job finishing, a
verdict is reached, an audit entry is written, and if promoted
v147 serves the next Lambda request. The decision is already
closed before anyone is awake.

The time difference sounds like it's about automation, but the
real point is *where the judgment lives*. The conventional flow
pushes judgment out to a committee. We push it into a code path.

## What happens inside the gate

Step into `_decide_promotion()`. This is the sequence v147 goes
through.

*Step 1 — operator override check.* Is `--force-promote` set? It
is not; this is scheduled retraining. Move on.

*Step 2 — champion existence.* The registry has a current champion,
v143. This is a real competition, not a bootstrap.

*Step 3 — fidelity floor.* Does the distilled student model clear
the fidelity floor against the teacher on each of thirteen tasks?
v147 clears all of them. If even one had failed, the function
would halt here and reject. The reason for this ordering is
coming back below.

*Step 4 — `ModelCompetition.evaluate()`.* Compare v143 and v147's
training metrics. Three criteria:

- Does the primary metric avg\_auc improve by at least 0.5%?
  v147 at 0.731 vs v143 at 0.724 — an improvement of 0.97%, pass.
- No secondary metric degrades by more than 2%? One task's AUC
  drops 0.8% — within tolerance, pass.
- Is the improvement statistically significant? t-test p-value
  0.012, below the 0.05 threshold, pass.

All three pass. `promotion_approved=True` lands, `registry.promote(
v147)` is called. An audit entry is written — `champion_version=
v143`, `challenger_version=v147`, `decision=promote`, reason =
competition summary, comparison = per-metric values, significance
= p-value. The function returns and the pipeline continues to the
next stage (serving-manifest update, CloudWatch notification).

Total elapsed time: under ten seconds. By 4 a.m. it is all
finished.

## Two weeks later, v148 stops at the same gate

Same `_decide_promotion()`, different ending.

At step 3, on two of the thirteen tasks the student-teacher KL
divergence exceeds the floor. Training produced a distribution
shift somewhere and the student diverged from the teacher as a
function. Look only at training metrics and v148's avg\_auc is
actually higher than v147's. But fidelity is checked *before*
competition, so the rejection is sealed here. Step 4 never runs.

Audit entry — `decision=reject`, reason = "2 fidelity failures:
task\_churn (KL=0.31 > 0.20), task\_next\_best (KL=0.28 > 0.20)".
v148 is stored in the registry with `promoted=False`, but it never
enters production.

Why fidelity sits before competition becomes visible here. If the
order were reversed — competition first, fidelity as a final check
— the temptation would emerge: "avg\_auc climbed that much, surely
a fidelity of 0.31 is tolerable?" Knowing the performance delta
makes it natural to nudge the fidelity floor. Putting fidelity
first means the floor operates independent of performance
information. Operational safety does not depend on competition
outcomes.

That sounds like a minor ordering detail, but it determines the
*shape of the answer* a year later when a regulator asks "why was
this model rejected?". Not "performance was insufficient after
comparison" but "fidelity floor violated, automatic rejection
regardless of performance" — the latter is a structural guarantee.

## Tuesday afternoon, an emergency

v147 has been in production a week. Tuesday afternoon at 2 p.m.,
the fairness monitor raises an alert — on a specific age × region
segment, the Disparate Impact ratio drops below threshold. The
next scheduled retrain is five days away. Can't wait.

The duty engineer finds v141 in the registry (a known-good version
that had successfully run production three weeks prior). One
explicit command:

```
python scripts/submit_pipeline.py --force-promote --version v141
```

`_decide_promotion()` runs again. This time it terminates at step
1 — with `--force-promote` set, every subsequent check is skipped
and v141 is promoted. Audit entry — `decision=force_promote`,
`trigger=manual`, reason = "DI breach emergency rollback to
v141", `operator` = engineer's ID.

Within two minutes, v141 is serving. The committee reviews the
audit entry later that Friday. The review is about "was this
override appropriate", not "did an override happen". The entry is
immutable; who intervened when and for what reason is permanently
fixed in the hash chain.

Making force-promote an explicit separate CLI option, not just
another flag, is the center of this design. There is no path
where a config file gets quietly edited to change the serving
version, nor a path that writes to the registry directly. Emergency
intervention must pass through the *explicit path*, and that path
necessarily writes an audit entry.

## The loop around the gate

MRM does not stop once v147 is in production. That is where the
second loop starts.

Every prediction is recorded in the HMAC hash chain (covered in
Ep 3). The fairness monitor computes Disparate Impact, Statistical
Parity Difference, and Equal Opportunity Difference on the
production stream in real time. The drift monitor aggregates PSI
and KL on feature and prediction distributions nightly.

When drift exceeds its threshold, the orchestrator automatically
triggers the next retraining job. When that job finishes, the
flow returns to the early-morning `_decide_promotion()`. Offline
gate → production monitors → retrain trigger → offline gate — a
closed loop.

What's interesting about this loop is that the human role inside
it is *not surveillance*. The loop runs itself. Human intervention
concentrates in two places: (1) emergency force-promote, and (2)
the meta-judgment about whether the loop's parameters —
`min_improvement`, `max_degradation`, the fidelity floor values,
drift thresholds — are still appropriate. The first falls to
engineers; the second to the MRM committee.

## What this flow changes about MRM

Ep 1 said "move MRM from a post-hoc report to a structural
property". This episode shows what that phrase looks like as a
flow.

Judgment no longer depends on the committee calendar. The fact
that v148 was rejected at 3 a.m. on a Monday is settled regardless
of whether anyone reads the email. A year after a promotion, when
a regulator asks, the audit log reconstructs that moment's
Champion metrics, Challenger metrics, significance test, and
rejection reason verbatim.

Conversely, what *remains the MRM committee's job* becomes
sharper. The gate only answers "is this better than the current
Champion". "Is the current Champion itself the right design?",
"Is a fidelity floor of 0.20 aligned with our risk tolerance?",
"Is 0.5% the right value for min\_improvement?" — those remain
human judgment. The committee's role shifts from *adjudicating
each Challenger* to *adjudicating the gate's design itself*.

That shift is Ep 1's core message. Human work did not decrease;
it concentrated on *what only humans can do*. It is why MRM can
function in a three-person organization that would never get
through a weekly queue of Challengers under the old model.

## Next

Ep 3 is about what else is being recorded around this flow — not
only promotion decisions but *every prediction* landing in the
HMAC chain, the role partition across seven audit tables, and how
EU AI Act Article 13-14 (transparency, human oversight) and KFCPA
§17 (financial-consumer dispute handling) mappings become code
paths rather than checklists.

Source material:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §4-5.
Implementation lives in `scripts/submit_pipeline.py`,
`core/evaluation/model_competition.py`, and
`core/monitoring/audit_logger.py`.
