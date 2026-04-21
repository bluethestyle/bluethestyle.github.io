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
invariants; this episode applies that frame to one specific gate —
Champion-Challenger.*

## The textbook version

In most organizations, Champion-Challenger lives like this:

1. The development team trains a new model (the Challenger).
2. It runs in parallel with the production model (the Champion) on
   shadow traffic or a holdout, accumulating comparison metrics for
   several weeks.
3. An MRM team reviews a comparison report — an Excel sheet, a
   PowerPoint, or a tab in an internal dashboard.
4. A committee decides "promote / hold / reject" at its monthly
   meeting.
5. If promotion is approved, the development team files a
   deployment ticket and the swap happens in the next release
   window.

This structure is reasonable in a world where models are replaced
quarterly. The delay between promotion decision and actual swap
functions as a safety buffer rather than a liability.

The problem is not the delay itself, but that *the decision
exists only on paper*. The criteria used, the metrics compared,
the statistical significance, the rationale when rejected — they
live in slides and meeting minutes. Five years from now, when a
regulator asks "why was the Challenger on 2026-04-23 rejected?",
someone has to go find the answer. With luck, they find it.
Without luck, the person who wrote it has left the company.

## What we changed

In our pipeline, Champion-Challenger is the `_decide_promotion()`
function in `scripts/submit_pipeline.py`. The moment a training
job finishes and the model is registered, this function is called
synchronously. It returns nothing — whether or not it ends up
calling `registry.promote(version)`, *once the verdict is reached,
the function returns and the pipeline continues*. There is no
queue of pending decisions.

The synchronous nature matters. The decision moment *is* the
invocation moment. There is no intermediate state called "pending
review at the next committee".

## Four decision shapes

Every invocation ends in one of four outcomes — and all four get
an HMAC-signed entry in the same audit log.

**1. `force_promote` — operator override.**
When the `--force-promote` CLI flag is present, the comparison,
significance test, and fidelity check are all skipped and the
model is promoted unconditionally. This exists for emergency
rollback scenarios. For example: a fairness violation is detected
in the Champion, or a regulatory directive lands, and an
alternative must go into production *right now*, regardless of
whether the improvement is statistically significant. The
operator sets `--force-promote` explicitly; the audit log records
`decision="force_promote"`, `trigger="manual"`, `reason="Operator
override via --force-promote"`. Six months later, anyone reviewing
the log can identify this entry as a moment of explicit human
intervention, not automated judgment.

**2. `bootstrap` — first registration.**
If the registry has no existing Champion, there is nothing to
compare against. The Challenger is bootstrap-promoted. It becomes
the Champion, but the audit log records `decision="bootstrap"`
so the distinction "promoted by winning a contest" versus "promoted
because it was the first one through the door" is explicitly
preserved. This distinction matters because bootstrap does not
carry the same level of assurance as a competition pass — until
the next Challenger goes through the gate, additional monitoring
is warranted.

**3. `reject` (fidelity failure) — safety floor.**
If the distilled student model violates the teacher-student
fidelity floor on any task (regardless of how good its training
metrics look), it is rejected *before* the competition stage. That
this comes before competition is a deliberate design choice. The
textbook version says "if performance is better, swap"; but
student-teacher fidelity is an assurance condition independent of
performance. If distribution shift during training caused the
student to diverge significantly from the teacher, the fact that
its avg_auc went up does not mean it should go to production. The
floor blocks "high-performing wrong models".

**4. `promote` / `reject` (competition) — the honest case.**
If none of the above apply, `ModelCompetition.evaluate()` runs.
The default criteria:
- primary metric (avg_auc) must improve by at least 0.5%
- no secondary metric may degrade by more than 2%
- t-test or bootstrap significance `p < 0.05`

All three satisfied → `promotion_approved=True` → promoted.
Any failure → Challenger stays in the registry but does not enter
production. The audit log records `decision="reject"` with the
specific failure condition in the `reason` field.

## Why the safety floor comes before competition

This ordering was the most deliberate design choice in the gate.
The instinct is to run competition first and apply fidelity checks
to the winner. But that ordering lets the competition outcome
bias the fidelity judgment. If the Challenger is significantly
better, the temptation is to ask "can we tolerate this small
fidelity delta?" Inverting the order — check fidelity first —
keeps the floor independent of performance information. Operational
safety does not depend on competition outcomes.

## Audit entries are not an afterthought

For each of the four verdicts, `_audit_promotion()` is invoked
and `AuditLogger.log_model_promotion()` appends one entry to the
HMAC hash-chained log. The fields recorded:

- `champion_version` (previous Champion, None if bootstrap)
- `challenger_version` (this Challenger)
- `decision` (`force_promote` / `bootstrap` / `promote` / `reject`)
- `reason` (for automated verdicts, `ModelCompetition`'s
  `decision_reason`; for manual, the operator's explanation)
- `comparison` (per-metric Champion / Challenger values)
- `significance` (t-test p-value, bootstrap p-value)
- `trigger` (`auto` / `manual`)

The invariant this field set establishes is what makes SR 11-7
Pillar 2's "effective challenge" requirement satisfiable as a
*reconstruction property*. When a regulator in 2030 asks "why was
the Challenger rejected on 2026-04-23?", the five-year-old
Champion metrics, Challenger metrics, significance test results,
and rejection reason are all in the log, verbatim. It does not
matter whether the person who ran the pipeline is still at the
company.

One deliberate detail in the `_audit_promotion` implementation:
audit log write failures do not block the promotion itself. The
log has a local fallback first; S3 retry is asynchronous. Reason:
if audit failures could halt operations, the audit infrastructure
becomes a single point of failure for production. The audit log
must be "reliably recoverable later", not "guaranteed right now".

## What this buys, and what it doesn't

What it buys:

- A retrievable answer to *which version went to production when
  and why*, available instantly
- Continuous A/B capability on promotion decision quality, not
  bounded by quarter edges
- An explicit emergency rollback path (`--force-promote`),
  distinguished from quietly mutating a config file
- Clear separation of bootstrap from competition pass, so initial
  low-trust states are not laundered into "approved"

What it does not buy:

- *Online competition* — A/B distribution comparison on accumulated
  production traffic is a separate trigger
  (`ModelMonitor.evaluate_champion_challenger`) and is not part of
  the offline gate. The offline gate looks only at immediate
  post-training metrics.
- *Validation of the design itself* — this gate answers "is it
  better than the current Champion". It does not answer "is the
  current Champion itself the right design". That question — what
  Ep 1 called "challenging the architecture itself" — remains MRM
  oversight's job.

The second limitation matters. Making Champion-Challenger a gate
does not mean MRM committees go away. Their role shifts from
*adjudicating each Challenger* to *adjudicating the gate's design
itself*. Is min_improvement set right at 0.5%? Is max_degradation
of 2% aligned with our risk tolerance? These meta-questions are
the committee's new job.

## Next

Ep 3 covers the other side of the audit story — how every
*prediction*, not just every promotion, becomes an entry in the
HMAC chain, the role partition across seven audit tables, and
how EU AI Act Article 13-14 (transparency, human oversight) and
KFCPA §17 (financial consumer dispute handling) mappings become
code paths rather than checklists.

Source material:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §4-5.
The implementation lives in the open-source repo at
`scripts/submit_pipeline.py`, `core/evaluation/model_competition.py`,
and `core/monitoring/audit_logger.py`.
