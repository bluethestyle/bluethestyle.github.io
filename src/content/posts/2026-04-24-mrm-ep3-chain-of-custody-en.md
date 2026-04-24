---
title: "[MRM Thread] Ep 3 — Auditing the Auditors: Chain of Custody and Consensus Arbitration"
date: 2026-04-24 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, audit, consensus, sr-11-7, regulation, financial-ai]
lang: en
excerpt: "Seven audit tables and an HMAC hash chain give you 'continuity of record'. But who verifies the record? The trap of the single-LLM auditor, multi-agent consensus with α/β/γ perspectives, a minority-report-that-never-gets-deleted design, and why AWS parallel voting and on-prem 2-round deliberation chose different paths."
series: mrm-thread
part: 3
alt_lang: /2026/04/24/mrm-ep3-chain-of-custody-ko/
next_title: "Ep 4 — FRIA: How the Korean AI Basic Act §35 Lives in Code"
next_desc: "Korea AI Basic Act §35 seven-dimension impact assessment, five-year retention, and why it's kept separate from the EU AI Act Article 9 FRIAEvaluator."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 3 of "The MRM Thread". Ep 2 covered the record of promotion
decisions. This episode goes one layer down at a time — into *what*
the record contains (seven audit tables, an HMAC chain), *who*
verifies it (Ops and Audit agents), and *how those verifiers keep
each other honest* (three-way parallel voting on AWS, 2-round
deliberation on-prem).*

## The question in June 2027

A scenario. A customer from April 2026 files a dispute at a
review window in June 2027. "Fifteen months ago I was recommended
this product. I believe the recommendation acted discriminatively
based on my income tier. On what basis was this product
recommended to me?" Under KFCPA §17 (suitability principle) the
institution must answer.

The conventional model offers a few answer shapes — qualitative
narratives like "we cannot reproduce the feature scores from that
moment, but the model generally ...", or SHAP approximations such
as "this customer belonged to the product-A preference segment".
For regulatory response, none of these are enough. They lack
point-in-time specificity and reconstruction ability.

Our pipeline answers in a different shape. At the moment the
recommendation left the system 15 months ago, we can *pinpoint
what was written to which audit log*, and we can *independently
certify that the log was not tampered with*. How those two layers
of specificity were stacked is the subject of this episode.

## How seven audit tables split the load

The audit-logger module exposes seven `log_*` methods, each
writing to its own table.

- `log_operation` — system state transitions (retrain start/end,
  promotion, serving-manifest swap)
- `log_model_inference` — when each prediction occurred, for whom,
  by which model version
- `log_data_access` — who touched which data
- `log_dimension_change` — feature-schema shifts (349D → 403D)
- `log_attribution` — per-prediction feature-contribution record
- `log_guardrail` — what Safety Gate blocked or rewrote
- `log_model_promotion` — Ep 2's promotion decisions

The seven-table split follows separation of concerns. Mixing
everything into one table tangles search, audit, and retention
policy. `log_data_access` sits under PIPA and the Credit
Information Act — different retention schedule. `log_model_promotion`
sits under SR 11-7 Pillar 2 — different access rights. A single
combined table would let the strictest policy dominate and explode
storage cost.

## How one request flows through the tables

Real-traffic collection with partner institutions began
2026-04-30, so the reconstruction properties described above
haven't yet been tested at operational scale. What exists is the
design. To see how the design works, walk through a typical
Korean branch scenario — a customer arriving to roll over a matured
time deposit, with the system proposing a savings + deposit combo.

Four tables get written in sequence during the inference:

- `log_data_access` is written first, when the branch teller
  (after authentication) submits a recommendation request on the
  customer's behalf. Operator ID, customer ID, and access reason
  ("branch-initiated recommendation") are recorded. This is the
  audit linkage point for PIPA Article 37-2.
- `log_model_inference` is written when the distilled LightGBM
  model runs the 13 tasks and produces scores. Model version
  pointer and feature-tensor hash are captured, so "which model
  judged which inputs" is reconstructable later.
- `log_attribution` follows immediately, holding top-K feature
  contributions and the expert-gate weights. This is the evidence
  base that answers "why this recommendation for this customer"
  under EU AI Act Article 13 and KFCPA §17.
- `log_guardrail` is written when the Safety Gate agent reviews
  the recommendation explanation against five criteria
  (regulatory compliance, suitability, hallucination, tone,
  factuality). The verdict (pass/modify/block) and per-criterion
  scores are recorded regardless of outcome.

The whole inference takes under a second. Four audit writes add
negligible overhead because each is a small standardized JSON
append rather than a network round-trip.

The remaining three tables do not write per request.
`log_operation` writes on system-state transitions (nightly
retrain start/end, serving-manifest swap, etc.), `log_dimension_change`
on feature-schema changes (e.g. 349D → 403D after a Phase-0 revision),
`log_model_promotion` on champion-challenger decisions (Ep 2).

## HMAC hash chain — why this isn't "just a log"

Each entry isn't simply written. When `log_operation()` is
invoked, three things happen at once.

1. **Entry serialization** — timestamp, event type, and payload
   are normalized to canonical JSON.
2. **HMAC signature** — the hash of the previous entry plus the
   current entry content is signed with the audit-chain secret key.
3. **Chain linkage** — the current entry's `prev_hash` field is
   populated with the previous entry's hash.

The consequence of this structure — *to modify entry n you must
re-sign n+1, n+2, ..., up through the latest entry*. That is
impossible without the HMAC key. Tamper-evidence is thus
guaranteed by structure, not by trust.

On AWS key management: the audit-signing key is stored in SSM
Parameter Store (SecureString), accessible to the Lambda runtime
only through IAM. Key exposure means the whole audit chain
collapses, so this is one of MRM's *distinct verification points* —
"we keep audit logs" only starts to count once followed by "and
the audit-signing key sits under a separate rotation policy".

That covers the first half of Ep 3. A record exists, and the
record is structurally guaranteed against tampering. But one
question is still open.

## Who verifies the log — the single-LLM auditor trap

Having an audit log doesn't guarantee "the log is correct". Who
checks every day that the hash chain hasn't broken? Who spots
anomaly signals? Who reads regulatory-keyword violations?

The simple answer is "run one LLM agent nightly to verify". The
initial prototype did exactly that — one Sonnet session reading
the last 24 hours of audit logs and returning either "no
anomalies" or "attend to the following".

That design's weakness showed up within a few weeks. Identical
inputs against the same prompt produced subtly shifting outputs.
Some nights it caught a particular anomaly; other nights the same
pattern slipped by. Even with lower temperature and hardened
system prompts, the *direction* of hallucination was
unpredictable. And one observation was more uncomfortable than
the rest — **when a single-LLM auditor reaches a wrong verdict
with high confidence, the wrong verdict gets dressed in fluent
language that makes it look more plausible than it should.**

The most dangerous scenario under regulatory audit isn't "the
auditor existed but missed a specific anomaly". It's "the
auditor existed, the auditor confidently reached a wrong verdict,
and the verdict was logged as basis for subsequent decisions".
A single-LLM auditor is structurally exposed to exactly that
scenario.

## Ops and Audit agents — splitting the viewpoint

The first move to address this was to split the auditor's
*viewpoint* into two.

- **Ops agent** — the *"is the pipeline operating healthy"*
  viewpoint. Looks at performance metrics, stability, cost, drift,
  cold-start latency. Runs nightly, classifies system state as
  GREEN / YELLOW / RED, emits an `attention_required` list.
- **Audit agent** — the *"are we meeting regulatory
  obligations"* viewpoint. Looks at HMAC-chain integrity,
  fairness-indicator breaches, explanation quality,
  regulatory-keyword hits. Runs weekly, emits risk level
  (LOW / MEDIUM / HIGH) and `focus_areas`.

Both agents read the same audit logs but *ask different
questions*. Ops asks "did p99 latency cross 300ms yesterday?";
Audit asks "among `log_guardrail` events yesterday, was the block
rate concentrated on a protected group?". What one might miss,
the other catches.

They share the same three-step loop:

```
Collect    (measurements per checkpoint)
  → Diagnose    (thresholds · trends · cross-correlation rules)
    → Report      (tell the on-call "where to look")
```

The difference lies in *which checkpoints* and *which rules*.
Ops reports take the shape of a list of
`finding + likely_cause + suggested_action`. Audit reports take
the shape of a structured bundle of
`focus_areas + regulatory_summary + reason_quality_dashboard`.

## Consensus arbitration — don't trust a single session

Splitting the viewpoint doesn't fix the *single-LLM wobble*.
One Ops session and one Audit session still render verdicts
alone.

The second device addresses that. Each agent's Diagnose step
isn't run alone — **multiple independent sessions run the same
diagnosis in parallel**. Each session is given a different system
prompt that assigns a distinct *perspective*.

- **α (Alpha) — conservative.** "What matters is not missing a
  risk signal. When in doubt, verdict WARN."
- **β (Beta) — statistical.** "Check whether sample size is
  adequate and whether movement is statistically significant."
- **γ (Gamma) — business.** "Judge by impact on real customer
  experience and business KPIs."

Each session returns one of PASS / WARN / FAIL, a confidence in
[0.0, 1.0], and a 500–800-token reasoning. These opinions land
at the **ConsensusArbiter**, which classifies the outcome into
one of three categories.

- **Consensus** — all sessions agree. Treated as *confirmed* and
  committed directly to the report.
- **Majority** — most sessions agree, a minority dissents. The
  majority verdict is adopted but carries a *priority review*
  flag.
- **Minority Report** — opinions split. The majority decision is
  still used, but *the minority opinion is preserved on a
  separate track, permanently*.

The structural win is one sentence. **If one session hallucinates
a wrong verdict, the other sessions — reasoning from different
perspectives — will not all land the same way, so the final
result gets a minority flag automatically.** The exact situation
in which a single-LLM auditor could confidently produce a wrong
answer is also the situation in which a second perspective is
structurally surfaced as a *second opinion*.

## Why AWS and on-prem chose different consensus methods

Under the same "multi-agent consensus" goal, AWS and on-prem
environments use *different consensus methods*. The split falls
naturally out of three constraints: model capability, cost
structure, and latency budget.

**AWS — independent parallel voting (jury model)**

On AWS, three Sonnet sessions run in parallel. They cannot see
each other's output. The α/β/γ perspectives each render a verdict
independently; ConsensusArbiter aggregates the three. Wall-clock
time about 5 seconds, cost roughly 3× a single session.

Three reasons this fits AWS. (1) Sonnet is strong enough that a
single session's verdict quality is already usable — running more
sessions is for *cross-checking the direction of wobble*, not for
raising baseline verdict quality. (2) Parallel execution is
roughly linear in cost and adds negligible latency, so "a 3-person
jury in 5 seconds" is operationally practical. (3) Sessions don't
see each other, so *conformity bias* — the drift where minority
opinions get pulled toward majority ones — is eliminated at the
root.

**On-prem — 2-round hybrid (independent vote → sequential
deliberation)**

On-prem runs a 14B Q4-quantized model on a single RTX 4070. That
model is weaker than Sonnet. Under this condition, blindly
applying the "3 independent sessions" recipe breaks down: three
weak models wobble independently and consensus itself becomes
unstable. Pure sequential deliberation (the Delphi method) goes
the other way — *later sessions drift toward earlier ones*, and
minority opinions disappear into conformity bias.

In operations/audit work, **"missing something" is much worse
than "raising a false alarm"**. A false alarm can be cleared by
the duty operator; a missed anomaly compounds into a regulatory
incident. So on-prem uses a 2-round structure that offsets both
weaknesses in alternation.

```
[Round 1: independent voting — minority preservation]
  ① → "filter issue"     (independent, no cross-talk)
  ② → "sample-size issue"(independent)
  ③ → "PASS"             (independent)
  ④ → "filter issue"     (independent)
  ⑤ → "seasonal pattern" (independent)

  tally: filter 2, sample 1, PASS 1, seasonal 1
  minority locked: ③(PASS), ⑤(seasonal) — cannot be deleted later

[Round 2: sequential deliberation — argument refinement (minority
  not deletable)]
  ⑥ → sees Round 1 in full; organizes the majority argument +
       evaluates the validity of the minority
  ⑦ → sees ⑥; renders final verdict + records supporting /
       rebutting evidence for each opinion
```

Round 1 *locks in the minority* via independent voting. Round 2
*lifts argument quality* via sequential deliberation. The single
rule — **a minority view locked in at Round 1 cannot be deleted
in Round 2**. If Round 2's ⑥ concludes "⑤'s seasonal-pattern
hypothesis is low-validity", that becomes the record "⑥ judged
validity low — grounds: ... / original opinion (⑤) preserved".
The final decision still belongs to a human.

On-prem default is 5 sessions (R1=5, R2=2); high-risk verdicts
escalate to 7 (R1=7, R2=2). Each session on a 14B Q4 model takes
~30–40 seconds. Only WARN/FAIL items are subject to consensus,
typically 5–10 per night — roughly 45 minutes total,
well within a post-check time window.

| Env | Model | R1 | R2 | Total sessions | Per-item time |
|---|---|---|---|---|---|
| AWS | Sonnet | 3 (parallel) | — | 3 | ~5 s |
| On-prem default | 14B Q4 | 5 (independent) | 2 (deliberation) | 7 | ~2 min |
| On-prem high-risk | 14B Q4 | 7 (independent) | 2 (deliberation) | 9 | ~2.5 min |

The two environments converge on a **shared final classification
schema**. Both AWS 3/3 and on-prem 5/5 map to *Consensus*; AWS
2/3 and on-prem 3/5+ to *Majority* (priority review); AWS 1/3 and
on-prem 1–2/5 to *Minority Report* (second-priority review).
Audit logs use the same field names regardless of environment,
so the MRM committee reviews one flow even when the underlying
machinery differs.

## Minority reports never get deleted — where SR 11-7 Effective Challenge lives

The two implementations differ, but they share one principle.
**Minority reports, once identified, are never deleted.**

A minority verdict that didn't win the majority is still
written as a separate `log_operation` entry and kept permanently.
When the MRM committee reviews quarterly, the targets include
"how many verdicts this quarter carried a minority flag?", "in
how many of those did the minority turn out to be right in
hindsight?", "rereading the minority reasoning now, do we see a
perspective we missed?".

The regulatory reason this matters: it maps cleanly onto SR 11-7's
*Effective Challenge* requirement. SR 11-7 Pillar 2 calls for
effective challenge to exist inside the organization. Traditionally
this was satisfied by 2nd-line-of-defense independent review. As
firms shift to AI-backed audit systems, there's a real risk of
drifting into "the LLM verifies everything, so we can reduce
human verification".

Our design goes the opposite direction. *We don't trust any
single LLM session.* Multiple independent sessions are
*structurally* made to challenge one another from different
perspectives, and the *product of that challenge* (the minority
report) is *not deleted*. Part of what humans used to verify gets
automated — but the machine-vs-machine challenge leaves behind
a *record* that humans review later.

## On-prem rule engine — the system must still work without LLMs

One more concern. What happens if Sonnet calls fail, or the
on-prem 14B weights get corrupted? That's a regulator-facing
question with no avoidable answer.

Our answer is dual-layer. The agent's essential verdict function
is designed to be complete with a **deterministic Python rule
engine** alone. A 48-item threshold-based checklist is baked into
the rule engine, and this alone is enough for Ops to render
GREEN/YELLOW/RED and for Audit to render LOW/MEDIUM/HIGH.
Reproducibility 100%, cost 0, external-network dependency 0.

The LLM consensus layer sits on top as a **convenience layer**.

- **Interpret & Discuss (Sonnet)** — translates rule-engine
  verdicts into natural language and answers "why this verdict"
  when the operator asks.
- **Impact Review (Sonnet)** — pre-reasons the audit-indicator
  impact of a configuration change.
- **Deep Audit (Opus, quarterly)** — long-form reasoning over
  trade-offs between multiple regulatory regimes.

AWS 3-way parallel voting and on-prem 2-round deliberation both
live inside this convenience layer. If Bedrock goes down, or the
local model dies, the rule engine keeps running and the audit
system does not *fully* stop. The AI auditor assists; the system
meets its regulatory obligations even without one.

## Turning regulation into code paths

EU AI Act Article 13 (transparency) requires "high-risk AI systems
shall be designed such that users can interpret the output". The
conventional answer was "we maintain a transparency document". In
our structure the answer is a `log_attribution` entry (the feature
ranking that drove that specific prediction) plus a `log_guardrail`
entry (which Safety Gate criteria were passed). Article 13 compliance
becomes *one query*.

Article 14 (human oversight) — in our structure the answer is the
`HumanReviewQueue` API endpoint, the kill-switch API, and the
`log_operation` record of kill-switch firings. The details are
covered in Ep 5.

Article 15 (accuracy, robustness, cybersecurity) — join
`log_model_inference` and `log_dimension_change` and you can
ask "what accuracy did model v143 produce, when, under which
schema?" on demand. The conventional "quarterly report" collapses
to a SQL one-liner.

KFCPA §17 (financial-consumer dispute handling) — the question in
the opening scenario. Joining `log_model_inference`,
`log_attribution`, and `log_data_access` reconstructs "on
2026-04-15 14:37, for this customer ID, model v143 recommended
which product with which feature combination". Reconstructable 15
months later — and because AuditAgent has verified the hash chain
every day since, the reconstruction is guaranteed to be
*untampered*.

## Still the MRM committee's job

The same pattern from Ep 2 repeats here. The architecture
resolves some things; others remain human judgment.

What the architecture resolves:

- **Reconstruction** of any prediction, decision, or configuration
  change at any prior point in time
- **Structural certainty** of answers to regulator queries 15
  months later
- The **multi-session consensus structure** that reduces
  single-LLM wobble
- A **24-hour upper bound** on tamper-detection latency via
  AuditAgent's daily hash-chain verification
- A **rule-engine fallback** that continues to meet compliance
  obligations when Bedrock or the local LLM is unavailable

What remains human work:

- Is the HMAC-key rotation cadence and policy appropriate?
- Are the Ops and Audit checkpoint configurations still aligned
  with the current regulatory environment? (New regulations often
  require checklist extension.)
- Do the α/β/γ persona prompts introduce any bias? The persona
  construction itself is a recurring review subject.
- Do recurring minority-report patterns signal an architectural
  redesign need?
- Is the Bedrock dependency managed as a third-party risk under
  SR 11-7 Pillar 4 (governance)?

The misreading "consensus runs itself, so committee reviews are
unnecessary" doesn't hold here either. Automation replaced the
*daily repetitive verification*; the *design and parameter
stewardship of the verification structure itself* stays with
risk-management professionals. If anything, the Minority Report
adds a new review item to the committee's standing list.

## Next

Ep 4 covers how the Korean AI Basic Act §35 FRIA (Fundamental
Rights Impact Assessment) lives in code: seven-dimension
assessment, five-year retention, and why it's kept as a *separate
class* from the EU AI Act Article 9 FRIAEvaluator — the temptation
to merge two schemes of different legal basis, and its risk.

Source:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §6
"Regulatory mapping"; implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial).
