---
title: "[FinAI Build] Ep 7 — Distillation and Serving: PLE → LightGBM → Lambda + 5 Bedrock Agents"
date: 2026-05-08 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, distillation, serving, lambda, bedrock]
lang: en
excerpt: "Teacher is PLE, student is per-task LightGBM, serving is AWS Lambda. Why this combination, what happens when teacher-student fidelity fails, and the role division across the 5-agent Bedrock pipeline."
series: three-months
part: 7
alt_lang: /2026/05/08/ep7-distillation-serving-ko/
next_title: "Ep 8 — Honest Negative Results and What Comes Next"
next_desc: "The adaTT null effect, GradSurgery rejection, Paper 3 WIP, and the real-data metrics pending after 2026-04-30 — a record of *what did not work* across three months."
next_status: draft
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*Part 7 of "Building a Financial AI in Three Months". Eps 4-6
settled the model architecture; this episode is about the path
the model takes *to actually reach a customer* — distillation and
serving. How a three-person team ends up serving real-time
financial recommendations on AWS Lambda.*

## Teacher is PLE, student is LightGBM

The training-time model is PLE with seven heterogeneous experts,
CGC gating, and thirteen task towers. Under 2M parameters (Ep 4),
so it's light — but not right for serving directly. Three problems.

**Problem 1 — PyTorch inference runtime.** Cold-starting torch on
Lambda costs 2-3 seconds. Unacceptable for a single-request path.

**Problem 2 — Memory footprint.** Seven experts + CGC gate + task
towers = ~150MB fully loaded. Lambda's concurrent-execution count
gets capped by memory budget.

**Problem 3 — Interpretability surface.** When a customer or
regulator asks "why this recommendation?", answers need to be
immediate. LightGBM has per-tree contribution decomposition built
in; PLE's expert gate weights are interpretable but the LGBM
feature attribution path is simpler.

The common answer to all three — **distill per-task into LightGBM**.
The PLE teacher generates soft probabilities per task; the LGBM
student is trained with those as its target. Result: cold start
under 300ms, ~30MB memory, feature attribution built in.

## Teacher-student fidelity floor

The key thing in distillation is *fidelity* — how close the student
sits to the teacher. This is the fidelity floor Ep 2 referenced in
the champion-challenger gate ("reject before competition if fidelity
floor fails").

Measured as per-task student-teacher KL divergence. Threshold set
in config (`distillation.fidelity_floor`, default 0.20). If any of
the 13 tasks exceeds this threshold, the entire challenger is
rejected.

Why the threshold matters — *even a student with strong training
metrics, if it learned a different function from the teacher, is a
different model*. In conventional knowledge-distillation research,
a student occasionally *outperforms* the teacher, and this is sometimes
celebrated. In the financial-recommendation context, it's a warning
sign. The teacher (PLE) was designed under regulatory, fairness,
and interpretability constraints; if the student *bypasses* those
constraints for better performance, the original design intent is
broken.

So *the student's role is to faithfully replicate the teacher*, not
outperform it. The fidelity floor enforces that principle
structurally.

## Teacher threshold gating — 2× random baseline

Not every task is distillable. When the teacher itself performs
poorly, the student only learns the teacher's noise. A
`distillation.teacher_threshold` (default 2× random baseline) is
applied: below this threshold, the student LGBM is trained on
*hard labels directly*, skipping distillation (an MRM safety
guard).

For a binary task with random-baseline AUC of 0.5, the teacher
needs AUC above 1.0 to qualify. Below that, the polygon is hard-label
training instead. This is how we avoid "a student that mimics
even the teacher's noise" reaching production.

## Three-layer serving fallback

In operation not everything goes right — distillation failure,
input anomalies, model-file corruption. A three-layer fallback
guarantees no service interruption:

- **Layer 1** — PLE → LGBM distilled model (the normal path, 99%+
  of traffic).
- **Layer 2** — LGBM trained directly on labels (used for tasks
  that failed the fidelity floor).
- **Layer 3** — financial-DNA rules (when the full model pipeline
  fails) — conservative recommendations based on age band, asset
  size, and transaction history.

Ep 5 also mentioned Layer 4 (human fallback) but that's opt-in.
Layers 1-3 are the default.

This layering guarantees the SLA requirement ("99.9% availability")
*at the architecture level*. No single point of failure exists.

## AWS Lambda + 5 Bedrock agents

Full serving stack:

```
Client (bank branch system)
  ↓
API Gateway
  ↓
Lambda (Python 3.11, 1GB memory)
  ↓
  ├─ LightGBM model load (three-layer fallback)
  ├─ Feature Selector agent (Bedrock Sonnet)
  ├─ Reason Generator agent (Bedrock Sonnet)
  ├─ Safety Gate agent (Bedrock Sonnet)
  └─ Return prediction + Ep 3 audit log entry

Async paths:
  OpsAgent (Bedrock Sonnet) — interprets CloudWatch logs
  AuditAgent (Bedrock Haiku) — nightly hash-chain verification
```

The division of labor across the five agents:

**1. Feature Selector (Sonnet)** — Selects from the LightGBM's
raw feature attributions those *worth explaining to a customer*.
"Three overseas online-purchase transactions recently" is meaningful;
"feature #247" is not. Converts the model's internal raw attributions
into a business-meaningful set.

**2. Reason Generator (Sonnet)** — Rewrites the selected features
into *natural-language honorific Korean a bank branch employee
can read to a customer*. "Your recent spending pattern shows
increasing foreign-currency payments, so we recommend a
foreign-currency deposit product." Not a template — tone is
adjusted for the customer context.

**3. Safety Gate (Sonnet)** — Validates the generated reason
against five criteria: (a) regulatory (no restricted-phrase list
violations), (b) suitability (within the customer's risk
tolerance), (c) no hallucination (matches actual feature values),
(d) appropriate tone (no high-pressure or exaggerated language),
(e) factuality (product name and terms match the internal
database). The last gate before leaving the Lambda handler.

**4. OpsAgent (Sonnet)** — Async, triggered nightly or by on-call.
Interprets drift-monitor and fairness-monitor output as a summary
for SRE / MLOps / business-team consumption. Drafts incident
tickets when a kill switch fires.

**5. AuditAgent (Haiku)** — mentioned in Ep 3. Verifies the HMAC
hash chain nightly. The computation is light enough that Haiku
suffices.

## Ops vs. Audit separation — why two agents

Originally we tried to merge OpsAgent and AuditAgent — they both
"read logs", after all.

What kept them separate — *different audiences*. OpsAgent's output
is consumed by SRE / MLOps / business teams — "performance drift
is approaching the threshold", an operational view. AuditAgent's
output is consumed by compliance / risk teams — "hash chain
verification failed at entry N", an audit view.

More critically, the *trust boundary* is different. OpsAgent
*reads and interprets* logs; AuditAgent *does not trust logs and
re-verifies them* (Ep 3). That difference is embedded in each
agent's prompt design. Merging them dilutes one of the two roles.

## How this code actually got written

The serving stack above reads like something a team of AWS
specialists shipped. In our case the specialist work happened in
Claude Code sessions.

The Lambda `handler.py` went through roughly thirty iterations.
Version 1 was a 60-line Claude Code draft that loaded a
LightGBM model and returned predictions. Version 30 is 400 lines
with three-layer fallback, audit logging, Bedrock agent
orchestration, and deterministic cold-start warming. The
intermediate versions were driven by real production failure
modes — a timeout here, an IAM permission denial there, an
edge-case schema change — each of which came into the codebase
as one Claude Code conversation ending in a PR.

The three-layer fallback was not designed upfront. Layer 1 was
all we had for the first two weeks. Layer 2 came after a
distillation failure on `task_churn` (fidelity floor violation,
Ep 2's rejection path) — we needed a route to still serve that
task without the distilled model. Layer 3 came after a broader
incident where a Phase 0 schema change propagated incorrectly and
the LGBM load itself failed for an afternoon; a pure-rules
fallback meant customers still got recommendations, just
conservative ones. Each layer is a fossil of a specific past
failure.

The 5-agent orchestration similarly grew incrementally. Feature
Selector and Reason Generator came first, as a direct
requirement from Paper 2 (explain recommendations to customers).
Safety Gate was added after a pre-launch review where an
engineer (with AI help) surfaced 40 edge cases in which the
Reason Generator could produce factually wrong text; the Safety
Gate rules came from that review. OpsAgent came after a 3am
alert that took 40 minutes to triage manually. AuditAgent came
last, tied to Ep 3's chain-of-custody requirement.

None of this was linear planning. Each addition resolved a
specific observed problem, and Claude Code wrote the
implementation while Opus argued the design. That's the pattern
— observation → Opus dialogue → Claude Code implementation →
production → next observation — that produced this stack over
three months of 3-person work.

## Lambda cost — why serverless is the answer

Serving a model 24/7 on a dedicated server with a GPU is infeasible
for a three-person team's budget. Lambda + LightGBM draws a
different picture:

- Per-request billing (near-zero cost when RPS is low)
- Cold start 300ms (LightGBM load) — within most financial
  recommendation SLAs
- Concurrent-execution limit adjustable via config, bounding cost

Estimated cost (at 1M recommendations/month): Lambda $15 + API
Gateway $3 + S3 / CloudWatch $10 = under $30/month. Roughly
1/100th of a dedicated server.

This is why *consumer GPU + serverless* is the combination that
makes this approach accessible to small Korean financial-services
teams. Training once, serving cheaply forever. Production AI
without large infrastructure.

## Next

Ep 8 (FinAI Build finale) — a record of *what did not work* across
the three months. The adaTT null effect at 13-task scale,
GradSurgery rejected on VRAM grounds, Paper 3 (Loss Dynamics) in
WIP, and the current state of awaiting real-data metrics after
2026-04-30. Closing with why honest negative results are an
important part of the project.

Source:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) §3
"Serving architecture" + §8 "Agent design". Implementation lives
in `core/distillation/`, `aws/lambda/handler.py`, and
`core/agents/`.
