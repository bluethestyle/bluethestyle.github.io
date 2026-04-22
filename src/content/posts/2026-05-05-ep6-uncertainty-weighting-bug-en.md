---
title: "[FinAI Build] Ep 6 — The Bug That Overwhelmed All Architectural Decisions"
date: 2026-05-05 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, debugging, methodology, financial-ai]
lang: en
excerpt: "For weeks sigmoid gating seemed to beat softmax. Fixing an uncertainty-weighting implementation bug flipped the result. A case study in how a training-environment bug contaminates architectural conclusions."
series: three-months
part: 6
alt_lang: /2026/05/05/ep6-uncertainty-weighting-bug-ko/
next_title: "Ep 7 — Distillation and Serving: PLE → LightGBM → Lambda + 5 Bedrock Agents"
next_desc: "Teacher-student fidelity, why LightGBM for serving, the serverless cost profile, and the 5-agent Bedrock pipeline composition."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf
source_label: "Development Story (EN, PDF) §11"
---

*Part 6 of "Building a Financial AI in Three Months". What
survived data-integrity cleanup from Ep 5 — the training
environment itself harboring a bug that *silently* contaminated
architectural conclusions we made for weeks. This episode is the
story of its discovery and the methodological lesson.*

## What we thought was the conclusion

Ep 2 briefly touched the sigmoid-gate adoption story — PLE
val_loss failing to converge → dialogue with Opus → finding the
NeurIPS 2024 sigmoid gate paper → implementation. Experiments
agreed: sigmoid consistently beat softmax.

Consistent results are strong evidence. Five ablation runs,
different seeds, different data splits — all had sigmoid ahead by
0.02-0.04 on NDCG and F1-macro. At this point we *documented the
conclusion* and used it as a premise for subsequent architectural
decisions (adaTT design, among others).

Conclusions beget follow-on work. A Paper 1 draft said "sigmoid
gating is appropriate for heterogeneous expert architecture". The
adaTT design proceeded on sigmoid-gate assumptions. "This is
settled" propagated into several downstream decisions.

## A few weeks later, the uncertainty weighting bug

While implementing adaTT, Engineer 2's team reviewed the
uncertainty weighting (Kendall et al., 2018) code and found a bug.
The bug itself is small — the formula multiplying task loss by a
learned $\log \sigma^2$ had a sign flipped. The correct form is:

$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

The implementation was:

$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t - \log \sigma_t$$

One sign. `+ log_sigma` became `- log_sigma`. When that flips, the
regularization term pushes in the opposite direction — training
drives $\sigma_t$ *down* instead of *up*. Result: for tasks whose
loss is intrinsically small (binary classification), $\sigma_t$
collapsed to near-zero, giving those tasks effective weights that
overwhelmed the others.

So uncertainty weighting, for weeks, was *effectively over-weighting
the binary classification tasks*.

## The result flipped after the fix

We fixed the bug and reran the five ablation runs. Result —
*softmax beat sigmoid on NDCG*.

One run, 0.02 ahead. Two runs, 0.03. Three runs, 0.01. The other
two runs were close. The direction fully reversed.

First suspicion: measurement noise. Ran ten more with different
seeds. Consistent direction. The "consistency" that made sigmoid
look superior was replaced by a *different* consistency.

## Why — the root cause

Revisiting the logic after the fact, it was clean.

**Under the broken uncertainty weighting:** the thirteen tasks
were not effectively equal-weighted. Binary classification
gradients (of which there are more in count) overwhelmed
multiclass and regression gradients. Under these conditions
softmax's competitive routing concentrated expert capacity on the
binary side, starving other task types. Sigmoid's
non-competitive routing kept all experts active, *accidentally
functioning as a firewall* preventing that starvation.

**Under correct uncertainty weighting:** multiclass and regression
gradients recovered their intended magnitudes. Softmax's
competitive routing then acted as *a structural barrier* between
task-type gradient flows, forcing expert specialization. Sigmoid,
conversely, no longer served as a firewall — it mixed gradients
freely, leaving each task type less specialized.

So the *direction* of the result depends on the training condition.
"Sigmoid is better" was a *valid adaptation* to a broken environment,
not an architectural superiority.

## Homogeneous MTL vs. heterogeneous MTL — the literature trap

The NeurIPS 2024 sigmoid gate paper showed sigmoid's advantage in a
2-4 task, same-task-type setting. Under the *homogeneous MTL*
assumption of that paper, competitive softmax does induce collapse
among structurally similar experts, and sigmoid is genuinely better.

Our project is *13 tasks, 3 task types (binary · multiclass ·
regression)* — heterogeneous MTL. In this regime competitive
routing doesn't induce collapse; it functions as a *structural
barrier* against gradient contamination across task types. The
literature result does not transfer. A boundary condition matters.

The lesson is simple — *if you don't check that the reference
paper's experimental conditions match your own project, you end
up porting correct conclusions into the wrong environment*. In our
case a training-environment bug compounded the contamination.

## What enabled the discovery

The arc of detection is itself the cleanest example of what Ep 2
called "Claude Code non-substitutability". The context from weeks
earlier — why that paper was searched for, what the experimental
numbers were, which downstream decisions had taken the result as a
premise — was *still accessible* when the new evidence (the
post-uncertainty-fix reversal) arrived. That triggered an
immediate "what does our earlier conclusion look like now" review.
Had a new session forced reconstruction of "why did we pick
sigmoid" from scratch, the re-investigation might never have
started.

## The methodological upshot

What this episode left in our team's methodology:

**1. Every architectural conclusion now carries a "training
conditions stable?" check.** Before documenting a "conclusion", we
verify loss weighting, scaler state, label alignment, and
scheduler configuration. The checklist was added to CLAUDE.md.

**2. Paper draft revisions were significant.** Paper 1's "sigmoid
gating is appropriate for heterogeneous expert architecture"
section was rewritten to explicitly note "the result can flip by
boundary condition". Paper 3 (Loss Dynamics) was seeded here.

**3. Re-interpretation of the adaTT null result.** Ep 8 will cover
this in detail, but the initially-reported "adaTT at 13 tasks
hurts PLE by -0.019 AUC" result was also measured under this bug.
After the fix, the adaTT on/off gap was -0.001 — within noise. So
adaTT wasn't "structurally failing"; the training environment was
contaminated.

**4. Expertise is not not-making-mistakes; it is being able to
detect them.** This line landed in the project's CLAUDE.md §1.
Missing a sign bug for weeks isn't a lack of expertise. The
*re-investigation flow* afterwards is.

## Open questions

- Were other "consistent results" also byproducts of training
  environment bugs? — Other Paper 1 results required re-verification,
  and significant portions were revised.
- Will the same reversal happen on real data? — Awaiting real-data
  metrics after 2026-04-30. If it reproduces, the methodological
  lesson strengthens; if not, that itself opens new questions.
- Is the uncertainty-weighting implementation now robust? — A
  sign-aware unit test plus runtime $\sigma$ monitoring was added
  to CI. The same bug does not return.

## Next

Ep 7 covers the work after the architecture stabilized —
distillation (PLE teacher → LGBM student) and serving (AWS Lambda
+ 5-agent Bedrock pipeline). Why LGBM for distillation, the
serverless cost structure, and the division of labor across the
five agents (Feature Selector · Reason Generator · Safety Gate
· OpsAgent · AuditAgent).

Source:
[Development Story §11 "The Bug That Overwhelmed All Architectural Decisions"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf).
