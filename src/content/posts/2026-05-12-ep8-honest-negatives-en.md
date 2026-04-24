---
title: "[FinAI Build] Ep 8 — Honest Negative Results and What Comes Next"
date: 2026-05-12 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, adatt, gradsurgery, negative-results, financial-ai]
lang: en
excerpt: "Record from four months — how adaTT converged to a null effect at 13-task scale, why GradSurgery was rejected on VRAM overhead, Paper 3 WIP status, and real-data metrics pending after 2026-04-30. Why what did not work matters as much as what did."
series: three-months
part: 8
alt_lang: /2026/05/12/ep8-honest-negatives-ko/
source_url: https://doi.org/10.5281/zenodo.19621884
source_label: "Paper 1 (Zenodo DOI)"
---

*The final episode of "Building a Financial AI in Four Months".
Seven episodes covered what worked — the ALS-replacement motivation
(Ep 1), AI collaboration organization (Ep 2), guardrails (Ep 3),
the seven experts (Ep 4), data-integrity hunting (Ep 5), the bug
that contaminated an architectural conclusion (Ep 6), distillation
and serving (Ep 7). This one is the record of *what did not work*
over the same four months.*

## adaTT — a null effect at 13-task scale

The initial Paper 1 draft reported "adaTT drops AUC by -0.019
relative to PLE-only at the 13-task configuration". The reading at
the time was that adaTT *structurally fails* at that scale, with
two supporting explanations — combinatorial instability across 156
directed affinity pairs, and PLE's representation-level separation
being undone by adaTT's loss-level re-mixing.

After the uncertainty-weighting bug from Ep 6 was fixed, we reran
the same experiment. The adaTT on/off gap shifted from -0.019 to
-0.001. *Within single-seed noise*.

The new reading — adaTT doesn't hurt PLE at 13-task scale, and
doesn't meaningfully help it either. A *null effect*. In this data
× architecture combination, adaTT has *no reason to be used*.

## "Structural failure" and "null effect" are different

This distinction matters. The original "structural failure" claim
risked generalizing to *other scales as well*. "Null effect"
doesn't generalize. adaTT may work in 2-4 task small-scale settings
— the literature reports such cases — and it just happens that no
effect was observed in our 13-task heterogeneous environment.

Paper 1 v2's phrasing changed accordingly: "adaTT at 13-task
heterogeneous MTL: null effect, within single-seed noise. An
earlier-reported negative result was an artifact of a contaminating
training-environment bug, now corrected." Going forward, this kind
of result is reported with *effect size + confidence interval +
explicit limitation*.

## GradSurgery — rejected on VRAM overhead

Another adaTT alternative we tried — GradSurgery based on PCGrad
task-type projection. A different approach to resolving task
gradient conflict at 13-task scale.

Experimentally the AUC difference was, like adaTT, within noise.
So performance-wise no significant difference. *VRAM overhead*
told a different story.

GradSurgery stores per-task gradients every step and performs
pairwise projection. 13 tasks × 2M parameters = 26M gradient
copies + 156-pair projection memory. Within the RTX 4070's 12GB
VRAM this overhead cut batch size in half.

Same AUC, 2× slower training. Rejection.

The general principle: *if implementation complexity or hardware
cost offsets the performance gain, reject*. GradSurgery isn't a
bad algorithm — the ROI was simply negative under our constraints.

## Paper 3 (Loss Dynamics) — WIP

Ep 6 noted "Paper 3 (Loss Dynamics) was seeded here". That Paper 3
is currently in WIP.

The core question — *can the dynamics of loss functions themselves
determine architectural conclusions*. Ep 6's sigmoid-softmax
reversal is a strong single data point; Paper 3 explores the
phenomenon systematically — a matrix of loss weighting schemes
(uniform, uncertainty, GradNorm, DWA) × gate structures (softmax,
sigmoid, mixture-of-experts), tracking how architectural
conclusions shift.

Still at abstract-draft stage. Experimental design complete,
execution in progress, Zenodo upload targeted at Q3 2026. When
results are in, Study Thread will cover them in detail.

## From synthetic to production data — the validation window isn't open yet

All four months of design and experimentation ran on a public
benchmark (Santander Customer Transaction Prediction) and a 1M-row
synthetic dataset generated to mimic its distribution. No matter
how rigorously the two paths were validated, there is no guarantee
that the same architectural conclusions will hold under actual
production traffic. Whether that gap closes can only be confirmed
on real production data.

Production-traffic collection for per-task AUC, F1-macro, MAE, NDCG,
and fairness indicators across five protected attributes began on
2026-04-30. At the time of writing (mid-May), the early-May
metrics accumulated so far are well short of statistically
meaningful volume. A few days of data cannot decide whether the
sigmoid-softmax reversal (Ep 6) reproduces in production.

The line between what appears on this blog and what does not is
clear.
* **Publishable:** per-task AUC, DI per protected attribute, drift
  trends, incident statistics — aggregated technical indicators.
* **Not publishable:** customer segment distributions, individual
  customer characteristics, internal operational policies — any
  information that could identify customers or business specifics.

The latter sits under the institution's internal information-
protection rules; only aggregated technical numbers appear here.

Whether our synthetic-data conclusions reproduce in production
remains open. Whether the sigmoid-softmax reversal (Ep 6) holds on
real data, which of the seven heterogeneous experts wobbles first
under production distributions — these depend on months of
accumulated metrics. If they reproduce, the methodological claims
are supported by real data; if not, the divergence itself becomes
the starting point for the next paper. Either way, follow-ups will
appear in Study Thread or Commentary.

## Things tried in four months that did not make the record

A few more. Not included in the paper drafts but attempted within
the four months:

- **9-expert configuration trial.** Added Gaussian Process expert
  and Dropout Bayesian expert on top of the seven. No performance
  difference; VRAM tight. Kept at 7.
- **Multi-head attention expert.** Tried as an additional expert.
  Parameter explosion (2× the existing seven), negligible AUC
  delta. Rejected.
- **2-axis task-group decomposition.** Simplification attempt
  replacing 4 task groups with 2 (engagement-lifecycle vs.
  consumption-value). Some tasks improved, others regressed. Net
  zero. No complexity-reduction gain. Kept at 4.
- **Causal expert's NOTEARS replaced with DirectLiNGAM.** Linear
  vs. nonlinear causal-discovery comparison. At 13-task scale, both
  showed convergence instability. Eventually stabilized with
  NOTEARS + recon loss patch (see memory:
  project_causal_w_collapse_fix).

These don't make the paper — *recording every null or marginal
result inflates the paper and dilutes the core message*. The blog
has room for them. Another team exploring the same path learns
"already tried" from them.

## The common thread across what didn't work

Looking back at the negatives documented here and in earlier
episodes — three label leakages (Ep 5), the uncertainty-weighting
sign bug (Ep 6), the initial "adaTT structurally fails"
misreading, GradSurgery's VRAM trade-off, the 9-expert and
multi-head attention false starts, the softmax-sigmoid reversal,
the DirectLiNGAM vs NOTEARS indecision — every one of them
surfaced through the same development pattern:

1. An initial result (often inflated) looked like a conclusion.
2. Follow-on work proceeded on the assumption that the
   conclusion was settled.
3. A new observation — usually weeks later — forced a
   re-examination.
4. In the re-examination, *the original reasoning was still
   accessible in context*, letting the team ask "how does our
   earlier conclusion look now?" rather than "why did we do that
   again?"

Step 4 is where Claude Code's long-lived context turned out to
matter most. Without it, each of these re-examinations would
have been a partial reconstruction from notes, with high odds of
missing a premise. Several of these negatives would have stayed
hidden under "we decided that back in week 3" and never been
revisited.

This is not a story about AI replacing engineers. It is a story
about *AI making the re-investigation cost low enough that
engineers actually do it*. The three-person team's willingness
to revisit past conclusions — a habit of honesty — was the human
contribution. Claude Code made the cost of that honesty
affordable under four-month timelines.

## Why negative results matter

A large portion of ML's reproducibility problem comes from
*negative results not being published*. Unfavorable results get
thrown out as "uninteresting", and the same mistakes recur at
other teams. In our case, the uncertainty-weighting bug (Ep 6)
was *likely something others had already hit*, but there was no
public "sigmoid won then turned out to be the bug" record, so we
rediscovered it from scratch.

This blog is an attempt to break that pattern, even slightly. If
"four months of work that didn't pan out" saves another team
*weeks*, the cost-benefit of documenting negatives is high.

## Closing the series

Across eight episodes: motivation for replacing the ALS system,
derivation of the 7-expert architecture, AI collaboration
methodology, the guardrail framework, data-integrity hunting, the
training-environment bug, distillation and serving, and this
episode's negative results.

The through-line — *small Korean financial-services teams (three
to five people) can put contemporary AI architecture into
production*. Without a large GPU cluster, without a dedicated MRM
department, with Claude Code as the partner. In exchange, every
step needs its reasoning — why this architecture, why this tool,
why this verification step.

Four months out, *more mistakes are visible than successes*. The
uncertainty-weighting sign bug, the initial "adaTT structurally
fails" misreading, v1 synthetic data's GAN limitations, the three
chained label leakages. Every mistake tightened the project's
methodology by a notch. Expertise isn't avoiding mistakes — it's
being able to detect them. Ep 6's line captures this whole
series.

## What comes next

FinAI Build ends here at eight episodes. Upcoming work on the blog:

- **Study Thread** — already in progress (6 PLE + 4 adaTT
  episodes). Next up in sequence: Causal OT, TDA, Temporal
  Ensemble, Economics Expert, etc.
- **Commentary** — irregular posts on specific incidents,
  regulatory revisions, paper reviews. No first episode yet.
- **Real-data updates** — interpretation of metrics accumulated
  after 2026-04-30 will land in Study Thread or a separate series.

Thanks for reading this far. Each episode was written to stand
alone, so return by interest. Questions or corrections via
[Seonkyu Jeong's ORCID](https://orcid.org/0009-0005-3291-9112).

Source:
[Paper 1 (Zenodo)](https://doi.org/10.5281/zenodo.19621884) +
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052).
Code: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).
