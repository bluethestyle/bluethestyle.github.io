---
title: "[FinAI Build] Ep 5 — The Data Integrity Hunt"
date: 2026-05-01 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, data-integrity, leakage, financial-ai]
lang: en
excerpt: "Before any architecture debate — three chained label-leakage detections, the deterministic-leakage rationale behind the 18→13 task reduction, and the self-replicating features that surfaced across synthetic-data iterations v2→v3→v4."
series: three-months
part: 5
alt_lang: /2026/05/01/ep5-data-integrity-hunt-ko/
next_title: "Ep 6 — The Bug That Overwhelmed All Architectural Decisions"
next_desc: "A fix to uncertainty-weighting flipped the softmax-vs-sigmoid result. The methodological lesson: training bugs can corrupt architectural conclusions."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf
source_label: "Development Story (EN, PDF) §9"
---

*Part 5 of "Building a Financial AI in Three Months". Through Ep 4
we covered which architectures got picked and why. This episode is
about confirming the *inputs* are correct — the unavoidable boring
work that, after three iterations, settled into something useful.*

## First leakage — duplicate has_nba column

When ablation v1 finished, the AUC was suspiciously high. The
conservatively-set baseline was 0.68 and a complex expert
configuration returned 0.87. A gap that wide is rarely an
algorithm story; it's a data story.

The culprit was the `has_nba` column. Next Best Action (NBA) is
one of the tasks we wanted to predict — "what product should we
recommend next to this customer". The raw CSV already contained a
field named `has_nba` that was being used as a feature. Part of
the answer to the target was being fed in as an input feature.

Removing it dropped AUC from 0.87 to 0.71. The moment of
discovery feels like *disappointment*, but in truth it's *relief*.
That's the point at which we can measure what the algorithm
actually does. This is the value of leakage detection — it lowers
the numbers, but the numbers become real.

## Second leakage — ground-truth glob ordering

After the `has_nba` fix, in the *same session*, Claude Code dug
into the next problem (the case Ep 2 referenced for "chained
detection within one session"). Ground-truth files were being
loaded with a glob in alphabetical order, and that order
correlated accidentally with the train/val split boundary.

Concretely — files with earlier customer IDs had fewer new
customers; later files had more recent signups. That distribution
difference created systematic bias in the validation set. The
model learned spurious rules like "recent signups buy product ABC"
and showed inflated validation accuracy.

Fix: replace the glob with an *explicit ID-based shuffle*. Once
the validation distribution matched train's, the AUC adjusted
again (0.71 → 0.66). Again, the lower number is *real*.

## Third leakage — generator label input

At the end of the same session came the third. Some of our
generators (feature-construction modules) were accepting *labels
as input during training* to build piecewise features. A generator
that outputs "probability this customer belongs to segment X" was
slyly referencing the validation label in its probability
computation.

Root cause: separation of concerns broke at one point between the
adapter and the generator (why CLAUDE.md §1.2 exists, covered in
Ep 3). The generator wasn't explicitly dropping the label column
from its dataframe; it was selecting only the columns it needed,
and the label accidentally remained in that select list.

AUC 0.66 → 0.62. Over three chained detections, a total of 0.25
AUC was revealed to be a mirage. One session, a few days of work.
If each leakage had been debugged separately, it would have taken
weeks — Claude Code's 1M-token context keeping "the context of
prior fixes alive as we pursued the next suspicion" was decisive.

## The 18→13 task reduction — deterministic leakage

Even after the three-chained leakage fix, some tasks stayed
suspicious. `income_tier`, `tenure_stage`, `spend_level`,
`engagement_score`, and similar tasks showed *0.99+ AUC across
every configuration* in the ablation. 0.99 isn't an amazing model;
it means *the task definition is itself a deterministic transform
of features*.

And it was. `income_tier` is `income` binned into five buckets;
`tenure_stage` is `tenure_months` in six-step buckets. The model
can *perfectly reconstruct* the label from the input, so AUC
saturates near 1.0. Leaving these tasks in the multi-task mix
meant (a) their loss was anomalously small, contributing nothing
to learning compared to other tasks, and (b) the transfer-learning
evaluation metric was contaminated.

Decision: drop the five deterministic tasks. 18 → 13. CLAUDE.md
§1.3 then added the clause "labels derived as simple transforms of
features are not tasks". Subsequent ablations excluded this class
from the start.

## Synthetic-data iteration v2 → v3 → v4

We ran parallel experiments on a million-customer synthetic
dataset. A model trained on initial v1 synthetic data didn't
transfer to real data — synthetic AUC 0.82, real-data AUC 0.54.

v2 improved feature-distribution matching. Still no transfer.
v3 added inter-feature correlation matching. Partial improvement.
v4 added *time-series dependency* matching. Transfer finally
succeeded.

The progression v1 → v4 was increasingly sophisticated statistical
matching, but the core problem was one thing — the synthetic
generator's *tendency to replicate itself*. v1 was GAN-based, and
a GAN learns the "easy parts" of real data and misses the "hard
parts". What the model learned from synthetic was overfitting to
"easy patterns of reality".

v4 explicitly encoded time-series dependency with a state-space
model, forcing the "hard parts" into the distribution. Real-data
transfer worked after that.

## Why this had to be solved first

Architecture debates — is HGCN better or is LightGCN better — only
make sense with clean inputs. If any of the above had remained, the
ablation results would measure not "algorithm differences" but
"which algorithm exploits the leakage more aggressively". A
shortcut to writing results that don't reproduce on real data, in
paper form.

*Data integrity hunts are unglamorous work.* There's no new
algorithm to try; the job is to *doubt* existing data, and visible
progress stays zero for days. With a three-person team under time
pressure, the hardest fight is against the pull of "let's see
results fast". What sustains the discipline here is *the memory of
how much AUC prior leakage detections destroyed*. Having watched
0.87 → 0.62 fall three times, "what else is hiding" becomes the
default stance.

## The methodological upshot

This experience added *five leakage-prevention clauses* to
CLAUDE.md:

1. Scaler fits only on the TRAIN split.
2. Temporal split requires `gap_days` (minimum 7).
3. Sequence data's last timestep must not overlap with the label —
   verified explicitly.
4. `LeakageValidator` must run before training.
5. Labels derived as simple transforms of features are not tasks.

This is why Ep 3's CLAUDE.md is a *constitution*: you don't have
to remember these clauses for every new experiment, and new team
members get them applied from day one.

## Next

Ep 6 covers a problem that persisted even after data integrity was
clean — an implementation bug in uncertainty weighting that turned
out to have been contaminating the sigmoid-vs-softmax architectural
conclusion. Fixing the bug flipped the result. The methodological
lesson: *training bugs can corrupt architectural conclusions*.

Source:
[Development Story §9 "Data Integrity Audit"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf).
