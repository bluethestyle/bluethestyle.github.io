---
title: "[FinAI Build] Ep 4 — The Seven Experts: Importing Structural Isomorphism Across Eleven Disciplines"
date: 2026-04-28 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, architecture, ple, expert-pool, financial-ai]
lang: en
excerpt: "Why seven experts, why these seven. The cross-disciplinary scan with Gemini surfaced eleven fields; the feasibility review with Opus narrowed to DeepFM, Temporal Ensemble, HGCN, PersLay, Causal, LightGCN, and Optimal Transport."
series: three-months
part: 4
alt_lang: /2026/04/28/ep4-seven-experts-ko/
next_title: "Ep 5 — The Data Integrity Hunt"
next_desc: "Three chained label-leakage detections, the 18→13 task reduction, and synthetic-data iterations v2→v3→v4 — what had to be solved before the architecture could even be measured."
next_status: published
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/expert_details_en.pdf
source_label: "Expert Details (EN, PDF)"
---

*Part 4 of "Building a Financial AI in Three Months". Ep 1 named
"heterogeneous expert networks" as a decision that emerged from the
PLE reframe. Ep 2 named "structural isomorphism" as a concept that
surfaced in Gemini dialogue. This episode is how those abstract
terms landed on seven concrete networks.*

## Why seven

The number wasn't decided first. The question was *how many
sufficiently heterogeneous mathematical perspectives* were needed,
and the answer came out to seven.

There were more than ten initial candidates. The cross-disciplinary
scan with Gemini surfaced eleven fields — hyperbolic geometry,
chemical reaction kinetics, SIR epidemic models, optimal transport,
persistent homology, structural causal inference, graph theory,
state-space time series, dot-product/attention, factorization
machines, and Gaussian mixtures.

Seven passed two constraints: *structurally different from what's
already in the pool*, and *fits the VRAM budget when seven are
stacked together*. Four were rejected:

- **Mamba alone** — too much memory on 17-month sequences. Absorbed
  into Temporal Ensemble instead.
- **Large Transformer experts** — can't stack seven on 12GB VRAM.
  The "brute force with parameters" path was architecturally
  blocked.
- **Gaussian Mixture Model experts** — structurally redundant with
  Causal + OT, which already cover distribution-comparison views.
- **Plain MLP ensembles** — "heterogeneous" by initialization
  only, which collapses to mean. Not actually heterogeneous.

What the remaining seven see, in order:

## The seven, each fills a gap

**1. DeepFM — feature interactions.** The most conventional seat.
It catches 2-way and higher-order feature interactions via
factorization machines plus a deep network. This is the *baseline*.
It's how we measure novelty elsewhere — if a task is dominated by
simple interactions, DeepFM wins, and that's the comparison point.
Without this seat, every result becomes "novel" and there's no
reference.

**2. Temporal Ensemble (Mamba + LNN + PatchTST) — temporal
dynamics.** Receives the 17-month customer behavior sequence. Not
a single model but an ensemble of three temporal architectures:
Mamba for long-range dependency, LNN (Liquid Neural Network) for
nonlinear adaptation, PatchTST for periodic pattern capture.
Each sees a different time structure; an HMM Triple-Mode gate
distributes weight per regime.

**3. HGCN — hierarchical structure (hyperbolic space).** MCC
(Merchant Category Code) is a category tree — food > restaurants >
Korean cuisine, for instance. Embedding trees in Euclidean space
distorts distances. In the Poincaré ball model's hyperbolic space,
tree embeddings are geometrically natural. The customer's
spending-category hierarchy is interpreted in this space.

**4. PersLay / TDA — topological shape.** Persistent homology
quantifies the *shape* of spending-time-and-amount distributions
as a number. A Vietoris-Rips complex is built, the persistence
diagram is embedded through a five-block multi-beta architecture.
Answers questions like "does this customer's spending pattern take
a similar shape each month, or are there irregular bursts?"

**5. Causal — structural causal inference.** NOTEARS-based DAG
learning. Discovers causal relationships among features from data.
Where other experts compose correlations, Causal answers "what
happens to Y if we intervene on X?" The only expert that makes
Paper 2's Counterfactual Champion-Challenger analysis possible.

**6. LightGCN — user-item bipartite graph.** Replaces the
collaborative filtering done by the ALS recommender with graph
convolutional operations. Without this seat, the guarantee
"matches previous system performance" weakens — so LightGCN also
serves as a *regression-to-previous-baseline* safety net.

**7. Optimal Transport — distribution comparison.** Sinkhorn
divergence between the probability distributions of two customers
(or customer segments). Where Causal deals with causal graphs, OT
treats distributions themselves as a metric space. Provides
independent signal for segment-change detection, drift measurement,
and fairness computations.

## The order of arrival matters separately

The list above is the final set, but the entries didn't arrive in
that order.

DeepFM and Temporal were forced in by the ALS-replacement requirement
from the start. LightGCN followed as baseline insurance. Those
three are the standard set from recommender-systems literature.

HGCN, PersLay, Causal, and OT arriving together is what set the
heterogeneous character of the expert pool. Dialogue with Gemini
asked "which aspect of customer behavior is structurally isomorphic
to chemical kinetics?" kinds of questions, and the four perspectives
— hierarchy, topology, causality, distribution — each turned out to
be independently meaningful.

Feasibility review with Opus checked each candidate. Does HGCN
actually work on the real MCC tree structure? What filter turns a
17-month sequence into a persistence diagram through PersLay? Does
NOTEARS converge in 349-dim space for Causal? Only those answering
YES stayed.

Each feasibility check was not just a conversation — it was a
Claude Code session. Opus would argue a candidate's theoretical
suitability; Claude Code would then write a minimal prototype and
run it on synthetic data that afternoon. HGCN's feasibility was
decided by a two-hour session implementing a Poincaré ball
embedding on a 27D slice of MCC hierarchy and watching the loss
curve converge; PersLay took three days of iteration on the
filtration function before the persistence diagram was stable
enough to feed an MLP. Standalone Mamba was tested on 17-month
sequences and rejected *on memory grounds* — the actual OOM error
in the Claude Code terminal settled the argument, leading to its
absorption into Temporal Ensemble.

This iteration pattern — *hypothesize with Opus, prototype with
Claude Code, decide with the numbers* — is how a three-person team
validated seven expert architectures in about six weeks. Each
prototype was often under 300 lines of throwaway code, but the
throughput of "idea → test → verdict" was what made narrowing from
eleven to seven feasible at this team size.

## Isn't seven overkill

That was the ablation-defining question. Over v12 iterations of
the 23-scenario ablation, every single-expert-removed configuration
was compared.

The result was illuminating. *Removing any single expert
measurably hurt AUC*. Removing HGCN hurt the tasks where MCC
hierarchy is central (spending_category, merchant_affinity).
Removing PersLay hurt tasks sensitive to spending-burst patterns
(consumption_cycle). Removing OT hurt segment-based tasks.

So the seven are *complementary*, not redundant. No single
mathematical perspective can carry all thirteen heterogeneous
tasks. That's the interpretation of the ablation result — the
transition from "heterogeneous experts is a paper idea" to
"it actually works".

## Why this structure fits the Korean financial-AI constraint

Each of the seven experts is lightweight (20k–200k parameters).
Seven combined is under 2M. That's how the whole ensemble fits
into 12GB of VRAM on an RTX 4070. Stacking seven Transformer
experts would not have fit two of them, let alone seven.

*Lightweight + structurally heterogeneous* is the condition that
makes this approach accessible to mid-size Korean financial-services
teams. Without a large GPU cluster, it's possible to build a model
where *domain knowledge is embedded in the architecture itself*
rather than brute-forced through parameters. Instead of porting
the "large-scale MoE" paradigm from international literature, the
constraint translates into a design opportunity.

## Next

Ep 5 covers what had to be solved *before* this architecture could
be measured at all — data integrity. Three chained label-leakage
detections, the background to the 18→13 task reduction (deterministic
leakage), synthetic-data iterations v2→v3→v4. Confirming *inputs
are correct* before arguing about architectures.

Source material:
[Expert Details (EN, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/expert_details_en.pdf)
+ [Development Story §5 "Design Philosophy"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf).
