---
title: "[Study Thread] Ep 1 — PLE Foundations: From Shared-Bottom to CGC"
date: 2026-04-19 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, mmoe, cgc, mtl, architecture]
lang: en
series: study-thread
part: 1
alt_lang: /2026/04/19/study-ep1-ple-foundations-ko/
next_title: "Ep 2 — Heterogeneous Expert Basket Design"
next_desc: "Why we replaced PLE's identical expert pool with eight structurally different domain experts. The FeatureRouter pattern that slices a ~349D input into per-expert feature subsets, the criteria for picking experts, and the design philosophy behind the pool/basket pattern itself."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/tech_ref_ple_adatt_en.pdf
source_label: "PLE + adaTT Tech Reference §1 (EN, PDF)"
---

*Part 1 of "Study Thread" — a parallel English/Korean series
summarizing the papers and math foundations behind the PLE
architecture used in this project.*

## Why care

We need to predict 13 tasks from one customer representation —
churn signal, next best action, MCC trends, six product-acquisition
probabilities, and so on. Multi-task learning (MTL) is the obvious
framing, but the architecture choice inside MTL is not obvious.
Shared-Bottom, MMoE, and PLE are three waves of MTL architectures,
each fixing a specific failure mode of the previous one.
Understanding *why PLE won* requires seeing what the earlier two
failed at. This episode walks the three waves and ends at the CGC
math we actually run in production.

## The case for multi-task learning

A single shared representation must serve many heads. If the
shared trunk overfits to one task it stops being useful to the
others, so only patterns that benefit *every* task survive. That
is the inter-task regularization argument, and it is the entire
reason MTL exists as a regime instead of just training $K$ models.
The total loss is a weighted sum:

$$\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k(f_k(\mathbf{h}_{shared}(\mathbf{x})))$$

The hard part is not the equation. The hard part is what
$\mathbf{h}_{shared}$ looks like as an architecture, because the
naive answer (one trunk, $K$ heads) breaks the moment your tasks
stop agreeing about what the representation should encode.

## Three waves

### Shared-Bottom (Caruana, 1997)

All tasks share a single trunk and then branch into per-task
heads:

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

Implementation is simple, parameter count is minimal. The failure
mode is **negative transfer**: when two tasks want the shared
trunk to encode different things, gradient updates from one task
actively hurt the other. With low-correlation tasks this gets
severe enough that you would have done better training each task
in isolation.

### MMoE (Ma et al., KDD 2018)

MMoE replaces the single trunk with $N$ experts of identical
structure. A per-task softmax gate decides which experts to mix:

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot f_i^{expert}(\mathbf{x}), \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x})$$

Each task is now allowed a different mix of experts, so two tasks
that disagree can route around each other. Better than
Shared-Bottom in practice. The new failure mode is **expert
collapse**: nothing forces gates from different tasks to actually
diverge, and at the typical learning rates they often converge to
the same expert. When that happens you have effectively rebuilt
Shared-Bottom with extra parameters.

### PLE (Tang et al., RecSys 2020)

PLE explicitly separates experts into **Shared Experts**
$\mathcal{E}^s$ and **Task-specific Experts** $\mathcal{E}^k$.
Each task's representation is a CGC-gated combination of
shared-pool outputs *and* its own task-specific outputs:

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

The structural separation is the whole point. Task-specific
experts learn patterns that only one task cares about without
interfering with the others. Shared experts learn the
cross-task signal. Expert collapse is much harder to fall into
because the roles are no longer symmetric. Stacking multiple PLE
layers gives you progressive extraction — low-level shared
features at the bottom, increasingly specialized features at the
top.

| | Shared-Bottom | MMoE | PLE |
|---|---|---|---|
| Expert structure | Single shared trunk | $N$ experts fully shared | Shared + Task-specific separated |
| Gating | None | Per-task softmax gate | CGC: shared + task expert combined |
| Negative transfer | High | Medium (expert collapse) | Low (explicit separation) |
| Expert collapse | N/A | High | Low |

## The CGC math

CGC stands for Customized Gate Control. It is the gate at the
heart of PLE — the thing that decides, per task, how much of each
expert's output to use.

This implementation has two CGC variants with different output
shapes.

**CGCLayer (weighted sum).** This is the original PLE-paper CGC.
A task-wise gate produces a weighted sum of all expert outputs,
and the result has fixed dimension `expert_hidden_dim` regardless
of how many experts you have:

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot \mathbf{e}_i, \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x}) \in \mathbb{R}^N$$

**CGCAttention (block scaling).** This is what we actually run.
With heterogeneous experts (different output dimensions, different
internal structure) a weighted sum throws away too much
information, so instead we concatenate all expert outputs and
scale each block by a per-task attention weight:

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

$$\tilde{\mathbf{h}}_{k,i} = w_{k,i} \cdot \mathbf{h}_i^{expert} \quad \text{for } i = 1, \dots, 8$$

$$\mathbf{h}_k^{cgc} = [\tilde{\mathbf{h}}_{k,1} \,\|\, \tilde{\mathbf{h}}_{k,2} \,\|\, \dots \,\|\, \tilde{\mathbf{h}}_{k,8}] \in \mathbb{R}^{576}$$

The output is the same 576D for every task, but with different
expert contribution weights per task. The Transformer analogy is
exact: the gate is the **Query**, the shared representation is
the **Key**, and each expert output is the **Value** that gets
selectively passed through.

## What the implementation actually needed

Two corrections that the original PLE paper does not have but you
will need the moment you put PLE into a real system.

**Entropy regularization.** Even with PLE's structural separation,
nothing in the loss explicitly penalizes a task gate that picks
one expert and ignores the other seven. We add an entropy
regularizer over the gate distribution:

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left(-\frac{1}{|\mathcal{T}|}\right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k), \quad H(\mathbf{w}_k) = -\sum_{i=1}^{8} w_{k,i} \cdot \log(w_{k,i})$$

With $\lambda_{ent} = 0.01$, minimizing the negative entropy
pushes gate distributions toward uniform, which keeps the experts
in the rotation. This is cheap and matters more than its weight
suggests.

**Dimension normalization.** When experts have heterogeneous
output dimensions — in our case, one 128D expert
(`unified_hgcn`) and seven 64D experts — the larger expert
dominates the post-concatenation magnitude before any gate
weighting kicks in. The fix is a per-expert scale factor:

$$\text{scale}_i = \sqrt{\text{mean\_dim} / \text{dim}_i}, \quad \text{mean\_dim} = (128 + 64 \times 7) / 8 = 72.0$$

This gives the 128D expert a scale of $\approx 0.750$
(attenuation) and the 64D experts a scale of $\approx 1.061$
(amplification). The gate now decides expert importance based on
content, not on whoever happens to have a wider output layer.

## What's next

Episode 2 moves to the heterogeneous expert basket itself — the
design choice of replacing PLE's identical expert pool with eight
structurally different domain experts (DeepFM, LightGCN, Unified
HGCN, Temporal, PersLay, Causal, Optimal Transport, RawScale).
The FeatureRouter pattern, the per-expert input dimensions, the
selection criteria, and why the pool/basket pattern is its own
design philosophy and not just an implementation trick.

Source material for this episode is §1 of the PLE + adaTT Tech
Reference (linked in the frontmatter). This series adapts the
reference for a public reading audience and adds the
implementation context that the reference omits.
