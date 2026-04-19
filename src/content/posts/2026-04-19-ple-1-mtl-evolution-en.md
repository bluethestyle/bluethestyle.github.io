---
title: "[Study Thread] PLE-1 — MTL and the Evolution Toward Gated Experts (Shared-Bottom → MMoE)"
date: 2026-04-19 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, mmoe, mtl, shared-bottom]
lang: en
series: study-thread
part: 1
alt_lang: /2026/04/19/ple-1-mtl-evolution-ko/
next_title: "PLE-2 — Progressive Layered Extraction: Explicit Expert Separation and CGC Gates"
next_desc: "PLE's fix: explicit separation of shared vs task-specific experts (Tang et al., 2020). Two CGC gate variants — weighted-sum (CGCLayer) and block-scaling (CGCAttention) — with the full math. Entropy regularization and dimension normalization to keep heterogeneous experts healthy."
next_status: draft
---

*PLE-1 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and
math foundations behind the PLE architecture used in this project.
Source: the on-prem `기술참조서/PLE_기술_참조서` document (the full
PDF will be attached to PLE-6). adaTT is split off into its own
ADATT-1 ~ ADATT-4 sub-thread.*

## Why care

We need to predict 13 tasks from one customer representation —
churn signal, next best action, MCC trends, six product-acquisition
probabilities, and so on. Multi-task learning (MTL) is the obvious
framing, but the architecture choice inside MTL is not obvious.
Shared-Bottom and MMoE are the first two waves of MTL
architectures, and each one fails in a specific way that motivates
the next. This post walks those two failures; PLE — the wave that
actually works — is the subject of PLE-2.

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

## Shared-Bottom (Caruana, 1997)

All tasks share a single trunk and then branch into per-task
heads:

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

Implementation is simple, parameter count is minimal. The failure
mode is **negative transfer**: when two tasks want the shared
trunk to encode different things, gradient updates from one task
actively hurt the other. With low-correlation tasks this gets
severe enough that you would have done better training each task
in isolation.

## MMoE (Ma et al., KDD 2018)

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

## Where this leaves us

For our 13-task setup neither wave is usable as-is. Shared-Bottom
collapses under task diversity — we have churn, ranking, and
regression targets that pull the trunk in incompatible directions.
MMoE in theory lets the tasks route around each other, but the
symmetric expert pool plus unconstrained gates means expert
collapse is the default outcome, not the exception. The fix that
actually holds up is structural: stop asking identical experts to
self-organize by gate pressure alone, and instead *bake the
separation into the architecture* — shared experts for cross-task
signal, task-specific experts for patterns only one task cares
about. That is PLE, and it is where **PLE-2** picks up: the
Progressive Layered Extraction architecture (Tang et al., RecSys
2020), the CGC gate in both of its variants, and the two
regularizers we needed to add before it would train stably on
heterogeneous experts.
