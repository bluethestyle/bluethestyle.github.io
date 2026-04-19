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
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document (the full PDF will be
attached to PLE-6). adaTT is split off into its own ADATT-1 ~ ADATT-4
sub-thread. This first post starts at the root motivation for multi-task
learning, shows the mathematical face of Negative Transfer, and then
walks through where Shared-Bottom and MMoE each break down — PLE, the
third wave that actually works, is the subject of **PLE-2**.*

## The Case for Multi-Task Learning

### Root motivation — learning one thing helps you learn the next

A recommender system has to produce dozens of predictions at once: CTR
(click-through rate), CVR (conversion), churn, LTV (lifetime value), and
so on. The obvious move is to train an independent model per task. But
in practice the tasks are related to each other.

- A customer with high CTR is also more likely to convert (a funnel relationship).
- A customer at high churn risk has low retention (inverse correlation).
- Spending patterns are the core signal for LTV.

Exploiting that relatedness gives a dramatic boost in data efficiency.
If a pattern discovered while learning task A is also useful for task B,
the same data buys you a richer representation. That is the core
motivation of multi-task learning (MTL).

> **Analogy — learning a foreign language.** Someone who already speaks
> Spanish will pick up Italian faster. The two languages share a large
> fraction of their roots, grammar, and phonetic rules. The
> "understanding of how Romance-family languages are put together"
> acquired from one *transfers* to the other. In MTL, that shared
> structural understanding is exactly what the Shared Expert is asked
> to carry.

### A statistical view — inductive bias and regularization

A single-task model trains only on its own data, which makes overfitting
a real risk. MTL has multiple tasks *implicitly regularize* each other
through a shared representation.

$$\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k(f_k(\mathbf{h}_{shared}(\mathbf{x})))$$

When this total loss is minimized, $\mathbf{h}_{shared}$ cannot easily
overfit to any single task. Only representations that are useful to
*every* task at the same time survive in $\mathbf{h}_{shared}$. This
is a different kind of regularization from L2 or Dropout — it is
*inter-task regularization*.

> **Undergraduate math — what is a weighted sum?** In
> $\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k$, the
> $w_k$ is the weight (importance) of task $k$ and $\mathcal{L}_k$ is
> its loss. A weighted sum is simply "multiply each item by its
> importance and add them up" — the same principle as a grade-point
> average. Concretely, if you get 90 in Korean (weight 2) and 80 in
> math (weight 3), the weighted average is
> $(2 \times 90 + 3 \times 80) / (2 + 3) = 420 / 5 = 84$ — an average
> that leans toward math. In MTL, if $w_{CVR} = 1.5$ and $w_{CTR} = 1.0$,
> the CVR loss contributes 1.5× more to the total, so the model pays
> more attention to CVR accuracy. Training of $\mathbf{h}_{shared}$
> happens through the gradient of this weighted sum,
> $\nabla \mathcal{L}_{MTL}$, so a task with a larger $w_k$ exerts a
> correspondingly stronger pull on the shared representation.

## Negative Transfer — the Dark Side of MTL

### Defining the problem

Is it always a win to train every task together? No. **Negative transfer**
is the phenomenon where loosely-related tasks pollute the shared
representation so badly that the joint model does *worse* than a
single-task baseline.

> **⚠ The seesaw effect — what this looks like in practice.** Push CTR
> performance up and churn performance drops; push churn up and CTR
> drops. That seesaw is the most common failure mode in MTL, and it
> happens because the two tasks' gradients point in opposite directions
> in the shared parameter space.

### An optimization view — gradient conflict

Let $\mathcal{L}_k$ be task $k$'s loss. The gradient of that loss with
respect to the shared parameters $\boldsymbol{\theta}_{shared}$ is:

$$\mathbf{g}_k = \nabla_{\boldsymbol{\theta}_{shared}} \mathcal{L}_k$$

When two tasks' gradients point in *the same direction*, the
relationship is cooperative (positive transfer). When they point in
*opposite directions*, it is conflict (negative transfer).

$$\cos(\mathbf{g}_i, \mathbf{g}_j) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

- $\cos > 0$: tasks $i$ and $j$ cooperate — training them together pays off.
- $\cos < 0$: the gradients conflict — the shared parameter update hurts both sides.
- $\cos \approx 0$: unrelated — joint training has little effect either way.

> **Undergraduate math — dot product and cosine similarity.** For two
> vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$, the dot product is
> $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$. Geometrically,
> $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta$,
> so the cosine similarity
> $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$
> measures the *directional similarity* of two vectors on the range
> $[-1, 1]$. Concretely, in 2D, if $\mathbf{g}_{CTR} = (3, 1)$ and
> $\mathbf{g}_{CVR} = (2, 2)$, then
> $\cos = (3 \times 2 + 1 \times 2) / (\sqrt{10} \times \sqrt{8}) = 8 / 8.94 \approx 0.89$
> — strong cooperation. With $\mathbf{g}_{Churn} = (-1, 2)$,
> $\cos = (3 \times (-1) + 1 \times 2) / (\sqrt{10} \times \sqrt{5}) = -1 / 7.07 \approx -0.14$
> — mild conflict. Whenever gradient cosine similarity goes negative,
> updating the shared parameters turns improvement for one task into
> degradation for the other — and that is Negative Transfer in action.

> **Intuition — tug-of-war.** The shared parameters are like a rope
> being pulled on by every task at once. If everyone pulls in the same
> direction, the rope moves forward quickly. If some pull the opposite
> way, the rope stays put, or even moves backward. PLE's core idea is
> simply "give each task its own extra rope." The Shared Expert is the
> common rope; the Task-specific Expert (in this project's
> implementation, the GroupTaskExpertBasket) is each task's private
> rope.

### An information-theoretic view — mutual information between tasks

Treating tasks $A$ and $B$'s labels as random variables $Y_A$, $Y_B$,
their relationship can be quantified by mutual information.

$$I(Y_A; Y_B) = \sum_{y_a, y_b} p(y_a, y_b) \log \frac{p(y_a, y_b)}{p(y_a) p(y_b)}$$

- High $I(Y_A; Y_B)$: the two tasks share a lot of common information → the same Expert is useful for both.
- Low $I(Y_A; Y_B)$: they are nearly independent → forcing them to share only injects noise.

An ideal MTL architecture lets tasks with high $I$ share a
representation while tasks with low $I$ stay separated. PLE's split
between Shared and Task-specific Experts, combined with the CGC gating
scheme, is exactly this principle baked into the architecture.

> **Undergraduate math — mutual information and KL-divergence.**
> Mutual information $I(X; Y)$ is in fact a *KL-divergence* between two
> distributions: $I(X; Y) = D_{KL}(p(x,y) \,\|\, p(x) p(y))$. The
> KL-divergence $D_{KL}(P \,\|\, Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$
> is "the information lost when you approximate distribution $P$ by
> distribution $Q$." So $I(X; Y)$ is the information loss between the
> joint distribution $p(x,y)$ and the independence assumption
> $p(x) p(y)$ — i.e. "the extra information that exists precisely
> because $X$ and $Y$ are not independent." Concretely: for CTR and
> CVR, a customer who clicks is more likely to convert, so
> $p(\text{click}=1, \text{convert}=1) \gg p(\text{click}=1) \times p(\text{convert}=1)$
> and $I(\text{CTR}; \text{CVR})$ is large. CTR and
> Brand\_prediction, on the other hand, are relatively independent, so
> their $I$ is small — forcing them to share the same Expert only adds
> noise.

## Architecture Evolution — From Shared-Bottom to PLE

### Shared-Bottom — the simplest MTL

Every task passes through a single trunk (a shared network) and then
branches into a per-task head (a tower).

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

- **Strength**: simple to implement and parameter-efficient.
- **Limitation**: it forces every task onto the same representation,
  which makes Negative Transfer severe whenever task relatedness is
  low. As an analogy, it is like handing every student the exact same
  single textbook.

### MMoE — put multiple experts in front of a gate

The MMoE of Ma et al. (KDD 2018) sets up $N$ experts with identical
structure and lets a per-task gate decide a weighted sum of their
outputs.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot f_i^{expert}(\mathbf{x}), \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x})$$

- **Strength**: each task can choose its own expert mixture.
- **Limitation**: **Expert Collapse** — all tasks' gates converge on
  the same expert, degenerating the model back into Shared-Bottom with
  extra parameters. It happens because gradients concentrate on the
  "popular expert," gradients vanish for the rest, and their learning
  stalls.

> **Historical context.** *Shared-Bottom* was formalized by *Caruana
> (Machine Learning, 1997)*, the pioneer of MTL. Rich Caruana's core
> insight was that "jointly training related tasks improves inductive
> bias," demonstrated on medical diagnosis — pneumonia mortality
> prediction improved when related tasks were used as auxiliaries. That
> framework shaped MTL for the next two decades. *MMoE* was proposed
> by *Ma, Zhao, Yi, Chen, Hong & Chi (KDD 2018)* at Google. On
> YouTube's engagement-vs-satisfaction prediction pair, Shared-Bottom
> suffered badly from Negative Transfer, and their answer — keep
> multiple experts but let a gate choose among them — mitigated it.
> The Mixture of Experts idea itself originated with
> *Jacobs, Jordan, Nowlan & Hinton (1991)*; MMoE is a reinterpretation
> of that idea in the multi-task context.

> **Analogy — the buffet restaurant.** MMoE is like a buffet
> restaurant. Several dishes (Experts) are laid out, and each guest
> (task) puts whatever they want on their plate (gate). The problem:
> every guest just piles on the steak (the popular Expert), and the
> rest of the food sits there until it is thrown away. PLE's answer is
> to explicitly offer "a base meal (Shared Expert)" plus "a personal
> specialty (Task Expert)" as separate dishes.

## Where this leaves us

For this project's 13-task setup neither wave is usable as-is.
Shared-Bottom collapses under task diversity — we have churn, ranking,
and regression targets that pull the trunk in incompatible directions.
MMoE in theory lets the tasks route around each other, but the
combination of a symmetric expert pool and unconstrained gates makes
Expert Collapse the default outcome rather than the exception. The fix
that actually holds up is structural: stop asking identical experts to
self-organize by gate pressure alone, and instead *bake the separation
into the architecture itself* — a Shared Expert for cross-task signal,
Task-specific Experts for patterns only one task cares about. That is
PLE, and it is where **PLE-2** picks up.
