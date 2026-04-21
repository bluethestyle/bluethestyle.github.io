---
title: "[Study Thread] ADATT-1 — Why adaTT: Adaptive Towers and the Transformer Attention Analogy"
date: 2026-04-20 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, attention, hypernetwork, mtl]
lang: en
series: study-thread
part: 7
alt_lang: /2026/04/20/adatt-1-adaptive-tower-motivation-ko/
next_title: "ADATT-2 — TaskAffinityComputer and Gradient Cosine Similarity"
next_desc: "The TaskAffinityComputer engine that actually measures task-to-task affinity, the mathematical definition of gradient cosine similarity with EMA smoothing, why cosine over Euclidean distance, and the torch.compiler.disable-handled gradient extraction path."
next_status: published
---

*First post of the adaTT sub-thread in the "Study Thread" series. Across
ADATT-1 → ADATT-4, in parallel Korean and English, I unpack the adaTT
(Adaptive Task Transfer) mechanism behind this project. The source is
the on-prem reference `기술참조서/adaTT_기술_참조서`, and the complete
PDF is attached to the final ADATT-4 post. Where the PLE sub-thread
dealt with task interference in the *feature* path, the adaTT
sub-thread deals with what is left in the *gradient* path — how we
measure the remaining conflicts and turn them back into cooperation.
This first post starts from a single design decision — "why an
adaptive tower?" — and walks through its analogy with Transformer
Attention and its position in the Conditional Computation /
Hypernetwork lineage.*

> **Provisional status — no measurable benefit on synthetic data.**
> Stated plainly: on the synthetic-data benchmarks so far, PLE with
> adaTT shows no clear performance gap against PLE without adaTT. If
> the real-data evaluation confirms the same, *removing adaTT* is the
> reasonable move. The four posts that follow are not written under
> an assumption of removal — they are a record of "why we chose this
> design." Even if the component turns out to be unused, the
> mathematical and engineering reasoning behind the choice is worth
> preserving.

## What PLE Did Not Fix — Gradient Conflict

PLE-2 split "which task looks at which expert" in the feature path via
CGC gating. By explicitly separating shared and task-specific expert
pools, the path by which one task's feature preference interferes with
another was largely closed off.

But *another* interference path remains. No matter how cleanly the gate
splits things up, the Shared Experts themselves take the backward pass
of every task. When CTR's loss pushes $\theta_{shared}$ in one
direction and Churn's loss pushes it the opposite way, the same
parameters receive contradictory gradients on every step, and the
shared backbone oscillates in the middle. The "implicit regularization"
that MTL is supposed to provide turns into *noise* at exactly this
point.

A fixed tower can neither see nor change this conflict. It simply
distributes fixed weights to per-task towers; it never even asks *"are
CTR's and Churn's gradients elbowing each other right now?"* adaTT
exists to ask exactly that question.

> **Three limits of a fixed tower.** (1) *One-way sharing* — when CTR
> optimisation hurts Churn, there is no detection mechanism. (2)
> *Task interactions ignored* — which task pairs help, which hurt?
> Nothing measures it. (3) *No adaptation to training phase* — fixed
> weights cannot track how task relationships shift between early and
> late training.

## Three Decisions

Responding to these limits demands three decisions in sequence.

*First — measure, do not guess.* Do not rely on manual task-pair
tagging or domain priors alone. Pull the signal from what is *directly
observed during training*. This leads us to gradient cosine similarity
later.

*Second — borrow the Attention philosophy.* Self-Attention's core
principle — "focus on what is related, ignore what is not" — transfers
directly to task adaptation. Swap tokens for tasks, and rearrange the
Query-Key-Value framing onto *task space*.

*Third — slot into the Conditional Computation lineage.* Hypernetworks
change weights themselves based on a condition $\mathbf{c}$. adaTT is
the lighter variant: it does not generate weights, only modulates
*transfer strength* conditionally. A full hypernetwork is replaced by
an $n^2$ transfer matrix.

In one sentence: *"fixed towers ignore task interference, adaptive
towers measure and control it."*

## Moving the Attention Philosophy into Task Space

Transformer Self-Attention splits the input into three roles.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

- $\mathbf{Q}$ — "what am I looking for?"
- $\mathbf{K}$ — "what information can I offer?"
- $\mathbf{V}$ — "what do I actually deliver?"
- $d_k$ — the Key dimension, a scaling factor to prevent softmax
  saturation.

> **Equation intuition.** A larger $\mathbf{Q} \mathbf{K}^\top$ means
> two vectors point in similar directions; softmax turns those scores
> into a probability distribution used to take a weighted sum of
> $\mathbf{V}$. One-line summary: "focus more on Keys whose direction
> matches your own." Dividing by $\sqrt{d_k}$ exists because the
> variance of the dot product scales linearly with $d_k$; without it,
> softmax saturates on extreme values and gradients vanish.

adaTT performs Attention over *tasks*, not tokens. Each role shifts by
one level.

| Role | Transformer Self-Attention | adaTT Task Transfer |
| --- | --- | --- |
| Query | query of the current token | gradient direction of the current task |
| Key | response capacity of another token | gradient direction of another task |
| Similarity | $\mathbf{Q} \mathbf{K}^\top / \sqrt{d_k}$ | gradient cosine similarity |
| Probabilisation | softmax | softmax (temperature $T$) |
| Value | actual information of another token | loss value of another task |
| Output | weighted-sum context | transfer loss |

adaTT's transfer-weight computation is, in essence, *Attention in task
space*: "from other tasks whose gradient direction is similar to mine,
pull their losses and take a weighted sum."

## Lineage — Conditional Computation and Hypernetworks

This idea does not stand alone. It lives inside the broader Conditional
Computation paradigm: "dynamically change the network's behaviour
depending on input or situation."

> **Historical context.** Bengio et al. (2013, *"Estimating or
> Propagating Gradients Through Stochastic Neurons for Conditional
> Computation"*) formalised "activate only part of the network
> depending on input." This evolved into Mixture of Experts (Shazeer
> et al., 2017) and eventually scaled to a trillion parameters in
> Switch Transformer (Fedus et al., 2022).

A conventional network applies the same $\mathbf{W}$ to every input.

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Conditional Computation lets the weights themselves vary with a
condition $\mathbf{c}$.

$$\mathbf{y} = \mathbf{W}(\mathbf{c}) \mathbf{x} + \mathbf{b}(\mathbf{c})$$

What goes into $\mathbf{c}$ is the branching point. The Hypernetwork
(Ha et al., ICLR 2017) puts a task embedding there and lets "a small
network generate a large network's weights" — paying for full weight
generation with a heavy generator network.

adaTT picks the other branch. Rather than generating weights, it uses
an *observed signal* — the gradient direction, measured directly in
the current training state — as the condition. With no separately
learnable representation like a task embedding, a shift in task
relationships is reflected immediately.

| Aspect | Hypernetwork | adaTT |
| --- | --- | --- |
| Weight generation | separate network produces full weights | gradient similarity determines transfer weights |
| Conditioning input | task-embedding vector | per-task gradient vector |
| Parameter count | large (the generator itself is large) | small ($n^2$ transfer matrix + prior) |
| Adaptation speed | tied to training | fast adaptation via EMA |

The shape is: *task $i$'s update direction gets a correction vector
that weighs peer tasks' gradients by affinity*. High-affinity pairs
contribute strongly; low-affinity or opposing pairs collapse to near
zero. The explicit formulas and derivations arrive step by step across
ADATT-2 and ADATT-3.

## Five decisions become five equations

The five core equations that show up across the following posts each
correspond to one design decision. Here I leave only the name and the
role of each — the equation lands in the post where the decision is
unpacked.

- **Cosine similarity** — gradient magnitudes vary with each task's
  loss scale, so the only thing that can be compared is *direction*.
  Magnitude-normalised, direction-only. (ADATT-2)
- **Softmax normalisation** — keeping transfer weights summing to 1
  preserves the meaning of the transfer strength $\lambda$ whether
  there are 16 or 32 tasks. A differentiable choice over hard argmax.
  (ADATT-3)
- **EMA smoothing** — single-batch gradients are noisy; past
  observations are folded in via exponential decay for stability.
  $O(1)$ memory, sliding-window-equivalent. (ADATT-2)
- **Transfer-Enhanced Loss** — peer loss enters as a residual on top
  of own loss. The residual form means a useless transfer naturally
  converges to zero weight — safe default. (ADATT-3)
- **Prior Blend** — early on, data confidence is low, so a domain
  prior carries half the weight; as training proceeds, the ratio
  shifts toward data. Bayesian "prior → posterior" transition reduced
  to a single blend ratio. (ADATT-3)

## "Measure, Select, Regulate"

adaTT's end-to-end behaviour fits in three words: *Measure*, *Select*,
*Regulate*.

1. *Measure* — every $N$ steps, extract per-task gradients and compute
   task-to-task affinity via cosine similarity. EMA filters batch
   noise to keep a stable affinity matrix $\mathbf{A}$. (Subject of
   ADATT-2.)
2. *Select* — blend affinity with a Group Prior, block negative
   transfer, and softmax-normalise to produce transfer weights
   $\mathbf{w}$. (Subject of ADATT-3.)
3. *Regulate* — a 3-Phase schedule (Warmup → Dynamic → Frozen) controls
   transfer strength and timing: observe only early, transfer
   dynamically mid-training, freeze late for stability. (Also ADATT-3.)

With these three steps combined, *sixteen tasks grow complementarily
instead of obstructing each other's learning*. Where a fixed tower
accepts "the shared backbone shakes, so be it," adaTT replaces it with
"measure in real time who helps whom, cut the harmful interference,
and transmit only the beneficial knowledge."

## Where We Stop

We laid out "why an adaptive tower?" as three decisions — measurement
first, Attention philosophy, and the Conditional Computation lineage.
We reduced the five core equations to one-line "why this form?"
summaries, and wrapped everything in the three words "Measure, Select,
Regulate." What remains is *how the Measure step actually works*: how
to distil a stable affinity matrix $\mathbf{A}$ out of batch-to-batch
gradient jitter, why `torch.compiler.disable` is needed, and what
exactly the `TaskAffinityComputer` engine's math and code path look
like. That is the subject of the next post, **ADATT-2**.
