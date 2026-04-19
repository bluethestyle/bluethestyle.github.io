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

*First post of the adaTT sub-thread inside the "Study Thread" series. In
parallel Korean and English across ADATT-1 → ADATT-4 I summarise the
adaTT (Adaptive Task Transfer) mechanism behind this project. The source
is the on-prem reference `기술참조서/adaTT_기술_참조서`, and the
complete PDF is attached to the final ADATT-4 post. This first part
starts from the root motivation — "why an adaptive tower?" — walks
through the analogy with Transformer Attention, locates adaTT inside
the conditional-computation / hypernetwork lineage, gives an intuitive
reading of the core equations, and closes on the overall narrative of
"measure, select, regulate."*

## Why "Adaptive Towers" — Starting from the Limits of Fixed Towers

The simplest Multi-Task Learning (MTL) architecture is *a single shared
backbone* with *a fixed, task-specific tower* on top of it. The design
is crisp, but it carries fundamental weaknesses.

> **Three limits of fixed towers.**
> 1. *One-way sharing*: shared backbone parameters influence every task
>    identically. Even if CTR optimisation degrades Churn prediction,
>    there is no mechanism to detect or regulate it.
> 2. *Task interactions ignored*: sixteen tasks implicitly compete over
>    the shared parameters, yet nothing measures which task pairs help
>    each other and which pairs hurt.
> 3. *No adaptation to training phase*: CTR and CVR may train in similar
>    directions early on but diverge onto very different features later.
>    A fixed weighting cannot track that drift.

adaTT answers each of these limits.

1. *Selective transfer*: knowledge is shared only between task pairs
   whose gradients align; opposite directions are blocked.
2. *Task-affinity measurement*: gradient cosine similarity gives a
   *quantitative* readout of task-to-task relationships.
3. *Dynamic adaptation*: a 3-Phase schedule modulates transfer strength
   with training progress.

Summarised in one sentence: *"fixed towers ignore task interference,
adaptive towers measure and control it."*

## Transformers and Attention — Why This Mechanism Fits Task Adaptation

Although adaTT is spelled "Adaptive Task-aware *Transfer*," the core
principle it leverages internally is deeply connected with the
philosophy of the Attention mechanism.

### The Query-Key-Value Principle of Self-Attention

Transformer Self-Attention splits the input into three roles.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

- $\mathbf{Q}$: Query — "what am I looking for?"
- $\mathbf{K}$: Key — "what information can I offer?"
- $\mathbf{V}$: Value — "what do I actually deliver?"
- $d_k$: key dimension (scaling factor)

The central insight is *"focus on what is related, ignore what is not."*
$\mathbf{Q} \mathbf{K}^\top$ computes a *similarity* between queries and
keys; softmax turns that similarity into a *probability distribution*
used to take a weighted sum of the values.

> **Undergrad math — dot product and the angle between vectors.** The
> dot product of two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$
> is $\mathbf{a} \cdot \mathbf{b} = \sum_{k=1}^n a_k b_k$. Combined with
> the high-school identity $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$,
> a *large* dot product means (1) the vectors are long or (2) the angle
> $\theta$ is small (directions are similar). The $(i,j)$ entry of
> $\mathbf{Q} \mathbf{K}^\top$ is the dot product of Query $i$ with Key
> $j$, so Attention *assigns high scores to Keys whose direction matches
> the Query*. Division by $\sqrt{d_k}$ is needed because the variance of
> the dot product scales linearly with $d_k$; without this normalisation
> softmax saturates on extreme values and we hit *gradient vanishing*.

### The Attention Analogy Inside adaTT

adaTT does not perform Self-Attention over tokens — it performs
*Attention over tasks*. The analogy lines up as follows.

| Role | Transformer Self-Attention | adaTT Task Transfer |
| --- | --- | --- |
| Query | query of the current token | gradient direction of the current task |
| Key | response capacity of another token | gradient direction of another task |
| Similarity | $\mathbf{Q} \mathbf{K}^\top / \sqrt{d_k}$ | gradient cosine similarity |
| Probabilisation | softmax | softmax (temperature $T$) |
| Value | actual information of another token | loss value of another task |
| Output | weighted-sum context | transfer loss |

In essence, adaTT's transfer-weight computation is *Attention in task
space*: "from other tasks (Keys) whose gradient direction is similar to
my own task (Query), pull their losses (Values) and take a weighted
sum."

## Conditional Computation and Hypernetworks — adaTT's Lineage

adaTT's idea belongs to the broader *Conditional Computation* paradigm:
"change the network's behaviour dynamically depending on the input or
situation."

### The Core Idea of Conditional Computation

> **Historical context — origins of Conditional Computation.** The idea
> of "activating only part of the network depending on the input" was
> formalised by Bengio et al. (2013, *"Estimating or Propagating
> Gradients Through Stochastic Neurons for Conditional Computation"*).
> The original goal was *reducing computation cost* — instead of
> activating every neuron on every input, a gating function selects only
> the necessary pieces. This idea evolved into Mixture of Experts (MoE,
> Shazeer et al., 2017) and eventually scaled to a trillion parameters
> in Google's Switch Transformer (Fedus et al., 2022). adaTT is the
> *task-level extension* of Conditional Computation: it conditionally
> decides not "which neurons to activate" but "which task's knowledge to
> transfer."

A conventional network applies the same weights $\mathbf{W}$ to every
input.

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Conditional Computation lets the weights themselves vary with a
*condition* $\mathbf{c}$.

$$\mathbf{y} = \mathbf{W}(\mathbf{c}) \mathbf{x} + \mathbf{b}(\mathbf{c})$$

The condition $\mathbf{c}$ can be a task ID, an input feature, a
training phase, or any other signal.

### Relation to Hypernetworks

The Hypernetwork (Ha et al., ICLR 2017) is the idea that "one network
generates the weights of another network." adaTT can be read as a
*lightweight variant* of this idea.

> **Historical context — the birth of the Hypernetwork.** Ha, Dai & Le
> (ICLR 2017, *"HyperNetworks"*) proposed the then-radical idea that "a
> small network (hypernetwork) directly produces the weights of a larger
> network (main network)." The inspiration was biological — just as a
> genotype does not directly determine behaviour but expresses a
> phenotype through protein-synthesis pathways, a hypernetwork decides
> the parameters of the main network *indirectly*. This paradigm led to
> Task-Conditioned HyperNetworks (von Oswald et al., NeurIPS 2020) and
> to the low-rank adaptation of LoRA (Hu et al., 2022). adaTT is a
> variant that drops hypernetwork-style *full weight generation* and
> instead determines transfer weights from an *observed signal* —
> gradient similarity — for maximum parameter efficiency.

| Aspect | Hypernetwork | adaTT |
| --- | --- | --- |
| Weight generation | separate network produces full weights | gradient similarity determines transfer weights |
| Conditioning input | task-embedding vector | per-task gradient vector |
| Parameter count | large (the generator itself is large) | small ($n^2$ transfer matrix + prior) |
| Adaptation speed | tied to training | fast adaptation via EMA |

adaTT's key advantage is that it *uses gradients themselves as the
conditioning signal*. Rather than a separately learnable representation
like a task embedding, it judges task relationships from gradient
directions observed directly in the current training state. This allows
changes in task relationships across training phases to be reflected
*without delay*.

### How Task Embeddings Regulate the Parameter Space

In a typical task-adaptive model, a task embedding
$\mathbf{e}_{\text{task}}$ plays the role of *selecting the working
region* in parameter space.

$$\mathbf{W}_{\text{effective}} = \mathbf{W}_{\text{shared}} + \mathbf{\Delta}(\mathbf{e}_{\text{task}})$$

In adaTT this role is played by the *transfer-weight matrix*
$\mathbf{R}$. For each task $i$, the row $\mathbf{R}_{i, :}$ encodes
"task $i$'s view of its relationship with the other tasks," and this
governs the direction of movement in the loss landscape.

> **Recent trend — task-specific adaptation research (2024–2025).** The
> idea of regulating the parameter space with task embeddings has been
> steadily refined. (1) *Task Arithmetic (Ilharco et al., ICLR 2023)*:
> add and subtract the weight-difference vectors of fine-tuned models
> ("task vectors") *arithmetically* to compose model capabilities —
> e.g. "sentiment-analysis + translation − toxic-output." (2)
> *TIES-Merging (Yadav et al., NeurIPS 2023)*: an algorithm that
> resolves sign conflicts when merging many task vectors. (3)
> *AdapterSoup (Chronopoulou et al., EACL 2023)*: weight-averages many
> LoRA adapters to adapt to a new task. adaTT's transfer-weight matrix
> $\mathbf{R}$ can be read as a *continuous and dynamic* version of this
> family of task-vector combinations.

Mathematically, the effect of transfer weights on the loss is

$$\nabla_\theta \mathcal{L}_i^{\text{adaTT}} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j.$$

The term $\lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$
is a *correction vector that adjusts the parameter-update direction for
task $i$*. Gradients of high-affinity tasks $j$ enter with large
weights, nudging the shared parameters toward a direction that is
simultaneously favourable for both sides.

## Intuitive Reading of the Core Equations

### Cosine Similarity — Why Compare Only Directions

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

*Intuition*: think of the gradients of two tasks as arrows in a
high-dimensional space.

- *Magnitude* captures the absolute sensitivity of the loss. If CTR loss
  is 0.01 and LTV loss is 1000, the gradient magnitudes differ by tens
  of thousands of times.
- *Direction* captures "which way to change the parameters so that the
  loss decreases."

adaTT cares about *direction*: "do the two tasks want to move
parameters the same way?" Cosine similarity normalises magnitudes so
that only direction is compared. Had we used Euclidean distance, the
task with the larger gradient would dominate and two tasks with
*identical* directions could still be judged "far apart."

> **Undergrad math — geometric meaning of cosine similarity.** In the
> plane, take vectors $\mathbf{a} = (3, 4)$ and $\mathbf{b} = (6, 8)$.
> Euclidean distance gives
> $\|\mathbf{a} - \mathbf{b}\| = \sqrt{9 + 16} = 5$, labelling them "far
> apart." But they point in *the same direction*. Cosine similarity
> returns $\cos\theta = (3 \cdot 6 + 4 \cdot 8) / (5 \cdot 10) = 50/50 = 1.0$,
> correctly recognising "identical direction." Geometrically, cosine
> similarity projects the two vectors onto the unit circle (or the unit
> sphere in higher dimensions) and measures the angle between them.
> $\cos\theta = 1$ means $\theta = 0°$ (same direction),
> $\cos\theta = 0$ means $\theta = 90°$ (orthogonal, unrelated),
> $\cos\theta = -1$ means $\theta = 180°$ (opposite). Gradient
> *magnitude* depends on the absolute scale of the loss, but
> *direction* carries the essential information — "which way do we push
> parameters to reduce loss?" — so a direction-only comparison via
> cosine similarity is the right choice.

### Softmax Normalisation — Why Form a Probability Distribution

$$w_{i \to j} = \frac{\exp(\mathbf{R}_{i,j} / T)}{\sum_{k \neq i} \exp(\mathbf{R}_{i,k} / T)}$$

*Intuition*: turning transfer weights into a *probability distribution*
buys three things.

1. *Unit sum*: transfer weights always sum to 1, so the scale of the
   transfer loss is invariant to the number of tasks. With sixteen tasks
   or thirty-two, the meaning of $\lambda = 0.1$ does not change.
2. *Competitive selection*: by nature of softmax, when one weight rises
   the others fall. This *focuses attention on the most helpful tasks*.
3. *Differentiability*: unlike hard argmax, softmax is continuous and
   differentiable, which enables gradient-based optimisation of the
   learnable parameter $\mathbf{W}$.

The role of temperature $T$ reads well as a thermometer.

- Low $T$ (cold): like a crystallising material, the distribution
  *focuses on a single task*.
- High $T$ (hot): like a gas, the distribution *spreads evenly over all
  tasks*.
- $T = 1.0$ is the liquid state — fluid enough to move yet still
  structured.

> **Undergrad math — anatomy of softmax.** For an input vector
> $\mathbf{z} = (z_1, z_2, z_3)$, softmax is
> $\sigma(z_i) = e^{z_i} / \sum_k e^{z_k}$. For
> $\mathbf{z} = (2, 1, 0)$,
> $\sigma = (e^2, e^1, e^0) / (e^2 + e + 1) \approx (7.39, 2.72, 1.0) / 11.1 \approx (0.67, 0.24, 0.09)$.
> Applying temperature $T$ gives $\sigma(z_i / T)$. At $T = 0.5$ the
> inputs are *doubled* to $(4, 2, 0)$, giving
> $\approx (0.87, 0.12, 0.02)$ — more concentrated on the maximum. At
> $T = 2.0$ the inputs *shrink* to $(1, 0.5, 0)$, giving
> $\approx (0.42, 0.34, 0.24)$ — nearly uniform. Mathematically, as
> $T \to 0^+$ softmax converges to a one-hot vector ($\text{argmax}$),
> and as $T \to \infty$ it converges to the uniform distribution $1/n$.
> This property is inherited from the Boltzmann distribution of
> statistical mechanics, $p_i \propto e^{-E_i / k_B T}$, where $T$ is
> the actual physical temperature.

### EMA Smoothing — Balancing Memory and Forgetting

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1 - \alpha) \cdot \mathbf{C}_t$$

*Intuition*: treat EMA as a kind of *memory system*.

- $\alpha = 0.9$ means "keep 90% of the old memory and take in 10% of
  the new observation."
- This corresponds to an *effective observation window* of about
  $1 / (1 - \alpha) = 10$.
- In other words, it behaves like a weighted average of roughly the
  last ten observations.

Why not a simple average?

- *Simple average*: every past observation gets equal weight. Noisy
  early-training gradients keep influencing estimates late into
  training.
- *EMA*: recent observations dominate. When task relationships drift
  during training, the estimate follows quickly.
- *Sliding window*: requires careful window management and is
  memory-heavy. EMA obtains the same effect with a single scalar
  $\alpha$.

> **Undergrad math — EMA as a recursive filter.** Unrolling
> $A_t = \alpha A_{t-1} + (1-\alpha) C_t$ gives
> $A_t = (1-\alpha) \sum_{k=0}^{t-1} \alpha^k C_{t-k} + \alpha^t A_0$.
> The weight of a past observation $C_{t-k}$ is $(1-\alpha) \alpha^k$,
> which shrinks *exponentially* as $k$ grows (the older the
> observation). With $\alpha = 0.9$ the weight is $0.09$ one step back,
> $0.059$ five steps back, $0.012$ twenty steps back, and $0.0005$ fifty
> steps back — effectively ignored. This is exactly the structure of a
> *first-order IIR (Infinite Impulse Response) low-pass filter* in
> signal processing. The transfer function is
> $H(z) = (1-\alpha) / (1 - \alpha z^{-1})$, and the cut-off frequency
> $f_c \approx (1-\alpha) / (2 \pi)$ strips high-frequency noise
> (per-batch gradient fluctuation) while passing the low-frequency
> trend (the real task relationships). Memory-wise, a sliding-window
> average needs $O(W)$ memory, whereas EMA only stores the current state
> $A_t$ and runs on $O(1)$ memory.

### Transfer-Enhanced Loss — Learning from Other Tasks

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$$

*Intuition*: read this formula as *the act of listening to advice*.

- $\mathcal{L}_i$: the judgement of task $i$ itself (raw loss).
- $\sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$: advice from the
  other tasks (a weighted combination).
- $\lambda = 0.1$: how seriously to take the advice (10% weight).
- $w_{i \to j}$: whose advice to trust more (affinity-based weights).

The conservative value $\lambda = 0.1$ means *"keep 90% of your own
judgement and take only 10% from your peers."* The safeguard
`max_transfer_ratio = 0.5` adds that "no matter how good the advice is,
never exceed 50% of your own judgement."

### Prior Blend — Balancing Experience and Data

$$\mathbf{R}_{\text{blended}} = (\mathbf{W} + \mathbf{A}) \cdot (1 - r) + \mathbf{P} \cdot r$$

*Intuition*: this formula is the act of *mixing the opinion of a
seasoned mentor (Prior) with what the data actually shows (Affinity)*.

- Early training ($r = 0.5$): "data is still sparse — half-trust the
  mentor's view (the domain knowledge)."
- Late training ($r = 0.1$): "data has accumulated — trust the observed
  results 90%."

This matches the central principle of Bayesian inference: *when the
prior is strong, rely on the prior; once enough data has accumulated,
follow the likelihood (observation).*

> **Undergrad math — Bayesian inference basics, prior and posterior.**
> Bayes' theorem is
> $P(\theta | D) = P(D | \theta) \cdot P(\theta) / P(D)$. Here
> $P(\theta)$ is the *prior* — belief before seeing data; $P(D | \theta)$
> is the *likelihood* — probability of observing $D$ given parameter
> $\theta$; $P(\theta | D)$ is the *posterior* — updated belief after
> seeing the data. Consider coin flipping: you start with a prior "the
> probability of heads is near 0.5," then flip ten times and see seven
> heads — the posterior shifts above 0.5. After a thousand flips the
> posterior converges close to the observed ratio (0.7). In adaTT, the
> Group Prior $\mathbf{P}$ is the prior belief, the gradient cosine
> similarity is the observed data, and the decay of the blend ratio $r$
> implements the Bayesian update of "as data accumulates, reduce
> reliance on the prior."

## The Overall Narrative — "Measure, Select, Regulate"

adaTT's end-to-end behaviour fits in three words: *Measure*, *Select*,
*Regulate*.

1. *Measure*: every $N$ steps, extract each task's gradient and compute
   task-to-task affinity via cosine similarity. EMA filters out noise
   to maintain a stable affinity matrix $\mathbf{A}$.
2. *Select*: blend the affinity matrix with a Group Prior, block
   negative transfer, and softmax-normalise to obtain transfer weights
   $\mathbf{w}$. "Whom to learn from" is decided from data.
3. *Regulate*: a 3-Phase schedule controls the strength and timing of
   transfer. Early training observes only (Warmup), mid-training
   transfers dynamically (Dynamic), and late training freezes for
   stability (Frozen).

These three steps together create an ecosystem in which *sixteen tasks
grow complementarily without obstructing one another's learning*.

> **Fixed tower vs adaptive tower — the key difference.** *Fixed tower*:
> "all tasks share the same backbone; if conflicts arise, so be it."
> *Adaptive tower (adaTT)*: "all tasks share the same backbone, but we
> measure in real time who helps whom, block harmful interference, and
> selectively transmit only the beneficial knowledge."

## Overview and Design Philosophy of adaTT

> **Core problem — negative transfer in Multi-Task Learning.** Training
> many tasks simultaneously in one network can cause *negative
> transfer*: when task A's gradient points opposite to task B's
> gradient, a shared-parameter update that improves one task degrades
> the other. In this system, which trains sixteen tasks concurrently,
> the problem is especially severe.

### Why Adaptive Task Transfer

Our Multi-Task Learning (MTL) architecture trains sixteen active tasks
at once. CTR, CVR, Churn, NBA and other tasks with *different business
goals* all share Shared Expert parameters, so training becomes unstable
unless their interactions are controlled.

Conventional approaches include:

- *Fixed Weighting*: manually set per-task weights (Kendall et al.,
  2018).
- *GradNorm*: dynamic weighting based on gradient magnitude (Chen et
  al., 2018).
- *PCGrad*: project away conflicting gradients (Yu et al., 2020).

None of them *directly measure* task-to-task similarity. adaTT
*measures* task affinity via gradient cosine similarity and uses that
signal to perform *selective knowledge transfer*.

> **Historical context — the evolution of task-specific adaptation in
> MTL.** The concept of Multi-Task Learning goes back to Caruana (1997,
> *"Multitask Learning"*). Early work relied on plain Hard Parameter
> Sharing (shared backbone + per-task heads), but interference across
> tasks was reported persistently. To address this, (1) Cross-Stitch
> Networks (Misra et al., CVPR 2016) linearly combine per-task network
> outputs; (2) MTAN (Liu et al., CVPR 2019) uses Attention to pick
> task-specific features out of shared ones; (3) PLE (Tang et al.,
> RecSys 2020) separates shared and task-specific components at the
> Expert level. Within this lineage, adaTT is distinguished by *using
> gradients themselves as the task-relation signal*.

> **Recent trend — 2024–2025 task-aware MTL architectures.** Recent MTL
> research is evolving in three directions. (1) *Gradient manipulation*:
> Nash-MTL (Navon et al., ICML 2022) and Aligned-MTL (Senushkin et al.,
> CVPR 2023) project gradients toward Pareto-optimal directions, and
> CAGrad (Liu et al., NeurIPS 2021) guarantees progress on the
> worst-case task. (2) *Task routing*: Mod-Squad (Chen et al., NeurIPS
> 2023) lets each task dynamically select a subset of experts, which is
> complementary to adaTT's affinity-based transfer. (3) *MTL in the
> foundation-model era*: LoRA-based per-task adapters (Hu et al., 2022)
> and MTLoRA (Agiza et al., 2024) deliver parameter-efficient task
> adaptation on huge models. adaTT's gradient-cosine-similarity-based
> approach extends naturally to such adapter routing.

### Core Ideas

adaTT boils down to three ideas.

1. *Gradient cosine similarity*: same-direction gradients signal
   positive transfer; opposite-direction gradients signal negative
   transfer.
2. *Group Prior*: a domain-knowledge-based task-group structure serves
   as a prior, stabilising early training.
3. *3-Phase schedule*: Warmup → Dynamic → Frozen — measure, then apply,
   then freeze affinity.

The 3-Phase schedule progresses roughly as follows. Phase 1 (Warmup)
uses only the Group Prior and performs affinity measurement. Phase 2
(Dynamic) blends prior and affinity to learn the transfer weights.
Phase 3 (Frozen) fixes the transfer weights to stabilise fine-tuning.
The transition points are set by the `warmup_epochs` and `freeze_epoch`
configuration values.

### Placement Within the System

adaTT sits in the *late stage of the forward pass* of the
PLE-Cluster-adaTT architecture. Predictions flow through Shared Experts
→ CGC Gating → Task Experts → Task Towers; once per-task losses are
computed, adaTT adds the inter-task transfer loss. Gradients are
extracted from the task losses to update the affinity matrix, and the
transfer loss is added on top of the original task losses to form the
Total Loss. In code, this coupling is implemented around
`ple_cluster_adatt.py:1290-1319`.

## Where This Post Stops

Starting from "why fixed towers fall short," we used the analogy with
Transformer Attention to reveal adaTT as Attention performed in task
space, and located it inside the Conditional Computation / Hypernetwork
lineage as the variant that uses gradients themselves as the
conditioning signal. We examined the five core equations — cosine
similarity, softmax normalisation, EMA smoothing, the
transfer-enhanced loss, and the prior blend — from a "why this form?"
perspective, and summarised the whole narrative in three words:
*measure, select, regulate.* What remains is *how* the measurement step
actually works: how to distil a stable affinity matrix $\mathbf{A}$
from the batch-to-batch jitter of gradients, why
`torch.compiler.disable` is needed, and what the precise mathematics
and code path of the `TaskAffinityComputer` engine — combining cosine
similarity with EMA — look like. That is the subject of the next post,
**ADATT-2**.
