---
title: "[Study Thread] PLE-2 — Progressive Layered Extraction: Explicit Expert Separation and CGC Gates"
date: 2026-04-19 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, tang2020, mtl]
lang: en
excerpt: "Picking up from MMoE's Expert Collapse — PLE's three chained decisions: explicit Shared/Task expert separation, the heterogeneous Shared Expert pool, and the CGC gate that learns how much of each expert to use per task."
series: study-thread
part: 2
alt_lang: /2026/04/19/ple-2-progressive-layered-extraction-ko/
next_title: "PLE-3 — Input Structure and Heterogeneous Shared Expert Pool (512D)"
next_desc: "The full PLEClusterInput field spec, 734D feature-tensor index mapping, and HMM mode routing. Plus the seven heterogeneous Shared Experts this project actually runs (Economics, Temporal, HMM, TDA, GMM, GCN, PersLay, UnifiedHGCN) — each interpreting the customer through a structurally different mathematical lens."
next_status: published
---

*PLE-2 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document. This second post
picks up exactly where PLE-1 left off — at MMoE's Expert Collapse —
and walks through the three chained decisions the architecture makes
in response.*

## What PLE-1 left on the table

MMoE was the answer to Shared-Bottom's gradient conflict. Instead of
forcing every task through one shared trunk, it puts multiple experts
between the trunk and the task towers, and a per-task gate learns
how much of each expert to use. In theory, conflict dissolves — if
task $i$ leans on Expert A and task $j$ leans on Expert B, they stop
stepping on each other's gradients.

In practice it does not play out that way. When training starts, one
of the experts — because of parameter init, input distribution, or
loss magnitude differences — receives a slightly larger gradient in
the first few steps. That expert becomes slightly more useful. Every
task's gate tilts slightly toward it. That expert receives still more
gradient, and gets more useful. A winner-take-all positive feedback
loop — *Expert Collapse* — kicks in. Eventually every task uses the
same expert; the other experts' parameters barely update. The model
has quietly degenerated back into Shared-Bottom.

The diagnosis is clean. The expert pool is *symmetric*. Every expert
shares the same architecture, the same input, the same initialization.
Asking a gate to "learn a division of labor" inside that symmetry is
basically asking 7 identical employees to do 7 different jobs —
whoever pulls ahead first attracts all the work.

PLE's answer is in two moves. Break the symmetry at the architecture
level, then layer a constrained gate on top.

## Decision 1 — split shared and task-private experts explicitly

Tang, Liu, Zhao & Gong (RecSys 2020) argue: don't hope the gate
learns to divide labor; bake the division into the architecture.
Concretely, split the experts into two categories.

- *Shared Expert* ($\mathcal{E}^s$): a public expert pool every task can attend to.
- *Task-specific Expert* ($\mathcal{E}^k$): a private expert only task $k$ can use.

Each task's gate receives both the Shared pool and its private
expert as inputs, and learns the best combination ratio.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

What this one line resolves:

1. **Negative Transfer dampens.** The task-private expert learns
   patterns only that task cares about, without getting trampled by
   gradients from other tasks. The surface area where conflicts can
   happen (shared parameters) shrinks.
2. **Expert Collapse is blocked structurally.** A task-private
   expert only ever sees its own task's gradient. The scenario "every
   task converges onto the same expert" becomes physically impossible.
3. **Role differentiation becomes automatic.** The Shared Expert ends
   up carrying "parameters that must remain useful to every task,"
   while the Task Expert carries "parameters that only need to help
   one task." The two kinds of representation separate by construction.

> **Historical context.** PLE was proposed by *Tang, Liu, Zhao & Gong
> (RecSys 2020)* on Tencent's video recommender system, where VCR,
> VTR, Share Rate and other engagement metrics had to be optimized
> simultaneously. MMoE was in production, but Expert Collapse and
> seesaw were severe — PLE was reported as the first MTL architecture
> to improve every task simultaneously relative to MMoE. It has since
> become the de facto industrial standard at Alibaba, JD.com,
> Kuaishou, ByteDance, and similar large platforms.

## Decision 2 — make the expert pool heterogeneous

The original PLE paper, however, uses *identically structured small
MLPs* for both Shared and Task-specific experts. The "more experts,
more capacity" idea survives, but no individual expert carries a
different inductive bias — what differs across experts is only the
gate weight each one receives.

That setup was fine for Tencent Video. The tasks there (VCR, VTR,
Share Rate) were essentially variants of the same user–item
engagement prediction, so a homogeneous pool was expressive enough.
But the 13 tasks in this project are far more heterogeneous — next
best product, churn prediction, customer value tiering, similar
customer search, brand prediction, LTV regression. Feeding all of
those through seven identical MLPs gives the gate very little
structural information to discriminate on. "MLP-3 helped CTR, MLP-5
helped Churn" is a signal hard to separate from noise.

So we went a step further: **make the Shared Expert pool
heterogeneous.** The seven experts are each chosen to represent a
structurally distinct mathematical perspective.

- *Hyperbolic geometry* (unified_hgcn) — hierarchies in hyperbolic space
- *Persistent homology* (perslay) — topological shape of transaction patterns
- *Factorization machine* (deepfm) — symmetric pairwise feature crosses
- *Temporal dynamics* (temporal) — time-series patterns
- *Bipartite graph* (lightgcn) — customer–merchant collaborative signal
- *Causal inference* (causal) — do-operator level features
- *Optimal transport* (optimal_transport) — distances between distributions

Three reasons to do this.

**Encode expressiveness as inductive bias, not parameter count.** A
single 12GB VRAM card cannot support multiple Transformer-scale
experts. If each expert borrows a structure that is already
well-optimized in its home domain (HGCN's hyperbolic geometry,
PersLay's persistent homology), we buy a lot of representational
capacity per parameter. Given a fixed parameter budget, structural
diversity beats depth for our problem.

**Make explainability structural, not post-hoc.** "unified_hgcn
contributed 35%, temporal contributed 28%" is not a SHAP
approximation — it is the *actual gate weight computed by the model*,
and the names are business-readable ("hierarchy", "temporal
pattern"). A homogeneous MLP ensemble cannot do this; "MLP-3
contributed 28%" is an explanation neither customers nor regulators
can read.

**Role differentiation across tasks becomes sharper.** Homogeneous
experts tend to converge onto similar features during training
(another face of Expert Collapse). Heterogeneous experts each carry
their own structural bias and naturally differentiate. When the gate
decides which lens to look through, it picks from a space of
genuinely distinguishable lenses.

> **Analogy — referral to a specialist.** A homogeneous MLP ensemble
> is like showing the same patient to seven internists and asking a
> gate "whose opinion do I listen to?" The opinions look too similar
> to pick between — the gate's decision degenerates into noise. A
> heterogeneous expert pool is seven genuinely different specialists
> — internist, surgeon, radiologist, psychologist, physiatrist,
> emergency physician, pharmacist — and the symptom structurally
> dictates which perspective matters.

This decision underwrites everything below. The seven specific
experts are introduced one by one in **PLE-3**, and the new problems
that heterogeneous output dims (64D / 128D) introduce are handled by
the two-stage CGCAttention construction plus `dim_normalize` in
**PLE-4**.

## Decision 3 — CGC: a softmax-weighted gate

With shared/task separation and a heterogeneous pool in place, the
*shape* of the gate is the next decision. Tang et al. propose
Customized Gate Control (CGC): task $k$'s gate computes a softmax
weighted sum on a concat axis combining the Shared pool output and
Task-$k$'s own expert output.

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^N$$

Why softmax weighting — the alternatives we considered:

- **Hard selection (top-1).** "Use only the highest-scoring expert."
  Compute-cheap but non-differentiable; needs straight-through
  estimators. More importantly, it cannot express natural mixtures
  like "task A wants 60% Shared and 40% Task."
- **Independent sigmoid weights.** "Each expert gets an independent
  0–1 score." No sum constraint, so degenerate solutions where
  everything goes to 1 or 0 are hard to rule out.
- **Attention ($\mathbf{QK}^T / \sqrt{d}$).** CGC is essentially a
  special case of Attention (Query = task projection, Key = expert
  representation, Value = expert output), so this is what we pick.
  The softmax-weighted sum is the shadow cast by Attention
  mathematics.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

In CGC, the Query is task $k$'s gate weight, the Key is the shared
representation $\mathbf{h}_{shared}$, and the Value is each expert's
output block. The only difference from a Transformer is that
attention is computed between *experts*, not between tokens in a
sequence. Same math, different unit.

> **Three reasons to prefer softmax.** (1) *Positivity*: $e^x > 0$,
> so there are no negative-weight pathologies. (2) *Sum to 1*: the
> weights form a probability distribution, directly readable as "the
> attention this task pays to each expert for this input." (3)
> *Differentiation is clean*: $\frac{d}{dx} e^x = e^x$, gradients
> are simple. For the same reasons softmax is the default
> classification head in undergraduate ML, it is the default gate for
> expert mixing too.

## Experts and gates, in three angles

The three decisions above (split, heterogeneity, softmax weighting)
compose into a system. Here it is from three different angles.

### Experts as different lenses

The seven Shared Experts are *the same customer data* read through
seven very different perspectives.

- **unified_hgcn**: product/category hierarchy in *hyperbolic space*.
- **perslay**: the *topological shape* of transaction data.
- **deepfm**: *crossed interactions* between features.
- **temporal**: *temporal patterns* and dynamics.
- **lightgcn**: customer–merchant *graph relations*.
- **causal**: *directional causal* structure between features.
- **optimal_transport**: *distances between distributions*.

Some tasks (CTR) care most about temporal patterns; others (Brand
Prediction) need hierarchy. Even given the same customer data, which
lens a task looks through is allowed to differ — that is the starting
premise of the heterogeneous pool.

#### Comparing the seven Shared Experts: input, target, irreplaceability

| Expert | Input | Learning target | Why it can't be replaced by another Expert | Output dim |
|---|---|---|---|---|
| DeepFM | Normalized 644D | Symmetric feature-pair interactions | Explicitly captures FM's $O(nk)$ 2nd-order crosses | 64D |
| LightGCN | Pre-computed 64D | Customer–merchant collaborative signal | "Similar customer" patterns via the bipartite graph | 64D |
| Unified HGCN | Pre-computed 47D | Merchant hierarchical structure (merchant nodes only) | MCC tree in hyperbolic space + co-visit correction | 128D |
| Temporal | Sequence $[B,180,16]$ + $[B,90,8]$ | Temporal pattern shifts | Mamba + LNN + Transformer ensemble | 64D |
| PersLay | Persistence Diagram | Topological structure | Loops / clusters / branches in consumption patterns | 64D |
| Causal | Normalized 644D | Directional causal structure between features (DAG) | Confounder removal, asymmetric causal structure | 64D |
| OT | Normalized 644D | Customer–prototype distribution distance | Encodes distributional geometry via the Wasserstein distance | 64D |

> **Why all seven Experts are needed.** The seven each capture a
> different facet of the same customer, and the CGC Gate learns the
> per-task optimal combination. Three Experts (DeepFM, Causal, OT)
> take the same normalized 644D as input, but extract fundamentally
> different mathematical structures — *symmetric / asymmetric /
> distance* — so they are not redundant.

### The gate as an attention distribution

The gate is an *attention mechanism* that decides, per task, how much
to trust each expert's opinion. The equation is above; unpacked:

- $\mathbf{W}_k \cdot \mathbf{h}_{shared}$: look at the current input
  and *score* each expert's relevance.
- $\text{Softmax}$: convert scores into a probability distribution
  summing to 1.
- $\mathbf{w}_k \in \mathbb{R}^7$: the *trust vector* that task $k$
  assigns to the 7 experts.

> **Analogy — a medical diagnosis committee.** Seven specialists
> examine a patient, each from their own field. The gate is the
> *primary physician* deciding whose opinion carries what weight
> when judging this patient's condition. Cardiac symptoms weight the
> internist, trauma weights the surgeon — the same picture as "each
> task needs a different mix of lenses," told without equations.

### PLE as a mixture-density model

Why is a weighted sum of expert outputs so strong? From a
function-approximation angle, each expert is a *basis function*
specialized for a region of input space.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot \mathbf{e}_i(\mathbf{x})$$

The gate plays the role of mixing coefficient — the same structure
as a statistical *mixture density model*:

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^N \pi_i(\mathbf{x}) \cdot p_i(\mathbf{y} \mid \mathbf{x})$$

Here $\pi_i(\mathbf{x})$ corresponds to the gate and $p_i$ to the
expert. Because each expert covers a different region of input space,
the overall model can *efficiently approximate more complex functions*
than a single network could. Using a heterogeneous pool makes that
"different regions" argument structural: the basis functions already
live in genuinely different mathematical spaces.

> **Recent trends.** Mixture of Experts (MoE) has emerged as a core
> architecture in 2024–2025 LLMs. Mistral's *Mixtral 8x7B* (2023),
> Google's *Switch Transformer* (Fedus et al., 2022), DeepSeek's
> *DeepSeek-MoE* (2024) are representative. They use Sparse MoE
> (top-k selection) to cut compute sharply vs. parameter count. This
> system's CGC is Dense MoE (all experts active) — acceptable because
> with only seven experts the compute saving from sparsity is small.

## But this design introduces still more problems

With the three decisions — explicit separation, heterogeneous pool,
softmax gate — MMoE's collapse is prevented. But new problems appear.

**One: heterogeneous output dims.** unified_hgcn is 128D; the other
six are 64D. Equal gate weights produce unequal L2 contributions, so
a subtle form of collapse — leaning toward the bigger expert — is
still available. PLE-4 fixes this with `dim_normalize`.

**Two: the Shared concat needs per-task recoloring.** The original
CGCLayer mixes Shared + Task experts along one axis, but our setup
also needs to let each task *repaint* the 512D Shared concat
differently. So a second stage — CGCAttention — is layered
orthogonally on top of the paper's CGCLayer. Also a PLE-4 subject.

**Three: the initial gate knows nothing.** From random init, if one
expert lucks into slightly better early gradients, we can converge
back into collapse. So we combine `domain_experts`-based bias
initialization (CTR → PersLay + Temporal + UHGCN, Brand_prediction
→ UHGCN) with **entropy regularization** as a second line of defense
— also PLE-4.

**Four: task dependencies are not yet expressed in the gate.** CTR
should inform CVR, Churn should inform Retention, but the CGC gate
only handles "how to mix experts." Cross-task signal passing goes
through a separate path — Logit Transfer — in PLE-5.

**Five: customers live at multiple time scales simultaneously.**
Clicks are daily. Churn risk is monthly. Lifestyle is yearly.
Handling all three inside one Shared pool is hard. HMM Triple-Mode
routing — also PLE-4.

## Where did "Progressive" go

The original PLE paper stacks multiple Extraction Layers
*progressively*. The task $k$ output at the $l$-th layer is:

$$\mathbf{h}_k^{(l)} = \text{Gate}_k^{(l)}\left(\mathbf{E}^{s,(l)}(\mathbf{h}^{(l-1)}), \mathbf{E}^{k,(l)}(\mathbf{h}_k^{(l-1)})\right)$$

Each layer the Shared representation refines "information useful to
every task" progressively, and the Task representation sharpens its
specialization. It is the same progression as a CNN refining edges
→ textures → objects → semantics.

This implementation uses a single Extraction Layer. Instead, the
four-stage pipeline — *CGC → GroupTaskExpertBasket → Logit Transfer
→ Task Tower* — splits the progressive-refinement role across
stages. CGC recolors the shared representation; GroupTaskExpertBasket
does task-private refinement; Logit Transfer carries explicit
task-to-task dependencies; Task Tower compresses to final output. We
spread depth across the pipeline rather than stacking it inside one
layer.

## Summary of PLE-1 → PLE-2, and what's next

| Stage | Problem | Fix | New problem |
|---|---|---|---|
| Shared-Bottom | Gradient conflict across tasks on a shared trunk | — | — |
| MMoE | Multiple experts + per-task gates | Expert Collapse (symmetric pool) |
| **PLE (2020)** | **Explicit Shared / Task split + CGC** | **Collapse blocked structurally** | Narrow division-of-labor in a homogeneous pool |
| **This project** | **7 heterogeneous experts + 2-stage CGCAttention** | **Structural division + explainability** | Heterogeneous dims, gate init, task dependencies, time scales |

PLE-3 takes the seven experts one by one — DeepFM, LightGCN, Unified
HGCN, Temporal, PersLay, Causal, Optimal Transport — and shows what
mathematical lens each one uses on the same customer. How we actually
wire them into the gate is PLE-4.
