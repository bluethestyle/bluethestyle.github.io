---
title: "[Study Thread] PLE-2 — Progressive Layered Extraction: Explicit Expert Separation and CGC Gates"
date: 2026-04-19 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, tang2020, mtl]
lang: en
series: study-thread
part: 2
alt_lang: /2026/04/19/ple-2-progressive-layered-extraction-ko/
next_title: "PLE-3 — Input Structure and Heterogeneous Shared Expert Pool (576D)"
next_desc: "The full PLEClusterInput field spec, 734D feature-tensor index mapping, and HMM mode routing. Plus the eight heterogeneous Shared Experts this project actually runs (Economics, Temporal, HMM, TDA, GMM, GCN, PersLay, UnifiedHGCN) — each interpreting the customer through a structurally different mathematical lens."
next_status: published
---

*PLE-2 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document. This second post is
about how PLE (Tang et al., RecSys 2020) fixes the failure modes of
Shared-Bottom and MMoE — the explicit balance between sharing and
separation, the roles of Experts and Gates, what the equations actually
say, and the narrative of "why PLE."*

## PLE: The Explicit Balance Between Sharing and Separation

Tang et al. (RecSys 2020)'s PLE splits experts into two kinds.

- *Shared Expert* ($\mathcal{E}^s$): a common expert that every task can access.
- *Task-specific Expert* ($\mathcal{E}^k$): a private expert that only task $k$ can access.

Each task's gate takes both the Shared Expert and its own private expert
as inputs, and learns the optimal combination ratio.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

What this design solves:

1. **Mitigates Negative Transfer**: the task-specific Expert can learn
   patterns that only that task cares about, without interference.
2. **Prevents Expert Collapse**: the Shared Expert is forced to learn
   information that must be useful to every task, while the Task Expert
   learns specialized information — roles separate naturally.
3. **Progressive**: multiple Extraction Layers can be stacked so that
   per-task representations are refined progressively, low-level to
   high-level.

> **Historical context.** PLE was proposed by *Tang, Liu, Zhao & Gong
> (RecSys 2020)* on Tencent's video recommender system. At the time,
> Tencent Video had to simultaneously optimize multiple engagement
> metrics: VCR (Video Completion Rate), VTR (Video Through Rate), Share
> Rate, and so on. MMoE was already in production, but Expert Collapse
> and the seesaw effect were severe, and PLE was born from the idea "what
> if we explicitly split Experts into shared and private ones?" In the
> paper, PLE is reported as the first MTL architecture to improve every
> task simultaneously relative to MMoE. It has since been adopted widely
> at Alibaba, JD.com, Kuaishou, ByteDance and other large Chinese
> platforms, and is now a de facto standard for industrial MTL.

## Where this project diverges from the paper — from homogeneous MLPs to heterogeneous experts

In the original PLE paper (Tang et al., 2020), both the Shared Experts
and the Task-specific Experts are *identically structured small MLPs*.
The idea of "more experts for richer representation" survives, but no
individual expert carries a different inductive bias from the others —
what differs across experts is only the gate weight each one receives.

This project takes one step further. **We make the Shared Expert pool
heterogeneous.** The eight Shared Experts are each chosen to represent
a *structurally distinct mathematical perspective*.

- *Hyperbolic geometry* (unified_hgcn) — hierarchies in hyperbolic space
- *Persistent homology* (perslay) — topological shape of transaction patterns
- *Factorization machine* (deepfm) — symmetric pairwise feature crosses
- *Temporal dynamics* (temporal) — time-series patterns
- *Bipartite graph* (lightgcn) — customer-product relations
- *Causal inference* (causal) — do-operator level features
- *Optimal transport* (optimal_transport) — distances between distributions
- *Power-law raw scale* (raw_scale) — finance-native scales preserved before normalization (v3.3)

### Why this choice

Three reasons.

**Encode expressiveness through inductive bias, not parameter count.** A
single 12GB VRAM card cannot support several Transformer-scale experts
stacked on top of each other. If each expert instead borrows a
structure that is already well-optimized in its home domain (HGCN's
hyperbolic geometry, PersLay's persistent homology, and so on), a
great deal of expressive capacity is bought per parameter.

**Make explainability structural, not post-hoc.** "unified_hgcn
contributed 35%, temporal contributed 28%" is not a SHAP approximation
— it is the *actual gate weight computed by the model*. And the
names themselves are business-readable ("hierarchy", "temporal
pattern"), which is not possible with a homogeneous MLP ensemble —
"MLP #3 contributed 28%" is no explanation for a customer or a
regulator.

**Strengthen inter-task regularization.** Homogeneous experts tend to
converge onto similar features during training (a subtler face of
Expert Collapse). Heterogeneous experts each carry their own
structural inductive bias, so their roles differentiate naturally.
When a gate decides "which lens to look through," it picks from a
space of genuinely distinguishable lenses.

> **Paper vs. implementation — the lesson.** The PLE paper was born
> from Tencent Video's engagement prediction, where homogeneous MLPs
> already beat MMoE meaningfully — because all tasks there (VCR, VTR,
> Share Rate) predicted essentially the same kind of user-item
> engagement. This project's 13 tasks are far more heterogeneous (next
> best product, churn prediction, customer value tiering, similar-customer
> search, and so on). Reflecting that heterogeneity *at the expert
> structure level* turned out to be the right move.

This decision is the premise for everything in the sections below —
the composition of the eight heterogeneous experts, the block-scaling
CGCAttention gate variant, and the dimension-normalization trick for
heterogeneous output dims.

## The Roles of Expert and Gate — An Intuitive View

### Expert: domain specialists looking at the world through different lenses

An Expert is a *specialist* that interprets the input data through one
specific perspective. This system's eight Shared Experts each provide a
view from a completely different domain.

- **unified_hgcn**: interprets product/category hierarchy in *hyperbolic space*.
- **perslay**: captures the *topological shape* of transaction data.
- **deepfm**: learns *crossed interactions* between features.
- **temporal**: captures *temporal patterns* and dynamics.
- **lightgcn**: represents *customer-item graph relations*.
- **causal**: extracts *causal relations* between features.
- **optimal_transport**: measures *distances between distributions*.
- **raw_scale**: preserves the *power-law distribution pattern* of pre-normalization raw features (v3.3).

Each is a different "lens" through which the same customer data is
viewed. Some tasks (like CTR) care about temporal patterns, while others
(like Brand Prediction) care about hierarchical relations.

#### Comparing the eight Shared Experts: input, learning target, irreplaceability

| Expert | Input | Learning target | Why it can't be replaced by another Expert | Output dim |
|---|---|---|---|---|
| DeepFM | Normalized 644D | Symmetric feature-pair interactions | Explicitly captures FM's $O(nk)$ 2nd-order crosses | 64D |
| LightGCN | Pre-computed 64D | Customer–merchant collaborative signal | "Similar customer" patterns via the bipartite graph | 64D |
| Unified HGCN | Pre-computed 47D | Merchant hierarchical structure (merchant nodes only) | MCC tree in hyperbolic space + co-visit correction | 128D |
| Temporal | Sequence $[B,180,16]$ + $[B,90,8]$ | Temporal pattern shifts | Mamba + LNN + Transformer ensemble | 64D |
| PersLay | Persistence Diagram | Topological structure | Loops / clusters / branches in consumption patterns | 64D |
| Causal | Normalized 644D | Directional causal structure between features (DAG) | Confounder removal, asymmetric causal structure | 64D |
| OT | Normalized 644D | Customer–prototype distribution distance | Encodes distributional geometry via the Wasserstein distance | 64D |
| RawScale | Raw 90D | Power-law distribution pattern (v3.3) | Preserves raw scale / distribution information lost to normalization | 64D |

> **Why all eight Experts are needed.** The eight Experts each capture a
> different facet of the same customer, and the CGC Gate learns the
> per-task optimal combination. Three Experts (DeepFM, Causal, OT) take
> the same normalized 644D as input, but extract fundamentally different
> mathematical structures — symmetric / asymmetric / distance — so they
> are not redundant. RawScaleExpert (v3.3) takes the pre-normalization
> raw 90D features as input and preserves power-law distribution
> information.

### Gate: how much should we listen to each specialist?

The Gate is an *attention mechanism* that decides, per task, "how much do
we trust each Expert's opinion?"

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

What this equation says is clear.

- $\mathbf{W}_k \cdot \mathbf{h}_{shared}$: look at the current input and *score* each Expert's relevance.
- $\text{Softmax}$: convert scores into a probability distribution summing to 1.
- $\mathbf{w}_k \in \mathbb{R}^8$: the *trust vector* that task $k$ assigns to the 8 Experts.

> **Analogy — a medical diagnosis committee.** Eight specialists
> (Experts) diagnose a patient (input data) each from their own field.
> The internist, the surgeon, the radiologist, and so on each write an
> opinion (Expert output). The Gate is the *primary doctor* deciding
> "when judging this patient's condition, how much weight should each
> specialist's opinion carry?" For cardiac symptoms, the internist's
> opinion gets high weight; for trauma, the surgeon gets high weight.

> **Undergraduate math — why does Softmax use $e^x$ (the natural
> exponential)?** Softmax turns an arbitrary real-valued vector into a
> probability distribution (positive, summing to 1):
> $\text{Softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. Three reasons to
> pick $e^x$: (1) *Positivity*: $e^x > 0$ for all real $x$, so there is
> no "negative probability" problem. (2) *Monotone increasing*: if
> $z_i > z_j$ then $e^{z_i} > e^{z_j}$, so score ordering is preserved.
> (3) *Ease of differentiation*: $\frac{d}{dx} e^x = e^x$, so gradient
> computation stays clean. *Concrete calculation*: a score vector
> $\mathbf{z} = [2.0, 1.0, 0.1]$ gives $e^{\mathbf{z}} = [7.39, 2.72, 1.11]$,
> with sum $= 11.22$ and $\text{Softmax} = [0.659, 0.242, 0.099]$ —
> score gaps have been translated into probability gaps. The larger the
> gap, the more sharply the probabilities diverge, thanks to the
> exponential growth of $e^x$. You could use another positive function
> (e.g. $x^2$), but the ordering breaks for negative inputs
> ($(-3)^2 > (-1)^2$) and the gradient vanishes near zero, so $e^x$ is
> the optimal choice.

## Mathematical Considerations — What the Equations Are Actually Saying

### The link between gating and attention

The Softmax weighted sum of a CGC gate has the exact same structure as
the Attention mechanism of the Transformer.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

In CGC:

- **Query**: task $k$'s gate weight $\mathbf{W}_k$ (what this task wants to know).
- **Key**: the shared representation $\mathbf{h}_{shared}$ (each Expert's relevance to the current input).
- **Value**: each Expert's output block $\mathbf{h}_i^{expert}$ (the actual information).

The difference is that a Transformer computes attention between tokens
in a sequence, while CGC computes *attention between Experts*. The
underlying mathematical principle is identical — "combine information
selectively in proportion to relevance" — applied to a different unit
(token vs. Expert).

### The function-approximation meaning of a weighted sum of Experts

Why is a weighted sum of Expert outputs so powerful? You can understand
it from the perspective of function approximation.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot \mathbf{e}_i(\mathbf{x})$$

Each Expert $\mathbf{e}_i$ can be viewed as a *basis function* specialized
for a particular region of the input space. The Gate $g_{k,i}$ is the
*mixing coefficient* of those basis functions. This is exactly what the
name Mixture of Experts carries — it is a *Mixture of Experts model*.

It has precisely the same structure as a statistical *mixture density
model*.

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^N \pi_i(\mathbf{x}) \cdot p_i(\mathbf{y} \mid \mathbf{x})$$

Here $\pi_i(\mathbf{x})$ corresponds to the gate and $p_i$ to the
Expert. Because each Expert covers a different region of input space,
the overall model can *efficiently approximate much more complex
functions* than a single network could.

> **Undergraduate math — the function-approximation meaning of weighted
> sums.** In mathematics, expressing an arbitrary function as a linear
> combination (weighted sum) of basis functions is the core of
> approximation theory. A classical example is Fourier series
> $f(x) = \sum a_n \cos(nx) + b_n \sin(nx)$, where a weighted sum over
> the trigonometric basis can approximate *any periodic function*. The
> Expert weighted sum $\mathbf{h} = \sum g_i \cdot \mathbf{e}_i(\mathbf{x})$
> runs on the same principle. Each Expert $\mathbf{e}_i$ is a "basis
> function" specialized to a particular pattern in input space, and the
> gate $g_i$ plays the role of mixing coefficient. The difference is
> that the basis itself is learned too. *Concrete example*: if the input
> $\mathbf{x}$ is a customer with strong temporal patterns,
> $g_{\text{temporal}} = 0.4$ grows larger; if graph relations matter,
> $g_{\text{lightgcn}} = 0.3$ grows larger. The resulting representation
> $\mathbf{h}$ is a mix of specialist opinions tuned to this particular
> customer.

> **Current trends.** Mixture of Experts (MoE) has emerged as a core
> architecture in the 2024-2025 LLM space. Mistral's *Mixtral 8x7B*
> (2023), Google's *Switch Transformer* (Fedus et al., 2022), and
> DeepSeek's *DeepSeek-MoE* (2024) are representative examples. These
> use Sparse MoE (top-k selection) to cut compute sharply relative to
> parameter count. In recommenders too, Alibaba's *Star Topology
> Adaptive Recommender* (STAR, CIKM 2021) and Kuaishou's *PEPNet* (KDD
> 2023) adopt MoE structures that select Experts depending on the input
> condition. This system's CGC gating belongs to the Dense MoE family
> (all Experts are used), and since the Expert count is small (8),
> compute efficiency holds even without sparse selection.

### How a Progressive structure affects information flow

The original PLE paper stacks multiple Extraction Layers. The task $k$
output of the $l$-th layer is:

$$\mathbf{h}_k^{(l)} = \text{Gate}_k^{(l)}\left(\mathbf{E}^{s,(l)}(\mathbf{h}^{(l-1)}), \mathbf{E}^{k,(l)}(\mathbf{h}_k^{(l-1)})\right)$$

As you pass through each layer:

1. The **Shared representation** *progressively refines* information that is useful to every task.
2. The **Task representation** becomes increasingly *specialized* to the task.
3. Gates are trained independently per layer, so *different combination strategies per abstraction level* are possible.

This is the same principle as a CNN refining information from low-level
(edges, textures) to high-level (objects, semantics).

This implementation uses a single Extraction Layer, but the three-stage
pipeline of *CGC → GroupTaskExpertBasket → Logit Transfer* effectively
performs the same role of Progressive information refinement.

## An Intuitive Reading of the Core Equations

This section interprets what each of the main equations in this document
actually means — "why is this equation needed here?" — from a
practitioner's viewpoint.

### PLE gating combination

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

**Reading**: "Task $k$'s representation is the weighted sum of the
shared specialists' opinions plus the weighted sum of its own private
specialist's opinions."

The first sum $\sum g_{k,i}^s \cdot \mathbf{e}_i^s$ selects only the
parts of *shared knowledge* that are useful to task $k$, and the second
sum $\sum g_{k,j}^k \cdot \mathbf{e}_j^k$ fetches specialized information
from *private knowledge*. The larger the gate value $g$, the louder the
corresponding Expert's voice.

### CGC Attention weights

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

**Reading**: "Look at the current input ($\mathbf{h}_{shared}$), score
the relevance of each of the 8 Experts, then normalize them with
Softmax. Since the Softmax output sums to 1, this is exactly 'the
attention distribution that task $k$ assigns to the Experts for the
current input'."

The role of the initial bias ($\mathbf{b}_k$) matters. Experts listed as
`domain_experts` get `bias_high=1.0`, the rest get `bias_low=-1.0`, so
that early in training attention concentrates on the Experts that
already match domain knowledge. As training proceeds, $\mathbf{W}_k$
updates and the distribution adjusts to fit the data.

### Entropy regularization

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left(-\frac{1}{|\mathcal{T}|}\right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k), \quad H(\mathbf{w}_k) = -\sum_{i=1}^{8} w_{k,i} \cdot \log(w_{k,i})$$

**Reading**: "If the gate distribution has low entropy (concentrated on
one Expert), penalize it — at the very least, encourage the task to make
balanced use of multiple Experts."

Why is this needed? Without entropy regularization, the gate quickly
converges onto the single Expert with the largest gradient signal, and
the other Experts stop training — *Expert Collapse*. This loss term is
the constraint "at least consult every specialist."

### Focal Loss

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

**Reading**: "Shrink the loss of easy examples that are already well
predicted ($p_t \approx 1$) almost to zero, and concentrate the learning
budget on hard examples that are still wrong ($p_t \approx 0$)."

The $(1 - p_t)^\gamma$ term is the heart of it. For $p_t = 0.9$ (well
predicted) we get $(1 - 0.9)^2 = 0.01$, a 100× down-weighting. For
$p_t = 0.1$ (wrong), $(1 - 0.1)^2 = 0.81$, i.e. essentially full weight.
Larger $\gamma$ attenuates easy examples more aggressively.

$\alpha_t$ corrects class imbalance. The Churn task's $\alpha = 0.6$
encodes the business judgement "missing a churner costs more than
misclassifying a non-churner, so give positive (churn) examples more
weight."

> **Undergraduate math — the attenuation effect of $(1-p_t)^\gamma$.**
> Raising a number in $[0, 1]$ to a higher power drives the value toward
> zero more quickly. That is the central mechanism of Focal Loss. For
> $\gamma = 0$, $(1-p_t)^0 = 1$ and Focal Loss $=$ standard
> cross-entropy (no attenuation). For $\gamma = 1$, linear attenuation:
> $p_t = 0.9 \to 0.1$, $p_t = 0.5 \to 0.5$. For $\gamma = 2$, quadratic
> attenuation: $p_t = 0.9 \to 0.01$, $p_t = 0.5 \to 0.25$. For
> $\gamma = 5$, aggressive attenuation: $p_t = 0.9 \to 0.00001$,
> $p_t = 0.5 \to 0.03$. So larger $\gamma$ means "ignore easy (high
> $p_t$) examples more strongly" and focus on the hard ones. In
> practice $\gamma = 2$ is by far the most common — a moderate setting
> that reduces the contribution of well-predicted examples by roughly
> 100×.

### Uncertainty Weighting

$$\mathcal{L}_k^{uw} = w_k \cdot (\exp(-s_k) \cdot \mathcal{L}_k + s_k)$$

**Reading**: "Automatically lower the weight of intrinsically hard
tasks (high uncertainty) and raise the weight of easy ones. The model
learns this balance on its own."

$s_k = \log(\sigma_k^2)$ is the *learnable uncertainty* of task $k$.

- Large $s_k$ (high uncertainty): $\exp(-s_k)$ shrinks and the loss
  contribution shrinks with it. At the same time $+s_k$ grows, which
  prevents $s_k$ from running off to infinity.
- Small $s_k$ (low uncertainty): $\exp(-s_k)$ grows and the loss is
  reflected strongly.

This mechanism reduces the need to hand-tune per-task weights. Tuning
the weights of 16 tasks one by one is combinatorially explosive;
Uncertainty Weighting replaces that with *automatic balancing*.

> **Undergraduate math — why $\exp$ and $\log$ always come as a pair.**
> $\exp(x) = e^x$ and $\log(x) = \ln(x)$ are inverses of each other:
> $\exp(\log(x)) = x$, $\log(\exp(x)) = x$. In Uncertainty Weighting,
> defining $s_k = \log(\sigma_k^2)$ and using $\exp(-s_k)$ gives two
> benefits: (1) $\sigma_k^2$ (variance) must be positive, but $s_k$ is
> an unconstrained real, so optimization is easier. (2)
> $\exp(-s_k) = \exp(-\log(\sigma_k^2)) = 1/\sigma_k^2$ is the
> *precision*. *Concrete calculation*: $s_k = 0$ gives
> $\exp(-s_k) = 1$ (standard loss, unchanged). $s_k = 2$ gives
> $\exp(-2) \approx 0.135$ (only 13.5% of the loss flows through —
> uncertain task is attenuated). $s_k = -1$ gives $\exp(1) \approx 2.718$
> (loss amplified 2.7× — confident task is emphasized). At the same
> time the $+s_k$ regularizer adds $+2$ when $s_k = 2$, encoding the
> constraint "declaring high uncertainty has a cost."

### Evidential Uncertainty

$$u = \frac{K}{S}, \quad S = \sum_{k=1}^{K} \alpha_k$$

**Reading**: "If the model has gathered a lot of evidence, it is
confident ($S$ large, $u$ small); if it lacks evidence, it honestly says
'I don't know' ($S$ small, $u$ large)."

A standard Softmax always outputs a probability distribution for any
input. It will confidently predict on out-of-distribution data,
potentially leading to dangerous decisions. The Evidential approach
models "probability over probability" — the Dirichlet distribution —
which quantifies the uncertainty of the prediction itself.

### Soft Routing

$$\mathbf{e}_{cluster} = \sum_{c=0}^{19} p_c \cdot \mathbf{E}_c \in \mathbb{R}^{32}$$

**Reading**: "Customers at the boundary of GMM clusters are not
hard-assigned to a single cluster; instead, cluster embedding vectors
are mixed in proportion to membership probability. The mixed embedding
is combined with the GroupEncoder output and passed through the
TaskHead."

This fixes the discontinuity problem of hard clustering. A customer at
the boundary of cluster 0 and cluster 1, if hard-assigned to id=0,
gets no access to cluster 1's knowledge. Soft routing mixes the
embeddings proportionally (e.g. $p_0 = 0.6$, $p_1 = 0.4$), making
boundary-customer representations much more stable.

## The Overall Narrative — "Why PLE"

### The arc of the story

**Starting point**: we have to predict 16 tasks at once. Training 16
independent models is data-inefficient and throws away the common
patterns between them.

**First attempt (Shared-Bottom)**: we built a single shared network.
Some tasks improved, but fundamentally different tasks like CTR and
Churn interfered with each other's training — *Negative Transfer*.

**Second attempt (MMoE)**: we put multiple Experts in front of a gate
and let the task choose. But all the gates ended up picking the same
Expert — *Expert Collapse* — and the model degenerated back into
something indistinguishable from Shared-Bottom.

**The fix (PLE)**: *explicitly split* Experts into shared and
task-specific ones, and let a CGC gate combine them at the optimal
ratio. The Shared Experts carry the base knowledge useful to every
task; the task-specific Experts carry the specialized knowledge only
one task needs.

**Extension (this project)**: on top of PLE's idea, we added 8
heterogeneous domain Experts (GCN, TDA, DeepFM, Temporal, Graph,
Causal, OT, RawScale), a GroupEncoder $+$ ClusterEmbedding (4 groups,
20 clusters), HMM Triple-Mode routing, the Logit Transfer chain,
Evidential uncertainty, and SAE interpretability — completing the
*PLE-Cluster-adaTT system specialized for AIOps recommendations*.

### A summary of design principles

| Principle | Implementation |
|---|---|
| Balance of sharing and separation | 8 Shared Experts + CGC gate + GroupTaskExpertBasket |
| Minimizing inter-task interference | Expert separation + entropy regularization + domain_experts bias |
| Inter-task knowledge transfer | Logit Transfer (explicit) + adaTT gradient transfer (adaptive) |
| Per-cluster specialization | GroupEncoder + ClusterEmbedding + Soft Routing |
| Time-scale separation | HMM Triple-Mode (daily / monthly) |
| Uncertainty awareness | Evidential Deep Learning (Dirichlet) |
| Automatic balancing | Uncertainty Weighting (auto task weights) |
| Interpretability | SAE (2304D sparse latent) |

## Theoretical Background of PLE

### Paper reference

**[RecSys 2020]** Tang, H., Liu, J., Zhao, M., & Gong, X. *"Progressive
Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for
Personalized Recommendations."*

### The Negative Transfer problem

The most serious problem in multi-task learning (MTL) is *Negative
Transfer*: loosely-related tasks pollute the representation space so
badly that joint training does *worse* than single-task training.

> **⚠ What Negative Transfer actually looks like.** In an AIOps system,
> CTR (click-through rate) and Churn (attrition) have to learn
> fundamentally different patterns — CTR is about short-term engagement,
> Churn is about long-term attrition signals. Under Shared-Bottom, when
> these two tasks share the same representation, one side's gradient
> disrupts the other's training — the *seesaw effect*.

### Comparing Shared-Bottom, MMoE, and PLE

| Aspect | Shared-Bottom | MMoE | PLE |
|---|---|---|---|
| Expert structure | Single Shared trunk | N Experts all shared | Shared Expert + Task-specific Expert, explicitly split |
| Gating | None | Per-task Softmax gate | CGC: gate combining Shared + Task Experts |
| Negative Transfer | High (every task interferes) | Moderate (Expert Collapse possible) | Low (explicit separation minimizes interference) |
| Expert Collapse | N/A | High (all tasks pick the same Expert) | Low (Shared/Task Experts are separated) |
| Scalability | Limited | Handled by adding Experts | Handled by adding Extraction Layers |

### The PLE gating formula

In PLE, the output for task $k$ is determined by a gated combination of
the Shared Expert set $\mathcal{E}^s$ and the Task-specific Expert set
$\mathcal{E}^k$.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

- $\mathbf{e}_i^s$: the $i$-th Shared Expert's output; $\mathbf{e}_j^k$: task $k$'s $j$-th Task Expert output.
- $g_{k,i}^s, g_{k,j}^k$: CGC gate weights (Softmax-normalized).

> **This project's PLE variant.** The original PLE paper implements
> Task-specific Experts as independent per-task MLPs, but in this system
> we instead have *the CGC Gate apply scale weights to the Shared Expert
> output blocks* and pass the result into a
> *GroupTaskExpertBasket (4 GroupEncoders + ClusterEmbedding)* — a
> two-stage structure. The GroupTaskExpertBasket takes over the role of
> the Task-specific Expert, achieving intra-group parameter sharing and
> per-cluster specialization simultaneously.
