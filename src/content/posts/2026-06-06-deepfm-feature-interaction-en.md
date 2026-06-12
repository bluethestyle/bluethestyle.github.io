---
title: "[Study Thread] DEEPFM-1 — Feature Interaction: Factorization Machines and the Shared-Embedding Trick"
date: 2026-06-06 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, deepfm, factorization-machine, feature-interaction, expert]
lang: en
excerpt: "The DeepFM sub-thread opens — why linear models miss the synergy between features, how a Factorization Machine learns every pairwise interaction with O(nk) latent vectors instead of O(n²) cross parameters, the O(nk) linearization trick, and how DeepFM shares one embedding between an FM head and a deep network. With the field layout and output dimension that wire this Expert into PLE."
series: study-thread
part: 13
alt_lang: /2026/06/06/deepfm-feature-interaction-ko/
next_title: "HGCN-1 — Curved Space for Hierarchies: Hyperbolic Graph Convolution"
next_desc: "Why a tree of merchants and customers refuses to embed cleanly in flat Euclidean space, how negative curvature buys exponential room, and how the unified_hgcn Expert reads graph structure in the Poincaré ball before handing 128D back to PLE."
next_status: draft
---

*First post of the DeepFM (Factorization Machine + deep network)
sub-thread in the "Study Thread" series. Across this and the following
posts, in parallel Korean and English, I unpack the DeepFM Expert — one
of the heterogeneous Shared Experts in this project. The source is the
on-prem reference `기술참조서/DeepFM_기술_참조서`, and the full PDF will
be attached to the final post of the sub-thread. Where the TDA
sub-thread asked what* shape *an Expert reads, this one asks something
more elementary and just as easy to get wrong: when two features only
matter* together *— a high-RFM customer* who also *leans digital — how
does a model see the* product *of features without drowning in
parameters? DeepFM's answer is a factorized inner product, shared
between a shallow and a deep half.*

> **One embedding, two readers.** DeepFM's whole trick is that a single
> set of field embeddings is consumed *twice* in parallel — once by a
> Factorization Machine that reads explicit 2nd-order interactions, once
> by a deep MLP that reads implicit higher-order ones. No separate
> feature pipeline, no hand-designed cross features (the headache that
> Google's Wide & Deep still carried), and end-to-end gradients flow
> into the *same* latent vectors from both sides. In this project the
> Expert ends one step further than the paper: it outputs not a scalar
> CTR but a **64D representation vector** that feeds the PLE gate.

## The Limit of Linear Models

The job of a recommendation model is blunt: will *this* customer take
*this* action? Individual features — age band, product category,
channel usage — carry signal on their own. But real behavior lives in
the relationships *between* features. "In their 30s" tells you a little.
"Leans digital" tells you a little. "In their 30s *and* leans digital"
tells you something neither does alone: a spike in online-investment
conversion that only appears when both fire together.

A linear model cannot see this. Writing $x_i$ for feature $i$,

$$ \hat{y} = w_0 + \sum_{i=1}^{n} w_i\, x_i $$

assumes every feature contributes *independently* and additively. There
is no term that turns on only when two features co-occur. To capture
synergy you must add the cross term $x_i x_j$ explicitly, which turns
the model into 2nd-order polynomial regression:

$$ \hat{y} = w_0 + \sum_{i=1}^{n} w_i\, x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} w_{ij}\, x_i x_j $$

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <!-- Linear: separate boxes summed -->
  <text x="140" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Linear — independent sum</text>
  <g>
    <rect x="55" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <rect x="122" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <rect x="189" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <text x="73" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₁</text>
    <text x="140" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₂</text>
    <text x="207" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₃</text>
    <text x="105" y="80" text-anchor="middle" font-size="14" fill="#64748b">+</text>
    <text x="172" y="80" text-anchor="middle" font-size="14" fill="#64748b">+</text>
  </g>
  <text x="140" y="135" text-anchor="middle" font-size="11" fill="#64748b">no x·x terms — synergy invisible</text>
  <!-- divider -->
  <line x1="280" y1="40" x2="280" y2="180" stroke="#e2e8f0" stroke-width="1"/>
  <!-- FM: pairwise lattice -->
  <text x="420" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">FM — pairwise lattice</text>
  <g fill="#0d9488"><circle cx="360" cy="62" r="7"/><circle cx="420" cy="62" r="7"/><circle cx="480" cy="62" r="7"/></g>
  <g stroke="#0d9488" stroke-width="1.4">
    <line x1="360" y1="62" x2="420" y2="62"/>
    <line x1="420" y1="62" x2="480" y2="62"/>
    <path d="M 360 68 Q 420 110 480 68" fill="none"/>
  </g>
  <text x="360" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₁</text>
  <text x="420" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₂</text>
  <text x="480" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₃</text>
  <text x="390" y="56" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₁,v₂⟩</text>
  <text x="450" y="56" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₂,v₃⟩</text>
  <text x="420" y="108" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₁,v₃⟩</text>
  <text x="420" y="135" text-anchor="middle" font-size="11" fill="#64748b">every pair carries a learned weight</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A linear model sums features in isolation; FM adds a weight to every pair. The question is how to afford those weights.</figcaption>
</figure>

## The Combinatorial Explosion

Adding cross terms is correct but expensive. With $n$ features there are
$n(n-1)/2$ cross parameters $w_{ij}$. On this project's normalized
feature space — **644D** in the V1-compatible layout — that is

$$ \frac{644 \times 643}{2} = 207{,}046 $$

cross parameters. Worse than the count is the *sparsity*: most $(i,j)$
pairs barely co-occur in the training data, so $w_{ij}$ cannot be
estimated stably. The cross-parameter matrix $W \in \mathbb{R}^{n\times n}$
is symmetric ($w_{ij}=w_{ji}$), and in a high-dimensional space "every
pair" grows quadratically — a face of the curse of dimensionality.

> **Historical context.** The fix descends straight from
> recommender-system matrix factorization. The 2006 Netflix Prize made
> SVD-style MF the workhorse of collaborative filtering (Funk's public
> SVD; Koren's BellKor ensemble in 2009). Steffen Rendle then
> generalized "user × item" to "any pair of features" and published
> *Factorization Machines* at ICDM 2010. FFM (2016), Wide & Deep
> (2016), and DeepFM (Guo et al., IJCAI 2017) followed — the line from
> hand-crafted cross features to fully automatic interaction learning.

## Factorization — From O(n²) to O(nk)

Matrix factorization's lesson: a low-rank product approximates a big
interaction matrix. FM applies it to the cross-parameter matrix
directly. Instead of storing each $w_{ij}$, give every feature $i$ a
*latent vector* $\mathbf{v}_i \in \mathbb{R}^k$ and approximate

$$ w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}, \qquad W \approx V V^{\!\top} $$

This is exactly a rank-$k$ approximation of the symmetric matrix $W$ —
the same idea as keeping the top $k$ singular values in a truncated SVD
(Eckart–Young), except FM learns it adaptively from data. The full
FM prediction becomes:

$$ \hat{y}_{\text{FM}} = w_0 + \sum_{i=1}^{n} w_i\, x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j $$

The payoff is dramatic. With $k=16$ on this project, the cross
parameters drop from ~207K to $n \times k = 644 \times 16 = 10{,}304$ —
roughly **20× fewer**.

| Approach | Cross parameters | Sparse-data behavior |
| --- | --- | --- |
| 2nd-order polynomial | $n(n-1)/2 = 207{,}046$ | fails — most pairs unseen |
| FM ($k=16$) | $n \times k = 10{,}304$ | strong — each $\mathbf{v}_i$ is shared |

Why does sharing rescue sparsity? Because $\mathbf{v}_i$ is trained
across *all* of feature $i$'s co-occurrences. Even if pair $(i,j)$ never
appears together, $\mathbf{v}_i$ was learned from $(i,k)$ and
$\mathbf{v}_j$ from $(j,l)$, so $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$
still yields a meaningful estimate. The inner product also has a clean
reading: $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \lVert\mathbf{v}_i\rVert\,\lVert\mathbf{v}_j\rVert\cos\theta_{ij}$
— a positive value is *synergy*, a negative value is *suppression*,
near-zero is *no interaction*.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="230" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Interaction = latent-vector inner product</text>
  <!-- origin -->
  <circle cx="100" cy="180" r="3" fill="#64748b"/>
  <text x="92" y="198" text-anchor="middle" font-size="10" fill="#64748b">0</text>
  <!-- synergy pair: same direction -->
  <line x1="100" y1="180" x2="200" y2="90" stroke="#0d9488" stroke-width="2"/>
  <line x1="100" y1="180" x2="225" y2="110" stroke="#0d9488" stroke-width="2"/>
  <text x="205" y="84" font-size="11" fill="#0d9488" font-weight="700">vᵢ</text>
  <text x="233" y="112" font-size="11" fill="#0d9488" font-weight="700">vⱼ</text>
  <path d="M 130 153 A 42 42 0 0 1 142 142" fill="none" stroke="#0d9488" stroke-width="1"/>
  <text x="160" y="150" font-size="10" fill="#0d9488">θ small → ⟨·,·⟩ &gt; 0 synergy</text>
  <!-- suppression pair: opposite -->
  <line x1="100" y1="180" x2="240" y2="180" stroke="#e11d48" stroke-width="2" stroke-dasharray="1 0"/>
  <line x1="100" y1="180" x2="120" y2="120" stroke="#e11d48" stroke-width="2"/>
  <text x="248" y="184" font-size="11" fill="#e11d48" font-weight="700">vₚ</text>
  <text x="108" y="114" font-size="11" fill="#e11d48" font-weight="700">v_q</text>
  <text x="300" y="180" font-size="10" fill="#e11d48">wide θ → ⟨·,·⟩ &lt; 0 suppression</text>
  <!-- caption math -->
  <text x="260" y="218" text-anchor="middle" font-size="11" fill="#64748b">⟨vᵢ, vⱼ⟩ = ‖vᵢ‖ ‖vⱼ‖ cos θᵢⱼ</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Each field becomes a point in 16D latent space. Aligned vectors interact positively (synergy); opposed vectors negatively (suppression).</figcaption>
</figure>

## The FM Trick — O(nk) in One Pass

There is still a problem hiding in the equation. Summed naively, the
pairwise term needs $n(n-1)/2$ inner products, each over $k$
dimensions — $O(n^2 k)$. A single algebraic identity collapses it to
$O(nk)$:

$$ \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j = \frac{1}{2}\sum_{f=1}^{k}\left[\left(\sum_{i=1}^{n} v_{i,f}\, x_i\right)^{\!2} - \sum_{i=1}^{n} \left(v_{i,f}\, x_i\right)^{2}\right] $$

> **Equation intuition.** The identity is just
> $\sum_{i<j} a_i a_j = \tfrac{1}{2}\big[(\sum_i a_i)^2 - \sum_i a_i^2\big]$,
> with $a_i = v_{i,f}\,x_i$. The *square of the sum* contains every
> self-term plus twice every cross-term; subtract the *sum of squares*
> (the self-terms) and twice the cross-terms is all that remains. So per
> latent dimension $f$ you compute one sum, square it, subtract a sum of
> squares — two linear passes — and you have every pairwise interaction.
> "Compute the whole, then remove the self-interaction."

In code this is `sum_sq − sq_sum`. And one project-specific twist
matters: the original FM collapses all $k$ channels into a single
scalar. This Expert *keeps the k-dimensional vector*. Each of the 16
channels carries a different view of field synergy:

$$ \mathbf{y}_{\text{FM}} = \frac{1}{2}\left[\left(\sum_{i=1}^{n} \mathbf{v}_i\right)^{\!2} - \sum_{i=1}^{n} \mathbf{v}_i^{2}\right] \in \mathbb{R}^{k},\qquad k=16 $$

## Why FM Alone Is Not Enough — The Deep Half

FM models *2nd-order* interactions and only those. The term
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ is a linear cross of
two features; three-way and higher patterns — "high RFM *and* heavy
digital *and* macro uncertainty → flight to safe assets" — are out of
reach. A deep MLP is the complement: as a universal function
approximator it learns *implicit* higher-order nonlinear interactions,
though it would need many parameters to recover even a simple 2nd-order
pattern the long way around.

| Capability | FM | Deep Network |
| --- | --- | --- |
| 2nd-order interaction | explicit, efficient | possible but wasteful |
| higher-order interaction | impossible | learned implicitly |
| nonlinearity | inner product only | ReLU etc. |
| parameter efficiency | very high ($O(nk)$) | lower |
| interpretability | per-pair attribution | black box |

The deep half consumes the *flattened* field embeddings and runs a
3-layer MLP, each layer linear → BatchNorm → ReLU → dropout, narrowing
the representation as it deepens:

$$ \mathbf{h}^{(l+1)} = \mathrm{ReLU}\big(\mathrm{BN}(W^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)})\big),\qquad \mathbf{h}^{(0)} = \mathrm{flatten}([\mathbf{v}_1;\dots;\mathbf{v}_n]) \in \mathbb{R}^{nk} $$

With $n=28$ fields and $k=16$, the flattened input is $28\times16=448$D,
compressed $448 \to 256 \to 128 \to 64$.

## DeepFM — One Embedding, Shared

The structural move that names the model: FM and Deep do **not** own
separate embeddings. They share one set of field embeddings, read it in
parallel, and back-propagate into it together. That is what removes the
hand-built cross-feature pipeline of Wide & Deep, and it is why both
halves stay consistent.

The fields themselves are the project's contribution. The 644D vector is
sliced into **28 semantic fields** — `rfm` (34D), four split category
fields (`customer_cat`/`product_cat`/`region_cat`/`channel_cat`, 16D
each), `transaction` (80D), `deposit`, `investment`, `mamba` (50D),
`economics`, `merchant_hierarchy` (21D), and so on — each projected to
16D by its own `nn.Linear(dᵢ, 16)`. That gives $\sum_i d_i \times 16 = 644\times16 = 10{,}304$
embedding parameters, with 28 fields yielding $28\times27/2 = 378$
pairwise FM interactions.

> **Historical context.** Splitting the old 64D `category` block into
> four 16D subfields (v3.11) was deliberate: FM never crosses *within* a
> field, so a monolithic category field hid the `product_cat × region_cat`
> and `customer_cat × channel_cat` interactions. Four subfields surface
> them at near-zero parameter cost (351 → 378 pairs). The bigger jump
> was switching from per-feature `nn.Embedding` to per-field `nn.Linear`,
> which shrank the Deep input from 10,304D to 448D and the Expert from
> roughly **10.9M to ~169K** parameters — a 98% cut in the MLP.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="300" fill="#f8fafc" rx="8"/>
  <!-- input -->
  <rect x="230" y="18" width="140" height="34" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="300" y="35" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">x  [B, 644]</text>
  <text x="300" y="47" text-anchor="middle" font-size="9" fill="#64748b">normalized 644D (V1-compat)</text>
  <!-- field embeddings -->
  <rect x="210" y="76" width="180" height="40" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="300" y="93" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Field Embeddings [B, 28, 16]</text>
  <text x="300" y="107" text-anchor="middle" font-size="9" fill="#64748b">28 × Linear(dᵢ → 16) — SHARED</text>
  <line x1="300" y1="52" x2="300" y2="76" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="300,76 296,68 304,68" fill="#cbd5e1"/>
  <!-- split to FM and Deep -->
  <line x1="260" y1="116" x2="150" y2="146" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="150,146 158,142 156,150" fill="#cbd5e1"/>
  <line x1="340" y1="116" x2="450" y2="146" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="450,146 442,142 444,150" fill="#cbd5e1"/>
  <!-- FM head -->
  <rect x="55" y="148" width="160" height="56" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="135" y="168" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">FM Layer</text>
  <text x="135" y="183" text-anchor="middle" font-size="9" fill="#64748b">sum_sq − sq_sum</text>
  <text x="135" y="196" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="700">[B, 16]</text>
  <!-- Deep head -->
  <rect x="385" y="148" width="160" height="56" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="465" y="166" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Deep Network</text>
  <text x="465" y="180" text-anchor="middle" font-size="9" fill="#64748b">flatten 448 → 256 → 128</text>
  <text x="465" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="700">[B, 64]</text>
  <!-- concat -->
  <line x1="135" y1="204" x2="270" y2="232" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="270,232 262,229 263,237" fill="#cbd5e1"/>
  <line x1="465" y1="204" x2="330" y2="232" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="330,232 338,229 337,237" fill="#cbd5e1"/>
  <rect x="220" y="234" width="160" height="32" rx="6" fill="#1e3a5f11" stroke="#1e3a5f" stroke-width="1"/>
  <text x="300" y="254" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">concat [FM ; Deep] → [B, 80]</text>
  <!-- output -->
  <line x1="300" y1="266" x2="300" y2="282" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="210" y="282" width="180" height="14" rx="4" fill="#0d9488"/>
  <text x="300" y="293" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Linear(80→64) → LN → SiLU  [B, 64]</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The shared-embedding architecture: one [B,28,16] embedding feeds both the FM head (16D) and the Deep network (64D); their concatenation projects to a 64D Expert output.</figcaption>
</figure>

The two halves are concatenated and projected to the final
representation:

$$ \mathbf{y}_{\text{DeepFM}} = \mathrm{SiLU}\big(\mathrm{LN}(W_{\text{out}}\,[\,\mathbf{y}_{\text{FM}}\,;\,\mathbf{y}_{\text{Deep}}\,] + \mathbf{b}_{\text{out}})\big) \in \mathbb{R}^{64} $$

with $[\,\mathbf{y}_{\text{FM}}\,;\,\mathbf{y}_{\text{Deep}}\,]$ the
$16+64 = 80$D concatenation and $W_{\text{out}} \in \mathbb{R}^{64\times80}$.
Two MTL-minded choices replace the paper's defaults: **LayerNorm**
instead of the sigmoid output for scale stability across tasks, and
**SiLU** ($x\,\sigma(x)$) instead of ReLU — because SiLU does not kill
negatives, so FM's *suppression* signal (negative inner products)
survives to the output.

## Where the DeepFM Expert Sits

The 64D vector is the whole point of the on-prem extension: the paper
emits a scalar CTR, this Expert emits a representation. Inside
`ple_cluster_adatt.py`, `_forward_shared_experts()` hands the feature
tensor to DeepFM (routed via `FeatureRouter` when registered, otherwise
the full features), and the 64D output joins the other Shared Experts at
the PLE CGC gate, which mixes them per task.

| Stage | Operation | Output dim |
| --- | --- | --- |
| 1 | input | `[B, 644]` |
| 2 | 28-field slice + embed | `[B, 28, 16]` |
| 3a | FM: sum_sq − sq_sum | `[B, 16]` |
| 3b | flatten | `[B, 448]` |
| 4 | Deep: 448→256→128→64 | `[B, 64]` |
| 5 | concat [FM ; Deep] | `[B, 80]` |
| 6 | output: Linear→LN→SiLU | `[B, 64]` |
| 7 | interpret projection | `[B, 4]` |

Alongside DeepFM, two other Experts read the *same* 644D and also emit
64D — the Causal Expert (asymmetric DAG, $W_{ij}\neq W_{ji}$) and the
Optimal-Transport Expert (a distance, $W(\mu,\nu)\geq 0$). DeepFM's
contribution is the one structure they cannot give: the *symmetric*
inner product $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$. The CGC gate
weights these views per task, so an LTV task can lean on DeepFM's cross
patterns while a churn task leans on causal structure. Tasks that name
`domain_experts: ["deepfm"]` — `ltv`, `spending_bucket`, and the like —
get a high initial gate bias toward it. A final `Linear(64→4)`
projection exposes interpretable low/high-order and sparse/dense
interaction channels for SAE analysis.

## Where We Stop

We started from a discomfort with additive linear models, watched the
cross-parameter count explode to ~207K, and saw factorization rescue it
with shared latent vectors at $O(nk)$ — then the algebraic trick that
makes even that a single pass. We split DeepFM into its FM half (16D of
explicit pairwise synergy) and its deep half (64D of implicit
higher-order pattern), and saw the one embedding they share. Finally we
placed the 64D Expert at the PLE gate, beside Causal and OT, each
reading the same features through a different mathematical lens.

What remains is the machinery and the alternatives: how the `FMLayer`
and `DeepNetwork` compose in code, why this project also ships a
**DCNv2** Expert for *explicit* higher-order crosses when the implicit
MLP is not enough, and how the field-interaction analysis actually reads
out which pairs synergize. But before we go deeper into flat
feature-interaction space, the next sub-thread leaves it entirely — for
a *curved* one. **HGCN-1** asks why a hierarchy of merchants and
customers refuses to fit in Euclidean space, and how negative curvature
in the Poincaré ball gives the `unified_hgcn` Expert exponential room
before it hands 128D back to PLE.
