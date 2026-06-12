---
title: "[Study Thread] HGCN-1 — Bending the Graph: Hyperbolic Geometry and the Poincaré-Ball Expert"
date: 2026-06-06 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, hgcn, hyperbolic, graph, poincare, expert]
lang: en
excerpt: "The HGCN sub-thread opens — why flat Euclidean space cannot hold a tree without exploding the dimension count, how negative curvature makes a graph's hierarchy fit, what the Poincaré ball actually is, and how the project does graph convolution on a curved space: lift each node to the tangent plane with a log map, aggregate neighbors there, send it back with an exp map, and project safely inside the ball. The merchant-hierarchy Expert, one of the project's Shared Experts."
series: study-thread
part: 14
alt_lang: /2026/06/06/hgcn-hyperbolic-graph-ko/
next_title: "CausalOT-1 — Moving Mass Along Causes: Optimal Transport Meets Causal Inference"
next_desc: "How the next Expert reads not correlation but cause: the optimal-transport view of distribution shift, why a transport plan is a counterfactual map, and how the project grounds a causal signal in a graph of customer behavior without a randomized experiment."
next_status: draft
---

*First post of the HGCN (Hyperbolic Graph Convolutional Network)
sub-thread in the "Study Thread" series. Across this and the following
posts, in parallel Korean and English, I unpack the merchant-hierarchy
Expert — one of the seven heterogeneous Shared Experts in this project.
The source is the on-prem reference `기술참조서/GCN_기술_참조서`, and the
full PDF will be attached to the final post of the sub-thread. Where the
TDA sub-thread asked what* shape *an Expert reads, this one asks a
question about the* space *the reading happens in. The merchant graph is
a tree — Root → MCC L1 → industry sub-category (grop) → industry-L2
leaf — and a tree, it turns out,
simply does not fit in flat space. The Expert's answer is to bend the
space until it does.*

> **The crux in one line.** A complete binary tree of depth $d$ has
> $2^d$ leaves, and packing them into Euclidean space with low
> distortion needs $O(2^d)$ dimensions. The project's merchant
> hierarchy has ~550K nodes; embedding the old Brand leaf level
> (~50,000 entries) in flat space without distortion would take tens of
> thousands of dimensions. In an **8-dimensional** Poincaré ball it fits with low
> distortion. That single fact — exponential capacity from negative
> curvature — is the whole reason this Expert lives in a curved space
> instead of $\mathbb{R}^d$.

## Why a Tree Will Not Fit in Flat Space

When we embed customers and products for collaborative filtering,
Euclidean space $\mathbb{R}^d$ is the natural home. "User A likes item
1" and "user B likes item 2" are *peer* relations — no hierarchy, every
direction equivalent, linear algebra applies directly. That is exactly
what the project's LightGCN path is built for.

But the *merchant classification* is not peer-to-peer. It is a tree:
Root → MCC Level-1 (8) → industry sub-category (grop, ~35) → industry
Level-2 (frcs_tind_cd, ~346, the leaf — the old Brand and Branch levels
have been retired). And trees have a
geometric problem that flat space cannot solve. The number of leaves
grows as $2^d$ with depth, so the *room* you need grows exponentially —
but Euclidean volume grows only polynomially with radius. The two rates
do not match. To lay out $2^d$ leaves at equal distances you need
$O(2^d)$ dimensions; the children crowd each other and the tree's metric
gets crushed.

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="250" fill="#f8fafc" rx="8"/>
  <!-- Euclidean side: neighborhoods grow polynomially -->
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Euclidean — room grows ∝ rⁿ</text>
  <circle cx="150" cy="135" r="30" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="58" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="86" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="3" fill="#1e3a5f"/>
  <g fill="#64748b">
    <circle cx="180" cy="135" r="3.5"/><circle cx="121" cy="135" r="3.5"/><circle cx="150" cy="105" r="3.5"/><circle cx="150" cy="165" r="3.5"/>
    <circle cx="208" cy="135" r="3.5"/><circle cx="92" cy="135" r="3.5"/><circle cx="190" cy="178" r="3.5"/><circle cx="110" cy="92" r="3.5"/>
    <circle cx="236" cy="135" r="3.5"/><circle cx="64" cy="135" r="3.5"/><circle cx="212" cy="195" r="3.5"/><circle cx="88" cy="75" r="3.5"/>
  </g>
  <text x="150" y="238" text-anchor="middle" font-size="11" fill="#64748b">leaves crowd — distortion</text>
  <line x1="290" y1="40" x2="290" y2="215" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Hyperbolic side: room grows exponentially -->
  <text x="430" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">Hyperbolic — room grows ∝ eʳ</text>
  <circle cx="430" cy="135" r="92" fill="#0d948808" stroke="#0d9488" stroke-width="1"/>
  <circle cx="430" cy="135" r="3" fill="#0d9488"/>
  <g stroke="#0d9488" stroke-width="1"><line x1="430" y1="135" x2="430" y2="60"/><line x1="430" y1="135" x2="495" y2="170"/><line x1="430" y1="135" x2="365" y2="170"/></g>
  <g stroke="#94a3b8" stroke-width="0.8">
    <line x1="430" y1="60" x2="408" y2="48"/><line x1="430" y1="60" x2="452" y2="48"/>
    <line x1="495" y1="170" x2="505" y2="148"/><line x1="495" y1="170" x2="512" y2="190"/>
    <line x1="365" y1="170" x2="348" y2="152"/><line x1="365" y1="170" x2="355" y2="192"/>
  </g>
  <g fill="#0d9488"><circle cx="430" cy="60" r="3.5"/><circle cx="495" cy="170" r="3.5"/><circle cx="365" cy="170" r="3.5"/></g>
  <g fill="#e11d48"><circle cx="408" cy="48" r="3"/><circle cx="452" cy="48" r="3"/><circle cx="505" cy="148" r="3"/><circle cx="512" cy="190" r="3"/><circle cx="348" cy="152" r="3"/><circle cx="355" cy="192" r="3"/></g>
  <text x="430" y="238" text-anchor="middle" font-size="11" fill="#0d9488">every level fits — low distortion</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A tree branches exponentially with depth. Euclidean volume grows polynomially, so children crowd; hyperbolic volume grows exponentially, matching the tree exactly. Root at the centre, leaves toward the boundary.</figcaption>
</figure>

## Negative Curvature — Space That Expands as You Walk Out

The fix is *curvature*. Three regimes are worth holding side by side:

| Curvature | Geometry | What happens as you move out |
| --- | --- | --- |
| positive | sphere | space *shrinks* — meridians converge at the pole |
| zero | Euclidean plane | space is *uniform* — graph paper everywhere |
| negative | hyperbolic space | space *expands exponentially* — room for ever-more children |

A tree adds nodes exponentially with depth; negative curvature adds
room exponentially with radius. The two growth laws *coincide*. That is
why the merchant hierarchy embeds into a tiny **8-dimensional** ball
with low distortion, where flat space would have demanded tens of
thousands of dimensions.

> **Historical context.** Hyperbolic geometry is one of the 19th
> century's great upheavals. For two thousand years Euclid's parallel
> postulate looked self-evident; in 1829 Lobachevsky and in 1832 Bolyai
> independently proved a consistent geometry with *infinitely many*
> parallels. Poincaré (1882) gave it a picture — the interior of a unit
> disk — which is the model machine learning still uses. The data-science
> turn came with Nickel & Kiela's *Poincaré Embeddings* (NeurIPS 2017),
> which packed WordNet's hierarchy into 5 hyperbolic dimensions more
> faithfully than 200 Euclidean ones; HGCN (Chami et al., 2019) then
> fused that idea with graph convolution.

## The Poincaré Ball

To compute, you need a concrete *model* of hyperbolic space. The project
uses the **Poincaré ball** — the open ball of radius $1/\sqrt{c}$:

$$ \mathbb{B}_c^d = \{\, \mathbf{x} \in \mathbb{R}^d : c\,\lVert \mathbf{x} \rVert^2 < 1 \,\} $$

with curvature $c = 1.0$ and dimension $d = 8$. Every point lives
*strictly inside* the unit ball; the boundary is infinitely far away in
the true hyperbolic metric. The intuition is clean: **origin = root**,
**boundary = leaf**. A general consumer who spends across everything
sits near the centre; a specialist who concentrates on one brand drifts
toward the rim.

The reason the centre and the rim feel so different is the *distance
function*. Two points $\mathbf{x},\mathbf{y}$ are separated by

$$ d_{\mathbb{B}}(\mathbf{x},\mathbf{y}) = \frac{1}{\sqrt{c}}\,\operatorname{arccosh}\!\left( 1 + \frac{2c\,\lVert \mathbf{x}-\mathbf{y} \rVert^2}{(1 - c\lVert\mathbf{x}\rVert^2)(1 - c\lVert\mathbf{y}\rVert^2)} \right) $$

> **Equation intuition.** The numerator $2c\lVert\mathbf{x}-\mathbf{y}\rVert^2$
> is just ordinary squared Euclidean distance. The denominator is the
> *conformal factor* $(1-c\lVert\mathbf{x}\rVert^2)(1-c\lVert\mathbf{y}\rVert^2)$
> — the multiplier that bends the space. Near the origin both factors
> are ≈ 1, so distance reduces to the familiar Euclidean one: root-level
> categories are easy to move between. Near the boundary a factor
> $\to 0$, so the *same* small Euclidean gap blows up into an enormous
> hyperbolic distance: two different brand leaves are genuinely far
> apart. The geometry encodes "siblings deep in the tree are not
> neighbours" for free.

<img src="/poincare-hyperbolic.webp" alt="Poincaré disk model — (a) high-density triangle cell tessellation mesh, (b) geodesic paths: diameter geodesic (straight) and circular arc geodesics (curved, meeting boundary orthogonally)" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

The tessellation on the left makes the metric visible: every triangle is
the *same hyperbolic size*, yet they shrink toward the rim — that is the
exponential expansion, drawn. On the right, geodesics (shortest paths)
are not straight lines except through the centre; off-centre they are
circular arcs that meet the boundary at right angles.

## Why You Cannot Just Add Two Hyperbolic Points

Here is the catch that shapes everything downstream. In the Poincaré
ball the everyday operations break. Add two points the Euclidean way and
the result can land *outside* the ball, where it is not even a valid
hyperbolic point. A plain weighted mean of neighbors — the heart of
graph convolution — is therefore not defined directly.

The standard fix, straight out of Riemannian geometry, is to never do
arithmetic on the ball itself. At any point you can lay down a flat
*tangent space* $T_{\mathbf{0}}\mathbb{B}_c^d \cong \mathbb{R}^d$, do the
linear work there, and map back:

$$ \mathbf{x} \in \mathbb{B}_c^d \;\xrightarrow{\ \log\ }\; \mathbf{v} \in T_{\mathbf{0}}\mathbb{B}_c^d \;\xrightarrow{\ \text{compute}\ }\; \mathbf{v}' \;\xrightarrow{\ \exp\ }\; \mathbf{x}' \in \mathbb{B}_c^d $$

Two maps move between the ball and its tangent plane, and because the
project always anchors at the origin $\mathbf{0}$, both have a clean
closed form. The **exponential map** sends a tangent vector outward
along a geodesic:

$$ \exp_{\mathbf{0}}(\mathbf{v}) = \tanh\!\big(\sqrt{c}\,\lVert\mathbf{v}\rVert\big)\,\frac{\mathbf{v}}{\sqrt{c}\,\lVert\mathbf{v}\rVert} $$

The **logarithmic map** is its inverse, pulling a ball point back to the
tangent plane:

$$ \log_{\mathbf{0}}(\mathbf{y}) = \operatorname{arctanh}\!\big(\sqrt{c}\,\lVert\mathbf{y}\rVert\big)\,\frac{\mathbf{y}}{\sqrt{c}\,\lVert\mathbf{y}\rVert} $$

The $\tanh$ in the exp map is what guarantees the output stays in
$(-1,1)$ — you cannot fall out of the ball. The $\operatorname{arctanh}$
in the log map is what blows up near the boundary, faithfully reporting
"this point is hyperbolically very far out." A small implementation note
worth its weight: the log map clamps its argument to $1-\varepsilon$,
because $\operatorname{arctanh}(1) = \infty$ would otherwise produce NaNs
on any node that drifts to the rim.

## Hyperbolic Graph Convolution — log → transform → aggregate → exp

Now the message passing. Chami et al. (2019) gave the recipe in one
sentence: *lift to the tangent space with a log map, do the Euclidean
graph work, send it back with an exp map.* The project's
`HyperbolicGCNLayer` runs exactly that, in five steps, for each layer
$k$:

$$ \mathbf{a}_i^{(k)} = \sum_{j \in \mathcal{N}(i)} w_{ij}\, W^{(k)} \log_{\mathbf{0}}\!\big(\mathbf{x}_j^{(k)}\big), \qquad \mathbf{x}_i^{(k+1)} = \operatorname{proj}\Big(\exp_{\mathbf{0}}\big(\mathbf{a}_i^{(k)}\big)\Big) $$

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="200" fill="#f8fafc" rx="8"/>
  <text x="310" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">One HyperbolicGCNLayer (for each layer k)</text>
  <!-- Step 1 hyperbolic in -->
  <rect x="18" y="70" width="92" height="60" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="64" y="62" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">ball</text>
  <text x="64" y="96" text-anchor="middle" font-size="12" fill="#1e3a5f">xᵢ⁽ᵏ⁾</text>
  <text x="64" y="112" text-anchor="middle" font-size="9" fill="#64748b">∈ 𝔹</text>
  <!-- Step 2 log -->
  <rect x="138" y="70" width="92" height="60" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="184" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 1: log</text>
  <text x="184" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">log₀(x)</text>
  <text x="184" y="112" text-anchor="middle" font-size="9" fill="#64748b">→ tangent</text>
  <!-- Step 3 linear -->
  <rect x="258" y="70" width="92" height="60" rx="6" fill="#fff1f2" stroke="#e11d48" stroke-width="1"/>
  <text x="304" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 2: W (id-init)</text>
  <text x="304" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">W v</text>
  <text x="304" y="112" text-anchor="middle" font-size="9" fill="#64748b">linear, no bias</text>
  <!-- Step 4 aggregate -->
  <rect x="378" y="70" width="92" height="60" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="424" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 3: aggregate</text>
  <text x="424" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">Σ wᵢⱼ hⱼ</text>
  <text x="424" y="112" text-anchor="middle" font-size="9" fill="#64748b">sym-norm</text>
  <!-- Step 5 exp + proj -->
  <rect x="498" y="70" width="104" height="60" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="550" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 4–5: exp · proj</text>
  <text x="550" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">exp₀(a)</text>
  <text x="550" y="112" text-anchor="middle" font-size="9" fill="#64748b">→ ball, clamp</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="110" y1="100" x2="136" y2="100"/><polygon points="136,100 128,96 128,104"/>
    <line x1="230" y1="100" x2="256" y2="100"/><polygon points="256,100 248,96 248,104"/>
    <line x1="350" y1="100" x2="376" y2="100"/><polygon points="376,100 368,96 368,104"/>
    <line x1="470" y1="100" x2="496" y2="100"/><polygon points="496,100 488,96 488,104"/>
  </g>
  <!-- loop back -->
  <path d="M 550 130 C 550 175, 64 175, 64 132" fill="none" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="307" y="190" text-anchor="middle" font-size="9" fill="#94a3b8">next layer k+1</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Hyperbolic message passing: log map lifts each node to the flat tangent space, a learnable linear W and symmetric-normalized neighbor sum do the Euclidean work, the exp map returns to the ball, and a projection keeps everything strictly inside.</figcaption>
</figure>

The five steps, end to end:

- **Step 1 — log map.** $\log_{\mathbf{0}}$ lifts every node embedding
  from the curved ball onto the flat tangent plane, where linear
  operations are mathematically legitimate.
- **Step 2 — linear transform.** A learnable $W^{(k)} =$
  `nn.Linear(dim, dim, bias=False)` reshapes the tangent vector. It is
  **initialized to the identity** (`nn.init.eye_`), so at step zero the
  layer is mathematically equivalent to the older parameter-free
  smoothing — a warm start that keeps early training stable.
- **Step 3 — aggregate.** A symmetric-normalized weighted sum over
  neighbors $\mathcal{N}(i)$ (a `scatter_add`) blends in graph
  structure. The normalization keeps a hub like a 1M-transaction brand
  from drowning out a 100-transaction one.
- **Step 4 — exp map.** $\exp_{\mathbf{0}}$ sends the aggregated tangent
  vector back to the ball; the built-in $\tanh$ already keeps it inside.
- **Step 5 — project.** A final `proj` clamps against floating-point
  drift so no point ever escapes the boundary.

> **Where the non-linearity hides.** Notice there is no ReLU or GELU
> *inside* the layer. That is deliberate: the $\tanh$ in the exp map and
> the $\operatorname{arctanh}$ in the log map are *already* non-linear,
> so the maps themselves play the role an activation would. The added
> $W$ is a pure linear transform in the tangent space — adding a second
> explicit activation on top would stack non-linearities and risk
> vanishing gradients. The geometry does the bending; the weight only
> rotates.

A final detail makes the optimizer hyperbolic-aware. Plain Euclidean
gradients ignore the space's expansion and over-shoot near the boundary,
pushing points out of the ball. The fix scales the gradient by the
inverse metric tensor:

$$ \nabla_{\text{Riem}}\,f(\mathbf{x}) = \frac{(1 - c\lVert\mathbf{x}\rVert^2)^2}{4}\,\nabla_{\text{Euclid}}\,f(\mathbf{x}) $$

Near the origin the factor is ≈ ¼; near the boundary it $\to 0$. The
principle in plain words: *the closer to the rim, the more cautiously
you step.* Without this correction the project's `_train_gcn()` produces
NaNs in the first epochs.

## Relation to LightGCN — Same Skeleton, Different Space

The project runs two GCN paths, and they are siblings, not rivals. Strip
the geometry away and both keep the *same* LightGCN skeleton: no
in-layer non-linearity, neighbor-averaging, and a final mean over all
layers $\frac{1}{L+1}\sum_{k=0}^{L}\mathbf{x}^{(k)}$ to fight
over-smoothing. What differs is the *graph* and the *space*.

| | LightGCN | H-GCN (this Expert) |
| --- | --- | --- |
| Nodes | customers + merchants (bipartite) | merchants only (MCC hierarchy tree) |
| Edges | customer ↔ merchant transactions | parent ↔ child industry hierarchy (old brand ↔ brand co-visit edges retired) |
| Learns | "who likes what" (collaborative) | "how merchants relate structurally" |
| Space | Euclidean $\mathbb{R}^{64}$ | Poincaré ball $\mathbb{B}^{8}$ |
| Output | 64D customer embedding (direct) | merchant emb → per-customer 47D (indirect) |

The two signals complement each other: LightGCN personalizes from
collaborative behavior, while H-GCN supplies structural merchant
relations that cover cold-start and sparse customers. One caution from
the reference is worth repeating — **H-GCN is not collaborative
filtering.** Even when transaction-derived *co-visitation edges* existed
(they were retired along with the brand leaf), they only nudged
merchant-to-merchant geometry; H-GCN never learns a customer's preference
the way LightGCN does.

> A note on operational state. As of 2026-04-24 the LightGCN path is
> *temporarily disabled* (its Stage-1 `collaborative_embeddings`
> artifact is absent), so the live Shared-Expert set is six, summing to
> 448D rather than 512D. The hyperbolic path described here, the
> `unified_hgcn` Expert, is active. The comparison above is the design
> contract, not a claim that both run today.

## Where the HGCN Expert Plugs Into PLE

Graph message passing is expensive — the full ~550K-node graph has to
sit in memory — so the project splits the work in two, the same
Stage-1/Stage-2 pattern Pinterest's PinSage popularized:

1. **Stage 1 (offline, Airflow batch).** A `HierarchyEmbeddingGenerator`
   trains the hyperbolic GCN by self-supervision over the merchant tree,
   then freezes a per-user **47D** embedding (Output A 20D + Output B
   27D, both built from weighted Poincaré means) to Parquet. Heavy, done
   on the graph's update cycle, not per task.
2. **Stage 2 (batch training/serving).** The `UnifiedHGCNExpert` simply
   *looks up* those 47D vectors and runs a light bottleneck
   `refine_mlp` — `Linear(47→128) → GELU → Linear(128→47)` with a
   residual — then an `output_proj` to the **128D** Expert
   representation. No graph propagation at inference; constant cost
   regardless of graph size.

That 128D output is one Shared Expert feeding the **PLE CGC gate**,
which mixes it per task against the other Experts across all **15**
tasks. Because the embedding carries the merchant hierarchy, it lends
structural signal to every task that touches *what* a customer buys and
*how specialized* they are — a depth indicator (origin distance) rides
along inside the same embedding. The Expert itself is **embedding-only**:
its former hierarchical brand-prediction head was removed when the
brand-prediction task was retired.

## Where We Stop

We started from a discomfort that flat space simply cannot hold a tree,
walked through negative curvature and the Poincaré ball that realizes
it, saw why hyperbolic distance makes root-categories close and
brand-leaves far, and unpacked the one move that makes graph convolution work
on a curved space — lift with a log map, do the linear graph work on the
flat tangent plane, return with an exp map, and project safely back
inside. Then we placed the Expert: a Stage-1 frozen 47D embedding,
refined to 128D, gated into all 15 tasks.

What remains is the other half of the geometry. We embedded *structure*
here — but structure is only correlation between merchants. The next
sub-thread turns to an Expert that reads *cause*: **CausalOT**, where
optimal transport supplies the map that turns a distribution shift into
a counterfactual, and a causal signal is grounded in the customer graph
without ever running a randomized experiment. That is the subject of the
next post, **CausalOT-1**.
