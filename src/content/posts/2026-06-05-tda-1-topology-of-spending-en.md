---
title: "[Study Thread] TDA-1 — The Shape of Spending: Topological Data Analysis and the PersLay Bridge"
date: 2026-06-05 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, perslay, persistent-homology, topology, expert]
lang: en
excerpt: "The TDA / PersLay sub-thread opens — why summary statistics miss the geometric structure of spending, how persistent homology reads the 'shape' of a customer's behavior, what a persistence diagram actually encodes, and the bridge PersLay builds from that diagram into a neural network. With the validation result that justified the Expert."
series: study-thread
part: 11
alt_lang: /2026/06/05/tda-1-topology-of-spending-ko/
next_title: "TDA-2 — PersLay as a Set Function: φ, w, ρ and the 5-Block Architecture"
next_desc: "How PersLay turns a variable-size, unordered point set into a fixed 64D vector: the RationalHat point transform, the persistence weight, permutation-invariant aggregation, and why the project splits Short/Long range × homology dimension into five independent blocks."
next_status: published
---

*First post of the TDA (Topological Data Analysis) / PersLay sub-thread
in the "Study Thread" series. Across this and the following posts, in
parallel Korean and English, I unpack the PersLay Expert — one of the
seven heterogeneous Shared Experts in this project. The source is the
on-prem reference `기술참조서/PersLay_기술_참조서`, and the full PDF will
be attached to the final post of the sub-thread. Where the PLE and
adaTT sub-threads dealt with how tasks share and transfer, this one
asks a different question: what kind of signal does an Expert read in
the first place? PersLay's answer is unusual — it reads the* shape *of
spending, the part of the data that means and variances throw away.*

> **Empirical status — validated, not speculative.** Before writing a
> line of model code, we ran a standalone validation on real session
> logs (90 days, 120 customers). The question was blunt: does customer
> behavior actually carry topological structure, or is this
> mathematically elegant but empirically empty? Persistence-diagram
> summaries separated behavior groups at silhouette **0.299**, against
> **0.192** for the matched raw-aggregate features — a **+0.108**
> improvement, with the TDA score clearing the pre-registered PASS bar
> of 0.15. The
> Expert earned its place. The numbers, and how we got them, are in the
> "Does spending have a shape?" section below.

## The Limit of Summary Statistics

When we describe a customer, we reach for means, variances, maxima,
medians. These capture the *central tendency* and *spread* of a
distribution — and almost nothing about its *geometric arrangement*.

Take two customers, A and B. A spends evenly across food, transport,
culture, and shopping; in the high-dimensional space of category
vectors, the points form a single connected blob. B spends only on
food and transport, but rotates — food early in the month, transport
late — so the points split into two separated clusters with a periodic
path running between them.

Their mean spend can be identical. Their variance can be identical. Yet
the *structure* of their behavior is qualitatively different: one
connected lump versus two lobes joined by a cycle. That difference is
*topological*, and no summary statistic can see it.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <!-- Customer A: single blob -->
  <text x="140" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Customer A — one cluster</text>
  <circle cx="110" cy="110" r="58" fill="#0d948815" stroke="#0d9488" stroke-width="1" stroke-dasharray="4 3"/>
  <g fill="#0d9488">
    <circle cx="95" cy="90" r="4"/><circle cx="120" cy="85" r="4"/><circle cx="140" cy="105" r="4"/>
    <circle cx="105" cy="120" r="4"/><circle cx="130" cy="130" r="4"/><circle cx="90" cy="115" r="4"/>
    <circle cx="118" cy="112" r="4"/><circle cx="150" cy="125" r="4"/><circle cx="100" cy="100" r="4"/>
  </g>
  <text x="140" y="200" text-anchor="middle" font-size="11" fill="#64748b">β₀ = 1, β₁ = 0</text>
  <!-- divider -->
  <line x1="280" y1="40" x2="280" y2="195" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Customer B: two clusters + loop -->
  <text x="420" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Customer B — two lobes + cycle</text>
  <circle cx="360" cy="100" r="34" fill="#e11d4815" stroke="#e11d48" stroke-width="1" stroke-dasharray="4 3"/>
  <circle cx="470" cy="130" r="34" fill="#e11d4815" stroke="#e11d48" stroke-width="1" stroke-dasharray="4 3"/>
  <path d="M 360 100 C 390 60, 440 60, 470 130 C 490 175, 400 180, 360 100 Z" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="5 4"/>
  <g fill="#e11d48">
    <circle cx="345" cy="90" r="4"/><circle cx="370" cy="95" r="4"/><circle cx="355" cy="115" r="4"/><circle cx="375" cy="112" r="4"/>
    <circle cx="458" cy="120" r="4"/><circle cx="482" cy="128" r="4"/><circle cx="465" cy="145" r="4"/><circle cx="478" cy="140" r="4"/>
  </g>
  <text x="420" y="200" text-anchor="middle" font-size="11" fill="#64748b">β₀ = 2, β₁ = 1</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Identical mean and variance, different shape. Topology counts the lumps (β₀) and the loops (β₁); statistics count neither.</figcaption>
</figure>

## What Topology Measures

Topology studies properties preserved under continuous deformation —
stretch, bend, and twist, but the *number of holes* never changes
(the old joke: a coffee cup is a donut). Applied to a point cloud, it
answers three questions that survive noise:

| Invariant | What it counts | In spending data |
| --- | --- | --- |
| $H_0$ ($\beta_0$) | connected components | separated spending clusters — e.g. a "food-centric" and a "travel-centric" group |
| $H_1$ ($\beta_1$) | loops (1-D holes) | cyclic spending — food → transport → culture → back to food |
| $H_2$ ($\beta_2$) | voids (2-D cavities) | high-dimensional empty regions where three-plus categories never co-occur |

The value of these features is not aesthetic. They are *coordinate
invariant* (rotate or shift the data and they do not move),
*noise robust* (a small perturbation cannot change a hole count), and
*multi-scale* (they track structure across every distance threshold at
once, not at one hand-picked cutoff).

> **Historical context.** Algebraic topology starts with Poincaré's
> homology theory in the 19th century and was axiomatized by
> Eilenberg–Steenrod in the mid-20th. The turn to *computational*
> topology came in the 1990s–2000s (Edelsbrunner, Harer, Carlsson),
> and from 2010 onward it entered data science as *Topological Data
> Analysis*. Roughly fifty years of pure mathematics flowing downhill
> into a practical tool for measuring the shape of point clouds.

## Persistent Homology — Watching Holes Appear and Die

A single distance cutoff is arbitrary. Persistent homology refuses to
pick one. Instead it grows a threshold $\varepsilon$ from $0$ upward
and watches the topology change — a process called a *filtration*.

- $\varepsilon = 0$: every point is isolated. $n$ components, no loops.
- $\varepsilon$ small: nearby points connect into edges; components merge.
- $\varepsilon$ medium: edges close into loops — an $H_1$ feature is *born*.
- $\varepsilon$ larger: triangles fill the loop in — that feature *dies*.
- $\varepsilon \to \infty$: everything is one blob.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="170" fill="#f8fafc" rx="8"/>
  <!-- shared point layout (a rough pentagon) -->
  <!-- panel coords helper: each panel 140 wide -->
  <!-- Panel 1: isolated -->
  <text x="75" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε = 0</text>
  <g fill="#4f46e5"><circle cx="75" cy="55" r="4"/><circle cx="108" cy="80" r="4"/><circle cx="95" cy="118" r="4"/><circle cx="55" cy="118" r="4"/><circle cx="42" cy="80" r="4"/></g>
  <text x="75" y="150" text-anchor="middle" font-size="10" fill="#64748b">5 components</text>
  <!-- Panel 2: edges -->
  <text x="225" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε small</text>
  <g stroke="#94a3b8" stroke-width="1.4">
    <line x1="225" y1="55" x2="258" y2="80"/><line x1="258" y1="80" x2="245" y2="118"/>
    <line x1="245" y1="118" x2="205" y2="118"/><line x1="205" y1="118" x2="192" y2="80"/><line x1="192" y1="80" x2="225" y2="55"/>
  </g>
  <g fill="#4f46e5"><circle cx="225" cy="55" r="4"/><circle cx="258" cy="80" r="4"/><circle cx="245" cy="118" r="4"/><circle cx="205" cy="118" r="4"/><circle cx="192" cy="80" r="4"/></g>
  <text x="225" y="150" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700">loop born (H₁)</text>
  <!-- Panel 3: loop filling -->
  <text x="375" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε medium</text>
  <polygon points="375,55 408,80 395,118 355,118 342,80" fill="#0d948822" stroke="#94a3b8" stroke-width="1.4"/>
  <g fill="#4f46e5"><circle cx="375" cy="55" r="4"/><circle cx="408" cy="80" r="4"/><circle cx="395" cy="118" r="4"/><circle cx="355" cy="118" r="4"/><circle cx="342" cy="80" r="4"/></g>
  <text x="375" y="150" text-anchor="middle" font-size="10" fill="#64748b">filling in…</text>
  <!-- Panel 4: filled -->
  <text x="525" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε large</text>
  <polygon points="525,55 558,80 545,118 505,118 492,80" fill="#0d948855" stroke="#0d9488" stroke-width="1.4"/>
  <g fill="#4f46e5"><circle cx="525" cy="55" r="4"/><circle cx="558" cy="80" r="4"/><circle cx="545" cy="118" r="4"/><circle cx="505" cy="118" r="4"/><circle cx="492" cy="80" r="4"/></g>
  <text x="525" y="150" text-anchor="middle" font-size="10" fill="#e11d48" font-weight="700">loop dies</text>
  <!-- arrows -->
  <g fill="#cbd5e1"><polygon points="156,88 146,83 146,93"/><polygon points="306,88 296,83 296,93"/><polygon points="456,88 446,83 446,93"/></g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A Vietoris–Rips filtration. As ε grows, a loop is born when edges close it and dies when triangles fill it. Persistence = death − birth.</figcaption>
</figure>

Each feature gets a *birth* $\varepsilon$ and a *death* $\varepsilon$.
The same structure, viewed as horizontal bars over the filtration axis,
is the *persistence barcode* — long bars are robust features, short
bars are noise.

<img src="/persistence-barcode.webp" alt="Persistence barcode — horizontal bars at varying heights show the lifespan of each topological feature across the filtration scale; longer bars indicate robust features, shorter bars are noise" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

## The Persistence Diagram

Plot each feature as a point $(b, d)$ — birth on the x-axis, death on
the y-axis — and you get a *persistence diagram*. Everything sits above
the diagonal $d = b$, and distance from that diagonal is the whole
story:

- **Far from the diagonal** — large persistence $d - b$. A structure
  that survives a wide range of scales: a *real* feature.
- **Near the diagonal** — tiny persistence. A structure that flickered
  into existence and vanished: *noise*.

<figure style="margin:24px auto;max-width:440px;">
<svg viewBox="0 0 440 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="440" height="320" fill="#f8fafc" rx="8"/>
  <!-- axes -->
  <line x1="60" y1="270" x2="400" y2="270" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="270" x2="60" y2="40" stroke="#64748b" stroke-width="1.2"/>
  <text x="230" y="300" text-anchor="middle" font-size="12" fill="#1e3a5f">birth (b)</text>
  <text x="22" y="160" text-anchor="middle" font-size="12" fill="#1e3a5f" transform="rotate(-90 22 160)">death (d)</text>
  <!-- diagonal -->
  <line x1="60" y1="270" x2="370" y2="55" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="350" y="80" font-size="10" fill="#94a3b8">d = b</text>
  <!-- noise band -->
  <polygon points="60,270 80,256 360,61 340,75" fill="#94a3b822"/>
  <text x="250" y="200" font-size="10" fill="#94a3b8" transform="rotate(-34 250 200)">noise band</text>
  <!-- robust points (far from diagonal) -->
  <g fill="#0d9488"><circle cx="110" cy="110" r="6"/><circle cx="95" cy="90" r="6"/><circle cx="140" cy="130" r="6"/></g>
  <text x="150" y="105" font-size="11" fill="#0d9488" font-weight="700">H₀ — robust clusters</text>
  <g fill="#e11d48"><circle cx="180" cy="120" r="6"/><circle cx="210" cy="145" r="6"/></g>
  <text x="225" y="128" font-size="11" fill="#e11d48" font-weight="700">H₁ — robust loop</text>
  <!-- noise points near diagonal -->
  <g fill="#94a3b8"><circle cx="150" cy="172" r="4"/><circle cx="200" cy="218" r="4"/><circle cx="250" cy="252" r="4"/><circle cx="120" cy="158" r="4"/><circle cx="290" cy="270" r="4"/></g>
  <!-- persistence arrow -->
  <line x1="110" y1="110" x2="110" y2="225" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="116" y="180" font-size="10" fill="#d97706">persistence = d − b</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A persistence diagram. The further a point is from the diagonal, the more robust the structure it represents.</figcaption>
</figure>

The reason we can trust this picture is the *stability theorem*:

$$ d_B\big(\mathrm{Dgm}(f), \mathrm{Dgm}(g)\big) \le \lVert f - g \rVert_\infty $$

> **Equation intuition.** $f, g$ are the functions defining two
> filtrations (here, distance matrices); $\mathrm{Dgm}$ is the diagram
> each produces; $d_B$ is the bottleneck distance between diagrams.
> The inequality says: perturb the input a little, and the diagram
> moves at most that little. Sensor noise or a measurement error
> cannot flip the topological summary — a guarantee that ordinary
> features like variance, which a single outlier can wreck, simply do
> not have.

## From Diagram to Network — Why You Cannot Just Feed It In

Here is the catch. A persistence diagram is *not* a vector. It is an
unordered set of points, of *variable size* (every customer produces a
different number of features), living in a metric space defined by
bottleneck / Wasserstein distance. A standard MLP wants a fixed-length,
ordered vector. The two do not meet.

PersLay (Carrière et al., *JMLR 2020*) is the bridge. It treats the
diagram as a *set* and applies the DeepSets recipe — a learnable,
permutation-invariant set function:

$$ F(D) = \rho\!\left( \sum_{(b,d)\in D} w(b,d)\,\cdot\,\phi(b,d) \right) $$

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="180" fill="#f8fafc" rx="8"/>
  <!-- D: diagram set -->
  <rect x="20" y="55" width="90" height="75" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="65" y="48" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">D (point set)</text>
  <g fill="#4f46e5"><circle cx="45" cy="78" r="3"/><circle cx="72" cy="70" r="3"/><circle cx="88" cy="95" r="3"/><circle cx="55" cy="105" r="3"/><circle cx="80" cy="115" r="3"/></g>
  <!-- phi -->
  <rect x="150" y="60" width="92" height="64" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="196" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">φ(b,d)</text>
  <text x="196" y="104" text-anchor="middle" font-size="9" fill="#64748b">point transform</text>
  <text x="196" y="48" text-anchor="middle" font-size="9" fill="#64748b">each point → vector</text>
  <!-- w -->
  <rect x="282" y="60" width="92" height="64" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="328" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">w(b,d)</text>
  <text x="328" y="104" text-anchor="middle" font-size="9" fill="#64748b">persistence weight</text>
  <text x="328" y="48" text-anchor="middle" font-size="9" fill="#64748b">noise → ~0</text>
  <!-- rho -->
  <rect x="414" y="60" width="92" height="64" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="460" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">ρ (Σ)</text>
  <text x="460" y="104" text-anchor="middle" font-size="9" fill="#64748b">order-invariant sum</text>
  <!-- output -->
  <rect x="540" y="68" width="46" height="48" rx="6" fill="#0d9488" />
  <text x="563" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">F(D)</text>
  <text x="563" y="104" text-anchor="middle" font-size="9" fill="#fff">64D</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="110" y1="92" x2="148" y2="92"/><polygon points="148,92 140,88 140,96"/>
    <line x1="242" y1="92" x2="280" y2="92"/><polygon points="280,92 272,88 272,96"/>
    <line x1="374" y1="92" x2="412" y2="92"/><polygon points="412,92 404,88 404,96"/>
    <line x1="506" y1="92" x2="538" y2="92"/><polygon points="538,92 530,88 530,96"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">PersLay as a set function: transform each point (φ), weight it by persistence (w), aggregate order-invariantly (ρ) into one fixed 64D vector.</figcaption>
</figure>

Three pieces do the work, and all three are *learnable*:

- **$\phi$ — point transform.** Each $(b,d)$ becomes a vector. The
  project's `RationalHatPhi` first hand-expands the point to six
  features — $[\,b,\ d,\ d-b,\ \tfrac{b+d}{2},\ b\cdot d,\ \tfrac{d}{b+\epsilon}\,]$
  — then runs a 2-layer MLP, letting the network learn which non-linear
  mix matters per task.
- **$w$ — weight.** With $w(b,d) = |d-b|^{p}$, points *on* the diagonal
  (persistence 0) get weight 0. Noise is suppressed for free, and zero
  padding is ignored without any mask.
- **$\rho$ — aggregation.** A sum (or mean / max / attention) collapses
  the weighted set into one vector — invariant to the order the points
  arrived in.

Because the whole thing is differentiable, the recommendation loss
back-propagates all the way into $\phi$ and $w$: the network *discovers*
which topological structures matter for CTR, for churn, for next-best
action — instead of us hand-designing a topological feature and hoping.

> **Why this beats fixed encodings.** Persistence Landscapes (Bubenik,
> 2015) and Persistence Images (Adams et al., 2017) also vectorize a
> diagram — but with a *fixed* transform, blind to the task. PersLay's
> one move is to make $\phi$, $w$, $\rho$ trainable, turning a static
> descriptor into a task-optimized representation.

## Does Spending Actually Have a Shape?

Elegant math is not a license to ship. Before committing a single
Expert to the model, we tested the load-bearing hypothesis directly:
*does session behavior carry topological structure that separates
customers?* If not, PersLay is dead weight.

The setup, deliberately blunt:

- **Data** — 90 days of real app session logs (2026-01-13 → 04-12),
  hash-sampled to **120 customers**, ~42 session points each.
- **Per customer** — build a point cloud from session vectors
  (duration, pageview, buycount, buyprice, …), run Ripser to get the
  persistence diagram, summarize it into TDA features (H₀/H₁ counts,
  total / max / mean lifetime, persistence entropy).
- **Test** — cluster customers with KMeans and score separation by
  silhouette. Compare *TDA persistence features* against *matched raw
  aggregates* (plain session statistics). Same customers, same
  pipeline, only the feature set changes.

<figure style="margin:24px auto;max-width:480px;">
<svg viewBox="0 0 480 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="480" height="250" fill="#f8fafc" rx="8"/>
  <text x="240" y="30" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Silhouette score — behavior-group separation</text>
  <!-- axis -->
  <line x1="120" y1="200" x2="430" y2="200" stroke="#64748b" stroke-width="1"/>
  <line x1="120" y1="60" x2="120" y2="200" stroke="#64748b" stroke-width="1"/>
  <!-- scale ticks -->
  <g font-size="9" fill="#94a3b8" text-anchor="end">
    <text x="114" y="203">0.0</text><text x="114" y="158">0.1</text><text x="114" y="113">0.2</text><text x="114" y="68">0.3</text>
  </g>
  <line x1="120" y1="155" x2="430" y2="155" stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3"/>
  <line x1="120" y1="110" x2="430" y2="110" stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3"/>
  <!-- PASS threshold 0.15 -->
  <line x1="120" y1="132.5" x2="430" y2="132.5" stroke="#d97706" stroke-width="1.2" stroke-dasharray="6 3"/>
  <text x="426" y="128" text-anchor="end" font-size="9.5" fill="#d97706" font-weight="700">PASS bar = 0.15</text>
  <!-- raw bar: 0.192 -> height 0.192*450 = 86.4 -->
  <rect x="175" y="113.6" width="70" height="86.4" fill="#94a3b8" rx="3"/>
  <text x="210" y="107" text-anchor="middle" font-size="13" font-weight="700" fill="#64748b">0.192</text>
  <text x="210" y="218" text-anchor="middle" font-size="10" fill="#64748b">raw aggregates</text>
  <text x="210" y="231" text-anchor="middle" font-size="9" fill="#94a3b8">(k = 6)</text>
  <!-- TDA bar: 0.299 -> height 134.55 -->
  <rect x="305" y="65.5" width="70" height="134.5" fill="#0d9488" rx="3"/>
  <text x="340" y="59" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">0.299</text>
  <text x="340" y="218" text-anchor="middle" font-size="10" fill="#0d9488" font-weight="700">TDA persistence</text>
  <text x="340" y="231" text-anchor="middle" font-size="9" fill="#94a3b8">(k = 2)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">TDA persistence features separate behavior groups markedly better than raw aggregates (+0.108), clearing the pre-set PASS bar. Verdict: PASS.</figcaption>
</figure>

> **What the silhouette score is.** The standard measure of how cleanly
> a clustering separates. For each customer, take two distances — $a$,
> the average distance to the other customers in *their own group*, and
> $b$, the average distance to the *nearest neighboring group*. That
> customer's score is $(b-a)/\max(a,b)$, and the final silhouette is
> the average over everyone. The range runs from $-1$ to $+1$: near
> $+1$ means groups are tight inside and far apart — a crisp split;
> near $0$ means blurred boundaries; negative means outright
> misassignment. By convention, $0.5$ and above reads as clear
> structure, $0.25{\sim}0.5$ as weak but real. The two bars above are
> the same experiment run on two different feature sets — the taller
> the bar, the more cleanly that feature set separates customers. Which
> is why 0.299 reads as "real," not "strong" — and why the caveat box
> below stays humble.

The result: **TDA silhouette 0.299** (best at $k=2$) versus **raw
aggregates 0.192** ($k=6$) — a **+0.108** lift, with the TDA score
comfortably past the pre-registered 0.15 PASS threshold. The logged verdict reads:
*"Persistence summaries separate behavior groups; PersLay-style TDA
features are justified for session behavior."* That sentence is the
reason the PersLay Expert exists in this codebase rather than as a
footnote.

> A caveat worth keeping. This was a *120-customer, single-window*
> probe, and the topology of session logs is not the topology of
> 12-month financial transactions. It justifies *building* the Expert;
> it does not certify production performance. The honest reading is
> "green light to invest," not "case closed."

## Where PersLay Sits

Computing persistence is expensive — a Vietoris–Rips complex grows like
$O(2^n)$ in the worst case — so the project splits the work in two:

1. **Offline (Airflow batch).** A `PersistenceExtractor` runs Ripser /
   Ripser++ on each customer's point cloud and writes the diagrams
   (birth, death, dimension) to Parquet. Heavy, GPU-accelerated, done
   once.
2. **Online (batch training/serving).** Five `PersLayBlock`s consume
   those diagrams and learn the $362\text{D} \to 64\text{D}$ mapping
   end-to-end. The 64D output feeds the PLE CGC gate, which mixes it
   with the other Experts per task.

PersLay is wired in as a `domain_experts` member for seven tasks —
**ctr, cvr** (engagement), **churn, retention, life_stage**
(lifecycle), and **nba, spending_category** (consumption) — the tasks
where the *shape* of behavior plausibly carries signal that means and
counts miss.

## Where We Stop

We started from a discomfort with summary statistics, walked through
persistent homology and the filtration that makes it multi-scale, read
the persistence diagram as a noise-robust summary, and saw how PersLay
bridges that unordered point set into a 64D vector a network can use.
Then we checked the one thing that matters — *does spending have a
shape?* — and got a PASS.

What remains is the machinery: how exactly `RationalHatPhi`,
the persistence weight, and the aggregation compose into a working
layer; and why the project does not run one PersLay but *five* — Short
versus Long range, crossed with homology dimension, each block a
separate set of parameters because a 90-day cluster and a 12-month void
are qualitatively different signals. That is the subject of the next
post, **TDA-2**.
