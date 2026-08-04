---
title: "[Study Thread] CAUSALOT-1 — Moving Mass Along Causes: Optimal Transport Meets Causal Inference"
date: 2026-06-06 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, causal, optimal-transport, notears, counterfactual, expert]
lang: en
excerpt: "The Causal + Optimal Transport sub-thread opens — why correlation is not enough to recommend, how the Causal Expert learns a directed acyclic graph with NOTEARS' differentiable acyclicity constraint, what its structural equation and counterfactuals actually compute, and how the OT side reads a customer as a distribution and measures Wasserstein distance with Sinkhorn. Two heads of one coin: 'why this' and 'how well it fits.'"
series: study-thread
part: 15
alt_lang: /2026/06/06/causal-ot-expert-ko/
next_title: "TEMPORAL-1 — Three Clocks for One Customer: Mamba, Liquid Networks, and a Patch Transformer in Ensemble"
next_desc: "How the Temporal Expert reads the time axis of spending — a state-space model (Mamba), a continuous-time liquid network, and a patch Transformer combined, and why three different notions of 'memory' beat one."
next_status: draft
---

*First post of the Causal + Optimal Transport (CausalOT) sub-thread in
the "Study Thread" series. Across this and the following posts, in
parallel Korean and English, I unpack the Causal Expert and the OT
Expert — two of the seven heterogeneous Shared Experts in this project.
The source is the on-prem reference `기술참조서/CausalOT_기술_참조서`, and
the full PDF will be attached to the final post of the sub-thread. Where
the TDA sub-thread asked what kind of* shape *an Expert reads, this one
asks two harder questions: does a recommendation actually* cause *the
outcome we want, and how* well *does this customer's behavior fit a
prototype? Correlation answers neither. A causal DAG and optimal
transport do.*

> **Two Experts, one design pattern.** v3.2 of the architecture added
> two new Shared Experts side by side: a **Causal Expert** that learns a
> directed-acyclic causal graph over latent variables and intervenes on
> it, and an **OT Expert** that turns each customer into a probability
> distribution and measures its Wasserstein distance to learnable
> prototypes. Both take the same **734D** feature tensor and
> both emit a **64D** representation into the CGC gate — yet they extract
> mathematically opposite structures: one *asymmetric and acyclic*
> (direction of causation), the other a *metric* (distributional
> distance). This post walks through both.

## Correlation Is Not Enough to Recommend

A conventional recommender learns *correlation*: "customers who bought A
also bought B." It works astonishingly well — until it doesn't. The
reference opens with a blunt example.

> Premium-card holders buy overseas travel insurance at a high rate.

A correlation-driven system will push travel insurance at every premium
cardholder. But the real structure may be a *confounder*:

```
premium card  ←  high income  →  travel insurance
```

High income causes *both*. The card does not *cause* the insurance
purchase. Hand out free premium cards and insurance uptake will not
move. This is the trap correlation cannot escape, and it is exactly what
Judea Pearl's *ladder of causation* formalizes — three rungs, each
answering a strictly harder question:

| Rung | Question | In this Expert |
| --- | --- | --- |
| 1 — Association | "what is related to what?" | the raw correlation a vanilla model sees |
| 2 — Intervention | "what happens if I *do* X?" | the structural equation $\hat{\mathbf z} = \mathbf z + \mathbf z(\mathbf W \odot \mathbf W)$ |
| 3 — Counterfactual | "what *would have* happened?" | `get_counterfactual` → factual / direct_only / full_cf |

<figure style="margin:24px auto;max-width:440px;">
<svg viewBox="0 0 440 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="440" height="240" fill="#f8fafc" rx="8"/>
  <text x="220" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Pearl's Ladder of Causation</text>
  <!-- rung 3 -->
  <rect x="70" y="48" width="300" height="40" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="86" y="66" font-size="12" font-weight="700" fill="#4f46e5">Rung 3 — Counterfactual</text>
  <text x="86" y="81" font-size="10" fill="#64748b">"what would have happened?"  ·  full_cf − direct_only</text>
  <!-- rung 2 -->
  <rect x="70" y="100" width="300" height="40" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <text x="86" y="118" font-size="12" font-weight="700" fill="#0d9488">Rung 2 — Intervention</text>
  <text x="86" y="133" font-size="10" fill="#64748b">"what if I do(X)?"  ·  ẑ = z + z(W⊙W)</text>
  <!-- rung 1 -->
  <rect x="70" y="152" width="300" height="40" rx="6" fill="#64748b15" stroke="#64748b" stroke-width="1.2"/>
  <text x="86" y="170" font-size="12" font-weight="700" fill="#64748b">Rung 1 — Association</text>
  <text x="86" y="185" font-size="10" fill="#64748b">"what correlates?"  ·  P(Y | X)</text>
  <!-- upward arrow -->
  <line x1="40" y1="190" x2="40" y2="58" stroke="#d97706" stroke-width="1.6"/>
  <polygon points="40,50 35,62 45,62" fill="#d97706"/>
  <text x="22" y="128" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700" transform="rotate(-90 22 128)">stronger claims</text>
  <text x="220" y="222" text-anchor="middle" font-size="10.5" fill="#64748b">A vanilla recommender lives on Rung 1. This Expert climbs to 2 and 3.</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Pearl's three rungs. Each rung asks a question the rung below cannot answer; only intervention and counterfactual distinguish causation from coincidence.</figcaption>
</figure>

> **Historical context.** Two giants built modern causal inference from
> different languages. *Jerzy Neyman* (1923) introduced potential
> outcomes; *Donald Rubin* (1974) extended them to observational studies
> as the "Rubin Causal Model." Separately, *Judea Pearl* (2000)
> grounded causation in graph theory through structural causal models
> and do-calculus, work that earned the 2011 Turing Award. The two
> frameworks speak different dialects — potential outcomes versus graphs
> — but are known to be mathematically equivalent. This project's Causal
> Expert is deliberately a hybrid: Pearl's graph view (the adjacency
> matrix $\mathbf W$) plus Rubin's individual-effect view.

## NOTEARS — Learning a DAG by Continuous Optimization

A causal graph must be a **DAG** — directed and *acyclic*. "A causes B,
B causes C, C causes A" is logically impossible (it violates time
order). The hard part is that searching over DAGs is combinatorial and
NP-hard: for just $d=10$ variables there are roughly $4.2\times10^{18}$
possible graphs.

NOTEARS (Zheng et al., *NeurIPS 2018*) is the move that broke this open.
It replaces the combinatorial "is it acyclic?" constraint with a single
*differentiable equality* on the weighted adjacency matrix $\mathbf W$:

$$ h(\mathbf W) = \operatorname{tr}\!\left(e^{\,\mathbf W \odot \mathbf W}\right) - d = 0 $$

Here $\mathbf W \odot \mathbf W$ is the Hadamard (element-wise) square —
which also forces non-negative causal strengths — $e^{(\cdot)}$ is the
matrix exponential, $\operatorname{tr}$ is the trace, and $d = 32$ is
the number of causal variables (`n_causal_vars`).

> **Equation intuition.** The $(i,i)$ diagonal entry of $e^{\mathbf M}$
> is the weighted sum of *every closed walk* from node $i$ back to
> itself, because $(\mathbf M^k)_{ii}$ counts length-$k$ loops and
> $e^{\mathbf M}=\sum_k \mathbf M^k/k!$ sums them over all lengths. In a
> DAG there are no such loops, so every diagonal entry collapses to the
> identity's contribution of $1$, the trace equals $d$, and
> $h(\mathbf W)=0$. A positive $h(\mathbf W)$ is a direct readout of "a
> cycle exists." NOTEARS' one trick is turning a combinatorial graph
> condition into an analytic equation a gradient can chase.

Computing a full matrix exponential is $O(d^3)$, so the project
approximates it with the first 10 Taylor terms — cheap and accurate
because $\mathbf W$ starts tiny (`randn(32,32)*0.01`), so high-order
terms vanish fast. Ten terms means "detect every cycle of length $\le
10$," and 10-hop cycles among 32 nodes essentially never occur in
practice.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <!-- left: cyclic, h(W)>0 -->
  <text x="140" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">cycle present → h(W) &gt; 0</text>
  <g>
    <circle cx="90"  cy="90"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="90"  y="94"  text-anchor="middle" font-size="11" fill="#1e3a5f">A</text>
    <circle cx="190" cy="80"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="190" y="84"  text-anchor="middle" font-size="11" fill="#1e3a5f">B</text>
    <circle cx="140" cy="160" r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="140" y="164" text-anchor="middle" font-size="11" fill="#1e3a5f">C</text>
    <line x1="105" y1="84" x2="175" y2="80" stroke="#e11d48" stroke-width="1.6"/><polygon points="175,80 166,76 167,84" fill="#e11d48"/>
    <line x1="185" y1="94" x2="150" y2="146" stroke="#e11d48" stroke-width="1.6"/><polygon points="150,146 155,137 159,144" fill="#e11d48"/>
    <line x1="128" y1="148" x2="97" y2="104" stroke="#e11d48" stroke-width="1.6"/><polygon points="97,104 105,110 99,114" fill="#e11d48"/>
  </g>
  <!-- divider -->
  <line x1="280" y1="40" x2="280" y2="185" stroke="#e2e8f0" stroke-width="1"/>
  <!-- right: DAG, h(W)=0 -->
  <text x="420" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">acyclic (DAG) → h(W) = 0</text>
  <g>
    <circle cx="370" cy="90"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="370" y="94"  text-anchor="middle" font-size="11" fill="#1e3a5f">A</text>
    <circle cx="470" cy="80"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="470" y="84"  text-anchor="middle" font-size="11" fill="#1e3a5f">B</text>
    <circle cx="420" cy="160" r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="420" y="164" text-anchor="middle" font-size="11" fill="#1e3a5f">C</text>
    <line x1="385" y1="84" x2="455" y2="80" stroke="#0d9488" stroke-width="1.6"/><polygon points="455,80 446,76 447,84" fill="#0d9488"/>
    <line x1="365" y1="104" x2="412" y2="146" stroke="#0d9488" stroke-width="1.6"/><polygon points="412,146 403,142 408,136" fill="#0d9488"/>
    <line x1="463" y1="94" x2="430" y2="146" stroke="#0d9488" stroke-width="1.6"/><polygon points="430,146 434,137 438,144" fill="#0d9488"/>
  </g>
  <text x="420" y="200" text-anchor="middle" font-size="10" fill="#64748b">edge weight = W²ᵢⱼ  (j → i strength)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The acyclicity penalty in one picture. A cycle A→B→C→A inflates the trace above d; a DAG leaves it exactly at d, so h(W)=0.</figcaption>
</figure>

During training the project does not enforce $h(\mathbf W)=0$ exactly
(the paper uses an augmented Lagrangian); it relaxes to a simple penalty
plus a sparsity term:

$$ \mathcal{L}_{\text{DAG}} = \lambda_{\text{acyclic}}\cdot h(\mathbf W) + \lambda_{\text{sparse}}\cdot \lVert \mathbf W \odot \mathbf W \rVert_1 $$

with `dag_lambda = 0.01` and `sparsity_lambda = 0.001`. The first term
forbids cycles; the second prunes the $32\times32=1024$ possible edges
down to the meaningful few. One warning from the reference: push
`dag_lambda` past 0.1 and $\mathbf W$ collapses to the zero matrix — the
Expert degenerates into an identity map ($\hat{\mathbf z}\approx\mathbf
z$) and the causal structure simply evaporates.

## The Learned W and the Structural Equation

Inside the Expert the pipeline is three stages: a **Compressor**
squeezes the input down to 32 causal variables
($734\to128\to32$ under V1), the **structural causal model** intervenes, and a
**Causal Encoder** lifts the result back to 64D ($32\to128\to64$). The
intervention itself is a single, almost suspiciously simple equation:

$$ \hat{\mathbf z} = \mathbf z + \mathbf z(\mathbf W \odot \mathbf W) $$

$\mathbf z$ is the 32-dim latent; $\mathbf W$ is a learnable $[32,32]$
adjacency matrix; $\mathbf W \odot \mathbf W$ gives non-negative edge
strengths where $W_{ij}^2$ is the causal influence of variable $j$ on
variable $i$. The product $\mathbf z(\mathbf W\odot\mathbf W)$ adjusts
each variable by the linear combination of its causal parents, and the
residual `z +` preserves the original signal. The result is a
customer representation that is *causally corrected*, not merely
correlated. After training, `get_causal_graph()` returns
$(\mathbf W \odot \mathbf W).\text{detach()}$ — a $[32,32]$ heatmap of
which latent factor drives which.

## Counterfactuals — What Would Have Happened

This is where the Expert reaches Rung 3. `get_counterfactual(x, j, v)`
applies a hard intervention $do(z_j = v)$ and runs the encoder *three*
ways — and the gap between two of them is the whole point:

$$
\begin{aligned}
\textbf{factual} &= \text{encoder}(\mathbf z + \mathbf z\,\mathbf W^2) \\
\textbf{direct\_only} &= \text{encoder}(\mathbf z' + \mathbf z\,\mathbf W^2) \\
\textbf{full\_cf} &= \text{encoder}(\mathbf z' + \mathbf z'\,\mathbf W^2)
\end{aligned}
$$

where $\mathbf z'$ is $\mathbf z$ with coordinate $j$ overwritten by
$v$. In **direct_only** the intervention touches only $z_j$ itself; the
DAG-mediated path $\mathbf z\,\mathbf W^2$ is held *frozen* at its
pre-intervention value. In **full_cf** the intervention is allowed to
*propagate* through the graph — the mediated term recomputes from
$\mathbf z'$. So the difference

$$ \Delta_{\text{mediated}} = \textbf{full\_cf} - \textbf{direct\_only} $$

is exactly the **effect that flows through the causal graph** — Pearl's
Rung-3 mediated effect. If $\mathbf W$ turns out to be merely
decorative (no real structure learned), the two branches collapse to the
same value and $\Delta_{\text{mediated}}\to 0$. The counterfactual probe
is, among other things, an honesty check on whether the DAG is doing any
work at all.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="250" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">do(z_j = v): three forward variants</text>
  <!-- root -->
  <rect x="20" y="105" width="96" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="68" y="123" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">latent z</text>
  <text x="68" y="138" text-anchor="middle" font-size="9" fill="#64748b">intervene z_j</text>
  <!-- factual -->
  <line x1="116" y1="118" x2="200" y2="60" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="40" width="200" height="44" rx="6" fill="#64748b15" stroke="#64748b" stroke-width="1.1"/>
  <text x="214" y="58" font-size="11" font-weight="700" fill="#64748b">factual</text>
  <text x="214" y="73" font-size="9.5" fill="#64748b">encoder(z + z·W²) — no intervention</text>
  <!-- direct_only -->
  <line x1="116" y1="125" x2="200" y2="128" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="106" width="200" height="44" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.1"/>
  <text x="214" y="124" font-size="11" font-weight="700" fill="#0d9488">direct_only</text>
  <text x="214" y="139" font-size="9.5" fill="#0d9488">encoder(z′ + z·W²) — path frozen</text>
  <!-- full_cf -->
  <line x1="116" y1="132" x2="200" y2="196" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="172" width="200" height="44" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.1"/>
  <text x="214" y="190" font-size="11" font-weight="700" fill="#4f46e5">full_cf</text>
  <text x="214" y="205" font-size="9.5" fill="#4f46e5">encoder(z′ + z′·W²) — path propagates</text>
  <!-- delta bracket -->
  <line x1="412" y1="128" x2="412" y2="194" stroke="#d97706" stroke-width="1.4"/>
  <line x1="412" y1="128" x2="406" y2="128" stroke="#d97706" stroke-width="1.4"/>
  <line x1="412" y1="194" x2="406" y2="194" stroke="#d97706" stroke-width="1.4"/>
  <text x="424" y="156" font-size="10" font-weight="700" fill="#d97706">Δ = full_cf − direct_only</text>
  <text x="424" y="170" font-size="9.5" fill="#64748b">= DAG-mediated effect</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Three forward passes from one intervention. Freezing vs. propagating the W² path isolates exactly the effect the causal graph mediates.</figcaption>
</figure>

The same forward pass feeds two more derived signals the project uses
elsewhere: a **causal coherence score** $\lVert \mathbf z - \mathbf z
\cdot \mathbf W^2 \rVert^2 / \lVert \mathbf z \rVert^2$ (the basis of the
`CausalGuardrail`, a Rung-1 in-distribution check) and a **CEH
attribution** head (Rung 2, grad×input of the task logit). One Expert,
all three rungs, almost no extra compute.

## The Optimal Transport Side — A Customer Is a Distribution

The OT Expert refuses to compress a customer into a single point. It
reads a customer as a *probability distribution* and asks how far that
distribution sits from a set of learned prototypes — using the most
geometrically honest distance there is.

Why not KL divergence or total variation? Because they ignore the
geometry of the underlying space. Put one distribution's mass on Seoul,
another's on Incheon, a third's on Busan. KL and TV report
$\text{dist}(P,Q)\approx\text{dist}(P,R)$ — if the supports don't
overlap, the distance is the same regardless of *how far apart* the mass
sits. Optimal transport fixes this: it accounts for the cost of *moving*
mass across the ground space, so Seoul↔Incheon is genuinely closer than
Seoul↔Busan.

<img src="/optimal-transport.webp" alt="Optimal Transport — source distribution μ (blue cluster) and target distribution ν (red cluster) connected by transport plan γ showing pair-wise sample matchings" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

The classical formulation is the **Monge–Kantorovich** problem. Monge
(1781) framed it as moving a pile of earth at minimum cost; Kantorovich
(1942) relaxed it to a linear program over transport plans (Nobel in
Economics, 1975):

$$ W(\boldsymbol\mu, \boldsymbol\nu) = \min_{\mathbf P \in \mathcal U(\boldsymbol\mu,\boldsymbol\nu)} \langle \mathbf P, \mathbf C\rangle_F, \qquad \mathcal U(\boldsymbol\mu,\boldsymbol\nu) = \{\mathbf P \ge 0 : \mathbf P\mathbf 1 = \boldsymbol\mu,\ \mathbf P^\top\mathbf 1 = \boldsymbol\nu\} $$

$\mathbf P_{ij}$ is the mass moved from $i$ to $j$, $\mathbf C_{ij}$ its
unit cost, and the minimum total cost is the **Wasserstein distance** —
the earth mover's distance. Unlike KL it stays finite even when supports
don't overlap, and it hands back not just a scalar but the full
transport plan $\mathbf P$, an interpretable map of *how* one
distribution morphs into the other.

In the Expert this becomes concrete:

- **Customer distribution.** $\boldsymbol\mu = \operatorname{softmax}(\text{DistProjector}(\mathbf x)) \in \Delta^{32}$ — the feature vector projected to a probability simplex over 32 latent categories.
- **Prototypes.** $\boldsymbol\nu_k = \operatorname{softmax}(\boldsymbol\ell_k) \in \Delta^{32}$, a bank of learnable reference distributions (class default 16, operational `n_ref=8`) — each a "typical customer type" the data clusters into (travel-heavy, savings-heavy, …) learned end-to-end, not hand-defined.
- **Cost matrix.** $\mathbf C = \mathbf M^\top\mathbf M$, a learnable ground metric forced positive-semidefinite by the $\mathbf M^\top\mathbf M$ factorization — so no entry rewards transport (which would make Sinkhorn produce nonsense plans).

## Sinkhorn — Entropic Regularization Buys Speed and Gradients

The bare Kantorovich LP has $d^2$ variables and is expensive at scale.
Cuturi (*NeurIPS 2013*) added an **entropic regularizer**, which turns
the problem into something Sinkhorn iterations solve with *linear*
convergence — and, crucially, makes it differentiable for end-to-end
training:

$$ \min_{\mathbf P \in \mathcal U(\boldsymbol\mu,\boldsymbol\nu)} \langle \mathbf P, \mathbf C\rangle - \varepsilon\, H(\mathbf P), \qquad H(\mathbf P) = -\sum_{i,j} P_{ij}\log P_{ij} $$

> **Equation intuition.** The entropy term $-\varepsilon H(\mathbf P)$
> penalizes transport plans that are too "spiky" (all mass on one
> route), nudging the plan toward smoothness. Large $\varepsilon$ blurs
> the plan toward uniform (you lose cost information); small
> $\varepsilon$ is sharp but numerically unstable. The project uses
> $\varepsilon = 0.1$ (`sinkhorn_epsilon`) — the reference flags below
> 0.01 as divergence/NaN territory and above 1.0 as blurry transport.

Solving it is alternating normalization of the dual variables, done in
the **log domain** for numerical safety (so the tiny Gibbs-kernel
entries $e^{-C_{ij}/\varepsilon}$ never underflow):

$$
\begin{aligned}
\mathbf u_{\text{new}} &= \log\boldsymbol\mu - \operatorname{logsumexp}\!\left(-\mathbf C/\varepsilon + \mathbf v\right) \\
\mathbf v_{\text{new}} &= \log\boldsymbol\nu - \operatorname{logsumexp}\!\left(-\mathbf C^\top/\varepsilon + \mathbf u\right)
\end{aligned}
$$

Each step matches the plan's row sums to $\boldsymbol\mu$ and its column
sums to $\boldsymbol\nu$; `logsumexp` is a log-domain softmax that holds
off floating-point underflow. The class default runs 10 iterations
(operational config 5). After convergence the transport plan is $\log
P_{ij} = u_i + \log K_{ij} + v_j$ and the distance is the Frobenius
inner product $W(\boldsymbol\mu,\boldsymbol\nu_k) = \langle \mathbf P,
\mathbf C\rangle_F$. Run it against all 16 prototypes and you get a
$[B,16]$ Wasserstein-distance vector — a *distributional coordinate
system* locating each customer by its distance to 16 reference points.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="170" fill="#f8fafc" rx="8"/>
  <!-- x input -->
  <rect x="16" y="62" width="78" height="46" rx="6" fill="#f0fdf4" stroke="#0d9488" stroke-width="1"/>
  <text x="55" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">x [734]</text>
  <text x="55" y="97" text-anchor="middle" font-size="9" fill="#64748b">features</text>
  <!-- softmax mu -->
  <rect x="120" y="62" width="92" height="46" rx="6" fill="#fce7f3" stroke="#e11d48" stroke-width="1"/>
  <text x="166" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">μ ∈ Δ³²</text>
  <text x="166" y="97" text-anchor="middle" font-size="9" fill="#64748b">softmax proj</text>
  <!-- prototypes nu + cost -->
  <rect x="238" y="22" width="100" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="288" y="40" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">16 × ν_k</text>
  <text x="288" y="54" text-anchor="middle" font-size="9" fill="#64748b">prototypes</text>
  <rect x="238" y="108" width="100" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="288" y="126" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">C = MᵀM</text>
  <text x="288" y="140" text-anchor="middle" font-size="9" fill="#64748b">PSD cost</text>
  <!-- sinkhorn -->
  <rect x="362" y="62" width="100" height="46" rx="6" fill="#fce7f3" stroke="#e11d48" stroke-width="1.2"/>
  <text x="412" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Sinkhorn</text>
  <text x="412" y="97" text-anchor="middle" font-size="9" fill="#64748b">log-domain ×10</text>
  <!-- wasserstein vec -->
  <rect x="486" y="62" width="96" height="46" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="534" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">W [B,16]</text>
  <text x="534" y="97" text-anchor="middle" font-size="9" fill="#64748b">→ 64D enc</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="94" y1="85" x2="118" y2="85"/><polygon points="118,85 110,81 110,89"/>
    <line x1="212" y1="85" x2="360" y2="85"/><polygon points="360,85 352,81 352,89"/>
    <line x1="462" y1="85" x2="484" y2="85"/><polygon points="484,85 476,81 476,89"/>
    <line x1="338" y1="42" x2="362" y2="74"/><polygon points="362,74 353,71 359,67"/>
    <line x1="338" y1="128" x2="362" y2="96"/><polygon points="362,96 359,105 353,99"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The OT Expert forward pass: project to a simplex, Sinkhorn against 16 learnable prototypes with a PSD cost, read out a 16-vector of Wasserstein distances, encode to 64D.</figcaption>
</figure>

A final 2-layer **Wasserstein Encoder** ($16\to128\to64$) lifts that
distance vector into the shared 64D space, learning nonlinear distance
*patterns* ("close to the travel type but far from the savings type").
Note one asymmetry with the Causal Expert: OT carries **no separate
regularization loss** — Sinkhorn's entropic term already regularizes
internally, and the prototypes and cost matrix simply learn through the
main task gradient.

## Where the Experts Plug Into PLE

Both Experts take the **734D** feature tensor under the V1 contract
(`input_dim: 734` in `model_config.yaml`) and emit **64D**. Under
operational V2 group routing splits them: Causal takes `base` +
`multi_source` + `domain` + `multidisciplinary` + `model_derived` at
**539D**, OT takes `extended_source` + `multi_source` at **175D**. v3.2 widened the CGC Gate
Attention from $[B,5]$ to $[B,7]$ to admit them alongside the existing
PersLay / DeepFM / Temporal / Unified H-GCN Experts. The gate then mixes
all seven per task across the project's 16 task towers.

Why keep Causal and OT *separate* rather than fusing them into one
Expert? The reference gives three reasons:

| Reason | Why it matters |
| --- | --- |
| Gradient interference | NOTEARS' acyclicity ($\operatorname{tr}(e^{\mathbf W\odot\mathbf W})=d$) and Sinkhorn's entropy ($\varepsilon H(\mathbf P)$) carve completely different loss surfaces; co-training slows both |
| Independent gating | the CGC gate can weight Causal high for churn, OT high for cross-sell — impossible if fused |
| Swappability | Causal can swap NOTEARS→GES/PC, OT can swap Sinkhorn→Sliced-Wasserstein, independently |

Three Experts (DeepFM, Causal, OT) read the *same* 734D under V1 yet contribute
disjoint structure: DeepFM the symmetric pair interaction
$\langle\mathbf v_i,\mathbf v_j\rangle$, Causal the asymmetric acyclic
direction $W_{ij}^2$, OT the metric distance $W(\boldsymbol\mu,
\boldsymbol\nu_k)$. Same input, three irreducibly different questions.

## Where We Stop

We started from a confounder — premium cards and travel insurance — and
climbed Pearl's ladder: association, then NOTEARS' differentiable
acyclicity that lets a gradient learn a DAG, then the structural
equation $\hat{\mathbf z}=\mathbf z+\mathbf z(\mathbf W\odot\mathbf W)$,
then counterfactuals where freezing versus propagating the $\mathbf W^2$
path isolates exactly the DAG-mediated effect. Then we crossed to the OT
side: a customer as a distribution, Monge–Kantorovich and Wasserstein,
and Sinkhorn's entropic regularization making it fast and
differentiable. Two Experts, two mathematical worldviews — "why this"
and "how well it fits" — feeding one gate.

What remains is *time*. Causal and OT both read a customer as a static
snapshot; neither sees the tempo of money — the rhythm of when spending
accelerates, stalls, or churns. The next sub-thread takes up the
**Temporal Expert**: a state-space model (Mamba), a continuous-time
liquid neural network, and a patch Transformer combined in ensemble —
three different clocks for one customer, and why three notions of
"memory" beat one. That is **TEMPORAL-1**.
