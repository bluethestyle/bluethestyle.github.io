---
title: "[Study Thread] TEMPORAL-1 — Three Clocks for One Customer: Mamba, Liquid Networks, and a Patch Transformer in Ensemble"
date: 2026-06-06 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, temporal, mamba, ssm, transformer, expert]
lang: en
excerpt: "The Temporal Expert sub-thread opens — why a customer's spending is a sequence, not a snapshot, and why the project reads it with not one but three sequence models. The O(n²) attention vs linear-time SSM tradeoff, Mamba's selective-scan recurrence, the Liquid Neural Network's input-dependent time constant, PatchTST's patch-wise attention, and the softmax gate that fuses all three into one 64D vector for the PLE."
series: study-thread
part: 16
alt_lang: /2026/06/06/temporal-ensemble-ko/
next_title: "ECON-1 — The Economics-Feature Expert: Reading the Macro Tide Under a Customer's Spend"
next_desc: "The next sub-thread turns from time to context: how an Expert injects macroeconomic and market signal — rates, inflation, sector indices — into a per-customer recommendation, and why a personal time series needs an exogenous economic frame to be read correctly."
next_status: draft
---

*First post of the Temporal Expert sub-thread in the "Study Thread"
series. Across this and the following posts, in parallel Korean and
English, I unpack the Temporal Ensemble Expert — one of the seven
heterogeneous Shared Experts in this project. The source is the on-prem
reference `기술참조서/Temporal_기술_참조서`, and the full PDF will be
attached to the final post of the sub-thread. Where the TDA sub-thread
asked what* shape *behavior has, this one asks what* rhythm *it has — the
order, the intervals, the trends that a single-snapshot feature throws
away. The project's answer is unusual: it does not pick one sequence
model. It runs three at once — a selective state-space model, a liquid
ODE network, and a patch Transformer — and lets a gate decide who to
trust.*

> **One Expert, three sub-models, on purpose.** Most teams pick a
> sequence architecture and commit. This Expert refuses to. Mamba carries
> long-range sequential dependency in linear time; the Liquid Neural
> Network corrects for irregular gaps between transactions through an
> input-dependent time constant; PatchTST matches global periodicity with
> patch-wise attention. A learnable softmax gate observes all three
> outputs and assigns weights per input — high PatchTST weight for a
> strongly periodic customer, high LNN weight for an erratic one. The
> design bet is that no single clock reads every customer, so the Expert
> carries three and reads the dial.

## Time Is the Fourth Dimension of a Customer

A classic recommender describes a user with *static* features: age,
occupation, preferred category. These are fixed regardless of *when* you
look. But real behavior moves along a *time axis* and never stops moving.

Take a customer who "eats out every Friday, pays utilities on the 1st of
every month, and has been slowly increasing coffee spend over the last
three months." Collapse that into a single snapshot — *current monthly
average spend* — and you destroy three load-bearing signals at once:
*periodicity*, *trend*, and *seasonality*.

| View | Static feature | Temporal feature |
| --- | --- | --- |
| Representation | fixed vector $\mathbf{x}\in\mathbb{R}^d$ | sequence $\mathbf{X}\in\mathbb{R}^{T\times d}$ |
| Information loss | time-axis averaging → pattern dies | order, interval, trend preserved |
| Model needed | MLP, embedding table | RNN, SSM, Transformer |
| Example | monthly avg: 1.5M KRW | daily series: [12, 0, 5, 0, 0, 85, 15, …] |

The Temporal Expert solves exactly this. It keeps transaction and session
data *as sequences*, learns the patterns buried in the time dimension,
and compresses them into a 64D representation the rest of the model can
use.

## The Tradeoff: O(n²) Attention vs Linear-Time Recurrence

There are two great families of sequence models, and they disagree about
how to reach across time.

A **Transformer** computes self-attention — every position attends to
every other position directly:

$$ \mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V} $$

This is wonderful for *global pattern matching*: the distance between two
timesteps does not matter, so a January habit and a December habit are
one matrix entry apart. The cost is that the $\mathbf{Q}\mathbf{K}^\top$
matrix is $L\times L$ — **$O(L^2)$** in time and memory. For a long
transaction history this is exactly where it hurts.

A **state-space model (SSM)**, by contrast, propagates information
*sequentially* through a hidden state, like an RNN — **$O(L)$**, linear
in sequence length. The price is the mirror image of the Transformer's:
information from the far past *decays* as it passes through the state,
making very long-range matching harder.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <!-- Left: attention all-pairs -->
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Attention — all pairs, O(n²)</text>
  <g id="attn">
    <circle cx="70"  cy="80" r="7" fill="#4f46e5"/><circle cx="130" cy="80" r="7" fill="#4f46e5"/>
    <circle cx="190" cy="80" r="7" fill="#4f46e5"/><circle cx="250" cy="80" r="7" fill="#4f46e5"/>
  </g>
  <g stroke="#4f46e5" stroke-width="0.8" opacity="0.5" fill="none">
    <path d="M70 80 Q100 150 130 80"/><path d="M70 80 Q130 175 190 80"/><path d="M70 80 Q160 200 250 80"/>
    <path d="M130 80 Q160 150 190 80"/><path d="M130 80 Q190 175 250 80"/>
    <path d="M190 80 Q220 150 250 80"/>
  </g>
  <text x="150" y="225" text-anchor="middle" font-size="11" fill="#64748b">every pair connected directly</text>
  <text x="150" y="242" text-anchor="middle" font-size="10" fill="#e11d48" font-weight="700">cost grows like L²</text>
  <!-- divider -->
  <line x1="300" y1="45" x2="300" y2="215" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Right: SSM chain -->
  <text x="450" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">SSM — recurrence chain, O(n)</text>
  <g>
    <circle cx="360" cy="120" r="7" fill="#0d9488"/><circle cx="420" cy="120" r="7" fill="#0d9488"/>
    <circle cx="480" cy="120" r="7" fill="#0d9488"/><circle cx="540" cy="120" r="7" fill="#0d9488"/>
  </g>
  <g stroke="#0d9488" stroke-width="1.6" fill="#0d9488">
    <line x1="367" y1="120" x2="413" y2="120"/><polygon points="413,120 405,116 405,124"/>
    <line x1="427" y1="120" x2="473" y2="120"/><polygon points="473,120 465,116 465,124"/>
    <line x1="487" y1="120" x2="533" y2="120"/><polygon points="533,120 525,116 525,124"/>
  </g>
  <g font-size="10" fill="#64748b" text-anchor="middle">
    <text x="360" y="145">h₁</text><text x="420" y="145">h₂</text><text x="480" y="145">h₃</text><text x="540" y="145">h₄</text>
  </g>
  <text x="450" y="225" text-anchor="middle" font-size="11" fill="#64748b">state carried forward one step at a time</text>
  <text x="450" y="242" text-anchor="middle" font-size="10" fill="#0d9488" font-weight="700">cost grows like L</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Two ways to reach across time. Attention links every pair directly (O(n²), strong long-range, expensive). An SSM threads a single hidden state through the sequence (O(n), cheap, but the past decays).</figcaption>
</figure>

| Generation | Model | Weakness | Successor |
| --- | --- | --- | --- |
| 2nd | LSTM / GRU (gated RNN) | $O(L)$ sequential bottleneck, vanishing gradient | Transformer |
| 3rd | Transformer (self-attention) | $O(L^2)$ cost, weak order info | SSM, PatchTST |
| 4th | SSM + ODE + Patch Transformer **ensemble** | more model complexity, gate-collapse risk | **this project's Temporal Expert** |

> **Historical context.** Transformers (Vaswani et al., 2017) revolutionized
> NLP and were then adapted to time series. But Zeng et al. (AAAI 2023)
> threw a punch — a plain linear model beat elaborate time-series
> Transformers — and the field re-balanced. S4 (Gu et al., ICLR 2022) brought
> state-space models to deep sequence modeling, Mamba (Gu & Dao, 2023)
> broke the linear-time-invariant limit with a *selective* mechanism, and
> PatchTST (Nie et al., 2023) made attention efficient by working on
> patches. This Expert does not crown a winner; it ensembles the survivors.

## Mamba — A State-Space Model That Chooses What to Remember

Start from the continuous linear time-invariant (LTI) system that
underlies every SSM:

$$ \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}u,\qquad y = \mathbf{C}\mathbf{x} + \mathbf{D}u $$

Here $\mathbf{x}\in\mathbb{R}^N$ is a hidden state, $u$ the input signal,
$\mathbf{A}$ the state-transition matrix, $\mathbf{B}$ / $\mathbf{C}$ the
input / output matrices. To run this on a discrete transaction sequence,
we *discretize* it with a step $\Delta$ using zero-order hold:

$$ \bar{\mathbf{A}} = \exp(\Delta\,\mathbf{A}),\qquad \bar{\mathbf{B}} = (\Delta\,\mathbf{A})^{-1}\big(\bar{\mathbf{A}} - \mathbf{I}\big)\cdot\Delta\,\mathbf{B} $$

which turns the differential equation into a discrete recurrence:

$$ \mathbf{h}_t = \bar{\mathbf{A}}\,\mathbf{h}_{t-1} + \bar{\mathbf{B}}\,\mathbf{x}_t,\qquad \mathbf{y}_t = \mathbf{C}_t\,\mathbf{h}_t $$

> **Equation intuition.** This recurrence is a generalization of an RNN.
> $\bar{\mathbf{A}}$ decides *how much of the old memory to keep*,
> $\bar{\mathbf{B}}$ *how much of the new input to admit*, and
> $\mathbf{C}_t$ is a read head pulling only the needed information out of
> the state. The discretization step $\Delta$ is the knob: large $\Delta$
> remembers an input for a long time (slow dynamics), small $\Delta$
> forgets it quickly (fast dynamics). In a classic SSM these matrices are
> constants — the same rule at every timestep — and the whole thing
> collapses to a convolution you can parallelize.

The trouble with that constancy: an LTI system processes *every* input by
the same rule. A 5,000-KRW coffee and a 5,000,000-KRW transfer get the
same memory treatment. Mamba's **S6 selective mechanism** fixes this by
making $\Delta$, $\mathbf{B}$, $\mathbf{C}$ *input-dependent*:

$$ \Delta = \mathrm{softplus}(\mathbf{W}_\Delta\mathbf{x} + \mathbf{b}_\Delta),\quad \mathbf{B} = \mathbf{W}_B\mathbf{x},\quad \mathbf{C} = \mathbf{W}_C\mathbf{x} $$

The softplus guarantees $\Delta > 0$ (a timestep must be positive). Now
the *transition rule itself depends on the content*: a large transaction
pushes $\Delta$ up and is strongly remembered; a routine small one keeps
$\Delta$ low and is processed as background. The price is that the model
is no longer time-invariant, so the convolution shortcut is gone — Mamba
recovers efficiency with a hardware-aware *selective scan*. In this
project's online instance, Mamba runs at `d_model=128`, `d_state=16` over
the 180-step transaction sequence.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="210" fill="#f8fafc" rx="8"/>
  <text x="300" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Selective recurrence, unrolled — hₜ = Ā hₜ₋₁ + B̄ xₜ</text>
  <!-- inputs -->
  <g font-size="11" fill="#64748b" text-anchor="middle">
    <text x="110" y="70">x₁</text><text x="240" y="70">x₂</text><text x="370" y="70">x₃</text><text x="500" y="70">x₄</text>
  </g>
  <g stroke="#64748b" stroke-width="1.2" fill="#64748b">
    <line x1="110" y1="78" x2="110" y2="108"/><polygon points="110,108 106,100 114,100"/>
    <line x1="240" y1="78" x2="240" y2="108"/><polygon points="240,108 236,100 244,100"/>
    <line x1="370" y1="78" x2="370" y2="108"/><polygon points="370,108 366,100 374,100"/>
    <line x1="500" y1="78" x2="500" y2="108"/><polygon points="500,108 496,100 504,100"/>
  </g>
  <!-- state boxes -->
  <g>
    <rect x="80"  y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="210" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="340" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="470" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
  </g>
  <g font-size="12" fill="#0d9488" font-weight="700" text-anchor="middle">
    <text x="110" y="135">h₁</text><text x="240" y="135">h₂</text><text x="370" y="135">h₃</text><text x="500" y="135">h₄</text>
  </g>
  <!-- A-bar transitions -->
  <g stroke="#1e3a5f" stroke-width="1.6" fill="#1e3a5f">
    <line x1="140" y1="130" x2="208" y2="130"/><polygon points="208,130 200,126 200,134"/>
    <line x1="270" y1="130" x2="338" y2="130"/><polygon points="338,130 330,126 330,134"/>
    <line x1="400" y1="130" x2="468" y2="130"/><polygon points="468,130 460,126 460,134"/>
  </g>
  <g font-size="10" fill="#1e3a5f" text-anchor="middle">
    <text x="174" y="122">Ā</text><text x="304" y="122">Ā</text><text x="434" y="122">Ā</text>
  </g>
  <!-- selectivity callouts -->
  <text x="240" y="180" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700">large txn → Δ↑ → strongly remembered</text>
  <text x="370" y="196" text-anchor="middle" font-size="10" fill="#64748b">small txn → Δ↓ → kept as background</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The state-space recurrence unrolled. Each step carries hₜ forward through Ā and admits the new input through B̄. Because Mamba makes Δ, B, C depend on the input, the memory/forget rule changes per transaction.</figcaption>
</figure>

## LNN — A Liquid Time Constant for Irregular Gaps

Transactions do not arrive on a clock. The gap between two purchases
might be ten minutes or ten days, and that *interval itself* is signal.
The **Liquid Neural Network** is a continuous-time (Neural ODE) model
whose time constant adapts to the input:

$$ \frac{d\mathbf{h}}{dt} = \frac{-\mathbf{h} + f(\mathbf{x},\mathbf{h})}{\tau(\mathbf{x},\mathbf{h})} $$

The intuition is "move from the current state toward a target state
$f(\mathbf{x},\mathbf{h})$, at speed $\tau$." A large $\tau$ means slow
change (a long-dormant customer whose state decays gently); a small
$\tau$ means fast reaction (an active spending burst). Crucially, the LNN
*generates $\tau$ from the input*, so the same customer relaxes at
different rates in different periods. To run it on discrete steps, the
project uses a single first-order Euler update — `LNNSingleStep`:

$$ \mathbf{h}_{t+1} = \mathbf{h}_t + \Delta t\cdot\frac{-\mathbf{h}_t + f(\mathbf{x}_t,\mathbf{h}_t)}{\tau(\mathbf{x}_t,\mathbf{h}_t)} $$

where $\Delta t$ is the *real* inter-event interval (in days), clamped to
`[0.001, 30.0]` — about 1.4 minutes to 30 days — both to keep it
meaningful and because Euler can oscillate when $\Delta t > \tau$. The
design intent matters here: the LNN is run *in series after* Mamba (it
corrects Mamba's final state with time-awareness), rather than as a full
parallel sequence ODE, because a full ODE would duplicate Mamba's work
and cost more for little gain.

## PatchTST — Attention, but on Patches

The third sub-model is the Transformer side of the bet. Plain
self-attention on every timestep is the $O(L^2)$ trap from earlier.
**PatchTST** sidesteps it by chopping the series into *patches* (the
project uses `patch_size=16`) and attending *patch-to-patch* rather than
point-to-point. The complexity drops to $O((L/P)^2)$, and — more
importantly — a patch is a small local window, which is a better unit for
spotting *global periodicity* (weekly, monthly cycles) than an individual
day. PatchTST takes the *raw* sequence as input independently, by design:
unlike the LNN, it is not fed Mamba's processed state — sharing that
input would shrink the gate's ability to differentiate the models, so
the project separates inputs to keep the ensemble diverse.

## The Gate — Three Outputs, One Vote

Now the three sub-models have each produced a representation, and the
ensemble must combine them. A learnable gate observes all three and emits
a convex weighting:

$$ \mathbf{g} = \mathrm{Softmax}\!\big(\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{z}_{\mathrm{cat}} + \mathbf{b}_1) + \mathbf{b}_2\big),\qquad \mathbf{y} = \sum_{i=1}^{3} g_i\cdot\mathrm{Proj}_i(\mathbf{z}_i) $$

where $\mathbf{z}_{\mathrm{cat}}\in\mathbb{R}^{384}$ is the concatenation
of the three model outputs ($192 + 96 + 96 = 384$), a 2-layer MLP maps
$384\to6\to3$, and softmax forces $g_1 + g_2 + g_3 = 1$. Each model's
output is first projected to a common 64D space ($\mathrm{Proj}_i$) so the
weighted sum is meaningful. Geometrically, $\mathbf{y}$ is a point inside
the triangle whose vertices are the three projected outputs, and
$\mathbf{g}$ is its barycentric coordinate.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="270" fill="#f8fafc" rx="8"/>
  <!-- three model boxes -->
  <rect x="30"  y="40" width="120" height="44" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1.2"/>
  <text x="90" y="60" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">Mamba</text>
  <text x="90" y="76" text-anchor="middle" font-size="10" fill="#64748b">192D</text>
  <rect x="30"  y="110" width="120" height="44" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="90" y="130" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">LNN</text>
  <text x="90" y="146" text-anchor="middle" font-size="10" fill="#64748b">96D</text>
  <rect x="30"  y="180" width="120" height="44" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1.2"/>
  <text x="90" y="200" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">PatchTST</text>
  <text x="90" y="216" text-anchor="middle" font-size="10" fill="#64748b">96D</text>
  <!-- concat -->
  <rect x="210" y="100" width="90" height="64" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>
  <text x="255" y="128" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">Concat</text>
  <text x="255" y="144" text-anchor="middle" font-size="10" fill="#64748b">384D</text>
  <!-- gate -->
  <rect x="340" y="100" width="100" height="64" rx="6" fill="#e11d4818" stroke="#e11d48" stroke-width="1.2"/>
  <text x="390" y="124" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Gate MLP</text>
  <text x="390" y="139" text-anchor="middle" font-size="9.5" fill="#64748b">384→6→3</text>
  <text x="390" y="153" text-anchor="middle" font-size="9.5" fill="#64748b">+ Softmax</text>
  <!-- output -->
  <rect x="478" y="108" width="60" height="48" rx="6" fill="#1e3a5f"/>
  <text x="508" y="130" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">y</text>
  <text x="508" y="145" text-anchor="middle" font-size="9.5" fill="#fff">64D</text>
  <!-- arrows model->concat -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="150" y1="62"  x2="208" y2="120"/><polygon points="208,120 199,118 203,111"/>
    <line x1="150" y1="132" x2="208" y2="132"/><polygon points="208,132 200,128 200,136"/>
    <line x1="150" y1="202" x2="208" y2="144"/><polygon points="208,144 203,153 199,146"/>
  </g>
  <!-- concat->gate->out -->
  <g fill="#94a3b8" stroke="#94a3b8" stroke-width="1.6">
    <line x1="300" y1="132" x2="338" y2="132"/><polygon points="338,132 330,128 330,136"/>
    <line x1="440" y1="132" x2="476" y2="132"/><polygon points="476,132 468,128 468,136"/>
  </g>
  <text x="458" y="124" text-anchor="middle" font-size="9" fill="#e11d48" font-weight="700">g₁,g₂,g₃</text>
  <!-- weighted sum note -->
  <text x="280" y="245" text-anchor="middle" font-size="11" fill="#64748b">y = Σ gᵢ · Projᵢ(zᵢ),  with g₁+g₂+g₃ = 1  (a convex vote)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Ensemble gating. The three outputs are concatenated to 384D; a small MLP + softmax produces a 3-way weight; each model is projected to 64D and convex-combined. The result is one 64D vector — and the gate's weights are a soft, per-input vote of confidence.</figcaption>
</figure>

> **Why this is a lightweight MoE.** This is the Mixture-of-Experts idea
> (Jacobs et al., 1991) in its smallest useful form — Dense MoE, where
> *all* experts are always active and only the weights move, as in Soft
> MoE (Google, 2023). With just three sub-models the project can skip the
> load-balancing loss that large sparse MoEs need, and instead manage
> *gate collapse* — one model hijacking all the weight — by monitoring the
> gate's Shannon entropy. A persistent entropy below ~0.3 is the warning
> sign that the ensemble has degenerated to a single model, logged to
> MLflow as `temporal_gate_entropy`.

## Where the Expert Plugs Into PLE

The Expert consumes two sequences and emits one vector. Inputs:
`txn_seq` of shape `[B, 180, 16]` (16 = card 8D + deposit 8D) and
`session_seq` of shape `[B, 90, 4]` (4 = `sess_duration` / `page_views` /
`elapsed_sec` / `hour`); each sub-model keeps separate
txn/session instances and concatenates them before the gate. Output: a
single **64D** vector that feeds the PLE CGC gate, which mixes it with the
other Experts per task.

There is a deliberate safety valve. For a batch with no sequence — a
cold-start customer — the Expert returns a **zero vector** of width 64.
The PLE's CGC gate then automatically down-weights it, and the other
Experts (DeepFM, LightGCN, and the rest) compensate. Even a missing
`session_seq` falls back to a zero tensor to keep shapes compatible.

The Expert is wired as a `domain_experts` member for **twelve** tasks,
spanning every group where rhythm carries signal:

| Group | Tasks | What temporal contributes |
| --- | --- | --- |
| Engagement | ctr, cvr, engagement | click-timing, purchase-journey, session patterns |
| Lifecycle | churn, retention, life_stage, ltv | frequency-decline trend, long-horizon trajectory |
| Value | balance_util, channel, timing | balance trend, channel time-of-day, 28-day (4×7) periodicity |
| Consumption | consumption_cycle, merchant_affinity | 7-type spend cycle, merchant-visit series |

## Where We Stop

We started from a discomfort with snapshots — a monthly average that
erases the Friday dinner, the 1st-of-the-month bill, the slow coffee
creep. We walked the core tradeoff (O(n²) attention vs O(n) recurrence),
then met the three clocks the project keeps: Mamba's selective scan for
long-range dependency, the Liquid Network's input-dependent $\tau$ for
irregular gaps, and PatchTST's patch attention for global periodicity.
Finally we saw the softmax gate fuse them into one 64D vector, with a
zero fallback and an entropy alarm to keep the ensemble honest, feeding
twelve tasks through the PLE.

What we did *not* do is open the boxes: the exact selective-scan
implementation, the LNN cell's tau-net, the PatchTST encoder internals,
and how the gate entropy is actually computed and monitored in training.
Those are the machinery of the next Temporal post. But the broader thread
now turns outward. A customer's time series does not move in a vacuum —
it moves inside an economy of rates, inflation, and sector cycles. The
next sub-thread, **ECON-1**, takes up the Economics-feature Expert: how to
read the macro tide flowing *under* a single customer's spend.
