---
title: "[Study Thread] TDAFEAT-1 — The Offline TDA Feature Pipeline: From Raw Logs to 70D in the Main Tensor"
date: 2026-06-07 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, persistent-homology, feature-engineering, offline]
lang: en
excerpt: "The TDA-1 post built the PersLay Expert — an online Shared Expert that consumes persistence diagrams end-to-end. This post is about the other half of the topology story: the offline, batch-precomputed TDA features. Raw transaction and app logs become 6D point clouds, Ripser turns them into persistence diagrams, and a fixed set of statistics and Persistence Entropy vectorizes them into the 70D block that lives inside the 734D main tensor. The vectorization choices, the actual feature breakdown, and the cost trade-offs that justify doing it offline."
series: study-thread
part: 18
alt_lang: /2026/06/07/tda-features-offline-ko/
next_title: "HMM-1 — Regime Features: Hidden Markov Models for Life-Stage Detection"
next_desc: "The Model-Derived block carries a 5D HMM summary. How a hidden Markov model reads a customer's transaction stream as a sequence of latent regimes, what the Viterbi path and posterior entropy encode, and why those five numbers complement the topological view of behavioral change."
next_status: draft
---

*A post in the "Study Thread" series, drawn from the on-prem reference
`기술참조서/TDA_피처_기술_참조서`. This is the companion to the TDA-1 /
PersLay post: where TDA-1 introduced persistent homology and the* online
*PersLay Expert that learns a 64D vector from persistence diagrams
end-to-end, this post covers the* offline *side — the batch-precomputed
TDA features that become a fixed 70D slice of the 734D main tensor. I
will not re-teach the homology basics here; for filtrations, persistence
diagrams, and the stability theorem, see TDA-1. The full PDF of the
reference will be attached to the final post of the sub-thread.*

> **Two pipelines, one mathematics.** Both the offline TDA features and
> the online PersLay Expert sit on persistent homology, but they are
> different machines doing different jobs. PersLay is a *trainable*
> Shared Expert: it ingests a persistence diagram and learns a 64D
> representation by back-propagating the recommendation loss. The
> offline TDA features are *fixed*: raw logs in, a deterministic 70D
> vector out, computed once in the Airflow batch and frozen into the
> main tensor. No gradients, no learning — just a hand-chosen
> vectorization of the diagram. This post is entirely about that second
> machine.

## Offline Features vs. Online Expert

The project deliberately runs topology through *two* doors. The reason
is cost asymmetry: computing a persistence diagram is expensive and
input-shaped (point clouds vary per customer), whereas a model wants a
fixed-width vector it can read every forward pass. So one path
precomputes a frozen descriptor; the other learns from the diagram
inside the network.

| Aspect | TDA features (this post) | PersLay Expert (TDA-1) |
| --- | --- | --- |
| Role | Offline feature extraction | Online PLE Shared Expert |
| Input | Raw transaction / app logs | Persistence diagram |
| Output | 70D → part of 734D main tensor | 64D → PLE CGC gate |
| When | Batch preprocessing (Airflow) | Training / inference (end-to-end) |
| Learnable? | No — fixed statistics | Yes — φ, w, ρ trainable |
| Vectorization | Persistence Entropy + lifetime statistics | RationalHat φ + persistence weight |

The two are not redundant. The 70D offline block lands directly in the
input tensor that *every* Expert and task reads, so even tasks that do
not wire in the PersLay Expert still get a topological signal for free.
And the offline diagram can serve as a precomputed fallback input to the
PersLay Expert when on-the-fly extraction is not available.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <!-- offline lane -->
  <rect x="16" y="40" width="588" height="86" rx="8" fill="#4f46e508" stroke="#4f46e5" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="30" y="60" font-size="12" font-weight="700" fill="#4f46e5">OFFLINE · Airflow batch · computed once, frozen</text>
  <rect x="30" y="72" width="92" height="42" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="76" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">raw logs</text>
  <text x="76" y="105" text-anchor="middle" font-size="9" fill="#64748b">txn / app</text>
  <rect x="158" y="72" width="92" height="42" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="204" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">Ripser</text>
  <text x="204" y="105" text-anchor="middle" font-size="9" fill="#64748b">→ diagram</text>
  <rect x="286" y="72" width="100" height="42" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="336" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">vectorize</text>
  <text x="336" y="105" text-anchor="middle" font-size="9" fill="#64748b">entropy + stats</text>
  <rect x="422" y="72" width="70" height="42" rx="6" fill="#d97706"/>
  <text x="457" y="91" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">70D</text>
  <text x="457" y="105" text-anchor="middle" font-size="8.5" fill="#fff">fixed</text>
  <rect x="516" y="66" width="74" height="54" rx="6" fill="#1e3a5f"/>
  <text x="553" y="88" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">734D</text>
  <text x="553" y="102" text-anchor="middle" font-size="8.5" fill="#cbd5e1">main tensor</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="122" y1="93" x2="156" y2="93"/><polygon points="156,93 148,89 148,97"/>
    <line x1="250" y1="93" x2="284" y2="93"/><polygon points="284,93 276,89 276,97"/>
    <line x1="386" y1="93" x2="420" y2="93"/><polygon points="420,93 412,89 412,97"/>
    <line x1="492" y1="93" x2="514" y2="93"/><polygon points="514,93 506,89 506,97"/>
  </g>
  <!-- online lane -->
  <rect x="16" y="146" width="588" height="86" rx="8" fill="#0d948808" stroke="#0d9488" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="30" y="166" font-size="12" font-weight="700" fill="#0d9488">ONLINE · training / inference · learns every forward pass</text>
  <rect x="30" y="178" width="92" height="42" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="76" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f">diagram</text>
  <text x="76" y="211" text-anchor="middle" font-size="9" fill="#64748b">point set</text>
  <rect x="158" y="178" width="120" height="42" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="218" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f">PersLay φ·w·ρ</text>
  <text x="218" y="211" text-anchor="middle" font-size="9" fill="#64748b">trainable</text>
  <rect x="314" y="178" width="70" height="42" rx="6" fill="#0d9488"/>
  <text x="349" y="197" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">64D</text>
  <text x="349" y="211" text-anchor="middle" font-size="8.5" fill="#fff">learned</text>
  <rect x="420" y="172" width="100" height="54" rx="6" fill="#1e3a5f"/>
  <text x="470" y="194" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">PLE CGC gate</text>
  <text x="470" y="208" text-anchor="middle" font-size="8.5" fill="#cbd5e1">per task</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="122" y1="199" x2="156" y2="199"/><polygon points="156,199 148,195 148,203"/>
    <line x1="278" y1="199" x2="312" y2="199"/><polygon points="312,199 304,195 304,203"/>
    <line x1="384" y1="199" x2="418" y2="199"/><polygon points="418,199 410,195 410,203"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Two lanes from the same homology. Top: the offline batch produces a frozen 70D slice of the 734D tensor. Bottom: the online PersLay Expert learns a 64D vector that feeds the PLE gate. This post is the top lane.</figcaption>
</figure>

## The Extraction Pipeline

The offline pipeline is a four-step narrative, run per customer inside
the Airflow batch. Nothing about it is learned — every step is a fixed
transform.

1. **Point cloud.** Each transaction (or app session) becomes a point
   in a **6D space** of amount, category, day-of-week, and hour. The
   cloud of all of a customer's points is their "spending terrain."
2. **Multi-resolution scan.** A Vietoris–Rips filtration grows the ball
   radius ε and watches structure appear and disappear — fine clusters
   at small ε, macro structure at large ε. (This is the filtration from
   TDA-1; I will not re-derive it.)
3. **Structural summary.** Ripser compresses the whole scan into a
   *persistence diagram* — birth, death, dimension — keeping only the
   long-lived features and discarding noise near the diagonal.
4. **Vectorize.** A fixed set of statistics plus Persistence Entropy
   turns the variable-size diagram into a fixed-width vector. *This* is
   what enters the model.

A fifth step, specific to the offline pipeline, tracks *temporal*
change: split a customer's history into earlier/later halves, compute
each diagram, and measure how much the topology moved with a Wasserstein
distance. That is the `phase_transition` block.

The point-cloud coordinates are not raw values. Amounts go through a log
transform (`ln(amount + 1)`) to tame the power-law tail; cyclic
variables (day-of-week, hour) are sin/cos encoded so that Sunday and
Monday sit one step apart on the unit circle rather than six apart on a
number line; and the MCC category — being nominal, not ordinal — is
spread to a uniform `PERCENT_RANK()` so distance is meaningful.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="200" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a5f">Per-customer offline extraction</text>
  <!-- step 1 -->
  <rect x="20" y="58" width="96" height="70" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="68" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4f46e5">6D cloud</text>
  <g fill="#4f46e5"><circle cx="48" cy="98" r="2.5"/><circle cx="66" cy="92" r="2.5"/><circle cx="84" cy="104" r="2.5"/><circle cx="58" cy="112" r="2.5"/><circle cx="80" cy="116" r="2.5"/><circle cx="70" cy="106" r="2.5"/></g>
  <text x="68" y="122" text-anchor="middle" font-size="8" fill="#64748b">amt·cat·dow·hr</text>
  <!-- step 2 -->
  <rect x="152" y="58" width="96" height="70" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="200" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#0d9488">VR filtration</text>
  <circle cx="200" cy="102" r="16" fill="none" stroke="#0d9488" stroke-width="1" stroke-dasharray="3 2"/>
  <circle cx="200" cy="102" r="26" fill="none" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="2 3"/>
  <text x="200" y="122" text-anchor="middle" font-size="8" fill="#64748b">grow ε</text>
  <!-- step 3 -->
  <rect x="284" y="58" width="96" height="70" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="332" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#d97706">diagram</text>
  <line x1="312" y1="118" x2="356" y2="88" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3 2"/>
  <g fill="#d97706"><circle cx="322" cy="100" r="2.5"/><circle cx="332" cy="94" r="2.5"/><circle cx="340" cy="108" r="2.5"/></g>
  <text x="332" y="122" text-anchor="middle" font-size="8" fill="#64748b">(b, d)</text>
  <!-- step 4 -->
  <rect x="416" y="58" width="96" height="70" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="464" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e3a5f">vectorize</text>
  <g fill="#1e3a5f"><rect x="438" y="92" width="6" height="22" rx="1"/><rect x="448" y="98" width="6" height="16" rx="1"/><rect x="458" y="88" width="6" height="26" rx="1"/><rect x="468" y="100" width="6" height="14" rx="1"/><rect x="478" y="94" width="6" height="20" rx="1"/></g>
  <text x="464" y="122" text-anchor="middle" font-size="8" fill="#64748b">E + stats</text>
  <!-- output -->
  <rect x="540" y="72" width="48" height="42" rx="6" fill="#d97706"/>
  <text x="564" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">70D</text>
  <text x="564" y="103" text-anchor="middle" font-size="8" fill="#fff">block</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="116" y1="93" x2="150" y2="93"/><polygon points="150,93 142,89 142,97"/>
    <line x1="248" y1="93" x2="282" y2="93"/><polygon points="282,93 274,89 274,97"/>
    <line x1="380" y1="93" x2="414" y2="93"/><polygon points="414,93 406,89 406,97"/>
    <line x1="512" y1="93" x2="538" y2="93"/><polygon points="538,93 530,89 530,97"/>
  </g>
  <text x="84" y="168" text-anchor="middle" font-size="9" fill="#94a3b8">log amount · sin/cos cyclic · PERCENT_RANK category</text>
  <text x="430" y="168" text-anchor="middle" font-size="9" fill="#94a3b8">Persistence Entropy + 5 lifetime statistics per H-dim</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The four fixed steps: raw events → 6D point cloud → Vietoris–Rips diagram → fixed vector. Coordinate transforms (bottom-left) make distance meaningful; the vectorizer (bottom-right) is six numbers per homology dimension.</figcaption>
</figure>

## How the Diagram Becomes a Vector

A persistence diagram is a *multiset* — a variable-size, unordered bag
of `(birth, death)` points — so it cannot enter a model directly. TDA-1
solved this with PersLay's *learnable* set function. The offline
pipeline takes the older, *fixed* route. The reference lists three
classical options:

| Method | Principle | Used here? |
| --- | --- | --- |
| Statistics | Mean/std/range of the lifetime distribution. Fastest, most direct. | **Yes — primary** |
| Persistence Entropy | Shannon entropy of the lifetime distribution. Diversity in one scalar. | **Yes** |
| Persistence Landscape | Tent functions → $L^p$ norm (Amplitude). Lives in a Banach space. | Initialized but **not** in output columns |

The system combines **statistics + Persistence Entropy**. Concretely,
for each homology dimension it emits **six features**: Persistence
Entropy (1) plus five lifetime statistics — mean, std, min, max, median.
A Landscape Amplitude computer (`Amplitude(metric="landscape")`) is
instantiated in the code but its output is *not* part of the current
production columns; I note this because the schema names hint at it.

The entropy is the most distinctive of the six. With $L_i = d_i - b_i$
the lifetime of the $i$-th feature:

$$ E = -\sum_{i=1}^{N} p_i \log p_i, \qquad p_i = \frac{d_i - b_i}{\sum_{j=1}^{N}(d_j - b_j)} $$

> **Equation intuition.** Normalize every feature's lifetime by the
> total lifetime, treat the result as a probability distribution, and
> take its Shannon entropy. If one giant structure hogs all the
> lifetime — all spending collapsed into a single cluster — $E$ is low.
> If many features share the lifetime evenly — varied, balanced
> spending regions — $E$ is high, up to $\log N$. It is the *diversity*
> of a customer's topological structure compressed into one number, and
> by Atienza et al. (2019) it is provably stable: small input
> perturbations cannot swing it.

The five lifetime statistics are blunter but complementary: the **mean**
says how much robust, wide-scale structure exists; the **std**
distinguishes a uniform set of features from a mix of stable structure
and transient noise; **min/max/median** sketch the rest of the lifetime
distribution. Six fixed numbers, no training, computed the same way for
every customer.

> **Historical context.** The "fixed vectorization" lineage is its own
> small history. Persistence Entropy was systematized by Rucco et al.
> (2016), applying Shannon's 1948 entropy to lifetime distributions;
> Atienza et al. (2019) later proved its stability. The Persistence
> Landscape — the option the project prepared but left out of the output
> — is Bubenik's 2015 contribution, the move that placed diagram
> summaries in a Banach space where means and hypothesis tests finally
> make sense. The offline pipeline deliberately picks the cheapest,
> most interpretable members of this family and leaves the *learnable*
> vectorization to PersLay.

## The 70D Breakdown

The 70D TDA block is the single largest occupant of the 159D Domain
feature group, and it splits into three sub-blocks:

| Sub-block | Dim | Source | Homology | Scope |
| --- | --- | --- | --- | --- |
| `tda_short` | 24D | 90-day app logs | $H_0, H_1$ | Global + Local |
| `tda_long` | 36D | 12-month card transactions | $H_0, H_1$ | Global + Local |
| `phase_transition` | 10D | earlier/later window diff | $H_0, H_1$ | temporal |

Both `tda_short` and `tda_long` follow the same dimensional arithmetic:
**6 features × 2 Betti ($H_0, H_1$) × 2 scopes (Global, Local) = 24D**.
The Global scope computes topology over a sampled population of all
customers (up to 10,000 sampled events), giving the *background* shape; the
Local scope uses only that one customer's window, giving their *own*
shape. $H_2$ (voids) is excluded everywhere: at the ~200-point
per-customer cloud size, voids do not form stably, and $H_2$ carries an
$O(n^3)$ cost from simplex-count blow-up.

> A documentation caveat worth flagging. The tensor-composition table
> lists `tda_long` as **36D**, and `feature_schema.yaml` enumerates
> names `tda_long_001`–`tda_long_036`, but the dimension formula and the
> actual `extract_long_features()` output are **24D** ($H_0, H_1$ × 6 ×
> 2). The reference notes the schema-name count and the produced-column
> count disagree — a known schema-side issue, not a second set of
> features. I report the 70D split as the schema labels it (24 + 36 +
> 10) while noting the real `tda_long` payload is 24 produced columns.

The `phase_transition` block is structurally different — it is not a
diagram summary but a *diff*. It splits into **PD Distance (4D) +
Transition Detection (6D)**:

- **PD Distance (4D)** — `pt_W1_distance_h0`, `pt_W1_distance_h1` (the
  Wasserstein-1 distance between the earlier and later diagrams in each
  dimension), their sum `pt_total_topological_change`, and
  `pt_max_structural_shift` (the larger of the two bottleneck shifts).
- **Transition Detection (6D)** — a sigmoid-squashed transition
  probability plus imminence, frequency, direction, magnitude, and a
  phase-classification confidence.

The transition probability is a regime detector borrowed straight from
physics:

$$ P_{\text{transition}} = \frac{1}{1 + e^{-2(\Delta_{\text{total}} - \tau)}} $$

with $\tau = 0.5$. When the total topological change $\Delta_{\text{total}}$
crosses the threshold, the sigmoid snaps the probability toward 1. The
functional form is the same as the Fermi–Dirac distribution of
statistical mechanics, and the snap at a critical threshold mimics a
physical phase transition — which is exactly why the feature is named
that way. A companion `_classify_phase` routine bins customers into five
regimes — Stable, Growing, Shrinking, Chaotic, and Transitioning — based
on Betti and entropy trends.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a5f">TDA 70D block composition</text>
  <!-- full bar 70 -->
  <!-- tda_short 24 -->
  <rect x="40" y="50" width="171" height="44" rx="4" fill="#4f46e5"/>
  <text x="125" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">tda_short</text>
  <text x="125" y="86" text-anchor="middle" font-size="10" fill="#dbeafe">24D</text>
  <!-- tda_long 36 -->
  <rect x="215" y="50" width="171" height="44" rx="4" fill="#0d9488"/>
  <text x="300" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">tda_long</text>
  <text x="300" y="86" text-anchor="middle" font-size="10" fill="#ccfbf1">36D (schema)</text>
  <!-- phase 10 -->
  <rect x="390" y="50" width="130" height="44" rx="4" fill="#d97706"/>
  <text x="455" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">phase_trans</text>
  <text x="455" y="86" text-anchor="middle" font-size="10" fill="#fef3c7">10D</text>
  <text x="280" y="112" text-anchor="middle" font-size="10" fill="#64748b">= 70D, the largest slice of the 159D Domain group</text>
  <!-- breakdown of a 24D block -->
  <text x="125" y="142" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4f46e5">24D = 6 × 2 Betti × 2 scope</text>
  <g font-size="9" fill="#64748b">
    <text x="125" y="160" text-anchor="middle">6 feat: entropy · mean · std</text>
    <text x="125" y="174" text-anchor="middle">· min · max · median</text>
    <text x="125" y="190" text-anchor="middle">Betti: H₀, H₁ — Scope: Global, Local</text>
  </g>
  <!-- breakdown of phase 10D -->
  <text x="455" y="142" text-anchor="middle" font-size="10.5" font-weight="700" fill="#d97706">10D = 4 + 6</text>
  <g font-size="9" fill="#64748b">
    <text x="455" y="160" text-anchor="middle">PD dist (4): W₁ H₀/H₁,</text>
    <text x="455" y="174" text-anchor="middle">total, max shift</text>
    <text x="455" y="190" text-anchor="middle">detect (6): prob, imminence…</text>
  </g>
  <line x1="290" y1="130" x2="290" y2="200" stroke="#e2e8f0" stroke-width="1"/>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The 70D block: tda_short (24D) + tda_long (36D as schema-labelled) + phase_transition (10D). Each 24D summary block is 6 fixed features across 2 homology dimensions and 2 scopes; the 10D phase block is a 4D diagram-distance plus a 6D transition detector.</figcaption>
</figure>

## Why Offline — The Cost Argument

Computing persistent homology is the expensive part, and it scales
badly: a Vietoris–Rips complex can have up to $2^n - 1$ simplices, and
the boundary-reduction is $O(n^3)$. Doing this *inside* the model on
every forward pass would be untenable. The offline batch pays the cost
once and freezes the result, which is the whole architectural reason
the 70D block exists separately from the PersLay Expert.

The pipeline keeps the cost bounded with a three-rung engine and
aggressive sampling:

- **Engine priority chain.** `PersistenceExtractor` auto-selects the
  fastest available backend: **Ripser++** (CUDA, fastest) → **Ripser**
  (C++ binding; 10–50× speedup when paired with a CuPy-computed GPU
  distance matrix) → **giotto-tda** (CPU, richest API). A Ripser++
  failure (CUDA mismatch) falls back to CPU Ripser automatically.
- **Point sampling.** `max_points = 1000` per customer, drawn by
  *time-stratified sampling*: split the time-ordered data into
  $k = \min(10, n/10)$ buckets, sample evenly within each, and preserve
  order. This caps the $O(n^2)$ distance matrix and the homology cost
  while keeping temporal coverage.
- **Memory.** The distance matrix is $O(n^2)$ — ~95 MB at $n=5000$,
  ~381 MB at $n=10000$ (float32) — so the CuPy path chunks it
  (chunk size 2000) and 12 GB+ of GPU memory is recommended. An
  optional Sparse Rips mode (`use_sparse`, off by default) trades
  accuracy for speed by ignoring simplices beyond a distance threshold.

For cold-start customers with too little history for stable topology,
the project swaps in a 4-stage progressive strategy (statistics-based
approximation when points are too few), but that is its own topic.

## Where We Stop

We split topology into two lanes — the learnable online PersLay Expert
of TDA-1 and the fixed offline features here — and followed the offline
lane end to end: raw logs into 6D point clouds, Ripser into persistence
diagrams, and a deliberately *fixed* vectorization (Persistence Entropy
plus five lifetime statistics per homology dimension) into a 70D block.
We saw that block decompose into `tda_short` (24D), `tda_long` (24
produced / 36 schema-labelled D), and `phase_transition` (10D), and saw
why doing all of this offline — once, in the Airflow batch — is the only
way to afford an $O(n^3)$ computation inside a 734D input tensor.

What this block does *not* capture is the *sequential* structure of a
customer's behavior — the latent regime they are in and the transitions
between regimes as a probabilistic process rather than a topological
diff. That is the job of the Hidden Markov Model summary in the
Model-Derived group: five numbers from a Viterbi path and posterior over
latent states. That is the subject of the next post, **HMM-1**.
