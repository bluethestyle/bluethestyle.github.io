---
title: "[Study Thread] GMM-1 — Soft Clustering: Responsibilities as Features and the Gaussian Mixture Behind Them"
date: 2026-06-07 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, gmm, mixture-model, clustering, em, features]
lang: en
excerpt: "The GMM feature sub-thread opens — why a single cluster id throws away most of the signal, how a Gaussian mixture assigns each customer a soft membership across 20 behavioral archetypes, what the EM responsibility actually computes, how BIC picks K=20, and the 22D vector that lands in the model's soft-routing basket. With the real input/output composition from the on-prem reference."
series: study-thread
part: 20
alt_lang: /2026/06/07/gmm-soft-clustering-ko/
next_title: "TimeSeries-1 — Temporal Features: From Raw Transaction Sequences to a Fixed Behavioral Footprint"
next_desc: "How the project turns a variable-length stream of dated transactions into stable per-customer temporal features: trend and seasonality decomposition, autocorrelation structure, change-point signals, and why a state-space view beats hand-rolled rolling windows."
next_status: draft
---

*First post of the GMM (Gaussian Mixture Model) feature sub-thread in
the "Study Thread" series. Across this and the following posts, in
parallel Korean and English, I unpack the GMM soft-clustering feature
module — one of the offline feature blocks that feeds the on-prem
recommendation model. The source is the on-prem reference
`기술참조서/GMM_피처_기술_참조서`, and the full PDF will be attached to the
final post of the sub-thread. Where the TDA sub-thread asked what shape
an Expert reads, this one asks something simpler and more operational:
when we cluster customers, what exactly should we hand the downstream
model — a label, or a distribution? GMM's answer is the second, and
the difference turns out to carry real information.*

> **Why soft, in one line.** A hard cluster id is one number: "customer
> A is type 3." A GMM membership is a full distribution: "customer A is
> 0.65 type 3, 0.20 type 7, 0.10 type 12, …" — a point on a 20D
> probability simplex instead of a vertex. In this project that vector
> is **20D of soft membership + cluster_id + entropy = 22D**, fed from a
> **K=20, full-covariance** mixture fit by EM on **40D** of pre-computed
> customer features. The reference puts it bluntly: a hard label carries
> ~1 bit; the soft vector carries up to ~4.32 bits ($\log_2 20$). That
> extra information is the whole point.

## Hard Labels Throw Away the Boundary

When we cluster customers, the reflex is to assign each one to a single
group and move on. K-means does exactly this — every customer gets the
nearest centroid, full stop. It is simple, fast, and it discards the one
thing that matters most about the customers who are hardest to serve:
*their ambiguity*.

A customer sitting halfway between a "food-centric" group and a
"travel-centric" group gets shoved arbitrarily to one side. The fact
that they are a boundary case — equally well described by two archetypes
— is erased the moment the label is written. And boundary customers are
not a rounding error; in financial behavior data, where clusters overlap
heavily, they are a large fraction of the book.

The reference enumerates K-means' three structural limits — hard
assignment itself, isotropic distance, and the lack of a probabilistic
reading — and contrasts the two methods across seven aspects:

| Aspect | K-means (hard) | GMM (soft) |
| --- | --- | --- |
| Assignment | one-hot (1 of 20 is 1) | probability vector (20D, sums to 1.0) |
| Information per feature | 1 bit (which cluster) | ~4.32 bits (K=20 entropy ceiling) |
| Boundary customers | arbitrary, unstable | probability spreads across neighbors |
| Cluster shape | spherical (isotropic) | ellipsoidal (covariance-shaped) |
| Confidence signal | none | quantified by entropy |
| Cold start | nearest centroid (biased) | uniform = unbiased neutral |
| Gradient compatibility | discontinuous (one-hot) | continuous (differentiable) |

The deepest of these is the first. A soft assignment is *coordinate on a
simplex*, not a choice of corner — and a continuous coordinate preserves
information that a one-hot quantization permanently destroys.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="250" fill="#f8fafc" rx="8"/>
  <!-- Hard k-means: spherical, hard boundary -->
  <text x="140" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">K-means — hard, spherical</text>
  <circle cx="100" cy="120" r="45" fill="#64748b18" stroke="#64748b" stroke-width="1.2"/>
  <circle cx="185" cy="135" r="45" fill="#64748b18" stroke="#64748b" stroke-width="1.2"/>
  <line x1="143" y1="70" x2="143" y2="190" stroke="#e11d48" stroke-width="1.4" stroke-dasharray="5 4"/>
  <g fill="#64748b"><circle cx="85" cy="110" r="3.5"/><circle cx="105" cy="100" r="3.5"/><circle cx="95" cy="135" r="3.5"/><circle cx="200" cy="125" r="3.5"/><circle cx="180" cy="150" r="3.5"/><circle cx="205" cy="145" r="3.5"/></g>
  <!-- boundary point forced left -->
  <circle cx="140" cy="120" r="5" fill="#d97706"/>
  <text x="140" y="212" text-anchor="middle" font-size="10" fill="#e11d48">boundary → forced to one side</text>
  <!-- divider -->
  <line x1="280" y1="40" x2="280" y2="210" stroke="#e2e8f0" stroke-width="1"/>
  <!-- GMM: ellipses, soft -->
  <text x="420" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">GMM — soft, ellipsoidal</text>
  <ellipse cx="380" cy="115" rx="52" ry="30" transform="rotate(-18 380 115)" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <ellipse cx="470" cy="140" rx="46" ry="28" transform="rotate(24 470 140)" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <g fill="#0d9488"><circle cx="360" cy="105" r="3.5"/><circle cx="385" cy="98" r="3.5"/><circle cx="372" cy="128" r="3.5"/></g>
  <g fill="#4f46e5"><circle cx="478" cy="132" r="3.5"/><circle cx="462" cy="152" r="3.5"/><circle cx="490" cy="145" r="3.5"/></g>
  <!-- boundary point with split membership -->
  <circle cx="425" cy="125" r="5" fill="#d97706"/>
  <text x="425" y="205" text-anchor="middle" font-size="10" fill="#d97706">γ = (0.55, 0.45) — both</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">K-means forces every point to a single spherical cluster across a hard line; GMM gives ellipsoidal clusters and lets a boundary customer hold partial membership in both.</figcaption>
</figure>

## The Gaussian Mixture

The model behind the soft assignment is the Gaussian mixture. It assumes
each customer's feature vector was generated by first *picking* one of
$K$ behavioral types — type $k$ with prior probability $\pi_k$ — and then
*drawing* from that type's multivariate Gaussian $\mathcal{N}(\mu_k,
\Sigma_k)$. The density of any observed customer $x$ is the weighted sum
over all types:

$$ p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \pi_k \ge 0,\ \sum_k \pi_k = 1 $$

Each component is a full multivariate normal:

$$ \mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2}\,|\Sigma_k|^{1/2}} \exp\!\left(-\tfrac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)\right) $$

with $D = 40$ here. The quadratic form in the exponent, $(x-\mu_k)^\top
\Sigma_k^{-1}(x-\mu_k)$, is the squared *Mahalanobis distance* — a
distance that, unlike Euclidean, corrects for feature correlation and
scale through $\Sigma_k^{-1}$. Two features that move together (say
transaction amount and frequency) are not double-counted; a customer
high on both is judged *closer* to the centroid than independence would
suggest. This is precisely why the project uses `covariance_type="full"`
rather than spherical: it lets each cluster be a tilted ellipsoid, not a
ball.

> **Historical context.** The mixture idea predates the algorithm to fit
> it by nearly a century — Karl Pearson decomposed a two-component
> normal mixture by the method of moments in 1894. The practical engine
> arrived in 1977, when Dempster, Laird, and Rubin formalized
> *Expectation–Maximization* as a general recipe for maximum-likelihood
> estimation with latent variables. The "which component generated this
> point" variable is exactly such a latent, and EM is what makes fitting
> a 20-component, 40-dimensional, full-covariance mixture on a million
> customers a few-minutes job rather than an intractable one.

## EM — Fitting the Mixture

We never observe which type generated a given customer; that assignment
is latent. EM handles this with two alternating steps that provably never
decrease the log-likelihood $\ln p(X \mid \Theta)$, where $\Theta =
\{\pi_k, \mu_k, \Sigma_k\}$.

**E-step.** Given the current parameters, compute each customer's
*responsibility* — the posterior probability that customer $x_n$ came
from type $k$:

$$ \gamma_{nk} = \frac{\pi_k\,\mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(x_n \mid \mu_j, \Sigma_j)} $$

This is Bayes' rule, nothing more: numerator is *prior of type $k$ ×
likelihood of $x_n$ under type $k$*, denominator normalizes over all
types so the responsibilities sum to 1. **This $\gamma_{nk}$ is the
feature** — it becomes the output columns `cluster_prob_00` through
`cluster_prob_19` verbatim.

**M-step.** Treat the soft responsibilities as fractional memberships
and re-estimate the parameters. With effective count $N_k = \sum_n
\gamma_{nk}$ (a real number, "≈1,251 customers' worth"):

$$ \pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k}\sum_n \gamma_{nk}\,x_n, \quad \Sigma_k = \frac{1}{N_k}\sum_n \gamma_{nk}\,(x_n-\mu_k)(x_n-\mu_k)^\top $$

Each update is a responsibility-weighted average; high-confidence
customers pull their type's centroid and shape more strongly. The two
steps alternate until the log-likelihood change drops below $10^{-3}$ or
`max_iter=200` is reached. A small diagonal regularizer
`reg_covar=1e-1` is added to each $\Sigma_k$ to guarantee positive
definiteness (so the Cholesky factor exists and $\Sigma_k^{-1}$ is
stable).

<figure style="margin:24px auto;max-width:540px;">
<svg viewBox="0 0 540 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="540" height="240" fill="#f8fafc" rx="8"/>
  <text x="270" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">The EM loop</text>
  <!-- E box -->
  <rect x="60" y="70" width="170" height="90" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2"/>
  <text x="145" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">E-step</text>
  <text x="145" y="120" text-anchor="middle" font-size="10" fill="#64748b">fix Θ, compute γₙₖ</text>
  <text x="145" y="138" text-anchor="middle" font-size="10" fill="#64748b">(responsibilities)</text>
  <!-- M box -->
  <rect x="310" y="70" width="170" height="90" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>
  <text x="395" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">M-step</text>
  <text x="395" y="120" text-anchor="middle" font-size="10" fill="#64748b">fix γₙₖ, update Θ</text>
  <text x="395" y="138" text-anchor="middle" font-size="10" fill="#64748b">(πₖ, μₖ, Σₖ)</text>
  <!-- arrows -->
  <path d="M 230 100 L 305 100" fill="none" stroke="#94a3b8" stroke-width="1.6"/>
  <polygon points="305,100 296,95 296,105" fill="#94a3b8"/>
  <path d="M 310 135 L 235 135" fill="none" stroke="#94a3b8" stroke-width="1.6"/>
  <polygon points="235,135 244,130 244,140" fill="#94a3b8"/>
  <!-- convergence note -->
  <text x="270" y="195" text-anchor="middle" font-size="10.5" fill="#1e3a5f">repeat until Δ ln L &lt; 10⁻³  or  max_iter = 200</text>
  <text x="270" y="214" text-anchor="middle" font-size="9.5" fill="#94a3b8">log-likelihood never decreases · reg_covar = 1e-1 keeps Σₖ positive definite</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">EM alternates: the E-step recomputes soft memberships from the current Gaussians, the M-step re-fits the Gaussians as membership-weighted statistics. Each pass climbs the likelihood.</figcaption>
</figure>

## The Responsibility, Read as a Feature

It is worth pausing on what $\gamma_{nk}$ actually gives the downstream
model. For one customer it is a 20-vector on the probability simplex — a
*convex profile* over archetypes. The reference's example is the cleanest
framing: telling the model

> "customer A is type 3"

is strictly less informative than telling it

> "customer A is 0.65 type 3, 0.20 type 7, 0.10 type 12."

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">One customer's responsibility vector γ (20D, sums to 1.0)</text>
  <!-- baseline -->
  <line x1="40" y1="180" x2="520" y2="180" stroke="#64748b" stroke-width="1"/>
  <!-- 20 bars; heights encode an illustrative soft membership -->
  <g>
    <!-- bar template: x, height(px), fill, label -->
    <rect x="46"  y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="70"  y="166" width="18" height="14"  fill="#cbd5e1"/>
    <rect x="94"  y="80"  width="18" height="100" fill="#0d9488"/>
    <rect x="118" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="142" y="170" width="18" height="10"  fill="#cbd5e1"/>
    <rect x="166" y="160" width="18" height="20"  fill="#cbd5e1"/>
    <rect x="190" y="148" width="18" height="32"  fill="#4f46e5"/>
    <rect x="214" y="175" width="18" height="5"   fill="#cbd5e1"/>
    <rect x="238" y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="262" y="168" width="18" height="12"  fill="#cbd5e1"/>
    <rect x="286" y="160" width="18" height="20"  fill="#d97706"/>
    <rect x="310" y="176" width="18" height="4"   fill="#cbd5e1"/>
    <rect x="334" y="173" width="18" height="7"   fill="#cbd5e1"/>
    <rect x="358" y="171" width="18" height="9"   fill="#cbd5e1"/>
    <rect x="382" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="406" y="170" width="18" height="10"  fill="#cbd5e1"/>
    <rect x="430" y="175" width="18" height="5"   fill="#cbd5e1"/>
    <rect x="454" y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="478" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="502" y="176" width="18" height="4"   fill="#cbd5e1"/>
  </g>
  <!-- annotations on the dominant bars -->
  <text x="103" y="72"  text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">0.65</text>
  <text x="103" y="196" text-anchor="middle" font-size="9" fill="#64748b">type 3</text>
  <text x="199" y="140" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">0.20</text>
  <text x="199" y="196" text-anchor="middle" font-size="9" fill="#64748b">type 7</text>
  <text x="295" y="152" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">0.10</text>
  <text x="295" y="196" text-anchor="middle" font-size="9" fill="#64748b">type 12</text>
  <text x="430" y="196" text-anchor="middle" font-size="9" fill="#94a3b8">17 other types ≈ 0.05 total</text>
  <text x="36" y="84" text-anchor="end" font-size="9" fill="#94a3b8">γ</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A single customer's soft membership: most mass on type 3, real secondary mass on 7 and 12, a long thin tail everywhere else. A hard label would keep only the tallest bar and discard the rest.</figcaption>
</figure>

The second is what the soft routing downstream consumes directly. And
because $\gamma_{nk}$ is a smooth, softmax-like function of the input — no
`argmin`, no discontinuity — gradients can in principle flow through it,
keeping the door open to end-to-end fine-tuning later even though today
the GMM is pre-trained and used as a fixed feature extractor.

Alongside the 20 probabilities, the module emits a scalar that summarizes
*how sure* the assignment is — the assignment entropy:

$$ H_n = -\sum_{k=1}^{K} \gamma_{nk}\,\ln(\gamma_{nk} + \epsilon), \qquad \epsilon = 10^{-10} $$

$H_n \in [0, \ln K]$. At $H_n = 0$ the customer is fully committed to one
type; at $H_n = \ln 20 \approx 2.996$ the distribution is uniform — the
customer is unclassifiable, which is exactly the cold-start signal. This
single number lets the recommender act conservatively when it is unsure,
matching the operational rule "when in doubt, choose the safe option."

> **Equation intuition.** Read $\gamma_{nk}$ as Bayes updating a belief.
> Before seeing the customer, your belief that they are type $k$ is the
> population share $\pi_k$. After seeing their 40D feature vector, you
> multiply by how plausible that vector is under type $k$'s Gaussian and
> renormalize. The entropy $H_n$ then measures how peaked the resulting
> belief is: a spike on one type is low entropy (confident), a flat
> spread is high entropy (a borderline or brand-new customer). Nothing
> here is heuristic — it is the posterior, and its sharpness, read off
> directly.

## Choosing K — BIC over AIC

A mixture can always fit better by adding components, so $K$ must be
chosen against a complexity penalty. The system uses the **Bayesian
Information Criterion**:

$$ \mathrm{BIC} = -2\ln\hat{L} + k\,\ln(n) $$

where $\hat{L}$ is the maximized likelihood, $k$ the number of free
parameters, and $n$ the number of customers; lower is better. BIC is
preferred over AIC ($-2\ln\hat{L} + 2k$) deliberately: with $n$ in the
hundreds of thousands to millions, AIC's flat $2k$ penalty would wave
through far too many clusters, while BIC's $k\ln(n)$ term scales with the
data and clamps down on overfitting — for any $n > 7$ it penalizes more
strongly than AIC. With `full` covariance the parameter count grows like
$O(K \cdot D^2)$; at $K=20$, $D=40$ the covariances alone are about
$20 \times 40 \times 41 / 2 = 16{,}400$ parameters, so the penalty is
not academic.

A one-time sweep (`analyze_optimal_k.py`) over $K \in [5, 30]$ landed on
$K = 20$ as the crossing point of minimum BIC and maximum silhouette:

| K | BIC | Silhouette | Note |
| --- | --- | --- | --- |
| 5 | high | 0.35 | under-split — heterogeneous customers mixed |
| 10 | mid | 0.38 | basic granularity |
| 15 | mid–low | 0.41 | good separation |
| **20** | **lowest** | **0.42** | **BIC/silhouette optimum — adopted** |
| 25 | low | 0.41 | marginal BIC gain, overfitting begins |
| 30 | low | 0.39 | empty clusters, covariance degeneracy risk |

A silhouette of 0.42 is "broadly good separation," reasonable for
financial customer data where cluster boundaries are genuinely fuzzy. K
is then frozen, with a monthly BIC monitor (`validate_k_range()` over
$[5, 30]$) that logs a warning when the optimal K drifts from the current
one by more than `K_CHANGE_THRESHOLD=3` — an actual change stays a
manual, supervised step, because moving K reshapes the output dimension.

## The Feature Vector — 22D, and Where It Lands

The module's output is fixed at `DEFAULT_K + 2 = 22D`, composed as:

| Feature | Dim | Meaning |
| --- | --- | --- |
| `cluster_prob_00` … `cluster_prob_19` | 20D | per-cluster membership $\gamma_{nk}$, sums to 1.0 |
| `cluster_id` | 1D | $\arg\max_k \gamma_{nk}$; cold start → 20 (dedicated unassigned id) |
| `cluster_entropy` | 1D | assignment uncertainty $H_n$ |

The input side is a 40D pre-computed continuous vector, Z-score
normalized (train-time $\mu_d, \sigma_d$ saved to `gmm_norm_params.npz`
and reused at inference for train–serve consistency), assembled by a
DuckDB join over thirteen sources: **Base 13D** (RFM, transaction
statistics, time patterns, category diversity), **Multi-source 10D**
(deposit, credit, investment, digital engagement), **Domain 10D**
(optional — TDA persistence entropy, phase transition, income
decomposition, and financial behavior), **Demographics 2D**, and
**Supplementary 5D**. Four of
the Domain features — `permanent_income_avg`,
`transitory_income_volatility`, `income_elasticity`, `spending_risk` —
come from the **Economics** module, so the batch ordering is
*Economics → GMM → 734D integration*, enforced in the DAG via an
`ExternalTaskSensor`.

> **The contract has since moved.** The 734D above is the V1 feature
> contract. On 2026-07-02 the project switched to the V2 strict contract
> and the operational input width is **4035D** — 734D was not discarded;
> it remains V2's _shared base of eight groups_, with the
> lag/rolling/product families (3301D) appended on top.

Downstream, the 20D probability block is passed as the `cluster_probs`
field of `PLEClusterInput` and drives the **soft routing** in
`GroupTaskExpertBasket`, which keeps 20 cluster sub-heads per task and
mixes them by the membership vector:

$$ o_\text{task} = \sum_{k=0}^{19} \gamma_{nk}\cdot h_{\text{subhead}_k}(z_\text{shared}) $$

This is a *convex combination* over the simplex: the output is an
interpolation of the 20 sub-head outputs, always inside their convex
hull, so soft routing buys stability for free. A hard `argmax` would use
a single vertex and discard the rest; a boundary customer instead blends
neighboring sub-heads, and a cold-start customer (uniform membership)
naturally receives the ensemble of all 20. The reference notes the
structural kinship to Mixture-of-Experts (Mixtral and friends) — with the
difference that this system runs *dense* routing over all 20 heads rather
than sparse top-2, affordable at $K=20$ and lossless for it.

## Where We Stop

We started from a discomfort with hard labels — that committing a
boundary customer to one cluster destroys the most useful thing about
them — and saw how a Gaussian mixture replaces the label with a soft
membership over 20 archetypes. We wrote down the mixture density and its
full-covariance Mahalanobis geometry, watched EM alternate between
responsibilities and weighted re-fits, read the E-step responsibility as
the actual 20D feature plus an entropy that flags the unsure, let BIC
pick $K=20$, and traced the resulting 22D vector into the soft-routing
basket where it ensembles 20 sub-heads per task.

What we have *not* done is leave the cross-sectional snapshot. GMM reads
one moment — a single 40D feature vector per customer — and asks which
mixture of types it resembles. It says nothing about *order*: whether
spending is trending, seasonal, accelerating, or about to break. That is
a different axis entirely, and the next module reads it directly from the
dated transaction stream. The subject of the next post is **TimeSeries
features**.
