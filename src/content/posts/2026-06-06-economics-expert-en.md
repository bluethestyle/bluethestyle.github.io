---
title: "[Study Thread] ECON-1 — The Price of a Habit: Economics-Derived Features and the Shared Expert That Reads Them"
date: 2026-06-06 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, economics, elasticity, behavioral-economics, features, expert]
lang: en
excerpt: "Why the economic structure behind a spend is a signal a black-box model under-uses — and how the project turns a century of microeconomics into 17 dimensions of feature: income elasticity, CRRA utility and consumption smoothing, a refund-ratio price-sensitivity proxy, HHI and Shannon diversification, and the permanent/transitory income split. Then where those 17D plug into the PLE-adaTT Domain Expert."
series: study-thread
part: 17
alt_lang: /2026/06/06/economics-expert-ko/
next_title: "TDA-3 — Offline at Scale: Extracting Persistence Features in the Airflow Batch"
next_desc: "How the project computes persistence diagrams for millions of customers without melting the cluster: the PersistenceExtractor batch, Ripser/Ripser++ on session point clouds, the Parquet diagram store, and the cost trade that pushes O(2^n) topology off the training hot path."
next_status: draft
---

*A post in the "Study Thread" series, this time on the economics-derived
feature block — the 17D slice the PLE-adaTT Domain Expert reads. In
parallel Korean and English, I unpack how a card-recommendation model
reads the* economic structure *behind a customer's spending, not just
its surface statistics. The source is the on-prem reference
`기술참조서/Economics_피처_기술_참조서`, and the full PDF will be attached to
the final post of this sub-thread. Where the TDA sub-thread asked what
the* shape *of behavior means, this one asks a flatter but older
question: what does a century of microeconomics — Marshall's elasticity,
Friedman's permanent income, CRRA utility — let us read off a card
ledger that a mean and a variance throw away?*

> **The core claim, stated plainly.** Two customers can spend an
> identical ₩3M a month and be economically opposite people. One earns
> ₩4M and spends steadily; the other earns ₩2M plus a quarterly bonus
> and spends in bursts. A descriptive statistic — mean, variance,
> skew — sees the same ₩3M. An *economics* feature sees a
> high-permanent-income saver versus a bonus-driven cashback candidate. This post is
> about the 17 dimensions the project computes to make that difference
> visible to the model, and the Expert that consumes them.

## Why Economic Structure Is a Signal the Model Under-Uses

When we describe a customer we reach for descriptive statistics — mean,
standard deviation, skew, kurtosis. These summarize the *shape* of a
number stream but say nothing about *why* that shape appeared. A
black-box model fed only raw aggregates has to *rediscover* the causal
structure of consumption from scratch, with no prior. It usually
settles for correlation.

Economics features hand the model that prior for free. The reference
gives three reasons they beat pure statistics:

- **They encode causal structure.** Economic theory states *how* income
  movement drives spending movement — a direction, not just a
  co-occurrence. Baking that into a feature lets the model learn
  behavioral *direction* rather than raw correlation.
- **They are interpretable.** A feature value of `income_elasticity =
  1.3` reads directly as "a 1% income rise lifts this customer's spend
  by 1.3% — luxury-leaning." That is an XAI-ready statement, not a
  latent coordinate.
- **They are domain-normalized.** Dimensionless economic ratios
  (elasticity, CV, HHI) are invariant to income scale, price level, and
  currency. A ₩30M-a-year and a ₩300M-a-year customer's elasticities are
  *directly comparable* — half the feature-scaling problem disappears
  before normalization even runs.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <text x="280" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Same mean spend, opposite economic structure</text>
  <!-- Customer A: steady -->
  <text x="140" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">Customer A — steady (₩4M income)</text>
  <line x1="40" y1="150" x2="240" y2="150" stroke="#cbd5e1" stroke-width="1"/>
  <polyline points="48,120 80,122 112,118 144,121 176,119 208,120 232,120" fill="none" stroke="#0d9488" stroke-width="2"/>
  <text x="140" y="178" text-anchor="middle" font-size="10" fill="#64748b">high permanent income · low volatility</text>
  <text x="140" y="194" text-anchor="middle" font-size="10" fill="#64748b">→ steady-discount card</text>
  <!-- divider -->
  <line x1="280" y1="44" x2="280" y2="205" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Customer B: bursty -->
  <text x="420" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">Customer B — bursty (₩2M + bonus)</text>
  <line x1="320" y1="150" x2="520" y2="150" stroke="#cbd5e1" stroke-width="1"/>
  <polyline points="328,140 360,142 392,90 424,141 456,139 488,84 512,140" fill="none" stroke="#e11d48" stroke-width="2"/>
  <text x="420" y="178" text-anchor="middle" font-size="10" fill="#64748b">low permanent income · high bonus_frequency</text>
  <text x="420" y="194" text-anchor="middle" font-size="10" fill="#64748b">→ cashback card</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Both customers average ₩3M/month. A mean cannot separate them; the permanent/transitory income split and bonus_frequency can — and they point at different cards.</figcaption>
</figure>

## The Demand Function Behind the Ledger

The spine of the whole feature set is the microeconomic demand
function. The reference writes consumer demand as

$$ Q_d = f(P,\ Y,\ P_s,\ P_c,\ T,\ E) $$

and then maps each abstract variable onto something the project can
actually observe on a card-and-deposit ledger:

| Theory variable | Observed data | Feature |
| --- | --- | --- |
| $Y$ — income | monthly deposit inflow | `permanent_income_avg`, `transitory_income_avg` |
| $Q_d$ — demand | monthly card spend total | basis for elasticity |
| $P$ — price | refund ratio (price-dissatisfaction proxy) | `price_sensitivity` |
| $T$ — taste | MCC-code spend distribution | `spending_diversification`, `category_hhi` |
| $E$ — expectation | first-half vs second-half spend ratio | `discount_rate_proxy` |

The move is the same every time: take a quantity microeconomics defines
on *infinitesimal* changes, and approximate it with a *discrete,
monthly* statistic that a SQL query can compute over a 36-month window.

## Elasticity — The Dimensionless Workhorse

Elasticity is "the percent change in one variable per percent change in
another." Income elasticity of demand is the canonical case:

$$ \varepsilon_Y = \frac{\partial Q}{\partial Y}\cdot\frac{Y}{Q} $$

Its sign and magnitude give a universal classification: $\varepsilon_Y >
1$ is a *luxury* (spend accelerates as income rises), $0 < \varepsilon_Y
< 1$ a *necessity*, and $\varepsilon_Y < 0$ an *inferior* good (spend
*falls* as income rises). Because the units cancel, the number is the
same in won or dollars, in levels or logs — which is exactly why it
travels across customers of wildly different scale.

> **Historical context.** Elasticity is Alfred Marshall's, formalized in
> *Principles of Economics* (1890), where he read it off the demand
> curve as a geometric property. The demand function itself goes back
> further — Antoine-Augustin Cournot (1838) was the first to write
> demand as a mathematical function of price. The behavioral-proxy
> trick — measuring price sensitivity from observed behavior when the
> elasticity itself is unmeasurable — descends from George Stigler's
> (1961) economics of *search cost*: a price-sensitive shopper searches
> more, and a refund is a kind of *after-the-fact* price search.

The continuous partial derivative is uncomputable on a ledger, so the
project approximates it with the average of monthly arc-elasticity
changes:

$$ \hat{\varepsilon}_Y = \frac{1}{T}\sum_{t=1}^{T}\frac{\Delta S_t / S_{t-1}}{\Delta Y_t / Y_{t-1}} $$

In code this collapses to one aggregate, with a `NULLIF` guard so a
zero-spend month never divides by zero:

```python
income_elasticity = AVG(
    (monthly_spending - prev_monthly_spending)
    / NULLIF(prev_monthly_spending, 0)
)
```

> **Equation intuition.** $S_t$ is month-$t$ spend, $Y_t$ month-$t$
> income. Each term asks "by what % did spending move relative to how
> much income moved this month?" and the feature averages that ratio
> over the window. It is the practical, discrete face of the
> textbook derivative — an *arc* elasticity standing in for a point one.

The other two elasticity-family features are proxies of the same spirit.
**Price sensitivity** uses the refund ratio: $\text{price\_sensitivity}
= 1 - \overline{\text{refund\_ratio}}$ — high means *insensitive* (few
refunds), low means a price-hunting customer. **Cross-category
elasticity** is the coefficient of variation of the monthly category
count, $\sigma(\text{category\_count})/\mu(\text{category\_count})$ — a
customer whose set of spending categories swings month to month is more
likely to respond to a *new* category's card benefit.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="300" fill="#f8fafc" rx="8"/>
  <text x="260" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Income elasticity = slope of the Engel curve</text>
  <!-- axes -->
  <line x1="70" y1="250" x2="470" y2="250" stroke="#64748b" stroke-width="1.2"/>
  <line x1="70" y1="250" x2="70" y2="56" stroke="#64748b" stroke-width="1.2"/>
  <text x="270" y="282" text-anchor="middle" font-size="12" fill="#1e3a5f">income Y</text>
  <text x="30" y="155" text-anchor="middle" font-size="12" fill="#1e3a5f" transform="rotate(-90 30 155)">spend Q</text>
  <!-- 45-degree reference -->
  <line x1="70" y1="250" x2="430" y2="70" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="430" y="64" font-size="9.5" fill="#94a3b8" text-anchor="end">ε = 1 (proportional)</text>
  <!-- luxury (convex, ε>1) -->
  <path d="M 70 250 Q 280 240 430 80" fill="none" stroke="#4f46e5" stroke-width="2.2"/>
  <text x="438" y="86" font-size="11" font-weight="700" fill="#4f46e5">ε &gt; 1 luxury</text>
  <!-- necessity (concave, ε<1) -->
  <path d="M 70 250 Q 200 120 430 110" fill="none" stroke="#0d9488" stroke-width="2.2"/>
  <text x="438" y="116" font-size="11" font-weight="700" fill="#0d9488">0 &lt; ε &lt; 1 necessity</text>
  <!-- inferior (downward, ε<0) -->
  <path d="M 70 200 Q 250 215 430 240" fill="none" stroke="#e11d48" stroke-width="2.2"/>
  <text x="438" y="244" font-size="11" font-weight="700" fill="#e11d48">ε &lt; 0 inferior</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Income elasticity is the local slope of the Engel curve relative to the proportional line: convex above (luxury), concave below (necessity), downward (inferior). The sign alone routes a customer toward premium vs discount cards.</figcaption>
</figure>

## Utility, Risk, and the Smoothing of Spend

Behind elasticity sits the consumer's *utility* function $u(C)$, with
the two defining properties of rational consumption: more is better and
each extra unit is worth less,

$$ u'(C) > 0, \qquad u''(C) < 0 $$

The project anchors on the CRRA (constant relative risk aversion) form,

$$ u(C) = \frac{C^{1-\gamma}}{1-\gamma}, \qquad \gamma > 0,\ \gamma \neq 1 $$

where $\gamma$ is the relative risk-aversion coefficient. Its
consequence is the load-bearing one: a higher-$\gamma$ customer *hates*
consumption swings and works to flatten them, which the project reads as
a higher `consumption_smoothing`. Crucially, $\gamma$ is *constant*
across consumption scale (Arrow–Pratt $-C\,u''/u' = \gamma$), which is
the theoretical license to compare smoothing across rich and poor
customers on the same axis.

> **Equation intuition.** The concavity $u'' < 0$ is the whole story.
> By Jensen's inequality it means $u(\mathbb{E}[C]) > \mathbb{E}[u(C)]$:
> facing two consumption paths with the same average, a rational
> consumer prefers the *certain* one. That preference for smoothness is
> exactly what `consumption_smoothing` measures — and it is why a
> volatile spender and a steady one, equal on the mean, separate here.

`consumption_smoothing` is computed as the *inverse* coefficient of
variation, $\mu/\sigma$ over monthly spend — which the reference notes
is structurally the signal-to-noise ratio, and in fact the
*Sharpe ratio* of a customer's consumption. High smoothing means a
predictable spender whose card-spend-threshold achievement is reliable.

## Portfolio Concentration — Borrowing from Industrial Organization

Two more features describe *how spread out* a customer's spending is,
borrowing tools from information theory and antitrust. **Spending
diversification** is Shannon entropy over the MCC-category spend shares,

$$ H = -\sum_{i=1}^{N} s_i \ln s_i $$

zero when all spend sits in one category, maximal ($\ln N$) when it is
evenly spread. **Category HHI** is the Herfindahl–Hirschman Index — the
same number the U.S. DOJ uses for merger review — applied to spend
shares instead of market shares:

$$ \text{HHI} = \sum_{i=1}^{N} s_i^2 $$

The two are complementary by design: HHI's square term is sensitive to
the *dominant* category, while entropy's log term reaches into the
*tail*. Together they let the model read both "how concentrated is the
main category" and "how many categories matter at all" — and the
recommendation rule is direct: HHI < 0.15 → a broad multi-benefit card;
HHI > 0.25 → a category-specialized card (fuel, telecom).

## The 17 Dimensions, Assembled

The Economics block is two extractors stacked into 17D. The table is the
whole feature set, grounded in the reference's summaries:

| Feature | Economic meaning | How computed |
| --- | --- | --- |
| `income_elasticity` | luxury / necessity / inferior tilt | mean of monthly arc-elasticity $\Delta S/S_{\text{prev}}$ |
| `price_sensitivity` | sensitivity to price (refund proxy) | $1 - \overline{\text{refund\_ratio}}$ |
| `cross_category_elasticity` | volatility of category breadth | CV of monthly category count |
| `spending_diversification` | spread of spend across categories | Shannon entropy $-\sum s_i \ln s_i$ |
| `category_hhi` | concentration in top categories | $\sum s_i^2$ |
| `spending_risk` | unpredictability of monthly total | CV of monthly spend |
| `discount_rate_proxy` | time preference (immediate vs deferred) | first-half / second-half spend ratio |
| `savings_propensity` | tendency to save | negative-to-positive net-spend ratio |
| `consumption_smoothing` | dislike of consumption swings | inverse CV, $\mu/\sigma$ |
| `permanent_income_avg` | long-run stable income level | $\text{mean}(\hat{Y}^P)$ (Friedman PIH) |
| `permanent_income_stability` | stability of the income level | CV, $\sigma(\hat{Y}^P)/\mu(\hat{Y}^P)$ |
| `permanent_income_growth` | income growth over the window | $(\hat{Y}^P_T - \hat{Y}^P_1)/\hat{Y}^P_1$ |
| `permanent_income_trend` | long-run trend direction | linear-regression slope |
| `transitory_income_avg` | mean transitory income (≈0 if no recurring bonus) | $\text{mean}(\hat{Y}^T)$ |
| `transitory_income_volatility` | income uncertainty | $\sigma(\hat{Y}^T)$ |
| `transitory_income_max` | biggest bonus-scale event | $\max(\hat{Y}^T)$ |
| `bonus_frequency` | how often big bonuses land | share of months with $\hat{Y}^T > 0.5\,\hat{Y}^P$ |

The first nine are the `financial_behavior` group (9D); the income split
is the `income_decomposition` group (8D), which decomposes raw deposit
inflow into a *permanent* and *transitory* component à la Friedman's
permanent income hypothesis — estimated in implementation, over a
36-month observation window, by one of three selectable methods: a
12-month moving average (the default), an HP filter ($\lambda =
14{,}400$ for monthly data), or a Kalman filter. Together: 8 + 9
= 17D.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="210" fill="#f8fafc" rx="8"/>
  <!-- raw -->
  <rect x="18" y="78" width="96" height="54" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1"/>
  <text x="66" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">card +</text>
  <text x="66" y="116" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">deposit ledger</text>
  <!-- two extractors -->
  <rect x="160" y="34" width="150" height="56" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="235" y="56" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">IncomeDecomposition</text>
  <text x="235" y="72" text-anchor="middle" font-size="10" fill="#64748b">PIH · HP · Kalman → 8D</text>
  <rect x="160" y="120" width="150" height="56" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="235" y="142" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">FinancialBehavior</text>
  <text x="235" y="158" text-anchor="middle" font-size="10" fill="#64748b">elasticity · utility · HHI → 9D</text>
  <!-- economics 17D -->
  <rect x="360" y="78" width="96" height="54" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="408" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">Economics</text>
  <text x="408" y="118" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">17D</text>
  <!-- domain / main -->
  <rect x="500" y="50" width="84" height="46" rx="6" fill="#f8fafc" stroke="#1e3a5f" stroke-width="1"/>
  <text x="542" y="70" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">Domain</text>
  <text x="542" y="85" text-anchor="middle" font-size="10" fill="#64748b">159D</text>
  <rect x="500" y="114" width="84" height="46" rx="6" fill="#1e3a5f"/>
  <text x="542" y="134" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Main Tensor</text>
  <text x="542" y="149" text-anchor="middle" font-size="10" fill="#cbd5e1">734D</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="114" y1="92" x2="158" y2="62"/><polygon points="158,62 149,62 154,70"/>
    <line x1="114" y1="118" x2="158" y2="148"/><polygon points="158,148 149,140 154,148"/>
    <line x1="310" y1="62" x2="358" y2="98"/><polygon points="358,98 349,92 350,100"/>
    <line x1="310" y1="148" x2="358" y2="112"/><polygon points="358,112 350,110 349,118"/>
    <line x1="456" y1="100" x2="498" y2="78"/><polygon points="498,78 489,78 494,86"/>
    <line x1="542" y1="96" x2="542" y2="112"/><polygon points="542,112 538,104 546,104"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Two extractors (8D + 9D) compose the 17D Economics block, which joins TDA/GMM/Mamba inside the 159D Domain group of the 734D main tensor.</figcaption>
</figure>

## Where the Economics Block Plugs Into PLE

The 17 dimensions do not float free. In the 734D main tensor (644D
normalized + 90D raw power-law), Economics is one slice of the 159D
**Domain** feature group, sitting alongside TDA (70D), GMM (22D), and
Mamba (50D). From there it feeds the PLE-adaTT model's **Domain Expert**,
which consumes the 17D and uses it to sharpen check-card recommendation
across the 18 tasks — letting the model tell apart "customers with the
same average spend but different structure," exactly the case a raw mean
collapses.

> **The contract has since moved.** The 734D above is the V1 feature
> contract. On 2026-07-02 the project switched to the V2 strict contract
> and the operational input width is **4035D** — 734D was not discarded;
> it remains V2's _shared base of eight groups_, with the
> lag/rolling/product families (3301D) appended on top.

The features earn their place downstream in two concrete ways. First, a
`DebitCardIncomeConstraints` layer turns the income split into hard and
soft rules — a premium-tier card is excluded when `permanent_income_avg`
< ₩3M; a high `bonus_frequency` nudges a cashback card up the ranking.
Second, the pipeline has an ordering dependency: four Economics features
(`permanent_income_avg`, `transitory_income_volatility`,
`income_elasticity`, `spending_risk`) are part of the *40D input to GMM
clustering*, so Economics must be computed *before* GMM runs — the
income structure literally shapes which cluster a customer lands in.

## Where We Stop

We started from a discomfort: a mean and a variance see two
economically opposite customers as the same person. We walked the
microeconomic demand function onto an observable ledger, read income
elasticity off an Engel-curve slope, grounded `consumption_smoothing` in
CRRA utility and Jensen's inequality, borrowed HHI and Shannon entropy
from industrial organization, and assembled the 17 dimensions —
`financial_behavior` (9D) over `income_decomposition` (8D) — that the
PLE-adaTT Domain Expert reads.

What we have *not* done is the heavy lifting of a different Expert
entirely: the offline machinery. The TDA sub-thread left an Expert whose
features are too expensive to compute in the training loop, and the
project answers that with an Airflow batch — a `PersistenceExtractor`
running Ripser/Ripser++ over millions of customers and writing diagrams
to Parquet, once, off the hot path. How that offline extraction is built
and why the cost trade is unavoidable is the subject of the next post,
**TDA-3**.
