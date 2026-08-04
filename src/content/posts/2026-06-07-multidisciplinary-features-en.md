---
title: "[Study Thread] MULTI-1 — Borrowed Instruments: Four Sciences, One Spending Stream, 24 Dimensions"
date: 2026-06-07 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, multidisciplinary, entropy, complexity, features, offline]
lang: en
excerpt: "Statistics is one lens on behavior. This post opens the Multidisciplinary feature module — 24 dimensions built by borrowing measuring instruments from four other sciences: chemical kinetics reads spending acceleration, SIR epidemiology reads category contagion, criminology's routine-activity theory reads burstiness and circadian rhythm, and wave physics reads the frequency spectrum of consumption. Same card-transaction stream, four orthogonal projections, and why structural isomorphism — not loose analogy — makes it legitimate."
series: study-thread
part: 22
alt_lang: /2026/06/07/multidisciplinary-features-ko/
next_title: "DISTILL-1 — The Teacher and the Student: Why We Train a Heavy PLE Only to Throw It Away"
next_desc: "From a high-capacity PLE-adaTT Teacher to a lightweight LGBM Student: what knowledge distillation actually transfers, why a closed-network batch system needs the small model in production, and how soft targets carry more than the hard labels ever could."
next_status: draft
---

*First post of the Multidisciplinary feature sub-thread in the "Study
Thread" series. In parallel Korean and English, I unpack one of the
small-but-strange feature groups in this project: a 24-dimensional block
built not by stacking more statistics, but by borrowing the measuring
instruments of four other sciences and pointing them at a single stream
of card transactions. The source is the on-prem reference
`기술참조서/Multidisciplinary_피처_기술_참조서`, and the full PDF will be
attached to the final post of the sub-thread. Where the TDA sub-thread
asked what shape behavior has, this one asks a sister question: when
statistics runs out of lenses, whose instruments do we borrow next?*

> **The thesis in one line.** Every scientific discipline has spent
> centuries sharpening a mathematical tool to capture one *kind* of
> pattern — chemistry the rate and barrier of a transformation,
> epidemiology the spread of a state, criminology the regularity and
> rupture of a routine, wave physics the interference of overlapping
> rhythms. Those tools were built for molecules, pathogens, crimes, and
> waves — but the *equations* don't know that. Point them at a customer's
> 90-day transaction window and they extract behavioral structure that
> means, variances, and frequencies simply cannot see. Twenty-four
> dimensions, four borrowed instruments, ~3.3% of the 734D tensor.

## Why One Lens Is Not Enough

Traditional feature engineering looks at data through a single lens:
statistics. Means, variances, frequencies, correlations — powerful, but
they reveal only one slice of the structure a behavior carries.

The analogy in the reference is exact: analyzing behavior from one
viewpoint is like photographing a statue head-on. The front shot tells
you nothing about depth, the texture of the back, the curvature of the
side. A multidisciplinary approach photographs the same object from
several angles at once and reconstructs its three-dimensional structure.

Each discipline brings an instrument that was ground over centuries to
detect a *specific* kind of pattern. The crucial point is that the four
projections are nearly *orthogonal* — they look at different axes of the
same card-transaction data — so combining them expands the feature space
efficiently rather than redundantly.

| Discipline | Borrowed instrument | What it sees that statistics misses |
| --- | --- | --- |
| Physical chemistry (reaction kinetics) | rate, barrier, catalysis, saturation | *acceleration* (2nd derivative), not just trend; the energy barrier to enter a new category; how a catalyst (payday) moves behavior |
| Epidemiology (SIR model) | compartmental flow S→I→R | population-level *dynamics*, not individual acts; a contagion threshold $R_0$ for category adoption |
| Criminology (routine-activity theory) | regularity vs. deviation | the *circular* nature of time (23:00 and 01:00 are close); burstiness vs. regularity; breakpoints where a routine ruptures |
| Wave physics (spectral analysis) | interference, synchronization | FFT *frequency-domain* decomposition — spectral entropy, dominant period, phase locking, cross-coherence |

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="250" fill="#f8fafc" rx="8"/>
  <text x="290" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">One transaction stream → four orthogonal projections</text>
  <!-- source -->
  <rect x="20" y="100" width="86" height="50" rx="6" fill="#1e3a5f"/>
  <text x="63" y="121" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">90-day</text>
  <text x="63" y="136" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">card txns</text>
  <!-- four modules -->
  <rect x="180" y="48" width="150" height="34" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1"/>
  <text x="255" y="69" text-anchor="middle" font-size="10.5" fill="#0d9488" font-weight="700">Chemical kinetics · 6D</text>
  <rect x="180" y="92" width="150" height="34" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1"/>
  <text x="255" y="113" text-anchor="middle" font-size="10.5" fill="#d97706" font-weight="700">SIR epidemic · 5D</text>
  <rect x="180" y="136" width="150" height="34" rx="6" fill="#e11d4818" stroke="#e11d48" stroke-width="1"/>
  <text x="255" y="157" text-anchor="middle" font-size="10.5" fill="#e11d48" font-weight="700">Crime pattern · 5D</text>
  <rect x="180" y="180" width="150" height="34" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1"/>
  <text x="255" y="201" text-anchor="middle" font-size="10.5" fill="#4f46e5" font-weight="700">Wave physics · 8D</text>
  <!-- combine -->
  <rect x="400" y="100" width="92" height="50" rx="6" fill="#4f46e5"/>
  <text x="446" y="121" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">24D</text>
  <text x="446" y="137" text-anchor="middle" font-size="9" fill="#fff">multidisc.</text>
  <!-- arrows source->modules -->
  <g stroke="#cbd5e1" stroke-width="1.4" fill="none">
    <path d="M 106 120 C 140 65, 150 65, 180 65"/>
    <path d="M 106 122 C 140 109, 150 109, 180 109"/>
    <path d="M 106 128 C 140 153, 150 153, 180 153"/>
    <path d="M 106 130 C 140 197, 150 197, 180 197"/>
  </g>
  <!-- arrows modules->combine -->
  <g stroke="#cbd5e1" stroke-width="1.4" fill="none">
    <path d="M 330 65 C 365 100, 375 110, 400 118"/>
    <path d="M 330 109 C 365 118, 375 120, 400 122"/>
    <path d="M 330 153 C 365 140, 375 132, 400 128"/>
    <path d="M 330 197 C 365 160, 375 145, 400 132"/>
  </g>
  <text x="540" y="128" text-anchor="middle" font-size="9" fill="#64748b" transform="rotate(90 540 128)">→ 734D tensor</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The same 90-day transaction window passes through four discipline-specific extractors; each reads a different axis, and the 24D union joins the 734D main tensor.</figcaption>
</figure>

## Analogy With Teeth — Structural Isomorphism

The obvious objection: a consumer is not a molecule, and category
adoption is not an epidemic. So why is borrowing a chemistry equation
anything but a cute metaphor?

The reference is blunt about this. The "analogy" here is not loose — it
rests on *structural isomorphism*. Two systems can have completely
different surface objects (molecules vs. consumers) while the *relational
structure* between those objects is mathematically identical. The time
for a reactant's concentration to halve and the time for a customer's
transaction frequency to halve are both governed by the same exponential
decay. The fraction of a population still susceptible and the fraction of
categories a customer has not yet adopted are both the *S* compartment of
the same compartmental model.

> **Equation intuition.** If the equation is the same, the pattern it
> captures is the same. What the symbols originally *referred* to —
> atoms, pathogens — has no bearing on whether the equation is valid for
> a new domain. This is exactly the move behind physics-informed ML and
> transfer learning: take a mathematical structure proven in one domain
> and apply it where the same relational structure recurs. The honest
> caveat (the reference states it too): the math transfers, but the
> *causal mechanism* of the origin discipline does not. These are
> pattern-capture tools, not causal explanations.

The single thread running through chemistry and epidemiology is the
exponential. Any process whose rate of change is proportional to its
current state evolves exponentially:

$$ \frac{dy}{dt} = \alpha y \quad\Longrightarrow\quad y(t) = y_0\, e^{\alpha t} $$

That one differential equation underwrites the Arrhenius rate
$k = A\,e^{-E_a/RT}$ in chemistry and the early infection growth
$I(t) \approx I_0\, e^{(\beta S_0 - \gamma)t}$ in epidemiology — the same
skeleton, two disciplines.

## Instrument 1 — Chemical Kinetics (6D)

Chemistry studies *how fast* a transformation proceeds and *what barrier*
it must clear. Mapped onto spending, a "reaction" is a category
transition, the activation energy $E_a$ is the friction of entering a new
category, and a catalyst is an external event (payday, a promotion) that
speeds things up without being consumed.

The signature feature is **spending acceleration** — the second
derivative of the trend, which a first-derivative trend line cannot see.
The project computes it as a discrete second finite difference over three
30-day windows:

$$ f''(t) \approx f(t+\Delta t) - 2f(t) + f(t-\Delta t) $$

In code that is `spending_acceleration = avg_w3 - 2*avg_w2 + avg_w1`,
where the three terms are the mean spend of the first, middle, and last
30 days. Positive means spending is *accelerating* (convex); negative
means it is decelerating (concave) — and decelerating spend is a known
*leading* indicator of churn, often weeks ahead of any drop in frequency
or amount.

> **Historical context.** The exponential rate law comes from Svante
> Arrhenius (1889), who fit the temperature dependence of sugar
> hydrolysis; it was later justified from Boltzmann's distribution
> $P(E)\propto e^{-E/k_BT}$. Half-life as a characteristic time was
> systematized by Ernest Rutherford studying radioactive decay in the
> 1900s. The same $T_{1/2} = \ln 2 / k$ that dates archaeological carbon
> now stands in for the median inter-transaction gap of a customer.

The full 6D: `new_category_activation_rate` (a proxy for inverse $E_a$),
`spending_half_life` ($T_{1/2}$ = median transaction gap),
`spending_acceleration`, `dormancy_reactivation_rate` (catalytic revival
of a dormant category), `catalyst_sensitivity` (payday elasticity =
early-month vs. late-month daily spend), and `saturation_proximity` (how
close max spend sits to mean + 1σ — the spending ceiling).

## Instrument 2 — SIR Epidemic Diffusion (5D)

Epidemiology partitions a population into compartments — Susceptible,
Infected, Recovered — and watches individuals flow between them. The
Kermack–McKendrick (1927) model is a system of differential equations:

$$ \frac{dS}{dt} = -\beta S I, \quad \frac{dI}{dt} = \beta S I - \gamma I, \quad \frac{dR}{dt} = \gamma I $$

with the basic reproduction number $R_0 = \beta/\gamma$ — the
dimensionless contagion threshold. $R_0 > 1$ and an epidemic spreads;
$R_0 < 1$ and it dies out. Translated to consumption, $R_0 > 1$ means
category adoption is self-reinforcing.

The mapping is structural, not poetic. A customer's **susceptible**
fraction is the share of the population's Top-15 MCCs they have *not* yet
used (`susceptible_count / 15`); the **infected** fraction is the share
of categories whose recent per-day frequency is *growing*; the
**recovered** fraction is the share they used before but have abandoned
in the last 30 days — adoption, then immunity/indifference. The growth
test corrects for unequal window lengths by comparing daily rates:

$$ \text{infected} = \mathbb{1}\!\left[\, \text{recent\_count} > \text{older\_count}\cdot \tfrac{30}{L-30} \,\right] $$

A customer whose `max_weekly_new_mcc` spikes — many first-time categories
in a single week — is the consumption equivalent of a *super-spreader*.
The 5D: `sir_susceptible`, `sir_infected`, `sir_recovered`,
`max_weekly_new_mcc` (contagion peak), `category_lifecycle_mean` (mean
days a category survives, first→last transaction).

## Instrument 3 — Crime Pattern / Routine-Activity Theory (5D)

Cohen & Felson's routine-activity theory (1979) holds that behavior is
governed by daily routine, and the *rupture* of routine creates the
opportunity for an abnormal event. Ported to spending, the question
becomes: how regular is a customer's routine, and where does it break?

The signature measure is **burstiness**, from Barabási's work on human
dynamics (2005). Human action is not a Poisson process — it follows short
bursts and long lulls, a heavy-tailed distribution. Normalized to
$[-1,1]$:

$$ B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} $$

where $\sigma_\tau, \mu_\tau$ are the standard deviation and mean of the
inter-transaction gaps. $B=-1$ is perfectly regular (a subscription-like
cadence), $B=0$ is the Poisson baseline (random), $B=+1$ is extreme
clustering (binge-then-silence).

> **Historical context.** Routine-activity theory was published by
> Lawrence Cohen and Marcus Felson in *American Sociological Review*
> (1979), shifting crime-prevention policy from "correcting offenders" to
> "situational prevention." Barabási formalized burstiness in *Nature*
> (2005), showing that human inter-event times follow a power law because
> people process tasks from a priority queue, not at random —
> high-priority tasks burst, low-priority ones wait in a heavy tail.

The other regularity instrument is **circular variance**, because time is
a circle, not a line: 23:00 and 01:00 are two hours apart, not
twenty-two. Mapping hour $h$ to an angle $\theta = 2\pi h/24$ and taking
the mean resultant length $\bar{R}$:

$$ \mathrm{CV} = 1 - \bar{R}, \quad \bar{R} = \sqrt{\Big(\tfrac1n\textstyle\sum \sin\theta_i\Big)^2 + \Big(\tfrac1n\textstyle\sum \cos\theta_i\Big)^2} $$

$\mathrm{CV}\to 0$ means a customer always transacts around the same hour
(say, lunchtime); $\mathrm{CV}\to 1$ means the hours are spread evenly —
a structure plain linear variance cannot recover. The 5D: `burstiness`,
`recurrence_period` (the autocorrelation-peak lag among 7/14/21/28/30
days), `routine_breakpoint_count` (weekly-total mean-crossings),
`circular_variance`, and `max_amount_zscore` — an outlier intensity
folded through a sigmoid $\sigma(z-3)$ so a single extreme transaction
saturates toward 1 instead of dominating the feature.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="230" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Burstiness — three spending rhythms on a 90-day axis</text>
  <!-- regular -->
  <text x="20" y="62" font-size="10.5" fill="#0d9488" font-weight="700">B = −1  regular</text>
  <line x1="150" y1="58" x2="500" y2="58" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#0d9488"><circle cx="180" cy="58" r="3.5"/><circle cx="225" cy="58" r="3.5"/><circle cx="270" cy="58" r="3.5"/><circle cx="315" cy="58" r="3.5"/><circle cx="360" cy="58" r="3.5"/><circle cx="405" cy="58" r="3.5"/><circle cx="450" cy="58" r="3.5"/></g>
  <!-- poisson -->
  <text x="20" y="122" font-size="10.5" fill="#64748b" font-weight="700">B = 0  Poisson</text>
  <line x1="150" y1="118" x2="500" y2="118" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#64748b"><circle cx="172" cy="118" r="3.5"/><circle cx="205" cy="118" r="3.5"/><circle cx="240" cy="118" r="3.5"/><circle cx="305" cy="118" r="3.5"/><circle cx="350" cy="118" r="3.5"/><circle cx="398" cy="118" r="3.5"/><circle cx="470" cy="118" r="3.5"/></g>
  <!-- bursty -->
  <text x="20" y="182" font-size="10.5" fill="#e11d48" font-weight="700">B = +1  bursty</text>
  <line x1="150" y1="178" x2="500" y2="178" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#e11d48"><circle cx="168" cy="178" r="3.5"/><circle cx="176" cy="178" r="3.5"/><circle cx="184" cy="178" r="3.5"/><circle cx="192" cy="178" r="3.5"/><circle cx="360" cy="178" r="3.5"/><circle cx="368" cy="178" r="3.5"/><circle cx="376" cy="178" r="3.5"/></g>
  <text x="260" y="214" text-anchor="middle" font-size="10" fill="#94a3b8">Same transaction count, three different gap distributions — only B tells them apart.</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Burstiness B = (σ−μ)/(σ+μ) collapses the coefficient of variation of inter-transaction gaps into [−1, 1]; the Poisson process (σ=μ) sits exactly at 0.</figcaption>
</figure>

## Instrument 4 — Wave Physics / Spectral Analysis (8D)

When several waves overlap, their phase relationship decides whether they
*reinforce* (constructive) or *cancel* (destructive). The wave-physics
module takes the daily-spend time series into the frequency domain with
the FFT and reads its structure. The signature is **spectral entropy** —
Shannon entropy applied to the normalized power spectrum:

$$ p(k) = \frac{|X(k)|^2}{\sum_{k'} |X(k')|^2}, \quad H = -\sum_{k=1}^{K} p(k)\,\log_2 p(k) $$

A customer who spends the same amount on the same weekday concentrates
energy at $f = 1/7$ and has *low* entropy (regular, predictable); a
customer whose timing and amounts scatter spreads energy across
frequencies and has *high* entropy. This is exactly Shannon's
uncertainty, ported from communication channels to spending rhythm.

> **Historical context.** Shannon entropy $H = -\sum p_i \log p_i$
> appeared in his 1948 "A Mathematical Theory of Communication," and is
> mathematically identical to Boltzmann's thermodynamic entropy
> $S = -k_B\sum p_i\ln p_i$ — von Neumann reportedly told Shannon to call
> it entropy because "no one knows what entropy really is, so you'll
> always win the argument." Phase-locking value descends from
> functional-connectivity analysis in neuroscience, where it measures
> whether two signals keep a consistent phase difference.

The other instruments here measure *relationships between categories* in
the frequency domain: cross-spectral coherence (do two categories beat at
the same frequency?), phase-locking value, and a
constructive-interference ratio for whether category pairs sit in phase.
The 8D:
`spectral_entropy`, `weekly_harmonic_power` (energy at the 1/7 and 2/7
bins), `cross_spectral_coherence`, `dominant_period` ($T = 1/f_{\text{peak}}$),
`spectral_centroid_shift` (first-half vs. second-half mean frequency),
`phase_locking_value`, `mean_phase_difference`, and
`constructive_interference_ratio`.

<figure style="margin:24px auto;max-width:480px;">
<svg viewBox="0 0 480 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="480" height="250" fill="#f8fafc" rx="8"/>
  <text x="240" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Spectral entropy — concentrated vs. scattered energy</text>
  <!-- axes left -->
  <line x1="50" y1="210" x2="220" y2="210" stroke="#64748b" stroke-width="1"/>
  <line x1="50" y1="60" x2="50" y2="210" stroke="#64748b" stroke-width="1"/>
  <text x="135" y="234" text-anchor="middle" font-size="9.5" fill="#0d9488" font-weight="700">low H — regular</text>
  <!-- concentrated: one tall bar at 1/7 -->
  <g fill="#0d9488">
    <rect x="62" y="205" width="14" height="5"/><rect x="84" y="200" width="14" height="10"/>
    <rect x="106" y="78" width="14" height="132"/><rect x="128" y="198" width="14" height="12"/>
    <rect x="150" y="203" width="14" height="7"/><rect x="172" y="206" width="14" height="4"/><rect x="194" y="204" width="14" height="6"/>
  </g>
  <text x="113" y="72" text-anchor="middle" font-size="8.5" fill="#0d9488">f = 1/7</text>
  <!-- axes right -->
  <line x1="270" y1="210" x2="440" y2="210" stroke="#64748b" stroke-width="1"/>
  <line x1="270" y1="60" x2="270" y2="210" stroke="#64748b" stroke-width="1"/>
  <text x="355" y="234" text-anchor="middle" font-size="9.5" fill="#e11d48" font-weight="700">high H — scattered</text>
  <!-- scattered: many similar bars -->
  <g fill="#e11d48">
    <rect x="282" y="150" width="14" height="60"/><rect x="304" y="135" width="14" height="75"/>
    <rect x="326" y="160" width="14" height="50"/><rect x="348" y="140" width="14" height="70"/>
    <rect x="370" y="152" width="14" height="58"/><rect x="392" y="145" width="14" height="65"/><rect x="414" y="158" width="14" height="52"/>
  </g>
  <text x="135" y="50" text-anchor="middle" font-size="9" fill="#64748b">power spectrum</text>
  <text x="355" y="50" text-anchor="middle" font-size="9" fill="#64748b">power spectrum</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Energy concentrated in one frequency bin (left) gives low spectral entropy — a regular 7-day rhythm; energy spread across bins (right) gives high entropy — an irregular pattern.</figcaption>
</figure>

## Where the 24D Lands

All four extractors share one input — the same 90-day card-transaction
window, filtered to `customer_id_encrypted IS NOT NULL AND
transaction_amount > 0` — and run as DuckDB in-memory SQL (with a numpy
FFT hybrid for the spectral block), each writing a ZSTD-Parquet file.
`feature_integrator.py` then LEFT-JOINs the four files on
`customer_id_encrypted` into the 24D block, which sits inside the 734D
main tensor (644D normalized + 90D raw power-law). The schema registry
`feature_schema.yaml` holds the 24 keys, `chemical_kinetics_001` through
`interference_008`.

> **The contract has since moved.** The 734D above is the V1 feature
> contract this post is grounded in. On 2026-07-02 the project switched
> to the V2 strict contract, and the operational input width is now
> **4035D**. 734D was not discarded — it remains V2's _shared base of
> eight groups_, with the lag/rolling/product families (3301D) appended
> on top to reach 4035D. The 24D multidisciplinary block keeps its slot
> inside that base.

Downstream, the 24D reaches exactly two Shared Experts — **DeepFM** and
**Causal**. Experts do not receive the whole tensor; they are routed by
feature group (`DEFAULT_EXPERT_ROUTING_V2` in `ple_cluster_adatt.py`),
and these two are the only ones whose routing includes the
`multidisciplinary` group. DeepFM takes all thirteen groups — the full
4035D. Causal takes `base` + `multi_source` + `domain` +
`multidisciplinary` + `model_derived` = 539D. The **OT** Expert takes
only `extended_source` + `multi_source` = 175D, so the
multidisciplinary block never reaches it.

DeepFM learns cross patterns between the
multidisciplinary features and the rest (e.g. `spending_acceleration` ×
churn probability) as field interactions; the Causal Expert tries to
recover causal direction among them as a DAG. The reference's expected
per-task contribution lines up the instruments with tasks: chemical
kinetics (acceleration/saturation) toward **churn** and **LTV**; epidemic
diffusion toward **NBA** and **cross-sell**; crime pattern (burstiness/
periodicity) toward **timing** and consumption-cycle tasks; interference
(spectral) toward **spending category** and merchant affinity.

A warning the reference is careful to repeat: 24D is only ~3.3% of 734D
(0.6% against V2's 4035D),
and analogy has limits. The features are pattern-capture instruments, not
causal explanations — a consumer is not a molecule. And many of them lean
on data quality (circular variance needs transaction-time, SIR ratios
need MCC mapping); when data is thin, a COALESCE default is returned, and
that default means "no pattern," not "normal."

## Where We Stop

We started from the limit of a single lens, made the case that borrowing
another science's instrument is structural isomorphism rather than loose
metaphor, and walked the four instruments in turn — chemical kinetics
reading spending acceleration, SIR reading category contagion,
routine-activity theory reading burstiness and circadian rhythm, and wave
physics reading the frequency spectrum. Twenty-four dimensions, four
orthogonal projections, one transaction stream, joined into the 734D
tensor.

What remains, across this whole project, is the model that *eats* that
tensor — and the moment where we train a heavy, high-capacity PLE-adaTT
Teacher and then, almost perversely, throw it away in favor of a small
LGBM Student. Why a closed-network batch system distills a big model down
to a little one, what knowledge actually transfers, and why soft targets
carry more than the hard labels ever could — that is the subject of the
next post, **DISTILL-1**.
