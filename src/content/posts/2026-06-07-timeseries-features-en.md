---
title: "[Study Thread] TSFEAT-1 — The Order in the Numbers: Classical Time-Series Features over a Spending History"
date: 2026-06-07 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, timeseries, seasonality, spectral, features, offline]
lang: en
excerpt: "Why a sum, a mean and a max throw away the one thing a spending history actually has — order. This post walks the offline time-series feature module: the stochastic decomposition of a consumption series into level / volatility / noise, autocorrelation and stationarity, the FFT spectral features, volatility shape, and the entropy complexity measures — the hand-crafted 18D that sits beside a learned 50D Mamba embedding to make 68D. With the real formulas and where each number lands in the 734D tensor."
series: study-thread
part: 21
alt_lang: /2026/06/07/timeseries-features-ko/
next_title: "MULTI-1 — Borrowed Instruments: Four Sciences, One Spending Stream, 24 Dimensions"
next_desc: "The cross-domain feature group — 24 dimensions borrowed from four other sciences: chemical kinetics for spending acceleration, SIR epidemiology for category contagion, criminology's routine-activity theory for burstiness and circadian rhythm, wave physics for the frequency spectrum — and why a recommender for spending reaches into sciences that never heard of a credit card."
next_status: draft
---

*Part of the "Study Thread" series, opening a short sub-thread on the
offline* time-series feature *module. Across this and the next post, in
parallel Korean and English, I unpack how a raw spending history becomes
a fixed-width feature block. The source is the on-prem reference*
`기술참조서/TimeSeries_피처_기술_참조서`, *and the full PDF will be attached
to the final post of the sub-thread. One disambiguation up front, because
the project has two things that both say "time series": this post is about
the* classical, hand-crafted *features — autocorrelation, FFT spectra,
CUSUM changepoints, entropy — pre-computed offline. It is* not *the
Temporal Expert, the deep Mamba→LNN model that trains end-to-end inside
the PLE. Same vocabulary, different instance, no shared weights. We will
keep tripping over that line, so we mark it clearly.*

> **What this module is, in one sentence.** A spending history is the
> only feature source in the whole system that carries *order* —
> shuffle the days of every other feature and nothing changes, but a
> shuffled transaction stream is a different customer. This module's job
> is to *quantify the order without destroying it*: an 18D block of
> hand-crafted statistics (distribution shape, frequency, changepoints,
> autocorrelation, complexity) sitting beside a 50D learned Mamba
> embedding, **68D** total, about 9.3% of the 734D main tensor and the
> single densest information group in it.

## Why Not Just Sum, Mean, Max?

Take two customers, A and B. Over 90 days each spends exactly ₩3,000,000.
Daily mean ₩33k, standard deviation ₩21k, max ₩120k — *identical on
every aggregate*.

- **Customer A** spends ₩40k every Monday and ₩60k every Friday, near
  nothing otherwise. A strong weekly rhythm.
- **Customer B** spends almost nothing for 80 days, then dumps the whole
  ₩3,000,000 into the final 10. One extreme regime change.

The sum, the mean, the standard deviation cannot tell them apart, because
those statistics are *permutation-invariant* — shuffle the days and they
do not move. The difference lives entirely in the **ordering**, and only
order-aware features see it: a high lag-7 autocorrelation for A (and ≈0
for B), a large CUSUM `max_shift_magnitude` for B, low spectral entropy
for A (energy concentrated at the weekly frequency) versus high for B.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Same sum, same mean, same max — different order</text>
  <!-- Customer A: weekly spikes -->
  <text x="140" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Customer A — weekly rhythm</text>
  <line x1="40" y1="120" x2="250" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <g fill="#0d9488">
    <rect x="52" y="78" width="6" height="42"/><rect x="82" y="62" width="6" height="58"/>
    <rect x="112" y="78" width="6" height="42"/><rect x="142" y="62" width="6" height="58"/>
    <rect x="172" y="78" width="6" height="42"/><rect x="202" y="62" width="6" height="58"/>
    <rect x="232" y="78" width="6" height="42"/>
  </g>
  <text x="140" y="138" text-anchor="middle" font-size="9.5" fill="#64748b">ρ(7) high · spectral entropy low</text>
  <!-- divider -->
  <line x1="280" y1="44" x2="280" y2="150" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Customer B: one burst -->
  <text x="420" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Customer B — late burst</text>
  <line x1="310" y1="120" x2="520" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <g fill="#e11d48">
    <rect x="318" y="114" width="5" height="6"/><rect x="338" y="116" width="5" height="4"/><rect x="358" y="115" width="5" height="5"/>
    <rect x="378" y="113" width="5" height="7"/><rect x="398" y="116" width="5" height="4"/>
    <rect x="458" y="66" width="6" height="54"/><rect x="472" y="58" width="6" height="62"/><rect x="486" y="70" width="6" height="50"/><rect x="500" y="60" width="6" height="60"/>
  </g>
  <text x="420" y="138" text-anchor="middle" font-size="9.5" fill="#64748b">CUSUM shift large · ρ(7) ≈ 0</text>
  <!-- shared aggregate box -->
  <rect x="120" y="170" width="320" height="46" rx="6" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>
  <text x="280" y="190" text-anchor="middle" font-size="10.5" fill="#1e3a5f" font-weight="700">aggregate view: Σ = ₩3.0M · mean ₩33k · σ ₩21k · max ₩120k</text>
  <text x="280" y="207" text-anchor="middle" font-size="10" fill="#64748b">— indistinguishable —</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Aggregates flatten the time axis. The order-aware features (autocorrelation, CUSUM, spectral entropy) are the only ones that separate A from B.</figcaption>
</figure>

The reference frames the whole module around four complementary lenses,
each answering a different question about the series:

| Lens | Question it asks | Technique | Feature block |
| --- | --- | --- | --- |
| Time domain | How does the value change over time? | autocorrelation, changepoints, moving average | AR 4D, Changepoint 3D |
| Frequency domain | At what period does it repeat? | FFT, spectral analysis | Freq 4D |
| Distribution / shape | What shape is the value distribution? | skewness, kurtosis, tails | Dist 4D |
| Information theory | How complex / predictable is it? | entropy, permutation entropy | Complex 3D |

Those four lenses are the hand-crafted **18D**. Above them sits a learned
**50D** Mamba state-space embedding that integrates the same information
non-linearly — the human-designed lenses and the model-learned lens, kept
side by side on purpose.

> **Historical context.** Almost every tool in this module predates
> machine learning by decades. CUSUM is E. S. Page's 1954 quality-control
> scheme for catching a shifted manufacturing mean. The Ljung–Box test is
> Ljung and Box's 1978 refinement of the 1970 Box–Pierce portmanteau.
> Approximate Entropy is Pincus's 1991 heart-rate-variability measure,
> and Sample Entropy its 2000 bias-corrected successor (Richman &
> Moorman). The FFT itself is Cooley–Tukey, 1965. We are not inventing
> features here — we are borrowing fifty years of signal processing and
> information theory and pointing them at a credit-card stream.

## The Decomposition — Level, Volatility, Noise

There is a decomposition at the heart of the module, but it is *not* the
STL or Hodrick–Prescott trend/seasonal split you may expect. (That HP
filter, with $\lambda = 14{,}400$ for monthly data, lives in the
*Economics* Expert's income-decomposition group — a different post, a
different module. Worth disambiguating, since both touch spending series.)
Here the series is read as a **stochastic process** and split into three
time-varying parts:

$$ X_t = \mu_t + \sigma_t\,\epsilon_t $$

- $\mu_t$ — the time-varying level (trend plus seasonality): *where* the
  spending sits. Captured by changepoint magnitude and the Mamba
  embedding.
- $\sigma_t$ — the time-varying volatility: *how uncertain* it is.
  Captured by kurtosis and entropy.
- $\epsilon_t$ — white noise, $E[\epsilon_t]=0$, $\mathrm{Var}(\epsilon_t)=1$:
  whose distributional shape (skew, fat tails) the distribution features read.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 330" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="330" fill="#f8fafc" rx="8"/>
  <text x="300" y="24" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Xₜ = μₜ + σₜ·εₜ — one series, three layers</text>
  <!-- observed Xt -->
  <text x="20" y="64" font-size="11" font-weight="700" fill="#1e3a5f">Xₜ (observed)</text>
  <polyline points="150,72 180,55 210,80 240,48 270,70 300,40 330,66 360,44 390,78 420,50 450,72 480,46 510,68 540,52"
    fill="none" stroke="#1e3a5f" stroke-width="1.6"/>
  <!-- mu_t level -->
  <text x="20" y="142" font-size="11" font-weight="700" fill="#4f46e5">μₜ (level / trend)</text>
  <path d="M 150 158 Q 300 120, 540 132" fill="none" stroke="#4f46e5" stroke-width="2"/>
  <text x="544" y="135" font-size="9" fill="#4f46e5">→ changepoint, Mamba</text>
  <!-- sigma_t volatility -->
  <text x="20" y="222" font-size="11" font-weight="700" fill="#d97706">σₜ (volatility)</text>
  <path d="M 150 232 Q 230 230, 300 215 Q 380 198, 460 226 Q 510 240, 540 220" fill="none" stroke="#d97706" stroke-width="2"/>
  <text x="544" y="223" font-size="9" fill="#d97706">→ kurtosis, entropy</text>
  <!-- epsilon noise -->
  <text x="20" y="300" font-size="11" font-weight="700" fill="#64748b">εₜ (noise shape)</text>
  <line x1="150" y1="296" x2="540" y2="296" stroke="#cbd5e1" stroke-width="1"/>
  <g stroke="#64748b" stroke-width="1.2">
    <line x1="165" y1="296" x2="165" y2="284"/><line x1="195" y1="296" x2="195" y2="305"/><line x1="225" y1="296" x2="225" y2="281"/>
    <line x1="255" y1="296" x2="255" y2="302"/><line x1="285" y1="296" x2="285" y2="288"/><line x1="315" y1="296" x2="315" y2="278"/>
    <line x1="345" y1="296" x2="345" y2="306"/><line x1="375" y1="296" x2="375" y2="287"/><line x1="405" y1="296" x2="405" y2="300"/>
    <line x1="435" y1="296" x2="435" y2="282"/><line x1="465" y1="296" x2="465" y2="304"/><line x1="495" y1="296" x2="495" y2="290"/><line x1="525" y1="296" x2="525" y2="283"/>
  </g>
  <text x="544" y="299" font-size="9" fill="#64748b">→ skew, kurtosis</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The stochastic-process view: the observed series is a moving level, modulated by a moving volatility, riding on shaped noise. Each layer maps to a different feature family.</figcaption>
</figure>

The practical point the reference is careful to make: **non-stationarity
is not a nuisance to be filtered away — it is the signal.** A
consumption series is almost never stationary (a raise lifts the mean, a
life change jolts the variance), and a shifting mean, a fattening tail,
or a collapsing autocorrelation are exactly the behavioral changes worth
recommending against. The goal is to *quantify the non-stationarity,
not erase it*.

## Autocorrelation and Stationarity

Permutation-invariant aggregates die here; autocorrelation is the first feature that
listens to the order. The autocorrelation function measures how much a
series *remembers* its own past at lag $h$ — the time-series cousin of
the Pearson correlation, with $X_t$ and $X_{t+h}$ standing in for the two
variables. The sample estimate the module computes is:

$$ \rho_k = \frac{\sum_{t} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_t (x_t - \bar{x})^2} $$

Two lags are extracted as features. **$\rho_1$** (`ar_lag1_autocorr`) is
spending momentum — "did yesterday's spend predict today's?"; values of
0.3–0.5 are ordinary inertia, above 0.7 a strong streak (a trip, a
binge). **$\rho_7$** (`ar_lag7_autocorr`) is weekly seasonality — high
when every Saturday looks like the last.

Raw ACF mixes in *indirect* correlation: $X_t$ and $X_{t+2}$ may correlate
only through $X_{t+1}$. The **partial** autocorrelation strips the
intervening lags out, and the module estimates the lag-1 partial via a
Yule–Walker approximation, $\phi_{1,1} \approx c_1/c_0$. Finally a
**Ljung–Box** statistic tests whether *any* of this structure is real:

$$ Q_{\mathrm{LB}} = n(n+2)\,\frac{r_1^2}{\,n-1\,} $$

Under the null "no autocorrelation," a large $Q_{\mathrm{LB}}$ says the
series is *not* random — which is itself a useful meta-feature: a
predictable customer earns higher recommendation confidence.

That is the `ar_*` block: **4D** — lag-1, lag-7, partial lag-1, Ljung–Box.

> **Equation intuition.** $\rho_k$ is a Pearson correlation where both
> variables are the *same* series, offset by $k$ steps. The numerator
> asks "do day $t$ and day $t{+}k$ tend to be high together and low
> together?"; the denominator is the series' own variance, rescaling the
> answer into $[-1,1]$. A weekly shopper lights up at $k=7$; a
> momentum spender at $k=1$; pure noise sits near zero at every lag.

## The Frequency Domain — FFT Features

The same second-order structure is visible from a second vantage point.
Time domain asks "what is the value at time $t$"; frequency domain asks
"how strong is the oscillation at frequency $f$" — and the Fourier
transform moves losslessly between them. The deep reason the module
carries *both* an ACF block and an FFT block: the autocovariance and the
spectral density are a Fourier pair (Wiener–Khinchin), mathematically
equivalent, but a network may find one representation easier to learn
than the other. Weekly periodicity shows up as lag-7 ACF *and* as a peak
at $f = 1/7 \approx 0.143$ cycles/day.

After normalizing the amount sequence, the module runs a real FFT and
forms the power spectrum $P(f) = |X(f)|^2$, then reads four features off
it. The **spectral centroid** is the energy-weighted mean frequency — the
spectrum's center of mass:

$$ f_{\mathrm{centroid}} = \frac{\sum_i f_i\,|X(f_i)|^2}{\sum_i |X(f_i)|^2} $$

A result of 0.033 cycles/day means a dominant ~30-day (monthly) rhythm;
0.143 means ~7-day (weekly). Around that centroid, the **spectral
bandwidth** $\sqrt{\sum_i (f_i - f_{\mathrm{centroid}})^2 P(f_i)/\sum_i P(f_i)}$
measures how spread the energy is — narrow for a single dominant cycle,
wide for a tangle of overlapping ones. The **dominant frequency** is just
the peak bin (excluding the DC term). And the **spectral entropy**
normalizes the spectrum into a distribution and takes its Shannon
entropy:

$$ H_{\mathrm{spectral}} = -\sum_i p_i \log p_i, \qquad p_i = \frac{|X(f_i)|^2}{\sum_j |X(f_j)|^2} $$

Low spectral entropy means energy concentrated at one frequency — a
strong, regular cycle. High means energy smeared across many — irregular,
aperiodic spending. That is the `freq_*` block: **4D**.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="260" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Power spectrum P(f) — a weekly spender</text>
  <!-- axes -->
  <line x1="60" y1="210" x2="520" y2="210" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="210" x2="60" y2="50" stroke="#64748b" stroke-width="1.2"/>
  <text x="290" y="244" text-anchor="middle" font-size="11" fill="#1e3a5f">frequency f (cycles/day)</text>
  <text x="24" y="130" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 24 130)">power |X(f)|²</text>
  <!-- spectrum stems -->
  <g stroke="#4f46e5" stroke-width="2">
    <line x1="92" y1="210" x2="92" y2="178"/><line x1="120" y1="210" x2="120" y2="190"/>
    <line x1="148" y1="210" x2="148" y2="200"/><line x1="176" y1="210" x2="176" y2="196"/>
    <line x1="204" y1="210" x2="204" y2="186"/>
    <line x1="232" y1="210" x2="232" y2="84"/>
    <line x1="260" y1="210" x2="260" y2="188"/><line x1="288" y1="210" x2="288" y2="198"/>
    <line x1="316" y1="210" x2="316" y2="194"/><line x1="344" y1="210" x2="344" y2="201"/>
    <line x1="372" y1="210" x2="372" y2="196"/><line x1="400" y1="210" x2="400" y2="203"/>
    <line x1="428" y1="210" x2="428" y2="199"/><line x1="456" y1="210" x2="456" y2="204"/>
    <line x1="484" y1="210" x2="484" y2="200"/>
  </g>
  <g fill="#4f46e5"><circle cx="232" cy="84" r="4"/></g>
  <!-- dominant peak label -->
  <line x1="232" y1="84" x2="300" y2="64" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="3 3"/>
  <text x="304" y="62" font-size="10.5" fill="#d97706" font-weight="700">dominant peak — f ≈ 0.143 (7-day)</text>
  <!-- centroid marker -->
  <line x1="255" y1="50" x2="255" y2="210" stroke="#0d9488" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="262" y="120" font-size="10" fill="#0d9488" font-weight="700">f_centroid</text>
  <!-- bandwidth bracket -->
  <line x1="200" y1="224" x2="312" y2="224" stroke="#e11d48" stroke-width="1"/>
  <line x1="200" y1="220" x2="200" y2="228" stroke="#e11d48" stroke-width="1"/>
  <line x1="312" y1="220" x2="312" y2="228" stroke="#e11d48" stroke-width="1"/>
  <text x="256" y="237" text-anchor="middle" font-size="9.5" fill="#e11d48">bandwidth (low → regular)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Energy concentrated in one sharp peak at the weekly frequency: low spectral entropy, narrow bandwidth, centroid near the dominant period. A textbook regular spender.</figcaption>
</figure>

## Volatility and Distribution Shape

The noise term $\epsilon_t$ has a shape, and that shape carries
information beyond its variance. The module measures it with two
standardized central moments. **Skewness** $\gamma_1 = \mu_3/\sigma^3$ is
the asymmetry — positive means a right tail (mostly small spends, the odd
large one), the typical retail signature. **Excess kurtosis**
$\gamma_2 = \mu_4/\sigma^4 - 3$ is the tail thickness, the $-3$ anchoring
a normal distribution at zero:

$$ \gamma_2 = \frac{\tfrac1n \sum_i (x_i - \bar{x})^4}{\left(\tfrac1n \sum_i (x_i - \bar{x})^2\right)^2} - 3 $$

Positive excess kurtosis means extreme spending events — a big purchase,
overseas travel — arrive *more often* than a normal distribution would
predict. Both moments are computed on two sequences, the transaction
*amounts* and the inter-transaction *intervals*, giving the `dist_*`
block its **4D**: amount-skew, amount-kurtosis, interval-skew,
interval-kurtosis.

Why kurtosis and not just variance? Because of **volatility clustering** —
the financial regularity that big moves follow big moves. It barely shows
in lag-1 autocorrelation (the signs cancel), but it surfaces cleanly in
fat tails and in the autocorrelation of *absolute* values. A heavy-tailed
customer is one to whom a high-price recommendation is defensible.

## Complexity — Entropy as a Predictability Meter

The last lens asks the meta-question: *how hard is this customer to
predict at all?* Three entropy measures answer it, all built on the same
idea — turn the series into a sequence of patterns, treat pattern
frequency as a probability distribution, take its Shannon entropy. They
become features precisely because uncertainty is itself information: a
high-entropy customer can have their recommendation confidence dialed
down or their exploration dialed up.

**Approximate Entropy** compares how often length-$m$ subpatterns recur
against length-$(m{+}1)$ ones (with $m=2$, tolerance $r = 0.2\sigma$):

$$ \mathrm{ApEn}(m,r) = \Phi^m(r) - \Phi^{m+1}(r) $$

A small difference (low ApEn) means patterns keep repeating when extended —
regular. A large one means they break — irregular. **Sample Entropy**,
$\mathrm{SampEn} = -\ln(A/B)$, is the bias-corrected version that excludes
self-matches and stays reliable on short sequences. **Permutation
Entropy** ignores magnitudes entirely and looks only at the *ordinal*
pattern of $d=3$ consecutive values, normalized to $[0,1]$ by $\ln(d!)$:

$$ H_{\mathrm{perm}} = \frac{-\sum_{\pi \in \Pi} p(\pi)\,\ln p(\pi)}{\ln(d!)} $$

Permutation entropy has a quiet superpower: because it reads only the
up/down *order*, it is invariant to log1p and standardization and robust
to outliers — the same value before and after the preprocessing pipeline
touches the data. That is the `complex_*` block: **3D**. (Note: ApEn and
SampEn are $O(n^2)$, so the implementation caps the sequence at
`MAX_ENTROPY_SEQ_LEN = 300` to keep the memory from exploding; permutation
entropy runs in $O(n)$.)

## The Feature Set and Where It Lands

Putting the five families together gives the hand-crafted **18D**, which
the project labels `lnn_*` (a naming quirk — these are signal-processing
statistics, *not* the ODE-based Liquid Neural Network that lives in the
model layer). Beside it sits the learned 50D Mamba embedding for the full
**68D**:

| Family | Prefix | Dim | What it captures |
| --- | --- | --- | --- |
| Distribution shape | `dist_*` | 4D | skew/kurtosis of amounts and intervals — tail asymmetry, extremity |
| Frequency | `freq_*` | 4D | dominant frequency, spectral entropy/centroid/bandwidth — periodicity |
| Changepoint | `changepoint_*` | 3D | CUSUM count, max shift magnitude, mean segment length — regime change |
| Autocorrelation | `ar_*` | 4D | lag-1, lag-7, partial lag-1, Ljung–Box — memory and seasonality |
| Complexity | `complex_*` | 3D | ApEn, SampEn, permutation entropy — predictability |
| **Hand-crafted total** | `lnn_*` | **18D** | the four lenses, made explicit |
| Mamba embedding | `mamba_temporal_*` | 50D | SSM latent, 256D → PCA → 50D, learned |
| **Module total** | — | **68D** | order-aware features |

The changepoint block deserves a closing word because it is the one
purely *time-domain* regime detector. It runs a **CUSUM** scan — a
running sum of deviations from the mean, reset to zero whenever it dips
back below, flagged as a changepoint when it crosses $h = 2\sigma$:

$$ S_k^{+} = \max\!\big(0,\ S_{k-1}^{+} + (x_k - \bar{x})\big) $$

— with a mirrored $S_k^{-}$ watching downward shifts — and emits the
count of changepoints, the largest before/after mean shift, and the mean
segment length. A recent, large `max_shift_magnitude`
flags a life change (a move, a new job) and a reason to pivot the
recommended product set.

Where does the 68D go? Into the **734D main tensor** (644D normalized +
90D raw power-law), pre-computed *offline* before the PLE ever trains —
the Mamba 50D as part of the 159D domain group, the LNN 18D as part of
the 27D model-derived group. It is, the reference notes, the *only* group
in the entire tensor that uses temporal order; strip it out and the model
goes blind to how a customer changes over time. Densest information per
dimension in the whole schema.

A practical caveat the reference is blunt about: in the default LIVE path
(`LNN_FAST_SQL_MODE=1`) the 18D is generated by DuckDB aggregate *proxies*
— the `freq_*` become count/interval statistics, several `ar_*` are
hard-zero, changepoints become a single-pass `|amount − prev| > 2σ` count.
The exact FFT / iterative-CUSUM / ApEn-SampEn-PE implementation described
above runs only on the Python path with `LNN_FAST_SQL_MODE=0`. Honest
engineering: the elegant version and the fast version are different code,
and you should know which one produced your numbers.

## Where We Stop

We started from the discomfort that a sum, a mean and a max throw away
the one thing a spending history uniquely has — order. We read the series
as a stochastic process split into level, volatility and noise (and
flagged that the STL/HP decomposition you might expect lives one module
over, in Economics). We walked the four lenses: autocorrelation and
stationarity in the time domain, the FFT spectral features in the
frequency domain, skew and kurtosis for distribution shape, three
entropies for complexity — and saw them assemble into the hand-crafted
18D that rides beside a learned 50D Mamba embedding to make 68D in the
734D tensor.

What we deliberately set aside is the *cross-disciplinary* feature group —
the block that stops borrowing from signal processing and starts
borrowing from chemistry, epidemiology, criminology and wave physics:
reaction kinetics read as spending acceleration, SIR contagion across
categories, routine-activity burstiness, a second pass at the frequency
spectrum. Why a recommender for spending reaches into sciences that never
heard of a credit card is the subject of the next post, **MULTI-1**.
