---
title: "[Study Thread] SCORING-1 — No Server to Call: Batch Inference, the Parquet Repository, and Why Raw Scores Aren't Probabilities"
date: 2026-06-08 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, inference, scoring, calibration, duckdb, serving]
lang: en
excerpt: "How a closed-network financial system serves recommendations with no realtime inference server at all — everything is batch-precomputed into a Parquet repository and served by DuckDB lookup. Why the distilled LGBM Student runs in a nightly batch, how the FD-TVS scorer fuses 15 heterogeneous task outputs into one business score, and why a raw model score of 0.7 is not the same as a 70% probability — the Platt and one-vs-rest calibration that fixes it."
series: study-thread
part: 24
alt_lang: /2026/06/08/inference-scoring-ko/
next_title: "REASON-1 — Saying Why Without Making It Up: Tiered Reason Generation for Financial Recommendations"
next_desc: "Once a customer has a score and a ranked task, the system has to say why in plain language a marketer can trust. How the L1 template layer grounds a recommendation in extracted facts, how the L2a LLM rewrite makes it human without hallucinating, and the verdict-gated pass→fail guard that silences risky copy."
next_status: draft
---

*Part of the "Study Thread" series. This post opens the Inference &
Scoring sub-thread, drawn from the on-prem reference
`기술참조서/추론_스코어링_기술_참조서`; the full PDF will be attached to the
final post of the sub-thread. The previous sub-threads asked how the
model learns and what each Expert reads. This one asks a blunter
operational question: once the model is trained, how does a prediction
actually reach a customer? The answer here is unusual, and the reason
is a constraint most ML systems never face — this is a* closed-network
*financial system with* no realtime inference server *at all. There is
nothing to send a request to. Everything a customer might be shown is
computed ahead of time, in a batch, and looked up.*

> **The constraint that shapes everything.** Most recommendation
> writing assumes a live service: a request arrives, a model scores it,
> a response goes back in tens of milliseconds. This system has none of
> that. The closed network forbids the usual serving stack — no
> realtime inference server, no online feature cache, no dynamic
> batching. So the architecture inverts: a nightly batch scores *every*
> customer across all 15 tasks, fuses those into a single business
> score, and writes the result to a Parquet repository. Serving is then
> a *query*, not an inference — DuckDB reads the precomputed row. The
> realtime recompute path does not exist in the live code. Every design
> choice below falls out of that one fact.

## Batch-Generate, Repository-Lookup — and Why

The first fork in any inference system is *when* you predict. The
reference lays out three options, and the project is forced down one of
them by its environment.

| Mode | What it does | Upside | Downside |
| --- | --- | --- | --- |
| **Batch inference** | Predict all customers at a fixed time | High throughput, GPU utilization, cost-efficient | Stale predictions between runs |
| Realtime inference | Predict one customer on request | Freshest context, instant response | High latency cost, infra complexity |
| Micro-batch | Small batches every few seconds to minutes | Compromise of the two | Queuing complexity |

The system adopts **batch inference only**, by closed-network
necessity. A daily batch produces predictions and FD-TVS scores for the
entire customer base, lands them in a Parquet store, and serving reads
that store through DuckDB. No realtime inference server, no online
feature cache (Redis), no dynamic batching — and critically, *no
realtime recompute path exists in the live code at all*. The system
cannot fall back to "just call the model" because there is no model to
call online.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <text x="310" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Batch-generate (nightly) → Repository → Lookup (serving)</text>
  <!-- batch lane -->
  <rect x="20" y="50" width="430" height="86" rx="8" fill="#1e3a5f08" stroke="#1e3a5f" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="30" y="68" font-size="10" font-weight="700" fill="#1e3a5f">BATCH (Airflow, once per day)</text>
  <rect x="32" y="78" width="92" height="44" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="78" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">LGBM</text>
  <text x="78" y="112" text-anchor="middle" font-size="9" fill="#64748b">Student inference</text>
  <rect x="148" y="78" width="92" height="44" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="194" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">Calibrate</text>
  <text x="194" y="112" text-anchor="middle" font-size="9" fill="#64748b">Platt / OvR</text>
  <rect x="264" y="78" width="92" height="44" rx="6" fill="#fef2f2" stroke="#e11d48" stroke-width="1"/>
  <text x="310" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">FD-TVS</text>
  <text x="310" y="112" text-anchor="middle" font-size="9" fill="#64748b">4-stage score</text>
  <rect x="372" y="78" width="66" height="44" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="405" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">write</text>
  <text x="405" y="112" text-anchor="middle" font-size="9" fill="#64748b">parquet</text>
  <!-- arrows batch -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="124" y1="100" x2="146" y2="100"/><polygon points="146,100 138,96 138,104"/>
    <line x1="240" y1="100" x2="262" y2="100"/><polygon points="262,100 254,96 254,104"/>
    <line x1="356" y1="100" x2="370" y2="100"/><polygon points="370,100 362,96 362,104"/>
  </g>
  <!-- repository -->
  <ellipse cx="310" cy="172" rx="60" ry="14" fill="#1e3a5f" />
  <rect x="250" y="172" width="120" height="30" fill="#1e3a5f"/>
  <ellipse cx="310" cy="202" rx="60" ry="14" fill="#1e3a5f"/>
  <text x="310" y="180" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Parquet</text>
  <text x="310" y="196" text-anchor="middle" font-size="9" fill="#cbd5e1">repository</text>
  <line x1="405" y1="122" x2="340" y2="160" stroke="#cbd5e1" stroke-width="1.4"/><polygon points="340,160 350,159 344,167" fill="#cbd5e1"/>
  <!-- lookup lane -->
  <rect x="470" y="150" width="130" height="74" rx="8" fill="#0d948808" stroke="#0d9488" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="480" y="168" font-size="10" font-weight="700" fill="#0d9488">SERVING</text>
  <text x="535" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">DuckDB</text>
  <text x="535" y="206" text-anchor="middle" font-size="9" fill="#64748b">SELECT … = query,</text>
  <text x="535" y="218" text-anchor="middle" font-size="9" fill="#64748b">not inference</text>
  <line x1="370" y1="187" x2="468" y2="187" stroke="#0d9488" stroke-width="1.6"/><polygon points="468,187 460,183 460,191" fill="#0d9488"/>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The whole serving surface is a table lookup. Inference happens once, offline; the customer-facing path never runs a model.</figcaption>
</figure>

The trade is explicit. You give up *freshness* — a prediction is as old
as the last batch run — and in exchange you get throughput, full GPU
utilization, predictable cost, and an operationally simple serving
layer that is just a query engine over files. For a closed-network bank
where the customer base is large, behavior moves on a daily-or-slower
cadence, and the regulatory environment values reproducibility over
millisecond latency, that trade is not a compromise. It is the right
shape.

> **Historical context.** Model serving evolved in three generations.
> First (2012–2016), models were loaded straight into an API server —
> `pickle` + Flask — until Sculley et al.'s 2014 "High-Interest Credit
> Card of Technical Debt" warned of the operational debt this incurs.
> Second (2017–2021), dedicated model servers arrived — TensorFlow
> Serving, NVIDIA Triton, TorchServe — whose key innovation was
> *dynamic batching*, queuing individual requests to maximize GPU use.
> Third (2021–), Kubernetes-native platforms (KServe, Seldon) turned
> deployment into declarative YAML. This project deliberately steps
> *off* that ladder: its closed network rules out the live serving
> stack, so it returns to the oldest idea of all — precompute
> everything and read it from disk — but with a modern columnar engine
> (DuckDB over Parquet) doing the reading.

## The Scoring Pipeline over DuckDB and Parquet

What actually runs in the nightly batch? The production inference path
deliberately does *not* run the heavy PLE-adaTT Teacher — a 734D,
six-Expert model whose forward pass is too slow for serving-scale volume.
Instead, knowledge distillation hands the work to a lightweight **LGBM
Student**: the `dag_lgbm_inference` DAG runs the distilled
tree-ensemble directly over the closed-network batch, writes prediction
Parquet to the repository, and serving reads it back through DuckDB.
That Teacher→Student split is the whole strategy — *richness in
training, efficiency in serving*.

A note on the road not taken: the reference also documents an
ONNX-export and Triton-packaging path (`src/serving/*`). That path is
built — the packaging artifacts generate, the `config.pbtxt` is
written, ONNX promises a 2–5× throughput lift over raw LightGBM — but
it is **not deployed** (`triton_status=not_deployed`,
`triton_packaging_allowed` defaults to false). It is a future option,
not the live path. The live path is, again, batch LGBM → Parquet →
DuckDB.

Once every task has a prediction, those predictions are still
*heterogeneous* — 15 tasks with three different output spaces:

- **Binary** (ctr, cvr, churn, retention): a sigmoid probability in $[0,1]$.
- **Multiclass** (nba, life_stage, channel, timing, spending_category, consumption_cycle): a softmax vector summing to 1.
- **Regression** (ltv, engagement, balance_util, spending_bucket, merchant_affinity): an unbounded real number — 0 to millions.

You cannot just add these up. If you naively sum them, LTV's absolute
magnitude (hundreds of thousands) swamps a CTR probability (a fraction
of one). The **FD-TVS Scoring Engine** exists to fuse them sanely. Its
Stage 1 is a Weighted Sum Model — the most basic multi-criteria
decision rule — that requires every input to be a probability in
$[0,1]$:

$$ S_{\text{task}} = \sum_{i=1}^{n} \beta_i \cdot p_i, \qquad \sum_i \beta_i = 1, \quad p_i \in [0,1] $$

> **Equation intuition.** With weights $\beta_i \ge 0$ summing to 1,
> this is a *convex combination* — a weighted center of mass of the
> task probabilities. The payoff is automatic: if every $p_i \in
> [0,1]$, then $S_{\text{task}} \in [0,1]$ too, because $S = \sum
> \beta_i p_i \le \sum \beta_i \cdot 1 = 1$ and likewise $\ge 0$. The
> score can never escape its range — *provided every input really is a
> probability*. That proviso is the entire reason the next section
> exists. Later stages (DNA fit, TDA vitality, risk penalty, fatigue
> decay, engagement boost) multiply onto this, so a single near-zero factor can veto the whole
> recommendation — risk-first by construction.

## Why Raw Scores Aren't Probabilities

Here is the load-bearing subtlety. When the LGBM Student outputs
"0.7," that number is *not* a calibrated probability. It is a model
score that happens to live in $[0,1]$. A model is **well-calibrated**
only when, among all the customers it scores at 0.7, exactly 70%
actually convert. Formally:

$$ P(Y = 1 \mid \hat{p} = q) = q \qquad \forall\, q \in [0,1] $$

Tree ensembles, like most classifiers, routinely violate this — they
run over-confident or under-confident. And in *this* system that is not
a cosmetic flaw, because of the FD-TVS Stage 1 weighted sum above: if
the CTR model is over-confident and the CVR model is under-confident,
the weighted fusion silently tilts toward CTR. Mis-calibration in one
task corrupts the *combined* business score. Calibration is not
optional polish here; it is a precondition for the fusion to be
meaningful.

The fix is a **post-hoc calibration** layer fit on a holdout split. The
project's `ProbabilityCalibrator` (`src/evaluation/calibration.py`)
supports three methods:

| Method | How it works | Fits | Notes |
| --- | --- | --- | --- |
| **Platt scaling** | Re-map the score through a sigmoid | 2 params $(A,B)$ | Good for binary, low-data |
| **Isotonic** | Fit a monotone step function | nonparametric | Flexible with much data; can overfit |
| **none** | Pass through (clip only) | — | Legacy behaviour |

Platt scaling, the binary workhorse, comes from John Platt (1999), who
needed to turn an SVM's bare decision value $f(x)$ into a probability.
The recipe is a one-dimensional logistic regression on the score:

$$ P(y{=}1\mid s) = \frac{1}{1+\exp(As+B)} $$

with $A, B$ estimated by maximum likelihood on the holdout. Because the
LGBM Booster does not follow the sklearn API, the project uses a
*score-wrap* mode: take `booster.predict(X_val)` to get raw scores, fit
the calibrator from those scores against the validation labels
(`fit_from_scores`), and at inference time push fresh scores through
`transform`.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="240" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Platt scaling — a sigmoid over the raw score</text>
  <!-- axes -->
  <line x1="60" y1="200" x2="470" y2="200" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="200" x2="60" y2="48" stroke="#64748b" stroke-width="1.2"/>
  <text x="265" y="228" text-anchor="middle" font-size="11" fill="#1e3a5f">raw score s</text>
  <text x="26" y="124" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 26 124)">P(y=1 | s)</text>
  <!-- gridline at 0.5 -->
  <line x1="60" y1="124" x2="470" y2="124" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3 3"/>
  <text x="48" y="128" text-anchor="end" font-size="9" fill="#94a3b8">0.5</text>
  <text x="48" y="204" text-anchor="end" font-size="9" fill="#94a3b8">0</text>
  <text x="48" y="52" text-anchor="end" font-size="9" fill="#94a3b8">1</text>
  <!-- sigmoid curve: P = 1/(1+exp(A s + B)), drawn as smooth S -->
  <path d="M 60 196 C 150 192, 200 186, 245 124 C 290 62, 360 54, 470 51" fill="none" stroke="#0d9488" stroke-width="2.2"/>
  <!-- midpoint marker -->
  <circle cx="245" cy="124" r="4.5" fill="#d97706"/>
  <text x="252" y="146" font-size="10" fill="#d97706">inflection at s = −B/A</text>
  <text x="330" y="92" font-size="10" fill="#0d9488" font-weight="700">2 params (A, B) by MLE</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Platt scaling fits a single logistic curve mapping the raw model score s to a calibrated probability — just two parameters, A (slope) and B (shift), estimated on the holdout.</figcaption>
</figure>

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="270" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Reliability diagram — before vs after calibration</text>
  <!-- axes -->
  <line x1="70" y1="230" x2="510" y2="230" stroke="#64748b" stroke-width="1.2"/>
  <line x1="70" y1="230" x2="70" y2="50" stroke="#64748b" stroke-width="1.2"/>
  <text x="290" y="258" text-anchor="middle" font-size="11" fill="#1e3a5f">mean predicted probability</text>
  <text x="30" y="140" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 30 140)">fraction positive</text>
  <!-- perfect diagonal -->
  <line x1="70" y1="230" x2="490" y2="50" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="430" y="70" font-size="9.5" fill="#94a3b8">perfectly calibrated</text>
  <!-- uncalibrated curve (over-confident, sags below diagonal) -->
  <path d="M 70 230 C 170 220, 250 205, 320 150 C 380 105, 440 78, 490 50" fill="none" stroke="#e11d48" stroke-width="2"/>
  <text x="350" y="200" font-size="10" fill="#e11d48" font-weight="700">raw score (mis-calibrated)</text>
  <!-- calibrated curve (hugs diagonal) -->
  <path d="M 70 230 C 160 195, 250 158, 320 138 C 390 116, 450 78, 490 52" fill="none" stroke="#0d9488" stroke-width="2"/>
  <text x="110" y="150" font-size="10" fill="#0d9488" font-weight="700">after Platt / OvR</text>
  <!-- ECE arrow gap -->
  <line x1="320" y1="150" x2="320" y2="138" stroke="#d97706" stroke-width="6"/>
  <text x="332" y="148" font-size="9.5" fill="#d97706">ECE gap ↓</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A reliability diagram bins predictions and plots predicted vs observed frequency. The red curve sags off the diagonal (over-confident); calibration pulls it back. ECE is the average vertical gap.</figcaption>
</figure>

How do we *measure* whether it worked? The reliability diagram bins the
predictions and compares, per bin, the mean predicted probability
against the observed fraction of positives. The **Expected Calibration
Error** is the average gap across bins, and the project reports
calibration quality as $1 - \text{ECE}$:

$$ \text{ECE} = \frac{1}{B}\sum_{b=1}^{B}\big|\,\text{acc}(b) - \text{conf}(b)\,\big|, \qquad \text{cal\_score} = 1 - \text{ECE} $$

where $\text{conf}(b)$ is the mean predicted probability in bin $b$ and
$\text{acc}(b)$ the observed positive fraction. A perfectly calibrated
model has ECE 0, score 1.

### Multiclass: one calibrator per class, then renormalize

Binary calibration maps one score. A multiclass head (nba's 12 classes,
timing's 28) emits a whole softmax vector, and you cannot Platt-scale a
vector directly. The project uses **one-vs-rest**: for each class $c$,
treat "is it class $c$?" as a binary problem, fit a `ProbabilityCalibrator`
on $p_{\cdot,c}$ against the indicator $\mathbb{1}[y=c]$, and store a
`{class → calibrator}` dict. At inference:

$$ \tilde{p}_{c} = \text{cal}_c\!\big(p_{c}\big), \qquad \hat{p}_{c} = \frac{\tilde{p}_{c}}{\sum_{k}\tilde{p}_{k}} $$

The final division is the crucial step: each class is calibrated
*independently*, so the per-class results no longer sum to 1 — the
renormalization restores a proper distribution. Degenerate classes (a
class absent from the validation split) are skipped and pass through
uncalibrated, so calibration never crashes on a sparse head.

> A caveat the reference is honest about. In the *baseline* postprocessor
> documented in the tech reference, raw predictions were used as-is, and
> adding a calibration layer was listed as a future score-quality task.
> The `ProbabilityCalibrator` described here is the on-prem
> implementation that closes that gap — opt-in via a
> `calibration_method` config (default `none` preserves legacy
> behaviour), with the calibrators pickled per task alongside the
> Student model. It is wiring that *exists*; whether a given production
> run enables it is a config decision, not a guarantee.

## Throughput, Cost, and Where This Sits

The economics of the batch design are straightforward. Because there is
no live request path, there is no tail-latency budget to defend and no
idle-but-provisioned inference fleet to pay for. The cost is a single
nightly compute window over the whole customer base, sized for
*throughput*, not latency — exactly the regime where batch inference
and GPU utilization shine. The serving cost collapses to running DuckDB
queries over Parquet files, which is cheap, trivially scalable, and
needs no specialized serving infrastructure.

The reference also flags where, *if* the undeployed Triton/ONNX path
were ever activated, the latency bottlenecks would live: JSON parsing of
200+ features in the preprocessor (CPU-bound), the ONNX tree-ensemble
forward (worse with 1000+ trees), and JSON serialization in the
postprocessor. Useful to know — but on the live batch path, none of
these sit on a customer-facing critical path, because there is no
customer-facing inference at all.

So the scoring layer sits squarely between two neighbors. *Upstream* is
distillation: the PLE-adaTT Teacher trains the rich representation, and
the LGBM Student inherits its discriminative power in a form fast enough
to batch-score. *Downstream* is the recommendation and reason layer: the
FD-TVS score and ranked task become the input to the reason generator,
which has to turn a number into a sentence a marketer can act on. There
is also a routing safety net we have only touched — the FallbackRouter's
three layers and the rule-based baseline that fills in for any missing
LGBM task — but that belongs to its own discussion.

## Where We Stop

We started from the one constraint that defines this system — *no
realtime inference server* — and watched the architecture invert around
it: a nightly batch that scores every customer, a Parquet repository
that holds the results, and a serving layer that is a DuckDB query
rather than a model call. We saw the distilled LGBM Student do the
batch inference, the FD-TVS engine fuse 15 heterogeneous task outputs
through a convex weighted sum, and — the load-bearing detail — why that
sum is only trustworthy if every input is a *real* probability, which
raw scores are not. Platt scaling repairs the binary heads,
one-vs-rest-plus-renormalize repairs the multiclass heads, and ECE
tells us whether the repair held.

What remains is the last mile: the score and its ranked task are still
just numbers. The system has to explain *why* — in language a marketer
can trust and a regulator can audit, without inventing facts the data
does not support. How the L1 template layer grounds a recommendation in
extracted facts, how the L2a LLM rewrite makes it readable, and the
verdict-gated guard that turns a risky draft from pass to fail — that is
the subject of the next post, **REASON-1**.
