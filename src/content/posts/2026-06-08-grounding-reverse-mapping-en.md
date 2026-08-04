---
title: "[Study Thread] GROUND-1 — The Dictionary Between Numbers and Reasons: Feature Reverse-Mapping"
date: 2026-06-08 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, grounding, feature-mapping, attribution, explainability]
lang: en
excerpt: "The model speaks in 734 dimensions; a recommendation reason must speak in human facts. This post unpacks reverse-mapping — the dictionary between the two: how a feature index maps back to a named financial quantity, how Integrated Gradients picks the dominant features, how a YAML-rule fact extractor turns those into deterministic narrative facts, and how the whole thing hands off to reason generation without inventing anything."
series: study-thread
part: 26
alt_lang: /2026/06/08/grounding-reverse-mapping-ko/
next_title: "SERVE-1 — Qwen on vLLM: Serving the LLM that Rewrites the Reason"
next_desc: "The grounded facts have to be turned into a sentence by an actual model. The next post turns to the serving side — running Qwen on vLLM in a closed network: the OpenAI-compatible endpoint, batched generation over millions of customers, JSON-mode constraints, and the latency/throughput trade-offs of a batch-only reason pipeline."
next_status: draft
---

*Part of the Grounding sub-thread in the "Study Thread" series, in
parallel Korean and English. The source is the on-prem reference
`기술참조서/그라운딩_피쳐역매핑_기술_참조서`, and the full PDF will be
attached to the final post of the sub-thread. The recommendation model
in this project does not speak. It takes a 734-dimensional feature
vector and emits a probability — a number. But a reason a sales agent
can read, or a regulator can file, has to be made of human facts:
"frequent overseas payments," "rising interest in travel merchants."
Reverse-mapping is the* dictionary *between those two languages — the
bridge from the model's numeric features back to meaning. It feeds the
grounded facts that the* next *thread, REASON, rewrites into a finished
sentence; here we build the facts themselves.*

> **Why this exists.** A black-box recommendation fails three ways at
> once. The sales floor ignores a recommendation it cannot explain. The
> compliance file cannot list "AI model output" as a suitability
> rationale. And a data scientist staring at a bare score cannot catch
> the model when it learns a spurious correlation — say, a region code
> standing in for income. Reverse-mapping closes all three: it turns
> the dominant features into financial language a human reads, a
> regulator accepts, and an analyst can audit. Without it, the trust
> loop — *predict → attribute → ground → assemble → write → persuade →
> convert → improve* — snaps right between "predict" and "the agent."

## The Gap: a Number Is Not a Reason

A trained PLE-adaTT model maps a 734-dimensional feature vector to
probability scores for eighteen tasks — CTR, CVR, churn, and the rest.
That score answers *what* to recommend. It says nothing about *why*.

And the features themselves are no help to a human. A feature named
`chemical_kinetics_003` might rank first in importance, but no sales
agent on earth knows what that means. The honest translation —
"rate of trying new merchant categories (category-switch acceleration)" — is
nowhere in the number. There are two gaps to cross, not one:

1. **The attribution gap.** Of 734 numbers, *which* ones drove this
   particular prediction? A raw feature vector treats all dimensions
   equally; we need to know which ones mattered.
2. **The naming gap.** Once we know feature index *i* mattered, what
   human-meaningful quantity does *i* actually stand for, and what does
   its *value* mean in financial terms?

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="230" fill="#f8fafc" rx="8"/>
  <text x="290" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Two gaps between the model and a reason</text>
  <!-- 734D vector -->
  <rect x="30" y="70" width="120" height="100" rx="6" fill="#1e3a5f12" stroke="#1e3a5f" stroke-width="1"/>
  <text x="90" y="60" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">734D vector</text>
  <g font-family="monospace" font-size="9" fill="#64748b">
    <text x="45" y="92">0.12  0.85  …</text>
    <text x="45" y="108">0.04  0.71  …</text>
    <text x="45" y="124">0.93  0.08  …</text>
    <text x="45" y="140">0.55  0.27  …</text>
    <text x="45" y="156">…just numbers</text>
  </g>
  <!-- gap 1 -->
  <text x="232" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">gap 1</text>
  <text x="232" y="114" text-anchor="middle" font-size="9" fill="#64748b">which mattered?</text>
  <text x="232" y="128" text-anchor="middle" font-size="9" fill="#64748b">(IG attribution)</text>
  <!-- top features -->
  <rect x="300" y="78" width="130" height="84" rx="6" fill="#d9770612" stroke="#d97706" stroke-width="1"/>
  <text x="365" y="60" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">top features</text>
  <g font-family="monospace" font-size="9" fill="#64748b">
    <text x="312" y="98">feat #341  +0.12</text>
    <text x="312" y="114">feat #088  +0.08</text>
    <text x="312" y="130">feat #602  +0.07</text>
    <text x="312" y="150">…still opaque</text>
  </g>
  <!-- gap 2 -->
  <text x="478" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">gap 2</text>
  <text x="478" y="114" text-anchor="middle" font-size="9" fill="#64748b">what does it</text>
  <text x="478" y="128" text-anchor="middle" font-size="9" fill="#64748b">mean? (mapping)</text>
  <!-- human fact -->
  <rect x="522" y="86" width="44" height="68" rx="6" fill="#0d9488" />
  <text x="544" y="116" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">human</text>
  <text x="544" y="130" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">fact</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="150" y1="120" x2="298" y2="120"/><polygon points="298,120 290,116 290,124"/>
    <line x1="430" y1="120" x2="520" y2="120"/><polygon points="520,120 512,116 512,124"/>
  </g>
  <text x="290" y="206" text-anchor="middle" font-size="11" fill="#64748b">"frequent overseas payments; rising travel-merchant interest"</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Two crossings: attribution finds which dimensions drove the score, reverse-mapping names what they mean. Only after both does a number become a fact.</figcaption>
</figure>

The first gap is closed by attribution; the second by reverse-mapping.
This post is mostly about the second — but the two are inseparable, so
we start with a quick word on the first.

## Attribution: Which Numbers Drove the Score

To know which features to translate, the system attributes the
prediction back to its inputs with **Integrated Gradients** (IG;
Sundararajan et al., 2017). IG asks a blunt question: as a feature
moves from a baseline (here, the zero vector) up to its actual value,
how much did the prediction move with it? Formally it integrates the
gradient along the straight path from baseline $\mathbf{x}'$ to input
$\mathbf{x}$:

$$ \mathrm{IG}_i(\mathbf{x}) = (x_i - x'_i)\,\int_0^1 \frac{\partial F\big(\mathbf{x}' + \alpha(\mathbf{x}-\mathbf{x}')\big)}{\partial x_i}\, d\alpha $$

The property that earns IG its place over plain gradients is
**completeness**: the attributions sum exactly to the prediction gap.

$$ \sum_{i=1}^{644} \mathrm{IG}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}') $$

If CTR is predicted at 0.73 and the baseline at 0.15, the 644
attributions sum to exactly 0.58 — nothing leaks, nothing is double
counted. Practically, that guarantee is what lets a reverse-mapped
reason claim to *explain* the score rather than gesture at it.

> **Design intuition.** Completeness is the Fundamental Theorem of
> Calculus wearing a different hat. $\int_a^b f'(t)\,dt = f(b)-f(a)$ —
> integrate a derivative and you recover the endpoints' difference. IG
> does the same in 644 dimensions: the line integral of the gradient
> along the baseline→input path is, by the Gradient Theorem, exactly
> $F(\mathbf{x})-F(\mathbf{x}')$ regardless of the path. That is why the
> attribution leaks nothing — and why a plain gradient, which saturates
> to near-zero in flat regions and *undercounts* real contributions, is
> not good enough.

The output is a 734-D attribution vector $\mathbf{a}$. Take the
top-$K$ by absolute value and you have the handful of features worth
translating. That hand-off — top features in, named facts out — is the
job of the reverse-mapping engine.

## Feature Reverse-Mapping: from Index to Named Quantity

Reverse-mapping is this project's own term for one transformation:

$$ \mathrm{ReverseMap}:\ \big(\mathbf{x}\in\mathbb{R}^d,\ \mathbf{a}\in\mathbb{R}^d\big)\ \longrightarrow\ \{(r_k, s_k, t_k)\}_{k=1}^{K} $$

where $\mathbf{x}$ is the raw feature vector, $\mathbf{a}$ the IG
attribution, $r_k$ a *feature range* name (`profile`, `domain`, …),
$s_k$ a summary score for that range, and $t_k$ the financial-language
text. The hard part is doing two things at once: **dimensionality
reduction** (644 floats → roughly ten sentences) *and* **meaning
assignment**, without throwing away the signal.

The engine — `FeatureReverseMapper` in
`feature_reverse_mapper.py` — does not treat 734 dimensions as a flat
soup. The vector has a *known layout*. The 734D reverse-mapping input
is the V1-compatible structure: 644 normalized features plus 90 raw
power-law features. The 644 normalized dimensions decompose into seven
contiguous ranges:

> **The contract has since moved.** The 734D above is the V1 feature
> contract. On 2026-07-02 the project switched to the V2 strict contract
> and the operational input width is **4035D** — 734D was not discarded;
> it remains V2's _shared base of eight groups_, with the
> lag/rolling/product families (3301D) appended on top.

$$ 238_{\text{profile}} + 91_{\text{multi\_source}} + 84_{\text{extended}} + 159_{\text{domain}} + 27_{\text{model\_derived}} + 24_{\text{multi\_disc}} + 21_{\text{merchant}} = 644 $$

Each range is a slice of contiguous indices with a name and a
description, held in a small `FeatureRange` dataclass:

```python
@dataclass
class FeatureRange:
    start: int       # e.g. 413 for domain
    end: int         # e.g. 572 for domain
    name: str        # "domain"
    description: str  # "TDA(70) + GMM(22) + Mamba(50) + Economics(17)"
```

So when IG hands the engine feature index 470, a lookup says: 470 falls
in `domain` (413–572), specifically the TDA sub-range — this is a
*topological persistence* feature, not a demographic one. The index is
no longer anonymous; it has a home, and the home decides the dictionary
used to translate it.

The core loop, repeated for every range, is one pattern:

$$ t_k = \mathcal{M}_k\big(g(\mathbf{x}[s_k:e_k])\big) $$

— slice the sub-range, aggregate it with $g$ (a `np.mean`, an
`np.argmax`, or a threshold compare), then look the result up in
$\mathcal{M}_k$, the range-specific dictionary that turns a number into
a phrase.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">slice → aggregate → look up</text>
  <!-- full vector with ranges -->
  <text x="300" y="50" text-anchor="middle" font-size="10" fill="#64748b">644D vector, laid out in named ranges</text>
  <g>
    <rect x="40"  y="60" width="120" height="26" fill="#1e3a5f22" stroke="#1e3a5f" stroke-width="0.8"/>
    <rect x="160" y="60" width="60"  height="26" fill="#0d948822" stroke="#0d9488" stroke-width="0.8"/>
    <rect x="220" y="60" width="55"  height="26" fill="#d9770622" stroke="#d97706" stroke-width="0.8"/>
    <rect x="275" y="60" width="100" height="26" fill="#4f46e522" stroke="#4f46e5" stroke-width="0.8"/>
    <rect x="375" y="60" width="40"  height="26" fill="#64748b22" stroke="#64748b" stroke-width="0.8"/>
    <rect x="415" y="60" width="35"  height="26" fill="#e11d4822" stroke="#e11d48" stroke-width="0.8"/>
    <rect x="450" y="60" width="30"  height="26" fill="#0d948822" stroke="#0d9488" stroke-width="0.8"/>
  </g>
  <text x="100" y="78" text-anchor="middle" font-size="8" fill="#1e3a5f">profile</text>
  <text x="190" y="78" text-anchor="middle" font-size="8" fill="#0d9488">multi_src</text>
  <text x="325" y="78" text-anchor="middle" font-size="8" fill="#4f46e5">domain</text>
  <!-- highlight one sub-range (RFM inside profile) -->
  <rect x="78" y="58" width="34" height="30" fill="none" stroke="#e11d48" stroke-width="2" rx="2"/>
  <text x="95" y="104" text-anchor="middle" font-size="9" fill="#e11d48" font-weight="700">RFM slice</text>
  <!-- arrow down -->
  <line x1="95" y1="108" x2="95" y2="132" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="95,132 91,124 99,124" fill="#cbd5e1"/>
  <!-- aggregate -->
  <rect x="40" y="135" width="160" height="44" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="120" y="154" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">g = np.mean</text>
  <text x="120" y="170" text-anchor="middle" font-size="9" fill="#64748b">Recency mean = 0.82</text>
  <!-- arrow -->
  <line x1="200" y1="157" x2="248" y2="157" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="248,157 240,153 240,161" fill="#cbd5e1"/>
  <!-- dictionary -->
  <rect x="250" y="128" width="170" height="60" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="335" y="146" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">M (dictionary)</text>
  <text x="335" y="162" text-anchor="middle" font-size="8.5" fill="#64748b">&gt; 0.7 → "very active recently"</text>
  <text x="335" y="176" text-anchor="middle" font-size="8.5" fill="#64748b">0.4–0.7 → "moderate" · &lt; 0.4 → "low"</text>
  <!-- arrow -->
  <line x1="420" y1="157" x2="468" y2="157" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="468,157 460,153 460,161" fill="#cbd5e1"/>
  <!-- text out -->
  <rect x="470" y="138" width="100" height="40" rx="6" fill="#0d9488"/>
  <text x="520" y="162" text-anchor="middle" font-size="9.5" font-weight="700" fill="#fff">"very active</text>
  <text x="520" y="174" text-anchor="middle" font-size="9.5" font-weight="700" fill="#fff">recently"</text>
  <text x="300" y="220" text-anchor="middle" font-size="10" fill="#64748b">the same pattern repeats over 7 ranges × many sub-ranges, then assembles into one explanation</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The reverse-mapping core loop: slice a known sub-range, aggregate to a scalar, look it up in a domain-designed dictionary. Repeated and assembled, 644 numbers become a paragraph.</figcaption>
</figure>

A concrete worked example. Take the RFM block (Recency / Frequency /
Monetary) that lives at offset 100–150 inside `profile`. The engine
slices it, takes the mean of each axis, and runs three threshold
lookups:

| Axis | Value | Threshold rule | Mapped label |
| --- | --- | --- | --- |
| Recency | 0.82 | `> 0.7` | very active recently (txn within 7 days) |
| Frequency | 0.55 | `0.4–0.7` | mid-frequency (5–15 txns/month) |
| Monetary | 0.31 | `< 0.4` | small spender (< ₩300k/month) |

joined with `/`: **"very active recently / mid-frequency / small
spender."** That is a number turned into a sentence a human reads
without a manual.

The same machine handles the harder ranges by reaching for richer
aggregators. The table below shows how a feature *group* maps to a
reverse-mapped fact, and which aggregation does the work:

| Feature group (dim) | Aggregator | Reverse-mapped fact (example) |
| --- | --- | --- |
| RFM (50D in profile) | per-axis mean + threshold | "very active recently / mid-frequency / small spender" |
| credit/investment (financial 88D) | named ratio + threshold | "credit-limit utilization high → credit-risk watch; investment-leaning" |
| TDA persistence (domain) | scalar threshold | "spending pattern stable (>80% retained); behavior-shift likely" |
| HMM triple-mode (48D) | `argmax` over 16D states | "lifecycle: maturity; purchase journey: consideration stage" |
| GMM cluster (22D) | `argmax` over probabilities | "primary segment: VIP (membership prob 73.2%)" |
| chemical_kinetics (6D in the 24D multi-disc block) | mean + interpretation range | "spending-acceleration high — trying new merchant categories" |
| MCC hierarchy (21D merchant) | Level-1/2 + radius | "high loyalty to a specific merchant" |

Notice the metaphor-laundering in the last rows. A name like
`chemical_kinetics_003` is borrowed-science jargon; the dictionary for
that range translates it into a *business concept* — "rate of trying
new categories." This is exactly the concept-based-explanation move:
explain in units a person reasons about ("travel appetite,"
"thrift pattern"), not in raw feature units.

> **Historical context.** The theory under all of this is older than the
> model. Attribution traces to Lloyd Shapley's 1953 *"A Value for
> n-Person Games"* — the unique fair way to split a cooperative payoff,
> defined by four axioms (efficiency, symmetry, dummy, additivity).
> That sat in pure economics for sixty years until Lundberg & Lee's 2017
> SHAP paper rediscovered it for machine learning. Integrated Gradients
> arrived the same year by an identical *axiomatic* route — define the
> properties a good attribution must satisfy, then derive the unique
> method — replacing Shapley's discrete subset sum with a continuous
> path integral. Reverse-mapping is the last, unglamorous mile: turning
> that fair, leak-free attribution into a sentence a branch manager can
> repeat.

## The fact_extractor: Deterministic Facts from YAML Rules

Reverse-mapping produces flowing financial prose, range by range. But
there is a second, complementary channel that produces *atomic,
checkable facts* — and it does so without ever calling an LLM. This is
the `FactExtractor` (`fact_extractor.py`), a Mem0-inspired,
rule-based fact compression layer ported from the AWS
`core/recommendation/reason/fact_extractor.py`.

The idea is deliberately humble. A customer's features arrive as a
plain dict. A YAML config lists rules; each rule is a name, a boolean
condition, and the features that condition needs. If the condition
holds, the rule's name becomes a fact string. That is the whole engine.

```yaml
# fact_extraction.yaml
rules:
  - name: "deposit-centric portfolio"
    condition: "deposit_balance_ratio >= 0.7"
    required_features: ["deposit_balance_ratio"]
  - name: "rising fund interest (last 3 months)"
    condition: "fund_view_count_3m >= 5"
    required_features: ["fund_view_count_3m"]
  - name: "risk-averse profile"
    condition: "risk_tolerance_score <= 0.3"
    required_features: ["risk_tolerance_score"]
```

```python
extractor = FactExtractor("configs/fact_extraction.yaml")
facts = extractor.extract({
    "deposit_balance_ratio": 0.75,
    "fund_view_count_3m": 8,
    "risk_tolerance_score": 0.2,
})
# → ["deposit-centric portfolio",
#    "rising fund interest (last 3 months)",
#    "risk-averse profile"]
```

Three properties make this worth having alongside the reverse-mapper:

- **Deterministic.** Same dict in, same facts out, every time — no
  sampling, no temperature, no drift. The facts can be cached, diffed,
  and audited.
- **Cheap and batchable.** `extract_batch()` runs the same rules over a
  DataFrame row by row; in this project it has been run across roughly
  5.3M customers in one pass. No model calls means no GPU and no
  per-customer latency.
- **Safe by construction.** A rule's condition is `eval`-ed, but inside
  a locked-down namespace — `__builtins__` is emptied and only a small
  allow-list (`abs`, `min`, `max`, `len`, `round`, `int`, `float`, …)
  plus the customer's own features is injected. A condition that
  references a missing feature, or throws, is silently skipped rather
  than crashing the batch.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="240" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">YAML rule → fact, deterministically</text>
  <!-- feature dict -->
  <rect x="28" y="60" width="130" height="120" rx="6" fill="#1e3a5f10" stroke="#1e3a5f" stroke-width="1"/>
  <text x="93" y="52" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">feature dict</text>
  <g font-family="monospace" font-size="8.5" fill="#64748b">
    <text x="40" y="84">deposit_ratio</text><text x="148" y="84" text-anchor="end" fill="#1e3a5f">0.75</text>
    <text x="40" y="106">fund_view_3m</text><text x="148" y="106" text-anchor="end" fill="#1e3a5f">8</text>
    <text x="40" y="128">risk_score</text><text x="148" y="128" text-anchor="end" fill="#1e3a5f">0.2</text>
    <text x="40" y="150">…</text>
  </g>
  <!-- rules -->
  <rect x="220" y="50" width="160" height="140" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="300" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">YAML rules (eval)</text>
  <g font-size="8.5" fill="#64748b">
    <text x="232" y="76">deposit_ratio ≥ 0.7</text><text x="368" y="76" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="104">fund_view_3m ≥ 5</text><text x="368" y="104" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="132">risk_score ≤ 0.3</text><text x="368" y="132" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="160">campaign_resp ≥ 0.6</text><text x="368" y="160" text-anchor="end" fill="#e11d48" font-weight="700">✗</text>
  </g>
  <text x="300" y="182" text-anchor="middle" font-size="8" fill="#64748b">locked namespace · missing-feature → skip</text>
  <!-- facts out -->
  <rect x="430" y="64" width="150" height="112" rx="6" fill="#0d948812" stroke="#0d9488" stroke-width="1"/>
  <text x="505" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">fact list</text>
  <g font-size="8.5" fill="#0f766e">
    <text x="442" y="88">• deposit-centric</text>
    <text x="442" y="110">• rising fund interest</text>
    <text x="442" y="132">• risk-averse</text>
  </g>
  <text x="505" y="164" text-anchor="middle" font-size="8" fill="#64748b">no LLM · cacheable · auditable</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="158" y1="120" x2="218" y2="120"/><polygon points="218,120 210,116 210,124"/>
    <line x1="380" y1="120" x2="428" y2="120"/><polygon points="428,120 420,116 420,124"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The fact_extractor evaluates each YAML rule against the customer dict in a locked namespace. Conditions that pass become facts; missing or failing ones are skipped. No model in the loop.</figcaption>
</figure>

The two channels are not redundant. Reverse-mapping reads the dominant
*attributed* features and writes range-level prose ("why this customer
scored high"); the fact_extractor reads the full feature dict against a
curated rule book and emits short, hard, individually-true claims. The
reason generator gets both — narrative for fluency, atomic facts for
grounding.

## How Attributions Become Candidate Facts

Put the pieces in order and the data flow is short. The top-$K$ IG
features are used in exactly two places, and the duality is the point:

1. As `top_features` in the `reverse_map()` return value — handed
   straight to the client and to downstream prose, answering *"why is
   this feature important?"*
2. As `ig_top_features` into `ContextAssemblyAgent.assemble()` — the
   *basis for tool selection*, answering *"to explain this feature more
   deeply, which source do I need?"*

That second use is where attribution starts steering the pipeline. The
assembly agent maps each top feature back to its range and lets the
range decide which tools to fire. If the dominant feature sits in
`multidisciplinary`, it calls the multidisciplinary interpreter; if it
sits in `extended_source`, it pulls consultation history. Only the
features that actually drove the score earn the cost of deep context.

| Top-feature range | Tools the agent may fire |
| --- | --- |
| `profile` | `reverse_map`, `query_context` |
| `multi_source` | `reverse_map`, `get_consultation` |
| `extended_source` | `reverse_map`, `get_consultation` |
| `domain` | `reverse_map`, `interpret_multi` |
| `multidisciplinary` | `interpret_multi`, `query_similar` |
| `model_derived` | `reverse_map`, `query_similar` |
| `merchant_hierarchy` | `reverse_map` |

A "richness tier" then caps how many tools may run (tier 1: up to 5
tools; tier 3: `reverse_map` only) so the context budget tracks how
much signal the customer actually carries. The output of all this — the
reverse-mapped prose, the extracted facts, the consultation snippets,
the similar-customer hits — is assembled into a single context bundle.

## The Hand-Off to Reason Generation

The grounding stage does not write the final sentence; it prepares
everything the writer needs. In batch order the components cooperate:

1. **Reverse-mapping** (`FeatureReverseMapper`) — normalized vector + IG
   → per-range financial language.
2. **Fact extraction** (`FactExtractor`) — feature dict → deterministic
   narrative facts, no LLM.
3. **Context storage** (`LanceContextVectorStore`) — embed the mapped
   text, store it with the facts and consultation summaries in a
   `customer_context` table.
4. **Context assembly** (`ContextAssemblyAgent`) — IG → tool selection →
   merge reverse-map / consultation / multidisciplinary / similar-
   customer sources into one LLM-ready bundle.
5. **Reason generation** (L1 template → L2a LLM rewrite) — the bundle
   plus the fact list become the input to the actual sentence.

The fact list is injected directly into the L2a/L2b prompt, and its job
there is narrow but important: **reduce hallucination.** The rewriting
LLM is free to make the language natural, but it is handed a set of
pre-verified, deterministic facts to stay anchored to — it rephrases,
it does not invent. This is the seam where GROUND-1 meets the REASON
thread: everything here produces the grounded material; the reason
generator rewrites it into prose a customer reads.

## Safeguards

A grounding pipeline that quietly lies is worse than none, so several
guards are built in:

- **Completeness as a runtime check.** Because IG attributions must sum
  to $F(\mathbf{x}) - F(\mathbf{x}')$, that identity can be asserted at
  runtime; a violated sum flags a broken attribution before it
  contaminates a reason.
- **Sandboxed rule evaluation.** The fact_extractor's `eval` runs with
  emptied `__builtins__` and an allow-listed namespace, and any rule
  that errors or references a missing feature is skipped — a bad rule
  degrades one fact, never the batch.
- **Graceful dimension mismatch.** If the feature vector length and the
  feature-name list disagree (a real hazard during the V1↔V2 dimension
  transition), `reverse_map()` logs a warning and keeps going rather
  than aborting — deliberate, so version skew never blocks the batch.
- **SQL fallback in batch.** `batch_reverse_map()` reads Parquet via
  DuckDB SQL; on a column mismatch it auto-switches to
  `_batch_reverse_map_simple()`, which needs only the encrypted
  customer id.
- **Output quality gate.** Downstream, `L2QualityValidator` scores
  generated reasons on factuality / relevance / naturalness via
  stratified sampling, and a `fail` verdict is treated as a *silent
  risk* and blocked — so a reason that drifts off its grounded facts
  does not reach a customer.

Together these make the grounding stage fail *loudly and locally*: a
broken attribution, a malformed rule, or a drifted sentence is caught
and contained, never silently shipped as a confident-sounding lie.

## Where We Stop

We started from a gap — the model turns a 734-dimensional vector into
a bare number, a reason needs human facts — and built the dictionary
across it. Integrated
Gradients picks the dominant features with a leak-free, axiom-backed
attribution; the reverse-mapper slices each known feature range and
looks its aggregate up in a domain-designed dictionary; the
fact_extractor distills the raw feature dict into deterministic,
auditable claims with a YAML rule book and no model in the loop. Both
streams flow into the context assembler, and out the far side comes a
bundle of grounded material with the facts the rewrite must not
contradict.

What we have *not* done is write the sentence. The grounded facts still
have to be turned into fluent, customer-ready Korean by an actual
language model — and in a closed network, that means standing up the
model ourselves. The next post crosses into the serving side:
**SERVE-1 — Qwen on vLLM**, the OpenAI-compatible endpoint, batched
generation over millions of customers, JSON-mode constraints, and the
latency/throughput trade-offs of a batch-only reason pipeline. The
dictionary is built; next we hire the writer.
