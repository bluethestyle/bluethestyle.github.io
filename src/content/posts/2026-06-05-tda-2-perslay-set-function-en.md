---
title: "[Study Thread] TDA-2 — PersLay as a Set Function: φ, w, ρ and the 5-Block Architecture"
date: 2026-06-05 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, perslay, deepsets, set-function, expert]
lang: en
excerpt: "The TDA sub-thread closes — how a variable-size, unordered set of (birth, death) points becomes one fixed 64D vector. The addition trick from DeepSets, the RationalHat point transform, the persistence weight that ignores noise for free, the aggregation choice, and why the project runs five independent PersLay blocks instead of one. With an honest footnote on which path actually runs in production."
series: study-thread
part: 12
alt_lang: /2026/06/05/tda-2-perslay-set-function-ko/
next_title: "DEEPFM-1 — Feature Interaction: Factorization Machines and the Shared-Embedding Trick"
next_desc: "The Expert sub-threads continue with DeepFM — why linear models cannot see feature interactions, how a factorization machine compresses 207,046 pairwise weights into 10,304 shared embedding parameters, and how one embedding feeds both the FM head and the deep tower."
next_status: published
---

*Second and final post of the TDA / PersLay sub-thread in the "Study
Thread" series. The source is the on-prem reference
`기술참조서/PersLay_기술_참조서`. TDA-1 made the case — spending has a
shape, persistence diagrams capture it, and a validation run said the
signal is real. What it deferred was the machine. This post opens it:
how exactly does an unordered, variable-size set of (birth, death)
points become one fixed 64-dimensional vector a neural network can
consume? The answer is a single equation read three times — once for
each learnable piece — and then multiplied by five.*

> **Where we are.** TDA-1 ended at the door: persistence diagrams are
> the right summary of spending shape, but a diagram is not a vector,
> and a network cannot eat it raw. By the end of this post the bridge
> equation $F(D) = \rho(\sum w \cdot \phi)$ should read not as
> notation but as three separate design decisions — *how to translate
> a point* ($\phi$), *how much to trust it* ($w$), *how to combine
> the points into one vector* ($\rho$) — each one learnable, each one
> chosen for a reason this project can name.

## The Problem, Stated Precisely

One more time, with precision. A persistence diagram is a set of
$(b, d)$ points, and it resists a neural network on three counts:

1. **Variable size.** One customer's diagram has 12 points, another's
   has 47. An MLP input layer has a fixed width. There is no slot
   arrangement that fits both.
2. **No order.** The points are a bag, not a sequence. Flattening
   them into a vector *invents* an order — and then the same diagram,
   flattened in a different order, produces a different vector. A
   model trained on that representation learns the artifact, not the
   shape.
3. **The wrong space.** Diagrams live in a metric space measured by
   bottleneck and Wasserstein distances, not in $\mathbb{R}^n$.
   Nothing about a standard layer respects that geometry.

Any fix must produce a **fixed-length** output, be **indifferent to
order**, and remain **differentiable** so the recommendation loss can
reach back into it.

## The One Trick: Addition Does Not Care About Order

The escape hatch is almost embarrassingly simple. Addition is
commutative:

$$ \phi(p_1) + \phi(p_2) + \phi(p_3) \;=\; \phi(p_3) + \phi(p_1) + \phi(p_2) $$

Transform each point *independently* with the same function $\phi$,
then **sum the results**. The sum does not know what order the points
arrived in — order-invariance, solved. And whether you sum 12 vectors
or 47, the result is one vector of the same width — variable size,
solved. Both obstacles fall to one arithmetic fact.

This is the DeepSets recipe (Zaheer et al., 2017), and it is not just
a convenient hack:

$$ F(X) = \rho\!\left( \sum_{x \in X} \phi(x) \right) $$

> **Historical context.** Zaheer et al. proved a representation
> theorem: *every* continuous permutation-invariant function on a
> countable set decomposes into exactly this sum-of-transforms shape —
> a universal-approximation result for sets. PersLay (Carrière et
> al., JMLR 2020) is DeepSets specialized to persistence diagrams,
> with one addition: a per-point weight $w$ that encodes what TDA
> knows about which points matter. Without the theorem, the
> architecture would be a heuristic; with it, summing is not a
> compromise — it is the canonical form.

PersLay's full equation just inserts that weight:

$$ F(D) = \rho\!\left( \sum_{(b,d)\in D} w(b,d)\,\cdot\,\phi(b,d) \right) $$

> **Equation intuition.** Think of a stack of receipts. $D$ is the
> stack — different customers have different counts, and the stack has
> no meaningful order. $\phi$ is a clerk who reads *one receipt at a
> time* and fills in a fixed form. $w$ is the clerk's judgment call —
> "this receipt is informative, that one is junk" — a number between
> 0 and large. $\rho$ staples all the weighted forms into one
> fixed-format report. However many receipts come in, whatever order
> they sit in, the report always has the same fields.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <!-- three point lanes -->
  <g font-size="10">
    <rect x="18" y="30" width="86" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
    <text x="61" y="51" text-anchor="middle" fill="#4f46e5" font-weight="700">(0.2, 0.9)</text>
    <rect x="18" y="100" width="86" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
    <text x="61" y="121" text-anchor="middle" fill="#4f46e5" font-weight="700">(0.5, 0.55)</text>
    <rect x="18" y="170" width="86" height="34" rx="5" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="4 3"/>
    <text x="61" y="191" text-anchor="middle" fill="#94a3b8" font-weight="700">(0, 0) pad</text>
  </g>
  <!-- phi boxes -->
  <g font-size="10">
    <rect x="150" y="30" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="51" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <rect x="150" y="100" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="121" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <rect x="150" y="170" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="191" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <text x="182" y="22" text-anchor="middle" fill="#64748b" font-size="9">same weights, per point</text>
  </g>
  <!-- w multipliers -->
  <g font-size="10">
    <rect x="260" y="30" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706"/>
    <text x="303" y="51" text-anchor="middle" fill="#d97706" font-weight="700">× w = 0.70</text>
    <rect x="260" y="100" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706"/>
    <text x="303" y="121" text-anchor="middle" fill="#d97706" font-weight="700">× w = 0.05</text>
    <rect x="260" y="170" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706" stroke-dasharray="4 3"/>
    <text x="303" y="191" text-anchor="middle" fill="#94a3b8" font-weight="700">× w = 0</text>
    <text x="303" y="22" text-anchor="middle" fill="#64748b" font-size="9">w = |d − b|</text>
  </g>
  <!-- sum -->
  <rect x="400" y="95" width="70" height="44" rx="6" fill="#f1f5f9" stroke="#1e3a5f"/>
  <text x="435" y="122" text-anchor="middle" font-size="16" font-weight="700" fill="#1e3a5f">Σ</text>
  <!-- output -->
  <rect x="520" y="93" width="80" height="48" rx="6" fill="#0d9488"/>
  <text x="560" y="113" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">one vector</text>
  <text x="560" y="129" text-anchor="middle" font-size="9" fill="#fff">same width, any n</text>
  <!-- arrows -->
  <g stroke="#cbd5e1" stroke-width="1.3" fill="#cbd5e1">
    <line x1="104" y1="47" x2="148" y2="47"/><polygon points="148,47 140,43 140,51"/>
    <line x1="104" y1="117" x2="148" y2="117"/><polygon points="148,117 140,113 140,121"/>
    <line x1="104" y1="187" x2="148" y2="187"/><polygon points="148,187 140,183 140,191"/>
    <line x1="214" y1="47" x2="258" y2="47"/><polygon points="258,47 250,43 250,51"/>
    <line x1="214" y1="117" x2="258" y2="117"/><polygon points="258,117 250,113 250,121"/>
    <line x1="214" y1="187" x2="258" y2="187"/><polygon points="258,187 250,183 250,191"/>
    <line x1="346" y1="47" x2="398" y2="105"/><polygon points="398,105 389,103 394,96"/>
    <line x1="346" y1="117" x2="398" y2="117"/><polygon points="398,117 390,113 390,121"/>
    <line x1="346" y1="187" x2="398" y2="129"/><polygon points="398,129 393,138 388,131"/>
    <line x1="470" y1="117" x2="518" y2="117"/><polygon points="518,117 510,113 510,121"/>
  </g>
  <text x="310" y="237" text-anchor="middle" font-size="10" fill="#64748b">order of the three lanes is irrelevant — the sum is identical</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The DeepSets skeleton with PersLay's weight. Each point is transformed independently (φ), scaled by its importance (w), and summed. Padding points get weight 0 and vanish from the sum.</figcaption>
</figure>

Now the three pieces, one at a time — each as this project actually
configures it.

## φ — Translating One Point

$\phi$ answers: *what does a single $(b, d)$ point become?* The
production choice, `RationalHatPhi`, refuses to make the network
discover arithmetic it can be handed for free. It first expands the
2-D point into six hand-built views:

$$ \phi_{\text{hat}}(b, d) = W_2\,\mathrm{ReLU}\!\big(W_1\,[\,b,\ d,\ d-b,\ \tfrac{b+d}{2},\ b \cdot d,\ \tfrac{d}{b+\epsilon}\,] + \mathbf{b}_1\big) + \mathbf{b}_2 $$

| # | Feature | What it asks of the point |
| --- | --- | --- |
| 0 | $b$ (birth) | At what scale does this structure appear? |
| 1 | $d$ (death) | Up to what scale does it survive? |
| 2 | $d-b$ (persistence) | How long does it live — real structure or noise? |
| 3 | $(b+d)/2$ (midpoint) | Around which scale is it centered? |
| 4 | $b \cdot d$ (product) | Born large *and* dying large — a macro structure? |
| 5 | $d/(b+\epsilon)$ (ratio) | How long does it live *relative to* its birth scale? |

Follow one concrete point through. For $(b, d) = (0.2,\ 0.9)$ the six
views are plain arithmetic:

$$ [\,0.2,\quad 0.9,\quad 0.7,\quad 0.55,\quad 0.18,\quad 4.5\,] $$

In order: birth, death, persistence ($0.9-0.2$), midpoint
($(0.2+0.9)/2$), product ($0.2 \times 0.9$), ratio ($0.9/0.2$). One
point went from 2 numbers to 6, and nothing has been *learned* yet.

These six enter the 2-layer MLP — a small network that takes 6 numbers
in and emits 64 numbers out, and those 64 are the point's final
translation. Think of them as *a description of this point's
topological character from 64 angles*: training tunes some slots to
react strongly to "did it live long," others to "this particular mix
of midpoint and birth."

Why hand-build the six views at all? That persistence $d-b$ matters
is TDA common knowledge. Hand the MLP only $b$ and $d$, and it must
first discover from data that *subtraction is useful*. Pre-computing
the views skips that discovery entirely, so the learning capacity is
spent only on *which combination matters for which task* — churn
prediction may lean on persistence and ratio, CTR on midpoint and
birth, and nobody has to decide that by hand.

> **The alternative: GaussianPhi.** The reference also implements a
> second transform — place $K = 16$ learnable Gaussian "detectors"
> $\mu_k$ with bandwidths $\sigma_k$ on the diagram plane, and emit
> each point's activation against all sixteen:
> $\phi_{\text{gauss}}(p) = W_{\text{proj}}\,[\,e^{-\lVert p-\mu_1\rVert^2/2\sigma_1^2},\ \dots,\ e^{-\lVert p-\mu_K\rVert^2/2\sigma_K^2}\,]$.
> It is kernel density estimation with trainable sensors: training
> drags the $\mu_k$ toward the high-persistence regions that matter.
> One engineering scar worth recording: under fp16 mixed precision
> the $\exp$ overflows, so the implementation casts the bandwidths to
> float32 and clamps the exponent to $[-10, 10]$ — without which the
> forward pass emits `inf` on day one. Production uses RationalHat,
> which has no such hazard.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="200" fill="#f8fafc" rx="8"/>
  <rect x="22" y="78" width="80" height="44" rx="6" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="62" y="97" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">(b, d)</text>
  <text x="62" y="112" text-anchor="middle" font-size="9" fill="#64748b">2 numbers</text>
  <!-- six expansion chips -->
  <g font-size="9.5" font-weight="700">
    <rect x="150" y="18" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="33" text-anchor="middle" fill="#0d9488">b</text>
    <rect x="150" y="46" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="61" text-anchor="middle" fill="#0d9488">d</text>
    <rect x="150" y="74" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="89" text-anchor="middle" fill="#0d9488">d − b</text>
    <rect x="150" y="102" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="117" text-anchor="middle" fill="#0d9488">(b+d)/2</text>
    <rect x="150" y="130" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="145" text-anchor="middle" fill="#0d9488">b·d</text>
    <rect x="150" y="158" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="173" text-anchor="middle" fill="#0d9488">d/(b+ε)</text>
  </g>
  <text x="196" y="12" text-anchor="middle" font-size="9" fill="#64748b">6 fixed views</text>
  <rect x="300" y="70" width="120" height="58" rx="6" fill="#fffbeb" stroke="#d97706"/>
  <text x="360" y="94" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">2-layer MLP</text>
  <text x="360" y="110" text-anchor="middle" font-size="9" fill="#64748b">learned mixing</text>
  <rect x="468" y="76" width="72" height="46" rx="6" fill="#0d9488"/>
  <text x="504" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">64D</text>
  <text x="504" y="111" text-anchor="middle" font-size="9" fill="#fff">per point</text>
  <g stroke="#cbd5e1" stroke-width="1.2" fill="#cbd5e1">
    <line x1="102" y1="100" x2="146" y2="100"/><polygon points="146,100 138,96 138,104"/>
    <line x1="242" y1="29" x2="298" y2="84"/><line x1="242" y1="57" x2="298" y2="92"/><line x1="242" y1="85" x2="298" y2="97"/>
    <line x1="242" y1="113" x2="298" y2="102"/><line x1="242" y1="141" x2="298" y2="108"/><line x1="242" y1="169" x2="298" y2="115"/>
    <line x1="420" y1="99" x2="466" y2="99"/><polygon points="466,99 458,95 458,103"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">RationalHatPhi: hand-expand one point into six interpretable views, then let a small MLP learn the task-specific mixture.</figcaption>
</figure>

## w — Scoring Each Point's Importance

$w$ answers: *how much should this point count?* Its output is one
number per point — the multiplier that will scale that point's $\phi$
vector. Three modes are implemented; production runs the first.

| Mode | Formula | Character |
| --- | --- | --- |
| `persistence` (production) | $w(b,d) = \lvert d-b \rvert^{p}$, $p = 1.0$ | longer-lived structures weigh more; diagonal points weigh zero |
| `linear` | $w(b,d) = 1$ | every point equal — a uniform baseline |
| `learned` | $w = \mathrm{Softplus}(\mathrm{MLP}([b,d]))$ | weight itself trained; Softplus keeps it non-negative |

Unpack the production formula first. $|d-b|$ is the point's
persistence — the "how long did it live" from TDA-1 — itself. $p$ is
just an exponent on top: at $p=1$ the weight is *directly
proportional* to lifetime (a point that lived 0.7 counts 0.7, a point
that lived 0.05 counts 0.05). Raise $p$ to 2 and long-lived points
are favored quadratically, widening the gap; lower it to 0.5 and the
gap softens. Production uses the simplest choice, direct proportion,
$p = 1.0$.

The formula ends at one absolute value, yet that simple choice buys
three things at once:

- **Noise suppression for free.** TDA-1's reading of the diagram —
  far from the diagonal means real, near means noise — becomes a
  *soft, continuous* version of itself. No threshold to tune; near-
  diagonal points just fade.
- **Padding for free.** Diagrams are padded to a fixed `max_pairs`
  per batch with $(0,0)$ entries. Their persistence is $|0-0| = 0$,
  so their weight is 0 — padding vanishes from the sum with **no mask
  arithmetic at all**.
- **Gradients undistorted.** With $p = 1$ and $d > b$ always,
  $\partial w / \partial d = 1$: the weight scales contributions
  without bending the gradient field. (Padding points get zero
  gradient through $w = 0$ — which is exactly the intent.)

## ρ — Combining the Points into One Vector

$\rho$ answers: *how do the several weighted point-vectors merge into
a single vector?* Four permutation-invariant options are implemented:

| Mode | Reads as | Cost |
| --- | --- | --- |
| `sum` (production) | the *total* topological structure | $O(n)$ |
| `mean` | the *average* structure, size-normalized | $O(n)$ |
| `max` | the single *most prominent* structure | $O(n)$ |
| `attention` | a *learned* focus over structures | $O(n^2)$ |

> **Equation intuition.** Sum grows with how much structure a
> customer has; mean asks what the typical structure looks like
> regardless of count; max bets everything on the strongest pattern;
> attention ($\alpha_i = \mathrm{softmax}(W_2 \tanh(W_1 \mathbf{x}_i))$)
> learns where to look — the most expressive and the most expensive.

Production runs `sum`, and the reference is blunt about why the
config was switched from attention: at `max_pairs = 200`, attention
materializes a $200 \times 200$ matrix per diagram per batch — real
VRAM, real latency — while the persistence weight *already* encodes
per-point importance. Paying quadratic cost to relearn what $w$
provides linearly was a bad trade; the swap to `sum` took the cost
from $O(n^2)$ to $O(n)$ with no quality regression worth the bill.

## One Pass by Hand

Run the three pieces on the smallest possible example. Some
customer's diagram $D$ holds three points — two real ones and one
batch-padding entry.

**Step 1 — compute each point's weight $w$.** The formula is just
$w = |d-b|$.

- $(0.2,\ 0.9)$ : $w = |0.9-0.2| = 0.70$ — a long-lived, real
  structure.
- $(0.5,\ 0.55)$ : $w = |0.55-0.5| = 0.05$ — right next to the
  diagonal, close to noise.
- $(0,\ 0)$ pad : $w = |0-0| = 0$ — its dismissal is already booked.

**Step 2 — translate each point with $\phi$.** The RationalHat of the
previous section turns each point into 64 numbers; to keep this
traceable by hand, pretend it emits 3. The $\phi$ values in the table
are illustrative numbers — what matters is the *shape*: one
same-length vector per point.

**Step 3 — multiply by the weight, then add everything.**
$w \cdot \phi$ multiplies one number (the weight) into *each slot* of
the vector, and the final sum adds *slot by slot*.

| Point | $w = \lvert d-b \rvert$ | $\phi(b,d)$ (illustrative) | $w \cdot \phi$ |
| --- | --- | --- | --- |
| $(0.2,\ 0.9)$ | $0.70$ | $[\,0.4,\ -1.1,\ 0.8\,]$ | $[\,0.28,\ -0.77,\ 0.56\,]$ |
| $(0.5,\ 0.55)$ | $0.05$ | $[\,1.2,\ 0.3,\ -0.5\,]$ | $[\,0.06,\ 0.015,\ -0.025\,]$ |
| $(0,\ 0)$ pad | $0$ | $[\,0.9,\ 0.2,\ 0.1\,]$ | $[\,0,\ 0,\ 0\,]$ |

Follow just the first slot: $0.28 + 0.06 + 0 = 0.34$. The other slots
add the same way:

$$ F = [\,0.34,\ -0.755,\ 0.535\,] $$

Every promised property is sitting inside this arithmetic.

- **Noise suppression** — the long-lived point ($w=0.70$) dominates
  the result; the near-diagonal one ($w=0.05$) leaves only a trace.
- **Padding ignored** — the third row zeroed out the moment it was
  multiplied. Gone, without a mask.
- **Order invariance** — add the rows in any order; the sum is the
  same.
- **Fixed width** — 3 points or 300, the output is one vector. Three
  slots here, 64 in the real layer.

## Five Blocks — Why Not One PersLay

A single PersLay layer would feed every point of every diagram
through one $\phi$ and one $w$. The project instead runs **five
independent `PersLayBlock`s**, each with its own parameters:

```python
class PersLayBlock(nn.Module):
    def forward(self, points, mask=None):
        phi_out = self.phi(points)           # [B, max_pairs, 64]
        weights = self.weight_fn(points)     # [B, max_pairs, 1]
        return self.rho(phi_out, weights, mask)  # [B, 64]
```

The split is *time range × homology dimension*. Short-range diagrams
(90-day app logs, up to 200 pairs) carry a `beta_idx` channel and are
routed to a **Short $\beta_0$** and a **Short $\beta_1$** block;
long-range diagrams (12-month transactions, up to 150 pairs) feed
**Long $\beta_0$ / $\beta_1$ / $\beta_2$**. A point participates only
in its own block — the masks compose as *valid-pair mask AND beta
mask*, with the persistence weight acting as a third, implicit
filter.

> **Design rationale: why per-β parameters.** The $(b,d)$
> distributions are qualitatively different across dimensions. In
> $H_0$, every point is born at $b = 0$ — all components exist from
> the start — so the block's geometry is one-sided. In $H_1$, loops
> only form at positive scale, so $b > 0$ always. One shared $\phi$
> would have to straddle both regimes; five specialized blocks each
> learn their own. And a 90-day cluster split is a different *kind*
> of signal from a 12-month consumption void — separate parameters
> let each be itself.

The five 64-D block outputs concatenate with two side inputs and
compress:

| Component | Width | Content |
| --- | --- | --- |
| Short $\beta_0 + \beta_1$ | $64+64 = 128$D | short-term clusters and loops |
| Long $\beta_0 + \beta_1 + \beta_2$ | $64+64+64 = 192$D | long-term clusters, loops, voids |
| Global stats MLP | $30 \to 32$D | whole-diagram summaries (entropy, lifetime stats) the point-wise path can miss |
| Phase transition | $10$D | regime-change features, passed through |
| **Total → output** | $362 \to 64$D | `final_mlp`: Linear(362,128) → LayerNorm → SiLU → Dropout → Linear(128,64) → LayerNorm |

<figure style="margin:24px auto;max-width:640px;">
<svg viewBox="0 0 640 330" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="640" height="330" fill="#f8fafc" rx="8"/>
  <!-- inputs -->
  <rect x="30" y="22" width="150" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="105" y="36" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">short_diagrams</text>
  <text x="105" y="49" text-anchor="middle" font-size="8.5" fill="#64748b">[B, 200, 3] — 90-day app logs</text>
  <rect x="300" y="22" width="150" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
  <text x="375" y="36" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">long_diagrams</text>
  <text x="375" y="49" text-anchor="middle" font-size="8.5" fill="#64748b">[B, 150, 3] — 12-month txns</text>
  <!-- five blocks -->
  <g font-size="9.5" font-weight="700">
    <rect x="22" y="96" width="76" height="40" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="60" y="113" text-anchor="middle" fill="#4f46e5">Short β₀</text><text x="60" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="112" y="96" width="76" height="40" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="150" y="113" text-anchor="middle" fill="#4f46e5">Short β₁</text><text x="150" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="252" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="290" y="113" text-anchor="middle" fill="#0d9488">Long β₀</text><text x="290" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="342" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="380" y="113" text-anchor="middle" fill="#0d9488">Long β₁</text><text x="380" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="432" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="470" y="113" text-anchor="middle" fill="#0d9488">Long β₂</text><text x="470" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
  </g>
  <!-- side inputs -->
  <rect x="528" y="84" width="96" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/>
  <text x="576" y="97" text-anchor="middle" font-size="9" font-weight="700" fill="#d97706">global_stats</text>
  <text x="576" y="109" text-anchor="middle" font-size="8" fill="#64748b">[B, 30] → MLP → 32D</text>
  <rect x="528" y="122" width="96" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/>
  <text x="576" y="135" text-anchor="middle" font-size="9" font-weight="700" fill="#d97706">phase_transition</text>
  <text x="576" y="147" text-anchor="middle" font-size="8" fill="#64748b">[B, 10] passthrough</text>
  <!-- concat -->
  <rect x="170" y="196" width="300" height="36" rx="6" fill="#f1f5f9" stroke="#1e3a5f"/>
  <text x="320" y="211" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e3a5f">concat — 128 + 192 + 32 + 10 = 362D</text>
  <text x="320" y="226" text-anchor="middle" font-size="8.5" fill="#64748b">five blocks + two side inputs</text>
  <!-- final mlp -->
  <rect x="196" y="262" width="248" height="40" rx="6" fill="#0d9488"/>
  <text x="320" y="279" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">final_mlp — 362 → 128 → 64D</text>
  <text x="320" y="294" text-anchor="middle" font-size="8.5" fill="#fff">LayerNorm · SiLU · Dropout → PLE CGC gate</text>
  <!-- arrows -->
  <g stroke="#cbd5e1" stroke-width="1.2" fill="#cbd5e1">
    <line x1="80" y1="56" x2="62" y2="94"/><polygon points="62,94 61,86 69,89"/>
    <line x1="130" y1="56" x2="148" y2="94"/><polygon points="148,94 141,89 149,86"/>
    <line x1="345" y1="56" x2="292" y2="94"/><polygon points="292,94 294,86 300,91"/>
    <line x1="375" y1="56" x2="379" y2="94"/><polygon points="379,94 375,87 383,87"/>
    <line x1="405" y1="56" x2="468" y2="94"/><polygon points="468,94 460,92 465,85"/>
    <line x1="60" y1="136" x2="218" y2="195"/><line x1="150" y1="136" x2="252" y2="195"/>
    <line x1="290" y1="136" x2="300" y2="194"/><line x1="380" y1="136" x2="350" y2="194"/><line x1="470" y1="136" x2="400" y2="194"/>
    <line x1="576" y1="114" x2="576" y2="120"/>
    <line x1="560" y1="152" x2="462" y2="200"/><polygon points="462,200 470,196 472,203"/>
    <line x1="320" y1="232" x2="320" y2="260"/><polygon points="320,260 316,252 324,252"/>
  </g>
  <text x="60" y="76" text-anchor="middle" font-size="8.5" fill="#94a3b8">β₀ points</text>
  <text x="152" y="76" text-anchor="middle" font-size="8.5" fill="#94a3b8">β₁ points</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The 5-Block PersLayExpert. Each block owns its φ, w, ρ; the beta_idx channel routes each point to its block. 362D concatenated, compressed to the 64D the PLE gate expects.</figcaption>
</figure>

One quiet extra: alongside the 64D, the Expert emits a 4-D
*interpretable projection* — pattern stability, periodicity strength,
anomaly, complexity. It never touches the loss; it exists so a human
debugging the model (or explaining it to the business) can read four
named dials instead of sixty-four anonymous ones.

## An Honest Footnote — Which Path Actually Runs

Everything above is the paper-faithful design, and the code implements
all of it. But the reference is candid about the *current* state, and
this series quotes its sources honestly: in the live configuration the
raw-diagram path is **switched off** (`use_raw_diagram: false`, with
the raw-diagram Parquet not currently injected), so production
inference runs the **pre-computed fallback** — the 70-D offline
summary (24-D short + 36-D long + 10-D phase) through a 3-layer MLP,
$70 \to 64 \to 64 \to 64$. If even those features are missing, the
Expert degrades to a zero vector rather than crashing the batch.

The fallback trades per-point detail for robustness — aggregated
statistics cannot see what individual $(b,d)$ points encode. The
5-Block path is the destination; the stats MLP is what currently
ships. Worth one line of honesty rather than a footnote nobody reads.

## Paper vs Project

| Aspect | Carrière et al. (2020) | This project |
| --- | --- | --- |
| Blocks | one PersLay layer | five independent blocks (Short β₀/β₁ + Long β₀/β₁/β₂) |
| Input | a single diagram | Short/Long diagrams + 30D global stats + 10D phase |
| Output | variable | fixed 64D (PLE gate contract) |
| Post-processing | none | final_mlp (362→64) + 4D interpretable projection |
| Modes | raw diagrams only | dual: raw 5-Block + pre-computed 70D fallback |
| Padding | variable-size input | fixed max_pairs + persistence-weight auto-ignore |

## Where We Stop

The TDA sub-thread closes here. TDA-1 argued that spending has a
shape and verified the claim on real sessions; TDA-2 opened the
machine — one commutative operation dissolving the set-input problem,
three learnable pieces ($\phi$, $w$, $\rho$) each chosen with a
stated reason, five specialized blocks because short-term clusters
and long-term voids are different signals, and one honest footnote
about which path is live today. The offline side of this story — how
the 70-D summary features are actually extracted at batch time — gets
its own post later in the series (TDAFEAT-1).

Next, the Expert tour continues with a very different kind of
specialist: **DEEPFM-1**, on factorization machines — why a linear
model cannot see that "high food spend × late-night sessions" means
something neither signal means alone, and how a shared embedding
makes 200,000 pairwise weights collapse into 10,000.
