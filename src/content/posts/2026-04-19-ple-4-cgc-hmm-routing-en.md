---
title: "[Study Thread] PLE-4 — Two CGC Gate Variants and HMM Triple-Mode Routing"
date: 2026-04-19 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, hmm, regularization]
lang: en
series: study-thread
part: 4
alt_lang: /2026/04/19/ple-4-cgc-hmm-routing-ko/
next_title: "PLE-5 — GroupTaskExpertBasket, Logit Transfer, Task Tower"
next_desc: "How GroupTaskExpertBasket v3.2 (GroupEncoder + ClusterEmbedding) produces per-task specialized experts, the three modes of Logit Transfer for explicit cross-task information flow, and the Task Tower architecture that turns it all into final predictions."
next_status: published
---

*PLE-4 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document. This fourth post
covers the heart of PLE — its two CGC gate variants, the original
paper's CGCLayer (weighted sum) and this implementation's CGCAttention
(block scaling) — along with the regularization that keeps Expert
Collapse away, and the HMM-based Triple-Mode routing that sits beside
the gate.*

## CGC (Customized Gate Control) — Per-Task Expert Weighting

### Theoretical background

CGC extends the gating mechanism of MMoE (Ma et al., KDD 2018). A
per-task independent gate learns task-specific affinities by applying
different weights to the Shared Expert outputs. The original PLE paper
(Tang et al., RecSys 2020) defines **CGCLayer**, which mixes expert
outputs as a *vector weighted sum*. This project's implementation uses
**CGCAttention**, which instead *scales each expert output block* by a
scalar weight. This fourth post walks through the difference between
the two variants, the entropy and dimension regularizers layered on top
of CGCAttention, and the HMM Triple-Mode routing that rides next to it.

### Variant 1 — CGCLayer: vector weighted sum (original PLE paper)

The original paper's CGCLayer follows the MMoE gate recipe directly. A
per-task gate produces a weighted sum over the expert outputs.

$$\mathbf{h}_k^{cgc} = \sum_{i=1}^N g_{k,i} \cdot \mathbf{h}_i^{expert}, \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{h}_{shared})$$

- **Prerequisite**: every expert must produce the *same output dimension*.
  The weighted sum only makes sense if all $\mathbf{h}_i^{expert}$ live
  in the same $\mathbb{R}^d$.
- **Strength**: a clean equation; each task can express a natural mixture
  like "Expert A 70%, Expert B 30%."
- **Limitation**: it does not fit a setup like this project where expert
  dimensions are *heterogeneous* (unified_hgcn 128D, and the rest —
  perslay, temporal, deepfm, lightgcn, etc. — at 64D). Forcing a common
  dimension means taking an information loss to make the math work.

### Variant 2 — CGCAttention: block scaling (this implementation)

`_build_cgc()` (lines 566–677) manages per-task independent
`nn.Sequential(Linear + Softmax)` modules inside an `nn.ModuleDict`.

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^7$$

$$\tilde{\mathbf{h}}_{k,i} = w_{k,i} \cdot \mathbf{h}_i^{expert} \quad \text{for } i = 1, \ldots, 7$$

$$\mathbf{h}_k^{cgc} = [\tilde{\mathbf{h}}_{k,1} \,\|\, \tilde{\mathbf{h}}_{k,2} \,\|\, \ldots \,\|\, \tilde{\mathbf{h}}_{k,7}] \in \mathbb{R}^{512}$$

Here $\mathbf{W}_k \in \mathbb{R}^{7 \times 512}$ is task $k$'s gate
weight, $\mathbf{h}_i^{expert}$ is the $i$-th expert's output block (64D
or 128D), and $w_{k,i}$ is the attention weight that task $k$ assigns
to expert $i$.

> **Equation intuition.** The first equation takes the 512D shared
> representation, produces a "relevance score" for each of the 7 experts,
> and runs those through a Softmax to turn them into probabilities. The
> second equation multiplies each expert block by its scalar probability
> to modulate its importance. The third equation concatenates the scaled
> blocks back into a vector of the same 512D shape. The net effect: the
> same 512D vector flows out, but every task gets a different mixture of
> contributions from each expert.

> **Dimension-preserving design.** CGCAttention maps a 512D input to a
> 512D output. Because it multiplies each expert block by a scalar
> (*block scaling*), it is backward-compatible with the rest of the
> pipeline. The weights sum to 1 (Softmax), so the output scale is
> preserved.

> **Historical context.** CGC (Customized Gate Control) was first named
> in the PLE paper (Tang et al., RecSys 2020) as an extension of the
> MMoE gate. Where MMoE (Ma et al., KDD 2018) treats every expert
> symmetrically, CGC separates the gating over Shared Experts from that
> over Task-specific Experts. The connection to attention mechanisms is
> inspired by the Scaled Dot-Product Attention in Transformer (Vaswani
> et al., NeurIPS 2017): the same principle — *selectively combine
> information in proportion to relevance* — applied across tokens
> (Transformer), across experts (CGC), and across heads (Multi-Head
> Attention). Unlike the original paper, our CGC adds block scaling and
> dimension normalization to cope with heterogeneous expert dimensions.

### Initial bias — domain_experts

`_build_cgc()` (lines 621–649) reads each task's `domain_experts`
config field and sets the initial bias accordingly. Weight is
initialized to zero, a task's "preferred" experts get
`bias_high = 1.0`, and the rest get `bias_low = -1.0`. This is a soft
prior that makes the initial Softmax output line up with domain
knowledge before a single gradient step.

```python
# ple_cluster_adatt.py:626-638
bias_high = float(cgc_config.get("bias_high", 1.0))
bias_low = float(cgc_config.get("bias_low", -1.0))
linear_layer.weight.zero_()          # weight starts at 0
for i, expert_name in enumerate(expert_names):
    if expert_name in domain_experts:
        linear_layer.bias[i] = bias_high   # preferred expert
    else:
        linear_layer.bias[i] = bias_low    # non-preferred
```

| Task | domain_experts (bias_high=1.0) |
|---|---|
| CTR | `perslay`, `temporal`, `unified_hgcn` |
| CVR | `perslay`, `temporal`, `unified_hgcn` |
| Churn | `perslay`, `temporal` |
| Retention | `perslay`, `temporal` |
| NBA | `perslay`, `unified_hgcn`, `lightgcn` |
| Life-stage | `perslay`, `temporal` |
| Balance_util | `temporal` |
| Engagement | `temporal` |
| LTV | `temporal`, `deepfm` |
| Channel | `temporal` |
| Timing | `temporal` |
| Spending_category | `unified_hgcn`, `perslay` |
| Consumption_cycle | `temporal` |
| Spending_bucket | `deepfm` |
| Brand_prediction | `unified_hgcn` |
| Merchant_affinity | `unified_hgcn`, `temporal` |

### Entropy regularization (v2.3) — keeping Expert Collapse away

`_cgc_entropy_regularization()` (lines 748–768) maximizes the entropy
of the CGC attention distribution to prevent **Expert Collapse**.

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left( -\frac{1}{|\mathcal{T}|} \right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k)$$

$$H(\mathbf{w}_k) = -\sum_{i=1}^{7} w_{k,i} \cdot \log(w_{k,i})$$

Here $\mathcal{T}$ is the set of active tasks and
$\lambda_{ent} = 0.01$ (config key `cgc.entropy_lambda`). *Minimizing*
negative entropy is the same as *maximizing* entropy, which spreads the
distribution out.

> **Equation intuition.** Entropy $H$ is a measure of "how evenly a
> gate distribution is spread." If all the weight piles onto a single
> expert, $H$ is small; if the weights are uniform, $H$ is large. This
> loss term minimizes $-H$, pushing $H$ upward — and so the gate is
> encouraged to actually reference multiple experts. A larger
> $\lambda_{ent}$ applies a stronger even-spreading pressure.

> **Undergraduate math — entropy from information theory.**
> Shannon (1948) derived "a measure of uncertainty" axiomatically. Given
> a probability distribution $\mathbf{w} = (w_1, \ldots, w_n)$, entropy
> is the unique function satisfying three axioms: (1) continuity — a
> small change in $w_i$ produces a small change in $H$; (2) maximality
> — $H$ is largest on the uniform distribution; (3) additivity — the
> entropy of independent events adds. *A concrete calculation*: for 7
> experts on a uniform $w_i = 1/7$,
> $H = -7 \times (1/7) \times \log(1/7) = \log(7) \approx 1.946$ bits
> (maximum entropy). Fully concentrated on one expert,
> $\mathbf{w} = (1, 0, \ldots, 0)$, gives
> $H = -1 \times \log(1) = 0$ (minimum entropy). For
> $\mathbf{w} = (0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06)$,
> $H \approx 1.32$ — about 68% of the maximum. Entropy regularization
> pushes this number back toward the maximum and spreads expert
> utilization.

> **⚠ Expert Collapse risk.** If CGC entropy lambda is 0, the
> regularizer is disabled. Attention then tends to concentrate on one
> expert during training (typically the 128D unified_hgcn), and the
> gradients of the remaining experts vanish. The default
> `entropy_lambda=0.01` works; empirically the stable range is
> 0.005–0.02.

### CGC Attention in the forward pass

`_apply_cgc_attention()` (lines 679–725) scales each per-expert block
by its attention weight.

```python
# ple_cluster_adatt.py:708-725
parts = []
offset = 0
for i, dim in enumerate(self._cgc_expert_dims):
    block = shared_concat[:, offset:offset + dim]
    # v3.3: dimension normalization — 128D experts are attenuated, 64D amplified
    if self._cgc_dim_normalize and dim != self._cgc_mean_dim:
        scale = math.sqrt(self._cgc_mean_dim / dim)
        block = block * scale
    part = block * attention_weights[:, i:i+1]  # broadcast
    parts.append(part)
    offset += dim
return torch.cat(parts, dim=-1)
```

### Dimension normalization (v3.3) — fixing heterogeneous experts

When `dim_normalize=true`, a scaling step corrects the contribution
imbalance caused by the asymmetric expert output dimensions (128D vs
64D).

$$\text{scale}_i = \sqrt{\frac{\text{mean\_dim}}{\text{dim}_i}}$$

$$\text{mean\_dim} = \frac{128 + 64 \times 6}{7} \approx 73.14$$

- unified_hgcn (128D): scale $= \sqrt{73.14 / 128} \approx 0.756$ (attenuated)
- other experts (64D): scale $= \sqrt{73.14 / 64} \approx 1.069$ (amplified)
- same attention $\Rightarrow$ same L2 contribution

> **Equation intuition.** unified_hgcn (128D) has twice the output
> dimension of the other experts (64D), so even under an identical
> attention weight its L2-norm contribution is disproportionately
> large. This scaling "shrinks large-dimension experts and grows
> small-dimension experts" so that, when attention is uniform at
> $w_{k,i} \approx 0.143$ ($1/7$), every expert actually contributes the same.

### CGC freeze synchronization

`on_epoch_end()` (lines 1921–1942) freezes the CGC attention parameters
once the adaTT `freeze_epoch` is reached.

```python
# ple_cluster_adatt.py:1934-1942
if (freeze_epoch is not None
        and epoch >= freeze_epoch
        and not self._cgc_frozen.item()):
    for param in self.task_expert_attention.parameters():
        param.requires_grad = False
    self._cgc_frozen.fill_(True)
```

> **Why sync CGC and adaTT.** If adaTT freezes its transfer weights but
> CGC keeps training, the two mechanisms can drift into conflicting
> directions. A simultaneous freeze keeps the late phase of training
> stable.

## HMM Triple-Mode Routing

### Three HMM modes

HMM Triple-Mode (v2.0) separates customer behavior along three
time-scales and injects the most relevant behavioral mode per task
group.

| Mode | Input | Time scale | Target tasks |
|---|---|---|---|
| Journey | 16D | daily | CTR, CVR, Engagement, Uplift |
| Lifecycle | 16D | monthly | Churn, Retention, Life-stage, LTV |
| Behavior | 16D | monthly | NBA, Balance_util, Channel, Timing, Spending_category, Consumption_cycle, Spending_bucket, Merchant_affinity, Brand_prediction |

Each mode is made of a 10D base state probability plus a 6D ODE
dynamics bridge.

> **Historical context — origin and evolution of the Hidden Markov
> Model.** The HMM was formalized by *Baum & Petrie (1966)* for
> statistical language modeling. The core idea is that "behind every
> observable event there is an unobservable hidden state, and transitions
> between states follow the Markov property." In the 1970s,
> *Rabiner & Juang* applied it systematically to speech recognition and
> popularized it; it then spread to bioinformatics (gene sequence
> analysis), finance (market regime estimation), NLP (POS tagging), and
> beyond. In this system, we estimate hidden "journey states,"
> "lifecycle states," and "behavior pattern states" behind customers'
> observable actions (transactions, logins), and inject the state at
> the most appropriate time-scale for each task. The ODE dynamics bridge
> is inspired by *Neural ODE (Chen et al., NeurIPS 2018)*; it extends
> the discrete HMM states into continuous time by interpolation.

### HMM projector architecture

`_build_hmm_projectors()` (lines 452–496) creates a projector per mode.

$$\mathbf{h}_{hmm}^m = \text{SiLU}(\text{LayerNorm}(\text{Linear}_{16 \to 32}(\mathbf{x}_{hmm}^m)))$$

Here $m \in \{\text{journey}, \text{lifecycle}, \text{behavior}\}$, and
each projector trains independently.

> **Equation intuition.** This equation takes the 16-dim state vector
> coming out of the HMM (10D state probability + 6D ODE dynamics) and
> lifts it to a 32-dim representation that the rest of the model can
> use. `Linear` expands the dimension, `LayerNorm` stabilizes the scale,
> and `SiLU` adds nonlinearity. Because each of the three modes
> (journey/lifecycle/behavior) gets its own projector, "daily journey
> patterns" and "monthly lifecycle patterns" end up learning distinct
> transforms.

> **Undergraduate math — SiLU and why nonlinearity matters.**
> *SiLU (Sigmoid Linear Unit)* is defined as
> $\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$.
> The sigmoid $\sigma(x)$ squashes input to $[0, 1]$ and acts as a
> "smooth switch." SiLU multiplies $x$ by this switch, which
> approximately lets positive inputs through unchanged while smoothly
> suppressing negative inputs. *Why is nonlinearity needed?* Stacking
> only linear maps collapses:
> $\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x}$
> is equivalent to a single linear map. Inserting a nonlinear function
> between layers is what gives the stack expressive power. *Comparison
> with other activations*: ReLU ($\max(0, x)$) has zero gradient for
> $x < 0$, causing "dead neuron" problems. GELU ($x \cdot \Phi(x)$ with
> the Gaussian CDF) is similar to SiLU but more expensive to compute.
> SiLU strikes a balance between ReLU's cheapness and GELU's smoothness,
> and together with Mish ($x \cdot \tanh(\text{Softplus}(x))$) has
> become a standard activation since 2020.

```python
# ple_cluster_adatt.py:482-486
self.hmm_projectors[mode] = nn.Sequential(
    nn.Linear(hmm_dim, proj_dim),     # 16 → 32
    nn.LayerNorm(proj_dim),
    nn.SiLU(),
)
```

### Learnable default embedding

For samples with no HMM features (an all-zero row), the model uses a
*learnable default embedding* instead of a zero vector (lines 488–493).

```python
# ple_cluster_adatt.py:488-493
self.hmm_default_embeddings = nn.ParameterDict({
    mode: nn.Parameter(torch.zeros(proj_dim))
    for mode in ["journey", "lifecycle", "behavior"]
})
```

`_forward_hmm_projectors()` (lines 1365–1414) masks on a per-sample
basis, projecting only valid samples and substituting the default
embedding for invalid ones. Journey (16D), Lifecycle (16D), and
Behavior (16D) each pass through an independent 16→32D projector and
are then routed to their task groups — CTR/CVR/Engagement,
Churn/Retention/Life-stage/LTV, and NBA/Balance/Channel/... respectively.

## Where this leads

CGCAttention is the mechanism that "pulls a different mixture out of a
shared representation for every task," and HMM Triple-Mode is the
mechanism that "routes behavioral signals of different time-scales to
the right task group." Both run on top of a *shared expert pool*. The
next post, **PLE-5**, moves in the opposite direction — it looks at
how the model builds *dedicated expert baskets per task* via
GroupTaskExpertBasket v3.2 (GroupEncoder + ClusterEmbedding), explicit
cross-task information flow through the three modes of Logit Transfer,
and the Task Tower that turns all of this into final predictions.
