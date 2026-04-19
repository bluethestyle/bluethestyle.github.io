---
title: "[Study Thread] PLE-5 — GroupTaskExpertBasket, Logit Transfer, Task Tower"
date: 2026-04-19 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, logit-transfer, task-tower, group-encoder]
lang: en
series: study-thread
part: 5
alt_lang: /2026/04/19/ple-5-basket-logit-tower-ko/
next_title: "PLE-6 — Interpretability, Uncertainty, Full Specs (+ Tech Reference PDF)"
next_desc: "Expert interpretability via Sparse Autoencoder (SAE), uncertainty quantification via Evidential Deep Learning, the full 18-task spec, paper-vs-implementation comparison, debugging guide — and a downloadable PDF of the full PLE tech reference to close out the series."
next_status: published
---

*PLE-5 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document. This fifth post cuts
through the back half of the PLE data flow — GroupTaskExpertBasket v3.2,
which builds group-scoped expert encoders with cluster-conditioned
embeddings; Logit Transfer, which passes explicit predictions between
tasks along a DAG; and the Task Tower that produces the final output.*

## GroupTaskExpertBasket — GroupEncoder + ClusterEmbedding (v3.2)

### GroupTaskExpertBasket vs ClusterTaskExpertBasket

In v3.2, setting `use_group_encoder=true` (the default) switches to
`GroupTaskExpertBasket`, which achieves an *88% parameter reduction*
over the legacy `ClusterTaskExpertBasket`.

| Item | ClusterTaskExpertBasket (legacy) | GroupTaskExpertBasket (v3.2) |
| --- | --- | --- |
| Architecture | Independent MLP per task × cluster | Shared GroupEncoder + ClusterEmbedding |
| Parameters | ~3.0M | ~362K |
| Cluster specialization | Independent sub-head weights | Cluster-embedding injection |
| Generalization | Low (per-cluster overfit) | High (shared encoder) |

### GroupEncoder architecture

`_build_task_experts()` (lines 498–560) constructs the
`GroupTaskExpertBasket`.

```python
# ple_cluster_adatt.py:533-543
self.task_experts = GroupTaskExpertBasket(
    input_dim=task_expert_input_dim,     # 512 + 32 = 544
    group_hidden_dim=128,
    group_output_dim=64,
    cluster_embed_dim=32,
    subhead_output_dim=32,
    n_clusters=20,
    task_names=self.task_names,
    task_groups=task_groups,             # extracted from adaTT config
)
```

The single-task forward through a GroupEncoder is:

$$\mathbf{e}_{cluster} = \text{Embedding}(\text{cluster\_id}) \in \mathbb{R}^{32}$$

$$\mathbf{x}_{input} = [\text{CGC\_output}_{512D} \,\|\, \text{HMM\_proj}_{32D} \,\|\, \mathbf{e}_{cluster,32D}] \in \mathbb{R}^{576}$$

$$\mathbf{h}_{expert} = \text{MLP}_{576 \to 128 \to 64 \to 32}(\mathbf{x}_{input})$$

The actual input_dim is 544D (shared + HMM) + 32D (cluster_embed) =
576D. Tasks inside the same group share the GroupEncoder; groups
themselves remain independent.

> **Intuition.** This is how the model is told "which cluster this
> customer belongs to." First the cluster ID is mapped to a 32D
> embedding; this is concatenated with the CGC output (512D) and the
> HMM projection (32D) for a combined 576D input. A 3-stage MLP
> (576→128→64→32) then compresses that into a per-task 32D
> representation. Tasks in the same task group (e.g. CTR/CVR in the
> Engagement group) share a single GroupEncoder, saving parameters,
> while the cluster embedding still differentiates per-customer
> behavior.

> **Undergraduate math — what is an embedding?**
> $\text{Embedding}(\text{cluster\_id}) \in \mathbb{R}^{32}$ is a
> *lookup table* that maps an integer index to a continuous vector.
> Internally it stores a matrix
> $\mathbf{E} \in \mathbb{R}^{20 \times 32}$, and when
> cluster_id $= c$, it pulls out the $c$-th row
> $\mathbf{e}_c \in \mathbb{R}^{32}$. This is mathematically equivalent
> to *one-hot encoding + a linear transform*: with one-hot
> $\mathbf{v}_c \in \mathbb{R}^{20}$ (only the $c$-th entry is 1),
> $\mathbf{v}_c^T \mathbf{E} = \mathbf{e}_c$. One-hot requires sparse
> vector ops though, so direct indexing is faster. The key point is
> that $\mathbf{E}$ is a *trainable parameter*: as training progresses,
> embeddings of similar clusters move closer together while dissimilar
> ones move apart, turning a discrete ID into a meaningful continuous
> representation. This is the same idea behind Word2Vec
> (Mikolov et al., 2013), which represented words as vectors.

### Soft routing (cluster_probs)

For samples that sit on a cluster boundary (where `cluster_probs` are
spread across multiple clusters), *soft routing* uses a weighted
average of several cluster embeddings.

$$\mathbf{e}_{cluster} = \sum_{c=0}^{19} p_c \cdot \mathbf{E}_c \in \mathbb{R}^{32}$$

$$\mathbf{h}_{expert} = \text{TaskHead}([\text{GroupEncoder}(\mathbf{x}) \,\|\, \mathbf{e}_{cluster}])$$

Here $p_c$ is the posterior probability of GMM cluster $c$, and
$\mathbf{E}_c$ is the trainable embedding vector (32D) for cluster $c$.

> **Intuition.** This equation is how boundary customers are handled
> smoothly. If a customer belongs to cluster 3 with 60%, cluster 7 with
> 30%, and the rest with 10%, the embeddings of each cluster are mixed
> in those proportions. In implementation this is a single matrix
> product `cluster_probs @ embedding.weight`
> ($[B, 20] \times [20, 32] = [B, 32]$). The mixed embedding is
> concatenated with the GroupEncoder output and passed through the
> TaskHead, so the cluster-conditioning signal enters softly. Unlike
> hard routing, which force-assigns each customer to a single cluster,
> boundary customers become insensitive to fluctuations in cluster
> assignment.

When `forward_single_task()` receives `cluster_probs`, it performs
soft routing instead of hard assignment (lines 1247–1250).

> **Undergraduate math — GMM and Bayes' theorem: the mathematical
> basis of soft routing.** The cluster probability $p_c$ is the
> posterior of a *Gaussian Mixture Model (GMM)*. A GMM assumes that
> data is generated from a mixture of $K$ Gaussian distributions:
> $p(\mathbf{x}) = \sum_{c=1}^K \pi_c \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$,
> where $\pi_c$ is the mixture weight (prior) and $\mathcal{N}$ is the
> Gaussian. Applying *Bayes' theorem*, the posterior probability that
> an observation $\mathbf{x}$ belongs to cluster $c$ is
> $p(c | \mathbf{x}) = \frac{\pi_c \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$,
> which is exactly the $p_c$ used in soft routing. Customers far from
> any cluster center ($\boldsymbol{\mu}_c$) have posteriors spread
> across several clusters, amplifying the effect of soft routing,
> while customers close to a center are concentrated on one cluster
> and behave almost like hard routing. The parameters
> $\pi_c, \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c$ are estimated
> offline by the EM (Expectation-Maximization) algorithm.

> **Recent trends — cluster-based recommendation in industry.**
> Cluster-based conditional embedding has become a core technique in
> large-scale recommenders in 2023–2025. Kuaishou's
> *POSO (Personalized Cold-Start, KDD 2022)* installs independent gates
> per user segment to mitigate cold-start; Alibaba's *CL4CTR (2023)*
> refines user representations via cluster-wise contrastive learning;
> ByteDance's *SAMD (KDD 2024)* combines cluster embeddings with MoE
> and improved CTR by 2.3% in TikTok's short-video recommendation.
> The 20-cluster embedding + soft routing design in this system aligns
> with those industry trends, and in the financial domain it
> explicitly models behavioral differences across customer segments
> (VIP, general, young, senior).

## Logit Transfer — Explicit Information Passing Between Tasks

### Defining transfer pairs

`_build_logit_transfer()` (lines 984–1055) registers transfer pairs
from the `task_relationships` config.

| Source | Target | Type | Strength | Business meaning |
| --- | --- | --- | --- | --- |
| CTR | CVR | Sequential | 0.5 | Only clickers convert (AARRR funnel) |
| Churn | Retention | Inverse | 0.5 | Inverse of churn = retention base |
| CVR | LTV | Feature | 0.5 | Conversion prob affects lifetime value |
| NBA | Spending_category | Feature | 0.5 | Next-best action drives spending category |
| Spending_category | Brand_prediction | Feature | 0.5 | Category drives brand choice |

### Transfer mechanism

```python
# ple_cluster_adatt.py:1266-1277 — logit transfer applied in forward()
for task_name in execution_order:
    tower_input = task_expert_outputs[task_name]
    if task_name in self.logit_transfer_sources:
        source_task = self.logit_transfer_sources[task_name]
        if source_task in predictions:
            src_out = predictions[source_task]
            if src_out.dim() == 1:
                src_out = src_out.unsqueeze(-1)
            proj = self.logit_transfer_proj[task_name](src_out)
            tower_input = tower_input + strength * proj
```

$$\mathbf{h}_{tower}^t = \mathbf{h}_{expert}^t + \alpha \cdot \text{SiLU}(\text{LayerNorm}(\text{Linear}(\text{pred}^s)))$$

$\alpha = 0.5$ (`transfer_strength`), $\text{pred}^s$ is the source
task's prediction. `Linear` maps source output_dim → task_expert
output_dim (32D), and the projection module is
`nn.Sequential(Linear, LayerNorm, SiLU)`.

> **Intuition.** This equation is the explicit transfer that "adds the
> upstream task's prediction to the downstream task's input." In the
> CTR→CVR transfer, for example, when the CTR model predicts "this
> customer has a high click probability," that signal passes through a
> projection and is added to the CVR tower's input. $\alpha = 0.5$
> controls the relative magnitude of the transfer signal versus the
> original Expert output. Intuitively, because the transfer is added
> as a residual, when the source information is not useful the
> projection weights converge toward 0 and the transfer is simply
> ignored.

### Execution order (topological sort)

`_derive_task_order_from_config()` (lines 1093–1155) automatically
derives the execution order by running Kahn's algorithm on the
dependency graph encoded in `task_relationships`.

The dependency graph decomposes into three independent chains:

- **Engagement chain**: CTR --(seq)→ CVR --(feat)→ LTV
- **Retention chain**: Churn --(inv)→ Retention
- **Consumption chain**: NBA --(feat)→ Spending_category --(feat)→ Brand_prediction

> **⚠ When execution order fails.** If the topological sort fails
> (cycle detected), `_get_task_execution_order_fallback()` falls back
> to a hard-coded order (lines 1069–1091). When a new transfer
> relationship is added, registering it in the `task_relationships`
> config is enough — the order is picked up automatically.

> **Undergraduate math — topological sort and DAGs.** Topological sort
> is an algorithm that linearizes the nodes of a *Directed Acyclic
> Graph (DAG)* without violating any edge direction. If edge
> $A \to B$ exists, $A$ must appear before $B$.
> *Kahn's algorithm (1962)* works as follows: (1) enqueue every node
> with in-degree 0; (2) dequeue a node, append it to the output, and
> remove its outgoing edges; (3) enqueue any node whose in-degree just
> became 0; (4) repeat until the queue is empty. If some nodes remain
> unvisited, a *cycle* exists. *Time complexity* is $O(V + E)$, linear
> in the number of nodes $V$ and edges $E$. In this system, the chain
> CTR $\to$ CVR $\to$ LTV graphically encodes the causal precedence
> "CTR must be predicted before it can transfer to CVR, and CVR must
> be predicted before it can transfer to LTV."

> **Historical context — residual connections and Logit Transfer.**
> The form `tower_input = tower_input + alpha * proj` used by Logit
> Transfer is structurally identical to the residual (skip) connection
> proposed by *He, Zhang, Ren & Sun (CVPR 2016)* in ResNet. He et al.
> showed that training a 152-layer network without vanishing gradients
> required a "shortcut that adds the input as-is," i.e.
> $\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$. The idea had
> been explored earlier in *Highway Networks (Srivastava et al., 2015)*,
> but ResNet's radical simplification
> ($\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$ with no gate)
> turned out to be more effective. In Logit Transfer too, the source
> task's prediction is added as a residual, so when the transfer is
> not useful the projection weights converge to 0 and only the
> original Expert output remains — a *safe default* by construction.
> $\alpha = 0.5$ is the scaling coefficient that caps the residual's
> relative size.

> **⚠ Logit Transfer vs adaTT — two complementary transfer
> mechanisms.** This system performs cross-task knowledge transfer at
> *two different levels* simultaneously.
>
> | Property | Logit Transfer | adaTT |
> | --- | --- | --- |
> | Layer of operation | Feature/logit level (during forward pass) | Loss level (before backward pass) |
> | What is passed | Source task's prediction / hidden representation | Pairwise gradient affinity |
> | Directionality | Directed DAG (CTR→CVR→LTV) | Full matrix (all task pairs) |
> | Learnability | Fixed structure (hand-designed) | Adaptive (affinity learned via EMA) |
> | Purpose | Explicit propagation of sequential dependencies | Automatic mitigation of Negative Transfer |
>
> Logit Transfer and adaTT pursue the same goal — "cross-task
> knowledge transfer" — at different levels. Logit Transfer passes
> predictions directly between tasks whose business logic is
> sequential (e.g. CTR→CVR), while adaTT adaptively modulates
> pairwise influence at the gradient level for every task pair. They
> are complementary and operate at the same time. See the separate
> *adaTT tech reference* for the full mechanism.

## Task Tower — Final Prediction

### Tower architecture

The `TaskTower` class (lines 244–293) uses a common MLP shape across
all tasks.

$$\mathbf{y} = \text{Linear}_{32 \to out} \circ \text{Dropout} \circ \text{SiLU} \circ \text{LayerNorm} \circ \text{Linear}_{64 \to 32} \circ \text{Dropout} \circ \text{SiLU} \circ \text{LayerNorm} \circ \text{Linear}_{32 \to 64}(\mathbf{h}_{expert})$$

Input is 32D (Task Expert output), hidden_dims are [64, 32], dropout
is 0.2. Regression tasks use activation=None, binary uses sigmoid,
multiclass uses softmax.

> **Intuition.** The Task Tower is the "last step" that turns the 32D
> Expert output into a final prediction. It first expands 32→64 to
> enlarge capacity, compresses 64→32, and finally projects to the
> output dimension. Inserting LayerNorm (scale stabilization) + SiLU
> (nonlinearity) + Dropout (overfit prevention) between layers keeps
> even this shallow MLP stable. The output activation depends on the
> task type: binary uses sigmoid for a 0–1 probability, multiclass
> uses softmax for a class distribution, and regression outputs a real
> value with no activation.

```python
# ple_cluster_adatt.py:268-280
layers = []
prev_dim = input_dim  # 32
for hidden_dim in hidden_dims:  # [64, 32]
    layers.extend([
        nn.Linear(prev_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
    ])
    prev_dim = hidden_dim
layers.append(nn.Linear(prev_dim, output_dim))
```

> **Undergraduate math — what LayerNorm actually does.** Layer
> Normalization normalizes the hidden vector of each sample
> independently:
> $\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$,
> where $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ (mean),
> $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$ (variance),
> $\gamma, \beta \in \mathbb{R}^d$ are trainable scale/shift
> parameters, and $\epsilon \approx 10^{-5}$ keeps the denominator
> positive. *Why normalize at all?* Each layer's inputs depend on the
> previous layer's outputs. As previous-layer parameters change during
> training, the input distribution keeps shifting — *Internal
> Covariate Shift* — and learning-rate tuning becomes hard. LayerNorm
> normalizes every layer's input to mean 0 and variance 1, stabilizing
> the distribution. *Difference from BatchNorm*: BatchNorm normalizes
> the same neuron across the batch (batch-size dependent), while
> LayerNorm normalizes all neurons inside one sample (batch-size
> independent). LayerNorm is more stable in serving environments like
> the Task Tower where batch size varies.

### Per-task loss types

| Task | Type | Loss | Out dim | Activation | Weight |
| --- | --- | --- | --- | --- | --- |
| CTR | Binary | Focal ($\gamma$=2.0, $\alpha$=0.25) | 1 | sigmoid | 1.0 |
| CVR | Binary | Focal ($\gamma$=2.0, $\alpha$=0.20) | 1 | sigmoid | 1.5 |
| Churn | Binary | Focal ($\gamma$=2.0, $\alpha$=0.60) | 1 | sigmoid | 1.2 |
| Retention | Binary | Focal ($\gamma$=2.0, $\alpha$=0.20) | 1 | sigmoid | 1.0 |
| NBA | Multiclass | NLL (post-softmax) | 12 | softmax | 2.0 |
| Life-stage | Multiclass | NLL | 6 | softmax | 0.8 |
| Balance_util | Regression | Huber ($\delta$=1.0) | 1 | none | 1.0 |
| Engagement | Regression | MSE | 1 | none | 0.8 |
| LTV | Regression | Huber ($\delta$=1.0) | 1 | none | 1.5 |
| Channel | Multiclass | NLL | 3 | softmax | 0.8 |
| Timing | Multiclass | NLL | 28 | softmax | 0.8 |
| Spending_category | Multiclass | NLL | 12 | softmax | 1.2 |
| Consumption_cycle | Multiclass | NLL | 7 | softmax | 0.8 |
| Spending_bucket | Regression | Huber ($\delta$=1.0) | 1 | none | 0.8 |
| Brand_prediction | Contrastive | InfoNCE ($\tau$=0.07) | 128 | none | 2.0 |
| Merchant_affinity | Regression | Huber ($\delta$=1.0) | 1 | none | 1.0 |

> **Undergraduate math — Huber Loss: a compromise between MSE and
> MAE.** Huber Loss is a robust loss function proposed by
> *Peter J. Huber (1964)*:
> $\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta \cdot (|y - \hat{y}| - \delta/2) & \text{otherwise} \end{cases}$.
> It behaves like MSE ($L_2$, squared) when the error is small
> ($|y - \hat{y}| \le \delta$) and like MAE ($L_1$, absolute) when the
> error is large. *Why isn't MSE enough?* Because
> MSE $= (y - \hat{y})^2$ squares the error, outliers make the loss
> explode and produce huge gradients, so the model is dragged around
> by a single outlier. MAE $= |y - \hat{y}|$ is robust to outliers but
> is non-differentiable at 0 and has a constant gradient ($\pm 1$),
> so convergence near 0 is slow. Huber Loss combines MSE's smooth
> gradient near 0 with MAE's outlier resistance further out.
> $\delta = 1.0$ sets the boundary: "within an error of 1, track
> precisely; beyond 1, defend against outliers." For tasks like LTV
> (lifetime value) prediction, where extreme high-spend customers
> exist, Huber Loss is more stable than MSE.

> **Recent trends — InfoNCE and contrastive learning: the Brand
> Prediction loss.** The *InfoNCE (Noise-Contrastive Estimation)* loss
> used for Brand Prediction was introduced by *Oord, Li & Vinyals
> (2018)* in CPC (Contrastive Predictive Coding):
> $\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\sum_{i=0}^N \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}$,
> where $\mathbf{q}$ is the query, $\mathbf{k}_+$ the positive key,
> $\mathbf{k}_i$ the negative keys, and $\tau = 0.07$ the temperature.
> *SimCLR (Chen et al., ICML 2020)* and *MoCo (He et al., CVPR 2020)*
> applied it to visual representation learning and triggered the
> contrastive-learning boom; in 2023–2025 it has been widely adopted
> in sequential recommendation (*SASRec-CL*, *CL4Rec*, etc.). Brand
> Prediction is trained with contrastive learning here because —
> rather than directly classifying thousands of brands — placing
> similar brands close together and dissimilar brands far apart in
> *brand-embedding space* scales and generalizes better.

### Focal Loss implementation

`_compute_task_losses()` (lines 1765–1780) computes a
probability-based Focal Loss. Because the TaskTower has already
applied sigmoid, the implementation is written in probability space
rather than logit space to *avoid double sigmoid*.

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}, \quad \alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

$\gamma = 2.0$ is the focusing parameter (how aggressively easy
examples are down-weighted); $\alpha$ is the positive-class weight,
differentiated per task (see the config for the detailed design).

> **Intuition.** Focal Loss is standard Cross-Entropy multiplied by a
> $(1 - p_t)^\gamma$ weight. Since $p_t$ is "the probability the model
> assigns to the correct label," examples the model gets right
> (large $p_t$) have their weight shrink sharply, while examples it
> gets wrong (small $p_t$) keep their weight. The intuition is a
> training strategy encoded as a loss function: "you don't improve by
> redoing easy problems — focus on the hard ones." $\alpha_t$ is the
> class-imbalance correction, lifting the weight on rare positives
> (e.g. churners) so they aren't missed.

> **Undergraduate math.** *Why is Cross-Entropy $-\log(p)$?* In
> information theory, the *information content* of an event is defined
> as $I(x) = -\log_2(p(x))$ — the lower the probability (the more
> surprising the event), the higher the information content. Example:
> a coin flip landing heads ($p=0.5$) carries $-\log_2(0.5) = 1$ bit,
> while rolling a 1 on a die ($p=1/6$) carries
> $-\log_2(1/6) \approx 2.58$ bits. *Cross-Entropy*
> $H(p, q) = -\sum p(x) \log q(x)$ is "the average number of bits
> needed to encode data from the true distribution $p$ using the
> model's distribution $q$." In binary classification, if the label is
> $y=1$ and the model predicts $p$, then $-\log(p)$ becomes "a larger
> penalty the lower the probability the model assigned to the correct
> answer." *Relation to Focal Loss*: in
> $\text{FL} = -(1-p_t)^\gamma \log(p_t)$, the $(1-p_t)^\gamma$ factor
> multiplies this penalty by a weight proportional to "how wrong the
> prediction is." *Concrete numbers*: at $p_t=0.9$,
> $\text{CE} = -\log(0.9) = 0.105$,
> $\text{FL} = 0.1^2 \times 0.105 = 0.00105$ (a 100× reduction). At
> $p_t=0.1$, $\text{CE} = -\log(0.1) = 2.303$,
> $\text{FL} = 0.9^2 \times 2.303 = 1.865$ (almost unchanged).

> **Historical context.** Focal Loss was proposed by
> *Lin, Goyal, Girshick, He & Dollár (ICCV 2017)* in the context of
> object detection. At the time, one-stage detectors (e.g. YOLO, SSD)
> lagged behind two-stage detectors (Faster R-CNN) in accuracy,
> because background (easy negatives) overwhelmingly outnumbered
> foreground (hard positives), letting easy-example gradients dominate
> learning. Focal Loss dynamically reduces the contribution of easy
> examples via the $(1-p_t)^\gamma$ term, and enabled RetinaNet — a
> one-stage detector — to beat two-stage detectors for the first time.
> Since then it has become a standard loss in virtually every domain
> with severe class imbalance: recommendation, medical imaging, NLP,
> and so on. CTR prediction (with a positive rate of 3–8%) is a
> textbook application of Focal Loss.

```python
# ple_cluster_adatt.py:1774-1780 — fp16 AMP-safe focal loss
p_f = pred.squeeze().float().clamp(1e-7, 1 - 1e-7)
t_f = target.float()
bce = -(t_f * torch.log(p_f) + (1 - t_f) * torch.log(1 - p_f))
p_t = p_f * t_f + (1 - p_f) * (1 - t_f)
alpha_t = alpha * t_f + (1 - alpha) * (1 - t_f)
focal_weight = alpha_t * (1 - p_t) ** gamma
loss = (focal_weight * bce).mean()
```

> **⚠ Focal alpha design criteria.** `focal_alpha` is determined by
> two factors: the *positive-class ratio* and the *business cost of a
> false negative*.
> - CTR (positive 3–8%, moderate FN cost): $\alpha = 0.25$ (standard)
> - CVR (positive 0.5–3%, high FN cost): $\alpha = 0.20$ (boost learning on the negative boundary)
> - Churn (positive 5–15%, very high FN cost): $\alpha = 0.60$ (avoid missing churners, maximize recall)
> - Retention (positive 85–95%, moderate FN cost): $\alpha = 0.20$ (detect the minority early-churn signal)

### Uncertainty Weighting (Kendall et al.)

When `loss_weighting.strategy: "uncertainty"` is set (lines
1818–1827), each task's *homoscedastic uncertainty* is modeled with a
trainable log-variance.

$$\mathcal{L}_k^{\text{uw}} = w_k \cdot (\exp(-s_k) \cdot \mathcal{L}_k + s_k)$$

$s_k = \log(\sigma_k^2)$ is the trainable log-variance
(`task_log_vars[k]`); $\exp(-s_k)$ is the precision (higher
uncertainty ⇒ smaller weight); the $s_k$ term is a regularizer that
prevents uncertainty from growing without bound. $s_k$ is clamped to
$[-4.0, 4.0]$ and the precision to $[10^{-3}, 100]$.

> **Intuition.** This equation is a mechanism that "if a task is
> intrinsically hard to predict, automatically lowers its weight so
> its loss doesn't dominate training." $\exp(-s_k)$ is the precision —
> when uncertainty is high (large $s_k$), it shrinks and the task's
> contribution to the total loss is reduced. At the same time the
> $+s_k$ term acts as a regularizer, preventing the model from
> "declaring every task uncertain" to drive the loss to 0. Instead of
> hand-tuning 16 task weights, the model finds the balance on its own.

**[NeurIPS 2018]** Kendall, A., Gal, Y., & Cipolla, R. *"Multi-Task
Learning Using Uncertainty to Weigh Losses for Scene Geometry and
Semantics"*

> **Historical context.** Uncertainty Weighting was proposed by
> *Kendall, Gal & Cipolla (CVPR 2018)*. It was originally designed for
> scene understanding in autonomous driving — automatically balancing
> the three tasks of depth estimation, surface-normal estimation, and
> semantic segmentation. The theoretical basis is maximum-likelihood
> estimation (MLE) of *homoscedastic uncertainty* (uncertainty of the
> task itself, not of individual data points). Assuming each task's
> likelihood is Gaussian and simplifying the log-likelihood naturally
> yields the form $\exp(-s_k) \cdot \mathcal{L}_k + s_k$. Previously
> the choices were manual grid search or methods like GradNorm
> (Chen et al., ICML 2018) that equalize gradient magnitudes;
> Uncertainty Weighting replaced those with a *trainable parameter*
> that requires no extra hyperparameter, and was widely adopted.

> **Recent trends.** In 2024–2025 the MTL loss-balancing literature
> has moved beyond Uncertainty Weighting in several directions.
> *Nash-MTL (Navon et al., ICML 2022)* models cross-task gradients as
> a Nash bargaining game to find Pareto-optimal solutions;
> *Aligned-MTL (Senushkin et al., CVPR 2023)* aligns gradient
> directions to minimize conflict; *Auto-Lambda (Liu et al., 2024)*
> adapts weights via meta-learning. In practice, though, Uncertainty
> Weighting remains the most widely used option due to its
> implementation simplicity and stable performance, and it remains the
> proven default for large-scale MTL with more than 10 tasks.

### Aggregating the total loss

Inside `forward()` (lines 1284–1344), the following losses are summed.

1. **Task losses**: the sum of adaTT-enhanced losses (or a simple sum)
2. **CGC entropy regularization**: $\lambda_{\text{ent}} \times \mathcal{L}_{\text{entropy}}$ (during training, when CGC is not frozen)
3. **Causal Expert DAG regularization**: acyclicity + sparsity
4. **SAE loss**: reconstruction + L1 sparsity (weight = 0.01, detached)

## Where this leaves us

GroupTaskExpertBasket v3.2 replaces the per-cluster × per-task
independent MLP with a shared GroupEncoder + ClusterEmbedding, cutting
parameters by 88% while preserving cluster-level specialization; soft
routing uses GMM posteriors to handle boundary customers smoothly.
Logit Transfer declares business-order sequences like CTR→CVR→LTV as
a DAG, derives the execution order automatically via Kahn's
algorithm, and — because it adds a residual projection — lets the
transfer naturally converge to 0 when the source information is not
useful. The Task Tower produces predictions through a shared
32→64→32→out MLP while applying per-task-type losses (Focal, Huber,
NLL, InfoNCE) differentially, and Uncertainty Weighting balances 16
task weights automatically through trainable log-variances. The next
post, **PLE-6**, closes out the series with Sparse Autoencoder
interpretability, Evidential Deep Learning uncertainty, and the full
18-task spec — accompanied by a downloadable PDF of the complete PLE
tech reference.
