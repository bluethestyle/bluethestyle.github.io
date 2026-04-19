---
title: "[Study Thread] PLE-6 — Interpretability, Uncertainty, and Full Specs"
date: 2026-04-19 17:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, sae, uncertainty, evidential, specs]
lang: en
series: study-thread
part: 6
alt_lang: /2026/04/19/ple-6-interpretability-uncertainty-specs-ko/
next_title: "ADATT-1 — Why adaTT: Adaptive Towers and the Transformer Attention Analogy"
next_desc: "The adaTT sub-thread opens. The motivation for 'adaptive towers' starting from the limits of fixed towers, why Transformer Attention is the right mechanism for task adaptation, and where adaTT sits in the lineage of conditional computation and hypernetworks."
next_status: draft
source_url: /PLE_기술_참조서.pdf
source_label: "PLE Tech Reference (KO, PDF · 56 pages)"
---

*The final, sixth post of the "Study Thread" PLE sub-thread. Across
PLE-1 → PLE-6, in parallel Korean and English editions, I've walked
through the papers and mathematical foundations behind the PLE
architecture of this project. The source is the on-prem project's
`기술참조서/PLE_기술_참조서` (PLE Tech Reference). This part six
covers Expert interpretability (Sparse Autoencoder), uncertainty
quantification (Evidential Deep Learning), the full 18-task spec,
paper-vs-implementation comparison, a debugging guide, and the
appendix — and at the bottom of this post you can download the full
PDF. adaTT itself will be treated in a separate sub-thread starting
from ADATT-1.*

## SAE (Sparse Autoencoder) — Expert Interpretability

### Purpose

Extract interpretable sparse features from the 512D combined
representation of the Shared Experts. Inspired by Anthropic's Sparse
Autoencoder approach, the goal is to decompose internal neural network
representations into *human-interpretable units*.

### Architecture

`_build_sae()` (lines 877~896) instantiates `SparseAutoencoder`.

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_{enc} \cdot \mathbf{h}_{shared} + \mathbf{b}_{enc}) \in \mathbb{R}^{2048}$$

$$\hat{\mathbf{h}} = \mathbf{W}_{dec} \cdot \mathbf{z} + \mathbf{b}_{dec} \in \mathbb{R}^{512}$$

$$\mathcal{L}_{SAE} = \|\mathbf{h}_{shared} - \hat{\mathbf{h}}\|_2^2 + \lambda_1 \|\mathbf{z}\|_1$$

- `expansion_factor=4`: latent\_dim = 512 × 4 = 2048
- `l1_lambda=0.001`: induces sparsity
- `tied_weights=true`: $\mathbf{W}_{dec} = \mathbf{W}_{enc}^T$ (parameter saving)
- `loss_weight=0.01`: contribution to total loss

> **Equation intuition.** The first equation is encoding — the 512D
> shared representation is expanded 4× into a 2048D sparse vector
> $\mathbf{z}$. Thanks to ReLU, most elements become zero, so only a
> small number of active elements indicate "which concepts are
> switched on in this customer's representation." The second is
> decoding — the original 512D is reconstructed from the sparse
> vector, minimising information loss. The third equation's loss is
> the sum of reconstruction error ($L_2$) and a sparsity constraint
> ($L_1$). Intuitively it is the balance between two goals: "explain
> the expert representation with as few concepts as possible, but
> don't lose the original information."

> **Undergraduate math — the difference between L1 and L2 norms.**
> $\|\mathbf{z}\|_1 = \sum_i |z_i|$ is the sum of absolute values of
> each element, while
> $\|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 = \sum_i (h_i - \hat{h}_i)^2$
> is the sum of squared differences. Minimising the L1 norm induces
> *exactly zero* for many elements — a sparse solution. This is
> because of L1's geometric property: the vertices of the L1 ball lie
> on the axes, so under constrained optimisation the solution tends
> to sit on an axis (= the other coordinates are zero). The L2 norm,
> in contrast, has the shape of a sphere, so solutions are spread
> evenly and zeros are rare. Concrete example:
> $\mathbf{z} = [3, 0, 0, 2, 0]$ gives $\|\mathbf{z}\|_1 = 5$ with
> only two non-zero elements — this sparse $\mathbf{z}$ is
> interpretable as "only 2 of 5 concepts active."

> **Historical context — autoencoders, from dimensionality reduction
> to interpretability.** The autoencoder concept began with *Rumelhart,
> Hinton & Williams (1986)* in the backpropagation paper: "training
> a network to reconstruct itself forms useful representations in
> intermediate hidden layers." It later evolved into the Denoising
> Autoencoder (*Vincent et al., ICML 2008*) and the Variational
> Autoencoder (VAE, *Kingma & Welling, ICLR 2014*). The *Sparse
> Autoencoder* is a variant that places an L1 penalty on the hidden
> representation, forcing only a small number of neurons to activate,
> systematised by *Andrew Ng* in his 2011 Stanford lectures. The core
> idea, similar to PCA (Principal Component Analysis), is to reduce
> dimensionality — but with non-linear transforms allowed, and even
> in an *overcomplete* representation
> ($\dim(\mathbf{z}) > \dim(\mathbf{x})$), L1 sparsity can extract
> meaningful features. This system's SAE uses 512D → 2048D
> overcomplete encoding, "disentangling" the Expert information that
> is mixed up inside 512 dimensions into 2048 interpretable units.

> **Recent trends.** Applying Sparse Autoencoders to neural network
> interpretation — *mechanistic interpretability* — was ignited by
> Anthropic's 2023 research ("Towards Monosemanticity", Bricken et al.,
> 2023), which applied an SAE to the residual stream of a large
> language model to extract interpretable features. In 2024-2025,
> OpenAI, DeepMind, EleutherAI and others are actively researching
> SAE-based interpretation, and Templeton et al. (Anthropic, 2024)
> extracted millions of interpretable features from Claude 3 Sonnet.
> In recommendation systems too, there is growing research into
> decomposing a model's internal representation with an SAE to
> explain "why was this product recommended?", and the explainability
> requirements of the EU AI Act (in force from 2024) are accelerating
> this trend.

### Main Path Gradient Blocking

In `forward()` (line 1216), `shared_concat.detach()` disconnects the
SAE's input. The SAE loss updates only the SAE's own weights and has
no effect on the Shared Experts' learning.

```python
# ple_cluster_adatt.py:1216
_, sae_latent, sae_loss = self.sae(shared_concat.detach())
```

> **Using the SAE latent.** `PLEClusterOutput.sae_latent` (a 2048D
> sparse vector) is used after inference in the *Expert Neuron
> Dashboard* to analyse activation patterns. For example, one may
> interpret "frequently active latent #147 corresponds to a 'card
> loan usage pattern'."

## Evidential Deep Learning — Uncertainty Quantification

### Purpose

Quantify the *epistemic uncertainty* of task predictions (how much the
model "doesn't know") to evaluate recommendation trust. Predictions
with high uncertainty are routed to *fallback logic* at serving time.

### Principle (Sensoy et al., NeurIPS 2018)

For classification tasks, predict the *parameters of a Dirichlet
distribution* instead of a softmax output.

$$\boldsymbol{\alpha} = \text{evidence} + 1 \quad (\boldsymbol{\alpha} \in \mathbb{R}^K_+)$$

$$S = \sum_{k=1}^K \alpha_k \quad (\text{Dirichlet strength})$$

$$\hat{p}_k = \alpha_k / S \quad (\text{expected probability})$$

$$u = K / S \quad (\text{epistemic uncertainty})$$

- $K$: number of classes; larger $S$ means more confident, larger $u$ means more uncertain
- if evidence = 0, $\boldsymbol{\alpha} = \mathbf{1}$ → uniform distribution → maximum uncertainty

> **Equation intuition.** A conventional softmax classifier "always
> outputs a single probability distribution for any input," which
> risks making confident predictions for patterns unseen during
> training. The evidential approach models a "distribution over
> probability distributions" (Dirichlet). $\boldsymbol{\alpha}$ is
> the concentration parameter of the Dirichlet: as evidence
> accumulates, $S = \sum \alpha_k$ grows, the distribution becomes
> peaked (confident), and the uncertainty $u = K/S$ shrinks.
> Intuitively, this is quantification of epistemic uncertainty: "be
> confident when enough evidence accumulates; say honestly 'I don't
> know' when evidence is absent."

> **Undergraduate math — the Dirichlet distribution: modelling
> "probabilities of probabilities."** The Dirichlet distribution
> $\text{Dir}(\mathbf{p} | \boldsymbol{\alpha})$ is a distribution
> over the probability simplex. It generates a $K$-dimensional
> probability vector $\mathbf{p} = (p_1, \dots, p_K)$
> ($\sum p_k = 1$, $p_k \geq 0$), with probability density
> $f(\mathbf{p} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_k)}{\prod \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}$.
> $\Gamma(n)$ is the gamma function, with $\Gamma(n) = (n-1)!$ for
> natural numbers (the continuous extension of factorial).
> Intuitively: if all $\alpha_k$ equal 1, any $\mathbf{p}$ is equally
> likely (uniform); if all $\alpha_k$ are large, mass concentrates
> near the centre $(1/K, \dots, 1/K)$ (confidence); if only a
> specific $\alpha_k$ is large, mass shifts toward that class.
> Concrete example ($K = 3$): $\boldsymbol{\alpha} = (1, 1, 1)$ gives
> a uniform distribution over the triangle;
> $\boldsymbol{\alpha} = (10, 10, 10)$ concentrates near
> $(1/3, 1/3, 1/3)$ — "I am confident the three class probabilities
> are similar"; $\boldsymbol{\alpha} = (100, 1, 1)$ concentrates near
> $(1, 0, 0)$ — "I am confident class 1 is almost certain." By
> letting the network predict the Dirichlet's $\boldsymbol{\alpha}$,
> we can quantify "the variance of the prediction probabilities
> themselves" and express uncertainty.

> **Historical context.** Evidential Deep Learning was proposed by
> *Sensoy, Kaplan & Kandemir (NeurIPS 2018)*. Combining
> Dempster-Shafer evidence theory (1968, 1976) and Subjective Logic
> (Jøsang, 2016) with neural networks, they set out to solve the
> overconfidence problem of softmax outputs. The core idea — modelling
> "probabilities of probabilities" — was inspired by the Bayesian
> tradition of placing Dirichlet priors on posterior distributions
> (Ferguson, 1973). Later, Amini et al. (NeurIPS 2020) extended the
> framework to regression, proposing *Evidential Regression* that
> quantifies uncertainty of continuous predictions using the
> Normal-Inverse-Gamma (NIG) distribution.

> **Recent trends.** In 2024-2025, the evidential DL field focuses
> on *improving calibration* and *better OOD (Out-of-Distribution)
> detection*. The Posterior Network of Pandey & Yu (AAAI 2023) and
> the Natural Posterior Network of Charpentier et al. combine
> Normalizing Flows to raise the accuracy of evidence estimation. On
> the industry side, uncertainty quantification is becoming a
> regulatory requirement in autonomous driving (Waymo, 2024),
> medical diagnosis (Google Health), and financial risk assessment,
> accelerating production adoption. Research applying evidential
> approaches to LLM hallucination detection (Ren et al., 2024) is
> also drawing attention.

### Implementation

`_build_evidential_layers()` (lines 898~931) builds a per-task
`EvidentialLayer`.

```python
# ple_cluster_adatt.py:921-927
self.evidential_layers[task_name] = EvidentialLayer(
    input_dim=self.task_expert_output_dim,  # 32D
    task_type=task_type,
    output_dim=output_dim,
    kl_lambda=0.01,
    annealing_epochs=10,
)
```

In `forward()` (lines 1253~1260), it is applied in parallel to the
Task Expert output (32D); `compute_evidential_loss()` (lines
1838~1841) adds the auxiliary KL loss.

$$\mathcal{L}_{evi} = \mathcal{L}_{task} + \lambda_{KL} \cdot \min(1, \text{epoch}/\text{anneal}) \cdot \text{KL}(\text{Dir}(\boldsymbol{\alpha}) \,\|\, \text{Dir}(\mathbf{1}))$$

- `kl_lambda=0.01`, `annealing_epochs=10`: start with a small KL contribution
- prevents the failure mode where too-strong KL early in training collapses all predictions to uniform

> **Equation intuition.** The loss has two parts. The original task
> loss ($\mathcal{L}_{task}$) drives prediction accuracy. The KL term
> is a pressure "to push $\alpha$ of classes without evidence back to
> 1 (the uninformative state)." The annealing coefficient
> $\min(1, \text{epoch}/\text{anneal})$ starts the KL contribution
> weak at the beginning of training, so the model first acquires
> basic classification capability and only later focuses on
> calibrating its uncertainty. Intuitively: "at first, focus on
> getting answers right; once you have some skill, learn to express
> your confidence honestly."

## Full 18-Task Spec

Below is the complete specification of all 18 tasks defined in the
system. 16 are currently active; uplift and category\_uplift are
deactivated.

| Task | Group | Loss | dim | HMM mode | Weight | Active |
|---|---|---|---|---|---|---|
| CTR | Engagement | Focal | 1 | journey | 1.0 | O |
| CVR | Engagement | Focal | 1 | journey | 1.5 | O |
| Engagement | Engagement | MSE | 1 | journey | 0.8 | O |
| Uplift | Engagement | MSE | 1 | journey | 1.0 | X |
| Churn | Lifecycle | Focal | 1 | lifecycle | 1.2 | O |
| Retention | Lifecycle | Focal | 1 | lifecycle | 1.0 | O |
| Life-stage | Lifecycle | NLL | 6 | lifecycle | 0.8 | O |
| LTV | Lifecycle | Huber | 1 | lifecycle | 1.5 | O |
| Balance\_util | Value | Huber | 1 | behavior | 1.0 | O |
| Channel | Value | NLL | 3 | behavior | 0.8 | O |
| Timing | Value | NLL | 28 | behavior | 0.8 | O |
| NBA | Consumption | NLL | 12 | behavior | 2.0 | O |
| Spending\_category | Consumption | NLL | 12 | behavior | 1.2 | O |
| Consumption\_cycle | Consumption | NLL | 7 | behavior | 0.8 | O |
| Spending\_bucket | Consumption | Huber | 1 | behavior | 0.8 | O |
| Category\_uplift | Consumption | MSE | 12 | behavior | 1.5 | X |
| Brand\_prediction | Consumption | InfoNCE | 128 | behavior | 2.0 | O |
| Merchant\_affinity | Consumption | Huber | 1 | behavior | 1.0 | O |

### Task Groups (adaTT config)

| Group | Members | intra strength | inter strength |
|---|---|---|---|
| Engagement | CTR, CVR, Engagement, (Uplift) | 0.8 | 0.3 |
| Lifecycle | Churn, Retention, Life-stage, LTV | 0.7 | 0.3 |
| Value | Balance\_util, Channel, Timing | 0.6 | 0.3 |
| Consumption | NBA, Spending\_category, Consumption\_cycle, Spending\_bucket, Merchant\_affinity, Brand\_prediction | 0.7 | 0.3 |

> **Undergraduate math — what the intra/inter transfer strengths
> mean.** The adaTT *intra strength* ($= 0.6 \sim 0.8$) is the
> gradient transfer ratio between tasks within the same group; the
> *inter strength* ($= 0.3$) is the ratio between tasks across
> different groups. Mathematically, the amount of gradient
> transferred from task $i$ to task $j$ is
> $\mathbf{g}_j^{transferred} = \mathbf{g}_j + \alpha_{ij} \cdot \text{proj}(\mathbf{g}_i, \mathbf{g}_j)$,
> where
> $\text{proj}(\mathbf{g}_i, \mathbf{g}_j) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \cdot \mathbf{g}_j$
> is the projection of $\mathbf{g}_i$ onto the direction of
> $\mathbf{g}_j$. A *projection* is the operation of "extracting
> only the component of vector $\mathbf{a}$ along the direction of
> vector $\mathbf{b}$,"
> $\text{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$.
> Intuitively, within a group (e.g. CTR and CVR) gradients point in
> similar directions, so large transfer ($\alpha = 0.8$) is
> beneficial; across groups (e.g. CTR and Churn) gradients can
> conflict, so small transfer ($\alpha = 0.3$) is safer.

> **Recent trends — the frontier of gradient-based multi-task
> optimisation.** adaTT's gradient-based transfer extends a research
> stream that began with *PCGrad (Yu et al., NeurIPS 2020)*. PCGrad
> eliminates conflicts by projecting conflicting gradients onto each
> other's normal direction, while *CAGrad (Liu et al., NeurIPS 2021)*
> finds a direction that guarantees a minimum improvement for every
> task. *Nash-MTL (Navon et al., ICML 2022)* formalises this as a
> Nash bargaining game and derives Pareto-optimal solutions. In
> 2024-2025, *Aligned-MTL (Senushkin et al., CVPR 2023)* uses the
> SVD of the gradient matrix to find aligned update directions, and
> *FairGrad (Mahapatra & Rajan, 2024)* even considers fairness across
> tasks. This system's adaTT is distinctive in explicitly exploiting
> *group structure*, and by setting intra/inter strengths separately
> it injects domain knowledge (e.g. the CTR-CVR relationship).

## Paper vs Implementation Comparison

### PLE (Tang et al., 2020) comparison

| Item | Original paper | This implementation |
|---|---|---|
| Expert structure | Shared + Task-specific MLP | 7 domain Shared Experts (GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT) |
| Extraction Layer | Stack of PLE Layers | Single layer (CGC → GroupTaskExpertBasket) |
| Task Expert | Independent MLP per task | GroupEncoder + ClusterEmbedding (20 clusters) |
| Gate | Shared+Task Expert → gate | Shared Expert block scaling (512D preserved) |
| Knowledge Transfer | Implicit (Expert sharing) | Explicit Logit Transfer + gradient-based adaTT |
| Cluster specialisation | None | GMM 20-cluster embedding + GroupEncoder |
| HMM routing | None | Triple-Mode (journey/lifecycle/behavior) |
| Loss weighting | Fixed | Uncertainty Weighting (Kendall et al.) |
| Uncertainty | None | Evidential Deep Learning (Dirichlet) |

### MMoE (Ma et al., KDD 2018) comparison

| Item | MMoE | This implementation |
|---|---|---|
| # Experts | N identical-structure experts | 7 heterogeneous (GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT) |
| Expert structure | Identical MLP | Domain-specialised architecture each |
| Gate | Linear(input → N) + Softmax | Linear(512 → 7) + Softmax (CGC) |
| Expert Collapse | Severe (all tasks pick same expert) | Mitigated (entropy regularisation + domain\_experts bias) |
| Initial bias | None (random) | Warm start based on domain\_experts |
| Task specialisation | Separation by gate alone | CGC + HMM routing + GroupTaskExpertBasket |

### Main architectural innovations

Design elements unique to this project:

1. *Heterogeneous Expert combination*: instead of single-structure experts, combine 7 heterogeneous domain experts — GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT.
2. *CGC dimension normalisation*: corrects asymmetric expert output dimensions (128D vs 64D)
3. *HMM Triple-Mode routing*: selectively injects an HMM mode matched to each task's time scale
4. *GroupTaskExpertBasket*: GroupEncoder + ClusterEmbedding yields 88% parameter reduction (v3.2)
5. *Logit Transfer chain*: execution order is derived automatically from topological sort
6. *Evidential + SAE*: uncertainty quantification plus expert-representation interpretability

## Debugging Guide

### Expert output diagnostics

| Symptom | Cause | How to check |
|---|---|---|
| A specific Expert outputs all-zero | Input data is None → zero fallback | Inspect the corresponding expert tensor in the `shared_expert_outputs` dict |
| unified\_hgcn output NaN | Poincare coordinate overflow | Check `hierarchy_features` value ranges, tune curvature |
| temporal output all-zero | `txn_seq` is None | Confirm sequence loading is enabled in DataLoader |
| perslay output unstable | Raw diagram padding error | Check `tda_short_mask` valid ratio |

### CGC Attention distribution analysis

| Symptom | Cause | Fix |
|---|---|---|
| A single Expert concentrates at 0.9+ | Expert Collapse | Increase `entropy_lambda` (0.01→0.02) |
| All experts uniform (~0.125) | CGC not learning, or excess entropy | Decrease `entropy_lambda`, check learning rate |
| High weight on Experts outside domain\_experts | CGC overcame the domain bias | May be normal — cross-domain transfer pattern |
| Attention shifts sharply late in training | CGC freeze not applied | Check `freeze_epoch` setting |

### Loss-related issues

| Symptom | Cause | Fix |
|---|---|---|
| Loss becomes NaN/Inf | fp16 underflow + focal loss | Confirm `.float().clamp(1e-7, 1-1e-7)` (M-2 FIX) |
| Specific task loss = 0 | Targets all -1 (missing) | `ignore_index=-1` working as intended, inspect data |
| Uncertainty weight diverges | `task_log_vars` clamp not applied | Confirm clamp(-4.0, 4.0) and precision clamp |
| Total loss surges | Evidential KL annealing complete | Check KL contribution beyond `annealing_epochs` |
| Loss shifts sharply after adaTT kicks in | Negative transfer detected | Tune `negative_transfer_threshold` |

### Gradient flow diagnostics

| Symptom | Cause | Fix |
|---|---|---|
| Shared Expert gradient = 0 | `freeze_shared` active in Phase 2 | Check `freeze_shared_in_phase2` |
| CGC gradient = 0 (during training) | CGC frozen (`freeze_epoch` reached) | Intentional freeze — normal |
| `_extract_task_gradients` OOM | `retain_graph=True` accumulation | Increase `adatt_grad_interval` (10→50) |
| Brand prediction gradient weak | InfoNCE temperature too high | Decrease `temperature` (0.07→0.05) |

### HMM-related issues

| Symptom | Cause | Fix |
|---|---|---|
| HMM projection outputs identical | All samples received the default embedding | Check HMM pipeline data generation |
| Only a specific mode is being trained | Too few target tasks for the other modes | Rebalance `target_tasks` |
| Default embedding not trained | Most samples have valid HMM | Normal (the default applies only to a minority) |

### Logit Transfer issues

| Symptom | Cause | Fix |
|---|---|---|
| CVR over-depends on CTR | `transfer_strength` too high | Reduce 0.5→0.3 |
| Transfer not applied | Source task deactivated | Confirm the source is in `self.task_names` |
| Wrong execution order | Topological sort failed → fallback | Look for the "hardcoded fallback used" warning in logs |

## Appendix

### Code file mapping

| File | Role |
|---|---|
| `models/ple_cluster_adatt.py` | PLEClusterAdaTT main model (~2125 lines) |
| `models/experts/registry.py` | ExpertRegistry, SharedExpertFactory |
| `models/experts/cluster_task_expert.py` | ClusterTaskExpertBasket, GroupTaskExpertBasket |
| `models/adatt.py` | AdaptiveTaskTransfer (gradient-based transfer) |
| `models/layers/sae_layer.py` | SparseAutoencoder |
| `models/layers/evidential_layer.py` | EvidentialLayer (Dirichlet/Beta/NIG) |
| `models/tasks/task_registry.py` | TaskRegistry, TaskManager, TASK\_GROUPS |
| `models/tasks/base_task.py` | BaseTask, TaskConfig, TaskOutput, TaskType |
| `models/tasks/classification_tasks.py` | CTR, CVR, Churn, Retention, NBA, ... |
| `models/tasks/regression_tasks.py` | Engagement, BalanceUtil, LTV, Uplift |
| `models/tasks/merchant_tasks.py` | BrandPrediction, MerchantAffinity, ContrastiveLoss |
| `configs/model_config.yaml` | Full model config (source of truth) |

### Parameter count estimates

| Module | Params | Note |
|---|---|---|
| Unified H-GCN | ~200K | 128D output, merchant hierarchy |
| PersLay | ~50K | Raw diagram + global stats |
| DeepFM | ~169K | v3.11: independent per-field embeddings |
| Temporal Ensemble | ~500K | Mamba + LNN + Transformer |
| LightGCN | ~20K | Pre-computed embeddings → lightweight |
| Causal | ~100K | NOTEARS DAG + causal encoder |
| Optimal Transport | ~100K | Sinkhorn + reference distribution |
| CGC (16 tasks) | ~57K | 16 × Linear(512→7) |
| HMM projectors | ~5K | 3 × Linear(16→32) |
| GroupTaskExpertBasket | ~362K | 4 GroupEncoder × 20 clusters |
| Task Towers (16) | ~80K | 16 × MLP(32→64→32→out) |
| adaTT | ~10K | transfer matrix + affinity |
| SAE | ~2.1M | 512D × 4 expansion (analysis-only) |
| Evidential | ~30K | 16 × Linear(32→out) |
| Auxiliary projectors | ~40K | coldstart + anonymous + gate |
| **Total (excl. SAE)** | **~1.65M** | trainable parameters |
| **Total (incl. SAE)** | **~3.75M** | SAE is detached (analysis-only) |

> **How to check parameter counts.** The `model.summary()` method
> (lines 1967~2073) prints parameter counts per module. The figures
> above are estimates; actual values depend on the config. Run
> `summary()` after initialising the model for exact numbers.

### Training settings summary

| Item | Value |
|---|---|
| Optimizer | AdamW (lr=0.0005, weight\_decay=0.01) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, Tmult=2) |
| Batch size | 16384 |
| Max epochs | 100 |
| Early stopping | patience=7 |
| Gradient clipping | 5.0 |
| Mixed precision | fp16 (AMP) |
| Phase 1 | Shared Expert training (15 epochs) |
| Phase 2 | Cluster Subhead fine-tuning (8 epochs, shared frozen) |
| adaTT warmup | 0 epoch (production: 10) |
| adaTT freeze | 1 epoch (production: 28) |
| CGC freeze | synced with adaTT `freeze_epoch` |

> **Undergraduate math — the mathematical structure of AdamW.** AdamW
> was proposed by *Loshchilov & Hutter (ICLR 2019)* as Adam with
> *decoupled weight decay*. The update rule of base Adam is
> $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$
> (1st moment = moving average of gradients),
> $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$
> (2nd moment = moving average of squared gradients),
> $\hat{\mathbf{m}}_t = \mathbf{m}_t / (1 - \beta_1^t)$,
> $\hat{\mathbf{v}}_t = \mathbf{v}_t / (1 - \beta_2^t)$ (bias
> correction),
> $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$.
> Here $\eta$ is the learning rate; $\beta_1 = 0.9$, $\beta_2 = 0.999$
> are typical. Intuitively, $\hat{\mathbf{m}}_t$ says "which
> direction to go" (1st moment = inertia), while
> $\sqrt{\hat{\mathbf{v}}_t}$ says "how much this direction's
> gradient fluctuates" (2nd moment = adaptive learning rate) —
> high-variance parameters are stabilised by automatically shrinking
> their effective learning rate. AdamW's key difference is to apply
> weight decay directly to parameters rather than through gradients:
> $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1}(1 - \eta \lambda) - \eta \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$
> ($\lambda = 0.01$ = `weight_decay`), correctly implementing L2
> regularisation.

> **Historical context — the cosine annealing learning rate
> scheduler.** Cosine annealing was proposed by *Loshchilov & Hutter
> (ICLR 2017)* in the SGDR (Warm Restarts) paper. The learning rate
> is decayed as a cosine,
> $\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T_0))$.
> The learning rate is periodically restored to its maximum (warm
> restart), repeatedly offering opportunities to escape local
> minima. $T_0 = 10$ is the length of the first cycle, and
> $T_{mult} = 2$ doubles the cycle length each time (10→20→40
> epochs). This gives smoother transitions than StepLR (step decay)
> and explores a wider loss landscape than exponential decay thanks
> to warm restarts. Since 2020, cosine schedulers have become the
> standard for most large-model training — the warm-up + cosine
> decay combination is used even in GPT-3, PaLM, and other LLMs.

### Core config paths

The single source of truth for the model is
`configs/model_config.yaml`. The mapping between major sections and
model methods:

| Config section | Description | Reading method |
|---|---|---|
| `global` | cluster count, dropout, input\_dim | `__init__` |
| `shared_experts` | 7 expert settings | `_build_shared_experts` |
| `cgc` | CGC activation, bias, entropy | `_build_cgc` |
| `hmm_triple_mode` | 3-mode routing, target tasks | `_build_hmm_projectors` |
| `task_experts.common` | GroupEncoder settings | `_build_task_experts` |
| `task_experts.tasks` | per-task settings for 16+2 | `_build_task_experts`, `_compute_task_losses` |
| `adatt` | transfer strengths, task groups | `_build_adatt` |
| `task_relationships` | Logit Transfer pairs | `_build_logit_transfer` |
| `task_towers` | tower structure, activation | `_build_task_towers` |
| `sae` | SAE activation, expansion | `_build_sae` |
| `evidential` | Evidential activation, KL | `_build_evidential_layers` |
| `training` | lr, batch, epochs, loss weighting | `__init__` (task\_log\_vars) |

## Download the full PLE Tech Reference

That closes a walk from PLE-1 through PLE-6 across the on-prem
`기술참조서/PLE_기술_참조서` in blog form — equations, historical
context, implementation details all included. The original PDF is a
longer reference document with typesetting, index, and page numbers
fully preserved.

> **📄 [Download the PLE Tech Reference (full PDF)](/PLE_기술_참조서.pdf)** · KO · ~56 pages
>
> Progressive Layered Extraction · CGC Gate · GroupTaskExpertBasket ·
> Logit Transfer · 2-Phase Training — if you want the entire PLE-family
> architecture of this project in a single document, the link above
> is the place.

## Closing the PLE sub-thread, onward to adaTT

That is the end of the PLE sub-thread. Starting from the limits of
Shared-Bottom and MMoE in PLE-1, through the explicit Shared/Task
Expert separation and the mathematical intuition behind it in PLE-2,
the input structure (PLEClusterInput · 734D) and seven heterogeneous
Shared Expert pool in PLE-3, the two CGC gate stages (Stage 1 CGCLayer
weighting Shared + Task together, Stage 2 CGCAttention block-scaling
the Shared concat) and HMM Triple-Mode routing in PLE-4,
GroupTaskExpertBasket,
Logit Transfer, and Task Tower in PLE-5, and in this sixth post
interpretability (SAE), uncertainty (Evidential), the 18-task spec,
the paper-vs-implementation innovations, debugging guide, and
appendix — all the major components of this project's PLE-family
architecture have been laid out in blog form.

The separate **ADATT-1** opens the adaTT sub-thread. It will cover the
motivation for "adaptive towers" starting from the limits of fixed
towers, why Transformer Attention is the right mechanism for task
adaptation, and where adaTT sits in the lineage of conditional
computation and hypernetworks.
