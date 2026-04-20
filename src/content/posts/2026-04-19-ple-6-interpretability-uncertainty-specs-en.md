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
next_status: published
source_url: /PLE_기술_참조서.pdf
source_label: "PLE Tech Reference (KO, PDF · 56 pages)"
---

*The final, sixth post of the "Study Thread" PLE sub-thread. Across
PLE-1 → PLE-6, in parallel Korean and English editions, I've walked
through the papers and mathematical foundations behind the PLE
architecture of this project. By PLE-5 the system is structurally
done — it trains, it predicts. But two questions remain: "can we see
what each expert actually learned?" and "can we quantify how much
we trust a prediction?" This sixth post is the response, followed by
a reference appendix of the full specs, and a PDF download to close
out the series.*

## What PLE-5 leaves on the table

The architecture is complete. CGC picks experts stably.
GroupTaskExpertBasket handles cluster-level specialization. Logit
Transfer passes sequential dependencies. Uncertainty Weighting
auto-balances 16 loss scales. The model runs, produces predictions,
and can be served.

Yet two questions still nag.

**Do we actually know what the experts learned?** PLE's core design
bet was "heterogeneous experts will learn complementary things." It's
easy to confirm that gate weights are spread out (thanks to entropy
regularization). That does not guarantee each expert learned
*meaningfully different* things — seven experts might be
re-expressing similar patterns in different coordinate systems. The
512D concat vector is hard to read, and naive activation analysis
doesn't tell you "what does this neuron mean."

**We don't know how much to trust each prediction.** Softmax always
produces a probability distribution — even for out-of-distribution
inputs, it confidently declares "70% churn probability." In financial
decisions — lending, risk, credit actions — overconfidence carries
legal and financial liability. At minimum we need a signal that says
"don't trust this prediction, fall back to rule."

Two questions, answered in order. Crucially, both answers attach in a
way that *does not affect the main prediction path*. Interpretability
and uncertainty are analysis tools, not prediction tools.

## Decision 1 — Sparse Autoencoder for expert interpretability

### The problem — how do we read a 512D concat?

After training, the 512D representation formed by concatenating the
seven experts' outputs is hard to read directly. Each dimension is
typically a mixture of several concepts active at once ("this neuron
is 50% high-value-customer signal, 30% seasonality, 20% brand
preference" — polysemantic representation is the norm, not the
exception).

A few alternatives:

- **PCA / ICA.** Linear methods struggle on nonlinear neural
  representations.
- **Attention heat-map interpretation.** The CGC gate weights already
  tell us "how much does task $k$ attend to expert $i$." But what
  concepts activate *inside* each expert remains opaque.
- **Sparse Autoencoder (SAE).** Expand the representation into an
  overcomplete latent space, then enforce L1 sparsity to extract a
  *monosemantic* decomposition where each latent unit ideally
  corresponds to one interpretable concept. Anthropic's *Towards
  Monosemanticity* (Bricken et al., 2023) showed this works on LLMs.

Option three. Lift 512D into a 2048D overcomplete latent with L1
sparsity.

### SAE architecture

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

> **Undergraduate math — why L1 induces sparsity, not L2.**
> Minimising $\|\mathbf{z}\|_1 = \sum_i |z_i|$ produces solutions
> where many elements are *exactly zero*. Geometrically, the vertices
> of the L1 ball lie on the axes, so constrained optimisation tends
> to land on an axis (other coordinates at zero). The L2 ball is a
> sphere, spreading solutions evenly with few exact zeros.
> $\mathbf{z} = [3, 0, 0, 2, 0]$ — only 2 of 5 concepts active — is
> a natural outcome of L1 regularization.

### Main path gradient blocking

The SAE is an analysis tool, not part of the prediction path. We
apply `shared_concat.detach()` on its input so SAE gradients do not
update the Shared Experts. The SAE loss trains only the SAE's own
parameters and does not perturb the main training dynamics.
`loss_weight=0.01` is an extra safety bolt limiting inertia.

> **Using the SAE latent.** `PLEClusterOutput.sae_latent` (a 2048D
> sparse vector) is used after inference in the *Expert Neuron
> Dashboard* to analyse activation patterns. For example, one may
> interpret "frequently active latent #147 corresponds to a 'card
> loan usage pattern'." Under emerging explainability regimes (EU AI
> Act), keeping a decomposable representation like this is valuable
> on its own.

> **Historical context — autoencoders.** Autoencoders originate in
> *Rumelhart, Hinton & Williams (1986)*: "training a network to
> reconstruct itself forms useful representations in intermediate
> hidden layers." They evolved into Denoising Autoencoders (Vincent
> et al., ICML 2008) and VAE (Kingma & Welling, ICLR 2014). Sparse
> Autoencoders were systematised by *Andrew Ng* in his 2011 Stanford
> lectures and were re-popularised when Anthropic's "Towards
> Monosemanticity" (Bricken et al., 2023) applied them to the
> residual stream of LLMs to extract interpretable features.

## Decision 2 — Evidential Deep Learning for epistemic uncertainty

### The problem — Softmax cannot say "I don't know"

A softmax classifier always outputs a probability distribution.
Faced with an out-of-distribution customer — a pattern outside the
training distribution — it still confidently prints "70% churn
probability." There is no signal telling serving to fall back to
rule on a boundary case.

Alternatives:

- **Monte Carlo Dropout (Gal & Ghahramani 2016).** Run inference with
  dropout on, multiple times, and read variance. Simple, but inference
  cost scales by $N$ and serving latency suffers.
- **Deep Ensemble (Lakshminarayanan et al., 2017).** Train $N$
  independent models and read prediction variance. Gold standard for
  reliability, but training cost $\times N$.
- **Evidential Deep Learning (Sensoy et al., NeurIPS 2018).** Predict
  the parameters of a Dirichlet distribution itself. One forward pass
  yields both "prediction" and "uncertainty of that prediction"
  together.

Option three. Inference cost is essentially identical to standard
softmax, and uncertainty falls out naturally.

### The principle

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

> **Undergraduate math — the Dirichlet distribution.** $\text{Dir}(\mathbf{p} | \boldsymbol{\alpha})$
> is a distribution over the probability simplex. If all $\alpha_k = 1$,
> it is uniform; if all are large, mass concentrates near the center
> $(1/K, \dots, 1/K)$ (confidence); if only a particular $\alpha_k$ is
> large, mass shifts toward that class. For instance
> $\boldsymbol{\alpha} = (10, 10, 10)$ says "I'm confident all three
> classes are equally likely," and $\boldsymbol{\alpha} = (100, 1, 1)$
> says "I'm confident class 1 is almost certain." When the network
> predicts $\boldsymbol{\alpha}$, it also quantifies the variance of
> the prediction itself.

> **Historical context.** Evidential Deep Learning was proposed by
> *Sensoy, Kaplan & Kandemir (NeurIPS 2018)*, combining Dempster-Shafer
> evidence theory (1968, 1976) and Subjective Logic (Jøsang 2016)
> with neural networks. In 2020 Amini et al. extended it to regression
> (*Evidential Regression*) with the Normal-Inverse-Gamma distribution.
> As of 2024–2025, uncertainty quantification is becoming a
> regulatory requirement in autonomous driving, medical diagnosis,
> and financial risk assessment, which has accelerated industrial
> adoption.

### Implementation and auxiliary loss

`_build_evidential_layers()` creates per-task `EvidentialLayer`
instances that attach in parallel to the Task Expert output (32D) and
predict $\boldsymbol{\alpha}$. `compute_evidential_loss()` adds an
auxiliary KL loss.

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

## Full 18-Task Spec — reference

With interpretability and uncertainty handled, what's left is the
*accounting* of how every decision above actually sits in the system.
What follows is the reference operators consult — not new decisions,
but the concrete values the previous five posts' decisions resolved
to.

The complete specification of all 18 tasks defined in the system. 16
are currently active; uplift and category_uplift are deactivated.

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

> **What intra / inter mean.** adaTT's *intra strength* ($= 0.6 \sim 0.8$)
> is the gradient transfer ratio between tasks within the same group;
> *inter strength* ($= 0.3$) is the ratio between tasks across groups.
> Same-group pairs (e.g. CTR, CVR) have similar gradient directions,
> so larger transfer is beneficial; cross-group pairs (e.g. CTR,
> Churn) can conflict, so smaller transfer is safer. Full adaTT math
> lives in the separate ADATT sub-thread.

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

Design elements unique to this project — one-line recaps of the
decisions from the previous five posts:

1. *Heterogeneous Expert combination* (PLE-2, PLE-3): seven heterogeneous domain experts in place of homogeneous MLPs
2. *CGC dimension normalisation* (PLE-4): corrects the 128D vs 64D asymmetry
3. *HMM Triple-Mode routing* (PLE-4): injects the right time-scale state into each task group
4. *GroupTaskExpertBasket* (PLE-5): GroupEncoder + ClusterEmbedding yields 88% parameter reduction
5. *Logit Transfer chain* (PLE-5): execution order derived automatically from topological sort
6. *Evidential + SAE* (PLE-6): uncertainty quantification plus expert-representation interpretability

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
| DeepFM | ~169K | independent per-field embeddings |
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
> prints parameter counts per module. The figures above are estimates;
> actual values depend on the config.

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

### Core config paths

The single source of truth for the model is
`configs/model_config.yaml`.

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
`기술참조서/PLE_기술_참조서` in blog form. Each post started from the
problem the previous post's solution had introduced and moved to the
next decision — a chain. The original PDF is a longer reference
document with typesetting, index, and page numbers fully preserved.

> **📄 [Download the PLE Tech Reference (full PDF)](/PLE_기술_참조서.pdf)** · KO · ~56 pages
>
> Progressive Layered Extraction · CGC Gate · GroupTaskExpertBasket ·
> Logit Transfer · 2-Phase Training — if you want the entire PLE-family
> architecture of this project in a single document, the link above
> is the place.

## Closing the PLE sub-thread, onward to adaTT

That is the end of the PLE sub-thread. Each post was a link in a
chain, starting from a problem the previous post's solution had
opened.

- **PLE-1**: Shared-Bottom → MMoE — gradient conflict leading to Expert Collapse.
- **PLE-2**: explicit Shared/Task split + heterogeneous experts + softmax gate — a three-part answer to MMoE collapse.
- **PLE-3**: the seven experts, one by one — what gap each fills and why they don't reduce to one another.
- **PLE-4**: dim-asymmetry and time-scale separation — two-stage CGC + HMM Triple-Mode.
- **PLE-5**: memory, task dependency, loss balance — GroupTaskExpertBasket, Logit Transfer, Uncertainty Weighting.
- **PLE-6**: interpretability and uncertainty — SAE and Evidential DL, plus the full-spec reference.

The separate **ADATT-1** opens the adaTT sub-thread. It will cover the
motivation for "adaptive towers" starting from the limits of fixed
towers, why Transformer Attention is the right mechanism for task
adaptation, and where adaTT sits in the lineage of conditional
computation and hypernetworks — same format, same chain of "problem
the last decision left → next decision."
