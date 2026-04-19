---
title: "[Study Thread] PLE-3 — Input Structure and Heterogeneous Shared Expert Pool (512D)"
date: 2026-04-19 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, expert-pool, hmm, shared-experts]
lang: en
series: study-thread
part: 3
alt_lang: /2026/04/19/ple-3-heterogeneous-expert-pool-ko/
next_title: "PLE-4 — Two CGC Gate Variants and HMM Triple-Mode Routing"
next_desc: "The weighted-sum CGCLayer vs the block-scaling CGCAttention — what each variant does and when to pick which. Plus entropy regularization to prevent Expert Collapse, dimension normalization to correct heterogeneous output asymmetry, and the full HMM Triple-Mode routing architecture."
next_status: published
---

*PLE-3 of the "Study Thread" series — a parallel English/Korean
sub-thread running PLE-1 → PLE-6 that summarizes the papers and math
foundations behind the PLE architecture used in this project. Source:
the on-prem `기술참조서/PLE_기술_참조서` document. This third post
covers the exact input structure the PLE model consumes (the
`PLEClusterInput` dataclass and its 734D features tensor) and the
composition and Forward dispatch strategy of what this project calls
the "heterogeneous Expert Pool" — seven structurally different Shared
Experts, each reading the customer through a different mathematical
lens.*

## PLEClusterInput — the input data structure

The `PLEClusterInput` dataclass (lines 62–199) encapsulates every
input the model sees. It supports batch-level device transfer and has
HMM-mode routing logic built in.

### Full field specification

| Field | Type | Shape | Description / Source |
|---|---|---|---|
| `features` | `Tensor` | `[B, 734]` | base 238 + multi_source 91 + domain 159 + multidisciplinary 24 + model_derived 27 + extended_source 84 + merchant 21 (= 644D normalized) + raw_power_law 90D |
| `cluster_ids` | `Tensor` | `[B]` | GMM cluster ID (0–19) |
| `cluster_probs` | `Tensor?` | `[B, 20]` | Cluster probabilities for soft routing (boundary-sample handling) |
| `hyperbolic_features` | `Tensor?` | `[B, 20]` | MCC(8D) + Product(8D) + Region(4D) Poincaré coordinates |
| `tda_features` | `Tensor?` | `[B, 70]` | tda_short(24) + tda_long(36) + phase_transition(10) |
| `tda_short_diagrams` | `Tensor?` | `[B, 200, 3]` | Raw Persistence Diagram (birth, death, beta_idx) |
| `tda_short_mask` | `Tensor?` | `[B, 200]` | Valid-pair mask (excludes padding) |
| `tda_long_diagrams` | `Tensor?` | `[B, 150, 3]` | Long-term Persistence Diagram |
| `tda_long_mask` | `Tensor?` | `[B, 150]` | Long-range valid-pair mask |
| `tda_global_stats` | `Tensor?` | `[B, 30]` | short_global 12D + long_global 18D |
| `tda_phase_transition` | `Tensor?` | `[B, 10]` | Phase-transition features |
| `hmm_journey` | `Tensor?` | `[B, 16]` | HMM Journey mode (10D base + 6D ODE dynamics) |
| `hmm_lifecycle` | `Tensor?` | `[B, 16]` | HMM Lifecycle mode |
| `hmm_behavior` | `Tensor?` | `[B, 16]` | HMM Behavior mode |
| `txn_seq` | `Tensor?` | `[B, 180, 16]` | Transaction sequence: card(8) + deposit(8) |
| `session_seq` | `Tensor?` | `[B, 90, 8]` | Session sequence |
| `collaborative_features` | `Tensor?` | `[B, 64]` | LightGCN pre-computed embedding |
| `hierarchy_features` | `Tensor?` | `[B, 20]` | H-GCN pre-computed Poincaré coordinates |
| `customer_segment` | `Tensor?` | `[B]` | 0=anonymous, 1=cold_start, 2=warm_start |
| `coldstart_features` | `Tensor?` | `[B, 40]` | Cold-start static features |
| `anonymous_features` | `Tensor?` | `[B, 15]` | Anonymous static features |
| `targets` | `Dict?` | variable | Per-task ground-truth labels (training only) |

### 734D `features` tensor index map

The ordering of the `continuous` list in `feature_schema.yaml` fixes
the column order the data loader assembles. The table below defines
the exact index ranges inside the `features` tensor.

| Feature group | Dim | Index range | Subtotal | Breakdown |
|---|---|---|---|---|
| Base | 238D | `[0, 237]` | 238 | RFM 34D + Category 64D + Transaction_Stats 80D + Temporal 60D |
| Multi-source | 91D | `[238, 328]` | 91 | Deposit 20D + Membership 15D + Investment 18D + Credit 12D + Digital 14D + Product 12D |
| Domain | 159D | `[329, 487]` | 159 | TDA_short 24D + TDA_long 36D + Phase_Transition 10D + GMM 22D + Mamba 50D + Economics 17D |
| Multidisciplinary | 24D | `[488, 511]` | 24 | Chemical 6D + Epidemic 5D + Interference 8D + Crime 5D |
| Model-derived | 27D | `[512, 538]` | 27 | Bandit 4D + HMM_summary 5D + LNN 18D |
| Extended source | 84D | `[539, 622]` | 84 | Insurance 25D + Consultation 18D + Campaign 12D + Overseas 6D + OtherChannel 23D |
| Merchant hierarchy | 21D | `[623, 643]` | 21 | MCC_L1 4D + MCC_L2 4D + Brand 8D + Stats 4D + Radius 1D |

Detailed indices inside the Domain group:

| Subgroup | Dim | Index range | Description |
|---|---|---|---|
| TDA-Short | 24D | `[329, 352]` | Short-term topological patterns from app logs (H0+H1, 90-day window) |
| TDA-Long | 36D | `[353, 388]` | Long-term topological patterns from financial transactions (H0+H1+H2, 12-month window) |
| Phase Transition | 10D | `[389, 398]` | W1 distance, topology-change magnitude, transition probability/direction/size, etc. |
| GMM Cluster | 22D | `[399, 420]` | GMM cluster membership probabilities + distance statistics |
| Mamba Temporal | 50D | `[421, 470]` | Mamba SSM time-series latent representation |
| Income Decomposition | 8D | `[471, 478]` | Income-structure decomposition (economics features) |
| Financial Behavior | 9D | `[479, 487]` | Financial-behavior indicators (economics features) |

Detailed indices inside the Model-derived group:

| Subgroup | Dim | Index range | Description |
|---|---|---|---|
| Bandit (MAB) | 4D | `[512, 515]` | Multi-Armed Bandit exploration/exploitation behavior statistics |
| HMM Summary | 5D | `[516, 520]` | Dominant state, duration, stability, entropy, change rate |
| LNN Model | 18D | `[521, 538]` | Distribution stats 4D + frequency 4D + change points 3D + autocorrelation 4D + complexity 3D |

> **`_FEATURE_GROUP_DIMS_ORDER` reordering complete.**
> `_FEATURE_GROUP_DIMS_ORDER` at `ple_cluster_adatt.py:407` has been
> realigned to the order in `feature_schema.yaml`:
> base → multi_source → *domain* → multidisciplinary →
> model_derived → extended_source → merchant. The validation chain is
> `feature_schema.yaml` → `task_feature_mapper.py:FEATURE_GROUP_DIMS`
> → `feature_integrator.py:EXPECTED_GROUP_DIMS_CAN`.

> **Undergraduate math — what a 734-dimensional input vector means.**
> The network's input $\mathbf{x} \in \mathbb{R}^{734}$ is a vector of
> 734 real numbers. The first 644 dimensions are normalized features;
> the last 90 are the raw, pre-normalization power-law features. In
> linear algebra, $\mathbb{R}^n$ is the $n$-dimensional real vector
> space, and each axis corresponds to one feature. For example, if
> $x_1$ is "monthly average spend" and $x_2$ is "recent login
> frequency", then a single customer is a *point* in a 734-dimensional
> space. Humans can only visualize up to three dimensions, but the
> mathematical operations (inner product, norm, projection) apply
> identically regardless of dimension. In 734D space you can still
> compute the cosine similarity between two customer vectors and
> quantify "how similar are their behavior patterns." *The curse of
> dimensionality*: in high-dimensional spaces data becomes sparse and
> distance-based methods become inefficient. This is exactly why
> Expert networks compress into 64D or 128D — the point of training
> is to keep only the information useful to the tasks while discarding
> the rest.

### HMM mode routing

The `set_hmm_routing()` class method (lines 173–186) is called once
at model initialization and builds a per-task HMM-mode mapping from
the `hmm_triple_mode` section of the config.

```python
# ple_cluster_adatt.py:172-186 — config is the single source of truth
@classmethod
def set_hmm_routing(cls, hmm_config: dict) -> None:
    routing: Dict[str, str] = {}
    for mode in ["journey", "lifecycle", "behavior"]:
        for task in hmm_config.get(mode, {}).get("target_tasks", []):
            routing[task.lower().replace("-", "_")] = mode
    cls._default_hmm_routing = routing
```

`get_hmm_for_task()` (lines 188–198) returns the corresponding HMM
tensor given a task name; any task not in the mapping falls back to
`"behavior"` mode as the default.

## Shared Expert composition (512D)

### Seven heterogeneous Shared Experts

`_build_shared_experts()` (lines 395–451) calls
`SharedExpertFactory.create_from_config()` and dynamically instantiates
whichever Experts are enabled in the config's `shared_experts` section.

| Expert name | Input | Output | Role |
|---|---|---|---|
| `unified_hgcn` | 47D | 128D | Hyperbolic GCN + merchant hierarchy (hgcn+merchant_hgcn unified) |
| `perslay` | 70D | 64D | Persistence Diagram processor (TDA topology features) |
| `deepfm` | normalized 644D | 64D | Feature Interaction (FM + Deep, per-field independent embeddings v3.11) |
| `temporal` | sequence | 64D | Temporal Ensemble (Mamba + LNN + Transformer) |
| `lightgcn` | 64D | 64D | Graph-based CF (pre-computed embedding) |
| `causal` | normalized 644D | 64D | SCM/NOTEARS-based causal-relation extraction |
| `optimal_transport` | normalized 644D | 64D | Sinkhorn-based Wasserstein-distance representation |

$$\mathbf{h}_{shared} = [\text{unified\_hgcn}_{128D} \,\|\, \text{perslay}_{64D} \,\|\, \text{deepfm}_{64D} \,\|\, \text{temporal}_{64D} \,\|\, \text{lightgcn}_{64D} \,\|\, \text{causal}_{64D} \,\|\, \text{OT}_{64D}]$$

$$\dim(\text{shared\_concat}) = 6 \times 64 + 1 \times 128 = 512D$$

Here $\|$ denotes tensor concatenation. DeepFM, Causal, and OT all
receive `inputs.features[:, :644]` (the normalized 644D).

> **Equation intuition.** This formula stitches together the
> analyses of seven different specialists in a single row.
> Intuitively: you take the outputs from a graph-structural view
> (128D), a topological view (64D), an FM-cross view (64D), and so
> on, and glue them into one 512-dimensional vector. The downstream
> CGC gate then has the raw material it needs to judge "whose
> opinion matters for this customer on this task."

> **What is current in the field.** Heterogeneous Expert composition
> is one of the core trends in recommender-system research in
> 2024–2025. Classical MoE work (MMoE, PLE) used MLP Experts with an
> identical structure; more recent work mixes GNN + Transformer +
> CNN and similarly varied architectures inside the Expert pool.
> Google's *Multi-Aspect Expert Model* (MAEM, KDD 2024) assigned
> Experts specialized in behavior / context / profile respectively
> and improved YouTube recommendations. Meta's *DHEN* (Deep
> Heterogeneous Expert Network, 2023) explicitly models interactions
> between heterogeneous Experts and was deployed on Instagram feed
> ranking. This system's seven heterogeneous Experts — GCN, PersLay,
> DeepFM, Temporal, LightGCN, Causal, OT — sit squarely
> in that lineage, and together they build a *multi-aspect*
> customer representation that no single-domain Expert could capture
> on its own.

### Per-Expert Forward dispatch

Inside `_forward_shared_experts()` (lines 1416–1567), each Expert is
fed a different input depending on its name.

```python
# ple_cluster_adatt.py:1435-1565 — per-Expert dispatch summary
for name, expert in self.shared_experts.items():
    if name in ("hgcn", "merchant_hgcn", "unified_hgcn"):
        # hierarchy_features(20D) + merchant slice(27D) = 47D
        out, hgcn_interpret, _ = expert(combined_input)
    elif name == "perslay":
        # Raw diagram mode, or pre-computed 70D fallback
        out, _ = expert(tda_short_diagrams / tda_features / zero)
    elif name == "deepfm":
        out, _ = expert(inputs.features[:, :644])   # normalized 644D
    elif name == "temporal":
        out, _ = expert(txn_seq, session_seq, ...)  # sequences
    elif name == "lightgcn":
        out, _ = expert(collaborative_features)  # pre-computed 64D
    elif name in ("causal", "optimal_transport"):
        out, _ = expert(inputs.features[:, :644])   # normalized 644D
```

### The zero-fallback strategy

Every Expert performs a *zero-tensor fallback* whenever its input
data is `None`. That keeps things safe when some samples in a batch
simply do not have the feature in question. CGC gating will later
learn to lower that Expert's weight automatically.

> **⚠ Zero fallback interacting with CGC.** If CGC assigns a large
> weight to an Expert whose output is zero, the overall
> representation gets diluted. The initial bias on
> `domain_experts` in CGC mitigates this, but if a particular
> Expert's input is missing for most batches early in training, a
> *dead expert* phenomenon can emerge.
> `_cgc_entropy_regularization` partially counteracts it by
> encouraging gate dispersion.

## Where this leads

So far we have looked at the exact *shape* of what the PLE model
consumes — a 734D mix of normalized and raw power-law features,
supplemented by GMM cluster probabilities, TDA persistence diagrams,
HMM triple-mode tensors, two kinds of sequences, and pre-computed
GCN embeddings. On top of that sits a pool of seven heterogeneous
Shared Experts, each of which reads the same customer through a
different mathematical lens — Hyperbolic GCN for hierarchy, PersLay
for topology, DeepFM for field crosses, Temporal for time series,
Causal for causal graphs, and Optimal Transport for distributional
distance. The 512D concatenation simply packs those seven views into
one vector; from there the real question becomes "which task should
trust which Expert, and by how much." That is the job of CGC gating,
and **PLE-4** takes up the math of its two variants — weighted-sum
CGCLayer and block-scaling CGCAttention — in detail.
