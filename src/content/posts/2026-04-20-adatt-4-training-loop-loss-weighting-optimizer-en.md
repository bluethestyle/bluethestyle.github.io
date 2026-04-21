---
title: "[Study Thread] ADATT-4 — Training Loop, Loss Weighting, Optimizer, and CGC Synchronization"
date: 2026-04-20 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, training-loop, loss-weighting, optimizer, specs]
lang: en
series: study-thread
part: 10
alt_lang: /2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-ko/
next_title: "Next sub-threads — Causal OT, TDA, Temporal, Economics Expert foundations"
next_desc: "PLE and adaTT are done; next up are the math foundations of each of the 7 heterogeneous Shared Experts. Sub-threads will open for CausalOT (causal inference + optimal transport), TDA (topological data analysis / PersLay), Temporal (Mamba + LNN + Transformer), and the Economics-feature-based Expert in that order."
next_status: draft
source_url: /adaTT_기술_참조서.pdf
source_label: "adaTT Tech Reference (KO, PDF)"
---

*The final, fourth post of the adaTT sub-thread in the "Study Thread"
series. Across ADATT-1 → ADATT-4, in parallel Korean and English, I
have walked through the adaTT mechanism behind this project. The
source is the on-prem reference `기술참조서/adaTT_기술_참조서`. Through
ADATT-3, I laid out adaTT's four design decisions — Transfer Loss,
Group Prior, 3-Phase Schedule, and Negative Transfer blocking — and
how they interlock. This fourth post covers how that structure plugs
into the *actual training loop*, what the synchronisation contract
with CGC is, and how performance is defended. The full PDF is at the
bottom.*

## What ADATT-3 Left on the Table

The design is done. Affinity measurement, transfer-weight computation,
3-Phase scheduling, negative blocking — all self-contained inside the
`adaTT` module. But this module does not train alone. The project's
Trainer runs its own 2-Phase training (Shared Pretrain → Cluster
Finetune), applies Uncertainty Weighting over 16 tasks, manages
learning rates via AdamW + SequentialLR, and trains CGC's gate
weights. How does adaTT coexist with these? That is the central
question of this post.

Six decisions resolve it.

## Decision 1 — 2-Phase Training Loop (a *Different Layer* from adaTT's 3-Phase)

A naming note first. adaTT's *internal* 3-Phase (Warmup → Dynamic →
Frozen) is the affinity schedule; the Trainer's 2-Phase (Phase 1
Pretrain → Phase 2 Finetune) is the training schedule. The two layers
are orthogonal, each with its own freeze timing.

*Phase 1 — Shared Expert Pretrain.* Train the whole model for
`shared_expert_epochs` (default 15): Shared Experts, CGC, Task Experts,
Task Towers are all trainable. adaTT is *active* — gradient extraction
and transfer loss both run.

*Phase 2 — Cluster Finetune.* Train only per-cluster Task Expert
subheads for `cluster_finetune_epochs` (default 8). Shared Experts are
frozen. adaTT is *disabled*. The reason is simple — adaTT's gradients
are computed w.r.t. Shared Expert parameters, and those being frozen
means gradients are zero, so `autograd.grad` is wasted work.

Why split? Cluster-specific learning done alongside Shared training
pollutes the shared representation. Training Shared fully first,
freezing it, and then fine-tuning cluster heads draws a clean line
between specialisation and generalisation.

### Reset on Phase Transition

At the Phase 2 boundary, the following are all reset.

| Reset item | Reason |
|---|---|
| Optimizer | Shared frozen → AdamW momentum is stale, restart |
| Scheduler | Phase-2-only warmup (2 epochs, shorter than Phase 1's 5) |
| GradScaler | AMP scaler state reset (loss scale shifts across phase) |
| Early stopping | `best_val_loss`, `patience_counter` both reset |
| CGC Attention | Shared frozen → CGC gating frozen together |

Three safety guarantees matter. (1) adaTT is *backed up* and disabled
at Phase 2 start, not *replaced* — the model keeps its original
reference. (2) On Phase 2 end, a `finally` block *always* restores
adaTT, preserving checkpoint / inference compatibility even under
exceptions. (3) Warmup shrinks from Phase 1's 5 epochs to Phase 2's 2
epochs. Phase 2 is short; a long warmup would be pointless.

## Decision 2 — How to Balance 16 Task Losses

The 16 tasks have wildly different loss scales. CTR / CVR focal losses
live in one range, LTV's huber in another, brand_prediction's InfoNCE
in yet another. Manually tuning per-task weights is a combinatorial
explosion. Kendall et al. (CVPR 2018)'s Uncertainty Weighting
automates this.

$$\mathcal{L}_i^{weighted} = \frac{1}{2 \sigma_i^2} \cdot \mathcal{L}_i + \frac{1}{2} \log \sigma_i^2$$

- $\sigma_i^2 = \exp(\text{log\_var}_i)$: task $i$'s learnable
  homoscedastic uncertainty.
- `log_var` clamp $[-4, 4]$, precision clamp $[0.001, 100]$.

> **Equation intuition.** The first term is the loss weighted by
> precision ($1/\sigma^2$) — uncertain tasks get smaller weight. The
> second term $\frac{1}{2} \log \sigma_i^2$ is a regularisation
> penalty — it prevents the cheat of letting $\sigma$ blow up to zero
> out the loss. The form falls out naturally from $-\log p$ of the
> Gaussian likelihood $\mathcal{N}(\hat{y}, \sigma^2)$.

The *order* relative to adaTT matters. Uncertainty Weighting is
applied *before* adaTT. The `task_losses` that enter adaTT already
have uncertainty weighting baked in. This is intentional — the
learning signal of business-critical tasks (like nba with fixed
weight 2.0) should propagate through transfer to other tasks too.

### Per-Task Fixed Weights

Some tasks carry an extra fixed weight on top of Uncertainty
Weighting, especially those differing in business priority, positive
rate, or false-negative cost.

| Task | weight | loss type | Note |
|---|---|---|---|
| ctr | 1.0 | focal ($\gamma$=2, $\alpha$=0.25) | standard |
| cvr | 1.5 | focal ($\gamma$=2, $\alpha$=0.20) | very low positive rate → weight boosted |
| churn | 1.2 | focal ($\gamma$=2, $\alpha$=0.60) | high FN cost → alpha boosted |
| nba | 2.0 | CE | 12 classes, business-critical |
| ltv | 1.5 | huber ($\delta$=1.0) | regression, outlier-robust |
| brand_prediction | 2.0 | contrastive (InfoNCE) | 50K brands |

## Decision 3 — Per-Expert Learning Rate and SequentialLR

The seven Shared Experts are structurally different. The 128D
unified_hgcn learns in hyperbolic space and needs a conservative lr;
the 64D DeepFM can converge faster. A single global lr is suboptimal
either way.

The fix is per-Expert `param_group`. Each Shared Expert's parameters
go into a separate group, with per-expert lr / weight_decay overrides
in `model_config.yaml`. In Phase 2, frozen Shared Experts have
`requires_grad=False` and are automatically excluded by
`_create_optimizer` — no wasted optimizer-state memory.

The scheduler is a Linear Warmup → CosineAnnealingWarmRestarts inside
a SequentialLR.

- `warmup_steps = 5` epochs, `start_factor = 0.1`: warmup start lr =
  $0.0005 \times 0.1 = 5 \times 10^{-5}$.
- `cosine_t0 = 10`, `cosine_t_mult = 2`: first period 10 epochs, then
  20 → 40.
- At Phase 2 entry, `warmup_steps = 2` and the scheduler is rebuilt.

Why plain cosine (not warm restart) in Phase 2? Phase 2 is short —
only 8 epochs by default. A warm-restart period structure never
completes before training ends; a single smooth decay on a short
window suits it better.

AdamW's other hyperparameters: `lr=5 \times 10^{-4}$,
`weight_decay=0.01`, `gradient_clip_norm=5.0`.

## Decision 4 — The CGC-adaTT Synchronisation Contract

CGC learns "which Shared Experts each task should attend to." adaTT
regulates "how gradients transfer between tasks." Both act *on the
same Shared Expert parameters*, so they must stay in sync or they
fight each other.

*Why freeze together.* Suppose adaTT has frozen its transfer weights.
If CGC keeps training, CTR's Expert attention shifts, which changes
the direction of the CTR gradient flowing into Shared. adaTT's frozen
weights — which captured "CTR→CVR positive transfer" — now reflect a
stale relationship. Both gate dynamics and transfer dynamics must stop
together for clean convergence.

Synchronisation happens at two points.

- At adaTT's `freeze_epoch`, the `_cgc_frozen` buffer flips to True
  and CGC attention's `requires_grad=False`.
- At Phase 2 start, the same treatment. With Shared frozen, training
  CGC gating alone would overfit because the input (Expert outputs)
  no longer changes.

`_cgc_frozen` is a `register_buffer`, so its freeze state survives
checkpoint save / restore.

## Decision 5 — Memory and Performance, Three Key Moves

adaTT's gradient extraction is expensive. Per-task gradients against
the Shared Expert parameters across 16 tasks drop training speed
sharply without optimisation. Three decisions make it manageable.

*`retain_graph=True`'s cost.* Calling `autograd.grad` sequentially for
16 tasks while keeping the graph pushes peak memory to about 2× the
forward pass. Architecturally unremovable — the Trainer's
`loss.backward()` must reuse the same graph. On an RTX 4070 12GB, 16
tasks × batch_size 16384 is the ceiling.

*`adatt_grad_interval = 10`.* Every-step extraction means 16 ×
`autograd.grad` calls per step. Since affinity is EMA-smoothed,
measuring every 10 steps is still stable. This setting alone reduces
gradient-extraction overhead to $1/10$. The value was added after
every-step extraction during warmup caused hangs.

*TF32 + cuDNN benchmark (not torch.compile).* `torch.compile` is
disabled project-wide. The combination of 15-task MTL + `retain_graph`
+ dynamic shapes produces hundreds of kernel compilations and makes
the first epoch take 30+ minutes. Instead we get 10–15% speedup via
TF32 + cuDNN benchmark.

AMP (fp16) is on by default — ~40% memory saving, ~20% speedup. But
focal loss is explicitly cast to float32 — in fp16, intermediate
products of `focal_weight * bce` can drop into subnormal range and
produce NaN (the M-2/M-3 FIX).

## Decision 6 — Gradient Accumulation and NaN Defence

Finally, training stability. Gradient clipping is set at
`clip_grad_norm_=5.0`, and `gradient_accumulation_steps=1` so
effective batch equals `batch_size`. A `math.isfinite(loss_val)` check
skips the batch and runs `optimizer.zero_grad()` on NaN / Inf loss to
prevent contaminated gradients. OOM is handled by the exception
handler in `trainer.py`, which also skips the batch.

---

That closes the six decisions that plug adaTT into the real pipeline.
The Trainer's 2-Phase and adaTT's internal 3-Phase are two orthogonal
schedules; Uncertainty Weighting runs before adaTT's transfer;
per-expert lr and SequentialLR distribute learning rates across
experts; CGC-adaTT synchronised freeze cleans up convergence;
`grad_interval=10` and TF32 defend performance; NaN guards defend
stability. Full parameter listings, debugging guide, and the
mathematical appendix (EMA convergence, Bayesian conjugacy, PCGrad
comparison, etc.) live in the PDF below.

## Download the Full adaTT Tech Reference

Between ADATT-1 and ADATT-4, I have walked through the on-prem
`기술참조서/adaTT_기술_참조서` in blog form: motivation, mathematical
foundations, affinity measurement, Transfer Loss, Group Prior, 3-Phase
Schedule, Negative Transfer blocking, training loop, loss weighting,
optimizer, and CGC synchronisation. The original PDF is a longer
reference document that preserves typesetting, indexing, and all
equation proofs. The debugging guide, the full settings-parameter
index, and the mathematical appendix proofs (A.1 EMA convergence, A.2
Bayesian interpretation of Group Prior, A.3 Softmax temperature, A.4
theoretical basis for Negative Transfer blocking, A.5 convergence
impact of Transfer-Enhanced Loss) I trimmed from the blog all live
there.

> **📄 [Download the full adaTT Tech Reference (PDF)](/adaTT_기술_참조서.pdf)** · KO
>
> Adaptive Task Transfer · Gradient Cosine Similarity · Transfer
> Loss · 3-Phase Schedule · Negative Transfer Detection — if you want
> the entire adaTT content of this project in one document, grab it at
> the link above.

## End of the adaTT Sub-Thread, Next Are the Heterogeneous Shared Experts

This is the end of the adaTT sub-thread. Each post reads as a chain
picking up the problem left by its predecessor.

- **ADATT-1**: what CGC in the feature path could not solve — gradient
  conflict. Why an adaptive tower?
- **ADATT-2**: the four decisions of measurement — gradients, cosine,
  EMA, `torch.compiler.disable`. The `TaskAffinityComputer` engine
  completed.
- **ADATT-3**: what to do with the measured affinity — Transfer Loss,
  Group Prior, 3-Phase Schedule, Negative Transfer blocking.
- **ADATT-4**: six decisions to plug the design into the real training
  loop — 2-Phase training, Uncertainty Weighting ordering, per-expert
  lr, CGC-adaTT synchronisation, memory / performance, NaN defence.

Six PLE posts and four adaTT posts — ten Study Thread posts total —
cover the MTL backbone of this project in blog form. PLE separated
task conflicts in the *feature path*; adaTT measured the remaining
conflicts in the *gradient path* and turned them back into cooperation
— two sub-threads taking on two faces of the same MTL problem.

> **Open experimental result — adaTT removal under consideration.**
> As flagged in ADATT-1: on the synthetic-data benchmark, PLE+adaTT
> shows no clear performance gap over PLE-only. The same comparison
> is now running on real data (card transaction logs). If the
> result reproduces, *removing adaTT* from the stack is the plan.
> These four posts stay in either case — a record of "why we tried
> this design, and the basis on which we pulled it" is, for the next
> person, a map marked 'already tried here, move on'. (Update: the
> real-data results will be shared in a separate post.)

From here, we move on to the mathematical foundations of each of the
seven heterogeneous Shared Experts. Sub-threads will open for CausalOT
(causal inference + optimal transport), TDA (topological data
analysis / PersLay), Temporal (Mamba + LNN + Transformer), and the
Economics-feature-based Expert in that order.
