---
title: "[Study Thread] ADATT-3 — Transfer Loss, Group Prior, and the 3-Phase Schedule"
date: 2026-04-20 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, transfer-loss, group-prior, schedule, negative-transfer]
lang: en
series: study-thread
part: 9
alt_lang: /2026/04/20/adatt-3-transfer-loss-group-prior-schedule-ko/
next_title: "ADATT-4 — Training Loop, Loss Weighting, Optimizer, and CGC Sync (+ Tech Reference PDF)"
next_desc: "2-Phase Training Loop, Loss Weighting strategies (Uncertainty / GradNorm / DWA), Optimizer and Scheduler configuration, CGC-adaTT synchronization, memory and performance optimization, debugging guide, full settings reference, and appendix — closing the adaTT sub-thread with a downloadable PDF of the full adaTT tech reference."
next_status: published
---

*Part 3 of the adaTT sub-thread in the "Study Thread" series. Across ADATT-1 → ADATT-4, in parallel Korean and English, I unpack the adaTT mechanism used in this project. The source is the on-prem project's `기술참조서/adaTT_기술_참조서` (adaTT Tech Reference). This Part 3 covers the full formula of adaTT's core Transfer Loss term and its transfer-weight computation, the G-01 FIX Transfer Loss Clamp, masking for tasks whose targets are missing in a batch, the task-group-based Prior matrix with Prior Blend Annealing, the transition logic of the 3-Phase Schedule (Warmup → Dynamic → Frozen), and finally the Negative Transfer detection and blocking mechanism.*

## 4. Transfer Loss Computation

`compute_transfer_loss()` is the heart of adaTT: for each task, it adds a transfer loss contributed by the other tasks on top of that task's original loss.

> Source: `adatt.py:283-353` — `compute_transfer_loss()` method

### 4.1 The Full Formula

The Transfer-Enhanced Loss for each task $i$ is:

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \rightarrow j} \cdot \mathcal{L}_j$$

- $\mathcal{L}_i$: task $i$'s original loss (focal, huber, MSE, etc.)
- $\lambda = 0.1$ (default, the `transfer_lambda` parameter)
- $w_{i \rightarrow j}$: transfer weight from task $j$ into task $i$ (softmax-normalized)

> **Intuition.** This formula says "each task should not only look at its own loss, but also partially consult the losses of other tasks it is affine to." $\lambda = 0.1$ means "weight other tasks' opinions at 10%," and $w_{i \rightarrow j}$ decides "whose opinion matters more." Tasks whose gradients point in similar directions receive larger weights, so they accelerate each other's learning.

### 4.2 Transfer Weight Computation in Detail

The transfer weight $w_{i \rightarrow j}$ is a composition of several terms.

```python
# adatt.py:355-396 — _compute_transfer_weights()
raw_weights = self.transfer_weights + affinity  # learnable + affinity

# Blend with Group Prior (annealing)
raw_weights = raw_weights * (1 - r) + self.group_prior * r

# Block negative transfer
raw_weights = torch.where(
    affinity > self.neg_threshold,  # -0.1
    raw_weights,
    torch.zeros_like(raw_weights),
)

# Zero the diagonal (no self-transfer)
raw_weights = raw_weights.masked_fill(self.diag_mask, 0.0)

# Softmax normalization
weights = F.softmax(raw_weights / max(self.temperature, 1e-6), dim=-1)
```

Mathematically:

$$\mathbf{R} = (\mathbf{W} + \mathbf{A}) \cdot (1 - r) + \mathbf{P} \cdot r$$

$$\mathbf{R}_{i,j} \leftarrow 0 \quad \text{if } \mathbf{A}_{i,j} < \tau_{\text{neg}}$$

$$\mathbf{R}_{i,i} = 0$$

$$w_{i \rightarrow j} = \text{softmax}(\mathbf{R}_{i,j} / T)$$

- $\mathbf{W}$: learnable transfer weights (`nn.Parameter`, initialized at 0)
- $\mathbf{A}$: EMA affinity matrix
- $\mathbf{P}$: Group Prior matrix
- $r$: Prior blend ratio (varies across phases)
- $\tau_{\text{neg}} = -0.1$: negative-transfer cutoff
- $T = 1.0$: softmax temperature

> **Intuition.** This formula expresses a 4-stage pipeline for the transfer weight. First sum the learnable weight $\mathbf{W}$ and the measured affinity $\mathbf{A}$, then blend with domain knowledge (the Prior $\mathbf{P}$). Next, zero out "harmful transfers," exclude self-transfer, and finally turn everything into a probability distribution via softmax. Intuitively, it combines "task relationships observed from data + prior knowledge from domain experts," while cutting out paths that are harmful.

### 4.3 G-01 FIX: Transfer Loss Clamp

A ratio cap prevents the transfer loss from dominating the original loss.

```python
# adatt.py:346-351
raw_transfer = self.transfer_lambda * transfer_loss
if self.max_transfer_ratio > 0:
    max_val = original_loss.detach() * self.max_transfer_ratio
    raw_transfer = torch.clamp(raw_transfer, max=max_val)
enhanced_losses[task_name] = original_loss + raw_transfer
```

> **⚠ max_transfer_ratio = 0.5.** The transfer loss cannot exceed *50%* of the original loss (`adatt.py:191`). Without this cap, whenever a specific task's loss happens to be tiny the transfer loss becomes disproportionately large and warps the learning direction. Using `original_loss.detach()` ensures the clamp boundary does not influence the gradient.

### 4.4 Masking Tasks with Missing Targets

Not every batch contains targets for every task. Tasks whose targets are missing are removed from the transfer weighting.

```python
# adatt.py:321-334
loss_list = []
loss_mask = []
for name in self.task_names:
    if name in task_losses:
        loss_list.append(task_losses[name])
        loss_mask.append(1.0)
    else:
        loss_list.append(torch.tensor(0.0, device=affinity.device, requires_grad=False))
        loss_mask.append(0.0)

# ...
masked_transfer_w = transfer_w[i] * loss_mask_tensor
transfer_loss = (masked_transfer_w * loss_tensor).sum()
```

> **Why a mask and not a zero loss.** Simply inserting a 0.0 loss does not make the post-softmax weight go to zero. Multiplying by `loss_mask_tensor` *completely blocks* that transfer path. This is safe in production, where the set of active targets per batch is variable (especially when some tasks are inactive).

## 5. Group Prior Structure

The Group Prior is a matrix that encodes domain knowledge as a mathematical prior. During early training, before inter-task affinity has been measured reliably, it supplies a reasonable direction for transfer.

> Source: `adatt.py:256-281` — `_build_group_prior()` method

### 5.1 Task Group Definitions

`model_config.yaml:611-628` defines four groups.

| Group | Members | Intra strength | Business meaning |
|---|---|---|---|
| engagement | ctr, cvr, engagement, uplift | 0.8 | Customer engagement / conversion |
| lifecycle | churn, retention, life_stage, ltv | 0.7 | Customer lifecycle |
| value | balance_util, channel, timing | 0.6 | Customer value / behavior patterns |
| consumption | nba, spending_category, consumption_cycle, spending_bucket, merchant_affinity, brand_prediction | 0.7 | Consumption pattern analysis |

`inter_group_strength: 0.3` — cross-group transfer is kept deliberately low.

### 5.2 Building the Prior Matrix

```python
# adatt.py:256-281
def _build_group_prior(self) -> torch.Tensor:
    # 1. Initialize with inter-group transfer strength
    prior = torch.ones(self.n_tasks, self.n_tasks) * self.inter_group_strength  # 0.3

    # 2. Set intra-group transfer strength
    for group_name, members in self.task_groups.items():
        strength = self.intra_group_strength.get(group_name, 0.5)
        indices = [self.task_names.index(m) for m in members if m in self.task_names]
        for i in indices:
            for j in indices:
                if i != j:
                    prior[i, j] = strength

    # 3. Diagonal = 0 (no self-transfer)
    prior.fill_diagonal_(0.0)

    # 4. Row normalization
    row_sums = prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
    prior = prior / row_sums
```

> **What row normalization means.** Row normalization forces the sum of transfer weights that each task $i$ receives from the others to be 1. The effect is analogous to softmax: transfer intensity stays consistent regardless of how many tasks there are.

### 5.3 Prior Blend Annealing

The blend ratio $r$ decreases linearly as training progresses:

$$r(e) = r_{\text{start}} - (r_{\text{start}} - r_{\text{end}}) \cdot \min\left(\frac{e - e_{\text{warmup}}}{e_{\text{freeze}} - e_{\text{warmup}}}, 1.0\right)$$

- $r_{\text{start}} = 0.5$: early-training prior weight (`prior_blend_start`, `model_config.yaml:607`)
- $r_{\text{end}} = 0.1$: late-training prior weight (`prior_blend_end`, `model_config.yaml:608`)
- $e_{\text{warmup}}$: end of warmup
- $e_{\text{freeze}}$: start of freeze

> **Intuition.** This formula says "as training advances, depend less on the domain expert's prior ($\mathbf{P}$) and trust the observed affinity from actual data more." As $r$ drops from 0.5 to 0.1, the prior's share falls from 50% to 10% while the observed-data share rises from 50% to 90%. Intuitively, this is a new employee who initially leans on seniors' opinions (the Prior), then trusts their own judgment (observation) more as experience accumulates.

This annealing can be read, from a *Bayesian* perspective, as a transition from prior to posterior: early on, data is scarce so we lean on domain knowledge (prior); as data accumulates, we trust the learned gradient-based affinity (likelihood).

> **Historical background — Origins of the Bayesian-weight idea.** The idea of blending a prior with data goes back to Thomas Bayes (1763, posthumous) and Pierre-Simon Laplace (1812, *Theorie analytique des probabilites*). The Bayesian perspective was introduced to neural networks by pioneers like MacKay (1992, *"A Practical Bayesian Framework for Backpropagation Networks"*) and Neal (1996, *"Bayesian Learning for Neural Networks"*). In modern deep learning, Dropout has been reinterpreted as approximate Bayesian inference (Gal & Ghahramani, ICML 2016), and Weight Uncertainty (Blundell et al., ICML 2015, *"Bayes by Backprop"*) directly learns uncertainty over weights. adaTT's Prior Blend Annealing is a *pragmatic, lightweight version* of this Bayesian tradition — instead of inferring a full Bayesian posterior, it mimics the prior-to-posterior transition with a single blend ratio $r$.

The Prior Blend Annealing schedule flows from **r = 0.5 (high Prior dependence)** → linear decrease (start of Phase 2) → **r = 0.1 (affinity trusted, Phase 3 entered)**.

## 6. The 3-Phase adaTT Schedule

adaTT splits training into three phases to control how affinity is measured and how transfer is applied.

> Source: `adatt.py:298-313` — phase branching inside `compute_transfer_loss()`

> **Historical background — Lineage of training schedules.** The idea of "running training in stages" was systematized in Bengio et al. (2009, *"Curriculum Learning"*) — the observation that learning becomes more effective when examples go from easy to hard. That idea was extended by (1) Pre-training + Fine-tuning (Erhan et al., 2010), (2) Layer-wise Training (Hinton et al., 2006, Deep Belief Networks), and (3) Warmup-then-Decay learning-rate schedules (Goyal et al., 2017). adaTT's 3-Phase (Warmup → Dynamic → Frozen) applies this tradition to *inter-task transfer*: Phase 1 observes task relationships (the "exploration" stage of curriculum), Phase 2 exploits them (the main body of training), Phase 3 stabilizes (the "freeze" stage of fine-tuning).

### 6.1 Phase 1: Warmup (Affinity Measurement Only)

```python
# adatt.py:300-304
if epoch < self.warmup_epochs:
    if task_gradients is not None:
        self.affinity_computer.compute_affinity(task_gradients)
    return task_losses  # return original losses unchanged
```

In Phase 1, we compute gradient cosine similarity and accumulate the affinity matrix, but *we do not add the transfer loss*. The original `task_losses` are returned untouched.

- **Period**: epoch 0 to `warmup_epochs` (10 for production, 0 for tests)
- **Purpose**: if transfer starts before enough affinity data has accumulated, random transfer destabilizes training
- **Config**: `model_config.yaml:598` — `warmup_epochs: 0` (test); recommended 10 in production

### 6.2 Phase 2: Dynamic Transfer

```python
# adatt.py:311-317
# Phase 2: dynamic transfer
if task_gradients is not None:
    self.affinity_computer.compute_affinity(task_gradients)

affinity = self.affinity_computer.get_affinity_matrix()
transfer_w = self._compute_transfer_weights(affinity)
```

In Phase 2, affinity is updated every step while the transfer loss is simultaneously applied. The Prior blend ratio $r$ decreases linearly from `prior_blend_start` to `prior_blend_end`.

- **Period**: `warmup_epochs` to `freeze_epoch`
- **Learnable parameters**: `self.transfer_weights` (`nn.Parameter`, `adatt.py:229-231`)
- **Prior blend annealing**: $r$ decreases from 0.5 to 0.1 (`adatt.py:373-379`)

### 6.3 Phase 3: Frozen (Weights Fixed)

```python
# adatt.py:307-308
if self.is_frozen:
    return self._apply_frozen_transfer(task_losses)
```

In Phase 3, transfer weights are fixed and gradients are no longer computed for them. `_apply_frozen_transfer` uses `transfer_w[i].detach()` (`adatt.py:425`).

- **Period**: `freeze_epoch` through end of training
- **Config**: `model_config.yaml:599` — `freeze_epoch: 1` (test); recommended 28 in production
- **Effect**: removes gradient overhead for transfer weights and stabilizes training

> **⚠ H-6 validation: freeze_epoch > warmup_epochs.** `adatt.py:219-223` raises a `ValueError` if `freeze_epoch <= warmup_epochs`. If Phase 2 is completely skipped, no learned affinity ever gets reflected in transfer, so running adaTT becomes pointless. This check catches configuration errors early.

### 6.4 The Phase Transition Trigger: on_epoch_end

```python
# adatt.py:431-452
def on_epoch_end(self, epoch: int) -> None:
    self.current_epoch.fill_(epoch)

    if self.freeze_epoch is not None and epoch >= self.freeze_epoch:
        if not self.is_frozen.item():
            self.is_frozen.fill_(True)
            logger.info(f"adaTT: transfer weights frozen (epoch {epoch})")
```

> **Why fill_().** Reassigning with `self.current_epoch = epoch` replaces the plain tensor and severs its connection with the buffer registered via `register_buffer`. `fill_()` is an in-place update that preserves `state_dict` and device management. `is_frozen` uses `fill_(True)` for the same reason (`adatt.py:441`).

## 7. Negative Transfer Detection and Blocking

### 7.1 What Is Negative Transfer

When two tasks' gradients point in *opposite directions*, an update of the shared parameters that improves one task *degrades* the other.

For instance, if CTR (click-through rate) and Churn have gradients in opposite directions, learning in the direction that improves CTR will worsen Churn prediction.

### 7.2 The Blocking Mechanism

Inside `_compute_transfer_weights()`, transfer paths whose affinity falls below the threshold are zeroed out.

```python
# adatt.py:383-388
raw_weights = torch.where(
    affinity > self.neg_threshold,     # -0.1 (default)
    raw_weights,                       # keep
    torch.zeros_like(raw_weights),     # block
)
```

$$\mathbf{R}_{i,j} = \begin{cases} \mathbf{R}_{i,j} & \text{if } \mathbf{A}_{i,j} > \tau_{\text{neg}} \\ 0 & \text{otherwise} \end{cases}$$

- $\tau_{\text{neg}} = -0.1$ (`negative_transfer_threshold`, `model_config.yaml:600`)

> **Intuition.** This formula is a kind of *gate*. If affinity is above the threshold, the transfer weight passes through as-is; if below, it is completely blocked. Intuitively, it is a safety valve that says "I will simply ignore advice from a task whose direction is opposite to mine."

> **Why the threshold is -0.1 (not 0).** Cosine similarity of 0 means "orthogonal" (unrelated), and a weak negative correlation can be noise. Setting it to $-0.1$ allows mild negative correlation while blocking only clearly-opposite gradients. A threshold of 0 would block so many paths that adaTT's effect would be diluted.

> **State of the art — Recent evolution in negative-transfer mitigation (2023–2025).** Negative transfer is a central MTL challenge and is being actively researched. (1) *Aligned-MTL (Senushkin et al., CVPR 2023)* proposes a refined projection scheme that aligns gradients to a common descent direction while preserving per-task contributions. (2) *ForkMerge (Ye et al., NeurIPS 2023)* dynamically forks and merges tasks during training, automatically detecting and avoiding intervals of negative transfer. (3) *Auto-Lambda (Liu et al., ICLR 2022)* uses meta-gradients on a validation set to automatically tune task weights, indirectly mitigating negative transfer. Compared with these, adaTT's threshold-based blocking ($\tau_{\text{neg}} = -0.1$) is *extremely cheap yet effective*, making it practitioner-friendly. Natural extensions include an adaptive threshold ($\tau_{\text{neg}}$ tuned to the training stage) or soft gating (continuous attenuation instead of binary blocking).

### 7.3 Negative Transfer Diagnostic API

```python
# adatt.py:460-476
def detect_negative_transfer(self) -> Dict[str, List[str]]:
    affinity = self.affinity_computer.get_affinity_matrix()
    negative_pairs = {}
    for i, task_i in enumerate(self.task_names):
        neg_list = []
        for j, task_j in enumerate(self.task_names):
            if i != j and affinity[i, j] < self.neg_threshold:
                neg_list.append(task_j)
        if neg_list:
            negative_pairs[task_i] = neg_list
    return negative_pairs
```

Example return value: `{"churn": ["ctr", "engagement"], "ltv": ["brand_prediction"]}` — letting you see which task pairs exhibit negative transfer.

### 7.4 Impact of Blocking on Training

| Situation | Effect |
|---|---|
| Blocking disabled | Task pairs with negative transfer inflate each other's loss, destabilizing training |
| Over-blocking ($\tau_{\text{neg}} = 0$) | Most transfer paths are blocked → adaTT is effectively disabled |
| Proper blocking ($\tau_{\text{neg}} = -0.1$) | Only clear negative transfer is blocked; neutral/positive transfers are preserved |

Transfer Loss, Group Prior, the 3-Phase Schedule, and Negative Transfer blocking — the four layers interlock. The Prior fills the initially-empty affinity matrix with domain knowledge; the phase schedule enforces an observe → transfer → freeze curriculum; and negative-transfer blocking severs harmful paths when measured affinity dips into the negative regime. The G-01 FIX Clamp caps the whole thing so the transfer term never overwhelms the original loss. **ADATT-4** carries this structure into the actual training loop, loss-weighting strategies, optimizer settings, and the synchronization contract with CGC.
