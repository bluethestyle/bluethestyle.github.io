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

*The final (4th) post of the adaTT sub-thread of the "Study Thread"
series. Across ADATT-1 through ADATT-4, in parallel KO/EN, I've walked
through the adaTT mechanism that sits under this project. The source is
the on-prem `기술참조서/adaTT_기술_참조서` (adaTT Tech Reference). This
fourth post covers how adaTT actually plugs into the real training
pipeline — the 2-Phase Training Loop, loss-weighting strategy
(Uncertainty · GradNorm · DWA), Optimizer / Scheduler setup, CGC-adaTT
synchronization, memory and performance optimization, the debugging
guide, the full config-parameter index, and the appendix — and the full
PDF is downloadable at the bottom of this post.*

## 2-Phase Training Loop

The Trainer splits the whole training run into two phases. adaTT is
*active only in Phase 1* and is disabled in Phase 2. Source:
`trainer.py:456-542` — `train()`, `trainer.py:580-654` —
`_train_phase()`.

### Phase 1 — Shared Expert Pretrain

- *Duration*: `shared_expert_epochs` (default 15, `model_config.yaml:793`)
- *Training target*: the whole model — Shared Experts, CGC, Task
  Experts, Task Towers
- *adaTT*: active — gradient extraction and transfer loss applied

```python
# trainer.py:483-488
logger.info("=== Phase 1: Shared Expert training ===")
self._train_phase(
    train_loader, val_loader,
    max_epochs=self.config.shared_expert_epochs,
    phase_name="phase1",
)
```

By the end of Phase 1, adaTT has accumulated enough affinity data. If
`freeze_epoch` falls inside Phase 1, the transfer weights are already
frozen in the later part of Phase 1.

### Phase 2 — Cluster Finetune

- *Duration*: `cluster_finetune_epochs` (default 8, `model_config.yaml:794`)
- *Training target*: per-cluster Task Expert subheads only
- *adaTT*: *disabled* — Shared Experts are frozen, so gradient
  extraction is pointless

```python
# trainer.py:493-496
_adatt_backup = self.model.adatt
if self.model.adatt is not None:
    self.model.adatt = None
    logger.info("adaTT disabled: Phase 2 (Shared frozen → affinity invalid)")
```

> **Why disable adaTT in Phase 2.** adaTT's gradients are computed with
> respect to *Shared Expert parameters* (`ple_cluster_adatt.py:1872`).
> If Shared Experts are frozen in Phase 2, those gradients are zero, so
> cosine similarity is meaningless. Calling `autograd.grad` on frozen
> parameters is also wasted compute.

### Reset on Phase transition

`_setup_phase2()` (`trainer.py:544-578`) resets the following items.

| Reset item | Reason |
|---|---|
| Optimizer | Shared Expert frozen → momentum must be re-initialized (avoid stale momentum) |
| Scheduler | Phase-2-only warmup (2 epochs, shorter than Phase 1's 5) |
| GradScaler | AMP scaler state reset (loss scale changes across phase) |
| Early stopping | `best_val_loss`, `patience_counter` both reset |
| CGC Attention | Shared frozen → CGC gating frozen together |

```python
# trainer.py:544-578
def _setup_phase2(self):
    if self.config.freeze_shared_in_phase2:
        self.model.freeze_shared_experts()
    # Optimizer reset
    self.optimizer = self._create_optimizer()
    # Scheduler reset (Phase 2 warmup)
    self.config.warmup_steps = self.config.phase2_warmup_steps  # 2 epochs
    self.scheduler = self._create_scheduler()
    # Early stopping reset
    self.best_val_loss = float("inf")
    self.patience_counter = 0
```

### Guaranteed adaTT restore

After Phase 2 finishes, adaTT is *always restored* for checkpoint /
inference compatibility:

```python
# trainer.py:504-508
finally:
    self.model.adatt = _adatt_backup  # restore even on exception
    if self.config.freeze_shared_in_phase2:
        self.model.unfreeze_shared_experts()
```

Wrapping in a `finally` block keeps the model state consistent even if
an exception is raised.

## Loss Weighting Strategy

adaTT's transfer loss is *added on top of* the existing loss-weighting
strategy. The per-task loss weights and uncertainty weighting are
applied first, and then adaTT's transfer loss is added to that result.
Source: `ple_cluster_adatt.py:1607-1845` — `_compute_task_losses()`.

### Loss computation pipeline

1. *Decide per-task loss type*: focal, huber, MSE, NLL, contrastive,
   etc. (`ple_cluster_adatt.py:1656`)
2. *Apply focal loss alpha / gamma*: per-task positive-class weighting
   (`ple_cluster_adatt.py:1768-1780`)
3. *Apply loss weight*: per-task fixed weight or Uncertainty Weighting
   (`ple_cluster_adatt.py:1818-1830`)
4. *Add evidential loss*: auxiliary uncertainty-estimation loss
   (`ple_cluster_adatt.py:1832-1841`)
5. *adaTT transfer loss*: gradient-based transfer term added
   (`ple_cluster_adatt.py:1310-1316`)
6. *CGC entropy regularization*: prevents Expert collapse
   (`ple_cluster_adatt.py:1321-1329`)

### Uncertainty Weighting (Kendall et al., 2018)

> **Historical background — how Uncertainty Weighting was born.**
> Kendall, Gal & Cipolla (CVPR 2018, *"Multi-Task Learning Using
> Uncertainty to Weigh Losses for Scene Understanding"*) proposed
> automating the cost of hand-tuning per-task weights via
> *homoscedastic uncertainty*, in the context of joint semantic
> segmentation + depth + instance segmentation. The key idea: taking
> $-\log p$ of the Gaussian likelihood
> $p(y | f(x), \sigma) = \mathcal{N}(f(x), \sigma^2)$ naturally gives
> $\frac{1}{2\sigma^2} \cdot \|y - f(x)\|^2 + \log \sigma$, so *a task
> with large loss grows its $\sigma$ and its weight shrinks
> automatically*.

> **Undergrad math — why does $1/(2\sigma^2)$ appear?** Assume
> observation $y$ follows a normal $\mathcal{N}(\hat{y}, \sigma^2)$
> centered at prediction $\hat{y}$. The pdf is
> $p(y) = \frac{1}{\sigma \sqrt{2\pi}} \exp(-(y-\hat{y})^2 / (2\sigma^2))$.
> Taking the negative log gives
> $-\log p(y) = (y-\hat{y})^2 / (2\sigma^2) + \log \sigma + \text{const}$.
> Since $(y-\hat{y})^2$ is the loss $\mathcal{L}$, we naturally get
> $\mathcal{L}^{weighted} = \mathcal{L} / (2\sigma^2) + \log \sigma$.
> Re-parametrizing $\sigma^2 = \exp(\text{log\_var})$ gives
> $\log \sigma = \frac{1}{2} \cdot \text{log\_var}$, matching the code.
> The *precision* $1/\sigma^2$ is the effective weight, and the
> $\log \sigma$ term prevents the cheating solution of letting $\sigma$
> blow up to zero out the loss.

$$\mathcal{L}_i^{weighted} = \frac{1}{2 \sigma_i^2} \cdot \mathcal{L}_i + \frac{1}{2} \log \sigma_i^2$$

- $\sigma_i^2 = \exp(\text{log\_var}_i)$: learnable uncertainty of task
  $i$
- `log_var` clamp: \[-4.0, 4.0\], precision clamp: \[0.001, 100.0\]
  (`ple_cluster_adatt.py:1822-1823`)

> **Equation intuition.** This equation says "reflect high-uncertainty
> tasks with low weight, and low-uncertainty tasks with high weight".
> $1 / (2 \sigma_i^2)$ is precision — if $\sigma_i^2$ is large, weight
> shrinks. The $\frac{1}{2} \log \sigma_i^2$ term is a regularization
> penalty preventing the cheat of sending $\sigma_i^2$ to infinity.
> Intuitively: "be lenient about mistakes on tasks you barely
> understand, be strict about mistakes on tasks you know well".

This weight is applied *before* adaTT. That is, the `task_losses` input
to adaTT already has uncertainty weighting baked in.

### Per-task loss weights

Fixed per-task weights defined in `model_config.yaml`:

| Task | weight | loss type | Note |
|---|---|---|---|
| ctr | 1.0 | focal ($\gamma$=2, $\alpha$=0.25) | standard |
| cvr | 1.5 | focal ($\gamma$=2, $\alpha$=0.20) | very low positive rate → weight boosted |
| churn | 1.2 | focal ($\gamma$=2, $\alpha$=0.60) | high FN cost → alpha boosted |
| retention | 1.0 | focal ($\gamma$=2, $\alpha$=0.20) | high positive rate |
| nba | 2.0 | CE | 12 classes, business-critical |
| ltv | 1.5 | huber ($\delta$=1.0) | regression, outlier-robust |
| brand\_prediction | 2.0 | contrastive | InfoNCE, 50K brands |
| spending\_category | 1.2 | CE | 12 categories |
| rest | 0.8--1.0 | various | task-dependent |

### Interaction of adaTT and loss weights

adaTT's `compute_transfer_loss` receives `task_losses` *after* each
task's loss weight has been applied. So a task with high loss weight
(nba: 2.0) exerts a larger transfer effect on other tasks.

This is *intentional*: business-critical tasks should propagate their
learning signal to other tasks.

## Optimizer and Scheduler

Source: `trainer.py:220-334` — `_create_optimizer()`,
`_create_scheduler()`.

### AdamW Optimizer

```python
# trainer.py:236-242
trainable_params = [p for p in self.model.parameters() if p.requires_grad]
return torch.optim.AdamW(
    trainable_params,
    lr=self.config.learning_rate,      # 0.0005
    weight_decay=self.config.weight_decay,  # 0.01
)
```

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | 0.0005 | global LR (`model_config.yaml:752`) |
| `weight_decay` | 0.01 | L2 regularization strength |
| `gradient_clip_norm` | 5.0 | gradient magnitude cap (`model_config.yaml:780`) |
| `gradient_accumulation_steps` | 1 | effective batch = 16384 $\times$ 1 |

### Per-Expert Learning Rate

Each Shared Expert can have its own LR:

```python
# trainer.py:249-261
for expert_name, expert_module in self.model.shared_experts.items():
    expert_params = [p for p in expert_module.parameters() if p.requires_grad]
    cfg = expert_lr_config.get(expert_name, {})
    lr = cfg.get("lr", self.config.learning_rate)
    wd = cfg.get("weight_decay", self.config.weight_decay)
    param_groups.append({"params": expert_params, "lr": lr, "weight_decay": wd})
```

This feature is provided as a commented-out example in
`model_config.yaml:756-770`. For instance, `unified_hgcn` (learning in
hyperbolic space) needs a conservative LR, whereas `deepfm` can
converge quickly with a relatively higher LR.

> **Automatic exclusion in Phase 2.** If Shared Experts are frozen in
> Phase 2, they become `requires_grad=False` and are automatically
> skipped by `_create_optimizer` (`trainer.py:250`). This avoids
> allocating optimizer-state memory for frozen parameters.

### Learning Rate Scheduler — SequentialLR

Linear Warmup → CosineAnnealingWarmRestarts:

```python
# trainer.py:296-318
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    self.optimizer, start_factor=0.1, total_iters=warmup_steps  # 5 epochs
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    self.optimizer, T_0=self.config.cosine_t0,    # 10
    T_mult=self.config.cosine_t_mult,              # 2
)
return torch.optim.lr_scheduler.SequentialLR(
    self.optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],  # switch to cosine after 5 epochs
)
```

| Parameter | Default | Description |
|---|---|---|
| `warmup_steps` | 5 | linear warmup duration (epochs) |
| `cosine_t0` | 10 | first cosine period length (`model_config.yaml:773`) |
| `cosine_t_mult` | 2 | period multiplier (10 → 20 → 40 epochs) |
| `start_factor` | 0.1 | warmup start LR = 0.0005 $\times$ 0.1 = 0.00005 |

### Phase-2 dedicated Scheduler

At Phase 2 start, the scheduler is reset and warmup is shortened to 2
epochs:

```python
# trainer.py:562-566
original_warmup = self.config.warmup_steps
self.config.warmup_steps = self.config.phase2_warmup_steps  # 2
self.scheduler = self._create_scheduler()
self.config.warmup_steps = original_warmup  # restore
```

## CGC-adaTT Synchronization

CGC (Customized Gate Control) and adaTT are different mechanisms but
both act on the same Shared Expert parameters, so *synchronization* is
mandatory. Source: `ple_cluster_adatt.py:1921-1942` — `on_epoch_end()`
CGC freeze sync.

### Role of CGC

CGC learns which Shared Experts each task should attend to more. If
adaTT controls *inter-task knowledge transfer*, CGC controls *Expert
selection*. Letting them learn independently could push the parameters
in conflicting directions.

### Sync strategy — simultaneous Freeze

At adaTT's `freeze_epoch`, CGC Attention is frozen together:

```python
# ple_cluster_adatt.py:1931-1942
# v2.3: CGC freeze -- synced with adaTT freeze_epoch
freeze_epoch = self.config.get("adatt", {}).get("freeze_epoch")
if (freeze_epoch is not None
        and epoch >= freeze_epoch
        and self.task_expert_attention is not None
        and not self._cgc_frozen.item()):
    for param in self.task_expert_attention.parameters():
        param.requires_grad = False
    self._cgc_frozen.fill_(True)
    logger.info(f"CGC Attention frozen at epoch {epoch}")
```

> **Why freeze together.** If adaTT freezes transfer weights but CGC
> keeps learning, CGC would change Expert weights and invalidate the
> affinity relationships adaTT measured. E.g., if adaTT judged "CTR→CVR
> positive transfer" but CGC then changes CTR's Expert selection, the
> gradient direction changes and the frozen transfer weights are no
> longer valid.

### CGC freeze in Phase 2

At the start of Phase 2, CGC is also frozen (`trainer.py:549-555`):

```python
# trainer.py:549-555
if hasattr(self.model, '_cgc_frozen') and not self.model._cgc_frozen.item():
    if self.model.task_expert_attention is not None:
        for param in self.model.task_expert_attention.parameters():
            param.requires_grad = False
        self.model._cgc_frozen.fill_(True)
        logger.info("CGC Attention frozen in Phase 2")
```

Training CGC gating in Phase 2 (where Shared Experts are frozen) is
pointless: the input (Expert outputs) doesn't change, so gating
training would just overfit.

### \_cgc\_frozen state tracking

```python
# ple_cluster_adatt.py:376
self.register_buffer("_cgc_frozen", torch.tensor(False))
```

Registered as a buffer so the freeze state is preserved across
checkpoint save / restore. Freeze events are also logged in MLflow
(`trainer.py:604-608`).

## Memory and Performance Optimization

adaTT's gradient extraction is expensive. Because per-Shared-Expert
gradients are computed for each of 16 tasks, training speed drops
sharply without optimization.

### Memory impact of retain\_graph=True

```python
# ple_cluster_adatt.py:1865-1868
# G-04 NOTE: retain_graph=True is architecturally required.
# Memory impact: n_tasks x shared_param_size.
# 16-task case: ~2x peak memory vs forward pass.
```

`retain_graph=True` keeps the computation graph in memory even after
each task's `autograd.grad` call. Since we call it sequentially over 16
tasks, peak memory grows to roughly *2x* the forward pass.

| Element | Memory | Note |
|---|---|---|
| Forward pass graph | 1x (baseline) | needed for normal training |
| retain\_graph overhead | ~1x | graph not freed → extra memory |
| 16 task gradients | ~0.3x | each = shared\_param\_size |
| **Total** | **~2.3x** | batch\_size 16384 fits on RTX 4070 12GB |

### adatt\_grad\_interval

The key optimization: reduce gradient-extraction frequency to cut
compute cost.

```python
# ple_cluster_adatt.py:373
self.adatt_grad_interval = adatt_config.get("grad_interval", 10)

# ple_cluster_adatt.py:1299-1304
if self.training and self.adatt is not None:
    if self.global_step % self.adatt_grad_interval == 0:
        task_gradients = self._extract_task_gradients(task_losses)
```

> **Rationale for default 10.** Extracting gradients every step means
> 16 $\times$ `autograd.grad` calls per step. Affinity is smoothed by
> EMA, so measuring every 10 steps is still stable. Setting
> `grad_interval=10` reduces gradient computation overhead by *1/10*.
> Previously, extracting every step during warmup caused hangs
> (`ple_cluster_adatt.py:1300-1301`).

### torch.compiler.disable

```python
# ple_cluster_adatt.py:1847
@torch.compiler.disable
def _extract_task_gradients(self, task_losses):
```

`torch.compile` is disabled in this project (`trainer.py:174-177`). The
combination of 15-task MTL + adaTT `retain_graph` + dynamic shapes
produces hundreds of kernel compilations, making the first epoch take
*30+ minutes*. Instead we get ~10-15% speedup via TF32 + cuDNN
benchmark:

```python
# trainer.py:170-172
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### AMP (Automatic Mixed Precision)

The forward pass runs under `torch.amp.autocast` in fp16:

```python
# trainer.py:709-711
with autocast(device_type=self.device.type):
    outputs = self.model(inputs, compute_loss=True)
    loss = outputs.total_loss / self.config.gradient_accumulation_steps
```

adaTT gradient extraction runs inside autocast, but focal loss is
explicitly cast to float32 (`ple_cluster_adatt.py:1774`):

```python
# ple_cluster_adatt.py:1774
p_f = pred.squeeze().float().clamp(1e-7, 1 - 1e-7)
```

> **⚠ M-2 + M-3 FIX — fp16 focal loss safety.** In fp16,
> `log(1e-7) = -16.1` is fine, but intermediate products of
> `focal_weight * bce` can drop into the subnormal range and produce
> NaN. We run the whole focal-loss computation in float32 to avoid
> this.

### Gradient Accumulation

```python
# trainer.py:723-730
if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config.gradient_clip_norm  # 5.0
    )
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
```

Currently `gradient_accumulation_steps=1`, so we update every batch.
Effective batch size = 16384 $\times$ 1 = 16384.

## Debugging Guide

### Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| NaN loss | fp16 focal loss underflow | M-2/M-3: check float32 cast |
| Early loss divergence | transfer loss dominates original | G-01: check `max_transfer_ratio` (0.5) |
| RuntimeError in Phase 2 | adaTT not disabled | check `model.adatt = None` in trainer.py |
| `ValueError` init fail | `freeze_epoch <= warmup_epochs` | H-6: config validation |
| Training hangs | gradient extracted every step | check `adatt_grad_interval` (default 10) |
| Checkpoint mismatch | `fill_()` not used | check in-place ops for buffer update |

### Reading the affinity matrix

How to inspect the affinity matrix during training:

```python
# Get affinity matrix
affinity = model.adatt.affinity_computer.get_affinity_matrix()
print(affinity)  # [n_tasks, n_tasks]

# Detect negative transfer
neg_pairs = model.adatt.detect_negative_transfer()
print(neg_pairs)  # {"churn": ["ctr"], ...}

# Current transfer weight matrix
transfer_w = model.adatt.get_transfer_matrix()
print(transfer_w)  # softmax-normalized [n_tasks, n_tasks]
```

Properties of a healthy affinity matrix:

- Positive affinity within the same group ($> 0.3$)
- Weak positive or neutral between different groups ($-0.1 \sim 0.3$)
- Diagonal = 1.0
- *Matrix does not saturate at -1 or +1 globally* (saturation → tune
  EMA decay)

### Monitoring transfer loss

Key metrics logged in MLflow:

```python
# trainer.py:810-811 — step logging includes outputs.task_losses
self._log_step(outputs, phase_name)
```

Monitoring checklist:

- `task_losses/<task>`: per-task enhanced loss (incl. transfer loss)
- `adatt_freeze_epoch`: freeze time logged
- `cgc_frozen_epoch`: CGC freeze time logged
- Total loss = $\sum$ enhanced losses + CGC entropy + SAE + evidential

### Phase transition debugging

Things to verify on a phase transition:

1. *Phase 1 → Freeze*: log `adaTT: transfer weights frozen (epoch N)`
2. *Phase 1 → Phase 2*: log `adaTT disabled: Phase 2`
3. *CGC freeze*: log `CGC Attention frozen at epoch N`
4. *Optimizer reset*: log `Optimizer reset: Phase 2 start`

```python
# Example phase-transition logs
# INFO: adaTT: transfer weights frozen (epoch 28)
# INFO: CGC Attention frozen at epoch 28 (synced with adaTT freeze_epoch)
# INFO: === Phase 2: Cluster Head Fine-tuning ===
# INFO: Shared Experts frozen
# INFO: adaTT disabled: Phase 2 (Shared frozen → affinity invalid)
# INFO: Optimizer reset: Phase 2 start
# INFO: Scheduler reset: Phase 2 start (warmup=2 epochs)
```

### Preventing loss explosion

Diagnostic sequence when loss spikes:

1. *NaN check*: `_train_epoch` checks `math.isfinite(loss_val)`
   (`trainer.py:781-786`)
2. *Transfer loss ratio*: did it exceed `max_transfer_ratio=0.5`?
3. *Gradient norm*: was `gradient_clip_norm=5.0` applied?
4. *VRAM OOM*: `trainer.py:762-775` skips batch on OOM

```python
# trainer.py:781-786
loss_val = outputs.total_loss.item()
if not math.isfinite(loss_val):
    logger.error(f"NaN/Inf loss at batch {batch_idx}!")
    self.optimizer.zero_grad()  # avoid gradient contamination
    continue
```

## Full Config Parameter Index

### adaTT core parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `enabled` | `true` | bool | enable adaTT |
| `transfer_lambda` | 0.1 | \[0, 1\] | transfer loss weight $\lambda$ |
| `temperature` | 1.0 | (0, $\infty$) | softmax temperature $T$ |
| `warmup_epochs` | 0 (test) / 10 (prod) | \[0, max\_epochs) | Phase 1 duration |
| `freeze_epoch` | 1 (test) / 28 (prod) | (warmup, max\_epochs) | Phase 3 start |
| `negative_transfer_threshold` | -0.1 | \[-1, 0\] | negative transfer cut-off |
| `ema_decay` | 0.9 | \[0, 1\] | affinity EMA decay |
| `prior_blend_start` | 0.5 | \[0, 1\] | early-training group prior ratio |
| `prior_blend_end` | 0.1 | \[0, 1\] | late-training group prior ratio |
| `transfer_strength` | 0.5 | \[0, 1\] | logit transfer strength (CTR→CVR, etc.) |

Source: `model_config.yaml:593-628`.

### Task group parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `task_groups.<group>.members` | -- | list | member tasks of the group |
| `task_groups.<group>.intra_strength` | 0.5 | \[0, 1\] | intra-group transfer strength |
| `inter_group_strength` | 0.3 | \[0, 1\] | inter-group transfer strength |

### Training parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `batch_size` | 16384 | \[1024, 65536\] | training batch size |
| `learning_rate` | 0.0005 | \[1e-5, 1e-2\] | global LR |
| `weight_decay` | 0.01 | \[0, 0.1\] | L2 regularization |
| `shared_expert_epochs` | 15 | \[5, 100\] | Phase 1 duration |
| `cluster_finetune_epochs` | 8 | \[3, 50\] | Phase 2 duration |
| `freeze_shared_in_phase2` | `true` | bool | freeze Shared Expert in Phase 2 |
| `early_stopping_patience` | 7 | \[1, 20\] | early stopping patience |
| `gradient_clip_norm` | 5.0 | \[0.1, 20\] | gradient clipping threshold |
| `use_amp` | `true` | bool | use mixed precision (fp16) |
| `cosine_t0` | 10 | \[5, 50\] | CosineAnnealing first period (epochs) |
| `cosine_t_mult` | 2 | \[1, 4\] | CosineAnnealing period multiplier |
| `warmup_steps` | 5 | \[0, 20\] | LR warmup epochs |
| `phase2_warmup_steps` | 2 | \[0, 10\] | Phase 2 LR warmup epochs |

Source: `model_config.yaml:750-805`, `trainer.py:50-96`.

### Performance-optimization parameters

| Parameter | Default | Description |
|---|---|---|
| `adatt_grad_interval` | 10 | gradient extraction step interval. smaller → more accurate, larger → faster |
| `gradient_accumulation_steps` | 1 | gradient accumulation count. effective batch = batch\_size $\times$ this |
| `gradient_checkpointing` | `false` | unnecessary for a 9.25M model (off → 10-20% speedup) |
| `use_amp` | `true` | fp16 AMP. ~40% memory, ~20% speed |

### adaTT internal constants

Hardcoded in code — not configurable:

| Constant | Value | Location / description |
|---|---|---|
| `max_transfer_ratio` | 0.5 | `adatt.py:191` — max ratio of transfer loss vs original |
| `norm clamp min` | 1e-8 | `adatt.py:123` — prevent gradient norm div-by-zero |
| `diag_mask` | eye(n) | `adatt.py:239-242` — mask for excluding self-transfer |
| `affinity_matrix init` | eye(n) | `adatt.py:79` — diagonal 1, rest 0 (neutral start) |

## Appendix — Mathematical Proofs and Theoretical Foundations

### A.1 Convergence properties of EMA affinity

EMA update rule:

$$\mathbf{A}_t = \alpha \mathbf{A}_{t-1} + (1 - \alpha) \mathbf{C}_t$$

where $\mathbf{C}_t$ is the observed cosine similarity matrix at time
$t$.

> **Equation intuition.** This says "mix a small dose of the new
> observation $\mathbf{C}_t$ into the running memory
> $\mathbf{A}_{t-1}$". With $\alpha = 0.9$, each step retains 90% of
> the old value and reflects 10% of the new one. Like watching a
> moving-average trend line rather than day-to-day stock fluctuations —
> noise is filtered out while the real trend is captured.

> **Undergrad math — geometric series and EMA convergence.** The key
> tool in the proof below is the *geometric series sum*:
> $\sum_{k=0}^{n-1} \alpha^k = (1 - \alpha^n) / (1 - \alpha)$ (for
> $|\alpha| < 1$). As $n \to \infty$, $\alpha^n \to 0$ so
> $\sum_{k=0}^{\infty} \alpha^k = 1/(1-\alpha)$. Plugging in
> $\alpha = 0.9$ gives $1/(1-0.9) = 10$. In the proof,
> $(1-\alpha) \sum_{k=0}^{t-1} \alpha^k$ sums to
> $(1-\alpha) \cdot (1-\alpha^t)/(1-\alpha) = 1 - \alpha^t$. As
> $t \to \infty$, $\alpha^t \to 0$, so the weights sum to 1 and EMA
> converges exactly to the true mean $\mathbf{C}^*$.

**Proposition A.1**: If $\mathbf{C}_t$ follows a stationary
distribution and $\mathbb{E}[\mathbf{C}_t] = \mathbf{C}^*$, then

$$\mathbb{E}[\mathbf{A}_t] \to \mathbf{C}^* \quad \text{as } t \to \infty$$

**Proof**: Expand $\mathbf{A}_t$:

$$\mathbf{A}_t = (1 - \alpha) \sum_{k=0}^{t-1} \alpha^k \mathbf{C}_{t-k} + \alpha^t \mathbf{A}_0$$

As $t \to \infty$, $\alpha^t \mathbf{A}_0 \to 0$, and

$$\mathbb{E}[\mathbf{A}_t] = (1 - \alpha) \sum_{k=0}^{t-1} \alpha^k \mathbb{E}[\mathbf{C}_{t-k}] = (1 - \alpha) \cdot \mathbf{C}^* \cdot \frac{1 - \alpha^t}{1 - \alpha} = \mathbf{C}^* (1 - \alpha^t) \to \mathbf{C}^*$$

Hence EMA affinity converges to the *true affinity*. $\square$

**Variance**:
$\text{Var}(\mathbf{A}_t) \approx \frac{1 - \alpha}{1 + \alpha} \text{Var}(\mathbf{C}_t)$.
At $\alpha = 0.9$ this is $\approx 5.3\%$ of the original, giving high
stability.

### A.2 Bayesian interpretation of Group Prior

adaTT's Group Prior can be interpreted as a *Bayesian prior
distribution*.

> **Historical background — history of conjugate priors.** Raiffa &
> Schlaifer (1961, *"Applied Statistical Decision Theory"*)
> systematized the notion of a *conjugate family*: if prior and
> posterior live in the same family, the posterior has a closed-form
> solution. Gaussian Processes (Rasmussen & Williams, 2006), Bayesian
> Linear Regression, the Kalman Filter all use this conjugate
> structure.

> **Undergrad math — Normal-Normal conjugacy.** Prior:
> $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$. Observation:
> $x | \theta \sim \mathcal{N}(\theta, \sigma^2)$. The posterior is
> $\theta | x \sim \mathcal{N}(\mu_n, \sigma_n^2)$ with
> $\mu_n = (\sigma^2 \mu_0 + \sigma_0^2 x) / (\sigma^2 + \sigma_0^2)$.
> Rearranging, $\mu_n = r \cdot \mu_0 + (1-r) \cdot x$ with
> $r = \sigma^2 / (\sigma^2 + \sigma_0^2)$. In adaTT,
> $\mu_0 = \mathbf{P}$ (Group Prior),
> $x = \mathbf{W} + \mathbf{A}$ (learned weights + affinity), and
> $r$ decaying from 0.5 to 0.1 mimics "as more data accumulates, the
> influence of $\sigma^2$ shrinks".

**Model**: for the true affinity $a_{i,j}$ between tasks $i, j$,

$$a_{i,j} | \mathbf{P}, \sigma^2 \sim \mathcal{N}(\mathbf{P}_{i,j}, \sigma^2)$$

where $\mathbf{P}$ is the Group Prior matrix.

**Observation**: cosine similarity $c_{i,j} = \cos(\theta_{i,j})$,

$$c_{i,j} | a_{i,j}, \tau^2 \sim \mathcal{N}(a_{i,j}, \tau^2)$$

**Posterior mean** (conjugate normal):

$$\mathbb{E}[a_{i,j} | c_{i,j}] = \frac{\tau^2}{\sigma^2 + \tau^2} \mathbf{P}_{i,j} + \frac{\sigma^2}{\sigma^2 + \tau^2} c_{i,j}$$

With $r = \tau^2 / (\sigma^2 + \tau^2)$:

$$\mathbb{E}[a_{i,j}] = r \cdot \mathbf{P}_{i,j} + (1 - r) \cdot c_{i,j}$$

This *exactly matches* adaTT's prior-blend formula (`adatt.py:381`):

```python
raw_weights = raw_weights * (1 - r) + self.group_prior * r
```

### A.3 Role of softmax temperature

$$w_{i \to j} = \frac{\exp(\mathbf{R}_{i,j} / T)}{\sum_{k \neq i} \exp(\mathbf{R}_{i,k} / T)}$$

- $T \to 0$: concentrate on the highest weight (hard selection)
- $T \to \infty$: uniform distribution (all tasks equal)
- $T = 1.0$ (default): middle ground, reflects affinity differences
  appropriately

Why we use $T = 1.0$ in this system: with 16 tasks, too sharp
($T < 0.5$) would concentrate transfer on just a few tasks, while too
uniform ($T > 2.0$) would fail to suppress negative transfer.

> **Modern trend — softmax temperature (2023-2025).** (1) *Knowledge
> Distillation*: DKD (Zhao et al., CVPR 2022) separates target vs
> non-target class logits with different temperatures. (2) *LLM
> Decoding*: generation temperature controls the *creativity vs
> accuracy* trade-off. (3) *Contrastive Learning*: SimCLR uses low
> $T = 0.07$ to concentrate on hard negatives. (4) *Gumbel-Softmax*
> (Jang et al., ICLR 2017): anneal $T$ during training to approximate
> discrete selection in a differentiable way.

### A.4 Theoretical basis for blocking negative transfer

Yu et al. (2020)'s PCGrad defines gradient conflict as
$\cos(\theta_{i,j}) < 0$. adaTT uses a looser threshold
$\tau_{neg} = -0.1$ to account for *noise margin*.

> **Undergrad math — vector projection and gradient conflict.**
> PCGrad's core operation is vector projection. Projecting $\mathbf{a}$
> onto $\mathbf{b}$ gives
> $\text{proj}_{\mathbf{b}} \mathbf{a} = (\mathbf{a} \cdot \mathbf{b}) / (\mathbf{b} \cdot \mathbf{b}) \cdot \mathbf{b}$.
> When $\cos \theta < 0$ (conflict), PCGrad *removes* the
> $\mathbf{g}_j$-direction component from $\mathbf{g}_i$:
> $\mathbf{g}_i' = \mathbf{g}_i - \text{proj}_{\mathbf{g}_j} \mathbf{g}_i$.
> adaTT takes a simpler and more conservative stance — instead of
> rewriting gradients, it *blocks the transfer itself*, zeroing out the
> harmful task's loss contribution without modifying gradients.

Due to SGD's stochastic noise, even task pairs with true affinity near
zero can show weak negative correlations like
$\cos(\theta_{i,j}) \approx -0.05$ in a given batch. $\tau_{neg} = -0.1$
is the sweet spot that tolerates this noise while still blocking
*clear* negative transfer.

### A.5 Convergence impact of transfer-enhanced loss

**Proposition A.5**: The gradient of transfer loss biases the shared
parameters toward positive-transfer tasks' learning directions.

$$\nabla_\theta \mathcal{L}_i^{adaTT} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$$

> **Equation intuition.** This says "task $i$'s parameter-update
> direction is its own gradient plus a correction vector that is the
> weighted sum of other tasks' gradients". Intuitively, I'm walking in
> my own direction, and my teammates whisper "this way is better" with
> a vector — I adjust my path by a tiny amount $\lambda = 0.1$ of that
> suggestion. Teammates with high affinity (large $w_{i \to j}$) are
> weighted more.

For task $j$ with $w_{i \to j} > 0$ and $\cos(\theta_{i,j}) > 0$, its
gradient is *added* in the direction of task $i$'s gradient, updating
shared parameters in a direction beneficial to both tasks.

The combination $\lambda = 0.1$ and `max_transfer_ratio=0.5` is a
conservative setting that gains the benefit of positive transfer
without greatly distorting the original loss's learning direction.

> **Cross-reference — relationship with Logit Transfer.** adaTT
> regulates inter-task transfer at the *backward-pass* (gradient)
> level, whereas this system also has a separate *Logit Transfer*
> mechanism operating at the *forward-pass* (logit) level, directly
> passing prediction values between tasks. Logit Transfer explicitly
> designs sequential business-logic dependencies like CTR→CVR→LTV as a
> one-way DAG, and it works complementarily with adaTT's omnidirectional
> adaptive transfer.

## Download the full adaTT Tech Reference

Between ADATT-1 and ADATT-4, I've gone through the on-prem
`기술참조서/adaTT_기술_참조서` (adaTT Tech Reference) in blog form:
motivation, math foundations, affinity measurement, transfer loss,
group prior, 3-phase schedule, negative transfer blocking, training
loop, loss weighting, optimizer, CGC synchronization, debugging guide —
and the original PDF is a longer reference document where the
equations, diagrams, and index are all preserved.

> **📄 [Download the full adaTT Tech Reference (PDF)](/adaTT_기술_참조서.pdf)** · KO
>
> Adaptive Task Transfer · Gradient Cosine Similarity · Transfer
> Loss · 3-Phase Schedule · Negative Transfer Detection — if you want
> the entire adaTT content of this project in one document, grab it at
> the link above.

## End of the adaTT sub-thread, next are the heterogeneous Shared Experts

This is the end of the adaTT sub-thread. From ADATT-1 (motivation for
adaptive towers and the Transformer Attention analogy), ADATT-2
(gradient cosine-similarity-based affinity measurement), ADATT-3
(Transfer Loss · Group Prior · 3-Phase Schedule · negative transfer
blocking), and this ADATT-4 (2-Phase Training Loop · Loss Weighting ·
Optimizer · CGC synchronization · memory optimization · debugging
guide · appendix) — 6 PLE posts and 4 adaTT posts, a total of 10 Study
Thread posts, cover the MTL backbone of this project in blog form.

From here, we move on to the mathematical foundations of each of the 7
heterogeneous Shared Experts. Sub-threads will open for CausalOT
(causal inference + optimal transport), TDA (topological data
analysis / PersLay), Temporal (Mamba + LNN + Transformer), and the
Economics-feature-based Expert in that order.
