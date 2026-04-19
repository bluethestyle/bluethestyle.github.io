---
title: "[Study Thread] ADATT-2 — TaskAffinityComputer and Gradient Cosine Similarity"
date: 2026-04-20 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, gradient, cosine-similarity, ema]
lang: en
series: study-thread
part: 8
alt_lang: /2026/04/20/adatt-2-task-affinity-gradient-cosine-ko/
next_title: "ADATT-3 — Transfer Loss, Group Prior, and the 3-Phase Schedule"
next_desc: "adaTT's Transfer Loss calculation (full formula + transfer weights, the G-01 FIX clamp, target-task masking), the task-group-based Prior matrix with Prior Blend Annealing, the 3-Phase Schedule (Warmup → Dynamic → Frozen), and the Negative Transfer detection/blocking mechanism."
next_status: published
---

*Part 2 of the adaTT sub-thread inside the "Study Thread" series. Across ADATT-1 → ADATT-4, in parallel Korean/English, I unpack the adaTT mechanism behind this project. The source is the on-prem `기술참조서/adaTT_기술_참조서` reference. This part answers how adaTT actually measures inter-task affinity — the class structure of TaskAffinityComputer, how compute_affinity() extracts gradient cosine similarity, EMA smoothing, why cosine is preferred over Euclidean distance, and the gradient-extraction path wrapped by torch.compiler.disable.*

## 2. TaskAffinityComputer — the Core Affinity Engine

`TaskAffinityComputer` is the foundational module of adaTT: it computes gradient cosine similarity between tasks and maintains an EMA-smoothed (Exponential Moving Average) affinity matrix for learning stability.

Source: `adatt.py:58-143` — the `TaskAffinityComputer` class.

### 2.1 Class Structure and Initialization

```python
# adatt.py:58-84
class TaskAffinityComputer(nn.Module):
    def __init__(self, task_names: List[str], ema_decay: float = 0.9):
        super().__init__()
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.ema_decay = ema_decay

        # EMA affinity matrix (for training stability)
        self.register_buffer("affinity_matrix", torch.eye(self.n_tasks))
        self.register_buffer("update_count", torch.tensor(0))
```

> **Design choice — register_buffer.** Two reasons for registering `affinity_matrix` via `register_buffer`. (1) It is included in `state_dict`, so checkpoint save/restore is automatic. (2) It is moved with `.to(device)` automatically. By registering it as a buffer rather than an `nn.Parameter`, the optimizer does not treat this matrix as a learnable target.

The core state consists of two buffers.

| Buffer | Initial value | Purpose |
| --- | --- | --- |
| `affinity_matrix` | `eye(n_tasks)` | EMA-smoothed inter-task affinity matrix [n×n] |
| `update_count` | `0` | Number of EMA updates (used to skip blending on the first step) |

### 2.2 The compute_affinity() Method

This method takes per-task flattened gradient vectors, computes the cosine similarity matrix, and blends it with the existing affinity via EMA.

```python
# adatt.py:86-138
def compute_affinity(self, task_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 1. Collect gradients (zero-pad for missing tasks)
    grad_list = []
    for name in self.task_names:
        if name in task_gradients:
            grad_list.append(task_gradients[name])
        else:
            grad_list.append(torch.zeros_like(reference_grad))

    # 2. Stack: [n_tasks, grad_dim]
    grad_matrix = torch.stack(grad_list, dim=0)

    # 3. Cosine similarity matrix
    norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = grad_matrix / norms
    affinity = torch.mm(normalized, normalized.t())

    # 4. EMA update
    with torch.no_grad():
        if self.update_count > 0:
            new_affinity = self.ema_decay * self.affinity_matrix \
                         + (1 - self.ema_decay) * affinity
            self.affinity_matrix.copy_(new_affinity.clamp(-1.0, 1.0))
        else:
            self.affinity_matrix.copy_(affinity.clamp(-1.0, 1.0))
        self.update_count.add_(1)
```

> **Undergraduate math — computing all pairwise cosine similarities with a single matrix multiplication.** Stack the gradients of $n$ tasks row-wise into a matrix $\mathbf{G} \in \mathbb{R}^{n \times d}$. Dividing each row $\mathbf{g}_i$ by its L2 norm gives the normalized matrix $\hat{\mathbf{G}}$ (with $\hat{\mathbf{g}}_i = \mathbf{g}_i / \|\mathbf{g}_i\|$). Then the $(i,j)$ entry of $\hat{\mathbf{G}} \hat{\mathbf{G}}^\top$ is $\sum_{k=1}^d \hat{g}_{i,k} \hat{g}_{j,k} = \hat{\mathbf{g}}_i \cdot \hat{\mathbf{g}}_j = \cos \theta_{i,j}$. In other words, a *single matrix multiplication* yields all $n^2$ cosine similarities at once. A double for-loop would scatter the $O(n^2 d)$ work across $n^2$ Python calls and be very slow, whereas `torch.mm` leverages the GPU's parallel compute units (CUDA cores) and is hundreds of times faster. For 16 tasks, this is $16 \times 16 = 256$ similarities computed by a single GEMM (General Matrix Multiplication) kernel.

> **⚠ N-03 FIX: preventing floating-point error accumulation.** `.clamp(-1.0, 1.0)` must be applied after every EMA update (`adatt.py:132,135`). Across thousands of EMA updates, accumulated floating-point error can push cosine values outside the $[-1, 1]$ range; without this clamping, downstream operations such as `arccos` can produce NaN.

### 2.3 Interpreting the Affinity Matrix

The affinity matrix $\mathbf{A} \in [-1, 1]^{n \times n}$ is symmetric ($\mathbf{A}_{i,j} = \mathbf{A}_{j,i}$).

| Range | Meaning | adaTT behavior |
| --- | --- | --- |
| $\mathbf{A}_{i,j} \approx 1$ | Strong positive affinity | Gradients of tasks i, j aligned → aggressive transfer |
| $\mathbf{A}_{i,j} \approx 0$ | Neutral | No inter-task correlation → weak transfer |
| $\mathbf{A}_{i,j} < -0.1$ | Negative affinity | Negative transfer detected → transfer blocked |

The diagonal $\mathbf{A}_{i,i} = 1$ is always 1 since it is the cosine similarity of a vector with itself.

## 3. Mathematical Foundations of Gradient Cosine Similarity

### 3.1 Mathematical Definition

Let $\mathbf{g}_i = \nabla_\theta \mathcal{L}_i$ and $\mathbf{g}_j = \nabla_\theta \mathcal{L}_j$ be the gradients of tasks $i, j$ with respect to the Shared Expert parameters $\theta$:

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

- $\mathbf{g}_i \in \mathbb{R}^d$: the flattened gradient vector of task $i$
- $d$: total number of Shared Expert parameters
- $\|\cdot\|$: L2 norm

In the implementation, all pairwise similarities are computed in one matrix operation:

```python
# adatt.py:122-125
norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
normalized = grad_matrix / norms
affinity = torch.mm(normalized, normalized.t())  # [n_tasks, n_tasks]
```

`clamp(min=1e-8)` prevents a zero-gradient task from causing a division-by-zero.

### 3.2 EMA Smoothing

Single-batch gradients are noisy, so an Exponential Moving Average is applied over time to obtain a stabilized affinity:

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1 - \alpha) \cdot \cos(\theta_t)$$

- $\alpha = 0.9$ (default, the `ema_decay` parameter)
- $\mathbf{A}_0$: the first observation is used as-is (EMA initialization, `adatt.py:134-135`)

> **Why $\alpha = 0.9$?** With $\alpha = 0.9$, the EMA approximates a weighted average over the last 10 observations (effective window $\approx 1/(1-\alpha) = 10$). Early in training, gradients are unstable, so stability is prioritized over fast convergence; but because task relationships can drift across epochs, we avoid an overly conservative value. Configured at `model_config.yaml:601`: `ema_decay: 0.9`.

### 3.3 Why Cosine Similarity (vs. Euclidean Distance)

Three reasons motivate cosine similarity.

1. **Scale invariance.** Loss magnitudes differ across tasks, and so do gradient norms. Cosine similarity compares direction only and is unaffected.
2. **Interpretability.** Normalized to $[-1, 1]$, it gives an intuitive reading: "same direction = positive transfer, opposite direction = negative transfer".
3. **Computational efficiency.** After normalization, a single matrix multiplication computes all pairs in $O(n^2 d)$.

Fifty et al., 2021 also use gradient cosine similarity as the standard metric for task-affinity measurement in their Task Affinity study.

> **Recent trends — gradient-based task-similarity measurement (2023--2025).** Beyond cosine similarity, several gradient-based metrics have been actively studied. (1) *TAG (Task Affinity Grouping, Fifty et al., ICML 2021)*: automates task grouping using the sign-change pattern of gradient inner products. (2) *Gradient Vaccine (Wang et al., ICLR 2021)*: applies *partial projection* based on cosine similarity when gradients conflict, achieving finer-grained control than PCGrad. (3) *Conflict-Averse Gradient Descent (CAGrad, Liu et al., NeurIPS 2021)*: treats cosine similarity as a minimization target, finding a common direction that maximizes the minimum angle against every task's gradient. (4) *Nash-MTL (Navon et al., ICML 2022)*: formulates inter-task competition as a Nash bargaining problem to derive a Pareto-optimal gradient. Among these, adaTT combines *similarity measurement + selective transfer* — the strength of its design is the modular separation of measurement from utilization.

### 3.4 Gradient Extraction Path

Gradients are extracted in the `_extract_task_gradients` method of `ple_cluster_adatt.py`. The key point is that gradients are computed *only with respect to the Shared Expert parameters*.

```python
# ple_cluster_adatt.py:1871-1903
shared_params = list(self.shared_experts.parameters())

for task_name, loss in eligible:
    grads = torch.autograd.grad(
        loss,
        shared_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_flat = torch.cat([
        g.flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(grads, shared_params)
    ])
    task_gradients[task_name] = grad_flat
```

> **⚠ Why retain_graph=True is mandatory.** `retain_graph=True` is *architecturally unremovable* (`ple_cluster_adatt.py:1865-1868`). `_extract_task_gradients` is invoked during the forward pass (after loss computation but before `backward()`), and the same computation graph must be reused by the Trainer's `loss.backward()`. Releasing the graph here would cause `RuntimeError: Trying to backward through the graph a second time` inside `backward()`.

### 3.5 The torch.compiler.disable Decorator

```python
# ple_cluster_adatt.py:1847
@torch.compiler.disable
def _extract_task_gradients(self, task_losses):
```

> **Why exclude this from torch.compile.** `torch.autograd.grad` has incomplete `requires_grad` tracking inside compiled graphs. The `torch.compiler.disable` decorator runs this method outside the compile graph to guarantee correct gradient extraction. In practice `torch.compile` itself is currently disabled (`trainer.py:177`), but the decorator is applied defensively so the path stays safe if compilation is enabled later.

## Where We Stop

ADATT-2 answers the question of *how* adaTT measures inter-task affinity. We extract per-task gradients against the Shared Expert parameter space, normalize them, compute all pairwise cosine similarities with a single matrix multiplication, smooth out batch noise with EMA, and maintain the result as the affinity matrix $\mathbf{A}$ clamped to $[-1, 1]$. What we have *not* yet touched is how this affinity is actually *used* to drive inter-task transfer during training. How much gradient do we borrow when affinity is positive? How do we block transfer when it is negative? How should the unstable early-training affinity and the converged late-training affinity be treated differently? Those are the questions picked up by **ADATT-3** — Transfer Loss, the Group Prior, and the 3-Phase Schedule.
