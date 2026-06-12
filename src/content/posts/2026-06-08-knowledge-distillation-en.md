---
title: "[Study Thread] DISTILL-1 — The Teacher Who Leaves: Knowledge Distillation from a Deep PLE Teacher into LightGBM Students"
date: 2026-06-08 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, distillation, lightgbm, teacher-student, serving]
lang: en
excerpt: "Why a closed-network batch system trains a 20GB deep PLE teacher and then throws it away at serving time — compressing it into per-task LightGBM students that answer from a repository lookup. Hinton's soft-label distillation, temperature and dark knowledge, the T²-scaled loss, the fidelity gate, and the 3-layer fallback that decides which model actually answers."
series: study-thread
part: 23
alt_lang: /2026/06/08/knowledge-distillation-ko/
next_title: "DISTILL-2 — Inference & Scoring: From Student Models to a Repository Lookup"
next_desc: "Once the LightGBM students are trained and registered, no model runs at request time. How the batch scores every customer × task ahead of time, writes the predictions to a DuckDB-over-Parquet store, and turns serving into a key lookup — plus the multiclass shape adapters and the consent gate that sit in the path."
next_status: draft
---

*First post of the Knowledge Distillation sub-thread in the "Study
Thread" series, in parallel Korean and English. The source is the
on-prem reference `기술참조서/지식증류_기술_참조서`, and the full PDF
will be attached to the final post of the sub-thread. The previous
sub-threads asked what an Expert reads (PersLay) and how tasks share
(PLE, adaTT). This one asks a colder, more operational question: once
you have spent days training a deep multi-task teacher, how do you serve
it to millions of customers on a closed network with no GPU inference
server at request time? The project's answer is blunt — you don't serve
it at all. You distill it into a fleet of small tree models and let the
teacher leave.*

> **The setup in one paragraph.** The PLE-adaTT teacher is a ~50M
> parameter deep multi-task model: Expert networks, cluster-aware heads,
> HMM features, 15 active tasks (18 defined; `uplift` and
> `category_uplift` are off, `brand_prediction` retired). It is accurate
> and it is expensive —
> **20GB VRAM**, **~50ms** for a 1,024-row batch, an **8GB+** Docker
> image. In a closed-network financial system that scores *millions* of
> customers in a nightly batch, running that teacher per request is a
> non-starter. Distillation transfers the teacher's *implicit
> knowledge* — its "dark knowledge" — into per-task **LightGBM**
> students that run in **8GB**, roughly **10× faster**, while holding
> the performance loss **within 3 percentage points**.

## Why Distill At All — Two Problems, One Move

There are two separate problems hiding in "serve a deep model," and
distillation solves both at once.

The first is *serving cost and complexity*. The teacher needs a GPU, a
PyTorch + CUDA runtime, and a fat image. The reference lists three hard
constraints that rule out direct serving: GPU memory (one model per GPU
at 20GB), inference latency (~50ms/batch against a 10ms SLA), and
deployment weight (8GB+ image from the PyTorch/cuDF dependency tree). A
LightGBM deployment is a ~200MB image, runs on CPU, and finishes the
same job in roughly a tenth of the time.

The second is *architecture freedom*. Distillation does not copy
weights — it copies *behavior*. The student can be a completely
different model class. That is precisely why this project can go from a
deep network (PLE) to a gradient-boosted tree ensemble (LightGBM): the
student only has to reproduce the teacher's output distribution, not its
internals. Quantization and pruning shrink a model within its own
family; distillation crosses families.

| Constraint | Teacher (PLE-adaTT) | Student (LightGBM) |
| --- | --- | --- |
| Parameters / size | ~50M, deep multi-task | hundreds of trees, per-task |
| Memory | 20GB VRAM | 8GB, CPU-only |
| Latency (1,024 batch) | ~50ms | ~10× faster |
| Deploy image | PyTorch + CUDA, 8GB+ | LightGBM, ~200MB |
| Trained on | features + labels | features + hard label + **soft label** |

> **Historical context.** The idea predates the name. *Bucilua,
> Caruana & Niculescu-Mizil (KDD 2006)* showed you could compress a
> large ensemble into a single small neural net by training it on the
> ensemble's predictions — "model compression." *Hinton, Vinyals & Dean
> (2015)* reframed it as **knowledge distillation**, adding the two
> ideas that made it a standard tool: a *temperature* knob that controls
> how much information the soft targets carry, and the name *dark
> knowledge* for the structure hiding in a teacher's softmax. Everything
> below is that 2015 framework, specialized to a tree student.

## Dark Knowledge — What a Soft Label Carries That a Hard Label Throws Away

A hard label is a grading sheet: class 3, full stop. The teacher's
softmax output is a *worldview*. When a 12-class head outputs

$$ p_{teacher} = [0.72,\ 0.14,\ 0.08,\ 0.03,\ 0.01,\ \dots] $$

it is not just saying "class 0." It is saying class 1 is a plausible
near-miss, class 2 a weaker one, and classes 5+ essentially
impossible. That *relative structure across classes* — which wrong
answers are almost-right — is what Hinton named **dark knowledge**:
real, learnable information that is invisible (dark) in the hard label
but present in the soft one. A $C$-class soft label encodes $(C-1)$
dimensions of relational signal where the hard label encodes a single
index.

Even in a binary task the effect survives. A teacher saying "click
probability 0.8" is not the same as "clicked." The 0.8 hands the student
a *probabilistic* statement — "almost certainly clicks, but with 20%
uncertainty" — and learning that distribution acts like a built-in label
smoothing, a regularizer that lifts the student's generalization. That
is the whole reason the student is trained on soft labels and not just
the ground truth.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <!-- teacher -->
  <rect x="30" y="70" width="120" height="90" rx="8" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <text x="90" y="105" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">PLE Teacher</text>
  <text x="90" y="124" text-anchor="middle" font-size="9" fill="#64748b">deep · 20GB VRAM</text>
  <text x="90" y="138" text-anchor="middle" font-size="9" fill="#64748b">15 task heads</text>
  <!-- soft label cloud -->
  <rect x="225" y="55" width="150" height="120" rx="8" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="300" y="48" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">soft label (T = 5.0)</text>
  <g fill="#4f46e5">
    <rect x="240" y="120" width="14" height="40"/><rect x="258" y="135" width="14" height="25"/>
    <rect x="276" y="145" width="14" height="15"/><rect x="294" y="150" width="14" height="10"/>
    <rect x="312" y="152" width="14" height="8"/><rect x="330" y="154" width="14" height="6"/><rect x="348" y="155" width="14" height="5"/>
  </g>
  <text x="300" y="100" text-anchor="middle" font-size="9" fill="#64748b">[0.72, 0.14, 0.08, …]</text>
  <text x="300" y="113" text-anchor="middle" font-size="9" fill="#64748b">"dark knowledge"</text>
  <!-- student -->
  <rect x="450" y="70" width="120" height="90" rx="8" fill="#d9770615" stroke="#d97706" stroke-width="1.2"/>
  <text x="510" y="105" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">LGBM Student</text>
  <text x="510" y="124" text-anchor="middle" font-size="9" fill="#64748b">trees · 8GB · CPU</text>
  <text x="510" y="138" text-anchor="middle" font-size="9" fill="#64748b">per task</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.6">
    <line x1="150" y1="115" x2="223" y2="115"/><polygon points="223,115 213,110 213,120"/>
    <line x1="375" y1="115" x2="448" y2="115"/><polygon points="448,115 438,110 438,120"/>
  </g>
  <text x="187" y="105" text-anchor="middle" font-size="9" fill="#94a3b8">infer</text>
  <text x="412" y="105" text-anchor="middle" font-size="9" fill="#94a3b8">imitate</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The teacher is run once to produce soft labels; the student learns to imitate that distribution, inheriting the dark knowledge the hard label cannot carry.</figcaption>
</figure>

## Temperature — Softening the Softmax So the Knowledge Becomes Visible

If the teacher's softmax is already peaked — say $[0.95, 0.04, 0.01]$ —
the soft label is barely softer than the hard one, and the dark
knowledge stays hidden. **Temperature** $T$ fixes this by softening the
distribution before the student sees it:

$$ p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$

At $T = 1$ this is the ordinary softmax. As $T$ grows the distribution
flattens — its entropy rises — and the *small* probabilities on the
near-miss classes grow into a usable training signal. The project uses
$T = 5.0$ when generating soft labels. The name is not an accident: the
formula is the Boltzmann distribution from statistical mechanics with
$z_i = -E_i$, where high temperature spreads probability across states
and $T \to 0^+$ collapses everything onto the single lowest-energy
(highest-logit) state.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <!-- T=1 sharp -->
  <text x="150" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">T = 1 (sharp)</text>
  <line x1="50" y1="190" x2="270" y2="190" stroke="#64748b" stroke-width="1"/>
  <g fill="#e11d48">
    <rect x="60" y="60" width="30" height="130"/><rect x="100" y="170" width="30" height="20"/>
    <rect x="140" y="178" width="30" height="12"/><rect x="180" y="182" width="30" height="8"/><rect x="220" y="184" width="30" height="6"/>
  </g>
  <text x="150" y="212" text-anchor="middle" font-size="9" fill="#94a3b8">one class dominates → hard-label-like</text>
  <!-- divider -->
  <line x1="290" y1="45" x2="290" y2="200" stroke="#e2e8f0" stroke-width="1"/>
  <!-- T=5 soft -->
  <text x="420" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">T = 5 (soft)</text>
  <line x1="310" y1="190" x2="530" y2="190" stroke="#64748b" stroke-width="1"/>
  <g fill="#0d9488">
    <rect x="320" y="100" width="30" height="90"/><rect x="360" y="130" width="30" height="60"/>
    <rect x="400" y="148" width="30" height="42"/><rect x="440" y="160" width="30" height="30"/><rect x="480" y="168" width="30" height="22"/>
  </g>
  <text x="420" y="212" text-anchor="middle" font-size="9" fill="#94a3b8">near-misses visible → dark knowledge</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Raising the temperature flattens the softmax. The relative heights of the non-top classes — the dark knowledge — become a learnable signal instead of rounding error.</figcaption>
</figure>

## The Distillation Loss — Two Targets, One $T^2$ Correction

The student is trained against two things at once: the *truth* (hard
label) and the *teacher's view* (soft label). The project's loss is
their weighted sum:

$$ \mathcal{L}_{distill} = \alpha\,\mathcal{L}_{hard} + (1 - \alpha)\,T^2\,\mathcal{L}_{soft} $$

For a binary task this instantiates as

$$ \mathcal{L}_{binary} = \alpha\,\mathrm{BCE}(\hat{y}, y) + (1-\alpha)\,T^2\,\mathrm{KL}\big(p_t^{T}\,\|\,p_s^{T}\big) $$

and for multiclass it is the same shape with cross-entropy on the hard
side and $\mathrm{KL}\big(\mathrm{softmax}(z_t/T)\,\|\,\mathrm{softmax}(z_s/T)\big)$ on the soft side. The
soft term is a **KL divergence** in the *forward* direction —
$\mathrm{KL}(\text{teacher}\,\|\,\text{student})$ — which forces the student to put
mass everywhere the teacher does (covering the teacher's important
regions); reverse KL would be mode-seeking and could ignore whole modes,
which is wrong for distillation.

$\alpha$ sets the mix. The default is **$\alpha = 0.3$** — "30% grounded
in fact, 70% leaning on the expert" — and it can be overridden via
`DISTILLATION_ALPHA` (the repo ships 0.4 as an alternate default). The
rule of thumb: the better the teacher, the lower the $\alpha$, because a
trustworthy teacher's distribution is worth more than the raw labels.

> **Equation intuition.** Why the $T^2$ on the soft term? Softening the
> logits by $T$ also shrinks the soft loss's gradient — each softmax
> derivative picks up a factor of $1/T$, and the KL term's gradient
> scales like $1/T^2$. Without correction, raising $T$ would silently
> shrink the soft term's influence, and $\alpha$ would no longer mean
> what it says. Multiplying by $T^2$ (e.g. $T=5 \Rightarrow T^2=25$)
> restores the gradient to the same scale as the hard loss, so $\alpha$
> controls the hard/soft balance honestly. This is exactly the
> $1/(N T^2)$ factor Hinton derives. **Regression is the exception:** it
> predicts continuous values, not a probability distribution, so there
> is no temperature softening and *no* $T^2$ — the loss is just
> $\alpha\,\mathrm{MSE}(\hat{y}_s, y) + (1-\alpha)\,\mathrm{MSE}(\hat{y}_s, \hat{y}_t)$.

The clean part: LightGBM lets you inject this directly. The custom
objective (`fobj`) hands LightGBM the per-row gradient and hessian of
the combined loss —
$\nabla = \alpha\,\nabla_{hard} + (1-\alpha)\,T^2\,\nabla_{soft}$ —
so the tree ensemble is grown to minimize the *distillation* loss, not a
stock objective.

## The Teacher → Soft Labels → Student Pipeline

The whole thing is an offline Airflow batch, orchestrated by
`distillation_entrypoint.py` across **10 DAG stages**. The spine:

1. **detect-mode** — has the teacher changed? `full_distillation`
   (regenerate everything) vs `weekly_retrain` (reuse cached soft labels
   + feature selection, retrain only the students).
2. **load-teacher** — pull the PLE teacher from MLflow
   (`models:/ple_cluster_adatt/Production`).
3. **generate-soft-labels** — run the teacher at **T = 5.0** over all 15
   active tasks via `SoftLabelGenerator`, writing soft targets to
   Parquet.
4. **select-features** — Integrated-Gradients-based selection down to
   **200D**, with a mandatory-keep list (persistence entropy, MPC,
   income elasticity, Sharpe, volatility, …) so domain-critical features
   survive.
5–6. **mark-timestamp / load-cached-labels** — bookkeeping for the
   smart-mode branch.
7. **train-students** — fit one LightGBM model per task on
   `features + hard label + soft label` with the $T^2$-scaled custom
   objective.
8. **validate** — the fidelity gate (next section).
9–10. **log-mlflow / package** — register the students and package them
   for serving.

The teacher is consumed in stages 2–3 and never appears again. From
stage 7 on, the system only knows the students.

## The Fidelity Gate — Does the Student Actually Reproduce the Teacher?

A small model is useless if it quietly diverges from the teacher.
Before any student ships, distillation is checked for *fidelity*. The
reference defines a 5-criteria `DistillationValidator` as a library
specification:

| Criterion | Threshold | Metric | Meaning |
| --- | --- | --- | --- |
| 1 | AUC gap $\le 0.03$ | Teacher−Student AUC | accuracy preserved |
| 2 | Spearman $\rho \ge 0.95$ | rank correlation | **ranking** preserved |
| 3 | ECE gap $\le 0.02$ | calibration | probability *magnitude* preserved |
| 4 | all pass | segment consistency | no per-segment blind spot |
| 5 | speed ratio $\le 0.1$ | student/teacher latency | the 10× speedup is real |

Criterion 2 is the one that matters most for a recommender: it does not
care about absolute probabilities, only that if the teacher ranks
customer A above B, the student does too — because serving is a ranked
recommendation, and a flipped order degrades quality even when AUC looks
fine. Any one failure marks the task's distillation as FAIL.

One honesty note from the source: this 5-criteria class is the
*specification*. The **live** Stage 8 path runs a leaner *strict
validation* that compares student predictions directly against teacher
soft labels — binary: correlation $\ge 0.01$ and MAE $\le 0.50$;
multiclass: argmax agreement $\ge 0.08$; regression: correlation
$\ge 0.01$. The rich validator exists in the library but is not on the
current Stage 8 critical path.

## Where the Student Lives — The 3-Layer Fallback

Distillation produces a student, but the batch system does not blindly
trust it. At batch start the project runs a **3-layer FallbackRouter**
that decides, per task, *which* model's prediction is authoritative —
and the fidelity gate above is exactly what gates Layer 1.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="300" fill="#f8fafc" rx="8"/>
  <!-- decision 1 -->
  <rect x="180" y="20" width="200" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="280" y="38" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">teacher quality OK</text>
  <text x="280" y="52" text-anchor="middle" font-size="10" fill="#64748b">AND student fidelity passes?</text>
  <!-- L1 -->
  <rect x="380" y="90" width="160" height="44" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1.2"/>
  <text x="460" y="110" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Layer 1</text>
  <text x="460" y="125" text-anchor="middle" font-size="9" fill="#64748b">distilled LGBM</text>
  <!-- decision 2 -->
  <rect x="180" y="100" width="160" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="260" y="118" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">fidelity passes,</text>
  <text x="260" y="132" text-anchor="middle" font-size="10" fill="#64748b">teacher under bar?</text>
  <!-- L2 -->
  <rect x="380" y="160" width="160" height="44" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="460" y="180" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Layer 2</text>
  <text x="460" y="195" text-anchor="middle" font-size="9" fill="#64748b">direct LGBM</text>
  <!-- L3 -->
  <rect x="180" y="230" width="160" height="44" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1.2"/>
  <text x="260" y="250" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">Layer 3</text>
  <text x="260" y="265" text-anchor="middle" font-size="9" fill="#64748b">rule / template fallback</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#94a3b8" stroke-width="1.3">
    <line x1="380" y1="50" x2="378" y2="105"/><polygon points="378,112 373,102 383,102"/>
    <line x1="280" y1="60" x2="260" y2="98"/><polygon points="260,98 257,87 267,90"/>
    <line x1="340" y1="120" x2="378" y2="170"/><polygon points="378,178 369,170 377,164"/>
    <line x1="220" y1="140" x2="240" y2="228"/><polygon points="241,228 232,224 240,218"/>
  </g>
  <text x="400" y="78" font-size="9" fill="#0d9488" font-weight="700">yes</text>
  <text x="230" y="82" font-size="9" fill="#64748b">no</text>
  <text x="300" y="158" font-size="9" fill="#4f46e5" font-weight="700">yes</text>
  <text x="200" y="190" font-size="9" fill="#d97706" font-weight="700">no / missing</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The FallbackRouter resolves each task to a layer at batch start. Layer 1 requires both teacher quality and student fidelity; Layer 2 is the un-distilled tree; Layer 3 is a rule/template baseline when nothing else qualifies.</figcaption>
</figure>

- **Layer 1 — distilled LGBM.** The teacher passed its quality gate
  *and* the student passed fidelity. The compressed model is trusted;
  this is the happy path the whole pipeline aims for.
- **Layer 2 — direct LGBM.** The teacher was below its quality bar (so
  its dark knowledge isn't worth inheriting), but a tree trained directly
  on the labels still passes. Serve the tree without the soft-label
  transfer.
- **Layer 3 — rule / template fallback.** No model qualifies, or the
  task's predictions are missing entirely. A `RuleBasedRecommender`
  produces a heuristic baseline so every task always has an answer.

There is also a customer-level override outside this per-task ladder: if
a causal guardrail trips for a given customer, *all* of that customer's
tasks are forced to Layer 3 — a deliberately conservative choice when
the model's reasoning looks out-of-distribution. The 15 per-task
students are the Layer-1 fleet; the router is what decides, task by task,
whether each one actually gets to answer.

## Where We Stop

We started from an operational bind — a 20GB, ~50ms teacher that cannot
be served per request on a closed network — and watched distillation
dissolve it: the teacher is run *once* to emit soft labels, its dark
knowledge is poured into per-task LightGBM students through a
$T^2$-scaled hard+soft loss, a fidelity gate checks that the small model
still ranks and calibrates like the large one, and a 3-layer fallback
decides which model is authoritative for each task. The teacher trained
the students and left.

What remains is the part that makes all of this worth it: *serving*.
With the students trained and registered, the closed-network system runs
**no model at request time at all**. The batch scores every customer ×
task ahead of time and writes the results to a repository the serving
layer simply *looks up*. How that pre-scoring works — the DuckDB-over-
Parquet store, the multiclass shape adapters, the consent gate in the
path — is the subject of the next post, **DISTILL-2: Inference &
Scoring**.
