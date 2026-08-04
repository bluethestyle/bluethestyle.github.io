---
title: "[Study Thread] HMM-1 — Hidden Regimes: Reading the State Behind the Spending"
date: 2026-06-07 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, hmm, markov, regime, features, offline]
lang: en
excerpt: "The HMM sub-thread opens — why a customer's observable transactions hide a latent behavioral state, how a Hidden Markov Model recovers that state with a transition matrix, Gaussian emissions, and an initial distribution, how Baum-Welch fits it and Viterbi decodes it, and exactly which 48D of state posteriors, transition statistics, dwell times, and trajectory dynamics land in the model's separate-input path."
series: study-thread
part: 19
alt_lang: /2026/06/07/hmm-regime-features-ko/
next_title: "GMM-1 — Soft Clustering: Responsibilities as Features and the Gaussian Mixture Behind Them"
next_desc: "From temporal regimes to a single snapshot: how a 20-component Gaussian mixture profiles a customer by type rather than stage, why soft responsibilities beat hard labels, and how the 22D lands inside the 734D main tensor's Domain block instead of a separate path."
next_status: draft
---

*First post of the HMM sub-thread in the "Study Thread" series. Across
this and the following posts, in parallel Korean and English, I unpack
the HMM-derived features — an offline feature module that feeds the PLE
model through a separate input path. The source is the on-prem
reference `기술참조서/HMM_피처_기술_참조서`, and the full PDF will be
attached to the final post of the sub-thread. Where the TDA sub-thread
read the* shape *of behavior, this one reads its* state *— the hidden
phase a customer occupies and the law by which they move between
phases. The model never observes that phase directly. It infers it.*

> **What this module actually ships.** Three Gaussian HMMs run in
> parallel over each customer's transaction sequence — **Journey**
> (5 states), **Lifecycle** (5 states), and **Behavior** (6 states) —
> and emit **16D each → 48D** total through a *separate input* path
> distinct from the 734D main tensor. A compressed **5D summary** also
> rides inside the main tensor's `model_derived` block. Each 16D is
> `n_states` state posteriors + meta features + a 6D ODE-dynamics
> bridge. Everything below grounds in the reference doc — the real
> transition matrices, the `n_iter=200 / tol=1e-2` Baum-Welch settings,
> the 3D observation vector. No invented numbers.

## Why a Latent State, Not Just the Numbers

What we directly observe in card-transaction data is thin: how much was
spent, how many times, across how many distinct categories. But behind
those numbers sits a *psychological* state the data never names. Two
customers who each spend ₩100,000 this month can be in completely
different places — one *exploring* a new service, one spending one last
time *just before churning*. The aggregate is identical. The state is
opposite.

A rule like "over ₩500k/month ⇒ active customer" sets an arbitrary
threshold and mangles everyone near the boundary. A Hidden Markov Model
refuses the hard cut. It returns a *soft assignment* —
"80% active, 15% growing, 5% at-risk" — and that probability vector is
itself the feature. The model recovers what it cannot see (the
behavioral phase) from what it can (the transactions), probabilistically.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <!-- hidden layer -->
  <text x="40" y="42" font-size="11" font-weight="700" fill="#4f46e5">hidden state z</text>
  <g>
    <circle cx="120" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="120" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₁</text>
    <circle cx="260" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="260" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₂</text>
    <circle cx="400" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="400" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₃</text>
    <circle cx="500" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4" stroke-dasharray="3 3"/><text x="500" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">…</text>
  </g>
  <!-- transition arrows -->
  <g stroke="#4f46e5" stroke-width="1.4" fill="none">
    <path d="M 140 60 L 240 60"/><polygon points="240,60 232,56 232,64" fill="#4f46e5"/>
    <path d="M 280 60 L 380 60"/><polygon points="380,60 372,56 372,64" fill="#4f46e5"/>
    <path d="M 420 60 L 478 60"/><polygon points="478,60 470,56 470,64" fill="#4f46e5"/>
  </g>
  <text x="200" y="50" text-anchor="middle" font-size="9" fill="#64748b">A (transition)</text>
  <!-- emission arrows -->
  <g stroke="#0d9488" stroke-width="1.4" fill="none">
    <path d="M 120 80 L 120 140"/><polygon points="120,140 116,132 124,132" fill="#0d9488"/>
    <path d="M 260 80 L 260 140"/><polygon points="260,140 256,132 264,132" fill="#0d9488"/>
    <path d="M 400 80 L 400 140"/><polygon points="400,140 396,132 404,132" fill="#0d9488"/>
  </g>
  <text x="150" y="115" font-size="9" fill="#0d9488">B (emission)</text>
  <!-- observed layer -->
  <text x="40" y="178" font-size="11" font-weight="700" fill="#0d9488">observed oₜ</text>
  <g fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2">
    <rect x="100" y="150" width="40" height="34" rx="4"/><rect x="240" y="150" width="40" height="34" rx="4"/><rect x="380" y="150" width="40" height="34" rx="4"/>
  </g>
  <g font-size="9" fill="#0d9488" text-anchor="middle">
    <text x="120" y="171">o₁</text><text x="260" y="171">o₂</text><text x="400" y="171">o₃</text>
  </g>
  <text x="290" y="206" text-anchor="middle" font-size="10" fill="#64748b">transitions are hidden; only transactions are seen — the HMM inverts the arrows</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A Hidden Markov Model. The state chain z evolves by transition matrix A; each state emits an observation o by distribution B. We see only o and must reconstruct z.</figcaption>
</figure>

## The Markov Property — Tomorrow Depends Only on Today

The whole structure rests on one assumption, the *Markov property*:

$$ P(q_{t+1} \mid q_t, q_{t-1}, \dots, q_1) = P(q_{t+1} \mid q_t) $$

A customer's *next* state depends only on their *current* state, not on
the entire history before it. A customer who is `MATURE` now has the
same transition odds next month whether they were `NEW` or `AT_RISK`
six months ago. This is plainly a simplification — long-term memory
does matter in reality — but it caps the parameter count at $O(N^2)$
(the transition matrix), which is what lets the model train stably over
hundreds of thousands of customer sequences. The project compensates
for the lost long-range memory by running three modes at three
different time scales in parallel, plus a 6D ODE-dynamics bridge that
re-encodes whole-sequence behavior separately.

> **Historical context.** The mathematics of HMMs was laid down by
> Leonard Baum and Ted Petrie in 1966 (*Annals of Mathematical
> Statistics*), where they formalized the statistical estimation of
> "probabilistic functions of Markov chains" — the prototype of what we
> now call Baum-Welch. But HMMs went mainstream through Lawrence
> Rabiner's 1989 tutorial in the *Proceedings of the IEEE*, which
> organized the field into three canonical problems — evaluation,
> decoding, learning — and pushed the method out of speech recognition
> into bioinformatics, NLP, and finance. The notation and the
> three-problem framing this post uses follow Rabiner. In finance the
> same machinery appears as Hamilton's (1989) Markov-switching regime
> model — bull/bear regimes are exactly customer growth/churn regimes
> under a different name.

## The Three Parameters: $\pi$, $A$, $B$

A Gaussian HMM is the triple $\lambda = (\pi, A, B)$:

| Symbol | Name | Meaning | Constraint |
| --- | --- | --- | --- |
| $\pi = \{\pi_i\}$ | initial distribution | $\pi_i = P(q_1 = S_i)$ — where a fresh customer starts | $\sum_i \pi_i = 1$ |
| $A = \{a_{ij}\}$ | transition matrix | $a_{ij} = P(q_{t+1}=S_j \mid q_t=S_i)$ — the law of behavioral change | each row sums to 1 |
| $B = \{b_j(\mathbf{o})\}$ | emission | $b_j(\mathbf{o}) = \mathcal{N}(\mathbf{o};\mu_j,\Sigma_j)$ — what a state's spending looks like | $\Sigma_j$ diagonal |

The observation at each time step is a **3D vector** — log-amount,
log-count, and category diversity:

$$ \mathbf{o}_t = \big(\ln(\text{txn\_amount}+1),\ \ln(\text{txn\_count}+1),\ \text{mcc\_diversity}\big) $$

The $\ln(x+1)$ squashes the extreme tail of spending into something a
Gaussian can model, and the $+1$ keeps $\ln(0)$ from blowing up on
zero-transaction months. The covariance is `diag` deliberately: with
only 3 observed dimensions, a full covariance would cost 6 parameters
per state against the diagonal's 3, and overfitting is the bigger risk
than the lost cross-correlations.

The joint probability of an observation sequence
$\mathbf{O}=(\mathbf{o}_1,\dots,\mathbf{o}_T)$ together with a state
path $\mathbf{Q}=(q_1,\dots,q_T)$ factorizes along the chain:

$$ P(\mathbf{O},\mathbf{Q}\mid\lambda) = \pi_{q_1}\,b_{q_1}(\mathbf{o}_1)\prod_{t=2}^{T} a_{q_{t-1}q_t}\,b_{q_t}(\mathbf{o}_t) $$

> **Equation intuition.** Read it left to right as a generative story:
> start in some state with prior $\pi$, emit the first observation with
> $b$, then for every later step pay a *transition* cost $a$ to move and
> an *emission* cost $b$ to produce what you see. It is a chain of
> multiplied probabilities — so one improbable link (a transition the
> matrix says is rare, or an observation the state would almost never
> emit) collapses the whole path's probability. The HMM's job is to
> sum or maximize this over all the hidden paths $\mathbf{Q}$ we never
> got to see.

## Three Modes, Three Time Scales

Rather than one HMM, the module runs three — each a different question
about the same transaction stream. The transition matrices below are
the real domain-initialized priors from the reference.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Lifecycle mode — 5 states, forward drift + churn absorption</text>
  <!-- nodes -->
  <g>
    <circle cx="80" cy="120" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="80" y="116" text-anchor="middle" font-size="9.5" font-weight="700" fill="#0d9488">NEW</text><text x="80" y="130" text-anchor="middle" font-size="8" fill="#64748b">0</text>
    <circle cx="200" cy="80" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="200" y="78" text-anchor="middle" font-size="9" font-weight="700" fill="#0d9488">GROW</text><text x="200" y="91" text-anchor="middle" font-size="8" fill="#64748b">1</text>
    <circle cx="330" cy="120" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="330" y="117" text-anchor="middle" font-size="8.5" font-weight="700" fill="#0d9488">MATURE</text><text x="330" y="131" text-anchor="middle" font-size="8" fill="#64748b">2</text>
    <circle cx="450" cy="80" r="30" fill="#d9770618" stroke="#d97706" stroke-width="1.4"/><text x="450" y="78" text-anchor="middle" font-size="8" font-weight="700" fill="#d97706">AT_RISK</text><text x="450" y="91" text-anchor="middle" font-size="8" fill="#64748b">3</text>
    <circle cx="540" cy="160" r="30" fill="#e11d4818" stroke="#e11d48" stroke-width="1.4"/><text x="540" y="158" text-anchor="middle" font-size="7.5" font-weight="700" fill="#e11d48">CHURNED</text><text x="540" y="171" text-anchor="middle" font-size="8" fill="#64748b">4</text>
  </g>
  <!-- arrows with probs -->
  <g stroke="#64748b" stroke-width="1.2" fill="none">
    <path d="M 106 105 L 174 90"/><polygon points="174,90 165,89 168,97" fill="#64748b"/>
    <path d="M 226 92 L 304 110"/><polygon points="304,110 295,105 297,113" fill="#64748b"/>
    <path d="M 356 108 L 426 92"/><polygon points="426,92 417,91 420,99" fill="#64748b"/>
    <path d="M 474 100 L 518 138"/><polygon points="518,138 509,133 512,142" fill="#64748b"/>
  </g>
  <g font-size="8.5" fill="#1e3a5f" font-weight="700">
    <text x="140" y="88">0.45</text><text x="262" y="95">0.35</text><text x="392" y="88">0.15</text><text x="488" y="122">0.28</text>
  </g>
  <!-- self loop on churned -->
  <path d="M 540 130 C 575 105, 600 130, 568 145" stroke="#e11d48" stroke-width="1.2" fill="none"/><polygon points="568,145 575,138 576,148" fill="#e11d48"/>
  <text x="582" y="118" font-size="8.5" font-weight="700" fill="#e11d48">0.80</text>
  <!-- recovery edge -->
  <path d="M 525 132 C 380 230, 200 220, 88 150" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3" fill="none"/><polygon points="88,150 96,150 92,158" fill="#94a3b8"/>
  <text x="300" y="232" text-anchor="middle" font-size="8.5" fill="#94a3b8">re-acquisition 0.05 (CHURNED → NEW)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Lifecycle transition matrix as a graph. MATURE self-transition is 0.70 (stability); CHURNED self-transition is 0.80 (sticky absorption) but leaks 0.05 back to NEW for re-acquisition.</figcaption>
</figure>

| Mode | States | Time scale | Target tasks |
| --- | --- | --- | --- |
| **Journey** (AICRA) | 5 — AWARENESS, CONSIDERATION, PURCHASE, RETENTION, ADVOCACY | day/week (short) | CTR, CVR |
| **Lifecycle** | 5 — NEW, GROWING, MATURE, AT_RISK, CHURNED | month/year (long) | Churn, Retention, Life-stage |
| **Behavior** | 6 — DORMANT, CONSERVATIVE, ROUTINE, EXPLORATORY, SPLURGE, INVESTOR | monthly pattern | NBA, balance_util |

Two details worth noting from the reference. The **Lifecycle** matrix
gives `MATURE` a 0.70 self-transition (mature customers are stable) and
`CHURNED` a 0.80 self-transition (churn is sticky), while still allowing
a 0.05 `CHURNED → NEW` edge for re-acquisition. The **Behavior** matrix
gives `SPLURGE` the *lowest* self-transition at 0.35 — impulse spending
is transient by nature, not a durable state. And **Journey** is the odd
one out at training time: its daily sequences are too sparse for EM, so
the project skips Baum-Welch on the transition matrix and keeps the
config priors fixed (`params="mc"` — only means/covariances learn),
which is why Journey's $a_{ij}$ values are uncalibrated domain priors,
not data-estimated.

## Fitting and Decoding: Baum-Welch and Viterbi

Three classic problems, three algorithms. The reference is explicit
about the project's settings: Baum-Welch runs with `n_iter=200` and
`tol=1e-2` (looser than the textbook `1e-4`, chosen for convergence
stability at hundreds-of-thousands scale).

**Fitting — Baum-Welch (EM).** The chicken-and-egg of "I'd know the
parameters if I knew the states, and vice versa" is solved by
iterating. The E-step computes, given current parameters, the posterior
state membership $\gamma_t(i)$ and the transition posterior
$\xi_t(i,j)$; the M-step re-estimates $\pi,A,\mu,\Sigma$ as
$\gamma$-weighted averages. The forward recursion that powers the
E-step is the load-bearing equation:

$$ \alpha_{t+1}(j) = \Big[\sum_{i=1}^{N}\alpha_t(i)\,a_{ij}\Big]\,b_j(\mathbf{o}_{t+1}),\qquad \alpha_1(i)=\pi_i\,b_i(\mathbf{o}_1) $$

Each iteration increases the log-likelihood $\ln P(\mathbf{O}\mid\lambda)$
monotonically until it converges to a local maximum — which is exactly
why the domain-expert priors for $\pi$, $A$, $\mu$ matter: they seed the
search near a good basin instead of a random one.

**Decoding — Viterbi.** Where the forward pass *sums* over all paths,
Viterbi *maximizes* — it finds the single most-likely state sequence
via the same dynamic-programming trellis, replacing the sum with a max
and keeping back-pointers $\psi$ to retrace the winning path. Enumerating
all $N^T$ paths is hopeless ($N=5, T=12$ is ~240 million), but the
Markov property collapses it to $O(N^2 T)$ — about 300 operations for
that case. That decoded path *is* the customer's estimated state
history, and it is the raw material for the meta features and the ODE
bridge.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <text x="300" y="24" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">Viterbi trellis — the best path through the state lattice</text>
  <!-- time labels -->
  <g font-size="9" fill="#64748b" text-anchor="middle">
    <text x="90" y="210">t₁</text><text x="210" y="210">t₂</text><text x="330" y="210">t₃</text><text x="450" y="210">t₄</text><text x="540" y="210">t₅</text>
  </g>
  <!-- state labels -->
  <g font-size="9" fill="#64748b" text-anchor="end">
    <text x="55" y="64">S₁</text><text x="55" y="114">S₂</text><text x="55" y="164">S₃</text>
  </g>
  <!-- faint full lattice edges -->
  <g stroke="#e2e8f0" stroke-width="1">
    <line x1="90" y1="60" x2="210" y2="60"/><line x1="90" y1="60" x2="210" y2="110"/><line x1="90" y1="60" x2="210" y2="160"/>
    <line x1="90" y1="110" x2="210" y2="60"/><line x1="90" y1="110" x2="210" y2="110"/><line x1="90" y1="110" x2="210" y2="160"/>
    <line x1="90" y1="160" x2="210" y2="60"/><line x1="90" y1="160" x2="210" y2="110"/><line x1="90" y1="160" x2="210" y2="160"/>
    <line x1="210" y1="60" x2="330" y2="60"/><line x1="210" y1="110" x2="330" y2="110"/><line x1="210" y1="160" x2="330" y2="160"/>
    <line x1="210" y1="60" x2="330" y2="110"/><line x1="210" y1="110" x2="330" y2="60"/><line x1="210" y1="110" x2="330" y2="160"/><line x1="210" y1="160" x2="330" y2="110"/>
    <line x1="330" y1="60" x2="450" y2="60"/><line x1="330" y1="110" x2="450" y2="110"/><line x1="330" y1="160" x2="450" y2="160"/>
    <line x1="330" y1="110" x2="450" y2="160"/><line x1="330" y1="160" x2="450" y2="110"/>
    <line x1="450" y1="60" x2="540" y2="60"/><line x1="450" y1="110" x2="540" y2="110"/><line x1="450" y1="160" x2="540" y2="160"/>
    <line x1="450" y1="160" x2="540" y2="110"/>
  </g>
  <!-- best path highlighted: S2 -> S1 -> S1 -> S3 -> S3 -->
  <g stroke="#d97706" stroke-width="2.6" fill="none">
    <line x1="90" y1="110" x2="210" y2="60"/>
    <line x1="210" y1="60" x2="330" y2="60"/>
    <line x1="330" y1="60" x2="450" y2="160"/>
    <line x1="450" y1="160" x2="540" y2="160"/>
  </g>
  <!-- all nodes -->
  <g fill="#cbd5e1">
    <circle cx="90" cy="60" r="6"/><circle cx="90" cy="160" r="6"/>
    <circle cx="210" cy="110" r="6"/><circle cx="210" cy="160" r="6"/>
    <circle cx="330" cy="110" r="6"/><circle cx="330" cy="160" r="6"/>
    <circle cx="450" cy="60" r="6"/><circle cx="450" cy="110" r="6"/>
    <circle cx="540" cy="60" r="6"/><circle cx="540" cy="110" r="6"/>
  </g>
  <!-- path nodes -->
  <g fill="#d97706">
    <circle cx="90" cy="110" r="7.5"/><circle cx="210" cy="60" r="7.5"/><circle cx="330" cy="60" r="7.5"/><circle cx="450" cy="160" r="7.5"/><circle cx="540" cy="160" r="7.5"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">A Viterbi decode. Of all the grey lattice paths, only one (amber) maximizes the joint probability — the customer's recovered state history, from which dwell time and transition stability are read off.</figcaption>
</figure>

## What Actually Gets Extracted: the 16D per Mode

Each mode emits exactly **16D**, structured the same way:
`n_states` state-probability dimensions + meta features + a 6D ODE
bridge. The reference's dimension check is `n_states + 5 + 6` for the
numeric features (string-valued `dominant_state_name` and `state_mode`
do not count toward the dimension). For Journey/Lifecycle the formula
gives $5 + 5 + 6 = 16$; Behavior the reference books as
$6 + 4 + 6 = 16$ — the extra sixth state absorbed by a slimmer meta
block.

> **Here the reference and the code diverge.** The dimension check in
> `hmm_features.py` uses a single mode-agnostic expression,
> `expected_dim = self.n_states + 5 + 6`; there is no branch that trims
> Behavior's meta block to 4D. By that formula Behavior is
> $6 + 5 + 6 = 17$D. The consumer-side contract
> (`task_feature_mapper.py`), meanwhile, declares `hmm_behavior: 16`.
> So **the producer says 17 and the contract says 16**, and the
> reference's "$6+4+6$" reads as a post-hoc reconciliation. The 48D
> three-mode total still holds as the contract value, but Behavior's
> internal split is not confirmed by the code.

| Group | Features | Source | Dim (J/L · B) |
| --- | --- | --- | --- |
| **State posteriors** | `state_prob_0 … state_prob_{N-1}` — the $\gamma_t(i)$ soft assignment | Forward-Backward | 5D · 6D |
| **Transition / dwell meta** | `state_duration`, `transition_stability`, `transition_entropy`, `dominant_state`, `state_change_rate` | Viterbi path | 5D · 4D |
| **ODE dynamics** | `ode_velocity`, `ode_acceleration`, `ode_lyapunov`, `ode_cycle_period`, `ode_attractor`, `ode_trajectory_len` | Viterbi trajectory | 6D · 6D |

The **state posteriors** are the heart of it: $\gamma_t(i) =
P(q_t=S_i\mid\mathbf{O},\lambda)$, the probability the customer sits in
state $i$ given the *whole* sequence (past and future evidence both,
via $\alpha$ and $\beta$). This is the soft assignment that encodes
uncertainty, is continuous and differentiable for the downstream net,
and is interpretable per-dimension.

The **meta features** come off the decoded Viterbi path:
`state_duration` counts how long the customer has sat in the current
state (a stability/inertia signal); `transition_entropy` is the
Shannon entropy of the transition-pair frequencies, normalized by
$\log(N^2)$, measuring how *varied* the transitions are; while
`state_change_rate` measures how *often* transitions happen — the two
are complementary.

The **ODE-dynamics bridge** (added in v3.2.0) treats the Viterbi
sequence as a continuous trajectory and extracts six kinematic
descriptors — velocity as mean $|\Delta q_t|$, acceleration as mean
$|\Delta^2 q_t|$, a Lyapunov-inspired second-half/first-half
instability ratio, autocorrelation-based cycle period, an attractor
concentration ratio, and normalized trajectory length. These are pure
sequence analysis, no extra training. They guard against the Markov
property's blindness to long-range structure: if a sequence is shorter
than 3 steps, all six return 0.0, and cycle detection only switches on
at length ≥ 6.

## Where the 48D Lands

The full picture, from the reference's `feature_schema.yaml`:

- The PLE model takes a **734D main tensor** (644D normalized + 90D raw
  power-law) *plus* a **68D separate input** (20D hyperbolic + **48D
  HMM Triple-Mode**). The HMM features ride the *separate path*, not the
  main tensor.
- Each mode's 16D is routed to a dedicated **HMM Triple-Mode Projector**
  inside the PLE, which projects it into the target tasks' Expert
  hidden dimension: Journey → CTR/CVR, Lifecycle →
  Churn/Retention/Life-stage, Behavior → NBA/balance_util.
- Separately, a compressed **5D summary** —
  `hmm_dominant_state`, `hmm_state_duration`, `hmm_transition_stability`,
  `hmm_transition_entropy`, `hmm_state_change_rate` — sits inside the
  main tensor's `model_derived` block (27D = HMM summary 5D + Bandit 4D
  + LNN 18D), so every task can reference the high-level behavioral
  signal globally, complementing rather than duplicating the 48D.

> **The contract has since moved.** The 734D above is the V1 feature
> contract. On 2026-07-02 the project switched to the V2 strict contract
> and the operational input width is **4035D** — 734D was not discarded;
> it remains V2's _shared base of eight groups_, with the
> lag/rolling/product families (3301D) appended on top.

This is an *offline* module by design. The HMM is fit and decoded in
the Airflow batch — fully independent of the GMM and Economics modules,
so it parallelizes in the DAG — and the resulting features are written
to Parquet, then read at serving time. No HMM inference runs online.

## Where We Stop

We started from the discomfort that aggregate spending hides the state
that explains it, set up the Markov property and the $(\pi, A, B)$
triple, walked the joint-probability factorization, fit it with
Baum-Welch and decoded it with Viterbi, and saw the actual 48D
(state posteriors + transition/dwell meta + ODE dynamics, 16D × 3
modes) land on the PLE's separate-input path with a 5D summary in the
main tensor.

What HMM gives is a *temporal* read: where a customer is in their arc
and which way they are moving, with the transition matrix carrying the
time dynamics. The next module asks the opposite question — not
"what stage, moving where," but "what *type*, right now." That is the
job of a Gaussian mixture: a single-snapshot, cross-sectional
soft-clustering that profiles a customer by type rather than stage, and
lands its 22D *inside* the main tensor's Domain block rather than a
separate path. That is the subject of the next post, **GMM-1**.
