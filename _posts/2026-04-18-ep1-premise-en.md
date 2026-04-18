---
layout: default
title: "Ep 1 — The Premise"
date: 2026-04-18
lang: en
series: three-months
part: 1
---

# Episode 1 — The Premise

*Part 1 of "Three Months, Three People" — a series on building
a financial recommendation system with Claude Code, on consumer
hardware, as three people on personal time.*

🇰🇷 [한국어 버전 →](/2026/04/18/ep1-premise-ko/)

---

## Who we were

Three people. One data scientist serving as PM / lead architect,
and two engineers. That was the team.

The two engineers were not formally contracted employees.
They were recent graduates participating as "youth advisory
supporters" while preparing for full-time employment — a Korean
program that lets new graduates work on real projects without
formal contracts, for a modest stipend.

So: three people, part-time commitment, and a nominal program
budget. No institutional backing beyond that.

## What we had

One RTX 4070. Twelve gigabytes of VRAM. A desktop card, not a
datacenter card. Installed in a local PC in an unventilated room
next to the server room — no adequate cooling, because the
proper facility was reserved for "real" projects.

That was the infrastructure.

We requested a GPU server. "There is nothing we can do."
We requested migrating off HIVE to Spark or Impala for data
processing. Denied. So we stayed on HIVE and wrote parallel query
logic from scratch to saturate what network bandwidth we could get.

The AI tool subscriptions — Claude Code, Gemini, Cursor — came
out of the PM's personal wallet. So did AWS spot instance costs
and storage. And occasional team meals.

Looking back, this sounds grim. At the time it just felt like the
starting condition. Something to work within, not cry about.

## What we were replacing

The existing system was a **collaborative filtering recommender**
built on **ALS** (Alternating Least Squares). It had been in
production for years. It worked, in the sense that it produced
recommendations. But it had two problems we could not ignore:

**It could not explain itself.** ALS gives you latent factors —
numerical dimensions that capture user-item similarity, but those
dimensions have no business meaning. A branch employee cannot tell
a customer "we recommended this because *latent factor 7 is
high*." And a regulator asking "why did your model recommend this
product to this customer?" gets no useful answer from a latent
factor.

**It could not keep up with task diversity.** As the business
added more use cases — next best product, churn prediction,
customer value tiering — ALS did not generalize. You ended up
with one ALS model per task, each tuned separately, each drifting
separately.

The goal was to replace it with something that could (a) produce
business-interpretable explanations by construction, and (b)
handle many tasks in one model with shared representation.

## What the production system actually looked like

Before I describe the architecture, one more number worth
mentioning.

The on-premises production system — the one this project was
replacing — was not a toy. It had:

- 80+ Airflow DAGs
- Champion-Challenger model competition
- Weekly automated retraining
- A 734-dimensional feature tensor
- 18 simultaneous tasks
- 62 data table ingestion jobs

Building and replacing a system at that scale, with three people
and a desktop GPU, is the sort of thing you would dismiss on paper.
In practice, it is what AI-augmented development actually enabled.

The public AWS benchmark version is somewhat smaller (13 tasks
after removing 5 deterministic-leakage / redundant ones; 349
features instead of 734) but the architecture and the engineering
patterns are the same.

## The architecture decision journey

Here is where it gets interesting. The final architecture — seven
heterogeneous expert networks sharing a PLE bottom — did not
appear fully formed. It emerged by rejection.

**Candidate 1: Black-Litterman.** This comes from portfolio theory.
It combines "views" from experts with a market-equilibrium prior
via Bayesian updating. It is a beautiful model in its home
domain. The PM brought it to the table because the structural
analogy was real: different "expert" signals, combined into a
posterior.

The problem showed up when we asked: *can we decompose how much
each expert contributed to a specific recommendation?* And the
answer, honest answer, was no. Bayesian updating blends the
inputs into a posterior distribution in which individual
contributions are no longer recoverable. Which meant we could not
explain a recommendation to a customer, to a branch employee, or
to a regulator — the three audiences whose approval matters in
financial services.

Rejected.

**Candidate 2: Model ensemble.** Train N independent models,
combine their outputs. Simple, effective, widely used.

Two problems. First, cost. N models means N management points,
N monitoring dashboards, N retraining pipelines, N times the
serving cost. For a three-person team, that is a nonstarter.
Second, the same explainability problem: "MLP #3 contributed 28%"
is not a business explanation. A customer does not care about
MLP #3. Neither does a regulator.

Rejected.

**The reframe that mattered.** If combining experts *outside* the
model fails on both cost and explainability, what if we combine
them *inside* a single model? One model, one serving endpoint,
one monitoring surface — but with multiple internally distinct
experts whose contributions can be attributed via gate weights
that have business meaning.

That is PLE's promise. Progressive Layered Extraction, from the
Tencent recommender literature. Experts are trained together
inside one model; a gating network decides how much each expert
contributes to each task. The gates are interpretable — they are
not abstract latent factors, they are weights you can read.

The rest of the architecture followed from this reframe. Seven
experts, each chosen to represent a structurally distinct
*mathematical perspective* on the customer (graph structure,
temporal dynamics, topological shape, causal inference, and so
on). A shared bottom that lets them cross-fertilize. Gates that
are *the* explanation, not an artifact of post-hoc analysis.

## Why the constraints mattered

Had we had a proper GPU cluster, we would not have built this
architecture. We would have stacked seven Transformer-based heavy
experts and brute-forced the problem with parameters. It would
have worked — most things work if you throw enough compute at
them — but it would have been forgettable.

The 12GB VRAM ceiling blocked that lazy approach at the
architectural level. We had to choose: each expert had to be
*lightweight and structurally distinct*. Encoding expressiveness
through inductive biases, not parameter count. Borrowing
structural isomorphisms from eleven academic disciplines —
hyperbolic geometry, chemical kinetics, epidemic SIR models,
optimal transport, persistent homology, and so on — because every
one of those fields had already solved a similar problem in its
home domain, and we could import the structure cheaply.

Selection pressure in biological evolution drives specialization.
Resource constraints in engineering do the same thing. Without
the constraint, there is no identity.

---

## Next episode

In [Episode 2](#) we will cover how three people actually
*organized* the AI collaboration — which tools got used for what,
how we avoided burning subscriptions on the wrong phase, and what
patterns emerged that transferred across the 3.5 months.

---

<div class="series-nav" style="margin-top: 2em; font-size: 0.9em;">
<a href="/series/three-months/">← Series index</a>
· Episode 2 (coming soon)
</div>

---

<small>
Drafted with Claude Code assistance. Ideas, experience, and final
review by the author. Source material:
<a href="https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf">
Development Story (EN, PDF)
</a>.
</small>
