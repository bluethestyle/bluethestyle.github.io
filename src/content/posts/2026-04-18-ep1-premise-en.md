---
title: "[FinAI Build] Ep 1 — The Premise"
date: 2026-04-18 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, architecture, claude-code, financial-ai, ple]
lang: en
excerpt: "What three people with one desktop GPU set out to replace — an 80-DAG ALS recommender that could not explain itself and could not scale past one task — and how that starting constraint shaped every later architectural choice."
series: three-months
part: 1
alt_lang: /2026/04/18/ep1-premise-ko/
next_title: "Ep 2 — Organizing the AI Collaboration"
next_desc: "How three people actually organized the AI collaboration — which tools got used for what, how we avoided burning subscriptions on the wrong phase, and what patterns emerged that transferred across 3.5 months."
next_status: published
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf
source_label: "Development Story (EN, PDF)"
---

*Part 1 of "Building a Financial AI in Three Months" — a series on
building a financial recommendation system with Claude Code, on
consumer hardware, as a three-person team.*

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

**It could not keep up with task diversity.** ALS is a
user-item interaction matrix factorization — an algorithm scoped
to collaborative-filtering recommendation, not a general-purpose
predictor. As the business added more use cases — churn
prediction, customer value tiering, next best product — each
task got its own separate model bolted on (logistic regression,
XGBoost, rule-based segmentation). You ended up with a
patchwork of models per task, each tuned separately, each drifting
separately, with no shared customer representation anywhere.

The goal was to replace it with something that could (a) produce
business-interpretable explanations by construction, and (b)
handle many tasks in one model with shared representation.

## What the production system actually looked like

Before the design story, a few numbers to frame the scale of
what we were replacing.

The on-premises production system was not a toy. It had:

- 80+ Airflow DAGs
- Champion-Challenger model competition
- Weekly automated retraining
- A 734-dimensional feature tensor
- 18 simultaneous tasks
- 62 data table ingestion jobs

The public AWS benchmark version is somewhat smaller (13 tasks
after removing 5 deterministic-leakage / redundant ones; 349
features instead of 734) but the architecture and the engineering
patterns are the same.

Building and replacing a system at that scale, with three people
and a desktop GPU, is the sort of thing you would dismiss on paper.
In practice, it is what AI-augmented development actually enabled.
Which brings us to what "three people and a desktop GPU" actually
meant.

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

## Why Claude Code changed what three people could attempt

The constraints alone would not have been enough. Three people on
an RTX 4070, even with ten years of combined experience, cannot
ship an 80-DAG ALS replacement in three months through willpower.
What made this project possible as a *scope*, not just an
*ambition*, was Claude Code sitting in the loop from day one.

This is not a promotional claim; it is a scoping claim. Before
Claude Code, a project of this shape would have required a team of
eight to fifteen. With Claude Code as the primary development
partner — writing boilerplate, drafting expert implementations,
tracing bugs across the codebase, maintaining context across days
of parallel work — the same effort fit into three people. The
series from Ep 2 onward is about how that actually worked in
practice: which phases got which tools, how three humans each led
an AI agent team, and which failure modes emerged from that
pattern.

The point of naming this up front — a Korean financial-services
team of three to five people at a mid-size institution can now
consider projects that used to require a dedicated ML organization.
The constraints remain (consumer GPU, personal wallet for
subscriptions, small team), but the ceiling on what those
constraints can ship has moved.

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

None of this was a single moment. The Black-Litterman rejection
happened in a whiteboard session. The ensemble rejection happened
over two weeks of cost-modeling. The PLE reframe happened while
the PM was reading papers one Saturday and typed a line into a
Claude conversation: "what if the experts aren't independent
models but differentiable pieces of one model?" That line was
the hinge. What followed was three weeks of Claude Code sessions
taking the reframe into code — first a minimal PLE prototype on
synthetic data to see whether the gates were actually
interpretable, then iterative expansion to seven experts, then
the full 13-task setup. The "architecture decision" is easy to
write about in hindsight as a clean logical sequence; in real
time it was weeks of implementation, failure, reimplementation.

## Why the constraints were a gift

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
