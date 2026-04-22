---
title: "[FinAI Build] Ep 3 — How We Adapted: Guardrails, Memory Bank, Contract Verification"
date: 2026-04-24 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, claude-code, architecture, financial-ai]
lang: en
excerpt: "The mechanisms that kept three parallel AI-agent teams from breaking at integration — the CLAUDE.md constitution, the migration from a manual memory bank to Claude Code auto-memory, and the interface-key diff check run after every parallel session."
series: three-months
part: 3
alt_lang: /2026/04/24/ep3-guardrails-ko/
next_title: "Ep 4 — The Seven Experts: Importing Structural Isomorphism Across Eleven Disciplines"
next_desc: "How HGCN, PersLay, Causal, OT, Temporal Ensemble, DeepFM, and LightGCN became the seven — which mathematical gaps each fills, and what alternatives got rejected."
next_status: published
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf
source_label: "Development Story (EN, PDF) §3"
---

*Part 3 of "Building a Financial AI in Three Months". Ep 2 said
three humans × AI teams built modules in parallel. This episode is
about the machinery that kept the parallel work from breaking at
integration — the CLAUDE.md constitution, the memory bank, and
the interface contract verification routine.*

## What integration looked like, three days in

A Thursday afternoon, early in the project. Three teams had been
running since Tuesday, each with its own AI agent. The PM team
built the model-configuration schema; Engineer 1's team built the
preprocessing pipeline that generates the feature schema;
Engineer 2's team drafted the training script that reads both and
composes the model.

Three humans met Thursday afternoon to integrate. Outcome: five
runtime errors within thirty minutes, all variants of the *same
root cause*. Each team's component stored the same feature-group
boundaries under a different key name — all three names were
*plausible* in their own context.

This is the characteristic failure mode of parallel AI work. Each
agent picks the most reasonable name in its own context. The
probability that three contextually-reasonable names agree
converges to zero.

Two possible responses. One — give up parallelism, go serial (a
three-day task becomes nine). Two — install *a priori machinery*
that forces parallel work to converge. We chose the second; that's
what this episode is about.

## CLAUDE.md — the constitution written before code

The project-root `CLAUDE.md` was written during Phase C
(Cursor-based environment setup), *before a single line of code*.
Ep 2 noted the ordering was deliberate. That file shaped more of
the project's character than any other decision.

The current version has grown across repeated revisions into more
than fifteen sections, but four early clauses did most of the work:

**Config-driven principle.** Every parameter comes from a YAML
config. No column names, boundary values, scenario lists, or AWS
constants hardcoded in Python. Why *constitutional*: AI agents
are particularly fluent at embedding plausible constants directly
in code. A `batch_size=5632` three weeks later with no comment
and no one remembers why. Forcing values into config resolves
*documentation and tunability in a single move*.

**Separation of concerns.** The adapter does raw → standardized
DataFrame, nothing more. The pipeline runner does preprocessing,
feature generation, label derivation, normalization, and tensor
persistence. The training script loads training-ready data,
builds the model, and trains. Boundaries blur past 500 lines;
past that, separation has failed.

The Thursday integration bug's root cause was exactly this.
Engineer 2's team had inlined some preprocessing in the training
script, and that preprocessing diverged from the schema Engineer
1's team produced. Enforcing separation ex post facto moved 230
lines of preprocessing back into the pipeline runner.

**Data leakage prevention.** Scaler fits only on the train split.
Temporal splits require a mandatory gap. A leakage validator must
run before training. This clause gets a full episode of its own —
Ep 5, "The Data Integrity Hunt".

**Four-step completion check.** To report "done", a change must
pass compile verification, interface contract verification
(next section), hardcoding scan, and separation-of-concerns
check. All four, or partial completion must be stated explicitly.
Without this discipline, agents happily declare "done" after one
or two of the four, and breakage compounds.

## Memory bank — from manual to auto-memory

If `CLAUDE.md` is the project's constitution, the mechanism for
cross-session continuity is the memory bank. How we implemented
this shifted *twice* during the project.

**Initial approach — a manual eight-file structure.** In late
2025 and early 2026, Claude Code's cross-session memory was
limited. We compensated by placing eight files in a
`.claude/memory-bank/` directory and reading the core ones to
the agent at each new session — project brief, active context,
progress checklist, tech stack notes, product context, recurring
patterns, task state, style guide. The eight-file split was
empirical — a single "everything" file overwhelmed the first
session and agents dropped critical context within weeks.

In the same period we maintained a sync between a canonical
`.claude/RULES.md` and derived copies like `.cursorrules` —
multi-tool workflows meant each tool had its own rules file, and
drift between them produced divergent agent outputs that
compounded at integration points.

**Mid-2026 — after Claude Code auto-memory.** In April 2026
Claude Code v2.1.59+ introduced auto-memory, accumulating
session-to-session knowledge automatically — build commands,
debugging insights, architecture notes, code-style preferences,
workflow habits. Most of what had required manual files moved
into the territory the agent now remembers on its own.

So today most of the manual eight files are retired. Two remain:
a top-level summary of project goals and constraints (too
important to rely on automatic memory alone), and a rolling
task-state file (changes too granularly for automatic memory to
track reliably). The rest is auto-memory. The `.cursorrules`
sync was similarly retired as tool consolidation around Claude
Code reduced the benefit of maintaining parallel rule files.

The transition itself teaches a lesson — *guardrails built when
the tool was weak should not be maintained out of inertia after
the tool becomes stronger*. The manual memory bank was essential
in early 2026; today it's a deprecated pattern.

## Interface contract verification — the mandatory post-parallel check

To prevent the Thursday integration failure — the same concept
being stored and consumed under different key names across three
teams — this routine runs after every parallel session:

1. For each modified file, extract which keys each save function
   writes.
2. For every load function on the same branch, extract which keys
   it reads.
3. Diff the two sets — keys written but unread, keys read but
   unwritten, keys on both sides under different names.

The routine is delegated to an AI agent. When two or three
parallel agents finish their work, a *fourth agent* is invoked
specifically for interface-contract verification. The parallel
agents each chose "plausible" names in their own contexts; the
verification agent operates in a *whole-view* context and
identifies mismatches.

Without it, the Thursday integration bug repeats weekly and three
hours of debugging disappear. With it, those three hours become
twenty minutes — the mismatch surfaces well before CI.

## Guardrails aren't only for AI collaboration

These three mechanisms — the CLAUDE.md constitution, a persistent
cross-session context, the contract verification routine — sound
like AI-specific infrastructure. They aren't. The same problems
*apply to three-human collaboration*: naming mismatches, context
loss, rule drift. We systematized them in the AI-collaboration
context because that's where we first hit them at scale, but the
principles are old wisdom for human teams.

This is the reason the structure ports to a small
financial-services team (three to five people) with minimal
friction. Even without a Claude Code subscription, the
CLAUDE.md-level project constitution, shared context documents,
and pre-integration key-matching routine still install. With AI
it's faster; without AI it still works.

## Next

Ep 4 covers the architecture that sat on top of these guardrails —
how the seven heterogeneous expert networks were actually chosen.
HGCN, PersLay, Causal, OT, Temporal Ensemble, DeepFM, LightGCN.
Why these seven? Which candidates were rejected? The process of
importing structural isomorphism from eleven academic disciplines.

Source material:
[Development Story (EN, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf)
§3 "Quality Management Strategy".
