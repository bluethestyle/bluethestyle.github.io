---
title: "[FinAI Build] Ep 3 — How We Adapted: Guardrails, Memory Bank, Contract Verification"
date: 2026-04-24 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, claude-code, architecture, financial-ai]
lang: en
excerpt: "The actual mechanisms that kept three parallel AI-agent teams from breaking at integration — the CLAUDE.md constitution's four clauses, the eight-file memory bank, and the interface-key diff check run after every parallel session."
series: three-months
part: 3
alt_lang: /2026/04/24/ep3-guardrails-ko/
next_title: "Ep 4 — The Seven Experts: Importing Structural Isomorphism Across Eleven Disciplines"
next_desc: "How HGCN, PersLay, Causal, OT, Temporal Ensemble, DeepFM, and LightGCN became the seven — which mathematical gaps each fills, and what alternatives got rejected."
next_status: draft
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
built `config_builder` and the `PLEConfig` schema. Engineer 1's
team built the Phase 0 pipeline that generates `feature_schema`.
Engineer 2's team drafted `train.py` that reads both and composes
the model.

Three humans met Thursday afternoon to integrate. Outcome: five
runtime errors within thirty minutes, all variants of the *same
root cause*. The PM team's schema stored feature-group boundaries
under the key `group_ranges`. Engineer 1's pipeline stored them as
`feature_group_ranges`. Engineer 2's `train.py` read them as
`ranges_by_group`. All three are *plausible* names.

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
Ep 2 noted the ordering was deliberate. This episode covers what
that file actually contains.

The current version has grown past seventeen sections, but the
four original core clauses are:

**§1.1 Config-Driven Principle.** Every parameter comes from a
YAML config. No column names, boundary values, scenario lists, or
AWS constants hardcoded in Python. Split-config pattern:
`configs/pipeline.yaml` (common) + `configs/datasets/{name}.yaml`
(dataset-specific).

Why *constitutional*: AI agents are particularly fluent at embedding
plausible constants directly in code. A `batch_size=5632` three
weeks later with no comment and no one remembers why. Forcing the
value into config solves *documentation and tunability in one
move*.

**§1.2 Separation of Concerns.** Adapter: raw → standardized
DataFrame, nothing more. PipelineRunner: preprocessing, feature
generation, label derivation, normalization, tensor persistence —
all of Phase 0. `train.py`: load training-ready data → build
model → train. Nothing else. Past 500 lines, separation has
failed.

The Thursday integration bug's root cause was exactly this.
Engineer 2's team had inlined some preprocessing in `train.py`,
and that preprocessing diverged from the Phase 0 schema Engineer
1's team produced. Enforcing §1.2 ex post facto moved 230 lines of
preprocessing from `train.py` back to Phase 0.

**§1.3 Data Leakage Prevention.** Scaler fits only on the TRAIN
split. Temporal splits require `gap_days` (minimum 7). The
`LeakageValidator` must run before training. This clause gets a
full episode of its own — Ep 5, "The Data Integrity Hunt".

**§4 Completion Check (four steps).** Criteria to report "done":
(a) every modified `.py` file passes `py_compile.compile(f,
doraise=True)`; (b) interface contract verified — keys saved by A
match keys read by B; (c) hardcoding scan — grep for column
names, AWS constants, magic numbers; (d) separation-of-concerns
check.

## The memory bank — continuity across sessions

If `CLAUDE.md` is the project's constitution, the eight files
under `.claude/memory-bank/` are the *between-session memory*.
Claude Code's context window (1M tokens) is large, but it does not
cross session boundaries. Re-explaining the project's state at
every new session dries out within days.

Division of labor across the eight files:

- `projectbrief.md` — project goals and constraints (replacing
  an on-prem system, three people, one RTX 4070).
- `activeContext.md` — "what are we doing right now" — the first
  file the agent reads in every new session.
- `progress.md` — milestone checklist (Phase 0 complete, 13-task
  reduction complete, ablation v4 complete, etc.).
- `techContext.md` — tech stack, versions, compatibility notes.
- `productContext.md` — business context, customer segments,
  regulatory requirements.
- `systemPatterns.md` — recurring design patterns (logger
  conventions, exception handling, async call protocol).
- `tasks.md` — in-progress, blocked, and completed tasks.
- `style-guide.md` — code style, naming conventions, commit
  message format.

The eight-file structure emerged *empirically* after a few weeks of
discovering that "one file with everything" overwhelms an agent's
first session and important context is dropped. Splitting the
files lets the agent acquire essentials in the first two or three
files and query the rest on demand.

## Synchronizing .claude/RULES.md and .cursorrules

Running multiple AI tools means each tool has its own rules file.
Claude Code has `CLAUDE.md`, Cursor has `.cursorrules`, and there's
another `.claude/RULES.md` under the agent directory. The content
overlaps but the formats differ, and manual synchronization drifts
within a few weeks — Cursor's agent ends up not following some
rules that Claude Code follows.

The fix: treat `.claude/RULES.md` as the *canonical source* and
derive the others. In practice this is a manual copy plus a diff
checker. Fuller automation is possible, but for a three-person
team the realistic goal is "detect drift within ten minutes", not
"100% automated".

## Interface contract verification — the mandatory post-parallel check

To make sure the `group_ranges` / `feature_group_ranges` /
`ranges_by_group` confusion never repeats, this routine runs after
every parallel session:

1. For each modified file, grep the `save_*` / `write_*` functions
   to extract which keys they write.
2. For every `load_*` / `read_*` function on the same branch,
   extract which keys they read.
3. Diff the two sets — keys written but unread, keys read but
   unwritten, keys present on both sides under different names.

This routine is delegated to an AI agent. When two or three
parallel agents finish their work, a *fourth agent* is invoked
specifically for interface-contract verification. The parallel
agents each chose "plausible" names in their own contexts; the
verification agent operates in a *whole-view* context and
identifies the mismatches.

The fourth agent matters. Without it, the Thursday integration bug
repeats and three hours of debugging is consumed weekly. With it,
those three hours become *twenty minutes* — the mismatch shows up
well before CI.

## Three-platform experiment branches

One more mechanism worth naming — the `exp/claude-auto-*`,
`exp/codex-auto-*`, and `exp/vertex-auto-*` branch families.
Automated experimentation that feeds the same request to three
platforms in parallel.

This isn't a contest for "which platform wins". The three platforms
*fail at different points*, and the failure patterns themselves
become a diagnostic for the project's weak spots. Claude Code is
strong at complex mathematical derivation but occasionally wrong
on YAML-parsing minutiae; Codex is the reverse. Triangulating the
three results surfaces "where our explanation is incomplete".

## Guardrails aren't only for AI collaboration

These three mechanisms — the CLAUDE.md constitution, the memory
bank, the contract verification routine — sound like AI-specific
infrastructure. They aren't. The same problems *apply to
three-human collaboration*: naming mismatches, context loss,
rule drift. We systematized them in the AI-collaboration context
because that's where we first hit them at scale, but the principles
are old wisdom for human teams.

This is the reason the structure ports to a small financial-services
team (three to five people) with minimal friction. Even without a
Claude Code subscription, the CLAUDE.md + memory bank + contract
verification routine still installs. With AI it's faster; without
AI it still works.

## Next

Ep 4 covers the architecture that sat on top of these guardrails —
how the seven heterogeneous expert networks were actually chosen.
HGCN (hyperbolic hierarchy), PersLay (persistent homology), Causal
(structural inference), OT (optimal transport), Temporal Ensemble,
DeepFM, LightGCN. Why these seven? Which candidates were
rejected? The process of importing structural isomorphism from
eleven academic disciplines.

Source material:
[Development Story (EN, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf)
§3 "Quality Management Strategy".
