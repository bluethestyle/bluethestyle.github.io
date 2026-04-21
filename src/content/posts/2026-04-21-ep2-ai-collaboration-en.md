---
title: "[FinAI Build] Ep 2 — Organizing the AI Collaboration"
date: 2026-04-21 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, claude-code, architecture, financial-ai]
lang: en
series: three-months
part: 2
alt_lang: /2026/04/21/ep2-ai-collaboration-ko/
next_title: "Ep 3 — Hardware, Budget, and What It Bought Us"
next_desc: "One RTX 4070, three personal laptops, subscriptions out of the PM's wallet, AWS spot costs. What the constraints actually forced at the design level."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/ai_collaboration_guide_en.pdf
source_label: "AI Collaboration Guide (EN, PDF)"
---

*Part 2 of "Building a Financial AI in Three Months". How three
people divided work across Claude Code, Gemini, and Cursor, where
we got it wrong, and what had hardened into practice by the end
of 3.5 months.*

## The mistake we made early

The starting conditions from Ep 1 — three people, one RTX 4070,
personal time — make AI tools look like *salvation*. "If I can
code at 3x speed, then three of us are a nine-person team." That
fantasy came easy.

For the first two weeks we used Claude Code as a *general-purpose
engineer*. Throw it a task, say "implement this", merge the
result. The code looked plausible — but rereading it a few hours
later, it rested on assumptions subtly different from our
architecture. Feature-group names written slightly differently,
config paths invented out of nothing, separation-of-concerns
violated by inlining preprocessing into `train.py`.

Catching these ate the time savings. Maybe more than ate them.
The sense that "AI codes 3x faster" was not real; the real
throughput was "AI writes code + we spend N hours correcting it".
Once correction time exceeded ~70% of writing time, the net value
went negative.

Admitting that mistake reframed the question. Not "what do we
make the AI do?" but "which AI capability goes into which phase
so that we actually get faster?"

## A three-phase reframe

The division of labor that stabilized over 3.5 months:

**Phase 1 — Design conversation (Claude web interface).**
Long conversations that don't write to files. Throw ideas,
explore trade-offs, surface hidden assumptions. "If this expert
is grounded in hyperbolic geometry, what does that buy and what
does it cost?", "Why does Black-Litterman ensemble get rejected?"
The output of this phase is *a decision with its rationale*,
which gets copied into a hand-edited design document or
markdown.

No code in this phase. Once code enters the conversation, the
conversation shifts into "is this code right?" review mode, and
trade-off exploration stops.

**Phase 2 — Implementation (Claude Code, terminal).**
Once a design is locked, hand it to Claude Code with *explicit
context* — which files, which interface contract, which existing
pattern to follow. Small chunks. One task per session. The agent
proposes a diff; a human reads it and rejects or approves.

The most important rule in this phase: *don't commit code you
didn't read*. A simple rule, but under fatigue it's easy to
break. Breaking it puts you back in the "70% correction time"
trap.

**Phase 3 — Review and debugging (Gemini + human).**
When a bug surfaces or a design decision feels shaky, Gemini's
long-context reading provides a second opinion. Paste the
symptom, the relevant code, and our hypothesis together; ask
"what other root causes are plausible?" Gemini is for *reading*,
not *writing*. Writing stays consolidated in Claude Code so
conventions don't fragment.

Cursor was used only by the two engineers when they needed
autocomplete flow. It never drove architectural decisions or
large edits.

## The subscription trap

We started with three Claude Pro subscriptions (\$20/mo each).
After a month, usage data was lopsided: the PM always hit limits;
the two engineers used 30–40% of their quota. We were collectively
paying for \$60/mo but the PM was capacity-constrained and the
engineers' capacity sat idle.

Rebalanced: PM upgraded to Claude Max (\$100/mo), the two engineers
stayed on Pro. Added Gemini 2.5 Pro (\$20/mo) under the PM for the
Gemini role. Cursor (\$20/mo) as a shared engineer seat.

Final cost structure (all from the PM's personal wallet):
- Claude Max (PM) — \$100/mo
- Claude Pro × 2 (engineers) — \$40/mo
- Gemini Pro — \$20/mo
- Cursor — \$20/mo
- AWS spot + S3 — avg \$60-80/mo

Total: about \$240-260/mo, ≈\$900 over 3.5 months. Roughly one
week of a mid-level engineer's wages. The "AI tools are
expensive" feeling never came from subscription prices; it came
from *time wasted by using the wrong tool at the wrong phase*.

## Context hygiene

The most expensive lesson across 3.5 months was about session
lifetime. A Claude Code session left open for days accumulates
context residue from earlier tasks — hypotheses since discarded,
approaches since rejected, pre-edit code. All of that interferes
with the current task.

Rules that stuck:
- Task done = session closed (next task gets a fresh session)
- If the topic shifts mid-session, force `/compact`
- Persistent rules go in `CLAUDE.md` at the project root, not
  into session history
- Large reference material (papers, on-prem docs) goes into the
  first message of a session, once; afterwards it's referenced
  rather than re-pasted

`CLAUDE.md` was especially critical. Our project's `CLAUDE.md`
is now six sections covering hardcoding bans, separation of
concerns, cost management, and orchestration efficiency.
Re-explaining these each session would be impossible. Writing
them once and letting the agent auto-reference them was the only
way to scale.

## Patterns that hardened over 3.5 months

**Plan-first.** New features always start with a plan document.
This exists to stop Claude Code from diving straight into
implementation. We developed a reflex — within five minutes of
starting, pause and ask "did we write the plan?" If not, roll
back and write it.

**Interface-contract verification.** After running parallel
agents on related work, we *always* run a separate interface
contract check — does "the key name A writes" match "the key
name B reads"? Agents each choose plausible names in their own
context; two agents' "plausible names" disagreeing is a failure
that only surfaces at runtime.

**Local first, SageMaker last.** Debugging code on SageMaker
costs \$0.50+ per job submission, instantly. Local GPU until a
1-epoch end-to-end run succeeds → only then submit to SageMaker.
Without this discipline, every mistake turns into ~10 jobs and
\$5 evaporates.

**Completion-report discipline.** To report a task as "done", it
has to pass four checks — compile, interface contract, hardcoding
scan, separation-of-concerns. Partial completion is reported
explicitly as "compile OK, interface unverified, hardcoding
unscanned". Without this habit the gap between "reported done"
and "actually working" compounds.

## Next

Ep 3 covers hardware and budget — how the single RTX 4070
concretely forced specific design choices, how we decided the
AWS spot vs on-demand vs local hybrid split, and what made the
\$900 out of the PM's wallet worth it over 3.5 months.

Source material lives in the repo at
[AI Collaboration Guide (EN, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/ai_collaboration_guide_en.pdf).
