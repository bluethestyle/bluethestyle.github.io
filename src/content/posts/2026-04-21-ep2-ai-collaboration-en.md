---
title: "[FinAI Build] Ep 2 — Organizing the AI Agents"
date: 2026-04-21 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, claude-code, architecture, financial-ai]
lang: en
excerpt: "Five phases, four tools — how Gemini, Claude Opus, Cursor, and Claude Code each took a specific slot (ideation / technical validation / scaffolding / implementation / paper writing). Why tool separation beat tool uniformity on both speed and quality."
series: three-months
part: 2
alt_lang: /2026/04/21/ep2-ai-collaboration-ko/
next_title: "Ep 3 — How We Adapted: Guardrails, Memory Bank, Contract Verification"
next_desc: "The actual mechanisms that kept three parallel AI agent teams coherent — the CLAUDE.md constitution, the 8-file memory bank, .claude/RULES.md synced with .cursorrules, and the interface-key matching that had to run after every parallel work session."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf
source_label: "Development Story (EN, PDF) §2"
---

*Part 2 of "Building a Financial AI in Three Months". How three
people divided the work across Gemini, Claude Opus, Cursor, and
Claude Code across five project phases — which tool did what,
where the structural-isomorphism insight came from, and why Claude
Code was the non-substitutable one in the implementation phase.*

## Phase-tool pairing as a frame

The constraints from Ep 1 — three people, one GPU, personal time —
did not only force the architecture. They also forced *how the
AI collaboration itself was organized*. The "one Claude
subscription covers everything" mindset collapsed within the first
few days. Each phase needed a different tool's different strength;
deliberate tool separation turned out to be better than tool
uniformity, on both speed and quality.

The five phases, in order:

## Phase A — Ideation (Gemini)

Gemini drove initial concept exploration. ALS replacement options,
Black-Litterman exploration, model-ensemble comparisons — scanning
architecture candidates *broadly* matched Gemini's broad knowledge
base.

The greatest value came from *cross-disciplinary feature
ideation*. Questions like "can chemical kinetics describe spending
behavior?", "is product adoption structurally equivalent to an
epidemic?" were posed to systematically scan which academic fields
had already solved structurally-isomorphic problems to financial
customer behavior. The PM contributed domain expertise (FRM, credit
analysis); Gemini contributed cross-domain connections.

The concept of *structural isomorphism* emerged from this process.
The decision to import features from eleven academic disciplines
was set during this phase and became the foundation of every
subsequent technical decision. Gemini was optimal not for depth
in any single technology, but for rapidly scanning "which field
has already solved a similar problem".

## Phase B — Technical validation (Claude Opus)

Translating ideas into concrete architectures required Claude
Opus. Work demanding technical depth concentrated here — loss
function design with mathematical rigor, data-leakage
verification, normalization pipeline design.

Each expert's feasibility was validated one by one. "Does HGCN
work on the MCC hierarchy?", "Is Mamba efficient enough for a
17-month sequence?" — deep technical dialogue with Opus. PLE vs
MMoE trade-off analysis, adaTT loss-level vs representation-level
transfer comparisons — all architecture-level analyses landed
here.

Opus also played the role of *challenging assumptions*. Its
counterargument — "is Black-Litterman really suitable?" —
accelerated the pivot toward PLE. The *expert collapse* problem
in homogeneous MoE was first identified in dialogue with Opus; this
identification would later (in Phase E) lead to the NeurIPS 2024
sigmoid gate paper — and that conclusion would itself get overturned
weeks later, which turned out to be part of the same thread.

Phase B's output: nineteen technical reference documents (`.typ`
files). These served as the *design specifications* each AI agent
would reference during the subsequent implementation phase.

## Phase C — Environment setup (Cursor)

GitHub code environment, project structure, initial boilerplate —
all Cursor. The IDE-integrated fast navigation and refactoring
was its strength.

The most important deliverable of this phase was not code. It was
six initial design documents (00-09 architecture specifications)
plus the establishment of `CLAUDE.md` guardrails. The
config-driven principle, separation of concerns, leakage
prevention rules — the "constitution" that every subsequent AI
agent would follow — were all established *before a single line
of code was written*. The order was deliberate: guardrails first,
agent execution second. The reverse order would have broken Phase
D's parallel implementation at every integration point.

## Phase D — Parallel implementation (Claude Code · Opus/Sonnet)

In the full implementation phase, each team member served as the
*"team lead"* for their own AI agents. Opus and Sonnet were run
in parallel inside Claude Code to implement different modules
simultaneously. Three humans, each leading one AI agent team.

- **PM / Lead Architect's AI team** — Opus for architecture-level
  decisions (PLE config, adaTT task groups, logit-transfer design),
  Sonnet for fast code implementation (generators, adapters,
  pipeline runner). The critical debugging sessions — detection of
  three label-leakage cases, diagnosis of four FP16 NaN root
  causes, GPU utilization optimization — originated on this team.
- **Engineer 1's AI team** — data ingestion pipeline, HIVE
  parallel query logic, feature engineering across ten generators
  (TDA, HMM, Mamba, Graph, GMM, etc.), feature-to-business
  reverse-mapping registry.
- **Engineer 2's AI team** — model training, mathematical
  verification, knowledge distillation (PLE → LGBM).

Three teams running in parallel while staying coherent was made
possible by Phase C's `CLAUDE.md` guardrails plus the *interface
contract verification process*. After every parallel work
session, it was mandatory to verify that the keys written by
file A matched the keys read by file B. Without this routine,
parallel AI agents accumulate subtle key-naming mismatches that
only surface at runtime integration.

## Phase E — Experimentation + papers (Claude Code extension)

Ablation experiments used Claude Code as a *real-time monitoring
tool*. Progress, GPU utilization, error detection — all watched
live and adjusted on the fly. This is how the PLE toggle bug was
caught in live debugging: `use_ple=false` was altering the expert
composition itself, making fair comparison impossible. Days of
results would have been invalidated without the live watch.

Literature research during experiment wait times was the other
shape of this phase. Observing that PLE's val_loss was failing to
converge, dialogue with Opus produced the hypothesis that *the
softmax gate's competitive nature was hindering convergence
among heterogeneous experts*. A search led to the NeurIPS 2024
sigmoid gate paper, providing theoretical grounding; the sigmoid
gate was implemented immediately. Experiments agreed — sigmoid
consistently outperformed softmax. That finding was documented as
a conclusion and carried into subsequent experiment design.

If the story had ended there, the Phase B expert-collapse thread
would have closed cleanly in Phase E. It did not. Weeks later,
after fixing an implementation bug in uncertainty weighting, the
result reversed — softmax began to outperform sigmoid on NDCG
metrics. In retrospect, the root cause traced to broken loss
weighting: with all 13 tasks equally weighted, the numerous
binary-classification gradients overwhelmed multiclass and
regression; sigmoid's non-competitive routing *happened* to act
as a firewall under those broken conditions. With correct loss
weighting, softmax's competitive routing instead acted as a
structural barrier between task-type gradient flows. The sigmoid
literature result — proven in *homogeneous* MTL settings — did
not transfer directly to our heterogeneous 13-task, 3-type regime.

The lesson is not sigmoid vs softmax but a methodological one:
*training bugs can corrupt architectural conclusions*. "Sigmoid
is better" was a valid adaptation to a broken training environment
— not an architectural preference. Without the root-cause
investigation, that conclusion would have persisted indefinitely.
What made the re-investigation possible was long-lived context —
the original sigmoid-adoption reasoning was still accessible when
the new evidence (post-uncertainty-fix reversal) arrived, so
revisiting the earlier conclusion was a continuation rather than
a rebuild.

Paper-writing phase produced four papers (English/Korean) and
twenty-two technical documents through iterative work with
Claude. In this phase AI was not a text generator but a *thought
partner* in constructing the project's meaning.

## Why Claude Code specifically

We split work across five tools, but three things made Claude
Code *non-substitutable* in the implementation, experiment, and
paper phases.

**1M-token context + within-session continuous tracking.** Tracing
three label-leakage cases in sequence was only possible because
days of context stayed alive. After fixing the first (duplicate
`has_nba` column), the second (ground-truth glob ordering) and
the third (generator label input) were found *in the same
session* because the context of the earlier fixes was still
accessible.

**Global survey + local trace, simultaneously.** Simultaneously
diagnosing four FP16 NaN sources (CGC entropy, OT Sinkhorn,
Causal DAG, logits) required surveying the entire model
architecture while tracing numerical operations inside each
expert. Impossible file-by-file.

**Observation → hypothesis → literature → implementation, and
the later re-investigation of the conclusion.** The sigmoid ↔
softmax arc is the clearest example. The forward chain (analysis
→ hypothesis → literature → implementation) ran inside a single
session; weeks later, with the original adoption reasoning still
accessible in context, the re-investigation after new evidence
could proceed as a continuation rather than a rebuild. Had we
switched tools across these chains, we would have lost not just
the forward flow but also the ability to *challenge the earlier
conclusion* once better evidence arrived.

## Patterns that hardened

Several patterns that we never designed but that emerged
naturally — in rough order of how often we said them out loud:

**AI does HOW, humans decide WHAT and WHY.** AI generated code
and text; architecture decisions stayed with humans. The
structural-isomorphism insight emerged from human-AI dialogue,
but *adopting it as a design principle* was a human judgment.

**Guardrails before agents.** `CLAUDE.md` was written *before*
code, not after. Constitution precedes legislation.

**Heterogeneous AI = heterogeneous experts.** The model's
heterogeneous-expert design philosophy was applied to development
tool selection as well. Each AI tool ran a specialized role, and
the combination reached quality and speed unreachable by any
single tool. The Phase A-E division of labor is the concrete
shape of this.

**Fail fast with AI.** Leakage, FP16 NaN, ablation filter
failures — bugs that would have taken days to find manually were
caught and fixed in minutes with AI agents. Fast failure, fast
learning.

## Next

Ep 3 covers the actual mechanisms that kept three parallel AI
agent teams coherent in Phase D — the specific clauses of
`CLAUDE.md`, the eight-file memory bank, the `.claude/RULES.md` ↔
`.cursorrules` synchronization, and what the interface contract
verification that ran after every parallel work session actually
looked like.

Source material:
[Development Story (EN, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/en/development_story_en.pdf)
§2 "Organizing AI Agents".
