---
title: "[Study Thread] REASON-1 — Saying Why Without Making It Up: Tiered Reason Generation for Financial Recommendations"
date: 2026-06-08 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, reason-generation, llm, grounding, hallucination, nlg]
lang: en
excerpt: "How the project turns a model score into a human-readable reason for every recommendation under one hard rule — zero fabricated facts in a regulated financial product. The L1 template → L2a LLM rewrite (→ L2b validation) tiered design, the triple grounding that anchors the LLM to real evidence, and the verdict pass/fail gate (json_object structured output) that blocks silent hallucination risk."
series: study-thread
part: 25
alt_lang: /2026/06/08/reason-generation-ko/
next_title: "REASON-2 — Grounding by Feature Reverse-Mapping: From a 734D Vector Back to a Sentence a Human Trusts"
next_desc: "The mechanics of grounding itself: how Integrated Gradients picks the top features, how a normalized 734D feature index reverse-maps to human-readable ranges and category labels, and how that reverse-mapped text becomes the 'ground truth' every downstream LLM call is checked against."
next_status: draft
---

*First post of the Reason Generation sub-thread in the "Study Thread"
series. Across this and the following posts, in parallel Korean and
English, I unpack how the project explains itself — how a recommendation
score becomes a sentence a customer and a call-center agent can read.
The source is the on-prem reference
`기술참조서/추천사유생성_기술_참조서`, and the full PDF will be attached
to the final post of the sub-thread. The PLE, adaTT, and TDA sub-threads
asked what the model reads and how tasks share. This one asks the
question that comes after the score is computed: how do you say* why *—
in a financial product where a single fabricated fact is a compliance
incident, not a typo?*

> **The constraint that shapes everything.** A general chatbot that
> hallucinates a detail is annoying. A financial recommendation engine
> that tells a customer "you are guaranteed 7% returns" when no such
> product exists is a regulatory violation under the Financial Consumer
> Protection Act. The reason-generation pipeline is built backwards from
> that fear: not "how do we make the text fluent?" but "how do we make
> it *impossible* to ship a fabricated or non-compliant sentence?" Every
> design choice below — the template floor, the rewrite ceiling, the
> grounding, the verdict gate — is an answer to that single question.

## Why Generate Reasons At All

A recommendation model can be flawless and still fail in practice if it
cannot say *why*. In a financial setting the "why" is not a nicety — it
is load-bearing in three directions at once:

- **Regulation.** Korea's AI Basic Act (Articles 31, 34) requires a
  high-risk AI system to explain the basis of its decisions; the
  Financial Consumer Protection Act (Article 19) imposes a duty to
  explain under the suitability principle. Every recommendation needs a
  reason, and every reason needs an auto-attached AI-disclosure notice.
- **Trust.** "Recommended by AI analysis" earns nothing. A reason
  grounded in the customer's actual spending pattern, consultation
  history, and life stage measurably lifts acceptance — the reference
  cites a 20–40% uplift from explained-vs-unexplained recommendations.
- **Audit.** When the financial supervisor inspects an individual case,
  the system must reconstruct *when, on what data, by which model, with
  what reason* — retroactively, per recommendation.

So the requirement is blunt: a reason for **all 12 million customers**,
every one of them defensible, none of them invented.

## The Tiered Design — and Why Tiered

The naive options both fail. Generate every reason with an LLM and you
pay a GPU bill the reference puts at roughly 1,000 GPU-hours — against
~162 for the tiered design — and accept hallucination risk across 12M
cases. Use only templates and you satisfy
the regulator but ship mechanical, one-size-fits-30 prose that cannot
reflect a complex customer. The project refuses the dichotomy and
**layers** the two:

<figure style="margin:24px auto;max-width:640px;">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="640" height="300" fill="#f8fafc" rx="8"/>
  <!-- inputs -->
  <rect x="24" y="24" width="120" height="48" rx="6" fill="#1e3a5f15" stroke="#1e3a5f" stroke-width="1"/>
  <text x="84" y="44" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">Model score</text>
  <text x="84" y="60" text-anchor="middle" font-size="9" fill="#64748b">PLE-adaTT</text>
  <rect x="24" y="84" width="120" height="48" rx="6" fill="#1e3a5f15" stroke="#1e3a5f" stroke-width="1"/>
  <text x="84" y="104" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">IG Top-5</text>
  <text x="84" y="120" text-anchor="middle" font-size="9" fill="#64748b">feature attribution</text>
  <!-- L1 -->
  <rect x="196" y="48" width="130" height="60" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.4"/>
  <text x="261" y="72" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">L1 Template</text>
  <text x="261" y="90" text-anchor="middle" font-size="9" fill="#64748b">all 12M · LLM 0 calls</text>
  <text x="261" y="102" text-anchor="middle" font-size="9" fill="#64748b">deterministic floor</text>
  <!-- richness -->
  <rect x="196" y="130" width="130" height="40" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1"/>
  <text x="261" y="148" text-anchor="middle" font-size="10" font-weight="700" fill="#64748b">Richness sort</text>
  <text x="261" y="162" text-anchor="middle" font-size="9" fill="#94a3b8">rich / moderate / sparse</text>
  <!-- L2a -->
  <rect x="380" y="40" width="140" height="60" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.4"/>
  <text x="450" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">L2a LLM Rewrite</text>
  <text x="450" y="79" text-anchor="middle" font-size="9" fill="#64748b">rich+moderate (~500K/wk)</text>
  <text x="450" y="91" text-anchor="middle" font-size="9" fill="#64748b">quality ceiling</text>
  <!-- L2b -->
  <rect x="380" y="118" width="140" height="56" rx="6" fill="#d9770615" stroke="#d97706" stroke-width="1.4"/>
  <text x="450" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">L2b Validation</text>
  <text x="450" y="157" text-anchor="middle" font-size="9" fill="#64748b">sampled · 3-axis post-hoc</text>
  <!-- audit -->
  <rect x="380" y="200" width="140" height="52" rx="6" fill="#e11d4815" stroke="#e11d48" stroke-width="1.2"/>
  <text x="450" y="222" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Audit Archive</text>
  <text x="450" y="238" text-anchor="middle" font-size="9" fill="#64748b">DuckDB + Parquet</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="144" y1="48" x2="194" y2="70"/><polygon points="194,70 185,67 187,76"/>
    <line x1="144" y1="108" x2="194" y2="86"/><polygon points="194,86 187,80 185,89"/>
    <line x1="261" y1="108" x2="261" y2="128"/><polygon points="261,128 257,120 265,120"/>
    <line x1="326" y1="142" x2="378" y2="70"/><polygon points="378,70 369,71 374,79"/>
    <line x1="326" y1="150" x2="378" y2="145"/><polygon points="378,145 370,141 369,150"/>
    <line x1="450" y1="100" x2="450" y2="116"/><polygon points="450,116 446,108 454,108"/>
    <line x1="450" y1="174" x2="450" y2="198"/><polygon points="450,198 446,190 454,190"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The two-layer full-coverage architecture. L1 is the deterministic floor under all 12M customers; L2a is the LLM quality ceiling for the context-rich; L2b samples both for post-hoc quality. Every case lands in the audit archive.</figcaption>
</figure>

The division of labor is the whole point:

| Layer | Target | Method | LLM calls | Cost |
| --- | --- | --- | --- | --- |
| **L1** | all 12M | template (6 categories × 5 variants = 30, hash-selected) | 0 | ~20 min, CPU only |
| **L2a** | rich + moderate (~500K/wk) | LLM rewrite (Ollama dual-route + 3-layer Safety Gate) | 1 | ~1.0 s/case |
| **L2b** | sampled (~67K) | quality validation (factuality, relevance, naturalness) | 1 | post-hoc |

L1 is the **floor**: a template engine that reverse-maps Integrated
Gradients top features to a reason for *every* customer, with zero GPU
cost (just a `customer_id` hash to pick one of 30 variants
deterministically — same customer, same wording, always reproducible).
That floor alone satisfies the equal-explanation duty of FCPA Art. 19.

L2a is the **ceiling**: for customers whose context is rich enough to
say something specific, an LLM rewrites the mechanical template draft
into natural prose. Its governing principle is stated bluntly in the
source — *on failure, keep the L1 original*. A rewrite that fails any
gate does not blank the reason; it falls back to the template. An empty
reason is **never** shipped.

L2b is the **monitor**: a slim, sampled, post-hoc quality check over
both L1-only output and L2a rewrites — it does not gate the batch, it
watches it.

> **Historical context.** The "form vs. meaning" worry under all of this
> is Bender & Koller's ACL 2020 paper *"Climbing towards NLU: On
> Meaning, Form, and Understanding in the Age of Data."* Their argument:
> a language model trained only on *form* — co-occurrence of symbols —
> cannot acquire *meaning* without a link to the world outside the text.
> Their Octopus thought experiment dramatizes it: an agent that has only
> ever seen the symbols, never the referents, can mimic fluent
> conversation yet has no idea what any of it points at. A "stochastic
> parrot" — the coinage is from Bender et al.'s separate 2021 paper, but
> the worry is the same — produces plausible form unanchored to fact. The whole
> grounding design below is the engineering reply: never let the LLM
> speak from form alone — force every sentence to stand on explicit,
> externally-supplied evidence.

## Grounding — Anchoring the LLM to Evidence

If the L2a LLM is the risk, grounding is the leash. *Grounding* means
forcing the model's output to rest on external ground truth rather than
on whatever the language prior wants to say. In this system "ground"
carries three meanings at once, all injected into the prompt before the
model writes a word:

1. **Feature grounding.** The Top-5 feature attributions from Integrated
   Gradients — the *actual* basis of the model's score — are written
   into the prompt, so the reason is anchored to why the model decided,
   not to a guess.
2. **Customer grounding.** Real customer data — segment, transaction
   pattern, consultation history — is injected to suppress
   hallucination by giving the model concrete facts to lean on.
3. **Regulatory grounding.** The system prompt enumerates the FCPA
   violation patterns, and rule-based checks enforce them, so compliance
   is a hard constraint rather than a hope.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <text x="310" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Grounded rewrite vs. ungrounded parrot</text>
  <!-- grounded path -->
  <rect x="24" y="56" width="150" height="150" rx="6" fill="#0d948810" stroke="#0d9488" stroke-width="1.2"/>
  <text x="99" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Grounded</text>
  <text x="40" y="98" font-size="9" fill="#64748b">evidence in prompt:</text>
  <text x="40" y="114" font-size="9" fill="#1e3a5f">• IG: spend_food ↑ (0.31)</text>
  <text x="40" y="128" font-size="9" fill="#1e3a5f">• segment: WARMSTART</text>
  <text x="40" y="142" font-size="9" fill="#1e3a5f">• 8 consultations / 1y</text>
  <line x1="40" y1="152" x2="158" y2="152" stroke="#cbd5e1" stroke-width="0.8"/>
  <text x="40" y="170" font-size="9" fill="#0d9488" font-weight="700">→ "Since you spend</text>
  <text x="40" y="183" font-size="9" fill="#0d9488" font-weight="700">  most on dining, this</text>
  <text x="40" y="196" font-size="9" fill="#0d9488" font-weight="700">  card fits your pattern."</text>
  <!-- check -->
  <circle cx="186" cy="130" r="14" fill="#0d9488"/>
  <path d="M 180 130 l 4 5 l 9 -11" stroke="#fff" stroke-width="2.4" fill="none"/>
  <!-- ungrounded path -->
  <rect x="262" y="56" width="150" height="150" rx="6" fill="#e11d4810" stroke="#e11d48" stroke-width="1.2"/>
  <text x="337" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Ungrounded</text>
  <text x="278" y="98" font-size="9" fill="#64748b">no evidence, form only:</text>
  <text x="278" y="118" font-size="9" fill="#94a3b8">(language prior fills</text>
  <text x="278" y="131" font-size="9" fill="#94a3b8"> the gap with plausible</text>
  <text x="278" y="144" font-size="9" fill="#94a3b8"> but invented detail)</text>
  <line x1="278" y1="152" x2="396" y2="152" stroke="#fecaca" stroke-width="0.8"/>
  <text x="278" y="170" font-size="9" fill="#e11d48" font-weight="700">→ "Guaranteed 7%</text>
  <text x="278" y="183" font-size="9" fill="#e11d48" font-weight="700">  returns, zero risk,</text>
  <text x="278" y="196" font-size="9" fill="#e11d48" font-weight="700">  you must sign up."</text>
  <!-- block -->
  <circle cx="424" cy="130" r="14" fill="#e11d48"/>
  <line x1="418" y1="124" x2="430" y2="136" stroke="#fff" stroke-width="2.4"/>
  <line x1="430" y1="124" x2="418" y2="136" stroke="#fff" stroke-width="2.4"/>
  <!-- verdict gate -->
  <rect x="470" y="96" width="130" height="68" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="535" y="120" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Verdict gate</text>
  <text x="535" y="138" text-anchor="middle" font-size="9" fill="#64748b">factuality + compliance</text>
  <text x="535" y="151" text-anchor="middle" font-size="9" fill="#64748b">→ pass / revise / reject</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="200" y1="130" x2="468" y2="120"/><polygon points="468,120 459,120 462,128"/>
    <line x1="438" y1="130" x2="468" y2="135"/><polygon points="468,135 460,131 459,140"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Same model score, two prompts. With evidence injected the rewrite restates real facts; without it the language prior invents a compliance violation. The verdict gate is the last check that catches the parrot.</figcaption>
</figure>

The mathematics is one line. Autoregressive generation factorizes as

$$ P(\mathbf{r} \mid \mathbf{p}) = \prod_{j=1}^{m} P\!\left(r_j \mid p_1,\dots,p_n,\; r_1,\dots,r_{j-1}\right) $$

— each reason token $r_j$ is conditioned on the whole prompt $\mathbf{p}$
and the tokens already written. The implication is exactly the design
thesis: *the quality of the prompt $\mathbf{p}$ determines the quality of
the reason $\mathbf{r}$.* The richer the IG attributions, customer
features, and consultation history packed into $\mathbf{p}$, the more
probability mass $P(\mathbf{r}\mid\mathbf{p})$ puts on factually faithful
reasons — and the less is left for the parrot.

> **Design intuition.** Grounding is not a filter applied *after*
> generation — it reshapes the distribution the model samples *from*.
> The rewrite runs at temperature $\tau = 0.3$: low enough that the
> output stays close to the high-probability, evidence-consistent tokens
> (the prompt instruction even says "do not add numbers like rates"),
> high enough that two runs phrase it differently without changing the
> facts. Grounding narrows *what* can be said; temperature controls *how
> freely* it is said. The verdict gate then catches whatever still slips
> through.

This is, in spirit, *Structured RAG* — the retrieval-augmented
generation pattern (Lewis et al., NeurIPS 2020), except what gets
"retrieved" into the context is not unstructured documents but
structured feature attributions and a customer profile.

## The Verdict Gate — Blocking Silent Risk

Grounding lowers the odds of a fabricated sentence; it does not
eliminate them. So a generated reason is never shipped on trust. A
separate critique step scores it and returns a **verdict**:

$$ \text{verdict} = \begin{cases} \text{pass} & \text{if } f \ge 0.8 \;\wedge\; c \ge 1.0 \\ \text{revise} & \text{if } f \ge 0.5 \;\wedge\; c \ge 1.0 \\ \text{reject} & \text{otherwise} \end{cases} $$

where $f$ is the **factuality** score (0–1) and $c$ is the
**compliance** score (0–1). The structure encodes the financial-sector
priority directly:

- **Compliance $c$ is a binary gate.** $c = 1.0$ means no FCPA violation
  at all; $c < 1.0$ means *at least one*. A single violation forces
  `reject` no matter how factual the text is. Regulation outranks
  quality, full stop.
- **Factuality $f$ is continuous.** At $\ge 0.8$ the reason matches the
  source data well enough to ship. Between 0.5 and 0.8 it has partial
  hallucination and is sent back for one revision. Below 0.5 the LLM
  hallucinated badly and the system falls back to the safe template.
- **Revision is capped at one pass.** A reason that earns `revise` twice
  becomes `reject`, bounding LLM calls at three (generate + critique +
  regenerate/re-critique) and preventing an infinite loop.

The critical operational detail is *how* a verdict like this is read out
of an LLM at all. The L2b quality-critique call (the post-hoc monitor we
meet below) passes `response_format={"type": "json_object"}` so the model
returns parseable structured output — necessary because Qwen3's
`<think>...</think>` blocks and free-text drift otherwise corrupt JSON
parsing. And there is a sharp lesson baked into its fallback: when
parsing the critique fails, the fallback verdict was changed from
`'pass'` to **`'fail'`**. The reasoning is the heart of the whole
pipeline — *a verdict you cannot read is not a safe verdict.* Defaulting
an unparseable critique to "pass" would let a hallucinated reason slip
through quality control silently. Defaulting to "fail" makes the failure
loud: the case is scored as a failure, escalated to the heavier critique
model, and surfaces in the quality report. This is the difference between
**silent risk** and **safe failure**.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <text x="280" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Unparseable critique: silent risk vs. safe failure</text>
  <!-- llm critique -->
  <rect x="200" y="48" width="160" height="44" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="280" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">L2b critique LLM</text>
  <text x="280" y="83" text-anchor="middle" font-size="9" fill="#64748b">json_object requested</text>
  <!-- parse fail -->
  <rect x="200" y="108" width="160" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>
  <text x="280" y="126" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">JSON parse fails</text>
  <text x="280" y="140" text-anchor="middle" font-size="9" fill="#64748b">(think block / free text)</text>
  <!-- two branches -->
  <rect x="40" y="170" width="220" height="48" rx="6" fill="#e11d4810" stroke="#e11d48" stroke-width="1.2"/>
  <text x="150" y="190" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">fallback = 'pass'  ✗ (old)</text>
  <text x="150" y="206" text-anchor="middle" font-size="9" fill="#64748b">hallucination passes QA silently</text>
  <rect x="300" y="170" width="220" height="48" rx="6" fill="#0d948810" stroke="#0d9488" stroke-width="1.2"/>
  <text x="410" y="190" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">fallback = 'fail'  ✓ (now)</text>
  <text x="410" y="206" text-anchor="middle" font-size="9" fill="#64748b">scored fail — escalated, reported</text>
  <!-- arrows -->
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="280" y1="92" x2="280" y2="106"/><polygon points="280,106 276,98 284,98"/>
    <line x1="240" y1="148" x2="150" y2="168"/><polygon points="150,168 159,165 156,173"/>
    <line x1="320" y1="148" x2="410" y2="168"/><polygon points="410,168 401,165 404,173"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">The fallback verdict flip. Defaulting an unreadable critique to 'pass' is silent risk; defaulting to 'fail' converts it into a loud failure that gets escalated and lands in the quality report.</figcaption>
</figure>

L2a adds its own **3-Layer Safety Gate** in front of acceptance, with the
same fail-safe spirit: Gate 1 rejects empty strings and JSON debris
(text starting with `{` or `[`); Gate 2 rejects the six FCPA violation
keyword patterns (e.g. "guaranteed returns," "principal protected,"
"n% returns," "no loss," "must sign up"); Gate 3 rejects length outside
30–200 chars or Korean ratio below 80%. Pass all three and the rewrite is
applied — otherwise the L1 template stands.

## The LLM Provider Abstraction

A closed-network financial deployment cannot call a hosted API, and it
should not hard-wire one model. The project routes all of this through a
single `LLMProviderFactory` with six interchangeable backends behind one
`generate()` signature:

| Backend | What it is | Notes |
| --- | --- | --- |
| `ollama` / `qwen` | local Ollama, OpenAI-compatible | default `qwen3:14b` at `host.docker.internal:11434/v1` |
| `exaone` | LG AI Research Exaone | vLLM or Ollama self-hosted (`EXAONE_BASE_URL`) |
| `solar` | Upstage Solar | `upstage_api` REST or `local` self-hosted |
| `local` | generic OpenAI-compatible `/v1/chat/completions` | any in-house server |
| `dummy` | test/mock | deterministic JSON for unit tests |

In production the L2a/L2b engines run an **Ollama dual-route**: a light,
fast primary (`exaone3.5:2.4b`) handles the bulk, with **escalation** to
`qwen3:14b` when the primary stumbles — a parse or quality failure
(Gate 1/3) in L2a, a non-pass or unparseable critique in L2b. The
`L2aRewriteResult` record carries this explicitly — `primary_model`,
`primary_gate`, `escalation_used`, `escalation_model` — so every case is
auditable down to which model wrote it and whether escalation fired. The
unified Factory currently backs the diagnostic consensus path while the
operational engines migrate onto it incrementally; the `generate()`
contract is shared, so the move is mechanical.

## Evaluation — Two Verdicts, Two Roles

The pipeline runs two distinct quality checks, and the difference between
them is instructive:

- **Self-Critique (real-time gatekeeper).** Two axes — factuality and
  compliance — threshold $0.8$, runs at $\tau = 0.1$ for near-deterministic
  judgment. It *blocks*: a `reject` here means the LLM reason never
  reaches the customer.
- **L2b validation (post-hoc monitor).** Three axes — factuality $f$,
  relevance $r$, naturalness $n$ — threshold $0.7$, on a sample. It
  *watches*: `needs_improvement` does not block the batch but accumulates
  as feedback for prompt improvement.

$$ \text{verdict}_{\text{L2b}} = \begin{cases} \text{pass} & \text{if } f \ge 0.7 \;\wedge\; r \ge 0.7 \;\wedge\; n \ge 0.7 \\ \text{needs\_improvement} & \text{if any score} \in [0.5, 0.7) \\ \text{fail} & \text{if any score} < 0.5 \end{cases} $$

Two design choices fall out of this. L2b adds the **naturalness** axis
the gatekeeper omits — because L2a's entire job is to turn mechanical
templates into natural sentences, so naturalness is the thing worth
monitoring. And L2b's bar is *lower* (0.7 vs 0.8) precisely because it is
a monitor, not a gate: the real-time gatekeeper must be stricter than the
post-hoc auditor. L2b samples two sources — a ~0.4% stratified slice of
L1-only customers (~51K, template-quality monitoring) and a 5% audit of
L2a rewrites (~16K) — and the operational KPIs the pipeline tracks make the
priorities concrete: `l1_coverage_rate = 1.0`, `l2a_gate_pass_rate ≥ 0.9`,
`l2b_factual_score ≥ 0.8`, `fallback_rate ≤ 0.1`.

## Where We Stop

We started from one hard constraint — zero fabricated facts in a
regulated financial product — and watched the whole pipeline fall out of
it. L1 is the deterministic template floor that covers all 12M and
satisfies the explanation duty. L2a is the LLM ceiling that lifts quality
for the context-rich, always keeping the template as a safety net. Triple
grounding reshapes what the LLM can say; the verdict gate blocks what
grounding misses, and the structured `json_object` read-out with its
`fail`-by-default fallback converts silent hallucination risk into loud
safe failure. Six interchangeable
backends keep it all inside the closed network, with a dual-route that
escalates only when it must.

What we deferred is the part that makes grounding *work* in the first
place: how a Top-5 attribution from Integrated Gradients over a
normalized 734D feature vector actually reverse-maps to a human-readable
range and category — `spend_food ↑ (0.31)` becoming "you spend most on
dining." That reverse-mapping is the ground truth every verdict is
checked against, and it is the subject of the next post, **REASON-2**.
