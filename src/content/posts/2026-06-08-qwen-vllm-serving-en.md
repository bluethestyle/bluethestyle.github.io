---
title: "[Study Thread] QWEN-1 — Serving the Rewriter: vLLM, Qwen, and the Provider Abstraction Behind Closed-Network Reason Generation"
date: 2026-06-08 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, vllm, qwen, llm-serving, pagedattention, on-prem]
lang: en
excerpt: "The final post of the tech-reference arc: how an on-prem LLM serving stack rewrites recommendation reasons without ever leaving a closed financial network. PagedAttention's OS-paging trick for the KV cache, continuous batching for throughput, AWQ 4-bit quantization to fit Qwen on a 12GB GPU, the OpenAI-compatible endpoint with JSON output, and the LLMProviderFactory that lets one call site swap between Ollama, Qwen, Exaone, Solar, and a generic local OpenAI-compatible backend."
series: study-thread
part: 27
alt_lang: /2026/06/08/qwen-vllm-serving-ko/
next_title: "Tech-reference arc complete — next, real-data evaluation"
next_desc: "The seven Experts, the offline feature pipeline, the teacher–student distillation, the scoring path, the reason generator, and now the serving stack are all documented. What remains is no longer architecture but evidence: does the whole machine actually move the needle on real customers? The thread turns from how it is built to whether it works."
next_status: draft
---

*Final post of the tech-reference arc in the "Study Thread" series, and
the close of the LLM serving sub-thread. In parallel Korean and
English, this one steps out of the model and into the* infrastructure
*that runs it. The source is the on-prem reference
`기술참조서/Qwen_vLLM_기술_참조서`, and the full PDF will be attached to
this post — it is not yet in the public assets folder, so for now treat
the citation as pending. Where the earlier sub-threads asked what an
Expert reads or how tasks share, this one asks the most operational
question of all: once the recommendation and its reason are written, how
do you actually run a language model to polish that reason — half a
million times a week, on one consumer GPU, inside a network that can
never call a cloud API?*

> **The constraint that shapes everything.** This is a financial
> closed network. There is no OpenAI, no Anthropic, no Gemini endpoint
> reachable from inside — every token is generated on hardware we own.
> The serving trilemma (latency, throughput, memory) is therefore not
> an abstraction but a hard wall: a single RTX 4070 with **12 GB** of
> VRAM, a weekly L2a rewrite load of **~500,000** reasons, and a model
> whose raw FP16 weights (**16 GB**) do not even fit. Everything below
> — quantization, paged KV cache, continuous batching, a provider
> abstraction — exists to get through that wall.

## The On-Prem Wall

When you describe an LLM serving problem in the cloud, you reach for
autoscaling and a managed endpoint. Neither exists here. The reason
generator and the L2a rewriter both consume a self-hosted model, and
the reference frames the difficulty as a *trilemma* — three resources
that trade against each other:

- **Latency** — time to first token (TTFT) and full completion. Reasons
  surface in face-to-face scenarios; a slow response degrades the
  experience.
- **Throughput** — requests per unit time. ~500k L2a rewrites a week;
  processed one at a time, that is over 140 GPU-hours.
- **Memory** — the model weights *and* every intermediate state must
  live inside one card's VRAM.

A naive inference server cannot optimise all three at once. The stack
the reference builds is, in order: **AWQ 4-bit quantization** to make
the model fit at all, **PagedAttention** to stop the KV cache from
wasting memory, **continuous batching** to stop the GPU from idling,
and a **provider abstraction** so the call site never hard-codes which
engine answers.

## Why the KV Cache Is the Real Bottleneck

A decoder-only model is autoregressive: token $t$ is predicted from all
$t-1$ tokens before it. Without caching, every new token would
recompute attention over the whole prefix — $O(T^2)$ work to emit $T$
tokens. The **KV cache** stores the already-computed Key and Value
vectors so each step only attends the new Query against the cache,
collapsing per-token cost to $O(d_k)$.

That cache is the memory hog. Its size grows linearly with sequence
length and with the model's shape:

$$ M_\text{KV} = 2 \cdot L \cdot H_\text{KV} \cdot D \cdot S \cdot B \cdot \text{sizeof}(\text{dtype}) $$

> **Equation intuition.** The leading $2$ is Key *and* Value; $L$ is the
> layer count, $H_\text{KV}$ the number of KV heads, $D$ the head
> dimension, $S$ the sequence length, $B$ the number of concurrent
> requests, and $\text{sizeof}$ the dtype width (FP16 = 2 bytes). For
> Qwen3-8B ($L=32$, $H_\text{KV}=8$ thanks to GQA, $D=128$, FP16,
> $S=2048$) this works out to roughly **268 MB per concurrent request**.
> With a ~4 GB KV budget that is about **15** requests in flight — and
> if Grouped-Query Attention did not shrink the KV heads from 32 to 8,
> that 4× saving would vanish and you would be down to three or four.

The trouble is not just the size — it is *how the size gets allocated*.
A traditional engine reserves a contiguous block sized to the *maximum*
sequence length for every request. A reason that uses 150 tokens still
holds a 2048-token reservation; the other 93% is **internal
fragmentation**, dead memory no one else can touch. Freed requests
leave gaps too small to reuse — **external fragmentation**. Real-world
utilisation lands at 60–80%, and that directly caps how many requests
fit at once.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <!-- LEFT: contiguous, fragmented -->
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Contiguous reservation</text>
  <text x="150" y="44" text-anchor="middle" font-size="10" fill="#64748b">one block per request, sized to max</text>
  <!-- request A reservation -->
  <rect x="40" y="60" width="220" height="34" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="4 3"/>
  <rect x="40" y="60" width="42" height="34" fill="#0d9488"/>
  <rect x="82" y="60" width="178" height="34" fill="#e2e8f0"/>
  <text x="171" y="82" text-anchor="middle" font-size="9" fill="#94a3b8">wasted (internal frag.)</text>
  <!-- request B reservation -->
  <rect x="40" y="104" width="220" height="34" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="4 3"/>
  <rect x="40" y="104" width="70" height="34" fill="#4f46e5"/>
  <rect x="110" y="104" width="150" height="34" fill="#e2e8f0"/>
  <text x="185" y="126" text-anchor="middle" font-size="9" fill="#94a3b8">wasted</text>
  <!-- gap -->
  <rect x="40" y="148" width="220" height="22" fill="#fde68a" stroke="#d97706" stroke-width="0.8" stroke-dasharray="3 2"/>
  <text x="150" y="163" text-anchor="middle" font-size="9" fill="#d97706">freed gap — too small to reuse (external frag.)</text>
  <text x="150" y="196" text-anchor="middle" font-size="11" fill="#e11d48" font-weight="700">utilisation 60–80%</text>
  <!-- divider -->
  <line x1="300" y1="50" x2="300" y2="200" stroke="#e2e8f0" stroke-width="1"/>
  <!-- RIGHT: paged blocks -->
  <text x="450" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">PagedAttention blocks</text>
  <text x="450" y="44" text-anchor="middle" font-size="10" fill="#64748b">fixed B-token blocks, placed anywhere</text>
  <g>
    <rect x="330" y="60" width="30" height="30" fill="#0d9488" rx="3"/><rect x="366" y="60" width="30" height="30" fill="#4f46e5" rx="3"/>
    <rect x="402" y="60" width="30" height="30" fill="#0d9488" rx="3"/><rect x="438" y="60" width="30" height="30" fill="#94a3b8" rx="3"/>
    <rect x="474" y="60" width="30" height="30" fill="#4f46e5" rx="3"/><rect x="510" y="60" width="30" height="30" fill="#0d9488" rx="3"/>
    <rect x="330" y="96" width="30" height="30" fill="#4f46e5" rx="3"/><rect x="366" y="96" width="30" height="30" fill="#94a3b8" rx="3"/>
    <rect x="402" y="96" width="30" height="30" fill="#0d9488" rx="3"/><rect x="438" y="96" width="30" height="30" fill="#4f46e5" rx="3"/>
    <rect x="474" y="96" width="30" height="30" fill="#94a3b8" rx="3"/><rect x="510" y="96" width="30" height="30" fill="#0d9488" rx="3"/>
  </g>
  <text x="450" y="150" text-anchor="middle" font-size="9.5" fill="#64748b">block table maps logical → physical</text>
  <text x="450" y="166" text-anchor="middle" font-size="9.5" fill="#64748b">last block waste ≤ B−1 tokens</text>
  <text x="450" y="196" text-anchor="middle" font-size="11" fill="#0d9488" font-weight="700">utilisation 96%+</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Contiguous reservation wastes the unused tail of every request and leaves unreusable gaps; paged blocks scatter fixed-size chunks anywhere and cap waste at one block.</figcaption>
</figure>

## vLLM's Core Trick — Paging the KV Cache

PagedAttention's idea is almost embarrassingly simple: *manage the KV
cache the way an operating system manages physical memory.* An OS hands
each program a contiguous virtual address space, but lays the actual
pages down non-contiguously and tracks the mapping in a page table.
PagedAttention does the same with KV **blocks** — fixed chunks holding
$B$ tokens' worth of K and V — and a **block table** mapping each
sequence's logical block numbers to physical ones.

| Concept | OS virtual memory | PagedAttention |
| --- | --- | --- |
| Unit | 4 KB page | KV block ($B$ tokens of K, V) |
| Mapping | page table | block table |
| Allocation | demand paging | lazy, on-demand per token |
| Sharing | copy-on-write | copy-on-write (beam search) |
| Fragmentation fix | non-contiguous pages | non-contiguous blocks |

Because blocks are allocated lazily — one new block only when the
current one fills — the *only* waste is the unused slots in a
sequence's final block: at most $B-1$ tokens. With $B=16$ that is about
0.7% of a 2048-token sequence. Utilisation jumps from 60–80% to **96%+**,
and since memory was the thing capping concurrency, that translates
almost directly into more requests in flight.

> **Historical context.** PagedAttention came from Woosuk Kwon and
> colleagues at UC Berkeley, published at **SOSP 2023** — the top
> operating-systems venue — as *"Efficient Memory Management for Large
> Language Model Serving with PagedAttention."* That an ML-serving
> paper landed at SOSP at all was unusual; its argument was that the
> real bottleneck in LLM serving is *memory management, not
> computation*, and that the OS community's virtual-memory toolkit maps
> cleanly onto the GPU KV cache. vLLM, the engine built around it,
> became the de facto open-source serving standard within a year.

The second half of the trick is **continuous batching**. Static
batching groups $N$ requests and holds every slot until the *longest*
one finishes — so a 175-token L2a rewrite sits idle waiting on a
500-token reason. Continuous batching schedules at the *token* level
instead: every decode step, completed requests are evicted immediately
and queued requests slot straight into the freed space, keeping the GPU
near its maximum batch the whole time.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="260" fill="#f8fafc" rx="8"/>
  <!-- static -->
  <text x="20" y="30" font-size="13" font-weight="700" fill="#1e3a5f">Static batching</text>
  <line x1="90" y1="120" x2="600" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <text x="600" y="135" text-anchor="end" font-size="9" fill="#94a3b8">time →</text>
  <!-- 4 slots, finish at different times but held to max -->
  <rect x="90" y="48" width="150" height="14" fill="#0d9488" rx="2"/><rect x="240" y="48" width="270" height="14" fill="#fecaca" rx="2"/>
  <rect x="90" y="66" width="420" height="14" fill="#4f46e5" rx="2"/>
  <rect x="90" y="84" width="90" height="14" fill="#d97706" rx="2"/><rect x="180" y="84" width="330" height="14" fill="#fecaca" rx="2"/>
  <rect x="90" y="102" width="200" height="14" fill="#64748b" rx="2"/><rect x="290" y="102" width="220" height="14" fill="#fecaca" rx="2"/>
  <line x1="510" y1="42" x2="510" y2="122" stroke="#e11d48" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="514" y="55" font-size="9" fill="#e11d48">batch ends</text>
  <text x="375" y="76" text-anchor="middle" font-size="8.5" fill="#e11d48" font-weight="700">idle GPU (pink)</text>
  <!-- continuous -->
  <text x="20" y="170" font-size="13" font-weight="700" fill="#1e3a5f">Continuous batching</text>
  <line x1="90" y1="248" x2="600" y2="248" stroke="#cbd5e1" stroke-width="1"/>
  <rect x="90" y="186" width="150" height="14" fill="#0d9488" rx="2"/><rect x="240" y="186" width="160" height="14" fill="#14b8a6" rx="2"/><rect x="400" y="186" width="110" height="14" fill="#2dd4bf" rx="2"/>
  <rect x="90" y="204" width="420" height="14" fill="#4f46e5" rx="2"/>
  <rect x="90" y="222" width="90" height="14" fill="#d97706" rx="2"/><rect x="180" y="222" width="120" height="14" fill="#f59e0b" rx="2"/><rect x="300" y="222" width="210" height="14" fill="#fbbf24" rx="2"/>
  <text x="300" y="216" text-anchor="middle" font-size="8.5" fill="#0d9488" font-weight="700">freed slots refilled immediately — no idle</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Static batching pins every slot to the longest request (pink = wasted GPU). Continuous batching evicts finished requests each step and refills from the queue, keeping the device busy.</figcaption>
</figure>

The combined effect is real but worth stating honestly: the reference
notes vLLM achieves roughly **35% higher throughput** than Ollama from
this PagedAttention + continuous-batching pairing — *and* that the
system today runs Ollama for closed-network operational convenience,
with a vLLM switch held as a future option. So the engine below is the
documented canonical design; the live L2a path is described at the end.

## Making Qwen Fit — AWQ 4-bit on 12 GB

The model is **Qwen3-8B**: a decoder-only Transformer with RoPE
rotary positions, GQA (32 query heads sharing 8 KV groups), and SwiGLU
activations, a 151,936-token vocabulary, and up to 32K context (capped
to 2,048 here). It was chosen not for raw reasoning power but for the
task at hand — the LLM is a *text editor*, not a feature interpreter;
the features are already turned into Korean by a rule-based mapper, and
the model only has to synthesise fluent, schema-valid output. So the
selection criteria were JSON-compliance, Korean quality, and batch
speed.

| Model | VRAM | JSON compliance | Korean | Speed |
| --- | --- | --- | --- | --- |
| **Qwen3-8B-AWQ** | ~5.5 GB | high | excellent | fast |
| Gemma2-9B | ~6.0 GB | moderate | moderate | moderate |
| Llama3-8B | ~5.5 GB | moderate | weak | fast |
| Mistral-7B | ~5.0 GB | moderate | weak | fast |

But 8B parameters in FP16 is **16 GB** — it does not fit on a 12 GB
card. **AWQ** (Activation-Aware Weight Quantization, Lin et al., MLSys
2024) gets it to **~5.5 GB** at 4-bit. Its insight is that not all
weights matter equally: roughly 1% are *salient*, and saliency tracks
the magnitude of the channel's activation, because output error scales
as $\sum_j \lVert \delta W_{j,\cdot}\rVert^2 \cdot \lVert X_{\cdot,j}\rVert^2$.
AWQ finds those channels from a small calibration set and scales them up
by $s_j = \lVert X_{\cdot,j}\rVert^\alpha$ ($\alpha\approx0.5$) before
quantizing, so their rounding grid is relatively finer; multiplying by
$s_j^{-1}$ at inference restores the scale while leaving the error
reduced. The budget on the card then reads: ~5.5 GB weights + ~4.0 GB
KV cache (at `--gpu-memory-utilization 0.85`) + ~1.5 GB CUDA overhead +
~1.0 GB margin = 12 GB.

## The OpenAI-Compatible Endpoint and JSON Output

vLLM exposes an HTTP API that obeys the OpenAI spec, so the existing
`openai.OpenAI` SDK calls it unchanged — only the `base_url` points at
a local server, and the API key is a throwaway because a local vLLM
needs none. The canonical launch:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

`--max-model-len 2048` is deliberate: prompts run ~800 input + ~500
output tokens, so 2,048 is ample, and a needlessly large window would
burn KV-cache memory and shrink concurrency. A client sending
`messages` gets them auto-converted into Qwen3's ChatML
(`<|im_start|>system / user / assistant`) server-side — so wrapping
ChatML tags yourself double-wraps the prompt.

For structured output the reference is candid about a limitation:
OpenAI's `response_format={"type": "json_object"}` is only *partially*
supported on vLLM. The system therefore belts-and-braces it —
prompt-level JSON coercion (system role demanding JSON-only, the schema
inlined, "no other text" instructed) plus a post-hoc regex extractor
that pulls the first valid `{...}` block (or a fenced JSON code block)
and validates it with `json.loads` before trusting it. The reason generator emits a
strict schema (`{"reasons": [...], "summary": "..."}`); the L2a
rewriter, by contrast, emits *plain text*, because polishing a sentence
should not produce JSON at all.

## The Provider Abstraction — One Call Site, Many Backends

The piece that makes this maintainable is `LLMProviderFactory` in
`src/grounding/llm_provider.py`. The closed network deliberately
**excludes** every cloud backend (Bedrock, OpenAI, Gemini) and exposes
only self-hostable ones. A single config key picks the engine:

```python
from src.grounding import LLMProviderFactory
provider = LLMProviderFactory.create({
    "llm_provider": {
        "backend": "qwen",   # ollama | qwen | exaone | solar | local | dummy
        "qwen": {"model": "qwen3:14b",
                 "endpoint": "http://host.docker.internal:11434/v1"},
    }
})
response = provider.generate(prompt, response_format={"type": "json_object"})
```

Every backend implements the same `generate()` signature, and
`response_format` is forwarded to the OpenAI-compatible call only where
the backend supports it. The factory hides where the tokens actually
come from:

| Backend | Engine / route | Default endpoint | Note |
| --- | --- | --- | --- |
| `ollama` | Ollama OpenAI-compatible `/v1` | `:11434/v1` | qwen3 / llama3 / exaone imported into Ollama |
| `qwen` | alias of `ollama`, model→`qwen3:14b` | `:11434/v1` | convenience default |
| `exaone` | self-hosted (vLLM serve) | `:8000/v1` | `lgai/exaone-3.5-32b-instruct` |
| `solar` | local self-host or Upstage API | `:8000/v1` | `mode: local` keeps it on-prem |
| `local` | any OpenAI-compatible server | configurable | generic escape hatch |
| `dummy` | hardcoded output | — | offline tests / final fallback |

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <!-- call site -->
  <rect x="40" y="100" width="120" height="48" rx="6" fill="#1e3a5f"/>
  <text x="100" y="122" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">L2a / reason</text>
  <text x="100" y="138" text-anchor="middle" font-size="9" fill="#cbd5e1">.generate(prompt)</text>
  <!-- factory -->
  <rect x="210" y="96" width="110" height="56" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="265" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Factory</text>
  <text x="265" y="134" text-anchor="middle" font-size="9" fill="#64748b">backend: …</text>
  <line x1="160" y1="124" x2="208" y2="124" stroke="#94a3b8" stroke-width="1.4"/><polygon points="208,124 200,120 200,128" fill="#94a3b8"/>
  <!-- backends -->
  <g font-size="10" font-weight="700">
    <rect x="400" y="40" width="160" height="30" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="480" y="60" text-anchor="middle" fill="#0d9488">ollama / qwen · :11434</text>
    <rect x="400" y="78" width="160" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/><text x="480" y="98" text-anchor="middle" fill="#d97706">exaone · vLLM :8000</text>
    <rect x="400" y="116" width="160" height="30" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="480" y="136" text-anchor="middle" fill="#4f46e5">solar · local :8000</text>
    <rect x="400" y="154" width="160" height="30" rx="5" fill="#f1f5f9" stroke="#64748b"/><text x="480" y="174" text-anchor="middle" fill="#64748b">local · configurable</text>
    <rect x="400" y="192" width="160" height="30" rx="5" fill="#fee2e2" stroke="#e11d48"/><text x="480" y="212" text-anchor="middle" fill="#e11d48">dummy · fallback</text>
  </g>
  <g stroke="#cbd5e1" stroke-width="1.2">
    <line x1="320" y1="124" x2="400" y2="55"/><line x1="320" y1="124" x2="400" y2="93"/>
    <line x1="320" y1="124" x2="400" y2="131"/><line x1="320" y1="124" x2="400" y2="169"/><line x1="320" y1="124" x2="400" y2="207"/>
  </g>
  <text x="265" y="186" text-anchor="middle" font-size="9" fill="#94a3b8">cloud backends excluded by design</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">One call site, one generate() signature; the factory resolves a config key into a self-hosted backend. No cloud route is reachable from the closed network.</figcaption>
</figure>

## How Reason-Rewrite (L2a) Actually Calls It

The L2a rewriter is the heaviest consumer. It does *not* run for every
customer: a richness gate sends only `rich` and `moderate` contexts to
the LLM and skips `sparse` ones, where polishing barely helps. For each
eligible draft it builds a prompt — the L1 draft truncated to 300
chars, plus ~200 chars of context — with hard instructions: keep the
facts, **add no numbers** (a hallucinated "5% return" is a compliance
violation), two-to-three sentences, plain text only.

The live operational route is an **Ollama dual-route**, not the single
vLLM server in the canonical diagram:

1. **Primary** — `exaone3.5:2.4b` over Ollama handles the bulk at low
   cost.
2. **Three-layer Safety Gate** — Gate 1 (parse: reject empty or JSON
   debris), Gate 2 (compliance: a regex blacklist for "guaranteed
   return", "principal protected", "N% return"…), Gate 3 (quality:
   30–200 chars, ≥80% Korean).
3. **Escalation** — on a Gate 1 or Gate 3 failure, the draft is retried
   on `qwen3:14b`; on a *Gate 2* (compliance) failure there is **no**
   escalation — the system falls straight back to the L1 original.

That last rule matters: a compliance breach is never "tried harder," it
is dropped, and if every path fails the rule-based L1 draft ships
unchanged so a customer never sees an empty reason. The
`max_concurrent=10` client cap dovetails with the server's ~15-request
KV ceiling, leaving margin for sequence-length swings. (The `qwen3:14b`
escalation also carries a known operational wrinkle — a thinking-mode
token trap that forces a generous `max_tokens` even though real output
is ~150–200 tokens — but that is a tuning note, not an architecture
one.)

## Where We Stop

This is the last brick. Trace the arc backwards: seven heterogeneous
Experts each reading a different kind of signal (PersLay's *shape*,
CausalOT's transport, the GCN's hyperbolic graph, the Temporal
ensemble, GMM/HMM regimes, the economics and multidisciplinary
features); an offline feature pipeline that materialises all of it; a
teacher–student distillation that compresses the heavy model into
servable students; a scoring path that turns scores into ranked
recommendations; a reason generator that explains them; and now a
serving stack that runs the polishing model on a single 12 GB GPU,
half a million times a week, without one packet leaving the closed
network. The *how* is, finally, documented end to end.

What is conspicuously not documented is the *whether*. Every result we
have leaned on so far — the PersLay silhouette probe, the per-technique
validations — was a sub-component check, a green light to build, never a
verdict on the whole machine. The open frontier is no longer
architecture; it is evidence on real data: does the assembled system,
top to bottom, actually move recommendation quality on real customers,
and by how much? That is where the thread turns next — from how it is
built to whether it works.
