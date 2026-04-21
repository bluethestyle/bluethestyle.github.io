// Real data mirrored from bluethestyle.github.io
export const SITE = {
  owner: "Seonkyu Jeong",
  role: "Independent researcher — Seoul",
  bio: "Notes, working papers, and long-form thinking on financial AI, model risk management, and agentic systems. GARP Financial Risk Manager (FRM).",
  bio2: "This site collects what does not fit into a journal paper or a GitHub README — decisions and their reasons, failed experiments, what collaborating with an AI system for three months actually looked like.",
  contact: "jsk320098 [at] gmail [dot] com",
  orcid: "0009-0005-3291-9112",
  github: "bluethestyle",
  counts: { posts: 44, cats: 4, tags: 39, years: 1 }
};

export const CURRENT_WORK = {
  title: "Heterogeneous Expert PLE for Financial Product Recommendation",
  desc: "A 13-task multi-task learning system with seven structurally distinct expert networks, distilled to LightGBM for AWS Lambda serving, with regulatory-grade audit infrastructure.",
  meta: "Built by three people on personal time · early 2026 · with Claude Code (Anthropic) as the primary development partner",
  links: [
    { label: "Paper 1 · Architecture & Ablation", href: "https://doi.org/10.5281/zenodo.19621884", tag: "DOI" },
    { label: "Paper 2 · Agentic Reason Generation & Compliance", href: "https://doi.org/10.5281/zenodo.19622052", tag: "DOI" },
    { label: "Source code", href: "https://github.com/bluethestyle/aws_ple_for_financial", tag: "MIT" },
  ]
};

export const COAUTHORS = [
  { name: "Seonkyu Jeong", role: "architecture, regulatory framing", lead: true },
  { name: "Euncheol Sim",  role: "engineering, experimentation" },
  { name: "Youngchan Kim", role: "engineering, evaluation" },
];

export const SERIES = [
  {
    slug: "three-months",
    title: "Building a Financial AI in Three Months",
    tag: "FinAI Build",
    desc: "Building a financial recommendation system with Claude Code, on consumer hardware, as three people on personal time.",
    ep: 2, total: 8,
    ko: "/series/three-months-ko/",
    en: "/series/three-months/",
  },
  {
    slug: "mrm-thread",
    title: "The MRM Thread",
    tag: "MRM Thread",
    desc: "Regulatory compliance and model risk management for AI recommendation systems, from a GARP FRM practitioner perspective.",
    ep: 2, total: 6,
    ko: "/series/mrm-thread-ko/",
    en: "/series/mrm-thread/",
  },
  {
    slug: "study-thread",
    title: "Study Thread — Papers & Math Foundations",
    tag: "Study Thread",
    desc: "Papers, math foundations, and reference reading behind the PLE architecture — studied and summarized in parallel English/Korean.",
    ep: 10,
    ko: "/series/study-thread-ko/",
    en: "/series/study-thread/",
  },
];

/**
 * Excerpts keyed by post URL. These are separate from the markdown
 * frontmatter because (a) historically they were authored here and
 * (b) migrating them into every markdown file is a large, mechanical
 * diff that's better done lazily. Posts that set `excerpt:` in their
 * frontmatter override the entry below.
 */
export const EXCERPTS = {
  "/2026/04/21/mrm-ep2-champion-challenger-ko/":
    "MRM 스레드 2편. Champion-Challenger 를 동기 코드 게이트로 구현한 이야기 — 4가지 판정 형태(force_promote / bootstrap / reject(fidelity) / promote·reject(competition)), 안전 플로어가 경쟁 이전에 오는 이유, 그리고 모든 판정이 HMAC 서명 감사 엔트리가 되는 SR 11-7 재구성 가능성.",
  "/2026/04/21/mrm-ep2-champion-challenger-en/":
    "MRM Thread ep 2. Champion-Challenger implemented as a synchronous code gate — the four verdict shapes (force_promote / bootstrap / reject(fidelity) / promote·reject(competition)), why the safety floor comes before competition, and how every verdict becomes an HMAC-signed audit entry for SR 11-7 reconstruction.",
  "/2026/04/21/ep2-ai-collaboration-ko/":
    "3개월 개발기 2편. Phase A-E 로 나뉜 단계-도구 페어링 — Gemini (아이디에이션) → Claude Opus (기술 검증, 19편 기술 참조 공동 작성) → Cursor (환경·CLAUDE.md 가드레일) → Claude Code Opus/Sonnet (3인×AI 팀 병렬 구현) → Claude Code Extension (ablation 모니터링·논문). 왜 Claude Code 가 구현 단계에서 대체 불가능했는지 구체적 예 3가지 (leakage 3연쇄, FP16 NaN 4동시 진단, sigmoid 관찰→가설→문헌→구현 플로우).",
  "/2026/04/21/ep2-ai-collaboration-en/":
    "FinAI Build ep 2. The five phase-tool pairings (A-E) — Gemini for ideation, Claude Opus for technical validation and 19 co-authored tech reference docs, Cursor for environment and CLAUDE.md guardrails, Claude Code Opus/Sonnet for three-person × AI-team parallel implementation, Claude Code extension for ablation monitoring and papers. Why Claude Code was non-substitutable in implementation: three concrete examples (chained leakage tracing, simultaneous FP16 NaN diagnosis, sigmoid-gate observation→hypothesis→literature→implementation flow).",
  "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-ko/":
    "adaTT 서브스레드 최종편. 2-Phase Training Loop, Loss Weighting(Uncertainty·GradNorm·DWA), Optimizer·Scheduler, CGC-adaTT 동기화, 메모리·성능 최적화, 디버깅 가이드, 설정 총람, 부록 + 전체 adaTT 기술 참조서 PDF 다운로드.",
  "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-en/":
    "Final post of the adaTT sub-thread. 2-Phase Training Loop, Loss Weighting (Uncertainty / GradNorm / DWA), Optimizer and Scheduler, CGC-adaTT sync, memory and performance, debugging guide, full settings reference, appendix — plus downloadable PDF of the full adaTT tech reference.",
  "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-ko/":
    "adaTT Transfer Loss 의 전체 공식과 전이 가중치, G-01 FIX Clamp, Target 마스킹, 태스크 그룹 기반 Prior 행렬 + Prior Blend Annealing, 3-Phase Schedule(Warmup → Dynamic → Frozen), 그리고 Negative Transfer 감지·차단 메커니즘.",
  "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-en/":
    "adaTT's Transfer Loss formula and transfer weights, the G-01 FIX clamp, target-task masking, the task-group-based Prior matrix with Prior Blend Annealing, the 3-Phase Schedule (Warmup → Dynamic → Frozen), and Negative Transfer detection/blocking.",
  "/2026/04/20/adatt-2-task-affinity-gradient-cosine-ko/":
    "태스크 간 친화도를 실제로 측정하는 TaskAffinityComputer 엔진, gradient cosine similarity 의 수학적 정의와 EMA 평활화, 유클리드 거리 대비 코사인을 쓰는 이유, 그리고 torch.compiler.disable 로 처리한 gradient 추출 경로까지.",
  "/2026/04/20/adatt-2-task-affinity-gradient-cosine-en/":
    "The TaskAffinityComputer engine that measures task-to-task affinity, gradient cosine similarity definition with EMA smoothing, why cosine over Euclidean distance, and the torch.compiler.disable-handled gradient extraction path.",
  "/2026/04/20/adatt-1-adaptive-tower-motivation-ko/":
    "adaTT 서브스레드 1편. '왜 적응형 타워인가' 의 근본 동기, Transformer Attention 과의 유사성, 조건부 계산·Hypernetwork 계보에서의 위치, 핵심 수식 직관, 그리고 '측정하고·선택하고·조절한다' 내러티브.",
  "/2026/04/20/adatt-1-adaptive-tower-motivation-en/":
    "Opening the adaTT sub-thread. The root motivation for adaptive towers, the Transformer Attention analogy, where adaTT sits in the conditional-computation / hypernetwork lineage, intuitions for the core equations, and the 'measure, select, modulate' narrative.",
  "/2026/04/19/ple-6-interpretability-uncertainty-specs-ko/":
    "PLE 서브스레드 최종편. SAE 기반 Expert 해석성, Evidential Deep Learning 불확실성 정량화, 18 태스크 전체 사양, 논문 vs 구현 비교, 디버깅 가이드, 부록 — 그리고 전체 PLE 기술 참조서 PDF 다운로드 포함.",
  "/2026/04/19/ple-6-interpretability-uncertainty-specs-en/":
    "Final post of the PLE sub-thread. SAE-based Expert interpretability, Evidential Deep Learning uncertainty quantification, 18-task spec, paper-vs-implementation innovations, debugging guide, appendix — plus a downloadable PDF of the full PLE tech reference.",
  "/2026/04/19/ple-5-basket-logit-tower-ko/":
    "GroupTaskExpertBasket 이 태스크별 전용 Expert 를 만드는 방식, 태스크 간 명시적 정보 전달 Logit Transfer 3가지 모드, 그리고 최종 예측을 수행하는 Task Tower 의 구조.",
  "/2026/04/19/ple-5-basket-logit-tower-en/":
    "How GroupTaskExpertBasket produces per-task specialized experts, three modes of Logit Transfer for explicit cross-task information flow, and the Task Tower architecture that produces final predictions.",
  "/2026/04/19/ple-4-cgc-hmm-routing-ko/":
    "두 단계 CGC — 1단계 CGCLayer(Shared + Task 함께 가중합, 논문 원형)와 2단계 CGCAttention(Shared concat 위 블록 스케일링)의 수식. Expert Collapse 를 막는 entropy 정규화, 이종 차원 비대칭을 보정하는 차원 정규화, 그리고 HMM Triple-Mode 라우팅의 전체 구조.",
  "/2026/04/19/ple-4-cgc-hmm-routing-en/":
    "Two-stage CGC — Stage 1 CGCLayer (Shared + Task weighted sum together, paper-exact) and Stage 2 CGCAttention (block-scaling on the Shared concat) — math for both. Entropy regularization to prevent Expert Collapse, dimension normalization to correct heterogeneous output asymmetry, and the full HMM Triple-Mode routing architecture.",
  "/2026/04/19/ple-3-heterogeneous-expert-pool-ko/":
    "PLE-2 에서 이종 Shared Expert pool 을 결정한 뒤, 왜 7명인가, 왜 이 7명인가. 각 자리가 어떤 수학적 빈틈을 메우는지, 어떤 대안이 후보였는지, 그리고 왜 이 후보가 뽑혔는지를 하나씩 짚는다.",
  "/2026/04/19/ple-3-heterogeneous-expert-pool-en/":
    "Once PLE-2 committed to a heterogeneous Shared Expert pool, the question became: why seven, and why these seven? For each seat, what gap it fills, what alternatives were on the table, and why this specific one won.",
  "/2026/04/19/ple-2-progressive-layered-extraction-ko/":
    "PLE(Tang et al., RecSys 2020)가 Shared-Bottom·MMoE의 실패를 어떻게 해결했는지 — 공유와 분리의 명시적 균형, Expert와 Gate의 직관적 역할, 수학적 고찰, 핵심 수식 해석, 그리고 '왜 PLE인가' 내러티브.",
  "/2026/04/19/ple-2-progressive-layered-extraction-en/":
    "How PLE (Tang et al., RecSys 2020) answered Shared-Bottom and MMoE's failures — the explicit balance of sharing and separation, the intuitive roles of Experts and Gates, mathematical discussion, interpretations of the core equations, and the 'why PLE' narrative end-to-end.",
  "/2026/04/19/ple-1-mtl-evolution-ko/":
    "Study Thread의 PLE 서브스레드 1편. 멀티태스크 학습의 동기(외국어 학습 비유), Negative Transfer의 수학적 정의(gradient 충돌·상호정보량·줄다리기 비유), 그리고 Shared-Bottom·MMoE가 각자 어디서 무너지는지(Caruana 1997 · Ma 2018 · Jacobs 1991 역사적 맥락)까지.",
  "/2026/04/19/ple-1-mtl-evolution-en/":
    "First post of the PLE sub-thread in Study Thread. MTL motivation (foreign-language transfer analogy), the math of Negative Transfer (gradient conflict, mutual information, tug-of-war analogy), and where Shared-Bottom and MMoE each break (with Caruana 1997 / Ma 2018 / Jacobs 1991 historical context).",
  "/2026/04/18/mrm-ep1-architecture-ko/":
    "시리즈 'MRM 스레드' 1편. AI 추천 시스템의 규제 준수와 모델 리스크 관리를 GARP FRM 실무자 관점에서 다룬다.",
  "/2026/04/18/mrm-ep1-architecture-en/":
    "Part 1 of 'The MRM Thread' — a short parallel series on regulatory compliance and model risk management for AI recommendation systems, written from a GARP FRM practitioner perspective.",
  "/2026/04/18/ep1-premise-ko/":
    "시리즈 '3개월간의 금융 AI 개발기' 1편. Claude Code 를 주 개발 파트너로, 소비자용 GPU 한 대로, 3명이 개인 시간에 금융 추천 시스템을 만든 이야기.",
  "/2026/04/18/ep1-premise-en/":
    "Part 1 of 'Building a Financial AI in Three Months' — a series on building a financial recommendation system with Claude Code, on consumer hardware, as three people on personal time.",
};

/**
 * Planned-but-unpublished posts. These have no markdown file yet —
 * shown greyed in Archives.
 */
export const DRAFTS = [
  { date: "2026-04-28", title: "[FinAI Build] Ep 3 — How We Adapted: Guardrails, Memory Bank, Contract Verification", cat: "FinAI Build",  lang: "EN", ex: "CLAUDE.md constitution, 8-file memory bank, .claude/RULES.md ↔ .cursorrules sync, interface contract checks.", tags: ["finai-build","claude-code"] },
  { date: "2026-04-28", title: "[MRM Thread] Ep 3 — Chain of Custody for an Agent Pipeline", cat: "MRM Thread", lang: "EN", ex: "Seven audit tables, HMAC hash chain, EU AI Act & KFCPA mappings.", tags: ["mrm","sr-11-7","audit"] },
];

export const TAGS = [
  { slug: "adatt",              name: "adatt",              count: 8, hot: true },
  { slug: "architecture",       name: "architecture",       count: 5, hot: true },
  { slug: "attention",          name: "attention",          count: 2 },
  { slug: "audit",              name: "audit",              count: 3 },
  { slug: "cgc",                name: "cgc",                count: 4 },
  { slug: "claude-code",        name: "claude-code",        count: 4, hot: true },
  { slug: "cosine-similarity",  name: "cosine-similarity",  count: 2 },
  { slug: "ema",                name: "ema",                count: 2 },
  { slug: "evidential",         name: "evidential",         count: 2 },
  { slug: "expert-pool",        name: "expert-pool",        count: 2 },
  { slug: "finai-build",        name: "finai-build",        count: 4 },
  { slug: "financial-ai",       name: "financial-ai",       count: 6, hot: true },
  { slug: "gradient",           name: "gradient",           count: 2 },
  { slug: "group-encoder",      name: "group-encoder",      count: 2 },
  { slug: "group-prior",        name: "group-prior",        count: 2 },
  { slug: "hmm",                name: "hmm",                count: 4 },
  { slug: "hypernetwork",       name: "hypernetwork",       count: 2 },
  { slug: "logit-transfer",     name: "logit-transfer",     count: 2 },
  { slug: "loss-weighting",     name: "loss-weighting",     count: 2 },
  { slug: "mmoe",               name: "mmoe",               count: 4 },
  { slug: "mrm",                name: "mrm",                count: 5, hot: true },
  { slug: "mtl",                name: "mtl",                count: 6, hot: true },
  { slug: "negative-transfer",  name: "negative-transfer",  count: 2 },
  { slug: "optimizer",          name: "optimizer",          count: 2 },
  { slug: "ple",                name: "ple",                count: 15, hot: true },
  { slug: "regularization",     name: "regularization",     count: 2 },
  { slug: "regulation",         name: "regulation",         count: 3 },
  { slug: "sae",                name: "sae",                count: 2 },
  { slug: "schedule",           name: "schedule",           count: 2 },
  { slug: "shared-bottom",      name: "shared-bottom",      count: 2 },
  { slug: "shared-experts",     name: "shared-experts",     count: 2 },
  { slug: "specs",              name: "specs",              count: 4 },
  { slug: "sr-11-7",            name: "sr-11-7",            count: 2 },
  { slug: "study-thread",       name: "study-thread",       count: 20, hot: true },
  { slug: "tang2020",           name: "tang2020",           count: 2 },
  { slug: "task-tower",         name: "task-tower",         count: 2 },
  { slug: "training-loop",      name: "training-loop",      count: 2 },
  { slug: "transfer-loss",      name: "transfer-loss",      count: 2 },
  { slug: "uncertainty",        name: "uncertainty",        count: 2 },
];

export const CATEGORIES = [
  { slug: "finai-build",  name: "FinAI Build",   ko: "3개월 개발기",  count: 2,  desc: "Building a financial AI in three months — engineering notes.",                                                color: "1" },
  { slug: "mrm-thread",   name: "MRM Thread",    ko: "MRM 스레드",    count: 2,  desc: "Model risk management for AI recommendation systems.",                                                         color: "2" },
  { slug: "study-thread", name: "Study Thread",  ko: "스터디 스레드", count: 20, desc: "Papers, math foundations, and reference reading behind the PLE architecture — studied and summarized.",      color: "3" },
  { slug: "commentary",   name: "Commentary",    ko: "논평",          count: 0,  desc: "Readings of regulatory drafts and papers. (empty)",                                                           color: "4" },
];

// RECENT is now derived from loadRecent() in src/lib/posts.js — pages
// that need it call the loader in their frontmatter script and pass
// the result as a prop. No hardcoded list here anymore.
