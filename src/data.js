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
    ep: 1, total: 8,
    ko: "/series/three-months-ko/",
    en: "/series/three-months/",
  },
  {
    slug: "mrm-thread",
    title: "The MRM Thread",
    tag: "MRM Thread",
    desc: "Regulatory compliance and model risk management for AI recommendation systems, from a GARP FRM practitioner perspective.",
    ep: 1, total: 6,
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

export const POSTS = [
  // ADATT-4
  { date: "2026-04-20", title: "[Study Thread] ADATT-4 — 학습 루프·Loss Weighting·Optimizer·CGC 동기화", cat: "Study Thread", lang: "KO", url: "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-ko/", ex: "adaTT 서브스레드 최종편. 2-Phase Training Loop, Loss Weighting(Uncertainty·GradNorm·DWA), Optimizer·Scheduler, CGC-adaTT 동기화, 메모리·성능 최적화, 디버깅 가이드, 설정 총람, 부록 + 전체 adaTT 기술 참조서 PDF 다운로드.", tags: ["study-thread","adatt","training-loop","loss-weighting","optimizer","specs"] },
  { date: "2026-04-20", title: "[Study Thread] ADATT-4 — Training Loop, Loss Weighting, Optimizer, and CGC Synchronization", cat: "Study Thread", lang: "EN", url: "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-en/", ex: "Final post of the adaTT sub-thread. 2-Phase Training Loop, Loss Weighting (Uncertainty / GradNorm / DWA), Optimizer and Scheduler, CGC-adaTT sync, memory and performance, debugging guide, full settings reference, appendix — plus downloadable PDF of the full adaTT tech reference.", tags: ["study-thread","adatt","training-loop","loss-weighting","optimizer","specs"] },
  // ADATT-3
  { date: "2026-04-20", title: "[Study Thread] ADATT-3 — Transfer Loss · Group Prior · 3-Phase Schedule", cat: "Study Thread", lang: "KO", url: "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-ko/", ex: "adaTT Transfer Loss 의 전체 공식과 전이 가중치, G-01 FIX Clamp, Target 마스킹, 태스크 그룹 기반 Prior 행렬 + Prior Blend Annealing, 3-Phase Schedule(Warmup → Dynamic → Frozen), 그리고 Negative Transfer 감지·차단 메커니즘.", tags: ["study-thread","adatt","transfer-loss","group-prior","schedule","negative-transfer"] },
  { date: "2026-04-20", title: "[Study Thread] ADATT-3 — Transfer Loss, Group Prior, and the 3-Phase Schedule", cat: "Study Thread", lang: "EN", url: "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-en/", ex: "adaTT's Transfer Loss formula and transfer weights, the G-01 FIX clamp, target-task masking, the task-group-based Prior matrix with Prior Blend Annealing, the 3-Phase Schedule (Warmup → Dynamic → Frozen), and Negative Transfer detection/blocking.", tags: ["study-thread","adatt","transfer-loss","group-prior","schedule","negative-transfer"] },
  // ADATT-2
  { date: "2026-04-20", title: "[Study Thread] ADATT-2 — TaskAffinityComputer와 Gradient Cosine Similarity", cat: "Study Thread", lang: "KO", url: "/2026/04/20/adatt-2-task-affinity-gradient-cosine-ko/", ex: "태스크 간 친화도를 실제로 측정하는 TaskAffinityComputer 엔진, gradient cosine similarity 의 수학적 정의와 EMA 평활화, 유클리드 거리 대비 코사인을 쓰는 이유, 그리고 torch.compiler.disable 로 처리한 gradient 추출 경로까지.", tags: ["study-thread","adatt","gradient","cosine-similarity","ema"] },
  { date: "2026-04-20", title: "[Study Thread] ADATT-2 — TaskAffinityComputer and Gradient Cosine Similarity", cat: "Study Thread", lang: "EN", url: "/2026/04/20/adatt-2-task-affinity-gradient-cosine-en/", ex: "The TaskAffinityComputer engine that measures task-to-task affinity, gradient cosine similarity definition with EMA smoothing, why cosine over Euclidean distance, and the torch.compiler.disable-handled gradient extraction path.", tags: ["study-thread","adatt","gradient","cosine-similarity","ema"] },
  // ADATT-1
  { date: "2026-04-20", title: "[Study Thread] ADATT-1 — adaTT 동기: 적응형 타워와 Transformer Attention 의 유사성", cat: "Study Thread", lang: "KO", url: "/2026/04/20/adatt-1-adaptive-tower-motivation-ko/", ex: "adaTT 서브스레드 1편. '왜 적응형 타워인가' 의 근본 동기, Transformer Attention 과의 유사성, 조건부 계산·Hypernetwork 계보에서의 위치, 핵심 수식 직관, 그리고 '측정하고·선택하고·조절한다' 내러티브.", tags: ["study-thread","adatt","attention","hypernetwork","mtl"] },
  { date: "2026-04-20", title: "[Study Thread] ADATT-1 — Why adaTT: Adaptive Towers and the Transformer Attention Analogy", cat: "Study Thread", lang: "EN", url: "/2026/04/20/adatt-1-adaptive-tower-motivation-en/", ex: "Opening the adaTT sub-thread. The root motivation for adaptive towers, the Transformer Attention analogy, where adaTT sits in the conditional-computation / hypernetwork lineage, intuitions for the core equations, and the 'measure, select, modulate' narrative.", tags: ["study-thread","adatt","attention","hypernetwork","mtl"] },
  // PLE-6
  { date: "2026-04-19", title: "[Study Thread] PLE-6 — 해석성·불확실성·전체 사양", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-6-interpretability-uncertainty-specs-ko/", ex: "PLE 서브스레드 최종편. SAE 기반 Expert 해석성, Evidential Deep Learning 불확실성 정량화, 18 태스크 전체 사양, 논문 vs 구현 비교, 디버깅 가이드, 부록 — 그리고 전체 PLE 기술 참조서 PDF 다운로드 포함.", tags: ["study-thread","ple","sae","uncertainty","evidential","specs"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-6 — Interpretability, Uncertainty, and Full Specs", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-6-interpretability-uncertainty-specs-en/", ex: "Final post of the PLE sub-thread. SAE-based Expert interpretability, Evidential Deep Learning uncertainty quantification, 18-task spec, paper-vs-implementation innovations, debugging guide, appendix — plus a downloadable PDF of the full PLE tech reference.", tags: ["study-thread","ple","sae","uncertainty","evidential","specs"] },
  // PLE-5
  { date: "2026-04-19", title: "[Study Thread] PLE-5 — GroupTaskExpertBasket · Logit Transfer · Task Tower", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-5-basket-logit-tower-ko/", ex: "GroupTaskExpertBasket v3.2 가 태스크별 전용 Expert 를 만드는 방식, 태스크 간 명시적 정보 전달 Logit Transfer 3가지 모드, 그리고 최종 예측을 수행하는 Task Tower 의 구조.", tags: ["study-thread","ple","logit-transfer","task-tower","group-encoder"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-5 — GroupTaskExpertBasket, Logit Transfer, Task Tower", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-5-basket-logit-tower-en/", ex: "How GroupTaskExpertBasket v3.2 produces per-task specialized experts, three modes of Logit Transfer for explicit cross-task information flow, and the Task Tower architecture that produces final predictions.", tags: ["study-thread","ple","logit-transfer","task-tower","group-encoder"] },
  // PLE-4
  { date: "2026-04-19", title: "[Study Thread] PLE-4 — CGC 게이팅의 두 단계(CGCLayer + CGCAttention)와 HMM Triple-Mode 라우팅", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-4-cgc-hmm-routing-ko/", ex: "두 단계 CGC — 1단계 CGCLayer(Shared + Task 함께 가중합, 논문 원형)와 2단계 CGCAttention(Shared concat 위 블록 스케일링)의 수식. Expert Collapse 를 막는 entropy 정규화, 이종 차원 비대칭을 보정하는 차원 정규화, 그리고 HMM Triple-Mode 라우팅의 전체 구조.", tags: ["study-thread","ple","cgc","hmm","regularization"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-4 — The Two-Stage CGC Gate (CGCLayer + CGCAttention) and HMM Triple-Mode Routing", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-4-cgc-hmm-routing-en/", ex: "Two-stage CGC — Stage 1 CGCLayer (Shared + Task weighted sum together, paper-exact) and Stage 2 CGCAttention (block-scaling on the Shared concat) — math for both. Entropy regularization to prevent Expert Collapse, dimension normalization to correct heterogeneous output asymmetry, and the full HMM Triple-Mode routing architecture.", tags: ["study-thread","ple","cgc","hmm","regularization"] },
  // PLE-3
  { date: "2026-04-19", title: "[Study Thread] PLE-3 — 입력 구조와 이종 Shared Expert Pool (512D)", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-3-heterogeneous-expert-pool-ko/", ex: "PLEClusterInput 의 전체 필드 사양과 734D features 텐서 인덱스 매핑, HMM 모드 라우팅. 그리고 본 프로젝트의 7개 이종 Shared Expert 가 각자 어떤 수학적 관점으로 고객을 해석하는지.", tags: ["study-thread","ple","expert-pool","hmm","shared-experts"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-3 — Input Structure and Heterogeneous Shared Expert Pool (512D)", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-3-heterogeneous-expert-pool-en/", ex: "The full PLEClusterInput field spec, 734D feature-tensor index mapping, and HMM mode routing. Plus the seven heterogeneous Shared Experts this project runs, each interpreting the customer through a structurally different mathematical lens.", tags: ["study-thread","ple","expert-pool","hmm","shared-experts"] },
  // PLE-2
  { date: "2026-04-19", title: "[Study Thread] PLE-2 — Progressive Layered Extraction: 명시적 전문가 분리와 CGC 게이트", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-2-progressive-layered-extraction-ko/", ex: "PLE(Tang et al., RecSys 2020)가 Shared-Bottom·MMoE의 실패를 어떻게 해결했는지 — 공유와 분리의 명시적 균형, Expert와 Gate의 직관적 역할, 수학적 고찰, 핵심 수식 해석, 그리고 '왜 PLE인가' 내러티브.", tags: ["study-thread","ple","cgc","tang2020","mtl"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-2 — Progressive Layered Extraction: Explicit Expert Separation and CGC Gates", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-2-progressive-layered-extraction-en/", ex: "How PLE (Tang et al., RecSys 2020) answered Shared-Bottom and MMoE's failures — the explicit balance of sharing and separation, the intuitive roles of Experts and Gates, mathematical discussion, interpretations of the core equations, and the 'why PLE' narrative end-to-end.", tags: ["study-thread","ple","cgc","tang2020","mtl"] },
  // PLE-1
  { date: "2026-04-19", title: "[Study Thread] PLE-1 — MTL과 게이트드 전문가로의 진화 (Shared-Bottom → MMoE)", cat: "Study Thread", lang: "KO", url: "/2026/04/19/ple-1-mtl-evolution-ko/", ex: "Study Thread의 PLE 서브스레드 1편. 멀티태스크 학습의 동기(외국어 학습 비유), Negative Transfer의 수학적 정의(gradient 충돌·상호정보량·줄다리기 비유), 그리고 Shared-Bottom·MMoE가 각자 어디서 무너지는지(Caruana 1997 · Ma 2018 · Jacobs 1991 역사적 맥락)까지.", tags: ["study-thread","ple","mmoe","mtl"] },
  { date: "2026-04-19", title: "[Study Thread] PLE-1 — MTL and the Evolution Toward Gated Experts (Shared-Bottom → MMoE)", cat: "Study Thread", lang: "EN", url: "/2026/04/19/ple-1-mtl-evolution-en/", ex: "First post of the PLE sub-thread in Study Thread. MTL motivation (foreign-language transfer analogy), the math of Negative Transfer (gradient conflict, mutual information, tug-of-war analogy), and where Shared-Bottom and MMoE each break (with Caruana 1997 / Ma 2018 / Jacobs 1991 historical context).", tags: ["study-thread","ple","mmoe","mtl"] },
  // 2026-04-18 posts
  { date: "2026-04-18", title: "[MRM 스레드] 에피소드 1 — MRM 은 검증이 아니라 아키텍처에 속한다",        cat: "MRM 스레드",   lang: "KO", url: "/2026/04/18/mrm-ep1-architecture-ko/", ex: "시리즈 'MRM 스레드' 1편. AI 추천 시스템의 규제 준수와 모델 리스크 관리를 GARP FRM 실무자 관점에서 다룬다.", tags: ["mrm","architecture","sr-11-7","regulation"] },
  { date: "2026-04-18", title: "[MRM Thread] Ep 1 — Why MRM Belongs in the Architecture",                 cat: "MRM Thread",   lang: "EN", url: "/2026/04/18/mrm-ep1-architecture-en/", ex: "Part 1 of 'The MRM Thread' — a short parallel series on regulatory compliance and model risk management for AI recommendation systems, written from a GARP FRM practitioner perspective.", tags: ["mrm","architecture","sr-11-7","regulation"] },
  { date: "2026-04-18", title: "[3개월 개발기] 에피소드 1 — 전제 조건",                                  cat: "3개월 개발기", lang: "KO", url: "/2026/04/18/ep1-premise-ko/", ex: "시리즈 '3개월간의 금융 AI 개발기' 1편. Claude Code 를 주 개발 파트너로, 소비자용 GPU 한 대로, 3명이 개인 시간에 금융 추천 시스템을 만든 이야기.", tags: ["finai-build","claude-code","ple","financial-ai"] },
  { date: "2026-04-18", title: "[FinAI Build] Ep 1 — The Premise",                                       cat: "FinAI Build",  lang: "EN", url: "/2026/04/18/ep1-premise-en/", ex: "Part 1 of 'Building a Financial AI in Three Months' — a series on building a financial recommendation system with Claude Code, on consumer hardware, as three people on personal time.", tags: ["finai-build","claude-code","ple","financial-ai"] },
  // planned but unpublished — shown greyed in Archives
  { date: "2026-04-25", title: "[FinAI Build] Ep 2 — Hardware & Budget", cat: "FinAI Build",  lang: "EN", ex: "One consumer GPU, three laptops, three months.", tags: ["finai-build","claude-code"], draft: true },
  { date: "2026-04-25", title: "[MRM Thread] Ep 2 — Three Lines of Defense, Rewritten", cat: "MRM Thread", lang: "EN", ex: "Where 3LoD breaks once an agent is in the loop.", tags: ["mrm","sr-11-7"], draft: true },
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

export const RECENT = [
  { title: "[Study Thread] ADATT-4 — 학습 루프·Loss Weighting·Optimizer·CGC 동기화",                        date: "2026-04-20", url: "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-ko/" },
  { title: "[Study Thread] ADATT-4 — Training Loop, Loss Weighting, Optimizer, and CGC Synchronization",    date: "2026-04-20", url: "/2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-en/" },
  { title: "[Study Thread] ADATT-3 — Transfer Loss · Group Prior · 3-Phase Schedule",                       date: "2026-04-20", url: "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-ko/" },
  { title: "[Study Thread] ADATT-3 — Transfer Loss, Group Prior, and the 3-Phase Schedule",                 date: "2026-04-20", url: "/2026/04/20/adatt-3-transfer-loss-group-prior-schedule-en/" },
];
