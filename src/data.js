// Real data mirrored from bluethestyle.github.io
export const SITE = {
  owner: "Seonkyu Jeong",
  role: "Independent researcher — Seoul",
  bio: "Notes, working papers, and long-form thinking on financial AI, model risk management, and agentic systems. GARP Financial Risk Manager (FRM).",
  bio2: "This site collects what does not fit into a journal paper or a GitHub README — decisions and their reasons, failed experiments, what collaborating with an AI system for four months actually looked like.",
  contact: "jsk320098@gmail.com",
  orcid: "0009-0005-3291-9112",
  github: "bluethestyle",
  counts: { posts: 43, cats: 4, tags: 123, years: 1 }
};

export const CURRENT_WORK = {
  title: "Heterogeneous Expert PLE for Financial Product Recommendation",
  titleKo: "금융 상품 추천을 위한 이종 전문가 PLE",
  desc: "A 13-task multi-task learning system with seven structurally distinct expert networks, distilled to LightGBM for AWS Lambda serving, with regulatory-grade audit infrastructure.",
  descKo: "서로 다른 구조의 7개 전문가 네트워크가 13개 태스크를 함께 학습하고, LightGBM 으로 증류해 AWS Lambda 로 서빙하는 멀티태스크 시스템. 규제 수준의 감사 인프라까지 포함한다.",
  meta: "A three-person team · early 2026 · with Claude Code (Anthropic) as the primary development partner",
  metaKo: "3명 팀 · 2026년 초 · 주 개발 파트너는 Claude Code (Anthropic)",
  links: [
    { label: "Paper 1 · Architecture & Ablation", labelKo: "Paper 1 · 아키텍처 & Ablation", href: "https://doi.org/10.5281/zenodo.19621884", tag: "DOI" },
    { label: "Paper 2 · Agentic Reason Generation & Compliance", labelKo: "Paper 2 · 에이전틱 사유 생성 & 컴플라이언스", href: "https://doi.org/10.5281/zenodo.19622052", tag: "DOI" },
    { label: "Source code", labelKo: "소스 코드", href: "https://github.com/bluethestyle/aws_ple_for_financial", tag: "MIT" },
  ]
};

export const COAUTHORS = [
  { name: "Seonkyu Jeong", nameKo: "정선규", role: "architecture, regulatory framing", roleKo: "아키텍처 · 규제 관점 정립", lead: true },
  { name: "Euncheol Sim",  nameKo: "심은철",  role: "engineering, experimentation", roleKo: "엔지니어링 · 실험" },
  { name: "Youngchan Kim", nameKo: "김영찬", role: "engineering, evaluation", roleKo: "엔지니어링 · 평가" },
];

export const SERIES = [
  {
    slug: "three-months",
    title: "Building a Financial AI in Four Months",
    titleKo: "4개월간의 금융 AI 개발기",
    tag: "FinAI Build",
    tagKo: "4개월 개발기",
    desc: "Building a financial recommendation system with Claude Code, on consumer hardware, as a three-person team.",
    descKo: "3명 팀이 소비자용 GPU 한 대와 Claude Code 만으로 금융 추천 시스템을 만들어낸 기록.",
    ep: 8, total: 8,
    ko: "/series/three-months-ko/",
    en: "/series/three-months/",
  },
  {
    slug: "mrm-thread",
    title: "From statistical-model MRM to architecture-first responsible AI",
    titleKo: "수리통계 모델 시대의 MRM 에서 architecture-first 의 RAI 로",
    tag: "MRM Thread",
    tagKo: "MRM 스레드",
    desc: "Six episodes tracing how the lifecycle-integration principle of model risk management evolves into responsible AI for the deep learning era — written from a GARP FRM practitioner perspective.",
    descKo: "MRM 의 lifecycle 통합 원칙이 ML/DL 시대에 architecture-first responsible AI 로 진화하는 여정을 6 에피소드로 추적한다 — GARP FRM 실무자 관점에서.",
    ep: 6, total: 6,
    ko: "/series/mrm-thread-ko/",
    en: "/series/mrm-thread/",
  },
  {
    slug: "study-thread",
    title: "Study Thread — Papers & Math Foundations",
    titleKo: "스터디 스레드 — 논문 & 수학 기초",
    tag: "Study Thread",
    tagKo: "스터디 스레드",
    desc: "Papers, math foundations, and reference reading behind the PLE architecture — studied and summarized in parallel English/Korean.",
    descKo: "PLE 아키텍처의 뿌리가 되는 논문·수학 기초·참고 문헌을 영/한 두 언어로 정리해 둔 학습 기록.",
    ep: 27,
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
  "/2026/05/12/ep8-honest-negatives-ko/":
    "4개월의 기록 — adaTT 가 13-task 에서 null 로 수렴한 과정, GradSurgery 가 VRAM 오버헤드로 미채택된 이유, 집필 당시 진행 중이던 Paper 3, 2026-04-30 이후 실데이터 메트릭 대기. 작동하지 않은 것이 작동한 것만큼 중요한 이유.",
  "/2026/05/12/ep8-honest-negatives-en/":
    "Record from four months — how adaTT converged to a null effect at 13-task scale, why GradSurgery was rejected on VRAM overhead, Paper 3 still in progress at the time of writing, and real-data metrics pending after 2026-04-30. Why what did not work matters as much as what did.",
  "/2026/05/08/ep7-distillation-serving-ko/":
    "teacher 는 PLE, student 는 task 별 LightGBM, 서빙은 AWS Lambda. 왜 이 조합인가, teacher-student fidelity 가 실패하면 어떻게 되는가, 그리고 Bedrock 위 5-에이전트 파이프라인의 역할 분담.",
  "/2026/05/08/ep7-distillation-serving-en/":
    "Teacher is PLE, student is per-task LightGBM, serving is AWS Lambda. Why this combination, what happens when teacher-student fidelity fails, and the role division across the 5-agent Bedrock pipeline.",
  "/2026/05/05/ep6-uncertainty-weighting-bug-ko/":
    "몇 주 동안 sigmoid gate 가 softmax 를 이기는 것처럼 보였다. uncertainty weighting 구현 버그가 수정되자 결과가 뒤집혔다. 훈련 환경의 버그가 어떻게 아키텍처 결론을 오염시키는가의 사례 연구.",
  "/2026/05/05/ep6-uncertainty-weighting-bug-en/":
    "For weeks sigmoid gating seemed to beat softmax. Fixing an uncertainty-weighting implementation bug flipped the result. A case study in how a training-environment bug contaminates architectural conclusions.",
  "/2026/05/01/ep5-data-integrity-hunt-ko/":
    "아키텍처 논쟁 전에 풀어야 했던 것 — label leakage 3건 연쇄 탐지, 18→13 태스크 축소의 결정론적 리키지 배경, 합성데이터 v2→v3→v4 iteration 에서 드러난 자기복제 피처.",
  "/2026/05/01/ep5-data-integrity-hunt-en/":
    "Before any architecture debate — three chained label-leakage detections, the deterministic-leakage rationale behind the 18→13 task reduction, and the self-replicating features that surfaced across synthetic-data iterations v2→v3→v4.",
  "/2026/04/28/ep4-seven-experts-ko/":
    "왜 7명인가, 왜 이 7명인가. Gemini 와의 학제간 스캔에서 11개 분야를 훑고, Opus 와의 기술 검증에서 선 안으로 들어온 DeepFM·Temporal·HGCN·PersLay·Causal·LightGCN·OT 의 도출 과정.",
  "/2026/04/28/ep4-seven-experts-en/":
    "Why seven experts, why these seven. The cross-disciplinary scan with Gemini surfaced eleven fields; the feasibility review with Opus narrowed to DeepFM, Temporal Ensemble, HGCN, PersLay, Causal, LightGCN, and Optimal Transport.",
  "/2026/04/24/mrm-ep3-chain-of-custody-ko/":
    "14개월 전 추천 분쟁이 들어왔을 때 답의 형태를 결정하는 7개 감사 테이블과 HMAC 해시 체인. EU AI Act 13·14조와 KFCPA §17 을 체크리스트가 아니라 역구성 가능한 코드 경로로 만드는 방식.",
  "/2026/04/24/mrm-ep3-chain-of-custody-en/":
    "Fifteen months after a recommendation, a customer disputes it. Seven audit tables and one HMAC hash chain determine the shape of the answer — making EU AI Act 13-14 and KFCPA §17 reconstructable code paths, not checklists.",
  "/2026/04/24/ep3-guardrails-ko/":
    "3인 × AI 팀이 병렬로 굴러가면서도 통합 지점에서 깨지지 않게 한 실제 장치들 — CLAUDE.md 헌법 4조, 8개 메모리 뱅크 파일, 매 병렬 작업 후 인터페이스 키 검증의 구체 모습.",
  "/2026/04/24/ep3-guardrails-en/":
    "The actual mechanisms that kept three parallel AI-agent teams from breaking at integration — the CLAUDE.md constitution's four clauses, the eight-file memory bank, and the interface-key diff check run after every parallel session.",
  "/2026/04/21/mrm-ep2-champion-challenger-ko/":
    "MRM 스레드 2편. Champion-Challenger 를 동기 코드 게이트로 구현한 이야기 — 4가지 판정 형태(force_promote / bootstrap / reject(fidelity) / promote·reject(competition)), 안전 플로어가 경쟁 이전에 오는 이유, 그리고 모든 판정이 HMAC 서명 감사 엔트리가 되는 SR 11-7 재구성 가능성.",
  "/2026/04/21/mrm-ep2-champion-challenger-en/":
    "MRM Thread ep 2. Champion-Challenger implemented as a synchronous code gate — the four verdict shapes (force_promote / bootstrap / reject(fidelity) / promote·reject(competition)), why the safety floor comes before competition, and how every verdict becomes an HMAC-signed audit entry for SR 11-7 reconstruction.",
  "/2026/04/21/ep2-ai-collaboration-ko/":
    "4개월 개발기 2편. Phase A-E 로 나뉜 단계-도구 페어링 — Gemini (아이디에이션) → Claude Opus (기술 검증, 19편 기술 참조 공동 작성) → Cursor (환경·CLAUDE.md 가드레일) → Claude Code Opus/Sonnet (3인×AI 팀 병렬 구현) → Claude Code Extension (ablation 모니터링·논문). 왜 Claude Code 가 구현 단계에서 대체 불가능했는지 구체적 예 3가지 (leakage 3연쇄, FP16 NaN 4동시 진단, sigmoid 관찰→가설→문헌→구현 플로우).",
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
    "시리즈 '4개월간의 금융 AI 개발기' 1편. Claude Code 를 주 개발 파트너로, 소비자용 GPU 한 대로, 3명 팀이 금융 추천 시스템을 만든 이야기.",
  "/2026/04/18/ep1-premise-en/":
    "Part 1 of 'Building a Financial AI in Four Months' — a series on building a financial recommendation system with Claude Code, on consumer hardware, as a three-person team.",
};

/**
 * Planned-but-unpublished posts. These have no markdown file yet —
 * shown greyed in Archives.
 */
export const DRAFTS = [];

export const TAGS = [
  { slug: "adatt", name: "adatt", count: 5, hot: true },
  { slug: "ai-basic-act", name: "ai-basic-act", count: 1 },
  { slug: "architecture", name: "architecture", count: 4 },
  { slug: "attention", name: "attention", count: 1 },
  { slug: "attribution", name: "attribution", count: 1 },
  { slug: "audit", name: "audit", count: 4 },
  { slug: "bedrock", name: "bedrock", count: 1 },
  { slug: "behavioral-economics", name: "behavioral-economics", count: 1 },
  { slug: "calibration", name: "calibration", count: 1 },
  { slug: "causal", name: "causal", count: 2 },
  { slug: "cgc", name: "cgc", count: 2 },
  { slug: "claude-code", name: "claude-code", count: 3 },
  { slug: "clustering", name: "clustering", count: 1 },
  { slug: "commentary", name: "commentary", count: 2 },
  { slug: "complexity", name: "complexity", count: 1 },
  { slug: "consensus", name: "consensus", count: 1 },
  { slug: "cosine-similarity", name: "cosine-similarity", count: 1 },
  { slug: "counterfactual", name: "counterfactual", count: 1 },
  { slug: "data-integrity", name: "data-integrity", count: 1 },
  { slug: "debugging", name: "debugging", count: 1 },
  { slug: "deepfm", name: "deepfm", count: 1 },
  { slug: "deepsets", name: "deepsets", count: 1 },
  { slug: "disclosure", name: "disclosure", count: 1 },
  { slug: "distillation", name: "distillation", count: 2 },
  { slug: "duckdb", name: "duckdb", count: 1 },
  { slug: "economics", name: "economics", count: 1 },
  { slug: "elasticity", name: "elasticity", count: 1 },
  { slug: "em", name: "em", count: 1 },
  { slug: "ema", name: "ema", count: 1 },
  { slug: "entropy", name: "entropy", count: 1 },
  { slug: "eu-ai-act", name: "eu-ai-act", count: 1 },
  { slug: "evidential", name: "evidential", count: 1 },
  { slug: "expert", name: "expert", count: 7, hot: true },
  { slug: "expert-pool", name: "expert-pool", count: 2 },
  { slug: "explainability", name: "explainability", count: 2 },
  { slug: "factorization-machine", name: "factorization-machine", count: 1 },
  { slug: "fd-tvs", name: "fd-tvs", count: 1 },
  { slug: "feature-engineering", name: "feature-engineering", count: 1 },
  { slug: "feature-interaction", name: "feature-interaction", count: 1 },
  { slug: "feature-mapping", name: "feature-mapping", count: 1 },
  { slug: "features", name: "features", count: 5, hot: true },
  { slug: "finai-build", name: "finai-build", count: 8, hot: true },
  { slug: "financial-ai", name: "financial-ai", count: 15, hot: true },
  { slug: "gmm", name: "gmm", count: 1 },
  { slug: "gradient", name: "gradient", count: 1 },
  { slug: "gradsurgery", name: "gradsurgery", count: 1 },
  { slug: "graph", name: "graph", count: 1 },
  { slug: "grounding", name: "grounding", count: 2 },
  { slug: "group-encoder", name: "group-encoder", count: 1 },
  { slug: "group-prior", name: "group-prior", count: 1 },
  { slug: "guardrail", name: "guardrail", count: 1 },
  { slug: "hallucination", name: "hallucination", count: 1 },
  { slug: "hgcn", name: "hgcn", count: 1 },
  { slug: "hmm", name: "hmm", count: 3 },
  { slug: "hyperbolic", name: "hyperbolic", count: 1 },
  { slug: "hypernetwork", name: "hypernetwork", count: 1 },
  { slug: "inference", name: "inference", count: 1 },
  { slug: "lambda", name: "lambda", count: 1 },
  { slug: "lancedb", name: "lancedb", count: 1 },
  { slug: "leakage", name: "leakage", count: 1 },
  { slug: "lightgbm", name: "lightgbm", count: 1 },
  { slug: "llm", name: "llm", count: 1 },
  { slug: "llm-serving", name: "llm-serving", count: 1 },
  { slug: "logit-transfer", name: "logit-transfer", count: 1 },
  { slug: "loss-weighting", name: "loss-weighting", count: 1 },
  { slug: "mamba", name: "mamba", count: 1 },
  { slug: "markov", name: "markov", count: 1 },
  { slug: "methodology", name: "methodology", count: 1 },
  { slug: "mixture-model", name: "mixture-model", count: 1 },
  { slug: "mmoe", name: "mmoe", count: 1 },
  { slug: "modularity", name: "modularity", count: 1 },
  { slug: "mrm", name: "mrm", count: 8, hot: true },
  { slug: "mtl", name: "mtl", count: 3 },
  { slug: "multidisciplinary", name: "multidisciplinary", count: 1 },
  { slug: "negative-results", name: "negative-results", count: 1 },
  { slug: "negative-transfer", name: "negative-transfer", count: 1 },
  { slug: "nlg", name: "nlg", count: 1 },
  { slug: "notears", name: "notears", count: 1 },
  { slug: "offline", name: "offline", count: 4 },
  { slug: "on-prem", name: "on-prem", count: 1 },
  { slug: "optimal-transport", name: "optimal-transport", count: 1 },
  { slug: "optimizer", name: "optimizer", count: 1 },
  { slug: "pagedattention", name: "pagedattention", count: 1 },
  { slug: "paper-3", name: "paper-3", count: 1 },
  { slug: "persistent-homology", name: "persistent-homology", count: 2 },
  { slug: "perslay", name: "perslay", count: 2 },
  { slug: "pia", name: "pia", count: 1 },
  { slug: "pipa", name: "pipa", count: 1 },
  { slug: "ple", name: "ple", count: 9, hot: true },
  { slug: "poincare", name: "poincare", count: 1 },
  { slug: "qwen", name: "qwen", count: 1 },
  { slug: "rag", name: "rag", count: 1 },
  { slug: "reason-generation", name: "reason-generation", count: 1 },
  { slug: "regime", name: "regime", count: 1 },
  { slug: "regularization", name: "regularization", count: 1 },
  { slug: "regulation", name: "regulation", count: 4 },
  { slug: "retrieval", name: "retrieval", count: 1 },
  { slug: "sae", name: "sae", count: 1 },
  { slug: "schedule", name: "schedule", count: 1 },
  { slug: "scoring", name: "scoring", count: 1 },
  { slug: "seasonality", name: "seasonality", count: 1 },
  { slug: "serving", name: "serving", count: 3 },
  { slug: "set-function", name: "set-function", count: 1 },
  { slug: "shared-bottom", name: "shared-bottom", count: 1 },
  { slug: "shared-experts", name: "shared-experts", count: 1 },
  { slug: "specs", name: "specs", count: 2 },
  { slug: "spectral", name: "spectral", count: 1 },
  { slug: "sr-11-7", name: "sr-11-7", count: 3 },
  { slug: "ssm", name: "ssm", count: 1 },
  { slug: "study-thread", name: "study-thread", count: 27, hot: true },
  { slug: "tang2020", name: "tang2020", count: 1 },
  { slug: "task-tower", name: "task-tower", count: 1 },
  { slug: "tda", name: "tda", count: 3 },
  { slug: "teacher-student", name: "teacher-student", count: 1 },
  { slug: "temporal", name: "temporal", count: 1 },
  { slug: "timeseries", name: "timeseries", count: 1 },
  { slug: "topology", name: "topology", count: 1 },
  { slug: "training-loop", name: "training-loop", count: 1 },
  { slug: "transfer-loss", name: "transfer-loss", count: 1 },
  { slug: "transformer", name: "transformer", count: 1 },
  { slug: "uncertainty", name: "uncertainty", count: 1 },
  { slug: "vllm", name: "vllm", count: 1 },
  { slug: "xai", name: "xai", count: 1 },
];

export const CATEGORIES = [
  { slug: "finai-build",  name: "FinAI Build",   ko: "4개월 개발기",  count: 8,  desc: "Building a financial AI in four months — engineering notes.",                                                descKo: "4개월간의 금융 AI 개발 기록 — 엔지니어링 노트.",                                                            color: "1" },
  { slug: "mrm-thread",   name: "MRM Thread",    ko: "MRM 스레드",    count: 6,  desc: "From statistical-model MRM to architecture-first responsible AI.",                                              descKo: "수리통계 모델 시대의 MRM 에서 architecture-first 의 RAI 로.",                                                color: "2" },
  { slug: "study-thread", name: "Study Thread",  ko: "스터디 스레드", count: 27, desc: "Papers, math foundations, and reference reading behind the PLE architecture — studied and summarized.",      descKo: "PLE 아키텍처의 배경이 되는 논문·수학 기초·참고 문헌을 학습하며 정리한 노트.",                               color: "3" },
  { slug: "commentary",   name: "Commentary",    ko: "논평",          count: 2,  desc: "Readings of regulatory drafts, architecture patterns, and paper findings adjacent to the main threads.",       descKo: "규제 초안·아키텍처 패턴·논문 발견을 메인 스레드 옆에서 짚는 글.",                                              color: "4" },
];

// RECENT is now derived from loadRecent() in src/lib/posts.js — pages
// that need it call the loader in their frontmatter script and pass
// the result as a prop. No hardcoded list here anymore.
