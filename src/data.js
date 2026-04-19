// Real data mirrored from bluethestyle.github.io
export const SITE = {
  owner: "Seonkyu Jeong",
  role: "Independent researcher — Seoul",
  bio: "Notes, working papers, and long-form thinking on financial AI, model risk management, and agentic systems. GARP Financial Risk Manager (FRM).",
  bio2: "This site collects what does not fit into a journal paper or a GitHub README — decisions and their reasons, failed experiments, what collaborating with an AI system for three months actually looked like.",
  contact: "jsk320098 [at] gmail [dot] com",
  orcid: "0009-0005-3291-9112",
  github: "bluethestyle",
  counts: { posts: 24, cats: 4, tags: 9, years: 1 }
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
    slug: "finai-build",
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
];

export const POSTS = [
  { date: "2026-04-18", title: "[MRM 스레드] 에피소드 1 — MRM 은 검증이 아니라 아키텍처에 속한다",        cat: "MRM 스레드",   lang: "KO", ex: "시리즈 'MRM 스레드' 1편. AI 추천 시스템의 규제 준수와 모델 리스크 관리를 GARP FRM 실무자 관점에서 다룬다.", tags: ["mrm","architecture","sr-11-7","regulation"] },
  { date: "2026-04-18", title: "[MRM Thread] Ep 1 — Why MRM Belongs in the Architecture",                 cat: "MRM Thread",   lang: "EN", ex: "Part 1 of 'The MRM Thread' — a short parallel series on regulatory compliance and model risk management for AI recommendation systems, written from a GARP FRM practitioner perspective.", tags: ["mrm","architecture","sr-11-7","regulation"] },
  { date: "2026-04-18", title: "[3개월 개발기] 에피소드 1 — 전제 조건",                                  cat: "3개월 개발기", lang: "KO", ex: "시리즈 '3개월간의 금융 AI 개발기' 1편. Claude Code 를 주 개발 파트너로, 소비자용 GPU 한 대로, 3명이 개인 시간에 금융 추천 시스템을 만든 이야기.", tags: ["finai-build","claude-code","ple","financial-ai"] },
  { date: "2026-04-18", title: "[FinAI Build] Ep 1 — The Premise",                                       cat: "FinAI Build",  lang: "EN", ex: "Part 1 of 'Building a Financial AI in Three Months' — a series on building a financial recommendation system with Claude Code, on consumer hardware, as three people on personal time.", tags: ["finai-build","claude-code","ple","financial-ai"] },
  // planned but unpublished — shown greyed in Archives
  { date: "2026-04-25", title: "[FinAI Build] Ep 2 — Hardware & Budget", cat: "FinAI Build",  lang: "EN", ex: "One consumer GPU, three laptops, three months.", tags: ["finai-build","claude-code"], draft: true },
  { date: "2026-04-25", title: "[MRM Thread] Ep 2 — Three Lines of Defense, Rewritten", cat: "MRM Thread", lang: "EN", ex: "Where 3LoD breaks once an agent is in the loop.", tags: ["mrm","sr-11-7"], draft: true },
];

export const TAGS = [
  { slug: "financial-ai", name: "financial-ai", count: 6, hot: true },
  { slug: "architecture", name: "architecture", count: 5, hot: true },
  { slug: "audit",        name: "audit",        count: 3 },
  { slug: "claude-code",  name: "claude-code",  count: 4, hot: true },
  { slug: "finai-build",  name: "finai-build",  count: 4 },
  { slug: "mrm",          name: "mrm",          count: 5, hot: true },
  { slug: "ple",          name: "ple",          count: 3 },
  { slug: "regulation",   name: "regulation",   count: 3 },
  { slug: "sr-11-7",      name: "sr-11-7",      count: 2 },
];

export const CATEGORIES = [
  { slug: "finai-build",  name: "FinAI Build",  ko: "3개월 개발기",  count: 2, desc: "Building a financial AI in three months — engineering notes.", color: "1" },
  { slug: "mrm-thread",   name: "MRM Thread",   ko: "MRM 스레드",    count: 2, desc: "Model risk management for AI recommendation systems.",        color: "2" },
  { slug: "field-notes",  name: "Field Notes",  ko: "현장 노트",     count: 0, desc: "Short observations from client work. (empty)",                 color: "3" },
  { slug: "commentary",   name: "Commentary",   ko: "논평",          count: 0, desc: "Readings of regulatory drafts and papers. (empty)",             color: "4" },
];

export const RECENT = [
  { title: "[FinAI Build] Ep 1 — The Premise", date: "2026-04-18" },
  { title: "[3개월 개발기] 에피소드 1 — 전제 조건", date: "2026-04-18" },
  { title: "[MRM Thread] Ep 1 — Why MRM Belongs in the Architecture", date: "2026-04-18" },
  { title: "[MRM 스레드] 에피소드 1 — MRM 은 검증이 아니라 아키텍처에 속한다", date: "2026-04-18" },
];
