---
title: "[MRM 스레드] 에피소드 6 — Modular Adaptability · 규제는 변해도 아키텍처는 변하지 않는다"
date: 2026-05-05 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, modularity, regulation, financial-ai, ai-basic-act, eu-ai-act]
lang: ko
excerpt: "한국 AI 기본법 시행령, EU AI Act amendment, 미래 미국 framework — 모두 도착할 것이다. 5개 규제 generator 는 같은 감사 로그 substrate 위 5개 모듈이지, 다시 작성되는 5개 문서가 아니다. 아키텍처 모듈성이 왜 앞 다섯 에피소드를 결과적으로 의미 있게 만드는 장기 베팅인가."
series: mrm-thread
part: 6
alt_lang: /2026/05/05/mrm-ep6-modular-adaptability-en/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 6편 — 마지막 에피소드. Ep 1–5 가 substrate 를 쌓았다. 아키텍처 안의 MRM (Ep 1), 챔피언-챌린저 게이트 (Ep 2), 감사 로그 층 (Ep 3), 설명 column 으로서의 inherent XAI (Ep 4), retrieval 층으로서의 RAG + LanceDB (Ep 5). 이 substrate 위에 앉는 5개 규제 generator — `KoreanFRIAAssessor`, `FRIAEvaluator`, `AnnexIVMapper`, `PIAEvaluator`, `PublicDisclosureGenerator` — 는 Paper 2 에 기술되어 있다. 이번 편은 그 전체 stack 의 장기 thesis 다 — 규제는 계속 변할 것이고, 아키텍처의 핵심은 그 변화를 cheap 하게 만드는 것. 변화가 cheap 하지 않으면 앞 다섯 에피소드는 의미 없다.*

## 12개월 후의 어느 시나리오

2027년 5월. 한국 AI 기본법 시행령이 — 1년 이상 calibration 기간이 2026-01-22 에 끝난 후 — 막 발표됐다. 시행령은 원본 법에 없던 두 가지 구체 의무를 추가한다.

1. §35 영향평가에 *cross-border 데이터 노출* 차원 신설. 원본 7-차원 평가가 다루지 않던 영역.
2. 보존 의무 변경 — 평가 기록 5년에 더해 평가에 feed 된 *원본 입력* 의 *별도* 3년 보존 + 특정 encryption-at-rest 요구.

독립적으로 EU 측에서는 AI Office 가 Annex IV Section 5 (학습 데이터 속성) 에 대한 clarification 을 발행해, 학습 set 의 *demographic 분포 공개* 를 통계 요약값만이 아니라 요구한다.

컴플라이언스가 문서인 시스템에서, 이건 6주 작업이다. FRIA 템플릿 재작성. EU FRIA 재작성. Annex IV Section 5 narrative 재작성. 보존 정책 변경을 클라우드 운영팀과 조율. 데이터 처리 SOP update. 컴플라이언스 팀 교육 schedule. 리스크 팀이 이 중 하나를 skip 한다 — bandwidth 가 없으니까.

컴플라이언스가 감사 로그 위 query 인 시스템에서, 이건 generator 당 한 PR 과 config 변경 한 번이다. 아래 substrate 는 움직이지 않는다.

이게 *modular adaptability* 가 실무에서 의미하는 바고, Ep 1–5 를 우리가 작성한 방식으로 만드는 사전 비용이 아깝지 않은 이유다.

## 왜 한 개가 아니라 5개 generator 인가

규제 층 설계 시 유혹은 단일 `ComplianceReporter` 클래스 — 규제 식별자를 받아 적합한 보고서를 반환하는 — 였다. 한 클래스, 한 메서드, jurisdiction 으로 parametrize. DRY 처럼 보인다.

우리는 이걸 일찍 거절했다. Ep 4 가 `KoreanFRIAAssessor` 와 `FRIAEvaluator` 를 차원이 비슷해 보여도 분리해 둔 것과 같은 이유로 — *서로 다른 법적 기반은 한 규제의 amendment 가 다른 규제의 컴플라이언스 자세에 ripple 되는 coupling 을 공유해서는 안 된다.*

5개 generator 는 의도적으로 5개 분리된 모듈이다.

- `KoreanFRIAAssessor` — 한국 AI 기본법 §35, 7-차원 영향평가, 5년 보존
- `FRIAEvaluator` — EU AI Act Art. 9, 5-차원 위험 관리 process 기록
- `AnnexIVMapper` — EU AI Act Art. 11 + Annex IV, 12-section 기술문서 증거 매핑
- `PIAEvaluator` — 한국 PIPA + GDPR Art. 35, 6-도메인 개인정보 영향평가
- `PublicDisclosureGenerator` — 금융위 AI 가이드라인, 5-section 분기 공시

각 모듈이 자체 차원, 보존 규칙, 출력 포맷, update 주기를 own 한다. 공유하는 건 아래 substrate 뿐이다 — 같은 감사 로그, 같은 XAI 설명 column, 같은 RAG 검색 인터페이스.

위 2027년 5월 시나리오가 도착하면, 변경은 scoped:

- 새 §35 cross-border-data 차원 → `KoreanFRIAAssessor` 에 메서드 추가, YAML config 에 등록, 감사 로그 위 대응 query 작성. 다른 4 generator 는 안 움직임.
- 새 §35 원본 입력 보존 → `KoreanFRIAAssessor` run 에 feed 되는 입력에 대해 감사 로그의 `log_data_access` 테이블에 보존 정책 추가. 다른 4 generator 안 움직임.
- Annex IV Section 5 demographic 공개 → `AnnexIVMapper` 에 학습 데이터 스냅샷 테이블에서 demographic 분포 끌어오는 query 추가. `KoreanFRIAAssessor` 안 움직임. `FRIAEvaluator` 안 움직임.

Substrate (Ep 3–5) 가 각 변경을 한 곳에서 흡수한다. 그 위 규제 모듈이 한 곳에서 변한다. 총 두 PR, 어느 것도 모델이나 추론 경로를 건드리지 않는다.

## 여기서 "모듈" 이 의미하는 것

generator 모듈이 어떻게 생겼는지 구체적으로 적어두는 게 좋다. *"모듈"* 이라는 단어가 느슨하게 쓰이니까.

각 generator 모듈은 4개 부분으로 구성된다.

**Scope 선언.** 감사 로그의 어느 섹션을 query 할지 (어느 `log_*` 테이블, 어느 시간 윈도우, 어느 slice 필터). 이건 코드가 아니라 YAML config 다. 재배포 없이 변경 가능하고, 변경 자체가 로그된다.

**집계 명세.** Query 가 무엇을 계산하는지. `KoreanFRIAAssessor` 의 경우 7개 스칼라 차원 + 각각의 evidence pointer. `AnnexIVMapper` 의 경우 12개 evidence bundle, 각각 특정 감사 로그 row 또는 config 스냅샷 file 의 pointer 로 구성. 명세는 versioned 되고 runtime 에 hash-stamp 된다. 그래서 평가 record 가 자신을 produce 한 정확한 명세 버전을 carry 한다.

**Serialisation 포맷.** 출력 형태 — 규제가 요구. §35 는 JSON, Annex IV 는 구조화 PDF + JSON, 금융위 분기 공시는 Excel-호환 CSV. 받는 당국이 다르니 포맷이 다르다. 5개 모두 한 공통 형태로 강제하는 건 가장 까다로운 포맷을 모두에게 강요하는 셈이다.

**보존과 접근 정책.** §35 는 5년 WORM, Art. 9 는 고정 보존 없으나 immutable (그래서 default 10년 유지), PIA 출력은 3년 audit-controlled 접근, 금융위 공시는 공개 게시. 각 정책이 storage 층의 configuration 이지 hand-coded 행동이 아니다.

이 4개 부분이 *모듈을 모듈로 만드는* 것이다. 어느 것도 다른 것의 구현에 의존하지 않는다. 나머지를 건드리지 않고 하나를 교체한다.

## Substrate 는 움직이지 않는다

이 전체 접근의 논지는 *모듈 아래의 substrate 가 규제 agnostic* 이라는 점이다. Stack 을 내려가며:

**감사 로그 (Ep 3).** 7개 테이블, HMAC chain, 다중 에이전트 컨센서스 중재. 어느 테이블도 *"컴플라이언스"* 가 무엇인지 모른다. 무엇이 일어났는지를 기록한다. 같은 로그가 §35, Art. 9, Annex IV, PIPA, 금융위 공시에 servce 한다. 그 규제들이 별도 이벤트 스트림이 아니라 기록된 이벤트 위 query 이기 때문이다.

**XAI 설명 column (Ep 4).** 게이트 가중치, CEH 귀속, 마할라노비스 OOD 플래그. 어느 것도 특정 투명성 의무를 모른다. 매 예측 구조화 reasoning 데이터를 produce 한다. 아키텍처가 그렇게 하기 때문이고, 규제 generator 가 그 데이터를 소비하는 건 그게 마침 그들 query 의 적합한 substrate 이기 때문이다.

**검색 층 (Ep 5).** RAG over LanceDB. 규제도 모른다. 같은 vector 유사도와 time-travel query 를 인적 감독 큐, 공정성 모니터, 반사실적 평가자, 분기 집계 generator 에 모두 제공한다.

이게 앞 다섯 에피소드의 payoff 다. 각 component 가 특정 규제 가정을 baked-in 하지 않고 만들어졌다. 새 규제가 도착하면 같은 substrate 위 새 모듈로 land 하지 재설계가 아니다.

## 우리가 베팅하는 세 규제

향후 18개월 안에 예상하는 변화 셋과, 아키텍처가 그것들을 어떻게 흡수할지:

**한국 AI 기본법 시행령.** 법은 2026-01-22 에 발효됐고 1년 이상 calibration 기간이 따랐다. 시행령이 §31 투명성 의무, §34 고영향 사업자 의무, §35 영향평가 specifics 를 detail 할 것이다. 우리 입장: 각각이 `KoreanFRIAAssessor` 의 새 차원, 공시 (`PublicDisclosureGenerator`) 의 새 필드, 또는 둘 다로 번역될 것. 새 감사 로그 필드가 필요하면, 기존 query 깨뜨리지 않고 추가된다 (감사 로그가 append-schema, Lance versioning, Ep 5).

**EU AI Act 단계적 시행.** AI Act 가 2025–2027 에 걸쳐 단계적으로 시행되고 있다. 각 단계가 clarification 과 일부 amendment 를 가져온다. AI Office 가 General Purpose AI 의무와 Annex IV 기술문서 요구사항에 특히 active 했다. 우리 입장: `FRIAEvaluator` 와 `AnnexIVMapper` 가 amendment 를 새 query 패턴과 새 evidence-pointer 매핑으로 흡수한다. 둘 다 YAML config 변경.

**미국 federal AI framework.** Trump 행정부의 2026-03 National Policy Framework 가 7 pillar 와 preemption 전략을 깔았지만 작성 시점 기준 종합 federal AI Act 는 통과 안 됨. 만약 통과되면, 금융 서비스 high-impact AI 에 대한 disclosure 와 risk-management 요구사항을 specify 할 것으로 예상. EU 와 한국 요구사항과 substantially overlap 하되 jurisdictional boundary 가 다를 것. 우리 입장: 6번째 generator, `USComplianceGenerator`, 등록. Substrate 는 안 움직임.

요점은 우리가 구체 변화를 정확히 예측했다는 게 아니다. 이 *어느* 것도 재설계 없이 흡수된다는 점이 요점이다. Substrate (Ep 3–5) 가 규제 agnostic 으로, 규제 층 (이번 편) 이 모듈러로 만들어졌으니까.

## 모듈성이 아닌 것

이 접근이 약속하지 않는 것 몇 가지:

**컴플라이언스 *내용* 작업을 줄이지 않는다.** `KoreanFRIAAssessor` 에 새 차원 추가하는 건 여전히 그 차원이 무엇을 의미하는지, 어떤 evidence 가 적절한지, 어떻게 계산하는지 이해해야 한다. 아키텍처는 *plumbing* 작업을 절약하지 *legal interpretation* 작업을 절약하지 않는다. 컴플라이언스 팀과 FRM 팀은 여전히 indispensable.

**컴플라이언스 자세를 모델 quality 로부터 독립시키지 않는다.** 모델 자체가 unfair 한 예측이나 unstable 한 설명을 produce 하면, 모듈러 보고가 얼마든 그걸 fix 하지 못한다. Ep 4 (XAI) 와 원본 Paper 2 series 의 Ep 6 (운영 스트림 위 공정성) 가 그 측면을 다룬다. Modular adaptability 는 *보고* 층을 다루지 *substantive* 층은 다른 분야다.

**대규모 아키텍처 shift 로부터 보호하지 않는다.** 규제 당국이 모든 금융 AI 가 특정 algorithmic framework 를 써야 한다 (가능성 낮지만 불가능 아님), 또는 explainability 가 특정 인증 explainer 에서 와야 한다 (마찬가지로 가능성 낮음) 라고 결정하면, 모듈러 substrate 가 도움 안 된다. 우리는 Ep 4 에서 명시적으로 inherent XAI 에 베팅했고, 그 베팅에 downside 가 있다.

**모든 규제 방향을 예측하지 않는다.** 특히 개인정보법은 피처가 처리되는 방식의 구조적 재설계를 요구할 수 있는 방향으로 움직이고 있다 (homomorphic encryption, federated learning, 보호 속성에 대한 on-device 추론). 현재 substrate 가 그것들을 네이티브 지원 안 한다. 그 방향이 binding 되면 substrate 자체를 확장해야 한다, 위에 모듈만 추가하는 게 아니라.

## 6 에피소드, 1 아이디어

MRM 스레드 시리즈는 본질적으로 한 아이디어를 articulate 하려는 시도였다 — *아키텍처 안에 사는 MRM 이 정기적 문서 안에 사는 MRM 보다 cheap 하고, defensible 하고, 더 오래 산다.*

Ep 1 이 MRM 을 처음부터 아키텍처에 넣어야 하는 이유의 사례를 폈다 — 모델이 LLM 에이전트 파이프라인일 때 정기-검증 model 이 무너진다. Ep 2 가 승격 게이트에서 그게 어떻게 보이는지 보여줬다 — 모든 승격 결정이 이미 감사 entry 다. Ep 3 가 감사 로그 층 자체로 들어갔다. 7개 테이블, HMAC chain, 다중 에이전트 컨센서스 — *누가 watcher 를 watch 하는가* 가 진짜 질문이라서. Ep 4 가 한 층 내려가 설명 column 으로 — inherent XAI 가 감사 로그가 *추론 근거* 를 담게 하는 (단순 이벤트가 아니라) 것이고, FD-TVS 가 그 위에 운영 스코어링 철학으로 얹는다. Ep 5 가 한 층 올라가 검색 인터페이스로 — RAG over LanceDB 가 감사 로그를 query 가능 지식 베이스로 만든다. live oversight, 공정성 모니터링, 분기 집계 워크플로우가 모두 그걸 필요로 한다.

이번 편 (Ep 6) 이 loop 를 닫는다. Substrate 가 규제 agnostic 이고 규제 층이 모듈러이기 때문에, 규제가 변할 때 — 변할 것이다 — 작업이 generator 당 한 PR 과 config 변경으로 scoped 된다, 재설계가 아니라. 앞 다섯 에피소드의 사전 작업이 이 마지막 주장이 holds 할 때만 의미를 갖는다.

## 다음에 올 것에 대한 메모

이 작업의 자연스러운 확장 두 가지가 우리가 아직 다루지 않은 것:

- *반사실적 explainability* (CCP, Paper 2 에 간단히 언급됨) — causal teacher 의 amplified DAG 위 Pearl Rung 3 추론. 아키텍처가 지원하지만 반사실적 설명의 규제적 status 가 아직 형성 중이다.
- *Cross-jurisdictional 화해 보고* — 같은 예측을 한국과 EU 당국 양쪽에 다른 framing 으로 보고해야 할 때, 5 generator 위 집계 패턴이 자체 주제가 된다.

둘 다 underlying 작업이 성숙되면 미래 MRM 스레드 에피소드일 가능성. 지금은 6-에피소드 시리즈가 핵심 논지다 — 아키텍처가 MRM 이고, MRM 이 아키텍처다.

소스: Paper 2 [Zenodo DOI](https://doi.org/10.5281/zenodo.19622052) §5–§6 가 모듈러 generator 설계와 substrate 보장을 다룬다. 구현은 open-source repo 의 [`core/compliance/`](https://github.com/bluethestyle/aws_ple_for_financial) 에 산다.
