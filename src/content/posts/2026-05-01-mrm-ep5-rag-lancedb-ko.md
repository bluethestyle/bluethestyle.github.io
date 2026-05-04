---
title: "[MRM 스레드] 에피소드 5 — RAG + LanceDB · 감사 인프라가 결국 검색 문제인 이유"
date: 2026-05-01 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, rag, lancedb, audit, retrieval, financial-ai]
lang: ko
excerpt: "감사 로그는 쓰기 전용이 아니다. 질의 가능한 지식 베이스다. 운영/감사 검색을 RAG + LanceDB — columnar, 버전 인식, time-travel — 로 구성한 이유, 그리고 그 결과로 인적 감독·운영 스트림 위 공정성 모니터링·분기 집계가 어떻게 풀리는가."
series: mrm-thread
part: 5
alt_lang: /2026/05/01/mrm-ep5-rag-lancedb-en/
next_title: "에피소드 6 — Modular Adaptability · 규제는 변해도 아키텍처는 변하지 않는다"
next_desc: "한국 AI 기본법 시행령, EU AI Act amendment, 미래 미국 framework — 모두 도착할 것이다. 5개 규제 generator 는 5개 모듈이지 5개 문서가 아니다. 아키텍처 모듈성이 왜 장기 베팅인가."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 5편. Ep 4 가 감사 로그가 매 예측 구조화 설명 — 게이트 가중치, CEH 귀속, 마할라노비스 OOD 플래그 — 을 담는 이유는 아키텍처가 그것들을 부산물로 produce 하기 때문이라고 정리했다. Ep 5 는 한 층 위로 — 그 로그가 어떻게 query 되는지로 — 올라간다. 우리가 피하고 싶었던 실수는 감사 로그를 *분기에 한 번씩 누군가 batch 로 읽는 쓰기 전용 sink* 로 다루는 것이었다. live 한 query 가능한 지식 베이스가 되어야, 우리가 이걸 만든 다른 모든 이유가 작동한다.*

## 인적 감독 큐의 어느 시나리오

리스크 담당자가 22:30 에 on-call 중이다. HumanReviewQueue 가 tier-2 알림을 띄운다. 어느 고객의 추천 한 건이 Causal Guardrail (Ep 4 의 마할라노비스 OOD 플래그) 발사로 escalate 됐다. 담당자는 5분 안에 결정해야 한다 — 모델 override, 야간 검토 hold, 진행 통과 중 어느 하나.

그 순간 담당자에게 필요한 것:

- 매 예측 전체 기록 (입력 피처, 게이트 가중치, CEH 귀속, OOD 거리 점수)
- *유사한 과거 예측들* — 같은 고객, 비슷한 피처 패턴, 그때는 모델이 뭐라고 했나?
- *최근 같은 상품 카테고리에서 OOD 플래그가 발사된 예측들* — 고립된 사건인가, 드리프트 시그널인가?
- 현재 모델 버전의 학습 데이터 스냅샷 reference — 입력이 실제로 모델이 봤어야 할 범위 밖인지 sanity-check 가능하게

이 네 가지 중 셋이 *검색 query* 다. 단일 record 조회가 아니다. primary key 가 없다. *유사도 조건* 과 *시간 윈도우* 가 있다.

감사 로그가 평면 append-only Parquet 테이블이면, 이 query 들은 분 단위 (full scan) 가 걸리고 담당자의 5분 결정 윈도우는 무너진다. 감사 로그가 vector-aware columnar store 위 RAG 로 노출되면, 초 단위로 끝난다.

## 감사 로그가 쓰기 전용이 아닌 이유

금융권 감사 로그에 대한 default mental model 은 *규제 당국의 archive* 다. 발생 시점에 이벤트를 쓰고, 엔트리에 hash chain (Ep 3) 을 걸고, 요구되는 기간 동안 보존하고 (Ep 4 가 5년 케이스 다룸), 요청이 오면 응답한다. 읽기 접근은 드물고 batch 다.

이 model 이 AI MRM 맥락에서 깨지는 이유 셋:

**Live 감독은 live 검색이 필요하다.** 위 예시 — 5분 윈도우 안에 유사 케이스 맥락이 필요한 리스크 담당자 — 는 분기 batch query 가 아니다. interactive decision-support 워크로드이고, 엄격한 latency 요구가 있다. 감사 로그를 cold storage 로 다루는 건 oversight 팀이 *별도의* 자체 working store 를 짓게 만들고, 두 개의 source-of-truth 가 가져오는 모든 divergence 문제가 따라온다.

**공정성 모니터링은 운영 스트림에서 돌아간다.** Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference — 이건 분기에 한 번 validation sample 위에서 계산되는 게 아니다. 공정성을 진지하게 다루는 운영 AI 시스템에서는 *연속적으로* 실 추론 스트림 위에서, 보호 속성 slice 별로 계산된다. 그 스트림은 감사 로그 안에 산다. 감사 로그가 near-real-time 으로 query 가능하지 않으면, 공정성 모니터는 더 stale 한 proxy 위에서 돌거나 자체 duplicate 스트림을 갖는다.

**규제 산출물 생성이 query 다.** Ep 4 에서 이미 5개 generator (FRIA / EU FRIA / Annex IV / PIA / 공시) 에 대해 이 논지를 폈다. 각 generator 는 감사 로그 위 집계 query 로 돈다. 그 query 가 시간 단위 걸리면 분기 산출물 파이프라인이 아무도 디버깅 못 하는 batch job 이 된다. 초 단위 끝나면, generator 로직 변경 시 on-demand 재실행 가능하다.

감사 로그는 데이터의 *최종* 목적지가 아니다. 다른 모든 것이 query 하는 *source of truth* 다.

## 두 개 store 의 안티 패턴

감사 로그가 live query 에 너무 느리다는 걸 깨달은 팀이 흔히 가는 path 는 두 번째 store 를 짓는 것이다. 감사 로그 = immutable Parquet archive. 운영 query = 별도 Postgres 또는 Elasticsearch. CDC 나 batch job 으로 sync.

문제는 divergence 다. 두 store 는 어느 순간이든 동일 이벤트의 *살짝 다른 view* 를 담게 된다. Sync 지연, schema drift, replication 실패, 보존 정책 mismatch — 각각이 외부 감독자가 잡아낼 수 있는 *audit-vs-ops 불일치* 의 source 다. 더 나쁘게는, 6개월 후 그 불일치가 발견됐을 때 어느 store 도 이유를 설명 못 한다.

우리가 따르는 single-store 규칙: *감사 로그가 유일한 기록 source 이고, 운영 query 가 그 위에서 직접 돈다.* 이게 부담을 storage 층으로 옮긴다 — 컴플라이언스용 immutable append-only 쓰기와 운영용 빠른 indexed 검색 둘 다 지원해야 한다. LanceDB 가 그걸 가능하게 만든 선택이었다.

## LanceDB 를 고른 이유

이 결합 워크로드에 중요한 non-obvious 속성 몇 가지.

**Vector 네이티브 인덱싱 + columnar 저장.** LanceDB 는 데이터를 Apache Arrow columnar 포맷으로 저장한다. 감사 로그 위 분석 query 에 적합한 모양이다 (시간 범위 필터, 보호 속성별 집계, 나머지 안 건드리고 단일 column scan). 그 위에 IVF-PQ vector 인덱스를 네이티브 지원해서, *설명 column* — 매 예측 게이트 가중치와 CEH 귀속 — 을 정확 매칭이 아닌 유사도로 query 가능하다.

**버전 인식 "time travel".** 매 쓰기가 새 versioned snapshot 을 만든다. *2026-04-15 14:00:00 UTC 시점의 감사 로그* 를 query 가능하지, *현재 시점의 감사 로그* 만이 아니다. 이게 감독자의 15개월 재구성 query 가 작동하게 만드는 것 — 그 시점의 모델 레지스트리, 그 시점의 추론 로그, 그 시점의 귀속 로그가 모두 일관되게 join 된다.

**설계 단계에서 append-only.** 새 쓰기는 새 버전이지, 기존 버전 덮어쓰기가 아니다. HMAC chaining (Ep 3) 과 결합해서, 감사 로그가 필요로 하는 immutability 속성을 storage 층과 싸우지 않고 얻는다.

**임베드가 싸다.** 별도 cluster 없음, operator 없음. in-process 또는 sidecar 로 돈다. ops/audit 인프라가 한 사람이 야간에 책임지는 여러 일 중 하나인 작은 팀에서, 이게 보이는 것보다 더 중요하다.

비용은 실재한다. LanceDB 는 Postgres 나 Elastic 보다 어리고, operator-facing 도구 성숙도가 낮고, 커뮤니티 ecosystem 이 작다. 우리는 그 비용을 두 store 대신 한 store 를 갖는 가격으로 받아들였다.

## 설명 column 위 RAG

retrieval-augmented 부분이 live oversight 워크플로우를 viable 하게 만든다.

리스크 담당자가 OOD 플래그 발사된 예측을 볼 때, RAG 층은 *그 특정 예측* 의 설명 vector (게이트 가중치 7-element vector + CEH 귀속 sparse 피처 기여 vector + OOD 거리 스칼라) 를 가져와서 지난 90일 추천 위에서 vector 유사도 검색을 돌린다. 결과는 *추론 방식이 구조적으로 유사한 예측들* 의 ranked list — 고객 ID 무관, 상품 무관, 설정 가능 시간 윈도우 scoped.

이건 SQL `WHERE customer_id = X` query 가 아니다. *"같은 방식으로 생각한 예측들 찾아라"* 다. 현재 OOD 플래그가 anomaly 인지 패턴인지 결정하는 담당자에게, 이게 정확히 답해야 할 질문이다.

같은 검색 패턴이 다른 세 워크로드에 servce:

- **드리프트 탐지.** 이번 주 예측들이 지난 분기와 같은 expert mix 를 끌고 있나? 게이트 가중치 vector 분포의 시간 변화는 전통적 drift 메트릭이 따라잡기 몇 주 전에 피처 분포 drift 의 leading indicator 다.

- **반사실적 검토.** 특정 예측 추천이 주어졌을 때, 한 피처를 perturb 한 유사 고객들은 무엇을 추천받았을까? RAG 가 비교 set 을 retrieve 하고, 반사실적 층 (Paper 2 에서 간단히 다룸) 이 그 위에서 돈다.

- **설명 일관성 체크.** 모델이 유사 입력에 대해 유사 설명을 주는가? 설명 column 위 RAG 가 모델의 *추론이 안정적* 임을 검증할 수 있게 한다. 출력 일관성과는 다르고 더 엄격한 속성이다.

## 공정성 path

연속 공정성 모니터는 보호 속성 slice scope 으로 감사 로그 위 streaming query 로 돈다. 5개 보호 속성 (성별, 연령대, 지역, 소득 계층, 장애 여부) 별 *Disparate Impact* 가 curated validation sample 이 아니라 실 운영 예측의 rolling 24시간 윈도우 위에서 계산된다.

LanceDB 가 아래 깔려 있어 떨어지는 디자인 선택 두 가지:

**반사실적 챔피언-챌린저.** 공정성 층은 *"현재 운영 모델이 공정한가"* 만 묻지 않는다. *"챌린저 모델이 같은 운영 스트림에서 더 또는 덜 공정했을 것인가"* 를 묻는다. 챌린저 모델 예측이 동일 retrieved 입력 batch 위에서 오프라인 계산되고 비교된다. RAG 가 매칭된 batch 를 retrieve 하고, 챔피언-챌린저 비교가 위에서 돈다. 비교 결과의 Parquet archive 도 LanceDB 테이블이라, 추론 로그와 동일한 방식으로 query 가능하다.

**실시간 임계값 breach.** 보호 속성 slice 가 rolling 윈도우에서 공정성 임계값을 넘으면, HumanReviewQueue 가 즉시 tier-3 알림을 받는다. OOD 플래그 발사 예측이 가는 같은 큐다, severity tier 만 다르게. 둘 다 같은 retrieval substrate 에서 흐른다.

요점은 우리가 화려한 공정성 모니터를 만들었다는 게 아니다. *retrieval substrate 가 다른 이유들로 이미 자리잡아 있어서 공정성 모니터링이 cheap 해졌다* 는 게 요점이다. 좋은 인프라 선택은 이렇게 보인다 — 처음에 설계 안 한 곳에서 dividend 를 계속 produce 한다.

## 인적 감독 path

EU AI Act Art. 14 의 인적 감독 요구사항이 우리 구현에서 ticket queue 가 아닌 API endpoint 집합으로 산다. 세 흐름:

**Kill switch.** 전체 시스템에 새 예측을 halt 시키는, 2-factor operator 인증이 필요한 단일 API call. Kill switch 이벤트 자체가 감사 로그에 `log_operation` 쓰기로 떨어져, *halt 이유* 가 나중에 복구 가능하다.

**Tier 2 / Tier 3 escalation.** HumanReviewQueue 가 tiered severity 를 갖는다. Tier 2 = OOD 플래그 발사, 공정성 slice 가 임계값에 근접, 또는 컨센서스 중재자가 dissent. Tier 3 = 공정성 임계값 breach, 상관된 예측에서 다중 OOD 플래그, 또는 kill switch 발동. 각 tier 가 자체 검색 템플릿을 갖는다 — 담당자가 tier 에 적합한 사전 fetch 된 유사 케이스 맥락을 본다.

**`auto_promote=false` default 자세.** 모델 승격이 명시적 operator 승인을 요구한다 (Ep 2 챔피언-챌린저 다룸). 이게 oversight 섹션에 사는 이유는, *operator 의 승인 결정 자체가 감사 로그 위 query* 이기 때문이다. 챌린저 모델의 공정성 모니터가 green 이었나? 테스트 윈도우에 OOD 플래그가 있었나? 컨센서스 중재자가 뭐라 말했나? RAG 가 승인 인터페이스의 일부로 관련 맥락 bundle 을 fetch 해, operator 결정이 informed 되되 부담스럽지 않게 한다.

세 흐름 모두 감사 로그로 다시 쓴다. 감독은 시스템 *밖에서* 일어나는 것이 아니다. 시스템 *안에서* 기록되고, 다른 모든 것처럼 query 가능하다.

## 풀리지 않는 것

RAG-over-LanceDB 접근의 정직한 한계 몇 가지:

**임베딩 드리프트.** 설명 column 의 vector 표현은 우리가 한 임베딩 선택에 의존한다. 모델 아키텍처가 substantially 변하면 (새 expert 추가, 게이트 차원 변경), 기존 설명 vector 가 새 vector 와 더 이상 비교 가능하지 않다. 우리는 versioned 임베딩 store 로 처리하지만, 사실은 그대로다 — 큰 아키텍처 개정을 가로지르는 장기 검색은 within-version 케이스보다 어렵다. Ep 6 (모듈성) 이 부분적으로 이걸 필요 이상 어렵게 만들지 않는 것에 대한 이야기다.

**Cold-start 워크로드.** 과거 예측 데이터가 없는 새 배포는 유사도 검색 혜택을 못 받는다. 첫 몇 주는 oversight 워크플로우가 단일 record lookup 으로 격하된다. 알려진 한계다. workaround 는 staging 단계에서 합성 벤치마크 예측으로 설명 store 를 seed 해, 첫 날부터 검색이 작업할 거리를 갖게 하는 것.

**Query 전문성.** Vector 유사도 query 는 SQL query 보다 *미묘하게* 틀리기 쉽다. 잘못된 거리 메트릭이나 잘못된 시간 윈도우로 *"유사 예측 찾아라"* 를 돌리면 false neighbour 더미가 돌아온다. 우리는 operator 인터페이스에 사전 정의된 검색 템플릿 작은 set 만 노출하고 free-text query box 는 아닌 방식으로 mitigate 한다. ad-hoc query 는 데이터 사이언스 팀에 제한된 노트북 인터페이스를 통한다.

## 다음

Ep 6 가 시리즈를 마무리한다. 장기 thesis 와 함께 — 규제는 변할 것이다. 한국 AI 기본법 시행령 detail 이 도착할 것이다. EU AI Act 가 amend 될 것이다. 미국 framework 가 통과되면 자체 generator 를 요구할 것이다. 5개 규제 산출물 (Paper 2 의 KoreanFRIAAssessor, FRIAEvaluator, AnnexIVMapper, PIAEvaluator, PublicDisclosureGenerator) 은 *모듈* 이지 문서가 아니다 — 그리고 아키텍처가 새 규제를 시스템 재설계가 아닌 같은 감사 로그 substrate 위 새 모듈로 받아들이도록 set up 되어 있다.

소스: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) 의 운영 아키텍처. LanceDB 선택과 검색 템플릿은 [open-source repo](https://github.com/bluethestyle/aws_ple_for_financial) 의 `core/audit/` 와 `core/retrieval/` 에 산다.
