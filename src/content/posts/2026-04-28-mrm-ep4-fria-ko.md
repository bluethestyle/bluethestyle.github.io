---
title: "[MRM 스레드] 에피소드 4 — FRIA: AI 기본법 §35 가 코드로 산다"
date: 2026-04-28 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, fria, regulation, financial-ai, ai-basic-act]
lang: ko
excerpt: "한국 AI 기본법 §35 의 7-차원 영향평가와 5년 보존 의무. EU AI Act Article 9 FRIAEvaluator 와 리포트는 통합하더라도 내부 저장은 왜 분리해야 하는가."
series: mrm-thread
part: 4
alt_lang: /2026/04/28/mrm-ep4-fria-en/
next_title: "에피소드 5 — 티켓 큐가 아니라 API 인 Human Oversight"
next_desc: "EU AI Act Art 14, kill switch API, HumanReviewQueue tier 2/3, 그리고 auto_promote=false 가 production posture 로 강제되는 이유."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 4편. Ep 3 이 감사 로그의 구조를 다뤘다면, 이번 편은
그 로그 위에 한 층 더 올라가는 *규제 산출물* 의 이야기 — FRIA
(Fundamental Rights Impact Assessment).*

## 2026년 4월, 금감원의 첫 요청

2026-01-22 AI 기본법 시행. 금융 상품 추천은 §35 고영향 AI 로 분류.
4월 어느 날 금감원 감사관이 영향평가 기록을 요청한다. "귀 기관의
추천 AI 가 AI 기본법 §35 가 정한 7-차원 영향평가를 수행했습니까?
평가 결과와 완화 조치를 5년간 보존 가능한 형태로 제출하십시오."

Ep 1 에서 "규제를 사후 문서가 아니라 아키텍처에 박는다" 고 했다.
FRIA 는 그 주장의 가장 까다로운 시험대다 — 왜냐하면 FRIA 는 *한
번 작성하고 끝* 이 아니라, 모델이 바뀔 때마다 재평가하고 5년간
감사 가능한 형태로 축적되어야 하기 때문.

## 왜 클래스 두 개인가

우리 코드베이스에 FRIA 관련 클래스가 두 개 있다:

- `core/compliance/fria_assessment.py::KoreanFRIAAssessor` —
  AI 기본법 §35, 7-차원, 5년 보존
- `core/monitoring/fria_evaluator.py::FRIAEvaluator` — EU AI Act
  Article 9, 5-차원, 지속 모니터링

겉보기에는 중복이다. 둘 다 "영향평가" 고, 둘 다 비슷한 차원을 평가
한다. 처음 설계할 때 "하나로 합치자" 는 유혹이 있었다.

합치지 않은 이유 — *법적 기반이 다르다*. AI 기본법 §35 와 EU AI Act
Article 9 는 각각의 사법 관할에서 독립적으로 성립하는 의무이고,
한쪽의 평가 결과로 다른 쪽을 충족시킬 수 없다. 차원 구성도 미묘하게
다르다:

- AI 기본법 §35 의 7-차원: 생명·안전, 기본권, 차별, 투명성, 인적
  감독, 개인정보, 책임성.
- EU AI Act Article 9 의 5-차원: risk identification, risk
  estimation, risk evaluation, risk management measures,
  residual risk.

한국 §35 는 *침해될 수 있는 권익* 을 열거하고, EU Art 9 은 *리스크
관리 절차* 를 열거한다. 관점이 다르다.

리포트를 외부에 제출할 때는 통합된 한 문서로 묶을 수 있다 (둘의
공통 구성요소가 많다). 하지만 *내부 저장* 은 분리. 한 클래스가 두
법을 동시에 책임지면, 한쪽 법이 개정될 때 다른 쪽 법 대응까지 같이
무너진다. 분리된 클래스 두 개가 `AnnexIVMapper` 같은 통합 리포터를
통해 합쳐지는 구조가 안정적.

## 7-차원 평가의 구조

`KoreanFRIAAssessor` 는 새 모델 등록 시 자동 호출된다. 각 차원에
대해 (score, evidence_ref, mitigation) 세 필드를 기록:

- **생명·안전** — 금융 상품 추천이 직접 생명/안전 위협은 낮음,
  하지만 부적절 상품 추천의 누적 재무 피해 가능성 중간.
- **기본권** — 차별 가능성 (다음 차원 참조), 금융 접근권 영향
  평가.
- **차별** — 5개 보호 속성 (성별, 연령대, 지역, 소득 구간, 장애
  여부) 에 대한 Disparate Impact · Statistical Parity Difference ·
  Equal Opportunity Difference. 이 평가는 Ep 6 에서 다룰 공정성
  모니터의 출력을 그대로 인용.
- **투명성** — 설명 생성 가능 비율 (전문가 게이트 가중치 기준),
  고객 대상 설명 문구 reading-level.
- **인적 감독** — kill switch 발동 이력, HumanReviewQueue tier
  2/3 처리량, escalation 응답 시간.
- **개인정보** — PIPA §37의2 프로파일링 opt-out 처리율, 신용정보법
  §36의2 대응 기록.
- **책임성** — 모델 버전·학습 데이터 버전·config 버전의 trace
  가능성 (Ep 3 감사 테이블 조인으로 답).

각 차원의 evidence 는 Ep 3 의 감사 테이블 row 를 포인터로 가진다 —
*평가 결과가 감사 로그 위에 서있다*. 이게 "FRIA 가 코드로 산다" 의
의미.

## 5년 보존 — WORM + 해시 체인

§35 는 *평가 결과를 5년간 보존* 하라고 요구한다. 보존만이 아니라
위변조 방지까지. 우리 구현:

- FRIA 결과는 `fria_assessments_v1` Parquet 테이블에 S3 에 저장
- 버킷은 Object Lock (WORM, Write Once Read Many) 모드
- 각 엔트리는 Ep 3 의 HMAC 체인에 한 번 더 서명
- 5년간 삭제 불가 (bucket policy)

"삭제 불가" 가 중요하다. 모델 교체 후 이전 평가가 부끄럽게 느껴져도,
감사 관점에서는 남아야 한다. 규제 대응의 핵심은 *평가 자체의 완벽함*
이 아니라 *평가가 있었고 그 흔적이 보존됐다* 는 증명 가능성이다.

## KoreanFRIAAssessor vs FRIAEvaluator — 실제 호출 흐름

새 모델이 레지스트리에 등록되면:

1. Ep 2 의 `_decide_promotion()` 게이트 통과
2. 승격 결정된 경우, `KoreanFRIAAssessor.assess()` 호출 → 7-차원
   평가 실행 → `fria_assessments_v1` 에 저장
3. 병행하여 `FRIAEvaluator.evaluate()` 호출 → 5-차원 평가 →
   `fria_eu_v1` 에 저장
4. `AnnexIVMapper.aggregate(korean_result, eu_result)` → Article 11
   기술문서용 통합 리포트 생성
5. 감사 로그에 `log_operation(event="fria_complete", ...)` 엔트리

각 단계가 실패해도 파이프라인 자체는 실패로 끝나지 않는다 — 감사
인프라 실패가 운영 중단으로 번지는 걸 피하려는 best-effort 호출
(Ep 2 의 `_audit_promotion` 과 같은 원칙). 대신 실패 사실이 `log_
operation` 에 별도 엔트리로 남아 AuditAgent 가 야간에 발견.

## 금감원 쿼리의 답

"5년간 보존된 FRIA 결과를 제출하라" — 쿼리 하나. `fria_assessments_v1`
테이블에서 관심 시점의 모델 버전 필터. 7-차원 평가 결과 + 각 차원의
evidence_ref (감사 테이블 row) + mitigation 기록이 세트로 나온다.

1년 뒤, 2년 뒤, 5년 뒤에도 같은 쿼리가 같은 답을 돌려준다. 중간에
조직의 담당자가 바뀌어도, 모델이 여러 번 교체되어도. 이게 §35 가
요구하는 *지속적 재구성 가능성* 이다.

## 여전히 사람의 일

아키텍처가 사주지 못하는 것:

- **차원 scoring 의 실제 값은 사람이 결정한다.** "차별 차원 점수
  0.72 가 적절한가?" 에 답하는 건 공정성 위원회. 자동화되는 건 점수
  계산이지 점수 판정이 아니다.
- **Mitigation 설계.** 평가 결과에 따라 어떤 조치를 취할 것인가 —
  임계값 조정, 학습 데이터 리밸런싱, 인적 리뷰 강화. 이건 FRM
  자격증 보유자의 전문성 영역.
- **법 개정 대응.** §35 가 개정되면 `KoreanFRIAAssessor` 의 차원
  구성이 바뀌어야 한다. 분리된 클래스 두 개가 있는 이유 — 한쪽
  개정이 다른 쪽을 건드리지 않아야 유지보수가 가능.

## 다음 편

Ep 5 는 FRIA 의 "인적 감독" 차원을 깊이 파고든다 — EU AI Act Article
14 가 말하는 human oversight 를 티켓 큐가 아니라 API 엔드포인트로
구현한 방식. kill switch, HumanReviewQueue tier 2/3, 그리고
`auto_promote=false` 가 production posture 로 강제되는 이유.

원문 자료: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
§6 "규제 매핑". 구현은 `core/compliance/fria_assessment.py`,
`core/monitoring/fria_evaluator.py`, `core/compliance/annex_iv_mapper.py`
에 있다.
