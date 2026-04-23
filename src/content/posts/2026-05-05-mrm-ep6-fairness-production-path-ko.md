---
title: "[MRM 스레드] 에피소드 6 — Fairness as a Production Path"
date: 2026-05-05 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, fairness, monitoring, regulation, financial-ai]
lang: ko
excerpt: "5개 보호 속성에 대한 Disparate Impact · Statistical Parity · Equal Opportunity 가 검증 샘플이 아닌 프로덕션 스트림에서 실시간 계산되는 구조, Counterfactual Champion-Challenger 의 역할, 그리고 Parquet archive 로 흐르는 증거의 길."
series: mrm-thread
part: 6
alt_lang: /2026/05/05/mrm-ep6-fairness-production-path-en/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 6편 (최종). Ep 1~5 가 감사 · 승격 · 기록 · FRIA ·
인간 감독을 차례로 다뤘다. 이번 편은 남은 큰 조각 — 공정성 ——
이 어떻게 *프로덕션 경로 위의 실시간 계산* 으로 구현됐는지, 그리고
Counterfactual Champion-Challenger 가 공정성 판단에 어떻게 쓰이는지.*

## 분기 리포트가 아니라 매 요청

종래 공정성 감사 패턴 — 분기마다 validation sample 에서 Disparate
Impact · Statistical Parity · Equal Opportunity 를 계산하고 리포트
로 제출. 금감원·국내 감독 당국도 이 수준을 *통상 기대* 한다.

문제는 분기 단위의 샘플이 *production 분포를 반영한다는 보장이 없음*.
Validation 샘플은 보통 random hold-out 이고, production 의 실제
트래픽은 계절성·마케팅 캠페인·새 고객군 유입 등으로 분포가 이동한다.
Validation 에서 DI 0.88 (양호) 인 모델이 production 에서 0.76
(경계치 위반) 으로 나오는 건 흔한 시나리오.

우리 구조는 *production 예측 스트림 위에서 실시간 계산*. 매 예측이
Ep 3 의 `log_model_inference` 에 쌓일 때, fairness monitor 가 지정된
슬라이딩 윈도 (기본 10,000 건 또는 24시간) 에서 DI/SPD/EOD 를 다시
계산한다. 임계치 위반이 감지되면 kill switch 자동 발동 (Ep 5).

## 5개 보호 속성

공정성 계산의 대상은 5개 고객 보호 속성:

1. **성별** (남/여/미상)
2. **연령대** (20대 이하, 30대, 40대, 50대, 60대 이상)
3. **지역** (서울/광역시/지방)
4. **소득 구간** (5분위)
5. **장애 여부** (self-reported, optional field)

각 속성별로 3가지 지표 동시 계산:

- **Disparate Impact (DI)** — 보호군 긍정률 / 비보호군 긍정률.
  4/5 rule (0.8 threshold) 적용. 대안 임계치는 config.
- **Statistical Parity Difference (SPD)** — 두 군 긍정률 차이.
  |SPD| ≥ 0.1 시 경고.
- **Equal Opportunity Difference (EOD)** — True Positive Rate 의
  군간 차이. |EOD| ≥ 0.1 시 경고.

5 × 3 = 15개 지표가 각 윈도에서 동시 모니터링. 하나라도 threshold
위반이면 해당 task 에 대한 kill switch 자동 발동.

## 슬라이딩 윈도, 왜 그렇게 하는가

매 예측마다 재계산하면 계산 부담이 크고, 몇 분 단위 단기 노이즈에
과민반응한다. 매 일 단위 계산은 반응이 너무 느리다. 두 극단 사이에서
10,000 예측 또는 24시간 슬라이딩 윈도가 실무적 sweet spot.

10,000 건 shuffling 은 AWS Lambda 에서 별도 worker 로 돈다. 메인
서빙 경로에 지연 없음. 계산 완료되면 CloudWatch metric 으로 push,
임계치 위반은 PagerDuty + `log_operation` 에 기록.

## Parquet archive 로 쌓는 증거

매 윈도의 15개 지표가 S3 Parquet 에 쌓인다. 날짜·속성·지표 기준
파티셔닝, snappy 압축, append-only.

왜 Parquet archive 인가 — 3가지 용도:

1. **규제 감사 증거.** "2026년 7월부터 9월까지의 공정성 지표 추세"
   요청에 대해 한 쿼리로 응답. DuckDB httpfs 확장으로 S3 직쿼리.
2. **FRIA 의 evidence_ref.** Ep 4 의 KoreanFRIAAssessor 가 새 모델
   승격 시 "차별" 차원의 점수를 이 archive 에서 계산. 평가가 정적
   문서가 아니라 *최근 production stream 의 관찰* 위에 서있게.
3. **Counterfactual C-C 입력.** 다음 섹션.

## Counterfactual Champion-Challenger

Ep 2 의 `_decide_promotion()` 은 학습 메트릭 (AUC 등) 기반 비교.
공정성도 포함 가능하지만 학습 세트에서의 공정성이라 *현실 분포* 의
공정성과 일치 보장 없음.

Counterfactual C-C 는 production archive 위에서 다른 질문에 답한다
— "만약 Challenger 가 실제로 서빙됐다면 공정성 지표가 어땠을까?"
이걸 실제로 서빙하지 않고 계산하는 방법이 Importance Sampling (IPS)
/ Self-Normalized IPS (SNIPS).

필요한 것 — production 에 기록된 *logged propensities* (Champion 이
각 대안 상품을 추천한 확률). 이게 로그되어 있으면 Challenger 의
hypothetical 공정성을 재구성 가능. Counterfactual evaluator 가
estimator 종류 (IPS / SNIPS), 최소 lift, bootstrap CI 등의 파라미터를
config 에서 읽어 실행한다.

이 덕분에 "Challenger 가 공정성에서 Champion 보다 나은가" 라는 질문에
*실제 production traffic 위에서* 답 가능. 단순 A/B 테스트처럼 실서빙
분할 없이. 대규모 서빙 분할이 부담스러운 3인 팀에게 특히 유용.

## 임계치는 누가 정하는가 — 여전히 사람

4/5 rule (DI 0.8 threshold) 은 US EEOC 의 표준이지만, 한국 금소법과
AI 기본법 §35 에서 이보다 엄격한 임계치를 요구할 수 있다. 임계치
결정은 여전히 리스크관리위원회 또는 담당 임원 결재선의 몫이고, 코드에 박아 넣지 않고
config-driven 으로만 관리한다.

임계치 변경은 반드시 회의록 → PR → review → merge 경로를 거쳐야 한다. 조용한
config 변경도 임계치 변경 이벤트로 감사 로그에 자동 기록되어 누가
언제 왜 변경했는지 직접 확인할 수 있다.

이 구조의 핵심 — *임계치 결정 자체가 감사 대상* 이다. 단순히 "임계치
위반이 있었는가" 만이 아니라 "이 임계치가 합리적인가" 도 분기 결재선의
심사 주제가 된다.

## 전체 체인이 설계상 어떻게 돌아가는가

실 트래픽 수집은 2026-04-30 에 시작됐으니, 아래 체인은 실제 프로덕션
incident 가 아니라 합성 공정성 위반 시나리오로 훈련된 상태다. 아래는
case study 가 아니라 설계 목표다 — 파트너 기관에 실 트래픽이 쌓이기
시작한 뒤 시퀀스가 *어떻게 돌아가야 하는가* 의 모양.

주담대 추천 task 에서 65세 이상 연령층의 Disparate Impact 가 금소법
§17 적합성 임계치 아래로 떨어지는 10,000 예측 rolling window 를
가정하자. 설계된 응답 체인:

*0분* — 공정성 모니터가 window 를 닫고 임계치 위반 감지, `log_operation`
엔트리 `event="fairness_breach"` 기록.

*초 단위 뒤* — 영향받은 task 에 자동 kill switch 발동. 서빙 Lambda
환경 변수 갱신, 해당 task 의 신규 추론 요청이 증류된 모델 출력 대신
Layer-3 금융 DNA 기반 룰 추천 (보수적 폴백) 을 반환. PagerDuty 가
on-call 엔지니어에게 통지. `#mrm-incidents` Slack 채널에도 알림.

*약 10분* — 엔지니어가 대시보드를 열고 SPD · EOD 가 DI 하락이 노이즈
가 아님을 확인. 감사 로그 archive 에서 영향 window 를 쿼리, 보호군
예측에서 어떤 피처가 결과를 끌었는지 분석.

*약 25분* — 아카이브 window 에 Counterfactual Champion-Challenger
분석 실행, 직전 Champion 이 같은 요청들에 대해 더 나은 DI 를 냈을지
테스트. 개선되면 그 버전이 롤백 후보로 떠오른다.

*약 35분* — 엔지니어가 직전 known-good Champion 으로 force-promote
실행, 롤백 사유 문서화. `log_model_promotion` 엔트리 —
`decision=force_promote`, `trigger=manual`, reason="주담대 task 65세+
세그먼트 DI 위반; 직전 Champion counterfactual DI 가 임계치 회복",
operator ID.

*약 37분* — 직전 Champion 서빙 중. 영향받은 task 재활성화.

*약 48분* — OpsAgent 가 체인을 지켜보다가 리스크관리위원회 또는 담당 임원 주례 검토용
incident ticket 초안 작성, 관련 감사 엔트리로 링크.

금감원의 72시간 서면 보고 window 한참 안에서 담당 결재선은 *사후 재구축된
증언* 이 아니라 *재구성 가능한 감사 증거* 에 근거한 전체 서사를 받는다.

45분이 달성 가능한지는 실제 incident 가 프로덕션에 들어온 뒤 검증된다.
설계 목표 — *한 번의 인간 판단, 나머지는 구조적* — 가 이번 편의
아키텍처가 가능케 하려는 것이다. 분 단위가 중요한 이유 — 24/7 ML 운영
rotation 이 없는 한국 중견 금융기관은 수 일의 해결 시간을 감당 못
한다. 이 아키텍처가 소규모 팀에게 single-digit 시간 해결을 가능케
만드는 설계다.

## 공정성 실패 시 복구 경로

공정성 위반 감지 → 자동 kill switch → 이후 복구는 사람 판단. 가능한
선택지:

1. **이전 known-good 모델로 즉시 롤백** (Ep 2 의 force-promote 시나
   리오)
2. **특정 세그먼트에만 fallback 룰 적용** (Layer 3)
3. **해당 task 만 비활성** (다른 task 는 정상 서빙)
4. **Counterfactual C-C 로 대안 모델 평가 후 수동 승격**

어느 경로를 택할지는 위반 유형과 심각도에 따라 다르고, on-call
엔지니어가 판정. 매 복구 이력은 `log_operation` 에 쌓여 사후 패턴
분석 대상.

## 이 편으로 스레드를 마감하며

6편에 걸쳐 "MRM 을 사후 보고서에서 구조적으로 보장되는 체계로" 바꾸는
구체적 모습을 다뤘다 — 아키텍처 (Ep 1), 승격 게이트 (Ep 2), 감사 체인
(Ep 3), FRIA (Ep 4), 인간 감독 (Ep 5), 공정성 (Ep 6).

관통하는 패턴 — *규제 대응이 핵심 코드의 부산물이 될 때* 가장 강하다.
문서로 증명하려 하면 문서가 코드와 drift 하면서 실효성을 잃고, 위원회
회의로 덮으려 하면 회의 주기가 사건 주기와 어긋난다. 핵심 코드에
박혀 있으면 시스템이 도는 한 준수가 유지되고, 시스템이 멈추면 그
자체가 감사 기록에 남는다.

3인 팀·소비자용 GPU·RTX 4070 의 제약 안에서 이런 구조가 가능했던
건 *규제 대응을 따로 만들지 않았기 때문* 이다. 아키텍처 결정의 부산
물로 감사 로그가 생기고, 승격 게이트의 부산물로 판정 기록이 생기고,
공정성 모니터의 부산물로 규제 증거가 쌓인다. 따로 하나 더 만드는
게 아니라 *이미 하는 일이 규제 요구에 응답하도록 구조* 가 배열된 것이다.

여전히 사람의 일 — 임계치 결정, 설계 심사, 사건 복구 판단, 법 개정
대응. 리스크관리위원회 또는 담당 임원 결재선이 *심사하는 대상* 이 리포트에서
코드와 로그로 이동했을 뿐, 역할 자체가 사라진 건 아니다. 감독 당국이
준수를 *아키텍처에서 직접 확인* 하길 기대하는 시대에, 이 이동은 자연스러운
정렬이다.

## 스레드는 여기서 닫지만

MRM 스레드는 이 6편으로 골격을 마쳤다. 이후에는 특정 사건·이슈에
따라 부정기적 commentary 형태로 이어질 예정 (Commentary 카테고리).

원문 자료 전체: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052).
코드: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).
