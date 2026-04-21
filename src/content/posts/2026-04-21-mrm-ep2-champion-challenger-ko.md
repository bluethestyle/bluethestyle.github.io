---
title: "[MRM 스레드] 에피소드 2 — 관문으로서의 Champion-Challenger"
date: 2026-04-21 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, sr-11-7, regulation, financial-ai, audit]
lang: ko
series: mrm-thread
part: 2
alt_lang: /2026/04/21/mrm-ep2-champion-challenger-en/
next_title: "에피소드 3 — 에이전트 파이프라인의 Chain of Custody"
next_desc: "7개 감사 테이블, HMAC 해시 체인, 그리고 EU AI Act Article 13-14 / KFCPA §17 매핑이 체크리스트가 아니라 코드 경로가 되는 방식."
next_status: draft
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 2편. Ep 1 에서 세운 "구조적 불변성으로서의 MRM" 프레임을
구체적인 한 가지 관문 — Champion-Challenger — 에 적용한다.*

## 교과서 버전

대부분의 조직에서 Champion-Challenger 는 다음과 같이 살아있다:

1. 개발 팀이 신규 모델 (Challenger) 을 학습시킨다.
2. 현 운영 모델 (Champion) 과 병렬로 돌리며 몇 주간 shadow
   traffic 또는 holdout 에서 비교 지표를 쌓는다.
3. MRM 팀이 비교 보고서를 검토한다 — 엑셀, PowerPoint, 혹은 사내
   대시보드의 탭 하나.
4. 위원회가 월례 회의에서 "승격 / 보류 / 기각" 을 의결한다.
5. 승격 승인되면 개발팀이 배포 티켓을 끊어 다음 릴리즈 윈도에 교체한다.

이 구조는 모델이 분기 단위로 교체되는 세계에서는 합리적이다.
승격 판단과 실제 교체 사이의 지연이 리스크라기보다 안전장치로 작동한다.

문제는 지연 자체가 아니라 *판단이 서면으로만 존재한다* 는 점이다.
승격을 결정한 기준, 비교한 지표, 통계적 유의성, 기각된 경우 사유 —
모두 슬라이드와 의사록에 있다. 5년 뒤 감독 당국이 "2026년 4월 23일
challenger 가 왜 승격 기각됐습니까?" 라고 물을 때, 답이 어디 있는지
찾으러 가야 한다. 운이 좋으면 찾고, 운이 나쁘면 담당자가 이직했다.

## 우리가 바꾼 것

우리 파이프라인에서 Champion-Challenger 는 `scripts/submit_pipeline.py`
의 `_decide_promotion()` 함수다. 학습 Job 이 끝나고 모델 레지스트리에
등록되는 순간, 이 함수가 동기적으로 호출된다. 반환값은 없다 — 내부에서
`registry.promote(version)` 을 호출하든 호출하지 않든, *판정이 끝나면
함수는 반환되고 파이프라인은 다음 단계로 간다*. 사람이 개입하는 큐가
없다.

리턴값이 없다는 점이 중요하다. 동기 함수다. 호출 시점이 곧 판정
시점이다. "다음 위원회에서 결정" 이란 상태가 존재하지 않는다.

## 4가지 판정 형태

판정은 항상 네 가지 중 하나로 끝난다 — 그리고 모두 같은 감사 로그에
HMAC 서명으로 기록된다.

**1. `force_promote` — 운영자 오버라이드.**
`--force-promote` CLI 플래그가 붙으면 비교·유의성 검증·fidelity
체크를 모두 건너뛰고 무조건 승격한다. 이게 필요한 이유는 긴급 롤백
시나리오 때문이다. 예를 들어 Champion 에 공정성 위반이 감지됐거나
감독 당국 지적이 들어왔을 때, 대안 모델을 통계적 유의성 여부와 무관하게
*지금 당장* 투입해야 한다. 이때 operator 는 명시적으로 `--force-promote`
를 걸고, 감사 로그에는 `decision="force_promote"`, `trigger="manual"`,
`reason="Operator override via --force-promote"` 가 남는다. 6개월 뒤
누가 봐도 "이건 자동 판정이 아니라 사람이 명시적으로 개입한 순간" 임을
식별할 수 있다.

**2. `bootstrap` — 최초 등록.**
레지스트리에 기존 Champion 이 없으면 비교할 대상이 없다. 부트스트랩
승격으로 기록된다. Challenger 가 Champion 이 된다, 단 감사 로그에는
`decision="bootstrap"` 이 남아 "이 모델은 경쟁을 통해서가 아니라
첫 진입자였기 때문에 운영에 들어갔다" 는 구분이 명시적으로 보존된다.
이 구분이 중요한 이유 — 부트스트랩은 "경쟁 통과" 수준의 보증을 제공하지
않으므로, 후속 Challenger 승격 시점까지는 추가 모니터링이 필요하다는
판단 근거가 된다.

**3. `reject` (fidelity 실패) — 안전 플로어.**
증류된 student 모델이 teacher 대비 특정 태스크에서 fidelity floor 를
위반하면 (학습 메트릭이 아무리 좋아도), 경쟁 단계 *이전에* 기각된다.
이게 경쟁 이전에 오는 것이 설계 의도다. 교과서 버전은 "성능이 더
좋으면 교체" 지만, student-teacher fidelity 는 성능과 독립적 보장
조건이다. 학습 중 어떤 형태로든 distribution shift 가 심하게 일어나서
student 가 teacher 와 다른 모델이 되어버렸다면, avg_auc 가 올라갔어도
운영에 들어가면 안 된다. "성능 좋은 엉뚱한 모델" 을 막는 floor.

**4. `promote` / `reject` (competition) — 정직한 경쟁.**
위 세 가지에 해당하지 않으면 `ModelCompetition.evaluate()` 이
실행된다. 기본 기준은:
- 기본 지표 (avg_auc) 가 0.5% 이상 상승
- 어떤 보조 지표도 2% 이상 하락하지 않음
- t-test 또는 bootstrap 기반 유의성 `p < 0.05`

세 조건이 모두 충족되면 `promotion_approved=True` 를 반환하고 승격된다.
하나라도 실패하면 Challenger 는 레지스트리에 남되 운영에 들어가지
않는다. 감사 로그에는 `decision="reject"` 와 함께 어느 조건에서
실패했는지가 `reason` 필드에 구체적으로 기록된다.

## 왜 안전 플로어가 경쟁 이전에 오는가

이 순서가 설계에서 가장 의식적으로 택한 결정이었다. 자연스러운 본능은
"경쟁을 먼저 돌리고 이긴 놈을 fidelity 체크" 다. 하지만 그 순서는
경쟁 결과를 안다는 사실 자체가 fidelity 판단을 편향시킨다. 성능이
유의하게 좋은 Challenger 라면 "이 정도 fidelity 차이는 용인할 수 있나?"
라는 유혹이 생긴다. 순서를 뒤집어 fidelity 를 먼저 보게 하면, 성능
정보와 독립적으로 floor 가 작동한다. 운영 안전은 경쟁 결과에 의존하지
않는다.

## 감사 엔트리는 부산물이 아니다

네 가지 판정 각각에 대해, `_audit_promotion()` 이 호출되어
`AuditLogger.log_model_promotion()` 이 HMAC 해시 체인 로그에 한
엔트리를 추가한다. 기록되는 필드:

- `champion_version` (직전 Champion, 없으면 None)
- `challenger_version` (이번 Challenger)
- `decision` (`force_promote` / `bootstrap` / `promote` / `reject`)
- `reason` (자동 판정이면 `ModelCompetition` 의 `decision_reason`,
  수동이면 operator 의 설명)
- `comparison` (지표별 Champion/Challenger 값)
- `significance` (t-test p-value, bootstrap p-value)
- `trigger` (`auto` / `manual`)

이 필드 집합이 갖는 불변성이 SR 11-7 Pillar 2 의 "effective challenge"
요구를 *재구성 가능성* 으로 만족시킨다. 감독 당국이 2030년에
"2026-04-23 의 challenger 기각 사유" 를 물으면, 5년 전의 Champion
metrics, Challenger metrics, 유의성 검정 결과, 기각 사유가 모두
그대로 로그에 있다. 담당자가 남아있든 이직했든 무관하다.

`_audit_promotion` 의 구현에서 의도적인 디테일 하나 — 감사 로그 쓰기가
실패해도 승격 자체는 차단하지 않는다. 로컬 fallback 으로 먼저 쓰고
S3 재시도는 비동기로 한다. 이유: 감사 실패가 운영 중단으로 번지면
감사 인프라가 운영의 단일 장애점이 된다. 감사 로그는 "나중에 반드시
복원 가능" 해야지 "지금 반드시 성공" 일 필요는 없다.

## 이게 사주는 것, 그리고 안 사주는 것

사주는 것:

- *어떤 버전이 언제 왜 운영에 들어갔는가* 에 대한 즉시 조회 가능한
  답이 존재
- 승격 결정 품질에 대한 A/B 가능 — 분기 경계 없이 연속 흐름
- 긴급 롤백 경로가 명시적 (force-promote) — 설정 파일 조용히 바꿔서
  교체하는 관행과 차별화
- 부트스트랩을 경쟁 통과와 혼동하지 않음 — 초기 신뢰 부족 상태를
  명시적으로 표식

안 사주는 것:

- *온라인 경쟁* — 실 트래픽 누적에 기반한 A/B 분포 비교는 별도
  트리거 (`ModelMonitor.evaluate_champion_challenger`) 이며 오프라인
  게이트에 포함되지 않는다. 오프라인 게이트는 학습 직후의 메트릭만
  본다.
- *설계 품질 자체의 검증* — 이 게이트는 "현 Champion 보다 나은가" 만
  답한다. "현 Champion 이 애초에 올바른 설계인가" 는 여전히 MRM 감독의
  역할이다. Ep 1 에서 언급한 "아키텍처 자체에 대한 이의 제기" 가 이
  지점에서 필요하다.

두 번째 한계가 중요하다. Champion-Challenger 를 게이트로 만들었다고
해서 MRM 위원회가 없어져도 된다는 뜻이 아니다. 위원회의 역할이
*각 Challenger 를 심사* 하는 것에서 *관문 자체의 설계를 심사* 하는
것으로 이동했을 뿐이다. min\_improvement 를 0.5% 로 잡은 것이 적절한가?
max\_degradation 2% 가 우리 리스크 허용 수준에 맞는가? 이런 메타
질문이 위원회의 새 일이다.

## 다음 편

Ep 3 은 감사 로그의 다른 측면을 다룬다 — 승격 결정만이 아니라
*모든 예측* 이 어떻게 HMAC 체인에 기록되는지, 7개 감사 테이블의
역할 분담, 그리고 EU AI Act Article 13-14 (투명성·인적 감독) 와
KFCPA §17 (금융소비자 분쟁 대응) 매핑이 체크리스트가 아닌 코드
경로가 되는 방식.

원문 자료는 [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
§4-5. 구현은 오픈소스 레포의 `scripts/submit_pipeline.py` 와
`core/evaluation/model_competition.py`, `core/monitoring/audit_logger.py`
에 있다.
