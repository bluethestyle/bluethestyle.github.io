---
title: "[MRM 스레드] 에피소드 5 — 티켓 큐가 아니라 API 인 Human Oversight"
date: 2026-05-01 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, human-oversight, kill-switch, regulation, financial-ai]
lang: ko
excerpt: "EU AI Act Article 14 의 인간 감독 요구를 티켓 큐가 아니라 API 엔드포인트로 구현한 방식 — kill switch, HumanReviewQueue tier 2/3, 그리고 auto_promote=false 가 production posture 로 강제되는 이유."
series: mrm-thread
part: 5
alt_lang: /2026/05/01/mrm-ep5-human-oversight-en/
next_title: "에피소드 6 — Fairness as a Production Path"
next_desc: "5개 보호 속성에 대한 Disparate Impact · Statistical Parity · Equal Opportunity 가 검증 샘플이 아닌 프로덕션 스트림에서 실시간 계산되는 구조와 Counterfactual C-C 의 역할."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"MRM 스레드" 5편. Ep 4 에서 FRIA 의 "인적 감독" 차원을 언급만 했다.
이번 편은 그 차원이 실제로 어떤 코드로 구현됐는지를 펼친다 — EU AI
Act Article 14 의 human oversight 가 티켓 큐가 아니라 API 엔드포인트
로 살아있는 구조.*

## 공정성 incident 의 응답 창

파트너 기관과의 실 트래픽 수집은 2026-04-30 에 시작됐으니, 아직
실제 공정성 incident 를 프로덕션에서 처리해 본 건 아니다. 다만
설계 의도는 명시적이다 — *on-call 엔지니어의 판단 창이 탐지 지연보다
중요하다*, 그리고 아키텍처는 그 창을 좁히는 방향으로 맞춰져 있다.

시스템이 대응하도록 설계된 incident 의 모양을 생각해 보자. 공정성
모니터에서 보호군 위반이 드러난다 — 예를 들어 65세 이상 연령층의
Disparate Impact 가 적금 상품 추천 task 에서 금소법 §17 적합성 임계치
아래로 떨어진다. 중견 금융기관의 소규모 ML 팀 관점에서 세 응답
경로가 있다: (1) 다음 예정 재학습까지 기다림 (보통 4-7일 뒤), (2)
인적 리뷰 backlog 로 라우팅 (한국 IT 운영 캘린더 기준 건강한 편이면
12-24시간), (3) 지금 개입.

앞 두 가지는 받아들일 수 없다 — 금감원 사고 보고 의무가 *중대 사고는
24시간 내 유선 보고, 72시간 내 서면 보고* 를 요구하기 때문. 5일을
기다리는 건 응답이 아니라 2차 사고다.

그럼 "지금 개입" 이 실제로 어떤 모습인가. 대부분 한국 금융기관에서는
incident 티켓을 큐에 넣고 on-call 인력이 업무시간에 triage 한다. 평균
해결 시간은 *일 단위*. 우리 아키텍처에서는 on-call 엔지니어의 API 호출
한 번 — 설계 선택이지 필연이 아니다.

## 조직 프로세스와 API 엔드포인트의 차이

EU AI Act Article 14 — "고위험 AI 시스템은 이용자가 효과적으로 감독할
수 있는 수단을 제공해야 한다." "효과적 감독 수단" 이 무엇인가에 대해
규제가 구체 형식을 강제하지는 않는다. 대부분 금융기관의 답은 *조직
프로세스* 다 — "매월 리스크관리위원회 또는 담당 임원 결재선에서 모니터링 출력을 검토합니다",
"이상 탐지 시 담당자가 이메일로 보고받습니다", "긴급 상황 시 JIRA
티켓으로 관리합니다."

위 모든 답이 Article 14 *문자상* 요건을 충족한다. 그런데 실제 사건이
터졌을 때 작동하는가 — 이게 다른 질문이다. 이메일을 담당자가 제 때
읽는가? JIRA 큐가 5일 backlog 가 있지는 않은가? 월례 위원회 일정이
밀려 3주 전 사건을 이제야 다루고 있지는 않은가?

우리 접근은 감독 수단을 *API 엔드포인트* 로 제공하는 것이다. 프로세스
가 아니라 *호출* 이다. 호출은 실시간이고 멱등하다. 큐에 묻혀 사라지지
않는다.

## Kill Switch — 한 줄 호출로 특정 태스크 비활성화

가장 단순한 API 는 kill switch. 파이프라인 전체 비활성화, 특정
태스크 (13 중 하나) 비활성화, 특정 모델 버전 roll back — 각각 한
번의 API 호출.

```
POST /admin/kill-switch
{
  "scope": "task",
  "target": "churn_signal",
  "reason": "DI breach in age 60+ segment",
  "operator": "engineer_id"
}
```

이 한 호출이 수행하는 것:
- 해당 task 서빙 경로 즉시 block (Lambda 환경 변수 업데이트)
- `log_operation` 에 `event="kill_switch_fired"` 엔트리 기록
- Slack 채널 + PagerDuty 동시 알림
- OpsAgent 가 자동으로 incident ticket 생성

*조직 프로세스로 구현하면 불가능한 것* — 새벽 3시에 공정성 모니터가
경고 울렸을 때, JIRA 큐의 티켓이 기다리지 않고 on-call 엔지니어가
위 API 를 호출한다. 행동 시점과 결정 시점이 일치. Ticket backlog
가 낳는 "알고는 있었는데 처리가 밀렸다" 시나리오가 구조적으로 불가능.

## HumanReviewQueue — Tier 2/3 자동 상위 보고

Kill switch 는 "이미 뭔가 잘못됐을 때" 의 도구다. 그 전 단계 — *의심
수준의 케이스를 사람에게 자동으로 넘기는* 메커니즘이 HumanReviewQueue.

우리 구현은 3-tier:

- **Tier 1** — 자동 승인. 모델의 신뢰도 (gate 엔트로피 낮음) + 공정성
  지표 내 + 적합성 필터 통과. 대부분의 예측이 여기.
- **Tier 2** — 인적 검토 필요. 공정성 마진이 임계치에 근접, 또는
  신뢰도 낮음, 또는 고령/저소득 보호 대상 고객. 큐에 적재되어 시간
  내 (업무일 기준 24시간) 리뷰어 처리.
- **Tier 3** — 즉시 차단 + 즉시 인적 개입. 명확한 공정성 위반 signal,
  환각 탐지, 규제 키워드 (내부자 정보, 자본 금융 상품 광고 제한어 등)
  등장. 고객에게 전달되지 않고 인적 판정 전까지 보류.

Tier 를 *자동으로 판정* 하는 게 핵심이다. 사람이 "이건 tier 2 야"
하고 분류하지 않는다 — 위 세 조건 중 하나가 맞으면 tier 가 자동으로 올라간다.
즉 tier 체계는 규칙 집합이고, 그 규칙 자체가 리스크관리위원회 또는 담당 임원 결재선의 심사
대상이 된다.

## auto_promote=false — Production Posture

Ep 2 에서 다룬 승격 게이트는 모든 조건 통과 시 자동 승격한다.
*기술적으로는*. 하지만 production 에서는 자동 승격 비활성 설정이
강제되어 있어서, 모든 챌린저가 metric 을 통과해도 자동 승격되지 않고
운영자의 force-promote override 없이는 현 챔피언이 유지된다.

이게 왜 필요한가 — *자동 승격이 가능한 조건* 은 기술적으로 모든
gate 가 구조적으로 보장된다는 가정 위에 있다. 현실에서는 그
가정 자체가 인간 판단의 대상이다. "이번 챌린저가 진짜 production
에 들어갈 준비가 됐는가" 는 metric 하나로 답해지지 않는다. 2026
상반기 기준, 우리 team 은 매 승격 결정을 *수동 사인오프* 로 게이트
한다.

이게 EU AI Act Article 14 의 *meaningful human oversight* 조항과
연결된다. 자동화된 승격은 "the system decides" 쪽에 가깝고, 수동
승격은 "a human decides" 쪽에 가깝다. 실데이터 메트릭 안정화되기
전까지는 후자 posture 가 적절하다는 판단.

## Layer 4 — Opt-in Human Fallback

서빙 경로에는 3 계층 폴백 라우터가 있다:

- **Layer 1** — PLE → LGBM 증류 모델 (일반 경로, 99%+ 트래픽)
- **Layer 2** — 증류 실패 시 LGBM 직접 학습 모델
- **Layer 3** — 금융 DNA 기반 룰 (모델 전체 실패 시)

2026 상반기에 Layer 4 가 추가됐다 — **인적 폴백**. config 의 tier-3
human-fallback 플래그가 켜져 있을 때만 활성. Layer 1-3 모두 신뢰도
낮을 때, 추천을 *생성하지 않고* 고객에게 "담당자에게 연결해드립니다"
메시지 제공.

Layer 4 가 opt-in 인 이유 — 모든 금융기관이 인적 리소스를 거기 투입
할 수 있는 건 아니다. 옵션으로 열어두되 default off. 활성화한 기관은
인적 리뷰 큐를 별도 운영해야 한다.

## Article 14 를 문서가 아닌 행동으로 답한다

금감원 감사관이 Article 14 준수 증거를 요청하면:

- kill switch API 호출 이력 → `log_operation` 테이블
- HumanReviewQueue tier 2/3 처리 이력 → 별도 테이블
- 승격 수동 사인오프 이력 → `log_model_promotion` 의 `trigger=manual`
  엔트리
- Layer 4 활성화 시, 인적 폴백 경로의 호출 횟수 → 별도 집계

각 기록이 실제 API 호출 또는 실제 이벤트의 부산물이다. "우리는 이런
프로세스를 *갖추고 있습니다*" 가 아니라 "우리는 이런 호출을 지난
12개월간 N번 *했습니다*" 로 답한다. 후자가 감사 대응력이 훨씬 높다.

## 여전히 사람의 일

API 엔드포인트를 제공하는 것과 *사람이 그 API 를 실제로 호출하는
것* 은 다르다. kill switch API 가 있어도, 새벽 3시에 on-call
엔지니어가 공정성 알람을 읽고 판단하지 않으면 무용지물.

그래서 "API 기반 Human Oversight" 는 자동화 논증이 아니라 *인적
개입의 장벽을 낮추는* 논증이다. 티켓 큐를 통과하는 대신 한 API 호출
이면 되도록 설계해서, on-call 엔지니어의 판단이 24시간 대기 없이
즉시 행동으로 옮겨진다. 그 대신 *판단 자체* 의 질은 여전히 인간
결정이다.

리스크관리위원회 또는 담당 임원 결재선의 역할도 바뀐다. 월례 회의에서 리포트를 검토하는 게 아니라
API 호출 기록을 보며 "이 kill switch 발동은 적절했나", "tier 2/3
비율이 건강한가", "수동 승격 거부 건수가 과다한가" 를 판정한다.
*검토 대상이 리포트에서 호출 로그로* 바뀐 것이다.

## 다음 편

Ep 6 은 Article 14 의 사촌 — Disparate Impact / Statistical Parity
/ Equal Opportunity 를 *프로덕션 스트림 위에서 실시간* 계산하는
공정성 아키텍처. 검증 샘플에서 한 번 측정하는 게 아니라 매 요청마다
측정하는 구조, 그리고 Counterfactual Champion-Challenger 가 여기서
어떻게 공정성 판단에 쓰이는지.

원문 자료: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
§7 "Human-in-the-Loop 설계", 구현은 [오픈소스 레포](https://github.com/bluethestyle/aws_ple_for_financial).
