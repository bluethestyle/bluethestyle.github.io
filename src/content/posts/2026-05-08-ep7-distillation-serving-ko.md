---
title: "[3개월 개발기] 에피소드 7 — 증류와 서빙: PLE → LightGBM → Lambda + 5 Bedrock 에이전트"
date: 2026-05-08 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, distillation, serving, lambda, bedrock]
lang: ko
excerpt: "teacher 는 PLE, student 는 task 별 LightGBM, 서빙은 AWS Lambda. 왜 이 조합인가, teacher-student fidelity 가 실패하면 어떻게 되는가, 그리고 Bedrock 위 5-에이전트 파이프라인의 역할 분담."
series: three-months
part: 7
alt_lang: /2026/05/08/ep7-distillation-serving-en/
next_title: "에피소드 8 — Honest Negative Results & What Comes Next"
next_desc: "adaTT null 효과, GradSurgery 기각, Paper 3 WIP, 그리고 2026-04-30 이후 실데이터 메트릭 대기 — 3개월의 작업에서 *작동하지 않은 것* 의 기록."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 (Zenodo DOI)"
---

*"3개월 개발기" 7편. Ep 4-6 이 모델 아키텍처를 정리했다면, 이번 편은
그 모델이 *실제로 고객 앞에 도달하는 경로* — 증류와 서빙. 3인 팀이
어떻게 AWS Lambda 위에서 실시간 금융 추천을 서빙할 수 있는지의 구체
적 이야기.*

## Teacher 는 PLE, Student 는 LightGBM

학습 단계의 모델은 PLE + 7 이종 전문가 + CGC gating + 13 태스크 타워.
파라미터 2M 미만 (Ep 4 참조) 이라 가볍긴 하지만, 서빙에 그대로 쓰기
엔 몇 가지 문제가 있다.

**문제 1 — PyTorch 인퍼런스 런타임.** Lambda 에서 torch 런타임을
초기화하는 콜드 스타트 비용이 2-3초. 단 하나의 요청에 이 시간을 기다
리게 할 수 없다.

**문제 2 — 메모리 footprint.** 7개 전문가 + CGC gate + 태스크 타워
= 전체 모델 로드에 약 150MB. Lambda 에서 동시 실행되는 인스턴스
수가 메모리 예산 때문에 제한됨.

**문제 3 — 해석 가능성.** 고객 문의 또는 감독 당국 감사 시 "왜 이
추천이?" 에 답해야 한다. LightGBM 은 per-tree 기여도 분해가 즉석에서
가능. PLE 의 expert gate 가중치도 해석 가능하지만 LGBM 의 feature
attribution 이 더 단순.

이 세 문제의 공통 답 — **task 별 LightGBM 으로 증류**. Teacher 인
PLE 가 매 태스크에 대해 soft probability 를 생성, Student LGBM 이
이걸 fitting target 으로 학습. 결과 — Lambda 콜드 스타트 300ms
이내, 메모리 30MB, feature attribution 내장.

## Teacher-Student Fidelity Floor

증류 과정에서 가장 중요한 게 *fidelity* — student 가 teacher 와
얼마나 가까운가. Ep 2 의 Champion-Challenger 게이트에서 "fidelity
floor 실패 시 경쟁 이전에 거부" 라고 한 그 fidelity.

측정은 각 태스크별 student-teacher KL divergence. 임계치는 config
(`distillation.fidelity_floor` 기본 0.20). 13 태스크 중 어느 하나
라도 이 임계치를 넘으면 해당 챌린저 전체가 승격 거부.

왜 이 임계치가 중요한가 — *student 의 학습 메트릭이 좋더라도, teacher
와 다른 함수로 학습됐다면 그건 다른 모델이다*. 종래 지식 증류 연구
에서 "student 가 teacher 를 가끔 *능가* 하는 현상" 이 보고되는데,
금융 추천 맥락에서는 이게 위험 신호다. Teacher (PLE) 가 규제·공정성·
해석가능성 조건 하에 설계됐는데, student 가 이 조건을 *우회* 해서
성능을 올린다면 원래 설계 의도가 깨진다.

그래서 *student 의 역할은 teacher 를 가깝게 복제하는 것* 이지 능가
하는 게 아니다. fidelity floor 가 이 원칙을 구조적으로 강제한다.

## Teacher Threshold Gating — 2× Random Baseline

모든 task 가 증류 가능한 건 아니다. teacher 자체 성능이 낮은 task
는 증류해도 student 가 잡음만 학습. `distillation.teacher_threshold`
(기본 2× random baseline) 를 두어, 이 미만이면 해당 task 는 *증류
하지 않고* student LGBM 을 hard label 로 직접 학습 (MRM 안전장치).

예를 들어 binary task 의 random baseline AUC 가 0.5 라면, teacher
AUC 가 1.0 이상이어야 증류 적용. 아니면 hard label 학습 polygon.
이게 프로덕션에서 "teacher 의 noise 까지 모방하는 student" 를 피하는
방법.

## 3계층 서빙 폴백

운영 환경에서 모든 것이 잘 될 수는 없다. 증류 실패, 입력 이상, 모델
파일 corruption 등. 3계층 폴백으로 서비스 중단 없는 구조:

- **Layer 1** — PLE → LGBM 증류 모델 (정상 경로, 99%+ 트래픽)
- **Layer 2** — 증류 실패 시 LGBM 직접 학습 모델 (fidelity floor 통과
  실패한 task 에 사용됨)
- **Layer 3** — 금융 DNA 기반 룰 (모델 전체 실패 시) — 고객의 연령대·
  자산 규모·거래 이력으로 보수적 추천만

Ep 5 에서 Layer 4 (인적 폴백) 도 언급했지만 그건 opt-in. Layer 1-3
은 기본 활성.

이 계층 구조는 SLA 요구사항 ("99.9% 가용성") 을 *아키텍처 수준* 에서
보증한다. 단일 장애점이 존재하지 않음.

## AWS Lambda + 5 Bedrock 에이전트

서빙 스택 전체 구성:

```
Client (은행 영업점 시스템)
  ↓
API Gateway
  ↓
Lambda (Python 3.11, 1GB memory)
  ↓
  ├─ LightGBM 모델 로드 (3계층 폴백)
  ├─ Feature Selector 에이전트 (Bedrock Sonnet)
  ├─ Reason Generator 에이전트 (Bedrock Sonnet)
  ├─ Safety Gate 에이전트 (Bedrock Sonnet)
  └─ 예측 반환 + Ep 3 감사 로그 기록

비동기 경로:
  OpsAgent (Bedrock Sonnet) — CloudWatch 로그 해석
  AuditAgent (Bedrock Haiku) — 야간 hash chain 검증
```

5 에이전트의 역할 분담:

**1. Feature Selector (Sonnet)** — LightGBM 이 반환한 feature
attribution 중 *고객 설명에 가치 있는 것* 을 선별. 예를 들어 "최근
해외 직구 3회" 는 의미 있는 근거지만 "피처 #247" 은 의미 없음. 모델
내부의 raw attribution 을 business-meaningful set 으로 변환하는 단계.

**2. Reason Generator (Sonnet)** — 선별된 피처를 *은행 영업점 직원이
고객에게 읽어줄 수 있는* 자연스러운 금융 경어체 한국어로 재작성.
"고객님의 최근 소비 패턴 중 해외 결제 비중이 증가하여 외화 예금
상품을 추천드립니다" 같은 문장. 단순 template 이 아니라 고객 컨텍스트
에 맞춰 톤 조정.

**3. Safety Gate (Sonnet)** — 생성된 설명이 (a) 규제 요건 (광고 제한어
없음), (b) 적합성 (고객 리스크 허용도 이내), (c) 환각 없음 (실제
피처 값과 일치), (d) 어조 적절 (고압적·과장 표현 없음), (e) 사실성
(상품명·조건이 내부 DB 와 일치) 다섯 기준 통과하는지 검증. Lambda
핸들러를 떠나기 전 마지막 관문.

**4. OpsAgent (Sonnet)** — 비동기, 야간 또는 온콜 트리거. 드리프트
모니터·공정성 모니터의 출력을 해석하여 SRE/MLOps/Biz 관점 요약 제공.
kill switch 자동 발동 시 incident ticket 초안 작성.

**5. AuditAgent (Haiku)** — Ep 3 에서 언급. 매일 야간 HMAC hash chain
검증. 가벼운 연산이라 Haiku 로 충분.

## Ops vs Audit 분리 — 왜 두 에이전트인가

처음엔 OpsAgent 와 AuditAgent 를 하나로 합치려 했다. 둘 다 "로그를
읽는" 역할이니까.

합치지 않은 이유 — *독자가 다르다*. OpsAgent 의 출력은 SRE·MLOps·
비즈니스 팀이 읽는다 — "성능 드리프트가 임계치 근접" 같은 운영
관점. AuditAgent 의 출력은 규제 대응·리스크 팀이 읽는다 — "hash
chain 엔트리 N번에서 검증 실패" 같은 감사 관점.

더 중요한 건 *신뢰 경계* 가 다르다는 점. OpsAgent 는 로그를 *읽고
해석* 하지만, AuditAgent 는 로그를 *신뢰하지 않고 재검증* 한다 (Ep 3).
이 차이가 에이전트의 prompt design 자체에 녹아있다. 하나로 합치면
둘 중 하나의 역할이 희석된다.

## 이 코드는 실제로 어떻게 쓰여졌나

위 서빙 스택은 AWS 전문가 팀이 출시한 것처럼 읽힌다. 우리 경우
그 전문가 작업은 *Claude Code 세션 안에서* 일어났다.

Lambda `handler.py` 는 약 30 회 iteration 을 거쳤다. v1 은 60줄
짜리 Claude Code 초안 — LightGBM 모델 로드 + 예측 반환만. v30 은
400줄 — 3계층 폴백, 감사 로깅, Bedrock 에이전트 오케스트레이션,
결정론적 콜드 스타트 warming 까지 포함. 그 사이의 버전들은 실제
production 실패 모드로 추동됐다 — 어느 날의 타임아웃, 다른 날의
IAM 권한 거부, 엣지 케이스 스키마 변경 — 각각이 하나의 Claude Code
대화로 들어와 PR 로 종결됐다.

3계층 폴백은 미리 설계된 게 아니다. 처음 2주간은 Layer 1 만 있었다.
Layer 2 는 `task_churn` 의 증류 실패 (fidelity floor 위반, Ep 2 의
거부 경로) 이후 추가 — 증류 모델 없이도 이 task 를 서빙할 경로가
필요했다. Layer 3 는 더 큰 incident 이후 — Phase 0 스키마 변경이
잘못 전파되어 LGBM 로드 자체가 오후 내내 실패한 적이 있었고, pure
rule 기반 폴백이 있으면 고객은 보수적이라도 추천을 받는다. 각 계층은
특정 과거 실패의 화석이다.

5-에이전트 오케스트레이션도 점진적으로 자랐다. Feature Selector 와
Reason Generator 가 먼저 — Paper 2 의 "고객에게 추천 근거 설명"
요구에서 직접 도출. Safety Gate 는 출시 전 review 에서 엔지니어가
(AI 도움으로) Reason Generator 가 사실 오류 텍스트를 뽑는 40개
엣지 케이스를 찾아낸 뒤 추가됐다. Safety Gate 의 규칙은 그 review
결과에서 나왔다. OpsAgent 는 새벽 3시 알람을 수동 triage 하는 데
40분 걸린 어느 날 이후 추가. AuditAgent 는 마지막, Ep 3 chain-of-
custody 요구사항과 연결.

이 중 어느 것도 선형 계획이 아니었다. 각 추가가 관측된 특정 문제를
해결했고, Claude Code 가 구현을 쓰는 동안 Opus 가 설계를 논했다.
이 패턴 — 관측 → Opus 대화 → Claude Code 구현 → 프로덕션 → 다음
관측 — 이 3개월·3인 작업으로 이 스택을 만들어낸 방식이다.

## Lambda 비용 — 서버리스가 답인 이유

RTX 4070 로 학습한 모델을 24/7 서빙하려면 전용 서버·GPU 가 필요하다.
3인 팀의 예산으로는 무리. Lambda + LightGBM 이 다른 그림을 그린다:

- 요청 건당 과금 (RPS 낮으면 비용 거의 0)
- 콜드 스타트 300ms (LightGBM 로드) — 대부분의 금융 추천 SLA 안
- 동시 실행 제한을 config 로 조절해서 비용 상한 설정

예상 비용 (월 100만 추천 기준) — Lambda $15 + API Gateway $3 +
S3/CloudWatch $10 = 월 $30 미만. 전용 서버 대비 1/100 수준.

이게 *소비자용 GPU + 서버리스* 조합이 한국 금융권 중소 규모 팀에
접근 가능한 구조인 이유. 학습은 한 번, 서빙은 저비용 영구. 대규모
인프라 없이 production AI 가 가능.

## 다음 편

Ep 8 (FinAI Build 최종편) — 이 3개월에서 *작동하지 않은 것* 의 기록.
adaTT 가 13-task 규모에서 null 효과로 기각된 과정, GradSurgery 가
VRAM 문제로 미채택된 이유, Paper 3 (Loss Dynamics) 가 WIP 상태인
이유, 그리고 2026-04-30 이후 실데이터 메트릭 대기 중인 현재 상태.
honest negative results 가 프로젝트의 중요 부분인 이유로 마무리.

원문 자료: [Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052)
§3 "서빙 아키텍처" + §8 "에이전트 설계". 구현은
`core/distillation/`, `aws/lambda/handler.py`,
`core/agents/` 하위 모듈.
