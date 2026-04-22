---
title: "[3개월 개발기] 에피소드 5 — 데이터 무결성 사냥"
date: 2026-05-01 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, data-integrity, leakage, financial-ai]
lang: ko
excerpt: "아키텍처 논쟁 전에 풀어야 했던 것 — label leakage 3건 연쇄 탐지, 18→13 태스크 축소의 결정론적 리키지 배경, 합성데이터 v2→v3→v4 iteration 에서 드러난 자기복제 피처."
series: three-months
part: 5
alt_lang: /2026/05/01/ep5-data-integrity-hunt-en/
next_title: "에피소드 6 — 모든 아키텍처 결정을 압도한 버그"
next_desc: "uncertainty weighting 구현 버그 수정 후 softmax 가 sigmoid 를 앞서는 역전. '훈련 환경의 버그가 아키텍처 결론을 오염시킬 수 있다' 는 방법론적 교훈."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf
source_label: "개발 스토리 (KO, PDF) §9"
---

*"3개월 개발기" 5편. Ep 4 까지는 어떤 아키텍처를 왜 골랐는지를
다뤘다. 이번 편은 그 아키텍처가 의미 있는 숫자를 내놓기 전에
*입력* 이 올바른지 확인하는 이야기 — 피할 수 없는 지루한 작업이
반복 3회 끝에 어떤 모습으로 안정됐는지.*

## 첫 번째 leakage — has_nba 중복 컬럼

ablation v1 이 끝났을 때 AUC 가 의심스럽게 높았다. 보수적으로 잡은
baseline 이 0.68 인데 복잡한 expert 구성이 0.87 을 뱉었다. 이
수준 차이는 알고리즘이 아니라 *데이터* 쪽 문제라는 시그널이다.

원인은 `has_nba` 컬럼이었다. Next Best Action (NBA) 는 우리가
예측하려는 태스크 중 하나 — "이 고객에게 다음에 추천할 상품은
무엇인가". 그런데 원본 CSV 에 이미 `has_nba` 라는 필드가 피처로
들어가 있었다. 타겟에 대한 정답의 일부를 피처로 집어넣은 채 학습한
셈이다.

제거 후 AUC 가 0.87 → 0.71 로 떨어졌다. 발견된 순간엔 *실망* 이지만,
사실은 *안도* 다. 알고리즘이 실제로 한 일을 측정할 수 있게 된
시점이다. 이게 leakage 탐지의 가치 — 숫자를 떨어뜨리지만 그 숫자가
진짜가 된다.

## 두 번째 leakage — ground truth glob 정렬

`has_nba` 수정 후 같은 세션에서 Claude Code 가 다음 문제를 파고
들었다 (Ep 2 에서 "같은 세션에서 연쇄 탐지" 라고 언급한 바로 그
케이스). Ground truth 파일을 glob 으로 불러올 때 alphabetical 순서
로 읽어들였는데, 이 순서가 train/val split 의 경계와 우연히 상관
관계를 가졌다.

구체적으로 — customer ID 가 앞선 파일일수록 신규 고객이 적고, 뒤
파일일수록 최근 가입이 많았다. 이 분포 차이가 validation set 에
체계적 편향을 만들었다. 모델이 "최근 가입 고객은 ABC 상품을 산다"
같은 스푸리어스 규칙을 학습하고, validation 에서 정답률이 높게
나왔다.

해결은 glob 을 *명시적 ID-기반 shuffle* 로 교체. validation 분포가
train 과 같아지자 AUC 가 또 한 번 조정됐다 (0.71 → 0.66). 다시
한 번, 낮은 수치가 *진짜* 다.

## 세 번째 leakage — generator label 입력

같은 세션 말미에 발견된 세 번째. 우리 generator 중 일부 (피처
생성기) 가 *학습 중 label 을 입력으로 받아* piecewise 피처를
만들고 있었다. 예를 들어 특정 generator 가 "고객이 이 세그먼트에
속할 확률" 을 출력하는데, 그 확률 계산이 validation label 을 슬쩍
참조하고 있었다.

원인은 adapter 와 generator 의 관심사 분리가 한 지점에서 깨진
것 (Ep 3 에서 CLAUDE.md §1.2 를 다룬 이유). generator 가 label
컬럼을 dataframe 에서 직접 drop 하지 않고, 자기 필요한 컬럼만
select 하는 구조였는데, 그 select 목록에 실수로 label 이 끼어
있었다.

AUC 0.66 → 0.62 로 또 떨어졌다. 세 번의 연쇄 탐지에서 총 0.25
AUC 가 "환상" 이었다는 게 드러났다. 세션 한 번, 며칠 작업. 각 leakage
를 따로 디버깅했다면 수 주 걸렸을 것이다 — Claude Code 의 1M 컨텍스트
가 "앞선 수정의 맥락을 기억한 상태에서 다음 의심" 으로 이어지는
능력이 결정적이었다.

## 18→13 태스크 축소 — 결정론적 리키지

Leakage 3연쇄 이후에도 일부 태스크는 여전히 의심스러웠다. 특히
`income_tier`, `tenure_stage`, `spend_level`, `engagement_score`
등은 ablation 에서 *어느 구성이든 0.99+ AUC* 를 보였다. 0.99 는
모델이 훌륭한 게 아니라 *태스크 정의 자체가 피처의 결정론적 변환*
임을 뜻한다.

실제로 — `income_tier` 는 `income` 피처를 5개 bin 으로 나눈 결과
였고, `tenure_stage` 는 `tenure_months` 의 6-단계 bucketing 이었다.
모델이 입력에서 label 을 *완벽 복원* 가능하니 AUC 가 1.0 에 수렴.
이런 태스크를 멀티태스크 학습에 남겨두면 (a) 해당 태스크의 loss 가
이질적으로 작아져 다른 태스크에 비해 학습에 기여가 없고 (b) 전이
학습의 평가 지표가 오염됐다.

결론 — 5개 결정론적 태스크 제거. 18 → 13. CLAUDE.md §1.3 에 "피처의
단순 변환으로 파생되는 레이블은 태스크로 사용하지 않는다" 조항이
이 이후 추가됐다. 이후 Ablation 에서 이런 유형은 처음부터 제외된다.

## 합성데이터 v2 → v3 → v4 iteration

실제 데이터 외에 100만 고객 규모 합성데이터로도 병행 실험했다.
그런데 초기 v1 합성데이터로 학습한 모델이 실제 데이터에 transfer
되지 않았다 — 합성 AUC 0.82, 실데이터 AUC 0.54.

v2 에서 피처 분포 매칭을 개선했다. 여전히 transfer 안 됨.
v3 에서 피처 간 상관관계까지 매칭. 일부 개선.
v4 에서 *시계열 의존성* 까지 매칭. 최종적으로 transfer 성공.

v1 → v4 의 차이는 점점 정교해진 통계적 매칭이었지만, 핵심 문제는
하나였다 — 합성 생성기가 *자기 자신을 복제하는* 경향. 초기 v1 은
GAN 기반이었는데, GAN 이 실제 데이터의 "쉬운 부분" 만 학습하고
"어려운 부분" 을 놓쳤다. 모델이 합성에서 배운 건 "현실의 쉬운 패턴" 에
대한 과적합이었다.

v4 는 시계열 의존성을 state-space 모델로 명시적으로 인코딩해서 "어려운
부분" 을 강제로 포함시켰다. 이후 실데이터 transfer 가 성공했다.

## 왜 이걸 먼저 풀어야 했나

아키텍처 논쟁 — HGCN 이 더 좋은가 LightGCN 이 더 좋은가 — 은 입력이
깨끗해야 의미가 있다. 위 사항 중 어느 하나라도 남아있으면 ablation
결과는 "알고리즘 차이" 가 아니라 "어느 알고리즘이 leakage 를 더
많이 exploit 하는가" 를 측정하게 된다. 실데이터에서 재현 안 되는
결과를 paper 에 쓰는 지름길.

*데이터 무결성 사냥은 매력 없는 작업이다.* 새 알고리즘을 시도하는
게 아니라 있는 데이터를 *의심* 하는 일이라, 며칠이 지나도 visible
progress 가 없다. 3인 팀의 시간 압박에서는 "빨리 결과 보고 싶다"
는 유혹과 싸우는 게 가장 어렵다. 이 유혹을 이긴 건 *이전 leakage
발견이 AUC 를 얼마나 떨어뜨렸는지의 기억* 이다 — 0.87 → 0.62 의
추락을 세 번 목격하면 "이번엔 또 뭐가 숨어있을까" 가 기본 자세가
된다.

## 방법론적 함의

이 경험이 CLAUDE.md 에 *리키지 방지 조항* 을 5개 추가하게 만들었다:

1. Scaler 는 TRAIN split 에서만 fit
2. Temporal split 에 gap_days 필수 (최소 7일)
3. 시퀀스 데이터 마지막 timestep 이 label 과 겹치지 않는지 검증
4. `LeakageValidator` 학습 전 강제 호출
5. 피처의 단순 변환으로 파생된 label 은 태스크 아님

이게 Ep 3 의 CLAUDE.md 가 "헌법" 인 이유 — 매 새 실험마다 위 조항
들을 기억하지 않아도 되고, 새 팀원이 들어와도 즉시 적용된다.

## 다음 편

Ep 6 은 데이터 무결성 이후에도 남아있던 문제를 다룬다 — uncertainty
weighting 의 구현 버그와 이 버그가 sigmoid vs softmax gate 아키텍처
결론을 오염시킨 이야기. 버그 수정 후 결론이 뒤집힌다. "훈련 환경의
버그가 아키텍처 결론을 오염시킬 수 있다" 는 방법론적 교훈.

원문 자료:
[개발 스토리 §9 "데이터 무결성 감사"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf).
