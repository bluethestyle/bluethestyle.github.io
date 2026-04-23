---
title: "[3개월 개발기] 에피소드 6 — 모든 아키텍처 결정을 압도한 버그"
date: 2026-05-05 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, debugging, methodology, financial-ai]
lang: ko
excerpt: "몇 주 동안 sigmoid gate 가 softmax 를 이기는 것처럼 보였다. uncertainty weighting 구현 버그가 수정되자 결과가 뒤집혔다. 훈련 환경의 버그가 어떻게 아키텍처 결론을 오염시키는가의 사례 연구."
series: three-months
part: 6
alt_lang: /2026/05/05/ep6-uncertainty-weighting-bug-en/
next_title: "에피소드 7 — 증류와 서빙: PLE → LightGBM → Lambda + 5 Bedrock 에이전트"
next_desc: "teacher-student fidelity, 왜 서빙은 LGBM 인가, serverless 비용 프로필, 그리고 AWS Bedrock 위 5-에이전트 파이프라인 구성."
next_status: published
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf
source_label: "개발 스토리 (KO, PDF) §11"
---

*"3개월 개발기" 6편. Ep 5 에서 데이터 무결성을 정리한 뒤에도 남아
있던 문제 — 모델 훈련 환경 자체의 버그가 몇 주 동안 우리가 내린
아키텍처 결론을 *조용히* 오염시키고 있었다. 이 편은 그 발견의
이야기와 방법론적 교훈.*

## 거의 결론이라고 생각했던 것

Ep 2 에서 sigmoid gate 발견 과정을 짧게 짚었다. PLE val_loss 가
수렴하지 않는 문제 → Opus 와의 대화 → NeurIPS 2024 sigmoid gate
논문 → 구현. 실험 결과도 sigmoid 가 softmax 를 일관되게 앞섰다.

일관된 결과란 강력한 증거다. 다섯 번의 ablation run, 서로 다른 seed,
서로 다른 데이터 split — 모두 sigmoid 가 NDCG 와 F1-macro 에서
softmax 를 0.02-0.04 앞섰다. 이 시점에서 우리는 *결론을 문서화* 하고
다음 아키텍처 결정 (adaTT 설계 등) 에 이 결과를 전제로 썼다.

결론을 쓴 다음 이어지는 작업들이 있다. Paper 1 초안에 "sigmoid gate
가 이종 전문가 아키텍처에 적합하다" 가 들어갔다. adaTT 설계가
sigmoid gate 를 전제로 진행됐다. "이건 이미 정해진 것" 이라는
전제가 후속 결정 여러 개에 전파됐다.

## 몇 주 뒤, uncertainty weighting 버그

adaTT 구현 중에 엔지니어 2 팀이 uncertainty weighting (Kendall
et al., 2018) 코드를 점검하다가 버그를 발견했다. 버그 자체는 작다 —
task loss 에 학습 가능한 $\log \sigma^2$ 를 곱하는 공식에서 부호
하나가 반대로 구현되어 있었다. 수식은 다음이어야 했다:

$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

구현은 다음이었다:

$$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t - \log \sigma_t$$

부호 하나. `+ log_sigma` 가 `- log_sigma` 로 되어 있었다. 이 부호가
뒤집히면 regularization 항이 정반대로 작동해서, 학습이 *$\sigma_t$
를 키우는* 방향이 아니라 *작게 만드는* 방향으로 밀린다. 결과 —
binary 분류처럼 loss 가 본래 작은 태스크의 $\sigma_t$ 가 극단적으로
작아지면서, 그 태스크의 effective weight 가 다른 태스크를 압도.

즉 uncertainty weighting 이 *실질적으로 binary 태스크를 과가중*
하는 버그가 수 주 동안 작동하고 있었다.

## 수정 후 결과가 뒤집혔다

버그를 고쳐서 5개 ablation run 을 다시 돌렸다. 결과 —
*softmax 가 sigmoid 를 NDCG 에서 앞섰다*.

한 run 에서 0.02 앞섰다. 두 run 에서 0.03. 세 run 에서 0.01.
나머지 두 run 에서는 비슷. 방향이 완전히 뒤집혔다.

처음엔 측정 오차라고 생각했다. seed 를 바꿔 10회 더 돌렸다. 일관된
방향. sigmoid 가 우월하던 시기의 "일관성" 이 *다른* 일관성으로 대체된
것이다.

## 왜 이런 일이 일어났나 — 근본 원인

돌아가서 분석해 보니 논리는 깔끔했다.

**깨진 uncertainty weighting 환경에서:** 13개 태스크가 효과적으로
균등 가중되지 않았다. binary 분류 7개 (수가 많음) 의 gradient 가
multiclass 3개, regression 3개의 gradient 를 압도. 이런 조건에서
softmax gate 의 경쟁적 라우팅은 *binary 태스크 한 쪽에 전문가 용량
을 몰아주는* 방향으로 작동하여, 그 외 태스크 유형이 잠식됐다. sigmoid
의 비경쟁적 라우팅은 *모든 전문가를 활성* 으로 유지해서, 잠식을
막는 *우연한* 방화벽 역할을 했다.

**올바른 uncertainty weighting 환경에서:** multiclass 와 regression
gradient 가 의도한 크기를 회복. 이제 softmax 의 경쟁적 라우팅이
*전문가 전문화를 강제* 하여, 태스크 유형 간 gradient 오염의 *구조적
장벽* 으로 작동. 반대로 sigmoid 는 오히려 장벽 없이 gradient 를
섞어서 각 태스크 유형에 덜 특화된 결과.

즉 결과의 방향 자체가 *훈련 조건에 의존* 한다. "sigmoid 가 더 낫다"
는 결론은 아키텍처적 우위가 아니라, 깨진 환경에 대한 *유효한 적응* 이었다.

## 동질적 MTL vs 이질적 MTL — 문헌의 함정

NeurIPS 2024 sigmoid gate 논문은 2-4 태스크, 동일 태스크 유형의
세팅에서 sigmoid 의 우위를 보였다. 이 논문이 가정한 *동질적 MTL*
환경에서는 경쟁적 softmax 가 구조적으로 유사한 전문가 간 붕괴를
일으키기 때문에 sigmoid 가 실제로 우월하다.

우리 프로젝트는 *13 태스크, 3 태스크 유형 (binary · multiclass
· regression)* 의 이질적 MTL 이다. 이 레짐에서는 경쟁적 라우팅이
붕괴를 일으키지 않고, 오히려 태스크 유형 간 gradient 오염에 대한
*구조적 장벽* 으로 기능한다. 문헌 결과가 *전이되지 않는* 경계 조건
이다.

이 교훈은 단순하다 — *레퍼런스 논문의 실험 조건이 자기 프로젝트와
일치하는지 확인하지 않으면, 옳은 결론을 옳지 않은 환경에 이식하게
된다*. 우리 경우엔 추가로 훈련 환경 버그까지 겹쳐 이중 오염이었다.

## 발견을 가능하게 한 것

이 흐름이 발견된 과정 자체가 Ep 2 에서 "Claude Code 대체 불가능성"
의 가장 좋은 예다. 몇 주 전의 sigmoid 채택 맥락 — 왜 그 논문을
찾았는지, 실험 결과가 어떻게 나왔는지, 어떤 후속 결정에 전제로
들어갔는지 — 이 *여전히 직접 확인 가능한 상태* 에서 새 증거 (uncertainty
fix 이후 성능 역전) 를 만났다. 그 즉시 "우리가 그때 내린 결론의 근거
가 지금은 어떻게 보이는가" 를 재검토하는 흐름으로 이어졌다. 새 세션
에서 "예전에 왜 sigmoid 를 골랐더라" 부터 재구축해야 했다면, 재조사
자체가 시작되지 않았을 가능성이 높다.

## 방법론적 교훈

이 에피소드가 우리 팀 방법론에 남긴 것:

**1. 모든 아키텍처 결론에 "훈련 조건이 안정적이었는가" 체크가 붙는다.**
"결론" 이라고 문서화하기 전에 loss weighting, scaler state, label
alignment, scheduler configuration 등 훈련 환경 요소가 depend 하지
않는지 점검. 점검 체크리스트가 CLAUDE.md 에 추가됐다.

**2. Paper 초안 수정 폭이 커졌다.** Paper 1 의 "sigmoid gate 가
이종 전문가 구조에 적합하다" 섹션이 전면 수정되어 "경계 조건에
따라 결과가 뒤집힐 수 있음" 이 명시됐다. Paper 3 (Loss Dynamics)
의 시드 아이디어가 여기서 나왔다.

**3. adaTT null 결과의 재해석.** Ep 8 에서 자세히 다루겠지만, 초기
에 "adaTT 가 13-task 에서 PLE 대비 -0.019 AUC" 로 보고됐던 결과도
이 버그의 영향 아래서 측정된 것. 버그 수정 후 adaTT on/off gap 은
-0.001 로 노이즈 범위. 즉 adaTT 가 "구조적으로 실패" 한 게 아니라
그때의 훈련 환경 자체가 오염되어 있었다.

**4. 전문성은 실수 안 하는 것이 아니라 실수를 발견할 수 있는 것
이다.** 이 문장이 프로젝트 CLAUDE.md §1 에 들어갔다. 부호 버그
하나를 몇 주 동안 놓친 게 전문성 부족이 아니다. 그 뒤의 *재조사
흐름* 이 전문성의 증거다.

## 남은 질문들

- 다른 "일관된 결과" 도 훈련 환경 버그의 부산물은 아닌가? — Paper
  1 의 나머지 결과도 재검증이 필요했고, 상당 부분 수정됐다.
- 실데이터에서 같은 역전이 일어날까? — 2026-04-30 이후 실데이터
  메트릭 대기 중. 만약 실데이터에서도 같은 역전이 재현되면 방법론적
  교훈이 강화되고, 재현되지 않으면 그 자체로 새 질문.
- uncertainty weighting 구현이 견고해졌는가? — 부호 단위 테스트 + 
  σ 값 모니터링이 CI 에 추가됐다. 같은 버그는 다시 들어오지 않는다.

## 다음 편

Ep 7 은 아키텍처가 안정된 뒤의 작업 — 증류 (PLE teacher → LGBM
student) 와 서빙 (AWS Lambda + 5-에이전트 Bedrock 파이프라인). 왜
LGBM 으로 증류하는가, serverless 비용 구조, 그리고 Feature Selector
· Reason Generator · Safety Gate · OpsAgent · AuditAgent 5 에이전트
의 역할 분담.

원문 자료:
[개발 스토리 §11 "모든 아키텍처 결정을 압도한 버그"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf).
