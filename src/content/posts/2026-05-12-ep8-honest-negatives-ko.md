---
title: "[3개월 개발기] 에피소드 8 — Honest Negative Results & What Comes Next"
date: 2026-05-12 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, adatt, gradsurgery, negative-results, financial-ai]
lang: ko
excerpt: "3개월의 기록 — adaTT 가 13-task 에서 null 로 수렴한 과정, GradSurgery 가 VRAM 오버헤드로 미채택된 이유, Paper 3 WIP 상태, 2026-04-30 이후 실데이터 메트릭 대기. 작동하지 않은 것이 작동한 것만큼 중요한 이유."
series: three-months
part: 8
alt_lang: /2026/05/12/ep8-honest-negatives-en/
source_url: https://doi.org/10.5281/zenodo.19621884
source_label: "Paper 1 (Zenodo DOI)"
---

*"3개월 개발기" 마지막 편. 7편에 걸쳐 작동한 것들을 다뤘다 — ALS
대체 이유 (Ep 1), AI 협업 조직화 (Ep 2), 가드레일 (Ep 3), 7 전문가
(Ep 4), 데이터 무결성 (Ep 5), 아키텍처 결론 오염 버그 (Ep 6), 증류와
서빙 (Ep 7). 이번 편은 같은 3개월에서 *작동하지 않은 것* 들의 기록.*

## adaTT — 13-task 규모에서의 null 효과

Paper 1 초안에 처음엔 "adaTT 가 13-task 구성에서 PLE 단독 대비
AUC 를 -0.019 떨어뜨린다" 가 들어갔다. adaTT 가 이 규모에서 *구조적
으로 실패* 한다는 해석이었고, 156개 친화도 방향 쌍의 조합론적 불안정
성과 PLE 의 표현 수준 분리가 adaTT 의 loss 수준 재혼합으로 무효화된다는
이중 설명이 붙었다.

Ep 6 에서 짚은 uncertainty weighting 버그가 수정된 후, 같은 실험을
재실행했다. 결과 — adaTT on/off gap 이 -0.019 에서 -0.001 로 이동.
*단일 시드 노이즈 범위* 안이다.

재해석의 결론 — adaTT 는 13-task 규모에서 PLE 를 해치지도 않고, 눈에 띄게 개선하지도 않는다. *null 효과*. 즉 이 데이터·
이 아키텍처 조합에서는 adaTT 를 *쓸 이유가 없다*.

## "구조적 실패" 와 "null 효과" 는 다르다

이 구분이 중요하다. Paper 1 초판의 "구조적 실패" 주장은 *다른 규모
에서도 adaTT 가 작동 안 한다* 고 일반화할 소지가 있었다. "null 효과"
는 그런 일반화를 하지 않는다. 2-4 task 소규모 설정에서 adaTT 가
유용할 수 있고 — 실제 문헌에서도 그런 결과가 있다 — 우리 13-task
이질적 환경에서만 효과가 관측되지 않았다.

Paper 1 v2 에서 이 표현이 바뀌었다. "adaTT at 13-task heterogeneous
MTL: null effect, within single-seed noise. Earlier reported negative
result was an artifact of a contaminating training-environment bug,
now corrected." 앞으로 이 유형의 결과를 보고할 때는 *효과 크기 +
신뢰구간 + 한계 명시* 가 기본.

## GradSurgery — VRAM 오버헤드로 미채택

adaTT 대안으로 실험했던 또 하나 — PCGrad task-type projection
기반 GradSurgery. 13-task 규모에서 adaTT 와 다른 접근으로 task
그래디언트 충돌을 해결하는 방법이다.

실험했다. AUC 차이는 adaTT 와 유사하게 노이즈 범위. 즉 성능 측면
에서 유의한 차이 없음. 그런데 *VRAM 오버헤드* 가 다른 이야기를
했다.

GradSurgery 는 매 step 에서 task 별 gradient 를 저장 후 pairwise
projection 을 수행. 13 task × 2M parameter = 26M gradient copy +
156 pair projection 메모리. RTX 4070 의 12GB VRAM 에서 이 오버헤드
가 batch_size 를 절반으로 제한했다.

즉 *같은 AUC 에 2배 느린 학습*. 채택 기각.

여기서 일반 원칙 — *구현 복잡도나 하드웨어 비용이 성능 향상을 상쇄
하면 기각* 한다. GradSurgery 가 나쁜 알고리즘이라는 게 아니라, 우리
제약 조건에서는 ROI 가 음수였다.

## Paper 3 (Loss Dynamics) — WIP

Ep 6 에서 "Paper 3 (Loss Dynamics) 의 시드 아이디어가 여기서 나왔
다" 고 언급했다. 그 Paper 3 가 현재 WIP 상태다.

핵심 질문은 — *손실 함수의 동역학 자체가 아키텍처 결론을 결정할 수
있는가*. Ep 6 의 sigmoid-softmax 역전 사례가 단일 데이터 포인트
로서 강력하다. Paper 3 는 이 현상을 체계적으로 탐색한다 — 다양한
loss weighting schemes (uniform, uncertainty, GradNorm, DWA) 와
다양한 gate 구조 (softmax, sigmoid, mixture-of-experts) 의 조합
매트릭스에서 아키텍처 결론이 어떻게 흔들리는지.

아직 초록 단계. 실험 설계 완료, 실행 진행 중, 2026년 3분기 Zenodo
업로드 목표. 결과가 나오면 Study Thread 에서 자세히 다룰 예정.

## 2026-04-30 이후 실데이터 메트릭 — 대기 중

이 모든 작업이 최종 검증받는 지점이 실데이터 결과다. 2026-04-30
부터 5 금융기관 파트너로부터 production traffic 위에서의 AUC /
F1-macro / MAE / NDCG / 공정성 지표가 수집되기 시작했다. 현재 글
쓰는 시점 (5월 중순) 기준, 5월 초의 초기 지표가 들어오고 있지만
아직 의미 있는 볼륨은 아니다.

공개할 수 있는 것과 없는 것의 경계는 명확하다. *공개* — 모델 성능
메트릭 (task 별 AUC 등), 공정성 지표 (protected attribute 별 DI),
drift 추이, incident 통계. *비공개* — 고객 세그먼트 분포, 특정
고객군 특성, 파트너 기관 식별 정보, 내부 운영 상세. 후자는 NDA
대상이고, blog 에는 *전자만 게재*.

실데이터 메트릭이 합성데이터에서의 결론을 재현하는가는 열려 있다.
특히 sigmoid-softmax 역전 (Ep 6) 이 실데이터에서도 재현되는지,
7 전문가 중 어느 조합이 실환경에 강건한지. 재현되면 방법론적 주장
이 강화되고, 재현되지 않으면 그 자체로 새 질문이 된다. 어느 쪽이든
Study Thread 또는 Commentary 에서 후속 작성 예정.

## 3개월 동안 시도했지만 기록에 남기지 않은 것들

몇 가지 더. Paper 초안에 들어가지 않았지만 3개월 안에 실제로 시도
된 것:

- **전문가 N=9 구성 시도.** 지속 호몰로지 + 하이퍼볼릭 + 인과 외에
  Gaussian Process expert, Dropout Bayesian expert 를 추가한 9개
  구성. 성능 차이 없음, VRAM 빠듯. 7 유지.
- **Multi-head attention expert.** 추가 expert 로 시도. 파라미터
  폭발 (기존 7 expert 의 2배), AUC 차이 미미. 기각.
- **Task 그룹 3축 대신 2축 분해.** 4개 task group 대신 2개
  (engagement-lifecycle vs. consumption-value) 로 단순화 시도. 일부
  task 성능 개선, 일부 악화. Net zero. 복잡도 감소 이익 없어 유지.
- **Causal expert 의 NOTEARS 대신 DirectLiNGAM.** Linear vs.
  nonlinear 인과 발견 비교. 13-task 규모에서 둘 다 수렴 불안정. 결국
  NOTEARS + recon loss 패치로 안정화 (memory: project_causal_w_
  collapse_fix 참조).

이런 사항들은 paper 에 들어가지 않는다 — *null 이거나 marginal
한 결과를 전부 기록하면 paper 가 팽창하고 핵심 메시지가 희석된다*.
blog 에는 기록 가치가 있다. 같은 경로를 탐색하려는 다른 팀이 "이미
해봤다" 를 알 수 있게 된다.

## 작동하지 않은 것들을 관통하는 공통 스레드

이번 편과 이전 편들에 기록한 negative 들을 돌아보면 — label leakage
3건 (Ep 5), uncertainty weighting 부호 버그 (Ep 6), 초기 "adaTT
구조적 실패" 오해석, GradSurgery 의 VRAM 트레이드오프, 9-전문가·
multi-head attention 시도 실패, softmax-sigmoid 역전, DirectLiNGAM
vs NOTEARS 미결정 — 모두 같은 개발 패턴을 거쳐 드러나 있었다:

1. 초기 결과 (때로 부풀려진) 가 결론처럼 보였다.
2. 그 결론이 정해진 것이라는 전제 위에 후속 작업이 진행됐다.
3. 보통 수 주 뒤 새로운 관측이 재검토를 강요했다.
4. 재검토 시점에서 *원래 추론이 여전히 컨텍스트에 접근 가능* 해서,
   팀은 "왜 예전에 그랬더라" 가 아니라 "이전 결론이 지금은 어떻게
   보이는가" 를 물을 수 있었다.

4단계에서 Claude Code 의 장기 컨텍스트가 가장 결정적으로 작동했다.
이게 없었다면 매 재검토가 노트에서의 부분 재구축이 됐을 것이고,
전제를 하나쯤 놓칠 확률이 높았다. 위 negative 중 여러 개가 "그건
3주 전에 정했던 거" 밑에 묻혀 다시 들여다보지 않았을 것이다.

이건 AI 가 엔지니어를 대체하는 이야기가 아니다. *AI 가 재조사 비용을
엔지니어가 실제로 감당할 수 있을 만큼 낮춘다* 는 이야기다. 3인 팀이
과거 결론을 기꺼이 재검토하는 습관 — 정직함 — 은 인간의 몫이었고,
Claude Code 는 3개월 일정 안에서 그 정직함의 *비용을 낮춰 주었다*.

## 왜 negative results 가 중요한가

ML 연구에서 reproducibility 문제의 상당 부분이 *negative results 가
publish 되지 않아서* 발생한다. 나쁜 결과는 "흥미롭지 않다" 는 이유로
버려지고, 같은 실수가 다른 팀에서 반복된다. 우리 경우엔 uncertainty
weighting 버그 (Ep 6) 가 *누군가 이미 겪었을 가능성이 높은 실수*
였지만, 공개된 "sigmoid 가 이겼는데 나중에 보니 버그 때문" 기록이
없었기 때문에 우리가 처음부터 다시 발견했다.

이 블로그는 그 패턴을 조금이라도 깨려는 시도다. "3개월 간 3인 팀이
시도했지만 작동 안 한 것" 이 다른 팀에게 *몇 주의 시간을 절약* 해
준다면, negative result 를 문서화하는 비용 대비 이익은 충분히 크다.

## 시리즈를 닫으며

8편에 걸쳐 기록한 것 — ALS 시스템 대체 동기, 7 전문가 아키텍처의
도출, AI 협업 방법론, 가드레일 체계, 데이터 무결성 사냥, 훈련 환경
버그, 증류와 서빙, 그리고 이번 편의 negative results.

관통하는 메시지 — *한국 금융권 중소 규모 팀 (3-5명) 이 최신 AI
아키텍처를 production 에 올리는 건 가능하다*. 대형 GPU 클러스터
없이, 전담 MRM 부서 없이, Claude Code 를 파트너로 삼아. 대신 매
단계에서 의사결정의 근거가 있어야 한다 — 왜 이 아키텍처인가, 왜
이 도구인가, 왜 이 검증 단계인가.

3개월 후의 시야에서 — 잘 된 것보다 *실수한 것이 더 많이 보인다*.
uncertainty weighting 부호 버그, 초기 "adaTT 구조적 실패" 오해석,
합성데이터 v1 의 GAN 한계, label leakage 3연쇄. 매 실수가 프로젝트
의 방법론을 한 칸씩 단단하게 만들었다. 전문성은 실수 안 하는 게
아니라 실수를 발견할 수 있는 것 — Ep 6 의 그 문장이 이 시리즈 전체를 관통하는 함축이다.

## 다음 단계

FinAI Build 시리즈는 여기서 8편으로 완결. 이후 블로그의 작업:

- **Study Thread** — 이미 진행 중 (PLE 6편 + adaTT 4편). 이후 Causal
  OT · TDA · Temporal Ensemble · Economics Expert 등 순차 작성.
- **Commentary** — 특정 사건·규제 개정·논문 리뷰에 대한 부정기 글.
  아직 첫 편 없음.
- **실데이터 업데이트** — 2026-04-30 이후 누적된 메트릭의 해석은
  Study Thread 또는 별도 시리즈로.

여기까지 읽어주신 분께 감사. 각 편이 독립적으로 읽히도록 썼으니,
관심사에 따라 골라 돌아와도 됩니다. 질문·지적은
[정선규 ORCID](https://orcid.org/0009-0005-3291-9112) 로.

원문 자료:
[Paper 1 (Zenodo)](https://doi.org/10.5281/zenodo.19621884) +
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052).
코드: [github.com/bluethestyle/aws_ple_for_financial](https://github.com/bluethestyle/aws_ple_for_financial).
