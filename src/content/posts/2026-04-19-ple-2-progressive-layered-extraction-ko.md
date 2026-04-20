---
title: "[Study Thread] PLE-2 — Progressive Layered Extraction: 명시적 전문가 분리와 CGC 게이트"
date: 2026-04-19 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, tang2020, mtl]
lang: ko
series: study-thread
part: 2
alt_lang: /2026/04/19/ple-2-progressive-layered-extraction-en/
next_title: "PLE-3 — 입력 구조와 이종 Shared Expert Pool (512D)"
next_desc: "PLEClusterInput 의 전체 필드 사양과 734D features 텐서 인덱스 매핑, HMM 모드 라우팅. 그리고 본 프로젝트의 7개 이종 Shared Expert (EconomicsExpert·TemporalExpert·HMMExpert·TDAExpert·GMMExpert·GCNExpert·PersLayExpert·UnifiedHGCN) 가 각자 어떤 수학적 관점으로 고객을 해석하는지까지."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 2편. 영문/국문 병렬로 PLE-1 →
PLE-6 에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를
정리한다. 출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이다.
이번 2편은 PLE-1 이 끝난 지점 — MMoE 의 Expert Collapse — 에서
시작해서, 거기에 대한 응답으로 세 가지 연쇄적 결정이 어떻게 나왔는지를
따라간다.*

## PLE-1 이 남긴 문제

MMoE 는 Shared-Bottom 의 gradient 충돌에 대한 대답이었다. 공유 trunk
하나로 모든 태스크를 끌고 가는 대신 Expert 를 여러 개 두고, 태스크별
gate 가 어느 Expert 를 얼마만큼 쓸지를 학습하게 한다. 이론적으로는
충돌이 해소된다 — 태스크 $i$ 가 Expert A 를 많이 쓰고 태스크 $j$ 가
Expert B 를 많이 쓰면 서로를 방해하지 않는다.

실제로는 그렇게 흘러가지 않았다. 학습이 시작되면 Expert 들 중 하나가
처음 몇 스텝 안에 다른 것들보다 더 큰 gradient 를 받는다 — 파라미터
초기화, 입력 분포, 태스크 손실의 크기가 조금씩 다르기 때문에. 그
Expert 가 약간 더 유용해지면 모든 태스크의 gate 가 그쪽으로 조금씩
기울고, 그 Expert 가 더 많은 gradient 를 받고, 점점 더 유용해진다.
승자독식의 양의 피드백 루프 — *Expert Collapse* — 다. 결국 모든 태스크가
같은 Expert 를 쓰고, 다른 Expert 들의 파라미터는 거의 업데이트되지
않는다. Shared-Bottom 으로 다시 수렴한 것이다.

문제의 진단은 분명하다. Expert pool 이 *대칭적* 이다. 모든 Expert 가
같은 구조, 같은 입력, 같은 초기화를 공유한다. 이 대칭 안에서 gate
하나에게 "분업을 학습해" 라고 요구하는 것은 사실상 "똑같은 직원 7명
중에서 각자 다른 일을 하라" 라고 말하는 것이다. 첫 번째 사람이 조금
더 잘 한다고 알려지면 모두가 그 사람에게 가서 일을 시킨다.

PLE 의 답은 두 단계다. 대칭을 아키텍처 수준에서 깨는 것, 그리고 그
위에 제약된 gate 를 얹는 것.

## 결정 1 — 공용과 전용 Expert 를 명시적으로 분리한다

Tang, Liu, Zhao & Gong (RecSys 2020) 은 PLE 논문에서 이렇게 말한다.
"gate 가 태스크별 분업을 학습하기를 기대하지 말고, 분업 자체를
아키텍처에 내장하자." 구체적으로 Expert 를 두 종류로 나눈다.

- *Shared Expert* ($\mathcal{E}^s$): 모든 태스크가 접근 가능한 공용 Expert
- *Task-specific Expert* ($\mathcal{E}^k$): 태스크 $k$ 만 접근 가능한 전용 Expert

각 태스크의 gate 는 Shared pool 과 자기 전용 Expert 를 모두 입력받아
최적 결합 비율을 학습한다.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

이 한 줄이 해결하는 것:

1. **Negative Transfer 완화**: 태스크 전용 Expert 는 해당 태스크에만
   특화된 패턴을 다른 태스크 gradient 의 간섭 없이 학습할 수 있다. 공유
   파라미터 공간이 줄어든 만큼 충돌 표면이 줄어든다.
2. **Expert Collapse 를 구조적으로 차단**: 태스크 전용 Expert 는 자기
   태스크의 gradient 만 받는다. "모두가 같은 Expert 로 몰린다" 는
   시나리오가 물리적으로 불가능해진다.
3. **공유/특화의 역할 분화가 자동화**: Shared Expert 는 "모든 태스크에
   유용해야만 살아남는 파라미터" 가 되고, Task Expert 는 "한 태스크에만
   유용하면 충분한 파라미터" 가 된다 — 두 종류의 표현이 구조적으로
   분리된다.

> **역사적 배경.** PLE 는 *Tang, Liu, Zhao & Gong (RecSys 2020)* 이
> Tencent 의 동영상 추천 시스템에서 제안하였다. Tencent Video 는 VCR
> (Video Completion Rate), VTR (Video Through Rate), Share Rate 등
> 다수의 참여도 지표를 동시에 최적화해야 했다. MMoE 를 적용했지만
> Expert Collapse 와 Seesaw 현상이 심각했고, PLE 는 MMoE 대비 모든
> 태스크에서 동시에 성능이 향상된 최초의 MTL 아키텍처로 보고되었다.
> 이후 Alibaba, JD.com, Kuaishou, ByteDance 등 중국 대형 플랫폼에서
> 산업용 MTL 의 사실상 표준이 되었다.

## 결정 2 — Expert 들을 이종 (heterogeneous) 으로 구성한다

그런데 원 PLE 논문은 Shared Expert 와 Task-specific Expert 모두 *동일한
구조의 작은 MLP* 로 쓴다. "Expert 여러 개를 두어 표현력을 늘린다" 는
아이디어는 살아있지만 각 Expert 의 귀납 편향 (inductive bias) 은 다르지
않다. 차이를 만드는 것은 gate 가중치뿐이다.

이 설정은 Tencent Video 에서는 문제가 없었다. 거기의 태스크들 (VCR,
VTR, Share Rate) 은 본질적으로 같은 종류의 사용자-아이템 상호작용을
예측했기 때문이다. 하지만 우리가 다루는 13개 태스크는 훨씬 이질적이다
— 다음 상품 추천, 이탈 예측, 고객 가치 계층화, 유사 고객 찾기, 브랜드
예측, LTV 회귀 등이 동시에 들어간다. 동종 MLP 7개 앞에 gate 를 붙여
이걸 처리하게 하면, gate 가 학습해야 하는 분업의 공간 자체가 너무 좁다.
"MLP-3 가 CTR 에 좋고 MLP-5 가 Churn 에 좋다" 는 결론은 잡음과
구분하기 어렵다.

그래서 한 걸음 더 나갔다. **Shared Expert pool 을 이종으로 구성한다.**
7개 각각이 구조적으로 다른 수학적 관점을 대표하도록 뽑았다.

- *쌍곡 기하학* (unified_hgcn) — 계층 구조를 쌍곡 공간에서 표현
- *지속 호몰로지* (perslay) — 거래 패턴의 위상적 형태
- *인자 분해 기계* (deepfm) — 피처 쌍의 대칭 교차
- *시간 동역학* (temporal) — 시계열 패턴
- *이분 그래프* (lightgcn) — 고객-가맹점 협업 신호
- *인과 추론* (causal) — do-연산자 수준 피처
- *최적 수송* (optimal_transport) — 분포 간 거리

왜 이렇게 했나 — 세 가지 이유다.

**표현력을 파라미터 수가 아니라 귀납 편향으로 인코딩한다.** 12GB VRAM
한 대로 Transformer 규모의 전문가를 여러 개 쌓는 것은 불가능했다.
하지만 각 Expert 가 자기 도메인에서 이미 최적화된 구조 (HGCN 의 쌍곡
기하, PersLay 의 지속 호몰로지) 를 빌려오면, 파라미터 수 대비 풍부한
표현을 확보할 수 있다. 같은 파라미터 예산에서 "구조적 다양성" 이 "깊이"
보다 우리 문제에 더 유용하다는 판단이다.

**설명 가능성을 구조적으로 확보한다.** "unified_hgcn 이 35%, temporal
이 28% 기여했다" 는 SHAP 근사가 아니라 *실제로 계산된 gate 가중치*
이고, 각 Expert 이름이 비즈니스적으로 읽힌다 ("계층 관계", "시간
패턴"). 동종 MLP 앙상블에서는 "MLP-3 이 28%" 가 고객에게도 감독
당국에게도 설명이 되지 않는다.

**태스크 간 역할 분화가 강해진다.** 동종 Expert 들은 학습 과정에서
비슷한 특성으로 수렴하기 쉽다 (Expert Collapse 의 또 다른 얼굴). 이종
Expert 는 각자 고유한 구조적 편향을 갖기 때문에 자연스럽게 분화된다.
gate 가 "어떤 관점의 Expert 를 선택할지" 결정할 때, 실제로 구분 가능한
관점 공간 안에서 선택하게 된다.

> **비유 — 진료 의뢰.** 동종 MLP 앙상블은 내과 전문의 7명에게 같은
> 환자를 보여주고 "누구의 의견을 들을지" 를 정하는 것과 같다. 의견들이
> 서로 비슷하니 gate 입장에서는 선택 기준이 잡음에 가깝다. 이종 Expert
> pool 은 내과의, 외과의, 영상의학과, 심리상담가, 재활의학과, 응급의학과,
> 약사 — 각자 분명히 다른 관점을 가진 7명이다. 증상에 따라 어느 관점이
> 필요한지가 구조적으로 명확하다.

이 결정이 이후 섹션들의 전제가 된다. 7개 이종 Expert 의 구성은 **PLE-3**
에서 한 명씩 소개하고, 이종 출력 차원 (64D / 128D) 이 만드는 새로운
문제는 **PLE-4** 에서 CGCAttention 두 단계 구조와 dim-normalize 로
푼다.

## 결정 3 — CGC: gate 를 Softmax 가중합으로

Shared + Task 분리와 이종 Expert 까지 왔으면, 이제 gate 의 *형태* 를
정해야 한다. Tang et al. 은 Customized Gate Control (CGC) 을 제시한다.
태스크 $k$ 의 gate 는 Shared pool 출력과 Task-$k$ 전용 출력을 concat 한
축 위에서 Softmax 가중합을 계산한다.

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^N$$

왜 Softmax 가중합인가 — 대안들이 있었다.

- **Hard selection (top-1)**: "가장 점수 높은 Expert 만 쓴다." 계산이
  가볍지만 미분 불가능해서 straight-through estimator 같은 우회 필요.
  더 중요한 건 "태스크 A 는 60% Shared, 40% Task" 같은 자연스러운
  혼합을 표현 못 한다.
- **Sigmoid 독립 가중치**: "Expert 마다 독립적으로 0~1 점수." 총합 제약이
  없어서 모두가 1 로 가거나 모두가 0 으로 가는 degenerate solution 을
  막기 어렵다.
- **Attention (Q·K/√d)**: "쿼리와 키의 내적으로 점수." CGC 가 본질적으로
  Attention 의 특수한 경우 (Query = task-specific projection, Key =
  Expert 표현, Value = Expert 출력) 이므로 이 선택이다. Softmax 가중합이
  결국 Attention 수학의 그림자다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

CGC 에서는 Query 가 태스크 $k$ 의 gate 가중치, Key 가 공유 표현
$\mathbf{h}_{shared}$, Value 가 각 Expert 의 출력 블록이다. 차이점은
Transformer 가 토큰 간 attention 을 계산하고 CGC 가 *Expert 간
attention* 을 계산한다는 것뿐이다. 같은 수학이 다른 단위에 적용된 것.

> **Softmax 를 선택한 세 가지 이유.** (1) *양수 보장*: $e^x > 0$ 이라
> 음의 가중치 문제가 없다. (2) *총합 = 1*: 가중치가 확률 분포가 되어
> "태스크 $k$ 가 이 입력에 대해 Expert 들에게 부여하는 주의 분포" 로
> 직접 해석된다. (3) *미분 편의*: $\frac{d}{dx} e^x = e^x$ 라 gradient
> 가 단순. 학부 수학에서 Softmax 가 분류 헤드의 기본 선택인 것과 같은
> 이유로, Expert gate 에서도 기본 선택이다.

## Expert 와 Gate 의 직관 — 세 가지 비유

위의 세 결정 (공용/전용 분리, 이종 구성, Softmax 가중합) 이 만드는
시스템을 세 가지 각도에서 다시 본다.

### Expert 는 서로 다른 렌즈

7개 Shared Expert 는 *동일한 고객 데이터* 를 7가지 전혀 다른 관점으로
본 결과다.

- **unified_hgcn**: 상품/카테고리의 계층 구조를 *쌍곡 공간* 에서 해석
- **perslay**: 거래 데이터의 *위상적 형태* 를 포착
- **deepfm**: 피처 간 *교차 상호작용* 을 학습
- **temporal**: *시간적 패턴* 과 동역학을 포착
- **lightgcn**: 고객-가맹점 *그래프 관계* 를 표현
- **causal**: 피처 간 *방향성 인과* 를 추출
- **optimal_transport**: *분포 간 거리* 를 측정

어떤 태스크에는 시간적 패턴이 중요하고 (CTR), 어떤 태스크에는 계층적
관계가 중요하다 (Brand Prediction). 같은 고객 데이터라도 "어느 렌즈로
보느냐" 가 태스크별로 달라야 한다는 게 이종 pool 의 출발점이다.

#### 7개 Shared Expert 비교: 입력, 학습 대상, 대체 불가능성

| Expert | 입력 | 학습 대상 | 다른 Expert로 대체 불가능한 이유 | 출력 차원 |
|---|---|---|---|---|
| DeepFM | 정규화 644D | 피처 쌍의 대칭 상호작용 | FM의 $O(nk)$ 2차 교차를 명시적으로 포착 | 64D |
| LightGCN | 사전 계산된 64D | 고객-가맹점 협업 신호 | 이분 그래프 기반 "비슷한 고객" 패턴 | 64D |
| Unified HGCN | 사전 계산된 47D | 가맹점 계층 구조 (가맹점 노드만) | 쌍곡 공간에서 MCC 트리 + 공동방문 보정 | 128D |
| Temporal | 시퀀스 $[B,180,16]$+$[B,90,8]$ | 시간적 패턴 변화 | Mamba+LNN+Transformer 앙상블 | 64D |
| PersLay | Persistence Diagram | 위상적 구조 | 소비 패턴의 루프/클러스터/분기점 | 64D |
| Causal | 정규화 644D | 피처 간 방향성 인과 (DAG) | 교란 변수 제거, 비대칭 인과 구조 | 64D |
| OT | 정규화 644D | 고객-프로토타입 분포 거리 | Wasserstein 거리로 분포 기하학 인코딩 | 64D |

> **왜 7개 Expert 모두 필요한가.** 7개는 동일 고객의 다른 측면을
> 포착하며, CGC Gate 가 태스크별로 최적 조합을 학습한다. 세 Expert
> (DeepFM, Causal, OT) 가 동일 정규화 644D 를 입력받지만 *대칭 /
> 비대칭 / 거리* 라는 근본적으로 다른 수학 구조를 추출하므로 중복이
> 아니다.

### Gate 는 주의 분포

Gate 는 태스크별로 "어떤 Expert 의 의견을 얼마나 신뢰할 것인가" 를
결정하는 *주의 메커니즘* 이다. 수식은 위에서 봤고, 풀어 읽으면:

- $\mathbf{W}_k \cdot \mathbf{h}_{shared}$: 현재 입력을 보고 각 Expert
  의 관련성을 *점수* 로 매긴다
- $\text{Softmax}$: 점수를 확률 분포로 변환하여 총합을 1 로 만든다
- $\mathbf{w}_k \in \mathbb{R}^7$: 태스크 $k$ 가 7개 Expert 에게 부여하는
  *신뢰도 벡터*

> **비유 — 의료 진단 위원회.** 7명의 전문의가 환자를 각자 관점에서
> 진단한다. Gate 는 "이 환자 상태를 판단할 때 어느 전문의 소견을 얼마나
> 비중 있게 볼 것인가" 를 정하는 주치의다. 심장 증상이면 내과의, 외상이면
> 외과의에게 높은 가중치 — 태스크별로 다른 렌즈 혼합이 필요하다는
> 그림을 수식 없이 말한 것.

### 혼합 밀도 모델로서의 PLE

왜 Expert 가중합이 강력한가? 함수 근사의 관점에서, 각 Expert 는 입력
공간의 특정 영역에 특화된 *기저 함수* 로 볼 수 있다.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot \mathbf{e}_i(\mathbf{x})$$

Gate 는 이 기저 함수들의 혼합 계수다. 통계학의 혼합 밀도 모델과 정확히
같은 구조:

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^N \pi_i(\mathbf{x}) \cdot p_i(\mathbf{y} \mid \mathbf{x})$$

$\pi_i(\mathbf{x})$ 가 gate, $p_i$ 가 Expert 에 대응한다. 각 Expert 가
입력 공간의 서로 다른 영역을 담당하므로 전체 모델은 단일 네트워크보다
*더 복잡한 함수를 효율적으로 근사* 할 수 있다. 이종 pool 을 쓰는 순간
이 "서로 다른 영역 담당" 이 구조적으로 강제된다 — 기저 함수들이 원래부터
다른 수학적 공간에 살고 있으니까.

> **최신 동향.** Mixture of Experts (MoE) 는 2024-2025 LLM 분야에서
> 핵심 아키텍처다. Mistral 의 *Mixtral 8x7B* (2023), Google 의 *Switch
> Transformer* (Fedus et al., 2022), DeepSeek 의 *DeepSeek-MoE* (2024)
> 가 대표적이다. 이들은 Sparse MoE (top-k 선택) 로 계산량을 줄인다. 본
> 시스템의 CGC 는 Dense MoE (모든 Expert 활용) 에 해당하며, Expert 수가
> 7개로 적어 sparse 선택 없이도 계산 효율이 유지된다.

## 그런데 이 설계는 또 다른 문제를 낳는다

세 결정 — 명시적 분리, 이종 구성, Softmax gate — 이 작동하면 MMoE 의
Collapse 는 막힌다. 그러나 새 문제들이 생긴다.

**첫째, 이종 출력 차원.** unified_hgcn 이 128D 이고 나머지 6개가 64D
라서, gate 가중치가 같아도 L2 기여가 2배 다르다. 학습이 큰 Expert 로
편향되는 미묘한 collapse 가 여전히 가능하다. 이건 PLE-4 의
`dim_normalize` 로 풀린다.

**둘째, Shared concat 을 태스크별로 다르게 재조합해야 한다.** CGCLayer
원형은 Shared + Task Expert 를 한 축에서 가중합하지만, 우리 설정에서는
Shared concat (512D) 을 각 태스크가 *다르게 색칠* 할 필요도 있다. 그래서
CGCAttention 을 원형 위에 직교하게 얹는다 — 이것도 PLE-4 의 주제다.

**셋째, 초기 gate 는 아무것도 모른다.** 랜덤 초기화 상태에서 한 Expert
가 우연히 앞서 나가면 다시 collapse 로 수렴할 수 있다. 그래서
`domain_experts` 기반의 bias 초기화 (CTR → PersLay + Temporal + UHGCN,
Brand_prediction → UHGCN) 와 **entropy 정규화** 로 추가 방어선을 친다 —
역시 PLE-4.

**넷째, 태스크 간 의존성은 아직 gate 에 표현되지 않았다.** CTR 이 CVR
에 영향을 줘야 하고 Churn 이 Retention 에 영향을 줘야 하지만, CGC gate
는 "Expert 를 어떻게 섞을지" 만 다룬다. 태스크 간 signal 전달은 별도
경로 — Logit Transfer — 가 담당한다. 이건 PLE-5 에서.

**다섯째, 고객 시간이 여러 속도로 흐른다.** 클릭 패턴은 일 단위, 이탈
위험은 월 단위, 라이프스타일은 연 단위. 하나의 Shared pool 안에서 이
세 스케일을 모두 다루기는 어렵다. HMM Triple-Mode 라우팅 — 역시 PLE-4.

## PLE 의 "Progressive" 는 어디로 갔는가

원 논문의 PLE 는 여러 Extraction Layer 를 *점진적* 으로 쌓는다. $l$번째
레이어의 태스크 $k$ 출력은:

$$\mathbf{h}_k^{(l)} = \text{Gate}_k^{(l)}\left(\mathbf{E}^{s,(l)}(\mathbf{h}^{(l-1)}), \mathbf{E}^{k,(l)}(\mathbf{h}_k^{(l-1)})\right)$$

각 레이어를 지날 때마다 Shared 표현은 "모든 태스크에 유용한 정보" 를
점진적으로 정제하고, Task 표현은 점점 태스크에 특화된다. CNN 에서
저수준 (에지, 텍스처) → 고수준 (객체, 의미) 으로 정보가 정제되는 것과
같은 원리다.

본 구현에서는 단일 Extraction Layer 를 쓴다. 대신 *CGC →
GroupTaskExpertBasket → Logit Transfer → Task Tower* 라는 4단계
파이프라인이 Progressive 한 정보 정제의 역할을 나눠 가진다. CGC 가 공유
표현을 재조합하고, GroupTaskExpertBasket 이 태스크 전용 정제를 수행하고,
Logit Transfer 가 태스크 간 의존성을 전달하고, Task Tower 가 최종
예측으로 압축한다. 깊이를 층에 넣지 않고 *파이프라인의 단계별 분업* 으로
흩뜨린 구조다.

## PLE-1 → PLE-2 요약, 그리고 다음 편

| 단계 | 문제 | 해결 | 새 문제 |
|---|---|---|---|
| Shared-Bottom | 공유 trunk 에서 태스크 gradient 충돌 | — | — |
| MMoE | Expert 여러 개 + task-별 gate | Expert Collapse (대칭 풀) |
| **PLE (2020)** | **Shared / Task 명시 분리 + CGC** | **Collapse 구조적 차단** | 동종 pool 의 분업 공간이 좁음 |
| **본 구현** | **이종 Expert 7개 + CGCAttention 2단계** | **구조적 분업 + 설명가능성** | 이종 차원, gate 초기화, 태스크 의존성, 시간 스케일 |

PLE-3 에서는 "이종 7명이 구체적으로 누구인가" 를 하나씩 본다.
DeepFM / LightGCN / Unified HGCN / Temporal / PersLay / Causal /
Optimal Transport — 각자가 어떤 수학적 렌즈로 같은 고객을 다르게
해석하는지를 한 번 훑고, 그들을 어떻게 묶어 gate 에 먹일지는 PLE-4 로
넘어간다.
