---
title: "[Study Thread] PLE-1 — MTL과 게이트드 전문가로의 진화 (Shared-Bottom → MMoE)"
date: 2026-04-19 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, mmoe, mtl, shared-bottom]
lang: ko
series: study-thread
part: 1
alt_lang: /2026/04/19/ple-1-mtl-evolution-en/
next_title: "PLE-2 — Progressive Layered Extraction: 명시적 전문가 분리와 CGC 게이트"
next_desc: "Shared/Task Expert를 명시적으로 분리한 PLE(Tang et al., 2020) 아키텍처. CGC 게이팅의 두 단계 — 1단계 CGCLayer(Shared + Task 가중합, 논문 원형)와 그 위에 얹는 2단계 CGCAttention(Shared concat 블록 스케일링) — 의 수식, Expert Collapse를 막는 entropy 정규화 및 이종 차원 보정까지."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 1편. 영문/국문 병렬로 PLE-1 → PLE-6
에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를 정리한다.
출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이고, 마지막 PLE-6
편에서 전체 PDF 를 첨부한다. adaTT 는 별도 서브스레드 (ADATT-1 ~ ADATT-4)
로 분리해 다룬다. 이번 1편은 멀티태스크 학습의 근본 동기에서 시작해,
Negative Transfer 의 수학적 얼굴을 짚고, Shared-Bottom 과 MMoE 가 각자
어디서 무너지는지를 따라간다 — 실제로 작동하는 세 번째 단계인 PLE 는
**PLE-2** 의 주제이다.*

## 왜 멀티태스크 학습인가

### 근본 동기 — 하나를 알면 다른 것도 더 잘 안다

추천 시스템은 CTR(클릭률), CVR(전환율), Churn(이탈), LTV(생애가치) 등
수십 가지 예측을 동시에 수행해야 한다. 직관적으로 생각하면 태스크마다
독립 모델을 만드는 것이 가장 자연스럽다. 하지만 실제로는 태스크들이
서로 관련되어 있다.

- CTR이 높은 고객은 CVR도 높을 가능성이 크다 (퍼널 관계)
- Churn 위험이 높은 고객은 Retention이 낮다 (역상관)
- 소비 패턴(Spending)은 LTV를 결정하는 핵심 신호다

이 관련성을 활용하면 데이터 효율이 극적으로 올라간다. 태스크 A를
학습하면서 발견한 패턴이 태스크 B에도 유용한 경우, 동일한 데이터로 더
풍부한 표현을 학습할 수 있다. 이것이 멀티태스크 학습(MTL)의 핵심
동기다.

> **비유 — 외국어 학습.** 스페인어를 배운 사람이 이탈리아어를 더 빨리
> 습득하는 것과 같다. 두 언어는 어근, 문법 구조, 발음 규칙을 상당 부분
> 공유한다. 하나를 학습하면서 획득한 "라틴어 계열 언어의 구조 이해"가
> 다른 언어 학습에 전이(transfer)된다. MTL에서 Shared Expert가 바로
> 이 "공통 구조 이해"를 담당한다.

### 통계적 관점 — 귀납적 편향과 정규화

단일 태스크 모델은 해당 태스크의 데이터만으로 학습하므로
과적합(overfitting) 위험이 높다. MTL은 여러 태스크가 공유 표현을
통해 서로의 학습을 *암묵적으로 정규화*한다.

$$\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k(f_k(\mathbf{h}_{shared}(\mathbf{x})))$$

이 총 손실을 최소화할 때, $\mathbf{h}_{shared}$ 는 어느 한 태스크에만
과적합하기 어렵다. 모든 태스크에 동시에 유용한 표현만이
$\mathbf{h}_{shared}$ 에 살아남는다. 이것은 L2 정규화나 Dropout과는
다른, *태스크 간 상호 정규화(inter-task regularization)* 다.

> **학부 수학 — 가중합(Weighted Sum)이란?** $\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k$
> 에서 $w_k$ 는 태스크 $k$ 의 가중치(중요도)이고, $\mathcal{L}_k$ 는
> 해당 태스크의 손실이다. 가중합은 "각 항목에 중요도를 곱한 뒤 더하는
> 것"으로, 일상에서 학점 평균과 같은 원리다. 구체적 예시로 국어
> 90점(가중치 2), 수학 80점(가중치 3)이면 가중 평균은
> $(2 \times 90 + 3 \times 80) / (2 + 3) = 420 / 5 = 84$ 점 — 수학에
> 더 비중을 둔 평균이다. MTL에서도 $w_{CVR} = 1.5$, $w_{CTR} = 1.0$
> 이면 CVR 태스크의 손실이 총 손실에 1.5배 더 많이 반영되어, 모델이
> CVR 예측 정확도를 더 중시하게 된다. $\mathbf{h}_{shared}$ 의 학습은
> 이 가중합의 gradient $\nabla \mathcal{L}_{MTL}$ 로 이루어지므로,
> $w_k$ 가 큰 태스크의 gradient 가 공유 표현에 더 강하게 영향을
> 미친다.

## Negative Transfer — 멀티태스크의 어두운 면

### 문제 정의

모든 태스크를 함께 학습하면 항상 좋을까? 아니다. **Negative Transfer**
는 관련성이 낮은 태스크가 공유 표현을 오염시켜 오히려 단일 태스크
모델보다 성능이 하락하는 현상이다.

> **⚠ Seesaw 현상 — 실무에서의 경험.** CTR 성능을 올리면 Churn 성능이
> 떨어지고, Churn을 올리면 CTR이 떨어지는 시소(seesaw) 현상은 MTL의
> 가장 흔한 실패 모드다. 이는 두 태스크의 gradient 가 공유 파라미터
> 공간에서 서로 반대 방향을 가리키기 때문이다.

### 최적화 관점 — Gradient 충돌

태스크 $k$ 의 손실을 $\mathcal{L}_k$ 라 하자. 공유 파라미터
$\boldsymbol{\theta}_{shared}$ 에 대한 각 태스크의 gradient 는 다음과
같다.

$$\mathbf{g}_k = \nabla_{\boldsymbol{\theta}_{shared}} \mathcal{L}_k$$

두 태스크의 gradient 가 *같은 방향* 을 가리키면
협력(positive transfer)이고, *반대 방향* 을 가리키면 충돌(negative
transfer)이다.

$$\cos(\mathbf{g}_i, \mathbf{g}_j) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

- $\cos > 0$: 태스크 $i$ 와 $j$ 의 gradient 가 협력 — 함께 학습하면 이득
- $\cos < 0$: gradient 가 충돌 — 공유 파라미터가 양쪽 모두에 해로운 방향으로 업데이트
- $\cos \approx 0$: 무관 — 함께 학습해도 큰 효과 없음

> **학부 수학 — 내적과 코사인 유사도.** 두 벡터
> $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ 의 내적은
> $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$ 이다. 기하학적으로
> $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta$
> 이므로, 코사인 유사도
> $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$
> 는 두 벡터의 *방향적 유사성* 을 $[-1, 1]$ 범위로 측정한다. 구체적
> 예시로 2차원에서 $\mathbf{g}_{CTR} = (3, 1)$,
> $\mathbf{g}_{CVR} = (2, 2)$ 이면
> $\cos = (3 \times 2 + 1 \times 2) / (\sqrt{10} \times \sqrt{8}) = 8 / 8.94 \approx 0.89$
> (강한 협력). $\mathbf{g}_{Churn} = (-1, 2)$ 이면
> $\cos = (3 \times (-1) + 1 \times 2) / (\sqrt{10} \times \sqrt{5}) = -1 / 7.07 \approx -0.14$
> (약한 충돌). 이처럼 gradient 벡터의 코사인 유사도가 음수면, 공유
> 파라미터를 업데이트할 때 한 태스크의 개선이 다른 태스크의 악화로
> 이어져 Negative Transfer가 발생한다.

> **직관적 해석 — 줄다리기.** 공유 파라미터는 여러 태스크가 동시에
> 잡아당기는 줄과 같다. 모든 태스크가 같은 방향으로 당기면 빠르게
> 전진하지만, 반대 방향으로 당기면 제자리걸음이거나 오히려 후퇴한다.
> PLE의 핵심 아이디어는 "각 태스크에게 자기만의 줄을 하나 더 주는
> 것"이다. Shared Expert는 공용 줄, Task-specific Expert(본 구현의
> GroupTaskExpertBasket)는 태스크 전용 줄에 해당한다.

### 정보이론 관점 — 태스크 간 상호정보량

태스크 $A$ 와 $B$ 의 레이블을 확률 변수 $Y_A$, $Y_B$ 라 하면, 두 태스크의
관계는 상호정보량(mutual information)으로 측정할 수 있다.

$$I(Y_A; Y_B) = \sum_{y_a, y_b} p(y_a, y_b) \log \frac{p(y_a, y_b)}{p(y_a) p(y_b)}$$

- $I(Y_A; Y_B)$ 가 높으면: 두 태스크가 공통 정보를 많이 공유 → 같은 Expert가 유용
- $I(Y_A; Y_B)$ 가 낮으면: 독립적 → 강제로 공유하면 noise만 주입

이상적인 MTL 아키텍처는 $I$ 가 높은 태스크끼리는 표현을 공유하고,
$I$ 가 낮은 태스크끼리는 분리해야 한다. PLE의 Shared/Task-specific
Expert 분리와 CGC 게이팅이 바로 이 원칙을 아키텍처 수준에서 구현한
것이다.

> **학부 수학 — 상호정보량과 KL-Divergence.** 상호정보량 $I(X; Y)$ 는
> 실은 두 분포 간의 *KL-Divergence* 로 표현된다:
> $I(X; Y) = D_{KL}(p(x,y) \,\|\, p(x) p(y))$. KL-Divergence
> $D_{KL}(P \,\|\, Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$ 는 "분포 $P$
> 를 분포 $Q$ 로 근사할 때 발생하는 정보 손실"이다. 따라서
> $I(X; Y)$ 는 "결합 분포 $p(x,y)$ 와 독립 가정 $p(x) p(y)$ 사이의 정보
> 손실", 즉 "$X$ 와 $Y$ 가 독립이 아님으로 인해 생기는 추가 정보"를
> 측정한다. 구체적 예시로 CTR과 CVR의 경우, 클릭한 고객이 전환할
> 확률이 높으므로
> $p(\text{click}=1, \text{convert}=1) \gg p(\text{click}=1) \times p(\text{convert}=1)$
> 이고, $I(\text{CTR}; \text{CVR})$ 이 크다. 반면 CTR과 Brand\_prediction은
> 상대적으로 독립적이어서 $I$ 가 작고, 같은 Expert를 강제 공유하면
> 노이즈만 추가된다.

## 아키텍처 진화 — Shared-Bottom에서 PLE까지

### Shared-Bottom — 가장 단순한 MTL

모든 태스크가 하나의 trunk(공유 네트워크)를 통과한 뒤, 태스크별
head(타워)로 분기하는 구조다.

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

- **장점**: 구현이 단순하고, 파라미터 효율적이다.
- **한계**: 모든 태스크가 동일한 표현을 강제로 공유하므로, 태스크 간
  관련성이 낮을 때 Negative Transfer가 심각하다. 비유하자면, 모든
  학생에게 동일한 교과서 한 권만 제공하는 것과 같다.

### MMoE — Expert를 여러 개 두고 선택하게 하자

Ma et al. (KDD 2018)의 MMoE는 $N$ 개의 동일 구조 Expert를 두고,
태스크별 gate가 Expert 출력의 가중합을 결정한다.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot f_i^{expert}(\mathbf{x}), \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x})$$

- **장점**: 태스크별로 다른 Expert 조합을 사용할 수 있다.
- **한계**: **Expert Collapse** — 모든 태스크의 gate가 동일한 Expert를
  선택하여 사실상 Shared-Bottom으로 퇴화하는 현상. 이는 gradient가
  "인기 Expert"에 집중되면서, 나머지 Expert의 gradient가 소실되어
  학습이 멈추기 때문이다.

> **역사적 배경.** *Shared-Bottom* 은 MTL의 원조인 *Caruana (Machine
> Learning, 1997)* 이 체계화한 구조다. Rich Caruana는 "관련 태스크를
> 함께 학습하면 귀납적 편향(inductive bias)이 개선된다"는 핵심 통찰을
> 제시하며, 의료 진단에서 폐렴 사망률 예측에 관련 태스크를 보조로
> 사용하여 성능이 향상됨을 보였다. 이후 20년간 MTL의 기본 프레임워크가
> 되었다. *MMoE* 는 *Ma, Zhao, Yi, Chen, Hong & Chi (KDD 2018)* 이
> Google에서 제안하였다. YouTube의 참여도 예측과 만족도 예측이라는 두
> 태스크에서 Shared-Bottom의 Negative Transfer가 심각했고, Expert를
> 복수로 두되 gate로 선택하는 아이디어로 이를 완화했다. Mixture of
> Experts 자체는 *Jacobs, Jordan, Nowlan & Hinton (1991)* 이 최초로
> 제안한 것으로, MMoE는 이를 멀티태스크 학습의 맥락에서 재해석한
> 것이다.

> **비유 — 뷔페 레스토랑.** MMoE는 뷔페 레스토랑과 같다. 여러
> 음식(Expert)이 준비되어 있고, 각 손님(태스크)이 자기 접시에 원하는
> 만큼 담는다(gate). 문제는 모든 손님이 스테이크(인기 Expert)만 담고
> 나머지 음식은 아무도 먹지 않아 폐기되는 상황이다. PLE의 해결책은
> "기본 식사(Shared Expert)"와 "개인 특선(Task Expert)"을 명시적으로
> 분리하여 제공하는 것이다.

## 여기서 멈추는 이유

본 프로젝트의 13개 태스크 설정에서 두 단계 모두 그대로는 쓸 수 없다.
Shared-Bottom은 태스크 다양성에 무너진다 — churn, 순위, 회귀 타깃이
공유 trunk을 서로 다른 방향으로 잡아당긴다. MMoE는 이론상 태스크들이
서로를 우회할 수 있지만, 대칭적인 expert pool과 제약 없는 gate의
조합에서는 Expert Collapse가 예외가 아니라 기본값이 된다. 실제로
버티는 해법은 구조적이다 — 동일한 Expert들에게 gate 압력만으로 스스로
분업을 학습시키려 하지 말고, *분리 자체를 아키텍처에 내장* 하는 것 —
cross-task 신호용 Shared Expert, 한 태스크만 신경 쓰는 패턴용
Task-specific Expert. 이것이 PLE이며, 여기서 **PLE-2** 가 이어받는다.
