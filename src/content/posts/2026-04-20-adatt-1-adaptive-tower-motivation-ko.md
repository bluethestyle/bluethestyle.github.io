---
title: "[Study Thread] ADATT-1 — adaTT 동기: 적응형 타워와 Transformer Attention 유비"
date: 2026-04-20 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, attention, hypernetwork, mtl]
lang: ko
series: study-thread
part: 7
alt_lang: /2026/04/20/adatt-1-adaptive-tower-motivation-en/
next_title: "ADATT-2 — TaskAffinityComputer와 Gradient Cosine Similarity"
next_desc: "태스크 간 친화도를 실제로 측정하는 TaskAffinityComputer 엔진, gradient cosine similarity 의 수학적 정의와 EMA 평활화, 유클리드 거리 대비 코사인 유사도를 쓰는 이유, 그리고 torch.compiler.disable 로 처리한 gradient 추출 경로까지."
next_status: published
---

*"Study Thread" 시리즈의 adaTT 서브스레드 1편. 영문/국문 병렬로
ADATT-1 → ADATT-4 에 걸쳐 본 프로젝트의 adaTT(Adaptive Task Transfer)
메커니즘을 정리한다. 출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서`
이고, 마지막 ADATT-4 편에서 전체 PDF 를 첨부한다. PLE 서브스레드가
피처 경로에서 태스크 간 충돌을 어떻게 분리했는지를 다뤘다면, adaTT
서브스레드는 *gradient 경로* 에서 남아 있는 충돌을 어떻게 측정하고
다시 협력으로 돌리는지를 다룬다. 이번 1편은 "왜 적응형 타워가 필요한가"
라는 설계 결정 하나에서 출발해, Transformer Attention 과의 유비,
Conditional Computation · Hypernetwork 계보에서의 위치까지 따라간다.*

> **잠정적 상태 — 합성 데이터에서 효용 미확인.** 솔직히 적어둔다.
> 현재까지 합성 데이터 벤치마크에서 adaTT 를 붙인 PLE 와 붙이지 않은
> PLE 사이에 뚜렷한 성능 차이가 관찰되지 않는다. 실 데이터에서도 동일한
> 결과라면 *adaTT 를 제거* 하는 방향이 합리적이다. 아래 4편은 제거를
> 전제로 쓰이지 않고 "왜 이 설계를 선택했는가" 를 남기는 기록에
> 가깝다 — 쓸모가 없다고 판명되더라도 그 판단의 수학적·공학적 근거는
> 가치가 있다고 본다.

## PLE 가 다 해결하지 못한 것 — gradient 충돌

PLE-2 는 피처 경로에서 "어떤 태스크가 어떤 Expert 를 보느냐" 를 CGC
gate 로 분리했다. Shared Expert 의 공통 표현과 Task Expert 의 전용
표현이 명시적으로 갈라지면서, 한 태스크의 피처 선호가 다른 태스크를
간섭하는 경로는 크게 줄었다.

하지만 *또 하나의* 간섭 경로가 남는다. gate 가 아무리 잘 갈라도 Shared
Expert 자체는 모든 태스크의 backward pass 를 통과한다. CTR 의 loss 가
$\theta_{shared}$ 를 어느 방향으로 밀면, Churn 의 loss 는 정반대
방향으로 밀 수 있다. 같은 파라미터가 매 step 마다 서로 반대 방향의
gradient 를 동시에 받으면, 공유 backbone 은 중간에서 진동한다.
"암묵적 정규화" 라는 MTL 의 장점이 이 순간에 *노이즈* 가 된다.

고정 타워로는 이 충돌을 볼 수도 없고 바꿀 수도 없다. 매 step 고정된
가중치로 태스크별 tower 에 분배할 뿐, *"지금 CTR gradient 와 Churn
gradient 가 서로를 밀치고 있는가"* 는 질문조차 던지지 않는다. adaTT 의
존재 이유는 이 한 질문을 던지는 것이다.

> **고정 타워의 세 한계.** (1) *일방적 공유* — CTR 최적화가 Churn 을
> 악화시켜도 감지 장치가 없다. (2) *태스크 간 상호작용 무시* — 어떤
> 태스크 쌍이 돕고 어떤 쌍이 해치는지 측정하지 않는다. (3) *학습 단계별
> 변화 미반영* — 초반과 후반의 태스크 관계 변화를 고정 가중치는 따라갈
> 수 없다.

## 세 개의 결정

이 한계들에 대응하려면 세 가지 결정을 차례로 내려야 한다.

*첫째 — 측정한다, 추측하지 않는다.* 태스크 관계를 수동으로 태깅하거나
도메인 지식에만 기대지 않고, 학습 *도중에* 직접 관측되는 신호에서 뽑는다.
이 결정은 뒤에서 gradient cosine similarity 로 이어진다.

*둘째 — Attention 의 사상을 빌린다.* "관련 있는 것에 집중하고, 관련
없는 것은 무시한다" 는 Self-Attention 의 핵심 원리는 태스크 적응에도
그대로 들어맞는다. 토큰 대신 태스크를 두고, Query-Key-Value 를 *태스크
공간* 에 재배치한다.

*셋째 — Conditional Computation 계보에 올라탄다.* Hypernetwork 계열은
조건 $\mathbf{c}$ 에 따라 가중치 자체를 바꾸는데, adaTT 는 가중치
생성이 아니라 *전이 강도* 만 조건부로 조절하는 경량 변형이다. 풀
hypernetwork 대신 $n^2$ 전이 행렬만 관리한다.

한 문장으로: *"고정된 타워는 태스크 간 간섭을 방관하지만, 적응형
타워는 간섭을 측정하고 제어한다."*

## Attention 의 사상을 태스크 공간으로

Transformer Self-Attention 은 입력을 세 역할로 분리한다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

- $\mathbf{Q}$ — "나는 무엇을 찾고 있는가"
- $\mathbf{K}$ — "나는 어떤 정보를 제공할 수 있는가"
- $\mathbf{V}$ — "실제로 전달할 정보"
- $d_k$ — Key 차원, softmax saturation 을 막는 스케일링 인자

> **수식 직관.** $\mathbf{Q} \mathbf{K}^\top$ 이 큰 값일수록 두 벡터의
> 방향이 비슷하다는 뜻이고, softmax 가 이걸 확률로 바꾼 뒤 $\mathbf{V}$
> 의 가중합을 취한다. "방향이 비슷한 Key 에 더 많이 집중" 이 한 줄
> 요약이다. $\sqrt{d_k}$ 로 나누는 이유는 차원이 커지면 내적의 분산이
> $d_k$ 에 비례해 커지고, 그대로 softmax 에 넣으면 극단값에 몰려서
> gradient 가 소실되기 때문이다.

adaTT 는 토큰이 아니라 *태스크 간* Attention 을 수행한다. 역할이 한
단계씩 미끄러진다.

| 역할 | Transformer Self-Attention | adaTT Task Transfer |
| --- | --- | --- |
| Query | 현재 토큰의 질의 | 현재 태스크의 gradient 방향 |
| Key | 다른 토큰의 응답 가능성 | 다른 태스크의 gradient 방향 |
| 유사도 | $\mathbf{Q} \mathbf{K}^\top / \sqrt{d_k}$ | gradient cosine similarity |
| 확률화 | softmax | softmax (temperature $T$) |
| Value | 다른 토큰의 정보 | 다른 태스크의 loss |
| 출력 | 가중 합산된 context | 전이 손실 (transfer loss) |

즉 adaTT 의 전이 가중치 계산은 본질적으로 *태스크 공간에서의 Attention*
이다. "내 gradient 방향과 비슷한 다른 태스크로부터 loss 를 가져와 가중
합산한다."

## 계보 — Conditional Computation 과 Hypernetwork

이 아이디어는 혼자 서 있지 않다. "입력이나 상황에 따라 네트워크의
동작을 동적으로 바꾼다" 는 Conditional Computation 패러다임 위에 있다.

> **역사적 배경.** Bengio et al. (2013, *"Estimating or Propagating
> Gradients Through Stochastic Neurons for Conditional Computation"*)
> 가 "입력에 따라 일부 뉴런만 활성화" 아이디어를 체계화했고, 이후
> Mixture of Experts (Shazeer et al., 2017) 와 Switch Transformer
> (Fedus et al., 2022) 로 발전해 1조 파라미터 규모에 도달했다.

전통 신경망은 모든 입력에 동일한 $\mathbf{W}$ 를 적용한다.

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Conditional Computation 은 조건 $\mathbf{c}$ 에 따라 가중치 자체가
변한다.

$$\mathbf{y} = \mathbf{W}(\mathbf{c}) \mathbf{x} + \mathbf{b}(\mathbf{c})$$

$\mathbf{c}$ 자리에 무엇을 넣느냐가 분기점이다. Hypernetwork (Ha et al.,
ICLR 2017) 는 태스크 임베딩을 넣어 "작은 네트워크가 큰 네트워크의
가중치를 생성" 하는 구조를 만든다 — 풀 가중치를 만들어내는 대가로
생성 네트워크 자체가 무거워진다.

adaTT 는 다른 선택을 한다. 가중치 생성 대신 *관측 신호* — 현재 학습
상태에서 직접 측정되는 gradient 방향 — 을 조건으로 쓴다. 태스크
임베딩 같은 별도 학습 표현이 없어도, 지금 이 순간 태스크 간 관계가
바뀌면 즉시 반영된다.

| 측면 | Hypernetwork | adaTT |
| --- | --- | --- |
| 가중치 생성 | 별도 네트워크가 전체 가중치 | gradient 유사도로 전이 가중치 |
| 조건 입력 | 태스크 임베딩 벡터 | 태스크별 gradient 벡터 |
| 파라미터 수 | 큰 (생성 네트워크 자체가 큰) | 작은 ($n^2$ 전이 행렬 + prior) |
| 적응 속도 | 학습에 의존 | EMA 로 빠른 적응 |

수학적으로 태스크 $i$ 의 gradient 에 미치는 영향은 다음 식으로 요약된다.

$$\nabla_\theta \mathcal{L}_i^{\text{adaTT}} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$$

뒤 항은 *태스크 $i$ 의 파라미터 업데이트 방향을 수정하는 보정 벡터*
다. 친화도가 높은 $j$ 의 gradient 가 큰 $w_{i \to j}$ 로 합산되어,
공유 파라미터가 양쪽 태스크 모두에 유리한 방향으로 움직이게 만든다.

## 다섯 수식이 왜 이 형태인가 — 각각의 결정 한 줄

뒤 편들에서 자세히 뜯어볼 다섯 개의 핵심 수식을 여기서는 "왜 이
형태여야 했는가" 의 한 줄로만 남긴다.

*코사인 유사도.* gradient 의 *크기* 는 loss 의 절대 스케일에 좌우되지만
*방향* 은 "파라미터를 어느 쪽으로 바꿔야 loss 가 줄어드는가" 라는
본질 정보다. 방향만 비교해야 태스크별 loss 스케일 차이에 끌려가지
않는다.

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

*Softmax 정규화.* 전이 가중치 합이 항상 1 이므로 태스크 수가 16 개든
32 개든 $\lambda = 0.1$ 의 의미가 유지된다. hard argmax 와 달리 미분
가능하여 $\mathbf{W}$ 를 학습할 수 있다.

$$w_{i \to j} = \frac{\exp(\mathbf{R}_{i,j} / T)}{\sum_{k \neq i} \exp(\mathbf{R}_{i,k} / T)}$$

*EMA 평활화.* 단일 배치 gradient 는 노이즈가 크다. $\alpha = 0.9$ 는
"과거 10 개 관측의 가중 평균" 과 근사하며, sliding window 대비 $O(1)$
메모리로 같은 효과를 낸다.

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1 - \alpha) \cdot \mathbf{C}_t$$

*Transfer-Enhanced Loss.* "자기 판단 90%, 동료 조언 10%" — $\lambda = 0.1$
로 원본 loss 의 학습 방향을 크게 왜곡하지 않으면서 positive transfer
이득만 챙긴다. `max_transfer_ratio=0.5` 가 추가 안전장치.

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$$

*Prior Blend.* 학습 초기엔 관측된 친화도가 불안정하니 도메인 지식을
절반, 후반엔 데이터를 90% 신뢰. Bayesian 의 "prior → posterior" 전환을
blend ratio $r$ 하나로 모사한 실용적 경량화.

$$\mathbf{R}_{\text{blended}} = (\mathbf{W} + \mathbf{A}) \cdot (1 - r) + \mathbf{P} \cdot r$$

> **학부 수학 — 내적과 방향.** $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$
> 이므로 내적이 크다는 건 (1) 벡터가 길거나 (2) 사잇각이 작다는 뜻이다.
> 코사인은 크기를 정규화하여 (2) 만 남긴다. $\cos = 1$ 같은 방향,
> $\cos = 0$ 직교, $\cos = -1$ 정반대.

## "측정하고 · 선택하고 · 조절한다"

adaTT 의 작동 원리를 세 단어로 요약하면 *측정 (Measure)*, *선택
(Select)*, *조절 (Regulate)* 이다.

1. *측정* — 매 $N$ step 마다 태스크별 gradient 를 추출해 코사인 유사도
   로 친화도를 계산한다. EMA 로 배치 노이즈를 걸러내어 안정적인 친화도
   행렬 $\mathbf{A}$ 를 유지한다. (ADATT-2 의 주제)
2. *선택* — 친화도와 Group Prior 를 혼합하고 negative transfer 를
   차단한 뒤 softmax 로 정규화하여 전이 가중치 $\mathbf{w}$ 를 결정한다.
   (ADATT-3 의 주제)
3. *조절* — 3-Phase 스케줄 (Warmup → Dynamic → Frozen) 로 전이 강도와
   시점을 제어한다. 초기엔 관찰만, 중반엔 동적 전이, 후반엔 안정을 위한
   고정. (ADATT-3 의 주제)

세 단계가 결합되면 *16 개 태스크가 서로의 학습을 방해하지 않으면서도
상호 보완적으로 성장하는* 생태계가 만들어진다. 고정 타워가 "공유
백본이 요동쳐도 어쩔 수 없다" 고 받아들이는 자리에서, adaTT 는
"누가 누구에게 도움이 되는지 실시간으로 측정하고, 해로운 간섭은 끊고,
유익한 지식만 전달한다" 로 대체한다.

## 여기서 멈추는 이유

여기까지 "왜 적응형 타워가 필요한가" 를 세 결정 — 측정 기반, Attention
사상, Conditional Computation 계보 — 로 정리했다. 다섯 개의 핵심 수식
이 왜 이 형태여야 하는지 한 줄씩 짚었고, 전체 동작을 "측정 · 선택 ·
조절" 로 묶었다. 그러나 *측정* 단계가 실제로 어떻게 구현되는가 — 배치
마다 흔들리는 gradient 에서 어떻게 안정적인 친화도 행렬 $\mathbf{A}$
를 뽑아내는가, 왜 `torch.compiler.disable` 이 필요한가, `TaskAffinityComputer`
엔진의 정확한 수학과 코드 경로는 — 이 질문은 다음 편 **ADATT-2** 의
주제다.
