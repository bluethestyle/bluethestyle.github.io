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

*"Study Thread" 시리즈의 adaTT 서브스레드 1편. 영문/국문 병렬로 ADATT-1 → ADATT-4
에 걸쳐 본 프로젝트의 adaTT(Adaptive Task Transfer) 메커니즘을 정리한다.
출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서` 이고, 마지막
ADATT-4 편에서 전체 PDF 를 첨부한다. 이번 1편은 "왜 적응형 타워인가"
의 근본 동기에서 출발해, Transformer Attention 과의 유비, 조건부
계산·Hypernetwork 계보에서 adaTT 가 어디에 위치하는지, 핵심 수식의
직관적 해석, 그리고 "측정하고, 선택하고, 조절한다" 는 전체 내러티브를
따라간다.*

## 왜 "적응형 타워"인가 — 고정 타워의 한계에서 출발하는 이야기

Multi-Task Learning(MTL)에서 가장 단순한 아키텍처는 *하나의 공유 백본*
위에 *태스크별 고정 타워*(Task-specific Tower)를 올리는 구조이다. 이
구조는 명쾌하지만 근본적인 약점을 안고 있다.

> **고정 타워의 세 가지 한계.**
> 1. *일방적 공유*: 공유 백본의 파라미터가 모든 태스크에 동일하게
>    영향을 미친다. CTR 최적화가 Churn 예측을 악화시켜도 이를 감지하거나
>    조절할 메커니즘이 없다.
> 2. *태스크 간 상호작용 무시*: 16개 태스크가 공유 파라미터를 두고
>    *암묵적 경쟁*을 벌이지만, 어떤 태스크 쌍이 서로 돕고 어떤 쌍이
>    해치는지 전혀 측정하지 않는다.
> 3. *학습 단계별 변화 미반영*: 학습 초기에는 CTR과 CVR이 비슷한
>    방향으로 학습되다가, 후반에는 전혀 다른 특성을 잡아낼 수 있다.
>    고정 가중치로는 이 변화를 추적할 수 없다.

adaTT는 이 세 가지 한계를 각각 다음과 같이 해결한다.

1. *선택적 전이*: gradient 방향이 일치하는 태스크 쌍만 지식을 공유하고,
   반대 방향이면 차단한다.
2. *태스크 친화도 측정*: gradient cosine similarity로 태스크 간 관계를
   *정량적으로* 측정한다.
3. *동적 적응*: 3-Phase 스케줄로 학습 단계에 따라 전이 강도를 조절한다.

핵심 내러티브를 한 문장으로 요약하면: *"고정된 타워는 태스크 간 간섭을
방관하지만, 적응형 타워는 간섭을 측정하고 제어한다."*

## Transformer와 Attention — 왜 이 메커니즘이 태스크 적응에 적합한가

adaTT가 "Adaptive Task-aware *Transfer*"라는 이름을 쓰지만, 내부적으로
활용하는 핵심 원리는 Attention 메커니즘의 사상(思想)과 깊이 연결되어
있다.

### Self-Attention의 Query-Key-Value 원리

Transformer의 Self-Attention은 세 가지 역할로 입력을 분리한다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

- $\mathbf{Q}$: Query — "나는 무엇을 찾고 있는가"
- $\mathbf{K}$: Key — "나는 어떤 정보를 제공할 수 있는가"
- $\mathbf{V}$: Value — "실제로 전달할 정보"
- $d_k$: Key 차원 (스케일링 팩터)

이 메커니즘의 핵심 통찰은 *"관련 있는 것에 집중하고, 관련 없는 것은
무시한다"* 이다. $\mathbf{Q} \mathbf{K}^\top$ 는 쿼리와 키 사이의
*유사도* 를 계산하고, softmax는 이 유사도를 *확률 분포* 로 변환하여
Value의 가중 합을 만든다.

> **학부 수학 — 내적(dot product)과 사잇각의 관계.** 두 벡터
> $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ 의 내적은
> $\mathbf{a} \cdot \mathbf{b} = \sum_{k=1}^n a_k b_k$ 이다. 고등학교에서
> 배운 $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$
> 와 연결하면, 내적 값이 *크다* 는 것은 (1) 벡터가 길거나 (2) 사잇각
> $\theta$ 가 작다(방향이 비슷하다)는 뜻이다. Attention에서
> $\mathbf{Q} \mathbf{K}^\top$ 의 $(i,j)$ 원소는 곧 $i$ 번 Query와
> $j$ 번 Key의 내적이므로, *방향이 비슷한 Key에 높은 점수를 부여* 하는
> 구조이다. $\sqrt{d_k}$ 로 나누는 이유는 차원 $d_k$ 가 커질수록 내적
> 값의 분산이 $d_k$ 에 비례하여 증가하기 때문에, 이를 정규화하지 않으면
> softmax 출력이 극단값에 몰리는 *gradient vanishing* 문제가 생기기
> 때문이다.

### adaTT에서의 Attention 유비(類比)

adaTT는 토큰 간의 Self-Attention이 아니라 *태스크 간의 Attention* 을
수행한다. 비유하면 다음과 같다.

| 역할 | Transformer Self-Attention | adaTT Task Transfer |
| --- | --- | --- |
| Query | 현재 토큰의 질의 | 현재 태스크의 gradient 방향 |
| Key | 다른 토큰의 응답 가능성 | 다른 태스크의 gradient 방향 |
| 유사도 | $\mathbf{Q} \mathbf{K}^\top / \sqrt{d_k}$ | gradient cosine similarity |
| 확률화 | softmax | softmax (temperature $T$) |
| Value | 다른 토큰의 실제 정보 | 다른 태스크의 loss 값 |
| 출력 | 가중 합산된 context | 전이 손실 (transfer loss) |

즉, adaTT의 전이 가중치 계산은 본질적으로 *태스크 공간에서의 Attention*
이다. "내 태스크(Query)와 gradient 방향이 비슷한(Key) 다른 태스크로부터
손실(Value)을 가져와 가중 합산한다."

## 조건부 계산과 Hypernetwork — adaTT의 계보

adaTT의 아이디어는 더 넓은 *Conditional Computation* 패러다임의
일부이다. 이 패러다임은 "입력이나 상황에 따라 네트워크의 동작을 동적으로
변경한다"는 사상이다.

### Conditional Computation의 핵심 아이디어

> **역사적 배경 — Conditional Computation의 기원.** "입력에 따라
> 네트워크의 일부만 활성화한다"는 아이디어는 Bengio et al. (2013,
> *"Estimating or Propagating Gradients Through Stochastic Neurons for
> Conditional Computation"*)에서 본격적으로 제안되었다. 당시 목표는
> *계산 비용 절감* 이었다 — 모든 뉴런을 매번 활성화하는 대신, 게이트
> 함수로 필요한 부분만 선택하여 연산량을 줄이는 것이었다. 이 아이디어는
> 이후 Mixture of Experts (MoE, Shazeer et al., 2017)로 발전하여 Google
> 의 Switch Transformer (Fedus et al., 2022)에서 1조 파라미터 규모로
> 확장되었다. adaTT는 "어떤 뉴런을 활성화할 것인가"가 아니라 "어떤
> 태스크의 지식을 전이할 것인가"를 조건부로 결정한다는 점에서,
> Conditional Computation의 *태스크 수준 확장* 이라고 할 수 있다.

전통적 신경망은 모든 입력에 대해 동일한 가중치 $\mathbf{W}$ 를
적용한다.

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Conditional Computation은 *조건 $\mathbf{c}$* 에 따라 가중치 자체가
변한다.

$$\mathbf{y} = \mathbf{W}(\mathbf{c}) \mathbf{x} + \mathbf{b}(\mathbf{c})$$

여기서 조건 $\mathbf{c}$ 는 태스크 ID, 입력 특성, 학습 단계 등 다양한
신호가 될 수 있다.

### Hypernetwork와의 관계

Ha et al. (ICLR 2017)의 Hypernetwork는 "네트워크가 다른 네트워크의
가중치를 생성한다"는 개념이다. adaTT는 이 아이디어의 *경량화된 변형*
으로 볼 수 있다.

> **역사적 배경 — Hypernetwork의 탄생.** Ha, Dai & Le (ICLR 2017,
> *"HyperNetworks"*)는 "작은 네트워크(hypernetwork)가 큰 네트워크(main
> network)의 가중치를 직접 생성한다"는 파격적인 아이디어를 제안했다.
> 영감의 원천은 생물학이었다 — 유전자(genotype)가 직접 행동을 결정하는
> 것이 아니라, 단백질 합성 경로를 거쳐 표현형(phenotype)을 만들어내는
> 것처럼, hypernetwork가 main network의 파라미터를 *간접적으로*
> 결정한다. 이 패러다임은 이후 Task-Conditioned HyperNetworks (von
> Oswald et al., NeurIPS 2020), LoRA (Hu et al., 2022)의 저랭크 적응
> 등으로 이어졌다. adaTT는 hypernetwork의 *풀 가중치 생성* 대신 gradient
> 유사도라는 *관측 신호* 로 전이 가중치를 결정하여 파라미터 효율성을
> 극대화한 변형이다.

| 측면 | Hypernetwork | adaTT |
| --- | --- | --- |
| 가중치 생성 | 별도 네트워크가 전체 가중치 생성 | gradient 유사도가 전이 가중치 결정 |
| 조건 입력 | 태스크 임베딩 벡터 | 태스크별 gradient 벡터 |
| 파라미터 수 | 큰 (생성 네트워크 자체가 큰) | 작은 ($n^2$ 전이 행렬 + prior) |
| 적응 속도 | 학습에 의존 | EMA로 빠른 적응 |

adaTT의 핵심 장점은 *gradient 자체를 조건 신호로 사용* 한다는 것이다.
태스크 임베딩 같은 별도의 학습 가능한 표현이 아니라, 현재 학습 상태에서
직접 관측되는 gradient 방향으로 태스크 관계를 판단한다. 이는 학습
단계에 따른 태스크 관계 변화를 *지연 없이* 반영할 수 있다.

### 태스크 임베딩이 파라미터 공간을 조절하는 메커니즘

일반적인 태스크 적응형 모델에서 태스크 임베딩 $\mathbf{e}_{\text{task}}$
는 파라미터 공간에서 *작동 영역을 선택* 하는 역할을 한다.

$$\mathbf{W}_{\text{effective}} = \mathbf{W}_{\text{shared}} + \mathbf{\Delta}(\mathbf{e}_{\text{task}})$$

adaTT에서 이 역할을 하는 것이 *전이 가중치 행렬* $\mathbf{R}$ 이다. 각
태스크 $i$ 에 대해 행 $\mathbf{R}_{i, :}$ 가 곧 "태스크 $i$ 의 관점에서
바라본 다른 태스크와의 관계"를 인코딩하며, 이것이 loss landscape에서의
이동 방향을 조절한다.

> **최신 동향 — Task-specific Adaptation 최신 연구 (2024–2025).** 태스크
> 임베딩으로 파라미터 공간을 조절하는 아이디어는 최근 더욱 정교해지고
> 있다. (1) *Task Arithmetic (Ilharco et al., ICLR 2023)*: 파인튜닝된
> 모델들의 가중치 차이 벡터(task vector)를 *산술적으로* 더하고 빼서
> 모델 능력을 조합하는 방법을 제안했다. 예를 들어 "감정 분석 능력 +
> 번역 능력 - 유해 출력 능력" 같은 연산이 가능하다. (2) *TIES-Merging
> (Yadav et al., NeurIPS 2023)*: 다수의 task vector를 병합할 때 부호
> 충돌(sign conflict)을 해결하는 알고리즘을 제안했다. (3)
> *AdapterSoup (Chronopoulou et al., EACL 2023)*: 다수의 LoRA 어댑터를
> 가중 평균하여 새로운 태스크에 적응한다. adaTT의 전이 가중치 행렬
> $\mathbf{R}$ 은 이러한 task vector 조합의 *연속적이고 동적인* 버전으로
> 해석할 수 있다.

수학적으로 볼 때, 전이 가중치가 loss에 미치는 영향은 다음과 같다.

$$\nabla_\theta \mathcal{L}_i^{\text{adaTT}} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$$

이 식에서 $\lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$
는 *태스크 $i$ 의 파라미터 업데이트 방향을 수정하는 보정 벡터* 이다.
친화도가 높은 태스크 $j$ 의 gradient가 큰 가중치로 합산되어, 공유
파라미터가 양쪽 태스크 모두에게 유리한 방향으로 이동하게 만든다.

## 핵심 수식의 직관적 해석

### 코사인 유사도 — 방향만 비교하는 이유

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

*직관*: 두 태스크의 gradient를 고차원 공간의 화살표로 생각하자.

- *크기* 는 loss의 절대적 민감도를 나타낸다. CTR loss가 0.01이고 LTV
  loss가 1000이면 gradient 크기가 수만 배 다르다.
- *방향* 은 "어느 쪽으로 파라미터를 바꾸면 loss가 줄어드는가"를
  나타낸다.

adaTT가 관심 있는 것은 *방향* 이다. "두 태스크가 파라미터를 같은
방향으로 바꾸고 싶어하는가?" 코사인 유사도는 벡터의 크기를 정규화하여
순수한 방향 비교만 수행한다. 만약 유클리드 거리를 사용하면, gradient가
큰 태스크가 지배적으로 작용하여 실제 방향이 같은데도 "거리가 멀다"고
판단할 수 있다.

> **학부 수학 — 코사인 유사도의 기하학적 의미.** 2차원 평면에서 벡터
> $\mathbf{a} = (3, 4)$, $\mathbf{b} = (6, 8)$ 을 생각하자. 유클리드
> 거리는 $\|\mathbf{a} - \mathbf{b}\| = \sqrt{9 + 16} = 5$ 로 "멀다"고
> 판정한다. 그러나 두 벡터는 *같은 방향* 이다. 코사인 유사도는
> $\cos\theta = (3 \cdot 6 + 4 \cdot 8) / (5 \cdot 10) = 50/50 = 1.0$ 으로
> "완전히 같은 방향"이라고 정확하게 판정한다. 기하학적으로 코사인
> 유사도는 두 벡터를 단위원(또는 고차원 단위구)에 사영(projection)한 뒤
> 사잇각을 측정하는 것과 같다. $\cos\theta = 1$ 이면 $\theta = 0°$ (같은
> 방향), $\cos\theta = 0$ 이면 $\theta = 90°$ (직교, 무관),
> $\cos\theta = -1$ 이면 $\theta = 180°$ (정반대 방향)이다. gradient의
> *크기* 는 loss의 절대적 스케일에 의존하지만, *방향* 은 "파라미터를
> 어느 쪽으로 바꿔야 loss가 줄어드는가"라는 본질적 정보를 담고 있으므로,
> 방향만 비교하는 코사인 유사도가 적합하다.

### Softmax 정규화 — 확률 분포를 형성하는 이유

$$w_{i \to j} = \frac{\exp(\mathbf{R}_{i,j} / T)}{\sum_{k \neq i} \exp(\mathbf{R}_{i,k} / T)}$$

*직관*: 전이 가중치를 *확률 분포* 로 만드는 것에는 세 가지 이유가 있다.

1. *합이 1*: 전이 가중치의 합이 항상 1이므로, 태스크 수가 늘어나도 전이
   손실의 스케일이 일정하다. 16개 태스크든 32개 태스크든, $\lambda = 0.1$
   의 의미가 변하지 않는다.
2. *경쟁적 선택*: softmax의 특성상 하나의 가중치가 올라가면 나머지가
   내려간다. 이는 *가장 도움이 되는 태스크에 집중* 하는 효과를 낸다.
3. *미분 가능성*: hard argmax와 달리 softmax는 연속적이고 미분 가능하여,
   전이 가중치 $\mathbf{W}$ (학습 가능 파라미터)의 gradient-based 최적화
   가 가능하다.

*Temperature $T$ 의 역할* 을 온도계에 비유하면 다음과 같다.

- $T$ 가 낮으면 (차가우면) 물질이 결정화되듯 *하나의 태스크에 집중*
  한다.
- $T$ 가 높으면 (뜨거우면) 물질이 기체가 되듯 *모든 태스크에 고르게
  분산* 된다.
- $T = 1.0$ 은 액체 상태 — 적당히 유동적이면서도 구조를 유지한다.

> **학부 수학 — Softmax 함수의 수학적 해부.** 입력 벡터
> $\mathbf{z} = (z_1, z_2, z_3)$ 에 대해 softmax는
> $\sigma(z_i) = e^{z_i} / \sum_k e^{z_k}$ 이다. 예를 들어
> $\mathbf{z} = (2, 1, 0)$ 이면
> $\sigma = (e^2, e^1, e^0) / (e^2 + e + 1) \approx (7.39, 2.72, 1.0) / 11.1 \approx (0.67, 0.24, 0.09)$
> 이다. 이제 temperature $T$ 를 적용하면 $\sigma(z_i / T)$ 가 된다.
> $T = 0.5$ 이면 입력이 $(4, 2, 0)$ 으로 *2배 확대* 되어
> $\approx (0.87, 0.12, 0.02)$ 으로 최대값에 더 집중하고, $T = 2.0$ 이면
> 입력이 $(1, 0.5, 0)$ 으로 *축소* 되어 $\approx (0.42, 0.34, 0.24)$ 으로
> 거의 균등해진다. 수학적으로 $T \to 0^+$ 이면 softmax는 one-hot
> 벡터($\text{argmax}$)에 수렴하고, $T \to \infty$ 이면 균등 분포 $1/n$
> 에 수렴한다. 이 성질은 통계역학의 Boltzmann 분포
> $p_i \propto e^{-E_i / k_B T}$ 에서 유래했으며, 여기서 $T$ 는 실제
> 물리적 온도이다.

### EMA 평활화 — 기억과 망각의 균형

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1 - \alpha) \cdot \mathbf{C}_t$$

*직관*: EMA를 일종의 *기억 시스템* 으로 생각하자.

- $\alpha = 0.9$ 는 "과거 기억의 90%를 유지하고, 새 관측의 10%만
  반영한다"는 뜻이다.
- 이는 *효과적 관측 창 (effective window)* $\approx 1 / (1 - \alpha) = 10$
  에 해당한다.
- 즉, 최근 10번의 관측을 가중 평균한 것과 비슷한 효과이다.

왜 단순 평균이 아닌 EMA인가?

- *단순 평균*: 모든 과거 관측에 동일한 가중치를 준다. 학습 초기의 (의미
  없는) gradient도 후반까지 영향을 미친다.
- *EMA*: 최근 관측에 더 큰 가중치를 준다. 태스크 간 관계가 학습 중에
  변하면 이를 빠르게 반영한다.
- *이동 평균 (sliding window)*: 정확한 윈도우 관리가 필요하고 메모리
  비용이 크다. EMA는 스칼라 하나($\alpha$)만으로 동일한 효과를 달성한다.

> **학부 수학 — EMA의 재귀 필터 해석.** EMA 공식
> $A_t = \alpha A_{t-1} + (1-\alpha) C_t$ 를 반복 대입하면
> $A_t = (1-\alpha) \sum_{k=0}^{t-1} \alpha^k C_{t-k} + \alpha^t A_0$ 이
> 된다. 각 과거 관측 $C_{t-k}$ 에 붙는 가중치는 $(1-\alpha) \alpha^k$ 로,
> $k$ 가 커질수록(과거로 갈수록) *기하급수적으로* 감소한다.
> $\alpha = 0.9$ 이면 1 step 전의 가중치는 $0.09$, 5 step 전은 $0.059$,
> 20 step 전은 $0.012$, 50 step 전은 $0.0005$ 로 사실상 무시된다. 이는
> 신호처리에서 *IIR (Infinite Impulse Response) 1차 저역통과 필터* 와
> 정확히 같은 구조이다. 전달 함수는 $H(z) = (1-\alpha) / (1 - \alpha z^{-1})$
> 이며, 차단 주파수 $f_c \approx (1-\alpha) / (2 \pi)$ 로 고주파
> 노이즈(배치별 gradient 변동)를 제거하고 저주파 추세(진짜 태스크 관계)
> 만 통과시킨다. 메모리 관점에서도, sliding window 평균이 $O(W)$ 메모리
> 가 필요한 반면 EMA는 현재 상태 $A_t$ 하나만 저장하면 되어 $O(1)$
> 메모리로 동작한다.

### Transfer-Enhanced Loss — 다른 태스크로부터 배우기

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$$

*직관*: 이 수식을 *조언을 듣는 과정* 으로 비유하자.

- $\mathcal{L}_i$: 태스크 $i$ 자신의 판단 (원본 loss)
- $\sum_{j \neq i} w_{i \to j} \cdot \mathcal{L}_j$: 다른 태스크들의 조언
  (가중 합산)
- $\lambda = 0.1$: 조언을 얼마나 심각하게 받아들일지 (10% 반영)
- $w_{i \to j}$: 누구의 조언을 더 신뢰할지 (친화도 기반 가중치)

$\lambda = 0.1$ 이라는 보수적인 값은 *"자기 판단의 90%를 유지하되,
동료의 조언을 10%만 반영한다"* 는 것이다. `max_transfer_ratio = 0.5` 는
"아무리 좋은 조언이라도 자기 판단의 50%를 초과하여 따르지 않는다"는
안전장치이다.

### Prior Blend — 경험과 데이터의 균형

$$\mathbf{R}_{\text{blended}} = (\mathbf{W} + \mathbf{A}) \cdot (1 - r) + \mathbf{P} \cdot r$$

*직관*: 이 수식은 *경험 많은 선배(Prior)와 실제 데이터(Affinity)의
의견을 혼합* 하는 과정이다.

- 학습 초기 ($r = 0.5$): "데이터가 아직 부족하니, 선배의 의견(도메인
  지식)을 절반 반영하자"
- 학습 후반 ($r = 0.1$): "이제 데이터가 충분히 쌓였으니, 실제 관측
  결과를 90% 신뢰하자"

이는 Bayesian 추론의 핵심 원리와 정확히 일치한다: *prior가 강할 때는
prior에 의존하고, 데이터가 충분해지면 likelihood(관측)를 따른다.*

> **학부 수학 — Bayesian 추론의 기초, 사전분포와 사후분포.** Bayes
> 정리는 $P(\theta | D) = P(D | \theta) \cdot P(\theta) / P(D)$ 이다.
> 여기서 $P(\theta)$ 는 *사전분포(prior)* — 데이터를 보기 전의 믿음이고,
> $P(D | \theta)$ 는 *우도(likelihood)* — 파라미터 $\theta$ 가 주어졌을
> 때 데이터가 관측될 확률이며, $P(\theta | D)$ 는 *사후분포(posterior)*
> — 데이터를 본 후 업데이트된 믿음이다. 동전 던지기를 예로 들자.
> 처음에 "앞면 확률이 0.5 근처"라는 prior를 갖고 있다가, 10번 던져 7번
> 앞면이 나오면 posterior가 0.5보다 높은 쪽으로 이동한다. 데이터가
> 1000번이면 posterior는 거의 관측 비율(0.7)에 수렴한다. adaTT에서 Group
> Prior $\mathbf{P}$ 가 "사전 믿음"이고, gradient 코사인 유사도가 "관측
> 데이터"이며, blend ratio $r$ 의 감소가 "데이터가 쌓이면 prior 의존도를
> 줄인다"는 Bayesian updating을 구현한다.

## 전체 내러티브 — "측정하고, 선택하고, 조절한다"

adaTT의 전체 작동 원리를 세 단어로 요약하면 *측정(Measure)*,
*선택(Select)*, *조절(Regulate)* 이다.

1. *측정*: 매 $N$ step마다 각 태스크의 gradient를 추출하고, 코사인
   유사도로 태스크 간 친화도를 계산한다. EMA로 노이즈를 걸러내어
   안정적인 친화도 행렬 $\mathbf{A}$ 를 유지한다.
2. *선택*: 친화도 행렬과 Group Prior를 혼합하고, negative transfer를
   차단한 뒤, softmax로 정규화하여 전이 가중치 $\mathbf{w}$ 를
   결정한다. "누구에게 배울 것인가"를 데이터 기반으로 결정하는 것이다.
3. *조절*: 3-Phase 스케줄로 전이의 강도와 시점을 제어한다. 학습
   초기에는 관찰만 하고(Warmup), 중반에는 동적으로 전이하며(Dynamic),
   후반에는 안정성을 위해 고정한다(Frozen).

이 세 단계가 결합되어, *16개 태스크가 서로의 학습을 방해하지 않으면서도
상호 보완적으로 성장하는* 생태계를 형성한다.

> **고정 타워 vs 적응형 타워 — 핵심 차이.** *고정 타워*: "모든 태스크가
> 같은 백본을 공유한다. 충돌이 나면 어쩔 수 없다." *적응형
> 타워(adaTT)*: "모든 태스크가 같은 백본을 공유하되, 누가 누구에게
> 도움이 되는지 실시간으로 측정하고, 해로운 간섭은 차단하며, 유익한
> 지식만 선별적으로 전달한다."

## adaTT 개요 및 설계 철학

> **핵심 문제 — Multi-Task Learning의 Negative Transfer.** 다수의
> 태스크를 하나의 네트워크로 동시 학습하면 *Negative Transfer* 가
> 발생할 수 있다. 태스크 A의 gradient가 태스크 B의 gradient와 반대
> 방향일 때, 공유 파라미터 업데이트가 한쪽 태스크의 성능을 향상시키면서
> 다른 쪽을 악화시키는 현상이다. 16개 태스크를 동시 학습하는 본
> 시스템에서는 이 문제가 특히 심각하다.

### 왜 Adaptive Task Transfer인가

본 프로젝트의 Multi-Task Learning (MTL) 아키텍처는 16개 활성 태스크를
동시에 학습한다. CTR, CVR, Churn, NBA 등 *서로 다른 비즈니스 목표* 를
가진 태스크들이 Shared Expert 파라미터를 공유하기 때문에, 태스크 간
상호작용을 제어하지 않으면 학습이 불안정해진다.

전통적인 접근법은 다음과 같다.

- *고정 가중치* (Fixed Weighting): 각 태스크에 수동으로 가중치를 설정
  (Kendall et al., 2018)
- *GradNorm*: gradient 크기 기반 동적 가중치 (Chen et al., 2018)
- *PCGrad*: gradient 충돌 시 사영(projection) (Yu et al., 2020)

이들은 *태스크 간 유사도* 를 직접 측정하지 않는다. adaTT는 gradient
cosine similarity로 태스크 간 친화도를 *측정* 하고, 이를 기반으로
*선택적 지식 전이* 를 수행한다.

> **역사적 배경 — MTL에서 Task-specific Adaptation의 발전사.**
> Multi-Task Learning의 개념은 Caruana (1997, *"Multitask Learning"*)
> 로 거슬러 올라간다. 초기에는 단순 Hard Parameter Sharing(공유 백본 +
> 태스크별 헤드)이 주류였으나, 태스크 간 간섭 문제가 지속적으로
> 보고되었다. 이를 해결하기 위해 (1) Cross-Stitch Networks (Misra et
> al., CVPR 2016) — 태스크별 네트워크 출력을 선형 결합, (2) MTAN (Liu
> et al., CVPR 2019) — Attention으로 공유 피처에서 태스크별 피처 선택,
> (3) PLE (Tang et al., RecSys 2020) — Expert 단위로 공유/전용 분리
> 등이 발전해왔다. adaTT는 이 계보에서 *gradient 자체를 태스크 관계
> 신호로 사용* 한다는 점에서 차별화된다.

> **최신 동향 — 2024–2025 Task-aware MTL 아키텍처 트렌드.** 최근 MTL
> 연구는 크게 세 방향으로 진화하고 있다. (1) *Gradient Manipulation*:
> Nash-MTL (Navon et al., ICML 2022), Aligned-MTL (Senushkin et al.,
> CVPR 2023) 등이 gradient를 파레토 최적 방향으로 사영하는 방법을
> 제안하며, CAGrad (Liu et al., NeurIPS 2021)는 최악 태스크의 gradient
> 방향을 보장한다. (2) *Task Routing*: Mod-Squad (Chen et al., NeurIPS
> 2023)은 각 태스크가 Expert 서브셋을 동적 선택하는 구조를 제안했고,
> adaTT의 친화도 기반 전이와 상보적이다. (3) *Foundation Model 시대의
> MTL*: LoRA 기반 태스크별 어댑터 (Hu et al., 2022), MTLoRA (Agiza et
> al., 2024) 등은 거대 모델에서 파라미터 효율적 태스크 적응을 수행한다.
> adaTT의 gradient cosine similarity 기반 접근은 이러한 어댑터
> 라우팅에도 확장 가능하다.

### 핵심 아이디어

adaTT의 핵심은 세 가지로 요약된다.

1. *Gradient Cosine Similarity*: 두 태스크의 gradient 방향이 같으면
   positive transfer, 반대면 negative transfer로 판정
2. *Group Prior*: 도메인 지식 기반의 태스크 그룹 구조를 prior로 활용하여
   학습 초기 안정성 확보
3. *3-Phase Schedule*: Warmup → Dynamic → Frozen 단계로 친화도 측정 →
   적용 → 고정

3-Phase 스케줄은 대략 다음과 같이 진행된다. Phase 1 — Warmup에서는
Group Prior만 사용하고 친화도를 측정만 한다. Phase 2 — Dynamic에서는
Prior와 Affinity를 혼합하여 전이 가중치를 학습한다. Phase 3 — Frozen
에서는 전이 가중치를 고정하여 미세조정을 안정화한다. 전환 시점은
`warmup_epochs`, `freeze_epoch` 설정값으로 결정된다.

### 시스템 내 위치

adaTT는 PLE-Cluster-adaTT 아키텍처의 *forward pass 후반* 에 위치한다.
Shared Experts → CGC Gating → Task Experts → Task Towers 순서로 예측값
이 생성된 후, 각 태스크의 손실이 계산되면 adaTT가 태스크 간 전이 손실을
추가한다. 태스크 손실들로부터 gradient를 추출하여 친화도 행렬을
업데이트하고, 전이 손실(transfer loss)을 원본 태스크 손실에 더해 Total
Loss를 구성하는 흐름이다. 구현상으로는 `ple_cluster_adatt.py:1290-1319`
구간에서 이 결합이 이루어진다.

## 여기서 멈추는 이유

지금까지 "왜 고정 타워가 부족한가"에서 출발해, Transformer Attention
과의 유비를 통해 adaTT가 태스크 공간에서 수행하는 Attention의 정체를
보였고, Conditional Computation · Hypernetwork 계보에서 gradient 자체를
조건 신호로 쓰는 adaTT의 위치를 규정했다. 핵심 수식 다섯 개 — 코사인
유사도, softmax 정규화, EMA 평활화, Transfer-Enhanced Loss, Prior
Blend — 를 "왜 이 형태인가"의 관점에서 뜯어보았고, "측정 → 선택 → 조절"
의 세 단어로 전체 내러티브를 요약했다. 하지만 *측정* 단계가 실제로
어떻게 구현되는가 — 배치마다 흔들리는 gradient에서 안정적인 친화도
행렬 $\mathbf{A}$ 를 어떻게 뽑아내는가, `torch.compiler.disable` 이 왜
필요한가, 코사인 유사도와 EMA를 결합한 `TaskAffinityComputer` 엔진의
정확한 수학과 코드 경로는 무엇인가 — 는 다음 편 **ADATT-2** 의
주제이다.
