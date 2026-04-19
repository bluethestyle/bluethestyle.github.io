---
title: "[Study Thread] PLE-2 — Progressive Layered Extraction: 명시적 전문가 분리와 CGC 게이트"
date: 2026-04-19 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, tang2020, mtl]
lang: ko
series: study-thread
part: 2
alt_lang: /2026/04/19/ple-2-progressive-layered-extraction-en/
next_title: "PLE-3 — 입력 구조와 이종 Shared Expert Pool (576D)"
next_desc: "PLEClusterInput 의 전체 필드 사양과 734D features 텐서 인덱스 매핑, HMM 모드 라우팅. 그리고 본 프로젝트의 8개 이종 Shared Expert (EconomicsExpert·TemporalExpert·HMMExpert·TDAExpert·GMMExpert·GCNExpert·PersLayExpert·UnifiedHGCN) 가 각자 어떤 수학적 관점으로 고객을 해석하는지까지."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 2편. 영문/국문 병렬로 PLE-1 →
PLE-6 에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를
정리한다. 출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이다.
이번 2편은 PLE(Tang et al., RecSys 2020)가 Shared-Bottom·MMoE의 실패를
어떻게 해결했는지 — 공유와 분리의 명시적 균형, Expert와 Gate의 역할,
수식이 말하는 것들, 그리고 "왜 PLE인가" 라는 내러티브를 따라간다.*

## PLE: 공유와 분리의 명시적 균형

Tang et al. (RecSys 2020)의 PLE는 Expert를 두 종류로 나눈다.

- *Shared Expert* ($\mathcal{E}^s$): 모든 태스크가 접근 가능한 공용 Expert
- *Task-specific Expert* ($\mathcal{E}^k$): 태스크 $k$만 접근 가능한 전용 Expert

각 태스크의 gate는 Shared Expert와 자기 전용 Expert를 모두 입력받아
최적 결합 비율을 학습한다.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

이 설계가 해결하는 문제들:

1. **Negative Transfer 완화**: 태스크 전용 Expert가 해당 태스크에만
   특화된 패턴을 간섭 없이 학습할 수 있다.
2. **Expert Collapse 방지**: Shared Expert는 "반드시 모든 태스크에
   유용한 정보"를 학습하고, Task Expert는 "특화 정보"를 학습하여 역할이
   자연스럽게 분리된다.
3. **Progressive (점진적)**: 여러 Extraction Layer를 쌓아 저수준→고수준으로
   점진적으로 태스크별 표현을 정제할 수 있다.

> **역사적 배경.** PLE는 *Tang, Liu, Zhao & Gong (RecSys 2020)* 이
> Tencent의 동영상 추천 시스템에서 제안하였다. 당시 Tencent Video는
> VCR(Video Completion Rate), VTR(Video Through Rate), Share Rate 등
> 다수의 참여도 지표를 동시에 최적화해야 했다. MMoE를 적용했지만 Expert
> Collapse와 Seesaw 현상이 심각하여, "Expert를 명시적으로 공유/전용으로
> 분리하면 어떨까?"라는 아이디어에서 PLE가 탄생했다. 논문에서 PLE는
> MMoE 대비 모든 태스크에서 동시에 성능이 향상된 최초의 MTL
> 아키텍처로 보고되었다. 이후 Alibaba, JD.com, Kuaishou, ByteDance 등
> 중국 대형 플랫폼에서 광범위하게 채택되어 산업용 MTL의 사실상
> 표준(de facto standard)이 되었다.

## 논문 대비 본 구현의 분기 — 동종 MLP에서 이종 Expert로

원본 PLE 논문(Tang et al., 2020)에서 Shared Expert와 Task-specific
Expert는 모두 *동일한 구조의 작은 MLP* 다. "Expert 여러 개를 두어
표현력을 늘린다"는 아이디어는 살아있지만, 각 Expert의 귀납
편향(inductive bias)은 다르지 않다. 차이를 만드는 것은 gate 가중치
뿐이다.

본 프로젝트는 여기서 한 걸음 더 나간다. **Shared Expert를 이종
(heterogeneous)으로 구성한다.** 8개의 Shared Expert 각각이 *구조적으로
다른 수학적 관점* 을 대표하도록 선택했다.

- *쌍곡 기하학* (unified_hgcn) — 계층 구조를 쌍곡 공간에서 표현
- *지속 호몰로지* (perslay) — 거래 패턴의 위상적 형태
- *인자 분해 기계* (deepfm) — 피처 쌍의 대칭 교차
- *시간 동역학* (temporal) — 시계열 패턴
- *이분 그래프* (lightgcn) — 고객-상품 관계
- *인과 추론* (causal) — do-연산자 수준 피처
- *최적 수송* (optimal_transport) — 분포 간 거리
- *멱법칙 원시 스케일* (raw_scale) — 정규화 전 금융 고유 스케일 (v3.3)

### 왜 이렇게 했나

세 가지 이유다.

**표현력을 파라미터 수가 아니라 귀납 편향으로 인코딩한다.** 12GB VRAM
한 대로 Transformer 규모의 전문가를 여러 개 쌓는 것은 불가능했다.
그러나 각 Expert 가 자기 도메인에서 이미 최적화된 구조(HGCN 의 쌍곡
기하, PersLay 의 지속 호몰로지 등)를 빌려오면, 파라미터 수 대비 풍부한
표현을 확보할 수 있다.

**설명 가능성을 구조적으로 확보한다.** "unified_hgcn 이 35%, temporal
이 28% 기여했다" 는 SHAP 근사가 아니라 *실제로 계산된 gate 가중치* 다.
그리고 각 Expert 이름 자체가 비즈니스적으로 읽힌다 ("계층 관계", "시간
패턴"). 동종 MLP 앙상블에서는 불가능한 특성이다 — "MLP 3번이 28%" 는
고객에게도 감독 당국에게도 설명이 되지 않는다.

**태스크 간 상호 정규화가 강해진다.** 동종 Expert 들은 학습 과정에서
비슷한 특성으로 수렴하기 쉽다 (Expert Collapse 의 또 다른 얼굴). 이종
Expert 는 각자 고유의 구조적 inductive bias 를 갖기 때문에 자연스럽게
역할이 분화된다. gate 가 "어떤 관점의 Expert 를 선택할지" 결정할 때,
실제로 구분 가능한 관점 공간 안에서 선택한다.

> **논문 vs 구현의 교훈.** PLE 논문은 Tencent Video 의 참여도
> 예측에서 탄생했다. 거기서는 동종 MLP 만으로도 MMoE 대비 의미 있는
> 개선이 가능했다 — 태스크들(VCR, VTR, Share Rate)이 본질적으로 같은
> 종류의 사용자-아이템 상호작용을 예측했기 때문이다. 본 프로젝트의 13개
> 태스크는 훨씬 이질적이다 (다음 상품 추천, 이탈 예측, 고객 가치
> 계층화, 유사 고객 찾기 등). 이 이질성을 *Expert 구조 수준에서도*
> 반영하는 것이 합리적이었다.

이 결정이 아래 섹션의 모든 내용 — 8개 이종 Expert 의 구성,
CGCAttention 블록 스케일링 방식, 이종 차원 정규화 — 의 전제가 된다.

## Expert와 Gate의 역할 — 직관적 이해

### Expert: 서로 다른 렌즈로 세상을 보는 전문가

Expert는 입력 데이터를 특정 관점으로 해석하는 *전문가* 다. 본 시스템의
8개 Shared Expert는 각각 전혀 다른 도메인의 시각을 제공한다.

- **unified_hgcn**: 상품/카테고리의 계층 구조를 *쌍곡 공간* 에서 해석
- **perslay**: 거래 데이터의 *위상적 형태(topological shape)* 를 포착
- **deepfm**: 피처 간 *교차 상호작용* 을 학습
- **temporal**: *시간적 패턴* 과 동역학을 포착
- **lightgcn**: 고객-상품 *그래프 관계* 를 표현
- **causal**: 피처 간 *인과 관계* 를 추출
- **optimal_transport**: *분포 간 거리* 를 측정
- **raw_scale**: 정규화 전 원시 피처의 *멱법칙 분포 패턴* 을 보존 (v3.3)

이들은 동일한 고객 데이터를 8가지 서로 다른 "렌즈"로 바라본 결과다.
어떤 태스크에는 시간적 패턴이 중요하고(CTR), 어떤 태스크에는 계층적
관계가 중요하다(Brand Prediction).

#### 8개 Shared Expert 비교: 입력, 학습 대상, 대체 불가능성

| Expert | 입력 | 학습 대상 | 다른 Expert로 대체 불가능한 이유 | 출력 차원 |
|---|---|---|---|---|
| DeepFM | 정규화 644D | 피처 쌍의 대칭 상호작용 | FM의 $O(nk)$ 2차 교차를 명시적으로 포착 | 64D |
| LightGCN | 사전 계산된 64D | 고객-가맹점 협업 신호 | 이분 그래프 기반 "비슷한 고객" 패턴 | 64D |
| Unified HGCN | 사전 계산된 47D | 가맹점 계층 구조 (가맹점 노드만) | 쌍곡 공간에서 MCC 트리 + 공동방문 보정 | 128D |
| Temporal | 시퀀스 $[B,180,16]$+$[B,90,8]$ | 시간적 패턴 변화 | Mamba+LNN+Transformer 앙상블 | 64D |
| PersLay | Persistence Diagram | 위상적 구조 | 소비 패턴의 루프/클러스터/분기점 | 64D |
| Causal | 정규화 644D | 피처 간 방향성 인과 (DAG) | 교란 변수 제거, 비대칭 인과 구조 | 64D |
| OT | 정규화 644D | 고객-프로토타입 분포 거리 | Wasserstein 거리로 분포 기하학 인코딩 | 64D |
| RawScale | 원시 90D | 멱법칙 분포 패턴 (v3.3) | 정규화 시 손실되는 원시 스케일/분포 정보 보존 | 64D |

> **왜 8개 Expert 모두 필요한가.** 8개 Expert는 동일 고객의 다른 측면을
> 포착하며, CGC Gate가 태스크별로 최적 조합을 학습한다. 세
> Expert(DeepFM, Causal, OT)가 동일 정규화 644D를 입력받지만
> 대칭/비대칭/거리라는 근본적으로 다른 수학 구조를 추출하므로 중복이
> 아니다. RawScaleExpert(v3.3)는 정규화 전 원시 90D 피처를 입력받아
> 멱법칙 분포 정보를 보존한다.

### Gate: 어떤 전문가의 의견을 얼마나 들을 것인가

Gate는 태스크별로 "어떤 Expert의 의견을 얼마나 신뢰할 것인가"를
결정하는 *주의(attention) 메커니즘* 이다.

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

이 수식이 말하는 것은 명확하다.

- $\mathbf{W}_k \cdot \mathbf{h}_{shared}$: 현재 입력을 보고 각 Expert의 관련성을 *점수* 로 매긴다
- $\text{Softmax}$: 점수를 확률 분포로 변환하여, 총합이 1이 되게 한다
- $\mathbf{w}_k \in \mathbb{R}^8$: 태스크 $k$가 8개 Expert에게 부여하는 *신뢰도 벡터*

> **비유 — 의료 진단 위원회.** 8명의 전문의(Expert)가 환자(입력
> 데이터)를 각자의 전문 분야에서 진단한다. 내과의, 외과의, 영상의학과
> 의사 등이 각각 소견서(Expert output)를 작성한다. Gate는 "이 환자의
> 상태를 판단할 때 어느 전문의의 소견을 얼마나 비중 있게 볼 것인가"를
> 결정하는 *주치의* 의 역할이다. 심장 관련 증상이면 내과의 소견에 높은
> 가중치를, 외상이면 외과의에게 높은 가중치를 부여한다.

> **학부 수학 — Softmax에서 왜 $e^x$(자연 지수 함수)를 사용하는가?**
> Softmax는 임의의 실수 벡터를 확률 분포(양수, 합=1)로 변환한다:
> $\text{Softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$. $e^x$ 를 선택하는
> 이유는 세 가지다: (1) *양수 보장*: $e^x > 0$ (모든 실수 $x$에 대해),
> 따라서 "음의 확률" 문제가 없다. (2) *단조 증가*: $z_i > z_j$이면
> $e^{z_i} > e^{z_j}$이므로 점수 순서가 보존된다. (3) *미분 편의*:
> $\frac{d}{dx} e^x = e^x$이므로 gradient 계산이 간결하다. *구체적
> 계산*: 점수 벡터 $\mathbf{z} = [2.0, 1.0, 0.1]$이면
> $e^{\mathbf{z}} = [7.39, 2.72, 1.11]$, 합 $= 11.22$,
> $\text{Softmax} = [0.659, 0.242, 0.099]$ — 점수 차이가 확률 차이로
> 변환된다. 점수 차이가 클수록 확률 차이가 급격해지는 것이 $e^x$의
> 지수적 증가 특성 덕분이다. 다른 양수 함수(예: $x^2$)를 쓸 수도
> 있지만, 음수 입력에서 순서가 꼬이고($(-3)^2 > (-1)^2$) gradient가 0
> 근처에서 소실되는 문제가 생겨 $e^x$가 최적의 선택이다.

## 수학적 고찰 — 수식이 말하는 것들

### Gating과 Attention의 연결

CGC gate의 Softmax 가중합은 Transformer의 Attention 메커니즘과 본질적으로
같은 구조다.

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

CGC에서는:

- **Query**: 태스크 $k$의 게이트 가중치 $\mathbf{W}_k$ (이 태스크가 원하는 정보)
- **Key**: 공유 표현 $\mathbf{h}_{shared}$ (현재 입력에 대한 각 Expert의 관련성)
- **Value**: 각 Expert의 출력 블록 $\mathbf{h}_i^{expert}$ (실제 정보)

차이점은 Transformer가 시퀀스의 각 토큰 간 attention을 계산하는 반면,
CGC는 *Expert 간 attention* 을 계산한다는 것이다. 동일한 수학적 원리 —
"관련성에 비례하여 정보를 선택적으로 결합" — 가 다른 단위(토큰 vs
Expert)에 적용된 것이다.

### Expert 가중합의 함수 근사론적 의미

Expert 출력의 가중합이 왜 강력한가? 이를 함수 근사(function
approximation)의 관점에서 이해할 수 있다.

$$\mathbf{h}_k = \sum_{i=1}^N g_{k,i} \cdot \mathbf{e}_i(\mathbf{x})$$

각 Expert $\mathbf{e}_i$는 입력 공간의 특정 영역에 특화된 *기저
함수(basis function)* 로 볼 수 있다. Gate $g_{k,i}$는 이 기저 함수들의
*혼합 계수* 다. 이는 Mixture of Experts의 이름에 담긴 의미 그대로,
*전문가 혼합 모델(Mixture of Experts model)* 이다.

통계학의 혼합 밀도 모델(mixture density model)과 정확히 같은 구조다.

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^N \pi_i(\mathbf{x}) \cdot p_i(\mathbf{y} \mid \mathbf{x})$$

여기서 $\pi_i(\mathbf{x})$가 gate, $p_i$가 Expert에 대응한다. 각
Expert가 입력 공간의 서로 다른 영역을 담당하므로, 전체 모델은 단일
네트워크보다 *더 복잡한 함수를 효율적으로 근사* 할 수 있다.

> **학부 수학 — 가중합의 함수 근사론적 의미.** 수학에서 임의의 함수를
> 기저 함수(basis function)의 선형 결합(가중합)으로 표현하는 것은 근사
> 이론의 핵심이다. 푸리에 급수
> $f(x) = \sum a_n \cos(nx) + b_n \sin(nx)$ 가 대표적인 예시로,
> 삼각함수라는 기저의 가중합으로 *모든 주기 함수* 를 근사할 수 있다.
> Expert의 가중합 $\mathbf{h} = \sum g_i \cdot \mathbf{e}_i(\mathbf{x})$
> 도 같은 원리다. 각 Expert $\mathbf{e}_i$는 입력 공간의 특정 패턴에
> 특화된 "기저 함수"이고, gate $g_i$가 혼합 계수 역할을 한다. 차이점은
> 기저 자체도 학습된다는 것이다. *구체적 예시*: 입력 $\mathbf{x}$가
> 시간적 패턴이 강한 고객이면 $g_{\text{temporal}} = 0.4$ 로 커지고,
> 그래프 관계가 중요하면 $g_{\text{lightgcn}} = 0.3$ 으로 커진다. 결과
> 표현 $\mathbf{h}$는 이 고객에게 가장 적합한 전문가 의견의 혼합이
> 된다.

> **최신 동향.** Mixture of Experts(MoE)는 2024-2025년 LLM 분야에서 핵심
> 아키텍처로 부상했다. Mistral의 *Mixtral 8x7B* (2023), Google의
> *Switch Transformer* (Fedus et al., 2022), DeepSeek의 *DeepSeek-MoE*
> (2024)가 대표적이다. 이들은 Sparse MoE(top-k 선택)를 사용하여 파라미터
> 수 대비 계산량을 크게 줄였다. 추천 시스템에서도 Alibaba의 *Star
> Topology Adaptive Recommender* (STAR, CIKM 2021), Kuaishou의 *PEPNet*
> (KDD 2023)이 입력 조건에 따라 Expert를 선택하는 MoE 구조를 채택했다.
> 본 시스템의 CGC 게이팅은 Dense MoE(모든 Expert 활용)에 해당하며,
> Expert 수가 8개로 적어 sparse 선택 없이도 계산 효율이 유지된다.

### Progressive 구조가 정보 흐름에 미치는 영향

원 논문의 PLE는 여러 Extraction Layer를 쌓는다. $l$번째 레이어의 태스크
$k$ 출력은 다음과 같다.

$$\mathbf{h}_k^{(l)} = \text{Gate}_k^{(l)}\left(\mathbf{E}^{s,(l)}(\mathbf{h}^{(l-1)}), \mathbf{E}^{k,(l)}(\mathbf{h}_k^{(l-1)})\right)$$

각 레이어를 지날 때마다:

1. **Shared 표현** 은 모든 태스크에 공통으로 유용한 정보를 *점진적으로 정제* 한다
2. **Task 표현** 은 점점 더 태스크에 *특화* 된다
3. Gate는 레이어마다 독립적으로 학습되어, *추상화 수준별로* 다른 결합 전략을 사용할 수 있다

이것은 CNN에서 저수준(에지, 텍스처)→고수준(객체, 의미)으로 정보가
정제되는 것과 동일한 원리다.

본 구현에서는 단일 Extraction Layer를 사용하지만, *CGC →
GroupTaskExpertBasket → Logit Transfer* 라는 3단계 파이프라인이 사실상
Progressive한 정보 정제의 역할을 수행한다.

## 핵심 수식의 직관적 해석

이 절에서는 본 문서에 등장하는 주요 수식이 실제로 무엇을 의미하는지,
"왜 이 수식이 여기에 필요한지"를 실무자 관점에서 해석한다.

### PLE 게이팅 결합

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

**해석**: "태스크 $k$의 표현은, 공용 전문가들의 의견을 가중 합산하고,
자기 전용 전문가의 의견도 가중 합산한 뒤, 둘을 더한 것이다."

첫째 합 $\sum g_{k,i}^s \cdot \mathbf{e}_i^s$은 *공유 지식* 에서 태스크
$k$에 유용한 부분만 골라내는 것이고, 둘째 합
$\sum g_{k,j}^k \cdot \mathbf{e}_j^k$는 *전용 지식* 에서 특화 정보를
가져오는 것이다. gate 값 $g$가 클수록 해당 Expert의 발언권이 크다.

### CGC Attention 가중치

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

**해석**: "현재 입력($\mathbf{h}_{shared}$)을 보고 8개 Expert 각각의
관련성 점수를 매긴 뒤, Softmax로 비율화한다. Softmax를 거치면 합이
1이 되므로, 이것은 곧 '태스크 $k$가 현재 입력에 대해 Expert들에게
부여하는 주의(attention) 분포'다."

초기 bias($\mathbf{b}_k$)의 역할이 중요하다. `domain_experts`로 지정된
Expert에는 `bias_high=1.0`을, 나머지에는 `bias_low=-1.0`을 설정하여,
학습 초기에 이미 도메인 지식에 부합하는 Expert에 주의가 집중되도록
유도한다. 학습이 진행되면서 $\mathbf{W}_k$가 업데이트되어 데이터에 맞게
수정된다.

### Entropy 정규화

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left(-\frac{1}{|\mathcal{T}|}\right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k), \quad H(\mathbf{w}_k) = -\sum_{i=1}^{8} w_{k,i} \cdot \log(w_{k,i})$$

**해석**: "gate 분포의 엔트로피가 낮으면(한 Expert에 집중) 페널티를
부여하여, 최소한 여러 Expert를 고르게 활용하도록 유도한다."

왜 필요한가? 엔트로피 정규화 없이 학습하면, gate가 가장 gradient가 큰
Expert 하나에 빠르게 수렴하여 나머지 Expert의 학습이 멈추는 *Expert
Collapse* 가 발생한다. 이 손실 항은 "모든 전문의의 소견을 최소한
참고는 하라"는 제약이다.

### Focal Loss

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

**해석**: "이미 잘 맞추고 있는 쉬운 예제($p_t \approx 1$)의 손실은
거의 0으로 줄이고, 틀리고 있는 어려운 예제($p_t \approx 0$)에 학습
에너지를 집중한다."

$(1 - p_t)^\gamma$ 항이 핵심이다. $p_t = 0.9$(잘 맞추는 경우)이면
$(1 - 0.9)^2 = 0.01$로 가중치가 100분의 1이 된다. $p_t = 0.1$(틀리는
경우)이면 $(1 - 0.1)^2 = 0.81$로 가중치가 거의 유지된다. $\gamma$가
클수록 쉬운 예제의 감쇠가 강해진다.

$\alpha_t$는 클래스 불균형을 보정한다. Churn 태스크의 $\alpha = 0.6$은
"이탈 고객을 놓치는 비용이 비이탈을 잘못 예측하는 비용보다 크므로,
양성(이탈) 예제에 더 높은 가중치를 부여한다"는 비즈니스 판단을 인코딩한
것이다.

> **학부 수학 — 거듭제곱 $(1-p_t)^\gamma$의 감쇠 효과.** 0과 1 사이의
> 수를 거듭제곱하면 지수가 클수록 값이 빠르게 0에 가까워진다. 이것이
> Focal Loss의 핵심 메커니즘이다. $\gamma = 0$이면 $(1-p_t)^0 = 1$로
> Focal Loss $=$ 표준 Cross-Entropy (감쇠 없음). $\gamma = 1$이면 선형
> 감쇠: $p_t = 0.9 \to 0.1$, $p_t = 0.5 \to 0.5$. $\gamma = 2$이면
> 제곱 감쇠: $p_t = 0.9 \to 0.01$, $p_t = 0.5 \to 0.25$. $\gamma = 5$
> 이면 급격 감쇠: $p_t = 0.9 \to 0.00001$, $p_t = 0.5 \to 0.03$. 즉
> $\gamma$가 클수록 "쉬운 예제(높은 $p_t$)를 더 강하게 무시"하고 어려운
> 예제에 학습을 집중한다. 실무적으로 $\gamma = 2$가 가장 널리
> 사용되며, 이는 "잘 맞추는 예제의 기여를 대략 100분의 1로 줄이는"
> 적당한 강도이다.

### Uncertainty Weighting

$$\mathcal{L}_k^{uw} = w_k \cdot (\exp(-s_k) \cdot \mathcal{L}_k + s_k)$$

**해석**: "본질적으로 예측이 어려운 태스크(높은 불확실성)의 가중치를
자동으로 낮추고, 쉬운 태스크의 가중치를 높인다. 이 균형을 모델이
스스로 학습한다."

$s_k = \log(\sigma_k^2)$는 태스크 $k$의 *학습 가능한 불확실성* 이다.

- $s_k$가 크면(불확실성 높음): $\exp(-s_k)$가 작아져 손실 기여가 줄어든다.
  동시에 $+s_k$ 항이 커져서 $s_k$를 무한히 키우는 것을 방지한다.
- $s_k$가 작으면(불확실성 낮음): $\exp(-s_k)$가 커져 손실을 적극 반영한다.

이 메커니즘 덕분에 수동으로 태스크 가중치를 튜닝할 필요가 줄어든다.
16개 태스크의 가중치를 일일이 조정하는 것은 조합 폭발이지만,
Uncertainty Weighting은 이를 *자동 균형* 으로 대체한다.

> **학부 수학 — $\exp$와 $\log$는 왜 짝으로 나타나는가?** $\exp(x) = e^x$
> 와 $\log(x) = \ln(x)$는 역함수 관계이다: $\exp(\log(x)) = x$,
> $\log(\exp(x)) = x$. Uncertainty Weighting에서
> $s_k = \log(\sigma_k^2)$로 정의하고 $\exp(-s_k)$를 사용하는 이유는:
> (1) $\sigma_k^2$(분산)은 반드시 양수여야 하는데, $s_k$는 제약 없는
> 실수이므로 최적화가 쉽다. (2)
> $\exp(-s_k) = \exp(-\log(\sigma_k^2)) = 1/\sigma_k^2$이므로
> precision(정밀도)이 된다. *구체적 계산*: $s_k = 0$이면
> $\exp(-s_k) = 1$ (표준 손실 그대로 반영). $s_k = 2$이면
> $\exp(-2) \approx 0.135$ (손실의 13.5%만 반영 — 불확실한 태스크
> 감쇠). $s_k = -1$이면 $\exp(1) \approx 2.718$ (손실 2.7배 증폭 —
> 확실한 태스크 강조). 동시에 $+s_k$ 정규화 항은 $s_k = 2$일 때 $+2$를
> 더해 "불확실하다고 선언하는 데 비용이 든다"는 제약을 건다.

### Evidential Uncertainty

$$u = \frac{K}{S}, \quad S = \sum_{k=1}^{K} \alpha_k$$

**해석**: "모델이 증거(evidence)를 많이 모았으면 확신하고($S$ 큼, $u$
작음), 증거가 부족하면 '모르겠다'고 솔직하게 말한다($S$ 작음, $u$
큼)."

기존 Softmax는 어떤 입력이든 항상 확률 분포를 출력한다. 학습 분포에서
벗어난(out-of-distribution) 데이터에도 자신 있게 예측하여 위험한 결정을
내릴 수 있다. Evidential 접근은 "확률의 확률" — 즉 Dirichlet 분포 —
를 모델링하여, 예측 자체의 불확실성까지 정량화한다.

### Soft Routing

$$\mathbf{e}_{cluster} = \sum_{c=0}^{19} p_c \cdot \mathbf{E}_c \in \mathbb{R}^{32}$$

**해석**: "GMM 클러스터 경계에 있는 고객은 하나의 클러스터에 강제
배정하지 않고, 여러 클러스터 임베딩 벡터를 소속 확률에 비례하여
혼합한다. 혼합된 임베딩이 GroupEncoder 출력과 결합되어 TaskHead를
통과한다."

이것은 hard clustering의 불연속성 문제를 해결한다. 클러스터 0과 1의
경계에 있는 고객이 id=0으로 배정되면, 클러스터 1의 지식을 전혀 활용하지
못한다. Soft routing은 $p_0 = 0.6$, $p_1 = 0.4$처럼 임베딩을 비례
혼합하여 경계 고객의 표현을 더 안정적으로 만든다.

## 전체 내러티브 — "왜 PLE인가"

### 이야기의 흐름

**시작점**: 16개 태스크를 동시에 예측해야 한다. 독립 모델 16개를 만들면
데이터 효율이 낮고, 공통 패턴을 활용하지 못한다.

**첫 시도 (Shared-Bottom)**: 하나의 공유 네트워크를 만들었다. 일부
태스크는 성능이 올라갔지만, CTR과 Churn처럼 본질적으로 다른 태스크가
서로의 학습을 방해하는 *Negative Transfer* 가 발생했다.

**두 번째 시도 (MMoE)**: Expert를 여러 개 두고 gate로 선택하게 했다.
하지만 모든 gate가 같은 Expert를 선택하는 *Expert Collapse* 가
발생하여, 사실상 Shared-Bottom과 다를 바 없어졌다.

**해결 (PLE)**: Expert를 공유(Shared)와 전용(Task-specific)으로
*명시적으로 분리* 하고, CGC gate가 두 종류의 Expert를 최적 비율로
결합하게 했다. 공유 Expert는 모든 태스크에 유용한 기본 지식을, 전용
Expert는 각 태스크에만 필요한 특화 지식을 담당한다.

**확장 (본 프로젝트)**: PLE의 아이디어 위에 8개 이종 도메인
Expert(GCN, TDA, DeepFM, Temporal, Graph, Causal, OT, RawScale),
GroupEncoder $+$ ClusterEmbedding(4 그룹, 20 클러스터), HMM Triple-Mode
라우팅, Logit Transfer 체인, Evidential 불확실성, SAE 해석성을 추가하여
*AIOps 추천 시스템에 특화된 PLE-Cluster-adaTT* 를 완성하였다.

### 설계 원칙 요약

| 원칙 | 구현 |
|---|---|
| 공유와 분리의 균형 | 8 Shared Expert + CGC gate + GroupTaskExpertBasket |
| 태스크 간 간섭 최소화 | Expert 분리 + Entropy 정규화 + domain_experts bias |
| 태스크 간 지식 전달 | Logit Transfer (명시적) + adaTT gradient 전이 (적응적) |
| 클러스터별 특화 | GroupEncoder + ClusterEmbedding + Soft Routing |
| 시간 스케일 분리 | HMM Triple-Mode (daily / monthly) |
| 불확실성 인식 | Evidential Deep Learning (Dirichlet) |
| 자동 균형 | Uncertainty Weighting (태스크 가중치 자동 조정) |
| 해석 가능성 | SAE (2304D sparse latent) |

## PLE 이론적 배경

### 논문 참조

**[RecSys 2020]** Tang, H., Liu, J., Zhao, M., & Gong, X. *"Progressive
Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for
Personalized Recommendations."*

### Negative Transfer 문제

멀티태스크 학습(MTL)에서 가장 심각한 문제는 *Negative Transfer* 이다.
관련성이 낮은 태스크가 표현 공간을 오염시켜, 단일 태스크 학습보다
오히려 성능이 하락하는 현상이다.

> **⚠ Negative Transfer의 실제 영향.** AIOps 시스템에서 CTR(클릭률)과
> Churn(이탈) 태스크는 본질적으로 다른 패턴을 학습해야 한다. CTR은 단기
> 참여도, Churn은 장기 이탈 신호에 집중한다. Shared-Bottom 구조에서 이
> 두 태스크가 동일한 표현을 공유하면, 한쪽 gradient가 다른 쪽의 학습을
> 방해하는 *seesaw 현상* 이 발생한다.

### Shared-Bottom, MMoE, PLE 비교

| 구분 | Shared-Bottom | MMoE | PLE |
|---|---|---|---|
| Expert 구조 | 단일 Shared trunk | N개 Expert 전체 공유 | Shared Expert + Task-specific Expert 명시 분리 |
| 게이팅 | 없음 | 태스크별 Softmax gate | CGC: Shared + Task Expert 결합 gate |
| Negative Transfer | 높음 (모든 태스크가 간섭) | 중간 (Expert Collapse 가능) | 낮음 (명시적 분리로 간섭 최소화) |
| Expert Collapse | 해당 없음 | 높음 (모든 태스크가 동일 Expert 선택) | 낮음 (Shared/Task Expert 분리) |
| 확장성 | 제한적 | Expert 수 증가로 대응 | Extraction Layer 추가로 대응 |

### PLE 게이팅 공식

PLE에서 태스크 $k$의 출력은 Shared Expert 집합 $\mathcal{E}^s$와
Task-specific Expert 집합 $\mathcal{E}^k$의 게이팅 결합으로 결정된다.

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

- $\mathbf{e}_i^s$: $i$번째 Shared Expert 출력, $\mathbf{e}_j^k$: $k$ 태스크의 $j$번째 Task Expert 출력
- $g_{k,i}^s, g_{k,j}^k$: CGC gate 가중치 (Softmax 정규화)

> **본 프로젝트에서의 PLE 변형.** 원 논문의 PLE는 Task-specific Expert를
> 태스크당 독립 MLP로 구현하지만, 본 시스템에서는 *CGC Gate가 Shared
> Expert 출력 블록에 스케일 가중치를 적용* 하고, 그 결과를
> *GroupTaskExpertBasket(4 GroupEncoder + ClusterEmbedding)* 으로
> 전달하는 2단 구조로 변형하였다. Task-specific Expert의 역할을
> GroupTaskExpertBasket이 대신하며, 그룹 내 파라미터 공유 + 클러스터별
> 특화를 동시에 달성한다.
