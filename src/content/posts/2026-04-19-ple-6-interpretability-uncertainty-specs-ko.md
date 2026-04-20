---
title: "[Study Thread] PLE-6 — 해석성·불확실성·전체 사양"
date: 2026-04-19 17:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, sae, uncertainty, evidential, specs]
lang: ko
series: study-thread
part: 6
alt_lang: /2026/04/19/ple-6-interpretability-uncertainty-specs-en/
next_title: "ADATT-1 — adaTT 동기: 적응형 타워와 Transformer Attention 유비"
next_desc: "adaTT 서브스레드 시작. 고정 타워의 한계에서 출발하는 '적응형 타워'의 동기, Transformer Attention이 왜 태스크 적응에 적합한 메커니즘인지, 그리고 조건부 계산·Hypernetwork 계보에서 adaTT가 어디에 위치하는지."
next_status: published
source_url: /PLE_기술_참조서.pdf
source_label: "PLE 기술 참조서 (KO, PDF · 56 pages)"
---

*"Study Thread" 시리즈의 PLE 서브스레드 마지막 6편. 영문/국문 병렬로
PLE-1 → PLE-6 에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학
기초를 정리해왔다. PLE-5 에서 시스템이 구조적으로 완성됐다. 학습이 되고
예측이 나온다. 그러나 두 가지 질문이 남는다 — "Expert 가 실제로 무엇을
학습했는지 볼 수 있는가", 그리고 "예측 신뢰도를 정량화할 수 있는가".
이번 6편은 그 두 질문에 대한 응답이고, 이어서 전체 사양을 reference 로
정리하고, 마지막으로 PDF 다운로드로 시리즈를 마무리한다.*

## PLE-5 가 남긴 두 질문

구조는 끝났다. CGC 가 Expert 를 안정적으로 고른다. GroupTaskExpertBasket
이 클러스터별 특성화를 다룬다. Logit Transfer 가 순차 의존성을 전달한다.
Uncertainty Weighting 이 16개 손실 스케일을 자동 균형한다. 모델이 돌고,
예측이 나오고, 서빙에 올릴 수 있다.

그런데 여기서 두 가지가 찜찜하게 남는다.

**첫째, Expert 가 뭘 학습했는지 정말 아는가.** PLE 의 핵심 설계 베팅은
"이종 Expert 가 서로 보완적인 것을 학습할 것이다" 였다. gate 가중치가
분산되어 있는 것 (entropy 정규화 덕분에) 은 확인 가능하지만, 그게 각
Expert 가 *의미 있게 다른* 것을 학습했음을 보증하지는 않는다. 7개가
비슷한 패턴을 다른 좌표계로 재표현하고 있을 수도 있다. 512D concat
벡터는 인간이 읽기 어렵고, 단순 활성화 패턴 분석으로는 "이 뉴런이 뭘
의미하는가" 에 답이 안 나온다.

**둘째, 예측을 얼마나 믿을 수 있는지 모른다.** Softmax 는 어떤 입력이든
항상 확률 분포를 출력한다. 학습 분포에서 벗어난 out-of-distribution
데이터에도 자신 있게 "70% 이탈 확률" 을 내놓는다. 금융 의사결정 — 여신,
리스크, 신용 조치 — 에서 overconfidence 는 법적·재무적 부담이다. 최소한
"이 예측을 믿지 말고 fallback 규칙으로 가라" 는 신호를 낼 수 있어야
한다.

두 질문에 순서대로 답한다. 중요한 건 두 답변이 모두 *메인 예측 경로에
영향을 주지 않는 방식* 으로 붙는다는 점이다. 해석성과 불확실성은 분석
도구지 예측 도구가 아니다.

## 결정 1 — Expert 해석성을 위한 Sparse Autoencoder

### 문제 — 512D concat 을 어떻게 읽는가

7개 Expert 의 출력을 concat 한 512D 표현이 학습 이후 무엇을 담고 있는지
직접 보기는 어렵다. 차원 하나하나는 보통 여러 개념의 섞인 활성이다 ("이
뉴런은 50% 는 고가치 고객 신호, 30% 는 계절성, 20% 는 브랜드 선호" 같은
뒤섞임이 흔하다 — polysemantic representation).

어떻게 풀 것인가 — 몇 가지 대안:

- **PCA / ICA.** 선형 방법이라 비선형 뉴럴넷 표현에서 의미 있는 분해가
  잘 안 나온다.
- **Attention heat-map 해석.** 이미 CGC gate 가중치가 있으니 "태스크
  $k$ 가 Expert $i$ 에 얼마나 주의를 주는가" 는 읽힌다. 하지만 Expert
  *내부* 에서 어떤 개념이 활성화되는지는 여전히 불투명.
- **Sparse Autoencoder (SAE).** 표현을 과완전 (overcomplete) latent
  공간으로 확장한 뒤 L1 제약으로 sparse 하게 분해하면, 각 latent
  unit 이 하나의 개념에 대응하는 monosemantic 표현을 얻을 수 있다.
  Anthropic 의 *Towards Monosemanticity* (Bricken et al., 2023) 가
  LLM 에서 이 접근의 실제 효용을 보였다.

세 번째를 택했다. 512D 를 2048D overcomplete latent 로 올리고 L1
정규화로 희소성을 유도한다.

### SAE 구조

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_{enc} \cdot \mathbf{h}_{shared} + \mathbf{b}_{enc}) \in \mathbb{R}^{2048}$$

$$\hat{\mathbf{h}} = \mathbf{W}_{dec} \cdot \mathbf{z} + \mathbf{b}_{dec} \in \mathbb{R}^{512}$$

$$\mathcal{L}_{SAE} = \|\mathbf{h}_{shared} - \hat{\mathbf{h}}\|_2^2 + \lambda_1 \|\mathbf{z}\|_1$$

- `expansion_factor=4`: latent\_dim = 512 × 4 = 2048
- `l1_lambda=0.001`: sparsity 유도
- `tied_weights=true`: $\mathbf{W}_{dec} = \mathbf{W}_{enc}^T$ (파라미터 절약)
- `loss_weight=0.01`: 총 손실에 기여하는 비율

> **수식 직관.** 첫째 식은 인코딩 — 512D 공유 표현을 4배 확장한 2048D
> 희소 벡터 $\mathbf{z}$ 로 변환한다. ReLU 덕분에 대부분의 원소가 0 이
> 되어, 활성화된 소수의 원소만이 "이 고객의 표현에서 어떤 개념이 켜져
> 있는가" 를 나타낸다. 둘째 식은 디코딩 — 희소 벡터에서 원래 512D 를
> 복원하여 정보 손실을 최소화한다. 셋째 식의 손실은 복원 오차 ($L_2$) 와
> 희소성 제약 ($L_1$) 의 합. 직관적으로 "가능한 적은 수의 개념으로
> 전문가 표현을 설명하되, 원래 정보를 잃지 말라" 의 균형이다.

> **학부 수학 — L1 vs L2 왜 L1 이 sparse 를 만드는가.** $\|\mathbf{z}\|_1
> = \sum_i |z_i|$ 를 최소화하면 많은 원소가 *정확히 0* 이 되는 sparse
> 해가 나온다. 기하학적으로 L1 ball 의 꼭짓점이 축 위에 있어서 제약
> 최적화 시 해가 축에 놓이기 쉽기 때문이다. L2 ball 은 구 모양이라
> 해가 고르게 퍼져서 0 이 잘 안 나온다. $\mathbf{z} = [3, 0, 0, 2, 0]$
> 같이 5개 개념 중 2개만 활성된 벡터는 L1 에서 자연스럽게 유도된다.

### 메인 경로 gradient 차단

SAE 는 분석 도구지 예측 경로의 일부가 아니다. 그래서 입력에
`shared_concat.detach()` 를 걸어서 SAE gradient 가 Shared Expert 를
업데이트하지 않게 한다. SAE 손실은 SAE 자체 파라미터만 학습하고, 메인
학습 dynamics 에는 관여하지 않는다. `loss_weight=0.01` 은 이 관성의
크기를 제한하는 추가 안전장치.

> **SAE latent 활용.** `PLEClusterOutput.sae_latent` (2048D sparse
> vector) 는 추론 후 *Expert Neuron Dashboard* 에서 활성화 패턴 분석에
> 사용된다. 예를 들어 "자주 활성화되는 latent #147 은 '카드론 이용
> 패턴' 에 대응" 과 같은 해석이 가능하다. EU AI Act 같은 설명 가능성
> 요건이 강화되는 맥락에서 이런 분해 가능한 표현을 유지하는 것 자체가
> 가치.

> **역사적 배경 — 오토인코더의 역사.** 오토인코더는 *Rumelhart, Hinton
> & Williams (1986)* 역전파 논문에서 "자기 복원을 학습하면 중간 은닉층에
> 유용한 표현이 형성된다" 는 관찰로 시작되었다. Denoising Autoencoder
> (Vincent et al., ICML 2008), VAE (Kingma & Welling, ICLR 2014) 로
> 발전. Sparse Autoencoder 는 *Andrew Ng* 이 2011 년 Stanford 강의에서
> 체계화하였고, Anthropic ("Towards Monosemanticity", Bricken et al.,
> 2023) 이 LLM 의 residual stream 에 적용하여 해석 가능한 특징을
> 추출하면서 재조명받았다.

## 결정 2 — Evidential Deep Learning 으로 epistemic uncertainty

### 문제 — Softmax 는 "모른다" 고 말 못 한다

Softmax 분류기는 어떤 입력이든 항상 확률 분포를 출력한다. Out-of-distribution
고객 — 학습 데이터 분포에서 벗어난 패턴 — 에도 자신 있게 "70% 이탈
확률" 을 내놓는다. 서빙 시 fallback 규칙으로 넘어가야 하는 경계 케이스를
감지할 신호가 없다.

대안들:

- **Monte Carlo Dropout (Gal & Ghahramani 2016).** 추론 시 dropout 을
  켜둔 채로 여러 번 돌려 예측 분산을 본다. 간단하지만 추론 비용이
  $N$ 배로 늘어나고 서빙 지연이 악화된다.
- **Deep Ensemble (Lakshminarayanan et al., 2017).** $N$ 개 모델을
  독립 학습하고 예측 분산을 본다. 가장 신뢰성 높지만 학습 비용이
  $N$ 배.
- **Evidential Deep Learning (Sensoy et al., NeurIPS 2018).** 확률
  자체를 Dirichlet 분포의 파라미터로 예측한다. 한 번의 forward pass
  로 "예측 + 그 예측의 불확실성" 을 동시에 얻는다.

세 번째를 택했다. 추론 비용이 표준 softmax 와 거의 같고, 불확실성
신호를 자연스럽게 내놓는다.

### 원리

분류 태스크에서 Softmax 출력 대신 *Dirichlet 분포의 파라미터* 를
예측한다.

$$\boldsymbol{\alpha} = \text{evidence} + 1 \quad (\boldsymbol{\alpha} \in \mathbb{R}^K_+)$$

$$S = \sum_{k=1}^K \alpha_k \quad (\text{Dirichlet strength})$$

$$\hat{p}_k = \alpha_k / S \quad (\text{expected probability})$$

$$u = K / S \quad (\text{epistemic uncertainty})$$

- $K$: 클래스 수, $S$ 클수록 확신, $u$ 클수록 불확실
- evidence 가 0 이면 $\boldsymbol{\alpha} = \mathbf{1}$ → 균등 분포 → 최대 불확실

> **수식 직관.** 기존 Softmax 분류기는 "어떤 입력이든 항상 하나의 확률
> 분포를 출력" 하여, 학습 데이터에 없던 패턴에도 자신 있게 예측하는
> 위험이 있다. Evidential 접근은 "확률 분포의 분포" (Dirichlet) 를
> 모델링한다. $\boldsymbol{\alpha}$ 는 Dirichlet 의 농도 파라미터로,
> 증거 (evidence) 가 많을수록 $S = \sum \alpha_k$ 가 커져 분포가
> 뾰족해지고 (확신), 불확실성 $u = K/S$ 가 줄어든다. "증거가 충분하면
> 확신하고, 증거가 없으면 솔직하게 모르겠다" 의 인식론적 불확실성
> 정량화.

> **학부 수학 — Dirichlet 분포.** $\text{Dir}(\mathbf{p} | \boldsymbol{\alpha})$
> 는 확률 심플렉스 위의 분포다. $\alpha_k$ 가 모두 1 이면 균등 분포,
> 모두 크면 중심 $(1/K, \dots, 1/K)$ 에 집중, 특정 $\alpha_k$ 만 크면
> 해당 클래스로 치우침. 예를 들어 $\boldsymbol{\alpha} = (10, 10, 10)$
> 이면 "세 클래스가 비슷하다고 확신", $\boldsymbol{\alpha} = (100, 1, 1)$
> 이면 "클래스 1 이 거의 확실하다고 확신". 신경망이 $\boldsymbol{\alpha}$
> 를 예측하면 "예측 확률의 분산" 까지 정량화된다.

> **역사적 배경.** Evidential Deep Learning 은 *Sensoy, Kaplan &
> Kandemir (NeurIPS 2018)* 이 Dempster-Shafer 증거 이론 (1968, 1976)
> 과 주관적 논리 (Subjective Logic, Jøsang 2016) 를 신경망에 접목하여
> 제안. 2020 년에는 Amini et al. 이 회귀 문제로 확장한 *Evidential
> Regression* 을 Normal-Inverse-Gamma (NIG) 분포로 제시. 2024-2025
> 년 현재 자율주행, 의료 진단, 금융 리스크 평가에서 불확실성 정량화가
> 규제 요건으로 부상하며 실무 채택이 가속.

### 구현 및 보조 손실

`_build_evidential_layers()` 에서 태스크별 `EvidentialLayer` 가
생성된다. Task Expert 출력 (32D) 에 병렬로 붙어서 $\boldsymbol{\alpha}$
를 예측하고, `compute_evidential_loss()` 가 보조 KL 손실을 가산한다.

$$\mathcal{L}_{evi} = \mathcal{L}_{task} + \lambda_{KL} \cdot \min(1, \text{epoch}/\text{anneal}) \cdot \text{KL}(\text{Dir}(\boldsymbol{\alpha}) \,\|\, \text{Dir}(\mathbf{1}))$$

- `kl_lambda=0.01`, `annealing_epochs=10`: 초기에는 KL 기여 작게 시작
- 학습 초반 KL 이 너무 강하면 모든 예측이 균등 분포로 수렴하는 문제 방지

> **수식 직관.** 이 손실은 두 부분이다. $\mathcal{L}_{task}$ 는 예측
> 정확도를 올리고, KL 항은 "증거가 없는 클래스의 $\alpha$ 를 1 (무정보
> 상태) 로 되돌리라" 는 압력이다. annealing 계수는 학습 초반에 KL
> 기여를 약하게 시작해서, 모델이 먼저 기본 분류 능력을 갖춘 후 불확실성
> 교정에 집중하게 한다 — "처음에는 정답 맞추기, 어느 정도 실력이
> 쌓이면 자신의 확신도를 정직하게 표현하는 법" 순서.

## 18 태스크 전체 사양 — reference

해석성과 불확실성까지 정리됐으면, 남은 것은 *이 모든 결정이 실제로
어떻게 세팅되어 있는가* 의 회계다. 아래는 operator 가 참조할 수 있는
reference 로, 새로운 결정이 아니라 앞 5편에서 내린 결정들의 결과값이다.

시스템에 정의된 전체 18개 태스크의 완전한 사양. 현재 16개가 활성화되어
있으며, uplift 와 category_uplift 은 비활성화 상태.

| 태스크 | 그룹 | Loss | dim | HMM 모드 | Weight | 활성화 |
|---|---|---|---|---|---|---|
| CTR | Engagement | Focal | 1 | journey | 1.0 | O |
| CVR | Engagement | Focal | 1 | journey | 1.5 | O |
| Engagement | Engagement | MSE | 1 | journey | 0.8 | O |
| Uplift | Engagement | MSE | 1 | journey | 1.0 | X |
| Churn | Lifecycle | Focal | 1 | lifecycle | 1.2 | O |
| Retention | Lifecycle | Focal | 1 | lifecycle | 1.0 | O |
| Life-stage | Lifecycle | NLL | 6 | lifecycle | 0.8 | O |
| LTV | Lifecycle | Huber | 1 | lifecycle | 1.5 | O |
| Balance\_util | Value | Huber | 1 | behavior | 1.0 | O |
| Channel | Value | NLL | 3 | behavior | 0.8 | O |
| Timing | Value | NLL | 28 | behavior | 0.8 | O |
| NBA | Consumption | NLL | 12 | behavior | 2.0 | O |
| Spending\_category | Consumption | NLL | 12 | behavior | 1.2 | O |
| Consumption\_cycle | Consumption | NLL | 7 | behavior | 0.8 | O |
| Spending\_bucket | Consumption | Huber | 1 | behavior | 0.8 | O |
| Category\_uplift | Consumption | MSE | 12 | behavior | 1.5 | X |
| Brand\_prediction | Consumption | InfoNCE | 128 | behavior | 2.0 | O |
| Merchant\_affinity | Consumption | Huber | 1 | behavior | 1.0 | O |

### 태스크 그룹 (adaTT config)

| 그룹 | 멤버 | intra 강도 | inter 강도 |
|---|---|---|---|
| Engagement | CTR, CVR, Engagement, (Uplift) | 0.8 | 0.3 |
| Lifecycle | Churn, Retention, Life-stage, LTV | 0.7 | 0.3 |
| Value | Balance\_util, Channel, Timing | 0.6 | 0.3 |
| Consumption | NBA, Spending\_category, Consumption\_cycle, Spending\_bucket, Merchant\_affinity, Brand\_prediction | 0.7 | 0.3 |

> **intra / inter 의미.** adaTT 의 *intra 강도* ($= 0.6 \sim 0.8$) 는
> 같은 그룹 내 태스크 간 gradient 전이 비율, *inter 강도* ($= 0.3$) 는
> 그룹 간 전이 비율. 같은 그룹 (예: CTR, CVR) 은 gradient 방향이 유사해
> 큰 전이가 유익하고, 다른 그룹 (예: CTR, Churn) 은 충돌 가능성이 있어
> 작은 전이가 안전하다. 상세 adaTT 수식은 별도 ADATT 서브스레드에서.

## 논문 vs 구현 비교

### PLE (Tang et al., 2020) 비교

| 항목 | 원 논문 → 본 구현 |
|---|---|
| Expert 구조 | Shared + Task MLP → 7개 도메인 이종 Expert (DeepFM · LightGCN · UHGCN · Temporal · PersLay · Causal · OT) |
| Extraction Layer | 다중 PLE Layer 스택 → 단일 레이어 (CGC → GroupTaskExpertBasket) |
| Task Expert | 태스크별 독립 MLP → GroupEncoder + ClusterEmbedding (20 clusters) |
| Gate | Shared+Task 단일 gate → 1단계 CGCLayer + 2단계 CGCAttention (블록 스케일링) |
| Knowledge Transfer | 암묵적 (Expert 공유) → 명시적 Logit Transfer + adaTT gradient |
| Cluster 특화 | 없음 → GMM 20-cluster 임베딩 + soft routing |
| HMM 라우팅 | 없음 → Triple-Mode (journey / lifecycle / behavior) |
| Loss Weighting | 고정 가중치 → Uncertainty Weighting (Kendall et al. 2018) |
| 불확실성 정량화 | 없음 → Evidential DL (Dirichlet posterior) |

### MMoE (Ma et al., KDD 2018) 비교

| 항목 | MMoE → 본 구현 |
|---|---|
| Expert 수 | 동일 구조 N개 → 이종 7개 (DeepFM · LightGCN · UHGCN · Temporal · PersLay · Causal · OT) |
| Expert 구조 | 동일 MLP → 각각 도메인 특화 아키텍처 |
| Gate | Linear(input → N) + Softmax → Linear(512 → 7) + Softmax (CGC) |
| Expert Collapse | 심각 (모든 태스크 동일 Expert 로 수렴) → 완화 (Entropy 정규화 + domain\_experts bias) |
| 초기 편향 | 없음 (무작위) → domain\_experts 기반 warm start |
| 태스크 특화 | gate 만으로 분리 → CGC + HMM routing + GroupTaskExpertBasket |

### 주요 아키텍처 혁신

본 프로젝트만의 고유한 설계 요소 — 앞 5편의 결정들을 한 줄씩 요약:

1. *이종 Expert 결합* (PLE-2, PLE-3): 단일 구조 대신 7개 이종 도메인 Expert
2. *CGC 차원 정규화* (PLE-4): Expert 출력 차원 비대칭 (128D vs 64D) 보정
3. *HMM Triple-Mode 라우팅* (PLE-4): 태스크별 시간 스케일 맞춤 주입
4. *GroupTaskExpertBasket* (PLE-5): GroupEncoder + ClusterEmbedding 으로 88% 파라미터 감소
5. *Logit Transfer 체인* (PLE-5): 위상 정렬 기반 자동 실행 순서 도출
6. *Evidential + SAE* (PLE-6): 예측 불확실성 정량화 + Expert 표현 해석성

## 디버깅 가이드

### Expert 출력 진단

| 증상 | 원인 | 확인 방법 |
|---|---|---|
| 특정 Expert 출력 all-zero | 입력 데이터 None → zero fallback | `shared_expert_outputs` dict에서 해당 Expert 텐서 확인 |
| unified\_hgcn 출력 NaN | Poincare 좌표 overflow | `hierarchy_features` 값 범위 확인, curvature 조정 |
| temporal 출력 all-zero | `txn_seq` None | DataLoader의 시퀀스 로딩 활성화 확인 |
| perslay 출력 불안정 | Raw diagram 패딩 오류 | `tda_short_mask` 유효 비율 확인 |

### CGC Attention 분포 분석

| 증상 | 원인 | 해결 |
|---|---|---|
| 단일 Expert에 0.9+ 집중 | Expert Collapse | `entropy_lambda` 증가 (0.01→0.02) |
| 모든 Expert 균등 (~0.125) | CGC 미학습 또는 과도한 entropy | `entropy_lambda` 감소, 학습률 확인 |
| domain\_experts 외 Expert에 높은 가중치 | CGC가 domain 편향을 극복함 | 정상일 수 있음 — cross-domain transfer 패턴 |
| 학습 후반 attention 급변 | CGC freeze 미적용 | `freeze_epoch` 설정 확인 |

### Loss 관련 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| Loss NaN/Inf 발생 | fp16 underflow + focal loss | `.float().clamp(1e-7, 1-1e-7)` 확인 (M-2 FIX) |
| 특정 태스크 loss 0 | target 전체 -1 (결측) | `ignore_index=-1` 정상 동작, 데이터 확인 |
| Uncertainty weight 발산 | `task_log_vars` clamp 미적용 | clamp(-4.0, 4.0) 및 precision clamp 확인 |
| 총 loss 급증 | Evidential KL annealing 완료 | annealing\_epochs 이후 KL 기여 확인 |
| adaTT 이후 loss 급변 | Negative transfer 감지 | `negative_transfer_threshold` 조정 |

### Gradient Flow 진단

| 증상 | 원인 | 해결 |
|---|---|---|
| Shared Expert gradient 0 | Phase 2에서 freeze\_shared 활성 | `freeze_shared_in_phase2` 확인 |
| CGC gradient 0 (학습 중) | CGC frozen (freeze\_epoch 도달) | 의도적 고정 — 정상 |
| `_extract_task_gradients` OOM | retain\_graph=True 누적 | `adatt_grad_interval` 증가 (10→50) |
| Brand prediction gradient 약함 | InfoNCE temperature 너무 높음 | `temperature` 감소 (0.07→0.05) |

### HMM 관련 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| HMM 프로젝션 출력 동일 | 모든 샘플에 default embedding | HMM 파이프라인 데이터 생성 확인 |
| 특정 모드만 학습 | 다른 모드 대상 태스크가 적음 | `target_tasks` 균형 확인 |
| Default embedding이 학습 안 됨 | 대부분 샘플이 유효 HMM 보유 | 정상 (default는 소수에만 적용) |

### Logit Transfer 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| CVR이 CTR에 과의존 | `transfer_strength` 너무 높음 | 0.5→0.3 감소 |
| Transfer 미적용 | Source 태스크 비활성화 | Source가 `self.task_names`에 포함 확인 |
| 실행 순서 오류 | 위상 정렬 실패 → 폴백 | 로그에서 "하드코딩 폴백 사용" 경고 확인 |

## 부록

### 코드 파일 매핑

| 파일 | 역할 |
|---|---|
| `models/ple_cluster_adatt.py` | PLEClusterAdaTT 메인 모델 (~2125 라인) |
| `models/experts/registry.py` | ExpertRegistry, SharedExpertFactory |
| `models/experts/cluster_task_expert.py` | ClusterTaskExpertBasket, GroupTaskExpertBasket |
| `models/adatt.py` | AdaptiveTaskTransfer (gradient 기반 전이) |
| `models/layers/sae_layer.py` | SparseAutoencoder |
| `models/layers/evidential_layer.py` | EvidentialLayer (Dirichlet/Beta/NIG) |
| `models/tasks/task_registry.py` | TaskRegistry, TaskManager, TASK\_GROUPS |
| `models/tasks/base_task.py` | BaseTask, TaskConfig, TaskOutput, TaskType |
| `models/tasks/classification_tasks.py` | CTR, CVR, Churn, Retention, NBA, ... |
| `models/tasks/regression_tasks.py` | Engagement, BalanceUtil, LTV, Uplift |
| `models/tasks/merchant_tasks.py` | BrandPrediction, MerchantAffinity, ContrastiveLoss |
| `configs/model_config.yaml` | 전체 모델 설정 (진실의 원천) |

### 파라미터 카운트 추정

| 모듈 | 파라미터 | 비고 |
|---|---|---|
| Unified H-GCN | ~200K | 128D output, merchant 계층 구조 |
| PersLay | ~50K | Raw diagram + global stats |
| DeepFM | ~169K | 필드별 독립 임베딩 |
| Temporal Ensemble | ~500K | Mamba + LNN + Transformer |
| LightGCN | ~20K | 사전 계산 임베딩 → 경량 |
| Causal | ~100K | NOTEARS DAG + 인과 인코더 |
| Optimal Transport | ~100K | Sinkhorn + 기준 분포 |
| CGC (16 태스크) | ~57K | 16 × Linear(512→7) |
| HMM 프로젝터 | ~5K | 3 × Linear(16→32) |
| GroupTaskExpertBasket | ~362K | 4 GroupEncoder × 20 clusters |
| Task Towers (16) | ~80K | 16 × MLP(32→64→32→out) |
| adaTT | ~10K | 전이 행렬 + affinity |
| SAE | ~2.1M | 512D × 4 expansion (분석 전용) |
| Evidential | ~30K | 16 × Linear(32→out) |
| Auxiliary 프로젝터 | ~40K | coldstart + anonymous + gate |
| **총계 (SAE 제외)** | **~1.65M** | 학습 대상 파라미터 |
| **총계 (SAE 포함)** | **~3.75M** | SAE는 detach (분석 전용) |

> **파라미터 카운트 확인 방법.** `model.summary()` 메서드가 모듈별
> 파라미터 수를 출력한다. 위 수치는 추정값이며 실제 값은 config 설정에
> 따라 달라질 수 있다.

### 학습 설정 요약

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (lr=0.0005, weight\_decay=0.01) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, Tmult=2) |
| Batch Size | 16384 |
| Max Epochs | 100 |
| Early Stopping | patience=7 |
| Gradient Clipping | 5.0 |
| Mixed Precision | fp16 (AMP) |
| Phase 1 | Shared Expert 학습 (15 epochs) |
| Phase 2 | Cluster Subhead Fine-tuning (8 epochs, shared frozen) |
| adaTT Warmup | 0 epoch (프로덕션: 10) |
| adaTT Freeze | 1 epoch (프로덕션: 28) |
| CGC Freeze | adaTT freeze\_epoch과 동기화 |

### Config 핵심 경로

모델의 진실의 원천은 `configs/model_config.yaml` 이다.

| Config 섹션 | 설명 | 읽는 메서드 |
|---|---|---|
| `global` | 클러스터 수, dropout, input\_dim | `__init__` |
| `shared_experts` | 7개 Expert 설정 | `_build_shared_experts` |
| `cgc` | CGC 활성화, bias, entropy | `_build_cgc` |
| `hmm_triple_mode` | 3모드 라우팅, 대상 태스크 | `_build_hmm_projectors` |
| `task_experts.common` | GroupEncoder 설정 | `_build_task_experts` |
| `task_experts.tasks` | 16+2 태스크 개별 설정 | `_build_task_experts`, `_compute_task_losses` |
| `adatt` | 전이 강도, 태스크 그룹 | `_build_adatt` |
| `task_relationships` | Logit Transfer 쌍 | `_build_logit_transfer` |
| `task_towers` | Tower 구조, activation | `_build_task_towers` |
| `sae` | SAE 활성화, expansion | `_build_sae` |
| `evidential` | Evidential 활성화, KL | `_build_evidential_layers` |
| `training` | lr, batch, epochs, loss weighting | `__init__` (task\_log\_vars) |

## 전체 PLE 기술 참조서 다운로드

여기까지 PLE-1 부터 PLE-6 까지 온프렘 `기술참조서/PLE_기술_참조서` 를
블로그 형식으로 관통했다. 각 편이 이전 편의 해결책이 남긴 문제에서
출발해 다음 결정으로 이어지는 사슬이었다. 원본 PDF 는 조판 · 색인 ·
페이지 번호가 살아있는 긴 참조 문서다.

> **📄 [PLE 기술 참조서 전체 PDF 다운로드](/PLE_기술_참조서.pdf)** · KO · 약 56 페이지
>
> Progressive Layered Extraction · CGC Gate · GroupTaskExpertBasket ·
> Logit Transfer · 2-Phase Training — 본 프로젝트의 PLE 계열 아키텍처
> 전체 내용을 하나의 문서로 보고 싶으면 위 링크에서 받으면 된다.

## PLE 서브스레드 종료, adaTT 로

여기까지가 PLE 서브스레드의 끝이다. 각 편이 이전 편의 결정이 남긴
문제에서 출발한 사슬로 읽힌다.

- **PLE-1**: Shared-Bottom → MMoE — gradient 충돌에서 Expert Collapse 로.
- **PLE-2**: 명시적 Shared/Task 분리 + 이종 Expert + Softmax gate — MMoE Collapse 의 3중 해답.
- **PLE-3**: 7명의 전문가를 하나씩 — 각자가 메우는 빈틈과 환원 불가능성.
- **PLE-4**: Dim-asymmetry 와 시간 스케일 분리 — CGC 2단계 + HMM Triple-Mode.
- **PLE-5**: 메모리, 태스크 의존성, 손실 균형 — GroupTaskExpertBasket, Logit Transfer, Uncertainty Weighting.
- **PLE-6**: 해석성과 불확실성 — SAE 와 Evidential DL, 그리고 전체 사양 reference.

다음 **ADATT-1** 부터는 별도의 adaTT 서브스레드가 시작된다. 고정 타워의
한계에서 출발하는 "적응형 타워" 의 동기, Transformer Attention 이 왜
태스크 적응에 적합한 메커니즘인지, 그리고 조건부 계산 · Hypernetwork
계보에서 adaTT 가 어디에 위치하는지 — 이번에도 같은 형식으로, 이전
결정이 남긴 문제에서 다음 결정으로 이어지는 사슬을 따라간다.
