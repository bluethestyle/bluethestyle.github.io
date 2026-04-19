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
기초를 정리해왔다. 출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서`
이다. 이번 6편은 Expert 해석성(Sparse Autoencoder), 불확실성
정량화(Evidential Deep Learning), 18개 태스크 전체 사양, 논문 vs 구현
비교, 디버깅 가이드, 부록까지 — 그리고 이 글 하단에서 전체 PDF 를
다운로드 받을 수 있다. adaTT 는 다음 ADATT-1 부터 별도 서브스레드로
다룬다.*

## SAE (Sparse Autoencoder) — Expert 해석성

### 목적

Shared Expert 결합 표현(512D)에서 해석 가능한 sparse feature를
추출한다. Anthropic의 Sparse Autoencoder 접근법에서 영감을 받아,
신경망 내부 표현을 *인간이 해석 가능한 단위* 로 분해하는 것이 목적이다.

### 아키텍처

`_build_sae()` (라인 877~896)에서 `SparseAutoencoder`를 생성한다.

$$\mathbf{z} = \text{ReLU}(\mathbf{W}_{enc} \cdot \mathbf{h}_{shared} + \mathbf{b}_{enc}) \in \mathbb{R}^{2048}$$

$$\hat{\mathbf{h}} = \mathbf{W}_{dec} \cdot \mathbf{z} + \mathbf{b}_{dec} \in \mathbb{R}^{512}$$

$$\mathcal{L}_{SAE} = \|\mathbf{h}_{shared} - \hat{\mathbf{h}}\|_2^2 + \lambda_1 \|\mathbf{z}\|_1$$

- `expansion_factor=4`: latent\_dim = 512 × 4 = 2048
- `l1_lambda=0.001`: sparsity 유도
- `tied_weights=true`: $\mathbf{W}_{dec} = \mathbf{W}_{enc}^T$ (파라미터 절약)
- `loss_weight=0.01`: 총 손실에 기여하는 비율

> **수식 직관.** 첫째 식은 인코딩 — 512D 공유 표현을 4배 확장한 2048D
> 희소 벡터 $\mathbf{z}$ 로 변환한다. ReLU 덕분에 대부분의 원소가 0이
> 되어, 활성화된 소수의 원소만이 "이 고객의 표현에서 어떤 개념이 켜져
> 있는가"를 나타낸다. 둘째 식은 디코딩 — 희소 벡터에서 원래 512D를
> 복원하여 정보 손실을 최소화한다. 셋째 식의 손실은 복원 오차($L_2$)와
> 희소성 제약($L_1$)의 합이다. 직관적으로, "가능한 적은 수의 개념으로
> 전문가 표현을 설명하되, 원래 정보를 잃지 말라"는 두 목표의 균형이다.

> **학부 수학 — L1 노름과 L2 노름의 차이.**
> $\|\mathbf{z}\|_1 = \sum_i |z_i|$ 는 각 원소의 절댓값 합이고,
> $\|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 = \sum_i (h_i - \hat{h}_i)^2$
> 는 차이의 제곱합이다. L1 노름을 최소화하면 많은 원소가 *정확히 0*
> 이 되는 희소(sparse) 해를 유도한다. 이는 L1의 기하학적 성질
> 때문인데, L1 공의 꼭짓점이 축 위에 있어서 제약 하 최적화 시 해가 축
> 위(= 나머지 좌표 0)에 놓이기 쉽다. 반면 L2 노름은 원(구)의 형태로
> 해가 고르게 분산되어 0이 잘 나오지 않는다. 구체적 예시로
> $\mathbf{z} = [3, 0, 0, 2, 0]$ 이면 $\|\mathbf{z}\|_1 = 5$, 0이 아닌
> 원소가 2개뿐이다. 이렇게 희소한 $\mathbf{z}$ 는 "5개 개념 중 2개만
> 활성"이라는 해석이 가능하다.

> **역사적 배경 — 오토인코더의 역사: 차원 축소에서 해석성까지.**
> 오토인코더의 개념은 *Rumelhart, Hinton & Williams (1986)* 의 역전파
> 논문에서 "자기 자신을 복원하도록 학습하면 중간 은닉층에 유용한
> 표현이 형성된다"는 관찰로 시작되었다. 이후 *Vincent et al. (ICML
> 2008)* 의 Denoising Autoencoder, *Kingma & Welling (ICLR 2014)* 의
> Variational Autoencoder(VAE)로 발전하였다. *Sparse Autoencoder* 는
> 은닉 표현에 L1 페널티를 부여하여 소수의 뉴런만 활성화되도록 강제하는
> 변형으로, *Andrew Ng* 이 2011년 Stanford 강의에서 체계화하였다. 핵심
> 아이디어는 PCA(주성분 분석)와 유사하게 차원을 축소하되, 비선형
> 변환을 허용하고, *과완전(overcomplete)* 표현
> ($\dim(\mathbf{z}) > \dim(\mathbf{x})$)에서도 L1 희소성으로 의미
> 있는 특징을 추출할 수 있다는 점이다. 본 시스템의 SAE는 512D → 2048D
> 과완전 인코딩을 사용하여, 512차원에 뒤섞여 있는 Expert 정보를
> 2048개의 해석 가능한 단위로 "풀어헤치는" 역할을 한다.

> **최신 동향.** Sparse Autoencoder를 신경망 해석에 적용하는
> *기계적 해석성(Mechanistic Interpretability)* 은 2023년 Anthropic의
> 연구("Towards Monosemanticity", Bricken et al., 2023)에서 대규모
> 언어 모델의 잔차 스트림에 SAE를 적용하여 해석 가능한 특징(feature)을
> 추출한 것이 기폭제가 되었다. 2024-2025년에는 OpenAI, DeepMind,
> EleutherAI 등에서도 SAE 기반 해석을 활발히 연구 중이며, Templeton
> et al. (Anthropic, 2024)은 Claude 3 Sonnet에서 수백만 개의 해석
> 가능한 특징을 추출했다. 추천 시스템 분야에서도 모델의 내부 표현을
> SAE로 분해하여 "왜 이 상품을 추천했는가"를 설명하는 연구가 증가하고
> 있으며, EU AI Act(2024 발효)의 설명 가능성 요건이 이러한 추세를
> 가속화하고 있다.

### Main Path Gradient 차단

`forward()` (라인 1216)에서 `shared_concat.detach()`로 SAE 입력을
분리한다. SAE 손실은 SAE 자체 가중치만 업데이트하며, Shared Expert의
학습에 영향을 주지 않는다.

```python
# ple_cluster_adatt.py:1216
_, sae_latent, sae_loss = self.sae(shared_concat.detach())
```

> **SAE latent 활용.** `PLEClusterOutput.sae_latent` (2048D sparse
> vector)는 추론 후 *Expert Neuron Dashboard* 에서 활성화 패턴 분석에
> 사용된다. 예를 들어 "자주 활성화되는 latent #147은 '카드론 이용
> 패턴'에 대응"과 같은 해석이 가능하다.

## Evidential Deep Learning — 불확실성 정량화

### 목적

태스크 예측의 *epistemic uncertainty* (모델이 "얼마나 모르는지")를
정량화하여 추천 신뢰도를 평가한다. 높은 불확실성을 가진 예측은 서빙 시
*fallback 로직* 으로 전환된다.

### 원리 (Sensoy et al., NeurIPS 2018)

분류 태스크에서 Softmax 출력 대신 *Dirichlet 분포의 파라미터* 를
예측한다.

$$\boldsymbol{\alpha} = \text{evidence} + 1 \quad (\boldsymbol{\alpha} \in \mathbb{R}^K_+)$$

$$S = \sum_{k=1}^K \alpha_k \quad (\text{Dirichlet strength})$$

$$\hat{p}_k = \alpha_k / S \quad (\text{expected probability})$$

$$u = K / S \quad (\text{epistemic uncertainty})$$

- $K$: 클래스 수, $S$ 클수록 확신, $u$ 클수록 불확실
- evidence가 0이면 $\boldsymbol{\alpha} = \mathbf{1}$ → 균등 분포 → 최대 불확실

> **수식 직관.** 기존 Softmax 분류기는 "어떤 입력이든 항상 하나의 확률
> 분포를 출력"하여, 학습 데이터에 없던 패턴에도 자신 있게 예측하는
> 위험이 있다. Evidential 접근은 "확률 분포의 분포"(Dirichlet)를
> 모델링한다. $\boldsymbol{\alpha}$ 는 Dirichlet 분포의 농도
> 파라미터로, evidence(증거)가 많을수록 $S = \sum \alpha_k$ 가 커져
> 분포가 뾰족해지고(확신), 불확실성 $u = K/S$ 가 줄어든다. 직관적으로,
> "증거가 충분히 쌓이면 확신하고, 증거가 없으면 솔직하게 모르겠다고
> 말한다"는 인식론적 불확실성의 정량화다.

> **학부 수학 — Dirichlet 분포: "확률의 확률"을 모델링하는 분포.**
> Dirichlet 분포 $\text{Dir}(\mathbf{p} | \boldsymbol{\alpha})$ 는
> 확률 심플렉스(simplex) 위의 분포다. $K$-차원 확률 벡터
> $\mathbf{p} = (p_1, \dots, p_K)$ ($\sum p_k = 1$, $p_k \geq 0$)를
> 생성하며, 확률 밀도 함수는
> $f(\mathbf{p} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_k)}{\prod \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}$
> 이다. $\Gamma(n)$ 은 감마 함수로, 자연수에서는 $\Gamma(n) = (n-1)!$
> (팩토리얼의 연속 확장)이다. 직관적 이해로는 $\alpha_k$ 가 모두 1이면
> 균등 분포(어떤 $\mathbf{p}$ 든 동등), $\alpha_k$ 가 모두 크면 중심
> $(1/K, \dots, 1/K)$ 근처에 집중(확신), 특정 $\alpha_k$ 만 크면 해당
> 클래스 쪽으로 치우침. 구체적 예시 ($K = 3$) 로
> $\boldsymbol{\alpha} = (1, 1, 1)$ 이면 삼각형 위에 균일하게
> 분포한다. $\boldsymbol{\alpha} = (10, 10, 10)$ 이면
> $(1/3, 1/3, 1/3)$ 근처에 집중 — "세 클래스 확률이 비슷하다고 확신".
> $\boldsymbol{\alpha} = (100, 1, 1)$ 이면 $(1, 0, 0)$ 근처에 집중 —
> "클래스 1이 거의 확실하다고 확신". 이처럼 Dirichlet 분포의
> $\boldsymbol{\alpha}$ 를 신경망이 예측하면, "예측 확률 자체의
> 분산"까지 정량화하여 불확실성을 표현할 수 있다.

> **역사적 배경.** Evidential Deep Learning은 *Sensoy, Kaplan &
> Kandemir (NeurIPS 2018)* 이 제안하였다. 이들은 Dempster-Shafer 증거
> 이론(1968, 1976)과 주관적 논리(Subjective Logic, Jøsang 2016)를
> 신경망에 접목하여, Softmax 출력의 과신(overconfidence) 문제를
> 해결하고자 했다. 핵심 아이디어는 "확률의 확률"을 모델링하는 것으로,
> 베이지안 통계에서 사후 분포(posterior)에 Dirichlet prior를 놓는
> 전통(Ferguson, 1973)에서 영감을 받았다. 이후 Amini et al. (NeurIPS
> 2020)이 회귀 문제로 확장한 *Evidential Regression* 을 제안하여
> Normal-Inverse-Gamma(NIG) 분포로 연속값 예측의 불확실성을
> 정량화하였다.

> **최신 동향.** 2024-2025년 Evidential DL 분야는 *교정(calibration)
> 개선* 과 *OOD(Out-of-Distribution) 탐지 성능 향상* 에 집중되고
> 있다. Pandey & Yu (AAAI 2023)의 Posterior Network과 Charpentier
> et al.의 Natural Posterior Network이 Normalizing Flow를 결합하여
> 증거 추정의 정확도를 높였다. 산업적으로는 자율주행(Waymo, 2024),
> 의료 진단(Google Health), 금융 리스크 평가에서 모델 불확실성
> 정량화가 규제 요건으로 부상하면서 실무 채택이 가속화되고 있다. 특히
> LLM의 hallucination 감지에 evidential 접근을 적용하는 연구(Ren
> et al., 2024)도 주목받고 있다.

### 구현

`_build_evidential_layers()` (라인 898~931)에서 태스크별
`EvidentialLayer`를 생성한다.

```python
# ple_cluster_adatt.py:921-927
self.evidential_layers[task_name] = EvidentialLayer(
    input_dim=self.task_expert_output_dim,  # 32D
    task_type=task_type,
    output_dim=output_dim,
    kl_lambda=0.01,
    annealing_epochs=10,
)
```

Forward (라인 1253~1260)에서 Task Expert 출력(32D)에 병렬로 적용되며,
`compute_evidential_loss()` (라인 1838~1841)로 보조 KL 손실을 가산한다.

$$\mathcal{L}_{evi} = \mathcal{L}_{task} + \lambda_{KL} \cdot \min(1, \text{epoch}/\text{anneal}) \cdot \text{KL}(\text{Dir}(\boldsymbol{\alpha}) \,\|\, \text{Dir}(\mathbf{1}))$$

- `kl_lambda=0.01`, `annealing_epochs=10`: 초기에는 KL 기여 작게 시작
- 학습 초반 KL이 너무 강하면 모든 예측이 균등 분포로 수렴하는 문제 방지

> **수식 직관.** 이 손실은 두 부분으로 구성된다. 첫째, 원래 태스크
> 손실($\mathcal{L}_{task}$)은 예측 정확도를 높인다. 둘째, KL 항은
> "증거가 없는 클래스의 $\alpha$ 를 1(무정보 상태)로 되돌리라"는
> 압력이다. annealing 계수 $\min(1, \text{epoch}/\text{anneal})$ 은
> 학습 초반에 KL 기여를 약하게 시작하여 모델이 먼저 기본적인 분류
> 능력을 갖춘 후에 불확실성 교정에 집중하도록 한다. 직관적으로,
> "처음에는 정답 맞추기에 집중하고, 어느 정도 실력이 쌓이면 자신의
> 확신도를 정직하게 표현하는 법을 배워라"이다.

## 18 태스크 전체 사양

아래는 시스템에 정의된 전체 18개 태스크의 완전한 사양이다. 현재
16개가 활성화되어 있으며, uplift과 category\_uplift이 비활성화
상태이다.

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

> **학부 수학 — intra/inter 전이 강도의 의미.** adaTT의 *intra 강도*
> ($= 0.6 \sim 0.8$)는 같은 그룹 내 태스크 간 gradient 전이 비율이고,
> *inter 강도* ($= 0.3$)는 다른 그룹 태스크 간 전이 비율이다.
> 수학적으로 태스크 $i$ 의 gradient가 태스크 $j$ 에 전이되는 양은:
> $\mathbf{g}_j^{transferred} = \mathbf{g}_j + \alpha_{ij} \cdot \text{proj}(\mathbf{g}_i, \mathbf{g}_j)$
> 여기서
> $\text{proj}(\mathbf{g}_i, \mathbf{g}_j) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \cdot \mathbf{g}_j$
> 는 $\mathbf{g}_i$ 를 $\mathbf{g}_j$ 방향으로 사영(projection)한
> 것이다. *사영(projection)* 은 "벡터 $\mathbf{a}$ 에서 벡터
> $\mathbf{b}$ 방향 성분만 추출"하는 연산으로,
> $\text{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$
> 로 정의된다. 직관적으로, 같은 그룹(예: CTR과 CVR)은 gradient 방향이
> 유사하여 큰 전이($\alpha = 0.8$)가 유익하고, 다른 그룹(예: CTR과
> Churn)은 gradient가 충돌할 수 있어 작은 전이($\alpha = 0.3$)가
> 안전하다.

> **최신 동향 — Gradient 기반 멀티태스크 최적화의 최전선.** adaTT의
> gradient 기반 전이는 *PCGrad (Yu et al., NeurIPS 2020)* 에서 시작된
> 연구 흐름의 연장이다. PCGrad는 충돌하는 gradient를 서로의
> 법선(normal) 방향으로 사영하여 충돌을 제거하고, *CAGrad (Liu et al.,
> NeurIPS 2021)* 는 모든 태스크의 최소 개선을 보장하는 방향을 찾는다.
> *Nash-MTL (Navon et al., ICML 2022)* 은 이를 Nash 협상 게임으로
> 정식화하여 Pareto 최적 해를 유도하였다. 2024-2025년에는 *Aligned-MTL
> (Senushkin et al., CVPR 2023)* 이 gradient 행렬의 SVD를 이용해 정렬된
> 업데이트 방향을 찾고, *FairGrad (Mahapatra & Rajan, 2024)* 가 태스크
> 간 공정성까지 고려하는 방법을 제안하였다. 본 시스템의 adaTT는 이 중
> *그룹 구조* 를 명시적으로 활용하는 점이 차별적이며, intra/inter
> 강도를 별도로 설정하여 도메인 지식(예: CTR-CVR 관계)을 반영한다.

## 논문 vs 구현 비교

### PLE (Tang et al., 2020) 비교

| 항목 | 원 논문 | 본 구현 |
|---|---|---|
| Expert 구조 | Shared + Task-specific MLP | 7개 도메인 Shared Expert (GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT) |
| Extraction Layer | 다중 PLE Layer 스택 | 단일 레이어 (CGC → GroupTaskExpertBasket) |
| Task Expert | 태스크별 독립 MLP | GroupEncoder + ClusterEmbedding (20 clusters) |
| Gate | Shared+Task Expert → gate | Shared Expert 블록 스케일링 (512D 유지) |
| Knowledge Transfer | 암묵적 (Expert 공유) | 명시적 Logit Transfer + adaTT gradient 기반 |
| Cluster 특화 | 없음 | GMM 20-cluster 임베딩 + GroupEncoder |
| HMM 라우팅 | 없음 | Triple-Mode (journey/lifecycle/behavior) |
| Loss Weighting | 고정 | Uncertainty Weighting (Kendall et al.) |
| 불확실성 | 없음 | Evidential Deep Learning (Dirichlet) |

### MMoE (Ma et al., KDD 2018) 비교

| 항목 | MMoE | 본 구현 |
|---|---|---|
| Expert 수 | 동일 구조 N개 | 이종 7개 (GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT) |
| Expert 구조 | 동일 MLP | 각각 도메인 특화 아키텍처 |
| Gate | Linear(input → N) + Softmax | Linear(512 → 7) + Softmax (CGC) |
| Expert Collapse | 심각 (모든 태스크가 동일 Expert) | 완화 (Entropy 정규화 + domain\_experts bias) |
| 초기 편향 | 없음 (무작위) | domain\_experts 기반 warm start |
| 태스크 특화 | gate만으로 분리 | CGC + HMM routing + GroupTaskExpertBasket |

### 주요 아키텍처 혁신

본 프로젝트만의 고유한 설계 요소:

1. *이종 Expert 결합*: 단일 구조 Expert 대신 GCN, PersLay, DeepFM, Temporal, LightGCN, Causal, OT 등 7개 이종 도메인 Expert를 결합
2. *CGC 차원 정규화*: Expert 출력 차원 비대칭(128D vs 64D) 보정
3. *HMM Triple-Mode 라우팅*: 태스크별 시간 스케일에 맞는 HMM 모드 선택적 주입
4. *GroupTaskExpertBasket*: GroupEncoder + ClusterEmbedding으로 88% 파라미터 감소 (v3.2)
5. *Logit Transfer 체인*: 위상 정렬 기반 자동 실행 순서 도출
6. *Evidential + SAE*: 예측 불확실성 정량화 + Expert 표현 해석성

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
| DeepFM | ~169K | v3.11: 필드별 독립 임베딩 |
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

> **파라미터 카운트 확인 방법.** `model.summary()` 메서드 (라인
> 1967~2073)가 모듈별 파라미터 수를 출력한다. 위 수치는 추정값이며,
> 실제 값은 config 설정에 따라 달라질 수 있다. 정확한 수치는 모델
> 초기화 후 `summary()` 출력으로 확인한다.

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

> **학부 수학 — AdamW 옵티마이저의 수학적 구조.** AdamW는 *Loshchilov
> & Hutter (ICLR 2019)* 이 제안한 옵티마이저로, Adam에 *분리된 가중치
> 감쇠(decoupled weight decay)* 를 적용한 것이다. 기본 Adam의 파라미터
> 업데이트 규칙은:
> $\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$
> (1차 모멘트 = gradient 이동평균),
> $\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$
> (2차 모멘트 = gradient 제곱 이동평균),
> $\hat{\mathbf{m}}_t = \mathbf{m}_t / (1 - \beta_1^t)$,
> $\hat{\mathbf{v}}_t = \mathbf{v}_t / (1 - \beta_2^t)$ (편향 보정),
> $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$.
> 여기서 $\eta$ 는 학습률, $\beta_1 = 0.9$, $\beta_2 = 0.999$ 가
> 일반적이다. 직관은 $\hat{\mathbf{m}}_t$ 가 "어느 방향으로 가야
> 하는가"(1차 모멘트 = 관성), $\sqrt{\hat{\mathbf{v}}_t}$ 가 "이
> 방향의 gradient가 얼마나 변동하는가"(2차 모멘트 = 적응적 학습률).
> 변동이 큰 파라미터는 학습률을 자동으로 줄여 안정화한다. AdamW의 핵심
> 차이는 가중치 감쇠를 gradient가 아닌 파라미터 자체에 직접 적용하여
> $\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1}(1 - \eta \lambda) - \eta \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$
> ($\lambda = 0.01$ = `weight_decay`)로 L2 정규화를 올바르게 구현하는
> 것이다.

> **역사적 배경 — Cosine Annealing 학습률 스케줄러.** Cosine Annealing
> 은 *Loshchilov & Hutter (ICLR 2017)* 이 SGDR(Warm Restarts) 논문에서
> 제안하였다. 학습률을 코사인 함수로 감소시킨다:
> $\eta_t = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min})(1 + \cos(\pi t / T_0))$.
> 학습률이 주기적으로 최댓값으로 복원(warm restart)되어, 국소
> 최솟값(local minimum)에서 탈출할 기회를 반복적으로 제공한다.
> $T_0 = 10$ 은 첫 주기 길이, $T_{mult} = 2$ 는 주기가 매번 2배씩
> 늘어남을 의미한다 (10→20→40 에포크). 이 방식은 StepLR(계단식 감소)
> 보다 부드러운 전이를 제공하고, Exponential Decay보다 warm restart
> 덕분에 더 넓은 손실 경관을 탐색한다. 2020년 이후 대부분의 대규모
> 모델 학습에서 cosine 스케줄러가 표준으로 자리잡았으며, GPT-3, PaLM
> 등 LLM 학습에서도 warm-up + cosine decay 조합이 사용된다.

### Config 핵심 경로

모델의 진실의 원천(Single Source of Truth)은
`configs/model_config.yaml` 이다. 주요 섹션과 모델 메서드의 매핑:

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
블로그 형식으로 관통했다. 수식 설명, 역사적 배경, 구현 디테일까지 —
원본 PDF 는 조판·색인·페이지 번호가 모두 살아있는 긴 참조 문서다.

> **📄 [PLE 기술 참조서 전체 PDF 다운로드](/PLE_기술_참조서.pdf)** · KO · 약 56 페이지
>
> Progressive Layered Extraction · CGC Gate · GroupTaskExpertBasket ·
> Logit Transfer · 2-Phase Training — 본 프로젝트의 PLE 계열 아키텍처
> 전체 내용을 하나의 문서로 보고 싶으면 위 링크에서 받으면 된다.

## PLE 서브스레드 종료, adaTT 로

여기까지가 PLE 서브스레드의 끝이다. PLE-1 에서 Shared-Bottom 과 MMoE
의 한계에서 출발해, PLE-2 의 명시적 Shared/Task Expert 분리와 수학적
직관, PLE-3 의 입력 구조(PLEClusterInput · 734D)와 7개 이종 Shared
Expert Pool, PLE-4 의 CGC 게이팅 두 단계(1단계 CGCLayer가 Shared+Task
가중합, 2단계 CGCAttention이 Shared concat 블록 스케일링)와 HMM
Triple-Mode 라우팅, PLE-5 의 GroupTaskExpertBasket · Logit
Transfer · Task Tower 까지 — 그리고 이번 6편에서 해석성(SAE),
불확실성(Evidential), 18 태스크 사양, 논문 대비 구현의 혁신, 디버깅
가이드, 부록까지 — 본 프로젝트 PLE 계열 아키텍처의 모든 주요 구성
요소를 블로그 형식으로 정리했다.

다음 **ADATT-1** 부터는 별도의 adaTT 서브스레드가 시작된다. 고정
타워의 한계에서 출발하는 "적응형 타워"의 동기, Transformer Attention
이 왜 태스크 적응에 적합한 메커니즘인지, 그리고 조건부 계산 ·
Hypernetwork 계보에서 adaTT 가 어디에 위치하는지를 다룬다.
