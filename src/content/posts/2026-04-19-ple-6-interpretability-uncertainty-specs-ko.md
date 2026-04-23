---
title: "[Study Thread] PLE-6 — 해석성·불확실성·전체 사양"
date: 2026-04-19 17:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, sae, uncertainty, evidential, specs]
lang: ko
excerpt: "PLE 서브스레드 마지막 — 전문가 해석성을 위한 Sparse Autoencoder, 예측별 불확실성을 정량화하는 Evidential Deep Learning, 18개 태스크 전체 사양과 논문 대 구현 비교. 56쪽 PLE 기술 참조서 PDF 첨부."
series: study-thread
part: 6
alt_lang: /2026/04/19/ple-6-interpretability-uncertainty-specs-en/
next_title: "ADATT-1 — adaTT 동기: 적응형 타워와 Transformer Attention 의 유사성"
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
벡터는 사람이 직접 읽기 어렵고, 단순 활성화 패턴 분석으로는 "이 뉴런이 뭘
의미하는가" 에 답이 나오지 않는다.

**둘째, 예측을 얼마나 믿을 수 있는지 모른다.** Softmax 는 어떤 입력이든
항상 확률 분포를 출력한다. 학습 분포를 벗어난 out-of-distribution
데이터에도 자신 있게 "70% 이탈 확률" 을 내놓는다. 금융 의사결정 — 여신,
리스크, 신용 조치 — 에서 과신(overconfidence)은 법적·재무적 부담이다. 최소한
"이 예측을 믿지 말고 fallback 규칙으로 가라" 는 신호를 낼 수 있어야
한다.

두 질문에 순서대로 답한다. 중요한 건 두 답변이 모두 *메인 예측 경로에
영향을 주지 않는 방식* 으로 붙는다는 점이다. 해석성과 불확실성은 분석
도구이지 예측 도구가 아니다.

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
  $k$ 가 Expert $i$ 에 얼마나 주의를 주는가" 는 직접 확인할 수 있다. 하지만 Expert
  *내부* 에서 어떤 개념이 활성화되는지는 여전히 불투명하다.
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
학습 dynamics 에는 관여하지 않는다. `loss_weight=0.01` 은 이 영향의
크기를 제한하는 추가 안전장치다.

> **SAE latent 활용.** `PLEClusterOutput.sae_latent` (2048D sparse
> vector) 는 추론 후 *Expert Neuron Dashboard* 에서 활성화 패턴 분석에
> 사용된다. 예를 들어 "자주 활성화되는 latent #147 은 '카드론 이용
> 패턴' 에 대응" 과 같은 해석이 가능하다. EU AI Act 같은 설명 가능성
> 요건이 강화되는 맥락에서 이런 분해 가능한 표현을 유지하는 것 자체가
> 가치 있다.

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

상세 스펙 (18-태스크 전체 config, 파라미터 카운트, 학습 하이퍼파라미터,
디버깅 가이드, 코드 파일 매핑, Config 섹션 맵 등) 은 아래 PDF 에 모두
담았다. 블로그는 여기서 멈추고, 실제 운용에 필요한 세부 내용은 PDF 에서
확인하는 것이 깔끔하다.

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
