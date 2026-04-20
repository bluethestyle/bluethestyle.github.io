---
title: "[Study Thread] PLE-5 — GroupTaskExpertBasket · Logit Transfer · Task Tower"
date: 2026-04-19 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, logit-transfer, task-tower, group-encoder]
lang: ko
series: study-thread
part: 5
alt_lang: /2026/04/19/ple-5-basket-logit-tower-en/
next_title: "PLE-6 — 해석성·불확실성·전체 사양 (+ 기술 참조서 PDF)"
next_desc: "Sparse Autoencoder(SAE)를 통한 Expert 해석성, Evidential Deep Learning의 불확실성 정량화, 18개 태스크 전체 사양, 논문 vs 구현 비교, 디버깅 가이드, 그리고 PLE 시리즈 마무리 — 전체 PLE 기술 참조서 PDF 다운로드 포함."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 5편. 영문/국문 병렬로 PLE-1 →
PLE-6 에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를
정리한다. 출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이다.
PLE-4 에서 CGC 게이팅과 HMM Triple-Mode 라우팅으로 Shared Expert pool
위에서의 라우팅을 정리했다. 그래도 세 가지 결정이 아직 열려있다 — 메모리,
태스크 간 의존성, 그리고 손실 균형. 이번 5편은 그 셋에 대한 응답이다.*

## PLE-4 에서 PLE-5 로 — 남은 세 결정

Gate 는 안정화됐다. 7명의 Expert 중 누구도 혼자 학습을 독점하지 않는다.
HMM 이 시간 스케일별로 신호를 올바른 태스크 그룹에 주입한다. 여기까지
좋다. 그런데 태스크별 *전용* 경로 쪽으로 넘어가면 세 가지 결정이 아직
남아있다.

**첫째, 메모리.** 레거시 구현은 태스크 × 클러스터마다 독립 MLP 를 두었다
— 16개 태스크 × 20개 클러스터 × 작은 MLP 각각 = 약 3M 파라미터. 대부분의
capacity 가 클러스터별로 중복되어 있었다. 클러스터 차이는 *입력 분포*
에서는 크지만 *결정 함수* 에서는 작다는 관찰이 이 낭비의 원인이다.

**둘째, 태스크 간 의존성이 아키텍처에 표현되지 않았다.** CTR 이 CVR
에 영향을 주고 Churn 이 Retention 에 영향을 주는 것은 비즈니스 로직상
자명하다. 하지만 CGC gate 는 "Expert 를 어떻게 섞을지" 만 다룬다.
네트워크가 "CTR 점수가 높은 고객은 CVR 점수도 높다" 는 관계를 스스로
발견하길 기다리는 것은 데이터 효율이 낮다.

**셋째, 16개 태스크의 손실 스케일 균형.** Focal Loss 의 CTR 과 MSE 의
Engagement 와 InfoNCE 의 Brand 가 동시에 역전파된다. 스케일 차이가
100 배 이상이다. 수작업으로 태스크별 가중치를 튜닝하는 것은 16차원
조합 폭발이고, 한 번 맞춰도 데이터 분포가 바뀌면 재튜닝이 필요하다.

PLE-5 는 이 세 결정을 순서대로 푼다.

## 결정 1 — GroupTaskExpertBasket: 88% 파라미터 감소

### 문제 — 클러스터당 독립 MLP 는 중복이다

레거시 `ClusterTaskExpertBasket` 은 태스크별로 20개 클러스터에 각자
독립 MLP 를 두었다. 직관은 "클러스터별로 다른 특성이 있으니 클러스터
전용 네트워크가 필요하다" 였다. 하지만 실험해보면 학습된 MLP 들이 상당히
비슷한 방향으로 수렴한다. 클러스터별 신호가 *결정 함수 전체를 바꿀 만큼*
크지는 않았다는 것 — 대부분의 고객 행동 패턴은 클러스터를 가로질러
공유된다.

그래서 설계를 뒤집었다. 같은 그룹 내 태스크들은 **GroupEncoder MLP 를
공유** 하고, 클러스터 정체성은 **32D Embedding** 으로 *주입* 한다.
Embedding 은 gating 이 아니라 입력의 일부로 들어가서, 네트워크가
클러스터별 어떻게 다르게 반응할지 자동 학습한다.

```mermaid
flowchart TB
  cgc[CGC output<br/>512D]
  hmm[HMM projection<br/>32D]
  cid[cluster_id] --> emb[Embedding table<br/>20 × 32D]
  cgc --> concat[concat<br/>576D]
  hmm --> concat
  emb --> concat
  concat --> mlp[Shared GroupEncoder MLP<br/>576 → 128 → 64 → 32]
  mlp --> out((h_expert · 32D))
  style cgc fill:#D8E0FF,stroke:#2E5BFF
  style hmm fill:#C9ECD9,stroke:#1C8C5A
  style emb fill:#FDD8D1,stroke:#E14F3A
  style out fill:#FFFFFF,stroke:#141414,stroke-width:2px
```

`use_group_encoder=true` (기본값) 설정 시 `GroupTaskExpertBasket` 을
사용하며, 레거시 `ClusterTaskExpertBasket` (태스크×클러스터 독립 MLP,
~3.0M 파라미터) 대비 **88% 파라미터 감소** (~362K) 를 달성한다. 같은
그룹 내 태스크는 GroupEncoder 를 공유하고, 그룹 간은 독립이다. 결정
함수의 중심은 공유 MLP 에서 나오고, 클러스터가 만드는 입력 분포 차이는
Embedding 이 담당한다.

### Soft Routing — 경계 고객의 연속성

GMM 클러스터 할당은 사후 확률이지만 기존 설계는 argmax 로 하나의
cluster_id 를 뽑았다. 이것은 경계 고객에게 연속성 문제를 만든다 —
클러스터 3 과 7 사이에 걸친 고객이 id=3 에 배정되면 id=7 의 지식을
전혀 못 쓴다. 그래서 사후 확률을 그대로 쓴다.

$$\mathbf{e}_{cluster} = \sum_{c=0}^{19} p_c \cdot \mathbf{E}_c \in \mathbb{R}^{32}$$

$$\mathbf{h}_{expert} = \text{TaskHead}([\text{GroupEncoder}(\mathbf{x}) \,\|\, \mathbf{e}_{cluster}])$$

$p_c$ 는 GMM 클러스터 $c$ 의 사후 확률, $\mathbf{E}_c$ 는 클러스터 $c$
의 학습 가능 임베딩 벡터(32D) 다. 구현은 `cluster_probs @ embedding.weight`
($[B, 20] \times [20, 32] = [B, 32]$) 한 번의 행렬곱이다.

> **수식 직관.** 클러스터 3 에 60%, 7 에 30% 식으로 소속된 경계 고객은
> 각 클러스터 임베딩이 그 비율대로 혼합되어 부드럽게 들어간다. 하나의
> 클러스터에 강제 배정하는 hard routing 과 달리 경계 고객 예측이 클러스터
> 할당 변동에 민감하지 않다.

> **Embedding.** $\text{Embedding}(c) = \mathbf{E}[c, :] \in \mathbb{R}^{32}$
> 는 one-hot $\mathbf{v}_c^T \mathbf{E}$ 와 수학적으로 동치인 학습 가능한
> 룩업 테이블이다. 인덱싱이 sparse 행렬곱보다 빠르다.

> **GMM 사후 확률.** $p_c = P(c | \mathbf{x}) = \pi_c \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \big/ \sum_j \pi_j \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$.
> $\pi_c, \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c$ 는 EM 으로 오프라인
> 사전 계산된다.

## 결정 2 — Logit Transfer: 태스크 간 명시적 의존성

### 문제 — 순차적 관계를 네트워크에게 찾게 하지 말자

CTR → CVR → LTV 는 비즈니스 퍼널 관계다. 클릭이 있어야 전환이 있고,
전환의 누적이 LTV 다. Churn → Retention 은 역상관. NBA → Spending_category
→ Brand_prediction 은 "다음 행동 → 어느 카테고리 → 어느 브랜드" 의 세분화
체인이다. 이것을 네트워크가 학습 데이터만으로 자동 발견하기를 기다리면
데이터 낭비다 — 우리가 이미 아는 것을 모델이 시간을 써서 재발견하는 셈.

어떻게 전달할 것인가 — 세 가지 대안이 있다.

- **선행 태스크의 logit 을 후행 태스크 feature 에 concat.** 단순하지만
  feature dim 을 바꾸고, 전이가 유용하지 않을 때 끄기 어렵다.
- **선행 태스크 최종 activation (sigmoid 후 확률) 을 후행에 gate 로.**
  유용하지만 gate 가 0 이 되면 전이 자체가 없어지는 all-or-nothing 구조.
- **선행 예측을 projection 해서 후행 입력에 residual 로 더한다.** 유용하지
  않으면 projection 가중치가 자연스럽게 0 으로 수렴해서 전이가 꺼진다 —
  safe default.

세 번째를 골랐다. He et al. (ResNet, CVPR 2016) 의 residual skip 과 같은
원리 — "가져다 쓸 정보가 없으면 identity mapping 에 수렴한다" 가 기본값.

### 전이 DAG

```mermaid
flowchart TB
  subgraph eng [Engagement chain]
    direction LR
    ctr[CTR] -->|sequential<br/>α=0.5| cvr[CVR]
    cvr -->|feature<br/>α=0.5| ltv[LTV]
  end
  subgraph ret [Retention chain]
    direction LR
    churn[Churn] -->|inverse<br/>α=0.5| retain[Retention]
  end
  subgraph cons [Consumption chain]
    direction LR
    nba[NBA] -->|feature<br/>α=0.5| scat[Spending_category]
    scat -->|feature<br/>α=0.5| brand[Brand_prediction]
  end
  eng ~~~ ret
  ret ~~~ cons
  style eng fill:#D8E0FF,stroke:#2E5BFF
  style ret fill:#FDD8D1,stroke:#E14F3A
  style cons fill:#C9ECD9,stroke:#1C8C5A
```

> **세 개의 독립 DAG.** Kahn's algorithm (1962, 진입 차수 0 → 큐, $O(V+E)$,
> 사이클 자동 감지) 이 위상 정렬로 실행 순서를 자동 도출한다 — CTR → CVR
> → LTV, Churn → Retention, NBA → Spending_category → Brand_prediction.
> 새 전이를 `task_relationships` config 에 등록하면 순서가 자동 갱신된다.

### 전이 메커니즘

선행 태스크의 예측을 후행 태스크 입력에 잔차로 더한다.

$$\mathbf{h}_{tower}^t = \mathbf{h}_{expert}^t + \alpha \cdot \text{SiLU}(\text{LayerNorm}(\text{Linear}(\text{pred}^s)))$$

$\alpha = 0.5$ (`transfer_strength`), `Linear` 는 source output_dim → 32D
projection 이다. 잔차 형태이므로 source 정보가 유용하지 않으면 프로젝션
가중치가 0 으로 수렴해 자연스러운 *safe default* 가 된다.

> **수식 직관.** CTR 모델이 "이 고객 클릭 확률이 높다" 를 출력하면 그
> 신호가 projection 을 거쳐 CVR 타워 입력에 더해진다. $\alpha = 0.5$ 는
> 전이 신호 대비 원래 Expert 출력의 상대적 강도를 조절한다.

> **⚠ Logit Transfer vs adaTT — 두 전이 메커니즘의 차이와 보완 관계.**
> 본 시스템은 태스크 간 지식 전달을 *두 가지 서로 다른 수준* 에서
> 동시에 수행한다.
>
> | 특성 | Logit Transfer | adaTT |
> | --- | --- | --- |
> | 작동 계층 | Feature/Logit 수준 (forward pass 중) | Loss 수준 (backward pass 전) |
> | 전달 내용 | 선행 태스크의 예측값/은닉 표현 | 태스크 간 gradient 친화도 |
> | 방향성 | 단방향 DAG (CTR→CVR→LTV) | 전방향 행렬 (모든 태스크 쌍) |
> | 학습 가능성 | 고정 구조 (수동 설계) | 적응적 (EMA 로 친화도 학습) |
> | 목적 | 순차적 의존성 명시적 전달 | Negative Transfer 자동 완화 |
>
> Logit Transfer 는 비즈니스 로직상 순차적인 태스크에 예측값을 직접
> 전달하고, adaTT 는 gradient 수준에서 모든 태스크 쌍의 상호 영향을
> 적응적으로 조절한다. 둘은 상호 보완적이며 동시에 작동한다. 상세
> adaTT 메커니즘은 *adaTT 기술 참조서* 를 참조한다.

## 결정 3 — Task Tower 와 손실 유형 분화

Task Expert 출력 (32D) 이 나오면 최종 예측으로 변환해야 한다. 하지만
CTR (이진 분류) 과 LTV (회귀) 와 Brand_prediction (128개 브랜드 중
contrastive) 은 동일한 head 를 쓸 수 없다. Task Tower 는 공통 얕은 MLP
한 덩어리 + 태스크 타입별 출력층 으로 이 문제를 푼다.

### Task Tower 구조

$$\mathbf{y} = \text{Linear}_{32 \to out} \circ \text{Block}_{64 \to 32} \circ \text{Block}_{32 \to 64}(\mathbf{h}_{expert})$$

$$\text{Block}_{a \to b}(\mathbf{x}) = \text{Dropout}(\text{SiLU}(\text{LayerNorm}(\text{Linear}_{a \to b}(\mathbf{x}))))$$

입력은 32D, hidden_dims 는 [64, 32], dropout 0.2. Regression 은
activation=None, Binary 는 sigmoid, Multiclass 는 softmax.

> **수식 직관.** 32→64 로 표현력을 키운 뒤 64→32 로 압축, 마지막에 출력
> 차원으로 사영. 각 층 사이의 LayerNorm + SiLU + Dropout 이 얕은 MLP
> 에서도 안정적 학습을 보장한다.

> **LayerNorm.** $\text{LN}(\mathbf{x}) = \gamma \cdot (\mathbf{x} - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$
> 로 한 *샘플 내* 모든 뉴런의 평균/분산으로 정규화한다 (BatchNorm 은
> *배치 내* 같은 뉴런으로 정규화 — 배치 크기 의존). 추론 시 배치 크기가
> 가변인 환경에서 LayerNorm 이 더 안정적이다.

### 태스크별 손실 유형

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 450" style="max-width:520px;width:100%;margin:24px auto;display:block;" font-family="JetBrains Mono, SUIT Variable, Pretendard Variable, ui-monospace, sans-serif">
  <defs><style>
    .grp-lbl { font-size: 13px; font-weight: 600; fill: #141414; }
    .grp-meta { font-size: 11px; fill: #6B6A63; }
    .task-chip { font-size: 11px; fill: #141414; }
    .bin { fill: #D8E0FF; stroke: #2E5BFF; }
    .multi { fill: #FDD8D1; stroke: #E14F3A; }
    .reg { fill: #C9ECD9; stroke: #1C8C5A; }
    .contra { fill: #EBD9E4; stroke: #8E4E6B; }
  </style></defs>

  <!-- Binary + Focal group -->
  <g transform="translate(20,20)">
    <text class="grp-lbl" x="0" y="14">Binary · Focal Loss</text>
    <text class="grp-meta" x="0" y="32">γ=2.0, α per-task (0.20 – 0.60)</text>
    <g transform="translate(0,42)">
      <rect class="bin" x="0" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="57.5" y="18" text-anchor="middle">CTR</text>
      <rect class="bin" x="125" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="182.5" y="18" text-anchor="middle">CVR  1.5w</text>
      <rect class="bin" x="250" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="307.5" y="18" text-anchor="middle">Churn  1.2w</text>
      <rect class="bin" x="375" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="432.5" y="18" text-anchor="middle">Retention</text>
    </g>
  </g>

  <!-- Multiclass + NLL group -->
  <g transform="translate(20,120)">
    <text class="grp-lbl" x="0" y="14">Multiclass · NLL</text>
    <text class="grp-meta" x="0" y="32">Softmax outputs (3 – 28 classes)</text>
    <g transform="translate(0,42)">
      <rect class="multi" x="0" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="57.5" y="18" text-anchor="middle">NBA (12)  2.0w</text>
      <rect class="multi" x="125" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="182.5" y="18" text-anchor="middle">Life-stage (6)</text>
      <rect class="multi" x="250" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="307.5" y="18" text-anchor="middle">Channel (3)</text>
      <rect class="multi" x="375" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="432.5" y="18" text-anchor="middle">Timing (28)</text>
    </g>
    <g transform="translate(0,78)">
      <rect class="multi" x="0" y="0" width="240" height="28" rx="4"/>
      <text class="task-chip" x="120" y="18" text-anchor="middle">Spending_category (12)  1.2w</text>
      <rect class="multi" x="250" y="0" width="240" height="28" rx="4"/>
      <text class="task-chip" x="370" y="18" text-anchor="middle">Consumption_cycle (7)</text>
    </g>
  </g>

  <!-- Regression group -->
  <g transform="translate(20,250)">
    <text class="grp-lbl" x="0" y="14">Regression · Huber (δ=1.0) / MSE</text>
    <text class="grp-meta" x="0" y="32">Robust to outliers — LTV outliers, etc.</text>
    <g transform="translate(0,42)">
      <rect class="reg" x="0" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="57.5" y="18" text-anchor="middle">Balance_util</text>
      <rect class="reg" x="125" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="182.5" y="18" text-anchor="middle">Engagement (MSE)</text>
      <rect class="reg" x="250" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="307.5" y="18" text-anchor="middle">LTV  1.5w</text>
      <rect class="reg" x="375" y="0" width="115" height="28" rx="4"/>
      <text class="task-chip" x="432.5" y="18" text-anchor="middle">Spending_bucket</text>
    </g>
    <g transform="translate(0,78)">
      <rect class="reg" x="0" y="0" width="240" height="28" rx="4"/>
      <text class="task-chip" x="120" y="18" text-anchor="middle">Merchant_affinity</text>
    </g>
  </g>

  <!-- Contrastive -->
  <g transform="translate(20,376)">
    <text class="grp-lbl" x="0" y="14">Contrastive · InfoNCE (τ=0.07)</text>
    <g transform="translate(0,28)">
      <rect class="contra" x="0" y="0" width="240" height="28" rx="4"/>
      <text class="task-chip" x="120" y="18" text-anchor="middle">Brand_prediction (128)  2.0w</text>
    </g>
  </g>
</svg>

> **태스크 16 개, 4 개 손실 유형으로 분화.** 가중치(`Nw`) 가 명시된
> 항목은 uncertainty weighting (아래) 이 추가로 자동 조정한다. 명시 없는
> 항목은 1.0 이 기본.

> **Huber Loss.** $\mathcal{L}_{\text{Huber}} = \frac{1}{2}(y - \hat{y})^2$
> ($|y - \hat{y}| \le \delta$), 그 밖은 $\delta(|y - \hat{y}| - \delta/2)$.
> $\delta = 1.0$ 은 오차 1 이내는 L2 (정밀 추적), 밖은 L1 (이상치 방어).
> LTV 처럼 극단값이 있는 회귀에 적합 (Huber, 1964).

> **InfoNCE.** $\mathcal{L} = -\log \exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) / \sum_i \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)$
> (Oord et al., 2018) — 대조 학습 손실. 유사 브랜드는 임베딩 공간에서
> 가깝게, 비유사 브랜드는 멀리 배치한다. 수천 개 브랜드를 직접 분류하지
> 않고 임베딩 공간에서 유사도를 학습하는 편이 확장성·일반화에 유리하다.

### Focal Loss 구현

TaskTower 가 이미 sigmoid 를 적용하므로 *이중 sigmoid 방지* 를 위해
확률값 기반으로 구현한다.

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}, \quad \alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

$\gamma = 2.0$ 은 focusing parameter (쉬운 예제 감쇠 강도), $\alpha$ 는
양성 클래스 가중치 (태스크별 차별화).

> **수식 직관.** Cross-Entropy 에 $(1 - p_t)^\gamma$ 가중치를 곱한다.
> 잘 맞히는 예제 ($p_t$ 큼) 에는 가중치가 급격히 줄고, 틀리는 예제
> ($p_t$ 작음) 에는 유지된다 — "쉬운 문제 그만 풀고 어려운 문제에
> 집중하라"의 손실 함수 버전. $\alpha_t$ 는 클래스 불균형을 보정한다
> (Lin et al., RetinaNet, ICCV 2017).

> **⚠ Focal Alpha 설계 기준.** `focal_alpha` 는 *양성 비율* 과 *비즈니스
> FN 비용* 의 두 요인으로 결정된다.
> - CTR (양성 3~8%, FN 비용 중간): $\alpha = 0.25$ (표준)
> - CVR (양성 0.5~3%, FN 비용 높음): $\alpha = 0.20$ (음성 경계 학습 강화)
> - Churn (양성 5~15%, FN 비용 매우 높음): $\alpha = 0.60$ (이탈 놓침 방지, recall 극대화)
> - Retention (양성 85~95%, FN 비용 중간): $\alpha = 0.20$ (소수 이탈 전조 탐지)

## 결정 3' — Uncertainty Weighting: 16개 가중치를 자동으로

태스크마다 손실 유형도 스케일도 다르다. CTR 의 Focal Loss 는 0.01 ~ 0.5
범위에서 움직이고, LTV 의 Huber Loss 는 고객 가치 단위에 따라 1 ~ 100
범위다. 이걸 단순 합산하면 스케일 큰 태스크가 gradient 를 독점한다.
수작업으로 16개 가중치를 튜닝하면 한 번에 맞춰도 데이터가 조금만 바뀌면
재튜닝이다.

### 결정 — 학습 가능한 log-variance 로 스스로 균형을 찾게 한다

Kendall, Gal & Cipolla (CVPR 2018) 가 태스크별 likelihood 를 Gaussian
으로 가정하고 homoscedastic uncertainty 의 MLE 를 유도하면 자연스럽게
다음 형태가 나온다:

$$\mathcal{L}_k^{\text{uw}} = w_k \cdot (\exp(-s_k) \cdot \mathcal{L}_k + s_k)$$

$s_k = \log(\sigma_k^2)$ 는 학습 가능한 log variance (`task_log_vars[k]`),
$\exp(-s_k)$ 는 precision (불확실성 높으면 가중치 낮춤), $s_k$ 항은
불확실성을 무한히 키우는 것을 방지하는 정규화 항이다. $s_k$ 는
$[-4.0, 4.0]$ 으로 clamp.

$+s_k$ 의 역할이 핵심이다. 이게 없으면 모델이 모든 태스크를 "극도로
불확실하다" 고 선언해서 $\exp(-s_k) \to 0$ 으로 가면 모든 손실이
사라진다. $+s_k$ 가 "불확실하다고 선언하는 데 비용이 든다" 는 제약이다.

> **수식 직관.** 태스크가 본질적으로 어렵면 그 손실이 전체 학습을
> 지배하지 않도록 자동으로 가중치를 낮춘다. $+s_k$ 항이 "모든 태스크를
> 불확실하다고 선언해 손실을 0 으로" 만드는 편법을 막는다. 16 개 태스크
> 가중치를 수동 튜닝하지 않고 모델이 균형을 찾는다.

> **이론적 기반.** Kendall, Gal & Cipolla (CVPR 2018) — 태스크별
> likelihood 를 Gaussian 으로 가정하면 homoscedastic uncertainty 의
> MLE 로부터 $\exp(-s_k) \cdot \mathcal{L}_k + s_k$ 형태가 자연스럽게
> 유도된다.

### 총 손실 집계

`forward()` 에서 다음 손실들이 합산된다.

1. **태스크 손실**: adaTT 적용 후 enhanced losses 합계 (또는 단순 합계)
2. **CGC Entropy 정규화**: $\lambda_{\text{ent}} \times \mathcal{L}_{\text{entropy}}$ (학습 시, CGC 미고정 시)
3. **Causal Expert DAG 정규화**: acyclicity + sparsity
4. **SAE 손실**: reconstruction + L1 sparsity (weight=0.01, detached)

## 정리하자면

세 결정이 PLE-5 의 구조를 만든다.

1. **GroupTaskExpertBasket** — 클러스터 × 태스크 독립 MLP 에서 Group
   공유 + Cluster Embedding 으로 전환, 88% 파라미터 감소 (3M → 362K).
   클러스터 신호가 "결정 함수 전체" 가 아니라 "입력 분포의 변화" 라는
   관찰이 이 재설계의 근거.
2. **Logit Transfer** — CTR→CVR→LTV 같은 비즈니스 순서를 DAG 로
   선언하고 Kahn's algorithm 으로 실행 순서를 자동 도출. 잔차 형태
   projection 으로 유용하지 않으면 가중치가 0 으로 수렴하는 safe default.
   네트워크가 재발견하기를 기다리지 않고 우리가 이미 아는 의존성을 직접
   전달하는 쇼트컷.
3. **Uncertainty Weighting** — 16개 태스크 가중치를 학습 가능한
   log-variance 로 자동 균형. Kendall et al. (2018) 의 Gaussian
   likelihood MLE 에서 유도되는 $\exp(-s_k) \cdot \mathcal{L}_k + s_k$
   형태. $+s_k$ 정규화가 "모든 태스크를 불확실하다고 선언하는" 편법을
   막는다.

다음 편인 **PLE-6** 에서는 시스템이 돌아간 후의 두 가지 남은 질문 —
"Expert 가 실제로 무엇을 학습했는지 볼 수 있는가 (SAE)", "예측 신뢰도를
정량화할 수 있는가 (Evidential Deep Learning)" — 그리고 전체 18개 태스크
사양 reference 로 시리즈를 마무리한다. 기술 참조서 PDF 다운로드 포함.
