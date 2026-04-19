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
이번 5편은 태스크 그룹 단위로 전용 Expert를 만드는 GroupTaskExpertBasket
v3.2, 태스크 간 명시적 정보 전달을 수행하는 Logit Transfer의 3가지
모드, 그리고 최종 예측을 만드는 Task Tower의 구조 — PLE 데이터 흐름의
후반부를 관통한다.*

## GroupTaskExpertBasket — GroupEncoder + ClusterEmbedding (v3.2)

### GroupTaskExpertBasket vs ClusterTaskExpertBasket

v3.2 에서 `use_group_encoder=true` (기본값) 설정 시
`GroupTaskExpertBasket` 을 사용하며, 기존 `ClusterTaskExpertBasket`
대비 *88% 파라미터 감소* 를 달성한다.

| 항목 | ClusterTaskExpertBasket (레거시) | GroupTaskExpertBasket (v3.2) |
| --- | --- | --- |
| 아키텍처 | 태스크×클러스터 독립 MLP | GroupEncoder 공유 + ClusterEmbedding |
| 파라미터 | ~3.0M | ~362K |
| 클러스터 특화 | 독립 서브헤드 가중치 | 클러스터 임베딩 주입 |
| 일반화 | 낮음 (클러스터별 과적합) | 높음 (공유 인코더) |

### GroupEncoder 아키텍처

`_build_task_experts()` (라인 498~560) 에서 `GroupTaskExpertBasket` 을
생성한다.

```python
# ple_cluster_adatt.py:533-543
self.task_experts = GroupTaskExpertBasket(
    input_dim=task_expert_input_dim,     # 512 + 32 = 544
    group_hidden_dim=128,
    group_output_dim=64,
    cluster_embed_dim=32,
    subhead_output_dim=32,
    n_clusters=20,
    task_names=self.task_names,
    task_groups=task_groups,             # adaTT config에서 추출
)
```

GroupEncoder 의 단일 태스크 forward 는 다음과 같다.

$$\mathbf{e}_{cluster} = \text{Embedding}(\text{cluster\_id}) \in \mathbb{R}^{32}$$

$$\mathbf{x}_{input} = [\text{CGC\_output}_{512D} \,\|\, \text{HMM\_proj}_{32D} \,\|\, \mathbf{e}_{cluster,32D}] \in \mathbb{R}^{576}$$

$$\mathbf{h}_{expert} = \text{MLP}_{576 \to 128 \to 64 \to 32}(\mathbf{x}_{input})$$

실제 input_dim = 544D (shared + HMM) + 32D (cluster_embed) = 576D. 그룹
내 태스크는 GroupEncoder 를 공유하고, 그룹 간은 독립이다.

> **수식 직관.** 이 수식은 "이 고객이 어떤 클러스터에 속하는가"라는
> 정보를 모델 내부에 주입하는 과정이다. 먼저 클러스터 ID 를 32D
> 임베딩으로 변환하고, CGC 출력(512D) + HMM 프로젝션(32D) 과 이어 붙여
> 총 576D 입력을 만든다. 이후 3단 MLP(576→128→64→32) 가 이를 압축하여
> 태스크별 최종 표현(32D) 을 생성한다. 같은 태스크 그룹(예: Engagement
> 그룹의 CTR/CVR)은 동일한 GroupEncoder 를 공유하여 파라미터를
> 절약하면서도 클러스터 임베딩으로 차별화한다.

> **학부 수학 — 임베딩(Embedding)이란 무엇인가.**
> $\text{Embedding}(\text{cluster\_id}) \in \mathbb{R}^{32}$ 는 정수
> 인덱스를 연속 벡터로 변환하는 *룩업 테이블* 이다. 내부적으로
> $\mathbf{E} \in \mathbb{R}^{20 \times 32}$ 행렬을 저장하고,
> cluster_id = $c$ 이면 $\mathbf{E}$ 의 $c$ 번째 행
> $\mathbf{e}_c \in \mathbb{R}^{32}$ 를 꺼내는 것이다. 이것은
> *one-hot 인코딩 + Linear 변환* 과 수학적으로 동치다: one-hot
> $\mathbf{v}_c \in \mathbb{R}^{20}$ (c 번째만 1) 을 만들면
> $\mathbf{v}_c^T \mathbf{E} = \mathbf{e}_c$. 그러나 one-hot 은 희소
> 벡터 연산이 필요하여 비효율적이므로, 직접 인덱싱이 더 빠르다. 핵심은
> 이 행렬 $\mathbf{E}$ 가 *학습 가능한 파라미터* 라는 점이다. 학습이
> 진행되면서 유사한 클러스터의 임베딩 벡터는 가까워지고, 다른
> 클러스터는 멀어져서, 이산적 ID 가 의미 있는 연속 표현으로 변환된다.
> Word2Vec (Mikolov et al., 2013) 에서 단어를 벡터로 표현한 것과 동일한
> 원리다.

### Soft Routing (cluster_probs)

클러스터 경계에 위치한 샘플(cluster_probs 가 분산된 경우) 에 대해
*soft routing* 으로 여러 클러스터 임베딩의 가중 평균을 사용한다.

$$\mathbf{e}_{cluster} = \sum_{c=0}^{19} p_c \cdot \mathbf{E}_c \in \mathbb{R}^{32}$$

$$\mathbf{h}_{expert} = \text{TaskHead}([\text{GroupEncoder}(\mathbf{x}) \,\|\, \mathbf{e}_{cluster}])$$

여기서 $p_c$ 는 GMM 클러스터 $c$ 의 사후 확률, $\mathbf{E}_c$ 는
클러스터 $c$ 의 학습 가능 임베딩 벡터(32D) 다.

> **수식 직관.** 이 수식은 경계 고객을 부드럽게 처리하는 핵심이다.
> 예를 들어 어떤 고객이 클러스터 3 에 60%, 클러스터 7 에 30%, 나머지
> 10% 확률로 소속되면, 각 클러스터의 임베딩 벡터를 이 비율대로
> 혼합한다. 구현에서는 `cluster_probs @ embedding.weight`
> ($[B, 20] \times [20, 32] = [B, 32]$) 로 한 번의 행렬곱으로 완료된다.
> 혼합된 임베딩이 GroupEncoder 출력과 결합되어 TaskHead 를 통과하므로,
> 클러스터 조건 신호가 부드럽게 주입된다. 하나의 클러스터에 강제
> 배정하는 hard routing 과 달리, 경계 고객의 예측이 클러스터 할당
> 변동에 민감하지 않게 된다.

`forward_single_task()` 호출 시 `cluster_probs` 인자가 전달되면 hard
assignment 대신 soft routing 을 수행한다 (라인 1247~1250).

> **학부 수학 — GMM 과 베이즈 정리: Soft Routing 의 수학적 기반.**
> 클러스터 확률 $p_c$ 는 *가우시안 혼합 모델(GMM)* 의 사후 확률이다.
> GMM 은 데이터가 $K$ 개의 가우시안 분포의 혼합에서 생성되었다고
> 가정한다:
> $p(\mathbf{x}) = \sum_{c=1}^K \pi_c \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$.
> 여기서 $\pi_c$ 는 혼합 가중치(사전 확률), $\mathcal{N}$ 은 가우시안
> 분포다. *베이즈 정리* 를 적용하면 관측 $\mathbf{x}$ 가 클러스터 $c$
> 에 속할 사후 확률은
> $p(c | \mathbf{x}) = \frac{\pi_c \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)}{\sum_{j=1}^K \pi_j \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$
> 이다. 이것이 바로 soft routing 에 사용되는 $p_c$ 이다. 클러스터
> 중심($\boldsymbol{\mu}_c$) 에서 먼 고객일수록 여러 클러스터의 사후
> 확률이 비슷해져 soft routing 의 효과가 커지고, 중심에 가까운 고객은
> 하나의 클러스터에 집중되어 hard routing 과 유사해진다.
> EM(Expectation-Maximization) 알고리즘으로
> $\pi_c, \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c$ 를 추정하며, 이는
> 오프라인에서 사전 계산된다.

> **최신 동향 — 클러스터 기반 추천의 산업 적용.** 클러스터 기반 조건
> 주입(conditional embedding) 전략은 2023-2025 년 대규모 추천
> 시스템에서 핵심 기법이 되었다. Kuaishou 의
> *POSO (Personalized Cold-Start, KDD 2022)* 는 사용자 세그먼트별 독립
> gate 를 두어 콜드스타트 문제를 완화하였고, Alibaba 의
> *CL4CTR (2023)* 은 클러스터별 대조 학습으로 사용자 표현을 정제하였다.
> ByteDance 의 *SAMD (KDD 2024)* 는 클러스터 임베딩과 MoE 를 결합하여
> TikTok 의 단기 동영상 추천에서 CTR 을 2.3% 개선했다. 본 시스템의
> 20-클러스터 임베딩 + soft routing 설계는 이러한 산업 트렌드와
> 부합하며, 특히 금융 도메인에서 고객 세그먼트(VIP, 일반, 청년, 시니어
> 등) 에 따른 행동 패턴 차이를 명시적으로 모델링하는 접근이다.

## Logit Transfer — 태스크 간 명시적 정보 전달

### 전이 쌍 정의

`_build_logit_transfer()` (라인 984~1055) 에서 `task_relationships`
config 로부터 전이 쌍을 등록한다.

| Source | Target | 유형 | 강도 | 비즈니스 의미 |
| --- | --- | --- | --- | --- |
| CTR | CVR | Sequential | 0.5 | 클릭한 고객만 전환 (AARRR 퍼널) |
| Churn | Retention | Inverse | 0.5 | 이탈 확률의 역 = 유지 기반 |
| CVR | LTV | Feature | 0.5 | 전환 확률이 생애가치에 영향 |
| NBA | Spending_category | Feature | 0.5 | 추천 행동이 소비 카테고리 결정 |
| Spending_category | Brand_prediction | Feature | 0.5 | 소비 카테고리가 브랜드 선택에 영향 |

### 전이 메커니즘

```python
# ple_cluster_adatt.py:1266-1277 — forward()에서 logit transfer 적용
for task_name in execution_order:
    tower_input = task_expert_outputs[task_name]
    if task_name in self.logit_transfer_sources:
        source_task = self.logit_transfer_sources[task_name]
        if source_task in predictions:
            src_out = predictions[source_task]
            if src_out.dim() == 1:
                src_out = src_out.unsqueeze(-1)
            proj = self.logit_transfer_proj[task_name](src_out)
            tower_input = tower_input + strength * proj
```

$$\mathbf{h}_{tower}^t = \mathbf{h}_{expert}^t + \alpha \cdot \text{SiLU}(\text{LayerNorm}(\text{Linear}(\text{pred}^s)))$$

$\alpha = 0.5$ (`transfer_strength`), $\text{pred}^s$ 는 source 태스크
예측값이다. `Linear` 는 source output_dim → task_expert_output_dim (32D)
이고, projection 모듈은 `nn.Sequential(Linear, LayerNorm, SiLU)` 다.

> **수식 직관.** 이 수식은 "선행 태스크의 예측 결과를 후행 태스크의
> 입력에 더해주는" 명시적 전이다. 예를 들어 CTR→CVR 전이에서, CTR
> 모델이 "이 고객은 클릭 확률이 높다"고 예측하면 그 정보가 프로젝션을
> 거쳐 CVR 타워의 입력에 추가된다. $\alpha = 0.5$ 는 원래 Expert 출력
> 대비 전이 신호의 상대적 강도를 조절한다. 직관적으로, 전이가
> 잔차(residual) 형태로 더해지므로 source 정보가 유용하지 않으면
> 프로젝션 가중치가 0 에 수렴하여 자연스럽게 무시된다.

### 실행 순서 (Topological Sort)

`_derive_task_order_from_config()` (라인 1093~1155) 에서
`task_relationships` 의 의존 관계를 Kahn's algorithm 으로 위상 정렬하여
실행 순서를 자동 도출한다.

의존 관계 그래프는 세 개의 독립적인 체인으로 구성된다:

- **Engagement 체인**: CTR --(seq)→ CVR --(feat)→ LTV
- **Retention 체인**: Churn --(inv)→ Retention
- **Consumption 체인**: NBA --(feat)→ Spending_category --(feat)→ Brand_prediction

> **⚠ 실행 순서 위반 시.** 위상 정렬 실패(사이클 감지) 시
> `_get_task_execution_order_fallback()` 이 하드코딩된 순서로 폴백한다
> (라인 1069~1091). 새 전이 관계 추가 시 `task_relationships` config 에
> 등록하면 자동으로 순서가 반영된다.

> **학부 수학 — 위상 정렬(Topological Sort) 과 DAG.** 위상 정렬은
> *방향 비순환 그래프(DAG: Directed Acyclic Graph)* 의 노드를 간선
> 방향을 위반하지 않는 순서로 나열하는 알고리즘이다. 간선
> $A \to B$ 가 있으면 $A$ 가 $B$ 보다 앞에 와야 한다.
> *Kahn's algorithm (1962)* 은 다음과 같이 동작한다: (1) 진입 차수
> (in-degree) 가 0 인 노드를 큐에 삽입한다. (2) 큐에서 꺼낸 노드를
> 결과에 추가하고, 해당 노드의 출력 간선을 제거한다. (3) 새로 진입
> 차수가 0 이 된 노드를 큐에 삽입한다. (4) 큐가 빌 때까지 반복한다.
> 모든 노드를 방문하지 못하면 *사이클* 이 존재한다. *시간 복잡도* 는
> $O(V + E)$ 로, 노드 수 $V$ 와 간선 수 $E$ 에 비례한다. 본 시스템에서
> CTR $\to$ CVR $\to$ LTV 체인은 "CTR 을 먼저 예측해야 CVR 에 전이할
> 수 있고, CVR 을 먼저 예측해야 LTV 에 전이할 수 있다"는 인과적 선후
> 관계를 그래프로 표현한 것이다.

> **역사적 배경 — 잔차 연결(Residual Connection) 과 Logit Transfer.**
> Logit Transfer 의 `tower_input = tower_input + alpha * proj` 형태는
> *He, Zhang, Ren & Sun (CVPR 2016)* 이 ResNet 에서 제안한 잔차 연결
> (skip connection) 과 동일한 구조다. He et al. 은 152 층 깊은
> 네트워크에서 기울기 소실 없이 학습하려면
> $\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$ 처럼 "입력을 그대로
> 더해주는 지름길" 이 필요함을 보였다. 이 아이디어는
> *Highway Networks (Srivastava et al., 2015)* 에서 먼저 탐구되었으나,
> ResNet 의 극단적 단순화
> ($\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$, gate 없음) 가 더
> 효과적이었다. Logit Transfer 에서도 source 태스크의 예측이 잔차
> 형태로 더해지므로, 전이 정보가 유용하지 않으면 프로젝션 가중치가 0
> 으로 수렴하여 원래 Expert 출력만 남게 되는 *안전한 기본값(safe
> default)* 을 보장한다. $\alpha = 0.5$ 는 이 잔차의 상대적 크기를
> 제한하는 스케일링 계수다.

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
> Logit Transfer 와 adaTT 는 '태스크 간 지식 전달'이라는 같은 목표를
> 다른 수준에서 수행한다. Logit Transfer 는 CTR→CVR 처럼 비즈니스
> 로직상 순차적인 태스크에 예측값을 직접 전달하고, adaTT 는 gradient
> 수준에서 모든 태스크 쌍의 상호 영향을 적응적으로 조절한다. 둘은 상호
> 보완적이며 동시에 작동한다. 상세 adaTT 메커니즘은 *adaTT 기술 참조서*
> 를 참조한다.

## Task Tower — 최종 예측

### Tower 아키텍처

`TaskTower` 클래스 (라인 244~293) 는 모든 태스크에 공통된 MLP 구조를
사용한다.

$$\mathbf{y} = \text{Linear}_{32 \to out} \circ \text{Dropout} \circ \text{SiLU} \circ \text{LayerNorm} \circ \text{Linear}_{64 \to 32} \circ \text{Dropout} \circ \text{SiLU} \circ \text{LayerNorm} \circ \text{Linear}_{32 \to 64}(\mathbf{h}_{expert})$$

입력은 32D (Task Expert 출력), hidden_dims 는 [64, 32], dropout 은 0.2
다. Regression 태스크는 activation=None, Binary 는 sigmoid, Multiclass
는 softmax 다.

> **수식 직관.** Task Tower 는 32D Expert 출력을 받아 최종 예측값으로
> 변환하는 "마지막 한 걸음"이다. 32→64 로 확장하여 표현력을 키운 뒤,
> 64→32 로 압축하고, 마지막에 출력 차원으로 사영한다. 각 층 사이에
> LayerNorm (스케일 안정화) + SiLU (비선형성) + Dropout (과적합 방지)
> 을 끼워 얕은 MLP 이면서도 안정적 학습이 가능하다. 태스크 유형에
> 따라 출력 활성화가 달라지며, Binary 는 sigmoid 로 0~1 확률을,
> Multiclass 는 softmax 로 클래스 분포를, Regression 은 활성화 없이
> 실수값을 출력한다.

```python
# ple_cluster_adatt.py:268-280
layers = []
prev_dim = input_dim  # 32
for hidden_dim in hidden_dims:  # [64, 32]
    layers.extend([
        nn.Linear(prev_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
    ])
    prev_dim = hidden_dim
layers.append(nn.Linear(prev_dim, output_dim))
```

> **학부 수학 — LayerNorm 의 수학적 정의와 역할.** Layer Normalization
> 은 각 샘플의 은닉 벡터를 독립적으로 정규화한다:
> $\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$.
> 여기서 $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ (평균),
> $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$ (분산),
> $\gamma, \beta \in \mathbb{R}^d$ 는 학습 가능한 스케일/시프트
> 파라미터, $\epsilon \approx 10^{-5}$ 는 분모 0 방지다. *왜 정규화가
> 필요한가?* 신경망의 각 레이어는 이전 레이어의 출력을 입력으로 받는데,
> 학습 중 이전 레이어의 파라미터가 변하면 입력 분포가 계속 바뀌는
> *내부 공변량 이동(Internal Covariate Shift)* 이 발생한다. 이는
> 학습률 설정을 어렵게 만든다. LayerNorm 은 각 레이어 입력을 평균 0,
> 분산 1 로 정규화하여 분포를 안정화한다. *BatchNorm 과의 차이*:
> BatchNorm 은 배치 내 같은 뉴런을 정규화하고(배치 크기 의존),
> LayerNorm 은 한 샘플 내 모든 뉴런을 정규화한다(배치 크기 무관).
> Task Tower 처럼 배치 크기가 가변적인 추론 환경에서는 LayerNorm 이
> 더 안정적이다.

### 태스크별 Loss 유형

| 태스크 | 유형 | Loss | 출력 dim | Activation | Weight |
| --- | --- | --- | --- | --- | --- |
| CTR | Binary | Focal ($\gamma$=2.0, $\alpha$=0.25) | 1 | sigmoid | 1.0 |
| CVR | Binary | Focal ($\gamma$=2.0, $\alpha$=0.20) | 1 | sigmoid | 1.5 |
| Churn | Binary | Focal ($\gamma$=2.0, $\alpha$=0.60) | 1 | sigmoid | 1.2 |
| Retention | Binary | Focal ($\gamma$=2.0, $\alpha$=0.20) | 1 | sigmoid | 1.0 |
| NBA | Multiclass | NLL (softmax 후) | 12 | softmax | 2.0 |
| Life-stage | Multiclass | NLL | 6 | softmax | 0.8 |
| Balance_util | Regression | Huber ($\delta$=1.0) | 1 | none | 1.0 |
| Engagement | Regression | MSE | 1 | none | 0.8 |
| LTV | Regression | Huber ($\delta$=1.0) | 1 | none | 1.5 |
| Channel | Multiclass | NLL | 3 | softmax | 0.8 |
| Timing | Multiclass | NLL | 28 | softmax | 0.8 |
| Spending_category | Multiclass | NLL | 12 | softmax | 1.2 |
| Consumption_cycle | Multiclass | NLL | 7 | softmax | 0.8 |
| Spending_bucket | Regression | Huber ($\delta$=1.0) | 1 | none | 0.8 |
| Brand_prediction | Contrastive | InfoNCE ($\tau$=0.07) | 128 | none | 2.0 |
| Merchant_affinity | Regression | Huber ($\delta$=1.0) | 1 | none | 1.0 |

> **학부 수학 — Huber Loss: MSE 와 MAE 의 절충.** Huber Loss 는
> *Peter J. Huber (1964)* 가 제안한 로버스트(robust) 손실 함수다:
> $\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta \cdot (|y - \hat{y}| - \delta/2) & \text{otherwise} \end{cases}$.
> 오차가 작을 때 ($|y - \hat{y}| \le \delta$) 는 MSE 처럼 $L_2$ (제곱),
> 클 때는 MAE 처럼 $L_1$ (절대값) 로 동작한다. *왜 MSE 만으로 부족한가?*
> MSE $= (y - \hat{y})^2$ 는 이상치(outlier) 에서 제곱으로 인해 손실이
> 폭발하여 gradient 가 매우 커지고, 모델이 이상치 하나에 과도하게
> 끌려간다. MAE $= |y - \hat{y}|$ 는 이상치에 강하지만, 0 에서 미분
> 불가능하고 gradient 가 상수 ($\pm 1$) 라서 0 근처에서 수렴이 느리다.
> Huber Loss 는 0 근처에서는 MSE 의 부드러운 gradient 를, 멀리에서는
> MAE 의 이상치 내성을 결합한다. $\delta = 1.0$ 은 "오차 1 이내는 정밀
> 추적, 1 이상은 이상치 방어" 라는 경계를 설정한다. LTV (생애가치)
> 예측처럼 극단적 고액 소비자가 존재하는 태스크에서 Huber Loss 가 MSE
> 보다 안정적이다.

> **최신 동향 — InfoNCE 와 대조 학습: Brand Prediction 의 손실 함수.**
> Brand Prediction 태스크에 사용된
> *InfoNCE (Noise-Contrastive Estimation)* 은
> *Oord, Li & Vinyals (2018)* 이 CPC (Contrastive Predictive Coding)
> 에서 제안한 손실이다:
> $\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\sum_{i=0}^N \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}$.
> 여기서 $\mathbf{q}$ 는 쿼리, $\mathbf{k}_+$ 는 양성 키, $\mathbf{k}_i$
> 는 음성 키, $\tau = 0.07$ 은 온도 파라미터다.
> *SimCLR (Chen et al., ICML 2020)* 과 *MoCo (He et al., CVPR 2020)* 가
> 이를 시각 표현 학습에 적용하여 대조 학습의 붐을 일으켰고, 2023-2025
> 년에는 추천 시스템의 *SASRec-CL*, *CL4Rec* 등 시퀀셜 추천에 광범위하게
> 채택되었다. 본 시스템에서 Brand Prediction 을 대조 학습으로 학습하는
> 이유는 수천 개의 브랜드를 직접 분류하는 대신, *브랜드 임베딩 공간*
> 에서 유사 브랜드를 가깝게, 비유사 브랜드를 멀리 배치하는 것이
> 확장성과 일반화에 유리하기 때문이다.

### Focal Loss 구현

`_compute_task_losses()` (라인 1765~1780) 에서 확률값 기반 Focal Loss
를 계산한다. TaskTower 가 이미 sigmoid 를 적용하므로 *이중 sigmoid 방지*
를 위해 logits 기반이 아닌 확률값 기반으로 구현한다.

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}, \quad \alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

$\gamma = 2.0$ 은 focusing parameter (쉬운 예제 감쇠 강도), $\alpha$
는 양성 클래스 가중치 (태스크별 차별화, 상세 설계는 config 참조) 다.

> **수식 직관.** Focal Loss 는 표준 Cross-Entropy 에
> $(1 - p_t)^\gamma$ 가중치를 곱한 것이다. $p_t$ 는 "모델이 정답에
> 부여한 확률"이므로, 잘 맞추는 예제 ($p_t$ 가 큼) 에는 가중치가
> 급격히 줄어들고, 틀리는 예제 ($p_t$ 가 작음) 에는 가중치가 유지된다.
> 직관적으로, "쉬운 문제를 계속 풀어봐야 실력이 안 오르니, 어려운
> 문제에 집중하라"는 학습 전략을 손실 함수로 구현한 것이다. $\alpha_t$
> 는 클래스 불균형 보정으로, 희소한 양성 클래스 (예: 이탈 고객) 를
> 놓치지 않도록 가중치를 높인다.

> **학부 수학.** *Cross-Entropy 는 왜 $-\log(p)$ 인가?* 정보이론에서
> 어떤 사건의 *정보량(information content)* 은 $I(x) = -\log_2(p(x))$
> 로 정의된다. 확률이 낮은 사건(놀라운 사건) 일수록 정보량이 크다.
> 예: 동전 앞면 ($p=0.5$) 의 정보량 $= -\log_2(0.5) = 1$ 비트, 주사위
> 1 ($p=1/6$) 의 정보량 $= -\log_2(1/6) \approx 2.58$ 비트.
> *Cross-Entropy* $H(p, q) = -\sum p(x) \log q(x)$ 는 "실제 분포 $p$ 를
> 따르는 데이터를 모델 분포 $q$ 로 인코딩할 때 필요한 평균 비트
> 수"이다. 이진 분류에서 정답이 $y=1$ 이고 모델 예측이 $p$ 이면,
> $-\log(p)$ 는 "정답에 모델이 부여한 확률이 낮을수록 큰 벌점" 이 된다.
> *Focal Loss 와의 관계*:
> $\text{FL} = -(1-p_t)^\gamma \log(p_t)$ 에서 $(1-p_t)^\gamma$ 는 이
> 벌점에 "예측 틀린 정도"에 비례하는 가중치를 곱한 것이다. *구체적
> 계산*: $p_t=0.9$ 이면 $\text{CE} = -\log(0.9) = 0.105$,
> $\text{FL} = 0.1^2 \times 0.105 = 0.00105$ (100 배 감소). $p_t=0.1$
> 이면 $\text{CE} = -\log(0.1) = 2.303$,
> $\text{FL} = 0.9^2 \times 2.303 = 1.865$ (거의 유지).

> **역사적 배경.** Focal Loss 는
> *Lin, Goyal, Girshick, He & Dollár (ICCV 2017)* 이 물체 탐지
> (Object Detection) 에서 제안하였다. 당시 one-stage 탐지기 (예: YOLO,
> SSD) 가 two-stage 탐지기 (Faster R-CNN) 보다 정확도가 낮았는데,
> 원인은 배경(easy negative) 이 전경(hard positive) 보다 압도적으로
> 많아 쉬운 예제의 gradient 가 학습을 지배하기 때문이었다. Focal Loss
> 는 $(1-p_t)^\gamma$ 항으로 쉬운 예제의 기여를 동적으로 줄여,
> one-stage 탐지기 (RetinaNet) 가 처음으로 two-stage 를 능가하는 결과를
> 이끌어냈다. 이후 추천 시스템, 의료 영상, 자연어 처리 등 클래스
> 불균형이 심한 거의 모든 분야에서 표준 손실 함수로 채택되었다. CTR
> 예측 (양성 비율 3~8%) 은 Focal Loss 의 전형적인 적용 사례다.

```python
# ple_cluster_adatt.py:1774-1780 — fp16 AMP 안전 focal loss
p_f = pred.squeeze().float().clamp(1e-7, 1 - 1e-7)
t_f = target.float()
bce = -(t_f * torch.log(p_f) + (1 - t_f) * torch.log(1 - p_f))
p_t = p_f * t_f + (1 - p_f) * (1 - t_f)
alpha_t = alpha * t_f + (1 - alpha) * (1 - t_f)
focal_weight = alpha_t * (1 - p_t) ** gamma
loss = (focal_weight * bce).mean()
```

> **⚠ Focal Alpha 설계 기준.** `focal_alpha` 는 *양성 비율* 과
> *비즈니스 FN 비용* 의 두 요인으로 결정된다.
> - CTR (양성 3~8%, FN 비용 중간): $\alpha = 0.25$ (표준)
> - CVR (양성 0.5~3%, FN 비용 높음): $\alpha = 0.20$ (음성 경계 학습 강화)
> - Churn (양성 5~15%, FN 비용 매우 높음): $\alpha = 0.60$ (이탈 놓침 방지, recall 극대화)
> - Retention (양성 85~95%, FN 비용 중간): $\alpha = 0.20$ (소수 이탈 전조 탐지)

### Uncertainty Weighting (Kendall et al.)

`loss_weighting.strategy: "uncertainty"` 설정 시 (라인 1818~1827)
태스크별 학습 가능한 log variance 로 *homoscedastic uncertainty* 를
모델링한다.

$$\mathcal{L}_k^{\text{uw}} = w_k \cdot (\exp(-s_k) \cdot \mathcal{L}_k + s_k)$$

$s_k = \log(\sigma_k^2)$ 는 학습 가능한 log variance (`task_log_vars[k]`)
, $\exp(-s_k)$ 는 precision (불확실성 높으면 가중치 낮춤), $s_k$ 항은
불확실성을 무한히 키우는 것을 방지하는 정규화 항이다. $s_k$ 는
$[-4.0, 4.0]$ 으로 clamp 되고, precision 은 $[10^{-3}, 100]$ 으로
clamp 된다.

> **수식 직관.** 이 수식은 "어떤 태스크가 본질적으로 예측하기 어려우면,
> 그 태스크의 손실이 전체 학습을 지배하지 않도록 자동으로 가중치를
> 낮추는" 메커니즘이다. $\exp(-s_k)$ 는 정밀도(precision) 로서,
> 불확실성이 높으면 ($s_k$ 큼) 작아져 손실 기여를 줄인다. 동시에
> $+s_k$ 항이 정규화 역할을 하여, 모델이 "모든 태스크를 불확실하다고
> 선언하여 손실을 0 으로 만드는" 편법을 방지한다. 16 개 태스크의
> 가중치를 수동으로 튜닝하는 대신, 모델이 자동으로 균형을 찾는다.

**[NeurIPS 2018]** Kendall, A., Gal, Y., & Cipolla, R. *"Multi-Task
Learning Using Uncertainty to Weigh Losses for Scene Geometry and
Semantics"*

> **역사적 배경.** Uncertainty Weighting 은
> *Kendall, Gal & Cipolla (CVPR 2018)* 이 제안하였다. 원래 자율주행의
> 장면 이해(scene understanding) 문제에서 깊이 추정, 표면 법선 추정,
> 의미 분할이라는 세 태스크의 가중치를 자동으로 조정하기 위해
> 고안되었다. 이론적 기반은 *homoscedastic uncertainty* (데이터
> 포인트가 아닌 태스크 자체의 불확실성) 의 최대 우도 추정(MLE) 이다.
> 각 태스크의 likelihood 를 Gaussian 으로 가정하면 log-likelihood 를
> 정리했을 때 자연스럽게 $\exp(-s_k) \cdot \mathcal{L}_k + s_k$ 형태가
> 유도된다. 이전에는 수동 grid search 나 GradNorm (Chen et al., ICML
> 2018) 처럼 gradient 크기를 균등화하는 방법이 사용되었으나,
> Uncertainty Weighting 은 추가 하이퍼파라미터 없이 *학습 가능한
> 파라미터* 로 대체하여 널리 채택되었다.

> **최신 동향.** 2024-2025 년 MTL loss balancing 분야는 Uncertainty
> Weighting 을 넘어서는 방법들이 연구되고 있다.
> *Nash-MTL (Navon et al., ICML 2022)* 은 태스크 간 gradient 를 Nash
> 협상(bargaining) 게임으로 모델링하여 Pareto 최적 해를 찾고,
> *Aligned-MTL (Senushkin et al., CVPR 2023)* 은 gradient 방향을
> 정렬하여 충돌을 최소화한다. *Auto-Lambda (Liu et al., 2024)* 는 메타
> 학습으로 가중치를 적응적으로 조정한다. 그러나 실무에서는 Uncertainty
> Weighting 이 구현 단순성과 안정적 성능으로 여전히 가장 널리 사용되며,
> 특히 태스크 수가 10 개 이상인 대규모 MTL 에서 검증된 선택지로 남아
> 있다.

### 총 손실 집계

`forward()` (라인 1284~1344) 에서 다음 손실들이 합산된다.

1. **태스크 손실**: adaTT 적용 후 enhanced losses 합계 (또는 단순 합계)
2. **CGC Entropy 정규화**: $\lambda_{\text{ent}} \times \mathcal{L}_{\text{entropy}}$ (학습 시, CGC 미고정 시)
3. **Causal Expert DAG 정규화**: acyclicity + sparsity
4. **SAE 손실**: reconstruction + L1 sparsity (weight=0.01, detached)

## 정리하자면

GroupTaskExpertBasket v3.2 는 클러스터×태스크 독립 MLP 를 GroupEncoder
공유 + ClusterEmbedding 구조로 치환하여 88% 파라미터를 줄이면서도
클러스터별 특성화를 유지한다. soft routing 은 GMM 사후 확률로 경계
고객을 부드럽게 다룬다. Logit Transfer 는 CTR→CVR→LTV 같은 비즈니스
순서를 DAG 로 선언하고 Kahn's algorithm 으로 실행 순서를 자동
도출하며, 잔차 형태 프로젝션으로 source 정보가 유용하지 않을 때
자연스럽게 0 으로 수렴한다. Task Tower 는 32→64→32→out 의 공통 MLP
로 예측을 만들되, 태스크 유형별로 Focal / Huber / NLL / InfoNCE 손실을
차등 적용하고, Uncertainty Weighting 으로 16 개 태스크의 가중치를
학습 가능한 log variance 로 자동 균형한다. 다음 편인 **PLE-6** 에서는
Sparse Autoencoder 해석성, Evidential Deep Learning 불확실성, 18 개
태스크 전체 사양을 정리하며 기술 참조서 PDF 로 시리즈를 마무리한다.
