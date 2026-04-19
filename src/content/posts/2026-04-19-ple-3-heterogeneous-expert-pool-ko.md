---
title: "[Study Thread] PLE-3 — 입력 구조와 이종 Shared Expert Pool (512D)"
date: 2026-04-19 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, expert-pool, hmm, shared-experts]
lang: ko
series: study-thread
part: 3
alt_lang: /2026/04/19/ple-3-heterogeneous-expert-pool-en/
next_title: "PLE-4 — CGC 게이팅의 두 변형과 HMM Triple-Mode 라우팅"
next_desc: "가중합 방식의 CGCLayer와 블록 스케일링 방식의 CGCAttention — 두 CGC 변형의 수식과 언제 무엇을 쓰는지. Expert Collapse를 막는 entropy 정규화, 이종 차원 비대칭을 보정하는 차원 정규화, 그리고 HMM Triple-Mode 라우팅의 전체 구조."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 3편. 영문/국문 병렬로 PLE-1 → PLE-6 에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를 정리한다. 출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이다. 이번 3편은 PLE 모델이 실제로 받는 입력 구조(PLEClusterInput, 734D features 텐서)와, 본 프로젝트가 "이종 Expert Pool" 이라 부르는 7개의 구조적으로 다른 Shared Expert — 각자 고객을 다른 수학적 관점으로 해석하는 — 의 구성과 Forward 디스패치 전략을 다룬다.*

## PLEClusterInput — 입력 데이터 구조

`PLEClusterInput` 데이터클래스(라인 62~199)는 모델의 모든 입력을
캡슐화한다. 배치 단위로 디바이스 이동이 가능하며, HMM 모드 라우팅
로직을 내장한다.

### 전체 필드 사양

| 필드명 | 타입 | 차원 | 설명 / 출처 |
|---|---|---|---|
| `features` | `Tensor` | `[B, 734]` | base 238 + multi_source 91 + domain 159 + multidisciplinary 24 + model_derived 27 + extended_source 84 + merchant 21 (= 644D normalized) + raw_power_law 90D |
| `cluster_ids` | `Tensor` | `[B]` | GMM 클러스터 ID (0~19) |
| `cluster_probs` | `Tensor?` | `[B, 20]` | Soft routing용 클러스터 확률 (경계 샘플 처리) |
| `hyperbolic_features` | `Tensor?` | `[B, 20]` | MCC(8D) + Product(8D) + Region(4D) Poincare 좌표 |
| `tda_features` | `Tensor?` | `[B, 70]` | tda_short(24) + tda_long(36) + phase_transition(10) |
| `tda_short_diagrams` | `Tensor?` | `[B, 200, 3]` | Raw Persistence Diagram (birth, death, beta_idx) |
| `tda_short_mask` | `Tensor?` | `[B, 200]` | 유효 쌍 마스크 (패딩 제외) |
| `tda_long_diagrams` | `Tensor?` | `[B, 150, 3]` | Long-term Persistence Diagram |
| `tda_long_mask` | `Tensor?` | `[B, 150]` | Long 유효 쌍 마스크 |
| `tda_global_stats` | `Tensor?` | `[B, 30]` | short_global 12D + long_global 18D |
| `tda_phase_transition` | `Tensor?` | `[B, 10]` | 상전이 피처 |
| `hmm_journey` | `Tensor?` | `[B, 16]` | HMM Journey 모드 (10D base + 6D ODE dynamics) |
| `hmm_lifecycle` | `Tensor?` | `[B, 16]` | HMM Lifecycle 모드 |
| `hmm_behavior` | `Tensor?` | `[B, 16]` | HMM Behavior 모드 |
| `txn_seq` | `Tensor?` | `[B, 180, 16]` | 거래 시퀀스: card(8) + deposit(8) |
| `session_seq` | `Tensor?` | `[B, 90, 8]` | 세션 시퀀스 |
| `collaborative_features` | `Tensor?` | `[B, 64]` | LightGCN 사전 계산 임베딩 |
| `hierarchy_features` | `Tensor?` | `[B, 20]` | H-GCN 사전 계산 Poincare 좌표 |
| `customer_segment` | `Tensor?` | `[B]` | 0=anonymous, 1=cold_start, 2=warm_start |
| `coldstart_features` | `Tensor?` | `[B, 40]` | 콜드스타트 static features |
| `anonymous_features` | `Tensor?` | `[B, 15]` | 익명 static features |
| `targets` | `Dict?` | 가변 | 태스크별 정답 레이블 (학습 시) |

### 734D `features` 텐서 인덱스 매핑

`feature_schema.yaml` 의 `continuous` 리스트 순서가 데이터로더에서
텐서 컬럼 순서를 결정한다. 아래 표는 `features` 텐서의 정확한 인덱스
범위를 정의한다.

| 피처 그룹 | 차원 | 인덱스 범위 | 소계 | 세부 구성 |
|---|---|---|---|---|
| Base | 238D | `[0, 237]` | 238 | RFM 34D + Category 64D + Transaction_Stats 80D + Temporal 60D |
| Multi-source | 91D | `[238, 328]` | 91 | Deposit 20D + Membership 15D + Investment 18D + Credit 12D + Digital 14D + Product 12D |
| Domain | 159D | `[329, 487]` | 159 | TDA_short 24D + TDA_long 36D + Phase_Transition 10D + GMM 22D + Mamba 50D + Economics 17D |
| Multidisciplinary | 24D | `[488, 511]` | 24 | Chemical 6D + Epidemic 5D + Interference 8D + Crime 5D |
| Model-derived | 27D | `[512, 538]` | 27 | Bandit 4D + HMM_summary 5D + LNN 18D |
| Extended source | 84D | `[539, 622]` | 84 | Insurance 25D + Consultation 18D + Campaign 12D + Overseas 6D + OtherChannel 23D |
| Merchant hierarchy | 21D | `[623, 643]` | 21 | MCC_L1 4D + MCC_L2 4D + Brand 8D + Stats 4D + Radius 1D |

Domain 내부 세부 인덱스:

| 서브그룹 | 차원 | 인덱스 범위 | 설명 |
|---|---|---|---|
| TDA-Short | 24D | `[329, 352]` | 앱 로그 기반 단기 위상 패턴 (H0+H1, 90일 윈도우) |
| TDA-Long | 36D | `[353, 388]` | 금융 거래 기반 장기 위상 패턴 (H0+H1+H2, 12개월 윈도우) |
| Phase Transition | 10D | `[389, 398]` | W1 거리, 위상 변화량, 전이 확률/방향/크기 등 |
| GMM Cluster | 22D | `[399, 420]` | GMM 클러스터 소속 확률 + 거리 통계 |
| Mamba Temporal | 50D | `[421, 470]` | Mamba SSM 시계열 잠재 표현 |
| Income Decomposition | 8D | `[471, 478]` | 소득 구조 분해 (경제학 피처) |
| Financial Behavior | 9D | `[479, 487]` | 금융 행동 지표 (경제학 피처) |

Model-derived 내부 세부 인덱스:

| 서브그룹 | 차원 | 인덱스 범위 | 설명 |
|---|---|---|---|
| Bandit (MAB) | 4D | `[512, 515]` | Multi-Armed Bandit 탐색/활용 행동 통계 |
| HMM Summary | 5D | `[516, 520]` | 지배적 상태, 지속기간, 안정성, 엔트로피, 변화율 |
| LNN Model | 18D | `[521, 538]` | 분포 통계 4D + 주파수 4D + 변화점 3D + 자기상관 4D + 복잡성 3D |

> **`_FEATURE_GROUP_DIMS_ORDER` 순서 수정 완료.**
> `ple_cluster_adatt.py:407` 의 `_FEATURE_GROUP_DIMS_ORDER` 를
> `feature_schema.yaml` 순서에 맞게 수정 완료:
> base → multi_source → *domain* → multidisciplinary →
> model_derived → extended_source → merchant. 검증 기준:
> `feature_schema.yaml` → `task_feature_mapper.py:FEATURE_GROUP_DIMS`
> → `feature_integrator.py:EXPECTED_GROUP_DIMS_CAN`.

> **학부 수학 — 734차원 입력 벡터의 의미.** 신경망의 입력
> $\mathbf{x} \in \mathbb{R}^{734}$ 는 734개의 실수로 구성된 벡터다.
> 앞 644D 는 정규화된 피처, 뒤 90D 는 정규화 전 원시 power-law 피처
> 이다. 선형대수에서 $\mathbb{R}^n$ 은 $n$ 차원 실수 벡터 공간으로,
> 각 축이 하나의 피처에 대응한다. 예를 들어 $x_1$ 이 "월평균 소비
> 금액", $x_2$ 가 "최근 로그인 빈도"라면, 한 고객은 734차원 공간의 한
> *점* 으로 표현된다. 인간은 3차원까지만 시각적으로 이해할 수 있지만,
> 수학적 연산(내적, 노름, 사영)은 차원 수에 관계없이 동일하게 적용된다.
> 734D 공간에서도 두 고객 벡터의 코사인 유사도를 계산하면 "행동 패턴이
> 얼마나 비슷한가"를 정량화할 수 있다. *차원의 저주(Curse of
> Dimensionality)*: 고차원 공간에서는 데이터가 희소해져 거리 기반
> 방법이 비효율적이 된다. 이것이 Expert 네트워크가 차원을 64D 나 128D
> 로 압축하는 이유이며, 압축 과정에서 태스크에 유용한 정보만 남기는
> 것이 학습의 핵심이다.

### HMM 모드 라우팅

`set_hmm_routing()` 클래스 메서드(라인 173~186)는 모델 초기화 시 1회
호출되어 config 의 `hmm_triple_mode` 섹션에서 태스크별 HMM 모드 매핑을
구축한다.

```python
# ple_cluster_adatt.py:172-186 — config가 single source of truth
@classmethod
def set_hmm_routing(cls, hmm_config: dict) -> None:
    routing: Dict[str, str] = {}
    for mode in ["journey", "lifecycle", "behavior"]:
        for task in hmm_config.get(mode, {}).get("target_tasks", []):
            routing[task.lower().replace("-", "_")] = mode
    cls._default_hmm_routing = routing
```

`get_hmm_for_task()` (라인 188~198)는 태스크 이름으로 해당 HMM 텐서를
반환하며, 매핑에 없는 태스크는 `"behavior"` 모드를 기본값으로 사용한다.

## Shared Expert 결합 (512D)

### 7개 Shared Expert 구성

`_build_shared_experts()` (라인 395~451)에서
`SharedExpertFactory.create_from_config()` 를 호출하여 config 의
`shared_experts` 섹션에서 활성화된 Expert 를 동적으로 생성한다.

| Expert 이름 | 입력 | 출력 | 역할 |
|---|---|---|---|
| `unified_hgcn` | 47D | 128D | Hyperbolic GCN + Merchant 계층 구조 (hgcn+merchant_hgcn 통합) |
| `perslay` | 70D | 64D | Persistence Diagram 처리 (TDA 위상 피처) |
| `deepfm` | 정규화 644D | 64D | Feature Interaction (FM + Deep, 필드별 독립 임베딩 v3.11) |
| `temporal` | 시퀀스 | 64D | Temporal Ensemble (Mamba + LNN + Transformer) |
| `lightgcn` | 64D | 64D | Graph-based CF (사전 계산 임베딩) |
| `causal` | 정규화 644D | 64D | SCM/NOTEARS 기반 인과 관계 추출 |
| `optimal_transport` | 정규화 644D | 64D | Sinkhorn 기반 Wasserstein 거리 표현 |

$$\mathbf{h}_{shared} = [\text{unified\_hgcn}_{128D} \,\|\, \text{perslay}_{64D} \,\|\, \text{deepfm}_{64D} \,\|\, \text{temporal}_{64D} \,\|\, \text{lightgcn}_{64D} \,\|\, \text{causal}_{64D} \,\|\, \text{OT}_{64D}]$$

$$\dim(\text{shared\_concat}) = 6 \times 64 + 1 \times 128 = 512D$$

여기서 $\|$ 는 텐서 결합(concatenation). DeepFM/Causal/OT 는
`inputs.features[:, :644]` (정규화 644D) 를 입력으로 받는다.

> **수식 직관.** 이 수식은 7명의 서로 다른 전문가가 각자의 분석 결과를
> 한 줄로 이어 붙인 것이다. 직관적으로, 그래프 구조 분석(128D)과 위상
> 분석(64D), FM 교차(64D) 등 이질적인 시각의 결과물을 하나의 512차원
> 벡터로 합치면, 후속 CGC 게이트가 "이 고객에게는 어떤 전문가의 의견이
> 중요한가"를 태스크별로 판단할 수 있는 재료가 된다.

> **최신 동향.** 이종(heterogeneous) Expert 결합은 2024-2025년 추천
> 시스템 연구의 핵심 트렌드다. 기존 MoE 연구(MMoE, PLE)는 동일 구조의
> MLP Expert 를 사용했지만, 최근에는 GNN + Transformer + CNN 등 서로
> 다른 아키텍처를 Expert 로 결합하는 시도가 늘고 있다. Google 의
> *Multi-Aspect Expert Model* (MAEM, KDD 2024) 은 행동/컨텍스트/프로필
> 각각에 특화된 Expert 를 두어 YouTube 추천 성능을 개선했다. Meta 의
> *DHEN* (Deep Heterogeneous Expert Network, 2023) 은 이종 Expert 의
> 상호작용을 명시적으로 모델링하여 Instagram 피드 랭킹에 적용했다.
> 본 시스템의 7개 이종 Expert(GCN, PersLay, DeepFM, Temporal, LightGCN,
> Causal, OT) 결합은 이러한 최신 흐름과 정확히 부합하며, 단일
> 도메인 Expert 로는 포착할 수 없는 다면적(multi-aspect) 고객 표현을
> 구축한다.

### Expert별 Forward 디스패치

`_forward_shared_experts()` (라인 1416~1567)에서 Expert 이름에 따라
서로 다른 입력을 전달한다.

```python
# ple_cluster_adatt.py:1435-1565 — Expert별 디스패치 요약
for name, expert in self.shared_experts.items():
    if name in ("hgcn", "merchant_hgcn", "unified_hgcn"):
        # hierarchy_features(20D) + merchant slice(27D) = 47D
        out, hgcn_interpret, _ = expert(combined_input)
    elif name == "perslay":
        # Raw diagram mode 또는 pre-computed 70D fallback
        out, _ = expert(tda_short_diagrams / tda_features / zero)
    elif name == "deepfm":
        out, _ = expert(inputs.features[:, :644])   # 정규화 644D
    elif name == "temporal":
        out, _ = expert(txn_seq, session_seq, ...)  # 시퀀스
    elif name == "lightgcn":
        out, _ = expert(collaborative_features)  # 사전 계산 64D
    elif name in ("causal", "optimal_transport"):
        out, _ = expert(inputs.features[:, :644])   # 정규화 644D
```

### Zero Fallback 전략

모든 Expert 는 입력 데이터가 `None` 일 때 *zero 텐서 fallback* 을
수행한다. 이는 배치 내에 해당 피처가 없는 샘플이 있을 때 안전하게
처리하기 위함이다. CGC 게이팅이 이후 단계에서 해당 Expert 의 가중치를
자동으로 낮춘다.

> **⚠ Zero Fallback과 CGC의 상호작용.** Zero 출력을 가진 Expert 에
> 대해 CGC 가 높은 가중치를 부여하면 전체 표현이 희석된다. CGC 의
> `domain_experts` 초기 bias 가 이 문제를 완화하지만, 학습 초기에 특정
> Expert 의 입력이 대부분의 배치에서 누락되면 *dead expert* 현상이
> 발생할 수 있다. `_cgc_entropy_regularization` 으로 분산을 유도하여
> 부분 완화한다.

## 여기서 이어지는 것

지금까지 PLE 모델이 실제로 받아들이는 *형상* 을 살펴봤다 — 734D 의
정규화/원시 혼합 벡터, 그리고 그 위에 얹히는 GMM 클러스터 확률, TDA
persistence diagram, HMM triple-mode 텐서, 두 종류의 시퀀스, 그리고
사전 계산된 GCN 임베딩들. 그 다음 단계가 7개의 이종 Shared Expert 로,
각자 다른 수학적 렌즈로 같은 고객을 본다 — Hyperbolic GCN 은 계층
구조로, PersLay 는 위상으로, DeepFM 은 필드 교차로, Temporal 은
시계열로, Causal 은 인과 그래프로, Optimal Transport 는 분포 간 거리
로. 512D 의
concatenation 은 이 7개의 관점을 한 벡터로 모아둔 것이고, 여기서부터는
"어떤 태스크가 어떤 Expert 의 의견을 얼마나 믿을 것인가" 가 문제가
된다. 그것이 CGC 게이팅의 역할이며, **PLE-4** 에서 두 가지 변형 —
가중합 방식의 CGCLayer 와 블록 스케일링 방식의 CGCAttention — 을
수식으로 따라간다.
