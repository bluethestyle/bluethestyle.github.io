---
title: "[Study Thread] PLE-4 — CGC 게이팅의 두 단계(CGCLayer + CGCAttention)와 HMM Triple-Mode 라우팅"
date: 2026-04-19 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, cgc, hmm, regularization]
lang: ko
series: study-thread
part: 4
alt_lang: /2026/04/19/ple-4-cgc-hmm-routing-en/
next_title: "PLE-5 — GroupTaskExpertBasket · Logit Transfer · Task Tower"
next_desc: "GroupTaskExpertBasket v3.2 (GroupEncoder + ClusterEmbedding)이 태스크별 전용 Expert를 만드는 방식, 태스크 간 명시적 정보 전달을 수행하는 Logit Transfer의 3가지 모드, 그리고 최종 예측을 수행하는 Task Tower의 구조."
next_status: published
---

*"Study Thread" 시리즈의 PLE 서브스레드 4편. 영문/국문 병렬로 PLE-1 → PLE-6
에 걸쳐 본 프로젝트의 PLE 아키텍처 뒤에 있는 논문과 수학 기초를 정리한다.
출처는 온프렘 프로젝트 `기술참조서/PLE_기술_참조서` 이다. 이번 4편은
PLE의 심장인 CGC 게이트를 두 단계로 구성한다 — 1단계: 원본 논문의
CGCLayer가 Shared + Task Expert를 함께 가중합으로 섞고, 2단계:
CGCAttention이 그 뒤 Shared Expert concat에 태스크별 블록 스케일링을
얹는다 — 의 수식과 Expert Collapse를 막는 정규화 기법들, 그리고 HMM
기반의 Triple-Mode 라우팅을 다룬다.*

## CGC (Customized Gate Control) — 태스크별 Expert 가중치

### 이론적 배경

CGC는 MMoE(Ma et al., KDD 2018)의 게이팅 메커니즘을 확장한 것으로,
태스크별 독립 게이트가 Shared Expert 출력에 서로 다른 가중치를
적용하여 태스크-Expert 친화도를 학습한다. 원본 PLE 논문(Tang et al.,
RecSys 2020)의 **CGCLayer**가 "태스크별 gate가 (Shared ∪ Task) Expert
전체를 벡터 가중합"으로 섞는 1단계 primary gate를 담당한다. 그 위에
본 구현의 **CGCAttention**이 "이미 concat된 Shared Expert 출력에
태스크별 블록 스케일링을 얹는" 2단계 secondary attention으로 붙는다.
이 4편은 두 단계의 역할, CGCAttention에 얹은 entropy 정규화와 차원
정규화, 그리고 그 뒤에 붙는 HMM Triple-Mode 라우팅을 다룬다.

### 1단계 — CGCLayer: 논문 원형의 Shared + Task 가중합

1단계 primary gate는 원본 논문의 CGCLayer를 그대로 사용한다. 태스크
$k$ 의 gate 는 Shared Expert 와 태스크 $k$ 전용 Expert 를 *함께*
concat 한 축 위에서 Softmax 가중합을 계산한다. 태스크별 전용 Expert 가
이 레이어 안에 살아 있다.

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot \mathbf{h}_i^{\text{all}}, \quad \mathbf{h}^{\text{all}} = [\mathbf{h}^{\text{task}}_k \,\|\, \mathbf{h}^{\text{shared}}]$$

$$\mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{h}_{shared}) \in \mathbb{R}^{N}, \quad N = |\text{shared}| + |\text{task}_k|$$

- **역할**: 태스크 $k$ 를 담당하는 *primary* 게이트. Shared Expert 와
  Task-$k$ Expert 를 하나의 concat 축 위에서 Softmax 로 가중합하여
  고정 차원 벡터 하나를 뽑는다. Tang et al. (RecSys 2020) 의 원 수식을
  그대로 따른다.
- **특성**: 태스크별 전용 Expert 가 바로 이 레이어 안에서 선택된다 —
  태스크마다 "A Shared 60%, B Shared 15%, Task-k 전용 25%" 처럼
  Shared 와 Task 풀을 동시에 섞는 자연스러운 혼합이 가능하다. 출력은
  태스크당 단일 고정 차원 벡터.
- **이종 차원 문제**: Shared Expert 풀은 출력 차원이 이종이다
  (unified_hgcn 128D, 나머지 perslay/temporal/deepfm/lightgcn 등 64D).
  이것이 2단계 CGCAttention 이 Shared concat 위에 얹힌 이유이며,
  1단계 CGCLayer 자체는 그대로 실행된다.

### 2단계 — CGCAttention: 이종 Shared Expert concat에 얹은 per-task block attention

2단계는 1단계와 직교적으로 붙는다. CGCLayer가 이미 태스크별 gated
출력을 뽑은 것과 *별도로*, CGCAttention은 Shared Expert concat (512D)
에 태스크별 블록 스케일링을 얹어 *shared만의 per-task 표현*을 만든다.
이 두 경로의 출력이 downstream (Logit Transfer + Task Tower input)
에서 함께 쓰인다.


`_build_cgc()` (라인 566~677)에서 태스크별 독립적인
`nn.Sequential(Linear + Softmax)` 모듈을 `nn.ModuleDict`로 관리한다.

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^7$$

$$\tilde{\mathbf{h}}_{k,i} = w_{k,i} \cdot \mathbf{h}_i^{expert} \quad \text{for } i = 1, \ldots, 7$$

$$\mathbf{h}_k^{cgc} = [\tilde{\mathbf{h}}_{k,1} \,\|\, \tilde{\mathbf{h}}_{k,2} \,\|\, \ldots \,\|\, \tilde{\mathbf{h}}_{k,7}] \in \mathbb{R}^{512}$$

여기서 $\mathbf{W}_k \in \mathbb{R}^{7 \times 512}$ 는 태스크 $k$ 의
gate 가중치이고, $\mathbf{h}_i^{expert}$ 는 $i$ 번째 Expert의 출력 블록
(64D 또는 128D), $w_{k,i}$ 는 태스크 $k$ 가 Expert $i$ 에 부여하는
attention 가중치이다.

> **수식 직관.** 첫 번째 식은 512D 공유 표현을 보고 7개 Expert 각각의
> "관련성 점수"를 산출한 뒤 Softmax로 확률화하는 과정이다. 두 번째
> 식은 이 확률(스칼라)을 각 Expert 출력 블록에 곱해 중요도를
> 조절한다. 세 번째 식은 가중 조절된 블록들을 다시 이어 붙여 원래와
> 동일한 512D를 복원한다. 결과적으로, 같은 512D 벡터라도 태스크마다
> Expert별 기여 비중이 다르게 조합된다.

> **차원 유지 설계.** CGCAttention은 512D 입력을 512D 출력으로
> 변환한다. Expert별 블록에 스칼라 가중치를 곱하는 *블록 스케일링
> 방식*이므로 기존 파이프라인과 하위 호환된다. 가중치 합이 1
> (Softmax)이므로 출력 스케일이 보존된다.

> **역사적 배경.** CGC(Customized Gate Control)는 PLE 논문(Tang et
> al., RecSys 2020)에서 MMoE의 gate를 확장한 개념으로 처음 명명되었다.
> MMoE(Ma et al., KDD 2018)의 gate가 모든 Expert를 동등하게 취급하는
> 반면, CGC는 Shared Expert와 Task-specific Expert를 구분하여
> 게이팅한다. Attention 메커니즘과의 연결은 Transformer(Vaswani et
> al., NeurIPS 2017)의 Scaled Dot-Product Attention에서 영감을 받은
> 것으로, "관련성에 비례하여 정보를 선택적으로 결합"하는 동일 원리가
> 토큰 간(Transformer), Expert 간(CGC), 헤드 간(Multi-Head Attention)
> 등 다양한 단위에 적용된 것이다. 본 구현의 CGC는 원 논문의 CGCLayer를
> 유지하면서, 추가로 블록 스케일링 + 차원 정규화를 적용하여 이종
> Expert 차원 불일치를 처리한다.

### 초기 bias 설정 — domain_experts

`_build_cgc()` (라인 621~649)에서 각 태스크의 config `domain_experts`
필드를 읽어 초기 bias를 설정한다. weight는 0으로 두고, 태스크가
"선호"하는 Expert에는 `bias_high = 1.0`, 나머지에는 `bias_low = -1.0`
을 준다. 학습 초기에 Softmax 출력이 도메인 지식에 부합하는 분포에서
출발하게 만드는 소프트 프라이어다.

```python
# ple_cluster_adatt.py:626-638
bias_high = float(cgc_config.get("bias_high", 1.0))
bias_low = float(cgc_config.get("bias_low", -1.0))
linear_layer.weight.zero_()          # weight는 0 시작
for i, expert_name in enumerate(expert_names):
    if expert_name in domain_experts:
        linear_layer.bias[i] = bias_high   # 선호 Expert
    else:
        linear_layer.bias[i] = bias_low    # 비선호 Expert
```

| 태스크 | domain_experts (bias_high=1.0) |
|---|---|
| CTR | `perslay`, `temporal`, `unified_hgcn` |
| CVR | `perslay`, `temporal`, `unified_hgcn` |
| Churn | `perslay`, `temporal` |
| Retention | `perslay`, `temporal` |
| NBA | `perslay`, `unified_hgcn`, `lightgcn` |
| Life-stage | `perslay`, `temporal` |
| Balance_util | `temporal` |
| Engagement | `temporal` |
| LTV | `temporal`, `deepfm` |
| Channel | `temporal` |
| Timing | `temporal` |
| Spending_category | `unified_hgcn`, `perslay` |
| Consumption_cycle | `temporal` |
| Spending_bucket | `deepfm` |
| Brand_prediction | `unified_hgcn` |
| Merchant_affinity | `unified_hgcn`, `temporal` |

### Entropy 정규화 (v2.3) — Expert Collapse 방지

`_cgc_entropy_regularization()` (라인 748~768)은 CGC attention 분포의
엔트로피를 최대화하여 **Expert Collapse**를 방지한다.

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left( -\frac{1}{|\mathcal{T}|} \right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k)$$

$$H(\mathbf{w}_k) = -\sum_{i=1}^{7} w_{k,i} \cdot \log(w_{k,i})$$

여기서 $\mathcal{T}$ 는 활성화된 태스크 집합, $\lambda_{ent} = 0.01$
(config: `cgc.entropy_lambda`) 이며, 음의 엔트로피를 *최소화*하면
엔트로피가 *증가*하여 분산이 유도된다.

> **수식 직관.** 엔트로피 $H$ 는 "gate 분포가 얼마나 고르게 퍼져
> 있는가"의 척도다. 하나의 Expert에 가중치가 몰리면 $H$ 가 작고,
> 고르면 $H$ 가 크다. 이 손실 항은 $-H$ 를 최소화하므로 $H$ 를 키우는
> 방향, 즉 gate가 여러 Expert를 골고루 참조하도록 유도한다.
> $\lambda_{ent}$ 가 클수록 균등 분산 압력이 강해진다.

> **학부 수학 — 엔트로피의 정보이론적 유도.** Shannon(1948)은
> "불확실성의 측도"를 공리적으로 유도했다. 확률 분포
> $\mathbf{w} = (w_1, \ldots, w_n)$ 에 대해 다음 세 공리를 만족하는
> 유일한 함수가 엔트로피다: (1) 연속성 — $w_i$ 가 조금 변하면 $H$ 도
> 조금 변한다, (2) 최대성 — 균등 분포일 때 $H$ 가 최대, (3) 결합 —
> 독립 사건의 엔트로피는 덧셈적. *구체적 계산 예시*: Expert 7개에 대해
> 균등 분포 $w_i = 1/7$ 이면
> $H = -7 \times (1/7) \times \log(1/7) = \log(7) \approx 1.946$ 비트
> (최대 엔트로피). 한 Expert에 집중
> $\mathbf{w} = (1, 0, \ldots, 0)$ 이면
> $H = -1 \times \log(1) = 0$ (최소 엔트로피). 만약
> $\mathbf{w} = (0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06)$ 이면
> $H \approx 1.32$ — 최대의 약 68%만 활용. 엔트로피 정규화는 이 값을
> 최대 쪽으로 밀어 Expert 활용을 분산시킨다.

> **⚠ Expert Collapse 위험.** CGC entropy lambda가 0이면 정규화
> 비활성화. 이 경우 학습 중 특정 Expert(특히 unified_hgcn 128D)에
> attention이 집중되어 나머지 Expert의 gradient가 소실될 수 있다.
> `entropy_lambda=0.01` 이 기본값이며, 실험적으로 0.005~0.02 범위가
> 안정적이다.

### CGC Attention 적용 (forward)

`_apply_cgc_attention()` (라인 679~725)에서 Expert별 블록에 가중치를 곱한다.

```python
# ple_cluster_adatt.py:708-725
parts = []
offset = 0
for i, dim in enumerate(self._cgc_expert_dims):
    block = shared_concat[:, offset:offset + dim]
    # v3.3: 차원 정규화 — 128D Expert는 감쇠, 64D는 증폭
    if self._cgc_dim_normalize and dim != self._cgc_mean_dim:
        scale = math.sqrt(self._cgc_mean_dim / dim)
        block = block * scale
    part = block * attention_weights[:, i:i+1]  # broadcast
    parts.append(part)
    offset += dim
return torch.cat(parts, dim=-1)
```

### 차원 정규화 (v3.3) — 이종 Expert 차원 보정

`dim_normalize=true` 일 때 Expert 출력 차원 비대칭(128D vs 64D)에 의한
기여도 불균형을 스케일링으로 보정한다.

$$\text{scale}_i = \sqrt{\frac{\text{mean\_dim}}{\text{dim}_i}}$$

$$\text{mean\_dim} = \frac{128 + 64 \times 6}{7} \approx 73.14$$

- unified_hgcn (128D): scale $= \sqrt{73.14 / 128} \approx 0.756$ (감쇠)
- 나머지 Expert (64D): scale $= \sqrt{73.14 / 64} \approx 1.069$ (증폭)
- 동일 attention = 동일 L2 기여

> **수식 직관.** unified_hgcn(128D)은 다른 Expert(64D)보다 출력 차원이
> 2배 크므로, 동일한 attention 가중치를 받더라도 L2 노름 기준 기여가
> 과대하다. 이 스케일링은 "차원이 큰 Expert는 줄이고 작은 Expert는
> 키워서" attention $w_{k,i} \approx 0.143$ (균등, $1/7$)일 때 모든 Expert의 실질
> 기여가 동등하도록 보정한다.

### CGC Freeze 동기화

`on_epoch_end()` (라인 1921~1942)에서 adaTT `freeze_epoch` 에 도달하면
CGC attention 파라미터도 함께 고정한다.

```python
# ple_cluster_adatt.py:1934-1942
if (freeze_epoch is not None
        and epoch >= freeze_epoch
        and not self._cgc_frozen.item()):
    for param in self.task_expert_attention.parameters():
        param.requires_grad = False
    self._cgc_frozen.fill_(True)
```

> **CGC-adaTT 동기화 이유.** adaTT가 전이 가중치를 고정한 뒤에도 CGC가
> 계속 학습하면, 두 메커니즘이 상충하는 방향으로 진화할 수 있다.
> 동시 고정으로 학습 후반부의 안정성을 보장한다.

## HMM Triple-Mode 라우팅

### 3개 HMM 모드

HMM Triple-Mode (v2.0)는 고객의 행동을 3가지 시간 스케일로 분리하여
태스크별로 가장 적합한 행동 모드를 주입한다.

| 모드 | 입력 | 시간 스케일 | 대상 태스크 |
|---|---|---|---|
| Journey | 16D | daily | CTR, CVR, Engagement, Uplift |
| Lifecycle | 16D | monthly | Churn, Retention, Life-stage, LTV |
| Behavior | 16D | monthly | NBA, Balance_util, Channel, Timing, Spending_category, Consumption_cycle, Spending_bucket, Merchant_affinity, Brand_prediction |

각 모드는 10D base 상태 확률 + 6D ODE dynamics bridge로 구성된다.

> **역사적 배경 — Hidden Markov Model의 기원과 발전.** HMM(Hidden
> Markov Model)은 *Baum & Petrie (1966)* 이 통계적 언어 모델링을 위해
> 정식화하였다. 핵심 아이디어는 "관측 가능한 사건(observation) 뒤에
> 관측 불가능한 은닉 상태(hidden state)가 존재하고, 상태 간 전이가
> Markov 성질을 따른다"는 것이다. 1970년대 *Rabiner & Juang* 이 음성
> 인식에 체계적으로 적용하여 HMM이 대중화되었고, 이후
> 생물정보학(유전자 서열 분석), 금융(시장 상태 추정), NLP(품사 태깅)
> 등에 확산되었다. 본 시스템에서는 고객의 관측 가능한 행동(거래,
> 로그인) 뒤에 숨겨진 "여정 상태(journey)", "생애주기
> 상태(lifecycle)", "행동 패턴 상태(behavior)"를 HMM으로 추정하고, 각
> 태스크에 가장 적합한 시간 스케일의 상태 정보를 주입한다. ODE
> dynamics bridge는 *Neural ODE (Chen et al., NeurIPS 2018)* 에서
> 영감을 받아 이산 HMM 상태를 연속 시간으로 보간(interpolation)하는
> 확장이다.

### HMM 프로젝터 구조

`_build_hmm_projectors()` (라인 452~496)에서 모드별 프로젝터를 생성한다.

$$\mathbf{h}_{hmm}^m = \text{SiLU}(\text{LayerNorm}(\text{Linear}_{16 \to 32}(\mathbf{x}_{hmm}^m)))$$

여기서 $m \in \{\text{journey}, \text{lifecycle}, \text{behavior}\}$
이고, 각 프로젝터는 독립 학습된다.

> **수식 직관.** 이 수식은 HMM이 출력한 16차원 상태 벡터(10D 상태 확률
> + 6D ODE 동역학)를 모델 내부에서 사용하기 좋은 32차원으로 확장하는
> 과정이다. Linear로 차원을 키운 뒤, LayerNorm으로 스케일을 안정화하고,
> SiLU 활성화로 비선형성을 부여한다. 세 모드
> (journey/lifecycle/behavior) 각각이 독립 프로젝터를 가지므로, "일별
> 여정 패턴"과 "월별 생애주기 패턴"이 서로 다른 변환을 학습한다.

> **학부 수학 — SiLU 활성화 함수와 비선형성의 필요성.**
> *SiLU(Sigmoid Linear Unit)* 는
> $\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}$ 로
> 정의된다. $\sigma(x)$ 는 시그모이드 함수로, 입력을 $[0, 1]$ 범위로
> 압축하는 "부드러운 스위치"다. SiLU는 $x$ 에 이 스위치를 곱하여
> "양수 입력은 거의 그대로, 음수 입력은 부드럽게 억제"한다. *왜
> 비선형 활성화가 필요한가?* Linear 변환만 쌓으면
> $\mathbf{W}_2 (\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x}$
> 로 하나의 Linear와 동치이다. 비선형 함수를 사이에 넣어야 레이어를
> 쌓는 의미가 생긴다. *활성화 함수 비교*: ReLU ($\max(0, x)$)는
> $x < 0$ 에서 기울기가 0이 되어 "뉴런 사망" 문제가 있고, GELU
> ($x \cdot \Phi(x)$, Gaussian CDF)는 SiLU와 유사하나 계산이 더 비싸다.
> SiLU는 ReLU의 경량성과 GELU의 부드러움을 절충한 것으로, Mish
> ($x \cdot \tanh(\text{Softplus}(x))$)와 함께 2020년 이후 표준 활성화
> 함수로 자리잡았다.

```python
# ple_cluster_adatt.py:482-486
self.hmm_projectors[mode] = nn.Sequential(
    nn.Linear(hmm_dim, proj_dim),     # 16 → 32
    nn.LayerNorm(proj_dim),
    nn.SiLU(),
)
```

### 학습 가능한 Default Embedding

HMM 피처가 없는 샘플(all-zero row)에 대해 zero 대신 *학습 가능한
default embedding* 을 사용한다 (라인 488~493).

```python
# ple_cluster_adatt.py:488-493
self.hmm_default_embeddings = nn.ParameterDict({
    mode: nn.Parameter(torch.zeros(proj_dim))
    for mode in ["journey", "lifecycle", "behavior"]
})
```

`_forward_hmm_projectors()` (라인 1365~1414)에서 샘플별 마스킹으로
유효 샘플만 프로젝션하고, 무효 샘플은 default embedding으로
대체한다. Journey(16D) / Lifecycle(16D) / Behavior(16D) 각각이 독립
프로젝터(16→32D)를 거쳐, CTR/CVR/Engagement, Churn/Ret/Life-stage/LTV,
NBA/Balance/Channel/... 순으로 태스크 그룹에 주입된다.

## 다음 단계

CGCAttention은 "공유 표현에서 태스크별로 다른 혼합을 뽑는" 장치이고,
HMM Triple-Mode는 "시간 스케일이 다른 행동 신호를 태스크 그룹별로
라우팅"하는 장치다. 두 경로 모두 *공유된 Expert 풀* 위에서 움직인다.
다음 **PLE-5** 에서는 반대 방향 — 태스크별로 아예 *전용 Expert 바구니*
를 만드는 GroupTaskExpertBasket v3.2(GroupEncoder + ClusterEmbedding),
그리고 태스크 타워들 사이에서 정보를 명시적으로 흘려주는 Logit Transfer
의 3가지 모드, 마지막 Task Tower 구조를 다룬다.
