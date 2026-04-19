---
title: "[Study Thread] 에피소드 1 — PLE 기초: Shared-Bottom부터 CGC까지"
date: 2026-04-19 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, mmoe, cgc, mtl, architecture]
lang: ko
series: study-thread
part: 1
alt_lang: /2026/04/19/study-ep1-ple-foundations-en/
next_title: "에피소드 2 — 이종 전문가 Basket 설계"
next_desc: "동일 구조 expert pool 대신 8개 이종 도메인 expert를 결합하는 패턴. FeatureRouter로 각 expert에 피처 서브셋을 라우팅하는 구조, expert 선정 기준, pool/basket 패턴 자체의 설계 철학."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/tech_ref_ple_adatt.pdf
source_label: "PLE + adaTT 기술 참조서 §1 (KO, PDF)"
---

*시리즈 "Study Thread" 1편. 본 프로젝트의 PLE 아키텍처 뒤에 있는
논문과 수학 기초를 영문/국문 병렬로 정리한다.*

## 왜 다루는가

하나의 고객 표현으로부터 13개 태스크를 동시에 예측해야 한다 —
churn signal, next best action, MCC 트렌드, 6개 상품 획득 확률 등.
Multi-task learning (MTL) 은 자연스러운 프레이밍이지만, MTL 안에서의
아키텍처 선택은 자명하지 않다. Shared-Bottom, MMoE, PLE 는 MTL
아키텍처의 세 단계 진화이며, 각 단계는 이전 단계의 특정 실패 모드를
교정한 것이다. *왜 PLE가 이겼는지*를 이해하려면 앞의 두 모델이
무엇에서 실패했는지 봐야 한다. 이번 편은 세 단계를 따라가서, 우리가
프로덕션에서 실제로 돌리고 있는 CGC 수식까지 도달한다.

## 왜 MTL인가

하나의 공유 표현이 여러 head 에 봉사해야 한다. 공유 trunk 가 어느 한
태스크에 과적합하면 나머지 태스크에 더 이상 유용하지 않으므로,
*모든* 태스크에 동시에 도움이 되는 패턴만 살아남는다. 이것이
태스크 간 상호 정규화 (inter-task regularization) 의 논거이며,
$K$ 개 모델을 따로 학습하는 대신 MTL 을 하나의 패러다임으로 두는
이유의 전부이다. 총 손실은 가중합이다:

$$\mathcal{L}_{MTL} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_k(f_k(\mathbf{h}_{shared}(\mathbf{x})))$$

어려운 부분은 수식이 아니다. 어려운 부분은 $\mathbf{h}_{shared}$ 가
아키텍처로서 어떤 모습이어야 하는가이다. 순진한 답 (한 trunk, $K$ 개
head) 은 태스크들이 표현이 무엇을 인코딩해야 하는지에 동의하지
않기 시작하는 순간 무너진다.

## 세 단계 진화

### Shared-Bottom (Caruana, 1997)

모든 태스크가 단일 trunk 을 공유한 뒤 태스크별 head 로 분기한다:

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

구현이 단순하고 파라미터 수가 최소다. 실패 모드는 **negative
transfer** — 두 태스크가 공유 trunk 에 서로 다른 것을 인코딩하길
원할 때, 한 태스크의 gradient 업데이트가 다른 태스크에 적극적으로
해를 끼친다. 상관이 낮은 태스크들에서는 이것이 심각해져서, 차라리
태스크별로 따로 학습하는 편이 나았을 정도가 된다.

### MMoE (Ma et al., KDD 2018)

MMoE 는 단일 trunk 을 동일 구조 $N$ 개 expert 로 대체한다.
태스크별 softmax gate 가 어떤 expert 를 섞을지 결정한다:

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot f_i^{expert}(\mathbf{x}), \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x})$$

이제 각 태스크가 다른 expert 조합을 가질 수 있어, 의견이 다른 두
태스크가 서로를 우회할 수 있다. 실무적으로 Shared-Bottom 보다
낫다. 새로운 실패 모드는 **expert collapse** — 손실 함수에서
서로 다른 태스크의 gate 가 실제로 발산하도록 강제하는 것이 없으며,
일반적인 학습률에서는 종종 같은 expert 로 수렴해버린다. 그렇게 되면
파라미터만 더 든 Shared-Bottom 을 다시 만든 셈이다.

### PLE (Tang et al., RecSys 2020)

PLE 는 expert 를 **Shared Expert** $\mathcal{E}^s$ 와
**Task-specific Expert** $\mathcal{E}^k$ 로 명시적으로 분리한다.
각 태스크의 표현은 공유 풀 출력과 *해당 태스크* 전용 출력의
CGC-게이팅된 결합이다:

$$\mathbf{h}_k = \sum_{i=1}^{|\mathcal{E}^s|} g_{k,i}^s \cdot \mathbf{e}_i^s + \sum_{j=1}^{|\mathcal{E}^k|} g_{k,j}^k \cdot \mathbf{e}_j^k$$

구조적 분리가 핵심의 전부이다. Task-specific expert 는 한 태스크만
신경 쓰는 패턴을 다른 태스크에 간섭하지 않고 학습한다. Shared
expert 는 cross-task 신호를 학습한다. 역할이 더 이상 대칭이 아니므로
expert collapse 에 빠지기가 훨씬 어렵다. PLE 레이어를 여러 층 쌓으면
progressive extraction 이 된다 — 아래쪽은 저수준 공유 피처, 위쪽은
점점 더 특화된 피처.

| 구분 | Shared-Bottom | MMoE | PLE |
|---|---|---|---|
| Expert 구조 | 단일 shared trunk | $N$ 개 expert 전체 공유 | Shared + Task-specific 분리 |
| 게이팅 | 없음 | 태스크별 softmax gate | CGC: shared + task expert 결합 |
| Negative transfer | 높음 | 중간 (expert collapse) | 낮음 (명시적 분리) |
| Expert collapse | N/A | 높음 | 낮음 |

## CGC의 수학

CGC 는 Customized Gate Control 의 약자이다. PLE 의 핵심에 있는
gate — 태스크별로 각 expert 의 출력을 얼마나 사용할지를 결정하는
바로 그 메커니즘이다.

본 구현에는 출력 형태가 다른 두 가지 CGC 변형이 존재한다.

**CGCLayer (가중합 방식).** 원본 PLE 논문의 CGC 이다. 태스크별
gate 가 모든 expert 출력의 가중합을 산출하며, expert 수에 무관하게
출력 차원이 `expert_hidden_dim` 으로 고정된다:

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot \mathbf{e}_i, \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x}) \in \mathbb{R}^N$$

**CGCAttention (블록 스케일링 방식).** 우리가 실제로 돌리는 변형이다.
이종 expert (출력 차원과 내부 구조가 모두 다른) 환경에서 가중합은
정보를 너무 많이 버리므로, 대신 모든 expert 출력을 연결한 뒤 각
블록을 태스크별 attention weight 로 스케일링한다:

$$\mathbf{w}_k = \text{Softmax}(\mathbf{W}_k \cdot \mathbf{h}_{shared} + \mathbf{b}_k) \in \mathbb{R}^8$$

$$\tilde{\mathbf{h}}_{k,i} = w_{k,i} \cdot \mathbf{h}_i^{expert} \quad \text{for } i = 1, \dots, 8$$

$$\mathbf{h}_k^{cgc} = [\tilde{\mathbf{h}}_{k,1} \,\|\, \tilde{\mathbf{h}}_{k,2} \,\|\, \dots \,\|\, \tilde{\mathbf{h}}_{k,8}] \in \mathbb{R}^{576}$$

출력은 모든 태스크에 대해 동일한 576D 이지만, 태스크마다 expert
기여 비중이 다르다. Transformer 와의 유비는 정확하다 — gate 는
**Query**, shared representation 은 **Key**, 각 expert 출력은
선택적으로 통과되는 **Value** 다.

## 실제 구현에서 배운 것들

원본 PLE 논문에는 없지만 PLE 를 실제 시스템에 넣는 순간 필요한 두
가지 보정.

**Entropy 정규화.** PLE 의 구조적 분리가 있더라도, 손실 함수에서
어느 한 expert 만 선택하고 나머지 7개를 무시하는 태스크 gate 를
명시적으로 처벌하는 항이 없다. Gate 분포에 대한 entropy 정규화 항을
추가한다:

$$\mathcal{L}_{entropy} = \lambda_{ent} \cdot \left(-\frac{1}{|\mathcal{T}|}\right) \sum_{k \in \mathcal{T}} H(\mathbf{w}_k), \quad H(\mathbf{w}_k) = -\sum_{i=1}^{8} w_{k,i} \cdot \log(w_{k,i})$$

$\lambda_{ent} = 0.01$ 에서, 음의 엔트로피를 최소화하면 gate
분포가 균등 방향으로 밀려나고, 그 결과 expert 들이 회전 안에 남게
된다. 비용은 적고 효과는 가중치에서 짐작되는 것보다 크다.

**차원 정규화.** Expert 출력 차원이 이질적일 때 — 우리의 경우
128D expert 1개 (`unified_hgcn`) 와 64D expert 7개 — 더 큰
expert 가 gate 가중이 적용되기 *전에* 이미 concat 후의 magnitude
를 지배한다. 해법은 expert 별 스케일 인자다:

$$\text{scale}_i = \sqrt{\text{mean\_dim} / \text{dim}_i}, \quad \text{mean\_dim} = (128 + 64 \times 7) / 8 = 72.0$$

128D expert 에는 $\approx 0.750$ (감쇠), 64D expert 에는
$\approx 1.061$ (증폭) 이 부여된다. 이제 gate 는 출력 폭이 우연히
넓은 expert 에 끌리는 대신, 내용에 기반해서 expert 의 중요도를
결정한다.

## 다음 편 예고

에피소드 2 는 이종 전문가 Basket 자체로 이동한다 — PLE 의 동일
구조 expert pool 을 8개의 구조적으로 다른 도메인 expert (DeepFM,
LightGCN, Unified HGCN, Temporal, PersLay, Causal, Optimal
Transport, RawScale) 로 대체한 설계 결정. FeatureRouter 패턴,
expert 별 입력 차원, 선정 기준, 그리고 pool/basket 패턴이 단순한
구현 트릭이 아니라 그 자체로 하나의 설계 철학인 이유.

이번 편의 원문 자료는 PLE + adaTT 기술 참조서 §1 (frontmatter
링크). 이 시리즈는 참조서를 일반 독자용으로 각색하고 참조서가
생략한 구현 맥락을 더한다.
