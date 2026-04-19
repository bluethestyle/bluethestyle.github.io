---
title: "[Study Thread] PLE-1 — MTL과 게이트드 전문가로의 진화 (Shared-Bottom → MMoE)"
date: 2026-04-19 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, ple, mmoe, mtl, shared-bottom]
lang: ko
series: study-thread
part: 1
alt_lang: /2026/04/19/ple-1-mtl-evolution-en/
next_title: "PLE-2 — Progressive Layered Extraction: 명시적 전문가 분리와 CGC 게이트"
next_desc: "Shared/Task Expert를 명시적으로 분리한 PLE(Tang et al., 2020) 아키텍처. CGC 게이트의 두 변형 — 가중합 방식의 CGCLayer와 블록 스케일링 방식의 CGCAttention — 의 수식, 그리고 Expert Collapse를 막는 entropy 정규화 및 이종 차원 보정까지."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/tech_ref_ple_adatt.pdf
source_label: "PLE + adaTT 기술 참조서 §1 (KO, PDF)"
---

*"Study Thread" 시리즈의 PLE 서브스레드 1편. PLE-1 → PLE-6 로
이어지는 영문/국문 병렬 서브스레드로, 본 프로젝트의 PLE 아키텍처
뒤에 있는 논문과 수학 기초를 정리한다. PLE-6 편에서는 PLE + adaTT
기술 참조서 전체 PDF 를 첨부한다.*

## 왜 다루는가

하나의 고객 표현으로부터 13개 태스크를 동시에 예측해야 한다 —
churn signal, next best action, MCC 트렌드, 6개 상품 획득 확률 등.
Multi-task learning (MTL) 은 자연스러운 프레이밍이지만, MTL 안에서의
아키텍처 선택은 자명하지 않다. Shared-Bottom 과 MMoE 는 MTL
아키텍처의 첫 두 단계이며, 각 단계는 특정 방식으로 실패하여 다음
단계를 유도한다. 이번 편은 그 두 실패를 따라간다 — 실제로 작동하는
세 번째 단계인 PLE 는 **PLE-2** 의 주제이다.

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

## Shared-Bottom (Caruana, 1997)

모든 태스크가 단일 trunk 을 공유한 뒤 태스크별 head 로 분기한다:

$$\mathbf{h} = f_{shared}(\mathbf{x}) \quad \rightarrow \quad \hat{y}_k = f_k^{tower}(\mathbf{h})$$

구현이 단순하고 파라미터 수가 최소다. 실패 모드는 **negative
transfer** — 두 태스크가 공유 trunk 에 서로 다른 것을 인코딩하길
원할 때, 한 태스크의 gradient 업데이트가 다른 태스크에 적극적으로
해를 끼친다. 상관이 낮은 태스크들에서는 이것이 심각해져서, 차라리
태스크별로 따로 학습하는 편이 나았을 정도가 된다.

## MMoE (Ma et al., KDD 2018)

MMoE 는 단일 trunk 을 동일 구조 $N$ 개 expert 로 대체한다.
태스크별 softmax gate 가 어떤 expert 를 섞을지 결정한다:

$$\mathbf{h}_k = \sum_{i=1}^{N} g_{k,i} \cdot f_i^{expert}(\mathbf{x}), \quad \mathbf{g}_k = \text{Softmax}(\mathbf{W}_k^{gate} \cdot \mathbf{x})$$

이제 각 태스크가 다른 expert 조합을 가질 수 있어, 의견이 다른 두
태스크가 서로를 우회할 수 있다. 실무적으로 Shared-Bottom 보다
낫다. 새로운 실패 모드는 **expert collapse** — 손실 함수에서
서로 다른 태스크의 gate 가 실제로 발산하도록 강제하는 것이 없으며,
일반적인 학습률에서는 종종 같은 expert 로 수렴해버린다. 그렇게 되면
파라미터만 더 든 Shared-Bottom 을 다시 만든 셈이다.

## 여기서 우리가 마주친 문제

본 프로젝트의 13개 태스크 설정에서 두 단계 모두 그대로는 쓸 수
없다. Shared-Bottom 은 태스크 다양성에 무너진다 — churn, 순위,
회귀 타깃이 공유 trunk 을 서로 다른 방향으로 잡아당긴다. MMoE 는
이론상 태스크들이 서로를 우회할 수 있지만, 대칭적인 expert pool
과 제약 없는 gate 의 조합에서는 expert collapse 가 예외가 아니라
기본값이 된다. 실제로 버티는 해법은 구조적이다 — 동일한 expert
들에게 gate 압력만으로 스스로 분업을 학습시키려 하지 말고,
*분리 자체를 아키텍처에 내장*하는 것 — cross-task 신호용 shared
expert, 한 태스크만 신경 쓰는 패턴용 task-specific expert. 이것이
PLE 이며, 여기서 **PLE-2** 가 이어받는다 — Progressive Layered
Extraction 아키텍처 (Tang et al., RecSys 2020), CGC 게이트의 두
변형, 그리고 이종 expert 위에서 안정적으로 학습되도록 하기 위해
추가해야 했던 두 가지 정규화까지.
