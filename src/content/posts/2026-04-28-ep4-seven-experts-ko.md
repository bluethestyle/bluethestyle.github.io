---
title: "[3개월 개발기] 에피소드 4 — 일곱 전문가: 11개 학문에서 구조적 동형사상을 수입하다"
date: 2026-04-28 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, architecture, ple, expert-pool, financial-ai]
lang: ko
excerpt: "왜 7명인가, 왜 이 7명인가. Gemini 와의 학제간 스캔에서 11개 분야를 훑고, Opus 와의 기술 검증에서 선 안으로 들어온 DeepFM·Temporal·HGCN·PersLay·Causal·LightGCN·OT 의 도출 과정."
series: three-months
part: 4
alt_lang: /2026/04/28/ep4-seven-experts-en/
next_title: "에피소드 5 — 데이터 무결성 사냥"
next_desc: "label leakage 3연쇄 탐지, 18→13 태스크 축소, 합성데이터 v2→v3→v4 iteration — 모델 아키텍처보다 먼저 해결해야 했던 것."
next_status: published
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/expert_details.pdf
source_label: "전문가 상세 (KO, PDF)"
---

*"3개월 개발기" 4편. Ep 1 에서 "이종 전문가 네트워크" 라는 결정이
PLE 리프레이밍에서 나왔다고 썼다. Ep 2 에서 "구조적 동형사상" 이라는
개념이 Gemini 와의 대화에서 부상했다고 썼다. 이번 편은 그 추상어들이
구체적으로 일곱 네트워크로 귀착된 과정.*

## 왜 7명인가

전문가 수 자체가 먼저 결정된 건 아니다. "충분히 이질적인 수학적
관점" 이 몇 개 필요한가에서 시작했고, 결과로 7이 나왔다.

초기 후보는 열 개 넘게 있었다. Gemini 와의 학제간 스캔으로 11개
분야가 올라왔다 — 하이퍼볼릭 기하, 화학 반응속도론, SIR 전염병 모델,
최적 수송, 지속 호몰로지, 구조적 인과 추론, 그래프 이론, 상태 공간
시계열, 점곱/attention, 팩터라이제이션 기계, Gaussian mixture.

그 중 7개가 VRAM 예산과 태스크 공통 기반 위에서 *구조적으로 서로
다른 것을 본다* 는 조건을 통과했다. 나머지 4개는 기각.

**기각된 것들:**
- Mamba (Selective State Space) 단독 — 17개월 시퀀스에 메모리 과다.
  Temporal Ensemble 안에 흡수 형태로만 남음.
- 대형 Transformer expert — 12GB VRAM 에 7개 쌓을 수 없음.
  파라미터 수로 밀어붙이는 방향 자체 배제.
- Gaussian Mixture Model expert — 기존 expert 들과 구조적 중복
  (Causal + OT 가 이미 분포 비교 관점을 커버).
- 단순 MLP ensemble — "이종" 이 아니라 초기화만 다름. 붕괴 위험.

남은 7개가 어떤 수학적 빈틈을 메우는지 순서대로.

## 7개 전문가가 보는 것

**1. DeepFM — 피처 상호작용.** 가장 평범한 자리. 2-way / higher-order
feature interaction 을 factorization machine + deep network 로
잡는다. 여기가 *baseline* 이다. 다른 전문가들의 새로움을 측정하는
reference 이고, DeepFM 이 이길 수 있는 태스크 (단순 상호작용이
지배적인 것) 에서는 DeepFM 이 이긴다. 이 자리를 비우면 모든 것이
"novel" 이 되어 비교 대상이 사라진다.

**2. Temporal Ensemble (Mamba + LNN + PatchTST) — 시계열 동역학.**
17개월 고객 행동 시퀀스를 받는다. 단일 모델이 아니라 세 시계열
아키텍처의 앙상블 — Mamba 는 장기 의존성, LNN (Liquid Neural Network)
은 비선형 적응, PatchTST 는 주기성 포착. 셋이 서로 다른 시간 구조를
보고, HMM Triple-Mode 라우팅이 레짐별로 가중치를 분배한다.

**3. HGCN — 계층 구조 (하이퍼볼릭 공간).** MCC (Merchant Category
Code) 는 카테고리 트리다 — 식음료 > 레스토랑 > 한식당 같은. 유클리드
공간에서 트리를 임베딩하면 거리 왜곡이 크다. Poincaré ball 모델의
하이퍼볼릭 공간에서는 트리 임베딩이 기하학적으로 자연스럽다. 고객의
소비 카테고리 계층이 이 공간에서 해석된다.

**4. PersLay / TDA — 위상적 형태.** 지속 호몰로지 (persistent
homology) 로 고객의 소비 시점·금액 분포의 *형태* 를 수치화한다.
Vietoris-Rips complex 를 생성하고 persistence diagram 을 5-block
multi-beta 아키텍처로 임베딩. "고객의 지출 패턴이 매달 유사한 형태를
가지는가, 아니면 불규칙한 bursting 이 있는가" 같은 질문에 답한다.

**5. Causal — 구조적 인과 추론.** NOTEARS 기반 DAG 학습. 피처 간
인과 관계를 데이터에서 자동 발견. 다른 전문가들이 "상관관계 조합"
이라면 Causal 은 "무엇을 개입하면 무엇이 변하는가" 를 답한다. Paper
2 의 Counterfactual C-C (반사실 Champion-Challenger) 를 뒷받침하는
유일한 전문가.

**6. LightGCN — 사용자-상품 이분 그래프.** 기존 ALS 추천기가 하던
collaborative filtering 을 그래프 컨볼루션으로 대체. 이 자리가 없으면
"이전 시스템 성능을 따라잡는" 보증이 약해진다. 즉 LightGCN 은
*regression to previous baseline* 안전장치 역할도 겸한다.

**7. Optimal Transport — 분포 비교.** Sinkhorn divergence 로 두
고객 (또는 고객 군) 의 확률분포를 비교. Causal 이 인과 그래프를
다룬다면 OT 는 분포 자체를 metric 으로 만든다. 세그먼트 변화 감지,
드리프트 측정, 공정성 지표 계산에서 독립적 신호 제공.

## "왜 이 순서로 선택됐는가" 는 다르다

위는 최종 리스트지만, 시간 순서대로 들어온 게 아니다.

DeepFM 과 Temporal 은 ALS 대체 요구에서 *필연적으로* 들어왔다.
LightGCN 은 baseline 보장 차원에서 그 뒤에. 여기까지는 표준 추천
시스템 문헌에서 나오는 3점 세트다.

HGCN·PersLay·Causal·OT 가 들어오면서 이종 expert pool 의 성격이
결정됐다. Gemini 와의 대화가 "화학 반응속도론은 고객 행동의 어떤
측면과 구조적으로 동형인가?" 같은 질문을 던진 결과, 계층·위상·인과·
분포 네 관점이 *독립적으로* 의미 있다는 합의가 생겼다.

Opus 와의 기술 검증에서 각 후보의 실현 가능성이 체크됐다. HGCN 은
MCC 계층의 실제 트리 구조에서 작동할까? PersLay 는 17개월 시퀀스를
어떤 필터로 지속 다이어그램으로 만들까? Causal 은 349차원 공간에서
NOTEARS 가 수렴할까? 각 질문에 답이 YES 일 때만 남았다.

각 feasibility 체크는 단순 대화가 아니라 *Claude Code 세션* 이었다.
Opus 가 이론적 적합성을 논하면, Claude Code 가 그 오후에 최소 프로
토타입을 작성해 합성 데이터로 돌렸다. HGCN 의 feasibility 는 2시간
세션에서 27D MCC 계층 슬라이스 위에 Poincaré ball 임베딩을 구현해
loss 곡선이 수렴하는 걸 확인하며 결정됐다. PersLay 는 filtration
함수를 3일간 반복 수정한 끝에 persistence diagram 이 MLP 에 먹일
만큼 안정됐다. 단독 Mamba 는 17개월 시퀀스에서 *메모리 이유* 로
기각됐다 — Claude Code 터미널의 실제 OOM 에러가 결론을 내려주었고,
Temporal Ensemble 안에 흡수되는 경로로 변경됐다.

이 반복 패턴 — *Opus 와 가설, Claude Code 로 프로토타입, 숫자로
결정* — 덕분에 3인 팀이 7개 전문가 아키텍처를 약 6주 만에 검증할 수
있었다. 각 프로토타입은 대개 300줄 미만의 일회용 코드였지만,
"아이디어 → 테스트 → 판정" 의 처리량이 11→7 축소를 이 팀 규모에서
가능하게 한 요소였다.

## 7개를 다 넣으면 오버킬 아닌가

이게 Ablation 의 핵심 질문이었다. v12 iteration 까지 반복한 ablation
23 시나리오에서, 각 전문가를 하나씩 뺀 구성을 전수 비교했다.

결과가 흥미로웠다. *어느 전문가를 빼도 AUC 가 의미 있게 떨어졌다*.
특히 HGCN 을 뺐을 때는 MCC 계층이 잘 녹아있는 태스크 (spending\_
category, merchant\_affinity) 에서 크게 하락. PersLay 를 뺐을 때는
소비 패턴 bursting 을 감지하는 태스크 (consumption\_cycle) 에서
크게 하락. OT 를 뺐을 때는 세그먼트 기반 태스크에서 하락.

즉 7개가 *잉여* 가 아니라 *상보* 였다. 한 가지 수학적 관점으로는
13 태스크의 이질성을 다 감당하지 못한다는 게 ablation 결과의 해석.
즉 "이종 expert 는 논문 속 아이디어" 에서 *실제로 작동하는 구조*
로 전환된 셈이다.

## 왜 이 구조가 한국 금융권 조건에 맞는가

7개 전문가는 각자 경량 (20k~200k 파라미터). 합해도 2M 미만. RTX
4070 의 12GB VRAM 에 앙상블 7개가 올라가는 이유가 여기다. 만약
Transformer expert 7개를 쌓았다면 단 2개도 못 올렸을 것이다.

*경량 + 구조적 이종* 이 한국 금융권 중소 규모 팀의 접근 가능 조건이다.
대형 GPU 클러스터 없이도 "도메인 지식이 아키텍처에 박힌" 모델을
만들 수 있다는 증명. 해외 문헌의 "large-scale MoE" 패러다임을 그대로
이식하는 대신, 제약을 기회로 바꾼 설계다.

## 다음 편

Ep 5 는 이 아키텍처가 돌기 *전에* 해결해야 했던 문제를 다룬다 —
데이터 무결성. label leakage 3건 연쇄 탐지, 18→13 태스크 축소의
배경 (결정론적 리키지), 합성데이터 v2→v3→v4 iteration. 아키텍처
선택 전에 *입력* 이 올바른지 확인하는 과정.

원문 자료:
[전문가 상세 (KO, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/expert_details.pdf)
+ [개발 스토리 §5 "설계 철학"](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf).
