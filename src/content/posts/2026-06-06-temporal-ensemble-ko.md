---
title: "[Study Thread] TEMPORAL-1 — 한 고객을 위한 세 개의 시계: Mamba, Liquid Network, Patch Transformer 의 앙상블"
date: 2026-06-06 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, temporal, mamba, ssm, transformer, expert]
lang: ko
excerpt: "Temporal Expert 서브스레드 시작 — 고객의 소비는 스냅샷이 아니라 시퀀스라는 것, 그리고 프로젝트가 그것을 하나가 아니라 세 개의 시퀀스 모델로 읽는 이유. O(n²) 어텐션 vs 선형 시간 SSM 트레이드오프, Mamba 의 selective-scan 점화식, Liquid Neural Network 의 입력 의존적 시간 상수, PatchTST 의 패치 단위 어텐션, 그리고 셋을 하나의 64D 벡터로 융합해 PLE 에 넘기는 softmax 게이트."
series: study-thread
part: 16
alt_lang: /2026/06/06/temporal-ensemble-en/
next_title: "ECON-1 — 경제 피처 Expert: 고객 소비 아래 흐르는 거시 조류 읽기"
next_desc: "다음 서브스레드는 시간에서 맥락으로 옮겨간다 — 금리, 물가, 섹터 지수 같은 거시경제와 시장 신호를 어떻게 고객별 추천에 주입하는가, 그리고 왜 개인 시계열이 제대로 읽히려면 외생적 경제 프레임이 필요한가."
next_status: draft
---

*"Study Thread" 시리즈의 Temporal Expert 서브스레드 1편. 이번 편부터
영문/국문 병렬로 본 프로젝트의 7개 이종 Shared Expert 중 하나인 Temporal
Ensemble Expert 를 정리한다. 출처는 온프렘 프로젝트
`기술참조서/Temporal_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막
편에 첨부한다. TDA 서브스레드가 행동의* 형태 *를 물었다면, 이번 편은
행동의* 리듬 *을 묻는다 — 단일 스냅샷 피처가 버리는 순서, 간격, 추세.
프로젝트의 답은 독특하다. 시퀀스 모델 하나를 고르지 않는다. 셋을 동시에
돌린다 — selective state-space 모델, liquid ODE 네트워크, patch
Transformer — 그리고 누구를 믿을지는 게이트가 정한다.*

> **하나의 Expert, 의도된 세 서브모델.** 대부분의 팀은 시퀀스
> 아키텍처 하나를 골라 밀어붙인다. 이 Expert 는 거부한다. Mamba 는 장기
> 순차 의존성을 선형 시간에 처리하고, Liquid Neural Network 는 거래 간의
> 불규칙한 간격을 입력 의존적 시간 상수로 보정하며, PatchTST 는 패치 단위
> 어텐션으로 글로벌 주기성을 매칭한다. 학습 가능한 softmax 게이트가 세
> 출력을 모두 관찰한 뒤 입력마다 가중치를 배분한다 — 주기성이 강한
> 고객에겐 PatchTST 가중치가, 불규칙한 고객에겐 LNN 가중치가 높아진다.
> 설계의 베팅은 단 하나의 시계로 모든 고객을 읽을 수 없다는 것이다.
> 그래서 Expert 는 시계 셋을 들고 다이얼을 읽는다.

## 시간은 고객의 네 번째 차원

고전적 추천 시스템은 사용자를 *정적* 피처로 묘사한다 — 나이, 직업, 선호
카테고리. 언제 보든 고정된 속성이다. 그러나 실제 행동은 *시간 축* 을 따라
끊임없이 움직인다.

"매주 금요일에 외식 결제를 하고, 매달 1일에 공과금을 납부하며, 최근
3개월간 커피 지출이 점진적으로 증가하는" 고객을 보자. 이 정보를 단일
스냅샷 — *현재 월 평균 지출* — 으로 축약하면 세 가지 핵심 신호가 한꺼번에
사라진다 — *주기성*, *트렌드*, *계절성*.

| 관점 | 정적 피처 | 시간적 피처 |
| --- | --- | --- |
| 표현 형태 | 고정 벡터 $\mathbf{x}\in\mathbb{R}^d$ | 시퀀스 $\mathbf{X}\in\mathbb{R}^{T\times d}$ |
| 정보 손실 | 시간 축 평균화 → 패턴 소멸 | 순서·간격·추세 보존 |
| 필요 모델 | MLP, 임베딩 테이블 | RNN, SSM, Transformer |
| 예시 | 월 평균: 150만원 | 일별 시계열: [12, 0, 5, 0, 0, 85, 15, …] |

Temporal Expert 는 바로 이 문제를 푼다. 거래와 세션 데이터를 *시퀀스* 로
유지한 채, 시간 차원에 묻힌 패턴을 학습하여 64D 표현으로 압축한다.

## 트레이드오프: O(n²) 어텐션 vs 선형 시간 점화식

두 거대한 시퀀스 모델 계열이 있고, 둘은 시간을 가로지르는 방법에서
갈라진다.

**Transformer** 는 self-attention 을 계산한다 — 모든 위치가 다른 모든
위치를 직접 참조한다.

$$ \mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V} $$

*글로벌 패턴 매칭* 에 탁월하다 — 두 타임스텝 사이의 거리가 무관하므로,
1월의 습관과 12월의 습관이 행렬에서 한 칸 차이다. 대가는
$\mathbf{Q}\mathbf{K}^\top$ 행렬이 $L\times L$ 이라는 것 — 시간과 메모리
모두 **$O(L^2)$**. 긴 거래 이력에서 바로 이 지점이 아프다.

반대로 **상태 공간 모델(SSM)** 은 RNN 처럼 은닉 상태를 통해 정보를
*순차적* 으로 전파한다 — **$O(L)$**, 시퀀스 길이에 선형이다. 대가는
Transformer 의 거울상이다. 먼 과거의 정보가 상태를 통과하며 *감쇠* 하여,
아주 먼 거리의 매칭이 어려워진다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">어텐션 — 모든 쌍, O(n²)</text>
  <g>
    <circle cx="70"  cy="80" r="7" fill="#4f46e5"/><circle cx="130" cy="80" r="7" fill="#4f46e5"/>
    <circle cx="190" cy="80" r="7" fill="#4f46e5"/><circle cx="250" cy="80" r="7" fill="#4f46e5"/>
  </g>
  <g stroke="#4f46e5" stroke-width="0.8" opacity="0.5" fill="none">
    <path d="M70 80 Q100 150 130 80"/><path d="M70 80 Q130 175 190 80"/><path d="M70 80 Q160 200 250 80"/>
    <path d="M130 80 Q160 150 190 80"/><path d="M130 80 Q190 175 250 80"/>
    <path d="M190 80 Q220 150 250 80"/>
  </g>
  <text x="150" y="225" text-anchor="middle" font-size="11" fill="#64748b">모든 쌍이 직접 연결</text>
  <text x="150" y="242" text-anchor="middle" font-size="10" fill="#e11d48" font-weight="700">비용이 L² 로 증가</text>
  <line x1="300" y1="45" x2="300" y2="215" stroke="#e2e8f0" stroke-width="1"/>
  <text x="450" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">SSM — 점화식 사슬, O(n)</text>
  <g>
    <circle cx="360" cy="120" r="7" fill="#0d9488"/><circle cx="420" cy="120" r="7" fill="#0d9488"/>
    <circle cx="480" cy="120" r="7" fill="#0d9488"/><circle cx="540" cy="120" r="7" fill="#0d9488"/>
  </g>
  <g stroke="#0d9488" stroke-width="1.6" fill="#0d9488">
    <line x1="367" y1="120" x2="413" y2="120"/><polygon points="413,120 405,116 405,124"/>
    <line x1="427" y1="120" x2="473" y2="120"/><polygon points="473,120 465,116 465,124"/>
    <line x1="487" y1="120" x2="533" y2="120"/><polygon points="533,120 525,116 525,124"/>
  </g>
  <g font-size="10" fill="#64748b" text-anchor="middle">
    <text x="360" y="145">h₁</text><text x="420" y="145">h₂</text><text x="480" y="145">h₃</text><text x="540" y="145">h₄</text>
  </g>
  <text x="450" y="225" text-anchor="middle" font-size="11" fill="#64748b">한 스텝씩 상태를 앞으로 전달</text>
  <text x="450" y="242" text-anchor="middle" font-size="10" fill="#0d9488" font-weight="700">비용이 L 로 증가</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">시간을 가로지르는 두 방식. 어텐션은 모든 쌍을 직접 잇고(O(n²), 장거리 강점, 비쌈), SSM 은 하나의 은닉 상태를 시퀀스에 꿰어 넣는다(O(n), 싸지만 과거가 감쇠).</figcaption>
</figure>

| 세대 | 모델 | 약점 | 후계자 |
| --- | --- | --- | --- |
| 2세대 | LSTM / GRU (게이트 RNN) | $O(L)$ 순차 병목, vanishing gradient | Transformer |
| 3세대 | Transformer (self-attention) | $O(L^2)$ 비용, 순서 정보 약함 | SSM, PatchTST |
| 4세대 | SSM + ODE + Patch Transformer **앙상블** | 모델 복잡도 증가, 게이트 붕괴 위험 | **본 프로젝트의 Temporal Expert** |

> **역사적 배경.** Transformer(Vaswani et al., 2017)는 NLP 에 혁명을
> 일으킨 뒤 시계열에 적응됐다. 그러나 Zeng et al.(AAAI 2023)이 한 방
> 먹였다 — 단순 선형 모델이 정교한 시계열 Transformer 를 이겼다 — 이후
> 분야가 균형을 되찾았다. S4(Gu et al., ICLR 2022)가 상태 공간 모델을 딥 시퀀스
> 모델링에 들여왔고, Mamba(Gu & Dao, 2023)가 *선택적* 메커니즘으로
> 선형 시불변(LTI)의 한계를 돌파했으며, PatchTST(Nie et al., 2023)가
> 패치 단위로 어텐션을 효율화했다. 이 Expert 는 승자를 가리지 않는다.
> 생존자들을 앙상블한다.

## Mamba — 무엇을 기억할지 고르는 상태 공간 모델

모든 SSM 의 바탕인 연속 선형 시불변(LTI) 시스템에서 출발하자.

$$ \frac{d\mathbf{x}}{dt} = \mathbf{A}\mathbf{x} + \mathbf{B}u,\qquad y = \mathbf{C}\mathbf{x} + \mathbf{D}u $$

여기서 $\mathbf{x}\in\mathbb{R}^N$ 은 은닉 상태, $u$ 는 입력 신호,
$\mathbf{A}$ 는 상태 전이 행렬, $\mathbf{B}$ / $\mathbf{C}$ 는 입력 / 출력
행렬이다. 이산 거래 시퀀스에 돌리려면 스텝 $\Delta$ 로 zero-order hold
*이산화* 한다.

$$ \bar{\mathbf{A}} = \exp(\Delta\,\mathbf{A}),\qquad \bar{\mathbf{B}} = (\Delta\,\mathbf{A})^{-1}\big(\bar{\mathbf{A}} - \mathbf{I}\big)\cdot\Delta\,\mathbf{B} $$

이것이 미분방정식을 이산 점화식으로 바꾼다.

$$ \mathbf{h}_t = \bar{\mathbf{A}}\,\mathbf{h}_{t-1} + \bar{\mathbf{B}}\,\mathbf{x}_t,\qquad \mathbf{y}_t = \mathbf{C}_t\,\mathbf{h}_t $$

> **수식 직관.** 이 점화식은 RNN 의 일반화다. $\bar{\mathbf{A}}$ 가
> *이전 기억을 얼마나 유지할지*, $\bar{\mathbf{B}}$ 가 *새 입력을 얼마나
> 수용할지* 를 결정하고, $\mathbf{C}_t$ 는 상태에서 필요한 정보만 뽑아내는
> 읽기 헤드다. 이산화 스텝 $\Delta$ 가 손잡이다 — $\Delta$ 가 크면 입력을
> 오래 기억하고(느린 동역학), 작으면 빠르게 잊는다(빠른 동역학). 고전
> SSM 에서는 이 행렬들이 모든 시점에서 동일한 상수라, 전체가 병렬화
> 가능한 컨볼루션으로 붕괴한다.

그 불변성의 문제 — LTI 시스템은 *모든* 입력을 같은 규칙으로 처리한다.
5천원 커피와 5백만원 이체가 같은 기억 처리를 받는다. Mamba 의 **S6 선택적
메커니즘** 은 $\Delta$, $\mathbf{B}$, $\mathbf{C}$ 를 *입력 의존적* 으로
만들어 이를 고친다.

$$ \Delta = \mathrm{softplus}(\mathbf{W}_\Delta\mathbf{x} + \mathbf{b}_\Delta),\quad \mathbf{B} = \mathbf{W}_B\mathbf{x},\quad \mathbf{C} = \mathbf{W}_C\mathbf{x} $$

softplus 가 $\Delta > 0$ 을 보장한다(시간 스텝은 양수). 이제 *전이 규칙
자체가 콘텐츠에 의존* 한다 — 대형 거래는 $\Delta$ 를 키워 강하게 기억되고,
일상의 소액 거래는 $\Delta$ 를 낮춰 배경으로 처리된다. 대가는 모델이 더
이상 시불변이 아니어서 컨볼루션 지름길이 사라진다는 것 — Mamba 는
hardware-aware *selective scan* 으로 효율을 되찾는다. 본 프로젝트의 온라인
인스턴스에서 Mamba 는 180-스텝 거래 시퀀스에 대해 `d_model=128`,
`d_state=16` 으로 돈다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="210" fill="#f8fafc" rx="8"/>
  <text x="300" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">선택적 점화식, 펼침 — hₜ = Ā hₜ₋₁ + B̄ xₜ</text>
  <g font-size="11" fill="#64748b" text-anchor="middle">
    <text x="110" y="70">x₁</text><text x="240" y="70">x₂</text><text x="370" y="70">x₃</text><text x="500" y="70">x₄</text>
  </g>
  <g stroke="#64748b" stroke-width="1.2" fill="#64748b">
    <line x1="110" y1="78" x2="110" y2="108"/><polygon points="110,108 106,100 114,100"/>
    <line x1="240" y1="78" x2="240" y2="108"/><polygon points="240,108 236,100 244,100"/>
    <line x1="370" y1="78" x2="370" y2="108"/><polygon points="370,108 366,100 374,100"/>
    <line x1="500" y1="78" x2="500" y2="108"/><polygon points="500,108 496,100 504,100"/>
  </g>
  <g>
    <rect x="80"  y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="210" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="340" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
    <rect x="470" y="110" width="60" height="40" rx="6" fill="#0d948822" stroke="#0d9488" stroke-width="1.2"/>
  </g>
  <g font-size="12" fill="#0d9488" font-weight="700" text-anchor="middle">
    <text x="110" y="135">h₁</text><text x="240" y="135">h₂</text><text x="370" y="135">h₃</text><text x="500" y="135">h₄</text>
  </g>
  <g stroke="#1e3a5f" stroke-width="1.6" fill="#1e3a5f">
    <line x1="140" y1="130" x2="208" y2="130"/><polygon points="208,130 200,126 200,134"/>
    <line x1="270" y1="130" x2="338" y2="130"/><polygon points="338,130 330,126 330,134"/>
    <line x1="400" y1="130" x2="468" y2="130"/><polygon points="468,130 460,126 460,134"/>
  </g>
  <g font-size="10" fill="#1e3a5f" text-anchor="middle">
    <text x="174" y="122">Ā</text><text x="304" y="122">Ā</text><text x="434" y="122">Ā</text>
  </g>
  <text x="240" y="180" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700">대형 거래 → Δ↑ → 강하게 기억</text>
  <text x="370" y="196" text-anchor="middle" font-size="10" fill="#64748b">소액 거래 → Δ↓ → 배경으로 유지</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">상태 공간 점화식을 펼친 모습. 각 스텝이 Ā 로 hₜ 를 앞으로 나르고 B̄ 로 새 입력을 수용한다. Mamba 가 Δ, B, C 를 입력 의존적으로 만들기에, 기억/망각 규칙이 거래마다 달라진다.</figcaption>
</figure>

## LNN — 불규칙 간격을 위한 Liquid 시간 상수

거래는 시계에 맞춰 도착하지 않는다. 두 구매 사이의 간격이 10분일 수도
10일일 수도 있고, 그 *간격 자체* 가 신호다. **Liquid Neural Network** 는
시간 상수가 입력에 적응하는 연속 시간(Neural ODE) 모델이다.

$$ \frac{d\mathbf{h}}{dt} = \frac{-\mathbf{h} + f(\mathbf{x},\mathbf{h})}{\tau(\mathbf{x},\mathbf{h})} $$

직관은 "현재 상태에서 목표 상태 $f(\mathbf{x},\mathbf{h})$ 로, $\tau$ 의
속도로 이동한다" 이다. $\tau$ 가 크면 변화가 느리고(장기 휴면 고객의
상태가 서서히 감쇠), 작으면 반응이 빠르다(활발한 소비 burst). 핵심은
LNN 이 *$\tau$ 를 입력에서 생성* 한다는 것 — 같은 고객도 시기에 따라 다른
속도로 이완한다. 이산 스텝에 돌리기 위해 프로젝트는 1차 Euler 업데이트
한 번 — `LNNSingleStep` — 을 쓴다.

$$ \mathbf{h}_{t+1} = \mathbf{h}_t + \Delta t\cdot\frac{-\mathbf{h}_t + f(\mathbf{x}_t,\mathbf{h}_t)}{\tau(\mathbf{x}_t,\mathbf{h}_t)} $$

여기서 $\Delta t$ 는 *실제* 이벤트 간 간격(일 단위)으로, `[0.001, 30.0]`
— 약 1.4분 ~ 30일 — 로 clamp 된다. 의미 유지를 위해서이기도 하고,
$\Delta t > \tau$ 일 때 Euler 가 진동할 수 있기 때문이기도 하다. 설계
의도가 여기서 중요하다 — LNN 은 Mamba *뒤에 직렬* 로 돈다(Mamba 의 최종
상태에 시간 인식 보정을 입힌다). 전체 병렬 시퀀스 ODE 로 돌리지 않는
이유는, 그러면 Mamba 의 작업이 중복되고 비용만 커지기 때문이다.

## PatchTST — 어텐션, 단 패치 단위로

세 번째 서브모델은 베팅의 Transformer 쪽이다. 모든 타임스텝에 평범한
self-attention 을 쓰면 앞서 본 $O(L^2)$ 함정에 빠진다. **PatchTST** 는
시퀀스를 *패치* 로 잘라(프로젝트는 `patch_size=16` 사용) 점 대 점이 아닌
*패치 대 패치* 로 어텐션을 걸어 이를 우회한다. 복잡도가
$O((L/P)^2)$ 로 떨어지고 — 더 중요하게는 — 패치는 작은 로컬 윈도우라,
개별 일자보다 *글로벌 주기성*(주간, 월간 사이클)을 포착하는 데 더 나은
단위다. PatchTST 는 설계상 *원본* 시퀀스를 독립적으로 입력받는다 —
LNN 처럼 Mamba 의 처리된 상태를 먹이면 게이트의 모델 차별화 능력이
줄어들기에, 프로젝트는 입력을 분리하여 앙상블 다양성을 지킨다.

## 게이트 — 세 출력, 하나의 투표

이제 세 서브모델이 각자 표현을 냈고, 앙상블이 그것들을 결합해야 한다.
학습 가능한 게이트가 셋을 모두 관찰하여 볼록 가중치를 내보낸다.

$$ \mathbf{g} = \mathrm{Softmax}\!\big(\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{z}_{\mathrm{cat}} + \mathbf{b}_1) + \mathbf{b}_2\big),\qquad \mathbf{y} = \sum_{i=1}^{3} g_i\cdot\mathrm{Proj}_i(\mathbf{z}_i) $$

여기서 $\mathbf{z}_{\mathrm{cat}}\in\mathbb{R}^{384}$ 는 세 모델 출력의
concat($192 + 96 + 96 = 384$)이고, 2층 MLP 가 $384\to6\to3$ 으로 매핑하며,
softmax 가 $g_1 + g_2 + g_3 = 1$ 을 강제한다. 각 모델 출력은 먼저 공통
64D 공간으로 프로젝션되어($\mathrm{Proj}_i$) 가중합이 의미를 갖는다.
기하학적으로 $\mathbf{y}$ 는 세 프로젝션 출력을 꼭짓점으로 하는 삼각형
내부의 한 점이며, $\mathbf{g}$ 는 그 무게중심 좌표다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="270" fill="#f8fafc" rx="8"/>
  <rect x="30"  y="40" width="120" height="44" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1.2"/>
  <text x="90" y="60" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">Mamba</text>
  <text x="90" y="76" text-anchor="middle" font-size="10" fill="#64748b">192D</text>
  <rect x="30"  y="110" width="120" height="44" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="90" y="130" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">LNN</text>
  <text x="90" y="146" text-anchor="middle" font-size="10" fill="#64748b">96D</text>
  <rect x="30"  y="180" width="120" height="44" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1.2"/>
  <text x="90" y="200" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">PatchTST</text>
  <text x="90" y="216" text-anchor="middle" font-size="10" fill="#64748b">96D</text>
  <rect x="210" y="100" width="90" height="64" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>
  <text x="255" y="128" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">Concat</text>
  <text x="255" y="144" text-anchor="middle" font-size="10" fill="#64748b">384D</text>
  <rect x="340" y="100" width="100" height="64" rx="6" fill="#e11d4818" stroke="#e11d48" stroke-width="1.2"/>
  <text x="390" y="124" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Gate MLP</text>
  <text x="390" y="139" text-anchor="middle" font-size="9.5" fill="#64748b">384→6→3</text>
  <text x="390" y="153" text-anchor="middle" font-size="9.5" fill="#64748b">+ Softmax</text>
  <rect x="478" y="108" width="60" height="48" rx="6" fill="#1e3a5f"/>
  <text x="508" y="130" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">y</text>
  <text x="508" y="145" text-anchor="middle" font-size="9.5" fill="#fff">64D</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="150" y1="62"  x2="208" y2="120"/><polygon points="208,120 199,118 203,111"/>
    <line x1="150" y1="132" x2="208" y2="132"/><polygon points="208,132 200,128 200,136"/>
    <line x1="150" y1="202" x2="208" y2="144"/><polygon points="208,144 203,153 199,146"/>
  </g>
  <g fill="#94a3b8" stroke="#94a3b8" stroke-width="1.6">
    <line x1="300" y1="132" x2="338" y2="132"/><polygon points="338,132 330,128 330,136"/>
    <line x1="440" y1="132" x2="476" y2="132"/><polygon points="476,132 468,128 468,136"/>
  </g>
  <text x="458" y="124" text-anchor="middle" font-size="9" fill="#e11d48" font-weight="700">g₁,g₂,g₃</text>
  <text x="280" y="245" text-anchor="middle" font-size="11" fill="#64748b">y = Σ gᵢ · Projᵢ(zᵢ),  g₁+g₂+g₃ = 1  (볼록 투표)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">앙상블 게이팅. 세 출력을 384D 로 concat 하고, 작은 MLP + softmax 가 3방향 가중치를 낸다. 각 모델은 64D 로 프로젝션되어 볼록 결합된다. 결과는 하나의 64D 벡터 — 게이트 가중치는 입력별 신뢰도 투표다.</figcaption>
</figure>

> **왜 경량 MoE 인가.** 이것은 Mixture-of-Experts 아이디어(Jacobs et al.,
> 1991)의 가장 작은 유용한 형태다 — *모든* 전문가가 항상 활성화되고
> 가중치만 움직이는 Dense MoE, Soft MoE(Google, 2023)와 같다. 서브모델이
> 셋뿐이라 대규모 sparse MoE 가 필요로 하는 load-balancing loss 를 건너뛸
> 수 있고, 대신 *gate collapse* — 한 모델이 모든 가중치를 독점 — 를
> 게이트의 Shannon 엔트로피 모니터링으로 관리한다. 엔트로피가 ~0.3 아래로
> 지속되면 앙상블이 단일 모델로 퇴화했다는 경고이며, MLflow 에
> `temporal_gate_entropy` 로 기록된다.

## Expert 가 PLE 에 꽂히는 자리

Expert 는 두 시퀀스를 받아 하나의 벡터를 낸다. 입력: `[B, 180, 16]`
형태의 `txn_seq`(16 = card 8D + deposit 8D)와 `[B, 90, 8]` 형태의
`session_seq`. 각 서브모델은 txn/session 전용 인스턴스를 별도로 유지하고
게이트 앞에서 concat 한다. 출력: 단일 **64D** 벡터가 PLE CGC gate 로
들어가 태스크별로 다른 Expert 와 혼합된다.

의도된 안전 밸브가 있다. 시퀀스가 없는 배치 — cold-start 고객 — 에서는
Expert 가 폭 64 의 **zero 벡터** 를 반환한다. 그러면 PLE 의 CGC gate 가
자동으로 가중치를 낮추고, 다른 Expert(DeepFM, LightGCN 등)가 보상한다.
`session_seq` 가 없는 경우에도 zero 텐서로 폴백하여 shape 호환성을
보장한다.

Expert 는 리듬이 신호를 담는 모든 그룹에 걸쳐 **12개** 태스크의
`domain_experts` 멤버로 연결된다.

| 그룹 | 태스크 | Temporal 의 기여 |
| --- | --- | --- |
| Engagement | ctr, cvr, engagement | 클릭 시점, 구매 여정, 세션 패턴 |
| Lifecycle | churn, retention, life_stage, ltv | 빈도 감소 추세, 장기 행동 궤적 |
| Value | balance_util, channel, timing | 잔액 추세, 채널 시간대, 28일(4×7) 주기성 |
| Consumption | consumption_cycle, merchant_affinity | 7종 소비 주기, 가맹점 방문 시계열 |

## 여기서 멈추는 이유

스냅샷에 대한 불편함 — 금요일 외식, 매달 1일 공과금, 느린 커피 증가를
지워버리는 월 평균 — 에서 출발했다. 핵심 트레이드오프(O(n²) 어텐션 vs
O(n) 점화식)를 따라간 뒤, 프로젝트가 들고 있는 세 시계를 만났다 —
장기 의존성을 위한 Mamba 의 selective scan, 불규칙 간격을 위한 Liquid
Network 의 입력 의존적 $\tau$, 글로벌 주기성을 위한 PatchTST 의 패치
어텐션. 마지막으로 softmax 게이트가 셋을 하나의 64D 벡터로 융합하는 것을,
그리고 zero 폴백과 엔트로피 경보로 앙상블을 정직하게 유지하며 PLE 를
통해 12개 태스크에 공급하는 것을 봤다.

하지 *않은* 것은 상자를 여는 일이다 — 정확한 selective-scan 구현, LNN
셀의 tau-net, PatchTST 인코더 내부, 그리고 학습 중 게이트 엔트로피가 실제
어떻게 계산되고 모니터링되는가. 이것이 다음 Temporal 편의 기계 장치다. 그러나
서브스레드는 이제 바깥으로 향한다. 고객의 시계열은 진공에서 움직이지
않는다 — 금리, 물가, 섹터 사이클의 경제 안에서 움직인다. 다음 서브스레드
**ECON-1** 은 경제 피처 Expert 를 다룬다 — 한 고객의 소비 *아래* 흐르는
거시 조류를 어떻게 읽는가.
