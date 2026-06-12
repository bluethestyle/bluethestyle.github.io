---
title: "[Study Thread] TDA-1 — 소비의 형태: 위상 데이터 분석과 PersLay 라는 다리"
date: 2026-06-05 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, perslay, persistent-homology, topology, expert]
lang: ko
excerpt: "TDA / PersLay 서브스레드 시작 — 요약 통계량이 놓치는 소비의 기하학적 구조, persistent homology 가 고객 행동의 '형태'를 읽는 방식, persistence diagram 이 실제로 담는 정보, 그리고 그 diagram 을 신경망으로 잇는 PersLay 라는 다리. Expert 채택을 정당화한 검증 결과와 함께."
series: study-thread
part: 11
alt_lang: /2026/06/05/tda-1-topology-of-spending-en/
next_title: "TDA-2 — 집합 함수로서의 PersLay: φ, w, ρ 와 5-Block 아키텍처"
next_desc: "가변 크기의 순서 없는 점 집합을 어떻게 고정 64D 벡터로 바꾸는가: RationalHat 점 변환, persistence 가중치, 순서 불변 집계, 그리고 Short/Long range × 호몰로지 차원을 다섯 개의 독립 블록으로 분리하는 이유."
next_status: published
---

*"Study Thread" 시리즈의 TDA(위상 데이터 분석) / PersLay 서브스레드 1편.
이번 편부터 영문/국문 병렬로 본 프로젝트의 7개 이종 Shared Expert 중
하나인 PersLay Expert 를 정리한다. 출처는 온프렘 프로젝트
`기술참조서/PersLay_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막
편에 첨부한다. PLE 와 adaTT 서브스레드가 태스크들이 어떻게 공유하고
전이하는가를 다뤘다면, 이번 서브스레드는 다른 질문을 던진다 — 애초에
Expert 는 무엇을 신호로 읽는가? PersLay 의 답은 독특하다. 평균과 분산이
버리는 부분, 소비의* 형태 *를 읽는다.*

> **실증적 상태 — 추측이 아니라 검증됨.** 모델 코드를 한 줄 쓰기 전에,
> 실제 세션 로그(90일, 고객 120명)로 단독 검증을 먼저 돌렸다. 질문은
> 직설적이었다 — 고객 행동에 실제로 위상적 구조가 있는가, 아니면
> 수학적으로 우아하지만 실증적으로는 비어 있는가? Persistence diagram
> 요약 피처는 행동 그룹을 실루엣 **0.299** 로 분리했고, 동일 조건의 raw
> 집계 피처는 **0.192** 였다 — **+0.108** 개선이고, TDA 점수가 사전
> 등록한 PASS 기준선 0.15 를 넘겼다. Expert 는 자리값을 했다. 숫자와 도출 과정은
> 아래 "소비에 형태가 있는가?" 절에 있다.

## 요약 통계량의 한계

고객을 묘사할 때 우리는 평균, 분산, 최대값, 중앙값을 꺼내 든다. 이들은
분포의 *중심 경향* 과 *산포* 를 잡아내지만, 그 *기하학적 배치* 에
대해서는 거의 아무것도 말하지 않는다.

두 고객 A, B 를 보자. A 는 식비, 교통, 문화, 쇼핑을 고르게 소비한다.
업종 벡터의 고차원 공간에서 점들이 하나의 연결된 덩어리를 이룬다.
B 는 식비와 교통만 소비하되 순환한다 — 월초엔 식비, 월말엔 교통 — 그래서
점들이 분리된 두 군집과 그 사이를 도는 주기적 경로로 갈라진다.

두 고객의 평균 소비가 같을 수 있다. 분산도 같을 수 있다. 그러나 행동의
*구조* 는 질적으로 다르다. 하나의 연결된 덩어리 vs 순환으로 이어진 두
엽(lobe). 이 차이가 바로 *위상적* 인 것이며, 어떤 요약 통계량으로도
볼 수 없다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <text x="140" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">고객 A — 하나의 군집</text>
  <circle cx="110" cy="110" r="58" fill="#0d948815" stroke="#0d9488" stroke-width="1" stroke-dasharray="4 3"/>
  <g fill="#0d9488">
    <circle cx="95" cy="90" r="4"/><circle cx="120" cy="85" r="4"/><circle cx="140" cy="105" r="4"/>
    <circle cx="105" cy="120" r="4"/><circle cx="130" cy="130" r="4"/><circle cx="90" cy="115" r="4"/>
    <circle cx="118" cy="112" r="4"/><circle cx="150" cy="125" r="4"/><circle cx="100" cy="100" r="4"/>
  </g>
  <text x="140" y="200" text-anchor="middle" font-size="11" fill="#64748b">β₀ = 1, β₁ = 0</text>
  <line x1="280" y1="40" x2="280" y2="195" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">고객 B — 두 엽 + 순환</text>
  <circle cx="360" cy="100" r="34" fill="#e11d4815" stroke="#e11d48" stroke-width="1" stroke-dasharray="4 3"/>
  <circle cx="470" cy="130" r="34" fill="#e11d4815" stroke="#e11d48" stroke-width="1" stroke-dasharray="4 3"/>
  <path d="M 360 100 C 390 60, 440 60, 470 130 C 490 175, 400 180, 360 100 Z" fill="none" stroke="#d97706" stroke-width="1.4" stroke-dasharray="5 4"/>
  <g fill="#e11d48">
    <circle cx="345" cy="90" r="4"/><circle cx="370" cy="95" r="4"/><circle cx="355" cy="115" r="4"/><circle cx="375" cy="112" r="4"/>
    <circle cx="458" cy="120" r="4"/><circle cx="482" cy="128" r="4"/><circle cx="465" cy="145" r="4"/><circle cx="478" cy="140" r="4"/>
  </g>
  <text x="420" y="200" text-anchor="middle" font-size="11" fill="#64748b">β₀ = 2, β₁ = 1</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">평균과 분산은 같지만 형태가 다르다. 위상수학은 덩어리(β₀)와 고리(β₁)를 세지만, 통계량은 둘 다 세지 못한다.</figcaption>
</figure>

## 위상수학이 측정하는 것

위상수학은 연속적 변형 — 늘이고, 구부리고, 비틀어도 — 에도 *구멍의
개수* 가 변하지 않는 성질을 연구한다(오래된 농담: 커피컵은 도넛이다).
점 구름에 적용하면, 노이즈에도 살아남는 세 가지 질문에 답한다.

| 불변량 | 세는 것 | 소비 데이터 해석 |
| --- | --- | --- |
| $H_0$ ($\beta_0$) | 연결 성분 | 분리된 소비 군집 — 예: "식비 중심" vs "여행 중심" 그룹 |
| $H_1$ ($\beta_1$) | 고리(1차원 구멍) | 순환적 소비 — 식비 → 교통 → 문화 → 다시 식비 |
| $H_2$ ($\beta_2$) | 공동(2차원 캐비티) | 3개 이상 업종이 함께 나타나지 않는 고차원 빈 영역 |

이 피처들의 가치는 미학이 아니다. *좌표 불변* 이고(데이터를 회전하거나 이동해도
변하지 않는다), *노이즈에 강건* 하며(작은 섭동이 구멍 수를 바꾸지
못한다), *다중 스케일* 이다(하나의 임계값이 아니라 모든 거리 임계값에서의
구조를 한꺼번에 추적한다).

> **역사적 배경.** 대수적 위상수학은 19세기 Poincaré 의 호몰로지 이론에서
> 출발해 20세기 중반 Eilenberg–Steenrod 공리계로 정립됐다. *계산* 위상수학
> 으로의 전환은 1990~2000년대(Edelsbrunner, Harer, Carlsson)에 일어났고,
> 2010년 이후 *Topological Data Analysis* 라는 이름으로 데이터 과학에
> 진입했다. 순수 수학 50년이 점 구름의 형태를 측정하는 실용 도구로
> 흘러내린 셈이다.

## Persistent Homology — 구멍의 생성과 소멸을 지켜보기

하나의 거리 임계값은 자의적이다. Persistent homology 는 하나를 고르길
거부한다. 대신 임계값 $\varepsilon$ 을 0 에서부터 키워가며 위상의 변화를
지켜본다 — 이를 *여과(filtration)* 라 부른다.

- $\varepsilon = 0$: 모든 점이 고립. 성분 $n$개, 고리 0개.
- $\varepsilon$ 작음: 가까운 점들이 edge 로 연결되고 성분이 합쳐진다.
- $\varepsilon$ 중간: edge 들이 닫혀 고리가 된다 — $H_1$ 특성이 *생성*.
- $\varepsilon$ 큼: 삼각형이 고리를 채운다 — 그 특성이 *소멸*.
- $\varepsilon \to \infty$: 모든 게 하나의 덩어리.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="170" fill="#f8fafc" rx="8"/>
  <text x="75" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε = 0</text>
  <g fill="#4f46e5"><circle cx="75" cy="55" r="4"/><circle cx="108" cy="80" r="4"/><circle cx="95" cy="118" r="4"/><circle cx="55" cy="118" r="4"/><circle cx="42" cy="80" r="4"/></g>
  <text x="75" y="150" text-anchor="middle" font-size="10" fill="#64748b">성분 5개</text>
  <text x="225" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε 작음</text>
  <g stroke="#94a3b8" stroke-width="1.4">
    <line x1="225" y1="55" x2="258" y2="80"/><line x1="258" y1="80" x2="245" y2="118"/>
    <line x1="245" y1="118" x2="205" y2="118"/><line x1="205" y1="118" x2="192" y2="80"/><line x1="192" y1="80" x2="225" y2="55"/>
  </g>
  <g fill="#4f46e5"><circle cx="225" cy="55" r="4"/><circle cx="258" cy="80" r="4"/><circle cx="245" cy="118" r="4"/><circle cx="205" cy="118" r="4"/><circle cx="192" cy="80" r="4"/></g>
  <text x="225" y="150" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700">고리 생성 (H₁)</text>
  <text x="375" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε 중간</text>
  <polygon points="375,55 408,80 395,118 355,118 342,80" fill="#0d948822" stroke="#94a3b8" stroke-width="1.4"/>
  <g fill="#4f46e5"><circle cx="375" cy="55" r="4"/><circle cx="408" cy="80" r="4"/><circle cx="395" cy="118" r="4"/><circle cx="355" cy="118" r="4"/><circle cx="342" cy="80" r="4"/></g>
  <text x="375" y="150" text-anchor="middle" font-size="10" fill="#64748b">채워지는 중…</text>
  <text x="525" y="24" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">ε 큼</text>
  <polygon points="525,55 558,80 545,118 505,118 492,80" fill="#0d948855" stroke="#0d9488" stroke-width="1.4"/>
  <g fill="#4f46e5"><circle cx="525" cy="55" r="4"/><circle cx="558" cy="80" r="4"/><circle cx="545" cy="118" r="4"/><circle cx="505" cy="118" r="4"/><circle cx="492" cy="80" r="4"/></g>
  <text x="525" y="150" text-anchor="middle" font-size="10" fill="#e11d48" font-weight="700">고리 소멸</text>
  <g fill="#cbd5e1"><polygon points="156,88 146,83 146,93"/><polygon points="306,88 296,83 296,93"/><polygon points="456,88 446,83 446,93"/></g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Vietoris–Rips 여과. ε 가 커지면서 edge 가 닫힐 때 고리가 태어나고, 삼각형이 채우면 죽는다. persistence = death − birth.</figcaption>
</figure>

각 특성은 *birth* $\varepsilon$ 과 *death* $\varepsilon$ 을 얻는다.
같은 구조를 여과 축 위의 수평 막대로 보면 *persistence barcode* 가 된다 —
긴 막대는 강건한 구조, 짧은 막대는 노이즈다.

<img src="/persistence-barcode.webp" alt="Persistence barcode — horizontal bars at varying heights show the lifespan of each topological feature across the filtration scale; longer bars indicate robust features, shorter bars are noise" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

## Persistence Diagram

각 특성을 $(b, d)$ 점으로 — birth 를 x축, death 를 y축에 — 찍으면
*persistence diagram* 이 된다. 모든 점은 대각선 $d = b$ 위쪽에 놓이고,
대각선으로부터의 거리가 모든 것을 말한다.

- **대각선에서 멀다** — persistence $d - b$ 가 크다. 넓은 스케일 범위에
  걸쳐 살아남는 구조: *진짜* 특성.
- **대각선 근처** — persistence 가 작다. 잠깐 나타났다 사라진 구조:
  *노이즈*.

<figure style="margin:24px auto;max-width:440px;">
<svg viewBox="0 0 440 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="440" height="320" fill="#f8fafc" rx="8"/>
  <line x1="60" y1="270" x2="400" y2="270" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="270" x2="60" y2="40" stroke="#64748b" stroke-width="1.2"/>
  <text x="230" y="300" text-anchor="middle" font-size="12" fill="#1e3a5f">birth (b)</text>
  <text x="22" y="160" text-anchor="middle" font-size="12" fill="#1e3a5f" transform="rotate(-90 22 160)">death (d)</text>
  <line x1="60" y1="270" x2="370" y2="55" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="350" y="80" font-size="10" fill="#94a3b8">d = b</text>
  <polygon points="60,270 80,256 360,61 340,75" fill="#94a3b822"/>
  <text x="250" y="200" font-size="10" fill="#94a3b8" transform="rotate(-34 250 200)">노이즈 띠</text>
  <g fill="#0d9488"><circle cx="110" cy="110" r="6"/><circle cx="95" cy="90" r="6"/><circle cx="140" cy="130" r="6"/></g>
  <text x="150" y="105" font-size="11" fill="#0d9488" font-weight="700">H₀ — 강건한 군집</text>
  <g fill="#e11d48"><circle cx="180" cy="120" r="6"/><circle cx="210" cy="145" r="6"/></g>
  <text x="225" y="128" font-size="11" fill="#e11d48" font-weight="700">H₁ — 강건한 고리</text>
  <g fill="#94a3b8"><circle cx="150" cy="172" r="4"/><circle cx="200" cy="218" r="4"/><circle cx="250" cy="252" r="4"/><circle cx="120" cy="158" r="4"/><circle cx="290" cy="270" r="4"/></g>
  <line x1="110" y1="110" x2="110" y2="225" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="116" y="180" font-size="10" fill="#d97706">persistence = d − b</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Persistence diagram. 점이 대각선에서 멀수록 그것이 나타내는 구조가 강건하다.</figcaption>
</figure>

이 그림을 신뢰할 수 있는 이유는 *안정성 정리(stability theorem)* 다.

$$ d_B\big(\mathrm{Dgm}(f), \mathrm{Dgm}(g)\big) \le \lVert f - g \rVert_\infty $$

> **수식 직관.** $f, g$ 는 두 여과를 정의하는 함수(여기서는 거리 행렬),
> $\mathrm{Dgm}$ 은 각각이 만드는 diagram, $d_B$ 는 두 diagram 사이의
> bottleneck 거리다. 부등식의 뜻은 — 입력을 조금 흔들면 diagram 도 그
> 조금만큼만 움직인다. 센서 노이즈나 측정 오차가 위상적 요약을 뒤집을 수
> 없다. 이상치 하나에 박살날 수 있는 분산 같은 일반 피처에는 없는
> 보장이다.

## Diagram 에서 신경망으로 — 그냥 넣을 수 없는 이유

여기 함정이 있다. Persistence diagram 은 벡터가 *아니다*. 순서 없는 점
집합이고, *크기가 가변* 이며(고객마다 특성 개수가 다르다), bottleneck /
Wasserstein 거리로 정의된 메트릭 공간에 산다. 일반 MLP 는 고정 길이의
순서 있는 벡터를 원한다. 둘은 만나지 않는다.

PersLay (Carrière et al., *JMLR 2020*) 가 그 다리다. diagram 을
*집합* 으로 보고 DeepSets 레시피 — 학습 가능한 순서 불변 집합 함수 —
를 적용한다.

$$ F(D) = \rho\!\left( \sum_{(b,d)\in D} w(b,d)\,\cdot\,\phi(b,d) \right) $$

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="180" fill="#f8fafc" rx="8"/>
  <rect x="20" y="55" width="90" height="75" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="65" y="48" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">D (점 집합)</text>
  <g fill="#4f46e5"><circle cx="45" cy="78" r="3"/><circle cx="72" cy="70" r="3"/><circle cx="88" cy="95" r="3"/><circle cx="55" cy="105" r="3"/><circle cx="80" cy="115" r="3"/></g>
  <rect x="150" y="60" width="92" height="64" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="196" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">φ(b,d)</text>
  <text x="196" y="104" text-anchor="middle" font-size="9" fill="#64748b">점 변환</text>
  <text x="196" y="48" text-anchor="middle" font-size="9" fill="#64748b">각 점 → 벡터</text>
  <rect x="282" y="60" width="92" height="64" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="328" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">w(b,d)</text>
  <text x="328" y="104" text-anchor="middle" font-size="9" fill="#64748b">persistence 가중치</text>
  <text x="328" y="48" text-anchor="middle" font-size="9" fill="#64748b">노이즈 → ~0</text>
  <rect x="414" y="60" width="92" height="64" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="460" y="88" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">ρ (Σ)</text>
  <text x="460" y="104" text-anchor="middle" font-size="9" fill="#64748b">순서 불변 합</text>
  <rect x="540" y="68" width="46" height="48" rx="6" fill="#0d9488" />
  <text x="563" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">F(D)</text>
  <text x="563" y="104" text-anchor="middle" font-size="9" fill="#fff">64D</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="110" y1="92" x2="148" y2="92"/><polygon points="148,92 140,88 140,96"/>
    <line x1="242" y1="92" x2="280" y2="92"/><polygon points="280,92 272,88 272,96"/>
    <line x1="374" y1="92" x2="412" y2="92"/><polygon points="412,92 404,88 404,96"/>
    <line x1="506" y1="92" x2="538" y2="92"/><polygon points="538,92 530,88 530,96"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">집합 함수로서의 PersLay: 각 점을 변환(φ)하고, persistence 로 가중(w)하고, 순서 불변으로 집계(ρ)하여 하나의 고정 64D 벡터로.</figcaption>
</figure>

세 조각이 일을 하고, 셋 모두 *학습 가능* 하다.

- **$\phi$ — 점 변환.** 각 $(b,d)$ 를 벡터로. 프로젝트의 `RationalHatPhi`
  는 먼저 점을 6개 피처로 수동 확장하고 —
  $[\,b,\ d,\ d-b,\ \tfrac{b+d}{2},\ b\cdot d,\ \tfrac{d}{b+\epsilon}\,]$ —
  2-layer MLP 를 태워, 태스크별로 어떤 비선형 조합이 중요한지 학습하게
  한다.
- **$w$ — 가중치.** $w(b,d) = |d-b|^{p}$ 로, 대각선 *위* 의 점
  (persistence 0)은 가중치 0 을 받는다. 노이즈가 공짜로 억제되고, zero
  padding 도 마스크 없이 무시된다.
- **$\rho$ — 집계.** 합(또는 mean / max / attention)이 가중된 집합을
  하나의 벡터로 압축한다 — 점이 도착한 순서에 불변이다.

전체가 미분 가능하므로 추천 손실이 $\phi$ 와 $w$ 안쪽까지 역전파된다.
신경망이 CTR 에, churn 에, next-best action 에 어떤 위상 구조가 중요한지
*스스로 발견* 한다 — 우리가 위상 피처를 손으로 설계해 두고 맞기를
바라는 대신.

> **고정 인코딩보다 나은 이유.** Persistence Landscapes (Bubenik, 2015)
> 와 Persistence Images (Adams et al., 2017) 도 diagram 을 벡터화한다 —
> 그러나 *고정* 변환이라 태스크에 무지하다. PersLay 의 한 수는 $\phi$,
> $w$, $\rho$ 를 학습 가능하게 만들어, 정적 기술자를 태스크 최적화
> 표현으로 바꾼 것이다.

## 소비에 실제로 형태가 있는가?

우아한 수학이 배포 허가증은 아니다. Expert 하나를 모델에 넣기 전에,
전체를 떠받치는 핵심 가설을 직접 시험했다 — *세션 행동에 고객을 분리하는
위상적 구조가 있는가?* 없다면 PersLay 는 짐덩어리다.

설계는 의도적으로 직설적이다.

- **데이터** — 실제 앱 세션 로그 90일(2026-01-13 → 04-12), 해시 샘플로
  **고객 120명**, 각 ~42개 세션 점.
- **고객별** — 세션 벡터(duration, pageview, buycount, buyprice, …)로
  점 구름을 만들고, Ripser 로 persistence diagram 을 구해, TDA 피처
  (H₀/H₁ 개수, 총/최대/평균 수명, persistence 엔트로피)로 요약.
- **시험** — KMeans 로 고객을 군집화하고 실루엣으로 분리도를 채점.
  *TDA persistence 피처* 와 *동일 조건 raw 집계*(평범한 세션 통계)를
  비교. 같은 고객, 같은 파이프라인, 피처 집합만 다르게.

<figure style="margin:24px auto;max-width:480px;">
<svg viewBox="0 0 480 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="480" height="250" fill="#f8fafc" rx="8"/>
  <text x="240" y="30" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">실루엣 점수 — 행동 그룹 분리도</text>
  <line x1="120" y1="200" x2="430" y2="200" stroke="#64748b" stroke-width="1"/>
  <line x1="120" y1="60" x2="120" y2="200" stroke="#64748b" stroke-width="1"/>
  <g font-size="9" fill="#94a3b8" text-anchor="end">
    <text x="114" y="203">0.0</text><text x="114" y="158">0.1</text><text x="114" y="113">0.2</text><text x="114" y="68">0.3</text>
  </g>
  <line x1="120" y1="155" x2="430" y2="155" stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3"/>
  <line x1="120" y1="110" x2="430" y2="110" stroke="#94a3b8" stroke-width="0.6" stroke-dasharray="3 3"/>
  <line x1="120" y1="132.5" x2="430" y2="132.5" stroke="#d97706" stroke-width="1.2" stroke-dasharray="6 3"/>
  <text x="426" y="128" text-anchor="end" font-size="9.5" fill="#d97706" font-weight="700">PASS 기준 = 0.15</text>
  <rect x="175" y="113.6" width="70" height="86.4" fill="#94a3b8" rx="3"/>
  <text x="210" y="107" text-anchor="middle" font-size="13" font-weight="700" fill="#64748b">0.192</text>
  <text x="210" y="218" text-anchor="middle" font-size="10" fill="#64748b">raw 집계</text>
  <text x="210" y="231" text-anchor="middle" font-size="9" fill="#94a3b8">(k = 6)</text>
  <rect x="305" y="65.5" width="70" height="134.5" fill="#0d9488" rx="3"/>
  <text x="340" y="59" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">0.299</text>
  <text x="340" y="218" text-anchor="middle" font-size="10" fill="#0d9488" font-weight="700">TDA persistence</text>
  <text x="340" y="231" text-anchor="middle" font-size="9" fill="#94a3b8">(k = 2)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">TDA persistence 피처가 raw 집계보다 행동 그룹을 뚜렷이 잘 분리한다(+0.108). 사전 설정 PASS 기준을 넘김. 판정: PASS.</figcaption>
</figure>

> **실루엣 점수란.** 군집화 결과가 "얼마나 깔끔하게 갈라졌는가" 를 재는
> 표준 지표다. 고객 한 명마다 두 가지 거리를 잰다 — *자기 그룹* 의 다른
> 고객들과의 평균 거리 $a$, 그리고 *가장 가까운 이웃 그룹* 까지의 평균
> 거리 $b$. 그 고객의 점수는 $(b-a)/\max(a,b)$ 이고, 전체 고객의 평균이
> 최종 실루엣 점수다. 범위는 $-1$ 에서 $+1$: $+1$ 에 가까울수록 그룹
> 안은 빽빽하고 그룹 사이는 멀다 — 또렷하게 나뉜 것이다. $0$ 근처면
> 경계가 흐릿하고, 음수면 아예 잘못 묶인 것이다. 통념상 $0.5$ 이상이면
> 뚜렷한 구조, $0.25{\sim}0.5$ 면 약하지만 실재하는 구조로 읽는다. 위
> 두 막대는 같은 실험을 두 피처 집합으로 따로 돌린 결과이고, 막대가
> 높을수록 그 피처가 고객을 더 또렷한 그룹으로 가른다는 뜻이다. 그래서
> 0.299 는 "강력" 이 아니라 "실재" — 아래 단서 박스가 겸손한 이유이기도
> 하다.

결과: **TDA 실루엣 0.299**($k=2$ 에서 최고) vs **raw 집계 0.192**
($k=6$) — **+0.108** 상승이고, TDA 점수가 사전 등록한 0.15 PASS 기준을
여유 있게 통과. 기록된 판정문은 이렇다 — *"Persistence 요약이 행동 그룹을
분리한다; PersLay 스타일 TDA 피처는 세션 행동에 정당화된다."* 이 한
문장이, PersLay Expert 가 이 코드베이스에 각주가 아니라 실제 모듈로
존재하는 이유다.

> 새겨둘 단서 하나. 이건 *고객 120명, 단일 윈도우* 프로브였고, 세션
> 로그의 위상은 12개월 금융 거래의 위상과 다르다. Expert 를 *만드는* 것을
> 정당화할 뿐, 운영 성능을 보증하지는 않는다. 정직한 독해는 "투자해도 좋다는
> 청신호" 이지 "사건 종결" 이 아니다.

## PersLay 의 위치

Persistence 계산은 비싸다 — Vietoris–Rips 복합체는 최악의 경우 $O(2^n)$
으로 폭발한다 — 그래서 프로젝트는 일을 둘로 나눈다.

1. **오프라인 (Airflow 배치).** `PersistenceExtractor` 가 각 고객의 점
   구름에 Ripser / Ripser++ 를 돌려 diagram(birth, death, 차원)을
   Parquet 로 쓴다. 무겁고, GPU 가속되며, 한 번만.
2. **온라인 (배치 학습/서빙).** 다섯 개의 `PersLayBlock` 이 그 diagram 을
   받아 $362\text{D} \to 64\text{D}$ 매핑을 end-to-end 로 학습한다. 64D
   출력은 PLE CGC gate 로 들어가 태스크별로 다른 Expert 와 혼합된다.

PersLay 는 7개 태스크의 `domain_experts` 멤버로 연결된다 —
**ctr, cvr**(engagement), **churn, retention, life_stage**(lifecycle),
**nba, spending_category**(consumption) — 평균과 개수가 놓치는 신호를
행동의 *형태* 가 그럴듯하게 담는 태스크들이다.

## 여기서 멈추는 이유

요약 통계량에 대한 불편함에서 출발해, persistent homology 와 그것을
다중 스케일로 만드는 여과를 따라갔고, persistence diagram 을 노이즈에
강건한 요약으로 읽었으며, PersLay 가 그 순서 없는 점 집합을 신경망이 쓸
수 있는 64D 벡터로 잇는 방식을 봤다. 그리고 가장 중요한 한 가지 —
*소비에 형태가 있는가* — 를 확인해 PASS 를 받았다.

남은 것은 기계 장치다. `RationalHatPhi`, persistence 가중치, 집계가
정확히 어떻게 조합되어 동작하는 레이어가 되는가. 그리고 왜 프로젝트는
PersLay 를 하나가 아니라 *다섯* 개 돌리는가 — Short vs Long range 에
호몰로지 차원을 교차한, 90일 군집과 12개월 공동은 질적으로 다른
신호이기에 각 블록이 별도 파라미터를 갖는 구조. 이것이 다음 편 **TDA-2** 의
주제다.
