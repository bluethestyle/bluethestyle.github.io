---
title: "[Study Thread] MULTI-1 — 빌려온 계측기: 네 학문, 하나의 소비 흐름, 24차원"
date: 2026-06-07 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, multidisciplinary, entropy, complexity, features, offline]
lang: ko
excerpt: "통계학은 행동을 보는 하나의 렌즈일 뿐이다. 이번 글은 다학제 피처 모듈을 연다 — 다른 네 학문의 계측기를 빌려와 만든 24차원. 화학 반응 속도론은 소비 가속도를, SIR 역학은 카테고리 전염을, 범죄학의 일상활동이론은 버스트성과 시간 리듬을, 파동 물리학은 소비의 주파수 스펙트럼을 읽는다. 같은 카드 거래 흐름, 네 개의 직교 투영, 그리고 느슨한 비유가 아닌 구조적 동형사상이 이를 정당화하는 이유."
series: study-thread
part: 22
alt_lang: /2026/06/07/multidisciplinary-features-en/
next_title: "DISTILL-1 — 선생과 학생: 무거운 PLE 를 학습해 놓고 왜 버리는가"
next_desc: "고용량 PLE-adaTT Teacher 에서 경량 LGBM Student 로: 지식 증류가 실제로 전이하는 것, 폐쇄망 배치 시스템이 운영에서 작은 모델을 필요로 하는 이유, 그리고 soft target 이 hard label 보다 더 많은 것을 담는 방식."
next_status: draft
---

*"Study Thread" 시리즈의 다학제 피처 서브스레드 1편. 영문/국문 병렬로,
이 프로젝트의 작지만 독특한 피처 그룹 하나를 정리한다. 통계량을 더 쌓는
대신, 다른 네 학문의 계측기를 빌려와 단 하나의 카드 거래 흐름에 들이댄
24차원 블록이다. 출처는 온프렘 프로젝트
`기술참조서/Multidisciplinary_피처_기술_참조서` 이고, 전체 PDF 는
서브스레드 마지막 편에 첨부한다. TDA 서브스레드가 행동이 어떤 형태를
갖는가를 물었다면, 이번 서브스레드는 자매 질문을 던진다 — 통계학의 렌즈가
바닥나면, 다음엔 누구의 계측기를 빌려오는가?*

> **한 줄 테제.** 모든 학문은 수백 년에 걸쳐 한 _종류_ 의 패턴을 잡아내는
> 수학적 도구를 정교하게 다듬어 왔다 — 화학은 변환의 속도와 장벽을,
> 역학은 상태의 확산을, 범죄학은 루틴의 규칙성과 파열을, 파동 물리학은
> 겹치는 리듬의 간섭을. 이 도구들은 분자, 병원체, 범죄, 파동을 위해
> 만들어졌지만 _수식_ 은 그걸 모른다. 고객의 90일 거래 윈도우에 들이대면,
> 평균과 분산과 빈도로는 결코 볼 수 없는 행동 구조를 추출한다. 24차원,
> 네 개의 빌려온 계측기, 734D 텐서의 약 3.3%.

## 하나의 렌즈로는 부족한 이유

전통적 피처 엔지니어링은 통계학이라는 하나의 렌즈로 데이터를 본다.
평균, 분산, 빈도, 상관 — 강력하지만, 행동이 담은 구조의 한 조각만
드러낸다.

참조서의 비유는 정확하다. 하나의 관점으로 행동을 분석하는 것은 조각상을
정면에서만 촬영하는 것과 같다. 정면 사진만으로는 깊이감, 뒷면의 질감,
측면의 곡률을 알 수 없다. 다학제 접근은 같은 대상을 여러 각도에서 동시에
촬영하여 입체적 구조를 복원한다.

각 학문은 특정 _종류_ 의 패턴을 탐지하기 위해 수백 년에 걸쳐 갈아낸
계측기를 가져온다. 핵심은 네 투영이 서로 거의 _직교_ 한다는 것이다 —
같은 카드 거래 데이터의 다른 축을 본다 — 그래서 결합하면 피처 공간이
중복 없이 효율적으로 확장된다.

| 학문 분야 | 빌려온 계측기 | 통계학이 놓치는 것 |
| --- | --- | --- |
| 물리화학 (반응 속도론) | 속도, 장벽, 촉매, 포화 | 추세가 아닌 _가속도_(2차 도함수); 새 카테고리 진입의 에너지 장벽; 촉매(급여일)가 행동을 움직이는 방식 |
| 역학 (SIR 모델) | 구획 흐름 S→I→R | 개별 행동이 아닌 집단 수준 _동역학_; 카테고리 채택의 전파 임계값 $R_0$ |
| 범죄학 (일상활동이론) | 규칙성 vs 이탈 | 시간의 _원형적_ 성격(23시와 1시는 가깝다); 버스트성 vs 규칙성; 루틴이 깨지는 변조점 |
| 파동 물리학 (스펙트럼 분석) | 간섭, 동기화 | FFT _주파수 영역_ 분해 — 스펙트럼 엔트로피, 지배 주기, 위상 동기화, 교차 코히어런스 |

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="250" fill="#f8fafc" rx="8"/>
  <text x="290" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">하나의 거래 흐름 → 네 개의 직교 투영</text>
  <rect x="20" y="100" width="86" height="50" rx="6" fill="#1e3a5f"/>
  <text x="63" y="121" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">90일</text>
  <text x="63" y="136" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">카드 거래</text>
  <rect x="180" y="48" width="150" height="34" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1"/>
  <text x="255" y="69" text-anchor="middle" font-size="10.5" fill="#0d9488" font-weight="700">화학 반응 속도론 · 6D</text>
  <rect x="180" y="92" width="150" height="34" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1"/>
  <text x="255" y="113" text-anchor="middle" font-size="10.5" fill="#d97706" font-weight="700">SIR 역학 · 5D</text>
  <rect x="180" y="136" width="150" height="34" rx="6" fill="#e11d4818" stroke="#e11d48" stroke-width="1"/>
  <text x="255" y="157" text-anchor="middle" font-size="10.5" fill="#e11d48" font-weight="700">범죄 패턴 · 5D</text>
  <rect x="180" y="180" width="150" height="34" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1"/>
  <text x="255" y="201" text-anchor="middle" font-size="10.5" fill="#4f46e5" font-weight="700">파동 물리학 · 8D</text>
  <rect x="400" y="100" width="92" height="50" rx="6" fill="#4f46e5"/>
  <text x="446" y="121" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">24D</text>
  <text x="446" y="137" text-anchor="middle" font-size="9" fill="#fff">다학제</text>
  <g stroke="#cbd5e1" stroke-width="1.4" fill="none">
    <path d="M 106 120 C 140 65, 150 65, 180 65"/>
    <path d="M 106 122 C 140 109, 150 109, 180 109"/>
    <path d="M 106 128 C 140 153, 150 153, 180 153"/>
    <path d="M 106 130 C 140 197, 150 197, 180 197"/>
  </g>
  <g stroke="#cbd5e1" stroke-width="1.4" fill="none">
    <path d="M 330 65 C 365 100, 375 110, 400 118"/>
    <path d="M 330 109 C 365 118, 375 120, 400 122"/>
    <path d="M 330 153 C 365 140, 375 132, 400 128"/>
    <path d="M 330 197 C 365 160, 375 145, 400 132"/>
  </g>
  <text x="540" y="128" text-anchor="middle" font-size="9" fill="#64748b" transform="rotate(90 540 128)">→ 734D 텐서</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">같은 90일 거래 윈도우가 네 개의 학문별 추출기를 통과한다. 각각 다른 축을 읽고, 24D 합집합이 734D 메인 텐서에 합류한다.</figcaption>
</figure>

## 이빨 달린 비유 — 구조적 동형사상

명백한 반론: 소비자는 분자가 아니고, 카테고리 채택은 전염병이 아니다.
그런데 왜 화학 방정식을 빌려오는 게 귀여운 은유 이상인가?

참조서는 이 점에 단호하다. 여기서 "아날로지"는 느슨한 비유가 아니라
_구조적 동형사상(structural isomorphism)_ 에 기반한다. 두 시스템의
표면적 대상(분자 vs 소비자)은 완전히 다르지만, 그 대상들 사이의 _관계
구조_ 가 수학적으로 동일할 수 있다. 반응물 농도가 절반이 되는 시간과
고객의 거래 빈도가 절반이 되는 시간은 모두 같은 지수 감쇠가 지배한다.
인구 중 아직 감수성인 비율과 고객이 아직 채택하지 않은 카테고리 비율은
모두 같은 구획 모델의 _S_ 구획이다.

> **수식 직관.** 수식이 같다면, 그 수식이 포착하는 패턴도 같다. 기호가
> 원래 무엇을 _가리켰는지_ — 원자, 병원체 — 는 그 수식이 새 도메인에
> 유효한지와 무관하다. 이것이 바로 physics-informed ML 과 전이 학습의
> 핵심 동작이다. 한 도메인에서 입증된 수학적 구조를 같은 관계 구조가
> 재현되는 곳에 적용한다. 정직한 단서(참조서도 명시한다): 수학은
> 전이되지만 원래 학문의 _인과 메커니즘_ 은 전이되지 않는다. 이들은 패턴
> 포착 도구이지 인과 설명이 아니다.

화학과 역학을 관통하는 한 줄기는 지수 함수다. 변화율이 현재 상태에
비례하는 모든 과정은 지수적으로 진행된다.

$$ \frac{dy}{dt} = \alpha y \quad\Longrightarrow\quad y(t) = y_0\, e^{\alpha t} $$

이 하나의 미분방정식이 화학의 Arrhenius 속도 $k = A\,e^{-E_a/RT}$ 와
역학의 초기 감염 성장 $I(t) \approx I_0\, e^{(\beta S_0 - \gamma)t}$ 을
모두 떠받친다 — 같은 뼈대, 두 학문.

## 계측기 1 — 화학 반응 속도론 (6D)

화학은 변환이 _얼마나 빨리_ 진행되며 _어떤 장벽_ 을 넘어야 하는지를
연구한다. 소비에 매핑하면 "반응"은 카테고리 전환, 활성화 에너지 $E_a$ 는
새 카테고리 진입의 마찰, 촉매는 소비되지 않으면서 반응을 가속하는 외부
이벤트(급여일, 프로모션)다.

대표 피처는 **소비 가속도** — 1차 도함수인 추세선으로는 볼 수 없는,
추세의 2차 도함수다. 프로젝트는 이를 세 개의 30일 윈도우에 대한 이산
2차 유한차분으로 계산한다.

$$ f''(t) \approx f(t+\Delta t) - 2f(t) + f(t-\Delta t) $$

코드로는 `spending_acceleration = avg_w3 - 2*avg_w2 + avg_w1` 이고, 세
항은 각각 첫/중간/마지막 30일의 평균 지출이다. 양수면 지출이 _가속_
중(볼록), 음수면 감속 중(오목)이다 — 그리고 소비 감속은 빈도나 금액
감소보다 수 주 앞서는, 알려진 이탈의 _선행_ 지표다.

> **역사적 배경.** 지수 속도 법칙은 Svante Arrhenius(1889)가 설탕
> 가수분해의 온도 의존성을 맞추며 도출했고, 훗날 Boltzmann 분포
> $P(E)\propto e^{-E/k_BT}$ 로부터 정당화됐다. 특성 시간으로서의
> 반감기는 1900년대 방사성 붕괴를 연구한 Ernest Rutherford 가 체계화했다.
> 고고학 탄소 연대를 매기는 그 $T_{1/2} = \ln 2 / k$ 가 이제 고객의 중앙값
> 거래 간격을 대신한다.

전체 6D: `new_category_activation_rate`(역 $E_a$ 의 프록시),
`spending_half_life`($T_{1/2}$ = 중앙값 거래 간격), `spending_acceleration`,
`dormancy_reactivation_rate`(휴면 카테고리의 촉매적 재활성),
`catalyst_sensitivity`(급여일 탄성 = 월초 vs 월말 일평균 지출),
`saturation_proximity`(최대 지출이 평균 + 1σ 에 얼마나 근접한지 — 소비
천장).

## 계측기 2 — SIR 역학 확산 (5D)

역학은 인구를 구획 — Susceptible, Infected, Recovered — 으로 나누고
개체가 그 사이를 흐르는 것을 지켜본다. Kermack–McKendrick(1927) 모델은
미분방정식 시스템이다.

$$ \frac{dS}{dt} = -\beta S I, \quad \frac{dI}{dt} = \beta S I - \gamma I, \quad \frac{dR}{dt} = \gamma I $$

기본 재생산수 $R_0 = \beta/\gamma$ 는 무차원 전파 임계값이다. $R_0 > 1$
이면 전염병이 확산되고 $R_0 < 1$ 이면 소멸한다. 소비로 옮기면 $R_0 > 1$
은 카테고리 채택이 자기강화적이라는 뜻이다.

매핑은 시적인 것이 아니라 구조적이다. 고객의 **감수성** 비율은 인구
Top-15 MCC 중 아직 _이용하지 않은_ 비율(`susceptible_count / 15`),
**감염** 비율은 최근 일평균 빈도가 _성장 중_ 인 카테고리 비율,
**회복** 비율은 이전엔 썼으나 최근 30일에 버린 비율 — 채택, 그리고
면역/무관심. 성장 판정은 일평균을 비교하여 윈도우 길이 차이를 보정한다.

$$ \text{infected} = \mathbb{1}\!\left[\, \text{recent\_count} > \text{older\_count}\cdot \tfrac{30}{L-30} \,\right] $$

`max_weekly_new_mcc` 가 치솟는 고객 — 한 주에 다수의 첫 거래 카테고리 —
은 소비판 _슈퍼 스프레더_ 다. 5D: `sir_susceptible`, `sir_infected`,
`sir_recovered`, `max_weekly_new_mcc`(전염 피크),
`category_lifecycle_mean`(카테고리 평균 생존 일수, 첫→마지막 거래).

## 계측기 3 — 범죄 패턴 / 일상활동이론 (5D)

Cohen & Felson 의 일상활동이론(1979)은 행동이 일상 루틴에 의해 결정되고,
루틴의 _파열_ 이 비정상 사건의 기회를 만든다고 본다. 소비로 옮기면 질문은
이렇게 된다 — 고객의 루틴은 얼마나 규칙적이며, 어디서 깨지는가?

대표 측정값은 Barabási 의 인간 동역학 연구(2005)에서 온 **버스트성**
이다. 인간 행동은 포아송 과정이 아니다 — 짧은 폭발과 긴 휴지가 반복되는
중후꼬리 분포를 따른다. $[-1,1]$ 로 정규화하면:

$$ B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} $$

여기서 $\sigma_\tau, \mu_\tau$ 는 거래 간격의 표준편차와 평균이다.
$B=-1$ 은 완벽히 규칙적(정기 결제 같은 박자), $B=0$ 은 포아송 기준선
(랜덤), $B=+1$ 은 극단적 군집(몰아 쓰고 침묵)이다.

> **역사적 배경.** 일상활동이론은 Lawrence Cohen 과 Marcus Felson 이
> _American Sociological Review_(1979)에 발표하여, 범죄 예방 정책을
> "범죄자 교정" 에서 "상황적 예방" 으로 전환시켰다. Barabási 는 버스트성을
> _Nature_(2005)에서 정식화하며, 인간이 과업을 무작위가 아니라 우선순위
> 대기열에서 처리하기에 사건 간격이 멱법칙을 따름을 보였다 — 고우선
> 과업은 폭발하고, 저우선 과업은 중후꼬리에서 대기한다.

또 하나의 규칙성 계측기는 **원형 분산** 이다. 시간은 선이 아니라 원이기
때문이다 — 23시와 1시는 22시간이 아니라 2시간 떨어져 있다. 시각 $h$ 를
각도 $\theta = 2\pi h/24$ 로 매핑하고 평균 결과 길이 $\bar{R}$ 을 취하면:

$$ \mathrm{CV} = 1 - \bar{R}, \quad \bar{R} = \sqrt{\Big(\tfrac1n\textstyle\sum \sin\theta_i\Big)^2 + \Big(\tfrac1n\textstyle\sum \cos\theta_i\Big)^2} $$

$\mathrm{CV}\to 0$ 은 고객이 항상 비슷한 시각(예: 점심시간)에 거래함을,
$\mathrm{CV}\to 1$ 은 시간대가 고르게 분산됨을 뜻한다 — 평범한 직선
분산으로는 복원할 수 없는 구조다. 5D: `burstiness`,
`recurrence_period`(7/14/21/28/30일 중 자기상관 피크 lag),
`routine_breakpoint_count`(주간 총액 평균 교차 횟수), `circular_variance`,
그리고 `max_amount_zscore` — 시그모이드 $\sigma(z-3)$ 로 접은 이상치
강도로, 단일 극단 거래가 피처를 지배하는 대신 1 쪽으로 포화된다.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="230" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">버스트성 — 90일 축 위의 세 가지 소비 리듬</text>
  <text x="20" y="62" font-size="10.5" fill="#0d9488" font-weight="700">B = −1  규칙적</text>
  <line x1="150" y1="58" x2="500" y2="58" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#0d9488"><circle cx="180" cy="58" r="3.5"/><circle cx="225" cy="58" r="3.5"/><circle cx="270" cy="58" r="3.5"/><circle cx="315" cy="58" r="3.5"/><circle cx="360" cy="58" r="3.5"/><circle cx="405" cy="58" r="3.5"/><circle cx="450" cy="58" r="3.5"/></g>
  <text x="20" y="122" font-size="10.5" fill="#64748b" font-weight="700">B = 0  포아송</text>
  <line x1="150" y1="118" x2="500" y2="118" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#64748b"><circle cx="172" cy="118" r="3.5"/><circle cx="205" cy="118" r="3.5"/><circle cx="240" cy="118" r="3.5"/><circle cx="305" cy="118" r="3.5"/><circle cx="350" cy="118" r="3.5"/><circle cx="398" cy="118" r="3.5"/><circle cx="470" cy="118" r="3.5"/></g>
  <text x="20" y="182" font-size="10.5" fill="#e11d48" font-weight="700">B = +1  군집적</text>
  <line x1="150" y1="178" x2="500" y2="178" stroke="#e2e8f0" stroke-width="1"/>
  <g fill="#e11d48"><circle cx="168" cy="178" r="3.5"/><circle cx="176" cy="178" r="3.5"/><circle cx="184" cy="178" r="3.5"/><circle cx="192" cy="178" r="3.5"/><circle cx="360" cy="178" r="3.5"/><circle cx="368" cy="178" r="3.5"/><circle cx="376" cy="178" r="3.5"/></g>
  <text x="260" y="214" text-anchor="middle" font-size="10" fill="#94a3b8">같은 거래 건수, 다른 간격 분포 — B만이 이들을 구별한다.</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">버스트성 B = (σ−μ)/(σ+μ)는 거래 간격의 변동계수를 [−1, 1]로 압축한다. 포아송 과정(σ=μ)은 정확히 0에 놓인다.</figcaption>
</figure>

## 계측기 4 — 파동 물리학 / 스펙트럼 분석 (8D)

여러 파동이 겹칠 때, 위상 관계가 보강(constructive)인지 상쇄
(destructive)인지를 결정한다. 파동 물리학 모듈은 일일 지출 시계열을 FFT 로
주파수 영역으로 옮겨 그 구조를 읽는다. 대표 피처는 **스펙트럼 엔트로피**
— 정규화된 파워 스펙트럼에 적용한 Shannon 엔트로피다.

$$ p(k) = \frac{|X(k)|^2}{\sum_{k'} |X(k')|^2}, \quad H = -\sum_{k=1}^{K} p(k)\,\log_2 p(k) $$

매주 같은 요일에 같은 금액을 쓰는 고객은 $f = 1/7$ 에 에너지가 집중되어
_낮은_ 엔트로피(규칙적, 예측 가능)를 갖고, 시점과 금액이 흩어진 고객은
에너지가 여러 주파수에 퍼져 _높은_ 엔트로피를 갖는다. 이는 Shannon 의
불확실성을 통신 채널에서 소비 리듬으로 그대로 옮겨온 것이다.

> **역사적 배경.** Shannon 엔트로피 $H = -\sum p_i \log p_i$ 는 1948년
> "A Mathematical Theory of Communication" 에 등장했고, Boltzmann 의
> 열역학 엔트로피 $S = -k_B\sum p_i\ln p_i$ 와 수학적으로 동일하다 — von
> Neumann 이 Shannon 에게 "아무도 엔트로피가 뭔지 정확히 모르니, 그렇게
> 부르면 논쟁에서 항상 이긴다" 고 했다는 일화가 유명하다. 위상 동기화
> 값(PLV)은 신경과학의 기능적 연결성 분석에서 내려왔으며, 두 신호가 일정한
> 위상차를 유지하는지를 측정한다.

여기 나머지 계측기들은 주파수 영역에서 _카테고리 간 관계_ 를 측정한다 —
교차 스펙트럼 코히어런스(두 카테고리가 같은 주파수로 진동하는가?), 위상
동기화 값, 그리고 카테고리 쌍이 동위상인지에 대한 보강 간섭 비율. 8D:
`spectral_entropy`, `weekly_harmonic_power`(1/7 및 2/7 빈의 에너지),
`cross_spectral_coherence`, `dominant_period`($T = 1/f_{\text{peak}}$),
`spectral_centroid_shift`(전반 vs 후반 평균 주파수), `phase_locking_value`,
`mean_phase_difference`, `constructive_interference_ratio`.

<figure style="margin:24px auto;max-width:480px;">
<svg viewBox="0 0 480 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="480" height="250" fill="#f8fafc" rx="8"/>
  <text x="240" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">스펙트럼 엔트로피 — 집중된 에너지 vs 분산된 에너지</text>
  <line x1="50" y1="210" x2="220" y2="210" stroke="#64748b" stroke-width="1"/>
  <line x1="50" y1="60" x2="50" y2="210" stroke="#64748b" stroke-width="1"/>
  <text x="135" y="234" text-anchor="middle" font-size="9.5" fill="#0d9488" font-weight="700">낮은 H — 규칙적</text>
  <g fill="#0d9488">
    <rect x="62" y="205" width="14" height="5"/><rect x="84" y="200" width="14" height="10"/>
    <rect x="106" y="78" width="14" height="132"/><rect x="128" y="198" width="14" height="12"/>
    <rect x="150" y="203" width="14" height="7"/><rect x="172" y="206" width="14" height="4"/><rect x="194" y="204" width="14" height="6"/>
  </g>
  <text x="113" y="72" text-anchor="middle" font-size="8.5" fill="#0d9488">f = 1/7</text>
  <line x1="270" y1="210" x2="440" y2="210" stroke="#64748b" stroke-width="1"/>
  <line x1="270" y1="60" x2="270" y2="210" stroke="#64748b" stroke-width="1"/>
  <text x="355" y="234" text-anchor="middle" font-size="9.5" fill="#e11d48" font-weight="700">높은 H — 분산</text>
  <g fill="#e11d48">
    <rect x="282" y="150" width="14" height="60"/><rect x="304" y="135" width="14" height="75"/>
    <rect x="326" y="160" width="14" height="50"/><rect x="348" y="140" width="14" height="70"/>
    <rect x="370" y="152" width="14" height="58"/><rect x="392" y="145" width="14" height="65"/><rect x="414" y="158" width="14" height="52"/>
  </g>
  <text x="135" y="50" text-anchor="middle" font-size="9" fill="#64748b">파워 스펙트럼</text>
  <text x="355" y="50" text-anchor="middle" font-size="9" fill="#64748b">파워 스펙트럼</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">에너지가 한 주파수 빈에 집중되면(왼쪽) 낮은 스펙트럼 엔트로피 — 규칙적 7일 리듬; 여러 빈에 퍼지면(오른쪽) 높은 엔트로피 — 불규칙한 패턴.</figcaption>
</figure>

## 24D 가 안착하는 곳

네 추출기는 하나의 입력을 공유한다 — 같은 90일 카드 거래 윈도우,
`customer_id_encrypted IS NOT NULL AND transaction_amount > 0` 으로
필터링 — 그리고 DuckDB 인메모리 SQL(스펙트럼 블록은 numpy FFT 하이브리드)
로 실행되어 각각 ZSTD-Parquet 파일을 쓴다. `feature_integrator.py` 가 네
파일을 `customer_id_encrypted` 로 LEFT JOIN 하여 24D 블록으로 만들고,
이는 734D 메인 텐서(644D normalized + 90D raw power-law) 안에 들어간다.
스키마 레지스트리 `feature_schema.yaml` 이 `chemical_kinetics_001` ~
`interference_008` 의 24개 키를 보관한다.

다운스트림에서 24D 는 전체 734D 의 일부로 세 Shared Expert —
**DeepFM**, **Causal**, **OT** — 에 자동 투입된다(모두 644D 정규화
슬라이스). DeepFM 은 다학제 피처와 나머지 간 교차 패턴(예:
`spending_acceleration` × 이탈 확률)을 field interaction 으로 학습하고,
Causal Expert 는 이들 사이의 인과 방향을 DAG 로 복원하려 한다. 참조서의
태스크별 예상 기여도는 계측기를 태스크에 대응시킨다 — 화학 반응 속도론
(가속도/포화)은 **churn** 과 **LTV** 로, 역학 확산은 **NBA** 와
**cross-sell** 로, 범죄 패턴(버스트성/주기성)은 **timing** 과 소비 주기
태스크로, 간섭(스펙트럼)은 **spending category** 와 merchant affinity 로.

참조서가 거듭 강조하는 경고: 24D 는 734D 의 약 3.3%에 불과하고,
아날로지에는 한계가 있다. 이 피처들은 패턴 포착 계측기이지 인과 설명이 아니다 —
소비자는 분자가 아니다. 또한 다수가 데이터 품질에 의존한다(원형 분산은
거래 시각이, SIR 비율은 MCC 매핑이 필요). 데이터가 부족하면 COALESCE
기본값이 반환되며, 그 기본값은 "정상" 이 아니라 "패턴 없음" 을 뜻한다.

## 여기서 멈추는 이유

하나의 렌즈의 한계에서 출발해, 다른 학문의 계측기를 빌리는 것이 느슨한
은유가 아니라 구조적 동형사상임을 논증하고, 네 계측기를 차례로 짚었다 —
소비 가속도를 읽는 화학 반응 속도론, 카테고리 전염을 읽는 SIR, 버스트성과
시간 리듬을 읽는 일상활동이론, 주파수 스펙트럼을 읽는 파동 물리학.
24차원, 네 개의 직교 투영, 하나의 거래 흐름, 734D 텐서에 합류.

이 프로젝트 전체에서 남은 것은 그 텐서를 _먹는_ 모델이다 — 그리고 무겁고
고용량인 PLE-adaTT Teacher 를 학습해 놓고, 거의 도착적으로, 작은 LGBM
Student 를 위해 그것을 버리는 순간. 폐쇄망 배치 시스템이 왜 큰 모델을
작은 모델로 증류하는가, 실제로 어떤 지식이 전이되는가, soft target 이
hard label 보다 더 많은 것을 담는 이유는 무엇인가 — 이것이 다음 편
**DISTILL-1** 의 주제다.
