---
title: "[Study Thread] TSFEAT-1 — 숫자 속의 순서: 소비 이력에 적용하는 고전 시계열 피처"
date: 2026-06-07 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, timeseries, seasonality, spectral, features, offline]
lang: ko
excerpt: "합계, 평균, 최대값은 소비 이력이 가진 유일한 것 — 순서 — 를 버린다. 이번 글은 오프라인 시계열 피처 모듈을 따라간다: 소비 시계열을 수준/변동성/잡음으로 나누는 확률과정 분해, 자기상관과 정상성, FFT 스펙트럼 피처, 변동성 형상, 엔트로피 복잡도 측도 — 학습된 50D Mamba 임베딩 옆에 놓여 68D 를 이루는 수작업 18D. 실제 수식과, 각 숫자가 734D 텐서 어디에 들어가는지까지."
series: study-thread
part: 21
alt_lang: /2026/06/07/timeseries-features-en/
next_title: "MULTI-1 — 빌려온 계측기: 네 학문, 하나의 소비 흐름, 24차원"
next_desc: "교차 도메인 피처 그룹 — 네 학문에서 빌려온 24차원: 화학 반응 속도론의 소비 가속도, SIR 역학의 카테고리 전염, 범죄학 일상활동이론의 버스트성과 시간 리듬, 파동 물리학의 주파수 스펙트럼 — 그리고 소비 추천 시스템이 왜 신용카드를 들어본 적 없는 학문에 손을 뻗는가."
next_status: draft
---

*"Study Thread" 시리즈의 일부로, 오프라인* 시계열 피처 *모듈을 다루는
짧은 서브스레드를 연다. 이번 편과 다음 편에서 영문/국문 병렬로, 원시
소비 이력이 어떻게 고정 폭 피처 블록이 되는지 정리한다. 출처는 온프렘
프로젝트* `기술참조서/TimeSeries_피처_기술_참조서` *이고, 전체 PDF 는
서브스레드 마지막 편에 첨부한다. 시작 전에 한 가지 구분 — 이 프로젝트에는
"시계열"이라 불리는 것이 둘 있다. 이 글은* 고전적인 수작업 *피처 — 자기상관,
FFT 스펙트럼, CUSUM 변환점, 엔트로피 — 를 오프라인에서 사전 계산하는
이야기다. PLE 내부에서 end-to-end 학습되는 딥 모델인* Temporal Expert
(Mamba→LNN) *가* 아니다. *같은 어휘, 다른 인스턴스, 가중치 공유 없음. 이
경계선에 계속 발이 걸릴 테니 분명히 표시해 둔다.*

> **이 모듈이 무엇인가, 한 문장으로.** 소비 이력은 시스템 전체에서
> *순서* 를 담은 유일한 피처 소스다 — 다른 모든 피처는 날짜를 섞어도
> 변하지 않지만, 섞인 거래 스트림은 다른 고객이다. 이 모듈의 일은
> *순서를 파괴하지 않으면서 정량화* 하는 것이다: 수작업 통계 18D 블록
> (분포 형상, 주파수, 변환점, 자기상관, 복잡도)이 학습된 50D Mamba
> 임베딩 옆에 놓여 총 **68D**, 734D 메인 텐서의 약 9.3% 이자 그 안에서
> 차원당 정보 밀도가 가장 높은 그룹이다.

## 왜 합계, 평균, 최대값이 아닌가?

두 고객 A, B. 90일간 둘 다 정확히 300만원을 쓴다. 일 평균 3.3만원,
표준편차 2.1만원, 최대값 12만원 — *모든 집계에서 동일하다*.

- **고객 A** 는 매주 월요일 4만원, 금요일 6만원을 쓰고 나머지는 거의
  없다. 강한 주간 리듬.
- **고객 B** 는 80일간 거의 안 쓰다가 마지막 10일에 300만원을 쏟아붓는다.
  극단적 레짐 변화 1회.

합계, 평균, 표준편차는 둘을 구분하지 못한다. 이 통계량들이 *순서 불변*
이기 때문이다 — 날짜를 섞어도 움직이지 않는다. 차이는 전적으로 **순서**
에 있고, 순서를 보는 피처만이 그 차이를 잡아낸다: A 의 높은 lag-7 자기상관(B 는
≈0), B 의 큰 CUSUM `max_shift_magnitude`, A 의 낮은 스펙트럼 엔트로피
(에너지가 주간 주파수에 집중) 대 B 의 높은 값.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">같은 합계·평균·최대값 — 다른 순서</text>
  <text x="140" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">고객 A — 주간 리듬</text>
  <line x1="40" y1="120" x2="250" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <g fill="#0d9488">
    <rect x="52" y="78" width="6" height="42"/><rect x="82" y="62" width="6" height="58"/>
    <rect x="112" y="78" width="6" height="42"/><rect x="142" y="62" width="6" height="58"/>
    <rect x="172" y="78" width="6" height="42"/><rect x="202" y="62" width="6" height="58"/>
    <rect x="232" y="78" width="6" height="42"/>
  </g>
  <text x="140" y="138" text-anchor="middle" font-size="9.5" fill="#64748b">ρ(7) 높음 · 스펙트럼 엔트로피 낮음</text>
  <line x1="280" y1="44" x2="280" y2="150" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="52" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">고객 B — 후반 폭발</text>
  <line x1="310" y1="120" x2="520" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <g fill="#e11d48">
    <rect x="318" y="114" width="5" height="6"/><rect x="338" y="116" width="5" height="4"/><rect x="358" y="115" width="5" height="5"/>
    <rect x="378" y="113" width="5" height="7"/><rect x="398" y="116" width="5" height="4"/>
    <rect x="458" y="66" width="6" height="54"/><rect x="472" y="58" width="6" height="62"/><rect x="486" y="70" width="6" height="50"/><rect x="500" y="60" width="6" height="60"/>
  </g>
  <text x="420" y="138" text-anchor="middle" font-size="9.5" fill="#64748b">CUSUM 이동 큼 · ρ(7) ≈ 0</text>
  <rect x="120" y="170" width="320" height="46" rx="6" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>
  <text x="280" y="190" text-anchor="middle" font-size="10.5" fill="#1e3a5f" font-weight="700">집계 관점: Σ = 300만 · 평균 3.3만 · σ 2.1만 · 최대 12만</text>
  <text x="280" y="207" text-anchor="middle" font-size="10" fill="#64748b">— 구분 불가 —</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">집계는 시간 축을 뭉갠다. 순서를 보는 피처(자기상관, CUSUM, 스펙트럼 엔트로피)만이 A 와 B 를 가른다.</figcaption>
</figure>

참조서는 모듈 전체를 네 개의 상보적 렌즈로 짠다. 각 렌즈가 시계열에
대해 다른 질문을 던진다:

| 렌즈 | 던지는 질문 | 기법 | 피처 블록 |
| --- | --- | --- | --- |
| 시간 도메인 | 값이 시간에 따라 어떻게 변하는가? | 자기상관, 변환점, 이동평균 | AR 4D, Changepoint 3D |
| 주파수 도메인 | 어떤 주기로 반복되는가? | FFT, 스펙트럼 분석 | Freq 4D |
| 분포/형상 | 값의 분포는 어떤 모양인가? | 왜도, 첨도, 꼬리 | Dist 4D |
| 정보이론 | 얼마나 복잡하고 예측 가능한가? | 엔트로피, 순열 엔트로피 | Complex 3D |

이 네 렌즈가 수작업 **18D** 다. 그 위에 학습된 **50D** Mamba 상태공간
임베딩이 놓여 같은 정보를 비선형으로 통합한다 — 사람이 설계한 렌즈와
모델이 학습한 렌즈를 일부러 나란히 둔 것이다.

> **역사적 배경.** 이 모듈의 거의 모든 도구는 머신러닝보다 수십 년
> 앞선다. CUSUM 은 제조 공정의 평균 이동을 잡으려는 E. S. Page 의 1954년
> 품질관리 기법이다. Ljung–Box 검정은 1970년 Box–Pierce portmanteau 를
> 1978년에 개량한 것이다. Approximate Entropy 는 Pincus 의 1991년 심박
> 변이도 측도이고, Sample Entropy 는 그 편향을 보정한 2000년 후속작
> (Richman & Moorman)이다. FFT 자체는 Cooley–Tukey, 1965년. 우리는 여기서
> 피처를 발명하는 게 아니라, 신호처리와 정보이론 50년을 빌려와 신용카드
> 스트림에 겨누는 것이다.

## 분해 — 수준, 변동성, 잡음

모듈의 한가운데에 분해가 있다. 그런데 흔히 기대하는 STL 이나
Hodrick–Prescott 의 추세/계절 분리가 *아니다*. (월별 데이터 기준
$\lambda = 14{,}400$ 인 그 HP 필터는 *Economics* Expert 의 소득 분해 그룹에
산다 — 다른 글, 다른 모듈. 둘 다 소비 시계열을 건드리니 구분해 둘 가치가
있다.) 여기서는 시계열을 **확률과정** 으로 읽어 세 개의 시변 성분으로
나눈다:

$$ X_t = \mu_t + \sigma_t\,\epsilon_t $$

- $\mu_t$ — 시변 수준(추세 + 계절성): 소비가 *어디에* 놓이는가.
  변환점 크기와 Mamba 임베딩이 포착.
- $\sigma_t$ — 시변 변동성: *얼마나 불확실한가*. 첨도와 엔트로피가 포착.
- $\epsilon_t$ — 백색 잡음, $E[\epsilon_t]=0$, $\mathrm{Var}(\epsilon_t)=1$:
  그 분포 형상(왜도, 두꺼운 꼬리)을 분포 피처가 읽는다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 330" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="330" fill="#f8fafc" rx="8"/>
  <text x="300" y="24" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Xₜ = μₜ + σₜ·εₜ — 한 시계열, 세 층</text>
  <text x="20" y="64" font-size="11" font-weight="700" fill="#1e3a5f">Xₜ (관측)</text>
  <polyline points="150,72 180,55 210,80 240,48 270,70 300,40 330,66 360,44 390,78 420,50 450,72 480,46 510,68 540,52"
    fill="none" stroke="#1e3a5f" stroke-width="1.6"/>
  <text x="20" y="142" font-size="11" font-weight="700" fill="#4f46e5">μₜ (수준 / 추세)</text>
  <path d="M 150 158 Q 300 120, 540 132" fill="none" stroke="#4f46e5" stroke-width="2"/>
  <text x="544" y="135" font-size="9" fill="#4f46e5">→ 변환점, Mamba</text>
  <text x="20" y="222" font-size="11" font-weight="700" fill="#d97706">σₜ (변동성)</text>
  <path d="M 150 232 Q 230 230, 300 215 Q 380 198, 460 226 Q 510 240, 540 220" fill="none" stroke="#d97706" stroke-width="2"/>
  <text x="544" y="223" font-size="9" fill="#d97706">→ 첨도, 엔트로피</text>
  <text x="20" y="300" font-size="11" font-weight="700" fill="#64748b">εₜ (잡음 형상)</text>
  <line x1="150" y1="296" x2="540" y2="296" stroke="#cbd5e1" stroke-width="1"/>
  <g stroke="#64748b" stroke-width="1.2">
    <line x1="165" y1="296" x2="165" y2="284"/><line x1="195" y1="296" x2="195" y2="305"/><line x1="225" y1="296" x2="225" y2="281"/>
    <line x1="255" y1="296" x2="255" y2="302"/><line x1="285" y1="296" x2="285" y2="288"/><line x1="315" y1="296" x2="315" y2="278"/>
    <line x1="345" y1="296" x2="345" y2="306"/><line x1="375" y1="296" x2="375" y2="287"/><line x1="405" y1="296" x2="405" y2="300"/>
    <line x1="435" y1="296" x2="435" y2="282"/><line x1="465" y1="296" x2="465" y2="304"/><line x1="495" y1="296" x2="495" y2="290"/><line x1="525" y1="296" x2="525" y2="283"/>
  </g>
  <text x="544" y="299" font-size="9" fill="#64748b">→ 왜도, 첨도</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">확률과정 관점: 움직이는 수준 위에 움직이는 변동성이 곱해지고, 형상을 가진 잡음이 얹혀 관측 시계열이 된다. 각 층이 서로 다른 피처 패밀리에 대응한다.</figcaption>
</figure>

참조서가 신중히 짚는 실무적 핵심: **비정상성은 걸러낼 골칫거리가 아니라
신호 자체다.** 소비 시계열은 거의 정상이 아니다(소득이 오르면 평균이
올라가고, 생활 변화가 분산을 흔든다). 그리고 평균의 이동, 두꺼워지는
꼬리, 무너지는 자기상관은 정확히 추천이 대응해야 할 행동 변화다. 목표는
*비정상성을 지우는 게 아니라 정량화* 하는 것이다.

## 자기상관과 정상성

순서 불변 집계는 여기서 죽는다. 자기상관은 순서에 귀 기울이는 첫 피처다.
자기상관함수는 시계열이 시차 $h$ 에서 자기 과거를 얼마나 *기억* 하는지를
측정한다 — $X_t$ 와 $X_{t+h}$ 를 두 변수로 놓은, 피어슨 상관의 시계열
사촌이다. 모듈이 계산하는 표본 추정치:

$$ \rho_k = \frac{\sum_{t} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_t (x_t - \bar{x})^2} $$

두 시차가 피처로 추출된다. **$\rho_1$** (`ar_lag1_autocorr`)은 소비 모멘텀
— "어제 소비가 오늘을 예측했는가"; 0.3–0.5 는 보통의 관성, 0.7 초과는 강한
연속 소비(여행, 폭소비). **$\rho_7$** (`ar_lag7_autocorr`)은 주간 계절성 —
매주 토요일이 지난 토요일과 닮으면 높다.

원시 ACF 는 *간접* 상관을 섞는다: $X_t$ 와 $X_{t+2}$ 가 $X_{t+1}$ 을 통해서만
상관할 수 있다. **편** 자기상관은 중간 시차를 제거하고, 모듈은 Yule–Walker
근사로 lag-1 편자기상관을 $\phi_{1,1} \approx c_1/c_0$ 로 추정한다. 끝으로
**Ljung–Box** 통계량이 이 구조가 *진짜인지* 검정한다:

$$ Q_{\mathrm{LB}} = n(n+2)\,\frac{r_1^2}{\,n-1\,} $$

귀무가설 "자기상관 없음" 하에서 큰 $Q_{\mathrm{LB}}$ 는 시계열이 랜덤이
*아니라는* 뜻이고 — 이는 그 자체로 유용한 메타 피처다: 예측 가능한 고객은
더 높은 추천 신뢰도를 받는다.

이것이 `ar_*` 블록이다: **4D** — lag-1, lag-7, 편 lag-1, Ljung–Box.

> **수식 직관.** $\rho_k$ 는 두 변수가 *같은* 시계열을 $k$ 만큼 어긋나게
> 놓은 피어슨 상관이다. 분자는 "$t$ 일과 $t{+}k$ 일이 함께 높고 함께 낮은
> 경향이 있는가"를 묻고, 분모는 시계열 자신의 분산으로 답을 $[-1,1]$ 로
> 재척도한다. 주간 쇼퍼는 $k=7$ 에서, 모멘텀 소비자는 $k=1$ 에서 불이
> 켜지고, 순수 잡음은 모든 시차에서 0 근처에 머문다.

## 주파수 도메인 — FFT 피처

같은 2차 구조를 두 번째 관점에서도 볼 수 있다. 시간 도메인은 "시점 $t$ 의
값이 얼마인가"를, 주파수 도메인은 "주파수 $f$ 의 진동이 얼마나 강한가"를
묻는다 — 그리고 Fourier 변환이 둘을 손실 없이 오간다. 모듈이 ACF 블록과
FFT 블록을 *둘 다* 두는 깊은 이유: 자기공분산과 스펙트럴 밀도는 Fourier
쌍(Wiener–Khinchin)으로 수학적으로 동등하지만, 신경망이 배우기에는 한
표현이 다른 표현보다 쉬울 수 있다. 주간 주기성은 lag-7 ACF 로도, $f = 1/7
\approx 0.143$ cycles/day 의 피크로도 나타난다.

금액 시퀀스를 정규화한 뒤 모듈은 실수 FFT 를 돌려 파워 스펙트럼 $P(f) =
|X(f)|^2$ 을 만들고, 거기서 네 피처를 읽는다. **스펙트럼 중심** 은 에너지
가중 평균 주파수 — 스펙트럼의 무게 중심이다:

$$ f_{\mathrm{centroid}} = \frac{\sum_i f_i\,|X(f_i)|^2}{\sum_i |X(f_i)|^2} $$

결과가 0.033 cycles/day 이면 대표 주기가 ~30일(월), 0.143 이면 ~7일(주)이다.
그 중심 주위로 **스펙트럼 대역폭** $\sqrt{\sum_i (f_i -
f_{\mathrm{centroid}})^2 P(f_i)/\sum_i P(f_i)}$ 이 에너지가 얼마나 퍼졌는지
측정한다 — 단일 지배 주기면 좁고, 여러 주기가 엉키면 넓다. **주요 주파수**
는 DC 항을 뺀 최대 피크 빈이다. 그리고 **스펙트럼 엔트로피** 는 스펙트럼을
분포로 정규화한 뒤 Shannon 엔트로피를 취한다:

$$ H_{\mathrm{spectral}} = -\sum_i p_i \log p_i, \qquad p_i = \frac{|X(f_i)|^2}{\sum_j |X(f_j)|^2} $$

낮은 스펙트럼 엔트로피는 에너지가 한 주파수에 집중 — 강하고 규칙적인
주기. 높으면 여러 주파수에 흩어짐 — 불규칙하고 비주기적인 소비. 이것이 `freq_*`
블록이다: **4D**.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="260" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">파워 스펙트럼 P(f) — 주간 소비자</text>
  <line x1="60" y1="210" x2="520" y2="210" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="210" x2="60" y2="50" stroke="#64748b" stroke-width="1.2"/>
  <text x="290" y="244" text-anchor="middle" font-size="11" fill="#1e3a5f">주파수 f (cycles/day)</text>
  <text x="24" y="130" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 24 130)">파워 |X(f)|²</text>
  <g stroke="#4f46e5" stroke-width="2">
    <line x1="92" y1="210" x2="92" y2="178"/><line x1="120" y1="210" x2="120" y2="190"/>
    <line x1="148" y1="210" x2="148" y2="200"/><line x1="176" y1="210" x2="176" y2="196"/>
    <line x1="204" y1="210" x2="204" y2="186"/>
    <line x1="232" y1="210" x2="232" y2="84"/>
    <line x1="260" y1="210" x2="260" y2="188"/><line x1="288" y1="210" x2="288" y2="198"/>
    <line x1="316" y1="210" x2="316" y2="194"/><line x1="344" y1="210" x2="344" y2="201"/>
    <line x1="372" y1="210" x2="372" y2="196"/><line x1="400" y1="210" x2="400" y2="203"/>
    <line x1="428" y1="210" x2="428" y2="199"/><line x1="456" y1="210" x2="456" y2="204"/>
    <line x1="484" y1="210" x2="484" y2="200"/>
  </g>
  <g fill="#4f46e5"><circle cx="232" cy="84" r="4"/></g>
  <line x1="232" y1="84" x2="300" y2="64" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="3 3"/>
  <text x="304" y="62" font-size="10.5" fill="#d97706" font-weight="700">주요 피크 — f ≈ 0.143 (7일)</text>
  <line x1="255" y1="50" x2="255" y2="210" stroke="#0d9488" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="262" y="120" font-size="10" fill="#0d9488" font-weight="700">f_centroid</text>
  <line x1="200" y1="224" x2="312" y2="224" stroke="#e11d48" stroke-width="1"/>
  <line x1="200" y1="220" x2="200" y2="228" stroke="#e11d48" stroke-width="1"/>
  <line x1="312" y1="220" x2="312" y2="228" stroke="#e11d48" stroke-width="1"/>
  <text x="256" y="237" text-anchor="middle" font-size="9.5" fill="#e11d48">대역폭 (좁음 → 규칙적)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">에너지가 주간 주파수의 날카로운 한 피크에 집중: 낮은 스펙트럼 엔트로피, 좁은 대역폭, 주요 주기 근처의 중심. 교과서적 규칙 소비자.</figcaption>
</figure>

## 변동성과 분포 형상

잡음 항 $\epsilon_t$ 는 형상을 가지고, 그 형상은 분산을 넘어선 정보를
담는다. 모듈은 표준화된 두 중심 적률로 이를 측정한다. **왜도**
$\gamma_1 = \mu_3/\sigma^3$ 은 비대칭 — 양수면 오른쪽 꼬리(대부분 소액,
가끔 대액), 전형적 리테일 시그니처다. **초과 첨도**
$\gamma_2 = \mu_4/\sigma^4 - 3$ 은 꼬리 두께이고, $-3$ 이 정규분포를 0 에
고정한다:

$$ \gamma_2 = \frac{\tfrac1n \sum_i (x_i - \bar{x})^4}{\left(\tfrac1n \sum_i (x_i - \bar{x})^2\right)^2} - 3 $$

양의 초과 첨도는 극단적 소비 이벤트 — 대형 구매, 해외여행 — 가 정규분포가
예측하는 것보다 *자주* 옴을 뜻한다. 두 적률 모두 거래 *금액* 과 거래
*간격* 두 시퀀스에 대해 계산되어 `dist_*` 블록의 **4D** 를 이룬다:
금액-왜도, 금액-첨도, 간격-왜도, 간격-첨도.

왜 분산이 아니라 첨도인가? **변동성 군집** 때문이다 — 큰 변동이 큰 변동을
잇는 금융적 규칙성. lag-1 자기상관에는 잘 안 나타나지만(부호가 상쇄)
두꺼운 꼬리와 *절대값* 의 자기상관에 깨끗이 드러난다. 두꺼운 꼬리의
고객은 고가 상품 추천이 정당화되는 고객이다.

## 복잡도 — 예측 가능성 계기로서의 엔트로피

마지막 렌즈는 메타 질문을 던진다: *이 고객은 애초에 얼마나 예측하기
어려운가?* 세 엔트로피 측도가 답하며, 모두 같은 아이디어 위에 있다 —
시계열을 패턴 시퀀스로 바꾸고, 패턴 빈도를 확률 분포로 보고, Shannon
엔트로피를 취한다. 이들이 피처가 되는 이유는 불확실성 자체가 정보이기
때문이다: 높은 엔트로피 고객은 추천 신뢰도를 낮추거나 exploration 을
강화할 수 있다.

**Approximate Entropy** 는 길이 $m$ 부분 패턴이 길이 $(m{+}1)$ 대비 얼마나
자주 반복되는지를 비교한다($m=2$, 허용오차 $r = 0.2\sigma$):

$$ \mathrm{ApEn}(m,r) = \Phi^m(r) - \Phi^{m+1}(r) $$

차이가 작으면(ApEn 낮음) 확장해도 패턴이 반복 — 규칙적. 크면 깨짐 —
불규칙. **Sample Entropy** $\mathrm{SampEn} = -\ln(A/B)$ 는 자기 매칭을
제외해 짧은 시퀀스에서도 안정적인 편향 보정판이다. **Permutation
Entropy** 는 크기를 완전히 무시하고 연속 $d=3$ 값의 *순서* 패턴만 보며,
$\ln(d!)$ 로 $[0,1]$ 정규화된다:

$$ H_{\mathrm{perm}} = \frac{-\sum_{\pi \in \Pi} p(\pi)\,\ln p(\pi)}{\ln(d!)} $$

Permutation Entropy 에는 조용한 강점이 있다: 오르내림 *순서* 만 읽으므로
log1p 와 표준화에 불변이고 이상치에 강건하다 — 전처리 파이프라인이 데이터를
건드리기 전후가 같은 값이다. 이것이 `complex_*` 블록이다: **3D**. (참고:
ApEn 과 SampEn 은 $O(n^2)$ 이므로 구현은 시퀀스를 `MAX_ENTROPY_SEQ_LEN = 300`
으로 캡해 메모리 폭발을 막는다. Permutation Entropy 는 $O(n)$ 이다.)

## 피처 셋과, 어디에 들어가는가

다섯 패밀리를 합치면 수작업 **18D** 가 되고, 프로젝트는 이를 `lnn_*` 로
라벨링한다(이름이 빚는 오해 — 이들은 신호처리 통계이지 모델 레이어에 사는
ODE 기반 Liquid Neural Network 가 *아니다*). 그 옆에 학습된 50D Mamba
임베딩이 놓여 총 **68D**:

| 패밀리 | 접두 | 차원 | 포착 대상 |
| --- | --- | --- | --- |
| 분포 형상 | `dist_*` | 4D | 금액·간격의 왜도/첨도 — 꼬리 비대칭, 극단성 |
| 주파수 | `freq_*` | 4D | 주요 주파수, 스펙트럼 엔트로피/중심/대역폭 — 주기성 |
| 변환점 | `changepoint_*` | 3D | CUSUM 개수, 최대 이동 크기, 평균 구간 길이 — 레짐 변화 |
| 자기상관 | `ar_*` | 4D | lag-1, lag-7, 편 lag-1, Ljung–Box — 기억과 계절성 |
| 복잡도 | `complex_*` | 3D | ApEn, SampEn, 순열 엔트로피 — 예측 가능성 |
| **수작업 합계** | `lnn_*` | **18D** | 네 렌즈를 명시화 |
| Mamba 임베딩 | `mamba_temporal_*` | 50D | SSM 잠재, 256D → PCA → 50D, 학습됨 |
| **모듈 합계** | — | **68D** | 순서를 보는 피처 |

변환점 블록은 마지막으로 짚어 둘 만하다. 유일한 순수 *시간 도메인* 레짐
탐지기이기 때문이다. **CUSUM** 스캔을 돌린다 — 평균으로부터의 이탈을
누적하다 평균 아래로 돌아오면 0 으로 리셋하고, $h = 2\sigma$ 를 넘으면
변환점으로 표시:

$$ S_k^{+} = \max\!\big(0,\ S_{k-1}^{+} + (x_k - \bar{x})\big) $$

— 하락 방향은 대칭형 $S_k^{-}$ 가 감시한다 — 그리고 변환점 개수, 전후
평균의 최대 이동, 평균 구간 길이를 내보낸다.
최근의 큰 `max_shift_magnitude` 는 생활 변화(이사, 새 직장)를 표시하고 추천
상품군을 전환할 근거가 된다.

68D 는 어디로 가는가? **734D 메인 텐서**(644D normalized + 90D raw
power-law)로, PLE 가 학습되기 *전* 에 *오프라인* 으로 사전 계산된다 —
Mamba 50D 는 159D 도메인 그룹의 일부, LNN 18D 는 27D 모델 파생 그룹의
일부. 참조서가 짚듯 이것은 텐서 전체에서 시간적 순서를 쓰는 *유일한*
그룹이다. 빼내면 모델은 고객이 시간에 따라 어떻게 변하는지에 눈이 먼다.
스키마 전체에서 차원당 정보 밀도가 가장 높다.

> **계약은 그 뒤로 갱신됐다.** 위 734D 는 V1 피처 계약이다. 프로젝트는
> 2026-07-02 자로 V2 strict 계약으로 전환했고, 운영 입력 폭은 **4035D** 다 —
> 734D 는 폐기된 게 아니라 V2 의 _공유 베이스 8그룹_ 으로 남고, 여기에
> lag/rolling/product 계열 3301D 가 덧붙어 4035D 가 된다.

참조서가 솔직히 밝히는 실무적 단서 하나: 기본 LIVE 경로
(`LNN_FAST_SQL_MODE=1`)에서 18D 는 DuckDB 집계 *proxy* 로 생성된다 —
`freq_*` 는 거래수/간격 통계가 되고, 일부 `ar_*` 는 상수 0, 변환점은 단일
패스 `|amount − prev| > 2σ` 카운트가 된다. 위에 서술한 정밀 FFT / 반복형
CUSUM / ApEn-SampEn-PE 구현은 `LNN_FAST_SQL_MODE=0` 인 Python 경로에서만
동작한다. 정직한 엔지니어링: 우아한 버전과 빠른 버전은 다른 코드이고,
당신의 숫자가 어느 쪽에서 나왔는지는 알아야 한다.

## 여기서 멈추는 이유

합계, 평균, 최대값이 소비 이력이 가진 유일한 것 — 순서 — 를 버린다는
불편함에서 출발했다. 시계열을 수준, 변동성, 잡음으로 나누는 확률과정으로
읽었고(그리고 흔히 기대하는 STL/HP 분해는 한 모듈 건너 Economics 에
산다는 점을 표시했다), 네 렌즈를 따라갔다: 시간 도메인의 자기상관과 정상성,
주파수 도메인의 FFT 스펙트럼 피처, 분포 형상의 왜도와 첨도, 복잡도의 세
엔트로피 — 그리고 이들이 수작업 18D 로 모여 학습된 50D Mamba 임베딩 옆에서
734D 텐서 안의 68D 를 이루는 것을 봤다.

일부러 미뤄둔 것은 *교차 학제* 피처 그룹이다 — 신호처리에서 빌리기를
멈추고 화학, 역학(전염병학), 범죄학, 그리고 파동 물리학에서 빌리기
시작하는 블록: 소비 가속도로 읽는 반응 속도론, 카테고리 간 SIR 전염,
일상활동이론의 버스트성, 주파수 스펙트럼의 두 번째 독해. 소비 추천
시스템이 왜 신용카드를 들어본 적 없는 학문에 손을 뻗는가가 다음 편
**MULTI-1** 의 주제다.
