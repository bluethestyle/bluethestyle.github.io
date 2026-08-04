---
title: "[Study Thread] GMM-1 — 소프트 클러스터링: 피처가 되는 책임(responsibility)과 그 뒤의 가우시안 혼합"
date: 2026-06-07 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, gmm, mixture-model, clustering, em, features]
lang: ko
excerpt: "GMM 피처 서브스레드 시작 — 단 하나의 클러스터 ID 가 신호 대부분을 버리는 이유, 가우시안 혼합이 각 고객을 20개 행동 원형(archetype)에 걸친 소프트 소속으로 나누는 방식, EM 의 책임(responsibility)이 실제로 계산하는 것, BIC 가 K=20 을 고르는 과정, 그리고 모델의 소프트 라우팅 바스켓에 들어가는 22D 벡터. 온프렘 참조서의 실제 입출력 구성과 함께."
series: study-thread
part: 20
alt_lang: /2026/06/07/gmm-soft-clustering-en/
next_title: "TimeSeries-1 — 시계열 피처: 원시 거래 시퀀스에서 고정된 행동 지문으로"
next_desc: "가변 길이의 날짜별 거래 흐름을 어떻게 안정적인 고객별 시계열 피처로 바꾸는가: 추세와 계절성 분해, 자기상관 구조, 변화점(change-point) 신호, 그리고 손수 짠 롤링 윈도우보다 상태공간 관점이 나은 이유."
next_status: draft
---

*"Study Thread" 시리즈의 GMM(가우시안 혼합 모델) 피처 서브스레드 1편.
이번 편부터 영문/국문 병렬로, 온프렘 추천 모델에 입력되는 오프라인
피처 블록 중 하나인 GMM 소프트 클러스터링 피처 모듈을 정리한다. 출처는
온프렘 프로젝트 `기술참조서/GMM_피처_기술_참조서` 이고, 전체 PDF 는
서브스레드 마지막 편에 첨부한다. TDA 서브스레드가 Expert 가 어떤 형태를
읽는가를 물었다면, 이번 편은 더 단순하고 운영적인 질문을 던진다 —
고객을 군집화할 때, 후속 모델에게 정확히 무엇을 건네야 하는가? 라벨인가,
아니면 분포인가? GMM 의 답은 후자이고, 그 차이가 실제로 정보를
담는다는 게 드러난다.*

> **왜 소프트인가, 한 줄로.** 하드 클러스터 ID 는 숫자 하나다 — "고객
> A 는 유형 3 이다." GMM 소속은 분포 전체다 — "고객 A 는 유형 3 에 0.65,
> 유형 7 에 0.20, 유형 12 에 0.10, …" — 꼭짓점이 아니라 20차원 확률
> 심플렉스 위의 한 점이다. 이 프로젝트에서 그 벡터는 **소프트 소속 20D +
> cluster_id + 엔트로피 = 22D** 이고, **40D** 의 사전 계산 고객 피처에
> **K=20, full 공분산** 혼합을 EM 으로 적합해 얻는다. 참조서는
> 직설적이다 — 하드 라벨은 약 1비트, 소프트 벡터는 최대 약 4.32비트
> ($\log_2 20$). 그 추가 정보가 핵심 전부다.

## 하드 라벨은 경계를 버린다

고객을 군집화할 때 반사적으로 각자를 한 그룹에 배정하고 넘어간다.
K-means 가 정확히 그렇다 — 모든 고객이 가장 가까운 중심을 받고 끝. 단순하고
빠르지만, 가장 다루기 어려운 고객에 대해 가장 중요한 한 가지를 버린다.
바로 *그들의 모호함* 이다.

"식비 중심" 그룹과 "여행 중심" 그룹 사이 한가운데에 있는 고객은 임의로
한쪽으로 떠밀린다. 두 원형으로 똑같이 잘 설명되는 경계 사례라는 사실은
라벨이 쓰이는 순간 지워진다. 그리고 경계 고객은 반올림 오차가 아니다 —
클러스터가 심하게 겹치는 금융 행동 데이터에서는 전체 고객의 상당 비중을
차지한다.

참조서는 K-means 의 세 가지 구조적 한계 — 하드 할당 자체, 등방적 거리,
확률 해석의 부재 — 를 짚고, 두 방식을 일곱 가지 관점에서 대비한다.

| 관점 | K-means (하드) | GMM (소프트) |
| --- | --- | --- |
| 할당 방식 | 원-핫 (20개 중 1개만 1) | 확률 벡터 (20D, 합=1.0) |
| 피처 정보량 | 1비트 (어느 클러스터인가) | 약 4.32비트 (K=20 엔트로피 상한) |
| 경계 고객 | 임의 할당, 불안정 | 인접 클러스터로 확률 분산 |
| 클러스터 형태 | 구형 (등방적) | 타원체 (공분산으로 형태 자유) |
| 신뢰도 신호 | 없음 | 엔트로피로 정량화 |
| 콜드스타트 | 최근접 중심 (편향) | 균등 분포 = 무편향 중립 |
| 그래디언트 호환 | 불연속 (원-핫) | 연속 (미분 가능) |

이 중 가장 깊은 것은 첫 줄이다. 소프트 할당은 꼭짓점의 선택이 아니라
*심플렉스 위의 좌표* 이며, 연속 좌표는 원-핫 양자화가 영구히 파괴하는
정보를 보존한다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="250" fill="#f8fafc" rx="8"/>
  <text x="140" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">K-means — 하드, 구형</text>
  <circle cx="100" cy="120" r="45" fill="#64748b18" stroke="#64748b" stroke-width="1.2"/>
  <circle cx="185" cy="135" r="45" fill="#64748b18" stroke="#64748b" stroke-width="1.2"/>
  <line x1="143" y1="70" x2="143" y2="190" stroke="#e11d48" stroke-width="1.4" stroke-dasharray="5 4"/>
  <g fill="#64748b"><circle cx="85" cy="110" r="3.5"/><circle cx="105" cy="100" r="3.5"/><circle cx="95" cy="135" r="3.5"/><circle cx="200" cy="125" r="3.5"/><circle cx="180" cy="150" r="3.5"/><circle cx="205" cy="145" r="3.5"/></g>
  <circle cx="140" cy="120" r="5" fill="#d97706"/>
  <text x="140" y="212" text-anchor="middle" font-size="10" fill="#e11d48">경계 → 한쪽으로 강제</text>
  <line x1="280" y1="40" x2="280" y2="210" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">GMM — 소프트, 타원체</text>
  <ellipse cx="380" cy="115" rx="52" ry="30" transform="rotate(-18 380 115)" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <ellipse cx="470" cy="140" rx="46" ry="28" transform="rotate(24 470 140)" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <g fill="#0d9488"><circle cx="360" cy="105" r="3.5"/><circle cx="385" cy="98" r="3.5"/><circle cx="372" cy="128" r="3.5"/></g>
  <g fill="#4f46e5"><circle cx="478" cy="132" r="3.5"/><circle cx="462" cy="152" r="3.5"/><circle cx="490" cy="145" r="3.5"/></g>
  <circle cx="425" cy="125" r="5" fill="#d97706"/>
  <text x="425" y="205" text-anchor="middle" font-size="10" fill="#d97706">γ = (0.55, 0.45) — 양쪽</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">K-means 는 모든 점을 하드 경계선 너머 단일 구형 클러스터로 강제하고, GMM 은 타원체 클러스터를 주며 경계 고객이 양쪽에 부분 소속을 갖게 한다.</figcaption>
</figure>

## 가우시안 혼합

소프트 할당 뒤의 모델은 가우시안 혼합이다. 각 고객의 피처 벡터가 다음
과정으로 생성됐다고 가정한다 — 먼저 $K$개 행동 유형 중 하나를 *고르고*
(유형 $k$ 를 사전 확률 $\pi_k$ 로), 그 유형의 다변량 가우시안
$\mathcal{N}(\mu_k, \Sigma_k)$ 에서 *뽑는다*. 관측된 고객 $x$ 의 밀도는
모든 유형에 대한 가중 합이다.

$$ p(x) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \pi_k \ge 0,\ \sum_k \pi_k = 1 $$

각 성분은 완전한 다변량 정규분포다.

$$ \mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{D/2}\,|\Sigma_k|^{1/2}} \exp\!\left(-\tfrac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)\right) $$

여기서 $D = 40$ 이다. 지수부의 이차형식 $(x-\mu_k)^\top
\Sigma_k^{-1}(x-\mu_k)$ 은 *마할라노비스 거리* 의 제곱이다 — 유클리드와
달리 $\Sigma_k^{-1}$ 을 통해 피처 간 상관과 스케일을 보정하는 거리다.
함께 움직이는 두 피처(예: 거래액과 빈도)는 이중으로 계산되지 않으며, 둘
다 높은 고객은 독립 가정 하에서보다 중심에 *더 가깝게* 평가된다. 이것이
프로젝트가 spherical 이 아니라 `covariance_type="full"` 을 쓰는 정확한
이유다 — 각 클러스터가 공이 아니라 기울어진 타원체가 될 수 있다.

> **역사적 배경.** 혼합 아이디어는 그것을 적합하는 알고리즘보다 거의 한
> 세기 앞선다 — Karl Pearson 이 1894년 모멘트법으로 2-성분 정규 혼합을
> 분해했다. 실용적인 엔진은 1977년에야 등장했다. Dempster, Laird, Rubin 이
> *Expectation–Maximization* 을 잠재 변수가 있는 최대우도 추정의 일반
> 레시피로 정식화했다. "이 점을 어느 성분이 생성했는가" 변수가 바로 그런
> 잠재 변수이며, EM 덕에 100만 고객에 20-성분, 40차원, full 공분산
> 혼합을 적합하는 일이 다루기 불가능한 문제가 아니라 수 분짜리 작업이
> 된다.

## EM — 혼합을 적합하기

어느 유형이 특정 고객을 생성했는지는 관측되지 않는다. 그 할당은 잠재
변수다. EM 은 로그 우도 $\ln p(X \mid \Theta)$($\Theta = \{\pi_k, \mu_k,
\Sigma_k\}$)를 결코 감소시키지 않는 것이 증명된 두 단계를 번갈아 수행해
이를 다룬다.

**E-step.** 현재 파라미터가 주어졌을 때, 각 고객의 *책임
(responsibility)* — 고객 $x_n$ 이 유형 $k$ 에서 왔을 사후 확률 — 을
계산한다.

$$ \gamma_{nk} = \frac{\pi_k\,\mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(x_n \mid \mu_j, \Sigma_j)} $$

베이즈 정리 그 자체다 — 분자는 *유형 $k$ 의 사전 비율 × 유형 $k$ 하에서
$x_n$ 의 우도*, 분모는 모든 유형에 대해 정규화하여 책임의 합이 1 이
되게 한다. **이 $\gamma_{nk}$ 가 곧 피처다** — 그대로 출력 컬럼
`cluster_prob_00` 부터 `cluster_prob_19` 가 된다.

**M-step.** 소프트 책임을 부분 소속으로 보고 파라미터를 재추정한다.
유효 샘플 수 $N_k = \sum_n \gamma_{nk}$ (실수, "약 1,251명 규모")로:

$$ \pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k}\sum_n \gamma_{nk}\,x_n, \quad \Sigma_k = \frac{1}{N_k}\sum_n \gamma_{nk}\,(x_n-\mu_k)(x_n-\mu_k)^\top $$

각 갱신은 책임 가중 평균이다 — 확신이 높은 고객일수록 그 유형의 중심과
형태를 더 강하게 끌어당긴다. 두 단계는 로그 우도 변화가 $10^{-3}$ 미만이
되거나 `max_iter=200` 에 도달할 때까지 번갈아 돈다. 각 $\Sigma_k$ 의
대각에 소량의 정규화 `reg_covar=1e-1` 을 더해 양정치성을 보장한다(그래야
Cholesky 인자가 존재하고 $\Sigma_k^{-1}$ 가 안정적이다).

<figure style="margin:24px auto;max-width:540px;">
<svg viewBox="0 0 540 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="540" height="240" fill="#f8fafc" rx="8"/>
  <text x="270" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">EM 루프</text>
  <rect x="60" y="70" width="170" height="90" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2"/>
  <text x="145" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">E-step</text>
  <text x="145" y="120" text-anchor="middle" font-size="10" fill="#64748b">Θ 고정, γₙₖ 계산</text>
  <text x="145" y="138" text-anchor="middle" font-size="10" fill="#64748b">(책임)</text>
  <rect x="310" y="70" width="170" height="90" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>
  <text x="395" y="98" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">M-step</text>
  <text x="395" y="120" text-anchor="middle" font-size="10" fill="#64748b">γₙₖ 고정, Θ 갱신</text>
  <text x="395" y="138" text-anchor="middle" font-size="10" fill="#64748b">(πₖ, μₖ, Σₖ)</text>
  <path d="M 230 100 L 305 100" fill="none" stroke="#94a3b8" stroke-width="1.6"/>
  <polygon points="305,100 296,95 296,105" fill="#94a3b8"/>
  <path d="M 310 135 L 235 135" fill="none" stroke="#94a3b8" stroke-width="1.6"/>
  <polygon points="235,135 244,130 244,140" fill="#94a3b8"/>
  <text x="270" y="195" text-anchor="middle" font-size="10.5" fill="#1e3a5f">Δ ln L &lt; 10⁻³  또는  max_iter = 200 까지 반복</text>
  <text x="270" y="214" text-anchor="middle" font-size="9.5" fill="#94a3b8">로그 우도는 결코 감소하지 않음 · reg_covar = 1e-1 로 Σₖ 양정치 유지</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">EM 은 번갈아 돈다 — E-step 이 현재 가우시안에서 소프트 소속을 다시 계산하고, M-step 이 소속 가중 통계로 가우시안을 다시 적합한다. 매 패스마다 우도가 오른다.</figcaption>
</figure>

## 피처로 읽는 책임

$\gamma_{nk}$ 가 후속 모델에게 실제로 무엇을 주는지 잠시 짚어 볼 가치가
있다. 한 고객에 대해 그것은 확률 심플렉스 위의 20-벡터다 — 원형에 대한
*볼록 프로필*. 참조서의 예가 가장 깔끔한 표현이다 — 모델에게

> "고객 A 는 유형 3 이다"

라고 말하는 것은

> "고객 A 는 유형 3 에 0.65, 유형 7 에 0.20, 유형 12 에 0.10"

라고 말하는 것보다 정보량이 엄밀히 적다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">한 고객의 책임 벡터 γ (20D, 합=1.0)</text>
  <line x1="40" y1="180" x2="520" y2="180" stroke="#64748b" stroke-width="1"/>
  <g>
    <rect x="46"  y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="70"  y="166" width="18" height="14"  fill="#cbd5e1"/>
    <rect x="94"  y="80"  width="18" height="100" fill="#0d9488"/>
    <rect x="118" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="142" y="170" width="18" height="10"  fill="#cbd5e1"/>
    <rect x="166" y="160" width="18" height="20"  fill="#cbd5e1"/>
    <rect x="190" y="148" width="18" height="32"  fill="#4f46e5"/>
    <rect x="214" y="175" width="18" height="5"   fill="#cbd5e1"/>
    <rect x="238" y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="262" y="168" width="18" height="12"  fill="#cbd5e1"/>
    <rect x="286" y="160" width="18" height="20"  fill="#d97706"/>
    <rect x="310" y="176" width="18" height="4"   fill="#cbd5e1"/>
    <rect x="334" y="173" width="18" height="7"   fill="#cbd5e1"/>
    <rect x="358" y="171" width="18" height="9"   fill="#cbd5e1"/>
    <rect x="382" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="406" y="170" width="18" height="10"  fill="#cbd5e1"/>
    <rect x="430" y="175" width="18" height="5"   fill="#cbd5e1"/>
    <rect x="454" y="172" width="18" height="8"   fill="#cbd5e1"/>
    <rect x="478" y="174" width="18" height="6"   fill="#cbd5e1"/>
    <rect x="502" y="176" width="18" height="4"   fill="#cbd5e1"/>
  </g>
  <text x="103" y="72"  text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">0.65</text>
  <text x="103" y="196" text-anchor="middle" font-size="9" fill="#64748b">유형 3</text>
  <text x="199" y="140" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">0.20</text>
  <text x="199" y="196" text-anchor="middle" font-size="9" fill="#64748b">유형 7</text>
  <text x="295" y="152" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">0.10</text>
  <text x="295" y="196" text-anchor="middle" font-size="9" fill="#64748b">유형 12</text>
  <text x="430" y="196" text-anchor="middle" font-size="9" fill="#94a3b8">나머지 17개 유형 합 ≈ 0.05</text>
  <text x="36" y="84" text-anchor="end" font-size="9" fill="#94a3b8">γ</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">한 고객의 소프트 소속: 질량 대부분이 유형 3 에, 유의미한 2차 질량이 7 과 12 에, 나머지 전체에 얇고 긴 꼬리. 하드 라벨은 가장 높은 막대 하나만 남기고 나머지를 버린다.</figcaption>
</figure>

후자가 후속 소프트 라우팅이 직접 소비하는 것이다. 그리고 $\gamma_{nk}$ 는
입력에 대한 매끄러운 softmax 유사 함수라서 — `argmin` 도, 불연속도 없다 —
그래디언트가 원칙적으로 통과할 수 있고, 오늘은 GMM 이 사전 학습된 고정
피처 추출기로 쓰이더라도 향후 end-to-end 파인튜닝의 문은 열려 있다.

20개 확률과 함께, 모듈은 할당이 *얼마나 확실한지* 를 요약하는 스칼라를
내보낸다 — 할당 엔트로피다.

$$ H_n = -\sum_{k=1}^{K} \gamma_{nk}\,\ln(\gamma_{nk} + \epsilon), \qquad \epsilon = 10^{-10} $$

$H_n \in [0, \ln K]$. $H_n = 0$ 이면 고객이 한 유형에 완전히 할당된
것이고, $H_n = \ln 20 \approx 2.996$ 이면 분포가 균등 — 분류 불가능한
고객이며, 이것이 바로 콜드스타트 신호다. 이 숫자 하나가 확신 없는
상황에서 추천기를 보수적으로 행동하게 해 주며, "확신이 없을 때 안전한
선택을 하라"는 운영 원칙과 정확히 부합한다.

> **수식 직관.** $\gamma_{nk}$ 를 베이즈가 믿음을 갱신하는 것으로 읽자.
> 고객을 보기 전, 그가 유형 $k$ 라는 믿음은 인구 비율 $\pi_k$ 다. 그의
> 40D 피처 벡터를 본 뒤, 그 벡터가 유형 $k$ 의 가우시안 하에서 얼마나
> 그럴듯한지를 곱하고 재정규화한다. 엔트로피 $H_n$ 은 그 결과 믿음이
> 얼마나 뾰족한지를 측정한다 — 한 유형에 솟은 스파이크는 낮은 엔트로피
> (확신), 평평한 분산은 높은 엔트로피(경계 또는 신규 고객). 어느 것도
> 휴리스틱이 아니다 — 사후 확률 그 자체와, 그 날카로움을 그대로 읽은
> 것이다.

## K 의 선택 — AIC 가 아니라 BIC

혼합은 성분을 더하면 언제나 더 잘 적합되므로, $K$ 는 복잡도 페널티에
맞서 골라야 한다. 시스템은 **베이지안 정보 기준(BIC)** 을 쓴다.

$$ \mathrm{BIC} = -2\ln\hat{L} + k\,\ln(n) $$

$\hat{L}$ 은 최대 우도, $k$ 는 자유 파라미터 수, $n$ 은 고객 수이며,
낮을수록 좋다. AIC ($-2\ln\hat{L} + 2k$) 가 아니라 BIC 를 의도적으로
쓴다 — $n$ 이 수십만~수백만이면 AIC 의 평평한 $2k$ 페널티는 클러스터를
지나치게 많이 통과시키지만, BIC 의 $k\ln(n)$ 항은 데이터에 비례해
과적합을 억누른다($n > 7$ 이면 AIC 보다 강하게 벌한다). `full` 공분산
에서 파라미터 수는 $O(K \cdot D^2)$ 로 늘고, $K=20$, $D=40$ 이면 공분산
파라미터만 약 $20 \times 40 \times 41 / 2 = 16{,}400$ 개이므로 페널티는
형식적인 게 아니다.

1회성 sweep (`analyze_optimal_k.py`) 으로 $K \in [5, 30]$ 을 탐색한
결과, $K = 20$ 이 BIC 최저점과 silhouette 최고점의 교차점으로 선택됐다.

| K | BIC | Silhouette | 비고 |
| --- | --- | --- | --- |
| 5 | 높음 | 0.35 | 과소 분할 — 이질적 고객 혼재 |
| 10 | 중간 | 0.38 | 기본 세분화 수준 |
| 15 | 중간–낮음 | 0.41 | 양호한 분리도 |
| **20** | **최저** | **0.42** | **BIC/silhouette 최적 교차점 — 채택** |
| 25 | 낮음 | 0.41 | BIC 소폭 개선이나 과적합 시작 |
| 30 | 낮음 | 0.39 | 빈 클러스터 발생, 공분산 축퇴 위험 |

silhouette 0.42 는 "대체로 양호한 분리"로, 클러스터 경계가 실제로
모호한 금융 고객 데이터에서는 합리적인 수치다. 이후 K 는 고정되고, 월간
BIC 모니터(`validate_k_range()`, $[5, 30]$)가 최적 K 가 현재 K 와
`K_CHANGE_THRESHOLD=3` 이상 벌어지면 경고를 로깅한다 — 실제 변경은
출력 차원을 바꾸므로 사람이 감독하는 수동 단계로 남는다.

## 피처 벡터 — 22D, 그리고 어디로 가는가

모듈 출력은 `DEFAULT_K + 2 = 22D` 로 고정이며, 구성은 다음과 같다.

| 피처 | 차원 | 의미 |
| --- | --- | --- |
| `cluster_prob_00` … `cluster_prob_19` | 20D | 클러스터별 소속 $\gamma_{nk}$, 합=1.0 |
| `cluster_id` | 1D | $\arg\max_k \gamma_{nk}$; 콜드스타트 → 20 (전용 unassigned id) |
| `cluster_entropy` | 1D | 할당 불확실성 $H_n$ |

입력은 40D 사전 계산 연속 벡터이며, Z-score 정규화된다(학습 시 $\mu_d,
\sigma_d$ 를 `gmm_norm_params.npz` 에 저장하고 추론 시 재사용해 학습-서빙
일관성 유지). DuckDB JOIN 으로 13개 소스를 통합한다 — **Base 13D**(RFM,
거래 통계, 시간 패턴, 카테고리 다양성), **Multi-source 10D**(예금, 신용,
투자, 디지털 참여), **Domain 10D**(선택적 — TDA persistence 엔트로피,
상전이, 소득 분해, 금융 행동), **Demographics 2D**, **Supplementary 5D**. Domain
피처 중 넷 — `permanent_income_avg`, `transitory_income_volatility`,
`income_elasticity`, `spending_risk` — 은 **Economics** 모듈에서 오므로
배치 순서는 *Economics → GMM → 734D 통합* 이 되어야 하며, DAG 에서
`ExternalTaskSensor` 로 보장된다.

> **계약은 그 뒤로 갱신됐다.** 위 734D 는 V1 피처 계약이다. 프로젝트는
> 2026-07-02 자로 V2 strict 계약으로 전환했고, 운영 입력 폭은 **4035D** 다 —
> 734D 는 폐기된 게 아니라 V2 의 _공유 베이스 8그룹_ 으로 남고, 여기에
> lag/rolling/product 계열 3301D 가 덧붙어 4035D 가 된다.

후속에서 20D 확률 블록은 `PLEClusterInput` 의 `cluster_probs` 필드로
전달되어 `GroupTaskExpertBasket` 의 **소프트 라우팅** 을 구동한다. 이는
태스크마다 20개 클러스터 서브헤드를 운용하고 소속 벡터로 혼합한다.

$$ o_\text{task} = \sum_{k=0}^{19} \gamma_{nk}\cdot h_{\text{subhead}_k}(z_\text{shared}) $$

이는 심플렉스 위의 *볼록 결합* 이다 — 출력은 20개 서브헤드 출력의
보간이며 항상 그 볼록 껍질 안에 놓이므로, 소프트 라우팅은 안정성을
공짜로 얻는다. 하드 `argmax` 는 꼭짓점 하나만 쓰고 나머지를 버리지만, 경계
고객은 인접 서브헤드를 혼합하고, 콜드스타트 고객(균등 소속)은 자연스럽게
20개 전체의 앙상블을 받는다. 참조서는 Mixture-of-Experts(Mixtral 류)와의
구조적 친연성을 짚는다 — 차이는 이 시스템이 희소 top-2 가 아니라 20개
헤드 전체에 대해 *밀집* 라우팅을 한다는 점이며, $K=20$ 규모에서 감당
가능하고 정보 손실이 없다.

## 여기서 멈추는 이유

경계 고객을 한 클러스터에 묶는 것이 그들에 대한 가장 유용한 정보를
파괴한다는, 하드 라벨에 대한 불편함에서 출발해 가우시안 혼합이
라벨을 20개 원형에 대한 소프트 소속으로 대체하는 방식을 봤다. 혼합 밀도와
그 full 공분산 마할라노비스 기하를 적었고, EM 이 책임과 가중 재적합 사이를
번갈아 도는 것을 지켜봤으며, E-step 책임을 실제 20D 피처로, 그리고 확신
없는 고객을 표시하는 엔트로피로 읽었고, BIC 가 $K=20$ 을 고르게 했으며,
그 결과 22D 벡터가 태스크마다 20개 서브헤드를 앙상블하는 소프트 라우팅
바스켓으로 흘러드는 것을 따라갔다.

우리가 *하지 않은* 것은 횡단면 스냅숏을 떠나는 일이다. GMM 은 한 순간을
읽는다 — 고객당 단일 40D 피처 벡터 — 그리고 그것이 어떤 유형들의 혼합을
닮았는지 묻는다. *순서* 에 대해서는 아무 말도 하지 않는다 — 소비가
추세인지, 계절적인지, 가속 중인지, 곧 꺾일지. 이는 전혀 다른 축이며,
다음 모듈은 그것을 날짜별 거래 흐름에서 직접 읽는다. 다음 편의 주제는
**시계열(TimeSeries) 피처** 다.
