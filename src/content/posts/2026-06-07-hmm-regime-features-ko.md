---
title: "[Study Thread] HMM-1 — 숨겨진 국면: 소비 뒤에 있는 상태를 읽다"
date: 2026-06-07 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, hmm, markov, regime, features, offline]
lang: ko
excerpt: "HMM 서브스레드 시작 — 고객의 관측 가능한 거래가 숨기고 있는 잠재 행동 상태, Hidden Markov Model 이 전이 행렬, Gaussian 방출, 초기 분포로 그 상태를 복원하는 방식, Baum-Welch 학습과 Viterbi 디코딩, 그리고 상태 사후 확률, 전이 통계, 체류 시간, 궤적 동역학으로 이루어진 48D 가 모델의 별도 입력 경로 어디에 안착하는가."
series: study-thread
part: 19
alt_lang: /2026/06/07/hmm-regime-features-en/
next_title: "GMM-1 — 소프트 클러스터링: 피처가 되는 책임(responsibility)과 그 뒤의 가우시안 혼합"
next_desc: "시계열 국면에서 한 시점 스냅샷으로: 20개 컴포넌트 가우시안 혼합이 고객을 단계가 아닌 유형으로 프로파일링하는 방식, 하드 레이블보다 소프트 책임도가 나은 이유, 그리고 22D 가 별도 경로가 아니라 734D 메인 텐서의 Domain 블록 안에 안착하는 구조."
next_status: draft
---

*"Study Thread" 시리즈의 HMM 서브스레드 1편. 이번 편부터 영문/국문
병렬로 HMM 기반 피처 — 별도 입력 경로를 통해 PLE 모델에 공급되는
오프라인 피처 모듈 — 를 정리한다. 출처는 온프렘 프로젝트
`기술참조서/HMM_피처_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막
편에 첨부한다. TDA 서브스레드가 행동의* 형태 *를 읽었다면, 이번
서브스레드는 행동의* 상태 *를 읽는다 — 고객이 점유한 숨겨진 국면과,
국면 사이를 움직이는 법칙. 모델은 그 국면을 직접 관측하지 않는다.
추론한다.*

> **이 모듈이 실제로 만드는 것.** 세 개의 Gaussian HMM 이 각 고객의
> 거래 시퀀스 위에서 병렬로 돈다 — **Journey**(5 상태),
> **Lifecycle**(5 상태), **Behavior**(6 상태) — 그리고 각각 **16D →
> 합 48D** 를, 734D 메인 텐서와 구별되는 *별도 입력* 경로로 내보낸다.
> 압축된 **5D 요약** 은 메인 텐서의 `model_derived` 블록 안에도 함께
> 탑승한다. 각 16D 는 `n_states` 개의 상태 사후 확률 + 메타 피처 + 6D
> ODE-dynamics 브리지다. 아래의 모든 내용은 참조서에 근거한다 — 실제
> 전이 행렬, `n_iter=200 / tol=1e-2` Baum-Welch 설정, 3D 관측 벡터.
> 지어낸 숫자는 없다.

## 왜 숫자가 아니라 잠재 상태인가

카드 거래 데이터에서 우리가 직접 관측할 수 있는 것은 빈약하다 — 얼마를, 몇 번,
몇 개의 서로 다른 카테고리에 썼는가. 그러나 그 숫자들 뒤에는 데이터가
결코 이름 붙이지 않는 *심리적* 상태가 있다. 이번 달 각각 10만 원을 쓴
두 고객이 전혀 다른 곳에 있을 수 있다 — 한 명은 새 서비스를 *탐색*
중이고, 한 명은 *이탈 직전* 마지막으로 소비한다. 집계는 동일하다.
상태는 정반대다.

"월 50만 원 이상이면 활성 고객" 같은 규칙은 자의적 임계값을 세우고
경계 근처의 고객을 모두 뭉개 버린다. Hidden Markov Model 은 그 하드 컷을
거부한다. *소프트 할당* — "80% 활성, 15% 성장, 5% 위험" — 을 반환하고,
그 확률 벡터 자체가 피처다. 모델은 볼 수 없는 것(행동 국면)을 볼 수
있는 것(거래)으로부터 확률적으로 복원한다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <text x="40" y="42" font-size="11" font-weight="700" fill="#4f46e5">숨겨진 상태 z</text>
  <g>
    <circle cx="120" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="120" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₁</text>
    <circle cx="260" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="260" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₂</text>
    <circle cx="400" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/><text x="400" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">z₃</text>
    <circle cx="500" cy="60" r="20" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4" stroke-dasharray="3 3"/><text x="500" y="64" text-anchor="middle" font-size="11" fill="#4f46e5">…</text>
  </g>
  <g stroke="#4f46e5" stroke-width="1.4" fill="none">
    <path d="M 140 60 L 240 60"/><polygon points="240,60 232,56 232,64" fill="#4f46e5"/>
    <path d="M 280 60 L 380 60"/><polygon points="380,60 372,56 372,64" fill="#4f46e5"/>
    <path d="M 420 60 L 478 60"/><polygon points="478,60 470,56 470,64" fill="#4f46e5"/>
  </g>
  <text x="200" y="50" text-anchor="middle" font-size="9" fill="#64748b">A (전이)</text>
  <g stroke="#0d9488" stroke-width="1.4" fill="none">
    <path d="M 120 80 L 120 140"/><polygon points="120,140 116,132 124,132" fill="#0d9488"/>
    <path d="M 260 80 L 260 140"/><polygon points="260,140 256,132 264,132" fill="#0d9488"/>
    <path d="M 400 80 L 400 140"/><polygon points="400,140 396,132 404,132" fill="#0d9488"/>
  </g>
  <text x="150" y="115" font-size="9" fill="#0d9488">B (방출)</text>
  <text x="40" y="178" font-size="11" font-weight="700" fill="#0d9488">관측 oₜ</text>
  <g fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2">
    <rect x="100" y="150" width="40" height="34" rx="4"/><rect x="240" y="150" width="40" height="34" rx="4"/><rect x="380" y="150" width="40" height="34" rx="4"/>
  </g>
  <g font-size="9" fill="#0d9488" text-anchor="middle">
    <text x="120" y="171">o₁</text><text x="260" y="171">o₂</text><text x="400" y="171">o₃</text>
  </g>
  <text x="290" y="206" text-anchor="middle" font-size="10" fill="#64748b">전이는 숨겨져 있고, 거래만 보인다 — HMM 은 화살표를 거꾸로 푼다</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Hidden Markov Model. 상태 체인 z 는 전이 행렬 A 로 진화하고, 각 상태는 분포 B 로 관측 o 를 방출한다. 우리는 o 만 보고 z 를 복원해야 한다.</figcaption>
</figure>

## 마르코프 성질 — 미래는 현재에만 의존한다

전체 구조는 하나의 가정, *마르코프 성질(Markov property)* 위에 선다.

$$ P(q_{t+1} \mid q_t, q_{t-1}, \dots, q_1) = P(q_{t+1} \mid q_t) $$

고객의 *다음* 상태는 *현재* 상태에만 의존하며, 그 이전의 전체 이력과는
독립이다. 지금 `MATURE` 인 고객은 6개월 전에 `NEW` 였든 `AT_RISK`
였든 내달 전이 확률이 동일하다. 이는 명백한 단순화 — 현실에서는 장기
기억이 중요하다 — 이지만, 파라미터 수를 $O(N^2)$(전이 행렬)로 묶어
수십만 고객 시퀀스에서도 안정적으로 학습하게 해준다. 잃어버린 장기
기억은, 프로젝트가 세 가지 시간 척도의 모드를 병렬로 돌리고, 시퀀스
전체 행동을 별도로 재인코딩하는 6D ODE-dynamics 브리지를 더해 보완한다.

> **역사적 배경.** HMM 의 수학은 1966년 Leonard Baum 과 Ted Petrie 가
> *Annals of Mathematical Statistics* 에서 정립했다 — "마르코프 체인의
> 확률적 함수"의 통계적 추정을 정식화했고, 이것이 오늘날 Baum-Welch 로
> 불리는 알고리즘의 원형이다. 그러나 HMM 이 널리 보급된 것은 Lawrence
> Rabiner 의 1989년 *Proceedings of the IEEE* 튜토리얼 덕분으로, 세 가지
> 근본 문제 — 평가, 디코딩, 학습 — 로 분야를 정리해 음성 인식을 넘어
> 생물정보학, NLP, 금융으로 확산시켰다. 이 글의 표기법과 3문제 구분도
> Rabiner 를 따른다. 금융에서는 같은 기계가 Hamilton(1989)의
> Markov-switching regime 모델로 나타난다 — 강세/약세 국면이 곧 이름만
> 바꾼 고객 성장/이탈 국면이다.

## 세 파라미터: $\pi$, $A$, $B$

Gaussian HMM 은 삼중쌍 $\lambda = (\pi, A, B)$ 이다.

| 기호 | 이름 | 의미 | 제약 |
| --- | --- | --- | --- |
| $\pi = \{\pi_i\}$ | 초기 분포 | $\pi_i = P(q_1 = S_i)$ — 신규 고객이 시작하는 곳 | $\sum_i \pi_i = 1$ |
| $A = \{a_{ij}\}$ | 전이 행렬 | $a_{ij} = P(q_{t+1}=S_j \mid q_t=S_i)$ — 행동 변화의 법칙 | 각 행의 합 = 1 |
| $B = \{b_j(\mathbf{o})\}$ | 방출 | $b_j(\mathbf{o}) = \mathcal{N}(\mathbf{o};\mu_j,\Sigma_j)$ — 상태의 소비 양상 | $\Sigma_j$ 대각 |

각 시점의 관측은 **3D 벡터** — log-금액, log-횟수, 카테고리 다양성 —
이다.

$$ \mathbf{o}_t = \big(\ln(\text{txn\_amount}+1),\ \ln(\text{txn\_count}+1),\ \text{mcc\_diversity}\big) $$

$\ln(x+1)$ 은 극단적 꼬리를 가진 소비를 Gaussian 이 다룰 수 있는
형태로 압축하고, $+1$ 은 거래 0 인 달에 $\ln(0)$ 이 발산하는 것을
막는다. 공분산을 `diag` 로 둔 것은 의도적이다 — 관측 차원이 3 뿐이라,
full 공분산은 상태당 6개 파라미터가 필요한데 대각은 3개로 족하고,
잃는 교차 상관보다 과적합이 더 큰 위험이기 때문이다.

관측 시퀀스 $\mathbf{O}=(\mathbf{o}_1,\dots,\mathbf{o}_T)$ 와 상태 경로
$\mathbf{Q}=(q_1,\dots,q_T)$ 의 결합 확률은 체인을 따라 인수분해된다.

$$ P(\mathbf{O},\mathbf{Q}\mid\lambda) = \pi_{q_1}\,b_{q_1}(\mathbf{o}_1)\prod_{t=2}^{T} a_{q_{t-1}q_t}\,b_{q_t}(\mathbf{o}_t) $$

> **수식 직관.** 왼쪽에서 오른쪽으로 생성 이야기로 읽는다 — 사전 $\pi$
> 로 어떤 상태에서 시작해 $b$ 로 첫 관측을 방출하고, 이후 매 시점마다
> 움직이는 *전이* 비용 $a$ 와 보이는 것을 만드는 *방출* 비용 $b$ 를
> 지불한다. 곱해진 확률의 체인이므로, 한 고리만 비개연적이어도(행렬이
> 드물다고 하는 전이거나, 그 상태가 거의 방출하지 않을 관측이거나) 전체
> 경로 확률이 무너진다. HMM 의 일은 우리가 끝내 보지 못한 숨겨진 경로
> $\mathbf{Q}$ 에 대해 이것을 합하거나 최대화하는 것이다.

## 세 모드, 세 시간 척도

하나의 HMM 대신 셋을 돌린다 — 같은 거래 흐름에 대한 서로 다른 질문.
아래 전이 행렬은 참조서의 실제 도메인 초기화 사전값이다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Lifecycle 모드 — 5 상태, 순방향 드리프트 + 이탈 흡수</text>
  <g>
    <circle cx="80" cy="120" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="80" y="116" text-anchor="middle" font-size="9.5" font-weight="700" fill="#0d9488">NEW</text><text x="80" y="130" text-anchor="middle" font-size="8" fill="#64748b">0</text>
    <circle cx="200" cy="80" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="200" y="78" text-anchor="middle" font-size="9" font-weight="700" fill="#0d9488">GROW</text><text x="200" y="91" text-anchor="middle" font-size="8" fill="#64748b">1</text>
    <circle cx="330" cy="120" r="30" fill="#0d948818" stroke="#0d9488" stroke-width="1.4"/><text x="330" y="117" text-anchor="middle" font-size="8.5" font-weight="700" fill="#0d9488">MATURE</text><text x="330" y="131" text-anchor="middle" font-size="8" fill="#64748b">2</text>
    <circle cx="450" cy="80" r="30" fill="#d9770618" stroke="#d97706" stroke-width="1.4"/><text x="450" y="78" text-anchor="middle" font-size="8" font-weight="700" fill="#d97706">AT_RISK</text><text x="450" y="91" text-anchor="middle" font-size="8" fill="#64748b">3</text>
    <circle cx="540" cy="160" r="30" fill="#e11d4818" stroke="#e11d48" stroke-width="1.4"/><text x="540" y="158" text-anchor="middle" font-size="7.5" font-weight="700" fill="#e11d48">CHURNED</text><text x="540" y="171" text-anchor="middle" font-size="8" fill="#64748b">4</text>
  </g>
  <g stroke="#64748b" stroke-width="1.2" fill="none">
    <path d="M 106 105 L 174 90"/><polygon points="174,90 165,89 168,97" fill="#64748b"/>
    <path d="M 226 92 L 304 110"/><polygon points="304,110 295,105 297,113" fill="#64748b"/>
    <path d="M 356 108 L 426 92"/><polygon points="426,92 417,91 420,99" fill="#64748b"/>
    <path d="M 474 100 L 518 138"/><polygon points="518,138 509,133 512,142" fill="#64748b"/>
  </g>
  <g font-size="8.5" fill="#1e3a5f" font-weight="700">
    <text x="140" y="88">0.45</text><text x="262" y="95">0.35</text><text x="392" y="88">0.15</text><text x="488" y="122">0.28</text>
  </g>
  <path d="M 540 130 C 575 105, 600 130, 568 145" stroke="#e11d48" stroke-width="1.2" fill="none"/><polygon points="568,145 575,138 576,148" fill="#e11d48"/>
  <text x="582" y="118" font-size="8.5" font-weight="700" fill="#e11d48">0.80</text>
  <path d="M 525 132 C 380 230, 200 220, 88 150" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3" fill="none"/><polygon points="88,150 96,150 92,158" fill="#94a3b8"/>
  <text x="300" y="232" text-anchor="middle" font-size="8.5" fill="#94a3b8">재유치 0.05 (CHURNED → NEW)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Lifecycle 전이 행렬을 그래프로. MATURE 자기 전이는 0.70(안정성), CHURNED 자기 전이는 0.80(끈적한 흡수)이되 0.05 를 NEW 로 흘려 재유치를 모델링한다.</figcaption>
</figure>

| 모드 | 상태 | 시간 척도 | 타겟 태스크 |
| --- | --- | --- | --- |
| **Journey** (AICRA) | 5 — AWARENESS, CONSIDERATION, PURCHASE, RETENTION, ADVOCACY | 일/주 (단기) | CTR, CVR |
| **Lifecycle** | 5 — NEW, GROWING, MATURE, AT_RISK, CHURNED | 월/년 (장기) | Churn, Retention, Life-stage |
| **Behavior** | 6 — DORMANT, CONSERVATIVE, ROUTINE, EXPLORATORY, SPLURGE, INVESTOR | 월별 패턴 | NBA, balance_util |

참조서에서 짚어둘 만한 두 가지. **Lifecycle** 행렬은 `MATURE` 에
0.70 자기 전이(성숙 고객은 안정적)를, `CHURNED` 에 0.80 자기 전이
(이탈은 끈적함)를 주되, 재유치를 위한 0.05 `CHURNED → NEW` 엣지를
허용한다. **Behavior** 행렬은 `SPLURGE` 에 *가장 낮은* 자기 전이 0.35
를 준다 — 충동 소비는 본성상 일시적이지 지속적 상태가 아니다. 그리고
**Journey** 는 학습 시점에서 예외다 — 일별 시퀀스가 EM 에 너무 sparse
하여 전이 행렬에 Baum-Welch 를 건너뛰고 config 사전값을 고정한다
(`params="mc"` — means/covars 만 학습). 그래서 Journey 의 $a_{ij}$ 는
데이터로 추정되지 않은, 교정되지 않은 도메인 사전값이다.

## 학습과 디코딩: Baum-Welch 와 Viterbi

세 고전적 문제, 세 알고리즘. 참조서는 프로젝트 설정을 명시한다 —
Baum-Welch 는 `n_iter=200`, `tol=1e-2`(교과서 `1e-4` 보다 완화, 수십만
규모의 수렴 안정성을 위한 선택).

**학습 — Baum-Welch (EM).** "상태를 알면 파라미터를 알고, 그 역도
성립한다"는 닭-달걀 문제를 반복으로 푼다. E-step 은 현재 파라미터에서
상태 소속 사후 확률 $\gamma_t(i)$ 와 전이 사후 확률 $\xi_t(i,j)$ 를
계산하고, M-step 은 $\pi,A,\mu,\Sigma$ 를 $\gamma$ 가중 평균으로 재추정
한다. E-step 을 떠받치는 전방 재귀가 전체의 무게를 받치는 핵심 수식이다.

$$ \alpha_{t+1}(j) = \Big[\sum_{i=1}^{N}\alpha_t(i)\,a_{ij}\Big]\,b_j(\mathbf{o}_{t+1}),\qquad \alpha_1(i)=\pi_i\,b_i(\mathbf{o}_1) $$

매 반복마다 로그우도 $\ln P(\mathbf{O}\mid\lambda)$ 가 단조 증가하여
극대값에 수렴한다 — 그래서 $\pi$, $A$, $\mu$ 의 도메인 전문가 사전값이
중요하다. 무작위 대신 좋은 분지(basin) 근처에서 탐색을 시작하게 한다.

**디코딩 — Viterbi.** 전방 패스가 모든 경로를 *합산* 하는 반면 Viterbi
는 *최대화* 한다 — 같은 동적 프로그래밍 격자에서 합을 max 로 바꾸고,
승리 경로를 되짚는 역추적 포인터 $\psi$ 를 유지해 가장 그럴듯한 단일
상태 시퀀스를 찾는다. $N^T$ 경로를 모두 열거하는 것은 가망이 없지만
($N=5, T=12$ 이면 약 2.4억), 마르코프 성질이 이를 $O(N^2 T)$ — 그
경우 약 300 연산 — 으로 줄인다. 그 디코딩된 경로가 *바로* 고객의 추정
상태 이력이며, 메타 피처와 ODE 브리지의 원재료다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <text x="300" y="24" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">Viterbi 격자 — 상태 격자를 통과하는 최적 경로</text>
  <g font-size="9" fill="#64748b" text-anchor="middle">
    <text x="90" y="210">t₁</text><text x="210" y="210">t₂</text><text x="330" y="210">t₃</text><text x="450" y="210">t₄</text><text x="540" y="210">t₅</text>
  </g>
  <g font-size="9" fill="#64748b" text-anchor="end">
    <text x="55" y="64">S₁</text><text x="55" y="114">S₂</text><text x="55" y="164">S₃</text>
  </g>
  <g stroke="#e2e8f0" stroke-width="1">
    <line x1="90" y1="60" x2="210" y2="60"/><line x1="90" y1="60" x2="210" y2="110"/><line x1="90" y1="60" x2="210" y2="160"/>
    <line x1="90" y1="110" x2="210" y2="60"/><line x1="90" y1="110" x2="210" y2="110"/><line x1="90" y1="110" x2="210" y2="160"/>
    <line x1="90" y1="160" x2="210" y2="60"/><line x1="90" y1="160" x2="210" y2="110"/><line x1="90" y1="160" x2="210" y2="160"/>
    <line x1="210" y1="60" x2="330" y2="60"/><line x1="210" y1="110" x2="330" y2="110"/><line x1="210" y1="160" x2="330" y2="160"/>
    <line x1="210" y1="60" x2="330" y2="110"/><line x1="210" y1="110" x2="330" y2="60"/><line x1="210" y1="110" x2="330" y2="160"/><line x1="210" y1="160" x2="330" y2="110"/>
    <line x1="330" y1="60" x2="450" y2="60"/><line x1="330" y1="110" x2="450" y2="110"/><line x1="330" y1="160" x2="450" y2="160"/>
    <line x1="330" y1="110" x2="450" y2="160"/><line x1="330" y1="160" x2="450" y2="110"/>
    <line x1="450" y1="60" x2="540" y2="60"/><line x1="450" y1="110" x2="540" y2="110"/><line x1="450" y1="160" x2="540" y2="160"/>
    <line x1="450" y1="160" x2="540" y2="110"/>
  </g>
  <g stroke="#d97706" stroke-width="2.6" fill="none">
    <line x1="90" y1="110" x2="210" y2="60"/>
    <line x1="210" y1="60" x2="330" y2="60"/>
    <line x1="330" y1="60" x2="450" y2="160"/>
    <line x1="450" y1="160" x2="540" y2="160"/>
  </g>
  <g fill="#cbd5e1">
    <circle cx="90" cy="60" r="6"/><circle cx="90" cy="160" r="6"/>
    <circle cx="210" cy="110" r="6"/><circle cx="210" cy="160" r="6"/>
    <circle cx="330" cy="110" r="6"/><circle cx="330" cy="160" r="6"/>
    <circle cx="450" cy="60" r="6"/><circle cx="450" cy="110" r="6"/>
    <circle cx="540" cy="60" r="6"/><circle cx="540" cy="110" r="6"/>
  </g>
  <g fill="#d97706">
    <circle cx="90" cy="110" r="7.5"/><circle cx="210" cy="60" r="7.5"/><circle cx="330" cy="60" r="7.5"/><circle cx="450" cy="160" r="7.5"/><circle cx="540" cy="160" r="7.5"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Viterbi 디코딩. 모든 회색 격자 경로 중 단 하나(앰버)만 결합 확률을 최대화한다 — 복원된 고객 상태 이력이며, 여기서 체류 시간과 전이 안정성을 읽어낸다.</figcaption>
</figure>

## 실제로 추출되는 것: 모드당 16D

각 모드는 정확히 **16D** 를 내보내며, 구조는 동일하다 — `n_states` 개의
상태 확률 차원 + 메타 피처 + 6D ODE 브리지. 참조서의 차원 검증식은 수치
피처 기준 `n_states + 5 + 6` 이다(문자열인 `dominant_state_name` 과
`state_mode` 는 차원에 미포함). Journey/Lifecycle 은 이 식대로
$5 + 5 + 6 = 16$ 이고, Behavior 는 참조서가 $6 + 4 + 6 = 16$ 으로
기재한다 — 늘어난 여섯 번째 상태만큼 메타 블록을 줄여, 모든 모드가
16D 를 유지한다.

| 그룹 | 피처 | 출처 | 차원 (J/L · B) |
| --- | --- | --- | --- |
| **상태 사후 확률** | `state_prob_0 … state_prob_{N-1}` — $\gamma_t(i)$ 소프트 할당 | Forward-Backward | 5D · 6D |
| **전이 / 체류 메타** | `state_duration`, `transition_stability`, `transition_entropy`, `dominant_state`, `state_change_rate` | Viterbi 경로 | 5D · 4D |
| **ODE 동역학** | `ode_velocity`, `ode_acceleration`, `ode_lyapunov`, `ode_cycle_period`, `ode_attractor`, `ode_trajectory_len` | Viterbi 궤적 | 6D · 6D |

**상태 사후 확률** 이 핵심이다 — $\gamma_t(i) =
P(q_t=S_i\mid\mathbf{O},\lambda)$, *전체* 시퀀스(과거와 미래의 증거
모두, $\alpha$ 와 $\beta$ 를 통해)가 주어졌을 때 고객이 상태 $i$ 에
있을 확률. 불확실성을 인코딩하고, 다운스트림 신경망에 대해 연속이고
미분 가능하며, 차원별로 해석 가능한 소프트 할당이다.

**메타 피처** 는 디코딩된 Viterbi 경로에서 나온다 — `state_duration`
은 현재 상태에 머문 길이(안정성/관성 신호)를 세고,
`transition_entropy` 는 전이 쌍 빈도의 Shannon 엔트로피를 $\log(N^2)$
로 정규화해 전이가 얼마나 *다양한가* 를 재며, `state_change_rate` 는
전이가 얼마나 *자주* 일어나는가를 잰다 — 둘은 보완적이다.

**ODE-dynamics 브리지**(v3.2.0 추가)는 Viterbi 시퀀스를 연속 궤적으로
보고 여섯 가지 운동학적 기술자를 추출한다 — 속도는 평균 $|\Delta q_t|$,
가속도는 평균 $|\Delta^2 q_t|$, Lyapunov 영감의 후반부/전반부 불안정성
비, 자기상관 기반 주기, 끌개 집중도 비, 정규화 궤적 길이. 추가 학습
없는 순수 시퀀스 분석이며, 마르코프 성질이 보지 못하는 장기 구조를
보완하는 장치다. 시퀀스가 3 스텝 미만이면 여섯 개 모두 0.0 을 반환하고,
주기 탐지는 길이 ≥ 6 에서만 켜진다.

## 48D 가 안착하는 곳

참조서의 `feature_schema.yaml` 기준 전체 그림.

- PLE 모델은 **734D 메인 텐서**(644D normalized + 90D raw power-law)
  *에 더해* **68D 별도 입력**(20D hyperbolic + **48D HMM Triple-Mode**)
  을 받는다. HMM 피처는 메인 텐서가 아니라 *별도 경로* 에 탑승한다.
- 각 모드의 16D 는 PLE 내부의 전용 **HMM Triple-Mode Projector** 로
  라우팅되어, 타겟 태스크 Expert 의 hidden 차원으로 사영된다 —
  Journey → CTR/CVR, Lifecycle → Churn/Retention/Life-stage,
  Behavior → NBA/balance_util.
- 별도로, 압축된 **5D 요약** — `hmm_dominant_state`,
  `hmm_state_duration`, `hmm_transition_stability`,
  `hmm_transition_entropy`, `hmm_state_change_rate` — 은 메인 텐서의
  `model_derived` 블록(27D = HMM 요약 5D + Bandit 4D + LNN 18D) 안에
  있어, 모든 태스크가 고수준 행동 신호를 전역적으로 참조하게 한다 —
  48D 와 중복이 아니라 보완이다.

이것은 설계상 *오프라인* 모듈이다. HMM 은 Airflow 배치에서 학습되고
디코딩되며 — GMM, Economics 모듈과 완전히 독립이라 DAG 에서 병렬화된다 —
결과 피처는 Parquet 으로 쓰여 서빙 시점에 조회된다. 온라인 HMM 추론은
없다.

## 여기서 멈추는 이유

집계 소비가 그것을 설명하는 상태를 숨긴다는 불편함에서 출발해,
마르코프 성질과 $(\pi, A, B)$ 삼중쌍을 세우고, 결합 확률 인수분해를
따라갔으며, Baum-Welch 로 학습하고 Viterbi 로 디코딩해, 실제 48D
(상태 사후 확률 + 전이/체류 메타 + ODE 동역학, 16D × 3 모드)가 PLE 의
별도 입력 경로에, 5D 요약이 메인 텐서에 안착하는 것을 봤다.

HMM 이 주는 것은 *시계열* 적 독해다 — 고객이 자기 궤적의 어디에 있고
어느 방향으로 움직이는가, 전이 행렬이 시간 역학을 담는다. 다음 모듈은
정반대 질문을 던진다 — "어떤 단계, 어디로" 가 아니라 "지금 어떤
*유형*". 그것이 가우시안 혼합의 일이다 — 한 시점 스냅샷, 횡단면
소프트 클러스터링으로 고객을 단계가 아닌 유형으로 프로파일링하고,
22D 를 별도 경로가 아니라 메인 텐서의 Domain 블록 *안* 에 안착시킨다.
이것이 다음 편 **GMM-1** 의 주제다.
