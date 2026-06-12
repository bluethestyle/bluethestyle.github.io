---
title: "[Study Thread] CAUSALOT-1 — 원인을 따라 질량을 옮기다: 최적 수송과 인과 추론의 만남"
date: 2026-06-06 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, causal, optimal-transport, notears, counterfactual, expert]
lang: ko
excerpt: "Causal + Optimal Transport 서브스레드 시작 — 추천에 상관관계만으로 부족한 이유, Causal Expert 가 NOTEARS 의 미분 가능한 비순환성 제약으로 DAG 를 학습하는 방식, 구조 방정식과 반사실이 실제로 계산하는 것, 그리고 OT 측이 고객을 분포로 읽어 Sinkhorn 으로 Wasserstein 거리를 재는 법. '왜 이 추천인가' 와 '얼마나 잘 맞는가', 한 동전의 두 면."
series: study-thread
part: 15
alt_lang: /2026/06/06/causal-ot-expert-en/
next_title: "TEMPORAL-1 — 한 고객을 위한 세 개의 시계: Mamba, Liquid Network, Patch Transformer 의 앙상블"
next_desc: "Temporal Expert 가 소비의 시간 축을 읽는 방식 — 상태공간 모델(Mamba), 연속시간 liquid 네트워크, patch Transformer 를 결합한 구조, 그리고 세 가지 다른 '기억' 개념이 하나보다 나은 이유."
next_status: draft
---

*"Study Thread" 시리즈의 Causal + Optimal Transport(CausalOT) 서브스레드
1편. 이번 편부터 영문/국문 병렬로 본 프로젝트의 7개 이종 Shared Expert 중
둘 — Causal Expert 와 OT Expert — 을 정리한다. 출처는 온프렘 프로젝트
`기술참조서/CausalOT_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막
편에 첨부한다. TDA 서브스레드가 Expert 가 어떤* 형태 *를 읽는가를
물었다면, 이번 서브스레드는 더 어려운 두 질문을 던진다 — 이 추천이
정말 원하는 결과를* 유발 *하는가, 그리고 이 고객의 행동이 프로토타입에
얼마나* 잘 *맞는가? 상관관계는 둘 다 답하지 못한다. 인과 DAG 와
최적수송은 답한다.*

> **두 Expert, 하나의 설계 패턴.** 아키텍처 v3.2 는 두 개의 Shared
> Expert 를 나란히 추가했다 — 잠재 변수 위에 비순환 인과 그래프를 학습해
> 개입하는 **Causal Expert**, 그리고 각 고객을 확률 분포로 바꿔 학습된
> 프로토타입과의 Wasserstein 거리를 재는 **OT Expert**. 둘 다 동일한
> 정규화 **644D** 피처 벡터를 받아 **64D** 표현을 CGC gate 로 내보낸다 —
> 그러나 추출하는 수학적 구조는 정반대다. 하나는 *비대칭이고 비순환*(인과의
> 방향), 다른 하나는 *거리 함수*(분포적 거리). 이 글은 둘 모두를 따라간다.

## 추천에 상관관계만으로는 부족하다

일반 추천기는 *상관관계* 를 학습한다 — "A 를 산 고객이 B 도 샀다." 놀랍도록
잘 작동한다. 안 될 때까지는. 참조서는 직설적인 예로 시작한다.

> 프리미엄 카드 보유 고객은 해외여행 보험 가입률이 높다.

상관관계 기반 시스템은 프리미엄 카드 보유자 전원에게 여행 보험을 밀어붙일
것이다. 그러나 실제 구조는 *교란 변수* 일 수 있다.

```
프리미엄 카드  ←  높은 소득  →  여행 보험
```

높은 소득이 *둘 다* 를 유발한다. 카드가 보험 가입을 *유발* 하는 게
아니다. 프리미엄 카드를 무료로 뿌려도 보험 가입률은 움직이지 않는다.
이것이 상관관계가 빠져나갈 수 없는 함정이며, 정확히 Judea Pearl 의
*인과의 사다리* 가 형식화하는 지점이다 — 세 단(rung), 각 단이 아래
단으로는 답할 수 없는 더 어려운 질문에 답한다.

| 단(Rung) | 질문 | 이 Expert 에서 |
| --- | --- | --- |
| 1 — 연관 | "무엇이 무엇과 관련되는가?" | 평범한 모델이 보는 raw 상관 |
| 2 — 개입 | "X 를 *하면* 어떻게 되는가?" | 구조 방정식 $\hat{\mathbf z} = \mathbf z + \mathbf z(\mathbf W \odot \mathbf W)$ |
| 3 — 반사실 | "*했더라면* 어땠을까?" | `get_counterfactual` → factual / direct_only / full_cf |

<figure style="margin:24px auto;max-width:440px;">
<svg viewBox="0 0 440 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="440" height="240" fill="#f8fafc" rx="8"/>
  <text x="220" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Pearl 의 인과 사다리</text>
  <rect x="70" y="48" width="300" height="40" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="86" y="66" font-size="12" font-weight="700" fill="#4f46e5">Rung 3 — 반사실</text>
  <text x="86" y="81" font-size="10" fill="#64748b">"했더라면?"  ·  full_cf − direct_only</text>
  <rect x="70" y="100" width="300" height="40" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <text x="86" y="118" font-size="12" font-weight="700" fill="#0d9488">Rung 2 — 개입</text>
  <text x="86" y="133" font-size="10" fill="#64748b">"do(X) 하면?"  ·  ẑ = z + z(W⊙W)</text>
  <rect x="70" y="152" width="300" height="40" rx="6" fill="#64748b15" stroke="#64748b" stroke-width="1.2"/>
  <text x="86" y="170" font-size="12" font-weight="700" fill="#64748b">Rung 1 — 연관</text>
  <text x="86" y="185" font-size="10" fill="#64748b">"무엇이 상관되는가?"  ·  P(Y | X)</text>
  <line x1="40" y1="190" x2="40" y2="58" stroke="#d97706" stroke-width="1.6"/>
  <polygon points="40,50 35,62 45,62" fill="#d97706"/>
  <text x="22" y="128" text-anchor="middle" font-size="10" fill="#d97706" font-weight="700" transform="rotate(-90 22 128)">더 강한 주장</text>
  <text x="220" y="222" text-anchor="middle" font-size="10.5" fill="#64748b">평범한 추천기는 Rung 1 에 산다. 이 Expert 는 2·3 으로 오른다.</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Pearl 의 세 단. 각 단은 아래 단이 답할 수 없는 질문을 던지며, 개입과 반사실만이 인과를 우연과 구분한다.</figcaption>
</figure>

> **역사적 배경.** 현대 인과추론은 서로 다른 언어로 두 거장이 세웠다.
> *Jerzy Neyman*(1923)이 잠재결과를 도입했고, *Donald Rubin*(1974)이
> 이를 관측 연구로 확장해 "Rubin Causal Model" 로 체계화했다. 별개로
> *Judea Pearl*(2000)은 구조적 인과 모형과 do-calculus 로 인과를 그래프
> 이론 위에 정립했고(2011 튜링상), 두 프레임워크는 — 잠재결과 vs 그래프 —
> 다른 방언이지만 수학적으로 동치임이 알려져 있다. 본 프로젝트의 Causal
> Expert 는 의도적 하이브리드다 — Pearl 의 그래프 관점(인접행렬
> $\mathbf W$) 과 Rubin 의 개체 효과 관점을 함께 쓴다.

## NOTEARS — 연속 최적화로 DAG 를 학습하기

인과 그래프는 **DAG** — 방향이 있고 *비순환* — 여야 한다. "A 가 B 를,
B 가 C 를, C 가 다시 A 를 유발한다" 는 논리적으로 불가능하다(시간 순서
위반). 어려운 점은 DAG 탐색이 조합론적이고 NP-hard 라는 것이다 — 변수가
$d=10$ 개뿐이어도 가능한 그래프가 약 $4.2\times10^{18}$ 개다.

NOTEARS(Zheng et al., *NeurIPS 2018*)가 이 문제를 깨뜨린 한 수다. 조합론적
"비순환인가?" 제약을, 가중 인접행렬 $\mathbf W$ 에 대한 하나의 *미분 가능한
등식* 으로 대체한다.

$$ h(\mathbf W) = \operatorname{tr}\!\left(e^{\,\mathbf W \odot \mathbf W}\right) - d = 0 $$

여기서 $\mathbf W \odot \mathbf W$ 는 Hadamard(원소별) 제곱으로 — 인과
강도를 비음수로 강제하기도 한다 — $e^{(\cdot)}$ 는 행렬 지수함수,
$\operatorname{tr}$ 은 trace, $d = 32$ 는 인과 변수 수(`n_causal_vars`)다.

> **수식 직관.** $e^{\mathbf M}$ 의 $(i,i)$ 대각 원소는 노드 $i$ 에서
> 자기 자신으로 돌아오는 *모든 닫힌 경로* 의 가중합이다. $(\mathbf
> M^k)_{ii}$ 가 길이 $k$ 순환을 세고 $e^{\mathbf M}=\sum_k \mathbf
> M^k/k!$ 가 모든 길이에 걸쳐 합하기 때문이다. DAG 에는 그런 순환이
> 없으므로 모든 대각 원소가 항등행렬 기여분 $1$ 로 붕괴하고, trace 가
> $d$ 가 되어 $h(\mathbf W)=0$ 이다. 양의 $h(\mathbf W)$ 는 "순환이
> 존재한다" 의 직접적 신호다. NOTEARS 의 한 수는 조합론적 그래프 조건을
> gradient 가 쫓을 수 있는 해석학적 등식으로 바꾼 것이다.

행렬 지수 전체를 계산하면 $O(d^3)$ 이므로, 프로젝트는 Taylor 급수 10항으로
근사한다 — $\mathbf W$ 가 작게 시작(`randn(32,32)*0.01`)해 고차항이 빠르게
사라지므로 싸고 정확하다. 10항은 "길이 $\le 10$ 인 모든 순환 감지" 를
뜻하며, 32개 노드에서 10-hop 순환은 현실적으로 발생하지 않는다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <text x="140" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">순환 존재 → h(W) &gt; 0</text>
  <g>
    <circle cx="90"  cy="90"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="90"  y="94"  text-anchor="middle" font-size="11" fill="#1e3a5f">A</text>
    <circle cx="190" cy="80"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="190" y="84"  text-anchor="middle" font-size="11" fill="#1e3a5f">B</text>
    <circle cx="140" cy="160" r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="140" y="164" text-anchor="middle" font-size="11" fill="#1e3a5f">C</text>
    <line x1="105" y1="84" x2="175" y2="80" stroke="#e11d48" stroke-width="1.6"/><polygon points="175,80 166,76 167,84" fill="#e11d48"/>
    <line x1="185" y1="94" x2="150" y2="146" stroke="#e11d48" stroke-width="1.6"/><polygon points="150,146 155,137 159,144" fill="#e11d48"/>
    <line x1="128" y1="148" x2="97" y2="104" stroke="#e11d48" stroke-width="1.6"/><polygon points="97,104 105,110 99,114" fill="#e11d48"/>
  </g>
  <line x1="280" y1="40" x2="280" y2="185" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="28" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">비순환(DAG) → h(W) = 0</text>
  <g>
    <circle cx="370" cy="90"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="370" y="94"  text-anchor="middle" font-size="11" fill="#1e3a5f">A</text>
    <circle cx="470" cy="80"  r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="470" y="84"  text-anchor="middle" font-size="11" fill="#1e3a5f">B</text>
    <circle cx="420" cy="160" r="16" fill="#fff" stroke="#64748b" stroke-width="1.4"/><text x="420" y="164" text-anchor="middle" font-size="11" fill="#1e3a5f">C</text>
    <line x1="385" y1="84" x2="455" y2="80" stroke="#0d9488" stroke-width="1.6"/><polygon points="455,80 446,76 447,84" fill="#0d9488"/>
    <line x1="365" y1="104" x2="412" y2="146" stroke="#0d9488" stroke-width="1.6"/><polygon points="412,146 403,142 408,136" fill="#0d9488"/>
    <line x1="463" y1="94" x2="430" y2="146" stroke="#0d9488" stroke-width="1.6"/><polygon points="430,146 434,137 438,144" fill="#0d9488"/>
  </g>
  <text x="420" y="200" text-anchor="middle" font-size="10" fill="#64748b">간선 가중치 = W²ᵢⱼ  (j → i 강도)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">비순환성 페널티를 한 그림으로. 순환 A→B→C→A 는 trace 를 d 위로 부풀리고, DAG 는 정확히 d 에 두어 h(W)=0.</figcaption>
</figure>

학습 중 프로젝트는 $h(\mathbf W)=0$ 을 엄격히 강제하지 않는다(논문은 증강
라그랑지안 사용). 단순 페널티 + 희소성 항으로 완화한다.

$$ \mathcal{L}_{\text{DAG}} = \lambda_{\text{acyclic}}\cdot h(\mathbf W) + \lambda_{\text{sparse}}\cdot \lVert \mathbf W \odot \mathbf W \rVert_1 $$

`dag_lambda = 0.01`, `sparsity_lambda = 0.001`. 첫째 항은 순환을 금지하고,
둘째 항은 가능한 $32\times32=1024$ 개 간선을 의미 있는 소수로 가지친다.
참조서의 경고 하나 — `dag_lambda` 를 0.1 이상으로 올리면 $\mathbf W$ 가
영행렬로 붕괴하고, Expert 는 항등 사상($\hat{\mathbf z}\approx\mathbf z$)으로
퇴화하며 인과 구조가 그냥 증발한다.

## 학습된 W 와 구조 방정식

Expert 내부 파이프라인은 세 단계다 — **Compressor** 가 644D 입력을 32개
인과 변수로 압축($644\to128\to32$)하고, **구조적 인과 모형(SCM)** 이
개입하며, **Causal Encoder** 가 결과를 다시 64D 로 올린다($32\to128\to64$).
개입 자체는 의심스러울 만큼 단순한 한 방정식이다.

$$ \hat{\mathbf z} = \mathbf z + \mathbf z(\mathbf W \odot \mathbf W) $$

$\mathbf z$ 는 32차원 잠재 벡터, $\mathbf W$ 는 학습 가능한 $[32,32]$
인접행렬, $\mathbf W \odot \mathbf W$ 는 비음수 간선 강도로 $W_{ij}^2$ 가
변수 $j \to$ 변수 $i$ 의 인과 영향이다. 곱 $\mathbf z(\mathbf W\odot\mathbf
W)$ 는 각 변수를 그 인과적 부모들의 선형 조합만큼 보정하고, 잔차 연결
`z +` 가 원본 신호를 보존한다. 결과는 단순 상관이 아니라 *인과적으로
보정된* 고객 표현이다. 학습 후 `get_causal_graph()` 는 $(\mathbf W \odot
\mathbf W).\text{detach()}$ — 어떤 잠재 요인이 어떤 요인을 끄는지 보여주는
$[32,32]$ 히트맵 — 을 반환한다.

## 반사실 — 했더라면 어땠을까

여기서 Expert 는 Rung 3 에 닿는다. `get_counterfactual(x, j, v)` 는 하드
개입 $do(z_j = v)$ 를 적용하고 인코더를 *세* 가지로 돌린다 — 그리고 그
중 두 가지의 격차가 핵심이다.

$$
\begin{aligned}
\textbf{factual} &= \text{encoder}(\mathbf z + \mathbf z\,\mathbf W^2) \\
\textbf{direct\_only} &= \text{encoder}(\mathbf z' + \mathbf z\,\mathbf W^2) \\
\textbf{full\_cf} &= \text{encoder}(\mathbf z' + \mathbf z'\,\mathbf W^2)
\end{aligned}
$$

$\mathbf z'$ 는 $\mathbf z$ 에서 좌표 $j$ 만 $v$ 로 덮어쓴 것이다.
**direct_only** 에서는 개입이 $z_j$ 자체만 건드리고, DAG 매개 경로
$\mathbf z\,\mathbf W^2$ 는 개입 *이전 값으로 고정* 된다. **full_cf**
에서는 개입이 그래프를 통해 *전파* 되도록 허용한다 — 매개 항이 $\mathbf
z'$ 로부터 재계산된다. 따라서 차이

$$ \Delta_{\text{mediated}} = \textbf{full\_cf} - \textbf{direct\_only} $$

는 정확히 **인과 그래프를 통해 흐르는 효과** — Pearl 의 Rung 3 매개
효과 — 다. 만약 $\mathbf W$ 가 단지 장식적(실제 구조를 학습하지 못함)으로
드러나면, 두 분기가 같은 값으로 붕괴해 $\Delta_{\text{mediated}}\to 0$ 이
된다. 반사실 프로브는 무엇보다도 DAG 가 일을 하긴 하는지에 대한 정직성
점검이다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="250" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">do(z_j = v): 세 가지 forward variant</text>
  <rect x="20" y="105" width="96" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="68" y="123" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">잠재 z</text>
  <text x="68" y="138" text-anchor="middle" font-size="9" fill="#64748b">z_j 개입</text>
  <line x1="116" y1="118" x2="200" y2="60" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="40" width="200" height="44" rx="6" fill="#64748b15" stroke="#64748b" stroke-width="1.1"/>
  <text x="214" y="58" font-size="11" font-weight="700" fill="#64748b">factual</text>
  <text x="214" y="73" font-size="9.5" fill="#64748b">encoder(z + z·W²) — 개입 없음</text>
  <line x1="116" y1="125" x2="200" y2="128" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="106" width="200" height="44" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.1"/>
  <text x="214" y="124" font-size="11" font-weight="700" fill="#0d9488">direct_only</text>
  <text x="214" y="139" font-size="9.5" fill="#0d9488">encoder(z′ + z·W²) — 경로 고정</text>
  <line x1="116" y1="132" x2="200" y2="196" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="200" y="172" width="200" height="44" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.1"/>
  <text x="214" y="190" font-size="11" font-weight="700" fill="#4f46e5">full_cf</text>
  <text x="214" y="205" font-size="9.5" fill="#4f46e5">encoder(z′ + z′·W²) — 경로 전파</text>
  <line x1="412" y1="128" x2="412" y2="194" stroke="#d97706" stroke-width="1.4"/>
  <line x1="412" y1="128" x2="406" y2="128" stroke="#d97706" stroke-width="1.4"/>
  <line x1="412" y1="194" x2="406" y2="194" stroke="#d97706" stroke-width="1.4"/>
  <text x="424" y="156" font-size="10" font-weight="700" fill="#d97706">Δ = full_cf − direct_only</text>
  <text x="424" y="170" font-size="9.5" fill="#64748b">= DAG 매개 효과</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">하나의 개입에서 세 번의 forward. W² 경로를 고정 vs 전파시키는 차이가, 인과 그래프가 매개하는 효과만 정확히 분리한다.</figcaption>
</figure>

같은 forward 가 프로젝트가 다른 곳에서 쓰는 두 신호를 더 내놓는다 —
**causal coherence score** $\lVert \mathbf z - \mathbf z \cdot \mathbf
W^2 \rVert^2 / \lVert \mathbf z \rVert^2$(`CausalGuardrail` 의 근거,
Rung 1 in-distribution 점검)와 **CEH attribution** 헤드(Rung 2, task
logit 의 grad×input). 한 Expert 에 세 단 모두, 추가 계산은 거의 없이.

## 최적수송 측 — 고객은 하나의 분포다

OT Expert 는 고객을 하나의 점으로 압축하길 거부한다. 고객을 *확률 분포*
로 읽고, 그 분포가 학습된 프로토타입 집합에서 얼마나 멀리 떨어져 있는지를 —
가장 기하학적으로 정직한 거리로 — 묻는다.

왜 KL 발산이나 총 변동이 아닌가? 기저 공간의 기하를 무시하기 때문이다.
한 분포의 질량을 서울, 다른 분포를 인천, 또 다른 분포를 부산에 두자. KL 과
TV 는 $\text{dist}(P,Q)\approx\text{dist}(P,R)$ 로 보고한다 — support 가
겹치지 않으면 질량이 *얼마나 멀리* 있든 거리가 같다. 최적수송은 이를
바로잡는다 — 기저 공간을 가로질러 질량을 *옮기는* 비용을 반영하므로,
서울↔인천이 서울↔부산보다 진짜로 가깝다.

<img src="/optimal-transport.webp" alt="Optimal Transport — source distribution μ (blue cluster) and target distribution ν (red cluster) connected by transport plan γ showing pair-wise sample matchings" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

고전적 정식화는 **Monge–Kantorovich** 문제다. Monge(1781)는 흙 무더기를
최소 비용으로 옮기는 문제로 봤고, Kantorovich(1942)는 이를 수송 계획에
대한 선형계획으로 완화했다(1975 노벨 경제학상).

$$ W(\boldsymbol\mu, \boldsymbol\nu) = \min_{\mathbf P \in \mathcal U(\boldsymbol\mu,\boldsymbol\nu)} \langle \mathbf P, \mathbf C\rangle_F, \qquad \mathcal U(\boldsymbol\mu,\boldsymbol\nu) = \{\mathbf P \ge 0 : \mathbf P\mathbf 1 = \boldsymbol\mu,\ \mathbf P^\top\mathbf 1 = \boldsymbol\nu\} $$

$\mathbf P_{ij}$ 는 $i$ 에서 $j$ 로 옮긴 질량, $\mathbf C_{ij}$ 는 그 단위
비용, 최소 총 비용이 **Wasserstein 거리**(earth mover's distance)다. KL 과
달리 support 가 겹치지 않아도 유한하게 유지되며, 스칼라뿐 아니라 전체 수송
계획 $\mathbf P$ — 한 분포가 다른 분포로 *어떻게* 변형되는지를 보여주는
해석 가능한 지도 — 까지 돌려준다.

Expert 에서 이것은 구체화된다.

- **고객 분포.** $\boldsymbol\mu = \operatorname{softmax}(\text{DistProjector}(\mathbf x)) \in \Delta^{32}$ — 644D 피처 벡터를 32개 잠재 카테고리 위의 확률 simplex 로 사영.
- **프로토타입.** $\boldsymbol\nu_k = \operatorname{softmax}(\boldsymbol\ell_k) \in \Delta^{32}$, 학습 가능한 기준 분포 뱅크(클래스 기본 16, 운영 `n_ref=8`) — 데이터가 군집화한 "전형적 고객 유형"(여행 중심, 저축 중심 …)을 손으로 정의하지 않고 end-to-end 로 학습.
- **비용 행렬.** $\mathbf C = \mathbf M^\top\mathbf M$, $\mathbf M^\top\mathbf M$ 분해로 양반정치(PSD)를 강제한 학습 가능한 ground metric — 어떤 원소도 수송을 보상하지 않게(그러면 Sinkhorn 이 무의미한 계획을 낸다).

## Sinkhorn — 엔트로피 정규화가 속도와 gradient 를 산다

원형 그대로의 Kantorovich LP 는 변수가 $d^2$ 개라 대규모에서 비싸다. Cuturi(*NeurIPS
2013*)가 **엔트로피 정규화** 를 더해, 문제를 Sinkhorn 반복이 *선형*
수렴으로 푸는 형태로 — 그리고 결정적으로 end-to-end 학습을 위해 미분
가능하게 — 바꿨다.

$$ \min_{\mathbf P \in \mathcal U(\boldsymbol\mu,\boldsymbol\nu)} \langle \mathbf P, \mathbf C\rangle - \varepsilon\, H(\mathbf P), \qquad H(\mathbf P) = -\sum_{i,j} P_{ij}\log P_{ij} $$

> **수식 직관.** 엔트로피 항 $-\varepsilon H(\mathbf P)$ 는 너무 "뾰족한"
> 수송 계획(한 경로에 질량 몰림)에 페널티를 줘 계획을 부드러움 쪽으로
> 민다. $\varepsilon$ 이 크면 계획이 균등에 가깝게 흐려지고(비용 정보
> 손실), 작으면 날카롭지만 수치적으로 불안정하다. 프로젝트는 $\varepsilon
> = 0.1$(`sinkhorn_epsilon`)을 쓴다 — 참조서는 0.01 미만을 발산/NaN
> 영역, 1.0 초과를 흐릿한 수송으로 표시한다.

푸는 과정은 쌍대 변수의 교대 정규화이며, 수치 안전을 위해 **log 도메인**
에서 한다(작은 Gibbs 커널 원소 $e^{-C_{ij}/\varepsilon}$ 가 언더플로우하지
않도록).

$$
\begin{aligned}
\mathbf u_{\text{new}} &= \log\boldsymbol\mu - \operatorname{logsumexp}\!\left(-\mathbf C/\varepsilon + \mathbf v\right) \\
\mathbf v_{\text{new}} &= \log\boldsymbol\nu - \operatorname{logsumexp}\!\left(-\mathbf C^\top/\varepsilon + \mathbf u\right)
\end{aligned}
$$

각 단계는 계획의 행 합을 $\boldsymbol\mu$ 에, 열 합을 $\boldsymbol\nu$ 에
맞춘다. `logsumexp` 는 부동소수점 언더플로우를 막는 log 도메인 softmax 다.
클래스 기본은 10회 반복(운영 config 5회). 수렴 후 수송 계획은 $\log
P_{ij} = u_i + \log K_{ij} + v_j$, 거리는 Frobenius 내적
$W(\boldsymbol\mu,\boldsymbol\nu_k) = \langle \mathbf P, \mathbf
C\rangle_F$ 다. 16개 프로토타입 전부에 돌리면 $[B,16]$ Wasserstein 거리
벡터 — 각 고객을 16개 기준점으로부터의 거리로 위치시키는 *분포적 좌표계*
— 가 나온다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="170" fill="#f8fafc" rx="8"/>
  <rect x="16" y="62" width="78" height="46" rx="6" fill="#f0fdf4" stroke="#0d9488" stroke-width="1"/>
  <text x="55" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">x [644]</text>
  <text x="55" y="97" text-anchor="middle" font-size="9" fill="#64748b">피처</text>
  <rect x="120" y="62" width="92" height="46" rx="6" fill="#fce7f3" stroke="#e11d48" stroke-width="1"/>
  <text x="166" y="82" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">μ ∈ Δ³²</text>
  <text x="166" y="97" text-anchor="middle" font-size="9" fill="#64748b">softmax 사영</text>
  <rect x="238" y="22" width="100" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="288" y="40" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">16 × ν_k</text>
  <text x="288" y="54" text-anchor="middle" font-size="9" fill="#64748b">프로토타입</text>
  <rect x="238" y="108" width="100" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="288" y="126" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">C = MᵀM</text>
  <text x="288" y="140" text-anchor="middle" font-size="9" fill="#64748b">PSD 비용</text>
  <rect x="362" y="62" width="100" height="46" rx="6" fill="#fce7f3" stroke="#e11d48" stroke-width="1.2"/>
  <text x="412" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Sinkhorn</text>
  <text x="412" y="97" text-anchor="middle" font-size="9" fill="#64748b">log-domain ×10</text>
  <rect x="486" y="62" width="96" height="46" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="534" y="82" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">W [B,16]</text>
  <text x="534" y="97" text-anchor="middle" font-size="9" fill="#64748b">→ 64D enc</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="94" y1="85" x2="118" y2="85"/><polygon points="118,85 110,81 110,89"/>
    <line x1="212" y1="85" x2="360" y2="85"/><polygon points="360,85 352,81 352,89"/>
    <line x1="462" y1="85" x2="484" y2="85"/><polygon points="484,85 476,81 476,89"/>
    <line x1="338" y1="42" x2="362" y2="74"/><polygon points="362,74 353,71 359,67"/>
    <line x1="338" y1="128" x2="362" y2="96"/><polygon points="362,96 359,105 353,99"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">OT Expert forward: simplex 로 사영 → PSD 비용으로 16개 프로토타입에 Sinkhorn → Wasserstein 거리 16-벡터 → 64D 인코딩.</figcaption>
</figure>

마지막 2층 **Wasserstein Encoder**($16\to128\to64$)가 그 거리 벡터를
공유 64D 공간으로 올려, 거리 *패턴* 의 비선형 관계("여행형엔 가깝지만
저축형엔 먼")를 학습한다. Causal Expert 와의 비대칭 하나 — OT 는 **별도
정규화 손실이 없다**. Sinkhorn 의 엔트로피 항이 이미 내부에서 정규화하고,
프로토타입과 비용 행렬은 주 태스크 gradient 로 학습된다.

## Expert 는 PLE 어디에 꽂히는가

두 Expert 모두 정규화 **644D** 피처 벡터(V1 호환 경로 — 운영 V2 에서는
대신 group 별 feature subset 을 받는다)를 받아 **64D** 를 낸다. v3.2 는
CGC Gate Attention 을 $[B,5]$ 에서 $[B,7]$ 로 넓혀, 기존 PersLay / DeepFM
/ Temporal / Unified H-GCN Expert 와 나란히 받아들였다. 게이트는 그 일곱을
프로젝트의 16개 task tower 에 걸쳐 태스크별로 혼합한다.

Causal 과 OT 를 하나로 융합하지 않고 *분리* 유지하는 이유? 참조서는 세
가지를 든다.

| 이유 | 왜 중요한가 |
| --- | --- |
| Gradient 간섭 | NOTEARS 비순환성($\operatorname{tr}(e^{\mathbf W\odot\mathbf W})=d$)과 Sinkhorn 엔트로피($\varepsilon H(\mathbf P)$)는 손실 곡면이 완전히 다르다 — 동시 학습 시 둘 다 느려짐 |
| 독립 게이팅 | CGC 게이트가 churn 엔 Causal 을, cross-sell 엔 OT 를 높게 가중 가능 — 융합 시 불가능 |
| 교체 용이성 | Causal 은 NOTEARS→GES/PC, OT 는 Sinkhorn→Sliced-Wasserstein 으로 독립 교체 가능 |

세 Expert(DeepFM, Causal, OT)가 *같은* 644D 를 읽되 서로소인 구조를
보탠다 — DeepFM 은 대칭 쌍 상호작용 $\langle\mathbf v_i,\mathbf
v_j\rangle$, Causal 은 비대칭 비순환 방향 $W_{ij}^2$, OT 는 거리 함수
$W(\boldsymbol\mu, \boldsymbol\nu_k)$. 같은 입력, 환원 불가능하게 다른 세
질문.

## 여기서 멈추는 이유

교란 변수 — 프리미엄 카드와 여행 보험 — 에서 출발해 Pearl 의 사다리를
올랐다. 연관, 그다음 gradient 가 DAG 를 학습하게 하는 NOTEARS 의 미분
가능한 비순환성, 그다음 구조 방정식 $\hat{\mathbf z}=\mathbf z+\mathbf
z(\mathbf W\odot\mathbf W)$, 그다음 $\mathbf W^2$ 경로를 고정 vs 전파하는
차이가 DAG 매개 효과를 정확히 분리하는 반사실. 그리고 OT 측으로 건너가 —
분포로서의 고객, Monge–Kantorovich 와 Wasserstein, 그리고 그것을 빠르고
미분 가능하게 만드는 Sinkhorn 의 엔트로피 정규화. 두 Expert, 두 수학적
세계관 — "왜 이 추천인가" 와 "얼마나 잘 맞는가" — 이 하나의 게이트로
들어간다.

남은 것은 *시간* 이다. Causal 과 OT 는 둘 다 고객을 정적 스냅샷으로 읽고,
어느 쪽도 돈의 박자 — 소비가 언제 가속하고, 멈추고, 이탈하는지의 리듬 —
를 보지 못한다. 다음 서브스레드는 **Temporal Expert** 를 다룬다 — 상태공간
모델(Mamba), 연속시간 liquid 신경망, patch Transformer 를 앙상블로 결합한,
한 고객을 위한 세 개의 시계, 그리고 세 가지 다른 "기억" 개념이 하나보다
나은 이유. 이것이 **TEMPORAL-1** 이다.
