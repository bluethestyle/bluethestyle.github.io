---
title: "[Study Thread] HGCN-1 — 그래프를 휘다: 쌍곡 기하학과 푸앵카레 볼 Expert"
date: 2026-06-06 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, hgcn, hyperbolic, graph, poincare, expert]
lang: ko
excerpt: "HGCN 서브스레드 시작 — 평평한 유클리드 공간이 차원 수를 폭발시키지 않고는 트리를 담을 수 없는 이유, 음의 곡률이 어떻게 그래프의 계층을 맞아떨어지게 하는가, 푸앵카레 볼이 실제로 무엇인가, 그리고 프로젝트가 곡면 위에서 그래프 컨볼루션을 하는 방식: log map 으로 각 노드를 접평면에 올리고, 거기서 이웃을 집계하고, exp map 으로 되돌린 뒤, 볼 내부로 안전하게 투영한다. 가맹점 계층 Expert, 프로젝트의 Shared Expert 중 하나."
series: study-thread
part: 14
alt_lang: /2026/06/06/hgcn-hyperbolic-graph-en/
next_title: "CausalOT-1 — 원인을 따라 질량을 옮기다: 최적 수송과 인과 추론의 만남"
next_desc: "다음 Expert 가 상관이 아니라 원인을 읽는 방식: 분포 이동을 보는 최적 수송의 관점, 수송 계획이 왜 반사실 사상인가, 그리고 무작위 실험 없이 고객 행동 그래프 위에 인과 신호를 정초하는 법."
next_status: draft
---

*"Study Thread" 시리즈의 HGCN(쌍곡 그래프 컨볼루션 신경망) 서브스레드
1편. 이번 편부터 영문/국문 병렬로 본 프로젝트의 7개 이종 Shared Expert
중 하나인 가맹점 계층 Expert 를 정리한다. 출처는 온프렘 프로젝트
`기술참조서/GCN_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막 편에
첨부한다. TDA 서브스레드가 Expert 가 어떤* 형태 *를 읽는가를 물었다면,
이번 서브스레드는 그 읽기가 일어나는* 공간 *에 대한 질문을 던진다. 가맹점
그래프는 트리다 — Root → MCC L1 → 업종 소분류(grop) → 업종 L2 리프 —
그리고 트리는,
알고 보면, 평평한 공간에 도무지 들어가지 않는다. Expert 의 답은 공간을
들어맞을 때까지 휘는 것이다.*

> **핵심 한 줄.** 깊이 $d$ 의 완전 이진 트리는 리프가 $2^d$ 개이고, 이를
> 유클리드 공간에 낮은 왜곡으로 채우려면 $O(2^d)$ 차원이 필요하다.
> 프로젝트의 가맹점 계층은 ~550K 노드이며, 구 Brand 리프 레벨(~50,000개)을
> 왜곡 없이 평평한 공간에 임베딩하려면 수만 차원이 든다. **8차원** 푸앵카레
> 볼에서는 낮은 왜곡으로 들어맞는다. 이 한 가지 사실 — 음의 곡률에서
> 나오는 지수적 용량 — 이, 이 Expert 가 $\mathbb{R}^d$ 대신 곡면 공간에
> 사는 이유 전부다.

## 왜 트리는 평평한 공간에 들어가지 않는가

협업 필터링을 위해 고객과 상품을 임베딩할 때, 유클리드 공간
$\mathbb{R}^d$ 은 자연스러운 집이다. "사용자 A 가 아이템 1 을 좋아한다"와
"사용자 B 가 아이템 2 를 좋아한다"는 *대등한* 관계다 — 계층이 없고, 모든
방향이 동등하며, 선형대수가 그대로 적용된다. 프로젝트의 LightGCN 경로가
바로 이를 위해 만들어졌다.

그러나 *가맹점 분류 체계* 는 대등하지 않다. 트리다 — Root → MCC
Level-1(8개) → 업종 소분류(grop, ~35개) → 업종 Level-2(frcs_tind_cd,
~346개, 리프 — 구 Brand, Branch 레벨은 폐기됨). 그리고 트리에는 평평한
공간이 풀 수 없는
기하학적 문제가 있다. 리프 수는 깊이에 따라 $2^d$ 로 늘어나므로 필요한
*공간* 은 지수적으로 늘어나는데 — 유클리드 부피는 반지름에 대해
다항식으로만 늘어난다. 두 증가율이 맞지 않는다. $2^d$ 개의 리프를 같은 간격으로
배치하려면 $O(2^d)$ 차원이 필요하고, 자식들이 서로 밀치며 트리의 메트릭이
뭉개진다.

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="250" fill="#f8fafc" rx="8"/>
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">유클리드 — 공간 ∝ rⁿ</text>
  <circle cx="150" cy="135" r="30" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="58" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="86" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="150" cy="135" r="3" fill="#1e3a5f"/>
  <g fill="#64748b">
    <circle cx="180" cy="135" r="3.5"/><circle cx="121" cy="135" r="3.5"/><circle cx="150" cy="105" r="3.5"/><circle cx="150" cy="165" r="3.5"/>
    <circle cx="208" cy="135" r="3.5"/><circle cx="92" cy="135" r="3.5"/><circle cx="190" cy="178" r="3.5"/><circle cx="110" cy="92" r="3.5"/>
    <circle cx="236" cy="135" r="3.5"/><circle cx="64" cy="135" r="3.5"/><circle cx="212" cy="195" r="3.5"/><circle cx="88" cy="75" r="3.5"/>
  </g>
  <text x="150" y="238" text-anchor="middle" font-size="11" fill="#64748b">리프가 밀친다 — 왜곡</text>
  <line x1="290" y1="40" x2="290" y2="215" stroke="#e2e8f0" stroke-width="1"/>
  <text x="430" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">쌍곡 — 공간 ∝ eʳ</text>
  <circle cx="430" cy="135" r="92" fill="#0d948808" stroke="#0d9488" stroke-width="1"/>
  <circle cx="430" cy="135" r="3" fill="#0d9488"/>
  <g stroke="#0d9488" stroke-width="1"><line x1="430" y1="135" x2="430" y2="60"/><line x1="430" y1="135" x2="495" y2="170"/><line x1="430" y1="135" x2="365" y2="170"/></g>
  <g stroke="#94a3b8" stroke-width="0.8">
    <line x1="430" y1="60" x2="408" y2="48"/><line x1="430" y1="60" x2="452" y2="48"/>
    <line x1="495" y1="170" x2="505" y2="148"/><line x1="495" y1="170" x2="512" y2="190"/>
    <line x1="365" y1="170" x2="348" y2="152"/><line x1="365" y1="170" x2="355" y2="192"/>
  </g>
  <g fill="#0d9488"><circle cx="430" cy="60" r="3.5"/><circle cx="495" cy="170" r="3.5"/><circle cx="365" cy="170" r="3.5"/></g>
  <g fill="#e11d48"><circle cx="408" cy="48" r="3"/><circle cx="452" cy="48" r="3"/><circle cx="505" cy="148" r="3"/><circle cx="512" cy="190" r="3"/><circle cx="348" cy="152" r="3"/><circle cx="355" cy="192" r="3"/></g>
  <text x="430" y="238" text-anchor="middle" font-size="11" fill="#0d9488">모든 레벨이 들어간다 — 낮은 왜곡</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">트리는 깊이에 따라 지수적으로 분기한다. 유클리드 부피는 다항식으로 늘어 자식이 밀치고, 쌍곡 부피는 지수적으로 늘어 트리와 정확히 일치한다. 원점이 루트, 리프는 경계 쪽.</figcaption>
</figure>

## 음의 곡률 — 걸어 나갈수록 팽창하는 공간

해법은 *곡률* 이다. 세 가지 상태를 나란히 놓고 보면 좋다.

| 곡률 | 기하 | 밖으로 이동할 때 일어나는 일 |
| --- | --- | --- |
| 양 | 구 | 공간이 *줄어든다* — 경선이 극에서 수렴 |
| 영 | 유클리드 평면 | 공간이 *균일* — 어디서나 모눈종이 |
| 음 | 쌍곡 공간 | 공간이 *지수적으로 팽창* — 자식을 위한 자리가 계속 생김 |

트리는 깊이에 따라 노드를 지수적으로 더하고, 음의 곡률은 반지름에 따라
자리를 지수적으로 더한다. 두 증가 법칙이 *일치* 한다. 그래서 가맹점
계층이, 평평한 공간이라면 수만 차원을 요구했을 것을, 작은 **8차원** 볼에
낮은 왜곡으로 임베딩된다.

> **역사적 배경.** 쌍곡 기하학은 19세기의 대격변 중 하나다. 2,000년간
> 유클리드의 평행선 공준은 자명해 보였으나, 1829년 Lobachevsky 와 1832년
> Bolyai 가 독립적으로 평행선이 *무한히 많은* 일관된 기하학을 증명했다.
> Poincaré(1882)는 그것에 그림을 주었다 — 단위 원반의 내부 — 이것이
> 머신러닝이 지금도 쓰는 모델이다. 데이터 과학으로의 전환은 Nickel &
> Kiela 의 *Poincaré Embeddings*(NeurIPS 2017)에서 왔는데, WordNet 의
> 계층을 200 유클리드 차원보다 5 쌍곡 차원에 더 충실하게 담았다. 이후
> HGCN(Chami et al., 2019)이 그 아이디어를 그래프 컨볼루션과 결합했다.

## 푸앵카레 볼

계산하려면 쌍곡 공간의 구체적 *모델* 이 필요하다. 프로젝트는 **푸앵카레
볼** 을 쓴다 — 반지름 $1/\sqrt{c}$ 의 열린 공이다.

$$ \mathbb{B}_c^d = \{\, \mathbf{x} \in \mathbb{R}^d : c\,\lVert \mathbf{x} \rVert^2 < 1 \,\} $$

곡률은 $c = 1.0$, 차원은 $d = 8$. 모든 점은 단위 볼 *내부에 엄격히*
존재하고, 경계는 참 쌍곡 메트릭에서 무한히 멀다. 직관은 깔끔하다 —
**원점 = 루트**, **경계 = 리프**. 모든 영역에 두루 소비하는 일반 소비자는
중심 근처에 앉고, 한 브랜드에 집중하는 전문 소비자는 가장자리로 흘러간다.

중심과 가장자리가 그토록 다르게 느껴지는 이유는 *거리 함수* 다. 두 점
$\mathbf{x},\mathbf{y}$ 는 다음으로 떨어져 있다.

$$ d_{\mathbb{B}}(\mathbf{x},\mathbf{y}) = \frac{1}{\sqrt{c}}\,\operatorname{arccosh}\!\left( 1 + \frac{2c\,\lVert \mathbf{x}-\mathbf{y} \rVert^2}{(1 - c\lVert\mathbf{x}\rVert^2)(1 - c\lVert\mathbf{y}\rVert^2)} \right) $$

> **수식 직관.** 분자 $2c\lVert\mathbf{x}-\mathbf{y}\rVert^2$ 은 그저 평범한
> 유클리드 거리의 제곱이다. 분모는 *conformal factor*
> $(1-c\lVert\mathbf{x}\rVert^2)(1-c\lVert\mathbf{y}\rVert^2)$ — 공간을 휘는
> 배율이다. 원점 근처에서는 두 인자가 ≈ 1 이라 거리가 친숙한 유클리드
> 거리로 환원된다: 루트 레벨 카테고리 사이는 이동이 쉽다. 경계 근처에서는
> 한 인자가 $\to 0$ 이라, *같은* 작은 유클리드 간격이 거대한 쌍곡 거리로
> 폭발한다: 서로 다른 브랜드 리프는 실제로 매우 멀다. 기하학이 "트리
> 깊숙한 곳의 형제는 이웃이 아니다"를 공짜로 인코딩한다.

<img src="/poincare-hyperbolic.webp" alt="Poincaré disk model — (a) high-density triangle cell tessellation mesh, (b) geodesic paths: diameter geodesic (straight) and circular arc geodesics (curved, meeting boundary orthogonally)" style="max-width:520px;width:100%;margin:24px auto;display:block;" loading="lazy" />

왼쪽의 타일 분할이 메트릭을 눈에 보이게 한다 — 모든 삼각형은 *같은 쌍곡
크기* 인데도 가장자리로 갈수록 줄어든다. 그것이 지수적 팽창을 그려낸
것이다. 오른쪽에서 측지선(최단 경로)은 중심을 지날 때를 빼면 직선이
아니다 — 중심을 벗어나면 경계와 직각으로 만나는 원호다.

## 두 쌍곡 점을 그냥 더할 수 없는 이유

여기 하류 전체를 좌우하는 함정이 있다. 푸앵카레 볼에서는 일상의 연산이
깨진다. 두 점을 유클리드 방식으로 더하면 결과가 볼 *밖* 으로 떨어질 수
있고, 거기서는 유효한 쌍곡 점조차 아니다. 따라서 이웃의 평범한 가중 평균
— 그래프 컨볼루션의 핵심 — 이 직접 정의되지 않는다.

Riemannian 기하학에서 곧장 나온 표준 해법은, 볼 자체에서는 절대 산술을
하지 않는 것이다. 어느 점에서나 평평한 *접선 공간*
$T_{\mathbf{0}}\mathbb{B}_c^d \cong \mathbb{R}^d$ 을 깔고, 거기서 선형
작업을 한 뒤, 되돌린다.

$$ \mathbf{x} \in \mathbb{B}_c^d \;\xrightarrow{\ \log\ }\; \mathbf{v} \in T_{\mathbf{0}}\mathbb{B}_c^d \;\xrightarrow{\ \text{compute}\ }\; \mathbf{v}' \;\xrightarrow{\ \exp\ }\; \mathbf{x}' \in \mathbb{B}_c^d $$

두 사상이 볼과 그 접평면 사이를 오가며, 프로젝트가 항상 원점 $\mathbf{0}$
에 고정하므로 둘 다 깔끔한 closed-form 을 갖는다. **지수 사상(exp map)**
은 접선 벡터를 측지선을 따라 밖으로 보낸다.

$$ \exp_{\mathbf{0}}(\mathbf{v}) = \tanh\!\big(\sqrt{c}\,\lVert\mathbf{v}\rVert\big)\,\frac{\mathbf{v}}{\sqrt{c}\,\lVert\mathbf{v}\rVert} $$

**로그 사상(log map)** 은 그 역함수로, 볼의 점을 접평면으로 끌어내린다.

$$ \log_{\mathbf{0}}(\mathbf{y}) = \operatorname{arctanh}\!\big(\sqrt{c}\,\lVert\mathbf{y}\rVert\big)\,\frac{\mathbf{y}}{\sqrt{c}\,\lVert\mathbf{y}\rVert} $$

exp map 의 $\tanh$ 가 출력을 $(-1,1)$ 안에 유지함을 보장한다 — 볼에서
떨어져 나갈 수 없다. log map 의 $\operatorname{arctanh}$ 는 경계 근처에서
폭발하여 "이 점은 쌍곡적으로 매우 멀리 나와 있다"를 충실히 보고한다.
새겨둘 만한 구현 메모 하나 — log map 은 인자를 $1-\varepsilon$ 으로
clamp 한다. $\operatorname{arctanh}(1) = \infty$ 가 가장자리로 흘러간 어떤
노드에서든 NaN 을 내기 때문이다.

## 쌍곡 그래프 컨볼루션 — log → 변환 → 집계 → exp

이제 메시지 패싱이다. Chami et al.(2019)은 그 레시피를 한 문장으로 주었다
— *log map 으로 접선 공간에 올리고, 유클리드 그래프 작업을 한 뒤, exp map
으로 되돌린다.* 프로젝트의 `HyperbolicGCNLayer` 가 바로 이것을, 매 레이어
$k$ 마다 다섯 단계로 실행한다.

$$ \mathbf{a}_i^{(k)} = \sum_{j \in \mathcal{N}(i)} w_{ij}\, W^{(k)} \log_{\mathbf{0}}\!\big(\mathbf{x}_j^{(k)}\big), \qquad \mathbf{x}_i^{(k+1)} = \operatorname{proj}\Big(\exp_{\mathbf{0}}\big(\mathbf{a}_i^{(k)}\big)\Big) $$

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="200" fill="#f8fafc" rx="8"/>
  <text x="310" y="26" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">한 HyperbolicGCNLayer (매 레이어 k)</text>
  <rect x="18" y="70" width="92" height="60" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="64" y="62" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">볼</text>
  <text x="64" y="96" text-anchor="middle" font-size="12" fill="#1e3a5f">xᵢ⁽ᵏ⁾</text>
  <text x="64" y="112" text-anchor="middle" font-size="9" fill="#64748b">∈ 𝔹</text>
  <rect x="138" y="70" width="92" height="60" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="184" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 1: log</text>
  <text x="184" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">log₀(x)</text>
  <text x="184" y="112" text-anchor="middle" font-size="9" fill="#64748b">→ 접선</text>
  <rect x="258" y="70" width="92" height="60" rx="6" fill="#fff1f2" stroke="#e11d48" stroke-width="1"/>
  <text x="304" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 2: W (항등초기화)</text>
  <text x="304" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">W v</text>
  <text x="304" y="112" text-anchor="middle" font-size="9" fill="#64748b">선형, bias 없음</text>
  <rect x="378" y="70" width="92" height="60" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="424" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 3: 집계</text>
  <text x="424" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">Σ wᵢⱼ hⱼ</text>
  <text x="424" y="112" text-anchor="middle" font-size="9" fill="#64748b">대칭 정규화</text>
  <rect x="498" y="70" width="104" height="60" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="550" y="62" text-anchor="middle" font-size="9" fill="#64748b">Step 4–5: exp · proj</text>
  <text x="550" y="96" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">exp₀(a)</text>
  <text x="550" y="112" text-anchor="middle" font-size="9" fill="#64748b">→ 볼, clamp</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="110" y1="100" x2="136" y2="100"/><polygon points="136,100 128,96 128,104"/>
    <line x1="230" y1="100" x2="256" y2="100"/><polygon points="256,100 248,96 248,104"/>
    <line x1="350" y1="100" x2="376" y2="100"/><polygon points="376,100 368,96 368,104"/>
    <line x1="470" y1="100" x2="496" y2="100"/><polygon points="496,100 488,96 488,104"/>
  </g>
  <path d="M 550 130 C 550 175, 64 175, 64 132" fill="none" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="307" y="190" text-anchor="middle" font-size="9" fill="#94a3b8">다음 레이어 k+1</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">쌍곡 메시지 패싱: log map 이 각 노드를 평평한 접선 공간에 올리고, 학습 가능한 선형 W 와 대칭 정규화 이웃 합이 유클리드 작업을 하며, exp map 이 볼로 되돌리고, 투영이 모든 것을 엄격히 내부에 둔다.</figcaption>
</figure>

다섯 단계를 처음부터 끝까지 보면,

- **Step 1 — log map.** $\log_{\mathbf{0}}$ 가 모든 노드 임베딩을 곡면 볼
  에서 평평한 접평면으로 올린다. 거기서 선형 연산이 수학적으로 정당하다.
- **Step 2 — 선형 변환.** 학습 가능한 $W^{(k)} =$
  `nn.Linear(dim, dim, bias=False)` 가 접선 벡터를 변형한다. **항등 행렬로
  초기화**(`nn.init.eye_`)되므로, 학습 0 단계에서는 레이어가 과거의
  parameter-free smoothing 과 수학적으로 동치다 — 초기 학습을 안정시키는
  warm start.
- **Step 3 — 집계.** 이웃 $\mathcal{N}(i)$ 에 대한 대칭 정규화 가중 합
  (`scatter_add`)이 그래프 구조를 섞는다. 정규화는 거래 100만 건 브랜드 같은
  허브가 100건짜리를 압도하는 것을 누른다.
- **Step 4 — exp map.** $\exp_{\mathbf{0}}$ 가 집계된 접선 벡터를 볼로
  되돌린다. 내장 $\tanh$ 가 이미 내부에 유지한다.
- **Step 5 — 투영.** 마지막 `proj` 가 부동소수점 드리프트를 clamp 하여 어떤
  점도 경계를 벗어나지 않게 한다.

> **비선형성이 숨은 곳.** 레이어 *내부* 에 ReLU 도 GELU 도 없음에 주목하라.
> 의도된 것이다 — exp map 의 $\tanh$ 와 log map 의 $\operatorname{arctanh}$
> 가 *이미* 비선형이라, 사상 자체가 활성화의 역할을 한다. 추가된 $W$ 는
> 접선 공간에서의 순수 선형 변환이다 — 그 위에 두 번째 명시적 활성화를
> 얹으면 비선형이 겹쳐 gradient vanishing 위험이 생긴다. 기하학이 휘는 일을
> 하고, 가중치는 회전만 한다.

마지막 한 가지 디테일이 옵티마이저를 쌍곡-인식형으로 만든다. 평범한
유클리드 gradient 는 공간의 팽창을 무시하고 경계 근처에서 과도하게 나아가
점을 볼 밖으로 민다. 해법은 gradient 에 메트릭 텐서의 역수를 곱하는 것이다.

$$ \nabla_{\text{Riem}}\,f(\mathbf{x}) = \frac{(1 - c\lVert\mathbf{x}\rVert^2)^2}{4}\,\nabla_{\text{Euclid}}\,f(\mathbf{x}) $$

원점 근처에서 계수는 ≈ ¼, 경계 근처에서는 $\to 0$. 평이하게 옮기면 —
*가장자리에 가까울수록 조심스럽게 한 걸음 디뎌라.* 이 보정이 없으면
프로젝트의 `_train_gcn()` 은 첫 epoch 들에서 NaN 을 낸다.

## LightGCN 과의 관계 — 같은 골격, 다른 공간

프로젝트는 두 GCN 경로를 돌리고, 둘은 경쟁자가 아니라 형제다. 기하학을
걷어내면 둘 다 *같은* LightGCN 골격을 유지한다 — 레이어 내 비선형 없음,
이웃 평균, 그리고 over-smoothing 을 막기 위한 전 레이어 평균
$\frac{1}{L+1}\sum_{k=0}^{L}\mathbf{x}^{(k)}$. 다른 것은 *그래프* 와
*공간* 이다.

| | LightGCN | H-GCN (이 Expert) |
| --- | --- | --- |
| 노드 | 고객 + 가맹점 (이분 그래프) | 가맹점만 (MCC 계층 트리) |
| 엣지 | 고객 ↔ 가맹점 거래 | 부모 ↔ 자식 업종 계층 (구 브랜드 ↔ 브랜드 공동방문 간선은 폐기) |
| 학습 | "누가 무엇을 좋아하는가" (협업) | "가맹점끼리 구조적으로 어떻게 관련되는가" |
| 공간 | 유클리드 $\mathbb{R}^{64}$ | 푸앵카레 볼 $\mathbb{B}^{8}$ |
| 출력 | 64D 고객 임베딩 (직접) | 가맹점 임베딩 → 고객별 47D (간접) |

두 신호는 상호 보완적이다 — LightGCN 은 협업 행동에서 개인화하고, H-GCN 은
cold-start 와 sparse 고객을 메우는 구조적 가맹점 관계를 공급한다. 참조서의
주의 하나는 거듭 새길 만하다 — **H-GCN 은 협업 필터링이 아니다.** 거래
기반 *공동방문 엣지* 가 존재하던 시절에도(brand 리프와 함께 폐기됨) 그것은
가맹점-가맹점 기하를 미세 보정할 뿐이었고, LightGCN 처럼 고객의 선호를
학습하지는 않는다.

> 운영 상태에 대한 메모. 2026-04-24 부로 LightGCN 경로는 *임시 비활성*
> 이다(Stage 1 `collaborative_embeddings` 산출물 부재). 그래서 현재 활성
> Shared Expert 집합은 6개이고, 합산은 512D 가 아니라 448D 다. 여기서
> 설명한 쌍곡 경로, 즉 `unified_hgcn` Expert 는 활성이다. 위 비교는 설계
> 계약이지, 둘 다 오늘 돈다는 주장이 아니다.

## HGCN Expert 가 PLE 에 꽂히는 위치

그래프 메시지 패싱은 비싸다 — ~550K 노드 전체 그래프를 메모리에 올려야
한다 — 그래서 프로젝트는 일을 둘로 나눈다. Pinterest PinSage 가 대중화한
바로 그 Stage-1/Stage-2 패턴이다.

1. **Stage 1 (오프라인, Airflow 배치).** `HierarchyEmbeddingGenerator`
   가 가맹점 트리 위 자기지도학습으로 쌍곡 GCN 을 학습한 뒤, 고객별
   **47D** 임베딩(Output A 20D + Output B 27D, 둘 다 가중 푸앵카레 평균으로
   구성)을 Parquet 으로 freeze 한다. 무겁고, 태스크 단위가 아니라 그래프
   갱신 주기에 맞춰 한 번만.
2. **Stage 2 (배치 학습/서빙).** `UnifiedHGCNExpert` 는 그 47D 벡터를
   그냥 *룩업* 하고 경량 bottleneck `refine_mlp` — residual 을 둔
   `Linear(47→128) → GELU → Linear(128→47)` — 를 돌린 뒤, `output_proj`
   로 **128D** Expert 표현을 낸다. 추론 시 그래프 전파 없음, 그래프 크기와
   무관한 일정 비용.

그 128D 출력은 **PLE CGC gate** 로 들어가는 하나의 Shared Expert 이고,
gate 가 그것을 **15개** 전 태스크에 걸쳐 다른 Expert 들과 태스크별로
혼합한다. 임베딩이 가맹점 계층을 담고 있으므로, 고객이 *무엇* 을 사고
*얼마나 전문화* 되어 있는가를 건드리는 모든 태스크에 구조적 신호를 보탠다
— 깊이 지표(원점 거리)가 같은 임베딩 안에 함께 실려 간다. Expert 자체는
**임베딩 전용** 이다 — 과거의 브랜드 계층 예측 헤드는 brand_prediction
태스크 폐기와 함께 제거되었다.

## 여기서 멈추는 이유

평평한 공간이 트리를 도무지 담지 못한다는 불편함에서 출발해, 음의 곡률과
그것을 실현하는 푸앵카레 볼을 따라갔고, 쌍곡 거리가 왜 루트-카테고리를
가깝게 브랜드-리프를 멀게 만드는지 봤으며, 곡면 위에서 그래프 컨볼루션을
가능케 하는 한 수 — log map 으로 올리고, 평평한 접평면에서 선형 그래프
작업을 하고, exp map 으로 되돌린 뒤, 안전하게 내부로 투영 — 를 풀었다.
그리고 Expert 를 배치했다 — Stage 1 에서 freeze 한 47D 임베딩을 128D 로
refine 하여 15개 전 태스크로 gating.

남은 것은 기하학의 다른 절반이다. 여기서 우리는 *구조* 를 임베딩했다 —
그러나 구조는 가맹점들 사이의 상관일 뿐이다. 다음 서브스레드는 *원인* 을
읽는 Expert 로 향한다 — **CausalOT**, 최적 수송이 분포 이동을 반사실로
바꾸는 사상을 공급하고, 무작위 실험을 한 번도 돌리지 않은 채 고객 그래프
위에 인과 신호를 정초하는 곳. 이것이 다음 편 **CausalOT-1** 의 주제다.
