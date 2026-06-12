---
title: "[Study Thread] DEEPFM-1 — 피처 상호작용: Factorization Machine 과 임베딩 공유라는 트릭"
date: 2026-06-06 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, deepfm, factorization-machine, feature-interaction, expert]
lang: ko
excerpt: "DeepFM 서브스레드 시작 — 선형 모델이 놓치는 피처 사이의 시너지, Factorization Machine 이 O(n²) 교차 파라미터 대신 O(nk) 잠재 벡터로 모든 쌍의 상호작용을 학습하는 방식, O(nk) 선형화 트릭, 그리고 DeepFM 이 FM 헤드와 deep 네트워크 사이에서 하나의 임베딩을 공유하는 구조. 이 Expert 를 PLE 에 연결하는 필드 구성과 출력 차원과 함께."
series: study-thread
part: 13
alt_lang: /2026/06/06/deepfm-feature-interaction-en/
next_title: "HGCN-1 — 계층을 위한 굽은 공간: Hyperbolic Graph Convolution"
next_desc: "가맹점과 고객의 트리가 왜 평평한 유클리드 공간에 깔끔히 임베딩되지 않는가, 음의 곡률이 어떻게 지수적 공간을 벌어주는가, 그리고 unified_hgcn Expert 가 Poincaré 볼에서 그래프 구조를 읽어 128D 를 PLE 로 되돌려주는 방식."
next_status: draft
---

*"Study Thread" 시리즈의 DeepFM(Factorization Machine + deep network)
서브스레드 1편. 이번 편부터 영문/국문 병렬로 본 프로젝트의 이종 Shared
Expert 중 하나인 DeepFM Expert 를 정리한다. 출처는 온프렘 프로젝트
`기술참조서/DeepFM_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막 편에
첨부한다. TDA 서브스레드가 Expert 가 어떤* 형태 *를 읽는가를 물었다면,
이번 서브스레드는 더 기본적이면서 그만큼 틀리기 쉬운 질문을 던진다 — 두
피처가 오직* 함께 *일 때만 의미를 가질 때, 즉 RFM 이 높은 고객이* 동시에
*디지털에 기우는 경우, 모델은 어떻게 피처의* 곱 *을 파라미터에 파묻히지
않고 볼 수 있는가? DeepFM 의 답은, 얕은 절반과 깊은 절반이 공유하는
분해된 내적이다.*

> **하나의 임베딩, 두 독자.** DeepFM 의 핵심 트릭은 단 하나의 필드
> 임베딩 집합을 병렬로 *두 번* 소비한다는 것이다 — 한 번은 명시적 2차
> 상호작용을 읽는 Factorization Machine 이, 한 번은 암묵적 고차
> 상호작용을 읽는 deep MLP 가. 별도의 피처 파이프라인도, 손으로 설계한
> 교차 피처도(구글의 Wide & Deep 이 여전히 짊어졌던 골칫거리) 없으며,
> end-to-end 그래디언트가 양쪽에서 *같은* 잠재 벡터로 흘러든다. 이
> 프로젝트의 Expert 는 논문보다 한 걸음 더 간다 — 스칼라 CTR 이 아니라
> PLE gate 로 들어가는 **64D 표현 벡터** 를 출력한다.

## 선형 모델의 한계

추천 모델의 일은 직설적이다 — *이* 고객이 *이* 행동을 할 것인가? 개별
피처(연령대, 상품 카테고리, 채널 활용도)도 그 자체로 신호를 담는다.
그러나 실제 행동은 피처 *사이* 의 관계에서 산다. "30대" 는 조금
말해준다. "디지털에 기운다" 도 조금 말해준다. "30대 *이면서* 디지털에
기운다" 는 둘 중 어느 것도 단독으로 말하지 못하는 것을 말해준다 — 둘이
함께 켜질 때만 나타나는 온라인 투자상품 전환율의 급증.

선형 모델은 이를 볼 수 없다. 피처 $i$ 를 $x_i$ 로 쓰면,

$$ \hat{y} = w_0 + \sum_{i=1}^{n} w_i\, x_i $$

는 모든 피처가 *독립적* 으로 가산 기여한다고 가정한다. 두 피처가 동시에
나타날 때만 켜지는 항이 없다. 시너지를 잡으려면 교차항 $x_i x_j$ 를
명시적으로 더해야 하고, 그러면 모델은 2차 다항 회귀가 된다.

$$ \hat{y} = w_0 + \sum_{i=1}^{n} w_i\, x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} w_{ij}\, x_i x_j $$

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <text x="140" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">선형 — 독립적 합</text>
  <g>
    <rect x="55" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <rect x="122" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <rect x="189" y="55" width="36" height="36" rx="5" fill="#64748b22" stroke="#64748b" stroke-width="1"/>
    <text x="73" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₁</text>
    <text x="140" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₂</text>
    <text x="207" y="79" text-anchor="middle" font-size="12" fill="#64748b">x₃</text>
    <text x="105" y="80" text-anchor="middle" font-size="14" fill="#64748b">+</text>
    <text x="172" y="80" text-anchor="middle" font-size="14" fill="#64748b">+</text>
  </g>
  <text x="140" y="135" text-anchor="middle" font-size="11" fill="#64748b">x·x 항 없음 — 시너지 안 보임</text>
  <line x1="280" y1="40" x2="280" y2="180" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">FM — 쌍별 격자</text>
  <g fill="#0d9488"><circle cx="360" cy="62" r="7"/><circle cx="420" cy="62" r="7"/><circle cx="480" cy="62" r="7"/></g>
  <g stroke="#0d9488" stroke-width="1.4">
    <line x1="360" y1="62" x2="420" y2="62"/>
    <line x1="420" y1="62" x2="480" y2="62"/>
    <path d="M 360 68 Q 420 110 480 68" fill="none"/>
  </g>
  <text x="360" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₁</text>
  <text x="420" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₂</text>
  <text x="480" y="50" text-anchor="middle" font-size="11" fill="#0d9488">x₃</text>
  <text x="390" y="56" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₁,v₂⟩</text>
  <text x="450" y="56" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₂,v₃⟩</text>
  <text x="420" y="108" text-anchor="middle" font-size="9" fill="#0d9488">⟨v₁,v₃⟩</text>
  <text x="420" y="135" text-anchor="middle" font-size="11" fill="#64748b">모든 쌍이 학습된 가중치를 가짐</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">선형 모델은 피처를 고립시켜 더하고, FM 은 모든 쌍에 가중치를 더한다. 문제는 그 가중치를 어떻게 감당하느냐다.</figcaption>
</figure>

## 조합 폭발

교차항을 더하는 것은 옳지만 비싸다. $n$ 개 피처가 있으면 교차 파라미터
$w_{ij}$ 는 $n(n-1)/2$ 개다. 이 프로젝트의 정규화 피처 공간 — V1 호환
구성의 **644D** — 에서는

$$ \frac{644 \times 643}{2} = 207{,}046 $$

개의 교차 파라미터가 필요하다. 개수보다 나쁜 것은 *희소성* 이다. 대부분의
$(i,j)$ 쌍이 학습 데이터에 거의 함께 등장하지 않아 $w_{ij}$ 를 안정적으로
추정할 수 없다. 교차 파라미터 행렬 $W \in \mathbb{R}^{n\times n}$ 은
대칭이고($w_{ij}=w_{ji}$), 고차원 공간에서 "모든 쌍" 은 이차적으로
증가한다 — 차원의 저주의 한 단면이다.

> **역사적 배경.** 해법은 추천 시스템의 행렬 분해에서 곧장 내려온다.
> 2006년 Netflix Prize 가 SVD 계열 MF 를 협업 필터링의 일꾼으로 만들었고
> (Funk 의 공개 SVD, 2009년 Koren 의 BellKor 앙상블), Steffen Rendle 이
> "사용자 × 아이템" 을 "임의의 피처 쌍" 으로 일반화해 2010년 ICDM 에서
> *Factorization Machines* 를 발표했다. FFM(2016), Wide & Deep(2016),
> DeepFM(Guo et al., IJCAI 2017) 이 뒤따랐다 — 수작업 교차 피처에서 완전
> 자동 상호작용 학습으로 이어지는 계보다.

## 분해 — O(n²) 에서 O(nk) 로

행렬 분해의 교훈 — 저랭크 곱이 큰 상호작용 행렬을 근사한다. FM 은 이를
교차 파라미터 행렬에 직접 적용한다. 각 $w_{ij}$ 를 저장하는 대신, 모든
피처 $i$ 에 *잠재 벡터* $\mathbf{v}_i \in \mathbb{R}^k$ 를 부여하고

$$ w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{i,f}\, v_{j,f}, \qquad W \approx V V^{\!\top} $$

로 근사한다. 이는 정확히 대칭 행렬 $W$ 의 랭크-$k$ 근사다 — truncated
SVD 에서 상위 $k$ 개 특이값만 남기는 것과 같은 아이디어(Eckart–Young)이며,
다만 FM 은 데이터로부터 적응적으로 학습한다. FM 전체 예측은 이렇게 된다.

$$ \hat{y}_{\text{FM}} = w_0 + \sum_{i=1}^{n} w_i\, x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j $$

효과는 극적이다. 이 프로젝트에서 $k=16$ 으로, 교차 파라미터는 ~207K 에서
$n \times k = 644 \times 16 = 10{,}304$ 로 — 약 **20배 감소** 한다.

| 접근법 | 교차 파라미터 수 | 희소 데이터 거동 |
| --- | --- | --- |
| 2차 다항 회귀 | $n(n-1)/2 = 207{,}046$ | 실패 — 대부분 쌍 미관측 |
| FM ($k=16$) | $n \times k = 10{,}304$ | 강건 — 각 $\mathbf{v}_i$ 가 공유됨 |

왜 공유가 희소성을 구하는가? $\mathbf{v}_i$ 가 피처 $i$ 의 *모든*
공동출현에 걸쳐 학습되기 때문이다. 쌍 $(i,j)$ 가 한 번도 함께 등장하지
않더라도, $\mathbf{v}_i$ 는 $(i,k)$ 에서, $\mathbf{v}_j$ 는 $(j,l)$ 에서
학습되었다면 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 는 여전히 의미
있는 추정치를 준다. 내적은 또한 깔끔한 해석을 가진다 —
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \lVert\mathbf{v}_i\rVert\,\lVert\mathbf{v}_j\rVert\cos\theta_{ij}$
로, 양수는 *시너지*, 음수는 *억제*, 0 근처는 *상호작용 없음* 이다.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="230" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">상호작용 = 잠재 벡터 내적</text>
  <circle cx="100" cy="180" r="3" fill="#64748b"/>
  <text x="92" y="198" text-anchor="middle" font-size="10" fill="#64748b">0</text>
  <line x1="100" y1="180" x2="200" y2="90" stroke="#0d9488" stroke-width="2"/>
  <line x1="100" y1="180" x2="225" y2="110" stroke="#0d9488" stroke-width="2"/>
  <text x="205" y="84" font-size="11" fill="#0d9488" font-weight="700">vᵢ</text>
  <text x="233" y="112" font-size="11" fill="#0d9488" font-weight="700">vⱼ</text>
  <path d="M 130 153 A 42 42 0 0 1 142 142" fill="none" stroke="#0d9488" stroke-width="1"/>
  <text x="160" y="150" font-size="10" fill="#0d9488">θ 작음 → ⟨·,·⟩ &gt; 0 시너지</text>
  <line x1="100" y1="180" x2="240" y2="180" stroke="#e11d48" stroke-width="2" stroke-dasharray="1 0"/>
  <line x1="100" y1="180" x2="120" y2="120" stroke="#e11d48" stroke-width="2"/>
  <text x="248" y="184" font-size="11" fill="#e11d48" font-weight="700">vₚ</text>
  <text x="108" y="114" font-size="11" fill="#e11d48" font-weight="700">v_q</text>
  <text x="300" y="180" font-size="10" fill="#e11d48">θ 큼 → ⟨·,·⟩ &lt; 0 억제</text>
  <text x="260" y="218" text-anchor="middle" font-size="11" fill="#64748b">⟨vᵢ, vⱼ⟩ = ‖vᵢ‖ ‖vⱼ‖ cos θᵢⱼ</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">각 필드는 16D 잠재 공간의 한 점이 된다. 같은 방향 벡터는 양의 상호작용(시너지), 반대 방향 벡터는 음의 상호작용(억제).</figcaption>
</figure>

## FM 트릭 — 한 번의 패스로 O(nk)

수식 안에 아직 문제가 숨어 있다. 순진하게 합산하면 쌍별 항은
$n(n-1)/2$ 개의 내적이 필요하고, 각각 $k$ 차원에 걸쳐 — $O(n^2 k)$ 다.
하나의 대수적 항등식이 이를 $O(nk)$ 로 무너뜨린다.

$$ \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i x_j = \frac{1}{2}\sum_{f=1}^{k}\left[\left(\sum_{i=1}^{n} v_{i,f}\, x_i\right)^{\!2} - \sum_{i=1}^{n} \left(v_{i,f}\, x_i\right)^{2}\right] $$

> **수식 직관.** 항등식은 단지
> $\sum_{i<j} a_i a_j = \tfrac{1}{2}\big[(\sum_i a_i)^2 - \sum_i a_i^2\big]$
> 에 $a_i = v_{i,f}\,x_i$ 를 대입한 것이다. *합의 제곱* 에는 모든 자기항
> 과 모든 교차항의 2배가 들어 있고, *제곱의 합*(자기항)을 빼면 교차항의
> 2배만 남는다. 따라서 잠재 차원 $f$ 마다 합을 하나 구해 제곱하고, 제곱의
> 합을 빼면 — 선형 패스 두 번이면 — 모든 쌍별 상호작용을 얻는다.
> "전체를 계산한 뒤 자기 상호작용을 빼기."

코드에서 이것은 `sum_sq − sq_sum` 이다. 그리고 프로젝트 고유의 변형
하나가 중요하다 — 원본 FM 은 모든 $k$ 채널을 하나의 스칼라로 합친다. 이
Expert 는 *k차원 벡터를 그대로 유지* 한다. 16개 채널 각각이 필드 시너지의
서로 다른 관점을 담는다.

$$ \mathbf{y}_{\text{FM}} = \frac{1}{2}\left[\left(\sum_{i=1}^{n} \mathbf{v}_i\right)^{\!2} - \sum_{i=1}^{n} \mathbf{v}_i^{2}\right] \in \mathbb{R}^{k},\qquad k=16 $$

## 왜 FM 만으로는 부족한가 — 깊은 절반

FM 은 *2차* 상호작용만, 오직 그것만 모델링한다.
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ 는 두 피처의 선형
교차이고, 3차 이상의 패턴 — "RFM 고점 *이면서* 디지털 활용 高 *이면서*
거시 불확실성 高 → 안전자산 선호 급증" — 은 닿을 수 없다. deep MLP 가 그
보완이다. 범용 함수 근사기로서 *암묵적* 고차 비선형 상호작용을 학습한다.
다만 단순한 2차 패턴조차 우회적으로 복원하려면 많은 파라미터가 든다.

| 역량 | FM | Deep Network |
| --- | --- | --- |
| 2차 상호작용 | 명시적, 효율적 | 가능하지만 낭비적 |
| 고차 상호작용 | 불가능 | 암묵적으로 학습 |
| 비선형성 | 내적만 가능 | ReLU 등 |
| 파라미터 효율 | 매우 높음 ($O(nk)$) | 낮음 |
| 해석 가능성 | 쌍별 기여 추적 | 블랙박스 |

깊은 절반은 *플래튼* 된 필드 임베딩을 받아 3층 MLP 를 돌린다. 각 층은
선형 → BatchNorm → ReLU → dropout 으로, 깊어질수록 표현을 좁힌다.

$$ \mathbf{h}^{(l+1)} = \mathrm{ReLU}\big(\mathrm{BN}(W^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)})\big),\qquad \mathbf{h}^{(0)} = \mathrm{flatten}([\mathbf{v}_1;\dots;\mathbf{v}_n]) \in \mathbb{R}^{nk} $$

$n=28$ 필드와 $k=16$ 으로 플래튼 입력은 $28\times16=448$D 이고,
$448 \to 256 \to 128 \to 64$ 로 압축된다.

## DeepFM — 하나의 임베딩, 공유

모델에 이름을 준 구조적 한 수 — FM 과 Deep 은 별도의 임베딩을 **갖지
않는다**. 하나의 필드 임베딩 집합을 공유해 병렬로 읽고, 함께 그 안으로
역전파한다. 이것이 Wide & Deep 의 수작업 교차 피처 파이프라인을
제거하고, 두 절반이 일관되게 유지되는 이유다.

필드 자체가 프로젝트의 기여다. 644D 벡터는 **28개 의미 필드** 로
슬라이싱된다 — `rfm`(34D), 4개로 분할된 category 필드
(`customer_cat`/`product_cat`/`region_cat`/`channel_cat`, 각 16D),
`transaction`(80D), `deposit`, `investment`, `mamba`(50D), `economics`,
`merchant_hierarchy`(21D) 등 — 각각 자기 `nn.Linear(dᵢ, 16)` 으로 16D 에
프로젝션된다. 이로써 임베딩 파라미터는
$\sum_i d_i \times 16 = 644\times16 = 10{,}304$ 개, 28 필드는
$28\times27/2 = 378$ 가지 쌍별 FM 상호작용을 준다.

> **역사적 배경.** 기존 64D `category` 블록을 4개의 16D 서브필드로 나눈
> 것(v3.11)은 의도적이다 — FM 은 필드 *내부* 를 교차하지 않으므로, 단일
> category 필드는 `product_cat × region_cat` 과 `customer_cat × channel_cat`
> 상호작용을 숨겼다. 4개 서브필드가 거의 0에 가까운 파라미터 비용으로
> 이를 드러낸다(351 → 378 쌍). 더 큰 도약은 피처별 `nn.Embedding` 에서
> 필드별 `nn.Linear` 로의 전환이었고, 이는 Deep 입력을 10,304D 에서 448D
> 로, Expert 를 약 **10.9M 에서 ~169K** 파라미터로 줄였다 — MLP 의 98%
> 감소.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="300" fill="#f8fafc" rx="8"/>
  <rect x="230" y="18" width="140" height="34" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="300" y="35" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">x  [B, 644]</text>
  <text x="300" y="47" text-anchor="middle" font-size="9" fill="#64748b">정규화 644D (V1 호환)</text>
  <rect x="210" y="76" width="180" height="40" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="300" y="93" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Field Embeddings [B, 28, 16]</text>
  <text x="300" y="107" text-anchor="middle" font-size="9" fill="#64748b">28 × Linear(dᵢ → 16) — 공유</text>
  <line x1="300" y1="52" x2="300" y2="76" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="300,76 296,68 304,68" fill="#cbd5e1"/>
  <line x1="260" y1="116" x2="150" y2="146" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="150,146 158,142 156,150" fill="#cbd5e1"/>
  <line x1="340" y1="116" x2="450" y2="146" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="450,146 442,142 444,150" fill="#cbd5e1"/>
  <rect x="55" y="148" width="160" height="56" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="135" y="168" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">FM Layer</text>
  <text x="135" y="183" text-anchor="middle" font-size="9" fill="#64748b">sum_sq − sq_sum</text>
  <text x="135" y="196" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="700">[B, 16]</text>
  <rect x="385" y="148" width="160" height="56" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="465" y="166" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Deep Network</text>
  <text x="465" y="180" text-anchor="middle" font-size="9" fill="#64748b">flatten 448 → 256 → 128</text>
  <text x="465" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f" font-weight="700">[B, 64]</text>
  <line x1="135" y1="204" x2="270" y2="232" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="270,232 262,229 263,237" fill="#cbd5e1"/>
  <line x1="465" y1="204" x2="330" y2="232" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="330,232 338,229 337,237" fill="#cbd5e1"/>
  <rect x="220" y="234" width="160" height="32" rx="6" fill="#1e3a5f11" stroke="#1e3a5f" stroke-width="1"/>
  <text x="300" y="254" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">concat [FM ; Deep] → [B, 80]</text>
  <line x1="300" y1="266" x2="300" y2="282" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="210" y="282" width="180" height="14" rx="4" fill="#0d9488"/>
  <text x="300" y="293" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Linear(80→64) → LN → SiLU  [B, 64]</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">임베딩 공유 아키텍처: 하나의 [B,28,16] 임베딩이 FM 헤드(16D)와 Deep 네트워크(64D) 양쪽에 공급되고, 둘의 연결이 64D Expert 출력으로 프로젝션된다.</figcaption>
</figure>

두 절반은 연결되어 최종 표현으로 프로젝션된다.

$$ \mathbf{y}_{\text{DeepFM}} = \mathrm{SiLU}\big(\mathrm{LN}(W_{\text{out}}\,[\,\mathbf{y}_{\text{FM}}\,;\,\mathbf{y}_{\text{Deep}}\,] + \mathbf{b}_{\text{out}})\big) \in \mathbb{R}^{64} $$

여기서 $[\,\mathbf{y}_{\text{FM}}\,;\,\mathbf{y}_{\text{Deep}}\,]$ 는
$16+64 = 80$D 연결이고 $W_{\text{out}} \in \mathbb{R}^{64\times80}$ 이다.
MTL 을 염두에 둔 두 선택이 논문의 기본값을 대체한다 — 태스크 간 스케일
안정화를 위한 sigmoid 출력 대신 **LayerNorm**, 그리고 ReLU 대신 **SiLU**
($x\,\sigma(x)$). SiLU 는 음수를 죽이지 않으므로 FM 의 *억제* 신호(음의
내적)가 출력까지 살아남는다.

## DeepFM Expert 의 위치

64D 벡터가 온프렘 확장의 핵심이다 — 논문은 스칼라 CTR 을 내보내지만, 이
Expert 는 표현을 내보낸다. `ple_cluster_adatt.py` 안에서
`_forward_shared_experts()` 가 피처 텐서를 DeepFM 에 건네고(등록 시
`FeatureRouter` 로 라우팅, 아니면 full features), 64D 출력은 다른 Shared
Expert 들과 함께 PLE CGC gate 에서 만나 태스크별로 혼합된다.

| 단계 | 연산 | 출력 차원 |
| --- | --- | --- |
| 1 | 입력 | `[B, 644]` |
| 2 | 28필드 슬라이스 + 임베딩 | `[B, 28, 16]` |
| 3a | FM: sum_sq − sq_sum | `[B, 16]` |
| 3b | flatten | `[B, 448]` |
| 4 | Deep: 448→256→128→64 | `[B, 64]` |
| 5 | concat [FM ; Deep] | `[B, 80]` |
| 6 | output: Linear→LN→SiLU | `[B, 64]` |
| 7 | interpret projection | `[B, 4]` |

DeepFM 옆에서 두 Expert 가 *같은* 644D 를 읽고 역시 64D 를 낸다 — Causal
Expert(비대칭 DAG, $W_{ij}\neq W_{ji}$)와 Optimal-Transport Expert(거리,
$W(\mu,\nu)\geq 0$). DeepFM 의 기여는 둘이 줄 수 없는 구조다 — *대칭* 내적
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$. CGC gate 가 이 관점들을
태스크별로 가중하므로, LTV 태스크는 DeepFM 의 교차 패턴에 기대고 churn
태스크는 인과 구조에 기댈 수 있다. `domain_experts: ["deepfm"]` 로
지정된 태스크 — `ltv`, `spending_bucket` 등 — 는 DeepFM 쪽으로 높은 초기
gate bias 를 받는다. 마지막 `Linear(64→4)` 프로젝션이 SAE 분석용으로
저차/고차, 희소/밀집 상호작용 채널을 드러낸다.

## 여기서 멈추는 이유

가산 선형 모델에 대한 불편함에서 출발해, 교차 파라미터 수가 ~207K 로
폭발하는 것을 봤고, 분해가 $O(nk)$ 의 공유 잠재 벡터로 그것을 구하는
것을 봤다 — 그리고 그조차 한 번의 패스로 만드는 대수적 트릭까지. DeepFM
을 FM 절반(명시적 쌍별 시너지 16D)과 깊은 절반(암묵적 고차 패턴 64D)으로
나누고, 둘이 공유하는 하나의 임베딩을 봤다. 끝으로 64D Expert 를 Causal,
OT 옆 PLE gate 에 놓았다 — 각자 같은 피처를 다른 수학적 렌즈로 읽으면서.

남은 것은 기계 장치와 대안이다. `FMLayer` 와 `DeepNetwork` 가 코드에서
어떻게 조합되는가, 암묵적 MLP 로 부족할 때를 위해 이 프로젝트가 왜
*명시적* 고차 교차를 위한 **DCNv2** Expert 도 함께 싣는가, 그리고 필드
상호작용 분석이 실제로 어떤 쌍이 시너지를 내는지 어떻게 읽어내는가. 그러나
평평한 피처 상호작용 공간을 더 파기 전에, 다음 서브스레드는 그곳을 아예
떠난다 — *굽은* 공간으로. **HGCN-1** 은 가맹점과 고객의 계층이 왜 유클리드
공간에 들어맞지 않는가, 그리고 Poincaré 볼의 음의 곡률이 어떻게
`unified_hgcn` Expert 에게 128D 를 PLE 로 되돌려주기 전 지수적 공간을
벌어주는가를 묻는다.
