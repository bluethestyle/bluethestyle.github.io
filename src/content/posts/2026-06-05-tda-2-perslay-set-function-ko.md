---
title: "[Study Thread] TDA-2 — 집합 함수로서의 PersLay: φ, w, ρ 와 5-Block 아키텍처"
date: 2026-06-05 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, perslay, deepsets, set-function, expert]
lang: ko
excerpt: "TDA 서브스레드 완결편 — 크기가 제각각이고 순서도 없는 (birth, death) 점 집합이 어떻게 하나의 고정 64D 벡터가 되는가. DeepSets 의 더하기 트릭, RationalHat 점 변환, 노이즈를 공짜로 무시하는 persistence 가중치, 집계 방식의 선택, 그리고 PersLay 를 하나가 아니라 다섯 블록으로 돌리는 이유. 지금 실제로 도는 경로에 대한 정직한 각주와 함께."
series: study-thread
part: 12
alt_lang: /2026/06/05/tda-2-perslay-set-function-en/
next_title: "DEEPFM-1 — 피처 상호작용: Factorization Machine 과 임베딩 공유라는 트릭"
next_desc: "Expert 서브스레드는 DeepFM 으로 이어진다 — 선형 모델이 피처 상호작용을 볼 수 없는 이유, factorization machine 이 207,046개의 쌍별 가중치를 10,304개의 공유 임베딩 파라미터로 압축하는 방식, 그리고 하나의 임베딩이 FM 헤드와 deep 타워 양쪽에 공급되는 구조."
next_status: published
---

*"Study Thread" 시리즈의 TDA / PersLay 서브스레드 2편이자 완결편. 출처는
온프렘 프로젝트 `기술참조서/PersLay_기술_참조서` 이다. TDA-1 은 논거를
세웠다 — 소비에는 형태가 있고, persistence diagram 이 그것을 포착하며,
검증 런이 신호가 실재한다고 답했다. 미뤄둔 것은 기계 장치다. 이번 편이
그것을 연다: 순서 없고 크기도 제각각인 (birth, death) 점 집합이 정확히
어떻게 신경망이 소비할 수 있는 하나의 고정 64차원 벡터가 되는가? 답은
하나의 수식을 세 번 — 학습 가능한 조각마다 한 번씩 — 읽는 것이고, 그
다음 다섯을 곱하는 것이다.*

> **지금 어디인가.** TDA-1 은 문 앞에서 끝났다: persistence diagram 이
> 소비 형태의 올바른 요약이지만, diagram 은 벡터가 아니고 신경망은
> 그것을 날것으로 먹지 못한다. 이번 편이 끝나면 다리 역할을 하는 수식
> $F(D) = \rho(\sum w \cdot \phi)$ 가 기호가 아니라 세 개의 독립된 설계
> 결정으로 읽혀야 한다 — *점 하나를 어떻게 번역하는가* ($\phi$), *그
> 점을 얼마나 신뢰하는가* ($w$), *점들을 어떻게 하나로 합치는가*
> ($\rho$).
> 셋 모두 학습 가능하고, 셋 모두 이 프로젝트가 이유를 댈 수 있는
> 선택이다.

## 문제를 정확하게 다시 쓰기

한 번 더, 이번엔 정밀하게. Persistence diagram 은 $(b, d)$ 점들의
집합이고, 세 가지 지점에서 신경망에 저항한다.

1. **크기가 가변이다.** 어떤 고객의 diagram 은 점이 12개, 다른 고객은
   47개다. MLP 입력층의 폭은 고정이다. 둘을 모두 받는 슬롯 배치는
   존재하지 않는다.
2. **순서가 없다.** 점들은 자루이지 수열이 아니다. 벡터로 펼치는 순간
   순서를 *발명* 하게 된다 — 그러면 같은 diagram 을 다른 순서로 펼친
   것이 다른 벡터가 된다. 그 표현으로 학습한 모델은 형태가 아니라
   부산물을 배운다.
3. **사는 공간이 다르다.** Diagram 은 bottleneck 과 Wasserstein 거리로
   재는 메트릭 공간에 살지, $\mathbb{R}^n$ 에 살지 않는다. 표준 레이어의
   어떤 부분도 그 기하를 존중하지 않는다.

어떤 해법이든 **고정 길이** 출력을 내고, **순서에 무관** 하며, 추천
손실이 안쪽까지 거슬러 올라올 수 있도록 **미분 가능** 해야 한다.

## 단 하나의 트릭: 더하기는 순서를 모른다

탈출구는 민망할 만큼 단순하다. 덧셈은 교환법칙을 따른다:

$$ \phi(p_1) + \phi(p_2) + \phi(p_3) \;=\; \phi(p_3) + \phi(p_1) + \phi(p_2) $$

각 점을 같은 함수 $\phi$ 로 *독립적으로* 변환한 뒤, 결과를 **전부
더한다**. 합은 점들이 어떤 순서로 도착했는지 모른다 — 순서 불변,
해결. 그리고 12개를 더하든 47개를 더하든 결과는 같은 폭의 벡터
하나다 — 가변 크기, 해결. 두 장애물이 산술의 한 가지 사실에 함께
무너진다.

이것이 DeepSets 레시피(Zaheer et al., 2017)이고, 단순히 편리한 꼼수가
아니다:

$$ F(X) = \rho\!\left( \sum_{x \in X} \phi(x) \right) $$

> **역사적 배경.** Zaheer et al. 은 표현 정리를 증명했다: 가산 집합
> 위의 *모든* 연속 순서 불변 함수는 정확히 이 변환-후-합 형태로
> 분해된다 — 집합 버전의 보편 근사 정리다. PersLay (Carrière et al.,
> JMLR 2020) 는 DeepSets 를 persistence diagram 에 특화한 것으로,
> 하나를 추가했다: 어떤 점이 중요한지에 대해 TDA 가 아는 바를 담는
> 점별 가중치 $w$. 정리가 없었다면 이 아키텍처는 휴리스틱이었겠지만,
> 정리 덕분에 합산은 타협이 아니라 표준형이다.

PersLay 의 전체 수식은 그 가중치를 끼워 넣은 것뿐이다:

$$ F(D) = \rho\!\left( \sum_{(b,d)\in D} w(b,d)\,\cdot\,\phi(b,d) \right) $$

> **수식 직관.** 영수증 뭉치를 떠올리자. $D$ 가 그 뭉치다 — 고객마다
> 장수가 다르고, 뭉치에 의미 있는 순서는 없다. $\phi$ 는 *영수증을 한
> 장씩* 읽고 고정 양식을 채우는 직원이다. $w$ 는 그 직원의 판단이다 —
> "이 영수증은 유익하고, 저건 쓰레기다" — 0 부터 큰 수까지의 숫자
> 하나. $\rho$ 는 가중치가 매겨진 양식들을 하나의 고정 서식 보고서로
> 철하는 일이다. 영수증이 몇 장 들어오든, 어떤 순서로 쌓여 있든,
> 보고서의 항목 수는 늘 같다.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <g font-size="10">
    <rect x="18" y="30" width="86" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
    <text x="61" y="51" text-anchor="middle" fill="#4f46e5" font-weight="700">(0.2, 0.9)</text>
    <rect x="18" y="100" width="86" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
    <text x="61" y="121" text-anchor="middle" fill="#4f46e5" font-weight="700">(0.5, 0.55)</text>
    <rect x="18" y="170" width="86" height="34" rx="5" fill="#f1f5f9" stroke="#94a3b8" stroke-dasharray="4 3"/>
    <text x="61" y="191" text-anchor="middle" fill="#94a3b8" font-weight="700">(0, 0) 패딩</text>
  </g>
  <g font-size="10">
    <rect x="150" y="30" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="51" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <rect x="150" y="100" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="121" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <rect x="150" y="170" width="64" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
    <text x="182" y="191" text-anchor="middle" fill="#0d9488" font-weight="700">φ</text>
    <text x="182" y="22" text-anchor="middle" fill="#64748b" font-size="9">같은 가중치, 점마다 적용</text>
  </g>
  <g font-size="10">
    <rect x="260" y="30" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706"/>
    <text x="303" y="51" text-anchor="middle" fill="#d97706" font-weight="700">× w = 0.70</text>
    <rect x="260" y="100" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706"/>
    <text x="303" y="121" text-anchor="middle" fill="#d97706" font-weight="700">× w = 0.05</text>
    <rect x="260" y="170" width="86" height="34" rx="5" fill="#fffbeb" stroke="#d97706" stroke-dasharray="4 3"/>
    <text x="303" y="191" text-anchor="middle" fill="#94a3b8" font-weight="700">× w = 0</text>
    <text x="303" y="22" text-anchor="middle" fill="#64748b" font-size="9">w = |d − b|</text>
  </g>
  <rect x="400" y="95" width="70" height="44" rx="6" fill="#f1f5f9" stroke="#1e3a5f"/>
  <text x="435" y="122" text-anchor="middle" font-size="16" font-weight="700" fill="#1e3a5f">Σ</text>
  <rect x="520" y="93" width="80" height="48" rx="6" fill="#0d9488"/>
  <text x="560" y="113" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">벡터 하나</text>
  <text x="560" y="129" text-anchor="middle" font-size="9" fill="#fff">n 과 무관, 폭 고정</text>
  <g stroke="#cbd5e1" stroke-width="1.3" fill="#cbd5e1">
    <line x1="104" y1="47" x2="148" y2="47"/><polygon points="148,47 140,43 140,51"/>
    <line x1="104" y1="117" x2="148" y2="117"/><polygon points="148,117 140,113 140,121"/>
    <line x1="104" y1="187" x2="148" y2="187"/><polygon points="148,187 140,183 140,191"/>
    <line x1="214" y1="47" x2="258" y2="47"/><polygon points="258,47 250,43 250,51"/>
    <line x1="214" y1="117" x2="258" y2="117"/><polygon points="258,117 250,113 250,121"/>
    <line x1="214" y1="187" x2="258" y2="187"/><polygon points="258,187 250,183 250,191"/>
    <line x1="346" y1="47" x2="398" y2="105"/><polygon points="398,105 389,103 394,96"/>
    <line x1="346" y1="117" x2="398" y2="117"/><polygon points="398,117 390,113 390,121"/>
    <line x1="346" y1="187" x2="398" y2="129"/><polygon points="398,129 393,138 388,131"/>
    <line x1="470" y1="117" x2="518" y2="117"/><polygon points="518,117 510,113 510,121"/>
  </g>
  <text x="310" y="237" text-anchor="middle" font-size="10" fill="#64748b">세 레인의 순서는 무의미하다 — 합은 동일하다</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">PersLay 의 가중치를 얹은 DeepSets 골격. 각 점을 독립적으로 변환(φ)하고, 중요도로 스케일(w)하고, 합산한다. 패딩 점은 가중치 0 을 받아 합에서 사라진다.</figcaption>
</figure>

이제 세 조각을 하나씩 — 이 프로젝트가 실제로 설정한 그대로.

## φ — 점 하나를 번역하기

$\phi$ 가 답하는 질문: *$(b, d)$ 점 하나는 무엇이 되는가?* 프로덕션
선택인 `RationalHatPhi` 는, 공짜로 쥐여줄 수 있는 산술을 신경망이
스스로 발견하게 두지 않는다. 2차원 점을 먼저 여섯 개의 수제 관점으로
확장한다:

$$ \phi_{\text{hat}}(b, d) = W_2\,\mathrm{ReLU}\!\big(W_1\,[\,b,\ d,\ d-b,\ \tfrac{b+d}{2},\ b \cdot d,\ \tfrac{d}{b+\epsilon}\,] + \mathbf{b}_1\big) + \mathbf{b}_2 $$

| # | 피처 | 점에게 묻는 것 |
| --- | --- | --- |
| 0 | $b$ (birth) | 이 구조는 어느 스케일에서 나타나는가? |
| 1 | $d$ (death) | 어느 스케일까지 살아남는가? |
| 2 | $d-b$ (persistence) | 얼마나 오래 사는가 — 진짜 구조인가 노이즈인가? |
| 3 | $(b+d)/2$ (midpoint) | 어느 스케일을 중심으로 활동하는가? |
| 4 | $b \cdot d$ (product) | 크게 태어나 크게 죽는가 — 거시 구조인가? |
| 5 | $d/(b+\epsilon)$ (ratio) | 자기 birth 스케일 *대비* 얼마나 오래 사는가? |

구체적인 점 하나로 따라가 보자. $(b, d) = (0.2,\ 0.9)$ 라면 여섯
관점의 값은 그냥 산수다:

$$ [\,0.2,\quad 0.9,\quad 0.7,\quad 0.55,\quad 0.18,\quad 4.5\,] $$

차례로 birth, death, persistence($0.9-0.2$), midpoint($(0.2+0.9)/2$),
product($0.2 \times 0.9$), ratio($0.9/0.2$). 점 하나가 숫자 2개에서
숫자 6개가 됐고, 여기까지는 학습된 것이 아무것도 없다.

이 6개가 2-layer MLP 로 들어간다. MLP 는 "숫자 6개를 받아서 숫자
64개를 내놓는 작은 신경망"이고, 그 출력 64개가 이 점의 최종 번역이다.
64개 숫자의 정체는 *이 점의 위상적 성격을 64가지 각도에서 적어 둔
묘사*라고 생각하면 된다 — 어떤 칸은 "오래 살았는가"에 크게 반응하도록,
어떤 칸은 "midpoint 와 birth 의 특정 조합"에 반응하도록, 값이 학습으로
정해진다.

왜 6개 확장을 손으로 해 주는가? persistence $d-b$ 가 중요하다는 것은
TDA 의 기본 상식이다. 그런데 MLP 에 $b$ 와 $d$ 만 던져 주면, "빼기가
유용하다"는 사실부터 데이터에서 발견해야 한다. 미리 계산해서 건네주면
그 발견 과정을 통째로 건너뛰고, 학습 용량은 *어떤 조합이 어느 태스크에
중요한지* 에만 쓰인다 — churn 예측은 persistence 와 ratio 에, CTR 은
midpoint 와 birth 에 기댈 수 있고, 그것을 사람이 정할 필요가 없다.

> **대안: GaussianPhi.** 참조 문서는 두 번째 변환도 구현해 두었다 —
> diagram 평면에 학습 가능한 가우시안 "탐지기" $K = 16$ 개($\mu_k$,
> 대역폭 $\sigma_k$)를 배치하고, 각 점의 활성값 열여섯 개를 내보낸다:
> $\phi_{\text{gauss}}(p) = W_{\text{proj}}\,[\,e^{-\lVert p-\mu_1\rVert^2/2\sigma_1^2},\ \dots,\ e^{-\lVert p-\mu_K\rVert^2/2\sigma_K^2}\,]$.
> 센서가 훈련되는 커널 밀도 추정이다: 학습이 $\mu_k$ 를 persistence 가
> 높은 중요 영역으로 끌어간다. 기록해 둘 만한 엔지니어링 흉터 하나 —
> fp16 mixed precision 에서 $\exp$ 가 오버플로하므로, 구현은 대역폭을
> float32 로 캐스트하고 지수를 $[-10, 10]$ 으로 클램프한다. 그것이
> 없으면 forward 가 첫날부터 `inf` 를 뱉는다. 프로덕션은 그런 위험이
> 없는 RationalHat 을 쓴다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="200" fill="#f8fafc" rx="8"/>
  <rect x="22" y="78" width="80" height="44" rx="6" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="62" y="97" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">(b, d)</text>
  <text x="62" y="112" text-anchor="middle" font-size="9" fill="#64748b">숫자 2개</text>
  <g font-size="9.5" font-weight="700">
    <rect x="150" y="18" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="33" text-anchor="middle" fill="#0d9488">b</text>
    <rect x="150" y="46" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="61" text-anchor="middle" fill="#0d9488">d</text>
    <rect x="150" y="74" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="89" text-anchor="middle" fill="#0d9488">d − b</text>
    <rect x="150" y="102" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="117" text-anchor="middle" fill="#0d9488">(b+d)/2</text>
    <rect x="150" y="130" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="145" text-anchor="middle" fill="#0d9488">b·d</text>
    <rect x="150" y="158" width="92" height="22" rx="4" fill="#f0fdfa" stroke="#0d9488"/><text x="196" y="173" text-anchor="middle" fill="#0d9488">d/(b+ε)</text>
  </g>
  <text x="196" y="12" text-anchor="middle" font-size="9" fill="#64748b">고정된 6개 관점</text>
  <rect x="300" y="70" width="120" height="58" rx="6" fill="#fffbeb" stroke="#d97706"/>
  <text x="360" y="94" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">2-layer MLP</text>
  <text x="360" y="110" text-anchor="middle" font-size="9" fill="#64748b">학습되는 혼합</text>
  <rect x="468" y="76" width="72" height="46" rx="6" fill="#0d9488"/>
  <text x="504" y="96" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">64D</text>
  <text x="504" y="111" text-anchor="middle" font-size="9" fill="#fff">점마다</text>
  <g stroke="#cbd5e1" stroke-width="1.2" fill="#cbd5e1">
    <line x1="102" y1="100" x2="146" y2="100"/><polygon points="146,100 138,96 138,104"/>
    <line x1="242" y1="29" x2="298" y2="84"/><line x1="242" y1="57" x2="298" y2="92"/><line x1="242" y1="85" x2="298" y2="97"/>
    <line x1="242" y1="113" x2="298" y2="102"/><line x1="242" y1="141" x2="298" y2="108"/><line x1="242" y1="169" x2="298" y2="115"/>
    <line x1="420" y1="99" x2="466" y2="99"/><polygon points="466,99 458,95 458,103"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">RationalHatPhi: 점 하나를 해석 가능한 6개 관점으로 수동 확장한 뒤, 태스크별 혼합은 작은 MLP 가 학습한다.</figcaption>
</figure>

## w — 점마다 중요도 매기기

$w$ 가 답하는 질문: *이 점을 얼마나 중요하게 칠 것인가?* 출력은 점마다
숫자 하나다 — 뒤에서 그 점의 $\phi$ 벡터에 곱해질 배율. 세 가지 모드가
구현되어 있고, 프로덕션은 첫 번째를 쓴다.

| 모드 | 공식 | 성격 |
| --- | --- | --- |
| `persistence` (프로덕션) | $w(b,d) = \lvert d-b \rvert^{p}$, $p = 1.0$ | 오래 산 구조일수록 무겁다; 대각선 위의 점은 0 |
| `linear` | $w(b,d) = 1$ | 모든 점 동일 — 균등 베이스라인 |
| `learned` | $w = \mathrm{Softplus}(\mathrm{MLP}([b,d]))$ | 가중치 자체를 학습; Softplus 가 비음수 보장 |

프로덕션 공식부터 풀어 두자. $|d-b|$ 는 그 점의 persistence — TDA-1
에서 본 "얼마나 오래 살았는가" — 그 자체다. $p$ 는 거기에 얹는
지수다: $p=1$ 이면 가중치가 수명에 *정비례* 한다 (0.7 을 산 점은 0.7
만큼, 0.05 를 산 점은 0.05 만큼만 반영). $p$ 를 2 로 올리면 오래 산
점이 제곱으로 우대받아 격차가 벌어지고, 0.5 로 내리면 격차가
완만해진다. 프로덕션은 가장 단순한 정비례, $p = 1.0$ 이다.

절댓값 하나로 끝나는 단순한 식이지만, 이 선택이 세 가지 이득을
한꺼번에 가져온다.

- **노이즈 억제가 공짜.** TDA-1 의 diagram 읽는 법 — 대각선에서 멀면
  진짜, 가까우면 노이즈 — 이 자신의 *연속적인* 버전이 된다. 조정할
  임계값이 없다; 대각선 근처의 점은 그냥 흐려진다.
- **패딩 처리가 공짜.** 배치 처리를 위해 diagram 은 고정 `max_pairs`
  까지 $(0,0)$ 으로 패딩되는데, 그 persistence 는 $|0-0| = 0$, 따라서
  가중치도 0 — **마스크 산술 없이** 패딩이 합에서 사라진다.
- **gradient 를 왜곡하지 않는다.** $p = 1$ 이고 항상 $d > b$ 이므로
  $\partial w / \partial d = 1$: 가중치는 기여도를 스케일할 뿐
  gradient 장을 구부리지 않는다. (패딩 점은 $w = 0$ 을 통해 gradient
  도 0 이 된다 — 정확히 의도된 동작이다.)

## ρ — 점들을 벡터 하나로 합치기

$\rho$ 가 답하는 질문: *가중치까지 곱해진 점-벡터 여러 개를 어떻게
벡터 하나로 합치는가?* 순서 불변 옵션 네 가지가 구현되어 있다.

| 모드 | 읽히는 의미 | 비용 |
| --- | --- | --- |
| `sum` (프로덕션) | 위상 구조의 *총량* | $O(n)$ |
| `mean` | 점 수에 정규화된 *평균적* 구조 | $O(n)$ |
| `max` | *가장 두드러진* 구조 하나 | $O(n)$ |
| `attention` | *학습된* 주의 배분 | $O(n^2)$ |

> **수식 직관.** sum 은 고객이 가진 구조가 많을수록 커진다. mean 은
> 개수와 무관하게 전형적 구조가 어떤지 묻는다. max 는 가장 강한 패턴
> 하나에 모든 것을 건다. attention
> ($\alpha_i = \mathrm{softmax}(W_2 \tanh(W_1 \mathbf{x}_i))$) 은 어디를
> 볼지 학습한다 — 가장 표현력이 높고 가장 비싸다.

프로덕션은 `sum` 이고, attention 에서 갈아탄 이유에 대해 참조 문서는
직설적이다: `max_pairs = 200` 에서 attention 은 배치마다 diagram 당
$200 \times 200$ 행렬을 만든다 — 실제 VRAM, 실제 지연 — 그런데 점별
중요도는 persistence 가중치가 *이미* 선형 비용으로 제공한다. $w$ 가
주는 것을 다시 배우려고 제곱 비용을 내는 것은 나쁜 거래였고, `sum`
전환은 비용을 $O(n^2)$ 에서 $O(n)$ 으로 내리면서 청구서를 정당화할
품질 손실이 없었다.

## 손으로 한 바퀴

지금까지의 세 조각을 가장 작은 예제로 직접 돌려 보자. 어떤 고객의
diagram $D$ 에 점이 세 개 있다 — 진짜 점 둘과, 배치 처리용 패딩 하나.

**1단계 — 점마다 가중치 $w$ 를 계산한다.** 공식은 $w = |d-b|$ 하나다.

- $(0.2,\ 0.9)$ : $w = |0.9-0.2| = 0.70$ — 오래 산 진짜 구조.
- $(0.5,\ 0.55)$ : $w = |0.55-0.5| = 0.05$ — 대각선 바로 옆, 노이즈에
  가깝다.
- $(0,\ 0)$ 패딩 : $w = |0-0| = 0$ — 이 순간 이미 무시가 예약됐다.

**2단계 — 점마다 $\phi$ 로 번역한다.** 앞 절의 RationalHat 이 각 점을
64개 숫자로 바꾸지만, 손으로 따라가기 좋게 여기서는 3개만 낸다고 치자.
아래 표의 $\phi$ 값 자체는 설명용 예시 숫자다 — 중요한 것은 *점 하나당
같은 길이의 벡터 하나가 나온다* 는 모양이다.

**3단계 — 가중치를 곱하고, 전부 더한다.** $w \cdot \phi$ 는 숫자
하나(가중치)를 벡터의 *각 칸에* 곱하는 것이고, 마지막 합은 *칸별로*
더하는 것이다.

| 점 | $w = \lvert d-b \rvert$ | $\phi(b,d)$ (예시값) | $w \cdot \phi$ |
| --- | --- | --- | --- |
| $(0.2,\ 0.9)$ | $0.70$ | $[\,0.4,\ -1.1,\ 0.8\,]$ | $[\,0.28,\ -0.77,\ 0.56\,]$ |
| $(0.5,\ 0.55)$ | $0.05$ | $[\,1.2,\ 0.3,\ -0.5\,]$ | $[\,0.06,\ 0.015,\ -0.025\,]$ |
| $(0,\ 0)$ 패딩 | $0$ | $[\,0.9,\ 0.2,\ 0.1\,]$ | $[\,0,\ 0,\ 0\,]$ |

첫째 칸만 따라가 보면 $0.28 + 0.06 + 0 = 0.34$. 나머지 칸도 같은
방식으로 더하면:

$$ F = [\,0.34,\ -0.755,\ 0.535\,] $$

이 산수 안에 약속했던 성질이 전부 들어 있다.

- **노이즈 억제** — 오래 산 점($w=0.70$)이 결과를 지배하고, 대각선 옆
  점($w=0.05$)은 흔적만 남긴다.
- **패딩 무시** — 셋째 행은 곱하는 순간 전부 0 이 됐다. 마스크 없이
  사라진 것이다.
- **순서 불변** — 행을 어떤 순서로 더해도 합은 같다.
- **고정 폭** — 점이 3개든 300개든 출력은 벡터 하나다. 여기서는 3칸,
  실제로는 64칸.

## 다섯 블록 — 왜 PersLay 하나가 아닌가

단일 PersLay 레이어라면 모든 diagram 의 모든 점을 하나의 $\phi$ 와
하나의 $w$ 에 통과시켰을 것이다. 프로젝트는 대신 **독립 파라미터를
가진 다섯 개의 `PersLayBlock`** 을 돌린다.

```python
class PersLayBlock(nn.Module):
    def forward(self, points, mask=None):
        phi_out = self.phi(points)           # [B, max_pairs, 64]
        weights = self.weight_fn(points)     # [B, max_pairs, 1]
        return self.rho(phi_out, weights, mask)  # [B, 64]
```

분할 기준은 *시간 범위 × 호몰로지 차원* 이다. Short-range diagram
(90일 앱 로그, 최대 200쌍)은 `beta_idx` 채널을 달고 **Short
$\beta_0$** 와 **Short $\beta_1$** 블록으로 라우팅되고, long-range
diagram (12개월 거래, 최대 150쌍)은 **Long $\beta_0$ / $\beta_1$ /
$\beta_2$** 로 들어간다. 각 점은 자기 블록에만 참여한다 — 마스크는
*유효쌍 마스크 AND beta 마스크* 로 합성되고, persistence 가중치가
세 번째 암묵적 필터로 작동한다.

> **설계 근거: 왜 β 별 독립 파라미터인가.** $(b,d)$ 분포가 차원마다
> 질적으로 다르다. $H_0$ 에서는 모든 점이 $b = 0$ 에서 태어난다 —
> 모든 성분은 처음부터 존재하므로 — 블록의 기하가 한쪽으로 쏠려 있다.
> $H_1$ 에서는 루프가 양의 스케일에서만 생기므로 항상 $b > 0$ 이다.
> 공유된 $\phi$ 하나는 두 체제에 다리를 걸쳐야 하지만, 특화된 다섯
> 블록은 각자 자기 것을 배운다. 그리고 90일짜리 군집 분리는 12개월짜리
> 소비 공동과 *종류가 다른* 신호다 — 별도 파라미터가 각자를 각자이게
> 한다.

다섯 개의 64D 블록 출력은 두 보조 입력과 이어 붙은 뒤 압축된다.

| 구성 요소 | 폭 | 내용 |
| --- | --- | --- |
| Short $\beta_0 + \beta_1$ | $64+64 = 128$D | 단기 군집과 루프 |
| Long $\beta_0 + \beta_1 + \beta_2$ | $64+64+64 = 192$D | 장기 군집, 루프, 공동 |
| Global stats MLP | $30 \to 32$D | 점별 경로가 놓칠 수 있는 diagram 전체 요약 (엔트로피, 수명 통계) |
| Phase transition | $10$D | 국면 전환 피처, passthrough |
| **합계 → 출력** | $362 \to 64$D | `final_mlp`: Linear(362,128) → LayerNorm → SiLU → Dropout → Linear(128,64) → LayerNorm |

<figure style="margin:24px auto;max-width:640px;">
<svg viewBox="0 0 640 330" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="640" height="330" fill="#f8fafc" rx="8"/>
  <rect x="30" y="22" width="150" height="34" rx="5" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="105" y="36" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">short_diagrams</text>
  <text x="105" y="49" text-anchor="middle" font-size="8.5" fill="#64748b">[B, 200, 3] — 90일 앱 로그</text>
  <rect x="300" y="22" width="150" height="34" rx="5" fill="#f0fdfa" stroke="#0d9488"/>
  <text x="375" y="36" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">long_diagrams</text>
  <text x="375" y="49" text-anchor="middle" font-size="8.5" fill="#64748b">[B, 150, 3] — 12개월 거래</text>
  <g font-size="9.5" font-weight="700">
    <rect x="22" y="96" width="76" height="40" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="60" y="113" text-anchor="middle" fill="#4f46e5">Short β₀</text><text x="60" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="112" y="96" width="76" height="40" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="150" y="113" text-anchor="middle" fill="#4f46e5">Short β₁</text><text x="150" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="252" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="290" y="113" text-anchor="middle" fill="#0d9488">Long β₀</text><text x="290" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="342" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="380" y="113" text-anchor="middle" fill="#0d9488">Long β₁</text><text x="380" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
    <rect x="432" y="96" width="76" height="40" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="470" y="113" text-anchor="middle" fill="#0d9488">Long β₂</text><text x="470" y="127" text-anchor="middle" fill="#64748b" font-weight="400">[B, 64]</text>
  </g>
  <rect x="528" y="84" width="96" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/>
  <text x="576" y="97" text-anchor="middle" font-size="9" font-weight="700" fill="#d97706">global_stats</text>
  <text x="576" y="109" text-anchor="middle" font-size="8" fill="#64748b">[B, 30] → MLP → 32D</text>
  <rect x="528" y="122" width="96" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/>
  <text x="576" y="135" text-anchor="middle" font-size="9" font-weight="700" fill="#d97706">phase_transition</text>
  <text x="576" y="147" text-anchor="middle" font-size="8" fill="#64748b">[B, 10] passthrough</text>
  <rect x="170" y="196" width="300" height="36" rx="6" fill="#f1f5f9" stroke="#1e3a5f"/>
  <text x="320" y="211" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e3a5f">concat — 128 + 192 + 32 + 10 = 362D</text>
  <text x="320" y="226" text-anchor="middle" font-size="8.5" fill="#64748b">다섯 블록 + 두 보조 입력</text>
  <rect x="196" y="262" width="248" height="40" rx="6" fill="#0d9488"/>
  <text x="320" y="279" text-anchor="middle" font-size="10.5" font-weight="700" fill="#fff">final_mlp — 362 → 128 → 64D</text>
  <text x="320" y="294" text-anchor="middle" font-size="8.5" fill="#fff">LayerNorm · SiLU · Dropout → PLE CGC gate</text>
  <g stroke="#cbd5e1" stroke-width="1.2" fill="#cbd5e1">
    <line x1="80" y1="56" x2="62" y2="94"/><polygon points="62,94 61,86 69,89"/>
    <line x1="130" y1="56" x2="148" y2="94"/><polygon points="148,94 141,89 149,86"/>
    <line x1="345" y1="56" x2="292" y2="94"/><polygon points="292,94 294,86 300,91"/>
    <line x1="375" y1="56" x2="379" y2="94"/><polygon points="379,94 375,87 383,87"/>
    <line x1="405" y1="56" x2="468" y2="94"/><polygon points="468,94 460,92 465,85"/>
    <line x1="60" y1="136" x2="218" y2="195"/><line x1="150" y1="136" x2="252" y2="195"/>
    <line x1="290" y1="136" x2="300" y2="194"/><line x1="380" y1="136" x2="350" y2="194"/><line x1="470" y1="136" x2="400" y2="194"/>
    <line x1="576" y1="114" x2="576" y2="120"/>
    <line x1="560" y1="152" x2="462" y2="200"/><polygon points="462,200 470,196 472,203"/>
    <line x1="320" y1="232" x2="320" y2="260"/><polygon points="320,260 316,252 324,252"/>
  </g>
  <text x="60" y="76" text-anchor="middle" font-size="8.5" fill="#94a3b8">β₀ 점들</text>
  <text x="152" y="76" text-anchor="middle" font-size="8.5" fill="#94a3b8">β₁ 점들</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">5-Block PersLayExpert. 각 블록이 자기만의 φ, w, ρ 를 가진다; beta_idx 채널이 각 점을 제 블록으로 보낸다. 362D 로 이어 붙여 PLE gate 가 기대하는 64D 로 압축.</figcaption>
</figure>

조용한 덤 하나: 64D 와 함께 Expert 는 4차원 *해석용 프로젝션* 도
내보낸다 — 패턴 안정성, 주기성 강도, 이상 패턴, 복잡도. 손실에는 전혀
관여하지 않는다; 모델을 디버깅하는 사람(또는 비즈니스에 설명하는
사람)이 이름 없는 64개 눈금 대신 이름 붙은 4개 다이얼을 읽을 수 있게
하기 위해 존재한다.

## 정직한 각주 — 지금 실제로 도는 경로

위의 전부는 논문에 충실한 설계이고, 코드는 그 전부를 구현하고 있다.
그러나 참조 문서는 *현재* 상태에 대해 솔직하고, 이 시리즈는 출처를
정직하게 인용한다: 라이브 설정에서 raw diagram 경로는 **꺼져 있다**
(`use_raw_diagram: false`, raw diagram Parquet 도 현재 미주입). 따라서
프로덕션 추론은 **사전 계산 폴백** 을 돈다 — 오프라인 70D 요약(short
24D + long 36D + phase 10D)을 3-layer MLP, $70 \to 64 \to 64 \to 64$
에 통과시키는 경로다. 그 피처마저 없으면 Expert 는 배치를 깨뜨리는
대신 zero 벡터로 강등된다.

폴백은 점별 세부 정보를 강건함과 맞바꾼다 — 집계된 통계는 개별
$(b,d)$ 점이 담은 것을 볼 수 없다. 5-Block 경로가 목적지이고, 통계
MLP 가 지금 출하되는 것이다. 아무도 읽지 않는 각주보다는 본문의 정직한
한 줄이 낫다.

## 논문 vs 구현

| 항목 | Carrière et al. (2020) | 본 프로젝트 |
| --- | --- | --- |
| 블록 | 단일 PersLay 레이어 | 독립 블록 5개 (Short β₀/β₁ + Long β₀/β₁/β₂) |
| 입력 | 단일 diagram | Short/Long diagram + 30D global stats + 10D phase |
| 출력 | 가변 | 고정 64D (PLE gate 규격) |
| 후처리 | 없음 | final_mlp (362→64) + 4D 해석 프로젝션 |
| 모드 | raw diagram 만 | 듀얼: raw 5-Block + 사전 계산 70D 폴백 |
| 패딩 | 가변 크기 입력 | 고정 max_pairs + persistence 가중치 자동 무시 |

## 여기서 멈추는 이유

TDA 서브스레드는 여기서 닫힌다. TDA-1 은 소비에 형태가 있다고 주장하고
실제 세션으로 검증했다; TDA-2 는 기계를 열었다 — 집합 입력 문제를
녹이는 교환법칙 하나, 각자 이유가 명시된 학습 가능한 세 조각($\phi$,
$w$, $\rho$), 단기 군집과 장기 공동이 다른 신호이기에 다섯으로 나뉜
특화 블록, 그리고 오늘 실제로 도는 경로에 대한 정직한 각주 하나. 이
이야기의 오프라인 측 — 70D 요약 피처가 배치 시점에 실제로 어떻게
추출되는가 — 는 시리즈 뒤쪽의 별도 편(TDAFEAT-1)이 맡는다.

다음은 아주 다른 종류의 전문가다: **DEEPFM-1**, factorization machine
편 — 선형 모델은 "높은 식비 지출 × 심야 세션" 이 각 신호 단독으로는
없는 의미를 갖는다는 것을 왜 볼 수 없는지, 그리고 공유 임베딩이 어떻게
20만 개의 쌍별 가중치를 1만 개로 무너뜨리는지.
