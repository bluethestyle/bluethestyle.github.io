---
title: "[Study Thread] TDAFEAT-1 — 오프라인 TDA 피처 파이프라인: 원시 로그에서 메인 텐서의 70D까지"
date: 2026-06-07 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, tda, persistent-homology, feature-engineering, offline]
lang: ko
excerpt: "TDA-1 편은 PersLay Expert — persistence diagram 을 end-to-end 로 받는 온라인 Shared Expert — 를 다뤘다. 이번 편은 위상 이야기의 나머지 절반, 배치에서 사전 계산되는 오프라인 TDA 피처다. 원시 거래/앱 로그가 6D 점 구름이 되고, Ripser 가 persistence diagram 으로 만들고, 통계량과 Persistence Entropy 가 이를 734D 메인 텐서 안에 들어가는 70D 블록으로 벡터화한다. 벡터화 선택, 실제 피처 구성, 그리고 오프라인으로 처리하는 것을 정당화하는 비용 트레이드오프."
series: study-thread
part: 18
alt_lang: /2026/06/07/tda-features-offline-en/
next_title: "HMM-1 — Regime 피처: 라이프스테이지 감지를 위한 은닉 마르코프 모델"
next_desc: "Model-Derived 블록은 5D HMM summary 를 담는다. 은닉 마르코프 모델이 고객의 거래 흐름을 잠재 regime 의 시퀀스로 읽는 방식, Viterbi 경로와 사후 엔트로피가 담는 정보, 그리고 그 다섯 숫자가 행동 변화의 위상적 관점을 어떻게 보완하는가."
next_status: draft
---

*"Study Thread" 시리즈의 한 편으로, 온프렘 참조서
`기술참조서/TDA_피처_기술_참조서` 를 출처로 한다. TDA-1 / PersLay 편의
짝이다. TDA-1 이 persistent homology 와* 온라인 *PersLay Expert
(persistence diagram 에서 64D 벡터를 end-to-end 로 학습)를 소개했다면,
이번 편은* 오프라인 *쪽을 다룬다 — 배치에서 사전 계산되어 734D 메인
텐서의 고정 70D 조각이 되는 TDA 피처다. 호몰로지 기초는 여기서 다시
가르치지 않는다. 여과(filtration), persistence diagram, 안정성 정리는
TDA-1 을 보라. 참조서 전체 PDF 는 서브스레드 마지막 편에 첨부한다.*

> **두 파이프라인, 하나의 수학.** 오프라인 TDA 피처와 온라인 PersLay
> Expert 는 둘 다 persistent homology 위에 서 있지만, 다른 일을 하는
> 다른 기계다. PersLay 는 *학습 가능한* Shared Expert 다 — persistence
> diagram 을 받아 추천 손실을 역전파해 64D 표현을 학습한다. 오프라인 TDA
> 피처는 *고정* 이다 — 원시 로그를 넣으면 결정론적 70D 벡터가 나오고,
> Airflow 배치에서 한 번 계산되어 메인 텐서에 동결된다. 그래디언트도
> 학습도 없다. 사람 손으로 고른 diagram 벡터화일 뿐이다. 이 글은 전적으로
> 두 번째 기계에 관한 것이다.

## 오프라인 피처 vs. 온라인 Expert

프로젝트는 위상을 의도적으로 *두* 개의 문으로 통과시킨다. 이유는 비용의
비대칭성이다. persistence diagram 계산은 비싸고 입력 모양에 의존하지만
(점 구름이 고객마다 다르다), 모델은 매 forward 마다 읽을 수 있는 고정
폭의 벡터를 원한다. 그래서 한 경로는 동결된 기술자를 사전 계산하고, 다른
경로는 신경망 안에서 diagram 으로부터 학습한다.

| 구분 | TDA 피처 (이 글) | PersLay Expert (TDA-1) |
| --- | --- | --- |
| 역할 | 오프라인 피처 추출 | 온라인 PLE Shared Expert |
| 입력 | 원시 거래 / 앱 로그 | Persistence diagram |
| 출력 | 70D → 734D 메인 텐서의 일부 | 64D → PLE CGC gate |
| 시점 | 배치 전처리 (Airflow) | 학습 / 추론 (end-to-end) |
| 학습? | 아니오 — 고정 통계량 | 예 — φ, w, ρ 학습 가능 |
| 벡터화 | Persistence Entropy + lifetime 통계 | RationalHat φ + persistence 가중치 |

둘은 중복이 아니다. 70D 오프라인 블록은 *모든* Expert 와 태스크가 읽는
입력 텐서에 직접 들어가므로, PersLay Expert 를 연결하지 않는 태스크도
위상 신호를 공짜로 얻는다. 그리고 오프라인 diagram 은 실시간 추출이
불가능할 때 PersLay Expert 의 사전 계산 대체 입력으로도 쓸 수 있다.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <rect x="16" y="40" width="588" height="86" rx="8" fill="#4f46e508" stroke="#4f46e5" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="30" y="60" font-size="12" font-weight="700" fill="#4f46e5">오프라인 · Airflow 배치 · 한 번 계산, 동결</text>
  <rect x="30" y="72" width="92" height="42" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="76" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">원시 로그</text>
  <text x="76" y="105" text-anchor="middle" font-size="9" fill="#64748b">거래 / 앱</text>
  <rect x="158" y="72" width="92" height="42" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="204" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">Ripser</text>
  <text x="204" y="105" text-anchor="middle" font-size="9" fill="#64748b">→ diagram</text>
  <rect x="286" y="72" width="100" height="42" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="336" y="91" text-anchor="middle" font-size="10" fill="#1e3a5f">벡터화</text>
  <text x="336" y="105" text-anchor="middle" font-size="9" fill="#64748b">엔트로피 + 통계</text>
  <rect x="422" y="72" width="70" height="42" rx="6" fill="#d97706"/>
  <text x="457" y="91" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">70D</text>
  <text x="457" y="105" text-anchor="middle" font-size="8.5" fill="#fff">고정</text>
  <rect x="516" y="66" width="74" height="54" rx="6" fill="#1e3a5f"/>
  <text x="553" y="88" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">734D</text>
  <text x="553" y="102" text-anchor="middle" font-size="8.5" fill="#cbd5e1">메인 텐서</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="122" y1="93" x2="156" y2="93"/><polygon points="156,93 148,89 148,97"/>
    <line x1="250" y1="93" x2="284" y2="93"/><polygon points="284,93 276,89 276,97"/>
    <line x1="386" y1="93" x2="420" y2="93"/><polygon points="420,93 412,89 412,97"/>
    <line x1="492" y1="93" x2="514" y2="93"/><polygon points="514,93 506,89 506,97"/>
  </g>
  <rect x="16" y="146" width="588" height="86" rx="8" fill="#0d948808" stroke="#0d9488" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="30" y="166" font-size="12" font-weight="700" fill="#0d9488">온라인 · 학습 / 추론 · 매 forward 학습</text>
  <rect x="30" y="178" width="92" height="42" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="76" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f">diagram</text>
  <text x="76" y="211" text-anchor="middle" font-size="9" fill="#64748b">점 집합</text>
  <rect x="158" y="178" width="120" height="42" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="218" y="197" text-anchor="middle" font-size="10" fill="#1e3a5f">PersLay φ·w·ρ</text>
  <text x="218" y="211" text-anchor="middle" font-size="9" fill="#64748b">학습 가능</text>
  <rect x="314" y="178" width="70" height="42" rx="6" fill="#0d9488"/>
  <text x="349" y="197" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">64D</text>
  <text x="349" y="211" text-anchor="middle" font-size="8.5" fill="#fff">학습됨</text>
  <rect x="420" y="172" width="100" height="54" rx="6" fill="#1e3a5f"/>
  <text x="470" y="194" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">PLE CGC gate</text>
  <text x="470" y="208" text-anchor="middle" font-size="8.5" fill="#cbd5e1">태스크별</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="122" y1="199" x2="156" y2="199"/><polygon points="156,199 148,195 148,203"/>
    <line x1="278" y1="199" x2="312" y2="199"/><polygon points="312,199 304,195 304,203"/>
    <line x1="384" y1="199" x2="418" y2="199"/><polygon points="418,199 410,195 410,203"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">같은 호몰로지에서 나온 두 레인. 위: 오프라인 배치가 734D 텐서의 동결된 70D 조각을 만든다. 아래: 온라인 PersLay Expert 가 PLE gate 로 들어가는 64D 벡터를 학습한다. 이 글은 위쪽 레인이다.</figcaption>
</figure>

## 추출 파이프라인

오프라인 파이프라인은 Airflow 배치 안에서 고객별로 실행되는 네 단계
서사다. 학습되는 것은 없다 — 모든 단계가 고정 변환이다.

1. **점 구름.** 각 거래(또는 앱 세션)는 금액, 카테고리, 요일, 시간의
   **6D 공간** 의 한 점이 된다. 한 고객의 모든 점들의 구름이 그의 "소비
   지형" 이다.
2. **다중 해상도 스캔.** Vietoris–Rips 여과가 공 반지름 ε 을 키우며
   구조의 생성과 소멸을 지켜본다 — 작은 ε 에서는 세밀한 클러스터, 큰
   ε 에서는 거시 구조. (TDA-1 의 그 여과다. 다시 유도하지 않는다.)
3. **구조적 요약.** Ripser 가 스캔 전체를 *persistence diagram* — birth,
   death, 차원 — 으로 압축한다. 오래 산 특성만 남기고 대각선 근처의
   노이즈는 버린다.
4. **벡터화.** 통계량 집합에 Persistence Entropy 를 더해 가변 크기의
   diagram 을 고정 폭 벡터로 바꾼다. *이것* 이 모델에 들어간다.

오프라인 파이프라인 고유의 다섯 번째 단계는 *시간적* 변화를 추적한다.
고객 이력을 전반기/후반기로 나눠 각각의 diagram 을 구하고, Wasserstein
거리로 위상이 얼마나 움직였는지 측정한다. 그것이 `phase_transition`
블록이다.

점 구름 좌표는 원시값이 아니다. 금액은 멱법칙 꼬리를 누르기 위해 로그
변환(`ln(amount + 1)`)을 거치고, 주기 변수(요일, 시간)는 sin/cos
인코딩되어 일요일과 월요일이 수직선에서 6칸 떨어지는 대신 단위원에서 한
칸 떨어지게 한다. MCC 카테고리는 순서형이 아니라 명목형이므로 거리가
의미를 갖도록 균등 `PERCENT_RANK()` 로 펼친다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="200" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a5f">고객별 오프라인 추출</text>
  <rect x="20" y="58" width="96" height="70" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="68" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4f46e5">6D 구름</text>
  <g fill="#4f46e5"><circle cx="48" cy="98" r="2.5"/><circle cx="66" cy="92" r="2.5"/><circle cx="84" cy="104" r="2.5"/><circle cx="58" cy="112" r="2.5"/><circle cx="80" cy="116" r="2.5"/><circle cx="70" cy="106" r="2.5"/></g>
  <text x="68" y="122" text-anchor="middle" font-size="8" fill="#64748b">금액·업종·요일·시간</text>
  <rect x="152" y="58" width="96" height="70" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="200" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#0d9488">VR 여과</text>
  <circle cx="200" cy="102" r="16" fill="none" stroke="#0d9488" stroke-width="1" stroke-dasharray="3 2"/>
  <circle cx="200" cy="102" r="26" fill="none" stroke="#94a3b8" stroke-width="0.8" stroke-dasharray="2 3"/>
  <text x="200" y="122" text-anchor="middle" font-size="8" fill="#64748b">ε 증가</text>
  <rect x="284" y="58" width="96" height="70" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="332" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#d97706">diagram</text>
  <line x1="312" y1="118" x2="356" y2="88" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3 2"/>
  <g fill="#d97706"><circle cx="322" cy="100" r="2.5"/><circle cx="332" cy="94" r="2.5"/><circle cx="340" cy="108" r="2.5"/></g>
  <text x="332" y="122" text-anchor="middle" font-size="8" fill="#64748b">(b, d)</text>
  <rect x="416" y="58" width="96" height="70" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="464" y="80" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e3a5f">벡터화</text>
  <g fill="#1e3a5f"><rect x="438" y="92" width="6" height="22" rx="1"/><rect x="448" y="98" width="6" height="16" rx="1"/><rect x="458" y="88" width="6" height="26" rx="1"/><rect x="468" y="100" width="6" height="14" rx="1"/><rect x="478" y="94" width="6" height="20" rx="1"/></g>
  <text x="464" y="122" text-anchor="middle" font-size="8" fill="#64748b">E + 통계</text>
  <rect x="540" y="72" width="48" height="42" rx="6" fill="#d97706"/>
  <text x="564" y="90" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">70D</text>
  <text x="564" y="103" text-anchor="middle" font-size="8" fill="#fff">블록</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.2">
    <line x1="116" y1="93" x2="150" y2="93"/><polygon points="150,93 142,89 142,97"/>
    <line x1="248" y1="93" x2="282" y2="93"/><polygon points="282,93 274,89 274,97"/>
    <line x1="380" y1="93" x2="414" y2="93"/><polygon points="414,93 406,89 406,97"/>
    <line x1="512" y1="93" x2="538" y2="93"/><polygon points="538,93 530,89 530,97"/>
  </g>
  <text x="84" y="168" text-anchor="middle" font-size="9" fill="#94a3b8">로그 금액 · sin/cos 주기 · PERCENT_RANK 업종</text>
  <text x="430" y="168" text-anchor="middle" font-size="9" fill="#94a3b8">호몰로지 차원별 Persistence Entropy + lifetime 통계 5개</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">네 개의 고정 단계: 원시 이벤트 → 6D 점 구름 → Vietoris–Rips diagram → 고정 벡터. 좌표 변환(좌하단)이 거리를 의미 있게 만들고, 벡터화기(우하단)는 호몰로지 차원당 6개 숫자다.</figcaption>
</figure>

## Diagram 이 벡터가 되는 방식

Persistence diagram 은 *다중집합* 이다 — 가변 크기의 순서 없는
`(birth, death)` 점 주머니 — 그래서 모델에 직접 들어갈 수 없다. TDA-1 은
이를 PersLay 의 *학습 가능한* 집합 함수로 해결했다. 오프라인 파이프라인은
더 오래된 *고정* 경로를 택한다. 참조서는 세 가지 고전적 선택지를 든다.

| 방법 | 원리 | 여기서 사용? |
| --- | --- | --- |
| 통계량 | lifetime 분포의 평균/표준편차/범위. 가장 빠르고 직접적. | **예 — 주력** |
| Persistence Entropy | lifetime 분포의 Shannon 엔트로피. 다양성을 단일 스칼라로. | **예** |
| Persistence Landscape | 텐트 함수 → $L^p$ 노름(Amplitude). Banach 공간에 거주. | 초기화되었으나 출력 컬럼에 **미포함** |

시스템은 **통계량 + Persistence Entropy** 를 조합한다. 구체적으로 각
호몰로지 차원에 대해 **6개 피처** 를 낸다 — Persistence Entropy(1) 와
lifetime 통계 5개(mean, std, min, max, median). Landscape Amplitude
계산기(`Amplitude(metric="landscape")`)는 코드에 초기화되어 있으나 그
출력은 현 운영 컬럼에 *포함되지 않는다*. 스키마 이름이 이를 암시하므로
짚어둔다.

여섯 중 엔트로피가 가장 독특하다. $i$-번째 특성의 수명을 $L_i = d_i - b_i$
라 하면:

$$ E = -\sum_{i=1}^{N} p_i \log p_i, \qquad p_i = \frac{d_i - b_i}{\sum_{j=1}^{N}(d_j - b_j)} $$

> **수식 직관.** 각 특성의 수명을 전체 수명 합으로 정규화하고, 그
> 결과를 확률 분포로 보아 Shannon 엔트로피를 취한다. 하나의 거대한
> 구조가 수명을 독점하면 — 모든 소비가 단일 클러스터로 붕괴 — $E$ 가
> 낮다. 여러 특성이 수명을 고르게 나누면 — 다양하고 균형 잡힌 소비
> 영역 — $E$ 가 높고, 최대 $\log N$ 까지 간다. 고객의 위상 구조의
> *다양성* 을 하나의 숫자로 압축한 것이며, Atienza et al. (2019) 에 의해
> 안정성이 증명되어 있다 — 작은 입력 섭동이 값을 뒤흔들 수 없다.

다섯 lifetime 통계는 더 무디지만 보완적이다. **평균** 은 강건하고
넓은 스케일의 구조가 얼마나 있는지 말하고, **표준편차** 는 균일한 특성
집합과, 안정 구조에 일시적 노이즈가 섞인 혼합을 구분하며, **min/max/median** 은
수명 분포의 나머지를 스케치한다. 학습 없는 여섯 개의 고정 숫자가 모든
고객에 대해 같은 방식으로 계산된다.

> **역사적 배경.** "고정 벡터화" 의 계보는 그 자체로 작은 역사다.
> Persistence Entropy 는 Rucco et al. (2016) 이 Shannon 의 1948년
> 엔트로피를 수명 분포에 적용해 체계화했고, Atienza et al. (2019) 가
> 나중에 그 안정성을 증명했다. 프로젝트가 준비해 두고 출력에서는 뺀
> Persistence Landscape 는 Bubenik 의 2015년 기여로, diagram 요약을
> 평균과 가설 검정이 비로소 성립하는 Banach 공간에 올려놓은 한 수다.
> 오프라인 파이프라인은 이 가족에서 가장 싸고 해석 가능한 멤버를
> 의도적으로 고르고, *학습 가능한* 벡터화는 PersLay 에 맡긴다.

## 70D 구성

70D TDA 블록은 159D Domain 피처 그룹에서 가장 큰 몫을 차지하며, 세 개의
하위 블록으로 나뉜다.

> **계약은 그 뒤로 갱신됐다.** 위 734D 는 V1 피처 계약이다. 프로젝트는
> 2026-07-02 자로 V2 strict 계약으로 전환했고, 운영 입력 폭은 **4035D** 다 —
> 734D 는 폐기된 게 아니라 V2 의 _공유 베이스 8그룹_ 으로 남고, 여기에
> lag/rolling/product 계열 3301D 가 덧붙어 4035D 가 된다.

| 하위 블록 | 차원 | 출처 | 호몰로지 | 스코프 |
| --- | --- | --- | --- | --- |
| `tda_short` | 24D | 90일 앱 로그 | $H_0, H_1$ | Global + Local |
| `tda_long` | 36D | 12개월 카드 거래 | $H_0, H_1$ | Global + Local |
| `phase_transition` | 10D | 전반기/후반기 윈도우 diff | $H_0, H_1$ | 시간적 |

`tda_short` 와 `tda_long` 둘 다 같은 차원 산술을 따른다 —
**6피처 × 2 Betti ($H_0, H_1$) × 2 스코프 (Global, Local) = 24D**.
Global 스코프는 전체 고객의 샘플링된 모집단(최대 10,000건)에 대해
위상을 계산해 *배경* 형태를 주고, Local 스코프는 그 한 고객의 윈도우만
써서 그의 *고유* 형태를 준다. $H_2$(공동)는 어디서도 쓰이지 않는다.
고객당 ~200점 규모의 구름에서 공동은 안정적으로 형성되지 않고, $H_2$ 는
심플렉스 수 폭발로 $O(n^3)$ 비용을 치른다.

> 짚어둘 문서상 단서 하나. 텐서 구성 표는 `tda_long` 을 **36D** 로
> 적고 `feature_schema.yaml` 도 `tda_long_001`–`tda_long_036` 이름을
> enumerate 하지만, 차원 산식과 실제 `extract_long_features()` 출력은
> **24D**($H_0, H_1$ × 6 × 2)다. 참조서는 이 어긋남을 두 번째 피처
> 집합이 아니라 스키마 쪽의 알려진 이슈, 즉 스키마 이름 수와 산출 컬럼
> 수의 불일치로 기록한다. 70D 분할은 스키마 라벨대로(24 + 36 + 10) 보고하되, 실제
> `tda_long` 페이로드는 24개 산출 컬럼임을 함께 밝힌다.

`phase_transition` 블록은 구조적으로 다르다 — diagram 요약이 아니라
*diff* 다. **PD Distance (4D) + Transition Detection (6D)** 로 나뉜다.

- **PD Distance (4D)** — `pt_W1_distance_h0`, `pt_W1_distance_h1`(각
  차원에서 전반기/후반기 diagram 사이의 Wasserstein-1 거리), 그 합
  `pt_total_topological_change`, 그리고 `pt_max_structural_shift`(두
  bottleneck 변동 중 큰 쪽).
- **Transition Detection (6D)** — sigmoid 로 눌린 상전이 확률에
  imminence, frequency, direction, magnitude, 그리고 phase 분류 신뢰도.

상전이 확률은 물리학에서 곧장 빌려온 regime 감지기다.

$$ P_{\text{transition}} = \frac{1}{1 + e^{-2(\Delta_{\text{total}} - \tau)}} $$

여기서 $\tau = 0.5$. 총 위상 변화량 $\Delta_{\text{total}}$ 이 임계값을
넘으면 sigmoid 가 확률을 1 쪽으로 튕긴다. 함수 형태는 통계역학의
Fermi–Dirac 분포와 같고, 임계값에서의 급격한 전환은 물리적 상전이를
닮았다 — 피처가 그렇게 명명된 이유다. 짝을 이루는 `_classify_phase`
루틴은 Betti 와 엔트로피 추세를 근거로 고객을 안정기, 성장기, 수축기,
혼란기, 전이기의 다섯 regime 으로 분류한다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="220" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a5f">TDA 70D 블록 구성</text>
  <rect x="40" y="50" width="171" height="44" rx="4" fill="#4f46e5"/>
  <text x="125" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">tda_short</text>
  <text x="125" y="86" text-anchor="middle" font-size="10" fill="#dbeafe">24D</text>
  <rect x="215" y="50" width="171" height="44" rx="4" fill="#0d9488"/>
  <text x="300" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">tda_long</text>
  <text x="300" y="86" text-anchor="middle" font-size="10" fill="#ccfbf1">36D (스키마)</text>
  <rect x="390" y="50" width="130" height="44" rx="4" fill="#d97706"/>
  <text x="455" y="70" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">phase_trans</text>
  <text x="455" y="86" text-anchor="middle" font-size="10" fill="#fef3c7">10D</text>
  <text x="280" y="112" text-anchor="middle" font-size="10" fill="#64748b">= 70D, 159D Domain 그룹의 최대 조각</text>
  <text x="125" y="142" text-anchor="middle" font-size="10.5" font-weight="700" fill="#4f46e5">24D = 6 × 2 Betti × 2 스코프</text>
  <g font-size="9" fill="#64748b">
    <text x="125" y="160" text-anchor="middle">6피처: 엔트로피 · 평균 · 표준편차</text>
    <text x="125" y="174" text-anchor="middle">· 최소 · 최대 · 중앙값</text>
    <text x="125" y="190" text-anchor="middle">Betti: H₀, H₁ — 스코프: Global, Local</text>
  </g>
  <text x="455" y="142" text-anchor="middle" font-size="10.5" font-weight="700" fill="#d97706">10D = 4 + 6</text>
  <g font-size="9" fill="#64748b">
    <text x="455" y="160" text-anchor="middle">PD 거리(4): W₁ H₀/H₁,</text>
    <text x="455" y="174" text-anchor="middle">합, 최대 변동</text>
    <text x="455" y="190" text-anchor="middle">감지(6): 확률, imminence…</text>
  </g>
  <line x1="290" y1="130" x2="290" y2="200" stroke="#e2e8f0" stroke-width="1"/>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">70D 블록: tda_short (24D) + tda_long (스키마 라벨 36D) + phase_transition (10D). 각 24D 요약 블록은 2개 호몰로지 차원과 2개 스코프에 걸친 6개 고정 피처이고, 10D phase 블록은 4D diagram 거리에 6D 전이 감지기를 더한 것이다.</figcaption>
</figure>

## 왜 오프라인인가 — 비용 논증

persistent homology 계산이 비싼 부분이고, 확장성이 나쁘다. Vietoris–Rips
복합체는 심플렉스가 최대 $2^n - 1$ 개일 수 있고, 경계 행렬 축소는
$O(n^3)$ 이다. 이를 매 forward 마다 모델 *안에서* 하는 것은 지속
불가능하다. 오프라인 배치는 비용을 한 번 치르고 결과를 동결한다. 이것이 70D
블록이 PersLay Expert 와 별도로 존재하는 아키텍처적 이유 전부다.

파이프라인은 3단 엔진과 공격적 샘플링으로 비용을 묶어 둔다.

- **엔진 우선순위 체인.** `PersistenceExtractor` 가 사용 가능한 가장
  빠른 백엔드를 자동 선택한다 — **Ripser++**(CUDA, 가장 빠름) →
  **Ripser**(C++ 바인딩; CuPy 로 계산한 GPU 거리 행렬과 조합 시 10–50×
  가속) → **giotto-tda**(CPU, 가장 풍부한 API). Ripser++ 실패(CUDA
  불일치)는 자동으로 CPU Ripser 로 폴백한다.
- **점 샘플링.** 고객당 `max_points = 1000`, *시간 계층화 샘플링* 으로
  추출한다 — 시간순 데이터를 $k = \min(10, n/10)$ 버킷으로 나누고, 각
  버킷에서 균등 샘플링하며, 순서를 보존한다. 이것이 $O(n^2)$ 거리 행렬과
  호몰로지 비용을 제한하면서 시간 커버리지를 유지한다.
- **메모리.** 거리 행렬은 $O(n^2)$ — $n=5000$ 에서 ~95MB, $n=10000$
  에서 ~381MB(float32) — 그래서 CuPy 경로는 이를 청크(청크 크기 2000)로
  나누고 12GB+ GPU 메모리를 권장한다. 선택적 Sparse Rips 모드
  (`use_sparse`, 기본 off)는 거리 임계값 너머의 심플렉스를 무시해
  정확도와 속도를 맞바꾼다.

안정적 위상에 필요한 이력이 부족한 콜드스타트 고객에게는 4단계
점진적 전략(점이 너무 적을 때 통계 기반 근사)을 대신 쓰지만, 그건 별도
주제다.

## 여기서 멈추는 이유

위상을 두 레인으로 나눠 — TDA-1 의 학습 가능한 온라인 PersLay Expert 와
여기 고정 오프라인 피처 — 오프라인 레인을 끝까지 따라갔다. 원시 로그가
6D 점 구름으로, Ripser 가 persistence diagram 으로, 그리고 의도적으로
*고정* 된 벡터화(호몰로지 차원당 Persistence Entropy 와 lifetime 통계
5개)가 70D 블록으로. 그 블록이 `tda_short`(24D), `tda_long`(산출 24 /
스키마 라벨 36D), `phase_transition`(10D)으로 분해되는 것을 봤고, 이
모두를 오프라인으로 — Airflow 배치에서 한 번 — 하는 것이 734D 입력 텐서
안에서 $O(n^3)$ 계산을 감당할 유일한 방법인 이유를 봤다.

이 블록이 *포착하지 못하는* 것은 고객 행동의 *순차적* 구조다 — 그가 어떤
잠재 regime 에 있는지, 그리고 regime 사이의 전이를 위상적 diff 가 아니라
확률 과정으로 보는 것. 그것이 Model-Derived 그룹의 은닉 마르코프 모델
summary 의 일이다 — Viterbi 경로와 잠재 상태에 대한 사후 분포에서 나온
다섯 숫자. 이것이 다음 편 **HMM-1** 의 주제다.
