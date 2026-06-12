---
title: "[Study Thread] ECON-1 — 습관의 가격: 경제학 파생 피처와 그것을 읽는 Shared Expert"
date: 2026-06-06 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, economics, elasticity, behavioral-economics, features, expert]
lang: ko
excerpt: "한 번의 소비 뒤에 숨은 경제 구조 — 블랙박스 모델이 충분히 쓰지 못하는 신호 — 와, 프로젝트가 한 세기의 미시경제학을 17차원 피처로 바꾸는 방식: 소득 탄력성, CRRA 효용과 소비 평활화, 환불 비율 가격 민감도 프록시, HHI 와 Shannon 다각화, 그리고 항상/일시 소득 분해. 그리고 그 17D 가 PLE-adaTT Domain Expert 로 들어가는 지점."
series: study-thread
part: 17
alt_lang: /2026/06/06/economics-expert-en/
next_title: "TDA-3 — 대규모 오프라인: Airflow 배치에서 persistence 피처 추출하기"
next_desc: "수백만 고객의 persistence diagram 을 클러스터를 녹이지 않고 계산하는 법: PersistenceExtractor 배치, 세션 점 구름에 대한 Ripser/Ripser++, Parquet diagram 저장소, 그리고 O(2^n) 위상 계산을 학습 핫패스에서 밀어내는 비용 트레이드."
next_status: draft
---

*"Study Thread" 시리즈의 한 편으로, 이번에는 경제학 파생 피처 블록 —
PLE-adaTT Domain Expert 가 읽는 17D 조각 — 을 다룬다. 영문/국문 병렬로,
체크카드 추천 모델이 소비의 표면 통계가 아니라 그 뒤의* 경제 구조 *를
어떻게 읽는지 정리한다. 출처는 온프렘 프로젝트
`기술참조서/Economics_피처_기술_참조서` 이고, 전체 PDF 는 이 서브스레드
마지막 편에 첨부한다. TDA 서브스레드가 행동의* 형태 *가 무엇을 뜻하는지
물었다면, 이번 편은 더 소박하지만 더 오래된 질문을 던진다 — Marshall 의
탄력성, Friedman 의 항상소득, CRRA 효용, 한 세기의 미시경제학은 평균과
분산이 버리는 카드 원장에서 무엇을 읽어낼 수 있는가?*

> **핵심 주장, 단도직입으로.** 두 고객이 똑같이 월 ₩3M 을 써도 경제적으로는
> 정반대 사람일 수 있다. 한 명은 ₩4M 을 벌며 꾸준히 쓰고, 다른 한 명은
> ₩2M + 분기 보너스로 몰아 쓴다. 기술 통계 — 평균, 분산, 왜도 — 는 같은
> ₩3M 을 본다. *경제학* 피처는 높은 항상소득의 저축형 vs 보너스 기반
> 캐시백 후보를 본다. 이 글은 그 차이를 모델에게 보이게 만드는 17개
> 차원과, 그것을 소비하는 Expert 에 관한 것이다.

## 경제 구조가 모델이 덜 쓰는 신호인 이유

고객을 묘사할 때 우리는 기술 통계 — 평균, 표준편차, 왜도, 첨도 — 를
꺼내 든다. 이들은 숫자 흐름의 *형태* 를 요약하지만 그 형태가 *왜*
나타났는지는 말하지 않는다. raw 집계만 받은 블랙박스 모델은 사전 지식
없이 소비의 인과 구조를 *맨바닥에서 재발견* 해야 한다. 대개 상관관계로
타협한다.

경제학 피처는 그 사전 지식을 공짜로 건넨다. 참조서는 순수 통계를 이기는
세 이유를 든다.

- **인과 구조를 인코딩한다.** 경제학 이론은 소득 변화가 소비 변화를
  *어떻게* 끌고 가는지 — 단순 공존이 아니라 방향 — 를 말한다. 그걸
  피처에 녹이면 모델은 raw 상관이 아니라 행동의 *방향성* 을 학습한다.
- **해석 가능하다.** `income_elasticity = 1.3` 이라는 값은 곧바로
  "소득 1% 상승 시 이 고객 소비는 1.3% 증가 — 사치재 성향" 으로 읽힌다.
  잠재 좌표가 아니라 XAI 친화적 진술이다.
- **도메인 정규화돼 있다.** 무차원 경제 비율(탄력성, CV, HHI)은 소득
  규모, 물가 수준, 화폐 단위에 불변이다. 연 ₩30M 고객과 연 ₩300M 고객의
  탄력성을 *직접 비교* 할 수 있어, 정규화가 돌기도 전에 피처 스케일링
  부담의 절반이 사라진다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <text x="280" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">같은 평균 소비, 정반대 경제 구조</text>
  <text x="140" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">고객 A — 안정 (소득 ₩4M)</text>
  <line x1="40" y1="150" x2="240" y2="150" stroke="#cbd5e1" stroke-width="1"/>
  <polyline points="48,120 80,122 112,118 144,121 176,119 208,120 232,120" fill="none" stroke="#0d9488" stroke-width="2"/>
  <text x="140" y="178" text-anchor="middle" font-size="10" fill="#64748b">높은 항상소득 · 낮은 변동</text>
  <text x="140" y="194" text-anchor="middle" font-size="10" fill="#64748b">→ 정기 할인 카드</text>
  <line x1="280" y1="44" x2="280" y2="205" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="56" text-anchor="middle" font-size="12" font-weight="700" fill="#e11d48">고객 B — 폭발적 (₩2M + 보너스)</text>
  <line x1="320" y1="150" x2="520" y2="150" stroke="#cbd5e1" stroke-width="1"/>
  <polyline points="328,140 360,142 392,90 424,141 456,139 488,84 512,140" fill="none" stroke="#e11d48" stroke-width="2"/>
  <text x="420" y="178" text-anchor="middle" font-size="10" fill="#64748b">낮은 항상소득 · 높은 bonus_frequency</text>
  <text x="420" y="194" text-anchor="middle" font-size="10" fill="#64748b">→ 캐시백 카드</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">두 고객 모두 월평균 ₩3M. 평균은 둘을 분리하지 못하지만, 항상/일시 소득 분해와 bonus_frequency 는 분리한다 — 그리고 서로 다른 카드를 가리킨다.</figcaption>
</figure>

## 원장 뒤의 수요 함수

피처 집합 전체의 척추는 미시경제학의 수요 함수다. 참조서는 소비자 수요를

$$ Q_d = f(P,\ Y,\ P_s,\ P_c,\ T,\ E) $$

로 쓰고, 각 추상 변수를 프로젝트가 카드와 예금 원장에서 실제로 관측 가능한
것에 대응시킨다.

| 이론 변수 | 관측 데이터 | 피처 |
| --- | --- | --- |
| $Y$ — 소득 | 월간 예금 입금액 | `permanent_income_avg`, `transitory_income_avg` |
| $Q_d$ — 수요 | 월간 카드 사용 총액 | 탄력성 계산 기반 |
| $P$ — 가격 | 환불 비율 (가격 불만족 프록시) | `price_sensitivity` |
| $T$ — 취향 | MCC 코드별 지출 분포 | `spending_diversification`, `category_hhi` |
| $E$ — 기대 | 전반기/후반기 소비 비율 | `discount_rate_proxy` |

매번 두는 수는 동일하다 — 미시경제학이 *무한소* 변화로 정의하는 양을,
SQL 쿼리가 36개월 윈도우에서 계산할 수 있는 *이산 월간* 통계로 근사한다.

## 탄력성 — 무차원의 일꾼

탄력성은 "한 변수의 퍼센트 변화당 다른 변수의 퍼센트 변화" 다. 수요의
소득 탄력성이 대표 사례다.

$$ \varepsilon_Y = \frac{\partial Q}{\partial Y}\cdot\frac{Y}{Q} $$

부호와 크기가 보편적 분류를 준다 — $\varepsilon_Y > 1$ 은 *사치재*
(소득 상승 시 소비가 더 빠르게 증가), $0 < \varepsilon_Y < 1$ 은 *필수재*,
$\varepsilon_Y < 0$ 은 *열등재* (소득 상승 시 소비 *감소*). 단위가
상쇄되므로 원이든 달러든, 수준이든 로그든 같은 값이다 — 바로 그래서
규모가 크게 다른 고객 사이에서도 그대로 통한다.

> **역사적 배경.** 탄력성은 Alfred Marshall 의 것으로, *Principles of
> Economics* (1890)에서 수요 곡선의 기하학적 성질로 읽어냈다. 수요 함수
> 자체는 더 거슬러 올라가 — Antoine-Augustin Cournot (1838)가 수요를
> 가격의 수학적 함수로 처음 표현했다. 탄력성 자체를 측정할 수 없을 때
> 관측된 행동에서 가격 민감도를 재는 *행동적 프록시* 기법은 George
> Stigler (1961)의 *탐색 비용(search cost)* 경제학에서 내려온다 — 가격에
> 민감한 소비자일수록 더 많이 탐색하며, 환불은 일종의 *사후적* 가격
> 탐색이다.

연속 편미분은 원장에서 계산할 수 없으므로, 프로젝트는 월간 호탄력성
변화의 평균으로 근사한다.

$$ \hat{\varepsilon}_Y = \frac{1}{T}\sum_{t=1}^{T}\frac{\Delta S_t / S_{t-1}}{\Delta Y_t / Y_{t-1}} $$

코드에서는 하나의 집계로 압축되며, 소비가 0인 달이 0으로 나누지 않도록
`NULLIF` 가드를 둔다.

```python
income_elasticity = AVG(
    (monthly_spending - prev_monthly_spending)
    / NULLIF(prev_monthly_spending, 0)
)
```

> **수식 직관.** $S_t$ 는 $t$월 소비, $Y_t$ 는 $t$월 소득이다. 각 항은
> "이번 달 소득이 움직인 만큼 대비 소비는 몇 % 움직였나?" 를 묻고,
> 피처는 그 비율을 윈도우 위에서 평균 낸다. 교과서의 미분을 대신하는
> 실무적이고 이산적인 얼굴 — 점 탄력성을 대신하는 *호(arc)* 탄력성이다.

나머지 두 탄력성 계열 피처도 같은 정신의 프록시다. **가격 민감도** 는
환불 비율을 쓴다 — $\text{price\_sensitivity} = 1 -
\overline{\text{refund\_ratio}}$ — 높으면 *둔감* (환불 적음), 낮으면
가격을 탐색하는 고객이다. **교차 카테고리 탄력성** 은 월간 카테고리 수의
변동계수, $\sigma(\text{category\_count})/\mu(\text{category\_count})$
다 — 소비 카테고리 집합이 달마다 출렁이는 고객은 *새* 카테고리의 카드
혜택에 반응할 가능성이 높다.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="300" fill="#f8fafc" rx="8"/>
  <text x="260" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">소득 탄력성 = Engel 곡선의 기울기</text>
  <line x1="70" y1="250" x2="470" y2="250" stroke="#64748b" stroke-width="1.2"/>
  <line x1="70" y1="250" x2="70" y2="56" stroke="#64748b" stroke-width="1.2"/>
  <text x="270" y="282" text-anchor="middle" font-size="12" fill="#1e3a5f">소득 Y</text>
  <text x="30" y="155" text-anchor="middle" font-size="12" fill="#1e3a5f" transform="rotate(-90 30 155)">소비 Q</text>
  <line x1="70" y1="250" x2="430" y2="70" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="430" y="64" font-size="9.5" fill="#94a3b8" text-anchor="end">ε = 1 (비례)</text>
  <path d="M 70 250 Q 280 240 430 80" fill="none" stroke="#4f46e5" stroke-width="2.2"/>
  <text x="438" y="86" font-size="11" font-weight="700" fill="#4f46e5">ε &gt; 1 사치재</text>
  <path d="M 70 250 Q 200 120 430 110" fill="none" stroke="#0d9488" stroke-width="2.2"/>
  <text x="438" y="116" font-size="11" font-weight="700" fill="#0d9488">0 &lt; ε &lt; 1 필수재</text>
  <path d="M 70 200 Q 250 215 430 240" fill="none" stroke="#e11d48" stroke-width="2.2"/>
  <text x="438" y="244" font-size="11" font-weight="700" fill="#e11d48">ε &lt; 0 열등재</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">소득 탄력성은 비례선 대비 Engel 곡선의 국소 기울기다 — 위로 볼록(사치재), 아래로 오목(필수재), 우하향(열등재). 부호만으로도 고객을 프리미엄 vs 할인 카드로 라우팅한다.</figcaption>
</figure>

## 효용, 위험, 그리고 소비의 평활화

탄력성 뒤에는 소비자의 *효용* 함수 $u(C)$ 가 있고, 합리적 소비의 두
정의적 성질을 만족한다 — 많을수록 좋고, 추가 1단위의 가치는 줄어든다.

$$ u'(C) > 0, \qquad u''(C) < 0 $$

프로젝트는 CRRA(상대적 위험회피 불변) 형태에 닻을 내린다.

$$ u(C) = \frac{C^{1-\gamma}}{1-\gamma}, \qquad \gamma > 0,\ \gamma \neq 1 $$

여기서 $\gamma$ 는 상대적 위험회피계수다. 무게가 실리는 것은 그 귀결이다 —
$\gamma$ 가 큰 고객은 소비 변동을 *싫어하고* 그것을 평탄화하려 하며,
프로젝트는 이를 높은 `consumption_smoothing` 으로 읽는다. 결정적으로
$\gamma$ 는 소비 규모에 걸쳐 *상수* 다(Arrow–Pratt $-C\,u''/u' =
\gamma$). 이것이 부유한 고객과 가난한 고객의 평활화를 같은 축에서 비교할
수 있게 하는 이론적 근거다.

> **수식 직관.** 오목성 $u'' < 0$ 이 전부다. Jensen 부등식에 의해
> $u(\mathbb{E}[C]) > \mathbb{E}[u(C)]$ 를 뜻한다 — 평균이 같은 두 소비
> 경로 앞에서 합리적 소비자는 *확실한* 경로를 선호한다. 이 평활성에 대한
> 선호가 정확히 `consumption_smoothing` 이 재는 것이며, 평균은 같지만
> 변동이 큰 소비자와 안정적인 소비자가 여기서 갈리는 이유다.

`consumption_smoothing` 은 월간 소비의 변동계수 *역수*,
$\mu/\sigma$ 로 계산된다 — 참조서는 이것이 구조적으로 신호 대 잡음비이며
사실상 소비의 *샤프 비율* 이라고 짚는다. 평활화가 높으면 예측 가능한
소비자이고, 카드 실적 조건 달성의 신뢰도가 높다.

## 포트폴리오 집중도 — 산업조직론에서 빌려오기

두 피처가 더 있다 — 소비가 *얼마나 퍼져 있는지* 를 묘사하며, 정보 이론과
독점규제에서 도구를 빌린다. **지출 다각화** 는 MCC 카테고리 지출 비중에
대한 Shannon 엔트로피로,

$$ H = -\sum_{i=1}^{N} s_i \ln s_i $$

모든 지출이 한 카테고리면 0, 균등 분배면 최대($\ln N$)다. **카테고리
HHI** 는 Herfindahl–Hirschman 지수 — 미국 DOJ 가 합병 심사에 쓰는 바로
그 수 — 를 시장 점유율 대신 지출 비중에 적용한 것이다.

$$ \text{HHI} = \sum_{i=1}^{N} s_i^2 $$

둘은 설계상 상보적이다 — HHI 의 제곱 항은 *지배적* 카테고리에 민감하고,
엔트로피의 로그 항은 *꼬리* 까지 닿는다. 함께 쓰면 "주 카테고리가 얼마나
집중됐나" 와 "도대체 몇 개 카테고리가 의미 있나" 를 모두 읽을 수 있고,
추천 규칙은 직접적이다 — HHI < 0.15 → 폭넓은 다혜택 카드, HHI > 0.25 →
카테고리 특화 카드(주유, 통신).

## 17개 차원, 조립

Economics 블록은 두 추출기가 17D 로 쌓인 것이다. 아래 표가 피처 집합
전체이며, 참조서의 요약에 근거한다.

| 피처 | 경제적 의미 | 계산 방식 |
| --- | --- | --- |
| `income_elasticity` | 사치재 / 필수재 / 열등재 성향 | 월간 호탄력성 $\Delta S/S_{\text{prev}}$ 의 평균 |
| `price_sensitivity` | 가격 민감도 (환불 프록시) | $1 - \overline{\text{refund\_ratio}}$ |
| `cross_category_elasticity` | 카테고리 폭의 변동성 | 월간 카테고리 수의 CV |
| `spending_diversification` | 카테고리 간 지출 분산도 | Shannon 엔트로피 $-\sum s_i \ln s_i$ |
| `category_hhi` | 상위 카테고리 집중도 | $\sum s_i^2$ |
| `spending_risk` | 월 총지출의 예측 불가능성 | 월간 지출의 CV |
| `discount_rate_proxy` | 시간 선호 (즉시 vs 지연) | 전반기 / 후반기 소비 비율 |
| `savings_propensity` | 저축 성향 | 음수 순지출 / 양수 순지출 비율 |
| `consumption_smoothing` | 소비 변동 회피 | CV 역수, $\mu/\sigma$ |
| `permanent_income_avg` | 장기 안정 소득 수준 | $\text{mean}(\hat{Y}^P)$ (Friedman PIH) |
| `permanent_income_stability` | 항상소득 안정성 | CV, $\sigma(\hat{Y}^P)/\mu(\hat{Y}^P)$ |
| `permanent_income_growth` | 관찰 기간 항상소득 성장률 | $(\hat{Y}^P_T - \hat{Y}^P_1)/\hat{Y}^P_1$ |
| `permanent_income_trend` | 장기 추세 방향 | 선형 회귀 기울기 |
| `transitory_income_avg` | 평균 일시소득 (정기 보너스 없으면 ≈0) | $\text{mean}(\hat{Y}^T)$ |
| `transitory_income_volatility` | 소득 불확실성 | $\sigma(\hat{Y}^T)$ |
| `transitory_income_max` | 최대 보너스 규모 사건 | $\max(\hat{Y}^T)$ |
| `bonus_frequency` | 큰 보너스 빈도 | $\hat{Y}^T > 0.5\,\hat{Y}^P$ 인 월 비율 |

앞의 아홉은 `financial_behavior` 그룹(9D), 소득 분해는
`income_decomposition` 그룹(8D)으로, raw 예금 입금액을 Friedman 의
항상소득가설에 따라 *항상* 성분과 *일시* 성분으로 분해한다 — 구현에서는
36개월 관찰 윈도우 위에서 12개월 이동평균(기본값), HP 필터(월간 데이터
$\lambda = 14{,}400$), Kalman 필터 중 하나를 선택해 추정한다. 합쳐서
8 + 9 = 17D.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="210" fill="#f8fafc" rx="8"/>
  <rect x="18" y="78" width="96" height="54" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1"/>
  <text x="66" y="100" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">카드 +</text>
  <text x="66" y="116" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">예금 원장</text>
  <rect x="160" y="34" width="150" height="56" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="235" y="56" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">IncomeDecomposition</text>
  <text x="235" y="72" text-anchor="middle" font-size="10" fill="#64748b">PIH · HP · Kalman → 8D</text>
  <rect x="160" y="120" width="150" height="56" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="235" y="142" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">FinancialBehavior</text>
  <text x="235" y="158" text-anchor="middle" font-size="10" fill="#64748b">탄력성 · 효용 · HHI → 9D</text>
  <rect x="360" y="78" width="96" height="54" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="408" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">Economics</text>
  <text x="408" y="118" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">17D</text>
  <rect x="500" y="50" width="84" height="46" rx="6" fill="#f8fafc" stroke="#1e3a5f" stroke-width="1"/>
  <text x="542" y="70" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">Domain</text>
  <text x="542" y="85" text-anchor="middle" font-size="10" fill="#64748b">159D</text>
  <rect x="500" y="114" width="84" height="46" rx="6" fill="#1e3a5f"/>
  <text x="542" y="134" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Main Tensor</text>
  <text x="542" y="149" text-anchor="middle" font-size="10" fill="#cbd5e1">734D</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="114" y1="92" x2="158" y2="62"/><polygon points="158,62 149,62 154,70"/>
    <line x1="114" y1="118" x2="158" y2="148"/><polygon points="158,148 149,140 154,148"/>
    <line x1="310" y1="62" x2="358" y2="98"/><polygon points="358,98 349,92 350,100"/>
    <line x1="310" y1="148" x2="358" y2="112"/><polygon points="358,112 350,110 349,118"/>
    <line x1="456" y1="100" x2="498" y2="78"/><polygon points="498,78 489,78 494,86"/>
    <line x1="542" y1="96" x2="542" y2="112"/><polygon points="542,112 538,104 546,104"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">두 추출기(8D + 9D)가 17D Economics 블록을 구성하고, 이는 734D 메인 텐서의 159D Domain 그룹 안에서 TDA/GMM/Mamba 와 합류한다.</figcaption>
</figure>

## Economics 블록이 PLE 에 꽂히는 지점

17개 차원은 허공에 떠 있지 않다. 734D 메인 텐서(644D normalized + 90D
raw power-law)에서 Economics 는 159D **Domain** 피처 그룹의 한 조각으로,
TDA(70D), GMM(22D), Mamba(50D) 와 나란히 놓인다. 거기서 PLE-adaTT 모델의
**Domain Expert** 로 들어가, 17D 를 소비해 18개 태스크에 걸친 체크카드
추천을 날카롭게 만든다 — "평균 소비가 같지만 구조가 다른 고객" 을
구분하게 해주는데, 이는 raw 평균이 뭉개버리는 바로 그 경우다.

피처는 다운스트림에서 두 구체적 방식으로 제값을 한다. 첫째,
`DebitCardIncomeConstraints` 레이어가 소득 분해를 하드와 소프트 규칙으로
바꾼다 — `permanent_income_avg` < ₩3M 이면 프리미엄 tier 카드 제외,
높은 `bonus_frequency` 는 캐시백 카드를 순위에서 끌어올린다. 둘째,
파이프라인에 순서 의존성이 있다 — Economics 피처 4개
(`permanent_income_avg`, `transitory_income_volatility`,
`income_elasticity`, `spending_risk`)가 *GMM 클러스터링의 40D 입력* 에
포함되므로, Economics 는 GMM 보다 *먼저* 계산돼야 한다 — 소득 구조가
말 그대로 고객이 어느 클러스터에 떨어질지를 빚는다.

## 여기서 멈추는 이유

우리는 불편함에서 출발했다 — 평균과 분산은 경제적으로 정반대인 두 고객을
같은 사람으로 본다. 미시경제학의 수요 함수를 관측 가능한 원장 위로
걸었고, 소득 탄력성을 Engel 곡선 기울기로 읽었으며,
`consumption_smoothing` 을 CRRA 효용과 Jensen 부등식에 근거시키고,
HHI 와 Shannon 엔트로피를 산업조직론에서 빌려, PLE-adaTT Domain Expert 가
읽는 17개 차원 — `income_decomposition`(8D) 위의 `financial_behavior`
(9D) — 을 조립했다.

아직 하지 *않은* 것은 전혀 다른 Expert 의 무거운 작업이다 — 오프라인
기계 장치. TDA 서브스레드는 학습 루프 안에서 계산하기엔 너무 비싼 피처를
가진 Expert 를 남겼고, 프로젝트는 그것을 Airflow 배치로 답한다 — 수백만
고객에 대해 Ripser/Ripser++ 를 돌리는 `PersistenceExtractor` 가
diagram 을 Parquet 로, 핫패스 밖에서, 한 번만 쓴다. 그 오프라인 추출이
어떻게 만들어지고 왜 그 비용 트레이드가 불가피한지가 다음 편 **TDA-3**
의 주제다.
