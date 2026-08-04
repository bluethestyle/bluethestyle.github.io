---
title: "[Study Thread] GROUND-1 — 숫자와 사유 사이의 사전: 피쳐 역매핑"
date: 2026-06-08 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, grounding, feature-mapping, attribution, explainability]
lang: ko
excerpt: "모델은 734차원으로 말하고, 추천 사유는 사람의 사실로 말해야 한다. 이 글은 그 둘 사이의 사전 — 역매핑을 푼다. 피쳐 인덱스를 이름 붙은 금융 수치로 되돌리는 법, Integrated Gradients 가 지배 피쳐를 고르는 법, YAML 규칙 기반 사실 추출기가 그것을 결정론적 narrative fact 로 바꾸는 법, 그리고 이 모든 것이 아무것도 지어내지 않고 사유 생성으로 넘어가는 법."
series: study-thread
part: 26
alt_lang: /2026/06/08/grounding-reverse-mapping-en/
next_title: "SERVE-1 — vLLM 위의 Qwen: 사유를 다시 쓰는 LLM 서빙하기"
next_desc: "그라운딩된 사실을 실제 문장으로 바꾸려면 진짜 모델이 필요하다. 다음 편은 서빙 쪽으로 — 폐쇄망에서 Qwen 을 vLLM 으로 돌리기: OpenAI 호환 엔드포인트, 수백만 고객에 대한 배치 생성, JSON-mode 제약, 그리고 배치 전용 사유 파이프라인의 지연/처리량 트레이드오프."
next_status: draft
---

*"Study Thread" 시리즈의 그라운딩 서브스레드 1편, 영문/국문 병렬.
출처는 온프렘 프로젝트 `기술참조서/그라운딩_피쳐역매핑_기술_참조서` 이고,
전체 PDF 는 서브스레드 마지막 편에 첨부한다. 이 프로젝트의 추천 모델은
말하지 않는다. 734차원 피쳐 벡터를 받아 확률 하나 — 숫자 — 를 내놓을
뿐이다. 그러나 상담사가 읽을 수 있는, 또는 규제 당국이 보관할 수 있는
사유는 사람의 사실로 만들어져야 한다 — "해외 결제가 활발합니다",
"여행 업종 관심이 높아지고 있습니다". 역매핑은 그 두 언어 사이의* 사전
*이다 — 모델의 수치 피쳐에서 의미로 되돌아가는 다리.* REASON *스레드가
완성된 문장으로 다시 쓸 그라운딩된 사실을, 여기서 먼저 만든다.*

> **왜 존재하는가.** 블랙박스 추천은 세 가지로 동시에 실패한다. 영업
> 현장은 설명할 수 없는 추천을 무시한다. 컴플라이언스 문서는 "AI 모델
> 출력"을 적합성 근거로 기재할 수 없다. 그리고 점수만 들여다보는 데이터
> 과학자는 모델이 허위 상관 — 가령 지역 코드가 소득을 대리하는 — 을
> 학습해도 잡아낼 수 없다. 역매핑은 셋을 한 번에 닫는다. 지배 피쳐를
> 사람이 읽는 금융 언어로, 규제가 인정하는 근거로, 분석가가 감사할 수
> 있는 형태로 바꾼다. 이것이 빠지면 신뢰 루프 — *예측 → 기여도 → 그라운딩
> → 조립 → 작성 → 설득 → 전환 → 개선* — 가 바로 "예측"과 "상담사" 사이에서
> 끊어진다.

## 간극: 숫자는 사유가 아니다

학습된 PLE-adaTT 모델은 734차원 피쳐 벡터를 18개 태스크 — CTR, CVR,
Churn 등 — 의 확률 점수로 매핑한다. 그 점수는 *무엇을* 추천할지에
답한다. *왜* 인지는 한 마디도 하지 않는다.

피쳐 자체도 사람에게 도움이 안 된다. `chemical_kinetics_003` 이라는
피쳐가 중요도 1위일 수 있지만, 그게 무슨 뜻인지 아는 상담사는 세상에
없다. 정직한 번역 — "새 업종 시도율(카테고리 전환 가속도)" — 은 숫자 어디에도
없다. 건너야 할 간극은 하나가 아니라 둘이다.

1. **기여도 간극.** 734개 숫자 중 *어느* 것이 이 예측을 지배했는가?
   원본 피쳐 벡터는 모든 차원을 동등하게 취급하지만, 우리는 어느 것이
   중요했는지 알아야 한다.
2. **명명 간극.** 피쳐 인덱스 *i* 가 중요했다는 걸 알았다 해도, *i* 가
   실제로 무슨 사람-친화 수치를 가리키며, 그 *값* 은 금융적으로 무엇을
   뜻하는가?

<figure style="margin:24px auto;max-width:580px;">
<svg viewBox="0 0 580 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="580" height="230" fill="#f8fafc" rx="8"/>
  <text x="290" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">모델과 사유 사이의 두 간극</text>
  <rect x="30" y="70" width="120" height="100" rx="6" fill="#1e3a5f12" stroke="#1e3a5f" stroke-width="1"/>
  <text x="90" y="60" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">734D 벡터</text>
  <g font-family="monospace" font-size="9" fill="#64748b">
    <text x="45" y="92">0.12  0.85  …</text>
    <text x="45" y="108">0.04  0.71  …</text>
    <text x="45" y="124">0.93  0.08  …</text>
    <text x="45" y="140">0.55  0.27  …</text>
    <text x="45" y="156">…그냥 숫자</text>
  </g>
  <text x="232" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">간극 1</text>
  <text x="232" y="114" text-anchor="middle" font-size="9" fill="#64748b">어느 게 중요?</text>
  <text x="232" y="128" text-anchor="middle" font-size="9" fill="#64748b">(IG 기여도)</text>
  <rect x="300" y="78" width="130" height="84" rx="6" fill="#d9770612" stroke="#d97706" stroke-width="1"/>
  <text x="365" y="60" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">상위 피쳐</text>
  <g font-family="monospace" font-size="9" fill="#64748b">
    <text x="312" y="98">feat #341  +0.12</text>
    <text x="312" y="114">feat #088  +0.08</text>
    <text x="312" y="130">feat #602  +0.07</text>
    <text x="312" y="150">…여전히 암호</text>
  </g>
  <text x="478" y="100" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">간극 2</text>
  <text x="478" y="114" text-anchor="middle" font-size="9" fill="#64748b">무슨 뜻?</text>
  <text x="478" y="128" text-anchor="middle" font-size="9" fill="#64748b">(역매핑)</text>
  <rect x="522" y="86" width="44" height="68" rx="6" fill="#0d9488" />
  <text x="544" y="116" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">사람의</text>
  <text x="544" y="130" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">사실</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="150" y1="120" x2="298" y2="120"/><polygon points="298,120 290,116 290,124"/>
    <line x1="430" y1="120" x2="520" y2="120"/><polygon points="520,120 512,116 512,124"/>
  </g>
  <text x="290" y="206" text-anchor="middle" font-size="11" fill="#64748b">"해외 결제 활발; 여행 업종 관심 상승"</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">두 번의 건넘: 기여도가 어느 차원이 점수를 지배했는지 찾고, 역매핑이 그게 무슨 뜻인지 이름 붙인다. 둘을 거친 뒤에야 숫자는 사실이 된다.</figcaption>
</figure>

첫 번째 간극은 기여도가, 두 번째는 역매핑이 닫는다. 이 글은 대부분 두
번째에 관한 것이지만, 둘은 떼어놓을 수 없으니 첫 번째부터 짧게 짚는다.

## 기여도: 어느 숫자가 점수를 지배했는가

어떤 피쳐를 번역할지 알려면, 시스템은 예측을 입력 피쳐로 되돌려
귀속시켜야 한다. 이를 위해 **Integrated Gradients**(IG; Sundararajan et al., 2017)를
쓴다. IG 는 직설적으로 묻는다 — 어떤 피쳐가 baseline(여기선 영벡터)에서
실제 값까지 올라가는 동안, 예측이 그만큼 얼마나 움직였는가? baseline
$\mathbf{x}'$ 에서 입력 $\mathbf{x}$ 까지의 직선 경로를 따라 gradient 를
적분한다.

$$ \mathrm{IG}_i(\mathbf{x}) = (x_i - x'_i)\,\int_0^1 \frac{\partial F\big(\mathbf{x}' + \alpha(\mathbf{x}-\mathbf{x}')\big)}{\partial x_i}\, d\alpha $$

IG 가 단순 gradient 대신 채택된 핵심 이유는 **완전성(completeness)** 이다.
기여도의 합이 정확히 예측 차이가 된다.

$$ \sum_{i=1}^{644} \mathrm{IG}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}') $$

CTR 예측이 0.73 이고 baseline 이 0.15 라면, 644개 기여도의 합은 정확히
0.58 이 된다 — 누수 없고, 중복 계산 없다. 실무적으로 이 보장이야말로
역매핑된 사유가 점수를 가리키는 데 그치지 않고 *설명한다* 고 주장할 수
있게 하는 근거다.

> **설계 직관.** 완전성은 다른 모자를 쓴 미적분학 기본정리다.
> $\int_a^b f'(t)\,dt = f(b)-f(a)$ — 도함수를 적분하면 양 끝값의 차이가
> 돌아온다. IG 는 644차원에서 같은 일을 한다. baseline→입력 경로를 따른
> gradient 의 선적분은, Gradient Theorem 에 의해 경로와 무관하게 정확히
> $F(\mathbf{x})-F(\mathbf{x}')$ 이다. 그래서 기여도가 누수되지 않는다 —
> 그리고 평평한 영역에서 0 에 가깝게 포화되어 실제 기여를 *과소평가* 하는
> 단순 gradient 로는 충분치 않은 이유이기도 하다.

출력은 734차원 기여도 벡터 $\mathbf{a}$ 다. 절대값 기준 상위 $K$ 개를
취하면, 번역할 가치가 있는 소수의 피쳐가 손에 들어온다. 그 핸드오프 —
상위 피쳐가 들어가고, 이름 붙은 사실이 나오는 — 가 역매핑 엔진의 일이다.

## 피쳐 역매핑: 인덱스에서 이름 붙은 수치로

역매핑은 이 프로젝트 고유 용어로, 하나의 변환을 가리킨다.

$$ \mathrm{ReverseMap}:\ \big(\mathbf{x}\in\mathbb{R}^d,\ \mathbf{a}\in\mathbb{R}^d\big)\ \longrightarrow\ \{(r_k, s_k, t_k)\}_{k=1}^{K} $$

여기서 $\mathbf{x}$ 는 원본 피쳐 벡터, $\mathbf{a}$ 는 IG 기여도, $r_k$ 는
*피쳐 범위* 이름(`profile`, `domain`, …), $s_k$ 는 그 범위의 요약 점수,
$t_k$ 는 금융 언어 텍스트다. 어려운 부분은 두 가지를 동시에 해내는
것이다 — **차원 축소**(644개 부동소수점 → 약 열 개 문장) *와* **의미
부여**, 그러면서도 신호를 버리지 않기.

엔진 — `feature_reverse_mapper.py` 의 `FeatureReverseMapper` — 은 734차원을
평평한 죽으로 취급하지 않는다. 벡터에는 *알려진 레이아웃* 이 있다.
734D 역매핑 입력은 V1 호환 구조다 — 644 정규화 피처 + 90 raw power-law
피처. 644 정규화 차원은 7개 연속 범위로 분해된다.

> **계약은 그 뒤로 갱신됐다.** 위 734D 는 V1 피처 계약이다. 프로젝트는
> 2026-07-02 자로 V2 strict 계약으로 전환했고, 운영 입력 폭은 **4035D** 다 —
> 734D 는 폐기된 게 아니라 V2 의 _공유 베이스 8그룹_ 으로 남고, 여기에
> lag/rolling/product 계열 3301D 가 덧붙어 4035D 가 된다.

$$ 238_{\text{profile}} + 91_{\text{multi\_source}} + 84_{\text{extended}} + 159_{\text{domain}} + 27_{\text{model\_derived}} + 24_{\text{multi\_disc}} + 21_{\text{merchant}} = 644 $$

각 범위는 이름과 설명을 가진 연속 인덱스 슬라이스이며, 작은
`FeatureRange` 데이터클래스에 담긴다.

```python
@dataclass
class FeatureRange:
    start: int       # 예: domain 의 경우 413
    end: int         # 예: domain 의 경우 572
    name: str        # "domain"
    description: str  # "TDA(70) + GMM(22) + Mamba(50) + Economics(17)"
```

그래서 IG 가 엔진에 피쳐 인덱스 470 을 넘기면, 조회가 말한다 — 470 은
`domain`(413–572), 그중 TDA 서브레인지에 속한다 — 이건 인구통계가 아니라
*위상적 persistence* 피쳐다. 인덱스는 더 이상 익명이 아니다. 거처가
있고, 그 거처가 번역에 쓸 사전을 결정한다.

모든 범위에 반복되는 핵심 루프는 하나의 패턴이다.

$$ t_k = \mathcal{M}_k\big(g(\mathbf{x}[s_k:e_k])\big) $$

— 서브레인지를 슬라이싱하고, $g$ 로 집계하고(`np.mean`, `np.argmax`, 또는
임계값 비교), 그 결과를 $\mathcal{M}_k$ — 숫자를 구절로 바꾸는 범위 전용
사전 — 에서 조회한다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">슬라이스 → 집계 → 조회</text>
  <text x="300" y="50" text-anchor="middle" font-size="10" fill="#64748b">644D 벡터, 이름 붙은 범위로 배치됨</text>
  <g>
    <rect x="40"  y="60" width="120" height="26" fill="#1e3a5f22" stroke="#1e3a5f" stroke-width="0.8"/>
    <rect x="160" y="60" width="60"  height="26" fill="#0d948822" stroke="#0d9488" stroke-width="0.8"/>
    <rect x="220" y="60" width="55"  height="26" fill="#d9770622" stroke="#d97706" stroke-width="0.8"/>
    <rect x="275" y="60" width="100" height="26" fill="#4f46e522" stroke="#4f46e5" stroke-width="0.8"/>
    <rect x="375" y="60" width="40"  height="26" fill="#64748b22" stroke="#64748b" stroke-width="0.8"/>
    <rect x="415" y="60" width="35"  height="26" fill="#e11d4822" stroke="#e11d48" stroke-width="0.8"/>
    <rect x="450" y="60" width="30"  height="26" fill="#0d948822" stroke="#0d9488" stroke-width="0.8"/>
  </g>
  <text x="100" y="78" text-anchor="middle" font-size="8" fill="#1e3a5f">profile</text>
  <text x="190" y="78" text-anchor="middle" font-size="8" fill="#0d9488">multi_src</text>
  <text x="325" y="78" text-anchor="middle" font-size="8" fill="#4f46e5">domain</text>
  <rect x="78" y="58" width="34" height="30" fill="none" stroke="#e11d48" stroke-width="2" rx="2"/>
  <text x="95" y="104" text-anchor="middle" font-size="9" fill="#e11d48" font-weight="700">RFM 슬라이스</text>
  <line x1="95" y1="108" x2="95" y2="132" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="95,132 91,124 99,124" fill="#cbd5e1"/>
  <rect x="40" y="135" width="160" height="44" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="120" y="154" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">g = np.mean</text>
  <text x="120" y="170" text-anchor="middle" font-size="9" fill="#64748b">Recency 평균 = 0.82</text>
  <line x1="200" y1="157" x2="248" y2="157" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="248,157 240,153 240,161" fill="#cbd5e1"/>
  <rect x="250" y="128" width="170" height="60" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="335" y="146" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">M (사전)</text>
  <text x="335" y="162" text-anchor="middle" font-size="8.5" fill="#64748b">&gt; 0.7 → "최근 매우 활발"</text>
  <text x="335" y="176" text-anchor="middle" font-size="8.5" fill="#64748b">0.4–0.7 → "보통" · &lt; 0.4 → "저조"</text>
  <line x1="420" y1="157" x2="468" y2="157" stroke="#cbd5e1" stroke-width="1.4"/>
  <polygon points="468,157 460,153 460,161" fill="#cbd5e1"/>
  <rect x="470" y="138" width="100" height="40" rx="6" fill="#0d9488"/>
  <text x="520" y="162" text-anchor="middle" font-size="9.5" font-weight="700" fill="#fff">"최근 매우</text>
  <text x="520" y="174" text-anchor="middle" font-size="9.5" font-weight="700" fill="#fff">활발"</text>
  <text x="300" y="220" text-anchor="middle" font-size="10" fill="#64748b">같은 패턴이 7개 범위 × 다수 서브레인지에 반복되어 하나의 설명으로 조립된다</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">역매핑 핵심 루프: 알려진 서브레인지를 슬라이싱하고, 스칼라로 집계하고, 도메인 설계 사전에서 조회한다. 반복되고 조립되어 644개 숫자가 한 단락이 된다.</figcaption>
</figure>

구체적 예시 하나. `profile` 안쪽 오프셋 100–150 에 있는 RFM 블록
(Recency / Frequency / Monetary)을 보자. 엔진은 이를 슬라이싱하고, 각 축의
평균을 내고, 세 번의 임계값 조회를 돌린다.

| 축 | 값 | 임계값 규칙 | 매핑된 레이블 |
| --- | --- | --- | --- |
| Recency | 0.82 | `> 0.7` | 최근 매우 활발 (7일 이내 거래) |
| Frequency | 0.55 | `0.4–0.7` | 중빈도 (월 5~15회) |
| Monetary | 0.31 | `< 0.4` | 소액 소비 (월 30만원 미만) |

`/` 로 연결하면 **"최근 매우 활발 / 중빈도 / 소액 소비".** 매뉴얼 없이
사람이 읽는 문장으로 바뀐 숫자다.

같은 기계가 더 풍부한 집계기를 동원해 어려운 범위도 처리한다. 아래 표는
피쳐 *그룹* 이 역매핑된 사실로 어떻게 매핑되며, 어떤 집계가 그 일을 하는지
보여준다.

| 피쳐 그룹 (차원) | 집계기 | 역매핑된 사실 (예) |
| --- | --- | --- |
| RFM (profile 내 50D) | 축별 평균 + 임계값 | "최근 매우 활발 / 중빈도 / 소액 소비" |
| 신용/투자 (financial 88D) | 명명 비율 + 임계값 | "한도 소진율 높음 → 신용 위험 주의; 투자 성향" |
| TDA persistence (domain) | 스칼라 임계값 | "소비 패턴 안정적(80% 이상 유지); 행동 전환 가능성 높음" |
| HMM triple-mode (48D) | 16D 상태에 `argmax` | "생애주기: 성숙기; 구매 여정: 고려 단계" |
| GMM 군집 (22D) | 확률에 `argmax` | "주요 세그먼트: VIP (소속 확률 73.2%)" |
| chemical_kinetics (다학제 24D 중 6D) | 평균 + 해석 구간 | "소비 가속도 높음 — 새 업종 시도 중" |
| MCC 계층 (가맹점 21D) | Level-1/2 + 반지름 | "특정 가맹점 충성도 높음" |

마지막 행들의 메타포 세탁에 주목하라. `chemical_kinetics_003` 같은
이름은 빌려온 과학 용어다. 그 범위의 사전은 이를 *비즈니스 개념* —
"새 업종 시도율" — 으로 번역한다. 이것이 바로 개념 기반 설명의
접근이다 — 원본 피쳐 단위가 아니라 사람이 추론하는 단위("여행 성향",
"절약 패턴")로 설명하는 것.

> **역사적 배경.** 이 모든 것 아래의 이론은 모델보다 오래됐다. 기여도는
> Lloyd Shapley 의 1953년 *"A Value for n-Person Games"* 로 거슬러
> 올라간다 — 협력 게임의 이익을 공정하게 나누는 유일한 방법, 네 공리
> (효율성, 대칭성, 더미, 가법성)로 정의된다. 이것이 순수 경제학에 60년 앉아
> 있다가 Lundberg & Lee 의 2017년 SHAP 논문에서 머신러닝용으로
> 재발견됐다. Integrated Gradients 는 같은 해 동일한 *공리적* 경로로
> 도착했다 — 좋은 기여도가 만족해야 할 성질을 먼저 정의하고 유일한 방법을
> 유도하되, Shapley 의 이산 부분집합 합을 연속 경로 적분으로 교체했다.
> 역매핑은 그 마지막, 화려하지 않은 한 걸음이다 — 그 공정하고 누수 없는
> 기여도를 지점장이 입으로 옮길 수 있는 문장으로 바꾸는.

## fact_extractor: YAML 규칙으로 만드는 결정론적 사실

역매핑은 범위별로 흐르는 금융 산문을 만든다. 그러나 *원자적이고 검증
가능한 사실* 을 만드는 두 번째, 보완적 채널이 있다 — 그리고 LLM 을 한
번도 호출하지 않고 해낸다. 이것이 `FactExtractor`(`fact_extractor.py`),
AWS `core/recommendation/reason/fact_extractor.py` 에서 이식된 Mem0
스타일 규칙 기반 사실 압축 레이어다.

발상은 의도적으로 소박하다. 고객의 피쳐가 평범한 dict 로 들어온다.
YAML config 가 규칙을 나열한다 — 각 규칙은 이름, 불리언 조건, 그 조건이
필요로 하는 피쳐. 조건이 참이면 규칙의 이름이 사실 문자열이 된다. 그게
엔진의 전부다.

```yaml
# fact_extraction.yaml
rules:
  - name: "예적금 중심 포트폴리오"
    condition: "deposit_balance_ratio >= 0.7"
    required_features: ["deposit_balance_ratio"]
  - name: "최근 3개월 펀드 관심 증가"
    condition: "fund_view_count_3m >= 5"
    required_features: ["fund_view_count_3m"]
  - name: "리스크 회피 성향"
    condition: "risk_tolerance_score <= 0.3"
    required_features: ["risk_tolerance_score"]
```

```python
extractor = FactExtractor("configs/fact_extraction.yaml")
facts = extractor.extract({
    "deposit_balance_ratio": 0.75,
    "fund_view_count_3m": 8,
    "risk_tolerance_score": 0.2,
})
# → ["예적금 중심 포트폴리오",
#    "최근 3개월 펀드 관심 증가",
#    "리스크 회피 성향"]
```

이것을 역매퍼와 나란히 둘 가치가 있게 만드는 세 가지 성질:

- **결정론적.** 같은 dict 가 들어가면 매번 같은 사실이 나온다 — 샘플링도,
  temperature 도, 드리프트도 없다. 사실은 캐시되고, diff 되고, 감사될 수
  있다.
- **싸고 배치 가능.** `extract_batch()` 가 같은 규칙을 DataFrame 에 row
  단위로 돌린다. 이 프로젝트에선 약 5.3M 고객에 대해 한 번에 돌렸다. 모델
  호출이 없으니 GPU 도, 고객당 지연도 없다.
- **구조적으로 안전.** 규칙 조건은 `eval` 되지만, 봉쇄된 네임스페이스
  안에서다 — `__builtins__` 는 비워지고, 작은 허용 리스트
  (`abs`, `min`, `max`, `len`, `round`, `int`, `float`, …)와 고객 자신의
  피쳐만 주입된다. 없는 피쳐를 참조하거나 예외를 던지는 조건은 배치를
  죽이지 않고 조용히 건너뛴다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="240" fill="#f8fafc" rx="8"/>
  <text x="300" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">YAML 규칙 → 사실, 결정론적으로</text>
  <rect x="28" y="60" width="130" height="120" rx="6" fill="#1e3a5f10" stroke="#1e3a5f" stroke-width="1"/>
  <text x="93" y="52" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">피쳐 dict</text>
  <g font-family="monospace" font-size="8.5" fill="#64748b">
    <text x="40" y="84">deposit_ratio</text><text x="148" y="84" text-anchor="end" fill="#1e3a5f">0.75</text>
    <text x="40" y="106">fund_view_3m</text><text x="148" y="106" text-anchor="end" fill="#1e3a5f">8</text>
    <text x="40" y="128">risk_score</text><text x="148" y="128" text-anchor="end" fill="#1e3a5f">0.2</text>
    <text x="40" y="150">…</text>
  </g>
  <rect x="220" y="50" width="160" height="140" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="300" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">YAML 규칙 (eval)</text>
  <g font-size="8.5" fill="#64748b">
    <text x="232" y="76">deposit_ratio ≥ 0.7</text><text x="368" y="76" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="104">fund_view_3m ≥ 5</text><text x="368" y="104" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="132">risk_score ≤ 0.3</text><text x="368" y="132" text-anchor="end" fill="#0d9488" font-weight="700">✓</text>
    <text x="232" y="160">campaign_resp ≥ 0.6</text><text x="368" y="160" text-anchor="end" fill="#e11d48" font-weight="700">✗</text>
  </g>
  <text x="300" y="182" text-anchor="middle" font-size="8" fill="#64748b">봉쇄 네임스페이스 · 없는 피쳐 → 건너뜀</text>
  <rect x="430" y="64" width="150" height="112" rx="6" fill="#0d948812" stroke="#0d9488" stroke-width="1"/>
  <text x="505" y="56" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">사실 리스트</text>
  <g font-size="8.5" fill="#0f766e">
    <text x="442" y="88">• 예적금 중심</text>
    <text x="442" y="110">• 펀드 관심 증가</text>
    <text x="442" y="132">• 리스크 회피</text>
  </g>
  <text x="505" y="164" text-anchor="middle" font-size="8" fill="#64748b">LLM 없음 · 캐시 가능 · 감사 가능</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="158" y1="120" x2="218" y2="120"/><polygon points="218,120 210,116 210,124"/>
    <line x1="380" y1="120" x2="428" y2="120"/><polygon points="428,120 420,116 420,124"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">fact_extractor 는 각 YAML 규칙을 봉쇄 네임스페이스 안에서 고객 dict 에 평가한다. 통과한 조건은 사실이 되고, 없거나 실패한 것은 건너뛴다. 루프에 모델이 없다.</figcaption>
</figure>

두 채널은 중복이 아니다. 역매핑은 지배 *기여된* 피쳐를 읽고 범위 수준
산문("이 고객이 왜 높은 점수를 받았는가")을 쓴다. fact_extractor 는 전체
피쳐 dict 를 큐레이션된 규칙집에 대조해 짧고 단단하며 개별적으로 참인
주장을 내놓는다. 사유 생성기는 둘 다 받는다 — 유창함을 위한 내러티브,
그라운딩을 위한 원자적 사실.

## 기여도가 후보 사실이 되는 법

조각을 순서대로 두면 데이터 흐름은 짧다. 상위 $K$ IG 피쳐는 정확히 두
곳에서 쓰이며, 그 이중성이 핵심이다.

1. `reverse_map()` 반환값의 `top_features` 로 — 클라이언트와 하위 산문에
   직접 넘겨져 *"이 피쳐가 왜 중요한가"* 에 답한다.
2. `ContextAssemblyAgent.assemble()` 의 `ig_top_features` 입력으로 —
   *도구 선택의 근거* 가 되어 *"이 피쳐를 더 깊이 설명하려면 어떤 소스가
   필요한가"* 에 답한다.

이 두 번째 쓰임에서 기여도가 파이프라인을 조종하기 시작한다. 조립
에이전트는 각 상위 피쳐를 그 범위로 되돌리고, 범위가 어떤 도구를 쏠지
결정하게 한다. 지배 피쳐가 `multidisciplinary` 에 있으면 다학제 해석기를
부르고, `extended_source` 에 있으면 상담 이력을 끌어온다. 점수를 실제로
지배한 피쳐만 깊은 컨텍스트의 비용을 쓸 자격을 얻는다.

| 상위 피쳐 범위 | 에이전트가 쏠 수 있는 도구 |
| --- | --- |
| `profile` | `reverse_map`, `query_context` |
| `multi_source` | `reverse_map`, `get_consultation` |
| `extended_source` | `reverse_map`, `get_consultation` |
| `domain` | `reverse_map`, `interpret_multi` |
| `multidisciplinary` | `interpret_multi`, `query_similar` |
| `model_derived` | `reverse_map`, `query_similar` |
| `merchant_hierarchy` | `reverse_map` |

그 다음 "풍부도 Tier" 가 몇 개의 도구가 돌 수 있는지 상한을 둔다(tier 1:
최대 5개; tier 3: `reverse_map` 만) — 컨텍스트 예산이 고객이 실제로
지닌 신호량을 따라가게. 이 모든 것의 출력 — 역매핑 산문, 추출된 사실,
상담 스니펫, 유사 고객 히트 — 이 단일 컨텍스트 번들로 조립된다.

## 사유 생성으로의 핸드오프

그라운딩 단계는 최종 문장을 쓰지 않는다. 작성자가 필요로 하는 모든 것을
준비한다. 배치 순서로 컴포넌트가 협력한다.

1. **역매핑** (`FeatureReverseMapper`) — 정규화 벡터 + IG → 범위별 금융
   언어.
2. **사실 추출** (`FactExtractor`) — 피쳐 dict → 결정론적 narrative fact,
   LLM 없음.
3. **컨텍스트 저장** (`LanceContextVectorStore`) — 매핑된 텍스트를
   임베딩해, 사실과 상담 요약과 함께 `customer_context` 테이블에 저장.
4. **컨텍스트 조립** (`ContextAssemblyAgent`) — IG → 도구 선택 → 역매핑 /
   상담 / 다학제 / 유사 고객 소스를 LLM 입력용 단일 번들로 병합.
5. **사유 생성** (L1 템플릿 → L2a LLM 리라이트) — 번들 + 사실 리스트가
   실제 문장의 입력이 된다.

사실 리스트는 L2a/L2b 프롬프트에 직접 주입되며, 거기서의 역할은 좁지만
중요하다 — **환각 감소.** 리라이트 LLM 은 언어를 자연스럽게 만드는 데
자유롭지만, 사전 검증된 결정론적 사실 집합을 앵커로 받는다 — 다시
표현할 뿐, 지어내지 않는다. 이것이 GROUND-1 이 REASON 스레드를 만나는
이음새다 — 여기서는 그라운딩된 재료를 모두 만들고, 사유 생성기가 그것을
고객이 읽는 산문으로 다시 쓴다.

## 안전장치

조용히 거짓말하는 그라운딩 파이프라인은 없느니만 못하므로, 여러 가드가
내장돼 있다.

- **런타임 완전성 검사.** IG 기여도는 $F(\mathbf{x}) - F(\mathbf{x}')$ 로
  합산되어야 하므로, 그 항등식을 런타임에 단언할 수 있다. 합이 깨지면
  사유를 오염시키기 전에 깨진 기여도를 표시한다.
- **샌드박스 규칙 평가.** fact_extractor 의 `eval` 은 비워진
  `__builtins__` 와 허용 리스트 네임스페이스로 돌며, 예외를 던지거나 없는
  피쳐를 참조하는 규칙은 건너뛴다 — 나쁜 규칙은 사실 하나를 망칠 뿐,
  배치를 망치지 않는다.
- **차원 불일치의 우아한 처리.** 피쳐 벡터 길이와 피쳐명 리스트가 어긋나면
  (V1↔V2 차원 전환기의 실제 위험), `reverse_map()` 은 경고 로그를 남기고
  계속 간다 — 의도적이다, 버전 스큐가 배치를 막지 못하게.
- **배치 SQL 폴백.** `batch_reverse_map()` 은 DuckDB SQL 로 Parquet 을
  읽고, 컬럼 불일치 시 암호화된 customer id 만 필요한
  `_batch_reverse_map_simple()` 로 자동 전환한다.
- **출력 품질 게이트.** 하류에서 `L2QualityValidator` 가 생성된 사유를
  층화 샘플링으로 사실성 / 관련성 / 자연스러움에 채점하고, `fail` 판정은
  *silent risk* 로 간주되어 차단된다 — 그라운딩된 사실에서 벗어난 사유가
  고객에게 닿지 않도록.

이것들이 함께 그라운딩 단계를 *시끄럽고 국소적으로* 실패하게 만든다 —
깨진 기여도, 잘못된 규칙, 드리프트한 문장은 잡혀서 격리될 뿐, 자신감 있게
들리는 거짓말로 조용히 배포되지 않는다.

## 여기서 멈추는 이유

간극에서 출발했다 — 모델은 734차원 벡터를 받아 숫자 하나를 내놓고,
사유에는 사람의 사실이 필요하다 — 그리고 그 위에 사전을 놓았다. Integrated Gradients 가 누수
없는, 공리에 뒷받침된 기여도로 지배 피쳐를 고르고, 역매퍼가 알려진 각
피쳐 범위를 슬라이싱해 도메인 설계 사전에서 집계를 조회하며,
fact_extractor 가 YAML 규칙집과 루프 내 모델 없이 원본 피쳐 dict 를
결정론적이고 감사 가능한 주장으로 증류한다. 두 흐름은 컨텍스트
조립기로 흘러들고, 반대편으로 그라운딩된 재료 번들이 — 리라이트가
거스르면 안 되는 사실과 함께 — 나온다.

*하지* 않은 것은 문장 쓰기다. 그라운딩된 사실은 여전히 실제 언어
모델에 의해 유창하고 고객 친화적인 한국어로 바뀌어야 한다 — 그리고
폐쇄망에서 그것은 모델을 우리가 직접 띄운다는 뜻이다. 다음 편은
서빙 쪽으로 건너간다 — **SERVE-1 — vLLM 위의 Qwen**, OpenAI 호환
엔드포인트, 수백만 고객에 대한 배치 생성, JSON-mode 제약, 그리고 배치
전용 사유 파이프라인의 지연/처리량 트레이드오프. 사전은 지었다. 다음은
작성자를 고용할 차례다.
