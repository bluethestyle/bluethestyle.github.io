---
title: "[Study Thread] SCORING-1 — 호출할 서버가 없다: 배치 추론, Parquet 저장소, 그리고 raw 스코어가 확률이 아닌 이유"
date: 2026-06-08 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, inference, scoring, calibration, duckdb, serving]
lang: ko
excerpt: "폐쇄망 금융 시스템이 실시간 추론 서버 하나 없이 추천을 서빙하는 방법 — 모든 것을 배치로 미리 계산해 Parquet 저장소에 적재하고 DuckDB 조회로 내보낸다. 증류된 LGBM Student 가 야간 배치로 도는 방식, FD-TVS 스코어러가 15개 이질적 태스크 출력을 하나의 비즈니스 점수로 융합하는 방식, 그리고 raw 스코어 0.7 이 70% 확률과 같지 않은 이유 — 이를 바로잡는 Platt 와 one-vs-rest 캘리브레이션."
series: study-thread
part: 24
alt_lang: /2026/06/08/inference-scoring-en/
next_title: "REASON-1 — 지어내지 않고 '왜'를 말하기: 금융 추천을 위한 계층형 사유 생성"
next_desc: "고객에게 점수와 순위 태스크가 정해지면, 시스템은 마케터가 신뢰할 수 있는 평이한 언어로 그 이유를 말해야 한다. L1 템플릿 레이어가 추출된 사실에 추천을 어떻게 정박시키는지, L2a LLM 리라이트가 환각 없이 어떻게 사람의 말로 만드는지, 그리고 위험한 카피를 침묵시키는 verdict 기반 pass→fail 가드."
next_status: draft
---

*"Study Thread" 시리즈의 추론과 스코어링 서브스레드 1편. 출처는 온프렘
프로젝트 `기술참조서/추론_스코어링_기술_참조서` 이고, 전체 PDF 는
서브스레드 마지막 편에 첨부한다. 앞선 서브스레드들이 모델이 어떻게
학습하는가, 각 Expert 가 무엇을 읽는가를 물었다면, 이번 편은 더
직설적인 운영 질문을 던진다 — 모델이 다 학습되고 나면, 예측은 실제로
어떻게 고객에게 도달하는가? 여기서의 답은 독특하고, 그 이유는 대부분의
ML 시스템이 마주하지 않는 제약이다. 이 시스템은* 폐쇄망 *금융 시스템이고,*
실시간 추론 서버가 아예 없다 *— 요청을 보낼 대상 자체가 없다. 고객에게
보여질 수 있는 모든 것은 미리, 배치로 계산되어, 조회된다.*

> **모든 것을 규정하는 제약.** 대부분의 추천 글은 라이브 서비스를
> 전제한다 — 요청이 들어오고, 모델이 스코어링하고, 수십 밀리초 안에
> 응답이 나간다. 이 시스템엔 그게 없다. 폐쇄망이 통상의 서빙 스택을
> 금지한다 — 실시간 추론 서버 없음, 온라인 피처 캐시 없음, dynamic
> batching 없음. 그래서 아키텍처가 뒤집힌다. 야간 배치가 *모든* 고객을
> 15개 태스크 전부에 걸쳐 스코어링하고, 이를 하나의 비즈니스 점수로
> 융합해, 결과를 Parquet 저장소에 쓴다. 그러면 서빙은 추론이 아니라
> *쿼리* 가 된다 — DuckDB 가 미리 계산된 행을 읽는다. 실시간 재계산
> 경로는 LIVE 코드에 존재하지 않는다. 아래의 모든 설계 선택이 이 한
> 가지 사실에서 따라 나온다.

## 배치 생성, 저장소 조회 — 그 이유

추론 시스템의 첫 갈림길은 *언제* 예측하는가이다. 참조서는 세 가지
선택지를 제시하고, 프로젝트는 환경에 의해 그중 하나로 강제된다.

| 방식 | 하는 일 | 장점 | 단점 |
| --- | --- | --- | --- |
| **배치 추론** | 정해진 시점에 전체 고객 일괄 예측 | 높은 throughput, GPU 활용 극대화, 비용 효율 | 예측 신선도 저하 (stale) |
| 실시간 추론 | 요청 즉시 개별 고객 예측 | 최신 맥락 반영, 즉각 응답 | 높은 latency 비용, 인프라 복잡도 |
| 마이크로배치 | 짧은 간격(초~분)의 소규모 배치 | 둘의 절충안 | 큐잉 관리 복잡 |

시스템은 폐쇄망 제약상 **배치 추론만** 채택했다. 일일 배치가 전체
고객의 예측과 FD-TVS 점수를 산출해 Parquet 저장소에 적재하고, 서빙은
그 저장소를 DuckDB 로 조회한다. 실시간 추론 서버 없음, 온라인 피처
캐시(Redis) 없음, dynamic batching 없음 — 그리고 결정적으로 *실시간
재계산 경로 자체가 LIVE 코드에 존재하지 않는다*. "그냥 모델을
호출한다"는 폴백이 불가능하다. 온라인으로 호출할 모델이 없기 때문이다.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <text x="310" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">배치 생성(야간) → 저장소 → 조회(서빙)</text>
  <rect x="20" y="50" width="430" height="86" rx="8" fill="#1e3a5f08" stroke="#1e3a5f" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="30" y="68" font-size="10" font-weight="700" fill="#1e3a5f">배치 (Airflow, 하루 1회)</text>
  <rect x="32" y="78" width="92" height="44" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1"/>
  <text x="78" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#4f46e5">LGBM</text>
  <text x="78" y="112" text-anchor="middle" font-size="9" fill="#64748b">Student 추론</text>
  <rect x="148" y="78" width="92" height="44" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="194" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">보정</text>
  <text x="194" y="112" text-anchor="middle" font-size="9" fill="#64748b">Platt / OvR</text>
  <rect x="264" y="78" width="92" height="44" rx="6" fill="#fef2f2" stroke="#e11d48" stroke-width="1"/>
  <text x="310" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">FD-TVS</text>
  <text x="310" y="112" text-anchor="middle" font-size="9" fill="#64748b">4단계 점수</text>
  <rect x="372" y="78" width="66" height="44" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="405" y="98" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">쓰기</text>
  <text x="405" y="112" text-anchor="middle" font-size="9" fill="#64748b">parquet</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="124" y1="100" x2="146" y2="100"/><polygon points="146,100 138,96 138,104"/>
    <line x1="240" y1="100" x2="262" y2="100"/><polygon points="262,100 254,96 254,104"/>
    <line x1="356" y1="100" x2="370" y2="100"/><polygon points="370,100 362,96 362,104"/>
  </g>
  <ellipse cx="310" cy="172" rx="60" ry="14" fill="#1e3a5f" />
  <rect x="250" y="172" width="120" height="30" fill="#1e3a5f"/>
  <ellipse cx="310" cy="202" rx="60" ry="14" fill="#1e3a5f"/>
  <text x="310" y="180" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">Parquet</text>
  <text x="310" y="196" text-anchor="middle" font-size="9" fill="#cbd5e1">저장소</text>
  <line x1="405" y1="122" x2="340" y2="160" stroke="#cbd5e1" stroke-width="1.4"/><polygon points="340,160 350,159 344,167" fill="#cbd5e1"/>
  <rect x="470" y="150" width="130" height="74" rx="8" fill="#0d948808" stroke="#0d9488" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="480" y="168" font-size="10" font-weight="700" fill="#0d9488">서빙</text>
  <text x="535" y="190" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">DuckDB</text>
  <text x="535" y="206" text-anchor="middle" font-size="9" fill="#64748b">SELECT … = 쿼리,</text>
  <text x="535" y="218" text-anchor="middle" font-size="9" fill="#64748b">추론 아님</text>
  <line x1="370" y1="187" x2="468" y2="187" stroke="#0d9488" stroke-width="1.6"/><polygon points="468,187 460,183 460,191" fill="#0d9488"/>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">서빙 표면 전체가 테이블 조회다. 추론은 오프라인에서 한 번 일어나고, 고객 대면 경로는 결코 모델을 돌리지 않는다.</figcaption>
</figure>

트레이드오프는 명시적이다. *신선도* 를 포기한다 — 예측은 마지막 배치
실행만큼 오래됐다 — 그 대가로 throughput, 완전한 GPU 활용, 예측 가능한
비용, 그리고 파일 위 쿼리 엔진에 불과한 운영상 단순한 서빙 레이어를
얻는다. 고객 기반이 크고, 행동이 일 단위 이상의 주기로 움직이며,
밀리초 latency 보다 재현성을 중시하는 규제 환경의 폐쇄망 은행에게, 이
트레이드오프는 타협이 아니다. 올바른 형태다.

> **역사적 배경.** 모델 서빙은 세 세대를 거쳐 진화했다. 1세대
> (2012~2016)는 모델을 API 서버에 직접 로드했다 — `pickle` + Flask —
> Sculley 등의 2014년 "기술 부채의 고이자 신용카드" 논문이 이 방식의
> 운영 부채를 경고하기 전까지. 2세대(2017~2021)는 전용 모델 서버를
> 가져왔다 — TensorFlow Serving, NVIDIA Triton, TorchServe — 그 핵심
> 혁신은 개별 요청을 큐에 모아 GPU 활용을 극대화하는 *dynamic batching*
> 이었다. 3세대(2021~)는 Kubernetes 네이티브 플랫폼(KServe, Seldon)이
> 배포를 선언적 YAML 로 바꿨다. 이 프로젝트는 의도적으로 그 사다리에서
> *내려온다*. 폐쇄망이 라이브 서빙 스택을 배제하므로, 가장 오래된
> 아이디어로 돌아간다 — 모든 것을 미리 계산해 디스크에서 읽는다 —
> 다만 읽는 쪽은 현대적 컬럼형 엔진(DuckDB-over-Parquet)이다.

## DuckDB와 Parquet 위의 스코어링 파이프라인

야간 배치에서 실제로 도는 것은 무엇인가? 프로덕션 추론 경로는 무거운
PLE-adaTT Teacher 를 의도적으로 돌리지 *않는다* — 734D 6-Expert 모델의
forward 는 서빙 규모 볼륨에 너무 느리다. 대신 knowledge distillation 이
일을 경량 **LGBM Student** 에 넘긴다. `dag_lgbm_inference` DAG 가 증류된
트리 앙상블을 폐쇄망 배치로 직접 돌려 예측 Parquet 을 저장소에 쓰고,
서빙은 그것을 DuckDB 로 되읽는다. 이 Teacher→Student 분리가 전략의
전부다 — *학습에서의 풍부함, 서빙에서의 효율성*.

> **계약은 그 뒤로 갱신됐다.** 위 734D 는 V1 피처 계약이다. 프로젝트는
> 2026-07-02 자로 V2 strict 계약으로 전환했고, 운영 입력 폭은 **4035D** 다 —
> 734D 는 폐기된 게 아니라 V2 의 _공유 베이스 8그룹_ 으로 남고, 여기에
> lag/rolling/product 계열 3301D 가 덧붙어 4035D 가 된다.

가지 않은 길 하나: 참조서는 ONNX 변환과 Triton 패키징 경로
(`src/serving/*`)도 문서화한다. 그 경로는 구현돼 있다 — 패키징
산출물이 생성되고, `config.pbtxt` 가 쓰이고, ONNX 는 순수 LightGBM 대비
2~5배 throughput 향상을 약속한다 — 그러나 **미배포** 상태다
(`triton_status=not_deployed`, `triton_packaging_allowed` 기본 false).
향후 옵션이지 LIVE 경로가 아니다. LIVE 경로는, 다시, 배치 LGBM →
Parquet → DuckDB 다.

모든 태스크가 예측을 갖게 되어도, 그 예측들은 여전히 *이질적* 이다 —
출력 공간이 세 가지인 15개 태스크:

- **Binary** (ctr, cvr, churn, retention): $[0,1]$ 의 sigmoid 확률.
- **Multiclass** (nba, life_stage, channel, timing, spending_category, consumption_cycle): 합이 1인 softmax 벡터.
- **Regression** (ltv, engagement, balance_util, spending_bucket, merchant_affinity): 제한 없는 실수값 — 0 ~ 수백만.

이걸 그냥 더할 수는 없다. 순진하게 합산하면 LTV 의 절대 크기(수십만)가
CTR 확률(1보다 작은 소수)을 압도한다. **FD-TVS Scoring Engine** 이
이들을 분별 있게 융합하기 위해 존재한다. 그 Stage 1 은 가중 합 모델
(Weighted Sum Model) — 가장 기본적인 다기준 의사결정 규칙 — 으로, 모든
입력이 $[0,1]$ 의 확률일 것을 요구한다:

$$ S_{\text{task}} = \sum_{i=1}^{n} \beta_i \cdot p_i, \qquad \sum_i \beta_i = 1, \quad p_i \in [0,1] $$

> **수식 직관.** 가중치 $\beta_i \ge 0$ 의 합이 1이면, 이는 *볼록 결합*
> 이다 — 태스크 확률들의 가중 무게 중심. 보상은 자동이다. 모든 $p_i \in
> [0,1]$ 이면 $S_{\text{task}} \in [0,1]$ 도 성립한다. $S = \sum
> \beta_i p_i \le \sum \beta_i \cdot 1 = 1$ 이고 마찬가지로 $\ge 0$ 이기
> 때문이다. 점수는 결코 범위를 벗어나지 못한다 — *모든 입력이 정말로
> 확률이라는 전제 하에서*. 그 전제가 바로 다음 절이 존재하는 이유
> 전부다. 이후 단계들(DNA 적합, TDA 활력, 리스크 페널티, 피로도 감쇠,
> engagement boost)은 이
> 위에 곱셈으로 얹혀, 0에 가까운 인자 하나가 추천 전체에 거부권을 행사할
> 수 있다 — 설계상 리스크 우선이다.

## raw 스코어가 확률이 아닌 이유

여기 하중을 지는 미묘함이 있다. LGBM Student 가 "0.7" 을 출력할 때, 그
숫자는 캘리브레이션된 확률이 *아니다*. $[0,1]$ 에 우연히 들어 있는
모델 스코어일 뿐이다. 모델이 0.7 로 스코어링한 고객들 가운데 정확히 70% 가
실제로 전환할 때에만 그 모델이 **잘 캘리브레이션** 됐다고 말할 수 있다.
형식적으로:

$$ P(Y = 1 \mid \hat{p} = q) = q \qquad \forall\, q \in [0,1] $$

트리 앙상블은, 대부분의 분류기가 그렇듯, 이를 일상적으로 위반한다 —
과대확신(over-confident)하거나 과소확신(under-confident)한다. 그리고
*이* 시스템에서 그것은 미관상 결함이 아니다. 위 FD-TVS Stage 1 가중 합
때문이다. CTR 모델이 과대확신하고 CVR 모델이 과소확신하면, 가중 융합은
조용히 CTR 쪽으로 기운다. 한 태스크의 오캘리브레이션이 *결합된*
비즈니스 점수를 오염시킨다. 여기서 캘리브레이션은 선택적 마감이 아니다.
융합이 의미를 가지기 위한 전제 조건이다.

해법은 holdout split 에서 적합하는 **사후 캘리브레이션** 레이어다.
프로젝트의 `ProbabilityCalibrator` (`src/evaluation/calibration.py`)는
세 방법을 지원한다:

| 기법 | 작동 방식 | 파라미터 | 특성 |
| --- | --- | --- | --- |
| **Platt scaling** | sigmoid 로 스코어 재매핑 | 2개 $(A,B)$ | Binary·소량 데이터에 적합 |
| **Isotonic** | 단조 계단 함수 적합 | 비모수 | 데이터 많을 때 유연; 과적합 주의 |
| **none** | 통과 (clip 만) | — | 기존 동작 |

Binary 의 주력인 Platt scaling 은 John Platt(1999)에서 왔다. 그는
SVM 의 결정값 $f(x)$ 를 확률로 바꿔야 했다. 레시피는 스코어 위의
1차원 로지스틱 회귀다:

$$ P(y{=}1\mid s) = \frac{1}{1+\exp(As+B)} $$

$A, B$ 는 holdout 에서 최대우도로 추정한다. LGBM Booster 는 sklearn
API 를 따르지 않으므로, 프로젝트는 *score-wrap* 모드를 쓴다.
`booster.predict(X_val)` 로 raw 스코어를 얻어, 그 스코어를 검증
라벨에 대해 적합하고(`fit_from_scores`), 추론 시에는 새 스코어를
`transform` 으로 통과시킨다.

<figure style="margin:24px auto;max-width:520px;">
<svg viewBox="0 0 520 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="520" height="240" fill="#f8fafc" rx="8"/>
  <text x="260" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">Platt scaling — raw 스코어 위의 sigmoid</text>
  <line x1="60" y1="200" x2="470" y2="200" stroke="#64748b" stroke-width="1.2"/>
  <line x1="60" y1="200" x2="60" y2="48" stroke="#64748b" stroke-width="1.2"/>
  <text x="265" y="228" text-anchor="middle" font-size="11" fill="#1e3a5f">raw 스코어 s</text>
  <text x="26" y="124" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 26 124)">P(y=1 | s)</text>
  <line x1="60" y1="124" x2="470" y2="124" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="3 3"/>
  <text x="48" y="128" text-anchor="end" font-size="9" fill="#94a3b8">0.5</text>
  <text x="48" y="204" text-anchor="end" font-size="9" fill="#94a3b8">0</text>
  <text x="48" y="52" text-anchor="end" font-size="9" fill="#94a3b8">1</text>
  <path d="M 60 196 C 150 192, 200 186, 245 124 C 290 62, 360 54, 470 51" fill="none" stroke="#0d9488" stroke-width="2.2"/>
  <circle cx="245" cy="124" r="4.5" fill="#d97706"/>
  <text x="252" y="146" font-size="10" fill="#d97706">변곡점 s = −B/A</text>
  <text x="330" y="92" font-size="10" fill="#0d9488" font-weight="700">MLE 로 2개 파라미터 (A, B)</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Platt scaling 은 raw 모델 스코어 s 를 보정된 확률로 매핑하는 단일 로지스틱 곡선을 적합한다 — 단 두 파라미터, A(기울기)와 B(이동), holdout 에서 추정.</figcaption>
</figure>

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 270" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="270" fill="#f8fafc" rx="8"/>
  <text x="280" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">신뢰도 다이어그램 — 보정 전 vs 후</text>
  <line x1="70" y1="230" x2="510" y2="230" stroke="#64748b" stroke-width="1.2"/>
  <line x1="70" y1="230" x2="70" y2="50" stroke="#64748b" stroke-width="1.2"/>
  <text x="290" y="258" text-anchor="middle" font-size="11" fill="#1e3a5f">평균 예측 확률</text>
  <text x="30" y="140" text-anchor="middle" font-size="11" fill="#1e3a5f" transform="rotate(-90 30 140)">실제 양성 비율</text>
  <line x1="70" y1="230" x2="490" y2="50" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="420" y="70" font-size="9.5" fill="#94a3b8">완벽히 보정됨</text>
  <path d="M 70 230 C 170 220, 250 205, 320 150 C 380 105, 440 78, 490 50" fill="none" stroke="#e11d48" stroke-width="2"/>
  <text x="350" y="200" font-size="10" fill="#e11d48" font-weight="700">raw 스코어 (오보정)</text>
  <path d="M 70 230 C 160 195, 250 158, 320 138 C 390 116, 450 78, 490 52" fill="none" stroke="#0d9488" stroke-width="2"/>
  <text x="110" y="150" font-size="10" fill="#0d9488" font-weight="700">Platt / OvR 후</text>
  <line x1="320" y1="150" x2="320" y2="138" stroke="#d97706" stroke-width="6"/>
  <text x="332" y="148" font-size="9.5" fill="#d97706">ECE 간격 ↓</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">신뢰도 다이어그램은 예측을 구간화해 예측 확률 vs 관측 빈도를 찍는다. 빨간 곡선은 대각선에서 처져 있고(과대확신), 보정이 그것을 끌어당긴다. ECE 는 평균 수직 간격이다.</figcaption>
</figure>

그게 통했는지는 어떻게 *측정* 하는가? 신뢰도 다이어그램이 예측을
구간화해, 구간별로 평균 예측 확률을 관측된 양성 비율과 비교한다.
**Expected Calibration Error** 는 구간들에 걸친 평균 간격이고,
프로젝트는 캘리브레이션 품질을 $1 - \text{ECE}$ 로 보고한다:

$$ \text{ECE} = \frac{1}{B}\sum_{b=1}^{B}\big|\,\text{acc}(b) - \text{conf}(b)\,\big|, \qquad \text{cal\_score} = 1 - \text{ECE} $$

여기서 $\text{conf}(b)$ 는 구간 $b$ 의 평균 예측 확률, $\text{acc}(b)$
는 관측 양성 비율이다. 완벽히 보정된 모델은 ECE 0, 점수 1 이다.

### Multiclass: 클래스마다 보정기 하나, 그다음 재정규화

Binary 보정은 스코어 하나를 매핑한다. Multiclass 헤드(nba 12 클래스,
timing 28 클래스)는 softmax 벡터 전체를 내보내고, 벡터를 직접
Platt-scale 할 수는 없다. 프로젝트는 **one-vs-rest** 를 쓴다. 각 클래스
$c$ 에 대해 "이게 클래스 $c$ 인가?" 를 binary 문제로 보고, $p_{\cdot,c}$
를 지시자 $\mathbb{1}[y=c]$ 에 대해 `ProbabilityCalibrator` 로 적합한 뒤,
`{class → calibrator}` dict 로 저장한다. 추론 시:

$$ \tilde{p}_{c} = \text{cal}_c\!\big(p_{c}\big), \qquad \hat{p}_{c} = \frac{\tilde{p}_{c}}{\sum_{k}\tilde{p}_{k}} $$

마지막 나눗셈이 결정적 단계다. 각 클래스를 *독립적으로* 보정하므로
클래스별 결과는 더 이상 합이 1이 아니다 — 재정규화가 적절한 분포를
복원한다. degenerate 클래스(검증 split 에 없는 클래스)는 건너뛰고
보정 없이 통과시켜, 희소 헤드에서도 보정이 결코 죽지 않는다.

> 참조서가 정직하게 밝히는 단서 하나. 기술 참조서에 문서화된 *baseline*
> postprocessor 에서는 raw 예측을 그대로 썼고, 캘리브레이션 레이어
> 추가는 향후 스코어 품질 과제로 적혀 있었다. 여기 설명한
> `ProbabilityCalibrator` 가 그 간극을 메우는 온프렘 구현이다 —
> `calibration_method` config 로 opt-in 이며(기본 `none` 은 기존 동작
> 유지), 보정기는 Student 모델과 함께 태스크별로 pickle 저장된다.
> *존재하는* 배선이다. 특정 프로덕션 실행이 이를 켜는지는 보증이 아니라
> config 결정이다.

## Throughput, 비용, 그리고 이 레이어의 위치

배치 설계의 경제학은 단순하다. 라이브 요청 경로가 없으니, 방어할 tail
latency 예산도 없고, 놀고 있어도 프로비저닝된 추론 플릿 비용도 없다.
비용은 전체 고객 기반에 대한 단일 야간 컴퓨트 윈도우 하나로, latency 가
아니라 *throughput* 에 맞춰 사이징된다 — 정확히 배치 추론과 GPU 활용이
빛나는 영역이다. 서빙 비용은 Parquet 파일 위에서 DuckDB 쿼리를 돌리는
것으로 수렴한다 — 저렴하고, 수평 확장이 자명하며, 특수 서빙 인프라가
필요 없다.

참조서는 *만약* 미배포 Triton/ONNX 경로가 활성화된다면 latency 병목이
어디 살지도 짚는다. preprocessor 의 200+ 피처 JSON 파싱(CPU-bound),
ONNX 트리 앙상블 forward(1000+ 트리일수록 악화), postprocessor 의
JSON 직렬화. 알아둘 만하다 — 그러나 LIVE 배치 경로에서는 이 중 어느
것도 고객 대면 임계 경로에 놓이지 않는다. 고객 대면 추론 자체가 없기
때문이다.

그래서 스코어링 레이어는 두 이웃 사이에 정확히 앉는다. *상류* 는
증류다. PLE-adaTT Teacher 가 풍부한 표현을 학습하고, LGBM Student 가 그
판별력을 배치 스코어링에 충분히 빠른 형태로 물려받는다. *하류* 는
추천과 사유 레이어다. FD-TVS 점수와 순위 태스크가 사유 생성기의 입력이
되어, 숫자를 마케터가 행동에 옮길 수 있는 문장으로 바꿔야 한다. 우리가
스치기만 한 라우팅 안전망 — FallbackRouter 의 3-layer 와 missing LGBM
태스크를 메우는 rule-based baseline — 도 있지만, 그건 별도의 논의에
속한다.

## 여기서 멈추는 이유

이 시스템을 규정하는 단 하나의 제약 — *실시간 추론 서버 없음* — 에서
출발해, 아키텍처가 그 주위로 뒤집히는 것을 봤다. 모든 고객을
스코어링하는 야간 배치, 결과를 담는 Parquet 저장소, 그리고 모델
호출이 아니라 DuckDB 쿼리인 서빙 레이어. 증류된 LGBM Student 가 배치
추론을 수행하고, FD-TVS 엔진이 15개 이질적 태스크 출력을 볼록 가중
합으로 융합하며 — 하중을 지는 디테일 — 그 합이 신뢰할 만하려면 모든
입력이 *진짜* 확률이어야 한다는 것, raw 스코어는 그렇지 않다는 것을
봤다. Platt scaling 이 binary 헤드를, one-vs-rest-플러스-재정규화가
multiclass 헤드를 고치고, ECE 가 그 수리가 버텼는지를 말해준다.

남은 것은 마지막 한 구간이다. 점수와 순위 태스크는 여전히 숫자일
뿐이다. 시스템은 *왜* 인지를 설명해야 한다 — 마케터가 신뢰하고 규제가
감사할 수 있는 언어로, 데이터가 뒷받침하지 않는 사실을 지어내지 않고.
L1 템플릿 레이어가 추출된 사실에 추천을 어떻게 정박시키는지, L2a LLM
리라이트가 그것을 읽히게 어떻게 만드는지, 그리고 위험한 초안을
pass 에서 fail 로 돌리는 verdict 기반 가드 — 이것이 다음 편 **REASON-1**
의 주제다.
