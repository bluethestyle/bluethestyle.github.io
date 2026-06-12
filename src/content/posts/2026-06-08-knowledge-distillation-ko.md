---
title: "[Study Thread] DISTILL-1 — 떠나는 교사: 깊은 PLE Teacher 에서 LightGBM Student 로의 지식 증류"
date: 2026-06-08 12:00:00 +0900
categories: [Study Thread]
tags: [study-thread, distillation, lightgbm, teacher-student, serving]
lang: ko
excerpt: "폐쇄망 배치 시스템이 왜 20GB 짜리 깊은 PLE teacher 를 학습해 놓고 서빙 시점엔 버리는가 — 요청 시 GPU 추론 서버 없이, 저장소 조회로 답하는 태스크별 LightGBM student 로 압축한다. Hinton 의 soft-label 증류, temperature 와 dark knowledge, T²-스케일 손실, fidelity 게이트, 그리고 어떤 모델이 실제로 답할지 정하는 3-layer fallback."
series: study-thread
part: 23
alt_lang: /2026/06/08/knowledge-distillation-en/
next_title: "DISTILL-2 — 추론과 스코어링: Student 모델에서 저장소 조회로"
next_desc: "LightGBM student 의 학습과 등록이 끝나면, 요청 시점에는 어떤 모델도 돌지 않는다. 배치가 모든 고객 × 태스크를 사전 스코어링해 DuckDB-over-Parquet 저장소에 쓰고, 서빙을 키 조회로 바꾸는 방식 — 그리고 경로에 놓인 multiclass shape 어댑터와 consent 게이트."
next_status: draft
---

*"Study Thread" 시리즈의 지식 증류(Knowledge Distillation) 서브스레드
1편, 영문/국문 병렬. 출처는 온프렘 프로젝트
`기술참조서/지식증류_기술_참조서` 이고, 전체 PDF 는 서브스레드 마지막
편에 첨부한다. 앞선 서브스레드들이 Expert 가 무엇을 읽는가(PersLay),
태스크가 어떻게 공유하는가(PLE, adaTT)를 다뤘다면, 이번엔 더 차갑고
운영적인 질문을 던진다 — 며칠을 들여 깊은 멀티태스크 teacher 를 학습한
뒤, 요청 시점에 GPU 추론 서버가 없는 폐쇄망에서 수백만 고객에게 어떻게
서빙할 것인가? 프로젝트의 답은 직설적이다 — 서빙하지 않는다. 작은 트리
모델 한 무리로 증류한 뒤, teacher 는 떠나보낸다.*

> **한 문단 셋업.** PLE-adaTT teacher 는 ~50M 파라미터의 깊은 멀티태스크
> 모델이다 — Expert 네트워크, cluster-aware 헤드, HMM 피처, 15개 활성
> 태스크(18개 정의; `uplift`, `category_uplift` 비활성, `brand_prediction`
> 폐기). 정확하지만
> 비싸다 — **20GB VRAM**, 1,024행 배치 기준 **~50ms**, **8GB+** Docker
> 이미지. 야간 배치로 *수백만* 고객을 스코어링하는 폐쇄망 금융
> 시스템에서, 요청마다 이 teacher 를 돌리는 건 불가능하다. 증류는
> teacher 의 *암묵적 지식* — "dark knowledge" — 을 태스크별 **LightGBM**
> student 로 전이해, **8GB** 에서 약 **10배 빠른** 추론을 달성하면서도
> 성능 손실을 **3%p 이내** 로 유지한다.

## 왜 증류인가 — 두 문제, 하나의 수

"깊은 모델을 서빙한다"는 말 속에는 별개의 두 문제가 숨어 있고, 증류는
둘을 한 번에 푼다.

첫째는 *서빙 비용과 복잡도* 다. Teacher 는 GPU, PyTorch + CUDA 런타임,
무거운 이미지를 요구한다. 참조서는 직접 서빙을 배제하는 세 가지 제약을
든다 — GPU 메모리(20GB 라 GPU 1장당 모델 1개), 추론 지연(~50ms/배치 vs
10ms SLA), 배포 무게(PyTorch/cuDF 의존성으로 8GB+ 이미지). LightGBM 은
~200MB 이미지로 CPU 에서 돌고, 같은 일을 약 1/10 시간에 끝낸다.

둘째는 *아키텍처 자유* 다. 증류는 가중치를 복사하지 않는다 — *행동* 을
복사한다. Student 는 완전히 다른 모델 종류일 수 있다. 이 프로젝트가 깊은
신경망(PLE)에서 그래디언트 부스팅 트리(LightGBM)로 건너갈 수 있는 이유가
바로 이것이다 — student 는 teacher 의 내부가 아니라 *출력 분포* 만
재현하면 된다. 양자화와 가지치기는 같은 모델 계열 안에서 줄이지만,
증류는 계열을 넘는다.

| 제약 | Teacher (PLE-adaTT) | Student (LightGBM) |
| --- | --- | --- |
| 파라미터 / 크기 | ~50M, 깊은 멀티태스크 | 수백 그루 트리, 태스크별 |
| 메모리 | 20GB VRAM | 8GB, CPU 전용 |
| 지연 (1,024 배치) | ~50ms | 약 10배 빠름 |
| 배포 이미지 | PyTorch + CUDA, 8GB+ | LightGBM, ~200MB |
| 학습 데이터 | 피처 + 라벨 | 피처 + hard label + **soft label** |

> **역사적 배경.** 아이디어는 이름보다 앞선다. *Bucilua, Caruana &
> Niculescu-Mizil (KDD 2006)* 은 큰 앙상블의 예측으로 작은 신경망 하나를
> 학습시켜 압축할 수 있음을 보였다 — "model compression". *Hinton,
> Vinyals & Dean (2015)* 이 이를 **knowledge distillation** 으로
> 재정식화하며, 표준 도구로 만든 두 아이디어를 더했다 — soft target 이
> 담는 정보량을 조절하는 *temperature* 노브, 그리고 teacher 의 softmax
> 에 숨은 구조를 가리키는 *dark knowledge* 라는 이름. 아래의 모든 것은
> 트리 student 에 특화된 그 2015 프레임워크다.

## Dark Knowledge — Soft Label 이 담고 Hard Label 이 버리는 것

Hard label 은 채점표다 — 클래스 3, 끝. Teacher 의 softmax 출력은
*세계관* 이다. 12-클래스 헤드가

$$ p_{teacher} = [0.72,\ 0.14,\ 0.08,\ 0.03,\ 0.01,\ \dots] $$

를 출력할 때, 이건 단지 "클래스 0" 이라 말하는 게 아니다. 클래스 1 은
그럴듯한 근접 오답, 2 는 더 약한 오답, 5 이상은 사실상 불가능이라 말하고
있다. 어떤 오답이 거의-정답인가 하는 그 *클래스 간 상대 구조* 가
Hinton 이 명명한 **dark knowledge** 다 — hard label 에서는 보이지
않지만(어둠 속) soft label 에는 실재하는, 학습 가능한 정보. $C$-클래스
soft label 은 $(C-1)$ 차원의 관계 신호를 담지만, hard label 은 인덱스
하나를 담을 뿐이다.

Binary 태스크에서도 효과는 살아남는다. Teacher 가 "클릭 확률 0.8" 이라
말하는 것은 "클릭함" 과 다르다. 0.8 은 student 에게 *확률적* 진술 —
"거의 확실히 클릭하지만 20% 불확실성이 있다" — 을 건네고, 그 분포를
학습하는 것은 내장된 label smoothing 처럼 작동해 student 의 일반화를
끌어올린다. Student 를 정답만이 아니라 soft label 로 학습시키는 이유가
바로 이것이다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <rect x="30" y="70" width="120" height="90" rx="8" fill="#0d948815" stroke="#0d9488" stroke-width="1.2"/>
  <text x="90" y="105" text-anchor="middle" font-size="13" font-weight="700" fill="#0d9488">PLE Teacher</text>
  <text x="90" y="124" text-anchor="middle" font-size="9" fill="#64748b">깊은 모델 · 20GB</text>
  <text x="90" y="138" text-anchor="middle" font-size="9" fill="#64748b">15 task head</text>
  <rect x="225" y="55" width="150" height="120" rx="8" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="300" y="48" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">soft label (T = 5.0)</text>
  <g fill="#4f46e5">
    <rect x="240" y="120" width="14" height="40"/><rect x="258" y="135" width="14" height="25"/>
    <rect x="276" y="145" width="14" height="15"/><rect x="294" y="150" width="14" height="10"/>
    <rect x="312" y="152" width="14" height="8"/><rect x="330" y="154" width="14" height="6"/><rect x="348" y="155" width="14" height="5"/>
  </g>
  <text x="300" y="100" text-anchor="middle" font-size="9" fill="#64748b">[0.72, 0.14, 0.08, …]</text>
  <text x="300" y="113" text-anchor="middle" font-size="9" fill="#64748b">"dark knowledge"</text>
  <rect x="450" y="70" width="120" height="90" rx="8" fill="#d9770615" stroke="#d97706" stroke-width="1.2"/>
  <text x="510" y="105" text-anchor="middle" font-size="13" font-weight="700" fill="#d97706">LGBM Student</text>
  <text x="510" y="124" text-anchor="middle" font-size="9" fill="#64748b">트리 · 8GB · CPU</text>
  <text x="510" y="138" text-anchor="middle" font-size="9" fill="#64748b">태스크별</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.6">
    <line x1="150" y1="115" x2="223" y2="115"/><polygon points="223,115 213,110 213,120"/>
    <line x1="375" y1="115" x2="448" y2="115"/><polygon points="448,115 438,110 438,120"/>
  </g>
  <text x="187" y="105" text-anchor="middle" font-size="9" fill="#94a3b8">추론</text>
  <text x="412" y="105" text-anchor="middle" font-size="9" fill="#94a3b8">모방</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">Teacher 를 한 번 돌려 soft label 을 만들고, student 는 그 분포를 모방하며 hard label 이 담지 못하는 dark knowledge 를 물려받는다.</figcaption>
</figure>

## Temperature — Softmax 를 풀어 지식을 보이게 하기

Teacher 의 softmax 가 이미 뾰족하면 — 예: $[0.95, 0.04, 0.01]$ — soft
label 은 hard 와 거의 다를 바 없고 dark knowledge 는 숨은 채 남는다.
**Temperature** $T$ 가 이를 해결한다 — student 가 보기 전에 분포를
평활화한다.

$$ p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$

$T = 1$ 이면 보통의 softmax 다. $T$ 가 커지면 분포가 평평해지고 — 엔트로피가
오르고 — 근접 오답 클래스의 *작은* 확률들이 쓸 만한 학습 신호로 자라난다.
프로젝트는 soft label 생성 시 $T = 5.0$ 을 쓴다. 이름은 우연이 아니다 —
이 수식은 통계역학의 볼츠만 분포에서 $z_i = -E_i$ 로 둔 것이며, 높은
온도는 확률을 상태들에 퍼뜨리고 $T \to 0^+$ 이면 모든 확률이 가장 낮은
에너지(= 가장 높은 로짓) 하나로 모인다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="240" fill="#f8fafc" rx="8"/>
  <text x="150" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">T = 1 (뾰족)</text>
  <line x1="50" y1="190" x2="270" y2="190" stroke="#64748b" stroke-width="1"/>
  <g fill="#e11d48">
    <rect x="60" y="60" width="30" height="130"/><rect x="100" y="170" width="30" height="20"/>
    <rect x="140" y="178" width="30" height="12"/><rect x="180" y="182" width="30" height="8"/><rect x="220" y="184" width="30" height="6"/>
  </g>
  <text x="150" y="212" text-anchor="middle" font-size="9" fill="#94a3b8">한 클래스가 지배 → hard label 에 가까움</text>
  <line x1="290" y1="45" x2="290" y2="200" stroke="#e2e8f0" stroke-width="1"/>
  <text x="420" y="30" text-anchor="middle" font-size="12" font-weight="700" fill="#1e3a5f">T = 5 (평활)</text>
  <line x1="310" y1="190" x2="530" y2="190" stroke="#64748b" stroke-width="1"/>
  <g fill="#0d9488">
    <rect x="320" y="100" width="30" height="90"/><rect x="360" y="130" width="30" height="60"/>
    <rect x="400" y="148" width="30" height="42"/><rect x="440" y="160" width="30" height="30"/><rect x="480" y="168" width="30" height="22"/>
  </g>
  <text x="420" y="212" text-anchor="middle" font-size="9" fill="#94a3b8">근접 오답이 보임 → dark knowledge</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">온도를 올리면 softmax 가 평평해진다. 최상위가 아닌 클래스들의 상대 높이 — dark knowledge — 가 반올림 오차가 아니라 학습 가능한 신호가 된다.</figcaption>
</figure>

## 증류 손실 — 두 타깃, 하나의 $T^2$ 보정

Student 는 두 가지를 동시에 겨냥해 학습한다 — *사실*(hard label)과
*teacher 의 견해*(soft label). 프로젝트의 손실은 그 가중 합이다.

$$ \mathcal{L}_{distill} = \alpha\,\mathcal{L}_{hard} + (1 - \alpha)\,T^2\,\mathcal{L}_{soft} $$

Binary 태스크에서는 이렇게 구체화된다.

$$ \mathcal{L}_{binary} = \alpha\,\mathrm{BCE}(\hat{y}, y) + (1-\alpha)\,T^2\,\mathrm{KL}\big(p_t^{T}\,\|\,p_s^{T}\big) $$

Multiclass 는 같은 형태로, hard 측은 cross-entropy, soft 측은
$\mathrm{KL}\big(\mathrm{softmax}(z_t/T)\,\|\,\mathrm{softmax}(z_s/T)\big)$ 이다. Soft 항은 *forward*
방향의 **KL divergence** — $\mathrm{KL}(\text{teacher}\,\|\,\text{student})$ — 로,
teacher 가 확률을 둔 모든 곳에 student 도 질량을 두도록 강제한다(teacher
의 중요 영역을 빠짐없이 커버). 반대 방향인 reverse KL 은 mode-seeking
이라 일부 모드를 무시할 수 있어 증류에 부적합하다.

$\alpha$ 가 배합을 정한다. 기본값은 **$\alpha = 0.3$** — "30% 는 사실에
근거, 70% 는 전문가에 의존" — 이며 `DISTILLATION_ALPHA` 로 오버라이드
가능하다(레포는 대체 기본값 0.4 도 함께 둔다). 경험칙 — teacher 가 좋을수록
$\alpha$ 를 낮춘다. 신뢰할 만한 teacher 의 분포가 원본 라벨보다 값지기
때문이다.

> **수식 직관.** soft 항의 $T^2$ 는 왜인가? 로짓을 $T$ 로 풀면 soft
> 손실의 gradient 도 함께 줄어든다 — 각 softmax 미분이 $1/T$ 인자를
> 받고, KL 항의 gradient 는 $1/T^2$ 로 스케일된다. 보정 없이 $T$ 를
> 올리면 soft 항의 영향력이 조용히 줄어들어 $\alpha$ 가 의미한 바를 잃게
> 된다. $T^2$ 를 곱하면(예: $T=5 \Rightarrow T^2=25$) gradient 가 hard
> 손실과 같은 스케일로 복원되어, $\alpha$ 가 hard/soft 비율을 정직하게
> 제어한다. 이것이 Hinton 이 유도한 $1/(N T^2)$ 인자 그대로다.
> **Regression 은 예외다** — 확률 분포가 아니라 연속 값을 예측하므로
> temperature 평활화가 없고 $T^2$ 도 *없다*. 손실은 단지
> $\alpha\,\mathrm{MSE}(\hat{y}_s, y) + (1-\alpha)\,\mathrm{MSE}(\hat{y}_s, \hat{y}_t)$ 다.

깔끔한 부분 — LightGBM 은 이를 직접 주입하게 해준다. Custom
objective(`fobj`)가 결합 손실의 행별 gradient 와 hessian —
$\nabla = \alpha\,\nabla_{hard} + (1-\alpha)\,T^2\,\nabla_{soft}$ —
을 LightGBM 에 건네므로, 트리 앙상블이 stock objective 가 아니라 *증류*
손실을 최소화하도록 성장한다.

## Teacher → Soft Label → Student 파이프라인

전체는 `distillation_entrypoint.py` 가 **10개 DAG 스테이지** 로
오케스트레이션하는 오프라인 Airflow 배치다. 척추는 이렇다.

1. **detect-mode** — teacher 가 바뀌었는가? `full_distillation`(전체
   재생성) vs `weekly_retrain`(캐시된 soft label + 피처 선택 재사용,
   student 만 재학습).
2. **load-teacher** — MLflow 에서 PLE teacher 로드
   (`models:/ple_cluster_adatt/Production`).
3. **generate-soft-labels** — 15개 활성 태스크 전체에 대해 teacher 를
   **T = 5.0** 으로 돌려 `SoftLabelGenerator` 가 soft target 을 Parquet
   으로 기록.
4. **select-features** — Integrated Gradients 기반 **200D** 선택, 필수
   보존 리스트(persistence entropy, MPC, income elasticity, Sharpe,
   volatility, …)로 도메인 핵심 피처를 살림.
5–6. **mark-timestamp / load-cached-labels** — smart-mode 분기를 위한
   기록 관리.
7. **train-students** — 태스크별 LightGBM 1개를
   `피처 + hard label + soft label` 로, $T^2$-스케일 custom objective
   로 학습.
8. **validate** — fidelity 게이트(다음 절).
9–10. **log-mlflow / package** — student 등록 및 서빙용 패키징.

Teacher 는 스테이지 2–3 에서 소비되고 다시는 등장하지 않는다. 스테이지
7 부터 시스템은 student 만 안다.

## Fidelity 게이트 — Student 가 실제로 Teacher 를 재현하는가?

작은 모델이 teacher 에서 조용히 벗어나면 쓸모가 없다. 어떤 student 든
배포 전에 *fidelity* 를 검사한다. 참조서는 5-criteria
`DistillationValidator` 를 라이브러리 명세로 정의한다.

| 기준 | 임계값 | 메트릭 | 의미 |
| --- | --- | --- | --- |
| 1 | AUC gap $\le 0.03$ | Teacher−Student AUC | 정확도 보존 |
| 2 | Spearman $\rho \ge 0.95$ | 순위 상관 | **순위** 보존 |
| 3 | ECE gap $\le 0.02$ | 보정 | 확률 *크기* 보존 |
| 4 | 전부 통과 | 세그먼트 일관성 | 세그먼트별 blind spot 없음 |
| 5 | speed ratio $\le 0.1$ | student/teacher 지연 | 10배 가속이 실재함 |

추천기에서 가장 중요한 건 기준 2 다 — 절대 확률값은 상관없고, teacher
가 고객 A 를 B 위에 두면 student 도 그래야 한다는 것만 본다. 서빙이 순위
기반 추천이라, AUC 가 멀쩡해 보여도 순위가 뒤집히면 품질이 무너지기
때문이다. 다섯 중 하나라도 실패하면 그 태스크의 증류는 FAIL 이다.

출처에서 가져온 정직한 단서 하나 — 이 5-criteria 클래스는 *명세* 다.
**LIVE** Stage 8 경로는 student 예측을 teacher soft label 과 직접
비교하는 더 가벼운 *strict validation* 을 돌린다 — binary: 상관
$\ge 0.01$ 및 MAE $\le 0.50$; multiclass: argmax 일치율 $\ge 0.08$;
regression: 상관 $\ge 0.01$. 풍부한 validator 는 라이브러리에 존재하나
현재 Stage 8 임계 경로에는 올라가 있지 않다.

## Student 가 사는 곳 — 3-Layer Fallback

증류는 student 를 만들어내지만, 배치 시스템은 그걸 맹목적으로 믿지
않는다. 배치 시작 시점에 프로젝트는 **3-layer FallbackRouter** 를 돌려,
태스크별로 *어떤* 모델의 예측이 권위 있는지 정한다 — 그리고 위의 fidelity
게이트가 바로 Layer 1 을 통과시키는 관문이다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="300" fill="#f8fafc" rx="8"/>
  <rect x="180" y="20" width="200" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="280" y="38" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">teacher 품질 OK</text>
  <text x="280" y="52" text-anchor="middle" font-size="10" fill="#64748b">AND student fidelity 통과?</text>
  <rect x="380" y="90" width="160" height="44" rx="6" fill="#0d948818" stroke="#0d9488" stroke-width="1.2"/>
  <text x="460" y="110" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">Layer 1</text>
  <text x="460" y="125" text-anchor="middle" font-size="9" fill="#64748b">distilled LGBM</text>
  <rect x="180" y="100" width="160" height="40" rx="6" fill="#f1f5f9" stroke="#1e3a5f" stroke-width="1"/>
  <text x="260" y="118" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a5f">fidelity 통과,</text>
  <text x="260" y="132" text-anchor="middle" font-size="10" fill="#64748b">teacher 미달?</text>
  <rect x="380" y="160" width="160" height="44" rx="6" fill="#4f46e518" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="460" y="180" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Layer 2</text>
  <text x="460" y="195" text-anchor="middle" font-size="9" fill="#64748b">direct LGBM</text>
  <rect x="180" y="230" width="160" height="44" rx="6" fill="#d9770618" stroke="#d97706" stroke-width="1.2"/>
  <text x="260" y="250" text-anchor="middle" font-size="11" font-weight="700" fill="#d97706">Layer 3</text>
  <text x="260" y="265" text-anchor="middle" font-size="9" fill="#64748b">rule / template fallback</text>
  <g fill="#cbd5e1" stroke="#94a3b8" stroke-width="1.3">
    <line x1="380" y1="50" x2="378" y2="105"/><polygon points="378,112 373,102 383,102"/>
    <line x1="280" y1="60" x2="260" y2="98"/><polygon points="260,98 257,87 267,90"/>
    <line x1="340" y1="120" x2="378" y2="170"/><polygon points="378,178 369,170 377,164"/>
    <line x1="220" y1="140" x2="240" y2="228"/><polygon points="241,228 232,224 240,218"/>
  </g>
  <text x="400" y="78" font-size="9" fill="#0d9488" font-weight="700">예</text>
  <text x="230" y="82" font-size="9" fill="#64748b">아니오</text>
  <text x="300" y="158" font-size="9" fill="#4f46e5" font-weight="700">예</text>
  <text x="200" y="190" font-size="9" fill="#d97706" font-weight="700">아니오 / 누락</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">FallbackRouter 는 배치 시작 시 각 태스크를 layer 로 확정한다. Layer 1 은 teacher 품질과 student fidelity 둘 다 요구하고, Layer 2 는 증류하지 않은 트리, Layer 3 은 아무것도 자격이 안 될 때의 rule/template baseline.</figcaption>
</figure>

- **Layer 1 — distilled LGBM.** Teacher 가 품질 게이트를 통과 *했고*
  student 가 fidelity 를 통과했다. 압축 모델을 신뢰한다 — 전체 파이프라인이
  지향하는 happy path 다.
- **Layer 2 — direct LGBM.** Teacher 가 품질 기준 미달(dark knowledge 를
  물려받을 가치가 없음)이지만, 라벨로 직접 학습한 트리는 여전히 통과한다.
  Soft-label 전이 없이 그 트리를 서빙한다.
- **Layer 3 — rule / template fallback.** 어떤 모델도 자격이 안 되거나
  태스크 예측이 통째로 누락됐다. `RuleBasedRecommender` 가 휴리스틱
  baseline 을 만들어 모든 태스크가 항상 답을 갖게 한다.

이 태스크별 사다리 바깥에 고객 단위 override 도 있다 — 특정 고객에게
causal guardrail 이 걸리면 그 고객의 *모든* 태스크가 Layer 3 으로
강제된다. 모델의 추론이 분포 밖으로 보일 때의 의도적으로 보수적인
선택이다. 15개 태스크별 student 가 Layer-1 함대이고, router 는 태스크마다
그 각각이 실제로 답할 자격이 있는지 정하는 장치다.

## 여기서 멈추는 이유

운영적 곤경에서 출발했다 — 폐쇄망에서 요청마다 서빙할 수 없는 20GB,
~50ms teacher. 그리고 증류가 그것을 녹이는 걸 봤다 — teacher 는 *한 번*
돌아 soft label 을 내뱉고, 그 dark knowledge 가 $T^2$-스케일 hard+soft
손실을 통해 태스크별 LightGBM student 로 부어지며, fidelity 게이트가 작은
모델이 여전히 큰 모델처럼 순위 매기고 보정하는지 확인하고, 3-layer
fallback 이 태스크마다 어떤 모델이 권위 있는지 정한다. Teacher 는
student 를 가르치고 떠났다.

남은 것은 이 모든 것을 값지게 만드는 부분 — *서빙* 이다. Student 의
학습과 등록이 끝나면, 폐쇄망 시스템은 요청 시점에 **어떤 모델도 돌리지
않는다**. 배치가 모든 고객 × 태스크를 미리 스코어링해 결과를, 서빙
계층이 그저 *조회* 하는 저장소에 쓴다. 그 사전 스코어링이 어떻게
동작하는가 — DuckDB-over-Parquet 저장소, multiclass shape 어댑터, 경로에
놓인 consent 게이트 — 가 다음 편 **DISTILL-2: 추론과 스코어링** 의
주제다.
