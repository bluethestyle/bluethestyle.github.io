---
title: "[Study Thread] REASON-1 — 지어내지 않고 '왜'를 말하기: 금융 추천을 위한 계층형 사유 생성"
date: 2026-06-08 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, reason-generation, llm, grounding, hallucination, nlg]
lang: ko
excerpt: "모델 점수를 모든 추천 건에 대해 사람이 읽을 수 있는 사유로 바꾸되, 규제 금융 상품에서 지어낸 사실은 단 하나도 허용하지 않는다는 단 하나의 규칙 아래에서. L1 템플릿 → L2a LLM 리라이트(→ L2b 검증) 계층 설계, LLM을 실제 근거에 묶는 3중 grounding, 그리고 silent 환각 위험을 차단하는 verdict pass/fail 게이트(json_object 구조화 출력)."
series: study-thread
part: 25
alt_lang: /2026/06/08/reason-generation-en/
next_title: "REASON-2 — 피처 역매핑으로 하는 Grounding: 734D 벡터에서 사람이 신뢰하는 문장으로"
next_desc: "Grounding 자체의 기계 장치: Integrated Gradients가 Top 피처를 어떻게 고르고, 정규화된 734D 피처 인덱스가 어떻게 사람이 읽을 수 있는 범위와 카테고리 레이블로 역매핑되며, 그 역매핑 텍스트가 어떻게 모든 하위 LLM 호출의 검증 기준이 되는 '근거(ground truth)'가 되는가."
next_status: draft
---

*"Study Thread" 시리즈의 추천 사유 생성 서브스레드 1편. 이번 편부터
영문/국문 병렬로, 본 프로젝트가 스스로를 설명하는 방식을 정리한다 —
추천 점수가 어떻게 고객과 상담 직원이 읽을 수 있는 문장이 되는가. 출처는
온프렘 프로젝트 `기술참조서/추천사유생성_기술_참조서` 이고, 전체 PDF 는
서브스레드 마지막 편에 첨부한다. PLE, adaTT, TDA 서브스레드가 모델이
무엇을 읽고 태스크들이 어떻게 공유하는가를 다뤘다면, 이번 서브스레드는
점수가 계산된 다음의 질문을 던진다 — 어떻게* 왜 *를 말하는가. 지어낸
사실 하나가 오타가 아니라 규제 위반 사고가 되는 금융 상품에서.*

> **모든 것을 규정하는 제약.** 디테일 하나를 환각하는 범용 챗봇은
> 성가신 정도다. 존재하지도 않는 상품을 두고 "수익률 7% 가 보장됩니다"
> 라고 고객에게 말하는 금융 추천 엔진은, 금융소비자보호법 위반이다.
> 추천 사유 생성 파이프라인은 바로 그 공포에서 거꾸로 설계됐다 — "어떻게
> 문장을 매끄럽게 만들까" 가 아니라 "어떻게 지어내거나 규정을 위반한
> 문장이 배포되는 것을 *불가능* 하게 만들까". 아래의 모든 설계 선택 —
> 템플릿 바닥, 리라이트 천장, grounding, verdict 게이트 — 은 이 단 하나의
> 질문에 대한 답이다.

## 애초에 왜 사유를 생성하는가

추천 모델이 아무리 완벽해도, *왜* 를 말하지 못하면 실무에서 무너진다.
금융 환경에서 "왜" 는 장식이 아니라 세 방향으로 동시에 하중을 진다.

- **규제.** AI 기본법 제31조·제34조는 고위험 AI 시스템이 의사결정 근거를
  설명할 것을 요구하고, 금융소비자보호법 제19조는 적합성 원칙에 따른
  설명 의무를 규정한다. 모든 추천에 사유가 필요하고, 모든 사유에 AI 생성
  고지 문구가 자동 부착돼야 한다.
- **신뢰.** "AI 분석 기반 추천" 이라는 단문은 아무것도 못 얻는다. 고객의
  실제 거래 패턴, 상담 이력, 생애 주기에 기반한 사유는 수용률을 유의미하게
  높인다 — 참조서는 설명 동반 추천의 20~40% 수용률 상승을 인용한다.
- **감사.** 금융감독원이 개별 건을 검사할 때, 시스템은 *언제, 어떤 데이터를
  근거로, 어떤 모델이, 어떤 사유를 생성했는가* 를 추천 건별로 소급
  재구성할 수 있어야 한다.

요구사항은 직설적이다 — **고객 1,200만 전량** 에 사유를, 하나하나
방어 가능하게, 단 하나도 지어내지 않고.

## 계층형 설계 — 그리고 왜 계층인가

단순한 두 선택지는 둘 다 실패한다. 모든 사유를 LLM 으로 생성하면 GPU
비용이 참조서 추산 약 1,000 GPU-h — 계층 설계의 ~162 GPU-h 대비 — 에
이르고, 1,200만 건 전체에 환각 위험을 떠안는다.
템플릿만 쓰면 규제는 충족하지만 30개로 찍어낸 기계적 문장이라 복잡한
고객을 담지 못한다. 프로젝트는 이 이분법을 거부하고 둘을 **계층화** 한다.

<figure style="margin:24px auto;max-width:640px;">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="640" height="300" fill="#f8fafc" rx="8"/>
  <rect x="24" y="24" width="120" height="48" rx="6" fill="#1e3a5f15" stroke="#1e3a5f" stroke-width="1"/>
  <text x="84" y="44" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">모델 점수</text>
  <text x="84" y="60" text-anchor="middle" font-size="9" fill="#64748b">PLE-adaTT</text>
  <rect x="24" y="84" width="120" height="48" rx="6" fill="#1e3a5f15" stroke="#1e3a5f" stroke-width="1"/>
  <text x="84" y="104" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a5f">IG Top-5</text>
  <text x="84" y="120" text-anchor="middle" font-size="9" fill="#64748b">피처 기여도</text>
  <rect x="196" y="48" width="130" height="60" rx="6" fill="#0d948815" stroke="#0d9488" stroke-width="1.4"/>
  <text x="261" y="72" text-anchor="middle" font-size="12" font-weight="700" fill="#0d9488">L1 Template</text>
  <text x="261" y="90" text-anchor="middle" font-size="9" fill="#64748b">1,200만 전량 · LLM 0회</text>
  <text x="261" y="102" text-anchor="middle" font-size="9" fill="#64748b">결정론적 바닥</text>
  <rect x="196" y="130" width="130" height="40" rx="6" fill="#f1f5f9" stroke="#64748b" stroke-width="1"/>
  <text x="261" y="148" text-anchor="middle" font-size="10" font-weight="700" fill="#64748b">Richness 분류</text>
  <text x="261" y="162" text-anchor="middle" font-size="9" fill="#94a3b8">rich / moderate / sparse</text>
  <rect x="380" y="40" width="140" height="60" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.4"/>
  <text x="450" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="#4f46e5">L2a LLM Rewrite</text>
  <text x="450" y="79" text-anchor="middle" font-size="9" fill="#64748b">rich+moderate (~500K/주)</text>
  <text x="450" y="91" text-anchor="middle" font-size="9" fill="#64748b">품질 천장</text>
  <rect x="380" y="118" width="140" height="56" rx="6" fill="#d9770615" stroke="#d97706" stroke-width="1.4"/>
  <text x="450" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#d97706">L2b Validation</text>
  <text x="450" y="157" text-anchor="middle" font-size="9" fill="#64748b">샘플링 · 3축 사후</text>
  <rect x="380" y="200" width="140" height="52" rx="6" fill="#e11d4815" stroke="#e11d48" stroke-width="1.2"/>
  <text x="450" y="222" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">Audit Archive</text>
  <text x="450" y="238" text-anchor="middle" font-size="9" fill="#64748b">DuckDB + Parquet</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="144" y1="48" x2="194" y2="70"/><polygon points="194,70 185,67 187,76"/>
    <line x1="144" y1="108" x2="194" y2="86"/><polygon points="194,86 187,80 185,89"/>
    <line x1="261" y1="108" x2="261" y2="128"/><polygon points="261,128 257,120 265,120"/>
    <line x1="326" y1="142" x2="378" y2="70"/><polygon points="378,70 369,71 374,79"/>
    <line x1="326" y1="150" x2="378" y2="145"/><polygon points="378,145 370,141 369,150"/>
    <line x1="450" y1="100" x2="450" y2="116"/><polygon points="450,116 446,108 454,108"/>
    <line x1="450" y1="174" x2="450" y2="198"/><polygon points="450,198 446,190 454,190"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">2-Layer 전량 생성 아키텍처. L1 은 1,200만 전량 아래 깔리는 결정론적 바닥, L2a 는 컨텍스트가 풍부한 고객을 위한 LLM 품질 천장, L2b 는 둘을 샘플링해 사후 품질을 본다. 모든 건이 감사 아카이브로 들어간다.</figcaption>
</figure>

이 분업이 곧 설계의 핵심이다.

| Layer | 대상 | 방식 | LLM 호출 | 비용 |
| --- | --- | --- | --- | --- |
| **L1** | 1,200만 전량 | 템플릿 (6 카테고리 × 5 변형 = 30개, 해시 선택) | 0회 | ~20분, CPU 전용 |
| **L2a** | rich + moderate (~500K/주) | LLM 리라이트 (Ollama dual-route + 3-Layer Safety Gate) | 1회 | ~1.0초/건 |
| **L2b** | 샘플링 (~67K) | 품질 검증 (사실성·관련성·자연스러움) | 1회 | 사후 |

L1 은 **바닥** 이다 — Integrated Gradients Top 피처를 역매핑해 *모든*
고객에게 사유를 GPU 비용 0 으로 생성한다(`customer_id` 해시로 30개 변형
중 하나를 결정론적으로 선택 — 동일 고객, 동일 문구, 항상 재현 가능).
이 바닥만으로도 금소법 제19조의 동등 설명 의무를 충족한다.

L2a 는 **천장** 이다 — 컨텍스트가 구체적으로 말할 만큼 풍부한 고객에 대해,
LLM 이 기계적 템플릿 초안을 자연스러운 문장으로 리라이트한다. 그 지배
원칙은 참조서에 직설적으로 적혀 있다 — *실패 시 L1 원본 유지*. 어떤
게이트라도 실패한 리라이트는 사유를 비우지 않고 템플릿으로 폴백한다. 빈
사유는 **절대** 배포되지 않는다.

L2b 는 **모니터** 다 — L1-only 출력과 L2a 리라이트 둘 다에 대한 슬림한
샘플링 사후 품질 검사다. 배치를 막지 않고, 지켜본다.

> **역사적 배경.** 이 모든 것 아래에 깔린 "형태 vs 의미" 의 우려는
> Bender & Koller 의 ACL 2020 논문 *"Climbing towards NLU: On Meaning,
> Form, and Understanding in the Age of Data"* 다. 주장은 이렇다 —
> *형태(form)*, 즉 기호의 공기(共起)만으로 학습한 언어 모델은 텍스트
> 바깥 세계와의 연결 없이는 *의미(meaning)* 를 획득할 수 없다. 그들의
> Octopus 사고 실험이 이를 극화한다 — 기호만 봤을 뿐 지시체(referent)는
> 본 적 없는 행위자는 유창한 대화를 흉내 낼 수 있지만, 어느 것 하나
> 무엇을 가리키는지는 알지 못한다. "확률적 앵무새(stochastic parrot)" —
> 이 표현 자체는 Bender 등이 2021년 별도 논문에서 만든 것이지만 문제의식은
> 같다 — 는 사실에 묶이지 않은 그럴듯한 형태를 만든다.
> 아래의 grounding 설계 전체가 그에
> 대한 공학적 답이다 — LLM 이 형태만으로 말하게 두지 않고, 모든 문장이
> 명시적으로 외부에서 공급된 근거 위에 서도록 강제한다.

## Grounding — LLM 을 근거에 묶기

L2a 의 LLM 이 위험이라면, grounding 은 목줄이다. *Grounding* 이란 모델의
출력을 언어 사전(prior)이 말하고 싶어 하는 것이 아니라 외부의 ground
truth 위에 서게 만드는 것이다. 본 시스템에서 "ground" 는 세 의미를 동시에
가지며, 모델이 한 단어를 쓰기 전에 모두 프롬프트에 주입된다.

1. **피처 grounding.** Integrated Gradients 의 Top-5 피처 기여도 — 모델
   점수의 *실제* 근거 — 를 프롬프트에 명시해, 사유가 추측이 아니라 모델이
   왜 그렇게 판단했는지에 묶이게 한다.
2. **고객 grounding.** 세그먼트, 거래 패턴, 상담 이력 등 실제 고객 데이터를
   주입해, 모델이 기댈 구체적 사실을 줌으로써 환각을 억제한다.
3. **규정 grounding.** 시스템 프롬프트에 금소법 위반 패턴을 열거하고
   Rule-based 검증으로 강제해, 규정 준수를 희망이 아니라 강제 제약으로
   만든다.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="250" fill="#f8fafc" rx="8"/>
  <text x="310" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">근거에 묶인 리라이트 vs 근거 없는 앵무새</text>
  <rect x="24" y="56" width="150" height="150" rx="6" fill="#0d948810" stroke="#0d9488" stroke-width="1.2"/>
  <text x="99" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#0d9488">근거에 묶임</text>
  <text x="40" y="98" font-size="9" fill="#64748b">프롬프트 속 근거:</text>
  <text x="40" y="114" font-size="9" fill="#1e3a5f">• IG: spend_food ↑ (0.31)</text>
  <text x="40" y="128" font-size="9" fill="#1e3a5f">• 세그먼트: WARMSTART</text>
  <text x="40" y="142" font-size="9" fill="#1e3a5f">• 상담 8회 / 1년</text>
  <line x1="40" y1="152" x2="158" y2="152" stroke="#cbd5e1" stroke-width="0.8"/>
  <text x="40" y="170" font-size="9" fill="#0d9488" font-weight="700">→ "식비 지출이 가장</text>
  <text x="40" y="183" font-size="9" fill="#0d9488" font-weight="700">  많으셔서 이 카드가</text>
  <text x="40" y="196" font-size="9" fill="#0d9488" font-weight="700">  패턴에 맞습니다."</text>
  <circle cx="186" cy="130" r="14" fill="#0d9488"/>
  <path d="M 180 130 l 4 5 l 9 -11" stroke="#fff" stroke-width="2.4" fill="none"/>
  <rect x="262" y="56" width="150" height="150" rx="6" fill="#e11d4810" stroke="#e11d48" stroke-width="1.2"/>
  <text x="337" y="76" text-anchor="middle" font-size="11" font-weight="700" fill="#e11d48">근거 없음</text>
  <text x="278" y="98" font-size="9" fill="#64748b">근거 없이 형태만:</text>
  <text x="278" y="118" font-size="9" fill="#94a3b8">(언어 사전이 빈틈을</text>
  <text x="278" y="131" font-size="9" fill="#94a3b8"> 그럴듯하지만 지어낸</text>
  <text x="278" y="144" font-size="9" fill="#94a3b8"> 디테일로 채운다)</text>
  <line x1="278" y1="152" x2="396" y2="152" stroke="#fecaca" stroke-width="0.8"/>
  <text x="278" y="170" font-size="9" fill="#e11d48" font-weight="700">→ "수익률 7% 보장,</text>
  <text x="278" y="183" font-size="9" fill="#e11d48" font-weight="700">  손실 없음, 반드시</text>
  <text x="278" y="196" font-size="9" fill="#e11d48" font-weight="700">  가입하셔야 합니다."</text>
  <circle cx="424" cy="130" r="14" fill="#e11d48"/>
  <line x1="418" y1="124" x2="430" y2="136" stroke="#fff" stroke-width="2.4"/>
  <line x1="430" y1="124" x2="418" y2="136" stroke="#fff" stroke-width="2.4"/>
  <rect x="470" y="96" width="130" height="68" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="535" y="120" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Verdict 게이트</text>
  <text x="535" y="138" text-anchor="middle" font-size="9" fill="#64748b">사실성 + 컴플라이언스</text>
  <text x="535" y="151" text-anchor="middle" font-size="9" fill="#64748b">→ pass / revise / reject</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="200" y1="130" x2="468" y2="120"/><polygon points="468,120 459,120 462,128"/>
    <line x1="438" y1="130" x2="468" y2="135"/><polygon points="468,135 460,131 459,140"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">같은 모델 점수, 두 프롬프트. 근거를 주입하면 리라이트는 실제 사실을 다시 진술하고, 없으면 언어 사전이 규정 위반을 지어낸다. verdict 게이트가 앵무새를 잡는 마지막 검사다.</figcaption>
</figure>

수학은 한 줄이다. 자기회귀 생성은 다음으로 분해된다.

$$ P(\mathbf{r} \mid \mathbf{p}) = \prod_{j=1}^{m} P\!\left(r_j \mid p_1,\dots,p_n,\; r_1,\dots,r_{j-1}\right) $$

— 각 사유 토큰 $r_j$ 는 프롬프트 전체 $\mathbf{p}$ 와 이미 쓰인 토큰들에
조건부로 생성된다. 함의는 정확히 설계 명제다 — *프롬프트 $\mathbf{p}$ 의
품질이 사유 $\mathbf{r}$ 의 품질을 결정한다.* IG 기여도, 고객 피처, 상담
이력을 $\mathbf{p}$ 에 풍부하게 담을수록, $P(\mathbf{r}\mid\mathbf{p})$ 는
사실에 충실한 사유에 더 많은 확률 질량을 싣고 — 앵무새에게 남는 몫은
줄어든다.

> **설계 직관.** Grounding 은 생성 *이후* 에 적용하는 필터가 아니다 —
> 모델이 샘플링하는 분포 자체를 다시 빚는다. 리라이트는 temperature
> $\tau = 0.3$ 으로 돈다 — 출력이 고확률이면서 근거에 일치하는 토큰 근처에 머물 만큼은
> 낮고(프롬프트 지시는 아예 "금리 같은 숫자를 추가하지 마세요" 라고
> 못박는다), 두 번 돌리면 사실은 그대로 두고 표현만 달라질 만큼은 높다.
> Grounding 이 *무엇을* 말할 수 있는지를 좁히고, temperature 가 *얼마나
> 자유롭게* 말할지를 제어한다. 그래도 빠져나가는 것은 verdict 게이트가
> 잡는다.

이는 정신적으로 *Structured RAG* 다 — 검색 증강 생성(Lewis et al.,
NeurIPS 2020) 패턴이되, 컨텍스트로 "검색" 되는 것이 비정형 문서가 아니라
구조화된 피처 기여도와 고객 프로파일이라는 점만 다르다.

## Verdict 게이트 — Silent 위험 차단

Grounding 은 지어낸 문장의 확률을 낮추지, 없애지는 않는다. 그래서 생성된
사유는 결코 신뢰만으로 배포되지 않는다. 별도의 critique 단계가 채점하고
**verdict** 를 반환한다.

$$ \text{verdict} = \begin{cases} \text{pass} & \text{if } f \ge 0.8 \;\wedge\; c \ge 1.0 \\ \text{revise} & \text{if } f \ge 0.5 \;\wedge\; c \ge 1.0 \\ \text{reject} & \text{otherwise} \end{cases} $$

여기서 $f$ 는 **사실성** 점수(0–1), $c$ 는 **컴플라이언스** 점수(0–1)
다. 구조가 금융권의 우선순위를 직접 인코딩한다.

- **컴플라이언스 $c$ 는 이진 게이트다.** $c = 1.0$ 은 금소법 위반이 전혀
  없음을, $c < 1.0$ 은 *하나라도* 있음을 뜻한다. 위반이 하나라도 있으면
  사실성이 아무리 높아도 즉시 `reject`. 규정이 품질을 앞선다, 예외 없이.
- **사실성 $f$ 는 연속적이다.** $\ge 0.8$ 이면 사유가 원본 데이터와 충분히
  일치해 그대로 배포 가능. 0.5~0.8 이면 부분 환각이 있어 1회 수정으로
  돌려보낸다. 0.5 미만이면 LLM 이 심하게 환각한 것이므로 안전한 템플릿으로
  대체한다.
- **수정은 1회로 제한된다.** `revise` 를 두 번 받은 사유는 `reject` 가
  되어, LLM 호출을 3회(생성 + critique + 재생성·재critique)로 묶고 무한
  루프를 막는다.

핵심 운영 디테일은 이런 verdict 를 LLM 에서 애초에 *어떻게* 읽어내는가다.
L2b 품질 critique 호출(아래에서 만날 사후 모니터)은
`response_format={"type": "json_object"}` 를 전달해 모델이 파싱
가능한 구조화 출력을 반환하게 한다 — Qwen3 의 `<think>...</think>` 블록과
자유 텍스트 혼입이 그러지 않으면 JSON 파싱을 깨뜨리기 때문이다. 그리고
그 폴백에 날카로운 교훈이 박혀 있다 — critique 파싱이 실패하면, 폴백 verdict
가 `'pass'` 에서 **`'fail'`** 로 바뀌었다. 그 논리가 파이프라인 전체의
심장이다 — *읽을 수 없는 verdict 는 안전한 verdict 가 아니다.* 파싱 안 되는
critique 를 "pass" 로 기본 처리하면 환각 사유가 품질 관리를 silent 하게
빠져나간다. "fail" 로 기본 처리하면 실패가 시끄러워진다 — 해당 건은 실패로
채점되고, 더 무거운 critique 모델로 escalation 되며, 품질 리포트에 드러난다.
이것이 **silent 위험** 과 **안전한 실패** 의 차이다.

<figure style="margin:24px auto;max-width:560px;">
<svg viewBox="0 0 560 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="560" height="230" fill="#f8fafc" rx="8"/>
  <text x="280" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">파싱 불가 critique: silent 위험 vs 안전한 실패</text>
  <rect x="200" y="48" width="160" height="44" rx="6" fill="#4f46e515" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="280" y="68" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">L2b critique LLM</text>
  <text x="280" y="83" text-anchor="middle" font-size="9" fill="#64748b">json_object 요청</text>
  <rect x="200" y="108" width="160" height="40" rx="6" fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>
  <text x="280" y="126" text-anchor="middle" font-size="10" font-weight="700" fill="#d97706">JSON 파싱 실패</text>
  <text x="280" y="140" text-anchor="middle" font-size="9" fill="#64748b">(think 블록 / 자유 텍스트)</text>
  <rect x="40" y="170" width="220" height="48" rx="6" fill="#e11d4810" stroke="#e11d48" stroke-width="1.2"/>
  <text x="150" y="190" text-anchor="middle" font-size="10" font-weight="700" fill="#e11d48">폴백 = 'pass'  ✗ (이전)</text>
  <text x="150" y="206" text-anchor="middle" font-size="9" fill="#64748b">환각이 silent 하게 QA 통과</text>
  <rect x="300" y="170" width="220" height="48" rx="6" fill="#0d948810" stroke="#0d9488" stroke-width="1.2"/>
  <text x="410" y="190" text-anchor="middle" font-size="10" font-weight="700" fill="#0d9488">폴백 = 'fail'  ✓ (현재)</text>
  <text x="410" y="206" text-anchor="middle" font-size="9" fill="#64748b">fail 채점 — escalation·리포트</text>
  <g fill="#cbd5e1" stroke="#cbd5e1" stroke-width="1.4">
    <line x1="280" y1="92" x2="280" y2="106"/><polygon points="280,106 276,98 284,98"/>
    <line x1="240" y1="148" x2="150" y2="168"/><polygon points="150,168 159,165 156,173"/>
    <line x1="320" y1="148" x2="410" y2="168"/><polygon points="410,168 401,165 404,173"/>
  </g>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">폴백 verdict 전환. 읽을 수 없는 critique 를 'pass' 로 기본 처리하면 silent 위험, 'fail' 로 처리하면 escalation 되고 품질 리포트에 드러나는 시끄러운 실패가 된다.</figcaption>
</figure>

L2a 는 수용 앞단에 자체 **3-Layer Safety Gate** 를 둔다. 같은 fail-safe
정신이다 — Gate 1 은 빈 문자열과 JSON 잔해(`{` 또는 `[` 로 시작하는
텍스트)를 차단하고, Gate 2 는 금소법 위반 6개 키워드 패턴(예: "확정 수익",
"원금 보장", "n% 수익", "손실 없음", "반드시 가입")을 차단하며, Gate 3 은
길이 30~200자 이탈 또는 한국어 비율 80% 미만을 차단한다. 셋을 모두 통과하면
리라이트가 적용되고 — 아니면 L1 템플릿이 그대로 선다.

## LLM Provider 추상화

폐쇄망 금융 배포는 호스팅 API 를 호출할 수 없고, 단일 모델을 하드와이어해서도
안 된다. 프로젝트는 이 모든 것을 하나의 `LLMProviderFactory` 로 라우팅한다 —
하나의 `generate()` 시그니처 뒤에 6개 교체 가능한 backend 가 있다.

| Backend | 무엇인가 | 비고 |
| --- | --- | --- |
| `ollama` / `qwen` | 로컬 Ollama, OpenAI-compatible | 기본 `qwen3:14b` @ `host.docker.internal:11434/v1` |
| `exaone` | LG AI Research Exaone | vLLM 또는 Ollama self-hosted (`EXAONE_BASE_URL`) |
| `solar` | Upstage Solar | `upstage_api` REST 또는 `local` self-hosted |
| `local` | generic OpenAI-compatible `/v1/chat/completions` | 임의의 사내 서버 |
| `dummy` | test/mock | 단위 테스트용 결정론적 JSON |

운영에서 L2a/L2b 엔진은 **Ollama dual-route** 로 돈다 — 가볍고 빠른 1차
(`exaone3.5:2.4b`)가 대량을 처리하고, 1차가 흔들리면 — L2a 에서는 파싱이나
품질 게이트(Gate 1/3) 실패, L2b 에서는 non-pass 또는 파싱 불가 critique —
`qwen3:14b` 로 **escalation** 한다. `L2aRewriteResult` 레코드가 이를 명시적으로
싣는다 — `primary_model`, `primary_gate`, `escalation_used`,
`escalation_model` — 그래서 모든 건이 어떤 모델이 썼고 escalation 이
발동했는지까지 감사 가능하다. 통합 Factory 는 현재 진단 consensus 경로를
받치고 있고 운영 엔진은 점진적으로 그 위로 이행 중이다. `generate()` 계약이
공유되므로 이행은 기계적이다.

## 평가 — 두 개의 verdict, 두 개의 역할

파이프라인은 서로 다른 두 품질 검사를 돌리며, 그 차이가 시사적이다.

- **Self-Critique (실시간 게이트키퍼).** 2축 — 사실성과 컴플라이언스 — 임계
  $0.8$, $\tau = 0.1$ 로 거의 결정론적으로 판정. 이건 *막는다* — 여기서
  `reject` 면 LLM 사유가 고객에게 도달하지 않는다.
- **L2b 검증 (사후 모니터).** 3축 — 사실성 $f$, 관련성 $r$, 자연스러움 $n$
  — 임계 $0.7$, 샘플 대상. 이건 *지켜본다* — `needs_improvement` 는 배치를
  막지 않고 프롬프트 개선 피드백으로 축적된다.

$$ \text{verdict}_{\text{L2b}} = \begin{cases} \text{pass} & \text{if } f \ge 0.7 \;\wedge\; r \ge 0.7 \;\wedge\; n \ge 0.7 \\ \text{needs\_improvement} & \text{if any score} \in [0.5, 0.7) \\ \text{fail} & \text{if any score} < 0.5 \end{cases} $$

여기서 두 설계 선택이 흘러나온다. L2b 는 게이트키퍼가 빠뜨린 **자연스러움**
축을 더한다 — L2a 의 일 전체가 기계적 템플릿을 자연스러운 문장으로 바꾸는
것이므로, 자연스러움이야말로 모니터링할 가치가 있다. 그리고 L2b 의 기준선이
*더 낮은* 이유(0.7 vs 0.8)는 정확히 그것이 게이트가 아니라 모니터이기
때문이다 — 실시간 게이트키퍼가 사후 감사관보다 엄격해야 한다. L2b 는 두
소스를 샘플링한다 — L1-only 고객의 ~0.4% 층화 슬라이스(~51K, 템플릿 품질
모니터링)와 L2a 리라이트의 5% 감사(~16K) — 그리고 파이프라인이 추적하는 운영
KPI 가 우선순위를 구체화한다 — `l1_coverage_rate = 1.0`,
`l2a_gate_pass_rate ≥ 0.9`, `l2b_factual_score ≥ 0.8`,
`fallback_rate ≤ 0.1`.

## 여기서 멈추는 이유

단 하나의 제약 — 규제 금융 상품에서 지어낸 사실 0 — 에서 출발해, 파이프라인
전체가 거기서 흘러나오는 것을 봤다. L1 은 1,200만 전량을 덮고 설명 의무를
충족하는 결정론적 템플릿 바닥이다. L2a 는 컨텍스트가 풍부한 고객의 품질을
끌어올리는 LLM 천장이되, 항상 템플릿을 안전망으로 남겨둔다. 3중 grounding
이 LLM 이 말할 수 있는 것을 다시 빚고, verdict 게이트가 grounding 이 놓친
것을 막으며, 구조화된 `json_object` 읽기와 `fail` 기본 폴백이 silent 환각
위험을 시끄럽고 안전한 실패로 바꾼다. 6개 교체 가능 backend 가 이 모두를 폐쇄망 안에 두며,
필요할 때만 escalation 하는 dual-route 가 따라붙는다.

미뤄둔 것은 애초에 grounding 을 *작동하게* 만드는 부분이다 — 정규화된 734D
피처 벡터에 대한 Integrated Gradients 의 Top-5 기여도가 어떻게 사람이 읽을
수 있는 범위와 카테고리로 실제 역매핑되는가 — `spend_food ↑ (0.31)` 이
"식비 지출이 가장 많으십니다" 가 되는 과정. 그 역매핑이 모든 verdict 가
검증되는 ground truth 이며, 다음 편 **REASON-2** 의 주제다.
