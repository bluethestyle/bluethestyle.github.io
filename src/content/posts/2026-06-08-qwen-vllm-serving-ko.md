---
title: "[Study Thread] QWEN-1 — 리라이터를 서빙하기: vLLM, Qwen, 그리고 폐쇄망 사유 생성 뒤의 Provider 추상화"
date: 2026-06-08 16:00:00 +0900
categories: [Study Thread]
tags: [study-thread, vllm, qwen, llm-serving, pagedattention, on-prem]
lang: ko
excerpt: "기술참조서 아크의 마지막 편 — 폐쇄 금융망 밖으로 단 한 토큰도 나가지 않으면서 추천 사유를 리라이트하는 온프렘 LLM 서빙 스택. KV 캐시에 OS 페이징을 적용한 PagedAttention, 처리량을 위한 연속 배칭, Qwen 을 12GB GPU 에 적재하기 위한 AWQ 4-bit 양자화, JSON 출력을 위한 OpenAI-compatible 엔드포인트, 그리고 하나의 호출 지점에서 Ollama, Qwen, Exaone, Solar, 그리고 범용 로컬 OpenAI-compatible 백엔드를 갈아끼우게 하는 LLMProviderFactory."
series: study-thread
part: 27
alt_lang: /2026/06/08/qwen-vllm-serving-en/
next_title: "기술참조서 아크 완결 — 다음은 실데이터 평가"
next_desc: "7개 Expert, 오프라인 피처 파이프라인, teacher–student 증류, 스코어링 경로, 사유 생성기, 그리고 이제 서빙 스택까지 모두 정리됐다. 남은 것은 더 이상 아키텍처가 아니라 증거다 — 이 기계 전체가 실제 고객에게 정말 효과를 내는가? 스레드는 '어떻게 만들었는가' 에서 '작동하는가' 로 방향을 튼다."
next_status: draft
---

*"Study Thread" 시리즈의 기술참조서 아크 마지막 편이자, LLM 서빙
서브스레드의 끝. 영문/국문 병렬로, 이번 편은 모델 바깥으로 나가
그것을 굴리는* 인프라 *로 들어선다. 출처는 온프렘 프로젝트
`기술참조서/Qwen_vLLM_기술_참조서` 이고, 전체 PDF 는 이 편에 첨부할
예정이다 — 아직 public 에셋 폴더에 들어가 있지 않으므로 지금은 인용을
보류 상태로 봐 달라. 앞선 서브스레드들이 Expert 가 무엇을 읽는가,
태스크가 어떻게 공유하는가를 물었다면, 이번 편은 가장 운영적인 질문을
던진다 — 추천과 그 사유가 일단 쓰이고 나면, 그 사유를 윤색하기 위해
언어 모델을 실제로 어떻게 굴리는가? 주당 50만 건을, 소비자용 GPU 한
장에서, 클라우드 API 를 절대 호출할 수 없는 망 안에서.*

> **모든 것을 결정하는 제약.** 이것은 금융 폐쇄망이다. 내부에서 닿는
> OpenAI 도, Anthropic 도, Gemini 엔드포인트도 없다 — 모든 토큰은 우리가
> 소유한 하드웨어에서 생성된다. 따라서 서빙 트릴레마(지연시간, 처리량,
> 메모리)는 추상이 아니라 단단한 벽이다. **12 GB** VRAM 의 RTX 4070 한
> 장, 주당 **~500,000** 건의 L2a 리라이트 부하, 그리고 원본 FP16
> 가중치(**16 GB**)조차 적재되지 않는 모델. 아래의 모든 것 — 양자화,
> paged KV 캐시, 연속 배칭, provider 추상화 — 은 그 벽을 통과하기 위해
> 존재한다.

## 온프렘이라는 벽

클라우드에서 LLM 서빙 문제를 묘사할 땐 오토스케일링과 매니지드
엔드포인트를 꺼내 든다. 여기엔 둘 다 없다. 사유 생성기와 L2a
리라이터는 모두 자체 호스팅 모델을 소비하며, 참조서는 그 어려움을
*트릴레마* — 서로 상충하는 세 자원 — 로 규정한다.

- **지연시간** — 첫 토큰까지의 시간(TTFT)과 전체 완료 시간. 사유는 대면
  시나리오에서 노출되며, 느린 응답은 경험을 떨어뜨린다.
- **처리량** — 단위 시간당 요청 수. 주당 ~50만 건 L2a 리라이트를 단건
  순차 처리하면 GPU 140시간을 넘긴다.
- **메모리** — 모델 가중치 *와* 모든 중간 상태가 한 장의 VRAM 안에
  들어가야 한다.

순진한 추론 서버는 셋을 동시에 최적화하지 못한다. 참조서가 쌓는
스택은 순서대로 — 모델이 적재되게 하는 **AWQ 4-bit 양자화**, KV 캐시가
메모리를 낭비하지 않게 하는 **PagedAttention**, GPU 가 놀지 않게 하는
**연속 배칭**, 그리고 호출 지점이 어느 엔진이 답하는지 절대 하드코딩하지
않게 하는 **provider 추상화** 다.

## 왜 KV 캐시가 진짜 병목인가

Decoder-only 모델은 자기 회귀적이다. $t$번째 토큰은 그 앞의 $t-1$개
토큰으로부터 예측된다. 캐시가 없으면 매 토큰마다 전체 접두사에 대해
attention 을 다시 계산해야 하고 — $T$개 토큰을 뱉는 데 $O(T^2)$ 의
연산. **KV 캐시** 는 이미 계산된 Key 와 Value 벡터를 저장해, 각 스텝이 새
Query 만 캐시에 대해 attention 하게 만들어 토큰당 비용을 $O(d_k)$ 로
줄인다.

그 캐시가 메모리 먹는 하마다. 크기는 시퀀스 길이와 모델 형태에 선형으로
커진다.

$$ M_\text{KV} = 2 \cdot L \cdot H_\text{KV} \cdot D \cdot S \cdot B \cdot \text{sizeof}(\text{dtype}) $$

> **수식 직관.** 앞의 $2$ 는 Key *와* Value, $L$ 은 레이어 수,
> $H_\text{KV}$ 는 KV 헤드 수, $D$ 는 헤드 차원, $S$ 는 시퀀스 길이,
> $B$ 는 동시 요청 수, $\text{sizeof}$ 는 dtype 폭(FP16 = 2 bytes)이다.
> Qwen3-8B($L=32$, GQA 덕에 $H_\text{KV}=8$, $D=128$, FP16, $S=2048$)
> 의 경우 **동시 요청 1건당 약 268 MB** 가 된다. ~4 GB 의 KV 예산이면
> 약 **15개** 가 동시에 떠 있을 수 있고 — Grouped-Query Attention 이 KV
> 헤드를 32에서 8로 줄이지 않았다면 그 4배 절감이 사라져 서너 개로
> 떨어진다.

문제는 크기만이 아니라 *크기가 어떻게 할당되는가* 다. 전통적 엔진은 매
요청마다 *최대* 시퀀스 길이에 맞춘 연속 블록을 예약한다. 150 토큰을
쓰는 사유도 2048 토큰분 예약을 점유하고, 나머지 93% 는 누구도 건드릴 수
없는 죽은 메모리 — **내부 단편화** 다. 완료된 요청은 재사용하기엔 너무
작은 빈틈을 남긴다 — **외부 단편화**. 실제 활용률은 60~80% 에 머물고,
그것이 곧 동시 요청 수의 상한을 정한다.

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="230" fill="#f8fafc" rx="8"/>
  <text x="150" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">연속 예약 방식</text>
  <text x="150" y="44" text-anchor="middle" font-size="10" fill="#64748b">요청당 한 블록, 최대 길이 기준</text>
  <rect x="40" y="60" width="220" height="34" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="4 3"/>
  <rect x="40" y="60" width="42" height="34" fill="#0d9488"/>
  <rect x="82" y="60" width="178" height="34" fill="#e2e8f0"/>
  <text x="171" y="82" text-anchor="middle" font-size="9" fill="#94a3b8">낭비 (내부 단편화)</text>
  <rect x="40" y="104" width="220" height="34" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="4 3"/>
  <rect x="40" y="104" width="70" height="34" fill="#4f46e5"/>
  <rect x="110" y="104" width="150" height="34" fill="#e2e8f0"/>
  <text x="185" y="126" text-anchor="middle" font-size="9" fill="#94a3b8">낭비</text>
  <rect x="40" y="148" width="220" height="22" fill="#fde68a" stroke="#d97706" stroke-width="0.8" stroke-dasharray="3 2"/>
  <text x="150" y="163" text-anchor="middle" font-size="9" fill="#d97706">해제된 빈틈 — 작아서 재사용 불가 (외부 단편화)</text>
  <text x="150" y="196" text-anchor="middle" font-size="11" fill="#e11d48" font-weight="700">활용률 60~80%</text>
  <line x1="300" y1="50" x2="300" y2="200" stroke="#e2e8f0" stroke-width="1"/>
  <text x="450" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#1e3a5f">PagedAttention 블록</text>
  <text x="450" y="44" text-anchor="middle" font-size="10" fill="#64748b">고정 B-토큰 블록, 어디에든 배치</text>
  <g>
    <rect x="330" y="60" width="30" height="30" fill="#0d9488" rx="3"/><rect x="366" y="60" width="30" height="30" fill="#4f46e5" rx="3"/>
    <rect x="402" y="60" width="30" height="30" fill="#0d9488" rx="3"/><rect x="438" y="60" width="30" height="30" fill="#94a3b8" rx="3"/>
    <rect x="474" y="60" width="30" height="30" fill="#4f46e5" rx="3"/><rect x="510" y="60" width="30" height="30" fill="#0d9488" rx="3"/>
    <rect x="330" y="96" width="30" height="30" fill="#4f46e5" rx="3"/><rect x="366" y="96" width="30" height="30" fill="#94a3b8" rx="3"/>
    <rect x="402" y="96" width="30" height="30" fill="#0d9488" rx="3"/><rect x="438" y="96" width="30" height="30" fill="#4f46e5" rx="3"/>
    <rect x="474" y="96" width="30" height="30" fill="#94a3b8" rx="3"/><rect x="510" y="96" width="30" height="30" fill="#0d9488" rx="3"/>
  </g>
  <text x="450" y="150" text-anchor="middle" font-size="9.5" fill="#64748b">블록 테이블이 논리 → 물리 매핑</text>
  <text x="450" y="166" text-anchor="middle" font-size="9.5" fill="#64748b">마지막 블록 낭비 ≤ B−1 토큰</text>
  <text x="450" y="196" text-anchor="middle" font-size="11" fill="#0d9488" font-weight="700">활용률 96%+</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">연속 예약은 매 요청의 미사용 꼬리를 낭비하고 재사용 불가능한 빈틈을 남긴다. paged 블록은 고정 크기 조각을 아무 데나 흩뿌리며 낭비를 한 블록으로 제한한다.</figcaption>
</figure>

## vLLM 의 핵심 트릭 — KV 캐시 페이징

PagedAttention 의 발상은 거의 민망할 만큼 단순하다 — *운영체제가 물리
메모리를 관리하는 방식을 KV 캐시에 그대로 적용한다.* OS 는 프로그램마다
연속적인 가상 주소를 주지만, 실제 페이지는 비연속적으로 깔고 그 매핑을
페이지 테이블로 추적한다. PagedAttention 도 KV **블록** — $B$ 토큰분의
K 와 V 를 담는 고정 조각 — 과, 각 시퀀스의 논리 블록 번호를 물리 번호로
잇는 **블록 테이블** 로 똑같이 한다.

| 개념 | OS 가상 메모리 | PagedAttention |
| --- | --- | --- |
| 관리 단위 | 4 KB 페이지 | KV 블록 ($B$ 토큰분 K, V) |
| 매핑 | 페이지 테이블 | 블록 테이블 |
| 할당 | 요구 페이징 | 토큰 단위 지연 할당 |
| 공유 | Copy-on-Write | Copy-on-Write (빔 서치) |
| 단편화 해결 | 비연속 페이지 | 비연속 블록 |

블록이 지연 할당되므로 — 현재 블록이 가득 찰 때만 새 블록 하나 — *유일한*
낭비는 시퀀스 마지막 블록의 빈 슬롯, 최대 $B-1$ 토큰뿐이다. $B=16$ 이면
2048 토큰 시퀀스의 약 0.7%. 활용률이 60~80% 에서 **96%+** 로 뛰고,
메모리가 동시성의 상한이었으므로 이는 거의 그대로 더 많은 동시 요청으로
이어진다.

> **역사적 배경.** PagedAttention 은 UC Berkeley 의 Woosuk Kwon 과
> 동료들이 운영체제 분야 최고 학회인 **SOSP 2023** 에서 발표한
> *"Efficient Memory Management for Large Language Model Serving with
> PagedAttention"* 에서 나왔다. ML 서빙 논문이 SOSP 에 채택된 것 자체가
> 이례적이었다. 그 논증은 LLM 서빙의 진짜 병목이 *연산이 아니라 메모리
> 관리* 라는 것, 그리고 OS 의 가상 메모리 도구함이 GPU KV 캐시에 깔끔히
> 매핑된다는 것이었다. 그 위에 지어진 엔진 vLLM 은 1년 만에 오픈소스
> 서빙의 사실상 표준이 됐다.

트릭의 후반부는 **연속 배칭** 이다. 정적 배칭은 $N$개 요청을 묶고 *가장
긴* 요청이 끝날 때까지 모든 슬롯을 점유한다 — 그래서 175 토큰 L2a
리라이트가 500 토큰 사유를 기다리며 논다. 연속 배칭은 대신 *토큰* 단위로
스케줄링한다. 매 디코드 스텝마다 완료된 요청을 즉시 제거하고 대기 큐의
요청을 빈자리에 바로 끼워 넣어, GPU 를 내내 최대 배치 가까이 유지한다.

<figure style="margin:24px auto;max-width:620px;">
<svg viewBox="0 0 620 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="620" height="260" fill="#f8fafc" rx="8"/>
  <text x="20" y="30" font-size="13" font-weight="700" fill="#1e3a5f">정적 배칭</text>
  <line x1="90" y1="120" x2="600" y2="120" stroke="#cbd5e1" stroke-width="1"/>
  <text x="600" y="135" text-anchor="end" font-size="9" fill="#94a3b8">시간 →</text>
  <rect x="90" y="48" width="150" height="14" fill="#0d9488" rx="2"/><rect x="240" y="48" width="270" height="14" fill="#fecaca" rx="2"/>
  <rect x="90" y="66" width="420" height="14" fill="#4f46e5" rx="2"/>
  <rect x="90" y="84" width="90" height="14" fill="#d97706" rx="2"/><rect x="180" y="84" width="330" height="14" fill="#fecaca" rx="2"/>
  <rect x="90" y="102" width="200" height="14" fill="#64748b" rx="2"/><rect x="290" y="102" width="220" height="14" fill="#fecaca" rx="2"/>
  <line x1="510" y1="42" x2="510" y2="122" stroke="#e11d48" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="514" y="55" font-size="9" fill="#e11d48">배치 종료</text>
  <text x="375" y="76" text-anchor="middle" font-size="8.5" fill="#e11d48" font-weight="700">유휴 GPU (분홍)</text>
  <text x="20" y="170" font-size="13" font-weight="700" fill="#1e3a5f">연속 배칭</text>
  <line x1="90" y1="248" x2="600" y2="248" stroke="#cbd5e1" stroke-width="1"/>
  <rect x="90" y="186" width="150" height="14" fill="#0d9488" rx="2"/><rect x="240" y="186" width="160" height="14" fill="#14b8a6" rx="2"/><rect x="400" y="186" width="110" height="14" fill="#2dd4bf" rx="2"/>
  <rect x="90" y="204" width="420" height="14" fill="#4f46e5" rx="2"/>
  <rect x="90" y="222" width="90" height="14" fill="#d97706" rx="2"/><rect x="180" y="222" width="120" height="14" fill="#f59e0b" rx="2"/><rect x="300" y="222" width="210" height="14" fill="#fbbf24" rx="2"/>
  <text x="300" y="216" text-anchor="middle" font-size="8.5" fill="#0d9488" font-weight="700">해제된 슬롯 즉시 재충전 — 유휴 없음</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">정적 배칭은 모든 슬롯을 가장 긴 요청에 묶는다(분홍 = 낭비된 GPU). 연속 배칭은 매 스텝 완료 요청을 빼고 큐에서 다시 채워, 장치를 계속 바쁘게 둔다.</figcaption>
</figure>

결합 효과는 실재하되 정직하게 말할 가치가 있다. 참조서는 이
PagedAttention + 연속 배칭 조합으로 vLLM 이 Ollama 대비 약 **35% 높은
처리량** 을 낸다고 적으면서, *동시에* 현재 시스템은 폐쇄망 운영 편의상
Ollama 로 서빙하고 vLLM 전환은 향후 옵션으로 둔다고 밝힌다. 즉 아래의
엔진은 문서화된 정본 설계이고, 실 L2a 경로는 이 글 끝에서 따로 설명한다.

## Qwen 을 적재하기 — 12 GB 위의 AWQ 4-bit

모델은 **Qwen3-8B** 다. RoPE 회전 위치, GQA(32개 query 헤드가 8개 KV
그룹 공유), SwiGLU 활성화를 갖춘 decoder-only Transformer 로, 151,936
토큰 어휘에 최대 32K 컨텍스트(여기선 2,048로 제한). 선정 기준은 순수
추론력이 아니라 과제 자체였다 — LLM 은 *피처 해석자* 가 아니라 *텍스트
편집자* 이고, 피처는 이미 규칙 기반 매퍼가 한국어로 바꿔놓았으며,
모델은 유창하고 스키마에 맞는 출력을 종합하기만 하면 된다. 그래서
기준은 JSON 준수율, 한국어 품질, 배치 속도였다.

| 모델 | VRAM | JSON 준수 | 한국어 | 속도 |
| --- | --- | --- | --- | --- |
| **Qwen3-8B-AWQ** | ~5.5 GB | 높음 | 우수 | 빠름 |
| Gemma2-9B | ~6.0 GB | 보통 | 보통 | 보통 |
| Llama3-8B | ~5.5 GB | 보통 | 미흡 | 빠름 |
| Mistral-7B | ~5.0 GB | 보통 | 미흡 | 빠름 |

그러나 8B 파라미터의 FP16 은 **16 GB** — 12 GB 카드에 안 들어간다.
**AWQ**(Activation-Aware Weight Quantization, Lin et al., MLSys 2024)가
4-bit 으로 **~5.5 GB** 까지 줄인다. 핵심 통찰은 모든 가중치가 동등하게
중요하진 않다는 것 — 약 1% 만 *salient* 하고, 그 중요도는 채널 활성화의
크기를 따른다. 출력 오차가
$\sum_j \lVert \delta W_{j,\cdot}\rVert^2 \cdot \lVert X_{\cdot,j}\rVert^2$
로 스케일되기 때문이다. AWQ 는 작은 캘리브레이션 셋에서 그 채널들을 찾아
양자화 전에 $s_j = \lVert X_{\cdot,j}\rVert^\alpha$($\alpha\approx0.5$)
배 확대해 반올림 격자를 상대적으로 촘촘하게 하고, 추론 시 $s_j^{-1}$ 을
곱해 스케일을 복원하되 오차만 줄여 둔다. 카드 위 예산은 이렇게 읽힌다 —
~5.5 GB 가중치 + ~4.0 GB KV 캐시(`--gpu-memory-utilization 0.85` 기준)
+ ~1.5 GB CUDA 오버헤드 + ~1.0 GB 마진 = 12 GB.

## OpenAI-Compatible 엔드포인트와 JSON 출력

vLLM 은 OpenAI 규격을 따르는 HTTP API 를 노출하므로 기존 `openai.OpenAI`
SDK 호출이 그대로 동작한다 — `base_url` 만 로컬 서버를 가리키고, 로컬
vLLM 은 키가 필요 없으니 API 키는 아무 값이나 넣는 더미다. 정본 실행 명령:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

`--max-model-len 2048` 은 의도적이다. 프롬프트는 입력 ~800 + 출력 ~500
토큰이라 2,048이면 충분하고, 불필요하게 큰 윈도우는 KV 캐시를 태워
동시성을 깎는다. `messages` 를 보내는 클라이언트는 서버 측에서 Qwen3 의
ChatML(`<|im_start|>system / user / assistant`)로 자동 변환된다 — 그래서
ChatML 태그를 직접 붙이면 이중 래핑이 된다.

구조화 출력에 대해 참조서는 한계를 솔직히 적는다 — OpenAI 의
`response_format={"type": "json_object"}` 는 vLLM 에서 *제한적* 으로만
지원된다. 그래서 시스템은 이중으로 단속한다 — 프롬프트 수준 JSON
유도(JSON 전용을 요구하는 시스템 롤, 스키마 인라인, "다른 텍스트 금지"
지시)에 더해, 첫 유효 `{...}` 블록(또는 펜스된 JSON 코드 블록)을 뽑아
`json.loads` 로 검증한 뒤에야 신뢰하는 후처리 정규식 추출기. 사유 생성기는 엄격한
스키마(`{"reasons": [...], "summary": "..."}`)를 뱉지만, L2a 리라이터는
반대로 *순수 텍스트* 를 뱉는다 — 문장 윤색이 JSON 을 만들어선 안 되기
때문이다.

## Provider 추상화 — 하나의 호출 지점, 여러 백엔드

이 모든 걸 유지보수 가능하게 만드는 조각이 `src/grounding/llm_provider.py`
의 `LLMProviderFactory` 다. 폐쇄망은 의도적으로 모든 클라우드
백엔드(Bedrock, OpenAI, Gemini)를 **제외** 하고 자체 호스팅 가능한 것만
노출한다. 단일 config 키가 엔진을 고른다.

```python
from src.grounding import LLMProviderFactory
provider = LLMProviderFactory.create({
    "llm_provider": {
        "backend": "qwen",   # ollama | qwen | exaone | solar | local | dummy
        "qwen": {"model": "qwen3:14b",
                 "endpoint": "http://host.docker.internal:11434/v1"},
    }
})
response = provider.generate(prompt, response_format={"type": "json_object"})
```

모든 백엔드가 동일한 `generate()` 시그니처를 구현하고, `response_format`
은 지원하는 백엔드에서만 OpenAI-compatible 호출로 전달된다. factory 는
토큰이 실제로 어디서 오는지를 숨긴다.

| 백엔드 | 엔진 / 경로 | 기본 엔드포인트 | 비고 |
| --- | --- | --- | --- |
| `ollama` | Ollama OpenAI-compatible `/v1` | `:11434/v1` | qwen3 / llama3 / exaone Ollama import |
| `qwen` | `ollama` alias, model→`qwen3:14b` | `:11434/v1` | 편의 기본값 |
| `exaone` | 자체 호스팅 (vLLM serve) | `:8000/v1` | `lgai/exaone-3.5-32b-instruct` |
| `solar` | 로컬 자체 호스팅 또는 Upstage API | `:8000/v1` | `mode: local` 이면 온프렘 |
| `local` | 임의의 OpenAI-compatible 서버 | 설정 가능 | 범용 탈출구 |
| `dummy` | 하드코딩 출력 | — | 오프라인 테스트 / 최종 폴백 |

<figure style="margin:24px auto;max-width:600px;">
<svg viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:system-ui,sans-serif;">
  <rect x="0" y="0" width="600" height="250" fill="#f8fafc" rx="8"/>
  <rect x="40" y="100" width="120" height="48" rx="6" fill="#1e3a5f"/>
  <text x="100" y="122" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">L2a / 사유</text>
  <text x="100" y="138" text-anchor="middle" font-size="9" fill="#cbd5e1">.generate(prompt)</text>
  <rect x="210" y="96" width="110" height="56" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.2"/>
  <text x="265" y="118" text-anchor="middle" font-size="11" font-weight="700" fill="#4f46e5">Factory</text>
  <text x="265" y="134" text-anchor="middle" font-size="9" fill="#64748b">backend: …</text>
  <line x1="160" y1="124" x2="208" y2="124" stroke="#94a3b8" stroke-width="1.4"/><polygon points="208,124 200,120 200,128" fill="#94a3b8"/>
  <g font-size="10" font-weight="700">
    <rect x="400" y="40" width="160" height="30" rx="5" fill="#f0fdfa" stroke="#0d9488"/><text x="480" y="60" text-anchor="middle" fill="#0d9488">ollama / qwen · :11434</text>
    <rect x="400" y="78" width="160" height="30" rx="5" fill="#fffbeb" stroke="#d97706"/><text x="480" y="98" text-anchor="middle" fill="#d97706">exaone · vLLM :8000</text>
    <rect x="400" y="116" width="160" height="30" rx="5" fill="#eef2ff" stroke="#4f46e5"/><text x="480" y="136" text-anchor="middle" fill="#4f46e5">solar · local :8000</text>
    <rect x="400" y="154" width="160" height="30" rx="5" fill="#f1f5f9" stroke="#64748b"/><text x="480" y="174" text-anchor="middle" fill="#64748b">local · 설정 가능</text>
    <rect x="400" y="192" width="160" height="30" rx="5" fill="#fee2e2" stroke="#e11d48"/><text x="480" y="212" text-anchor="middle" fill="#e11d48">dummy · 폴백</text>
  </g>
  <g stroke="#cbd5e1" stroke-width="1.2">
    <line x1="320" y1="124" x2="400" y2="55"/><line x1="320" y1="124" x2="400" y2="93"/>
    <line x1="320" y1="124" x2="400" y2="131"/><line x1="320" y1="124" x2="400" y2="169"/><line x1="320" y1="124" x2="400" y2="207"/>
  </g>
  <text x="265" y="186" text-anchor="middle" font-size="9" fill="#94a3b8">클라우드 백엔드는 설계상 제외</text>
</svg>
<figcaption style="text-align:center;font-size:12px;color:#64748b;margin-top:4px;">하나의 호출 지점, 하나의 generate() 시그니처. factory 가 config 키를 자체 호스팅 백엔드로 해소한다. 폐쇄망에서 클라우드 경로는 닿지 않는다.</figcaption>
</figure>

## 사유-리라이트(L2a)는 실제로 어떻게 호출하는가

L2a 리라이터가 가장 무거운 소비자다. 모든 고객에게 도는 게 *아니다* —
풍부도 게이트가 `rich` 와 `moderate` 컨텍스트만 LLM 으로 보내고, 윤색
효과가 미미한 `sparse` 는 건너뛴다. 자격 있는 초안마다 프롬프트를 짠다 —
300자로 자른 L1 초안 + ~200자 컨텍스트 — 단단한 지시와 함께: 사실
유지, **숫자 추가 금지**(환각된 "연 5% 수익" 은 컴플라이언스 위반),
2~3문장, 순수 텍스트만.

실 운영 경로는 정본 다이어그램의 단일 vLLM 서버가 아니라 **Ollama
dual-route** 다.

1. **Primary** — Ollama 위 `exaone3.5:2.4b` 가 저비용으로 대부분을
   처리한다.
2. **3-Layer Safety Gate** — Gate 1(파싱: 빈 텍스트와 JSON 잔해 거부),
   Gate 2(컴플라이언스: "확정 수익", "원금 보장", "N% 수익"… 정규식
   블랙리스트), Gate 3(품질: 30~200자, 한국어 80% 이상).
3. **Escalation** — Gate 1 또는 Gate 3 실패 시 `qwen3:14b` 로 재시도.
   단 *Gate 2*(컴플라이언스) 실패 시엔 escalation **없이** 즉시 L1
   원본으로 폴백한다.

마지막 규칙이 중요하다 — 컴플라이언스 위반은 결코 "더 열심히 시도" 하지
않고 버려지며, 모든 경로가 실패하면 규칙 기반 L1 초안이 그대로 배포되어
고객이 빈 사유를 보는 일이 없다. 클라이언트의 `max_concurrent=10` 상한은
서버의 ~15 요청 KV 천장과 맞물려 시퀀스 길이 변동에 마진을 둔다.
(`qwen3:14b` escalation 엔 알려진 운영 결함이 하나 붙는다 — thinking
모드 토큰 트랩 때문에 실제 출력이 ~150~200 토큰인데도 넉넉한
`max_tokens` 를 강제해야 한다 — 그러나 이건 아키텍처가 아니라 튜닝
메모다.)

## 여기서 멈추는 이유

이것이 마지막 벽돌이다. 아크를 거꾸로 짚어 보자 — 각자 다른 종류의
신호를 읽는 7개의 이종 Expert(PersLay 의 *형태*, CausalOT 의 수송,
GCN 의 쌍곡 그래프, Temporal 앙상블, GMM/HMM 레짐, economics 와
multidisciplinary 피처); 그 모든 것을 구체화하는 오프라인 피처
파이프라인; 무거운 모델을 서빙 가능한 student 로 압축하는 teacher–student
증류; 점수를 순위 추천으로 바꾸는 스코어링 경로; 그것을 설명하는 사유
생성기; 그리고 이제 윤색 모델을 단일 12 GB GPU 에서, 주당 50만 번,
폐쇄망 밖으로 단 한 패킷도 내보내지 않고 굴리는 서빙 스택. *어떻게* 는
마침내 처음부터 끝까지 문서화됐다.

뚜렷하게 문서화되지 *않은* 것은 *과연 작동하는가* 다. 지금까지 기댄 모든
결과 — PersLay 실루엣 프로브, 기법별 검증 — 은 하위 구성요소 점검,
만들어도 좋다는 청신호였지, 기계 전체에 대한 판정은 아니었다. 열린
프런티어는 더 이상 아키텍처가 아니라 실데이터 위의 증거다 — 조립된
시스템이 위에서 아래까지 실제 고객의 추천 품질을 정말 움직이는가,
얼마나? 스레드가 다음으로 방향을 트는 지점이 거기다 — 어떻게 만들었는가
에서 작동하는가로.
