---
title: "[MRM 스레드] 에피소드 4 — 설명이 아키텍처가 될 때: Inherent XAI 와 FD-TVS 스코어링"
date: 2026-04-28 12:00:00 +0900
categories: [MRM Thread]
tags: [mrm, xai, explainability, ple, fd-tvs, financial-ai]
lang: ko
excerpt: "사후적 XAI(SHAP·LIME)는 운영 환경에서 불안정하고 모델과 분리되어 있다. 우리는 아키텍처 단계에서 설명을 만들기로 결정했다 — 게이트 가중치, CEH 귀속, 마할라노비스 OOD 가 추론 경로 자체의 부산물로 떨어진다. FD-TVS 는 그 위에 얹은 운영 스코어링 철학이다."
series: mrm-thread
part: 4
alt_lang: /2026/04/28/mrm-ep4-xai-foundation-en/
next_title: "에피소드 5 — RAG + LanceDB · 감사 인프라가 결국 검색 문제인 이유"
next_desc: "감사 로그는 쓰기 전용이 아니다. 질의 가능한 지식 베이스다. 운영/감사 검색을 RAG + LanceDB 로 구성한 이유, 그리고 그 결과로 인적 감독·공정성 모니터링·분기 집계가 어떻게 풀리는가."
next_status: published
source_url: https://doi.org/10.5281/zenodo.19621884
source_label: "Paper 1 + Paper 3 (Zenodo DOIs)"
---

*"MRM 스레드" 4편. Ep 3 가 감사 로그 층 — 7개 테이블, HMAC 체인, 컨센서스 중재 — 을 다뤘다면, 이번 편은 그 한 층 아래로 내려간다. 누가 *무엇이 로그에 들어갈지* 정하는가? 매 예측의 설명, 귀속, 신뢰도 플래그는 어디서 오는가? 추론 경로에 사후적으로 덧붙인 모듈에서가 아니다. 모두 아키텍처 단계의 XAI 선택의 결과로 떨어진다. Ep 4 는 *왜 설명이 아키텍처여야 하는가* 의 사례 분석이고, 그 위에 FD-TVS 가 무엇을 더하는지에 대한 이야기다.*

## 사후적 XAI 의 세 가지 구조적 문제

규제 당국이 "왜 이 모델이 이 예측을 내놓았는가" 를 물어왔을 때, 대부분의 금융 AI 팀이 본능적으로 꺼내는 도구는 SHAP·LIME·Integrated Gradients 같은 사후적 귀속 모듈이다. 추론 파이프라인에 attribution 모듈을 하나 끼우고, 매 예측마다 상위 K개 기여 피처를 노출시킨 뒤, 이걸 explainability 라고 부른다. 운영 환경에서 이 접근은 세 가지 구조적 문제를 안고 있다.

**불안정성.** Salih 등(2023)을 비롯한 후속 연구들은 SHAP 과 LIME 이 background distribution 선택, 샘플 크기, 심지어 random seed 변화에 민감하다는 점을 documented 했다. 같은 예측이 explainer 호출 방식에 따라 *상이한* 상위 기여 피처를 내놓을 수 있다. 일회성 연구물에서는 감수할 만하다. 그러나 규제 당국이 15개월이 지난 시점에 특정 예측 한 건을 재구성하려는 상황에서, 불안정성은 논문 caveat 가 아니라 컴플라이언스 부채다.

**추론 시점 연산 비용.** SHAP 계열은 비싸다. 분당 수천 건의 추천을 서빙하는 CPU Lambda 추론 경로 위에서 매 예측 SHAP 은 예산을 박살낸다. 대부분의 운영 배포는 결국 샘플링 처리(매 예측 보장 손실)나 일부 부분집합에 대한 사전 계산(보편성 손실) 중 하나로 후퇴한다. 어느 쪽이든 "모든 예측에 explainability 보장" 약속은 조용히 깨진다.

**모델과의 분리.** SHAP 과 LIME 은 모델을 블랙박스로 취급한다. 귀속 결과는 *별도의* 근사기가 입력 주변에서 모델 행동을 추정한 결과다. 모델과 explainer 의 답이 다를 때 — 실제로 종종 다르다 — 규제 당국·고객·감독 위원회에 보여지는 답은 explainer 쪽이다. 모델의 실제 reasoning 은, 불투명한 MLP 에서 그런 것이 존재한다 가정해도, 영영 보이지 않는다.

이 세 문제는 누적된다. 동일 예측에 대한 동일 질문에 *수년 후에도 답해야 하는* 규제 AI 시스템에서, 사후적 XAI 는 위치가 계속 움직이는 바닥이다.

## 아키텍처 XAI 라는 다른 선택

우리가 일찍 내린 결정은 설명 작업을 아키텍처 자체로 밀어 올리는 것이었다. 예측 *후에* 설명을 계산하는 게 아니라, 설명이 예측의 *일부* 가 되도록.

Heterogeneous Expert PLE (Paper 1) 는 7개의 구조적으로 상이한 shared expert — DeepFM, Temporal Ensemble, Hyperbolic GCN, PersLay, Causal, LightGCN, Optimal Transport — 위에 설계되어 있다. 각 expert 는 *명명된 수학적 연산* 이지, 무작위 초기값을 가진 generic MLP 가 아니다. CGC(Customised Gate Control) 는 각 태스크의 예측을 expert basket 위로 명시적인 expert 별 가중치와 함께 routing 한다.

여기서 얻는 것은 게이트 가중치 자체가 *설명* 이라는 점이다. 시스템이 "고객 X 의 상품 P 교차 판매 확률 0.78" 을 예측할 때, 그 예측에 첨부된 게이트 가중치는 — 예측의 순간에, 같은 forward pass 에서 — *Temporal 35% (지출 추세) + HGCN 28% (상품 계층 적합성) + Causal 15% (개입 추론) + ...* 형태로 그 예측이 무엇 때문에 나왔는지 말해준다. 별도 explainer 호출이 없다. routing decision 이 기록되니까 설명도 기록된다.

이게 *Inherent XAI* 가 실무에서 의미하는 바다. 설명은 UI 층이 아니다. 아키텍처 결정이다.

## 매 예측마다 세 층의 설명

단일 forward pass 가 예측 옆에 자동으로 적립되는 세 층의 설명을 produce 한다.

**게이트 가중치** — expert 별 기여도. 각 expert 가 명명된 inductive bias 를 인코딩하므로, 게이트 가중치는 비즈니스 가독 narrative 로 직접 매핑된다. *"Temporal 35%"* 는 *"최근 지출 패턴"* 이지 *"hidden unit 47 활성화"* 가 아니다. 이 가중치를 우리는 추천 사유 생성층(Paper 2)의 고객 대면 설명 근거로 사용한다.

**CEH 귀속** — Causal Explainability Head, Causal expert 내부의 피처 단위 기여도. 특정 예측에서 Causal expert 가 dominant route 일 때, CEH 는 causal DAG 안의 어느 피처가 결론을 끌었는지 노출한다. 게이트 가중치 아래 더 미세한 귀속 층이다 — *"Causal 38%"* 만으로 부족할 때, 감독자가 *어떤 causal pathway* 인지 알고 싶을 때를 위한.

**Causal latent 위 마할라노비스 OOD** — 매 예측의 신뢰도 플래그. Causal expert 의 잠재 공간 위에서 in-distribution reference 대비 마할라노비스 거리를 계산해, 매 예측마다 binary trust 플래그를 emit 한다. 합성 OOD probe 에서 5% 오탐률 기준 100% 탐지율. 해석: 이 플래그가 켜지면, 그 예측은 모델이 학습되지 않은 피처 공간 영역에 있고, 고객 대면 설명은 다운그레이드되거나 보류되어야 한다.

세 층 모두 추론 시점에 계산되고, 세 층 모두 감사 로그로 떨어진다. 어느 것도 별도 사후 explainer 호출을 요구하지 않는다. 추론 경로가 부산물로 produce 한다.

## 왜 이게 규제 대응의 토대인가

이 부분이 위쪽 규제층과 연결되는 지점이다.

Paper 2 가 다룬 5개 규제 산출물 — 한국 AI 기본법 §35 영향평가, EU AI Act Art. 9 위험 기록, Annex IV 기술문서 증거 매핑, PIPA + GDPR Art. 35 개인정보 영향평가, 금융위 AI 가이드라인 분기 공시 — 는 모두 동일한 매 예측 구조화 로그를 소비한다. 작성된 문서가 아니라 집계 query 다.

그러나 이 패턴이 작동하는 이유는 매 예측 로그가 *입력과 출력만 담는 게 아니라 구조화된 설명 데이터를 담기 때문이다.* 만약 예측 기록이 `(input_vector, output_score, timestamp)` 만이었다면, 어떤 집계 query 도 *"왜 모델이 고객 Y 에게 X 를 결정했는가"* 에 답할 수 없다 — 답이 데이터 안에 없으니까. 5개 generator 가 query 일 수 있는 이유는 로그가 *추론 근거* 를 담기 때문이고, 로그가 추론 근거를 담는 이유는 아키텍처가 추론 근거를 출력으로 produce 하기 때문이다.

Inherent XAI 가 토대다. 감사 로그가 2층이다. 5개 규제 generator 가 지붕이다. 토대를 사후적 SHAP 으로 바꾸면 2층과 3층이 무너진다 — 매 예측 로그가 구조화 설명 column 을 잃고, 그러면 집계 query 가 substrate 를 잃고, 그러면 규제 산출물이 다시 손으로 쓰는 문서로 회귀한다.

EU AI Act Art. 13 (투명성 의무) 와 한국 AI 기본법 §31 (투명성) 은 *explainer 를 갖고 있는 것* 으로 만족되지 않는다. *어느 예측에 대해서든 안정적이고 재구성 가능한 설명을 요청 시 produce 할 수 있는 것* 으로 만족된다. 우리가 아는 한, 모델 재학습 cycle 을 거쳐도 그 약속을 유지할 수 있는 유일한 아키텍처가 inherent XAI 다.

## FD-TVS — XAI 위에 얹은 스코어링 철학

이 시스템의 온프렘 선례는 스코어링 층에서 상품 단위 가중치를 사용했다. 각 금융 상품에 정적 가중치가 매뉴얼로 설정되어 있었다. 신상품이 launch 되면 config 가 손으로 update 되었다. 스코어링 층은 평면 lookup 이었다.

이건 세 측면에서 fragile 했다. 신상품 launch 마다 매뉴얼 재설정 필요. 고객 세그먼트 차이 (25세 첫 예금 고객 vs 60세 자산가) 가 스코어링 가중치에 반영되지 않음 — 세그먼트 행동은 모델 예측에서 잡혔기를 바라며 스코어링 층은 agnostic. 행동 변화 (예: 생애 이벤트 trigger 로 인한 특정 피처 급증) 가 스코어를 영향 줄 메커니즘 없음.

FD-TVS — Financial DNA Targeted Value Scoring — 는 그 스코어링 층의 재설계다. 세 가지 철학적 전환:

**상품 단위 → 태스크 단위.** 가중치는 *태스크* (교차 판매 의향, 이탈 위험, 적합성 fit 등) 에 붙는다. *상품* 이 아니다. 신상품은 기존 태스크 구조를 상속받고 재설정 필요 없음. 위에서 다룬 XAI 게이트 가중치가 여기에 직접 feed 된다 — 태스크 선택은 그 예측의 expert 별 routing 에 의해 informed 된다.

**세그먼트 인식 (`segment_task_weights`).** 각 고객 세그먼트가 태스크 가중치 위에 자체 multiplier 를 갖는다. 1.0–1.5 범위로 clipping. clipping 은 의도적이다. 1.0 미만을 허용하면 세그먼트 휴리스틱이 태스크 시그널을 suppress 할 수 있어, 모델이 1차 시그널 source 라는 역할이 깨진다. 1.5 초과를 허용하면 세그먼트 override 가 모델을 dominate 할 수 있다. 1.0–1.5 범위는 말한다 — *세그먼트는 multiplier 로서 중요하지, override 는 아니다.*

**행동 인식 (`dynamic_weight_rules`).** 특정 피처 임계값이 스코어링 시점에 특정 태스크 가중치를 boost 할 수 있다. 이탈과 상관 있는 피처 급증 → 이탈 태스크 가중치 ↑. 비활성 계좌에서 작은 입금 시퀀스 → 예금 상품 태스크 가중치 ↑. 반응적 스코어링이다 — *행동 자체가 가중치 조정을 trigger 하는 시그널* 이지, 정기적 재튜닝이 아니다.

세 가지 모두 `pipeline.yaml` 에 산다. 운영팀은 코드 변경 없이 세그먼트 표를 조정하거나 행동 규칙을 추가할 수 있다. 이게 중요하다 — 스코어링 정책 조정이 시간 단위로 ship 가능하지, 주 단위가 아니다. 모든 조정은 config 버전이 감사 로그에 stamp 되어 MRM 스택의 나머지가 따르는 동일한 15개월 재구성 윈도우를 obey 한다.

XAI 와의 연결은 직접적이다. XAI 층은 시스템에게 *왜* 그 예측이 났는지 (게이트 가중치 × CEH × OOD) 를 알려준다. FD-TVS 는 시스템에게 *그 예측이 최종 점수에서 얼마나 가중되어야 하는지, 이 고객이 누구이고 현재 어떻게 행동하는지를 고려해서* 알려준다. 두 층 모두 입력을 로그한다. 고객 대면 설명은 *"최근 지출 패턴(Temporal 35%) 과 상품 계층 적합성(HGCN 28%) 때문에 추천드렸고, 귀하 세그먼트가 이 카테고리에 보여온 선호로 가중되었습니다"* 같은 단일 문자열이 된다 — 모든 component 가 감사 로그로부터 독립적으로 복구되므로 defensible 한.

## XAI 토대가 enable 하는 것

Ep 5, 6 으로 가는 길:

**Ep 5 (RAG + LanceDB)** 는 매 예측 설명 로그가 어떻게 대규모로 query 되는지 다룬다. 설명 column 위 vector retrieval 이 *"지난 분기 Temporal 이 dominate 하고 OOD 가 발사된 예측들을 찾아라"* 같은 질의를 초 단위로 답할 수 있게 한다. 설명 column 이 먼저 존재해야 retrieval 이 의미 있는 일을 한다.

**Ep 6 (Modular adaptability)** 는 규제가 변할 때 이 아키텍처가 어떻게 견디는지 다룬다. 새 규제 = 동일 설명 로그 위 새 집계 query. XAI 토대는 규제 agnostic 이고, 규제 층은 swap 가능하다.

## 자동화되지 않는 것

아키텍처 XAI 선택이 사주지 않는 것 몇 가지:

**사람 손의 점수 검증.** *"이 고객 추천에 Temporal 35% 기여가 적절한 설명인가"* 는 추천 검토 위원회나 고객 대면 RM 의 판단 영역이다. 자동화된 건 그 기여도가 기록되고 안정적이고 재구성 가능하다는 점이다. 자동화되지 않은 건 그 설명이 비즈니스 sense 가 있는지다.

**경계 케이스 해석성.** 7개 expert 가 모두 비슷한 비율로 기여할 때 (게이트 엔트로피가 최대치 근방), 게이트 가중치 설명은 *"전부가 조금씩 기여했다"* 가 된다. 정직하지만 만족스럽지 않다. 우리는 high-entropy 예측을 별도 해석 카테고리로 다룬다 — *low-confidence, high-entropy* 예측은 OOD 플래그 발사 여부와 무관하게 oversight 층에서 인적 검토로 플래그된다.

**아키텍처 lock-in.** 전체 논지가 heterogeneous expert basket 이 안정적으로 유지된다는 가정에 의존한다. 향후 iteration 이 7개 expert 를 단일 transformer 로 교체하면, 게이트 가중치 설명이 사라지고 다시 사후적 XAI 로 돌아간다. 이건 단기 구현 선택이 아니라 장기 아키텍처 commitment 다. 5개 규제 generator (Ep 6) 는 이 commitment 가 유지된다는 가정 위에 설계되어 있다.

## 다음

Ep 5 는 한 층 위로 — 매 예측 설명 로그를 query 가능 시스템으로 만드는 retrieval 층으로 — 올라간다. 운영/감사 인프라를 RAG + LanceDB 로 선택한 이유, columnar version-aware retrieval 이 공정성 모니터링·인적 감독 escalation·분기 집계에 무엇을 가져다주는지, 그리고 감사 로그가 왜 쓰기 전용이 아닌지.

소스: [Paper 1 (Zenodo)](https://doi.org/10.5281/zenodo.19621884) 의 heterogeneous expert architecture 와 게이트 가중치 explainability, [Paper 3 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) 의 CEH 와 Causal Guardrail (마할라노비스 OOD); FD-TVS 스코어링 config 는 [`configs/pipeline.yaml`](https://github.com/bluethestyle/aws_ple_for_financial) 의 `scoring.segment_task_weights` 와 `scoring.dynamic_weight_rules` 에 산다.
