---
title: "[Study Thread] ADATT-4 — 학습 루프·Loss Weighting·Optimizer·CGC 동기화"
date: 2026-04-20 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, training-loop, loss-weighting, optimizer, specs]
lang: ko
excerpt: "adaTT 서브스레드 마무리 — 2-Phase Training Loop, Loss Weighting 전략 (Uncertainty · GradNorm · DWA), Optimizer · Scheduler 설정, CGC ↔ adaTT 동기화, 메모리·성능 노트. adaTT 기술 참조서 PDF 첨부."
series: study-thread
part: 10
alt_lang: /2026/04/20/adatt-4-training-loop-loss-weighting-optimizer-en/
next_title: "다음 서브스레드 — Causal OT · TDA · Temporal · Economics Expert 기초"
next_desc: "PLE + adaTT 가 끝났으니 다음은 7개 이종 Shared Expert 각자의 수학적 기초로. 우선 CausalOT(인과 추론 + 최적 수송), TDA(위상 데이터 분석 / PersLay), Temporal(Mamba + LNN + Transformer), Economics 피처 기반 Expert 순서로 각각 서브스레드를 열 예정."
next_status: draft
source_url: /adaTT_기술_참조서.pdf
source_label: "adaTT 기술 참조서 (KO, PDF)"
---

*"Study Thread" 시리즈의 adaTT 서브스레드 마지막 4편. 영문/국문 병렬로
ADATT-1 → ADATT-4 에 걸쳐 본 프로젝트의 adaTT 메커니즘을 정리해왔다.
출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서` 이다. ADATT-3
까지 adaTT 의 네 가지 설계 결정 — Transfer Loss · Group Prior · 3-Phase
Schedule · Negative Transfer 차단 — 이 서로 어떻게 맞물리는지 정리했고,
이번 4편은 이 구조가 *실제 학습 루프* 에 어떻게 꽂히는지, CGC 와의
동기화 계약은 무엇이고, 성능을 어떻게 지켜내는지를 다룬다. 글 하단에서
전체 PDF 도 받을 수 있다.*

## ADATT-3 이 남긴 질문

설계는 끝났다. 친화도 측정, 전이 가중치 계산, 3-Phase 스케줄, 음수 차단
— 모두 `adaTT` 모듈 안에서 자기 완결적으로 돈다. 하지만 이 모듈은 혼자
학습하지 않는다. 본 프로젝트의 Trainer 는 자체의 2-Phase 학습 (Shared
Pretrain → Cluster Finetune) 을 돌리고, 16 태스크에 Uncertainty Weighting
을 적용하며, AdamW + SequentialLR 로 학습률을 관리하고, CGC 가 gate
가중치를 학습한다. adaTT 가 이들과 어떻게 공존하는가 — 이번 편의 핵심
질문이다.

여섯 가지 결정으로 풀린다.

## 결정 1 — 2-Phase Training Loop (adaTT 의 3-Phase 와는 *다른 층*)

혼동 주의. adaTT *내부* 의 3-Phase (Warmup → Dynamic → Frozen) 는
친화도 스케줄이고, Trainer 의 2-Phase (Phase 1 Pretrain → Phase 2
Finetune) 는 학습 단계다. 두 층은 서로 직교하며, 각자의 freeze 시점이
따로 존재한다.

*Phase 1 — Shared Expert Pretrain.* `shared_expert_epochs` (기본 15)
동안 전체 모델을 학습한다. Shared Experts, CGC, Task Experts, Task
Towers 모두 업데이트 대상. adaTT 는 *활성* — gradient 추출과 transfer
loss 가 동작한다.

*Phase 2 — Cluster Finetune.* `cluster_finetune_epochs` (기본 8) 동안
클러스터별 Task Expert 서브헤드만 학습한다. Shared Experts 는 frozen.
adaTT 는 *비활성* 이다. 이유는 단순하다 — adaTT 의 gradient 는 Shared
Expert 파라미터에 대해 계산되는데 그게 frozen 이면 gradient 가 0 이고,
`autograd.grad` 호출은 낭비다.

왜 이렇게 나누는가. 클러스터별 학습을 Shared 과 같이 돌리면 클러스터
특화 신호가 공유 표현을 오염시킨다. Shared 를 먼저 충분히 학습한 뒤
고정하고 클러스터 헤드만 미세조정하는 순서가 특화 / 일반화의 경계를
깨끗하게 긋는다.

### Phase 전환 시 리셋

Phase 2 로 넘어갈 때 다음을 모두 리셋한다.

| 리셋 항목 | 이유 |
|---|---|
| Optimizer | Shared frozen → AdamW 모멘텀이 stale, 새로 시작 |
| Scheduler | Phase 2 전용 warmup (2 epoch, Phase 1 의 5 epoch 보다 짧음) |
| GradScaler | AMP 스케일러 상태 초기화 (phase 전환 시 loss 스케일 변화) |
| Early stopping | `best_val_loss`, `patience_counter` 모두 초기화 |
| CGC Attention | Shared frozen → CGC gating 도 같이 freeze |

중요한 세 가지 안전장치가 있다. (1) adaTT 는 Phase 2 시작 시
`model.adatt = None` 으로 *교체* 되지 않고 *백업* 된 뒤 비활성화된다.
(2) Phase 2 종료 시점에 `finally` 블록에서 adaTT 가 반드시 복원된다 —
예외 발생 시에도 모델 상태가 일관되게 유지되어 체크포인트 / 추론 호환성이
보장된다. (3) 학습률 warmup 은 Phase 1 의 5 epoch 에서 Phase 2 의 2
epoch 로 단축된다. Phase 2 는 짧기 때문이다 — 긴 warmup 은 의미 없다.

## 결정 2 — 16 Task 의 Loss 를 어떻게 균형 잡는가

16 태스크의 loss 스케일이 제각각이다. CTR / CVR 의 focal loss 와 LTV 의
huber loss 는 범위가 다르고, brand_prediction 의 InfoNCE 는 또 다르다.
수동으로 태스크별 가중치를 튜닝하는 건 조합 폭발이다. Kendall et al.
(CVPR 2018) 의 Uncertainty Weighting 이 이 문제를 자동화한다.

$$\mathcal{L}_i^{weighted} = \frac{1}{2 \sigma_i^2} \cdot \mathcal{L}_i + \frac{1}{2} \log \sigma_i^2$$

- $\sigma_i^2 = \exp(\text{log\_var}_i)$: 태스크 $i$ 의 학습 가능한
  homoscedastic uncertainty.
- `log_var` clamp: $[-4, 4]$, precision clamp: $[0.001, 100]$.

> **수식 직관.** 첫 항은 정밀도 ($1/\sigma^2$) 로 가중된 loss — 불확실한
> 태스크는 가중치가 작아진다. 둘째 항 $\frac{1}{2} \log \sigma_i^2$ 는
> 정규화 페널티 — $\sigma$ 가 무한히 커져 loss 를 0 으로 만드는 치팅을
> 막는다. 가우시안 likelihood $\mathcal{N}(\hat{y}, \sigma^2)$ 의 $-\log p$
> 에서 자연스럽게 도출된 형태.

중요한 건 adaTT 와의 *순서* 다. Uncertainty Weighting 은 adaTT *이전* 에
적용된다. 즉 adaTT 의 `task_losses` 입력에는 이미 불확실성 가중치가
반영된 값이 들어온다. 이건 의도된 동작이다 — 비즈니스적으로 중요한
태스크 (nba: 2.0 같은 높은 고정 가중치) 의 학습 신호가 전이를 통해 다른
태스크로도 전파되어야 한다.

### 태스크별 고정 가중치

일부 태스크는 Uncertainty Weighting 위에 추가로 고정 가중치가 걸린다.
특히 비즈니스 중요도 / 양성 비율 / FN 비용이 다른 태스크들.

| 태스크 | weight | loss type | 비고 |
|---|---|---|---|
| ctr | 1.0 | focal ($\gamma$=2, $\alpha$=0.25) | 표준 |
| cvr | 1.5 | focal ($\gamma$=2, $\alpha$=0.20) | 양성 비율 극소 → weight 상향 |
| churn | 1.2 | focal ($\gamma$=2, $\alpha$=0.60) | FN 비용 높음 → alpha 상향 |
| nba | 2.0 | CE | 12 classes, 비즈니스 핵심 |
| ltv | 1.5 | huber ($\delta$=1.0) | regression, outlier 대응 |
| brand_prediction | 2.0 | contrastive (InfoNCE) | 50K 브랜드 |

## 결정 3 — Per-Expert Learning Rate 와 SequentialLR

Shared Expert 7 개는 구조가 다르다. 128D unified_hgcn 은 하이퍼볼릭
공간에서 학습하므로 보수적인 lr 이 필요하고, 64D DeepFM 은 상대적으로
빠른 수렴이 가능하다. 전역 단일 lr 을 쓰면 어느 쪽이든 최적에서 벗어난다.

해결은 Expert 별 param_group. 각 Shared Expert 의 파라미터를 별도
param_group 으로 묶고, `model_config.yaml` 에서 expert 별 lr / weight_decay
override 를 선언한다. Phase 2 에서 Shared Expert 가 frozen 이면
`requires_grad=False` 가 되어 `_create_optimizer` 에서 자동 제외 — 불필요한
optimizer state 메모리를 쓰지 않는다.

스케줄러는 Linear Warmup → CosineAnnealingWarmRestarts 의 SequentialLR.

- `warmup_steps = 5` (epoch 단위), `start_factor = 0.1`: warmup 시작
  lr = $0.0005 \times 0.1 = 5 \times 10^{-5}$
- `cosine_t0 = 10`, `cosine_t_mult = 2`: 첫 주기 10 epoch, 이후 20 → 40
- Phase 2 전환 시 `warmup_steps = 2` 로 단축 후 스케줄러 재생성

왜 Phase 2 는 warm restart 가 아닌 plain cosine 인가. Phase 2 는 짧다
(8 epoch 기본). Warm restart 의 주기 구조가 다 돌기 전에 학습이 끝난다.
짧은 구간의 부드러운 감쇠가 더 유리하다.

AdamW 의 다른 하이퍼: `lr=5 \times 10^{-4}$, `weight_decay=0.01`,
`gradient_clip_norm=5.0`.

## 결정 4 — CGC-adaTT 동기화 계약

CGC 는 "각 태스크가 어떤 Shared Expert 에 attention 을 줄 것인가" 를
학습한다. adaTT 는 "태스크 간 gradient 가 어떻게 전이될 것인가" 를
제어한다. 두 메커니즘이 *같은 Shared Expert 파라미터에 대해* 작동하므로
동기화되지 않으면 상충된다.

*왜 동시에 freeze 해야 하는가.* adaTT 가 전이 가중치를 고정한 상황을
가정하자. CGC 가 계속 학습하면 CTR 의 Expert attention 이 바뀌고, 이는
Shared 로 흘러가는 CTR gradient 의 방향을 바꾼다. adaTT 가 "CTR→CVR
positive transfer" 라고 측정해서 고정해놓은 가중치는 이제 잘못된
관계를 반영한다. gate dynamic 과 transfer dynamic 이 *둘 다* 멈춰야
수렴이 깨끗하다.

동기화는 두 지점에서 일어난다.

- adaTT `freeze_epoch` 에서 `_cgc_frozen` 버퍼가 True 로 플립되며 CGC
  attention 의 `requires_grad=False`.
- Phase 2 시작 시 동일 처리. Shared 가 frozen 인데 CGC 만 계속 학습하면
  입력 (Expert 출력) 이 변하지 않으므로 gating 학습이 과적합으로 귀결
  된다.

`_cgc_frozen` 은 `register_buffer` 로 등록되어 체크포인트에 freeze 상태가
보존된다.

## 결정 5 — 메모리와 성능, 세 가지 핵심

adaTT 의 gradient 추출은 비싸다. 16 태스크 각각에 대해 Shared Expert
파라미터의 gradient 를 계산하므로, 최적화 없이는 학습 속도가 크게
떨어진다. 세 가지 결정이 이걸 관리 가능하게 만든다.

*`retain_graph=True` 의 비용.* 16 태스크에 대해 순차적으로 `autograd.grad`
를 호출하면서 graph 를 유지하므로, peak memory 가 forward pass 대비
약 2 배로 증가한다. 아키텍처상 제거 불가 — Trainer 의 `loss.backward()`
가 같은 graph 를 재사용해야 하기 때문이다. 16 태스크 × RTX 4070 12GB
기준 batch_size 16384 가 한계.

*`adatt_grad_interval = 10`.* 매 step gradient 를 추출하면 16 ×
`autograd.grad` 호출이 매 step 발생한다. 친화도는 EMA 로 평활화되므로
10 step 간격 측정으로도 충분히 안정적이다. 이 설정만으로 gradient 계산
오버헤드가 $1/10$ 로 감소한다. 과거에 warmup 중 매 step 추출하다 hang 이
발생해 추가된 설정이다.

*TF32 + cuDNN benchmark (not torch.compile).* `torch.compile` 은 이
프로젝트에서 비활성. 15 태스크 MTL + `retain_graph` + dynamic shape 조합
으로 커널 컴파일 수가 수백 개에 달해 첫 epoch 에 30 분 이상 소요된다.
대신 TF32 + cuDNN benchmark 로 10–15% 속도를 확보한다.

AMP (fp16) 는 기본 활성 — 메모리 약 40% 절감, 속도 약 20% 향상. 단
focal loss 계산은 float32 로 명시적 캐스팅 — fp16 에서 `focal_weight *
bce` 중간 결과가 subnormal 범위에 들어가면 NaN 이 발생할 수 있기 때문이다
(M-2/M-3 FIX).

## 결정 6 — Gradient Accumulation 과 NaN 방어

마지막으로 학습 안정성. Gradient clipping 은 `clip_grad_norm_=5.0` 으로
걸려 있고, `gradient_accumulation_steps=1` 이라 실질 배치는 batch_size
그대로. `math.isfinite(loss_val)` 체크로 NaN / Inf loss 발생 시 해당
배치를 스킵하고 `optimizer.zero_grad()` 로 오염된 gradient 를 초기화
한다. OOM 은 `trainer.py` 의 예외 핸들러에서 배치 스킵 처리.

---

여기까지가 adaTT 를 실제 파이프라인에 꽂는 여섯 결정이다. Trainer 의
2-Phase 와 adaTT 내부 3-Phase 는 서로 직교하는 두 스케줄이고,
Uncertainty Weighting 은 adaTT 의 transfer 이전에 적용되며, per-expert
lr 과 SequentialLR 이 학습률을 Expert 별로 분배하고, CGC-adaTT 동기화
freeze 가 수렴을 정리하며, `grad_interval=10` 과 TF32 가 성능을 유지하고,
NaN 방어가 안정성을 지탱한다. 세부 설정값의 전체 목록, 디버깅 guide,
수학 증명 (EMA 수렴, Bayesian conjugacy, PCGrad 비교 등) 은 아래 PDF 에
담았다.

## 전체 adaTT 기술 참조서 다운로드

ADATT-1 부터 ADATT-4 까지 `기술참조서/adaTT_기술_참조서` 를 블로그
형식으로 관통했다. 동기, 수학적 기초, 친화도 측정, Transfer Loss, Group
Prior, 3-Phase Schedule, Negative Transfer 차단, 학습 루프, Loss
Weighting, Optimizer, CGC 동기화까지 — 원본 PDF 는 조판 · 색인 · 수식
증명이 모두 살아있는 긴 참조 문서다. 블로그에서 덜어낸 디버깅 guide,
설정 파라미터 총람, 부록의 수학 증명 (A.1 EMA 수렴, A.2 Group Prior 의
Bayesian 해석, A.3 Softmax temperature, A.4 Negative Transfer 차단 근거,
A.5 Transfer-Enhanced Loss 의 수렴 영향) 은 모두 PDF 에서 확인할 수 있다.

> **📄 [adaTT 기술 참조서 전체 PDF 다운로드](/adaTT_기술_참조서.pdf)** · KO
>
> Adaptive Task Transfer · Gradient Cosine Similarity · Transfer Loss ·
> 3-Phase Schedule · Negative Transfer Detection — 본 프로젝트의 adaTT
> 전체 내용을 하나의 문서로 보고 싶으면 위 링크에서 받으면 된다.

## adaTT 서브스레드 종료, 다음은 이종 Shared Expert

여기까지가 adaTT 서브스레드의 끝이다. 각 편이 이전 편의 결정이 남긴
문제에서 출발한 사슬로 읽힌다.

- **ADATT-1**: 피처 경로의 CGC 가 다 못 푼 것 — gradient 충돌. 적응형
  타워가 왜 필요한가.
- **ADATT-2**: 측정의 네 결정 — gradient, cosine, EMA,
  `torch.compiler.disable`. `TaskAffinityComputer` 엔진 완성.
- **ADATT-3**: 측정된 친화도를 어떻게 쓸 것인가 — Transfer Loss, Group
  Prior, 3-Phase Schedule, Negative Transfer 차단.
- **ADATT-4**: 설계를 실제 학습 루프에 꽂는 여섯 결정 — 2-Phase training,
  Uncertainty Weighting 순서, per-expert lr, CGC-adaTT 동기화, 메모리 /
  성능, NaN 방어.

PLE 6 편과 adaTT 4 편, 총 10 편의 Study Thread 로 본 프로젝트의 MTL
백본을 블로그 형식으로 정리했다. PLE 는 *피처 경로* 에서 태스크 간 충돌을
분리하고, adaTT 는 *gradient 경로* 에서 남은 충돌을 측정해 협력으로
돌렸다 — 두 서브스레드가 같은 MTL 문제의 두 얼굴을 맡은 셈이다.

> **열린 실험 결과 — adaTT 제거를 검토 중.** ADATT-1 에서 미리 밝힌
> 대로, 합성 데이터 벤치마크에서 PLE+adaTT 와 PLE-only 사이에 뚜렷한
> 성능 차이가 관찰되지 않았다. 현재 실 데이터(카드 거래 로그) 에서 같은
> 비교를 진행 중이며, 결과가 재현되면 adaTT 를 스택에서 *제거* 할 생각이다.
> 그렇게 되더라도 이 4 편은 그대로 남겨둔다 — "왜 이 설계를 시도했고,
> 어떤 근거로 뺐는가" 의 기록이 다음 사람에게는 "다시 해보지 말 것" 의
> 안내가 된다. (업데이트: 실 데이터 실험 결과는 별도 포스트에서 공유
> 예정.)

다음부터는 7 개 이종 Shared Expert 각자의 수학적 기초로 넘어간다. 우선
CausalOT (인과 추론 + 최적 수송), TDA (위상 데이터 분석 / PersLay),
Temporal (Mamba + LNN + Transformer), Economics 피처 기반 Expert 순서로
각각 서브스레드를 열 예정이다.
