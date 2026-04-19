---
title: "[Study Thread] ADATT-4 — 학습 루프·Loss Weighting·Optimizer·CGC 동기화"
date: 2026-04-20 15:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, training-loop, loss-weighting, optimizer, specs]
lang: ko
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
출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서` 이다. 이번 4편은
adaTT 가 실제 학습 파이프라인에 어떻게 맞물리는지 — 2-Phase Training
Loop, Loss Weighting 전략(Uncertainty · GradNorm · DWA), Optimizer·Scheduler
설정, CGC-adaTT 동기화, 메모리·성능 최적화, 디버깅 가이드, 설정
매개변수 총람, 부록까지 — 그리고 이 글 하단에서 전체 PDF 를 다운로드
받을 수 있다.*

## 2-Phase Training Loop

Trainer는 전체 학습을 두 개의 Phase로 구분한다. adaTT는 *Phase 1에서만
활성화*되며, Phase 2에서는 비활성화된다. 소스: `trainer.py:456-542` —
`train()` 메서드, `trainer.py:580-654` — `_train_phase()`.

### Phase 1 — Shared Expert Pretrain

- *기간*: `shared_expert_epochs` (기본 15, `model_config.yaml:793`)
- *학습 대상*: 전체 모델 — Shared Experts, CGC, Task Experts, Task Towers
- *adaTT*: 활성 상태 — gradient 추출 및 transfer loss 적용

```python
# trainer.py:483-488
logger.info("=== Phase 1: Shared Expert 학습 ===")
self._train_phase(
    train_loader, val_loader,
    max_epochs=self.config.shared_expert_epochs,
    phase_name="phase1",
)
```

Phase 1이 끝나면 adaTT는 충분한 친화도 데이터를 축적한 상태이다.
`freeze_epoch`이 Phase 1 내에 있으면 Phase 1 후반부에서 이미 가중치가
고정된다.

### Phase 2 — Cluster Finetune

- *기간*: `cluster_finetune_epochs` (기본 8, `model_config.yaml:794`)
- *학습 대상*: 클러스터별 Task Expert 서브헤드만
- *adaTT*: *비활성화* — Shared Expert가 frozen이므로 gradient 추출 무의미

```python
# trainer.py:493-496
_adatt_backup = self.model.adatt
if self.model.adatt is not None:
    self.model.adatt = None
    logger.info("adaTT 비활성화: Phase 2 (Shared frozen으로 친화도 무효)")
```

> **왜 Phase 2에서 adaTT를 비활성화하는가.** adaTT의 gradient는
> *Shared Expert 파라미터*에 대해 계산된다 (`ple_cluster_adatt.py:1872`).
> Phase 2에서 Shared Expert가 frozen이면 gradient가 0이 되므로 코사인
> 유사도 계산이 무의미하다. 또한 frozen 파라미터에 대한 `autograd.grad`
> 호출은 불필요한 계산 오버헤드이다.

### Phase 전환 시 리셋

`_setup_phase2()` (`trainer.py:544-578`)에서 다음 항목이 리셋된다.

| 리셋 항목 | 이유 |
|---|---|
| Optimizer | Shared Expert frozen → 모멘텀 초기화 필요 (stale momentum 방지) |
| Scheduler | Phase 2 전용 warmup (2 에포크, Phase 1의 5 에포크보다 짧음) |
| GradScaler | AMP 스케일러 상태 초기화 (Phase 전환 시 loss 스케일 변화) |
| Early stopping | best\_val\_loss, patience\_counter 모두 초기화 |
| CGC Attention | Shared Expert frozen → CGC gating도 함께 freeze |

```python
# trainer.py:544-578
def _setup_phase2(self):
    if self.config.freeze_shared_in_phase2:
        self.model.freeze_shared_experts()
    # Optimizer 리셋
    self.optimizer = self._create_optimizer()
    # Scheduler 리셋 (Phase 2 전용 warmup)
    self.config.warmup_steps = self.config.phase2_warmup_steps  # 2 에포크
    self.scheduler = self._create_scheduler()
    # Early stopping 리셋
    self.best_val_loss = float("inf")
    self.patience_counter = 0
```

### adaTT 복원 보장

Phase 2 종료 후 adaTT는 *반드시 복원*된다 (체크포인트/추론 호환성):

```python
# trainer.py:504-508
finally:
    self.model.adatt = _adatt_backup  # 예외 발생 시에도 복원
    if self.config.freeze_shared_in_phase2:
        self.model.unfreeze_shared_experts()
```

`finally` 블록으로 감싸 예외 발생 시에도 모델 상태가 일관되게 유지된다.

## Loss Weighting 전략

adaTT의 transfer loss는 기존 loss weighting 전략 위에 *추가적으로*
동작한다. 즉, 태스크별 loss weight와 uncertainty weighting이 먼저
적용되고, 그 결과에 adaTT의 transfer loss가 가산된다. 소스:
`ple_cluster_adatt.py:1607-1845` — `_compute_task_losses()`.

### Loss 계산 파이프라인

1. *태스크별 loss 유형 결정*: focal, huber, MSE, NLL, contrastive 등
   (`ple_cluster_adatt.py:1656`)
2. *Focal Loss alpha/gamma 적용*: 태스크별 차별화된 양성 클래스 가중치
   (`ple_cluster_adatt.py:1768-1780`)
3. *Loss weight 적용*: 태스크별 고정 가중치 또는 Uncertainty Weighting
   (`ple_cluster_adatt.py:1818-1830`)
4. *Evidential loss 가산*: 불확실성 추정 보조 손실
   (`ple_cluster_adatt.py:1832-1841`)
5. *adaTT transfer loss*: gradient 기반 전이 손실 추가
   (`ple_cluster_adatt.py:1310-1316`)
6. *CGC entropy regularization*: Expert collapse 방지
   (`ple_cluster_adatt.py:1321-1329`)

### Uncertainty Weighting (Kendall et al., 2018)

> **역사적 배경 — Uncertainty Weighting의 탄생.** Kendall, Gal & Cipolla
> (CVPR 2018, *"Multi-Task Learning Using Uncertainty to Weigh Losses
> for Scene Understanding"*)는 컴퓨터 비전에서 세만틱 분할 + 깊이
> 추정 + 인스턴스 분할을 동시 학습하는 문제에서, 태스크별 가중치를
> 수동 튜닝하는 비용을 *homoscedastic uncertainty*로 자동화하는 방법을
> 제안했다. 이 아이디어의 핵심은 가우시안 likelihood
> $p(y | f(x), \sigma) = \mathcal{N}(f(x), \sigma^2)$ 에서 $-\log p$ 를
> 취하면 자연스럽게
> $\frac{1}{2\sigma^2} \cdot \|y - f(x)\|^2 + \log \sigma$ 형태가 되어,
> *loss가 큰 태스크의 $\sigma$ 가 커지고 가중치가 줄어드는* 자기 조절
> 메커니즘이 만들어진다는 것이다.

> **학부 수학 — 왜 $1/(2\sigma^2)$ 형태가 나오는가.** 관측값 $y$ 가 예측
> $\hat{y}$ 를 중심으로 한 정규분포 $\mathcal{N}(\hat{y}, \sigma^2)$ 를
> 따른다고 가정하자. 확률밀도함수는
> $p(y) = \frac{1}{\sigma \sqrt{2\pi}} \exp(-(y-\hat{y})^2 / (2\sigma^2))$
> 이다. 음의 로그를 취하면
> $-\log p(y) = (y-\hat{y})^2 / (2\sigma^2) + \log \sigma + \text{const}$
> 가 된다. 여기서 $(y-\hat{y})^2$ 이 loss $\mathcal{L}$ 에 해당하므로,
> $\mathcal{L}^{weighted} = \mathcal{L} / (2\sigma^2) + \log \sigma$ 가
> 자연스럽게 도출된다. $\sigma^2 = \exp(\text{log\_var})$ 로
> 재파라미터화하면 $\log \sigma = \frac{1}{2} \cdot \text{log\_var}$
> 이므로 코드의 구현과 일치한다. *정밀도* $1/\sigma^2$ 가 곧 loss의
> 가중치가 되며, $\sigma$ 가 무한히 커져 loss를 0으로 만드는 것을
> $\log \sigma$ 정규화 항이 방지한다.

$$\mathcal{L}_i^{weighted} = \frac{1}{2 \sigma_i^2} \cdot \mathcal{L}_i + \frac{1}{2} \log \sigma_i^2$$

- $\sigma_i^2 = \exp(\text{log\_var}_i)$: 태스크 $i$ 의 학습 가능한 불확실성
- `log_var` clamp: \[-4.0, 4.0\], precision clamp: \[0.001, 100.0\]
  (`ple_cluster_adatt.py:1822-1823`)

> **수식 직관.** 이 수식은 "불확실성이 높은 태스크의 loss는 낮은 가중치로,
> 불확실성이 낮은 태스크의 loss는 높은 가중치로 반영한다"는 것을 말한다.
> $1 / (2 \sigma_i^2)$ 는 정밀도(precision)로, $\sigma_i^2$ 가 크면
> 가중치가 작아진다. $\frac{1}{2} \log \sigma_i^2$ 항은 $\sigma_i^2$ 를
> 무한히 키워 loss를 0으로 만드는 치팅을 방지하는 정규화 페널티이다.
> 직관적으로, "잘 모르는 태스크의 실수에는 관대하되, 잘 아는 태스크의
> 실수에는 엄격하게" 학습하는 전략이다.

이 가중치는 adaTT *이전에* 적용된다. 즉, adaTT의 `task_losses` 입력에는
이미 uncertainty weighting이 반영된 값이 들어온다.

### 태스크별 Loss Weight 현황

`model_config.yaml`에서 정의된 태스크별 고정 가중치:

| 태스크 | weight | loss type | 비고 |
|---|---|---|---|
| ctr | 1.0 | focal ($\gamma$=2, $\alpha$=0.25) | 표준 |
| cvr | 1.5 | focal ($\gamma$=2, $\alpha$=0.20) | 양성 비율 극소 → weight 상향 |
| churn | 1.2 | focal ($\gamma$=2, $\alpha$=0.60) | FN 비용 높음 → alpha 상향 |
| retention | 1.0 | focal ($\gamma$=2, $\alpha$=0.20) | 양성 비율 높음 |
| nba | 2.0 | CE | 12 classes, 비즈니스 핵심 |
| ltv | 1.5 | huber ($\delta$=1.0) | regression, 이상치 대응 |
| brand\_prediction | 2.0 | contrastive | InfoNCE, 50K 브랜드 |
| spending\_category | 1.2 | CE | 12 categories |
| 나머지 | 0.8--1.0 | 다양 | 태스크별 상이 |

### adaTT와 Loss Weight의 상호작용

adaTT의 `compute_transfer_loss`는 각 태스크의 loss weight가 *이미 적용된*
후의 `task_losses`를 입력으로 받는다. 따라서 높은 loss weight를 가진
태스크(nba: 2.0)가 다른 태스크에 더 큰 전이 효과를 미친다.

이는 *의도된 동작*이다: 비즈니스적으로 중요한 태스크의 학습 시그널이
다른 태스크에도 전파되어야 한다.

## Optimizer 및 Scheduler 설정

소스: `trainer.py:220-334` — `_create_optimizer()`, `_create_scheduler()`.

### AdamW Optimizer

```python
# trainer.py:236-242
trainable_params = [p for p in self.model.parameters() if p.requires_grad]
return torch.optim.AdamW(
    trainable_params,
    lr=self.config.learning_rate,      # 0.0005
    weight_decay=self.config.weight_decay,  # 0.01
)
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `learning_rate` | 0.0005 | 전역 학습률 (`model_config.yaml:752`) |
| `weight_decay` | 0.01 | L2 정규화 강도 |
| `gradient_clip_norm` | 5.0 | Gradient 크기 제한 (`model_config.yaml:780`) |
| `gradient_accumulation_steps` | 1 | 실질 배치 = 16384 $\times$ 1 |

### Per-Expert Learning Rate

Shared Expert마다 다른 학습률을 설정할 수 있다:

```python
# trainer.py:249-261
for expert_name, expert_module in self.model.shared_experts.items():
    expert_params = [p for p in expert_module.parameters() if p.requires_grad]
    cfg = expert_lr_config.get(expert_name, {})
    lr = cfg.get("lr", self.config.learning_rate)
    wd = cfg.get("weight_decay", self.config.weight_decay)
    param_groups.append({"params": expert_params, "lr": lr, "weight_decay": wd})
```

이 기능은 `model_config.yaml:756-770`에 주석 처리된 예시로 제공되며,
하이퍼볼릭 공간에서 학습하는 `unified_hgcn`은 보수적인 lr이 필요하고,
`deepfm`은 상대적으로 높은 lr로 빠르게 수렴할 수 있다.

> **Phase 2에서의 자동 제외.** Phase 2에서 Shared Expert가 frozen되면
> `requires_grad=False`가 되므로, `_create_optimizer`에서 해당 파라미터가
> 자동으로 제외된다 (`trainer.py:250`). 이는 불필요한 optimizer state
> 메모리 할당을 방지한다.

### Learning Rate Scheduler — SequentialLR

Linear Warmup → CosineAnnealingWarmRestarts:

```python
# trainer.py:296-318
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    self.optimizer, start_factor=0.1, total_iters=warmup_steps  # 5 에포크
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    self.optimizer, T_0=self.config.cosine_t0,    # 10
    T_mult=self.config.cosine_t_mult,              # 2
)
return torch.optim.lr_scheduler.SequentialLR(
    self.optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],  # 5 에포크 후 cosine으로 전환
)
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `warmup_steps` | 5 | Linear warmup 기간 (에포크 단위) |
| `cosine_t0` | 10 | 첫 cosine 주기 길이 (`model_config.yaml:773`) |
| `cosine_t_mult` | 2 | 주기 배수 (10 → 20 → 40 에포크) |
| `start_factor` | 0.1 | Warmup 시작 LR = 0.0005 $\times$ 0.1 = 0.00005 |

### Phase 2 전용 Scheduler

Phase 2 시작 시 scheduler가 리셋되며, warmup은 2 에포크로 짧아진다:

```python
# trainer.py:562-566
original_warmup = self.config.warmup_steps
self.config.warmup_steps = self.config.phase2_warmup_steps  # 2
self.scheduler = self._create_scheduler()
self.config.warmup_steps = original_warmup  # 복원
```

## CGC-adaTT 동기화

CGC (Customized Gate Control)와 adaTT는 서로 다른 메커니즘이지만, 동일한
Shared Expert 파라미터에 대해 작동하므로 *동기화*가 필수적이다. 소스:
`ple_cluster_adatt.py:1921-1942` — `on_epoch_end()` CGC freeze 동기화.

### CGC의 역할

CGC는 각 태스크가 어떤 Shared Expert에 더 attention을 줄지 학습한다.
adaTT가 태스크 간 *지식 전이*를 제어한다면, CGC는 *Expert 선택*을
제어한다. 두 메커니즘이 독립적으로 학습하면 상충하는 방향으로 파라미터를
업데이트할 수 있다.

### 동기화 전략 — 동시 Freeze

adaTT의 `freeze_epoch`에서 CGC Attention도 함께 frozen된다:

```python
# ple_cluster_adatt.py:1931-1942
# v2.3: CGC freeze -- adaTT freeze_epoch과 동기화
freeze_epoch = self.config.get("adatt", {}).get("freeze_epoch")
if (freeze_epoch is not None
        and epoch >= freeze_epoch
        and self.task_expert_attention is not None
        and not self._cgc_frozen.item()):
    for param in self.task_expert_attention.parameters():
        param.requires_grad = False
    self._cgc_frozen.fill_(True)
    logger.info(f"CGC Attention frozen at epoch {epoch}")
```

> **왜 동시에 freeze하는가.** adaTT가 전이 가중치를 고정했는데 CGC가
> 계속 학습하면, CGC가 Expert 가중치를 변경하여 adaTT가 측정한 친화도
> 관계가 무효화된다. 예를 들어 adaTT가 "CTR→CVR positive transfer"라고
> 판단했는데, CGC가 CTR의 Expert 선택을 변경하면 gradient 방향이 달라져
> 고정된 전이 가중치가 더 이상 유효하지 않게 된다.

### Phase 2에서의 CGC Freeze

Phase 2 시작 시에도 CGC가 freeze된다 (`trainer.py:549-555`):

```python
# trainer.py:549-555
if hasattr(self.model, '_cgc_frozen') and not self.model._cgc_frozen.item():
    if self.model.task_expert_attention is not None:
        for param in self.model.task_expert_attention.parameters():
            param.requires_grad = False
        self.model._cgc_frozen.fill_(True)
        logger.info("CGC Attention frozen in Phase 2")
```

Shared Expert가 frozen인 Phase 2에서 CGC gating을 학습하는 것은 무의미하다:
입력(Expert 출력)이 변하지 않으므로 gating 학습이 과적합을 유발한다.

### \_cgc\_frozen 상태 추적

```python
# ple_cluster_adatt.py:376
self.register_buffer("_cgc_frozen", torch.tensor(False))
```

`register_buffer`로 등록하여 체크포인트 저장/복원 시 freeze 상태가
유지된다. MLflow에서도 freeze 이벤트를 로깅한다 (`trainer.py:604-608`).

## 메모리 및 성능 최적화

adaTT의 gradient 추출은 계산 비용이 높다. 16개 태스크 각각에 대해 Shared
Expert 파라미터의 gradient를 계산하므로, 최적화 없이는 학습 속도가 크게
저하된다.

### retain\_graph=True의 메모리 영향

```python
# ple_cluster_adatt.py:1865-1868
# G-04 NOTE: retain_graph=True는 아키텍처상 필수
# 메모리 영향: n_tasks x shared_param_size.
# 16개 태스크 기준 forward pass 대비 약 2x peak memory.
```

`retain_graph=True`는 각 태스크의 `autograd.grad` 호출 후에도 computation
graph를 메모리에 유지한다. 16개 태스크에 대해 순차적으로 호출하므로, peak
memory는 forward pass 대비 약 *2배*로 증가한다.

| 요소 | 메모리 | 비고 |
|---|---|---|
| Forward pass graph | 1x (기준) | 일반적인 학습에서도 필요 |
| retain\_graph 추가 | ~1x | graph가 해제되지 않아 추가 메모리 |
| 16 task gradients | ~0.3x | 각 gradient는 shared\_param\_size |
| **합계** | **~2.3x** | RTX 4070 12GB에서 batch\_size 16384 가능 |

### adatt\_grad\_interval

gradient 추출 빈도를 줄여 계산 비용을 낮추는 핵심 최적화:

```python
# ple_cluster_adatt.py:373
self.adatt_grad_interval = adatt_config.get("grad_interval", 10)

# ple_cluster_adatt.py:1299-1304
if self.training and self.adatt is not None:
    if self.global_step % self.adatt_grad_interval == 0:
        task_gradients = self._extract_task_gradients(task_losses)
```

> **기본값 10의 근거.** 매 step마다 gradient를 추출하면
> 16 $\times$ `autograd.grad` 호출이 발생한다. 친화도는 EMA로
> 평활화되므로, 10 step 간격으로 측정해도 충분히 안정적이다.
> `grad_interval=10`으로 설정하면 gradient 계산 오버헤드가 *1/10*로
> 감소한다. 기존에는 warmup 중 매 step 추출하여 hang이 발생했었다
> (`ple_cluster_adatt.py:1300-1301`).

### torch.compiler.disable

```python
# ple_cluster_adatt.py:1847
@torch.compiler.disable
def _extract_task_gradients(self, task_losses):
```

현재 프로젝트에서 `torch.compile`은 비활성화되어 있다
(`trainer.py:174-177`). 15-태스크 MTL + adaTT `retain_graph` + dynamic
shape 조합으로 커널 컴파일 수가 수백 개에 달해 첫 epoch에 *30분 이상*
소요되기 때문이다. 대신 TF32 + cuDNN benchmark로 10~15% 속도를 확보한다:

```python
# trainer.py:170-172
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### AMP (Automatic Mixed Precision)

Forward pass는 `torch.amp.autocast` 하에서 fp16으로 실행된다:

```python
# trainer.py:709-711
with autocast(device_type=self.device.type):
    outputs = self.model(inputs, compute_loss=True)
    loss = outputs.total_loss / self.config.gradient_accumulation_steps
```

adaTT의 gradient 추출은 autocast 내에서 이루어지지만, focal loss 계산은
float32로 명시적으로 캐스팅한다 (`ple_cluster_adatt.py:1774`):

```python
# ple_cluster_adatt.py:1774
p_f = pred.squeeze().float().clamp(1e-7, 1 - 1e-7)
```

> **⚠ M-2 + M-3 FIX — fp16 focal loss 안전성.** fp16에서
> `log(1e-7) = -16.1`은 정상이지만, `focal_weight * bce`의 중간 결과가
> subnormal 범위에 들어가면 NaN이 발생할 수 있다. 전체 focal loss 계산을
> float32로 수행하여 이 문제를 방지한다.

### Gradient Accumulation

```python
# trainer.py:723-730
if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config.gradient_clip_norm  # 5.0
    )
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
```

현재 `gradient_accumulation_steps=1`이므로 매 배치마다 업데이트된다.
실질 배치 크기 = 16384 $\times$ 1 = 16384.

## 디버깅 가이드

### 일반적인 실패 모드

| 증상 | 원인 | 해결 |
|---|---|---|
| NaN loss 발생 | fp16 focal loss underflow | M-2/M-3: float32 캐스팅 확인 |
| 학습 초기 loss 발산 | transfer loss가 원본 지배 | G-01: `max_transfer_ratio` 확인 (0.5) |
| Phase 2에서 RuntimeError | adaTT 비활성화 누락 | trainer.py에서 `model.adatt = None` 확인 |
| `ValueError` 초기화 실패 | `freeze_epoch <= warmup_epochs` | H-6: config 검증 |
| 학습 hang (무응답) | 매 step gradient 추출 | `adatt_grad_interval` 설정 확인 (기본 10) |
| 체크포인트 불일치 | `fill_()` 미사용 | buffer 업데이트 시 in-place 연산 확인 |

### 친화도 행렬 해석

학습 중 친화도 행렬을 확인하는 방법:

```python
# 친화도 행렬 조회
affinity = model.adatt.affinity_computer.get_affinity_matrix()
print(affinity)  # [n_tasks, n_tasks]

# Negative transfer 감지
neg_pairs = model.adatt.detect_negative_transfer()
print(neg_pairs)  # {"churn": ["ctr"], ...}

# 현재 전이 가중치 행렬
transfer_w = model.adatt.get_transfer_matrix()
print(transfer_w)  # softmax 정규화된 [n_tasks, n_tasks]
```

건강한 친화도 행렬의 특성:

- 같은 그룹 내 태스크 간 양의 친화도 ($> 0.3$)
- 다른 그룹 간 약한 양 또는 중립 ($-0.1 \sim 0.3$)
- 대각선 원소 = 1.0
- *전체 행렬이 -1 또는 +1로 포화되지 않음* (포화 시 EMA 감쇠율 조정
  필요)

### Transfer Loss 모니터링

MLflow에서 로깅되는 핵심 메트릭:

```python
# trainer.py:810-811 — step 로깅 시 outputs.task_losses 포함
self._log_step(outputs, phase_name)
```

모니터링 체크리스트:

- `task_losses/<task>`: 태스크별 enhanced loss (transfer loss 포함)
- `adatt_freeze_epoch`: freeze 시점 로깅
- `cgc_frozen_epoch`: CGC freeze 시점 로깅
- 총 loss = $\sum$ enhanced losses + CGC entropy loss + SAE loss +
  evidential loss

### Phase 전환 디버깅

Phase 전환 시 확인 사항:

1. *Phase 1 → Freeze*: `adaTT: 전이 가중치 고정 (epoch N)` 로그 확인
2. *Phase 1 → Phase 2*: `adaTT 비활성화: Phase 2` 로그 확인
3. *CGC Freeze*: `CGC Attention frozen at epoch N` 로그 확인
4. *Optimizer 리셋*: `Optimizer 리셋: Phase 2 시작` 로그 확인

```python
# Phase 전환 로그 예시
# INFO: adaTT: 전이 가중치 고정 (epoch 28)
# INFO: CGC Attention frozen at epoch 28 (synced with adaTT freeze_epoch)
# INFO: === Phase 2: Cluster Head Fine-tuning ===
# INFO: Shared Experts 동결
# INFO: adaTT 비활성화: Phase 2 (Shared frozen으로 친화도 무효)
# INFO: Optimizer 리셋: Phase 2 시작
# INFO: 스케줄러 리셋: Phase 2 시작 (warmup=2 에포크)
```

### Loss 폭발 방지

Loss가 급격히 증가하는 경우의 진단 순서:

1. *NaN 확인*: `_train_epoch`에서 `math.isfinite(loss_val)` 체크
   (`trainer.py:781-786`)
2. *Transfer loss 비율 확인*: `max_transfer_ratio=0.5` 초과 여부
3. *Gradient norm 확인*: `gradient_clip_norm=5.0` 적용 여부
4. *VRAM OOM 확인*: `trainer.py:762-775`에서 OOM 발생 시 배치 스킵 처리

```python
# trainer.py:781-786
loss_val = outputs.total_loss.item()
if not math.isfinite(loss_val):
    logger.error(f"NaN/Inf loss at batch {batch_idx}!")
    self.optimizer.zero_grad()  # gradient 오염 방지
    continue
```

## 설정 매개변수 총람

### adaTT 핵심 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---|---|---|---|
| `enabled` | `true` | bool | adaTT 활성화 여부 |
| `transfer_lambda` | 0.1 | \[0, 1\] | 전이 손실 가중치 $\lambda$ |
| `temperature` | 1.0 | (0, $\infty$) | Softmax temperature $T$ |
| `warmup_epochs` | 0 (test) / 10 (prod) | \[0, max\_epochs) | Phase 1 기간 |
| `freeze_epoch` | 1 (test) / 28 (prod) | (warmup, max\_epochs) | Phase 3 시작 |
| `negative_transfer_threshold` | -0.1 | \[-1, 0\] | Negative transfer 차단 임계값 |
| `ema_decay` | 0.9 | \[0, 1\] | 친화도 EMA 감쇠 계수 |
| `prior_blend_start` | 0.5 | \[0, 1\] | 학습 초기 group prior 비율 |
| `prior_blend_end` | 0.1 | \[0, 1\] | 학습 후반 group prior 비율 |
| `transfer_strength` | 0.5 | \[0, 1\] | 로짓 전이 강도 (CTR→CVR 등) |

소스: `model_config.yaml:593-628`.

### 태스크 그룹 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---|---|---|---|
| `task_groups.<group>.members` | -- | list | 그룹 소속 태스크 이름 리스트 |
| `task_groups.<group>.intra_strength` | 0.5 | \[0, 1\] | 그룹 내 전이 강도 |
| `inter_group_strength` | 0.3 | \[0, 1\] | 그룹 간 전이 강도 |

### Training 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---|---|---|---|
| `batch_size` | 16384 | \[1024, 65536\] | 학습 배치 크기 |
| `learning_rate` | 0.0005 | \[1e-5, 1e-2\] | 전역 학습률 |
| `weight_decay` | 0.01 | \[0, 0.1\] | L2 정규화 강도 |
| `shared_expert_epochs` | 15 | \[5, 100\] | Phase 1 기간 |
| `cluster_finetune_epochs` | 8 | \[3, 50\] | Phase 2 기간 |
| `freeze_shared_in_phase2` | `true` | bool | Phase 2에서 Shared Expert 동결 |
| `early_stopping_patience` | 7 | \[1, 20\] | Early stopping patience |
| `gradient_clip_norm` | 5.0 | \[0.1, 20\] | Gradient clipping 임계값 |
| `use_amp` | `true` | bool | Mixed Precision (fp16) 사용 |
| `cosine_t0` | 10 | \[5, 50\] | CosineAnnealing 첫 주기 (에포크) |
| `cosine_t_mult` | 2 | \[1, 4\] | CosineAnnealing 주기 배수 |
| `warmup_steps` | 5 | \[0, 20\] | LR warmup 에포크 수 |
| `phase2_warmup_steps` | 2 | \[0, 10\] | Phase 2 LR warmup 에포크 수 |

소스: `model_config.yaml:750-805`, `trainer.py:50-96`.

### 성능 최적화 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `adatt_grad_interval` | 10 | gradient 추출 간격 (step 단위). 작을수록 정확, 클수록 빠름 |
| `gradient_accumulation_steps` | 1 | gradient 누적 횟수. 실질 배치 = batch\_size $\times$ 이 값 |
| `gradient_checkpointing` | `false` | 9.25M 모델에 불필요 (끄면 10~20% 속도 향상) |
| `use_amp` | `true` | fp16 Mixed Precision. 메모리 ~40% 절감, 속도 ~20% 향상 |

### adaTT 내부 상수

코드에 하드코딩된 상수로, config에서 변경할 수 없다:

| 상수 | 값 | 위치 및 설명 |
|---|---|---|
| `max_transfer_ratio` | 0.5 | `adatt.py:191` — transfer loss의 원본 대비 최대 비율 |
| `norm clamp min` | 1e-8 | `adatt.py:123` — gradient norm 0-division 방지 |
| `diag_mask` | eye(n) | `adatt.py:239-242` — 자기 전이 제외용 마스크 |
| `affinity_matrix 초기값` | eye(n) | `adatt.py:79` — 대각선 1, 나머지 0 (중립 시작) |

## 부록 — 수학 증명 및 이론적 근거

### A.1 EMA 친화도의 수렴 특성

EMA 업데이트 규칙:

$$\mathbf{A}_t = \alpha \mathbf{A}_{t-1} + (1 - \alpha) \mathbf{C}_t$$

여기서 $\mathbf{C}_t$ 는 시점 $t$ 의 관측된 코사인 유사도 행렬이다.

> **수식 직관.** 이 수식은 "과거의 기억($\mathbf{A}_{t-1}$)에 새
> 관측($\mathbf{C}_t$)을 조금씩 섞어 넣는다"는 것을 말한다.
> $\alpha = 0.9$ 이면 매번 기존 값의 90%를 유지하고 새 값의 10%만
> 반영한다. 직관적으로, 하루하루의 주가 변동이 아닌 이동 평균 추세선을
> 보는 것과 같다 — 노이즈를 걸러내고 진짜 경향을 포착한다.

> **학부 수학 — 등비급수와 EMA 수렴.** 아래 증명에서 핵심이 되는 수학은
> *등비급수의 합*이다. 고등학교에서 배운 공식:
> $\sum_{k=0}^{n-1} \alpha^k = (1 - \alpha^n) / (1 - \alpha)$ ($|\alpha| < 1$).
> $n \to \infty$ 이면 $\alpha^n \to 0$ 이므로
> $\sum_{k=0}^{\infty} \alpha^k = 1/(1-\alpha)$ 이다. $\alpha = 0.9$ 를
> 대입하면 $1/(1-0.9) = 10$ 이다. 증명에서
> $(1-\alpha) \sum_{k=0}^{t-1} \alpha^k$ 의 합은
> $(1-\alpha) \cdot (1-\alpha^t)/(1-\alpha) = 1 - \alpha^t$ 이 된다.
> $t \to \infty$ 이면 $\alpha^t \to 0$ 이므로 가중치의 합이 1로 수렴하고,
> EMA가 정확히 true mean $\mathbf{C}^*$ 에 수렴한다.

**명제 A.1**: $\mathbf{C}_t$ 가 정상 (stationary) 분포를 따르고
$\mathbb{E}[\mathbf{C}_t] = \mathbf{C}^*$ 이면,

$$\mathbb{E}[\mathbf{A}_t] \to \mathbf{C}^* \quad \text{as } t \to \infty$$

**증명**: $\mathbf{A}_t$ 를 전개하면

$$\mathbf{A}_t = (1 - \alpha) \sum_{k=0}^{t-1} \alpha^k \mathbf{C}_{t-k} + \alpha^t \mathbf{A}_0$$

$t \to \infty$ 에서 $\alpha^t \mathbf{A}_0 \to 0$ 이고,

$$\mathbb{E}[\mathbf{A}_t] = (1 - \alpha) \sum_{k=0}^{t-1} \alpha^k \mathbb{E}[\mathbf{C}_{t-k}] = (1 - \alpha) \cdot \mathbf{C}^* \cdot \frac{1 - \alpha^t}{1 - \alpha} = \mathbf{C}^* (1 - \alpha^t) \to \mathbf{C}^*$$

따라서 EMA 친화도는 *true affinity*에 수렴한다. $\square$

**분산**:
$\text{Var}(\mathbf{A}_t) \approx \frac{1 - \alpha}{1 + \alpha} \text{Var}(\mathbf{C}_t)$.
$\alpha = 0.9$ 이면 분산이 원래의 $\approx 5.3\%$ 로 감소하여 높은
안정성을 제공한다.

### A.2 Group Prior의 Bayesian 해석

adaTT의 Group Prior는 *Bayesian 추론*의 prior distribution으로 해석할 수
있다.

> **역사적 배경 — Conjugate Prior의 역사.** Raiffa & Schlaifer (1961,
> *"Applied Statistical Decision Theory"*)가 *conjugate family* 개념을
> 체계화했으며, prior와 posterior가 같은 분포 족에 속하면 posterior를
> 해석적(closed-form)으로 구할 수 있다는 이점이 있다. 머신러닝에서는
> 가우시안 프로세스 (Rasmussen & Williams, 2006), Bayesian Linear
> Regression, Kalman Filter 등이 모두 이 conjugate 구조를 활용한다.

> **학부 수학 — Normal-Normal Conjugacy 유도.** Prior:
> $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$. 관측:
> $x | \theta \sim \mathcal{N}(\theta, \sigma^2)$. Posterior는
> $\theta | x \sim \mathcal{N}(\mu_n, \sigma_n^2)$ 이며,
> $\mu_n = (\sigma^2 \mu_0 + \sigma_0^2 x) / (\sigma^2 + \sigma_0^2)$ 이다.
> 이를 정리하면 $\mu_n = r \cdot \mu_0 + (1-r) \cdot x$, 여기서
> $r = \sigma^2 / (\sigma^2 + \sigma_0^2)$ 이다. adaTT에서
> $\mu_0 = \mathbf{P}$(Group Prior),
> $x = \mathbf{W} + \mathbf{A}$(학습된 가중치 + 친화도)이며, $r$ 이 0.5에서
> 0.1로 감소하는 것은 "관측 데이터의 양이 늘어나면서 $\sigma^2$ 의 영향이
> 줄어든다"는 것을 모사한다.

**모델**: 태스크 $i, j$ 의 true affinity $a_{i,j}$ 에 대해

$$a_{i,j} | \mathbf{P}, \sigma^2 \sim \mathcal{N}(\mathbf{P}_{i,j}, \sigma^2)$$

여기서 $\mathbf{P}$ 는 Group Prior 행렬이다.

**관측**: 코사인 유사도 $c_{i,j} = \cos(\theta_{i,j})$,

$$c_{i,j} | a_{i,j}, \tau^2 \sim \mathcal{N}(a_{i,j}, \tau^2)$$

**Posterior mean** (conjugate normal):

$$\mathbb{E}[a_{i,j} | c_{i,j}] = \frac{\tau^2}{\sigma^2 + \tau^2} \mathbf{P}_{i,j} + \frac{\sigma^2}{\sigma^2 + \tau^2} c_{i,j}$$

이를 $r = \tau^2 / (\sigma^2 + \tau^2)$ 로 정의하면:

$$\mathbb{E}[a_{i,j}] = r \cdot \mathbf{P}_{i,j} + (1 - r) \cdot c_{i,j}$$

이것은 adaTT의 prior blend 공식 (`adatt.py:381`)과 *정확히 동일*하다:

```python
raw_weights = raw_weights * (1 - r) + self.group_prior * r
```

### A.3 Softmax Temperature의 역할

$$w_{i \to j} = \frac{\exp(\mathbf{R}_{i,j} / T)}{\sum_{k \neq i} \exp(\mathbf{R}_{i,k} / T)}$$

- $T \to 0$: 가장 높은 가중치에 집중 (hard selection)
- $T \to \infty$: 균등 분포 (모든 태스크 동일 가중치)
- $T = 1.0$ (기본값): 중간 지점, 친화도 차이를 적절히 반영

본 시스템에서 $T = 1.0$ 을 사용하는 이유: 16개 태스크에서 너무 sharp한
선택 ($T < 0.5$)은 소수 태스크에만 전이가 집중되고, 너무 uniform한 선택
($T > 2.0$)은 negative transfer를 충분히 차단하지 못한다.

> **최신 동향 — Softmax Temperature (2023–2025).** (1) *Knowledge
> Distillation*: DKD (Zhao et al., CVPR 2022)가 target class와 non-target
> class의 logit을 분리하여 별도 temperature를 적용한다. (2) *LLM
> Decoding*: generation temperature가 *창의성 vs 정확성* 트레이드오프를
> 제어. (3) *Contrastive Learning*: SimCLR에서 $T = 0.07$ 같은 낮은
> temperature가 hard negative에 집중. (4) *Gumbel-Softmax* (Jang et al.,
> ICLR 2017): $T$ 를 annealing하여 이산적 선택을 미분 가능하게 근사.

### A.4 Negative Transfer 차단의 이론적 근거

Yu et al. (2020)의 PCGrad에서 gradient conflict를
$\cos(\theta_{i,j}) < 0$ 으로 정의한다. adaTT는 이보다 완화된 임계값
$\tau_{neg} = -0.1$ 을 사용하는데, 이는 *noise margin*을 고려한 것이다.

> **학부 수학 — 벡터 사영과 gradient conflict.** PCGrad의 핵심 연산은
> 벡터 사영이다. 벡터 $\mathbf{a}$ 를 $\mathbf{b}$ 위에 사영하면
> $\text{proj}_{\mathbf{b}} \mathbf{a} = (\mathbf{a} \cdot \mathbf{b}) / (\mathbf{b} \cdot \mathbf{b}) \cdot \mathbf{b}$
> 이다. PCGrad는 $\cos \theta < 0$ (충돌)일 때 gradient $\mathbf{g}_i$
> 에서 $\mathbf{g}_j$ 방향 성분을 *제거*한다:
> $\mathbf{g}_i' = \mathbf{g}_i - \text{proj}_{\mathbf{g}_j} \mathbf{g}_i$.
> adaTT는 이런 사영 대신 *전이 자체를 차단*하는 더 단순하고 보수적인
> 전략을 취한다 — gradient를 변형하지 않고, 해로운 태스크의 loss 기여를
> 0으로 만든다.

SGD의 stochastic noise로 인해 true affinity가 0에 가까운 태스크 쌍도
배치에 따라 $\cos(\theta_{i,j}) \approx -0.05$ 등의 약한 음의 상관을
보일 수 있다. $\tau_{neg} = -0.1$ 은 이러한 noise를 허용하면서 *명확한*
negative transfer만 차단하는 sweet spot이다.

### A.5 Transfer-Enhanced Loss의 수렴 영향

**명제 A.5**: Transfer loss의 gradient는 positive transfer 태스크의 학습
방향으로 공유 파라미터를 편향시킨다.

$$\nabla_\theta \mathcal{L}_i^{adaTT} = \nabla_\theta \mathcal{L}_i + \lambda \sum_{j \neq i} w_{i \to j} \nabla_\theta \mathcal{L}_j$$

> **수식 직관.** 이 수식은 "태스크 $i$ 의 파라미터 업데이트 방향이 자기
> 자신의 gradient에 다른 태스크들의 gradient를 가중 합산한 보정 벡터를
> 더한 것"임을 말한다. 직관적으로, 나 혼자 걸어가던 방향에 동료들이
> "이쪽이 더 좋다"고 제안하는 벡터를 $\lambda = 0.1$ 만큼 반영하여 경로를
> 미세 조정하는 것이다. 친화도가 높은 동료의 제안($w_{i \to j}$ 가 큰
> 경우)이 더 크게 반영된다.

$w_{i \to j} > 0$ 이고 $\cos(\theta_{i,j}) > 0$ 인 태스크 $j$ 의 gradient가
태스크 $i$ 의 gradient 방향으로 *가산*되어, 공유 파라미터가 양쪽 태스크
모두에 유리한 방향으로 업데이트된다.

$\lambda = 0.1$ 과 `max_transfer_ratio=0.5` 의 조합은 원본 loss의 학습
방향을 크게 왜곡하지 않으면서도 positive transfer의 이점을 얻을 수 있는
보수적 설정이다.

> **교차 참조 — Logit Transfer와의 관계.** adaTT는 *backward
> pass*(gradient) 수준에서 태스크 간 전이를 조절하는 반면, 본 시스템에는
> *forward pass*(logit) 수준에서 태스크 간 예측값을 직접 전달하는 *Logit
> Transfer* 메커니즘이 별도로 존재한다. Logit Transfer는 CTR→CVR→LTV
> 같은 비즈니스 로직상 순차적 의존성을 단방향 DAG로 명시적으로
> 설계하며, adaTT의 전방향 적응적 전이와 상호 보완적으로 작동한다.

## 전체 adaTT 기술 참조서 다운로드

여기까지 ADATT-1 부터 ADATT-4 까지 온프렘 `기술참조서/adaTT_기술_참조서` 를
블로그 형식으로 관통했다. 동기부여, 수학적 기초, 친화도 측정, Transfer
Loss, Group Prior, 3-Phase Schedule, Negative Transfer 차단, 학습 루프,
Loss Weighting, Optimizer, CGC 동기화, 디버깅 가이드까지 — 원본 PDF 는
수식·다이어그램·색인이 모두 살아있는 긴 참조 문서다.

> **📄 [adaTT 기술 참조서 전체 PDF 다운로드](/adaTT_기술_참조서.pdf)** · KO
>
> Adaptive Task Transfer · Gradient Cosine Similarity · Transfer Loss ·
> 3-Phase Schedule · Negative Transfer Detection — 본 프로젝트의 adaTT
> 전체 내용을 하나의 문서로 보고 싶으면 위 링크에서 받으면 된다.

## adaTT 서브스레드 종료, 다음은 이종 Shared Expert

여기까지가 adaTT 서브스레드의 끝이다. ADATT-1 의 적응형 타워 동기와
Transformer Attention 유비, ADATT-2 의 gradient 코사인 유사도 기반
친화도 측정, ADATT-3 의 Transfer Loss · Group Prior · 3-Phase Schedule ·
Negative Transfer 차단, 그리고 이번 4편의 2-Phase Training Loop · Loss
Weighting · Optimizer · CGC 동기화 · 메모리 최적화 · 디버깅 가이드 ·
부록까지 — PLE 6편과 adaTT 4편, 총 10편의 Study Thread로 본 프로젝트의
MTL 백본을 블로그 형식으로 정리했다.

다음부터는 7개 이종 Shared Expert 각자의 수학적 기초로 넘어간다. 우선
CausalOT(인과 추론 + 최적 수송), TDA(위상 데이터 분석 / PersLay),
Temporal(Mamba + LNN + Transformer), Economics 피처 기반 Expert 순서로
각각 서브스레드를 열 예정이다.
