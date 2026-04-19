---
title: "[Study Thread] ADATT-3 — Transfer Loss · Group Prior · 3-Phase Schedule"
date: 2026-04-20 14:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, transfer-loss, group-prior, schedule, negative-transfer]
lang: ko
series: study-thread
part: 9
alt_lang: /2026/04/20/adatt-3-transfer-loss-group-prior-schedule-en/
next_title: "ADATT-4 — 학습 루프·Loss Weighting·Optimizer·CGC 동기화 (+ 기술 참조서 PDF)"
next_desc: "2-Phase Training Loop, Loss Weighting 전략(Uncertainty·GradNorm·DWA), Optimizer 및 Scheduler 설정, CGC-adaTT 동기화, 메모리·성능 최적화, 디버깅 가이드, 설정 매개변수 총람, 부록 — adaTT 서브스레드 마무리 + 전체 adaTT 기술 참조서 PDF 다운로드 포함."
next_status: published
---

*"Study Thread" 시리즈의 adaTT 서브스레드 3편. 영문/국문 병렬로 ADATT-1 → ADATT-4 에 걸쳐 본 프로젝트의 adaTT 메커니즘을 정리한다. 출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서` 이다. 이번 3편은 adaTT 의 핵심 손실 항인 Transfer Loss 의 전체 공식과 전이 가중치 계산, G-01 FIX Transfer Loss Clamp, Target 미존재 태스크 마스킹, 태스크 그룹 기반 Prior 행렬과 Prior Blend Annealing, 3-Phase Schedule(Warmup → Dynamic → Frozen) 의 전환 로직, 그리고 Negative Transfer 감지·차단 메커니즘까지 다룬다.*

## 4. Transfer Loss 계산 메커니즘

`compute_transfer_loss()`는 adaTT의 핵심 메서드로, 각 태스크의 원본 손실에 다른 태스크로부터의 전이 손실을 가산한다.

> 소스: `adatt.py:283-353` — `compute_transfer_loss()` 메서드

### 4.1 전체 공식

각 태스크 $i$에 대한 Transfer-Enhanced Loss:

$$\mathcal{L}_i^{\text{adaTT}} = \mathcal{L}_i + \lambda \cdot \sum_{j \neq i} w_{i \rightarrow j} \cdot \mathcal{L}_j$$

- $\mathcal{L}_i$: 태스크 $i$의 원본 손실 (focal, huber, MSE 등)
- $\lambda = 0.1$ (기본값, `transfer_lambda` 파라미터)
- $w_{i \rightarrow j}$: 태스크 $i$에 대한 태스크 $j$의 전이 가중치 (softmax 정규화)

> **수식 직관.** 이 수식은 "각 태스크가 자기 자신의 loss만 보는 것이 아니라, 친화도가 높은 다른 태스크의 loss도 일부 참고한다"는 것을 말한다. $\lambda = 0.1$은 다른 태스크의 의견을 10%만 반영하겠다는 뜻이고, $w_{i \rightarrow j}$는 "누구의 의견을 더 들을 것인가"를 결정한다. 직관적으로, gradient 방향이 비슷한 태스크일수록 가중치가 높아져 서로의 학습을 가속시킨다.

### 4.2 전이 가중치 계산 상세

전이 가중치 $w_{i \rightarrow j}$는 여러 요소의 조합이다.

```python
# adatt.py:355-396 — _compute_transfer_weights()
raw_weights = self.transfer_weights + affinity  # 학습 가능 + 친화도

# Group Prior 결합 (annealing)
raw_weights = raw_weights * (1 - r) + self.group_prior * r

# Negative Transfer 차단
raw_weights = torch.where(
    affinity > self.neg_threshold,  # -0.1
    raw_weights,
    torch.zeros_like(raw_weights),
)

# 대각선 0 (자기 전이 제외)
raw_weights = raw_weights.masked_fill(self.diag_mask, 0.0)

# Softmax 정규화
weights = F.softmax(raw_weights / max(self.temperature, 1e-6), dim=-1)
```

수학적으로:

$$\mathbf{R} = (\mathbf{W} + \mathbf{A}) \cdot (1 - r) + \mathbf{P} \cdot r$$

$$\mathbf{R}_{i,j} \leftarrow 0 \quad \text{if } \mathbf{A}_{i,j} < \tau_{\text{neg}}$$

$$\mathbf{R}_{i,i} = 0$$

$$w_{i \rightarrow j} = \text{softmax}(\mathbf{R}_{i,j} / T)$$

- $\mathbf{W}$: 학습 가능한 전이 가중치 (`nn.Parameter`, 초기값 0)
- $\mathbf{A}$: EMA 친화도 행렬
- $\mathbf{P}$: Group Prior 행렬
- $r$: Prior blend ratio (Phase에 따라 변화)
- $\tau_{\text{neg}} = -0.1$: Negative transfer 차단 임계값
- $T = 1.0$: Softmax temperature

> **수식 직관.** 이 수식은 전이 가중치를 결정하는 4단계 파이프라인을 말한다. 먼저 학습 가능한 가중치 $\mathbf{W}$와 측정된 친화도 $\mathbf{A}$를 합산하고, 도메인 지식(Prior $\mathbf{P}$)과 혼합한다. 그런 다음 "해로운 전이"를 0으로 차단하고, 자기 자신은 제외한 뒤, softmax로 확률 분포를 만든다. 직관적으로, "데이터에서 관측한 태스크 관계 + 도메인 전문가의 사전 지식"을 결합하되, 해로운 경로는 끊어버리는 과정이다.

### 4.3 G-01 FIX: Transfer Loss Clamp

Transfer loss가 원본 loss를 지배하지 않도록 비율 제한이 적용된다.

```python
# adatt.py:346-351
raw_transfer = self.transfer_lambda * transfer_loss
if self.max_transfer_ratio > 0:
    max_val = original_loss.detach() * self.max_transfer_ratio
    raw_transfer = torch.clamp(raw_transfer, max=max_val)
enhanced_losses[task_name] = original_loss + raw_transfer
```

> **⚠ max_transfer_ratio = 0.5.** Transfer loss가 원본 loss의 *50%*를 초과할 수 없다 (`adatt.py:191`). 이 제한 없이 학습하면, 특정 태스크의 loss가 매우 작을 때 transfer loss가 상대적으로 과도하게 커져 학습 방향이 왜곡된다. `original_loss.detach()`를 사용하여 clamp 경계값이 gradient에 영향을 주지 않는다.

### 4.4 Target 미존재 태스크 마스킹

모든 배치에 모든 태스크의 target이 있는 것은 아니다. Target이 없는 태스크는 전이 가중치에서 제외된다.

```python
# adatt.py:321-334
loss_list = []
loss_mask = []
for name in self.task_names:
    if name in task_losses:
        loss_list.append(task_losses[name])
        loss_mask.append(1.0)
    else:
        loss_list.append(torch.tensor(0.0, device=affinity.device, requires_grad=False))
        loss_mask.append(0.0)

# ...
masked_transfer_w = transfer_w[i] * loss_mask_tensor
transfer_loss = (masked_transfer_w * loss_tensor).sum()
```

> **왜 0이 아닌 마스크를 사용하는가.** 단순히 0.0 loss를 넣으면 softmax 후 해당 가중치가 0이 되지 않는다. `loss_mask_tensor`로 곱하여 해당 전이 경로를 *완전히 차단*한다. 이는 배치별로 target이 가변적인 실환경(특히 비활성 태스크 존재 시)에서 안전하다.

## 5. Group Prior 구조

Group Prior는 도메인 지식을 수학적 prior로 인코딩한 행렬이다. 학습 초기에 태스크 간 친화도가 충분히 측정되기 전까지, 합리적인 전이 방향을 제공한다.

> 소스: `adatt.py:256-281` — `_build_group_prior()` 메서드

### 5.1 태스크 그룹 정의

`model_config.yaml:611-628`에서 4개 그룹이 정의된다.

| 그룹 | 멤버 | intra 강도 | 비즈니스 의미 |
|---|---|---|---|
| engagement | ctr, cvr, engagement, uplift | 0.8 | 고객 참여/전환 관련 |
| lifecycle | churn, retention, life_stage, ltv | 0.7 | 고객 생애주기 관련 |
| value | balance_util, channel, timing | 0.6 | 고객 가치/행동 패턴 |
| consumption | nba, spending_category, consumption_cycle, spending_bucket, merchant_affinity, brand_prediction | 0.7 | 소비 패턴 분석 |

`inter_group_strength: 0.3` — 그룹 간 전이 강도는 낮게 유지.

### 5.2 Prior 행렬 구성

```python
# adatt.py:256-281
def _build_group_prior(self) -> torch.Tensor:
    # 1. 그룹 간 전이 강도로 초기화
    prior = torch.ones(self.n_tasks, self.n_tasks) * self.inter_group_strength  # 0.3

    # 2. 그룹 내 전이 강도 설정
    for group_name, members in self.task_groups.items():
        strength = self.intra_group_strength.get(group_name, 0.5)
        indices = [self.task_names.index(m) for m in members if m in self.task_names]
        for i in indices:
            for j in indices:
                if i != j:
                    prior[i, j] = strength

    # 3. 대각선 = 0 (자기 자신 전이 없음)
    prior.fill_diagonal_(0.0)

    # 4. 행 정규화
    row_sums = prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
    prior = prior / row_sums
```

> **행 정규화의 의미.** 행 정규화는 각 태스크 $i$가 다른 태스크로부터 받는 전이 가중치의 합을 1로 만든다. 이는 softmax와 유사한 효과를 가지며, 태스크 수에 무관하게 전이 강도를 일관되게 유지한다.

### 5.3 Prior Blend Annealing

Prior blend ratio $r$은 학습 진행에 따라 선형적으로 감소한다.

$$r(e) = r_{\text{start}} - (r_{\text{start}} - r_{\text{end}}) \cdot \min\left(\frac{e - e_{\text{warmup}}}{e_{\text{freeze}} - e_{\text{warmup}}}, 1.0\right)$$

- $r_{\text{start}} = 0.5$: 학습 초기 prior 비율 (`prior_blend_start`, `model_config.yaml:607`)
- $r_{\text{end}} = 0.1$: 학습 후반 prior 비율 (`prior_blend_end`, `model_config.yaml:608`)
- $e_{\text{warmup}}$: warmup 종료 에포크
- $e_{\text{freeze}}$: freeze 시작 에포크

> **수식 직관.** 이 수식은 "학습이 진행될수록 도메인 전문가의 사전 지식($\mathbf{P}$)에 대한 의존도를 줄이고, 실제 데이터에서 관측한 친화도를 더 신뢰한다"는 것을 말한다. $r$이 0.5에서 0.1로 감소하면, Prior 비중이 50%에서 10%로 줄어들고 관측 데이터 비중이 50%에서 90%로 늘어난다. 직관적으로, 신입 사원이 처음에는 선배 의견(Prior)에 의존하다가 경험이 쌓이면 자기 판단(관측)을 더 믿는 것과 같다.

이 annealing은 *Bayesian 관점*에서 prior에서 posterior로의 전환으로 해석할 수 있다: 학습 초기에는 데이터가 부족하므로 domain knowledge (prior)에 의존하고, 데이터가 누적될수록 학습된 gradient 기반 친화도 (likelihood)를 신뢰한다.

> **역사적 배경 — Bayesian Weight 개념의 기원.** Prior와 데이터를 혼합하는 아이디어는 Thomas Bayes (1763, 사후 출판)와 Pierre-Simon Laplace (1812, *Theorie analytique des probabilites*)로 거슬러 올라간다. 신경망에 Bayesian 관점을 도입한 것은 MacKay (1992, *"A Practical Bayesian Framework for Backpropagation Networks"*)와 Neal (1996, *"Bayesian Learning for Neural Networks"*)이 선구자이다. 현대 딥러닝에서는 Dropout이 근사적 Bayesian inference로 해석되고 (Gal & Ghahramani, ICML 2016), Weight Uncertainty (Blundell et al., ICML 2015, *"Bayes by Backprop"*)이 가중치의 불확실성을 직접 학습한다. adaTT의 Prior Blend Annealing은 이러한 Bayesian 전통의 *실용적 경량화*이다 — 풀 Bayesian posterior를 추론하는 대신, blend ratio $r$ 하나로 prior-to-posterior 전환을 모사한다.

Prior Blend Annealing 스케줄은 **r = 0.5 (Prior 의존도 높음)** → 선형 감소 (Phase 2 시작) → **r = 0.1 (Affinity 신뢰, Phase 3 진입)** 로 진행된다.

## 6. 3-Phase adaTT Schedule

adaTT는 학습을 세 단계로 구분하여 친화도 측정과 전이를 제어한다.

> 소스: `adatt.py:298-313` — `compute_transfer_loss()` 내 Phase 분기

> **역사적 배경 — 학습 스케줄링의 계보.** "학습을 단계별로 나누어 진행한다"는 아이디어는 Bengio et al. (2009, *"Curriculum Learning"*)에서 체계화되었다 — 쉬운 예제부터 어려운 예제 순으로 학습하면 최종 성능이 향상된다는 발견이다. 이 아이디어는 (1) Pre-training + Fine-tuning (Erhan et al., 2010), (2) Layer-wise Training (Hinton et al., 2006, Deep Belief Networks), (3) Warmup-then-Decay 학습률 스케줄 (Goyal et al., 2017) 등으로 확장되었다. adaTT의 3-Phase(Warmup → Dynamic → Frozen)는 이 전통을 *태스크 간 전이*에 적용한 것이다: Phase 1에서 태스크 관계를 관찰하고(curriculum의 "탐색" 단계), Phase 2에서 관계를 활용하며(학습 본체), Phase 3에서 안정화한다(fine-tuning의 "고정" 단계).

### 6.1 Phase 1: Warmup (친화도 측정만)

```python
# adatt.py:300-304
if epoch < self.warmup_epochs:
    if task_gradients is not None:
        self.affinity_computer.compute_affinity(task_gradients)
    return task_losses  # 원본 손실 그대로 반환
```

Phase 1에서는 gradient cosine similarity를 계산하여 친화도 행렬을 축적하되, *전이 손실은 추가하지 않는다*. 원본 `task_losses`를 변경 없이 반환한다.

- **기간**: epoch 0 ~ `warmup_epochs` (프로덕션 기준 10, 테스트용 0)
- **목적**: 충분한 친화도 데이터 축적 없이 전이를 시작하면 random transfer로 학습이 불안정해진다
- **설정**: `model_config.yaml:598` — `warmup_epochs: 0` (테스트), 프로덕션 권장 10

### 6.2 Phase 2: Dynamic Transfer (동적 전이)

```python
# adatt.py:311-317
# Phase 2: 동적 전이
if task_gradients is not None:
    self.affinity_computer.compute_affinity(task_gradients)

affinity = self.affinity_computer.get_affinity_matrix()
transfer_w = self._compute_transfer_weights(affinity)
```

Phase 2에서는 매 step마다 친화도를 업데이트하면서 동시에 전이 손실을 적용한다. Prior blend ratio $r$이 `prior_blend_start`에서 `prior_blend_end`로 선형 감소한다.

- **기간**: `warmup_epochs` ~ `freeze_epoch`
- **학습 가능 파라미터**: `self.transfer_weights` (`nn.Parameter`, `adatt.py:229-231`)
- **Prior blend annealing**: $r$ 값이 0.5에서 0.1로 감소 (`adatt.py:373-379`)

### 6.3 Phase 3: Frozen (가중치 고정)

```python
# adatt.py:307-308
if self.is_frozen:
    return self._apply_frozen_transfer(task_losses)
```

Phase 3에서는 전이 가중치를 고정하고 더 이상 gradient를 계산하지 않는다. `_apply_frozen_transfer`에서 `transfer_w[i].detach()`를 사용한다 (`adatt.py:425`).

- **기간**: `freeze_epoch` ~ 학습 종료
- **설정**: `model_config.yaml:599` — `freeze_epoch: 1` (테스트), 프로덕션 권장 28
- **효과**: gradient 계산 오버헤드 제거, 학습 안정화

> **⚠ H-6 검증: freeze_epoch > warmup_epochs.** `adatt.py:219-223`에서 `freeze_epoch <= warmup_epochs`이면 `ValueError`를 발생시킨다. Phase 2가 완전히 스킵되면 학습된 친화도가 전혀 전이에 반영되지 않으므로, adaTT를 사용하는 의미가 없어진다. 이 검증은 설정 오류를 조기에 차단한다.

### 6.4 Phase 전환 트리거: on_epoch_end

```python
# adatt.py:431-452
def on_epoch_end(self, epoch: int) -> None:
    self.current_epoch.fill_(epoch)

    if self.freeze_epoch is not None and epoch >= self.freeze_epoch:
        if not self.is_frozen.item():
            self.is_frozen.fill_(True)
            logger.info(f"adaTT: 전이 가중치 고정 (epoch {epoch})")
```

> **fill_() 사용 이유.** `self.current_epoch = epoch`처럼 plain tensor를 재할당하면 `register_buffer`로 등록된 buffer와의 연결이 끊어진다. `fill_()`은 in-place 업데이트로 `state_dict`, device 관리를 유지한다. `is_frozen`도 마찬가지로 `fill_(True)`를 사용한다 (`adatt.py:441`).

## 7. Negative Transfer 감지 및 차단

### 7.1 Negative Transfer란 무엇인가

두 태스크의 gradient가 *반대 방향*을 가리킬 때, 공유 파라미터의 업데이트가 한 태스크의 성능을 개선하면서 다른 태스크의 성능을 *저하*시키는 현상이다.

예를 들어 CTR (클릭률)과 Churn (이탈률)의 gradient가 반대 방향이라면, CTR을 개선하는 방향으로 학습할 때 Churn 예측 성능이 악화된다.

### 7.2 차단 메커니즘

`_compute_transfer_weights()`에서 친화도가 임계값 이하인 전이 경로를 0으로 차단한다.

```python
# adatt.py:383-388
raw_weights = torch.where(
    affinity > self.neg_threshold,     # -0.1 (기본값)
    raw_weights,                       # 유지
    torch.zeros_like(raw_weights),     # 차단
)
```

$$\mathbf{R}_{i,j} = \begin{cases} \mathbf{R}_{i,j} & \text{if } \mathbf{A}_{i,j} > \tau_{\text{neg}} \\ 0 & \text{otherwise} \end{cases}$$

- $\tau_{\text{neg}} = -0.1$ (`negative_transfer_threshold`, `model_config.yaml:600`)

> **수식 직관.** 이 수식은 일종의 *게이트*이다. 친화도가 임계값보다 높으면 전이 가중치를 그대로 통과시키고, 낮으면 0으로 완전 차단한다. 직관적으로, "나와 반대 방향으로 가는 태스크의 조언은 아예 듣지 않는다"는 안전장치이다.

> **왜 -0.1이 임계값인가 (0이 아닌).** 코사인 유사도 0은 "직교" (무관)를 의미하며, 약간의 음의 상관도 노이즈일 수 있다. $-0.1$로 설정하여 약한 음의 상관은 허용하고, 명확한 반대 방향 gradient만 차단한다. 임계값이 0이면 너무 많은 전이 경로가 차단되어 adaTT의 효과가 약화된다.

> **최신 동향 — Negative Transfer 완화 기법의 최신 진화 (2023–2025).** Negative transfer 문제는 MTL의 핵심 난제로 활발히 연구되고 있다. (1) *Aligned-MTL (Senushkin et al., CVPR 2023)*: gradient를 공통 하강 방향(common descent direction)으로 정렬하되, 태스크별 기여도를 보존하는 정교한 사영 기법을 제안했다. (2) *ForkMerge (Ye et al., NeurIPS 2023)*: 학습 도중 태스크를 동적으로 분리(fork)했다가 병합(merge)하여 negative transfer 구간을 자동 감지하고 회피한다. (3) *Auto-Lambda (Liu et al., ICLR 2022)*: 검증 셋에서의 meta-gradient로 태스크 가중치를 자동 조절하여 negative transfer를 간접적으로 완화한다. adaTT의 임계값 기반 차단($\tau_{\text{neg}} = -0.1$)은 이들에 비해 *계산 비용이 극히 낮으면서도 효과적*이라는 점에서 실무 친화적이다. 향후 개선 방향으로는 적응적 임계값($\tau_{\text{neg}}$를 학습 단계에 따라 조절)이나 soft gating(binary 차단 대신 연속적 감쇠)을 고려할 수 있다.

### 7.3 Negative Transfer 진단 API

```python
# adatt.py:460-476
def detect_negative_transfer(self) -> Dict[str, List[str]]:
    affinity = self.affinity_computer.get_affinity_matrix()
    negative_pairs = {}
    for i, task_i in enumerate(self.task_names):
        neg_list = []
        for j, task_j in enumerate(self.task_names):
            if i != j and affinity[i, j] < self.neg_threshold:
                neg_list.append(task_j)
        if neg_list:
            negative_pairs[task_i] = neg_list
    return negative_pairs
```

반환 예시: `{"churn": ["ctr", "engagement"], "ltv": ["brand_prediction"]}` 형태로 어떤 태스크 쌍에서 negative transfer가 감지되었는지 확인할 수 있다.

### 7.4 차단이 학습에 미치는 영향

| 상황 | 효과 |
|---|---|
| 차단 미적용 | Negative transfer가 있는 태스크 쌍이 서로의 loss를 증가시켜 학습 불안정 |
| 과도한 차단 ($\tau_{\text{neg}} = 0$) | 대부분의 전이 경로 차단 → adaTT가 사실상 비활성화 |
| 적절한 차단 ($\tau_{\text{neg}} = -0.1$) | 명확한 negative transfer만 차단, 중립/양성 전이는 유지 |

Transfer Loss, Group Prior, 3-Phase Schedule, Negative Transfer 차단 — 이 네 층은 서로 맞물린다. Prior 는 초기의 빈 친화도 행렬을 도메인 지식으로 채우고, Phase 스케줄은 관찰 → 전이 → 고정의 curriculum을 강제하며, Negative Transfer 차단은 측정된 친화도가 음수로 꺾이는 구간에서 해로운 경로를 끊는다. G-01 FIX Clamp 는 이 모든 것이 원본 손실을 덮지 않게 비율 상한을 건다. **ADATT-4** 에서는 이 구조가 실제 학습 루프·Loss Weighting·Optimizer 설정과 어떻게 맞물리는지, 그리고 CGC 와의 동기화 계약까지 이어받는다.
