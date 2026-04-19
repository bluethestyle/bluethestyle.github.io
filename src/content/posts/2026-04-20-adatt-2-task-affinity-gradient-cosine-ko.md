---
title: "[Study Thread] ADATT-2 — TaskAffinityComputer와 Gradient Cosine Similarity"
date: 2026-04-20 13:00:00 +0900
categories: [Study Thread]
tags: [study-thread, adatt, gradient, cosine-similarity, ema]
lang: ko
series: study-thread
part: 8
alt_lang: /2026/04/20/adatt-2-task-affinity-gradient-cosine-en/
next_title: "ADATT-3 — Transfer Loss · Group Prior · 3-Phase Schedule"
next_desc: "adaTT 의 Transfer Loss 계산 메커니즘(전체 공식과 전이 가중치, G-01 FIX Transfer Loss Clamp, Target 미존재 태스크 마스킹), 태스크 그룹 기반 Prior 행렬과 Prior Blend Annealing, 3-Phase Schedule(Warmup → Dynamic → Frozen), 그리고 Negative Transfer 감지·차단 메커니즘."
next_status: published
---

*"Study Thread" 시리즈의 adaTT 서브스레드 2편. 영문/국문 병렬로 ADATT-1 → ADATT-4 에 걸쳐 본 프로젝트의 adaTT 메커니즘을 정리한다. 출처는 온프렘 프로젝트 `기술참조서/adaTT_기술_참조서` 이다. 이번 2편은 adaTT 가 태스크 간 친화도를 실제로 어떻게 측정하는지 — TaskAffinityComputer 의 클래스 구조, compute_affinity() 가 gradient cosine similarity 를 뽑아내는 방식, EMA 평활화, 유클리드 거리 대신 코사인을 쓰는 이유, 그리고 torch.compiler.disable 로 처리한 gradient 추출 경로까지를 다룬다.*

## 2. TaskAffinityComputer — 핵심 친화도 엔진

`TaskAffinityComputer` 는 adaTT의 기반 모듈로, 태스크 간 gradient cosine similarity 를 계산하고 EMA (Exponential Moving Average) 로 안정화된 친화도 행렬을 유지한다.

소스: `adatt.py:58-143` — `TaskAffinityComputer` 클래스.

### 2.1 클래스 구조와 초기화

```python
# adatt.py:58-84
class TaskAffinityComputer(nn.Module):
    def __init__(self, task_names: List[str], ema_decay: float = 0.9):
        super().__init__()
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.ema_decay = ema_decay

        # EMA 친화도 행렬 (학습 안정성)
        self.register_buffer("affinity_matrix", torch.eye(self.n_tasks))
        self.register_buffer("update_count", torch.tensor(0))
```

> **설계 선택 — register_buffer.** `affinity_matrix` 를 `register_buffer` 로 등록한 이유는 두 가지다. (1) `state_dict` 에 포함되어 체크포인트 저장/복원 시 자동 관리되고, (2) `.to(device)` 호출 시 자동으로 GPU/CPU 로 이동한다. `nn.Parameter` 가 아닌 buffer 로 등록하여 optimizer 가 이 행렬을 학습 대상으로 취급하지 않는다.

핵심 상태는 다음 두 버퍼로 구성된다.

| 버퍼 | 초기값 | 용도 |
| --- | --- | --- |
| `affinity_matrix` | `eye(n_tasks)` | EMA 평활된 태스크 간 친화도 행렬 [n×n] |
| `update_count` | `0` | EMA 업데이트 횟수 (첫 업데이트 시 EMA 스킵 판정) |

### 2.2 compute_affinity() 메서드

이 메서드는 각 태스크의 flattened gradient 벡터를 받아 코사인 유사도 행렬을 계산하고, EMA 로 기존 친화도와 혼합한다.

```python
# adatt.py:86-138
def compute_affinity(self, task_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 1. gradient 수집 (누락 태스크는 zero padding)
    grad_list = []
    for name in self.task_names:
        if name in task_gradients:
            grad_list.append(task_gradients[name])
        else:
            grad_list.append(torch.zeros_like(reference_grad))

    # 2. Stack: [n_tasks, grad_dim]
    grad_matrix = torch.stack(grad_list, dim=0)

    # 3. 코사인 유사도 행렬
    norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = grad_matrix / norms
    affinity = torch.mm(normalized, normalized.t())

    # 4. EMA 업데이트
    with torch.no_grad():
        if self.update_count > 0:
            new_affinity = self.ema_decay * self.affinity_matrix \
                         + (1 - self.ema_decay) * affinity
            self.affinity_matrix.copy_(new_affinity.clamp(-1.0, 1.0))
        else:
            self.affinity_matrix.copy_(affinity.clamp(-1.0, 1.0))
        self.update_count.add_(1)
```

> **학부 수학 — 행렬 곱으로 모든 쌍의 코사인 유사도를 한 번에 계산하기.** $n$ 개 태스크의 gradient 를 행으로 쌓은 행렬 $\mathbf{G} \in \mathbb{R}^{n \times d}$ 를 생각하자. 각 행 $\mathbf{g}_i$ 를 L2 norm 으로 나누면 정규화 행렬 $\hat{\mathbf{G}}$ 를 얻는다 ($\hat{\mathbf{g}}_i = \mathbf{g}_i / \|\mathbf{g}_i\|$). 이때 $\hat{\mathbf{G}} \hat{\mathbf{G}}^\top$ 의 $(i,j)$ 원소는 $\sum_{k=1}^d \hat{g}_{i,k} \hat{g}_{j,k} = \hat{\mathbf{g}}_i \cdot \hat{\mathbf{g}}_j = \cos \theta_{i,j}$ 이다. 즉, *단 한 번의 행렬 곱* 으로 $n^2$ 개 코사인 유사도를 동시에 구할 수 있다. 만약 이중 for 루프로 쌍마다 계산하면 $O(n^2 d)$ 연산이 $n^2$ 번의 Python 호출로 분산되어 느리지만, 행렬 곱 `torch.mm` 은 GPU 의 병렬 연산 유닛(CUDA cores)을 활용하여 수백 배 빠르게 처리한다. 16 개 태스크 기준으로 $16 \times 16 = 256$ 개의 유사도를 단일 GEMM (General Matrix Multiplication) 커널로 계산한다.

> **⚠ N-03 FIX: 부동소수점 오차 누적 방지.** EMA 업데이트 후 반드시 `.clamp(-1.0, 1.0)` 을 적용한다 (`adatt.py:132,135`). 수천 번의 EMA 누적 과정에서 부동소수점 오차가 쌓여 코사인 유사도가 $[-1, 1]$ 범위를 벗어나는 것을 방지한다. 이 clamping 없이 `arccos` 등 후속 연산을 수행하면 NaN 이 발생할 수 있다.

### 2.3 친화도 행렬 해석

친화도 행렬 $\mathbf{A} \in [-1, 1]^{n \times n}$ 는 대칭 행렬이다 ($\mathbf{A}_{i,j} = \mathbf{A}_{j,i}$).

| 범위 | 의미 | adaTT 행동 |
| --- | --- | --- |
| $\mathbf{A}_{i,j} \approx 1$ | 강한 양의 친화도 | 태스크 i, j 의 gradient 가 같은 방향 → 적극 전이 |
| $\mathbf{A}_{i,j} \approx 0$ | 중립 | 태스크 간 연관 없음 → 약한 전이 |
| $\mathbf{A}_{i,j} < -0.1$ | 음의 친화도 | Negative transfer 감지 → 전이 차단 |

대각 원소 $\mathbf{A}_{i,i} = 1$ 은 자기 자신과의 코사인 유사도이므로 항상 1 이다.

## 3. Gradient Cosine Similarity 수학적 기초

### 3.1 수학적 정의

두 태스크 $i$, $j$ 의 Shared Expert 파라미터 $\theta$ 에 대한 gradient 를 각각 $\mathbf{g}_i = \nabla_\theta \mathcal{L}_i$, $\mathbf{g}_j = \nabla_\theta \mathcal{L}_j$ 라 하면:

$$\cos(\theta_{i,j}) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \cdot \|\mathbf{g}_j\|}$$

- $\mathbf{g}_i \in \mathbb{R}^d$: 태스크 $i$ 의 flattened gradient 벡터
- $d$: Shared Expert 전체 파라미터 수
- $\|\cdot\|$: L2 norm

구현에서는 행렬 연산으로 모든 태스크 쌍의 유사도를 한 번에 계산한다:

```python
# adatt.py:122-125
norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
normalized = grad_matrix / norms
affinity = torch.mm(normalized, normalized.t())  # [n_tasks, n_tasks]
```

`clamp(min=1e-8)` 은 zero gradient 태스크의 0-division 을 방지한다.

### 3.2 EMA 평활화

단일 배치의 gradient 는 노이즈가 크므로, 시간에 걸쳐 안정화된 친화도를 얻기 위해 Exponential Moving Average 를 적용한다:

$$\mathbf{A}_t = \alpha \cdot \mathbf{A}_{t-1} + (1 - \alpha) \cdot \cos(\theta_t)$$

- $\alpha = 0.9$ (기본값, `ema_decay` 파라미터)
- $\mathbf{A}_0$: 첫 번째 관측값을 그대로 사용 (EMA 초기화, `adatt.py:134-135`)

> **왜 $\alpha = 0.9$ 인가.** $\alpha = 0.9$ 이면 과거 10 개 관측의 가중 평균에 근사한다 (effective window $\approx 1/(1-\alpha) = 10$). 학습 초기에는 gradient 가 불안정하므로 빠른 수렴보다 안정성을 우선하되, 태스크 관계가 epoch 에 따라 변할 수 있으므로 너무 보수적이지 않은 값을 선택한다. `model_config.yaml:601` 에서 설정: `ema_decay: 0.9`.

### 3.3 왜 코사인 유사도인가 (vs. 유클리드 거리)

코사인 유사도를 사용하는 이유는 세 가지다.

1. **스케일 불변성**: 태스크별 loss 규모가 다르면 gradient 크기도 다르다. 코사인 유사도는 방향만 비교하므로 이 차이에 영향받지 않는다.
2. **해석 용이성**: $[-1, 1]$ 범위로 정규화되어 "같은 방향 = positive transfer, 반대 방향 = negative transfer" 라는 직관적 해석이 가능하다.
3. **효율적 계산**: 정규화 후 행렬 곱 한 번으로 $O(n^2 d)$ 에 계산 완료.

Fifty et al., 2021 의 Task Affinity 연구에서도 gradient cosine similarity 를 태스크 유사도 측정의 표준 지표로 사용한다.

> **최신 동향 — Gradient 기반 태스크 유사도 측정 (2023--2025).** 코사인 유사도 외에도 다양한 gradient 기반 지표가 연구되고 있다. (1) *TAG (Task Affinity Grouping, Fifty et al., ICML 2021)*: gradient 내적의 부호 변화 패턴으로 태스크 그룹핑을 자동화한다. (2) *Gradient Vaccine (Wang et al., ICLR 2021)*: gradient 충돌 시 코사인 유사도에 기반한 *부분 사영* 을 적용하여 PCGrad 보다 세밀한 제어를 달성한다. (3) *Conflict-Averse Gradient Descent (CAGrad, Liu et al., NeurIPS 2021)*: 코사인 유사도를 최소화 대상으로 삼아 모든 태스크의 gradient 와 최소 각도를 최대화하는 공통 방향을 찾는다. (4) *Nash-MTL (Navon et al., ICML 2022)*: 태스크 간 경쟁을 Nash 협상 문제로 공식화하여 파레토 최적 gradient 를 구한다. adaTT 는 이 중에서 *유사도 측정 + 선택적 전이* 를 결합한 접근으로, 측정과 활용을 분리한 모듈러 설계가 강점이다.

### 3.4 Gradient 추출 경로

gradient 는 `ple_cluster_adatt.py` 의 `_extract_task_gradients` 메서드에서 추출된다. 핵심은 *Shared Expert 파라미터에 대해서만* gradient 를 계산한다는 것이다.

```python
# ple_cluster_adatt.py:1871-1903
shared_params = list(self.shared_experts.parameters())

for task_name, loss in eligible:
    grads = torch.autograd.grad(
        loss,
        shared_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_flat = torch.cat([
        g.flatten() if g is not None else torch.zeros_like(p).flatten()
        for g, p in zip(grads, shared_params)
    ])
    task_gradients[task_name] = grad_flat
```

> **⚠ retain_graph=True 의 필수성.** `retain_graph=True` 는 *아키텍처상 제거 불가* 하다 (`ple_cluster_adatt.py:1865-1868`). `_extract_task_gradients` 는 forward pass 도중 (loss 계산 후, `backward()` 전) 에 호출되며, 동일한 computation graph 를 Trainer 의 `loss.backward()` 에서 재사용해야 한다. 여기서 graph 를 해제하면 `backward()` 시 `RuntimeError: Trying to backward through the graph a second time` 이 발생한다.

### 3.5 torch.compiler.disable 데코레이터

```python
# ple_cluster_adatt.py:1847
@torch.compiler.disable
def _extract_task_gradients(self, task_losses):
```

> **왜 torch.compile 에서 제외하는가.** `torch.autograd.grad` 는 컴파일된 그래프 내에서 `requires_grad` 추적이 불완전하다. `torch.compiler.disable` 데코레이터로 이 메서드를 컴파일 그래프 밖에서 실행하여 gradient 추출이 정상 작동하도록 보장한다. 실제로는 `torch.compile` 자체가 비활성화되어 있지만 (`trainer.py:177`), 향후 활성화 시에도 안전하도록 방어적으로 적용한 것이다.

## 여기서 멈추는 이유

ADATT-2 는 adaTT 가 "태스크 간 친화도를 *어떻게 측정하는가*" 에 대한 답이다. Shared Expert 파라미터 공간에서 태스크별 gradient 를 뽑아내고, 정규화 후 단일 행렬 곱으로 전체 쌍의 코사인 유사도를 구한 뒤, EMA 로 배치 노이즈를 평활한다 — 그리고 그 결과를 $[-1, 1]$ 범위의 친화도 행렬 $\mathbf{A}$ 로 유지한다. 하지만 아직 이 친화도를 어떻게 *활용* 해서 실제 태스크 간 전이를 일으키는지는 다루지 않았다. 친화도가 양수면 gradient 를 얼마나 빌려오고, 음수면 어떻게 차단하며, 학습 초기의 불안정한 친화도와 후반의 수렴된 친화도를 어떻게 다르게 다룰 것인가 — 이것이 **ADATT-3** 에서 이어받는 Transfer Loss, Group Prior, 3-Phase Schedule 의 주제이다.
