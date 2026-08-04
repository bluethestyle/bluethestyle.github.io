# CLAUDE.md — bluethestyle.github.io

기술 블로그(Astro 5). 이 문서는 **포스트의 기술적 주장을 무엇에 대조해서 검증하는가** 를 고정한다.
포스트 내용을 수정하기 전에 반드시 읽는다.

## 1. 저장소 구조

- 빌드: Astro 5 + MDX, `remark-math` / `rehype-katex` (수식), `astro.config.mjs` 의 `remarkMermaid` 가
  ` ```mermaid ` 블록을 `<pre class="mermaid">` 로 변환 → `Base.astro` 가 클라이언트에서 렌더.
- 포스트: [src/content/posts/](src/content/posts/) — **43편 × ko/en = 86 파일**.
  파일명 규약 `YYYY-MM-DD-<slug>-<ko|en>.md`.
- 시리즈: `study-thread`(27) · `three-months`(8) · `mrm-thread`(6) · commentary 2편(시리즈 미지정).
- 프론트매터 키: `title, date, categories, tags, lang, excerpt, series, part,
  alt_lang, next_title, next_desc, next_status`.
  `alt_lang` 은 반대 언어판의 **URL 경로**(`/YYYY/MM/DD/<slug>-<lang>/`)여야 한다.

## 2. 원본 프로젝트와 근거 우선순위

포스트가 서술하는 시스템의 실물은 온프렘 프로젝트 `C:\workspace\gotothemoon` 이다.

**충돌 시 우선순위 — 높은 쪽이 이긴다:**

1. **실제 코드** — `gotothemoon/workspace/code/src/**` (단, `src/_legacy_merlin/**` 는 격리된 死코드이므로 근거로 쓰지 않는다)
2. **설계서** — `gotothemoon/docs/설계서/*.md`
3. **기술참조서** — `gotothemoon/docs/기술참조서/*.typ` (포스트가 명시하는 출처)

### 불일치 3분류 — 처리 방식이 다르다

| 유형 | 판정 | 조치 |
|---|---|---|
| ① 오류 | 블로그가 원본과 불일치 | 블로그 수정 |
| ② 노후 | 작성 시점엔 맞았으나 프로젝트가 이후 변경됨 | 블로그 수정 + 변경 시점 명시 |
| ③ 원본 미결 | 코드/문서 자체가 모순이거나 마이그레이션 중 | **수정하지 말고 보고** — 결정은 엔지니어 몫 |

③ 을 ① 처럼 자동 수정하는 것이 이 저장소에서 가장 큰 위험이다.

## 3. 포스트 ↔ 근거 문서 매핑

`study-thread` 27편은 기술참조서 18종과 사실상 1:1 대응한다. (ko/en 쌍은 동일 근거를 공유)

| # | 포스트 slug | 근거 문서 (`docs/기술참조서/`) |
|---|---|---|
| 1–3, 5–6 | `ple-1` ~ `ple-6` | `PLE_기술_참조서.typ` |
| 4 | `ple-4-cgc-hmm-routing` | `PLE_기술_참조서.typ` (본문에 출처 미표기 — 추정) |
| 7–10 | `adatt-1` ~ `adatt-4` | `adaTT_기술_참조서.typ` |
| 11–12 | `tda-1-topology-of-spending`, `tda-2-perslay-set-function` | `PersLay_기술_참조서.typ` |
| 13 | `deepfm-feature-interaction` | `DeepFM_기술_참조서.typ` |
| 14 | `hgcn-hyperbolic-graph` | `GCN_기술_참조서.typ` |
| 15 | `causal-ot-expert` | `CausalOT_기술_참조서.typ` |
| 16 | `temporal-ensemble` | `Temporal_기술_참조서.typ` |
| 17 | `economics-expert` | `Economics_피처_기술_참조서.typ` |
| 18 | `tda-features-offline` | `TDA_피처_기술_참조서.typ` |
| 19 | `hmm-regime-features` | `HMM_피처_기술_참조서.typ` |
| 20 | `gmm-soft-clustering` | `GMM_피처_기술_참조서.typ` |
| 21 | `timeseries-features` | `TimeSeries_피처_기술_참조서.typ` |
| 22 | `multidisciplinary-features` | `Multidisciplinary_피처_기술_참조서.typ` |
| 23 | `knowledge-distillation` | `지식증류_기술_참조서.typ` |
| 24 | `inference-scoring` | `추론_스코어링_기술_참조서.typ` |
| 25 | `reason-generation` | `추천사유생성_기술_참조서.typ` |
| 26 | `grounding-reverse-mapping` | `그라운딩_피쳐역매핑_기술_참조서.typ` |
| 27 | `qwen-vllm-serving` | `Qwen_vLLM_기술_참조서.typ` |

`three-months` · `mrm-thread` · commentary 는 서사/회고 성격으로 단일 근거 문서가 없다.
기술 주장이 나오면 코드와 설계서에 직접 대조한다.

## 4. 핵심 코드 근거 위치

| 주제 | 파일 |
|---|---|
| 피처 차원 계약 (V1/V2) | `workspace/code/src/features/integration/feature_contract.py` |
| PLE + adaTT 모델 본체 | `workspace/code/src/models/ple_cluster_adatt.py` |
| 태스크 정의 · 클래스 수 · 그룹 | `workspace/code/src/config/active_tasks.py` |
| 차원 불일치 이력 | `gotothemoon/FEATURE_DIM_RECONCILIATION.md` (2026-05-29) |

## 5. 교차 주장 재정 (확정, 2026-08-04)

여러 배치를 관통하는 숫자를 **코드에 직접 대조해 확정**한 결과. Phase 2 배치는 이 판정을
전제로 진행하며, 배치 안에서 재판정하지 않는다.

근거: `feature_contract.py` 의 `V2_BASE_GROUPS` / `V2_APPEND_FEATURE_GROUPS` /
`SEPARATE_INPUT_GROUPS` 실값, `task_feature_mapper.py`, `ple_cluster_adatt.py`.

| 숫자 | 코드 근거 | 판정 |
|---|---|---|
| **644D** | `V2_BASE_GROUPS` 앞 7그룹 합 = 238+91+159+24+27+84+21 | ✅ 정확 |
| **90D** | `("raw_power_law", 90)` | ✅ 정확 |
| **734D** | 644 + 90 = 734 = **V2 공유 베이스 8그룹** | ⚠️ 구조는 정확, "메인 텐서 폭" 서술은 노후 |
| **4035D** | 734 + `V2_APPEND`(1440+1440+360+30+31 = 3301) | 🔴 **블로그 0회 — 누락** |
| **159D** domain | *"domain stays at the V1-frozen 159 slot"* | ✅ V2 에서도 유지 |
| **64D** | `group_output_dim`/`output_dim` 기본 64, `_cgc_mean_dim = 64.0` | ✅ 현행 (128D 와 비대칭 병존) |
| **22D** | `task_feature_mapper.py:110` `"gmm_cluster": 22` | ✅ 정확 |
| **48D** HMM | `hmm_journey`/`lifecycle`/`behavior` 각 16 = 48, `PLEClusterInput.hmm_*` 별도 전달 | ✅ 정확 |
| **5D** hmm_summary | `model_derived` 27 = hmm_summary 5 + bandit 4 + lnn 18 | ✅ 정확 |
| **70D / 58D** TDA | 아래 참조 | ✅ **둘 다 정확** — 같은 슬롯의 두 관점 |

### 70D vs 58D — 모순이 아니다

`domain` 159 = **147 real + 12 pad**. real 내역은 tda_short 24 + tda_long 24 +
phase_transition 10 + gmm_cluster 22 + mamba_temporal 50 + economics 17 = 147.

- **스키마 라벨** 기준 TDA = 24 + **36** + 10 = **70**
- **실제 산출** 기준 TDA = 24 + **24** + 10 = **58**
- **70 − 58 = 12 = domain 그룹의 pad 와 정확히 일치**

즉 12 pad 의 정체가 `tda_long` 의 라벨(36) 대비 산출(24) 부족분이다. 블로그의 70D 는
슬롯 폭을, 코드의 58D 는 실산출을 말하는 것으로 **같은 대상의 두 관점**이다.

`configs/model_config.yaml` 의 `perslay.input_dim` 주석이 이를 날짜와 함께 확증한다 —
*"[2026-06-24: 70→58, tda_long 실측 24D 정정(2026-04-23) 반영. 구 70은 tda_long
36D(β2/H2) 미생산분]"*. **PersLay Expert 의 입력은 58D 가 확정값**이며, 2026-06-05 자
`tda-2` 포스트의 70D 폴백 서술은 그 변경 이전 시점이다(해당 포스트에 주석 추가 완료).
[tda-features-offline-ko.md](src/content/posts/2026-06-07-tda-features-offline-ko.md) 는 이미
"스키마 라벨 36 / 산출 24" 를 본문에 기록하고 있다. **수정 대상 아님.**
(단 오프라인 도메인 블록과 PersLay Expert 입력 `[batch,58]` 은 층위가 다르니 혼동 금지.)

### Expert 라우팅 — 교차 재정 (배치마다 재판정 금지)

Expert 는 텐서 전체를 받지 않는다. **피처 그룹 단위로 라우팅**된다.
SoT = `ple_cluster_adatt.py` 의 `DEFAULT_EXPERT_ROUTING_V2`
(`configs/model_config.yaml` 의 `v2_expert_dims` 가 참조값으로 일치).

| Expert | 라우팅 그룹 | 폭 |
|---|---|---|
| `deepfm` / `xdeepfm` / `autoint` | 13개 그룹 전부 | **4035D** |
| `causal` | base + multi_source + domain + multidisciplinary + model_derived | **539D** |
| `optimal_transport` (= `ot`) | extended_source + multi_source | **175D** |
| `lightgcn` | 64 + product_top30 | **95D** |

- Expert 출력은 별개다 — `output_dim` 기본 **64D**(일부 128D). 입력 폭과 혼동 금지.
- **"DeepFM · Causal · OT 가 같은 644D 정규화 벡터를 입력받는다" 는 서술은 틀렸다.**
  `model_config.yaml` 의 V1 값은 셋 다 `input_dim: 734` 다 (L225 · L286 · L318).
  644D 는 *normalized 소집합*(= 734 − raw_power_law 90)이지 Expert 입력 폭이 아니다.

  | Expert | V1 `input_dim` | V2 override |
  |---|---|---|
  | `deepfm` | 734 | **4035** |
  | `causal` | 734 | **539** |
  | `optimal_transport` | 734 | **175** |

  즉 V1 에서는 셋이 같은 폭(734)을 받은 것이 맞지만 그 값은 644 가 아니고,
  V2 에서는 **더 이상 같지도 않다**.

### DeepFM 필드 분할 — 교차 재정

`DEEPFM_FIELD_SPEC` (`src/models/experts/deepfm_expert.py:171`) 실측 = **31 필드, 합 771D**.
파일 자신의 주석도 *"이 기본 spec의 합은 771D다"* 라고 명시한다.

⚠️ `configs/model_config.yaml:220` 주석은 *"644D를 28개 서브그룹(필드)으로 분할"* 이라고
쓰지만 **낡았다**. 같은 파일 L228 은 *"(31필드)"* 라고 바르게 적어 자기모순이다.
우선순위(코드 > 설정 주석)에 따라 **31 필드 / 771D** 를 정답으로 한다.
블로그가 "28개 의미 필드" 라고 쓴 것은 이 낡은 yaml 주석을 따라간 결과다.
- 어떤 피처 블록이 어느 Expert 에 닿는지 주장할 때는 반드시 위 라우팅 표를 확인한다.
  예: `multidisciplinary` 24D 는 **DeepFM 과 Causal 에만** 닿고 OT 에는 닿지 않는다.

### 유일한 실질 결함 — V2 운영 계약 층 누락

블로그의 차원 서술은 **V1 기준으로 정확하지만 V1 에서 끝나 있다.** 코드는 세 곳에서
블로그와 동일한 프레이밍에 V2 를 덧붙여 쓴다:

> `feature_reverse_mapper.py:78` · `ai_risk_classifier.py:552` · `triton_feature_preprocessor.py:93`
> — *"SoT(feature_contract.py): V1 normalized=644(domain 159) / main 734 / **운영 strict V2 4035**"*

그리고 `feature_contract.py` 는 2026-07-02 자로 V1 런타임 은퇴(`is_v2_enabled()` 항상 `True`).

**따라서 수정은 치환이 아니라 가산이다** — 734D 를 4035D 로 바꾸는 게 아니라,
734D 를 V2 의 공유 베이스로 자리매김하고 운영 계약이 4035D 임을 덧붙인다.
대상은 734D 를 언급하는 9편(p17~p22, p24~p26)의 ko/en 쌍.

## 6. 원본 프로젝트 쪽 결함 (블로그 문제 아님 — 보고용)

- `ple_cluster_adatt.py:118` 주석이 *"strict V2 4057D"* 라고 쓰지만, `feature_contract.py` 는
  4057/domain=181 확장을 2026-06-16 에 되돌리고 **4035** 로 확정했다. 주석이 노후.
- `distillation/soft_label_generator.py:1512` 는 `life_stage: {"num_classes": 6}` 인데,
  `config/active_tasks.py:70` 은 2026-07-08 자로 **3-class** `{student, adult, senior}`.
- `task_feature_mapper.py:87` 주석이 *"hmm_journey/lifecycle/behavior: 각 10D"* 라고 쓰지만,
  같은 파일 L158–160 은 각 **16D**(합 48D)이고 L124 도 *"HMM Triple-Mode 48D"* 라고 명시한다.
  L87 주석이 노후.
- `configs/model_config.yaml:82` 는 `expert_input_version: v1` 인데, `feature_contract.py` 의
  `is_v2_enabled()` 는 2026-07-02 부터 항상 `True` 라 실제로는 V2 경로만 탄다. 설정값이 死값.
- `hmm_features.py:886` 의 차원 검증은 모드 무관 `n_states + 5 + 6` 이라 Behavior 가 **17D**
  인데, `task_feature_mapper.py:160` 은 `hmm_behavior: 16` 을 선언한다. 산출 17 / 계약 16 불일치.
- `feature_reverse_mapper.py:74-87` 의 `FEATURE_RANGES` 는 **632D**(domain 147) 레이아웃인데
  계약은 644D(domain 159)다. 파일 주석이 *"미마이그레이션 … 역매핑 정합은 엔지니어 결정"*
  이라고 스스로 밝힌다.
- `ple_dataset.py:15` 는 LNN 18D 를 *"ODE latent 16D + stability 1D + convergence 1D"* 라
  적지만, 실제 산출 모듈 `lnn_features.py` 는 *"분포 4D + 주파수 4D + 변화점 3D +
  자기상관 4D + 복잡도 3D"* 인 **통계 피처**다. 두 설명이 다르다.

## 6b. 미결 — 사용자 판단 필요 (자동 수정 금지)

**"7개 이종 Shared Expert" 서술 vs 코드의 활성 6개.**
`configs/model_config.yaml` 의 `shared_experts` 중 `enabled: true` 는
**perslay · deepfm · temporal · causal · optimal_transport · unified_hgcn = 6개**다.
**`lightgcn` 은 `enabled: false`** 이고 `hgcn` / `merchant_hgcn` / `din` / `xdeepfm` /
`autoint` 도 전부 `false`.

블로그는 여러 편에서 *"7개 이종 Expert (DeepFM · LightGCN · UHGCN · Temporal ·
PersLay · Causal · OT)"* 로 서술한다 — p2 · p3 · p6, `three-months` ep3 · ep4.
반면 p24 는 *"734D 6-Expert 모델"* 이라 써서 블로그 내부에서도 갈린다.

판단이 갈리는 지점:
- p3 는 *"왜 이 7명을 뽑았나"* 라는 **설계 근거** 글이라, 설계된 pool 을 7개로
  서술하는 것이 반드시 오류는 아니다.
- 반면 p2 의 Expert 비교표나 p6 의 사양표는 **현재 구조** 를 기술하므로
  활성 6개와 어긋난다.
- ~~`lightgcn` 의 `enabled: false` 가 영구 폐기인지 일시 비활성인지 판정 불가~~
  → **해소됨.** `configs/model_config.yaml:525` 주석이 명시한다:
  *"Shared Experts 출력 합(현행 활성 6개 = 5×64 + unified_hgcn 128 = **448D**;
  **lightgcn 복구 시 512D**)"*. "복구 시" 라는 표현이 **일시 비활성**임을 알려준다.
  `hgcn` 의 영구 폐기(*"deprecated → unified_hgcn 으로 통합"*)와는 성격이 다르다.

**이 때문에 연쇄로 어긋나는 숫자들 (수정 완료):**

| 값 | 7개 전원 기준 (블로그) | 현행 활성 6개 |
|---|---|---|
| Shared concat 폭 | 512D | **448D** (5×64 + 128) |
| SAE overcomplete latent | 2048D | **1792D** (`expansion_factor = 4`) |

PLE-4 · PLE-6 에 위 실측을 주석으로 가산했다. HGCN-1 이 이미 448D 를 쓰고 있어
블로그 내부에서도 512 / 448 이 갈려 있었다.

→ **여전히 남는 저자 결정:** "7개 Expert" 라는 서사 표현을 그대로 둘지
("설계된 7 / 활성 6" 병기 등). PLE-3 처럼 *설계 근거* 를 다루는 글은 7이 맞고,
PLE-2 비교표·PLE-6 사양표처럼 *현재 구조* 를 기술하는 글은 어긋난다.

**태스크 개수 — 13 / 15 / 16 / 17 이 공존한다.**
`src/config/active_tasks.py` 의 `TASK_METADATA` 는 **17개** 정의, 그중 `uplift` 와
`category_uplift` 가 `"enabled": False` 라 **활성 15개**다.

블로그 표기는 갈린다:
- `ep1-premise` · `ple-1` · `ple-4` 는 **13개** 라 쓰되, *공개 AWS 벤치마크 버전* 임을
  명시한다 (ple-4: *"이 도식은 온프렘 기술참조서의 일반 로스터를 쓴다. 운영 벤치마크는
  13개 태스크다"*). 이 구분이 있으므로 오류로 단정할 수 없다.
- 다만 `ple-2` 는 *"우리가 다루는 13개 태스크"* 라고 써서 온프렘 프로젝트의 태스크로 읽힌다.
- `causal-ot-expert` 는 *"16개 task tower"* 라 쓰는데 이 숫자는 코드 어디에도 없다.

→ 온프렘(15/17) 과 공개 벤치마크(13) 를 어느 글에서 어느 기준으로 쓸지는 저자 결정 사항.

## 6c. 검증 커버리지 (2026-08-04 기준)

**코드 대조 배치를 완료한 study-thread 글 — 21편 (전 27편 중).**
PLE-2 · PLE-3 · PLE-4 · PLE-5 · PLE-6 · ADATT-1~4 · TDA-1 · TDA-2 · DEEPFM-1 ·
HGCN-1 · CAUSALOT-1 · TEMPORAL-1 · ECON-1 · TDAFEAT-1 · HMM-1 · GMM-1 ·
TSFEAT-1 · MULTI-1 · GROUND-1 · QWEN-1

**AWS 저장소 — `C:\workspace\aws_ple_for_financial` (검증 완료).**
MRM 스레드 · Commentary · 4개월 개발기가 근거로 삼는 공개 저장소
(`github.com/bluethestyle/aws_ple_for_financial`). `core/…` 경로 표기가 이쪽이다.
식별자 316개를 두 저장소에 대조한 결과 **AWS 에만 존재 13건, 어디에도 없음 3건**이었고,
3건 중 2건(`ComplianceReporter` · `USComplianceGenerator`)은 각각 *기각한 안티패턴* 과
*가상의 미래 모듈* 이라 부재가 정상이다. 실제 오류는 `core/retrieval/` 1건이었다.

**AWS 저장소 대조로 확인된 사실:**
- 공개 벤치마크는 **12개 태스크** (README `12-task`). 초기 13에서 정합됨
  (`docs(paper1~3): 13→12 태스크 정합` 커밋 4건). 블로그의 "13개" 는 노후 → 수정 완료.
- 피처 차원 **~349D input / 403D after Phase 0** — MRM-3 의 "349D → 403D" 서술과 일치.
- 온프렘 로스터는 `docs/design/onprem_experiment_design.md` 가 **"734D (16 tasks)"** 로
  적는다. CAUSALOT-1 의 "16개 task tower" 는 여기 근거가 있다(미확인 아님).
- `min_improvement: 0.005` / `max_degradation: 0.02` (`configs/pipeline.yaml`) — MRM-2 일치.
- `segment_task_weights` 1.0~1.5 클리핑, `dynamic_weight_rules` — `AGENTS.md`/`CLAUDE.md`
  설계 원칙과 일치. 단 **파일 경로가 틀렸다**(→ `configs/datasets/santander.yaml`). 수정 완료.
- `core/monitoring/pia_evaluator.py` · `public_disclosure_generator.py` 실재 — Commentary 일치.
- `tenure_stage` · `spend_level` 실재하며 `santander.yaml:93` 이
  *"REMOVED: tenure_stage — reconstructable"* 라 적어 4개월 개발기 Ep5 서사와 부합.

**⚠️ (구) MRM 스레드 6편 · Commentary 2편 — gotothemoon 으로는 검증 불가.**
이 글들은 **다른 저장소**를 근거로 한다:
`https://github.com/bluethestyle/aws_ple_for_financial` (MRM-5 가 명시적으로 인용).
`core/audit/` · `core/retrieval/` · `core/monitoring/` 같은 `core/…` 경로 표기가
그 저장소의 레이아웃이다 (gotothemoon 은 `src/…` 를 쓰고 `src/core/` 하위에는
`agent`/`pipeline`/`recommendation` 만 있다).

전수 식별자 검사에서 gotothemoon 에 없는 것으로 나온 **16건**은 전부 이 계열이며,
**오류가 아니라 검증 범위 밖**이다:
`KoreanFRIAAssessor`(22회) · `ComplianceReporter` · `USComplianceGenerator` ·
`core/monitoring/pia_evaluator.py` · `core/monitoring/public_disclosure_generator.py` ·
`core/audit/` · `core/retrieval/` · `configs/pipeline.yaml` ·
`scoring.segment_task_weights` · `scoring.dynamic_weight_rules` ·
`min_improvement` · `max_degradation` · `tenure_stage` · `spend_level` 등.
`PIAEvaluator` / `FRIAEvaluator` / `AnnexIVMapper` / `PublicDisclosureGenerator` 는
gotothemoon 의 **문서(백서·리포트·DAG)** 에는 있으나 `src/` 코드에는 없다.

→ 이 8편을 검증하려면 AWS 저장소가 필요하다.

**4개월 개발기 8편** — 주장 밀도 최저(합 13건), 서사 위주. 미착수.

**절차 — 완료됨:**
- **ko/en 정합성** — v1 검사의 "19쌍 불일치"는 **전부 허위**였다. 아라비아 숫자 패턴
  개수만 비교해서 한국어의 후치 수사(`성분 5개` vs `5 components`)와 단위 차이
  (`27차원` vs `27D`), 영문 수사 표기(`seven experts`)가 불일치로 잡혔다.
  정규화 후 재검사 결과 실제 내용 드리프트는 **1건** — PLE-3 KO 에서 LightGCN 이
  한 단순화를 NGCF 가 한 것처럼 주어가 뒤바뀌어 있었다(EN 은 정상). 수정 완료.
- **KaTeX** — display 298 · inline 1,774 = **총 2,072개**. 구문 문제 **0건**
  (`$$` 짝 · 중괄호 · `\left`/`\right` 균형 · 미지원 매크로). 산술도 검산 완료 —
  dim-normalize scale(0.756 / 1.069), 코사인 예시 0.89, 24/734=3.3%, 24/4035=0.6%,
  68/734=9.3%, 28×27/2=378, 31×30/2=465, 771×16=12,336, ln20≈2.996, 448×4=1792.
- **학술 인용 41건 대조** — Tang RecSys 2020 · Ma KDD 2018 · Jacobs 1991 ·
  Rendle ICDM 2010 · Kipf&Welling 2017 · He SIGIR 2020 · Koren 2009 ·
  Carrière AISTATS 2020 · Zheng NeurIPS 2018 · Cuturi NeurIPS 2013 · Shannon 1948 ·
  Baum&Petrie 1966 · Kendall CVPR 2018 · Gal&Ghahramani 2016 · Sensoy NeurIPS 2018 ·
  MacKay 1992 · Neal 1996 · Neyman 1923 · Rubin 1974 · Pearl 2000 · Kantorovich 1942 ·
  Stigler 1961 · Nickel&Kiela NeurIPS 2017 · Cohen&Felson 1979 · Barabási Nature 2005 ·
  Platt 1999 · Bucilua/Caruana/Niculescu-Mizil **KDD 2006**(Model Compression — 정확) ·
  Hinton/Vinyals/Dean 2015 · Hamilton 1989 · Kermack&McKendrick 1927 등 **전부 정확**.
  수정 1건: NGCF 의 저자 표기 `He 2019` → **`Wang et al., SIGIR 2019`**(He 는 공저자).

**남은 미완:**
- 기술참조서 18종 중 통독한 문서 없음(필요 부분만 grep). 다만 참조서는 3순위 근거이고
  실제로 734D · DeepFM 28필드 · HMM 6+4+6 이 전부 참조서가 낡은 경우였으므로
  검증 가치가 낮다. 참조서에만 있는 것은 학술 인용이었고 그건 위에서 처리했다.

### 4개월 개발기 8편 — 검증 완료

주장 밀도가 낮아(추출 13건) 후순위였으나 정량 주장은 전부 대조했다.
`RTX 4070 · 12GB VRAM`, `5-agent Bedrock`(3 serving + 2 ops/audit), `941K users`,
`17-month`, `containers/lambda/`, `LightGBM per-task students`, adaTT `−0.019` →
버그 수정 후 `−0.001` — **AWS README 와 전부 일치**.

⚠️ **이 시리즈의 "13개 태스크" 는 고치지 말 것.** 개발 당시의 실험 조건이고
AWS README:221 도 *"−0.019 in the 13-task heterogeneous setting"* 으로 그 시점을
보존한다. 13→12 정합은 그 이후다. *현재 상태* 를 주장하는 문장
("운영 벤치마크는 13개 태스크다")만 12 로 고쳤다.

> **식별자 검사 주의.** 초기 검사 스크립트는 (1) 언더스코어 없는 CamelCase 를
> 필터로 배제했고 (2) `rg` 서브프로세스 실패를 예외로 삼켜 결과가 공백이었다.
> "223개 전부 존재" 라는 초기 결론은 **무효**였다. 현재 스크립트
> (`ident_full.py`)는 저장소를 직접 읽어 316개를 대조한다.

### PersLay 서브스레드 검증 결과 (TDA-1 · TDA-2) — 전부 일치

| 주장 | 코드 근거 |
|---|---|
| `PersLayBlock` 5개 (Short β₀/β₁ + Long β₀/β₁/β₂) | `perslay_expert.py:352-358` |
| concat 128 + 192 + 32 + 10 = **362D** → 64D | `perslay_expert.py:315` 주석과 완전 일치 |
| `final_mlp` 362 → 128 → 64 | `perslay_expert.py:376` |
| Short 최대 **200**쌍 / Long **150**쌍 | `model_config.yaml:214-215` (코드 기본 500/300 을 yaml 이 override — 블로그가 운영값을 쓴 것이 옳다) |
| `beta_idx` 채널로 블록 라우팅 | 텐서 3번째 컬럼 `[batch, max_pairs, 3]` |
| 4D 해석용 프로젝션 | `INTERPRET_DIM = 4`, `interpret_proj` |
| `w(b,d) = |d−b|^p`, p = 1.0 | `WeightFunction(method="persistence", power=1.0)` |
| 패딩 (0,0) 이 weight 로 자동 무시 | `perslay_expert.py:245` 주석 동일 |
| 프로덕션 ρ = `sum` (attention 에서 전환) | `rho_type: "sum"` + 전환 사유 주석 |
| φ = `RationalHatPhi` | `perslay_expert.py:30` |
| raw diagram 경로 dead | `use_raw_diagram: false` — 블로그가 이미 정직하게 기술 |
| `domain_experts` 에 PersLay 를 둔 **7개 태스크** (ctr·cvr·churn·retention·life_stage·nba·spending_category) | 7개 전부 확인. `uplift`/`category_uplift` 도 포함하나 `enabled: False` 라 제외한 것이 옳다 |

유일한 수정: TDA-2 의 70D 폴백 → PersLay `input_dim` 이 2026-06-24 자로 58D 로
정정된 사실을 주석으로 가산 (§5 참조).

## 7. 검토 절차

1. **Phase 0** — 주장 대장 기계 추출(차원/클래스 수/상태 수/코드 식별자). 현재 **1,911건**.
2. **Phase 1** — 위 §3 매핑 + §2 우선순위 고정.
3. **Phase 2** — **원본 문서 단위 배치** 검증. 배치 1개 = `.typ` 1개 + 해당 ko/en 포스트.
   포스트 단위가 아니라 근거 단위로 묶어야 컨텍스트가 감당된다.

   ⚠️ **배치 안에서도 코드가 1순위다.** 참조서 대조만으로 배치를 끝내지 말 것.
   `.typ` 는 포스트가 무엇을 근거로 썼는지 알려줄 뿐이고, 참조서 자체가 노후일 수 있다
   (실제로 §5 의 734D 는 참조서가 낡은 경우였다). 배치마다 최소한 다음을 코드로 확인한다:
   - 차원·클래스 수·상태 수의 실값 (`feature_contract.py`, `task_feature_mapper.py`, `active_tasks.py`)
   - 피처 키 이름 (`configs/feature_schema.yaml`)
   - **어느 Expert/모듈이 그 블록을 소비하는가** (`DEFAULT_EXPERT_ROUTING_V2`)
   
   배치 1(p22)에서 참조서만 봤을 때는 "전부 일치" 였으나, 코드를 보고서야
   "24D 가 세 Expert(DeepFM/Causal/OT)에 644D 슬라이스로 투입된다" 는 서술이
   **사실 오류**임이 드러났다. 소비처 주장은 참조서로 검증되지 않는다.
4. **Phase 3** — ko/en 정합성(숫자·구조·주장 일치) 별도 패스.
5. **Phase 4** — 배치 단위 수정 + 배치 단위 커밋.

전체를 한 번에 읽는 통짜 검토는 금지. 블로그 1.72MB + 기술참조서 1.87MB ≈ 90만 토큰으로,
컨텍스트에 넣더라도 추론 여지가 남지 않아 검증이 아니라 훑기가 된다.
