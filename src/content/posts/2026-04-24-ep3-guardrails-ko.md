---
title: "[3개월 개발기] 에피소드 3 — 우리의 적응 방식: 가드레일·메모리 뱅크·계약 검증"
date: 2026-04-24 12:00:00 +0900
categories: [FinAI Build]
tags: [finai-build, claude-code, architecture, financial-ai]
lang: ko
excerpt: "3인 × AI 팀이 병렬로 굴러가면서도 통합 지점에서 깨지지 않게 한 실제 장치들 — CLAUDE.md 헌법 4조, 8개 메모리 뱅크 파일, 매 병렬 작업 후 인터페이스 키 검증의 구체 모습."
series: three-months
part: 3
alt_lang: /2026/04/24/ep3-guardrails-en/
next_title: "에피소드 4 — 일곱 전문가: 11개 학문에서 구조적 동형사상을 수입하다"
next_desc: "HGCN · PersLay · Causal · OT · Temporal Ensemble · DeepFM · LightGCN 일곱 선택이 어떻게 도출됐나 — 각 전문가가 메우는 수학적 빈틈과 기각된 후보들."
next_status: draft
source_url: https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf
source_label: "개발 스토리 (KO, PDF) §3"
---

*"3개월 개발기" 3편. Ep 2 에서 3인 × AI 팀이 병렬로 모듈을
만들었다고 썼다. 이번 편은 그 병렬이 통합 지점에서 깨지지 않게 한
실제 장치들 — CLAUDE.md 헌법, 메모리 뱅크, 인터페이스 계약 검증 —
의 구체적 모습.*

## 3일 뒤 통합의 풍경

프로젝트 초기 어느 목요일. 3개 팀이 화요일부터 각자 AI 에이전트에
작업을 맡겼다. PM 팀은 config_builder 와 PLEConfig 스키마를,
엔지니어 1 팀은 feature_schema 를 생성하는 Phase 0 파이프라인을,
엔지니어 2 팀은 그 둘을 읽어 모델을 구성하는 train.py 초안을.

3명이 목요일 오후에 모여 통합을 시도했다. 결과 — 30분 만에 런타임
에러 5개. 모두 *같은 원인* 의 변형이었다. PM 팀의 스키마는 `group_ranges`
라는 키로 feature 그룹 경계를 저장했다. 엔지니어 1 팀의 파이프라인은
`feature_group_ranges` 로 저장했다. 엔지니어 2 팀의 train.py 는
또 다른 이름 `ranges_by_group` 으로 읽었다. 셋 다 *그럴듯한* 이름이다.

이게 병렬 AI 작업의 고유 실패 모드다. 각 에이전트는 자기 맥락에서
가장 합리적인 이름을 선택한다. 세 맥락의 합리적 이름 셋이 일치할
확률은 0 에 수렴한다.

해결책은 두 가지가 있다. 하나 — 병렬을 포기하고 직렬로 돌린다 (3일
걸릴 작업이 9일). 둘 — 병렬이 수렴하도록 *사전 장치* 를 깐다. 우리가
택한 둘째 접근이 이번 편의 주제다.

## CLAUDE.md 헌법 — 코드보다 먼저

프로젝트 루트의 `CLAUDE.md` 는 Phase C (Cursor 기반 환경 정비)
단계에서 *코드 한 줄 작성되기 전에* 쓰였다. 순서가 의도적이었음을
Ep 2 에서 짚었는데, 실제 그 파일이 뭘 담고 있는지가 이번 편이다.

현재 버전은 17개 섹션 이상으로 확장돼 있지만, 초기 4개 핵심 조항은
이것이다:

**§1.1 Config-Driven 원칙.** 모든 파라미터는 YAML config 에서
읽는다. Python 코드에 컬럼명·경계값·시나리오 목록·AWS 상수를 하드
코딩하지 않는다. `configs/pipeline.yaml` (공통) + `configs/datasets/{name}.yaml`
(데이터셋별) 의 split-config 패턴을 쓴다.

왜 이게 *헌법급* 인가 — AI 에이전트는 "그럴듯한 상수" 를 코드에
박아넣는 데 너무 능숙하다. `batch_size=5632` 같은 숫자가 주석 없이
train.py 에 박히면 3주 뒤 왜 그 값인지 아무도 모른다. Config 에
강제 위치시키면 *문서화와 변경 가능성이 한 번에 해결* 된다.

**§1.2 관심사 분리 (Separation of Concerns).** Adapter 는 raw →
standardized DataFrame 만. PipelineRunner 는 전처리·피처 생성·레이블
파생·정규화·텐서 저장 (Phase 0 전체). train.py 는 training-ready
데이터 로드 → 모델 빌드 → 학습만. 500 줄 넘으면 관심사 분리가 안
된 것으로 판정.

위 목요일 통합 사고의 근원이 여기 있었다. 엔지니어 2 팀이 train.py
안에 전처리 일부를 인라인으로 박았고, 그 전처리가 엔지니어 1 팀의
Phase 0 이 생성한 스키마와 다르게 동작했다. §1.2 를 사후에 강제
하면서 train.py 의 전처리 230 줄이 Phase 0 으로 이관됐다.

**§1.3 데이터 리키지 방지.** Scaler 는 TRAIN split 에서만 fit.
Temporal split 에 gap_days 필수 (최소 7일). LeakageValidator 를
학습 전에 반드시 호출. 이 조항은 Ep 5 (데이터 무결성 사냥) 에서
자세히 다룬다.

**§4 코드 검수 4단계.** "완료" 보고 기준 — (a) 수정된 모든 `.py` 에
`py_compile.compile(f, doraise=True)` 실행 (b) 인터페이스 계약 검증
— A 가 저장하는 키와 B 가 읽는 키가 일치하는지 (c) 하드코딩 스캔 —
컬럼명·AWS 상수·매직 넘버 검색 (d) 관심사 분리 검증.

## 메모리 뱅크 — 세션 사이의 연속성

`CLAUDE.md` 가 프로젝트의 헌법이라면, `.claude/memory-bank/` 의 8개
파일은 *세션 간 기억 장치* 다. Claude Code 의 맥락 창 (1M 토큰) 이
크긴 하지만 세션 사이를 넘어가지 않는다. 새 세션을 열 때마다 에이전트
에게 이전 상태를 재설명하면 며칠 안에 말라붙는다.

8 파일의 역할 분담:

- `projectbrief.md` — 프로젝트 목표와 제약 (온프렘 시스템 대체,
  3인 팀, RTX 4070)
- `activeContext.md` — "지금 무엇을 하고 있는가" — 매 세션 시작
  에이전트가 가장 먼저 읽는 파일
- `progress.md` — 이정표 체크리스트 (Phase 0 완료, 13-task 축소
  완료, ablation v4 완료 등)
- `techContext.md` — 기술 스택·버전·호환성 노트
- `productContext.md` — 비즈니스 맥락·고객 세그먼트·규제 요구사항
- `systemPatterns.md` — 반복 출현하는 설계 패턴 (logger 사용법,
  exception handling, async 호출 규약)
- `tasks.md` — 현재 in-progress · 차단 · 완료 태스크
- `style-guide.md` — 코드 스타일·이름 규칙·커밋 메시지 포맷

8 파일 구조가 나온 건 *경험적으로*, "한 파일에 모두 담으면 첫 세션
에서 에이전트가 overwhelming 해서 핵심을 놓친다" 는 것을 몇 주 안에
배우고 난 뒤였다. 파일을 쪼개면 에이전트가 첫 두세 파일에서 핵심 맥락
을 확보하고, 필요할 때 나머지를 조회하는 흐름이 생긴다.

## .claude/RULES.md 와 .cursorrules 의 동기화

여러 AI 도구를 쓰면 각 도구가 자기 규칙 파일을 가진다 — Claude Code
는 `CLAUDE.md`, Cursor 는 `.cursorrules`, .claude 하위에 또 `RULES.md`.
내용이 겹치지만 포맷이 다르다. 수동 동기화를 하면 몇 주 만에 어긋나서
Cursor 에이전트가 Claude Code 가 따르는 규칙 중 일부를 지키지 않는
상황이 생긴다.

해결은 *`.claude/RULES.md` 를 canonical source* 로 두고 다른 파일들이
거기서 파생되는 구조. 실제로는 수동 복사 + 차이 체크 스크립트로 유지.
더 깔끔한 자동화가 가능하지만 3인 팀의 우선순위에서는 이 수준이
적당하다 — 100% 자동화보다 "어긋났을 때 10분 안에 감지" 가 실용 목표.

## 인터페이스 계약 검증 — 병렬 작업 후의 필수 루틴

위에서 언급한 `group_ranges` / `feature_group_ranges` / `ranges_by_group`
혼란을 다시는 겪지 않도록, 모든 병렬 작업 세션 종료 시 다음 루틴이
돌아간다:

1. 수정된 각 파일의 `save_*` / `write_*` 함수가 어떤 키로 dict 를
   저장하는지 grep 으로 추출
2. 같은 브랜치의 다른 파일의 `load_*` / `read_*` 가 어떤 키로 읽는지
   추출
3. 두 집합의 diff 를 확인 — 저장만 하고 안 읽히는 키, 읽으려는데
   없는 키, 양쪽에 있지만 이름이 다른 키

이 루틴을 AI 에이전트에게 위임한다. 병렬 에이전트 2-3개가 작업을
끝내면, *네 번째 에이전트* 가 인터페이스 계약 검증 전용으로 호출된다.
병렬 에이전트들은 각자의 맥락에서 "그럴듯한" 이름을 선택하지만, 검증
에이전트는 *전체를 조망* 하는 맥락에서 불일치를 식별한다.

이 네 번째 에이전트의 가치는 중요하다. 없으면 목요일 통합 사고가
반복되고, 디버깅 3시간이 매주 소모된다. 있으면 그 3시간이 *20분* 으로
줄어든다 — 불일치가 CI 단계 훨씬 전에 발견되기 때문.

## 3개 플랫폼 실험 브랜치

마지막으로 언급할 장치 — `exp/claude-auto-*`, `exp/codex-auto-*`,
`exp/vertex-auto-*` 세 계열의 자동화 실험 브랜치. 같은 요청을 세
플랫폼에 동시 던져 결과를 비교하는 용도다.

이게 단순히 "어느 게 더 좋은가" 답을 구하려는 게 아니다. 세 플랫폼이
*서로 다른 지점에서 실패* 하기 때문에, 그 실패 패턴 자체가 프로젝트의
약점 진단이 된다. 예를 들어 Claude Code 는 복잡한 수학 유도에서
강하지만 YAML 파싱 세부에서 때때로 틀린다. Codex 는 반대. 세 결과를
교차하면 "우리 설명이 부족한 부분" 이 드러난다.

## 가드레일이 AI 협업에만 쓰이는 건 아니다

이 세 장치 — CLAUDE.md 헌법, 메모리 뱅크, 계약 검증 — 는 AI 에이전트
때문에 만들어진 것처럼 들리지만, 실은 *3인 팀 협업에도 그대로 적용*
된다. 3명이 병렬로 작업할 때의 문제 (이름 불일치, 맥락 유실, 규칙
drift) 는 AI 3명 병렬의 문제와 동형이다. 우리가 이걸 AI 협업 맥락에서
먼저 체계화했을 뿐, 원칙은 인간 팀 협업의 old wisdom 과 같다.

이게 한국 금융권 중소 규모 팀 (3-5명 규모) 에 이 구조가 이식 가능한
이유다. Claude Code 구독이 없어도 CLAUDE.md + 메모리 뱅크 + 계약
검증 루틴은 그대로 설치된다. AI 가 있으면 더 빠르고 없어도 작동한다.

## 다음 편

Ep 4 는 이 가드레일 위에서 실제로 구성된 아키텍처 — 7개 이종
전문가 네트워크 — 가 어떻게 도출됐는지 다룬다. HGCN (하이퍼볼릭
계층), PersLay (지속 호몰로지), Causal (구조적 인과), OT (최적
수송), Temporal Ensemble, DeepFM, LightGCN. 왜 이 7개인가. 기각된
후보들은 무엇이었나. 11개 학문 분야에서 구조적 동형사상을 수입한
과정.

원문 자료:
[개발 스토리 (KO, PDF)](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf)
§3 "품질 관리 전략".
