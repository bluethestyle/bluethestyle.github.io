---
layout: page
title: "시리즈: MRM 스레드"
permalink: /series/mrm-thread-ko/
---

# MRM 스레드

*AI 추천 시스템의 모델 리스크 관리. GARP FRM 실무자 관점에서.*

🇬🇧 [English index →](/series/mrm-thread/)

---

[*3개월간의 금융 AI 개발기*](/series/three-months-ko/) 시리즈와 병행되는 짧은 시리즈. 다른 독자층을 타겟으로 한다 — 금융권 실무자, 리스크 매니저, 감독 당국 스태프, MRM 팀.

핵심 주장: AI 추천 시스템이 단일 모델이 아니라 에이전트 파이프라인이 되면서, *검증 우선(validation-first)* MRM 모델은 무너지기 시작한다. 대안은 MRM 의무를 *아키텍처 자체* 에 밀어넣는 것 — 준수 특성을 사후 보고서가 아니라 구조적 불변성으로 만드는 것.

이 시리즈는 [Paper 2](https://doi.org/10.5281/zenodo.19622052) 에 기술된 프로덕션 시스템을 실제 사례로 삼아 이 주장을 세 각도로 다룬다.

---

## 에피소드

1. **[MRM 은 검증이 아니라 아키텍처에 속한다](/2026/04/18/mrm-ep1-architecture-ko/)** — 프레임: 모델이 LLM 에이전트 시스템일 때 무엇이 부서지는가, 아키텍처적 MRM 이 실제로 어떤 모습인가
2. *보고서가 아닌 관문으로서의 Champion-Challenger* (준비 중) — `ModelCompetition.evaluate()`, `--force-promote` override, 모든 승격 결정에 대한 HMAC 감사 엔트리
3. *에이전트 시스템과 규제적 chain of custody* (준비 중) — 7개 감사 테이블, HMAC 해시 체인 검증, EU AI Act 13-14 및 KFCPA §17 을 코드 경로로서

---

## 이 시리즈는 누구를 위한 것인가

- **GARP FRM 자격자** 중 금융 AI 또는 모델 리스크에 종사하는 분
- 은행, 카드사, 보험사의 **MRM / 2선** 팀
- 금감원, 금융위 또는 해외 상대 기관의 **감독 스태프**
- MRM 정렬 배포 패턴을 찾는 **금융 AI 엔지니어**
- "모델" 이 파이프라인일 때 SR 11-7 이 진화해야 하는 이유가 궁금한 모든 분

---

## 이 시리즈가 아닌 것

이 시리즈는 MRM 이 폐기되어야 한다거나 검증 팀이 해체되어야 한다고 주장하지 않는다. *감사의 표면* 이 "학습 후 모델 출력" 에서 "아키텍처 불변성, 지속적으로" 로 이동해야 한다고 주장한다.

검증 우선 모델은 작동한다. 지난 20년간 신용·시장 리스크 모델에 대해 작동했다. 그것이 부서지기 시작하는 이유는 그것이 틀려서가 아니라 그것이 검증하는 대상이 바뀌었기 때문이다.

---

## 원문 자료

이 시리즈의 주 원문은 [Paper 2: *From Prediction to Persuasion*](https://doi.org/10.5281/zenodo.19622052) — 특히 §5 (Operational Agent Pipeline), §6 (Regulatory Compliance), §7 (Experiments, compliance audit section). 논문에는 전체 매핑 표와 코드 참조가 포함되어 있다. 이 시리즈는 그 뒤에 있는 설계 결정을 서사화한다.

---

## 날짜별 글 목록

<ul>
{% for post in site.posts %}
  {% if post.series == "mrm-thread" and post.lang == "ko" %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
  {% endif %}
{% endfor %}
</ul>
