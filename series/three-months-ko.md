---
layout: page
title: "시리즈: 3개월간의 금융 AI 개발기"
permalink: /series/three-months-ko/
---

# 3개월간의 금융 AI 개발기

*Claude Code 를 주 개발 파트너로, 소비자용 GPU 한 대로, 3명이 개인 시간에 금융 추천 시스템을 만든 이야기. 태그: `[3개월 개발기]`*

🇬🇧 [English index →](/series/three-months/)

---

2026년 1월부터 4월까지, 우리 3명 — PM 겸 리드 아키텍트 1명과 엔지니어 2명 — 은 7개의 이종 전문가 네트워크로 구성된 13-태스크 멀티태스크 학습 추천기를 구축했다. LightGBM 으로 증류하여 Lambda 서빙에 맞게 만들고, 규제 수준의 감사 인프라를 아키텍처에 내장했다. 이 모든 것을 소비자용 GPU 한 대 (RTX 4070, 12GB VRAM), 개인 시간, Claude Code 주 개발 파트너로 해냈다.

이 시리즈는 그 이야기다.

논문 ([Zenodo](https://doi.org/10.5281/zenodo.19621884)) 은 *무엇을* 만들었는지 설명한다. [GitHub 저장소](https://github.com/bluethestyle/aws_ple_for_financial) 는 *어떻게* 작동하는지 보여준다. 이 시리즈는 그 어디에도 담기지 않는 것을 다룬다: 왜 그런 결정을 내렸는지, 무엇이 부서졌는지, 무엇이 놀라웠는지, 다시는 하지 않을 것, 그리고 3개월간의 인간-AI 협업이 실제로 어떤 느낌이었는지.

---

## 에피소드

1. **[전제 조건](/2026/04/18/ep1-premise-ko/)** — 팀, 인프라, 제약 조건, 그리고 ALS 에서 PLE 로 도달한 아키텍처 의사결정 여정
2. *파트너십* (준비 중) — 3명이 AI 협업을 어떻게 조직했는가
3. *가드레일* (준비 중) — CLAUDE.md, 메모리 뱅크, 인터페이스 계약
4. *기술적 도전* (준비 중) — 데이터 무결성, 수치 안정성, 파이프라인 엔지니어링
5. *설계 철학* (준비 중) — 왜 전문가가 7명인가
6. *데이터 무결성 감사* (준비 중) — v3에서 v4로, 결정론적 리키지, HGCN vs LightGCN
7. *아키텍처보다 중요했던 버그* (준비 중) — Uncertainty weighting, Softmax vs Sigmoid, 스케일에서의 adaTT, GradSurgery
8. *결과와 교훈* (준비 중) — 무엇을 만들었나, 무엇을 배웠나

---

## 병행 시리즈

이 시리즈의 전반부가 완료되면 두 번째 짧은 시리즈 [*MRM 스레드*](/series/mrm-thread-ko/) 가 병행된다. 규제·모델 리스크 관리 각도로 다룬다: 왜 MRM 이 검증이 아니라 아키텍처에 속하는가, Champion-Challenger 가 보고서가 아닌 관문으로 작동하는 방식, SR 11-7·EU AI Act·AI 기본법 매핑이 실무에서 어떤 모습인가. 다른 독자층 — 금융권 실무자, 리스크 매니저, 감독 당국 스태프 — 을 타겟으로 하지만 같은 프로젝트에서 나온다.

---

## 원문 자료

이 시리즈는 연구 저장소의 [development_story.pdf](https://github.com/bluethestyle/aws_ple_for_financial/blob/main/docs/typst/ko/development_story.pdf) (그리고 영문판) 를 재구성·확장한 것이다. 개발 스토리는 프로젝트 이력에 대한 가장 완결된 내부 기록이다. 한 문서로 읽기를 원하시면 PDF 가 있다.

시리즈 형식은 연속성을 조금 희생하는 대신 호흡 공간을 제공한다 — 각 에피소드는 5-10분 읽기 분량, 한 번에 한 주제.

---

## 날짜별 글 목록

<ul>
{% for post in site.posts %}
  {% if post.series == "three-months" and post.lang == "ko" %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small> — {{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
  {% endif %}
{% endfor %}
</ul>
