import os

path = r'src\content\posts\2026-04-24-ep3-guardrails-ko.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_str = '''이 세 가지 장치 — \CLAUDE.md\ 규약, 세션 간 맥락 유지, 계약 검증 — 는 마치 AI
에이전트만을 위해 고안된 것처럼 보입니다. 하지만 실은 **인간으로 구성된 3인 팀의
협업에도 똑같이 적용**할 수 있는 원칙들입니다. 사람이 병렬로 작업할 때 흔히 겪는
문제들(변수명 불일치, 맥락 유실, 규칙이 점차 어긋나는 현상)은 AI 3기가 병렬로
작업할 때 발생하는 문제와 정확히 일치하거든요. 우리는 이 해결책을 AI 협업 과정에서
먼저 체계화했을 뿐, 그 근본은 오랫동안 이어져 온 인간 팀 협업의 지혜와 맞닿아
있습니다.

바로 이 점이 한국 금융권의 중소 규모(3~5인) 팀에도 이 구조를 성공적으로
이식할 수 있는 이유입니다. 값비싼 Claude Code 구독이나 거창한 AI 도구가 없더라도
상관없습니다. \CLAUDE.md\ 수준의 명확한 프로젝트 규약집, 팀원 간의 공유 맥락 문서,
그리고 코드를 합치기 전 인터페이스를 검증하는 루틴만으로도 충분히 제 몫을 다하니까요.
AI가 있다면 훨씬 더 빠르겠지만, **AI가 없어도 이 시스템은 견고하게 작동합니다.**'''

new_str = '''이 세 가지 장치 — \CLAUDE.md\ 규약, 세션 간 맥락 유지, 계약 검증 — 는 마치 AI
에이전트만을 위해 고안된 것처럼 들리지만, 실은 **인간 팀의 협업에도 똑같이
적용**되는 원칙이다. 사람이 병렬로 작업할 때 흔히 겪는 문제(변수명 불일치,
맥락 유실, 규칙이 점차 어긋나는 현상)는 AI 3기가 병렬로 작업할 때 발생하는
문제와 동형이다. 우리가 이 해결책을 AI 협업 과정에서 먼저 체계화했을 뿐, 그
근본은 인간 팀 협업의 오랜 지혜와 맞닿아 있다.

이게 한국 금융권의 중소 규모(3~5인) 팀에도 이 구조가 성공적으로 이식 가능한
이유다. 값비싼 Claude Code 구독이나 거창한 AI 도구가 없어도 무방하다.
\CLAUDE.md\ 수준의 명확한 프로젝트 규약집, 팀원 간의 공유 맥락 문서, 그리고
매 통합 전 인터페이스를 검증하는 루틴은 그대로 작동한다. AI가 있으면 훨씬 더
빠르겠지만, **AI가 없어도 이 시스템은 견고하다.**'''

old_str_crlf = old_str.replace('\n', '\r\n')
if old_str_crlf in content:
    content = content.replace(old_str_crlf, new_str)
elif old_str in content:
    content = content.replace(old_str, new_str)
else:
    print('Could not find the target string to replace.')
    exit(1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced successfully.')
