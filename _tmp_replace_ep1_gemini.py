import codecs
import re

file_path = "src/content/posts/2026-04-18-ep1-premise-ko.md"

with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

old_str = "이를 위해 우리는 무려 11개의 서로 다른 학문 분야에서 수학적 동형사상(Isomorphism)의 구조를 빌려왔다. 하이퍼볼릭 기하학, 화학 동역학, SIR 전염병 확산 모델, 최적 수송(Optimal Transport) 이론, 지속 호몰로지(Persistent Homology) 등 각 분야가 자기만의 도메인에서 이미 훌륭하게 풀어낸 구조적 해법들을, 우리는 아주 저렴한 연산 비용으로 우리 시스템에 수입해 올 수 있었다."

new_str = "이를 위해 우리는 무려 11개의 서로 다른 학문 분야에서 **구조적 동형사상(Isomorphism)**의 개념을 빌려왔다. 재미있게도 이 아이디어는 다른 어떤 AI 모델도 아닌, **Gemini(제미나이)와의 끈질긴 아이디에이션(Ideation) 과정**을 통해 새롭게 발굴해 낸 핵심 해법이었다. 하이퍼볼릭 기하학, 화학 동역학, SIR 전염병 확산 모델, 최적 수송(Optimal Transport) 이론, 지속 호몰로지(Persistent Homology) 등 각 분야가 자기만의 도메인에서 이미 훌륭하게 풀어낸 구조적 해법들을, 우리는 아주 저렴한 연산 비용으로 우리 시스템에 수입해 올 수 있었다."

if old_str in content:
    content = content.replace(old_str, new_str)
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("String not found")
