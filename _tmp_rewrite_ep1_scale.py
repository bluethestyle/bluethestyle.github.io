import codecs
import re

file_path = "src/content/posts/2026-04-18-ep1-premise-ko.md"

with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

old_block = '''## 프로덕션 시스템의 실제 규모

추상적인 설계 이야기를 꺼내기 전에, 우리가 대체하려 했던 기존 시스템이 도대체 어느 정도 규모였는지 숫자로 짚고 넘어가자.

우리가 걷어내려 했던 온프레미스(On-premise) 프로덕션 시스템은 결코 장난감 수준이 아니었다.

- 80개 이상의 Airflow DAG
- 챔피언-챌린저(Champion-Challenger) 모델 경쟁 체제
- 매주 돌아가는 자동 재학습 파이프라인
- 734차원의 피처 텐서(Feature Tensor)
- 18개 태스크 동시 처리
- 62개 데이터 테이블의 인제스천(Ingestion)

공개된 AWS 벤치마크 버전은 이보다 조금 작게 축소되어 있다(결정론적 리키지 및 중복 태스크 5개를 제거하여 13개 태스크로 줄였고, 피처 역시 734차원에서 349차원으로 축소했다). 하지만 그 근간을 이루는 아키텍처와 엔지니어링 패턴은 동일하다.

이 거대한 규모의 시스템을 고작 3명의 인원과 데스크톱 GPU 한 대만으로 구축하고 대체하겠다는 계획은, 서류상으로만 보면 당장 기각당할 만한 무모한 조합이다. 하지만 이를 현실로 만든 것은 바로 **AI 증강 개발(AI-augmented development)**이었다. 그렇다면 그 "3명과 데스크톱 GPU"의 실체는 정확히 무엇이었을까.'''

new_block = '''## 우리가 구축한 시스템의 실제 규모

추상적인 설계 이야기를 꺼내기 전에, 우리가 과거의 ALS를 밀어내고 새롭게 구축한 시스템이 도대체 어느 정도 규모였는지 숫자로 짚고 넘어가자.

우리가 사내 온프레미스(On-premise) 환경에 정식 프로덕션으로 띄운 새로운 추천 시스템(이후 오픈소스로 공개된 AWS 버전의 오리지널)은 결코 장난감 수준이 아니었다.

- 80개 이상의 Airflow DAG
- 챔피언-챌린저(Champion-Challenger) 모델 경쟁 체제
- 매주 돌아가는 자동 재학습 파이프라인
- 734차원의 피처 텐서(Feature Tensor)
- 18개 태스크 동시 처리
- 62개 데이터 테이블의 인제스천(Ingestion)

(현재 오픈소스로 공개된 AWS 벤치마크 버전은 이보다 조금 축소되어 있다. 결정론적 리키지 및 중복 태스크 5개를 제거하여 13개 태스크로 줄였고, 피처 역시 734차원에서 349차원으로 다이어트했다. 하지만 그 근간을 이루는 아키텍처와 엔지니어링 패턴은 온프레미스 원본과 완전히 동일하다.)

이 거대한 규모의 시스템을 처음부터 끝까지 고작 3명의 인원과 데스크톱 GPU 한 대만으로 구축해 냈다는 사실은, 서류상으로만 보면 당장 기각당할 만한 무모한 소리처럼 들린다. 하지만 이를 현실로 만든 것은 바로 **AI 증강 개발(AI-augmented development)**이었다. 그렇다면 그 "3명과 데스크톱 GPU"의 실체는 정확히 무엇이었을까.'''

content = content.replace(old_block.replace('\n', '\r\n'), new_block)
content = content.replace(old_block, new_block)

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(content)
print("Rewrite complete")
