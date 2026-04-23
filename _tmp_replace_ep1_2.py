import codecs
import re

file_path = "src/content/posts/2026-04-18-mrm-ep1-architecture-ko.md"

with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

old_str = '''위에서 언급한 다섯 가지 속성은 사실 애초 프로젝트 계획에는 없던 것들이다. 원래 계획은 아주 관습적이었다. 모델을 만들고, 검증 리포트를 작성해서, 분기마다 MRM 팀에 넘기는 식이었다. 방향을 바꾸게 된 계기는 에이전트 파이프라인이 점차 형태를 갖춰가면서 마주친 구체적인 실패 사례들 때문이었다. 

첫 번째 Reason Generator 테스트에서 *그럴듯해 보이지만 사실은 틀린* 추천 설명이 생성되었을 때, 우리는 "만약 이대로 출시했다면 MRM 팀이 과연 이 오류를 잡아낼 수 있었을까?"라고 자문했다. 솔직한 대답은 "못 잡는다"였다. 그들은 결과로 나온 출력물만 볼 뿐, 그 출력물이 생성되는 내부 과정까지 들여다보지는 않기 때문이다. 바로 이 지점에서 "구조적 출력으로서의 설명 가능성"이라는 개념이 탄생했다.'''

new_str = '''위에서 언급한 다섯 가지 속성은 사실 애초 프로젝트 계획에는 없던 것들이다. 원래 계획은 아주 관습적이었다. 전담 MRM 조직조차 없는 현실 속에서, 일단 모델을 만들고 형식적인 검증 리포트를 작성해 분기마다 관련 부서(리스크 관리나 컴플라이언스)나 위원회에 서면으로 보고하는 식이었다. 방향을 바꾸게 된 계기는 에이전트 파이프라인이 점차 형태를 갖춰가면서 마주친 구체적인 실패 사례들 때문이었다. 

첫 번째 Reason Generator 테스트에서 *그럴듯해 보이지만 사실은 틀린* 추천 설명이 생성되었을 때, 우리는 "만약 이대로 출시했다면 사후 검토를 맡은 리스크 담당자가 과연 이 오류를 잡아낼 수 있었을까?"라고 자문했다. 솔직한 대답은 "못 잡는다"였다. 사후 검토자는 결과로 나온 출력물만 볼 뿐, 그 출력물이 생성되는 내부 과정까지 속속들이 들여다보지는 않기 때문이다. 바로 이 지점에서 "구조적 출력으로서의 설명 가능성"이라는 개념이 탄생했다.'''

# normalize spaces for matching just in case
old_str_normalized = re.sub(r'\s+', '', old_str)

def build_regex(s):
    return r'\s*'.join(map(re.escape, s.split()))

regex = build_regex(old_str)
if re.search(regex, content):
    content = re.sub(regex, new_str.replace('\\', '\\\\'), content)
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('Could not find the target string.')
