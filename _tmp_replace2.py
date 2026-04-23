import os
import re

path = r'src\content\posts\2026-04-24-mrm-ep3-chain-of-custody-ko.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_str = '''파트너 기관과의 실 트래픽 수집은 2026-04-30 에 시작됐으니, 위 재구성
속성이 아직 대규모에서 검증된 건 아니다. 존재하는 건 설계다. 설계가
어떻게 협조하는지를 보려면 한국 영업점의 전형적 시나리오를 따라가면
된다 — 만기 정기예금 재예치를 위해 내점한 고객에게 추천 시스템이
적금 + 예금 조합을 제안하는 케이스.

추론 경로에서 4개 테이블이 순차적으로 건드려진다:

- \log_data_access\ 가 먼저 발화한다. 영업점 직원이 인증 후 고객을
  대신해 요청을 제출할 때. operator ID, 고객 ID, 접근 사유
  ("영업점 발단 추천") — PIPA §37의2 감사 연결점.
- \log_model_inference\ — 증류된 LightGBM 이 13개 태스크를 실행해
  점수를 산출할 때 발화. 모델 버전 포인터와 피처 텐서 해시가
  기록되어 이후 재구성 시 "어느 모델이 어떤 입력을 봤는가" 가
  pinning 된다.
- \log_attribution\ — 직후 발화, 상위 K 피처 기여도 + expert gate
  가중치. EU AI Act Article 13 과 금소법 §17 의 "왜 이 고객에게 이
  추천을" 에 답하는 자료.
- \log_guardrail\ — Safety Gate 에이전트가 설명을 규제·적합성·환각
  ·어조·사실성 5개 기준으로 검토할 때 발화. 판정 (pass/modify/block)
  과 기준별 점수가 결과 무관하게 기록된다.

추론 전체 경로는 sub-second. 4개 감사 쓰기는 오버헤드가 미미한데,
각각이 작은 canonical JSON payload 를 append 하는 것이기 때문 (무거운
네트워크 왕복이 아니다).

나머지 3개 테이블은 요청별로 건드리지 않는다. \log_operation\ 은
시스템 상태 전이 (야간 재학습 시작·종료, 서빙 매니페스트 교체) 시,
\log_dimension_change\ 는 피처 스키마 변화 (예: Phase 0 개정 후
349D → 403D) 시, \log_model_promotion\ 은 Champion-Challenger 판정
(Ep 2) 시. 이들은 드문 이벤트라 매 요청 오버헤드에 들어가지 않는다.

Ep 3 서두의 시나리오 — 15개월 뒤 감독 당국 쿼리 — 가 현실이 되면,
재구성 조인은 위 4개 테이블에서 읽은 뒤 관련 \log_operation\ /
\log_model_promotion\ 엔트리로 거슬러 올라가 "어느 시스템 버전이
이 추천을 냈는가" 를 특정한다. 이 조인은 Parquet 아카이브에서 수초
내 실행되지, 수 일의 수동 재구성이 아니다.'''

new_str = '''물론 파트너 기관과의 실제 트래픽 수집은 2026-04-30에야 시작되었으므로, 위에서 설명한 재구성
속성이 아직 대규모 운영 환경에서 완벽히 검증된 것은 아니다. 현재 존재하는 것은 설계뿐이다. 이
설계가 실제 환경에서 어떻게 작동하는지 이해하기 위해 한국 영업점의 전형적인 시나리오를 하나
따라가 보자. 만기된 정기예금을 재예치하러 내점한 고객에게 추천 시스템이 '적금 + 예금 조합'을
제안하는 상황이다.

이 추론 과정에서는 4개의 테이블이 순차적으로 기록된다.

- \log_data_access\가 가장 먼저 기록된다. 영업점 직원이 인증을 마치고 고객을 대신해 추천을
  요청할 때 발생한다. 작업자 ID, 고객 ID, 접근 사유("영업점 발단 추천")가 남으며, 이는
  개인정보보호법(PIPA) 제37조의2를 충족하기 위한 감사 연결점이 된다.
- \log_model_inference\는 경량화(Distillation)된 LightGBM 모델이 13개 태스크를 실행해
  점수를 산출할 때 기록된다. 모델 버전 포인터와 피처 텐서(Feature Tensor) 해시값이 남기
  때문에, 나중에 "어떤 모델이 어떤 입력값을 보고 판단했는가"를 정확히 추적할 수 있다.
- \log_attribution\은 직후에 기록되며, 상위 K개 피처의 기여도와 전문가 게이트(Expert Gate)
  가중치를 담는다. 이는 EU AI 법안 제13조와 금융소비자보호법 제17조에서 요구하는 "왜 이
  고객에게 이런 추천을 했는가"에 답하기 위한 근거 자료다.
- \log_guardrail\은 안전 게이트(Safety Gate) 에이전트가 규제 준수·적합성·환각 여부·어조·
  사실성 등 5가지 기준으로 추천 설명을 검토할 때 기록된다. 최종 판정(pass/modify/block)과
  기준별 점수가 결과와 무관하게 모두 남는다.

전체 추론 과정은 1초도 걸리지 않는다(sub-second). 4개의 감사 로그를 남기지만 오버헤드는
아주 미미한 수준이다. 무거운 네트워크 통신을 발생시키는 것이 아니라, 각각 작고 표준화된
JSON 페이로드를 덧붙이는(append) 방식이기 때문이다.

나머지 3개 테이블은 개별 요청마다 기록되지 않는다. \log_operation\은 시스템 상태가 변할 때
(야간 재학습 시작 및 종료, 서빙 매니페스트 교체 등), \log_dimension_change\는 피처 스키마가
변경될 때(예: Phase 0 개정 후 349D에서 403D로 변경), \log_model_promotion\은 챔피언-챌린저
판정(Ep 2 참고)이 일어날 때 각각 기록된다. 이들은 드물게 발생하는 이벤트이므로, 매 요청마다
오버헤드를 발생시키지 않는다.

Ep 3 서두에서 언급한 "15개월 뒤 감독 당국의 쿼리" 상황이 현실이 된다면 어떨까? 시스템은 위
4개 테이블의 기록을 읽어들인 뒤, 관련된 \log_operation\ 및 \log_model_promotion\ 항목으로
거슬러 올라가 "당시 어떤 버전의 시스템이 이 추천을 내놓았는가"를 특정해 낸다. 이 모든 조인(Join)
과정은 누군가 며칠씩 매달려 수동으로 재구성하는 것이 아니라, Parquet 아카이브에서 단 몇 초
만에 자동으로 실행된다.'''

old_str_normalized = re.sub(r'\s+', '', old_str)
new_str_normalized = new_str

content_normalized = re.sub(r'\s+', '', content)

def build_regex(s):
    return r'\s*'.join(map(re.escape, s.split()))

regex = build_regex(old_str)
if re.search(regex, content):
    content = re.sub(regex, new_str, content)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Replaced successfully.')
else:
    print('Could not find the target string to replace.')
