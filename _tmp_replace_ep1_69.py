import codecs
import re

file_path = "src/content/posts/2026-04-18-ep1-premise-ko.md"

with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

old_str = "아무리 합산 경력이 10년이 넘고 의지가 충만하다 한들, 3명이 RTX 4070 한 대만으로 80개의 DAG가 얽힌 ALS 시스템을 3개월 만에 대체할 수는 없는 노릇이다."
new_str = "아무리 합산 경력이 10년이 넘고 의지가 충만하다 한들, 3명이 RTX 4070 한 대만으로 기존 ALS 시스템을 걷어내고 **80개의 DAG가 얽힌 거대한 새 시스템을 3개월 만에 구축해 낼 수는** 없는 노릇이다."

if old_str in content:
    content = content.replace(old_str, new_str)
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("String not found")
