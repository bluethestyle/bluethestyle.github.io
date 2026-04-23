import codecs
import re

file_path = "src/content/posts/2026-04-18-mrm-ep1-architecture-ko.md"

with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

old_str = "1. **공격 표면이 다단계로 이루어져 있다.** 환각 현상은"
new_str = "1. **오류와 위험이 발생하는 지점이 다단계로 나뉘어 있다.** 환각 현상은"

if old_str in content:
    content = content.replace(old_str, new_str)
    with codecs.open(file_path, 'w', 'utf-8') as f:
        f.write(content)
    print('Replaced successfully')
else:
    print('Could not find the target string.')
