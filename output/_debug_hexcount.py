from pathlib import Path
import re
p = Path(__file__).resolve().parent.parent / 'iv_data.txt'
print('path:', p)
try:
    s = p.read_text(encoding='utf-8')
except Exception as e:
    print('read_text error:', e)
    s = p.read_bytes().decode('utf-8', errors='replace')
print('chars:', len(s))
print('lines:', s.count('\n')+1)
pairs = re.findall(r"[0-9A-Fa-f]{2}", s)
print('pairs:', len(pairs))
print('first20:', pairs[:20])
