import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from micronmt.api import Translator
ckpt = os.environ.get('CKPT', 'runs/realbaseline_128x2_80c_ru_en/best.pt')
tr = Translator(ckpt)
print('Model loaded. Type text (ru->en by default). Ctrl+C to exit.', flush=True)
for line in sys.stdin:
    text = line.strip()
    if not text:
        continue
    try:
        res = tr.translate(text, 'ru', 'en')
        print('>>', res['translation'], flush=True)
    except Exception as e:
        print('ERROR:', e, flush=True)
