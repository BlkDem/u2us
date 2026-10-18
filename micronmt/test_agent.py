"""Automated translation test agent: runs functional, regression and quality checks."""
import argparse, json, time
from pathlib import Path
from .api import Translator

DEFAULT_CASES = [
    {'id':'basic_ru_en','src':'Привет, мир!','source':'ru','target':'en','expected':['Hello, world!','Hello world!']},
    {'id':'basic_en_ru','src':'Hello, world!','source':'en','target':'ru','expected':['Привет, мир!','Привет мир!']},
    {'id':'context','src':'I decided to stay there.','context':'I booked a room in a small hotel.','source':'en','target':'ru','expected':[]},
    {'id':'identifier','src':'Deploy API_ENDPOINT on NVIDIA hardware.','source':'en','target':'ru','must_contain':['API_ENDPOINT','NVIDIA']},
]

def run(checkpoint, cases, beam_size=1, terminology=None):
    translator=Translator(checkpoint, terminology=terminology); results=[]
    for case in cases:
        started=time.perf_counter()
        try:
            result=translator.translate(case['src'],case['source'],case['target'],case.get('context'),beam_size=beam_size)
            text=result['translation']; lower=text.lower(); ok=True; reasons=[]
            expected=[x.lower() for x in case.get('expected',[])]; must=[x.lower() for x in case.get('must_contain',[])]
            if expected and not any(x in lower for x in expected): ok=False; reasons.append('expected_phrase_missing')
            missing=[x for x in must if x not in lower]
            if missing: ok=False; reasons.append('protected_token_missing:'+','.join(missing))
            results.append({'id':case['id'],'ok':ok,'translation':text,'latency_ms':(time.perf_counter()-started)*1000,'reasons':reasons})
        except Exception as exc:
            results.append({'id':case['id'],'ok':False,'translation':'','error':str(exc),'reasons':['exception']})
    return results

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--cases'); p.add_argument('--out'); p.add_argument('--beam-size',type=int,default=1); p.add_argument('--terms'); a=p.parse_args()
 cases=DEFAULT_CASES if not a.cases else json.loads(Path(a.cases).read_text(encoding='utf8'))
 terms=json.loads(Path(a.terms).read_text(encoding='utf8')) if a.terms else None
 results=run(a.checkpoint,cases,a.beam_size,terms); passed=sum(x['ok'] for x in results); report={'checkpoint':a.checkpoint,'passed':passed,'total':len(results),'pass_rate':passed/max(1,len(results)),'results':results}
 print(json.dumps(report,ensure_ascii=False,indent=2));
 if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 raise SystemExit(0 if passed==len(results) else 1)
if __name__=='__main__': main()
