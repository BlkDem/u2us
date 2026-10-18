"""Generate English inputs, translate them with the local model, and write an E2E report."""
import argparse, json, time
from pathlib import Path
from micronmt.api import Translator

DEFAULT_TEXTS = [
    'The deployment pipeline failed after the configuration was changed.',
    'I decided to stay at the hotel for another two nights.',
    'If the service becomes unavailable, the application should retry the request.',
    'The report points out that the results are difficult to interpret without context.',
    'Please keep the API_ENDPOINT unchanged during the migration.',
    'The team is looking into why the database connection drops under heavy load.',
    'The contract requires the supplier to notify us before making any material changes.',
    'Although the solution is compact, it must preserve the meaning of long and ambiguous sentences.',
    'She said that the issue had already been escalated to the infrastructure team.',
    'The new model should run locally without sending user data to an external service.',
]

def main():
    p=argparse.ArgumentParser(description='End-to-end offline RU/EN model test')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='reports/e2e.jsonl')
    p.add_argument('--report',default='reports/e2e_report.json')
    p.add_argument('--texts-file',help='Optional UTF-8 file with one English sentence per line')
    p.add_argument('--beam-size',type=int,default=1)
    p.add_argument('--max-len',type=int)
    a=p.parse_args()
    texts=[x.strip() for x in Path(a.texts_file).read_text(encoding='utf8').splitlines() if x.strip()] if a.texts_file else DEFAULT_TEXTS
    translator=Translator(a.checkpoint,max_len=a.max_len); results=[]
    for i,text in enumerate(texts):
        started=time.perf_counter(); error=None; translation=''
        try: translation=translator.translate(text,source='en',target='ru',max_len=a.max_len,beam_size=a.beam_size)['translation']
        except Exception as exc: error=str(exc)
        results.append({'id':i+1,'source_language':'en','target_language':'ru','source':text,'translation':translation,'latency_ms':round((time.perf_counter()-started)*1000,3),'error':error})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in results)+'\n',encoding='utf8')
    ok=sum(not x['error'] and bool(x['translation'].strip()) for x in results); report={'checkpoint':a.checkpoint,'total':len(results),'successful':ok,'success_rate':ok/max(1,len(results)),'avg_latency_ms':sum(x['latency_ms'] for x in results)/max(1,len(results)),'output':a.output}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True);Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2)); raise SystemExit(0 if ok==len(results) else 1)
if __name__=='__main__':main()
