import argparse,json,subprocess,tempfile,sys
from pathlib import Path

def run(checkpoint,data,source,out):
 subprocess.run([sys.executable,'-m','micronmt.predict','--checkpoint',checkpoint,'--input',data,'--output',str(out),'--source',source],check=True)
 with tempfile.TemporaryDirectory() as directory:
  report=str(Path(directory)/'metrics.json')
  subprocess.run([sys.executable,'-m','micronmt.evaluate','--reference',data,'--hypothesis',str(out),'--out',report],check=True)
  return json.loads(Path(report).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--ru-en',required=True);p.add_argument('--en-ru',required=True);p.add_argument('--out',default='reports/bidirectional.json');p.add_argument('--limit',type=int,default=0);a=p.parse_args();Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory() as d:
  paths=[]
  for name,source in [('ru',a.ru_en),('en',a.en_ru)]:
   target=Path(d)/(name+'-data.jsonl'); lines=Path(source).read_text(encoding='utf8').splitlines(); target.write_text('\n'.join(lines[:a.limit or None])+'\n',encoding='utf8'); paths.append(target)
  result={'checkpoint':a.checkpoint,'ru_en':run(a.checkpoint,str(paths[0]),'ru',Path(d)/'ru.jsonl'),'en_ru':run(a.checkpoint,str(paths[1]),'en',Path(d)/'en.jsonl')}
 Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
