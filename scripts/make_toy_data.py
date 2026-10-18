import argparse,json,random
from pathlib import Path
pairs=[('Привет, мир!','Hello, world!'),('Я решил остаться здесь.','I decided to stay here.'),('Разверните приложение.','Deploy the application.'),('Это сложная задача.','This is a difficult task.')]
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True)
for name,rows in [('train',pairs*100),('valid',pairs)]:
 with open(Path(a.out)/(name+'.jsonl'),'w') as f:
  for s,t in rows: f.write(json.dumps({'src':s,'tgt':t},ensure_ascii=False)+'\n')
