"""Full existing RU→EN train, constant-LR continuation of diagnostic protocol; no test."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import yaml
from micronmt.model import MicroTransformer
from micronmt.checkpoint import tokenizer_metadata
from micronmt.sp_tokenizer import SentencePieceTokenizer
from scripts.diagnose_training import CONFIG, batch, evaluate, lengths, sha, dump, rows_write


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    report, run = ROOT/'reports/full_validation', ROOT/'runs/full_validation'
    if report.exists() or run.exists():
        raise SystemExit('Refusing existing full-validation outputs')
    report.mkdir(); run.mkdir()
    tokpath = ROOT/'data/tokenizer/real_baseline/shared.model'
    tok = SentencePieceTokenizer(tokpath)
    paths = {s: ROOT/f'data/real_baseline/ru_en/{s}.jsonl' for s in ('train', 'valid')}
    rows = {s: [json.loads(l) for l in p.read_text().splitlines()] for s,p in paths.items()}
    assert len(rows['train']) == 8000
    all_lengths = {s: [lengths(r,tok) for r in rs] for s,rs in rows.items()}
    max_len = next(n for n in (128,192,256,384,512,768,1024) if all(
        sum(max(pair)<=n for pair in ls)/len(ls)>=.99 for ls in all_lengths.values()))
    stats = {}
    for s,ls in all_lengths.items():
        stats[s] = {'rows':len(ls),'sha256':sha(paths[s]),'joint_fit_fraction':sum(max(p)<=max_len for p in ls)/len(ls)}
        for side,i in [('src',0),('tgt',1)]:
            values=torch.tensor([p[i] for p in ls],dtype=torch.float64)
            stats[s][side]={'quantiles':{str(q):float(torch.quantile(values,q)) for q in (0.,.5,.9,.95,.99,1.)},'truncated_rows':int((values>max_len).sum()),'truncated_tokens':int((values-max_len).clamp(min=0).sum())}
        rows_write(report/f'{s}_outliers.jsonl',[dict(index=i,lengths=p,**rows[s][i]) for i,p in enumerate(ls) if max(p)>max_len])
    dump(report/'lengths.json',dict(max_len=max_len,splits=stats,policy='smallest candidate with >=99% joint fit in each split; remaining outliers explicitly truncated, EOS appended'))
    cohorts={'diagnostic_short100':[json.loads(l) for l in (ROOT/'reports/diagnostics/valid_short.jsonl').read_text().splitlines()], 'full_first100':rows['valid'][:100]}
    for name,rs in cohorts.items(): rows_write(report/f'{name}.jsonl',rs)
    config=dict(CONFIG,max_len=max_len,init_mode='controlled')
    training=dict(batch_size=32,seed=42,updates=1500,lr=.001,label_smoothing=.1,scheduler='constant',optimizer='AdamW defaults weight_decay=.01',gradient_clip=1.,eval_every=250)
    (ROOT/'configs/full_validation_ru_en.yaml').write_text(yaml.safe_dump(dict(**config,**training),sort_keys=False))
    random.seed(42); torch.manual_seed(42)
    model=MicroTransformer(tok.vocab_size,**config)
    # Independent fork seed424242 matches B's embedding exactly, preserving constructor RNG.
    old=torch.load(ROOT/'runs/diagnostics/B/initial.pt',weights_only=True)
    parity=torch.equal(old['embedding.weight'],model.embedding.weight)
    generator=torch.Generator().manual_seed(42); batches=[]
    while len(batches)<1500:
        order=torch.randperm(len(rows['train']),generator=generator).tolist()
        batches.extend(order[i:i+32] for i in range(0,len(order),32))
    batches=batches[:1500]; dump(report/'batch_indices.json',batches)
    summary=dict(status='running',production_approved=False,license='unknown',config=config,training=training,data=stats,tokenizer_sha256=sha(tokpath),batch_sha256=sha(report/'batch_indices.json'),cohort_hashes={n:sha(report/f'{n}.jsonl') for n in cohorts},initialization=dict(embedding_matches_diagnostic_B=parity,seed=424242,rng='fork_rng preserves outer constructor RNG; dropout reset42; max_len changes position table and subsequent constructor draws, so other tensors/dropout shapes are not B parity'),test_accessed=False)
    assert parity
    torch.save(model.state_dict(),run/'initial.pt')
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    torch.manual_seed(42)
    history=[]; best=-1; selected=None
    for step in range(1501):
        if step:
            model.train(); src,tgt=batch([rows['train'][i] for i in batches[step-1]],tok,max_len)
            opt.zero_grad(); logits=model(src,tgt[:,:-1])
            loss=F.cross_entropy(logits.flatten(0,1),tgt[:,1:].flatten(),ignore_index=0,label_smoothing=.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if step%250==0:
            entry=dict(step=step,elapsed_seconds=time.perf_counter()-started,cohorts={n:evaluate(model,rs,tok,max_len) for n,rs in cohorts.items()})
            history.append(entry); rows_write(report/'history.jsonl',history)
            payload=dict(model=model.state_dict(),config=config,tokenizer=tokenizer_metadata(tok),direction='ru2en',step=step)
            torch.save(payload,run/f'step_{step:04d}.pt')
            if step and entry['cohorts']['diagnostic_short100']['chrF']>best:
                best=entry['cohorts']['diagnostic_short100']['chrF']; selected=entry
                torch.save(payload,run/'best.pt')
            print(step, 'chrF',entry['cohorts']['diagnostic_short100']['chrF'],'full100',entry['cohorts']['full_first100']['chrF'],flush=True)
    torch.save(dict(**payload,optimizer=opt.state_dict(),torch_rng_state=torch.get_rng_state()),run/'last.pt')
    summary.update(status='completed',selected_step=selected['step'],best={n:{k:v for k,v in m.items() if k!='samples'} for n,m in selected['cohorts'].items()},elapsed_seconds=time.perf_counter()-started,best_sha256=sha(run/'best.pt'),limitations=['No test accessed; both cohorts are validation, not independent quality estimates.','Full train plus length/position-table and RNG-shape changes: differences from B cannot be attributed to one factor.','Single seed, license unknown, production not approved.'])
    dump(report/'summary.json',summary)


if __name__=='__main__': main()
