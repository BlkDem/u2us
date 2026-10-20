"""Offline RU→EN audit and matched initialization experiment; never reads test."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import sacrebleu
from micronmt.model import MicroTransformer
from micronmt.sp_tokenizer import SentencePieceTokenizer
from scripts.train_controlled import repetition

CONFIG = dict(d_model=64, nhead=4, encoder_layers=2, decoder_layers=2,
              dim_feedforward=128, dropout=.1, max_len=64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def rows_write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC', s).split())


def lengths(row, tok):
    return [len(tok.processor.encode(prefix+row[key]))+2 for key,prefix in [('src','<ru2en> '),('tgt','')]]


def select_short(rows, tok, count):
    return [dict(r, original_index=i) for i,r in enumerate(rows)
            if all(isinstance(r.get(k), str) and r[k].strip() for k in ('src','tgt'))
            and max(lengths(r,tok)) <= 64][:count]


def controlled_init(model):
    with torch.random.fork_rng():
        torch.manual_seed(424242)
        with torch.no_grad():
            model.embedding.weight.normal_(0, model.d_model**-.5)
            model.embedding.weight[0].zero_()
    assert model.output.weight is model.embedding.weight


def token_masks(target, content_ids):
    valid = target.ne(0)
    content = valid & content_ids[target]
    return valid, content


def batch(rows, tok, max_len=64):
    arrays = [[tok.encode(('<ru2en> ' if k=='src' else '')+r[k],max_len) for r in rows] for k in ('src','tgt')]
    return tuple(torch.tensor([a+[0]*(max(map(len, side))-len(a)) for a in side]) for side in arrays)


def content_table(tok):
    return torch.tensor([any(c.isalnum() for c in tok.processor.id_to_piece(i).replace('▁','')) and i>5 for i in range(tok.vocab_size)])


@torch.inference_mode()
def evaluate(model, rows, tok, max_len=64):
    model.eval()
    table=content_table(tok)
    totals=dict(tokens=0, correct=0, content_tokens=0, content_correct=0, punctuation_tokens=0, punctuation_correct=0)
    ce=ls=logsum=logsquare=logn=0
    for start in range(0,len(rows),32):
        src,tgt=batch(rows[start:start+32],tok,max_len)
        logits=model(src,tgt[:,:-1]); target=tgt[:,1:]
        valid,content=token_masks(target,table)
        punctuation=valid & ~content & target.gt(5)
        pred=logits.argmax(-1)
        for prefix,mask in [('',valid),('content_',content),('punctuation_',punctuation)]:
            totals[prefix+'tokens']+=int(mask.sum())
            totals[prefix+'correct']+=int(((pred==target)&mask).sum())
        ce+=float(F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=0,reduction='sum'))
        ls+=float(F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=0,reduction='sum',label_smoothing=.1))
        active=logits[valid].double(); logsum+=float(active.sum()); logsquare+=float(active.square().sum()); logn+=active.numel()
    hyps=[]; generated=[]; eos=[]; refs=[r['tgt'] for r in rows]
    for start in range(0,len(rows),20):
        src,_=batch(rows[start:start+20],tok,max_len)
        outputs=model.greedy(src,max_len=max_len).tolist()
        for ids in outputs:
            hit=3 in ids[1:]; n=ids[1:].index(3) if hit else len(ids)-1
            generated.append(n); eos.append(hit); hyps.append(tok.decode(ids[1:n+1]))
    reference_lengths=[len(tok.processor.encode(r)) for r in refs]
    result={**totals,'ce':ce/totals['tokens'],'smoothed_ce':ls/totals['tokens'],
            'logit_std':math.sqrt(max(0,logsquare/logn-(logsum/logn)**2)),
            'token_accuracy':totals['correct']/totals['tokens'],
            'content_token_accuracy':totals['content_correct']/max(1,totals['content_tokens']),
            'punctuation_token_accuracy':totals['punctuation_correct']/max(1,totals['punctuation_tokens']),
            'chrF':sacrebleu.corpus_chrf(hyps,[refs]).score,'BLEU':sacrebleu.corpus_bleu(hyps,[refs]).score,
            'mean_generated_tokens':sum(generated)/len(rows),'mean_reference_tokens':sum(reference_lengths)/len(rows),
            'length_ratio':sum(generated)/sum(reference_lengths),'eos_rate':sum(eos)/len(rows),
            'immediate_eos_rate':sum(n==0 and e for n,e in zip(generated,eos))/len(rows),
            'premature_eos_rate':sum(e and n < .5*r for n,r,e in zip(generated,reference_lengths,eos))/len(rows),
            'nonempty_rate':sum(bool(h.strip()) for h in hyps)/len(rows),
            'repetition_rate':sum(repetition(h) for h in hyps)/len(rows),
            'samples':[dict(source=r['src'],reference=r['tgt'],hypothesis=h,generated_tokens=n,eos=e) for r,h,n,e in zip(rows,hyps,generated,eos)]}
    return result


def audit(rows,tok):
    records=[]
    for i,r in enumerate(rows[:100]):
        valid=isinstance(r,dict) and all(isinstance(r.get(k),str) for k in ('src','tgt'))
        record=dict(index=i,row=r,schema_valid=valid)
        if valid:
            record['encoded_lengths_including_bos_eos_and_source_tag']=lengths(r,tok)
            flags=[]; details={}
            for k in ('src','tgt'):
                s=r[k]; ids=tok.processor.encode(s); letters=[c for c in s if c.isalpha()]
                cy=sum('CYRILLIC' in unicodedata.name(c,'') for c in letters)/max(1,len(letters))
                la=sum('LATIN' in unicodedata.name(c,'') for c in letters)/max(1,len(letters))
                decoded=tok.processor.decode(ids)
                details[k]=dict(cyrillic_fraction=cy,latin_fraction=la,unknown_tokens=ids.count(1),tokens=len(ids),normalized_roundtrip_discrepancy=normalize(s)!=normalize(decoded),roundtrip=decoded,eos_preserved=tok.encode(s,64)[-1]==3)
                if not s.strip(): flags.append(k+'_empty')
                if (cy if k=='src' else la)<.5: flags.append(k+'_language_heuristic')
            if normalize(r['src'])==normalize(r['tgt']): flags.append('identical_sides')
            if max(len(r['src']),len(r['tgt']))/max(1,min(len(r['src']),len(r['tgt'])))>3: flags.append('character_length_ratio_gt3')
            record.update(details=details,alignment_red_flags=flags)
        records.append(record)
    summary=dict(examples=len(records),schema_invalid=sum(not r['schema_valid'] for r in records),heuristic_flagged=sum(bool(r.get('alignment_red_flags')) for r in records))
    for k,ix in [('src',0),('tgt',1)]:
        ds=[r['details'][k] for r in records if r['schema_valid']]
        summary[k]=dict(unknown_rate=sum(d['unknown_tokens'] for d in ds)/max(1,sum(d['tokens'] for d in ds)),normalized_roundtrip_discrepancies=sum(d['normalized_roundtrip_discrepancy'] for d in ds),eos_preserved=sum(d['eos_preserved'] for d in ds),truncated=sum(r['encoded_lengths_including_bos_eos_and_source_tag'][ix]>64 for r in records if r['schema_valid']))
    return summary,records


def main():
    torch.set_num_threads(2)
    report=ROOT/'reports/diagnostics'; run=ROOT/'runs/diagnostics'
    if report.exists() or run.exists(): raise SystemExit('Refusing existing diagnostic output directories')
    report.mkdir(); run.mkdir()
    start=time.perf_counter()
    tokpath=ROOT/'data/tokenizer/real_baseline/shared.model'; tok=SentencePieceTokenizer(tokpath)
    paths={s:ROOT/f'data/real_baseline/ru_en/{s}.jsonl' for s in ('train','valid')}
    rows={s:[json.loads(l) for l in p.read_text().splitlines()] for s,p in paths.items()}
    audits={}
    for split in rows:
        audits[split],records=audit(rows[split],tok)
        rows_write(report/f'{split}_first100_for_human_review.jsonl',records)
    dump(report/'data_audit.json',dict(results=audits,notice='Automated heuristics only; these samples are saved for future human review, not reviewed. Direction is RU→EN by schema and script heuristics; semantic alignment is not established. EOS is forcibly appended by encode, including when truncated.'))
    selected={s:select_short(rows[s],tok,3000 if s=='train' else 100) for s in rows}
    assert 2000<=len(selected['train'])<=4000 and len(selected['valid'])==100
    for s in selected: rows_write(report/f'{s}_short.jsonl',selected[s])
    random.seed(42); torch.manual_seed(42)
    a=MicroTransformer(tok.vocab_size,**CONFIG,init_mode='legacy'); b=copy.deepcopy(a); controlled_init(b)
    changed=[k for k,v in a.state_dict().items() if not torch.equal(v,b.state_dict()[k])]
    assert set(changed)=={'embedding.weight','output.weight'}
    generator=torch.Generator().manual_seed(42); batches=[]; epochs=0
    while len(batches)<1500:
        order=torch.randperm(len(selected['train']),generator=generator).tolist(); epochs+=1
        batches.extend([order[i:i+32] for i in range(0,len(order),32)])
    batches=batches[:1500]; dump(report/'batch_indices.json',batches)
    summary=dict(status='running',production_approved=False,license='unknown',python=sys.executable,threads=2,seed=42,config=CONFIG,training=dict(batch_size=32,lr=.001,label_smoothing=.1,optimizer='AdamW defaults; weight_decay=.01',schedule='constant',updates=1500,eval_every=250,gradient_clip=1.,sampling='seed42 shuffled epochs, identical index batches and dropout RNG in both arms'),data={s:dict(input_rows=len(rows[s]),selected_rows=len(selected[s]),input_sha256=sha(paths[s]),selected_sha256=sha(report/f'{s}_short.jsonl')) for s in rows},tokenizer_sha256=sha(tokpath),batch_sha256=sha(report/'batch_indices.json'),changed_state_keys=changed,initialization='A unchanged PyTorch initialization; B only shared embedding normal std=64^-0.5=.125, pad row zero, independent seed424242. This counteracts std~1 tied output logits and sqrt(d) input scaling without changing other tensors.',metric_definitions=dict(content='target piece contains alphanumeric, excluding IDs0..5',punctuation='noncontent target IDs>5; EOS counted only in overall accuracy',premature_eos='EOS before half reference token count',ce='unsmoothed token-weighted CE excluding PAD',logit_std='population std across all vocabulary logits on non-PAD target positions'),results={})
    for name,model in [('A',a),('B',b)]:
        arm=run/name; arm.mkdir(); began=time.perf_counter()
        torch.manual_seed(42)
        initial=evaluate(model,selected['valid'],tok)
        fixed=evaluate(model,rows['valid'][:100],tok)
        dump(report/f'{name}_initial_fixed_valid100.json',fixed)
        history=[dict(step=0,elapsed_seconds=time.perf_counter()-began,**initial)]
        torch.save(model.state_dict(),arm/'initial.pt')
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        best=-1; best_entry=None
        for step,indices in enumerate(batches,1):
            model.train(); src,tgt=batch([selected['train'][i] for i in indices],tok)
            opt.zero_grad(); logits=model(src,tgt[:,:-1])
            loss=F.cross_entropy(logits.flatten(0,1),tgt[:,1:].flatten(),ignore_index=0,label_smoothing=.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            if step%250==0:
                entry=dict(step=step,elapsed_seconds=time.perf_counter()-began,**evaluate(model,selected['valid'],tok))
                history.append(entry); rows_write(report/f'{name}_history.jsonl',history)
                if entry['chrF']>best:
                    best=entry['chrF']; best_entry=entry
                    torch.save(dict(model=model.state_dict(),config=CONFIG,step=step,valid_chrf=best),arm/'best.pt')
                print(name,step,'chrF',entry['chrF'],'CE',entry['ce'],flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=1500),arm/'last.pt')
        summary['results'][name]=dict(elapsed_seconds=time.perf_counter()-began,selected_step=best_entry['step'],best={k:v for k,v in best_entry.items() if k!='samples'},initial={k:v for k,v in initial.items() if k!='samples'},initial_fixed_valid100={k:v for k,v in fixed.items() if k!='samples'},best_sha256=sha(arm/'best.pt'))
        dump(report/'summary.json',summary)
    summary.update(status='completed',elapsed_seconds=time.perf_counter()-start,winner=max(summary['results'],key=lambda n:summary['results'][n]['best']['chrF']),limitations=['No test accessed; validation selection is not an independent generalization estimate.','Both arms use identical filtered short data: initialization is isolated, filtering versus previous runs is not.','Single seed and small model; language/alignment checks are heuristics, not human review.','No production model changes and no third experiment.'])
    dump(report/'summary.json',summary)


if __name__=='__main__': main()
