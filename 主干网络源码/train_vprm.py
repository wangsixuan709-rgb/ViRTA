"""Independent Visual Patch Restoration Module (VPRM) pretraining. Test data are used only by the test command."""
import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from vprm_data import VPRMPairs, CLASSES, prepare, LOCAL_DEGRADATION_PROBABILITY
from vprm_encoder import VPRMViTEncoder, STAGES, FORMAT, check_checkpoint


def lr_factor(step, warmup=1000, total=20000, start=.01, minimum=.01):
    if step <= warmup: return start+(1-start)*step/warmup
    progress=min(1.,(step-warmup)/(total-warmup))
    return minimum+.5*(1-minimum)*(1+math.cos(math.pi*progress))


def feature_loss(student, teacher):
    if len(student)!=3 or len(teacher)!=3: raise ValueError('Expected three ViT stages: 4, 8, 12')
    if any(s.shape != t.shape for s,t in zip(student,teacher)): raise ValueError('Teacher/student shapes must match')
    losses=torch.stack([F.mse_loss(s.float(),t.detach().float()) for s,t in zip(student,teacher)])
    return losses.mean(),losses


def similarities(a,b):
    return F.cosine_similarity(a.float(),b.float(),dim=1,eps=1e-8).mean((1,2))


class EarlyStop:
    def __init__(self,start=12000,patience=4000,delta=.001,window=5):
        self.start,self.patience,self.delta=start,patience,delta
        self.history=deque(maxlen=window); self.anchor=None; self.last_progress=None
    def update(self,step,score):
        if not math.isfinite(score): raise ValueError('Nonfinite validation')
        self.history.append(score); smooth=sum(self.history)/len(self.history)
        if step>=self.start and len(self.history)==self.history.maxlen:
            if self.anchor is None or smooth>self.anchor+self.delta:
                self.anchor,self.last_progress=smooth,step
        return smooth,self.last_progress is not None and step-self.last_progress>=self.patience
    def state_dict(self): return {'history':list(self.history),'anchor':self.anchor,'last_progress':self.last_progress}
    def load_state_dict(self,s):
        self.history.extend(s['history']);self.anchor=s['anchor'];self.last_progress=s['last_progress']


def atomic_save(state,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)


def log(path,item):
    text=json.dumps(item,allow_nan=False)
    with Path(path).open('a') as f:f.write(text+'\n')
    print(text,flush=True)


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loader(args,split,round_id):
    ds=VPRMPairs(args.manifest,split,round_id,args.seed,
                 local_degradation_probability=getattr(args,'local_degradation_probability',LOCAL_DEGRADATION_PROBABILITY))
    g=torch.Generator().manual_seed(args.seed+round_id)
    return DataLoader(ds,batch_size=args.batch_size,shuffle=split=='train',num_workers=args.workers,
                      pin_memory=args.device=='cuda',generator=g,drop_last=False,
                      multiprocessing_context='spawn' if args.workers else None)


@torch.no_grad()
def evaluate(model,args,split,round_id,detail_path=None):
    model.eval();rows=[]
    for batch in loader(args,split,round_id):
        hq,lq=batch['hq'].to(args.device),batch['lq'].to(args.device)
        teacher=model.forward_stages(hq,False)[-1]
        baseline=model.forward_stages(lq,False)[-1]
        restored=model.forward_stages(lq,True)[-1]
        bs,rs=similarities(baseline,teacher).cpu().tolist(),similarities(restored,teacher).cpu().tolist()
        for i,path in enumerate(batch['path']):
            rows.append({'path':path,'kind':CLASSES[int(batch['kind'][i])],'severity':int(batch['severity'][i]),
                         'sample_seed':int(batch['sample_seed'][i]),'corruption_seed':int(batch['corruption_seed'][i]),
                         'resize_size':int(batch['resize_size'][i]),'baseline':bs[i],'restored':rs[i],'gain':rs[i]-bs[i]})
    degraded=[r for r in rows if r['kind']!='clean'];clean=[r for r in rows if r['kind']=='clean']
    if not degraded:raise ValueError('No degraded validation samples')
    result={k:float(np.mean([r[k] for r in degraded])) for k in ('baseline','restored','gain')}
    result.update({'count':len(rows),'nonclean_count':len(degraded),'round':round_id,'split':split,
                   'clean_similarity':float(np.mean([r['restored'] for r in clean])) if clean else None})
    result['by_class']={k:{'count':len(v),'restored':float(np.mean([r['restored'] for r in v])),
                           'gain':float(np.mean([r['gain'] for r in v]))}
                        for k in CLASSES if (v:=[r for r in rows if r['kind']==k])}
    result['by_severity']={str(k):{'count':len(v),'restored':float(np.mean([r['restored'] for r in v])),
                                  'gain':float(np.mean([r['gain'] for r in v]))}
                           for k in range(1,6) if (v:=[r for r in degraded if r['severity']==k])}
    if detail_path:
        Path(detail_path).parent.mkdir(parents=True,exist_ok=True)
        Path(detail_path).write_text(json.dumps(rows,indent=2,allow_nan=False))
    return result


def model_state(model,args,step):
    return {'format':FORMAT,'language':args.language,'stages':STAGES,'vprm':model.vprm.state_dict(),'backbone':model.features.state_dict(),
            'step':step,'config':vars(args),'manifest_sha256':fingerprint(args.manifest)}


def compare(model,args,out,step,best_path):
    rounds=[1_000_000+step*3+j for j in range(args.compare_seeds)]
    candidates=[evaluate(model,args,'val',r,out/'comparisons'/f'{step}-candidate-{r}.json') for r in rounds]
    previous=None
    if best_path.exists():
        current={k:v.detach().cpu().clone() for k,v in model.vprm.state_dict().items()}
        best=torch.load(best_path,map_location='cpu',weights_only=False)
        check_checkpoint(best,args.language)
        if best['manifest_sha256'] != fingerprint(args.manifest):raise ValueError('Incumbent belongs to a different split')
        try:
            model.vprm.load_state_dict(best['vprm'])
            previous=[evaluate(model,args,'val',r,out/'comparisons'/f'{step}-incumbent-{r}.json') for r in rounds]
        finally:model.vprm.load_state_dict(current)
        del current,best
    difference=None if previous is None else float(np.mean([c['restored']-b['restored'] for c,b in zip(candidates,previous)]))
    replace=previous is None or difference>args.replace_delta
    if replace:atomic_save(model_state(model,args,step),best_path)
    return {'event':'comparison','step':step,'rounds':rounds,'paired_gain':difference,'replace':replace,
            'candidate_mean':float(np.mean([c['restored'] for c in candidates]))}


def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((out/name).exists() for name in ('latest.pt','best.pt','config.json')):
        raise FileExistsError('Use --resume or a fresh output directory; existing runs are never overwritten')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if args.device=='cuda':torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    model=VPRMViTEncoder(args.language,checkpoint_segments=True).to(args.device)
    optimizer=torch.optim.AdamW(model.vprm.parameters(),lr=args.peak_lr,weight_decay=args.weight_decay,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_factor(s,args.warmup_steps,args.max_steps,args.start_lr/args.peak_lr,args.min_lr/args.peak_lr))
    early=EarlyStop(args.early_start,args.patience_steps,args.min_delta)
    step,epoch,last_compare=0,0,0;best_path=out/'best.pt'
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        check_checkpoint(ck,args.language)
        if fingerprint(args.manifest)!=ck['manifest_sha256']:raise ValueError('Split manifest changed')
        for k in ('max_steps','warmup_steps','peak_lr','start_lr','min_lr','seed','batch_size','accumulation','early_start','patience_steps','min_delta','replace_delta','compare_seeds','compare_every','weight_decay','precision','grad_clip'):
            if ck['config'][k]!=getattr(args,k):raise ValueError(f'Resume mismatch: {k}')
        if ck['config'].get('local_degradation_probability',0.0)!=args.local_degradation_probability:
            raise ValueError('Resume mismatch: local_degradation_probability; use a fresh output for the new data policy')
        if Path(args.resume).resolve().parent != out.resolve():raise ValueError('Resume checkpoint must belong to the output directory')
        if not best_path.exists():raise FileNotFoundError('Resume in original output directory with best.pt')
        model.features.load_state_dict(ck['backbone']);model.vprm.load_state_dict(ck['vprm'])
        optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);early.load_state_dict(ck['early'])
        step,epoch,last_compare=ck['step'],ck['next_epoch'],ck['last_compare']
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        del ck
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    while step<args.max_steps:
        data=loader(args,'train',epoch);model.train();optimizer.zero_grad(set_to_none=True)
        sums=np.zeros(3);count=0;clean_sum=degraded_sum=0.;nc=nd=0;n=len(data.dataset)
        for index,batch in enumerate(data):
            hq,lq=batch['hq'].to(args.device),batch['lq'].to(args.device)
            with torch.no_grad():targets=model.forward_stages(hq,False)
            context=torch.autocast('cuda',dtype=torch.bfloat16) if args.precision=='bf16' else nullcontext()
            with context:
                restored=model.forward_stages(lq,True);loss,layers=feature_loss(restored,targets)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
            # Weight every example equally, including the partial final accumulation window.
            window_start=(index//args.accumulation)*args.accumulation*args.batch_size
            window_size=min(args.accumulation*args.batch_size,n-window_start)
            (loss*(len(hq)/window_size)).backward()
            sums+=layers.detach().cpu().numpy()*len(hq);count+=len(hq)
            with torch.no_grad():
                per=torch.stack([(s.float()-t.float()).square().flatten(1).mean(1) for s,t in zip(restored,targets)]).mean(0)
                mask=batch['kind'].to(args.device)==14
                clean_sum+=per[mask].sum().item();degraded_sum+=per[~mask].sum().item();nc+=int(mask.sum());nd+=int((~mask).sum())
            if (index+1)%args.accumulation==0 or index+1==len(data):
                grad=torch.nn.utils.clip_grad_norm_(model.vprm.parameters(),args.grad_clip,error_if_nonfinite=True)
                used_lr=optimizer.param_groups[0]['lr']
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                if step%args.log_every==0:log(out/'train.jsonl',{'step':step,'epoch':epoch,'lr_used':used_lr,'next_lr':scheduler.get_last_lr()[0],'loss':float(loss.detach()),'grad_norm':float(grad)})
                if step>=args.max_steps:break
        epoch+=1
        val=evaluate(model,args,'val',epoch,out/'validation'/f'epoch-{epoch}.json')
        smooth,stop=early.update(step,val['restored'])
        log(out/'metrics.jsonl',{'event':'epoch','step':step,'epoch':epoch,'samples':count,'layer_mse':(sums/count).tolist(),
                              'clean_mse':clean_sum/nc if nc else None,'degraded_mse':degraded_sum/nd if nd else None,
                              'val':val,'smoothed':smooth,'early':early.state_dict(),'lr':scheduler.get_last_lr()[0]})
        if not best_path.exists() or step-last_compare>=args.compare_every or stop or step>=args.max_steps:
            log(out/'metrics.jsonl',compare(model,args,out,step,best_path));last_compare=step
        state=model_state(model,args,step)
        state.update({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'early':early.state_dict(),
                      'next_epoch':epoch,'last_compare':last_compare,'torch_rng':torch.get_rng_state(),
                      'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                      'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else []})
        atomic_save(state,out/'latest.pt')
        if stop:log(out/'metrics.jsonl',{'event':'early_stop','step':step});break
    print('Training ended. best.pt uses validation only; testing is a separate command.',flush=True)


def test(args):
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if ck['manifest_sha256']!=fingerprint(args.manifest):raise ValueError('Split manifest differs')
    args.seed=ck['config']['seed']  # Keep the selected experiment's sampling protocol.
    args.local_degradation_probability=ck['config'].get('local_degradation_probability',0.0)
    check_checkpoint(ck,args.language)
    model=VPRMViTEncoder(args.language).to(args.device)
    model.features.load_state_dict(ck['backbone']);model.vprm.load_state_dict(ck['vprm']);del ck
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results=[evaluate(model,args,'test',2_000_000+j,out/f'test-seed-{j}.json') for j in range(args.test_seeds)]
    summary={k:{'mean':float(np.mean([r[k] for r in results])),
                'std':float(np.std([r[k] for r in results],ddof=1)) if len(results)>1 else 0.}
             for k in ('baseline','restored','gain','clean_similarity')}
    (out/'test-summary.json').write_text(json.dumps({'results':results,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','train','test'])
    p.add_argument('--root',default='clean-image');p.add_argument('--manifest',default='vprm_runs/split.json');p.add_argument('--output',default=None)
    p.add_argument('--language',choices=['en','zh'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--local-degradation-probability',type=float,default=LOCAL_DEGRADATION_PROBABILITY,
                   help='Fraction of non-clean samples using smooth clean/light/heavy regions; default 0.5')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-steps',type=int,default=20000);p.add_argument('--warmup-steps',type=int,default=1000)
    p.add_argument('--peak-lr',type=float,default=1e-4);p.add_argument('--start-lr',type=float,default=1e-6);p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--early-start',type=int,default=12000);p.add_argument('--patience-steps',type=int,default=4000)
    p.add_argument('--min-delta',type=float,default=.001);p.add_argument('--replace-delta',type=float,default=.001)
    p.add_argument('--compare-seeds',type=int,default=3);p.add_argument('--compare-every',type=int,default=1000)
    p.add_argument('--log-every',type=int,default=20);p.add_argument('--test-seeds',type=int,default=5)
    p.add_argument('--resume');p.add_argument('--checkpoint')
    args=p.parse_args()
    if args.output is None:args.output=f"vprm_runs/{args.language}/pretrain"
    if args.command=='prepare':print(prepare(args.root,args.manifest,args.seed));return
    if not Path(args.manifest).is_file():
        p.error('Split manifest not found; first run train_vprm.py prepare --language '
                f'{args.language} --root {args.root} --manifest {args.manifest}')
    if min(args.batch_size,args.accumulation,args.compare_seeds,args.test_seeds,args.compare_every,args.log_every,args.patience_steps)<1:p.error('Counts must be positive')
    if args.workers<0 or args.min_delta<0 or args.replace_delta<0 or args.grad_clip<=0:p.error('Invalid worker count or thresholds')
    if not 0<=args.local_degradation_probability<=1:p.error('local-degradation-probability must be in [0,1]')
    if args.command=='test':
        if not args.checkpoint:p.error('--checkpoint required')
        test(args);return
    if not 0<args.warmup_steps<args.max_steps:p.error('Require 0 < warmup < max_steps')
    if not 0<args.start_lr<=args.peak_lr or not 0<args.min_lr<=args.peak_lr:p.error('Invalid LR range')
    if args.precision=='bf16' and (args.device!='cuda' or not torch.cuda.is_bf16_supported()):p.error('BF16 requires supported CUDA')
    train(args)

if __name__=='__main__':main()
