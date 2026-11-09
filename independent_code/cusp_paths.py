"""Reusable near-QC tensor and ellipticity gate. No job submission.

P = floating-point estimate; only representation statements elsewhere are C.
All workloads have explicit CPU budgets and checkpoint after each row.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import psutil
from geometry import Tiling,family_lifts,slope_components
from frame import Frame,tensor_summary

ROOT=Path(__file__).resolve().parent

def offset_for(index):
    if index=='main_input':
        return np.random.default_rng(7).uniform(-.5,.5,4)*.37+np.array([.1173,.2711,-.0618,.3307])-.5
    return np.array([.137,.271,.319,.413]) if index==0 else np.random.default_rng(1049+int(index)).uniform(-.5,.5,4)

def calculate(p,q,family,sub=4,offset=0,dq=False):
    start=time.process_time(); wall=time.perf_counter()
    t=Tiling(family_lifts(p,q,family),offset=offset_for(offset))
    f=Frame(t,sub=sub)
    components=slope_components(t.S)
    out=dict(classification='P',source_script='cusp_paths.py',p=p,q=q,family=family,
             sub=sub,offset=offset,N=len(t.n),slope=t.S.tolist(),slope_components=components.tolist(),
             amplitude=float(np.linalg.norm(components)),radius=.03,EA=1.,EI=f.EI,
             unloaded=tensor_summary(f.C0),factor_bytes=f.factor_bytes,protocols={})
    for protocol in ['stress','strain']:
        f.set_protocol(protocol); crit=f.critical()
        C,neg=f.macro(.65*f.lambdaE)
        row=dict(preload=f.pre.copy(),critical=crit,
                 at_fixed_normalized_load_065=tensor_summary(C) if C is not None else None)
        if dq:
            theta=crit['theta']; n=np.array([np.cos(theta),np.sin(theta)])
            samples=[]
            for deltaq in [.004,.002,.001,.0005]:
                r=f.bloch_threshold(deltaq*(t.cell@n)); r['DQ']=deltaq; samples.append(r)
            xx=np.array([r['DQ']**2 for r in samples]); yy=np.array([r['normalized'] for r in samples])
            fit=np.polyfit(xx,yy,1)
            last=np.polyfit(xx[-2:],yy[-2:],1)
            row['DQ_check']=dict(branch='fixed critical acoustic direction',rows=samples,
                                 all_fit_intercept=float(fit[1]),last_two_intercept=float(last[1]),
                                 exact_zero_Schur=crit['normalized'])
        out['protocols'][protocol]=row
    out['cpu_s']=time.process_time()-start; out['wall_s']=time.perf_counter()-wall
    out['rss_bytes']=psutil.Process().memory_info().rss
    print(json.dumps(dict(family=family,p=p,q=q,N=len(t.n),sub=sub,offset=offset,
                          amp=out['amplitude'],stress=out['protocols']['stress']['critical']['normalized'],
                          strain=out['protocols']['strain']['critical']['normalized'],cpu_s=out['cpu_s'])),flush=True)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['base','refine','large','phase'],default='base')
    ap.add_argument('--cpu-budget',type=float,default=2400); args=ap.parse_args()
    if args.mode=='base':
        cases=[(p,q,fam,4,0,False) for fam in ['B1','B2','E2a','E2diag'] for p,q in [(1,1),(2,3),(5,7),(12,17)]]
    elif args.mode=='refine':
        cases=[(2,3,'B1',sub,'main_input',sub==2) for sub in [1,2,4,8]]
        cases += [(5,7,fam,sub,0,False) for fam in ['B1','B2','E2a','E2diag'] for sub in [2,8]]
        cases += [(5,7,'B1',2,'main_input',True),(12,17,'B1',2,'main_input',False)]
    elif args.mode=='large': cases=[(29,41,fam,4,0,False) for fam in ['B1','B2','E2a','E2diag']]
    else:
        cases=[(p,q,fam,4,seed,False) for fam in ['B1','B2','E2a','E2diag'] for p,q in [(5,7),(12,17)] for seed in [1,2]]
    path=ROOT/f'cusp_{args.mode}.json'
    result=[]; start=time.process_time()
    for case in cases:
        if time.process_time()-start>args.cpu_budget: break
        result.append(calculate(*case))
        path.write_text(json.dumps(dict(classification='P',source_script='cusp_paths.py',mode=args.mode,
                                       cpu_s=time.process_time()-start,completed=len(result),requested=len(cases),rows=result),indent=2),encoding='utf-8')
    print('TOTAL CPU seconds',time.process_time()-start,flush=True)

if __name__=='__main__': main()
