"""n = 7 family-A points (C4 seeds 1,2; C6 seed 1) and a DQ check at n = 6 C4 seed 1
(DQ = 0.001, 0.0005 Richardson). Appends to runs.jsonl (n=7) and writes step_check.json."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json, math, sys, os
os.environ["OMP_NUM_THREADS"]="1"
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import run_twelvefold as R, twelvefold_core as D
def dq(_):
    L1,L2=R.cell('A',6,'C4'); tl=D.tiling(L1,L2,offset=R.offset_of(1)); f=D.Frame(tl,sub=2); f.set_load()
    out={}
    for dqs in ((0.002,0.001),(0.001,0.0005)):
        ph,ext,tab=f.longwave(nphi=12,dqs=dqs); out[str(dqs)]=D.harmonics(ph,ext)
    return ('dq',out)
def j(spec): return ('job',R.job(spec))
if __name__=='__main__':
    specs=[('A',7,'C4',1,2),('A',7,'C4',2,2),('A',7,'C6',1,2)]
    with Pool(4) as p:
        res=[p.apply_async(j,(s,)) for s in specs]+[p.apply_async(dq,(0,))]
        for r in res:
            kind,val=r.get()
            if kind=='dq':
                Path('step_check.json').write_text(json.dumps(dict(status='P',case='A n=6 C4 seed 1 sub 2',results=val),indent=1)); print('dq',val,flush=True)
            else:
                with open('runs.jsonl','a') as fh: fh.write(json.dumps(val)+'\n')
                print(val['n'],val['kind'],val['seed'],val['V'],val['harm'],val['seconds'],flush=True)
