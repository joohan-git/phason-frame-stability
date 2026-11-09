"""n = 7 C4 seeds 1-3 and C6 seed 2 with DQ = (0.001, 0.0005) (q L / 2 pi = 0.1, same as n = 6 at the
production DQ).  -> order7_step.jsonl"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json, os, time
os.environ["OMP_NUM_THREADS"]="1"
from multiprocessing import Pool
import numpy as np, run_twelvefold as R, twelvefold_core as D
def job(spec):
    fam,n,kind,seed=spec; t=time.time()
    L1,L2=R.cell(fam,n,kind); tl=D.tiling(L1,L2,offset=R.offset_of(seed)); f=D.Frame(tl,sub=2); f.set_load()
    ph,ext,tab=f.longwave(nphi=12,dqs=(0.001,0.0005)); h=D.harmonics(ph,ext); al,be=D.strain_parts(tl['M'])
    return dict(fam=fam,n=n,kind=kind,seed=seed,sub=2,dqs=[0.001,0.0005],V=len(tl['lifts']),alpha=[al.real,al.imag],beta=[be.real,be.imag],harm=h,pc_phi=ext.tolist(),seconds=time.time()-t)
if __name__=='__main__':
    with Pool(4) as p:
        for r in p.imap_unordered(job,[('A',7,'C4',1),('A',7,'C4',2),('A',7,'C4',3),('A',7,'C6',2)]):
            open('order7_step.jsonl','a').write(json.dumps(r)+'\n'); print(r['kind'],r['seed'],r['V'],r['harm'],r['seconds'],flush=True)
