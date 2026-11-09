"""Independent numerical/census audits, separate from preregistered scores."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,time,argparse,itertools
from pathlib import Path
import numpy as np
from run_cells import ROOT,save,sha,mechanics
from patch_atlas import periodic_signatures,signature,census
from geometry import Tiling
from penrose_core import Penrose,richardson
from beam_check import Frame as FullFrame

def load_t(row):
    z=np.load(ROOT/'snapshots'/(row['id']+'.npz'))
    return Tiling(z['L'],n=z['n']) if row['symmetry']=='AB' else Penrose(z['L'],z['gamma'],z['n'])

def brute_signature(t,root):
    # Direct membership in the unwrapped integer module, independent of adj.
    d=t.n.shape[1];eye=np.eye(d,dtype=int);steps=np.r_[eye,-eye]
    seen={(0,)*d};front=seen.copy();lookup=set(map(tuple,t.n))
    for _ in range(2):
        nxt=set()
        for off in front:
            for step in steps:
                v=tuple(np.array(off)+step);rep,_=t.wrap(t.n[root]+v)
                if tuple(rep) in lookup and v not in seen:nxt.add(v)
        seen.update(nxt);front=nxt
    return signature(seen)

def run(limit=None):
    start=time.process_time();manifest=json.loads((ROOT/'cell_manifest.json').read_text())
    checks=[];pending=[]
    path=ROOT/'cell_audits.json'
    old=json.loads(path.read_text()) if path.exists() else dict(checks=[],cpu_s=0)
    done={r['id'] for r in old['checks']};count=0
    for spec in manifest['rows']:
        # Fixed before inspection: two endpoints of every family/symmetry.
        if (spec['rep'],int(spec['group'].split('_')[-1])) not in [(0,0),(1,5)]:continue
        if spec['id'] in done:continue
        raw=ROOT/'raw'/(spec['id']+'.json')
        if not raw.exists():pending.append(spec['id']);continue
        row=json.loads(raw.read_text());t=load_t(row);sigs=periodic_signatures(t)
        ids=np.unique(np.linspace(0,len(t.n)-1,17,dtype=int))
        assert all(brute_signature(t,int(i))==sigs[i] for i in ids)
        full=FullFrame(t.x,t.edges(),t.cell,div=2,EI=.03**2/4)
        vals=[]
        # 8x8 grid contains 4x4 grid; evaluate only additional distinct k.
        seen=set()
        for i,j in itertools.product(range(8),repeat=2):
            key=min((i,j),((-i)%8,(-j)%8))
            if key in seen:continue
            seen.add(key)
            if i%2==0 and j%2==0:continue
            p,res=full.eigen([i/8,j/8]);vals.append(dict(k=[i/8,j/8],pc=p/(np.pi**2*.03**2/4),residual=res))
        dense_pc=min([row['mechanics']['pc']]+[v['pc'] for v in vals])
        rich=richardson(t,sub=2,angles=8,finite=False)
        check=dict(id=spec['id'],label='P',brute_census_vertices=len(ids),brute_census_matches=True,
                   pc_grid4=row['mechanics']['pc'],pc_grid8=dense_pc,
                   grid_refinement_change=dense_pc-row['mechanics']['pc'],extra_bloch=vals,
                   richardson=rich,c0_condensed=row['mechanics']['c0'],
                   c0_difference=rich['c0']-row['mechanics']['c0'])
        old['checks'].append(check);count+=1;save(path,old)
        print('AUDIT',spec['id'],'dense_change',check['grid_refinement_change'],'c0_diff',check['c0_difference'],flush=True)
        if limit is not None and count>=limit:break
    old['pending']=pending;old['cpu_s']+=time.process_time()-start;save(path,old)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);a=ap.parse_args();run(a.limit)
