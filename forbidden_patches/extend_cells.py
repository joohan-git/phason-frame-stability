"""Extensions; never overwrite or replace the preregistered primary set."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,math
import numpy as np
from run_cells import ROOT,run_one,save,sha
from geometry import family_lifts,P as P4,slope,slope_components
from penrose_core import P as P5,D,components

def fresh_specs():
    """168 prospective new cases: 96 AB and 72 Penrose; no old outcomes."""
    rng=np.random.default_rng(866108);rows=[];seen=set()
    for sym in ['AB','Penrose']:
        P=P4 if sym=='AB' else P5;d=P.shape[1]
        families=['approximant','few_flip','equilibrated']+(['disc_cluster'] if d==4 else [])
        for family in families:
            for group in range(12):
                while True:
                    theta=rng.uniform(-.3,.3);c,s=np.cos(theta),np.sin(theta)
                    B=np.array([[c,-s],[s,c]])@np.array([[rng.uniform(12,22),rng.uniform(-3,3)],[0,rng.uniform(12,22)]])
                    L=np.rint(B@P+rng.normal(0,.2,(2,d))).astype(int)
                    if d==5:
                        for j in range(2):
                            axis=int(rng.integers(5));L[j,axis]-=int(L[j]@D)*int(D[axis])
                    minors=[int(round(np.linalg.det(L[:,[i,j]]))) for i,j in itertools.combinations(range(d),2)]
                    if min(minors)<=0 or math.gcd(*minors)!=1 or not 800<=sum(minors)<=2400:continue
                    delta=float(np.linalg.norm(slope_components(slope(L)))) if d==4 else components(L)['delta']
                    if not .003<=delta<=.05 or tuple(L.ravel()) in seen:continue
                    seen.add(tuple(L.ravel()));break
                for rep in range(2):
                    seed=866000000+len(rows)*101
                    rows.append(dict(id=f'fresh_{sym}_{family}_{group:02d}_{rep}',symmetry=sym,family=family,
                        group=f'fresh_{sym}_{family}_{group:02d}',rep=rep,L=L.tolist(),N_expected=sum(minors),
                        delta=delta,phase_seed=seed,mc_seed=seed+23,extension_kind='fresh',
                        flips=[1,2,4,8,16,32][group%6] if family=='few_flip' else 0,
                        disc_radius=[3,5,7,9][group%4] if family=='disc_cluster' else -1,
                        burn_sweeps=50000,blocks=40,block_sweeps=1250))
    return rows

def specs(kind):
    primary=json.loads((ROOT/'cell_manifest.json').read_text())
    if kind=='fresh':
        # Read frozen integer periods and offsets directly. Re-running
        # floating-point catalogue selection across BLAS libraries is unneeded.
        return json.loads((ROOT/'fresh_manifest.json').read_text())['rows']
    if kind=='pell':
        return [dict(id=f'extension_Pell_{p}_{q}_{rep}',symmetry='AB',family='F4_Pell',
                     group=f'Pell_{p}_{q}',p=p,q=q,rep=rep,L=family_lifts(p,q).tolist(),
                     phase_seed=865000000+p*100+rep,mc_seed=0)
                for p,q,reps in [(70,99,3),(169,239,1)] for rep in range(reps)]
    if kind=='mixing':
        rows=[]
        for r in primary['rows']:
            if r['family']!='equilibrated':continue
            r=r.copy();r['id']='extension_'+r['id'];r['mc_seed']+=200000
            r.update(burn_sweeps=50000,block_sweeps=1250,blocks=40);rows.append(r)
        return rows
    if kind=='large':
        rng=np.random.default_rng(865108)
        rows=[]
        for r in primary['rows']:
            r=r.copy();r['id']='extension_large_'+r['id'];r['group']='large_'+r['group']
            # Fourfold area, same rational slope; suitable for finite-size and
            # long-wave/disorder checks, NOT new independent approximant slopes.
            r['L']=(2*np.array(r['L'])).tolist();r['phase_seed']+=300000;r['mc_seed']+=300000
            if r['family']=='few_flip':r['flips']*=4
            if r['family']=='disc_cluster':r['disc_radius']*=2
            r.pop('N_expected',None);rows.append(r)
        return rows
    raise ValueError(kind)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('kind',choices=['pell','mixing','large','fresh'])
    ap.add_argument('--index',type=int);ap.add_argument('--all',action='store_true');ap.add_argument('--cpu-budget',type=float,default=7200)
    ap.add_argument('--list',action='store_true');a=ap.parse_args()
    rows=specs(a.kind)
    if a.list:
        for i,r in enumerate(rows):print(i,r['id'])
        return
    assert a.index is not None or a.all,'Select --index or --all'
    selected=rows if a.all else [rows[a.index]]
    path=ROOT/'extensions';path.mkdir(exist_ok=True)
    atlas=json.loads((ROOT/'patch_atlas.json').read_text())
    start=time.process_time()
    for spec in selected:
        target=path/(spec['id']+'.json')
        if target.exists():continue
        if time.process_time()-start>a.cpu_budget:break
        row=run_one(spec,set(atlas['symmetries'][spec['symmetry']]['patches']))
        row['analysis_role']='separate extension, not substituted in the primary scores'
        row['parent_manifest_sha256']=sha(ROOT/'cell_manifest.json')
        if a.kind=='fresh':
            row['fresh_manifest_sha256']=sha(ROOT/'fresh_manifest.json')
        save(target,row)
        print(spec['id'],row['census']['phi2'],row['mechanics']['c0'],row['cpu_s'],flush=True)
if __name__=='__main__':main()
