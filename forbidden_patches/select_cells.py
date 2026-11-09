"""Outcome-blind design. This manifest is frozen before any pressures."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import itertools, math, json, hashlib
from pathlib import Path
import numpy as np
from geometry import P as P4,Q as Q4,slope,slope_components,family_lifts
from penrose_core import P as P5,D,components

ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make():
    rng=np.random.default_rng(861108620); rows=[]; seen=set()
    for sym in ('AB','Penrose'):
        P=P4 if sym=='AB' else P5; d=P.shape[1]
        families=['approximant','few_flip','equilibrated']+(['disc_cluster'] if d==4 else [])
        for family in families:
            for group in range(6):
                while True:
                    angle=rng.uniform(-.25,.25); c,s=np.cos(angle),np.sin(angle)
                    B=np.array([[c,-s],[s,c]])@np.array([[rng.uniform(6,11),rng.uniform(-2,2)],[0,rng.uniform(6,11)]])
                    L=np.rint(B@P+rng.normal(0,.15,(2,d))).astype(int)
                    if d==5:
                        for j in range(2):
                            a=int(rng.integers(5));L[j,a]-=int(L[j]@D)*int(D[a])
                    minors=[int(round(np.linalg.det(L[:,[i,j]]))) for i,j in itertools.combinations(range(d),2)]
                    if min(minors)<=0 or math.gcd(*minors)!=1 or not 260<=sum(minors)<=850:continue
                    delta=(float(np.linalg.norm(slope_components(slope(L)))) if d==4 else components(L)['delta'])
                    if not .004<=delta<=.075:continue
                    key=tuple(L.ravel())
                    if key in seen:continue
                    seen.add(key);break
                for rep in range(2):
                    idx=len(rows); seed=862000000+idx*97+rep
                    rows.append(dict(id=f'{sym}_{family}_{group:02d}_{rep}',symmetry=sym,
                                     family=family,group=f'{sym}_{family}_{group:02d}',rep=rep,
                                     L=L.tolist(),N_expected=sum(minors),delta=delta,
                                     phase_seed=seed,mc_seed=seed+17,
                                     flips=[1,2,4,8,16,32][group] if family=='few_flip' else 0,
                                     disc_radius=[2,3,4,5,6,7][group] if family=='disc_cluster' else -1))
    f4=[]
    for p,q in [(2,3),(5,7),(12,17),(29,41)]:
        for rep in range(3):
            f4.append(dict(id=f'F4_B1_{p}_{q}_{rep}',symmetry='AB',family='F4_Pell',
                           group=f'Pell_{p}_{q}',p=p,q=q,rep=rep,L=family_lifts(p,q).tolist(),
                           phase_seed=863000000+p*100+rep,mc_seed=0))
    policy=dict(
        patches='Induced graph of vertices <=2 edges from root; signed lift directions; no rotation quotient; unwrap torus.',
        pressure='EA=1, beam radius=.03, sub=2, uniform hydrostatic stress. Minimum of optimized acoustic threshold and Bloch scan.',
        primary_bloch_grid='Gamma and (i/4,j/4), i,j=0,1,2,3, modulo k~-k (10 non-Gamma points).',
        pc_QC_AB=.7998,pc_QC_Penrose=.8106,
        Penrose_reference_source='the main code gives .8106; own pre-existing free-reference estimate .81054; constant normalization does not affect ranks.',
        atlas_radii=[35,60,90,110],
        F1='PASS iff N_AB>=40, all four families present, and Spearman(phi2,Delta)>=.9.',
        F2='PASS iff rho_phi2 >= every signed competitor rho. Report paired bootstrap 95% CIs; registration specifies no CI decision rule.',
        F3='Fit g(x)=a*x+b*x^2, a,b>=0, AB only. Penrose predictions g(phi2), scale fixed to 1 (no Penrose fit). PASS iff N>=30, all three families, rho>=.85.',
        F4='Test zero-phi2 Pell sequence; isotropic angular mean c0 (8 directions). Fit Delta_iso=c+a*abs(delta)+b*delta^2. 95% bootstrap CI for a, both all orders and three finest. Finite consistency is not proof of universal/asymptotic statement; report binary with scope.',
        bootstrap='10000 paired, stratified by family and clustered by period (two phase/MC replicates); percentile 95% CI; seed 864108.',
        alternatives='Centered internal variance and true Euclidean max/mean excess from each rational-cut projected cube. Moved fraction relative to same-period/same-phase canonical vertex set. These are periodic well-defined.',
        equilibration='Uniform-vertex legal flips, beta=0; 5000 burn-in sweeps, then 20 blocks of 500 sweeps. Retain traces, paired-chain diagnostics by period.',
        discs='Uniform beta=0 legal flips restricted to old AND new site in disc; 15000 global-vertex sweeps.',
        heldout='Fresh RNG, independent randomly rounded oblique primitive periods, no old observations or snapshots; F4 controls excluded from held-out cohorts.',
        cpu_limit_s=14400)
    files=['geometry.py','frame.py','beam_check.py','rigidity_check.py','penrose_core.py','penrose_mc.py','law_ab_mc.py']
    out=dict(label='P',seed=861108620,policy=policy,prereg_sha256=digest(ROOT/'registered_plan.md') if (ROOT/'registered_plan.md').exists() else None,
             copied_sources=[dict(file=f,source='../../independent/'+f,sha256=digest(ROOT/f)) for f in files],rows=rows,F4=f4)
    path=ROOT/'cell_manifest.json'
    if path.exists():assert json.loads(path.read_text())==out,'Manifest already frozen with different contents'
    else:path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Frozen',len(rows),'test cases and',len(f4),'F4 controls',digest(path))
if __name__=='__main__':make()
