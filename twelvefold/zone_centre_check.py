"""character of the critical long-wave mode in the NG frame.
Compares Richardson q->0 limit lambda(phi) with lambda at Gamma (q = 0, translations removed),
and samples a few finite q in the BZ; reports participation ratio of node rotations.  -> zone_centre_check.json"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json, math, time, numpy as np, twelvefold_core as D
from pathlib import Path
rows=[]
U=D.GEN+np.eye(4,dtype=np.int64)
for n in (4,5):
  for kind in ('C4','C6'):
    L1=np.linalg.matrix_power(U,n)@np.array([1,0,0,0]); L2=np.linalg.matrix_power(D.GEN,3 if kind=='C4' else 2)@L1
    t=time.time(); tl=D.tiling(L1,L2); f=D.Frame(tl,sub=2); f.set_load()
    ph,ext,tab=f.longwave(nphi=12)
    lg=f.lam((0,0))
    bz=[f.lam(k) for k in ((0.5,0),(0,0.5),(0.5,0.5),(0.25,0),(0.1,0.1))]
    al,be=D.strain_parts(tl['M'])
    r=dict(n=n,kind=kind,V=len(tl['lifts']),alpha=[al.real,al.imag],beta=[be.real,be.imag],
           longwave_ext=ext.tolist(),lam_dq002=tab[0.002].tolist(),lam_gamma=lg,lam_bz=bz,seconds=time.time()-t)
    rows.append(r); print(json.dumps({k:r[k] for k in ('n','kind','V','lam_gamma','lam_bz','seconds')}),'ext min/max',min(ext),max(ext),flush=True)
Path('zone_centre_check.json').write_text(json.dumps(dict(status='P',rows=rows),indent=1))
