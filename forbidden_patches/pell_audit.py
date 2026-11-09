"""Angular resolution and finite-order model sensitivity for F4."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,time
import numpy as np
from run_cells import ROOT,save,mechanics
from check_cells import load_t

def main():
    start=time.process_time();path=ROOT/'pell_audit.json'
    out=json.loads(path.read_text()) if path.exists() else dict(rows=[],cpu_s=0,label='P')
    done={r['id'] for r in out['rows']}
    for p,q in [(2,3),(5,7),(12,17),(29,41)]:
        key=f'F4_B1_{p}_{q}_0'
        if key in done:continue
        row=json.loads((ROOT/'raw'/(key+'.json')).read_text());t=load_t(row)
        mech=mechanics(t,angles=16,bloch=False)
        out['rows'].append(dict(id=key,delta=row['delta'],c0_8=row['mechanics']['c0'],c0_16=mech['c0'],
                               difference=mech['c0']-row['mechanics']['c0'],mechanics=mech))
        save(path,out);print(key,out['rows'][-1]['difference'],flush=True)
    x=np.array([r['delta'] for r in out['rows']]);y=np.array([1-r['c0_16']/.7998 for r in out['rows']])
    # Four parameters at four sizes: a sensitivity diagnostic, not a new test.
    A=np.c_[np.ones(len(x)),x,x*x,x*x*x]
    out['cubic_sensitivity_coefficients']=np.linalg.lstsq(A,y,rcond=None)[0].tolist()
    out['cpu_s']+=time.process_time()-start;save(path,out)
if __name__=='__main__':main()
