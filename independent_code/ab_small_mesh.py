"""Paired mesh sensitivity of small mean losses, not just absolute pressure."""
from ab_common import *

if __name__=='__main__':
    start=time.process_time();base=canonical();f=Frame(base,sub=8);p0=f.critical()['normalized'];rows=[]
    for seed in range(31001,31025):
        t=base.copy();rng=np.random.default_rng(seed);t.mc(rng,accepted=1)
        f=Frame(t,sub=8);p=f.critical()['normalized']
        rows.append(dict(seed=seed,k=1,sub=8,pressure=p,relative_drop=1-p/p0))
        if seed<31007:
            t.mc(rng,accepted=9);f=Frame(t,sub=8);p=f.critical()['normalized']
            rows.append(dict(seed=seed,k=10,sub=8,pressure=p,relative_drop=1-p/p0))
        save('ab_small_mesh_raw.json',dict(classification='P',source_script='ab_small_mesh.py',
            N=len(base.n),baseline=p0,rows=rows,cpu_s=time.process_time()-start))
    print('DONE',time.process_time()-start,flush=True)
