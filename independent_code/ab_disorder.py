"""Fresh 24-seed small-disorder ladder at main's periods/offset; all values P."""
from ab_common import *
import argparse

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-budget',type=float,default=2400)
    ap.add_argument('--seeds',type=int,default=24);args=ap.parse_args()
    start=time.process_time();base=canonical();win=Window(base)
    baseline=pressure(base,sub=4)
    rows=[];steps=[1,3,10,30,100,300,1000]
    for seed in range(31001,31001+args.seeds):
        t=base.copy();rng=np.random.default_rng(seed);last=0
        for k in steps:
            if time.process_time()-start>args.cpu_budget:break
            move=t.mc(rng,accepted=k-last);last=k
            st=win.stats(t)
            # Full extra sampled k only once cluster modes become plausible;
            # Gamma and exact acoustic are evaluated at every observation.
            ks=[[.5,0],[0,.5],[.5,.5]] if k>=100 else []
            mech=pressure(t,sub=4,ks=ks)
            row=dict(seed=seed,k=k,move=move,**st,mechanics=mech,
                     relative_drop=1-mech['minimum_sampled']/baseline['minimum_sampled'])
            rows.append(row)
            print(json.dumps(dict(seed=seed,k=k,moved=st['moved_fraction'],pc=mech['minimum_sampled'],
                                  drop=row['relative_drop'])),flush=True)
            save('ab_disorder_raw.json',dict(classification='P',source_script='ab_disorder.py',
                N=len(base.n),sub=4,matched_offset=True,baseline=baseline,rows=rows,
                cpu_s=time.process_time()-start,requested_seeds=args.seeds))
        if time.process_time()-start>args.cpu_budget:break
