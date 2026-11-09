"""Independent fixed-window correlation on 48 fresh configurations, all P.

Three deliberately different disorder protocols; no equilibrium claim.
Primary pressure matches main window statistic (Gamma); other branches saved.
"""
from ab_common import *
import argparse

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-budget',type=float,default=2400);args=ap.parse_args()
    start=time.process_time();base=canonical();win=Window(base);rows=[]
    settings=[]
    for k in [10,30,100,300,1000,3000]:
        for rep in range(4):settings.append(('global',k,rep))
    for radius in [5.,8.,11.]:
        for rep in range(4):settings.append(('region',radius,rep))
    for radius in [3.,4.5,6.]:
        for rep in range(4):settings.append(('biased',radius,rep))
    for index,(protocol,severity,rep) in enumerate(settings):
        if time.process_time()-start>args.cpu_budget:break
        seed=41001+index;rng=np.random.default_rng(seed);mark=time.process_time()
        if protocol=='global':
            t=base.copy();moves=[t.mc(rng,accepted=int(severity))]
        elif protocol=='region':
            t=RegionTiling(base.L,n=base.n);t.region_radius=severity
            moves=[t.mc(rng,attempts=80*len(t.n))]
        else:
            t=base.copy();moves=[]
            for beta in [2.,10.,40.]:
                moves.append(t.mc(rng,attempts=30*len(t.n),beta=beta,plant=(severity,(0,2))))
        st=win.stats(t);mc_cpu=time.process_time()-mark
        mech=pressure(t,sub=4,ks=[[.5,0],[0,.5],[.5,.5]])
        row=dict(index=index,seed=seed,protocol=protocol,severity=severity,replicate=rep,
                 moves=moves,mc_cpu_s=mc_cpu,**st,mechanics=mech)
        rows.append(row)
        print(json.dumps(dict(index=index,protocol=protocol,severity=severity,moved=st['moved_fraction'],
                              excess=st['mean_excess'],Gamma=mech['Gamma'],pc=mech['minimum_sampled'])),flush=True)
        save('ab_window_raw.json',dict(classification='P',source_script='ab_window.py',
                N=len(base.n),sub=4,matched_offset=True,rows=rows,cpu_s=time.process_time()-start,
                requested=len(settings),baseline_window=win.stats(base),
                definition='Primary excess = positive normalized facet violation, matching main. Euclidean and recentered controls also saved.'))
