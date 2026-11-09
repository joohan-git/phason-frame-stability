"""Fresh independent uniform and disk-confined Penrose flip ensembles.

Every (seed,count) starts from the same unflipped cell, with a fresh MC seed.
3000 attempted updates per vertex is an explicit run length, NOT a proof of
equilibration. Later time checks are stored separately from correlation rows.
"""
from penrose_core import *
from penrose_mc import mc

def main():
    start=time.process_time();path=ROOT/'penrose_disorder_raw.json'
    data=json.loads(path.read_text()) if path.exists() else dict(source=Path(__file__).name,rows=[])
    done={r['id'] for r in data['rows']};base=Penrose(named('fib_6'),phase(449));window=Window(base)
    if 'baseline' not in data:
        data['baseline']=dict(geometry=base.validate(),gamma=base.gamma.tolist(),L=base.L.tolist(),
                              threshold=threshold(base),angular=angular(base),richardson=richardson(base),window=window.stats(base))
        save(path.name,data);print('baseline',data['baseline']['threshold']['pc'],flush=True)
    for seed in range(501,513):
        counts=[1,3,10,30,100,300,1000,3000] if seed<=503 else [1,3,10]
        for count in counts:
            key=f'uniform_{seed}_k{count}'
            if key in done:continue
            t=base.copy();m=mc(t,seed*10000+count,accepted=count)
            row=dict(id=key,protocol='uniform',seed=seed,flips=count,mc=m,window=window.stats(t),threshold=threshold(t))
            data['rows'].append(row);save(path.name,data)
            print(key,row['threshold']['pc'],row['window']['max_excess'],flush=True)
    for seed in [501,502,503]:
        key=f'long_{seed}'
        if key in done:continue
        t=base.copy();m=mc(t,seed*10000+9001,attempts=3000*len(t.n))
        row=dict(id=key,protocol='long_uniform',seed=seed,sweeps=3000,mc=m,window=window.stats(t),threshold=threshold(t),geometry=t.validate())
        np.savez_compressed(ROOT/f'penrose_{key}.npz',n=t.n,L=t.L,gamma=t.gamma)
        data['rows'].append(row);save(path.name,data)
        print(key,row['threshold']['pc'],row['window'],flush=True)
        m2=mc(t,seed*10000+9002,attempts=1500*len(t.n))
        row2=dict(id=key+'_later',protocol='mixing_check',seed=seed,sweeps=4500,mc=m2,window=window.stats(t),threshold=threshold(t))
        data['rows'].append(row2);save(path.name,data)
        print(row2['id'],row2['threshold']['pc'],flush=True)
    for radius in [4.,7.,10.]:
        for seed in [511,512]:
            key=f'region_R{radius}_{seed}'
            if key in done:continue
            t=base.copy();m=mc(t,seed*10000+int(radius),attempts=3000*len(t.n),radius=radius)
            row=dict(id=key,protocol='region',radius=radius,seed=seed,sweeps=3000,mc=m,window=window.stats(t),threshold=threshold(t),geometry=t.validate())
            np.savez_compressed(ROOT/f'penrose_{key}.npz',n=t.n,L=t.L,gamma=t.gamma)
            data['rows'].append(row);save(path.name,data)
            print(key,row['threshold']['pc'],row['window'],flush=True)
    # Full uncondensed eigen and mesh checks on one strongly disordered cell.
    if 'mesh_check' not in data:
        z=np.load(ROOT/'penrose_long_501.npz');t=Penrose(z['L'],z['gamma'],z['n'])
        data['mesh_check']=dict(sub4=threshold(t,sub=4),full_sub2=richardson(t,sub=2,angles=4),full_sub4=richardson(t,sub=4,angles=4))
        save(path.name,data)
    data['last_invocation_cpu_s']=time.process_time()-start;save(path.name,data)

if __name__=='__main__':main()
