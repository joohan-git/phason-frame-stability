"""Fine-cell reference and residual mesh sensitivity of the offset penalty."""
from penrose_core import *

def main():
    start=time.process_time();path=ROOT/'penrose_fine_raw.json';data=dict(source=Path(__file__).name,rows=[])
    cases=[('a_6',seed,2,0.) for seed in range(401,405)]
    cases += [('a_6',401,4,0.),('fib_6',401,4,0.)]
    cases += [('fib_5',401,8,s) for s in [0.,.1,.5]]
    for name,seed,sub,s in cases:
        t=Penrose(named(name),phase(seed,s));a=angular(t,sub)
        row=dict(id=f'{name}_{seed}_sub{sub}_s{s}',group=name,V=len(t.n),L=t.L.tolist(),seed=seed,sub=sub,s=s,
                 components=components(t.L),angular=a,continuous_pc=Frame(t,sub=sub).critical()['normalized'])
        data['rows'].append(row);save(path.name,data);print(row['id'],a['c0'],a['h4'],row['continuous_pc'],flush=True)
    data['last_invocation_cpu_s']=time.process_time()-start;save(path.name,data)
if __name__=='__main__':main()
