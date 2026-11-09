"""Uniform torus-phase check of the matched-pair selection test.

Choose four independent uniform grid offsets and solve D.gamma=0 for the
fifth. This is Haar measure on the four-dimensional offset subtorus. It
checks sensitivity to the finite phase sample in the main slope scan.
"""
from penrose_core import *

def main():
    start=time.process_time();path=ROOT/'penrose_haar_raw.json'
    data=json.loads(path.read_text()) if path.exists() else dict(source=Path(__file__).name,rows=[])
    done={r['id'] for r in data['rows']}
    for name in ['a_3','fib_2','a_4','fib_3','a_5','fib_4']:
        for seed in range(4401,4425):
            key=f'{name}_haar{seed}'
            if key in done:continue
            g=np.zeros(5);g[:4]=np.random.default_rng(seed).uniform(-.5,.5,4);g[4]=-D[:4]@g[:4]
            L=named(name);t=Penrose(L,g)
            row=dict(id=key,group=name,seed=seed,gamma=g.tolist(),L=L.tolist(),V=len(t.n),components=components(L),angular=angular(t))
            data['rows'].append(row);save(path.name,data)
            print(key,row['angular']['c0'],row['angular']['h4'][0],flush=True)
    data['last_invocation_cpu_s']=time.process_time()-start;save(path.name,data)
if __name__=='__main__':main()
