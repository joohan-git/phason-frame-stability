"""Fresh phase averages, exact-size slope controls and generalized offsets.

All pressure curves use the exact condensed q=0 Schur limit; an independently
assembled uncondensed Bloch pencil checks Richardson and finite wavevectors.
Run from independent; no files outside independent are read except our own AB modules.
"""
from penrose_core import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',default='slopes');args=ap.parse_args()
    path=ROOT/('penrose_'+args.part+'_raw.json')
    data=json.loads(path.read_text()) if path.exists() else dict(source=Path(__file__).name,rows=[])
    done={r['id'] for r in data['rows']};start=time.process_time()
    def run(key,L,seed,sub=2,s=0,rich=False,group=''):
        if key in done:return
        t=Penrose(L,phase(seed,s));row=dict(id=key,group=group,L=L.tolist(),V=len(t.n),seed=seed,s=s,components=components(L))
        row['angular']=angular(t,sub)
        if rich:row['richardson']=richardson(t,sub)
        row['continuous_pc']=Frame(t,sub=sub).critical()['normalized']
        row['geometry']=t.validate()
        data['rows'].append(row);data['cpu_s']=sum(r['angular']['cpu_s']+r.get('richardson',{}).get('cpu_s',0) for r in data['rows'])
        save(path.name,data);print(key,len(t.n),row['angular']['c0'],row['angular']['h4'],row['continuous_pc'],flush=True)
    if args.part=='slopes':
        for name in ['a_3','fib_2','a_4','fib_3','a_5','fib_4','a_2','fib_5','fib_6']:
            for seed in range(401,409):
                run(f'{name}_phase{seed}',named(name),seed,rich=(seed==401 and name not in ['a_2']),group=name)
    elif args.part=='controls':
        # Identical V within each set, independently found by integer search.
        fixed={1440:[(18,7,5,8),(8,3,11,18),(13,5,7,11),(11,4,8,13),(7,3,13,21)],
               1288:[(8,3,10,16),(13,5,6,10),(10,4,8,13),(6,2,13,21)]}
        for V,cells in fixed.items():
            for j,cell in enumerate(cells):
                for seed in range(401,405):
                    run(f'fixed{V}_{j}_{seed}',lifts(*cell),seed,rich=(seed==401 and j in [0,len(cells)-1]),group=f'fixed{V}_{j}')
                if j in [0,len(cells)-1]:run(f'fixed{V}_{j}_mesh4',lifts(*cell),401,sub=4,rich=True,group=f'fixed{V}_{j}')
        for name in ['a_3','a_4','a_5','fib_3','fib_4','fib_5']:
            run(name+'_mesh4',named(name),401,sub=4,rich=True,group=name)
        run('a_3_supercell',2*named('a_3'),401,rich=True,group='supercell')
        # Exact rotation of the canonical lift, including the cube centre.
        G=np.zeros((5,5),dtype=np.int64);G[1:,0:4]=np.eye(4,dtype=np.int64);G[0,4]=-1
        key='rotation_a4'
        if key not in done:
            L=named('a_4');g=phase(401);t=Penrose(L,g);tr=Penrose(L@G.T,G@(g+.5)-.5)
            original=angular(t,angles=32);rotated=angular(tr,angles=32)
            h=complex(*original['h4']);expected=h*np.exp(4j*np.pi/5)
            data['rotation']=dict(original=original,rotated=rotated,expected_h4=[expected.real,expected.imag],
                                  error=float(abs(complex(*rotated['h4'])-expected)))
            save(path.name,data)
    elif args.part=='offsets':
        for name in ['fib_4','fib_5']:
            for s in [0.,.05,.1,.2,.3,.4,.5,-.1]:
                for seed in range(401,405):
                    run(f'{name}_s{s}_{seed}',named(name),seed,s=s,rich=(seed==401 and s in [0.,.1,.5]),group=name)
            for s in [0.,.1,.5]:run(f'{name}_s{s}_mesh4',named(name),401,sub=4,s=s,rich=True,group=name)
    data['last_invocation_cpu_s']=time.process_time()-start;save(path.name,data)

if __name__=='__main__':main()
