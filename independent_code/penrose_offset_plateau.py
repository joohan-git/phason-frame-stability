"""Check whether 'strict maximum over s' is literal in a finite approximant."""
from penrose_core import *

def main():
    start=time.process_time();rows=[]
    for name in ['fib_4','fib_5']:
        base=Penrose(named(name),phase(401));ref={tuple(n) for n in base.n}
        for sign in [-1,1]:
            for exp in range(1,7):
                ss=sign*10.**(-exp);t=Penrose(base.L,phase(401,ss));now={tuple(n) for n in t.n}
                rows.append(dict(name=name,s=ss,V=len(t.n),changed_vertices=len(now-ref),
                                 identical_integer_lift_set=(now==ref),
                                 identical_edge_table=(np.array_equal(np.array(t.edges()),np.array(base.edges())))))
    save('penrose_offset_plateau_raw.json',dict(source=Path(__file__).name,rows=rows,cpu_s=time.process_time()-start))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
