"""Mesh check of a near-zero single-flip pressure change; q=0 Schur limit."""
from penrose_core import *
from penrose_mc import mc
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=5010001);ap.add_argument('--out',default='penrose_small_mesh_raw.json');ap.add_argument('--bloch',action='store_true');args=ap.parse_args()
    start=time.process_time();base=Penrose(named('fib_6'),phase(449));flip=base.copy();mc(flip,args.seed,accepted=1)
    rows=[]
    for sub in [2,4,8,16]:
        vals=[]
        for t in [base,flip]:
            f=Frame(t,sub=sub);vals.append(f.critical())
        row=dict(sub=sub,baseline=vals[0]['normalized'],flipped=vals[1]['normalized'],change=vals[1]['normalized']-vals[0]['normalized'],details=vals)
        rows.append(row);save(args.out,dict(source=Path(__file__).name,seed=args.seed,rows=rows,cpu_s=time.process_time()-start))
        print(sub,row['baseline'],row['flipped'],row['change'],flush=True)
    if args.bloch:
        checks=[]
        for name,t in [('baseline',base),('flip',flip)]:
            rr=richardson(t,sub=2,angles=16)
            checks.append(dict(name=name,richardson=rr))
            print(name,'Richardson',rr['sampled_pc'],flush=True)
        data=dict(source=Path(__file__).name,seed=args.seed,rows=rows,bloch_checks=checks,cpu_s=time.process_time()-start)
        save(args.out,data)
if __name__=='__main__':main()
