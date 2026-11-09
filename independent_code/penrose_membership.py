"""Can a single legal flip be another canonical approximant phase?

Solve containment in a translated window while fixing the diagonal offset,
then independently regenerate the pentagrid tiling to check equality of the
complete integer lift set. This is a numerical feasibility check plus an
exact comparison of the resulting finite integer sets, not an interval proof.
"""
from penrose_core import *
from penrose_mc import mc
from scipy.optimize import linprog

def main():
    start=time.process_time();base=Penrose(named('fib_6'),phase(449));w=Window(base);rows=[]
    for seed in [5010001,5070001]:
        t=base.copy();mc(t,seed,accepted=1);y=t.n@w.perp.T-w.center
        violation=np.max(y@w.normals.T+w.constants,axis=0)
        A=np.c_[-w.normals,np.ones(len(w.normals))]
        diagonal=w.perp@D
        opt=linprog([0.,0.,0.,-1.],A_ub=A,b_ub=-violation,
                    A_eq=[np.r_[diagonal,0.]],b_eq=[0.],bounds=[(None,None)]*4,method='highs')
        row=dict(seed=seed,success=opt.success,margin=float(opt.x[3]),internal_shift=opt.x[:3].tolist())
        gamma=base.gamma+w.perp.T@opt.x[:3]
        if opt.x[3]>1e-8:
            candidate=Penrose(base.L,gamma)
            row['regenerated_identical_lift_set']={tuple(n) for n in t.n}=={tuple(n) for n in candidate.n}
            row['regenerated_difference_count']=len({tuple(n) for n in t.n}-{tuple(n) for n in candidate.n})
        else:row['regenerated_identical_lift_set']=None
        row['diagonal_offset']=float(D@gamma);rows.append(row)
    out=dict(source=Path(__file__).name,rows=rows,cpu_s=time.process_time()-start)
    save('penrose_membership_raw.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
