"""Round-3 helpers. Independent geometry/mechanics; no main imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import itertools,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import brentq,minimize
from scipy.linalg import eigvalsh
from geometry import Tiling,family_lifts,P
from frame import Frame

ROOT=Path(__file__).resolve().parent
OFFSET=np.random.default_rng(7).uniform(-.5,.5,4)*.37+np.array([.1173,.2711,-.0618,.3307])-.5

def save(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')

def canonical(p=12,q=17,matched=True):
    return Tiling(family_lifts(p,q),offset=OFFSET if matched else None)

def directional(f,theta):
    c,s=np.cos(theta),np.sin(theta)
    L=np.array([[c,0],[s,0],[0,c],[0,s]])
    def indicator(load):
        C,neg=f.macro(load)
        return -f.EI if C is None or neg else eigvalsh(L.T@C@L)[0]
    hi=min(1.15*f.lambdaE,.94*f.pole)
    root=brentq(indicator,0.,hi,xtol=1e-14,rtol=1e-11)
    return float(root/f.lambdaE)

def pressure(t,sub=4,ks=()):
    start=time.process_time();f=Frame(t,sub=sub);f.set_protocol('stress')
    crit=f.critical()
    gamma=f.bloch_threshold([0.,0.],upper=1.2*crit['critical_load'])
    modes=[f.bloch_threshold(k,upper=1.2*crit['critical_load']) for k in ks]
    return dict(acoustic_or_Gamma=crit['normalized'],theta=crit['theta'],kind=crit['kind'],
                Gamma=gamma['normalized'],finite_k=modes,
                minimum_sampled=min([crit['normalized'],gamma['normalized']]+[v['normalized'] for v in modes]),
                pressure_per_unit=f.pre['pressure_per_unit'],cpu_s=time.process_time()-start)

class Window:
    def __init__(self,base,offset=OFFSET):
        self.ref={tuple(n) for n in base.n}
        # SVD defines an independent orthonormal internal basis. All distances
        # and max normalized-facet violations are invariant under its rotation.
        self.perp=np.linalg.svd(base.L.astype(float),full_matrices=True)[2][2:]
        self.center=self.perp@(offset+.5)
        cube=np.array(list(itertools.product([-.5,.5],repeat=4)))@self.perp.T
        hull=ConvexHull(cube);self.poly=cube[hull.vertices]
        self.normals=hull.equations[:,:2];self.constants=hull.equations[:,2]
        test=self.violations(base.n)
        assert max(test)<1e-8, max(test)

    def violations(self,lifts,shift=None):
        y=lifts@self.perp.T-self.center
        if shift is not None:y=y-shift
        return np.max(y@self.normals.T+self.constants,axis=1)

    def stats(self,t):
        y=t.n@self.perp.T-self.center
        signed=self.violations(t.n);ex=np.maximum(signed,0.)
        moved=float(np.mean([tuple(n) not in self.ref for n in t.n]))
        assert abs(moved-np.mean(signed>1e-9))<1e-8
        # Actual Euclidean polygon distance: max facet violation is only a
        # lower bound near corners. Report both, with the main metric primary.
        distances=[]
        for a,b in zip(self.poly,np.roll(self.poly,-1,axis=0)):
            v=b-a;u=np.clip((y-a)@v/(v@v),0,1)
            distances.append(np.linalg.norm(y-a-u[:,None]*v,axis=1))
        eu=np.where(signed>0,np.min(distances,axis=0),0.)
        def objective(shift):
            return np.mean(np.maximum(np.max((y-shift)@self.normals.T+self.constants,axis=1),0.)**2)
        opt=minimize(objective,np.zeros(2),method='BFGS',options={'gtol':1e-10})
        rec=np.maximum(self.violations(t.n,opt.x),0.)
        return dict(moved_fraction=moved,mean_excess=float(ex.mean()),max_excess=float(ex.max()),
                    euclidean_mean_excess=float(eu.mean()),recentered_mean_excess=float(rec.mean()),
                    fitted_window_shift=opt.x.tolist(),recentered_objective=float(opt.fun),
                    internal_variance=float(np.mean(np.sum((y-y.mean(0))**2,axis=1))))

class RegionTiling(Tiling):
    """Same legal moves, but reject transitions whose old OR new site is outside."""
    region_radius=None
    def attempt(self,rng,beta=0.,plant=None,hard=False):
        if self.region_radius is None:return super().attempt(rng,beta,plant,hard)
        # Peek at exactly the selected vertex without consuming an extra RNG
        # draw. Restore state before calling the tested parent transition.
        state=rng.bit_generator.state
        i=int(rng.integers(len(self.n)))
        rng.bit_generator.state=state
        from geometry import SIGNED
        old=self.n[i]
        steps=SIGNED[list(self.adj[i].values())]
        new=old+steps.sum(axis=0)
        if len(steps)!=3 or any(np.linalg.norm(self.nearest_delta(v@P.T))>self.region_radius for v in [old,new]):
            rng.integers(len(self.n));self.attempted+=1;return False
        return super().attempt(rng,beta,plant,hard)
