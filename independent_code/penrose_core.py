"""independent independent rank-five pentagrid dual and frame verification utilities.

Only our AB frame / beam_check mechanics are reused. No Penrose
implementation or unreported input from the other solver is read/imported.
Lengths=1, EA=1, r=.03, p reported in pi^2 EI units. CPU threads fixed to one.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import itertools,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import brentq
from scipy.linalg import eigvalsh
from frame import Frame
from beam_check import Frame as FullFrame, geometry_check

ROOT=Path(__file__).resolve().parent
TH=np.pi*np.arange(5)/5
P=np.array([np.cos(TH),np.sin(TH)])
Q=np.array([np.cos(7*TH),np.sin(7*TH)])
D=(-1.)**np.arange(5)
U=np.eye(5,dtype=np.int64); SIGNED=np.r_[U,-U]
LAM=np.pi**2*.03**2/4

def save(name,data): (ROOT/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
def lifts(B,C,E,F):return np.array([[2*(B-C),B,C,-C,-B],[0,E,F,F,E]],dtype=np.int64)
FIB=[0,1,1,2,3,5,8,13,21,34,55,89]
def named(name):
    family,k=name.split('_');k=int(k)
    if family=='a':return lifts(FIB[k+3],FIB[k+1],FIB[k+1],FIB[k+2])
    if family=='fib':return lifts(FIB[k+2],FIB[k],FIB[k+1],FIB[k+2])
    raise ValueError(name)
def phase(seed=401,s=0.):
    g=np.random.default_rng(seed).uniform(-.5,.5,5)
    # D.gamma=0 gives the four nondegenerate canonical Penrose layers.
    # Center of projected unit cube is gamma + 1/2, with D.center=1/2+s.
    return g+(s-D@g)*D/5
def components(L):
    S=Q@L.T@np.linalg.inv(P@L.T)
    a=complex((S[0,0]+S[1,1])/2,(S[1,0]-S[0,1])/2)
    b=complex((S[0,0]-S[1,1])/2,(S[1,0]+S[0,1])/2)
    v=D@L.T@np.linalg.inv(P@L.T)/np.sqrt(5)
    return dict(a=[a.real,a.imag],b=[b.real,b.imag],v=v.tolist(),delta=float(np.sqrt(abs(a)**2+abs(b)**2)))

class Penrose:
    def __init__(self,L,gamma=None,n=None):
        self.L=np.asarray(L,dtype=np.int64);self.cell=self.L@P.T
        self.invcell=np.linalg.inv(self.cell);self.reduce_map=P.T@self.invcell
        self.gamma=phase() if gamma is None else np.asarray(gamma,float)
        self.n=self.generate() if n is None else n.copy()
        self.refresh()
    def wrap(self,n):
        sh=np.floor(n@self.reduce_map+.5+1e-10).astype(np.int64)
        return n-sh@self.L,sh
    def refresh(self):
        self.lookup={tuple(v):i for i,v in enumerate(self.n)}
        self.adj=np.full((len(self.n),10),-1,dtype=np.int64)
        for axis,step in enumerate(U):
            reps,_=self.wrap(self.n+step)
            for i,rep in enumerate(reps):
                j=self.lookup.get(tuple(rep),-1)
                if j>=0:self.adj[i,axis]=j;self.adj[j,axis+5]=i
    def generate(self):
        vertices=set();expected=0;gamma=self.gamma
        for i,j in itertools.combinations(range(5),2):
            B=self.L[:,[i,j]].T
            det=int(B[0,0]*B[1,1]-B[0,1]*B[1,0]);assert det>0
            expected+=det
            corners=np.array([[0,0],[0,1],[1,0],[1,1]])@B.T+gamma[[i,j]]
            bounds=[range(int(np.floor(corners[:,h].min()))-1,int(np.ceil(corners[:,h].max()))+1) for h in [0,1]]
            grid=np.array(list(itertools.product(*bounds)),dtype=np.int64)
            uv=(grid-gamma[[i,j]])@np.linalg.inv(B).T
            keep=np.all((uv>=-1e-10)&(uv<1-1e-10),axis=1);grid=grid[keep];uv=uv[keep]
            assert len(grid)==det
            raw=np.ceil(uv@self.L+gamma-1e-9).astype(np.int64);raw[:,i]=grid[:,0];raw[:,j]=grid[:,1]
            for ai,aj in itertools.product([0,1],repeat=2):
                reps,_=self.wrap(raw+ai*U[i]+aj*U[j]);vertices.update(map(tuple,reps))
        n=np.array(sorted(vertices),dtype=np.int64);assert len(n)==expected,(len(n),expected)
        return n
    @property
    def x(self):return self.n@P.T
    def edges(self):
        out=[]
        for axis in range(5):
            good=self.adj[:,axis]>=0;ids=np.where(good)[0];_,sh=self.wrap(self.n[ids]+U[axis])
            out.extend(zip(ids,self.adj[ids,axis],sh[:,0],sh[:,1]))
        assert len(out)==2*len(self.n)
        return out
    def copy(self):return Penrose(self.L,self.gamma,self.n)
    def validate(self):
        out=geometry_check(self.x,self.edges(),self.cell)
        minors=[int(round(np.linalg.det(self.L[:,[i,j]]))) for i,j in itertools.combinations(range(5),2)]
        out['thin_faces_from_periods']=sum(d for d,(i,j) in zip(minors,itertools.combinations(range(5),2)) if (j-i) in [1,4])
        out['fat_faces_from_periods']=len(self.n)-out['thin_faces_from_periods']
        out['integer_layers']=sorted(set((self.n@D).tolist()))
        return out

def angular(t,sub=2,angles=8):
    start=time.process_time();f=Frame(t,sub=sub);vals=[]
    for theta in np.arange(angles)*np.pi/angles:
        c,s=np.cos(theta),np.sin(theta);A=np.array([[c,0],[s,0],[0,c],[0,s]])
        def fun(p):
            C,neg=f.macro(p)
            return -f.EI if C is None else eigvalsh(A.T@C@A)[0]
        vals.append(brentq(fun,0,min(1.15*LAM,.94*f.pole),xtol=1e-13,rtol=2e-10)/LAM)
    out=harmonics(vals)
    out.update(sub=sub,pre=f.pre,cpu_s=time.process_time()-start)
    return out

def harmonics(vals):
    vals=np.asarray(vals);theta=np.arange(len(vals))*np.pi/len(vals)
    h={str(k):[float(2*np.mean(vals*np.cos(k*theta))),float(2*np.mean(vals*np.sin(k*theta)))] for k in [2,4]}
    return dict(c0=float(vals.mean()),h2=h['2'],h4=h['4'],sampled_pc=float(vals.min()),directions=vals.tolist())

def richardson(t,sub=2,dq=.002,angles=8,finite=True):
    start=time.process_time();f=FullFrame(t.x,t.edges(),t.cell,div=sub,EI=.03**2/4)
    rows=[]
    # dq is dimensionless Bloch-coordinate radius; map each physical direction
    # n to k proportional cell@n. Normalisation does not affect q->0 limit.
    for theta in np.arange(angles)*np.pi/angles:
        direction=t.cell@np.array([np.cos(theta),np.sin(theta)])
        direction/=np.linalg.norm(direction)
        p1,r1=f.eigen(dq*direction);p2,r2=f.eigen(dq/2*direction)
        rows.append(dict(theta=float(theta),p_coarse=p1/LAM,p_fine=p2/LAM,
                         extrapolated=(4*p2-p1)/3/LAM,residual=max(r1,r2)))
    out=harmonics([r['extrapolated'] for r in rows]);out.update(rows=rows,sub=sub,dq=dq)
    if finite:
        out['finite_k']=[dict(k=list(k),pc=(pr:=f.eigen(k))[0]/LAM,residual=pr[1]) for k in [(0,0),(.5,0),(0,.5),(.5,.5)]]
    out['cpu_s']=time.process_time()-start
    return out

def threshold(t,sub=2,finite=True):
    start=time.process_time();f=Frame(t,sub=sub);ac=f.critical()
    # Condensed real Gamma and half/half matrices have reliable real inertia.
    gam=f.bloch_threshold([0,0],upper=1.05*ac['critical_load'])
    rows=[gam]
    if finite:
        # Exp(i*pi*integer) has tiny numerical imaginary entries; explicitly
        # real assembly avoids treating roundoff as Hermitian pivot failure.
        original=f.assemble
        def real_half(em,k=None):
            A=original(em,k)
            if k is not None and np.max(abs(2*np.asarray(k)-np.round(2*np.asarray(k))))<1e-12:A=A.real.copy()
            return A
        f.assemble=real_half
        rows += [f.bloch_threshold(k,upper=1.05*ac['critical_load']) for k in [(.5,0),(0,.5),(.5,.5)]]
    return dict(pc=min([ac['normalized']]+[r['normalized'] for r in rows]),
                acoustic_or_Gamma=ac['normalized'],kind=ac['kind'],theta=ac['theta'],finite_k=rows,
                sub=sub,cpu_s=time.process_time()-start)

class Window:
    def __init__(self,t):
        self.ref={tuple(n) for n in t.n}
        self.perp=np.linalg.svd(t.L.astype(float),full_matrices=True)[2][2:]
        self.center=self.perp@(t.gamma+.5)
        cube=np.array(list(itertools.product([-.5,.5],repeat=5)))@self.perp.T
        hull=ConvexHull(cube)
        self.normals=hull.equations[:,:3];self.constants=hull.equations[:,3]
        self.triangles=cube[hull.simplices]
        y=t.n@self.perp.T-self.center
        assert np.max(y@self.normals.T+self.constants)<1e-8
    def stats(self,t):
        y=t.n@self.perp.T-self.center
        signed=np.max(y@self.normals.T+self.constants,axis=1);outside=signed>1e-9
        dist=np.full(len(y),np.inf)
        for a,b,c in self.triangles:
            v=b-a;w=c-a;normal=np.cross(v,w);normal/=np.linalg.norm(normal)
            height=(y-a)@normal;z=y-height[:,None]*normal
            vv=v@v;ww=w@w;vw=v@w;den=vv*ww-vw*vw
            u=((z-a)@v*ww-(z-a)@w*vw)/den
            q=((z-a)@w*vv-(z-a)@v*vw)/den
            inside=(u>=-1e-12)&(q>=-1e-12)&(u+q<=1+1e-12)
            dist=np.minimum(dist,np.where(inside,abs(height),np.inf))
            for aa,bb in [(a,b),(b,c),(c,a)]:
                edge=bb-aa;alpha=np.clip((y-aa)@edge/(edge@edge),0,1)
                dist=np.minimum(dist,np.linalg.norm(y-aa-alpha[:,None]*edge,axis=1))
        dist=np.where(outside,dist,0.)
        moved=np.mean([tuple(n) not in self.ref for n in t.n])
        return dict(max_excess=float(dist.max()),mean_excess=float(dist.mean()),
                    max_facet_excess=float(max(0,signed.max())),mean_facet_excess=float(np.maximum(signed,0).mean()),
                    internal_variance=float(np.mean(np.sum((y-y.mean(0))**2,axis=1))),
                    moved_fraction=float(moved),outside_fraction=float(outside.mean()),
                    internal_center=(y.mean(0)).tolist())

if __name__=='__main__':
    t=Penrose(named('fib_3'))
    print(t.validate());print(components(t.L));print(angular(t))
    print(richardson(t));print(Window(t).stats(t))
