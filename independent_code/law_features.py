"""Preregistered lift-only features; no mechanical data enter this module."""
import os
for _k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[_k]='1'
import itertools,time
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from geometry import P as P4,Q as Q4
from penrose_core import P as P5,Q as Q5,D

FEATURES=['variance','max_excess','mean_excess','max_gradient','cluster','symmetry_cluster']

class LiftFeatures:
    def __init__(self,base,gamma,radius=4.):
        self.R=radius;self.rank=base.n.shape[1];self.P=P4 if self.rank==4 else P5
        self.U=np.linalg.svd(base.L.astype(float),full_matrices=True)[2][2:]
        self.center=self.U@(np.asarray(gamma)+.5);self.cell=base.cell
        self.area=abs(np.linalg.det(base.cell));self.N=len(base.n)
        cube=np.array(list(itertools.product([-.5,.5],repeat=self.rank)))@self.U.T
        hull=ConvexHull(cube);self.normals=hull.equations[:,:-1];self.constants=hull.equations[:,-1]
        self.boundary=cube[hull.vertices] if self.rank==4 else cube[hull.simplices]
        Q=Q4/np.sqrt(2) if self.rank==4 else np.r_[Q5/np.sqrt(2.5),D[None,:]/np.sqrt(5)]
        self.M=(Q-Q@base.L.T@np.linalg.inv(self.P@base.L.T)@self.P)@self.U.T
        h=base.n@self.U.T;self.var0=float(np.mean(np.sum((h-h.mean(0))**2,axis=1)));self.sigma=np.sqrt(self.var0)
        gg,g4=self.gradients(base);self.grad0=float(gg.max());self.g40=float(g4.max())
        assert self.distances(base).max()<1e-8

    def distances(self,t):
        y=t.n@self.U.T-self.center
        outside=np.max(y@self.normals.T+self.constants,axis=1)>1e-9
        dist=np.full(len(y),np.inf)
        if self.rank==4:
            for a,b in zip(self.boundary,np.roll(self.boundary,-1,axis=0)):
                e=b-a;u=np.clip((y-a)@e/(e@e),0,1)
                dist=np.minimum(dist,np.linalg.norm(y-a-u[:,None]*e,axis=1))
        else:
            for a,b,c in self.boundary:
                v=b-a;w=c-a;normal=np.cross(v,w);normal/=np.linalg.norm(normal)
                height=(y-a)@normal;z=y-height[:,None]*normal
                vv=v@v;ww=w@w;vw=v@w;den=vv*ww-vw*vw
                u=((z-a)@v*ww-(z-a)@w*vw)/den;q=((z-a)@w*vv-(z-a)@v*vw)/den
                inside=(u>=-1e-12)&(q>=-1e-12)&(u+q<=1+1e-12)
                dist=np.minimum(dist,np.where(inside,abs(height),np.inf))
                for aa,bb in [(a,b),(b,c),(c,a)]:
                    e=bb-aa;u=np.clip((y-aa)@e/(e@e),0,1)
                    dist=np.minimum(dist,np.linalg.norm(y-aa-u[:,None]*e,axis=1))
        return np.where(outside,dist,0.)

    def gradients(self,t):
        x=t.x;h=t.n@self.U.T
        shifts=np.array(list(itertools.product([-1,0,1],repeat=2)))@t.cell
        xx=np.concatenate([x+s for s in shifts]);hh=np.tile(h,(9,1));tree=cKDTree(xx)
        gg=[];g4=[]
        for center in x:
            ids=tree.query_ball_point(center,self.R);dx=xx[ids]-center
            W=np.exp(-np.sum(dx*dx,axis=1)/self.R**2)
            design=np.c_[np.ones(len(ids)),dx]
            G=np.linalg.lstsq(design*W[:,None],hh[ids]*W[:,None],rcond=None)[0][1:].T
            H=self.M@G;gg.append(np.linalg.norm(G))
            if self.rank==4:g4.append(np.hypot(H[0,0]-H[1,1],H[1,0]+H[0,1])/2)
            else:g4.append(np.sqrt(((H[0,0]+H[1,1])**2+(H[1,0]-H[0,1])**2)/4+H[2]@H[2]))
        return np.asarray(gg),np.asarray(g4)

    def evaluate(self,t):
        start=time.process_time();h=t.n@self.U.T;var=float(np.mean(np.sum((h-h.mean(0))**2,axis=1)))
        d=self.distances(t);gg,g4=self.gradients(t);mask=d>1e-9
        parents=np.arange(len(t.n))
        def find(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for i,j,_,_ in t.edges():
            if mask[i] and mask[j]:parents[find(j)]=find(i)
        clusters={}
        for i in np.flatnonzero(mask):clusters.setdefault(find(i),[]).append(i)
        largest=max(map(len,clusters.values()),default=0);combination=0.
        for ids in clusters.values():
            extent=np.sqrt(len(ids)*self.area/self.N)/self.R
            amp=d[ids].mean()/self.sigma+max(0.,self.R*(g4[ids].max()-self.g40)/self.sigma)
            combination=max(combination,float(amp*(-np.expm1(-extent))))
        x=[max(0.,np.sqrt(var/self.var0)-1),float(d.max()/self.sigma),float(d.mean()/self.sigma),
           max(0.,self.R*(float(gg.max())-self.grad0)/self.sigma),np.sqrt(largest*self.area/self.N)/self.R,combination]
        return dict(x=dict(zip(FEATURES,map(float,x))),variance=var,max_excess=float(d.max()),mean_excess=float(d.mean()),
                    largest_cluster=largest,defects=int(mask.sum()),gradient_max=float(gg.max()),g4_max=float(g4.max()),
                    sigma=self.sigma,variance_reference=self.var0,gradient_reference=self.grad0,g4_reference=self.g40,
                    radius=self.R,cpu_s=time.process_time()-start)
