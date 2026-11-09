"""Independent multigrid dual generator and reversible local flip MC.

No main implementation is imported. Canonical torus vertices are obtained
from intersections of four periodic grid families, not a window BFS.
Physical edges remain the AB unit star. Writes are the caller's responsibility.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import itertools
import time
import numpy as np
from scipy.spatial import cKDTree

A=2**-.5
P=np.array([[1,A,0,-A],[0,A,1,A]])
Q=np.array([[1,-A,0,A],[0,A,-1,A]])
U=np.eye(4,dtype=np.int64)
SIGNED=np.r_[U,-U]
ROT=np.array([[0,0,0,-1],[1,0,0,0],[0,1,0,0],[0,0,1,0]],dtype=np.int64)


def family_lifts(p,q,family='B1'):
    if family in ['B1','B2']:
        L=np.array([[q,p,0,-p],[0,p,q,p]],dtype=np.int64)
    elif family in ['E2a','E2b','E2diag']:
        L=np.array([[q,p,0,-p],[0,q,2*p,q]],dtype=np.int64)
    else: raise ValueError(family)
    if family in ['B2','E2diag']: L=L@(np.eye(4,dtype=np.int64)+ROT).T
    if family=='E2b': L=L@ROT.T
    return L


def slope(L):
    return Q@L.T@np.linalg.inv(P@L.T)


def slope_components(S):
    # w=(a+ib)z+(c+id)conjugate(z)
    return np.array([(S[0,0]+S[1,1])/2,(S[1,0]-S[0,1])/2,
                     (S[0,0]-S[1,1])/2,(S[0,1]+S[1,0])/2])


class Tiling:
    def __init__(self,L,offset=None,n=None):
        self.L=np.asarray(L,dtype=np.int64)
        self.cell=self.L@P.T; self.invcell=np.linalg.inv(self.cell)
        self.reduce_map=P.T@self.invcell
        self.S=slope(self.L)
        if n is None:
            self.n=self.generate(offset)
        else: self.n=np.asarray(n,dtype=np.int64).copy()
        self.lookup={tuple(v):i for i,v in enumerate(self.n)}
        self.adj=[{} for _ in self.n]
        for i,v in enumerate(self.n):
            for axis,step in enumerate(U):
                raw=v+step; rep,_=self.wrap(raw)
                j=self.lookup.get(tuple(rep))
                if j is not None:
                    self.adj[i][j]=axis; self.adj[j][i]=axis+4
        if min(map(len,self.adj))<3:
            # Pure one-vertex lattices have parallel/self adjacency; they are
            # supported mechanically by edges(), but never used for MC.
            self.mc_safe=False
        else: self.mc_safe=True
        self.attempted=0; self.accepted=0

    def wrap(self,n):
        shift=np.floor(n@self.reduce_map+.5+1e-10).astype(np.int64)
        return n-shift@self.L,shift

    def generate(self,offset):
        gamma=np.array([.137,.271,.319,.413]) if offset is None else np.asarray(offset,float)
        vertices=set(); expected=0
        for i,j in itertools.combinations(range(4),2):
            B=self.L[:,[i,j]].T
            det=int(B[0,0]*B[1,1]-B[0,1]*B[1,0])
            if det==0: continue
            if det<0: raise ValueError('Nonpositive oriented tile minor')
            expected+=det
            corners=np.array([[0,0],[0,1],[1,0],[1,1]])@B.T+gamma[[i,j]]
            bounds=[range(int(np.floor(corners[:,h].min()))-1,int(np.ceil(corners[:,h].max()))+1) for h in [0,1]]
            inv=np.linalg.inv(B)
            grid=np.array(list(itertools.product(*bounds)),dtype=np.int64)
            uv=(grid-gamma[[i,j]])@inv.T
            keep=np.all((uv>=-1e-10)&(uv<1-1e-10),axis=1)
            grid=grid[keep]; uv=uv[keep]
            assert len(grid)==det,(i,j,len(grid),det)
            for kk,y in zip(grid,uv):
                raw=np.ceil(y@self.L+gamma-1e-9).astype(np.int64)
                raw[i],raw[j]=kk
                for ai,aj in itertools.product([0,1],repeat=2):
                    v=raw+ai*U[i]+aj*U[j]
                    rep,_=self.wrap(v); vertices.add(tuple(rep))
        n=np.array(sorted(vertices),dtype=np.int64)
        assert len(n)==expected,(len(n),expected)
        return n

    @property
    def x(self): return self.n@P.T

    def edges(self):
        result=[]
        for i,v in enumerate(self.n):
            for axis,step in enumerate(U):
                rep,sh=self.wrap(v+step); j=self.lookup.get(tuple(rep))
                if j is not None: result.append((i,j,int(sh[0]),int(sh[1])))
        assert len(result)==2*len(self.n)
        return result

    def copy(self): return Tiling(self.L,n=self.n)

    def nearest_delta(self,x):
        frac=x@self.invcell
        return (frac-np.round(frac))@self.cell

    def edge_penalty(self,raw,step,radius,target):
        if int(np.argmax(abs(step))) in target: return 0.
        midpoint=(raw+.5*step)@P.T
        d=self.nearest_delta(midpoint)
        # A smooth field drives unwanted ribbons out through the boundary.
        # The same energy difference is used in forward/reverse transitions.
        return max(0.,1.-float(d@d)/(radius*radius))

    def attempt(self,rng,beta=0.,plant=None,hard=False):
        self.attempted+=1
        i=int(rng.integers(len(self.n)))
        if len(self.adj[i])!=3: return False
        codes=list(self.adj[i].values()); steps=SIGNED[codes]
        delta=steps@P.T
        angle=np.sort(np.arctan2(delta[:,1],delta[:,0]))
        if np.max(np.diff(np.r_[angle,angle[0]+2*np.pi]))>=np.pi-1e-9: return False
        old=self.n[i].copy(); raw=old+steps.sum(axis=0)
        new,_=self.wrap(raw)
        if tuple(new) in self.lookup: return False
        opposite=[]
        for h in range(3):
            rep,_=self.wrap(raw-steps[h]); j=self.lookup.get(tuple(rep))
            if j is None: return False
            opposite.append(j)
        actual=set()
        for step in SIGNED:
            rep,_=self.wrap(raw+step); j=self.lookup.get(tuple(rep))
            if j is not None and j!=i: actual.add(j)
        if actual!=set(opposite): return False
        if plant is not None:
            radius,target=plant
            before=sum(self.edge_penalty(old,step,radius,target) for step in steps)
            after=sum(self.edge_penalty(raw,-step,radius,target) for step in steps)
            if hard:
                if any(self.edge_penalty(raw,-step,radius,target)>0 for step in steps): return False
            change=after-before
            if not hard and change>0 and rng.random()>=np.exp(-beta*change): return False
        for j in list(self.adj[i]): del self.adj[j][i]
        self.adj[i]={}
        del self.lookup[tuple(old)]; self.lookup[tuple(new)]=i; self.n[i]=new
        for code,j in zip(codes,opposite):
            self.adj[i][j]=(code+4)%8; self.adj[j][i]=code
        self.accepted+=1
        return True

    def mc(self,rng,attempts=None,accepted=None,beta=0.,plant=None,max_factor=500,hard=False):
        if not self.mc_safe: raise ValueError('Cell too small for flip MC')
        a0,b0=self.attempted,self.accepted
        limit=attempts if attempts is not None else accepted*max_factor
        for _ in range(limit):
            self.attempt(rng,beta,plant,hard)
            if accepted is not None and self.accepted-b0>=accepted: break
        return dict(attempted=self.attempted-a0,accepted=self.accepted-b0)

    def plant_quality(self,radius,target):
        # A radius reduced by one edge length is reported as the verified
        # core; no pure-patch claim if a forbidden edge crosses it.
        wrong=[]; correct=0
        for i,v in enumerate(self.n):
            for axis,step in enumerate(U):
                rep,_=self.wrap(v+step)
                if tuple(rep) not in self.lookup: continue
                d=self.nearest_delta((v+.5*step)@P.T); rr=float(np.linalg.norm(d))
                if rr<radius:
                    if axis in target: correct+=1
                    else: wrong.append(rr)
        pure_radius=max(0.,min(wrong,default=radius)-1.)
        return dict(requested_radius=radius,wrong_edges=len(wrong),target_edges=correct,
                    verified_pure_radius=pure_radius,
                    requested_patch_verified=not wrong and correct>0)

    def phason_variance(self,radius=3.):
        x=self.x; h=self.n@Q.T-x@self.S.T
        images=np.array(list(itertools.product([-1,0,1],repeat=2)))@self.cell
        xx=np.concatenate([x+d for d in images]); hh=np.tile(h,(len(images),1))
        tree=cKDTree(xx); gradients=[]
        for center in x:
            ids=tree.query_ball_point(center,radius)
            dx=xx[ids]-center
            design=np.c_[np.ones(len(ids)),dx]
            weights=np.exp(-2*np.sum(dx*dx,axis=1)/radius**2)
            coef=np.linalg.lstsq(design*weights[:,None]**.5,hh[ids]*weights[:,None]**.5,rcond=None)[0]
            gradients.append(coef[1:].T)
        g=np.array(gradients); mean=g.mean(axis=0)
        return dict(radius=radius,variance=float(np.mean(np.sum((g-mean)**2,axis=(1,2)))),
                    mean=mean.tolist(),height_variance=float(np.mean(np.sum((h-h.mean(0))**2,axis=1))))


if __name__=='__main__':
    from beam_check import geometry_check
    for family in ['B1','B2','E2a','E2b','E2diag']:
        L=family_lifts(2,3,family); t=Tiling(L)
        print(family,len(t.n),slope_components(t.S),geometry_check(t.x,t.edges(),t.cell),flush=True)
        if family=='B1':
            print(t.mc(np.random.default_rng(1),accepted=40))
            print(geometry_check(t.x,t.edges(),t.cell),t.phason_variance(2.))
