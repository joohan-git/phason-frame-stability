"""Offline AB slope test, equal physical edges, rigid 2D beam joints.

Uses the portfolio's unmodified rigidity_check.py generator.
EA=1, EI=1e-4, edge length=1. Relaxed unit HYDROSTATIC STRESS preload.
Tangent is the standard axial-force Euler-Bernoulli beam-column proxy:
K(p)=K-p H; it omits initial-shear/end-moment terms of final K08 theory.
Bloch minima are numerical upper estimates, not interval certificates.
All outputs are written beside this script. No network or external writes.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize, minimize_scalar, brentq
from rigidity_check import approximant, projections, UNIT

ROOT = Path(__file__).resolve().parent


def element(delta, EI=1e-4):
    L = la.norm(delta)
    c, s = delta / L
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
    T = la.block_diag(R, R)
    K = np.zeros((6, 6))
    K[np.ix_([0, 3], [0, 3])] = np.array([[1, -1], [-1, 1.]]) / L
    ids = [1, 2, 4, 5]
    K[np.ix_(ids, ids)] = EI / L**3 * np.array([
        [12, 6*L, -12, 6*L], [6*L, 4*L*L, -6*L, 2*L*L],
        [-12, -6*L, 12, -6*L], [6*L, 2*L*L, -6*L, 4*L*L]])
    G = np.zeros((6, 6))
    G[np.ix_(ids, ids)] = np.array([
        [36, 3*L, -36, 3*L], [3*L, 4*L*L, -3*L, -L*L],
        [-36, -3*L, 36, -3*L], [3*L, -L*L, -3*L, 4*L*L]])/(30*L)
    return T.T @ K @ T, T.T @ G @ T


def torus_edges(n, lifts, cell):
    physical = projections()[0]
    lookup = {tuple(v): i for i, v in enumerate(n)}
    edges = []
    for i, v in enumerate(n):
        for step in UNIT:
            raw = v + step
            wrap = np.floor((raw @ physical.T) @ la.inv(cell) + .5 + 1e-9).astype(int)
            j = lookup.get(tuple(raw - wrap @ lifts))
            if j is not None:
                edges.append((i, j, *wrap))
    return edges


def flip_torus(n, p, q, count=1, seed=261108):
    """Legal three-rhombus flips keep period lifts (hence exact slope) fixed."""
    rng = np.random.default_rng(seed)
    n = n.copy()
    physical = projections()[0]
    lifts = np.array([[q, p, 0, -p], [0, p, q, p]])
    cell = lifts @ physical.T
    invcell = la.inv(cell)
    def wrap(raw):
        sh = np.floor((raw @ physical.T) @ invcell + .5 + 1e-9).astype(int)
        return raw - sh @ lifts
    accepted = 0
    for attempt in range(count * len(n) * 100):
        lookup = {tuple(v): j for j, v in enumerate(n)}
        i = int(rng.integers(len(n)))
        steps = [d for d in np.r_[UNIT, -UNIT] if tuple(wrap(n[i]+d)) in lookup]
        if len(steps) != 3:
            continue
        d = np.array(steps)
        ang = np.sort(np.arctan2(*(d @ physical.T)[:, ::-1].T))
        if np.max(np.diff(np.r_[ang, ang[0]+2*np.pi])) >= np.pi-1e-8:
            continue
        new = wrap(n[i]+d.sum(axis=0))
        if tuple(new) in lookup:
            continue
        opposite = [tuple(wrap(n[i]+d[j]+d[k])) for j, k in [(0,1),(1,2),(0,2)]]
        if not all(v in lookup for v in opposite):
            continue
        actual = {tuple(wrap(new+v)) for v in np.r_[UNIT,-UNIT]
                  if tuple(wrap(new+v)) in lookup and tuple(wrap(new+v)) != tuple(n[i])}
        if actual != set(opposite):
            continue
        n[i] = new
        accepted += 1
        if accepted == count:
            break
    assert accepted == count, (accepted, count)
    edges = torus_edges(n, lifts, cell)
    assert len(edges) == 2*len(n)
    return n, n @ physical.T, edges, cell


def geometry_check(x, edges, cell):
    """Torus half-edge walk: every face is an AB square/rhombus, total area matches."""
    directed = []
    adj = [[] for _ in x]
    length = []
    degree = np.zeros(len(x), int)
    for i,j,a,b in edges:
        d = x[j]-x[i]+np.array([a,b]) @ cell
        length.append(la.norm(d))
        degree[i] += 1; degree[j] += 1
        for u,v,dd in [(i,j,d),(j,i,-d)]:
            h = len(directed); directed.append((u,v,dd))
            adj[u].append(h)
    for aa in adj:
        aa.sort(key=lambda h: np.arctan2(directed[h][2][1], directed[h][2][0]))
    visited=set(); areas=[]; sizes=[]
    for start in range(len(directed)):
        if start in visited:
            continue
        h=start; pts=[np.zeros(2)]
        while h not in visited:
            visited.add(h)
            u,v,d=directed[h]
            pts.append(pts[-1]+d)
            aa=adj[v]; h=aa[(aa.index(h^1)-1)%len(aa)]
        assert h==start and la.norm(pts[-1])<1e-8
        pts=np.array(pts[:-1]); sizes.append(len(pts))
        nxt=np.roll(pts,-1,axis=0)
        areas.append(float(np.sum(pts[:,0]*nxt[:,1]-pts[:,1]*nxt[:,0])/2))
    assert set(sizes)=={4}, set(sizes)
    assert np.max(np.abs(np.array(length)-1))<1e-10
    assert abs(sum(areas)-abs(la.det(cell)))<1e-8
    assert len(x)-len(edges)+len(areas)==0
    return dict(vertices=len(x),edges=len(edges),faces=len(areas),
                degree_histogram={str(i):int(sum(degree==i)) for i in np.unique(degree)},
                square_faces=int(sum(abs(np.array(areas)-1)<1e-8)),
                rhombus_faces=int(sum(abs(np.array(areas)-2**-.5)<1e-8)),
                area=float(abs(la.det(cell))),edge_length_range=[min(length),max(length)])


class Frame:
    def __init__(self, x, edges, cell, div=1, EI=1e-4):
        self.original_n=len(x); self.cell=cell; self.area=abs(la.det(cell))
        nodes=x.tolist(); elements=[]
        for i,j,a,b in edges:
            delta=x[j]-x[i]+np.array([a,b])@cell
            ids=[i]
            for h in range(1,div):
                ids.append(len(nodes)); nodes.append((x[i]+h/div*delta).tolist())
            ids.append(j)
            for h in range(div):
                elements.append((ids[h],ids[h+1],np.array([a,b]) if h==div-1 else np.zeros(2,int),delta/div))
        self.n=len(nodes); self.ndof=3*self.n
        self.dof=np.array([np.r_[3*i+np.arange(3),3*j+np.arange(3)] for i,j,*_ in elements])
        self.shifts=np.array([v[2] for v in elements]); self.delta=np.array([v[3] for v in elements])
        em=[element(d,EI) for d in self.delta]
        self.Ke=np.array([v[0] for v in em]); self.Ge=np.array([v[1] for v in em])
        self.rows=np.repeat(self.dof,6,axis=1).ravel(); self.cols=np.tile(self.dof,(1,6)).ravel()
        self.free=np.arange(2,self.ndof)
        self.K=self.assemble(self.Ke).real; self.Kf=self.K[self.free][:,self.free].tocsc()
        fac=sla.splu(self.Kf)
        strains=[np.array([[1.,0],[0,0]]),np.array([[0,0],[0,1.]]),np.array([[0,1.],[1.,0]])/np.sqrt(2)]
        affine=np.zeros((3,len(elements),6))
        for j,E in enumerate(strains): affine[j,:,3:5]=self.delta@E.T
        forces=np.array([self.scatter(np.einsum('eij,ej->ei',self.Ke,d)) for d in affine]).T
        w=np.zeros((self.ndof,3)); w[self.free]=-fac.solve(forces[self.free])
        relaxed=affine+np.transpose(w[self.dof],(2,0,1))
        C=np.einsum('aei,eij,bej->ab',relaxed,self.Ke,relaxed)/self.area
        eps=la.solve(C,[-1.,-1.,0.],assume_a='pos')
        de=np.einsum('a,aei->ei',eps,relaxed)
        ef=np.einsum('eij,ej->ei',self.Ke,de)
        direction=self.delta/la.norm(self.delta,axis=1)[:,None]
        tension=np.einsum('ei,ei->e',ef[:,3:5],direction)
        stress=ef[:,3:5].T@self.delta/self.area
        residual=la.norm(self.scatter(ef))/max(la.norm(ef),1e-30)
        assert residual<1e-8, residual
        assert la.norm(stress+np.eye(2))<1e-8, stress
        self.He=-tension[:,None,None]*self.Ge
        self.affine4=np.zeros((4,len(elements),6))
        for j in range(4):
            F=np.zeros((2,2)); F.flat[j]=1
            self.affine4[j,:,3:5]=self.delta@F.T
        self.pre=dict(equilibrium_residual=float(residual),stress=stress.tolist(),
                      strain_per_unit_pressure=eps.tolist(),C=C.tolist(),
                      compression_range=[float(-tension.max()),float(-tension.min())],
                      max_end_moment=float(abs(ef[:,[2,5]]).max()))

    def scatter(self,f):
        out=np.zeros(self.ndof); np.add.at(out,self.dof.ravel(),f.ravel()); return out

    def assemble(self,em,k=(0.,0.)):
        em=em.astype(complex,copy=True)
        ph=np.exp(2j*np.pi*(self.shifts@np.asarray(k)))
        em[:,:3,3:]*=ph[:,None,None]; em[:,3:,:3]*=ph.conj()[:,None,None]
        return sp.coo_matrix((em.ravel(),(self.rows,self.cols)),shape=(self.ndof,self.ndof)).tocsr()

    def eigen(self,k):
        k=np.asarray(k)
        gamma=la.norm(k-np.round(k))<1e-12
        free=self.free if gamma else np.arange(self.ndof)
        K=self.assemble(self.Ke,k)[free][:,free].tocsc()
        H=self.assemble(self.He,k)[free][:,free].tocsc()
        if gamma: K=K.real.copy(); H=H.real.copy()
        if len(free)<350:
            scale=1/np.sqrt(K.diagonal().real)
            val,v=la.eigh(H.toarray()*scale[:,None]*scale[None,:],K.toarray()*scale[:,None]*scale[None,:],
                         subset_by_index=(len(free)-1,len(free)-1),check_finite=False)
            v=scale*v[:,0]; mu=val[0]
        else:
            fac=sla.splu(K)
            inv=sla.LinearOperator(K.shape,matvec=fac.solve,dtype=K.dtype)
            val,vec=sla.eigsh(H,k=1,M=K,Minv=inv,which='LA',tol=2e-8,
                             v0=np.random.default_rng(17).normal(size=len(free)),maxiter=5000)
            mu=val[0]; v=vec[:,0]
        pressure=1/mu
        res=la.norm(K@v-pressure*(H@v))/(la.norm(K@v)+pressure*la.norm(H@v))
        return float(pressure),float(res)

    def acoustic_tensor_min(self,pressure):
        """Exact k->0 Schur-complement limit of THIS beam-column proxy."""
        em=self.Ke-pressure*self.He
        M=self.assemble(em).real.copy()[self.free][:,self.free].tocsc()
        fac=sla.splu(M)
        f=np.array([self.scatter(np.einsum('eij,ej->ei',em,d)) for d in self.affine4]).T
        C=np.einsum('aei,eij,bej->ab',self.affine4,em,self.affine4)
        C-=f[self.free].T@fac.solve(f[self.free]); C/=self.area
        C=(C+C.T)/2
        def evaluate(t):
            c,s=np.cos(t),np.sin(t)
            L=np.array([[c,0],[s,0],[0,c],[0,s]])
            return float(la.eigvalsh(L.T@C@L,check_finite=False)[0])
        angles=np.arange(48)*np.pi/48
        vals=np.array([evaluate(t) for t in angles]); order=np.argsort(vals)
        best=(vals[order[0]],angles[order[0]])
        for idx in order[:3]:
            t=angles[idx]
            opt=minimize_scalar(evaluate,bounds=(t-np.pi/48,t+np.pi/48),method='bounded',
                                options={'xatol':1e-10})
            if opt.fun<best[0]: best=(float(opt.fun),float(opt.x))
        return best

    def acoustic_threshold(self,estimate):
        # This is used only if a sign change occurs below the Gamma pole.
        gamma=self.eigen((0,0))[0]
        hi=min(1.2*estimate,gamma*(1-1e-5))
        if self.acoustic_tensor_min(hi)[0]>0: return None
        root=brentq(lambda p:self.acoustic_tensor_min(p)[0],0,hi,xtol=2e-13)
        value,theta=self.acoustic_tensor_min(root)
        return dict(pressure=float(root),pressure_over_EI=float(root/1e-4),
                    angle_radians=float(theta),acoustic_eigenvalue=float(value),
                    gamma_pressure=float(gamma))


def scan(p,q,div=1,flips=0,grid=4,refine=True):
    start=time.perf_counter()
    n,x,edges,cell=approximant(p,q)
    if flips: n,x,edges,cell=flip_torus(n,p,q,flips)
    geo=geometry_check(x,edges,cell)
    f=Frame(x,edges,cell,div=div)
    pts=[np.array([i,j])/grid for i in range(grid) for j in range(grid)]
    pts += [r*np.array([np.cos(t),np.sin(t)]) for r in [.001,.0003] for t in np.arange(12)*np.pi/12]
    vals=[]
    for k in pts:
        lam,res=f.eigen(k); vals.append((lam,k,res))
    vals.sort(key=lambda a:a[0]); best=vals[0]
    if refine:
        for lam,k,res in vals[:2]:
            if la.norm(k)<.01: continue
            opt=minimize(lambda kk:f.eigen(kk)[0],k,method='Nelder-Mead',
                         options={'maxiter':70,'xatol':1e-5,'fatol':1e-10})
            laa,res=f.eigen(opt.x)
            if laa<best[0]: best=(laa,opt.x,res)
        # Refine the direction of long-wavelength branches; a coarse angular
        # grid can miss the lowest elastic mode in a flipped approximant.
        for radius in [.001,.0003]:
            angular=sorted([v for v in vals if abs(la.norm(v[1])-radius)<1e-8],key=lambda a:a[0])
            for lam,k,res in angular[:2]:
                theta=np.arctan2(k[1],k[0])
                def at_angle(t): return f.eigen(radius*np.array([np.cos(t),np.sin(t)]))[0]
                opt=minimize_scalar(at_angle,bounds=(theta-np.pi/12,theta+np.pi/12),method='bounded',
                                    options={'xatol':2e-5,'maxiter':35})
                kk=radius*np.array([np.cos(opt.x),np.sin(opt.x)])
                laa,res=f.eigen(kk)
                vals.append((laa,kk,res))
                if laa<best[0]: best=(laa,kk,res)
    acoustic=f.acoustic_threshold(best[0])
    row=dict(p=p,q=q,t=p/q,flips=flips,subdivisions=div,geometry=geo,preload=f.pre,
             pressure=best[0],pressure_over_EI=best[0]/1e-4,k=best[1].tolist(),residual=best[2],
             max_sampled_residual=max(v[2] for v in vals),grid_min=vals[0][0],
             sample_points=len(pts),ndof=f.ndof,seconds=time.perf_counter()-start,
             acoustic_schur=acoustic,
             acoustic_min_by_radius={str(r):min(v[0] for v in vals if abs(la.norm(v[1])-r)<1e-8)
                                     for r in [.001,.0003]})
    print(json.dumps({key:row[key] for key in ['p','q','flips','subdivisions','pressure_over_EI','k','residual','seconds']}),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--quick',action='store_true'); parser.add_argument('--checks',action='store_true')
    args=parser.parse_args(); rows=[]
    cases=([(2,3,1,0),(2,3,4,0),(5,7,4,0),(2,3,1,1),(2,3,2,1),(2,3,4,1),
            (2,3,1,40),(2,3,2,40),(2,3,4,40)] if args.checks else
           [(1,3,2,0),(2,3,2,0),(5,7,2,0),(4,5,2,0),(1,1,2,0)])
    output=ROOT/('slope_checks.json' if args.checks else 'slope_results.json')
    for p,q,div,flips in cases:
        rows.append(scan(p,q,div,flips,grid=4 if args.quick else 6,refine=not args.quick))
        output.write_text(json.dumps(dict(model=__doc__,rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__': main()
