"""Independent condensed Euler-Bernoulli beam-column calculation, EA=1.

Extends independent r1 element code. Interior Hermite DOFs are eliminated exactly
at each load, reducing a subdivided frame to three DOFs per original vertex.
The tangent is axial-prestress-only, as in the main pilot; NOT full K08.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize_scalar,brentq
from beam_check import element
from geometry import P,Q

BASIS=np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1,1,0]],float).T/np.sqrt(2)
VD=np.array([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]],float)


def tensor_summary(C4):
    C=BASIS.T@C4@BASIS
    two=np.array([C[0,1],C[0,2]])
    four=np.array([(C[1,1]-C[2,2])/2,C[1,2]])
    angles=np.arange(64)*np.pi/32
    tt=[]
    for th in angles:
        n=np.array([np.cos(th),np.sin(th)]); t=np.array([-n[1],n[0]])
        z=np.outer(t,n).ravel(); tt.append(z@C4@z)
    tt=np.array(tt)
    harmonics={}
    for k in [2,4]:
        a=2*np.mean(tt*np.cos(k*angles)); b=2*np.mean(tt*np.sin(k*angles))
        harmonics[str(k)]=dict(cos=float(a),sin=float(b),amplitude=float(np.hypot(a,b)))
    return dict(C4=C4.tolist(),trace_deviator_rotation_basis=C.tolist(),
                elastic_spin2=two.tolist(),elastic_spin2_amplitude=float(la.norm(two)),
                elastic_spin4=four.tolist(),elastic_spin4_amplitude=float(la.norm(four)),
                rotation_row_norm=float(la.norm(C[3])),TT_mean=float(tt.mean()),TT_harmonics=harmonics)


class Frame:
    def __init__(self,tiling,sub=4,radius=.03):
        start=time.process_time()
        self.t=tiling; self.x=tiling.x; self.cell=tiling.cell; self.area=abs(la.det(self.cell))
        self.edges=np.array(tiling.edges(),dtype=np.int64)
        self.n=len(self.x); self.ndof=3*self.n; self.free=np.arange(2,self.ndof)
        self.sub=sub; self.EI=radius**2/4; self.lambdaE=np.pi**2*self.EI
        self.shifts=self.edges[:,2:]; self.delta=self.x[self.edges[:,1]]-self.x[self.edges[:,0]]+self.shifts@self.cell
        assert np.max(abs(la.norm(self.delta,axis=1)-1))<1e-8
        self.dofs=np.concatenate([3*self.edges[:,0,None]+np.arange(3),3*self.edges[:,1,None]+np.arange(3)],axis=1)
        self.rows=np.repeat(self.dofs,6,axis=1).ravel(); self.cols=np.tile(self.dofs,(1,6)).ravel()
        self.T=np.zeros((len(self.edges),6,6))
        c,s=self.delta.T
        for off in [0,3]:
            self.T[:,off,off]=c; self.T[:,off,off+1]=s
            self.T[:,off+1,off]=-s; self.T[:,off+1,off+1]=c; self.T[:,off+2,off+2]=1
        kl,gl=element(np.array([1.,0.]),self.EI)
        self.Ke=np.einsum('eji,jk,ekl->eil',self.T,kl,self.T)
        self.affine=np.zeros((4,len(self.edges),6))
        for j in range(4):
            F=np.zeros((2,2)); F.flat[j]=1
            self.affine[j,:,3:5]=self.delta@F.T
        self.K0=self.assemble(self.Ke)
        self.K0f=self.K0[self.free][:,self.free].tocsc()
        fac=self.factor(self.K0f)
        self.factor_bytes=int(fac.L.data.nbytes+fac.L.indices.nbytes+fac.L.indptr.nbytes+
                              fac.U.data.nbytes+fac.U.indices.nbytes+fac.U.indptr.nbytes)
        f=self.rhs(self.Ke)
        w=np.zeros((self.ndof,4)); w[self.free]=-fac.solve(f[self.free])
        self.relaxed=np.transpose(w[self.dofs],(2,0,1))+self.affine
        self.C0=np.einsum('aei,eij,bej->ab',self.relaxed,self.Ke,self.relaxed)/self.area
        self.C0=(self.C0+self.C0.T)/2
        self.prepare_subdivision()
        self.set_protocol('stress')
        self.init_cpu_s=time.process_time()-start

    @staticmethod
    def factor(M):
        return sla.splu(M.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})

    def prepare_subdivision(self):
        size=2*(self.sub+1)
        K=np.zeros((size,size)); G=K.copy()
        ke,ge=element(np.array([1/self.sub,0.]),self.EI)
        ids=[1,2,4,5]; ke=ke[np.ix_(ids,ids)]; ge=ge[np.ix_(ids,ids)]
        for j in range(self.sub):
            ix=np.arange(2*j,2*j+4)
            K[np.ix_(ix,ix)]+=ke; G[np.ix_(ix,ix)]+=ge
        self.Kbend=K; self.Gbend=G
        self.ends=np.array([0,1,size-2,size-1]); self.inside=np.arange(2,size-2)
        if len(self.inside):
            ii=np.ix_(self.inside,self.inside)
            self.internal_mu=la.eigvalsh(G[ii],K[ii])[-1]
        else: self.internal_mu=0.

    def set_protocol(self,protocol):
        self.protocol=protocol
        if protocol=='strain': F=-np.eye(2)
        elif protocol=='stress':
            sym=BASIS[:,:3]
            load=np.array([-np.sqrt(2),0,0])
            coef=la.solve(sym.T@self.C0@sym,load,assume_a='pos')
            F=(sym@coef).reshape(2,2)
        else: raise ValueError(protocol)
        de=np.einsum('a,aei->ei',F.ravel(),self.relaxed)
        ef=np.einsum('eij,ej->ei',self.Ke,de)
        self.force=np.einsum('ei,ei->e',ef[:,3:5],self.delta)
        self.compression=-self.force
        self.end_wrenches=ef
        sigma=ef[:,3:5].T@self.delta/self.area
        eq=self.scatter(ef)
        self.pre=dict(protocol=protocol,F=F.tolist(),stress=sigma.tolist(),
                      stress_deviator_norm=float(la.norm(sigma-np.trace(sigma)/2*np.eye(2))),
                      residual=float(la.norm(eq)/la.norm(ef)),
                      axial_compression_min=float(self.compression.min()),
                      axial_compression_max=float(self.compression.max()),
                      axial_compression_mean=float(self.compression.mean()),
                      pressure_per_unit=float(-np.trace(sigma)/2))
        assert self.pre['residual']<1e-8
        if protocol=='stress': assert la.norm(sigma+np.eye(2))<1e-8
        self.pole=(1/(max(self.compression.max(),1e-30)*self.internal_mu) if self.internal_mu else np.inf)

    def element_matrices(self,load):
        if load==0: return self.Ke
        B=self.Kbend[None,:,:]-load*self.compression[:,None,None]*self.Gbend[None,:,:]
        ee=B[:,self.ends[:,None],self.ends]
        if len(self.inside):
            ei=B[:,self.ends[:,None],self.inside]
            ii=B[:,self.inside[:,None],self.inside]
            ee=ee-ei@np.linalg.solve(ii,np.swapaxes(ei,1,2))
        local=np.zeros((len(self.edges),6,6))
        ids=np.array([1,2,4,5]); local[:,ids[:,None],ids]=ee
        local[:,0,0]=local[:,3,3]=1; local[:,0,3]=local[:,3,0]=-1
        return np.einsum('eji,ejk,ekl->eil',self.T,local,self.T)

    def scatter(self,f):
        out=np.zeros(self.ndof); np.add.at(out,self.dofs.ravel(),f.ravel()); return out

    def rhs(self,em):
        return np.array([self.scatter(np.einsum('eij,ej->ei',em,d)) for d in self.affine]).T

    def assemble(self,em,k=None):
        if k is not None and la.norm(k-np.round(k))>1e-12:
            em=em.astype(complex,copy=True); ph=np.exp(2j*np.pi*(self.shifts@np.asarray(k)))
            em[:,:3,3:]*=ph[:,None,None]; em[:,3:,:3]*=ph.conj()[:,None,None]
        return sp.coo_matrix((em.ravel(),(self.rows,self.cols)),shape=(self.ndof,self.ndof)).tocsr()

    @staticmethod
    def factor_negative(fac):
        if not np.array_equal(fac.perm_r,fac.perm_c):
            raise RuntimeError('No symmetric LU inertia: row/column permutations differ')
        diag=fac.U.diagonal()
        if np.max(abs(diag.imag))>1e-7*max(np.max(abs(diag)),1e-30):
            raise RuntimeError('Hermitian pivot has significant imaginary part')
        return int(np.sum(diag.real<0))

    def macro(self,load,require_stable=True):
        em=self.element_matrices(load)
        M=self.assemble(em)[self.free][:,self.free].tocsc()
        fac=self.factor(M)
        neg=self.factor_negative(fac)
        if neg and require_stable: return None,neg
        f=self.rhs(em)
        C=np.einsum('aei,eij,bej->ab',self.affine,em,self.affine)
        C-=f[self.free].T@fac.solve(f[self.free]); C/=self.area
        return (C+C.T)/2,neg

    @staticmethod
    def acoustic_min(C):
        def ev(theta):
            c,s=np.cos(theta),np.sin(theta)
            L=np.array([[c,0],[s,0],[0,c],[0,s]])
            return float(la.eigvalsh(L.T@C@L,check_finite=False)[0])
        angles=np.arange(32)*np.pi/32
        vals=np.array([ev(t) for t in angles]); order=np.argsort(vals)
        best=(float(vals[order[0]]),float(angles[order[0]]))
        for idx in order[:2]:
            theta=angles[idx]
            opt=minimize_scalar(ev,bounds=(theta-np.pi/32,theta+np.pi/32),method='bounded',
                                options={'xatol':2e-8})
            if opt.fun<best[0]: best=(float(opt.fun),float(opt.x))
        return best

    def critical(self):
        start=time.process_time(); calls=0
        def indicator(p):
            nonlocal calls
            calls+=1
            C,neg=self.macro(p)
            if neg: return -self.EI
            return self.acoustic_min(C)[0]
        hi=min(1.1*self.lambdaE,self.pole*.95)
        while indicator(hi)>0:
            if hi>=self.pole*.94: raise RuntimeError('No onset below internal beam pole')
            hi=min(hi*1.5,self.pole*.95)
        root=brentq(indicator,0,hi,xtol=1e-11,rtol=1e-8)
        atroot,root_neg=self.macro(root)
        root_acoustic=None if atroot is None else self.acoustic_min(atroot)[0]
        C,neg=self.macro(root*(1-1e-6))
        if C is None: kind='Gamma_or_pole'; theta=None; C0=None
        else:
            value,theta=self.acoustic_min(C)
            kind=('acoustic' if root_acoustic is not None and abs(root_acoustic)<self.EI*1e-3
                  else 'near_Gamma_or_acoustic')
            C0=tensor_summary(C)
        before,_=self.macro(root*.99)
        result=dict(critical_load=float(root),normalized=float(root/self.lambdaE),kind=kind,theta=theta,
                    macro_just_below=C0,macro_at_99pct=tensor_summary(before) if before is not None else None,
                    root_acoustic_value=root_acoustic,root_negative_pivots=root_neg,
                    calls=calls,cpu_s=time.process_time()-start)
        if self.protocol=='stress' and C is not None:
            result['gibbs_rotation_residual']=tensor_summary(C+root*(1-1e-6)*VD)['rotation_row_norm']
        return result

    def bloch_threshold(self,k,upper=None):
        k=np.asarray(k,float); gamma=la.norm(k-np.round(k))<1e-12
        free=self.free if gamma else np.arange(self.ndof)
        calls=0
        def negative(p):
            nonlocal calls
            calls+=1
            M=self.assemble(self.element_matrices(p),k)[free][:,free].tocsc()
            return self.factor_negative(self.factor(M))>0
        hi=min(upper or self.lambdaE*1.5,self.pole*.95)
        while not negative(hi):
            if hi>=self.pole*.94: raise RuntimeError('No Bloch onset below internal pole')
            hi=min(1.5*hi,self.pole*.95)
        lo=0.
        for _ in range(32):
            mid=(lo+hi)/2
            if negative(mid): hi=mid
            else: lo=mid
        return dict(k=k.tolist(),load=(lo+hi)/2,normalized=(lo+hi)/2/self.lambdaE,
                    bracket_width=hi-lo,calls=calls)
