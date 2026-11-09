"""Independent rooted, direction-labelled radius-two census (no imports from the main code).

A signature is the sorted set of integer lift offsets reached in <=2 steps.
The induced edges are precisely pairs differing by a signed unit vector, so
this representation retains all edge directions and the entire induced graph.
Periodic BFS carries lift offsets, never identifies different torus images.
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'
import itertools, json, time, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from geometry import P as P4
from penrose_core import P as P5, phase

ROOT=Path(__file__).resolve().parent

def signature(offsets):
    return np.asarray(sorted(offsets),dtype=np.int8).tobytes().hex()

def adjacency(t):
    d=t.n.shape[1]
    if isinstance(t.adj,np.ndarray):
        return [[(int(j),c) for c,j in enumerate(row) if j>=0] for row in t.adj]
    return [list(row.items()) for row in t.adj]

def periodic_signatures(t,radius=2):
    d=t.n.shape[1]; steps=np.r_[np.eye(d,dtype=int),-np.eye(d,dtype=int)]
    adj=adjacency(t); out=[]
    zero=(0,)*d
    for root in range(len(t.n)):
        seen={zero}; front=[(root,zero)]
        for _ in range(radius):
            nxt=[]
            for i,off in front:
                for j,c in adj[i]:
                    v=tuple(np.asarray(off)+steps[c])
                    if v not in seen: seen.add(v); nxt.append((j,v))
            front=nxt
        out.append(signature(seen))
    return out

def canonical_patch(symmetry,radius,seed):
    """Finite irrational multigrid dual, generated directly in physical space."""
    P=P4 if symmetry=='AB' else P5; d=P.shape[1]; scale=d/2
    gamma=(np.random.default_rng(seed).uniform(-.5,.5,d) if d==4 else phase(seed))
    # A dual vertex is scale*y plus a bounded cube projection; padding 2*d
    # guarantees every vertex within the requested physical radius is present.
    extent=(radius+2*d)/scale
    lim=int(np.ceil(extent+2)); vals=np.arange(-lim,lim+1)
    grid=np.array(list(itertools.product(vals,repeat=2)))
    eye=np.eye(d,dtype=int); vertices=set()
    for i,j in itertools.combinations(range(d),2):
        y=(grid-gamma[[i,j]])@np.linalg.inv(P[:,[i,j]])
        mask=np.linalg.norm(y,axis=1)<=extent; yy=y[mask]; gg=grid[mask]
        raw=np.ceil(yy@P+gamma-1e-9).astype(int)
        raw[:,i]=gg[:,0]; raw[:,j]=gg[:,1]
        for a,b in itertools.product([0,1],repeat=2):
            vertices.update(map(tuple,raw+a*eye[i]+b*eye[j]))
    n=np.array(sorted(vertices),dtype=int)
    return n,gamma

def finite_census(n,P,radius):
    d=n.shape[1]; steps=np.r_[np.eye(d,dtype=int),-np.eye(d,dtype=int)]
    lookup={tuple(v):i for i,v in enumerate(n)}
    adj=[[] for _ in n]
    for i,v in enumerate(n):
        for c,step in enumerate(steps):
            j=lookup.get(tuple(v+step))
            if j is not None: adj[i].append((j,c))
    centers=np.flatnonzero(np.linalg.norm(n@P.T,axis=1)<=radius-3)
    counts=Counter(); zero=(0,)*d
    for root in centers:
        seen={zero}; front=[(root,zero)]
        for _ in range(2):
            nxt=[]
            for i,off in front:
                assert len(adj[i])>=3, 'Atlas boundary reached'
                for j,c in adj[i]:
                    v=tuple(np.asarray(off)+steps[c])
                    if v not in seen: seen.add(v); nxt.append((j,v))
            front=nxt
        counts[signature(seen)]+=1
    return counts,len(centers)

def census(t,atlas):
    sigs=periodic_signatures(t); counts=Counter(sigs)
    forbidden=sum(v for k,v in counts.items() if k not in atlas)
    return dict(phi2=forbidden/len(sigs),forbidden_vertices=forbidden,
                vertices=len(sigs),distinct_patches=len(counts),
                forbidden_types=sum(k not in atlas for k in counts),
                histogram=dict(sorted(counts.items())))

def build():
    started=time.process_time(); out=dict(label='P',definition=__doc__,symmetries={})
    for sym in ('AB','Penrose'):
        known=set(); stages=[]; atlas_counts={}
        for radius,seed in [(35,861101),(60,861102),(90,861103),(110,861104)]:
            n,g=canonical_patch(sym,radius,seed)
            counts,num=finite_census(n,P4 if sym=='AB' else P5,radius)
            stages.append(dict(radius=radius,seed=seed,interior_vertices=num,
                               types=len(counts),new_types=len(set(counts)-known),
                               missing_previous=len(known-set(counts))))
            known.update(counts);atlas_counts.update(counts)
            print(sym,stages[-1],flush=True)
        out['symmetries'][sym]=dict(stages=stages,patches=sorted(known),
                                    stabilized=all(s['new_types']==0 for s in stages[-2:]),
                                    type_count=len(known))
    out['cpu_s']=time.process_time()-started
    (ROOT/'patch_atlas.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out

if __name__=='__main__': build()
