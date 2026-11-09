"""Independent reversible, uniform-vertex attempted flips for rank-five tilings.

This is a Numba translation of our previously checked AB legal-move test,
generalised to ten directed edges. No other solver's MC is imported.
"""
from penrose_core import *
from numba import njit,types
from numba.typed import Dict

@njit(cache=True)
def key5(a):return (a[0],a[1],a[2],a[3],a[4])
@njit(cache=True)
def reduce5(raw,L,rm):
    s0=0.;s1=0.
    for a in range(5):s0+=raw[a]*rm[a,0];s1+=raw[a]*rm[a,1]
    i=int(np.floor(s0+.5+1e-10));j=int(np.floor(s1+.5+1e-10))
    return raw-i*L[0]-j*L[1]
@njit(cache=True)
def in_region(raw,rm,cell,radius):
    s0=0.;s1=0.
    for a in range(5):s0+=raw[a]*rm[a,0];s1+=raw[a]*rm[a,1]
    s0-=np.round(s0);s1-=np.round(s1)
    x=s0*cell[0,0]+s1*cell[1,0];y=s0*cell[0,1]+s1*cell[1,1]
    return x*x+y*y<=radius*radius

@njit(cache=True)
def run_mc(n,adj,lookup,L,rm,cell,steps,physical,attempts,accepted_target,seed,radius):
    np.random.seed(seed);accepted=0;tried=0
    for attempt in range(attempts):
        tried+=1;i=np.random.randint(len(n));codes=np.where(adj[i]>=0)[0]
        if len(codes)!=3:continue
        old=n[i].copy();raw=old.copy();ang=np.zeros(3)
        for h in range(3):
            code=codes[h];raw+=steps[code]
            dx=0.;dy=0.
            for a in range(5):dx+=steps[code,a]*physical[0,a];dy+=steps[code,a]*physical[1,a]
            ang[h]=np.arctan2(dy,dx)
        ang.sort()
        if max(ang[1]-ang[0],ang[2]-ang[1],ang[0]+2*np.pi-ang[2])>=np.pi-1e-9:continue
        if radius>0 and (not in_region(old,rm,cell,radius) or not in_region(raw,rm,cell,radius)):continue
        new=reduce5(raw,L,rm)
        if key5(new) in lookup:continue
        opposite=np.full(3,-1,np.int64)
        for h in range(3):
            op=reduce5(raw-steps[codes[h]],L,rm);opkey=key5(op)
            if opkey in lookup:opposite[h]=lookup[opkey]
        if np.min(opposite)<0:continue
        good=True;actual=0
        for code in range(10):
            nb=reduce5(raw+steps[code],L,rm);nbkey=key5(nb)
            if nbkey in lookup:
                j=lookup[nbkey]
                if j==i:continue
                actual+=1
                if j!=opposite[0] and j!=opposite[1] and j!=opposite[2]:good=False
        if not good or actual!=3:continue
        for h in range(3):
            code=codes[h];j=adj[i,code];adj[j,(code+5)%10]=-1;adj[i,code]=-1
        del lookup[key5(old)];lookup[key5(new)]=i;n[i]=new
        for h in range(3):
            code=codes[h];j=opposite[h];adj[i,(code+5)%10]=j;adj[j,code]=i
        accepted+=1
        if accepted_target>0 and accepted>=accepted_target:break
    return tried,accepted

def mc(t,seed,attempts=None,accepted=None,radius=-1):
    start=time.process_time()
    lookup=Dict.empty(key_type=types.UniTuple(types.int64,5),value_type=types.int64)
    for i,n in enumerate(t.n):lookup[tuple(n)]=i
    limit=int(attempts if attempts is not None else max(10000,accepted*20000))
    tried,got=run_mc(t.n,t.adj,lookup,t.L,t.reduce_map,t.cell,SIGNED,P,limit,accepted or -1,seed,radius)
    t.lookup={tuple(n):i for i,n in enumerate(t.n)}
    if accepted is not None:assert got==accepted,(tried,got,accepted)
    return dict(attempted=tried,accepted=got,cpu_s=time.process_time()-start,seed=seed,radius=radius)

if __name__=='__main__':
    t=Penrose(named('fib_6'));w=Window(t)
    print('before',t.validate(),flush=True)
    print('mc',mc(t,44001,attempts=100*len(t.n)),flush=True)
    print('after',t.validate(),w.stats(t),flush=True)
    print('mechanics',threshold(t),flush=True)
