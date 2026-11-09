"""Resumable runner for the frozen manifest; one CPU thread."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse,itertools,json,time,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import brentq
from scipy.linalg import eigvalsh
from geometry import Tiling,slope,slope_components
from frame import Frame
from penrose_core import Penrose,phase
from beam_check import Frame as FullFrame,geometry_check
from law_ab_mc import mc as mc4
from penrose_mc import mc as mc5
from patch_atlas import census

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    # The destination can be briefly locked by another process.
    for attempt in range(40):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.1+min(attempt*.05,.5))

class Metrics:
    def __init__(self,base,gamma):
        self.ref=set(map(tuple,base.n));self.U=np.linalg.svd(base.L.astype(float),full_matrices=True)[2][2:]
        self.center=(np.asarray(gamma)+.5)@self.U.T;self.rank=base.n.shape[1]
        cube=np.array(list(itertools.product([-.5,.5],repeat=self.rank)))@self.U.T
        hull=ConvexHull(cube);self.equations=hull.equations
        self.boundary=cube[hull.vertices] if self.rank==4 else cube[hull.simplices]
        assert self.evaluate(base)['max_excess']<1e-8
    def evaluate(self,t):
        y=t.n@self.U.T-self.center
        outside=np.max(y@self.equations[:,:-1].T+self.equations[:,-1],axis=1)>1e-9
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
        dist=np.where(outside,dist,0.)
        return dict(internal_variance=float(np.mean(np.sum((y-y.mean(0))**2,axis=1))),
                    max_excess=float(dist.max()),mean_excess=float(dist.mean()),
                    moved_fraction=float(np.mean([tuple(v) not in self.ref for v in t.n])))

def make_base(spec):
    g=(np.random.default_rng(spec['phase_seed']).uniform(-.5,.5,4)
       if spec['symmetry']=='AB' else phase(spec['phase_seed']))
    t=Tiling(spec['L'],offset=g) if spec['symmetry']=='AB' else Penrose(spec['L'],g)
    return t,g

def bloch_points(grid=4):
    seen=set();out=[]
    for i,j in itertools.product(range(grid),repeat=2):
        key=min((i,j),((-i)%grid,(-j)%grid))
        if key not in seen:seen.add(key);out.append([i/grid,j/grid])
    return out

def mechanics(t,sub=2,angles=8,bloch=True,grid=4):
    start=time.process_time();f=Frame(t,sub=sub);crit=f.critical();vals=[]
    for th in np.arange(angles)*np.pi/angles:
        c,s=np.cos(th),np.sin(th);A=np.array([[c,0],[s,0],[0,c],[0,s]])
        def indicator(load):
            C,neg=f.macro(load)
            return -f.EI if C is None else eigvalsh(A.T@C@A,check_finite=False)[0]
        hi=min(1.2*f.lambdaE,.94*f.pole)
        vals.append(float(brentq(indicator,0,hi,xtol=1e-13,rtol=1e-10)/f.lambdaE))
    finite=[]
    if bloch:
        full=FullFrame(t.x,t.edges(),t.cell,div=sub,EI=f.EI)
        for k in bloch_points(grid):
            p,res=full.eigen(k);finite.append(dict(k=k,pc=p/f.lambdaE,residual=res))
    pc=min([crit['normalized']]+[v['pc'] for v in finite])
    return dict(pc=pc,acoustic_or_Gamma=crit['normalized'],kind=crit['kind'],theta=crit['theta'],
                c0=float(np.mean(vals)),directional=vals,finite_k=finite,sub=sub,pre=f.pre,
                cpu_s=time.process_time()-start,
                caveat='pc is minimum on sampled BZ plus exact acoustic limit; not a certified global BZ minimum.')

def run_one(spec,atlas):
    start=time.process_time();t,g=make_base(spec);metrics=Metrics(t,g)
    initial=census(t,atlas);initial.pop('histogram')
    move=mc4 if spec['symmetry']=='AB' else mc5;traces=[];moves=[]
    family=spec['family']
    if family=='few_flip':moves.append(move(t,spec['mc_seed'],accepted=spec['flips']))
    elif family=='disc_cluster':
        moves.append(move(t,spec['mc_seed'],attempts=15000*len(t.n),radius=spec['disc_radius']))
    elif family=='equilibrated':
        burn=spec.get('burn_sweeps',5000);step=spec.get('block_sweeps',500)
        moves.append(move(t,spec['mc_seed'],attempts=burn*len(t.n)))
        for block in range(spec.get('blocks',20)):
            moves.append(move(t,spec['mc_seed']+block+1,attempts=step*len(t.n)))
            trace=metrics.evaluate(t); trace['phi2']=census(t,atlas)['phi2'];trace['sweeps']=burn+step*(block+1)
            traces.append(trace)
    geom=geometry_check(t.x,t.edges(),t.cell);patch=census(t,atlas)
    feat=metrics.evaluate(t)
    snapshots=ROOT/'snapshots';snapshots.mkdir(exist_ok=True)
    np.savez_compressed(snapshots/(spec['id']+'.npz'),L=t.L,n=t.n,gamma=g)
    mech=mechanics(t,bloch=family!='F4_Pell')
    ref=.7998 if spec['symmetry']=='AB' else .8106
    result=dict(**spec,label='P',source='run_cells.py; snapshots/'+spec['id']+'.npz',gamma=g.tolist(),
                N=len(t.n),geometry=geom,initial_census=initial,census=patch,features=feat,
                mechanics=mech,Delta=1-mech['pc']/ref,Delta_iso=1-mech['c0']/ref,
                moves=moves,equilibration_trace=traces,cpu_s=time.process_time()-start)
    if family=='F4_Pell':result['delta']=float(np.linalg.norm(slope_components(slope(t.L))))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['test','f4'],default='test')
    ap.add_argument('--index',type=int);ap.add_argument('--limit',type=int)
    ap.add_argument('--cpu-budget',type=float,default=11000);args=ap.parse_args()
    manifest=json.loads((ROOT/'cell_manifest.json').read_text());atlas=json.loads((ROOT/'patch_atlas.json').read_text())
    if (ROOT/'registered_plan.md').exists():
        assert sha(ROOT/'registered_plan.md')==manifest['prereg_sha256']
    for item in manifest['copied_sources']:assert sha(ROOT.parent/'independent_code'/item['file'])==item['sha256']
    specs=manifest['rows' if args.stage=='test' else 'F4']
    if args.index is not None:specs=[specs[args.index]]
    path=ROOT/'raw';path.mkdir(exist_ok=True);start=time.process_time();completed=0
    for spec in specs:
        target=path/(spec['id']+'.json')
        if target.exists():continue
        if time.process_time()-start>args.cpu_budget or (args.limit is not None and completed>=args.limit):break
        row=run_one(spec,set(atlas['symmetries'][spec['symmetry']]['patches']))
        row['manifest_sha256']=sha(ROOT/'cell_manifest.json');row['atlas_sha256']=sha(ROOT/'patch_atlas.json')
        save(target,row);completed+=1
        print(spec['id'],'N',row['N'],'phi2',row['census']['phi2'],'pc',row['mechanics']['pc'],
              'c0',row['mechanics']['c0'],'CPU',round(row['cpu_s'],2),'total',round(time.process_time()-start,2),flush=True)
    save(ROOT/f'invocation_{args.stage}_{args.index}_{time.time_ns()}.json',
         dict(cpu_s=time.process_time()-start,completed=completed,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),args=vars(args)))
if __name__=='__main__':main()
