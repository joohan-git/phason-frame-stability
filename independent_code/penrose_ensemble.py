"""Independent Penrose drift arrays, restartable on fixed attempted-MC blocks.

No submission from this script. One thread, one seed per array task. The
finite Bloch sample is an estimate, never a certified full-zone minimum.
"""
import os
for _k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_k]='1'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import psutil
from penrose_core import Penrose,named,phase,threshold,components
from penrose_mc import mc

ROOT=Path(__file__).resolve().parent
def peak():
    m=psutil.Process().memory_info()
    if hasattr(m,'peak_wset'):return int(m.peak_wset)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def atomic(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for k in range(30):
        try:tmp.replace(path);return
        except PermissionError:
            if k==29:raise
            time.sleep(.1*(k+1))
def heights(t):
    U=np.linalg.svd(t.L.astype(float),full_matrices=True)[2][2:]
    h=t.n@U.T;h-=h.mean(0);frac=t.x@t.invcell
    modes=[]
    for wave in [(1,0),(0,1),(1,1)]:
        z=np.mean(h*np.exp(-2j*np.pi*(frac@wave))[:,None],axis=0)
        modes.append(dict(k=list(wave),power=float(np.sum(abs(z)**2))))
    return dict(internal_variance=float(np.mean(np.sum(h*h,axis=1))),low_modes=modes)
def make_plan(args):
    assert 8<=args.seeds<=32
    rows=[]
    sizes=[('fib_4',644),('fib_5',1686),('fib_6',4414),('fib_7',11556),('fib_8',30254)]
    if args.supercells:sizes += [('fib_4',644),('fib_5',1686),('fib_6',4414)]
    for i,(name,N) in enumerate(sizes):
        m=1 if i<5 else 2
        for seed in range(args.seeds+1):
            rows.append(dict(name=name,supercell=m,N=N*m*m,mode='canonical' if seed==0 else 'random',seed=seed,
                             phase_seed=91501,tag=f'{name}_sc{m}_seed{seed:02d}'))
    atomic(args.plan,dict(source_script=Path(__file__).name,classification='P',tasks=rows,seeds_per_size=args.seeds))
    print('Wrote',len(rows),'tasks to',args.plan)
def run(args):
    plan=json.loads(Path(args.plan).read_text());task=plan['tasks'][args.task_id]
    out=Path(args.outdir)/task['tag'];out.mkdir(parents=True,exist_ok=True)
    path=out/'result.json';checkpoint=out/'checkpoint.npz';start=time.process_time();wall=time.perf_counter()
    signature={k:vars(args)[k] for k in ['burn','spacing','samples','block','sub']}
    if checkpoint.exists():
        z=np.load(checkpoint);result=json.loads(str(z['result_json']))
        assert result['task']==task and result['settings']==signature
        t=Penrose(z['L'],z['gamma'],z['n']);current=int(z['sweeps']);block_id=int(z['block_id'])
        z.close()  # Atomic replacement fails while the old NPZ is open.
        assert current==result['completed_sweeps']
    else:
        t=Penrose(task['supercell']*named(task['name']),phase(task['phase_seed']));current=0;block_id=0
        assert len(t.n)==task['N']
        result=dict(source_script=Path(__file__).name,classification='P',task=task,settings=signature,N=len(t.n),period_lifts=t.L.tolist(),
                    phason_components=components(t.L),samples=[],diagnostics=[],invocations=[],completed_sweeps=0,
                    equilibration_certified=False,geometry=t.validate(),
                    interpretation='Acoustic/Gamma/three boundary Bloch points sampled; no full-zone or infinite-size certificate.')
    def checkpoint_now():
        result['completed_sweeps']=current
        # Store a complete matched state first; result is reconstructible from checkpoint metadata on a job interruption.
        tmp=out/'checkpoint.tmp.npz'
        np.savez_compressed(tmp,L=t.L,gamma=t.gamma,n=t.n,sweeps=current,block_id=block_id,result_json=json.dumps(result))
        for retry in range(30):
            try:tmp.replace(checkpoint);break
            except PermissionError:
                if retry==29:raise
                time.sleep(.1*(retry+1))
        atomic(path,result)
    def measure():
        if any(r['sweeps']==current for r in result['samples']):return
        before=time.process_time();mech=threshold(t,sub=args.sub)
        result['samples'].append(dict(sweeps=current,mechanics=mech,heights=heights(t),cpu_s=time.process_time()-before,peak_rss_bytes=peak()))
        print(task['tag'],'sample',current,mech['pc'],flush=True)
    if task['mode']=='canonical':
        measure();result['complete']=True;checkpoint_now()
    else:
        total=args.burn+(args.samples-1)*args.spacing
        while current<total:
            target=args.burn if current<args.burn else args.burn+(1+(current-args.burn)//args.spacing)*args.spacing
            block=min(args.block,target-current,total-current)
            # Re-seeding at deterministic block boundaries is restart-invariant.
            seed=(810000+100003*task['seed']+7919*block_id+int(task['N']))%(2**31-1)
            move=mc(t,seed,attempts=block*len(t.n));current+=block;block_id+=1
            result['diagnostics'].append(dict(sweeps=current,mc=move,**heights(t)))
            if current>=args.burn and (current-args.burn)%args.spacing==0:measure()
            checkpoint_now()
            if time.process_time()-start>args.cpu_budget:break
        result['complete']=current>=total
        if result['complete']:result['final_geometry']=t.validate()
    result['invocations'].append(dict(cpu_s=time.process_time()-start,wall_s=time.perf_counter()-wall,peak_rss_bytes=peak()))
    checkpoint_now();print('complete',result['complete'],'cpu',result['invocations'][-1]['cpu_s'],flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','run']);ap.add_argument('--plan',default=str(ROOT/'penrose_ensemble_plan.json'))
    ap.add_argument('--seeds',type=int,default=16);ap.add_argument('--supercells',action='store_true');ap.add_argument('--task-id',type=int,default=0)
    ap.add_argument('--outdir',default=str(ROOT/'ensemble_out'));ap.add_argument('--burn',type=int,default=3000);ap.add_argument('--spacing',type=int,default=1500)
    ap.add_argument('--samples',type=int,default=2);ap.add_argument('--block',type=int,default=100);ap.add_argument('--sub',type=int,default=4)
    ap.add_argument('--cpu-budget',type=float,default=25000);args=ap.parse_args()
    if args.mode=='plan':make_plan(args)
    else:run(args)
if __name__=='__main__':main()
