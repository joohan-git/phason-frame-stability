"""Prospective data pipeline; training first, locked test second."""
import os
for _k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[_k]='1'
import argparse,hashlib,json,time,datetime
from pathlib import Path
import numpy as np
import psutil
from ab_common import canonical,OFFSET
from geometry import Tiling,family_lifts
from penrose_core import Penrose,named,phase,threshold,richardson
from penrose_mc import mc as mc5
from law_ab_mc import mc as mc4
from law_features import LiftFeatures
from beam_check import geometry_check

ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    for attempt in range(30):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1*(attempt+1))
def check_prereg():
    lock=json.loads((ROOT/'law_prereg_lock.json').read_text());assert not (ROOT/'law_prereg.md').exists() or digest(ROOT/'law_prereg.md')==lock['sha256'];return lock
def rss():
    m=psutil.Process().memory_info()
    if hasattr(m,'peak_wset'):return int(m.peak_wset)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def base_for(kind):
    if kind=='train':return canonical(),OFFSET
    if kind=='ab_test':return canonical(17,24),OFFSET
    g=phase(90501);return Penrose(named('fib_6'),g),g
def settings(stage):
    if stage=='train':
        levels=[('accepted',k) for k in [1,10,100,1000]]+[('sweeps',s) for s in [20,200,1000,3000]]
        return [dict(id=f'train_uniform_{j}_{r}',kind='train',protocol=p,severity=v,rep=r,seed=510000+100*j+r,j=j) for j,(p,v) in enumerate(levels) for r in range(6)]+[
            dict(id=f'train_disc_{j}_{r}',kind='train',protocol='disc',severity=v,rep=r,seed=520000+100*j+r,j=j) for j,v in enumerate([3,5,7]) for r in range(4)]
    rows=[]
    for kind,seed0 in [('penrose_test',610000),('ab_test',710000)]:
        levels=[('accepted',k) for k in [1,30,300,3000]]+[('sweeps',s) for s in [200,3000]]
        rows += [dict(id=f'{kind}_uniform_{j}_{r}',kind=kind,protocol=p,severity=v,rep=r,seed=seed0+100*j+r,j=j) for j,(p,v) in enumerate(levels) for r in range(3)]
        rows += [dict(id=f'{kind}_disc_{j}_{r}',kind=kind,protocol='disc',severity=v,rep=r,seed=seed0+10000+100*j+r,j=j) for j,v in enumerate([4,7]) for r in range(3)]
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','test','validate']);ap.add_argument('--cpu-budget',type=float,default=11000);args=ap.parse_args()
    mark=time.process_time();wall=time.perf_counter();prereg=check_prereg()
    if args.stage=='test':
        lock=json.loads((ROOT/'law_model_lock.json').read_text());assert lock['prereg_sha256']==prereg['sha256']
        assert lock['training_sha256']==digest(ROOT/'law_train_raw.json')
        assert lock['feature_script_sha256']==digest(ROOT/'law_features.py')
    path=ROOT/f'law_{args.stage}_raw.json'
    out=json.loads(path.read_text()) if path.exists() else dict(source_script=Path(__file__).name,prereg_sha256=prereg['sha256'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=[],references={},invocations=[])
    if args.stage=='test':out.setdefault('model_lock_sha256',digest(ROOT/'law_model_lock.json'))
    done={r['id'] for r in out['rows']};cache={};snap=ROOT/'law_snapshots';snap.mkdir(exist_ok=True)
    def get(kind):
        if kind not in cache:
            base,g=base_for(kind);features=LiftFeatures(base,g);cache[kind]=(base,features,g)
            if kind not in out['references']:
                out['references'][kind]=dict(N=len(base.n),L=base.L.tolist(),gamma=np.asarray(g).tolist(),features=features.evaluate(base),mechanics=threshold(base,sub=4),geometry=geometry_check(base.x,base.edges(),base.cell))
                save(path,out)
        return cache[kind]
    if args.stage=='train':
        old=json.loads((ROOT/'ab_window_raw.json').read_text())
        for source in old['rows'][:24]:
            assert source['protocol']=='global';key=f"existing_{source['seed']}"
            if key in done:continue
            begin=time.process_time();base,feat,g=get('train');t=base.copy()
            move=t.mc(np.random.default_rng(source['seed']),accepted=int(source['severity']))
            f=feat.evaluate(t)
            assert abs(f['variance']-source['internal_variance'])<1e-9
            assert abs(f['mean_excess']-source['euclidean_mean_excess'])<1e-9
            pc=source['mechanics']['minimum_sampled'];p0=out['references']['train']['mechanics']['pc']
            row=dict(id=key,kind='train',protocol='existing_accepted',severity=source['severity'],rep=source['replicate'],seed=source['seed'],fold=source['replicate']%5,
                     features=f,mechanics=dict(pc=pc,source='ab_window_raw.json; ab_window.py'),y=pc/p0,mc=move,cpu_s=time.process_time()-begin)
            out['rows'].append(row);save(path,out);print(key,pc,flush=True)
    if args.stage in ['train','test']:
        for spec in settings(args.stage):
            if spec['id'] in done:continue
            if time.process_time()-mark>args.cpu_budget:break
            begin=time.process_time();base,feat,g=get(spec['kind']);t=base.copy();run=mc4 if t.n.shape[1]==4 else mc5
            p=spec['protocol'];v=spec['severity']
            move=run(t,spec['seed'],accepted=int(v)) if p=='accepted' else run(t,spec['seed'],attempts=int((v if p=='sweeps' else 3000)*len(t.n)),radius=v if p=='disc' else -1)
            f=feat.evaluate(t);geom=geometry_check(t.x,t.edges(),t.cell);mech=threshold(t,sub=4)
            p0=out['references'][spec['kind']]['mechanics']['pc']
            row=dict(**spec,fold=spec['rep']%5,features=f,mechanics=mech,y=mech['pc']/p0,mc=move,geometry=geom,N=len(t.n),cpu_s=time.process_time()-begin,peak_rss_bytes=rss())
            np.savez_compressed(snap/(spec['id']+'.npz'),L=t.L,n=t.n,gamma=g)
            out['rows'].append(row);save(path,out)
            print(spec['id'],'pc',mech['pc'],'y',row['y'],'cpu',row['cpu_s'],flush=True)
    else:
        for key in ['train_uniform_0_0','train_uniform_7_0','train_disc_2_0']:
            if key in done:continue
            z=np.load(snap/(key+'.npz'));t=Tiling(z['L'],n=z['n'])
            row=dict(id=key,sub2=threshold(t,sub=2),richardson_sub4=richardson(t,sub=4,angles=4))
            out['rows'].append(row);save(path,out);print('validation',key,flush=True)
        for r in range(3):
            key=f'mixing_{r}'
            if key in done:continue
            z=np.load(snap/f'train_uniform_7_{r}.npz');t=Tiling(z['L'],n=z['n']);base,feat,g=get('train')
            move=mc4(t,530000+r,attempts=1500*len(t.n))
            out['rows'].append(dict(id=key,mc=move,features=feat.evaluate(t),mechanics=threshold(t,sub=4)));save(path,out)
    out['invocations'].append(dict(cpu_s=time.process_time()-mark,wall_s=time.perf_counter()-wall,peak_rss_bytes=rss()))
    save(path,out)

if __name__=='__main__':main()
