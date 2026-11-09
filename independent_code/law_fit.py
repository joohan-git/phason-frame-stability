"""Exactly the frozen fitting and evaluation rules. No exploratory models."""
import os
for _k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[_k]='1'
import argparse,json,hashlib,datetime,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from law_features import FEATURES

ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,d): (ROOT/name).write_text(json.dumps(d,indent=2),encoding='utf-8')
def fit(x,y,alpha):
    z=x**alpha
    def error(beta):return float(np.mean((np.exp(-beta*z)-y)**2))
    r=minimize_scalar(error,bounds=(0,100),method='bounded',options={'xatol':1e-11})
    return min([0.,float(r.x),100.],key=error)
def lock_model():
    assert not (ROOT/'law_model_lock.json').exists(),'Model already frozen; do not overwrite'
    pr=read('law_prereg_lock.json');assert not (ROOT/'law_prereg.md').exists() or digest(ROOT/'law_prereg.md')==pr['sha256']
    d=read('law_train_raw.json');rows=d['rows'];assert len(rows)==84
    assert len({r['id'] for r in rows})==84
    y=np.array([r['y'] for r in rows]);folds=np.array([r['fold'] for r in rows]);scores=[]
    for name in FEATURES:
        x=np.array([r['features']['x'][name] for r in rows])
        for alpha in [.5,1.,2.]:
            pred=np.zeros(len(rows));bb=[]
            for fold in range(5):
                keep=folds!=fold;beta=fit(x[keep],y[keep],alpha);bb.append(beta)
                pred[~keep]=np.exp(-beta*x[~keep]**alpha)
            scores.append(dict(feature=name,alpha=alpha,cv_mse=float(np.mean((pred-y)**2)),fold_betas=bb))
    best=scores[0]
    for score in scores[1:]:
        if score['cv_mse']<best['cv_mse']-1e-12:best=score
    x=np.array([r['features']['x'][best['feature']] for r in rows]);beta=fit(x,y,best['alpha'])
    lock=dict(source_script=Path(__file__).name,locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              prereg_sha256=pr['sha256'],training_sha256=digest(ROOT/'law_train_raw.json'),feature_script_sha256=digest(ROOT/'law_features.py'),
              fit_script_sha256=digest(__file__),model='exp(-beta * x**alpha)',feature=best['feature'],alpha=best['alpha'],beta=beta,
              cv_candidates=scores,training_ids=[r['id'] for r in rows],training_n=len(rows),test_disorder_seen=False,
              calibration='Only unflipped reference pressure at each exact test period/phase; no disorder refit.')
    write('law_model_lock.json',lock);print(json.dumps({k:lock[k] for k in ['feature','alpha','beta','locked_utc']},indent=2))
def metrics(rows):
    y=np.array([r['y'] for r in rows]);pred=np.array([r['predicted_y'] for r in rows]);x=np.array([r['x'] for r in rows])
    sst=float(np.sum((y-y.mean())**2));sse=float(np.sum((pred-y)**2))
    return dict(n=len(rows),r2=1-sse/sst if sst>0 else None,mae=float(np.mean(abs(pred-y))),rmse=float(np.sqrt(sse/len(y))),
                spearman=float(spearmanr(x,y).statistic),actual_range=[float(y.min()),float(y.max())])
def evaluate():
    start=time.process_time();lock=read('law_model_lock.json');d=read('law_test_raw.json')
    assert not (ROOT/'law_prereg.md').exists() or digest(ROOT/'law_prereg.md')==lock['prereg_sha256']
    assert digest(ROOT/'law_train_raw.json')==lock['training_sha256']
    assert digest(ROOT/'law_features.py')==lock['feature_script_sha256']
    assert digest(__file__)==lock['fit_script_sha256']
    rows=[]
    for r in d['rows']:
        x=r['features']['x'][lock['feature']]
        rows.append(dict(id=r['id'],kind=r['kind'],protocol=r['protocol'],severity=r['severity'],rep=r['rep'],x=x,y=r['y'],predicted_y=float(np.exp(-lock['beta']*x**lock['alpha']))))
    assert len(rows)==48 and len({r['id'] for r in rows})==48
    groups={'pooled':rows,**{k:[r for r in rows if r['kind']==k] for k in ['ab_test','penrose_test']}}
    out=dict(source_script=Path(__file__).name,model_lock_sha256=digest(ROOT/'law_model_lock.json'),metrics={k:metrics(v) for k,v in groups.items()},predictions=rows)
    out['per_protocol']={k:metrics([r for r in rows if r['kind']+'_'+r['protocol']==k]) for k in sorted({r['kind']+'_'+r['protocol'] for r in rows})}
    m=out['metrics'];out['gate_checks']=dict(pooled_r2=m['pooled']['r2']>=.8,ab_r2=m['ab_test']['r2']>=.7,penrose_r2=m['penrose_test']['r2']>=.7,pooled_mae=m['pooled']['mae']<=.08)
    out['primary_pass']=all(out['gate_checks'].values());out['secondary_spearman_pass']=m['pooled']['spearman']<=-.9
    rng=np.random.default_rng(80501);boot={k:[] for k in groups}
    for _ in range(2000):
        sample=[]
        for kind in ['ab_test','penrose_test']:
            rr=[]
            for rep in rng.integers(0,3,size=3):rr += [r for r in groups[kind] if r['rep']==rep]
            sample+=rr;boot[kind].append(metrics(rr))
        boot['pooled'].append(metrics(sample))
    out['bootstrap_ci95']={k:{metric:np.nanpercentile([r[metric] for r in bb],[2.5,97.5]).tolist() for metric in ['r2','mae','spearman']} for k,bb in boot.items()}
    out['bootstrap_note']='Whole replicate blocks, independently within each class; three blocks per class limit interval resolution.'
    out['cpu_s']=time.process_time()-start;write('law_evaluation.json',out)
    print(json.dumps({k:out[k] for k in ['primary_pass','secondary_spearman_pass','metrics','gate_checks']},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['lock','evaluate']);args=ap.parse_args()
    lock_model() if args.mode=='lock' else evaluate()
