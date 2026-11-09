"""Assemble labelled evidence and standalone plots from completed outputs."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,zipfile,datetime,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(obj,source):
    if isinstance(obj,bool) or obj is None:return obj
    if isinstance(obj,(int,float,np.integer,np.floating)):
        assert np.isfinite(obj)
        return dict(value=float(obj) if isinstance(obj,(float,np.floating)) else int(obj),classification='P',source_script=source)
    if isinstance(obj,dict):return {k:label(v,source) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [label(v,source) for v in obj]
    return obj
def main():
    names=['law_prereg_lock.json','law_model_lock.json','law_train_raw.json','law_test_raw.json','law_evaluation.json','law_validate_raw.json','law_smoke.json','penrose_ensemble_benchmark.json','law_reuse_pressure_check.json']
    d={name:read(name) for name in names};lock=d['law_model_lock.json'];train=d['law_train_raw.json'];test=d['law_test_raw.json'];ev=d['law_evaluation.json'];bench=d['penrose_ensemble_benchmark.json']
    assert not (ROOT/'law_prereg.md').exists() or sha(ROOT/'law_prereg.md')==lock['prereg_sha256']==d['law_prereg_lock.json']['sha256']
    assert sha(ROOT/'law_train_raw.json')==lock['training_sha256']
    assert sha(ROOT/'law_model_lock.json')==test['model_lock_sha256']==ev['model_lock_sha256']
    assert datetime.datetime.fromisoformat(lock['locked_utc'])<datetime.datetime.fromisoformat(test['started_utc'])
    assert len(train['rows'])==84 and len(test['rows'])==48
    # Timings for a fresh worker process, including its measured peak memory.
    timing=[]
    for name in ['straight','medium','large']:
        r=next(v['result'] for v in bench['runs'] if v['name']==name)
        timing.append(dict(name=name,N=r['N'],mech_cpu_s=float(np.mean([s['mechanics']['cpu_s'] for s in r['samples']])),
                           mc_cpu_s=sum(v['mc']['cpu_s'] for v in r['diagnostics']),attempts=sum(v['mc']['attempted'] for v in r['diagnostics']),
                           process_cpu_s=sum(v['cpu_s'] for v in r['invocations']),peak_rss_MiB=max(v['peak_rss_bytes'] for v in r['invocations'])/2**20))
    pen=[r for r in test['rows'] if r['kind']=='penrose_test']
    strong=[r for r in pen if r['protocol']=='sweeps' and r['severity']==3000]
    timing.append(dict(name='heldout_4414',N=4414,mech_cpu_s=float(np.mean([r['mechanics']['cpu_s'] for r in strong])),
                       mc_cpu_s=sum(r['mc']['cpu_s'] for r in strong),attempts=sum(r['mc']['attempted'] for r in strong),
                       process_cpu_s=float(np.mean([r['cpu_s'] for r in strong])),peak_rss_MiB=max(r['peak_rss_bytes'] for r in strong)/2**20))
    timing.sort(key=lambda r:r['N'])
    logN=np.log([r['N'] for r in timing]);logT=np.log([r['mech_cpu_s'] for r in timing]);power,intercept=np.polyfit(logN,logT,1)
    memory_base=bench['runs'][0]['result']['invocations'][0]['peak_rss_bytes']/2**20
    memory_increment=np.maximum([r['peak_rss_MiB']-memory_base for r in timing],1.)
    memory_power,memory_intercept=np.polyfit(logN,np.log(memory_increment),1)
    # MC measured on long held-out runs avoids short-run JIT overhead.
    rate=sum(r['mc']['cpu_s'] for r in strong)/sum(r['mc']['attempted'] for r in strong)
    estimates=[]
    for N in [644,1686,4414,11556,30254]:
        mech=float(np.exp(intercept)*N**power);seconds=4500*N*rate+2*mech
        estimates.append(dict(N=N,mechanics_per_measurement_cpu_s=mech,random_task_cpu_h=seconds/3600,random_task_planning_range_cpu_h=[seconds/3600,3*seconds/3600],
                              sixteen_seeds_plus_reference_core_h=(16*seconds+mech)/3600,
                              estimated_peak_RSS_MiB=float(memory_base+np.exp(memory_intercept)*N**memory_power),
                              requested_memory_GiB=4 if N<=4414 else 16))
    estimate=dict(source_script=Path(__file__).name,measurements=timing,mechanics_size_exponent=float(power),mc_seconds_per_attempt=rate,per_size=estimates,
                  memory_baseline_MiB=memory_base,memory_increment_size_exponent=float(memory_power),
                  total_core_h=sum(r['sixteen_seeds_plus_reference_core_h'] for r in estimates),
                  note='P extrapolation, one core per task, two measurements at 3000/4500 sweeps; excludes scheduler wait and storage. Planning upper range is three times fitted time, not a statistical bound. Memory requests are provisioning, not measurements.')
    (ROOT/'law_resource_estimate.json').write_text(json.dumps(estimate,indent=2),encoding='utf-8')
    cpu=sum(v['cpu_s'] for name in ['law_train_raw.json','law_test_raw.json','law_validate_raw.json'] for v in d[name]['invocations'])+bench['cpu_s']+d['law_smoke.json']['cpu_s']+ev['cpu_s']+d['law_reuse_pressure_check.json']['cpu_s']
    cpu_summary=dict(recorded_cpu_s=cpu,overhead_allowance_s=300,total_with_allowance_cpu_h=(cpu+300)/3600,
                     note='Sum of recorded process CPU; 300 seconds explicitly allow interrupted-start work, imports, fitting, plotting, bundling and untimed checks. No simulation is omitted intentionally; not an OS-wide audit.')
    worst=max(ev['predictions'],key=lambda r:abs(r['predicted_y']-r['y']))
    residual_note=dict(status='Post-evaluation descriptive diagnostic only; no refit and no change to the registered gate.',
                       worst_case=worst,max_absolute_normalized_error=abs(worst['predicted_y']-worst['y']),
                       interpretation='Mixed-ensemble transfer passes, but long-sweep within-protocol prediction is poor. A universal rare-region law is not established.')
    from geometry import slope,slope_components
    from penrose_core import components
    periods={}
    for kind,ref in {**train['references'],**test['references']}.items():
        L=np.array(ref['L'])
        if L.shape[1]==4:
            S=slope(L);periods[kind]=dict(N=ref['N'],L=L.tolist(),phason_slope=S.tolist(),components=slope_components(S).tolist(),slope_Frobenius_norm=float(np.linalg.norm(S)))
        else:periods[kind]=dict(N=ref['N'],L=L.tolist(),components=components(L))
    wrapped=dict(schema='Every numeric leaf is P (floating-point numerical evidence/resource estimate), with source_script; no interval-certified C pressure claims.',
                 source_script=Path(__file__).name,confirmatory_gate=ev['primary_pass'],preregistration_unchanged=True,model_frozen_before_test=True,
                 datasets={name:label(obj,obj.get('source_script','law_run.py')) for name,obj in d.items()},resources=label(estimate,Path(__file__).name),cpu_accounting=label(cpu_summary,Path(__file__).name),periods=label(periods,Path(__file__).name),descriptive_residual_diagnostic=label(residual_note,Path(__file__).name))
    scripts=list(ROOT.glob('law_*.py'))
    wrapped['script_SHA256']={p.name:sha(p) for p in scripts}
    wrapped['data_SHA256']={name:sha(ROOT/name) for name in names}
    (ROOT/'law_results.json').write_text(json.dumps(wrapped,indent=2),encoding='utf-8')
    with (ROOT/'law_heldout_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['id','kind','protocol','severity','rep','x','y','predicted_y','classification','source_script']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in ev['predictions']:writer.writerow(dict(**r,classification='P',source_script='law_fit.py'))
    print(json.dumps(dict(cpu=cpu_summary,resources=estimate),indent=2))
if __name__=='__main__':main()
