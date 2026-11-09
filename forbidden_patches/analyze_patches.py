"""Fixed scores and explicitly separated exploratory diagnostics."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,time,collections
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from scipy.optimize import nnls
from run_cells import ROOT,save,sha

FEATURES=['phi2','internal_variance','max_excess','mean_excess','moved_fraction']
def rho(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)
def interval(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return np.quantile(a,[.025,.975]).tolist() if len(a) else None
def rowcorr(x,y):
    a=rankdata(x,axis=1);b=rankdata(y,axis=1)
    a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.sqrt((a*a).sum(1)*(b*b).sum(1))
    return np.divide((a*b).sum(1),den,out=np.full(len(a),np.nan),where=den>0)
def indices(rows,B=10000,seed=864108):
    rng=np.random.default_rng(seed);parts=[]
    for family in sorted({r['family'] for r in rows}):
        groups=sorted({r['group'] for r in rows if r['family']==family})
        ids=np.array([[i for i,r in enumerate(rows) if r['group']==g] for g in groups])
        parts.append(ids[rng.integers(0,len(groups),(B,len(groups)))].reshape(B,-1))
    return np.concatenate(parts,axis=1)
def feature(r,key):return r['census']['phi2'] if key=='phi2' else r['features'][key]
def bootstrap_scores(rows,target='Delta'):
    y=np.array([r[target] for r in rows]);idx=indices(rows);result={};dist={}
    for key in FEATURES:
        x=np.array([feature(r,key) for r in rows]);b=rowcorr(x[idx],y[idx]);dist[key]=b
        result[key]=dict(spearman=rho(x,y),bootstrap95=interval(b),finite_bootstrap_replicates=int(np.isfinite(b).sum()))
    diffs={key:dict(point_difference=result['phi2']['spearman']-result[key]['spearman'],
                    bootstrap95=interval(dist['phi2']-dist[key])) for key in FEATURES[1:]}
    return result,diffs

def mixing(rows):
    out=[]
    for group in sorted({r['group'] for r in rows if r['family']=='equilibrated'}):
        pair=[r for r in rows if r['group']==group];stats={}
        for key in ['phi2','internal_variance']:
            chains=np.array([[t[key] for t in r['equilibration_trace']] for r in pair])
            # Split each of the two independent phase/MC chains into halves.
            half=chains.shape[1]//2;split=np.concatenate([chains[:,:half],chains[:,half:]],axis=0)
            n=split.shape[1];W=np.mean(np.var(split,axis=1,ddof=1));B=n*np.var(split.mean(1),ddof=1)
            R=np.sqrt(((n-1)/n*W+B/n)/W) if W>0 else (1. if B==0 else np.inf)
            drift=chains[:,half:].mean()-chains[:,:half].mean()
            stats[key]=dict(split_Rhat=float(R) if np.isfinite(R) else None,constant_trace=bool(np.ptp(chains)==0),
                            first_half_mean=float(chains[:,:half].mean()),second_half_mean=float(chains[:,half:].mean()),
                            drift=float(drift),pooled_sd=float(np.std(chains)))
        out.append(dict(group=group,observables=stats,
                        both_Rhat_below_1_1=all(s['split_Rhat'] is not None and s['split_Rhat']<1.1 for s in stats.values())))
    return out

def f4_fit(rows,finest=None,B=10000):
    ordered=sorted(rows,key=lambda r:r['delta'],reverse=True)
    levels=sorted({r['delta'] for r in ordered},reverse=True)
    if finest:levels=levels[-finest:];ordered=[r for r in ordered if r['delta'] in levels]
    x=np.array([r['delta'] for r in ordered]);y=np.array([r['Delta_iso'] for r in ordered])
    A=np.c_[np.ones(len(x)),x,x*x];coef=np.linalg.lstsq(A,y,rcond=None)[0]
    rng=np.random.default_rng(864109);boot=[];invalid=0
    groups=[[i for i,r in enumerate(ordered) if r['delta']==v] for v in levels]
    for _ in range(B):
        draw=rng.integers(0,len(groups),len(groups))
        if len(set(draw))<3:invalid+=1;continue
        ids=np.concatenate([rng.choice(groups[j],len(groups[j]),replace=True) for j in draw])
        boot.append(np.linalg.lstsq(A[ids],y[ids],rcond=None)[0][1])
    return dict(orders=len(levels),configurations=len(ordered),coefficient_intercept_linear_quadratic=coef.tolist(),
                linear_bootstrap95=interval(boot),rms=float(np.sqrt(np.mean((A@coef-y)**2))),
                valid_bootstrap=len(boot),rank_deficient_resamples=invalid,
                uncertainty_scope='Pairs bootstrap over Pell orders and offsets; rank-deficient resamples excluded. Does not bound omitted higher orders, reference bias, or asymptotic extrapolation.')

def main():
    start=time.process_time();manifest=json.loads((ROOT/'cell_manifest.json').read_text());rows=[];missing=[]
    for s in manifest['rows']:
        p=ROOT/'raw'/(s['id']+'.json')
        if p.exists():rows.append(json.loads(p.read_text()))
        else:missing.append(s['id'])
    if missing:raise RuntimeError(f'Primary cohort incomplete: {len(missing)} missing')
    ab=[r for r in rows if r['symmetry']=='AB'];pen=[r for r in rows if r['symmetry']=='Penrose']
    scores,diff=bootstrap_scores(ab);pscores,_=bootstrap_scores(pen)
    ax=np.array([r['census']['phi2'] for r in ab]);ay=np.array([r['Delta'] for r in ab])
    coef=nnls(np.c_[ax,ax*ax],ay)[0]
    px=np.array([r['census']['phi2'] for r in pen]);py=np.array([r['Delta'] for r in pen]);pred=coef[0]*px+coef[1]*px*px
    pi=indices(pen,seed=864110);pr=rho(pred,py);pred_ci=interval(rowcorr(pred[pi],py[pi]))
    # The AB-only functional form is monotone, so Spearman is invariant to
    # its positive coefficients and to any allowed positive overall scale.
    f4=[]
    for s in manifest['F4']:
        p=ROOT/'raw'/(s['id']+'.json')
        if p.exists():f4.append(json.loads(p.read_text()))
    allfit=f4_fit(f4);finefit=f4_fit(f4,3)
    f4zero=all(r['census']['phi2']==0 for r in f4)
    # No precise F4 statistical rule is registered. Preserve this limitation
    # explicitly; the finite-data necessary-condition test uses the 95% CI.
    f4compatible=f4zero and all(f['linear_bootstrap95'][0]<=0<=f['linear_bootstrap95'][1] for f in [allfit,finefit])
    verdicts=dict(
        F1=dict(result='PASS' if len(ab)>=40 and scores['phi2']['spearman']>=.9 else 'FAIL',label='P',
                N=len(ab),criterion='Spearman >= 0.9, all four families, N >= 40',**scores['phi2']),
        F2=dict(result='PASS' if all(v['point_difference']>=0 for v in diff.values()) else 'FAIL',label='P',
                criterion='phi2 signed Spearman at least as large as each alternative; paired 95% CIs reported',
                correlations=scores,paired_differences=diff,
                interpretation='Registration does not specify whether CI dominance or point-estimate dominance determines success; both point and interval evidence are shown.'),
        F3=dict(result='PASS' if len(pen)>=30 and pr>=.85 else 'FAIL',label='P',N=len(pen),spearman=pr,bootstrap95=pred_ci,
                criterion='AB-only functional form; Penrose Spearman >= .85, N >= 30',
                model=dict(form='a*phi2+b*phi2**2; a,b>=0; zero intercept',coefficients=coef.tolist(),scale_factor=1.,
                           fit_source='48 AB held-out configurations; no Penrose fitting',
                           AB_rms=float(np.sqrt(np.mean((coef[0]*ax+coef[1]*ax*ax-ay)**2))),
                           Penrose_rms=float(np.sqrt(np.mean((pred-py)**2))))),
        F4=dict(result='PASS' if f4compatible else 'FAIL',label='P',configurations=len(f4),all_phi2_zero=f4zero,
                all_orders=allfit,finest_three=finefit,
                criterion='Zero forbidden -> no linear isotropic penalty; necessary finite-data diagnostic checks whether fitted linear coefficient includes zero at 95%.',
                interpretation='Binary numerical assessment, not a proof/counterproof of an asymptotic universal law. Registration supplies no order range, fitting model, or equivalence tolerance; these operational choices were frozen before pressures.'))
    for f in verdicts.values():f['sources']=['cell_manifest.json','raw/*.json','analyze_patches.py']
    mix=mixing(rows)
    summaries=[]
    for sym in ['AB','Penrose']:
        for family in ['approximant','few_flip','equilibrated','disc_cluster']:
            rr=[r for r in rows if r['symmetry']==sym and r['family']==family]
            if not rr:continue
            x=[r['census']['phi2'] for r in rr];y=[r['Delta'] for r in rr]
            summaries.append(dict(symmetry=sym,family=family,N=len(rr),phi2_range=[min(x),max(x)],
                                  Delta_range=[min(y),max(y)],spearman=rho(x,y)))
    # Exploratory alternative target and family-wise correlations never alter F1-F3.
    iso,_=bootstrap_scores(ab,'Delta_iso')
    atlas=json.loads((ROOT/'patch_atlas.json').read_text())
    audits=json.loads((ROOT/'cell_audits.json').read_text()) if (ROOT/'cell_audits.json').exists() else None
    extensions=[json.loads(p.read_text()) for p in (ROOT/'extensions').glob('*.json')] if (ROOT/'extensions').exists() else []
    extfit=f4_fit(f4+[r for r in extensions if r['family']=='F4_Pell']) if extensions else None
    extfine=f4_fit(f4+[r for r in extensions if r['family']=='F4_Pell'],3) if extensions else None
    verdicts['F4']['registered_claim_status']='not_established; asymptotic statement remains unresolved'
    verdicts['F4']['not_a_robust_falsification']=True
    verdicts['F4']['finite_diagnostic_result']=verdicts['F4']['result']
    verdicts['F4']['interpretation']+=(' FAIL means not established here. The separately computed larger Pell order '
        'makes the five-order linear-coefficient bootstrap include zero and changes the finest-three fit; '
        'therefore the finite four-order failure is not a robust asymptotic falsification.') if extfit and extfit['orders']>4 else ''
    long_rows=[r for r in extensions if r['family']=='equilibrated' and r['id'].startswith('extension_') and not r['id'].startswith('extension_large_')]
    complete_groups={g for g in {r['group'] for r in long_rows} if sum(r['group']==g for r in long_rows)==2}
    long_diagnostics=mixing([r for r in long_rows if r['group'] in complete_groups]) if complete_groups else []
    replacements={r['id'].removeprefix('extension_'):r for r in long_rows}
    sensitivity=[]
    for sym in ['AB','Penrose']:
        original=[r for r in rows if r['symmetry']==sym]
        alt=[replacements.get(r['id'],r) for r in original]
        sensitivity.append(dict(symmetry=sym,replaced=sum(r['id'] in replacements for r in original),
                                phi2_spearman=rho([r['census']['phi2'] for r in alt],[r['Delta'] for r in alt]),
                                label='Exploratory replacement sensitivity, never substituted in primary scores'))
    f4audit=json.loads((ROOT/'pell_audit.json').read_text()) if (ROOT/'pell_audit.json').exists() else None
    integrity=json.loads((ROOT/'validation.json').read_text()) if (ROOT/'validation.json').exists() else None
    fresh=[r for r in extensions if r['id'].startswith('fresh_')]
    replication=dict(completed=len(fresh),requested=168,manifest='fresh_manifest.json',
                     role='Prospective independent replication; never substituted in frozen primary decisions')
    if len(fresh)==168:
        fa=[r for r in fresh if r['symmetry']=='AB'];fp=[r for r in fresh if r['symmetry']=='Penrose']
        fs,fd=bootstrap_scores(fa);pp=np.array([r['census']['phi2'] for r in fp]);py2=np.array([r['Delta'] for r in fp])
        gp=coef[0]*pp+coef[1]*pp*pp;idx=indices(fp,seed=864110)
        replication.update(AB_correlations=fs,AB_paired_differences=fd,
                           F1='PASS' if fs['phi2']['spearman']>=.9 else 'FAIL',
                           F2='PASS' if all(z['point_difference']>=0 for z in fd.values()) else 'FAIL',
                           F3='PASS' if rho(gp,py2)>=.85 else 'FAIL',
                           Penrose_spearman=rho(gp,py2),Penrose_bootstrap95=interval(rowcorr(gp[idx],py2[idx])),
                           AB_model_coefficients_unchanged=coef.tolist())
    # CPU is process CPU, single BLAS/Numba thread. Include row costs rather
    # than double-counting completed-invocation totals that contain those rows.
    cpu=dict(primary_rows=sum(r['cpu_s'] for r in rows),F4_rows=sum(r['cpu_s'] for r in f4),
             atlas=atlas['cpu_s'],audits=audits['cpu_s'] if audits else 0,
             extensions=sum(r['cpu_s'] for r in extensions),F4_angular_audit=f4audit['cpu_s'] if f4audit else 0,
             integrity_check=integrity['cpu_s'] if integrity else 0)
    output=dict(task='forbidden-patch law',label_definitions={'C':'Exact finite combinatorial identity/count, conditional on the supplied graph and atlas.',
                  'P':'Numerical point estimate or statistical inference; no rigorous spectral or infinite-atlas certificate.'},
                preregistration=dict(path='../registered_plan.md',sha256=manifest['prereg_sha256'],unchanged=True),
                policy=manifest['policy'],manifest_sha256=sha(ROOT/'cell_manifest.json'),verdicts=verdicts,
                implementation_notes=['The frozen grid specification is implemented exactly. Its parenthetical prose says ten non-Gamma points, but the 4x4 grid has ten inversion-inequivalent points INCLUDING Gamma (nine non-Gamma).',
                                      'Bootstrap RNG streams: F1/F2 base 864108; F4 864109; F3 864110.'],
                atlas={s:dict(label='P',source='patch_atlas.py; patch_atlas.json',**{k:v for k,v in a.items() if k!='patches'}) for s,a in atlas['symmetries'].items()},
                combinatorial_checks=dict(label='C',claim='Integer V-E+F is zero for the stored torus face enumerations; Pell forbidden counts are exactly zero against the stored finite atlas. Geometric generation and spectral calculations remain P.',source='raw/*.json; patch_atlas.py'),
                equilibration=dict(label='P',diagnostics=mix,all_observed_Rhat_below_1_1=all(r['both_Rhat_below_1_1'] for r in mix),
                                   limitation='Finite observable diagnostics cannot prove full mixing or ergodicity of the flip sector.'),
                exploratory=dict(label='P',family_correlations=summaries,AB_isotropic_target=iso,
                                 extension_F4_fit=extfit,extension_F4_finest_three=extfine,
                                 extended_equilibration=dict(configurations=len(long_rows),diagnostics=long_diagnostics,
                                                             replacement_sensitivity=sensitivity),
                                 F4_angular_audit=f4audit,
                                 extension_sources=['extensions/'+r['id']+'.json' for r in extensions]),
                audits=audits,integrity_validation='validation.json' if integrity else None,
                prospective_replication=replication,CPU_seconds=cpu,
                configurations=[dict(id=r['id'],symmetry=r['symmetry'],family=r['family'],group=r['group'],label='P',
                                     source='raw/'+r['id']+'.json',snapshot='snapshots/'+r['id']+'.npz',N=r['N'],L=r['L'],
                                     phase_seed=r['phase_seed'],mc_seed=r['mc_seed'],phi2=r['census']['phi2'],
                                     forbidden_vertices=r['census']['forbidden_vertices'],features=r['features'],
                                     pc=r['mechanics']['pc'],c0=r['mechanics']['c0'],Delta=r['Delta'],Delta_iso=r['Delta_iso']) for r in rows+f4],
                limitations=['Finite atlas stability does not certify the infinite canonical language.',
                             'Bloch grid plus acoustic optimization is a sampled threshold, not a global spectral certificate.',
                             'New seeds and randomly generated oblique periods; no prior outputs reused. Unpublished prior random-period lists were not accessed, so collision exclusion against those lists is not certified.',
                             'F4 is asymptotic/universal; finite-order regression and its bootstrap do not by themselves settle the infinite-order claim.',
                             'Bootstrap describes this finite designed ensemble, not all possible tilings.'])
    output['CPU_seconds']['analysis']=time.process_time()-start
    output['CPU_seconds']['total_accounted']=sum(output['CPU_seconds'].values())
    save(ROOT/'forbidden_results.json',output)
    print(json.dumps(verdicts,indent=2));print('CPU',output['CPU_seconds']);print('mixing',output['equilibration']['all_observed_Rhat_below_1_1'])
if __name__=='__main__':main()
