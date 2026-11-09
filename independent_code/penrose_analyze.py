"""Auditable statistics and C/P-labelled assembly for the independent check."""
from penrose_core import *
from collections import defaultdict
from scipy.stats import spearmanr,rankdata,pearsonr,t as student
from scipy.optimize import curve_fit
import hashlib

def stats(v):
    v=np.asarray(v,float);n=len(v);sd=float(v.std(ddof=1)) if n>1 else 0.
    err=student.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else 0.
    return dict(n=n,mean=float(v.mean()),sd=sd,ci95_mean=[float(v.mean()-err),float(v.mean()+err)],min=float(v.min()),max=float(v.max()))
def groups(rows):
    g=defaultdict(list)
    for r in rows:g[r['group']].append(r)
    return g
def summary(rows):
    out={}
    for name,rr in groups(rows).items():
        out[name]=dict(V=rr[0]['V'],components=rr[0]['components'],
                      h4_re=stats([r['angular']['h4'][0] for r in rr]),
                      h4_im=stats([r['angular']['h4'][1] for r in rr]),
                      c0=stats([r['angular']['c0'] for r in rr]),
                      pc=stats([r.get('continuous_pc',r['angular']['sampled_pc']) for r in rr]),
                      kappa_a_eff=stats([r['angular']['h4'][0]/r['components']['a'][0] for r in rr]))
    return out
def pairfits(rows):
    g=groups(rows);out=[];rng=np.random.default_rng(44401)
    for A,B in [('a_3','fib_2'),('a_4','fib_3'),('a_5','fib_4')]:
        ra,rb=g[A],g[B];n=min(len(ra),len(rb));a=ra[0]['components']['a'][0]
        ba,bb=ra[0]['components']['b'][0],rb[0]['components']['b'][0]
        ha=np.array([r['angular']['h4'][0] for r in ra]);hb=np.array([r['angular']['h4'][0] for r in rb])
        def fit(x,y):
            kb=(y-x)/(bb-ba);ka=(x-kb*ba)/a;return ka,kb,kb/ka
        value=fit(ha.mean(),hb.mean());ix=rng.integers(n,size=(5000,n));boot=np.array(fit(ha[:n][ix].mean(1),hb[:n][ix].mean(1))).T
        out.append(dict(pair=[A,B],a=a,b_ratio=bb/ba,kappa_a=value[0],kappa_b=value[1],ratio=value[2],
                        bootstrap_ci95=np.percentile(boot,[2.5,97.5],axis=0).T.tolist(),
                        caveat='Two-cell linear estimate confounds b^2 and finite-slope corrections.'))
    names=['a_3','fib_2','a_4','fib_3','a_5','fib_4'];X=[];Y=[]
    for name in names:
        rr=g[name];a=rr[0]['components']['a'][0];b=rr[0]['components']['b'][0]
        X.append([a,b,b*b]);Y.append([r['angular']['h4'][0] for r in rr])
    X=np.array(X);Y=np.array(Y);w=1/np.abs(X[:,0])
    op=np.linalg.pinv(X*w[:,None]);coef=op@(Y.mean(1)*w)
    ix=rng.integers(Y.shape[1],size=(5000,Y.shape[1]));yb=Y[:,ix].mean(2)
    cb=op@(yb*w[:,None]);ratio=cb[1]/cb[0]
    return dict(pairs=out,joint=dict(coefficients=coef.tolist(),coefficient_names=['kappa_a','kappa_b','mu_b_squared'],
             kappa_b_over_a=float(coef[1]/coef[0]),ci95_coefficients=np.percentile(cb,[2.5,97.5],axis=1).T.tolist(),
             ci95_ratio=np.percentile(ratio,[2.5,97.5]).tolist(),weight='residual multiplied by 1/abs(a)',
             rms=float(np.sqrt(np.mean((X@coef-Y.mean(1))**2)))))
def correlation(rows):
    metrics=['max_excess','mean_excess','internal_variance','moved_fraction','max_facet_excess','mean_facet_excess']
    pressure=np.array([r['threshold']['pc'] for r in rows]);out={};rng=np.random.default_rng(44402)
    for name in metrics:
        x=np.array([r['window'][name] for r in rows]);rho,p=spearmanr(x,pressure)
        boot=[]
        for _ in range(2000):
            ix=rng.integers(len(rows),size=len(rows));boot.append(spearmanr(x[ix],pressure[ix]).statistic)
        out[name]=dict(rho=float(rho),p=float(p),bootstrap_ci95=np.nanpercentile(boot,[2.5,97.5]).tolist())
    moved=np.array([r['window']['moved_fraction'] for r in rows]);Z=np.c_[np.ones(len(rows)),rankdata(moved)]
    yy=rankdata(pressure);yy-=Z@np.linalg.lstsq(Z,yy,rcond=None)[0]
    for name in ['max_excess','mean_excess','internal_variance']:
        xx=rankdata([r['window'][name] for r in rows]);xx-=Z@np.linalg.lstsq(Z,xx,rcond=None)[0]
        out[name]['partial_rho_given_moved']=float(pearsonr(xx,yy).statistic)
    return dict(n=len(rows),metrics=out,ids=[r['id'] for r in rows])

def numeric_labels(obj,source,classification='P'):
    if isinstance(obj,bool) or obj is None:return obj
    if isinstance(obj,(int,float,np.integer,np.floating)):
        return dict(value=float(obj) if isinstance(obj,(float,np.floating)) else int(obj),classification=classification,source_script=source)
    if isinstance(obj,list):return [numeric_labels(v,source,classification) for v in obj]
    if isinstance(obj,dict):return {k:numeric_labels(v,source,classification) for k,v in obj.items()}
    return obj

def main():
    start=time.process_time();files=['penrose_slopes_raw.json','penrose_controls_raw.json','penrose_offsets_raw.json','penrose_haar_raw.json','penrose_disorder_raw.json']
    raw={name:json.loads((ROOT/name).read_text()) for name in files}
    slopes=raw[files[0]]['rows'];controls=raw[files[1]]['rows'];offsets=raw[files[2]]['rows'];haar=raw[files[3]]['rows'];disorder=raw[files[4]]
    assert len(slopes)==72 and len(controls)==47 and len(offsets)==70 and len(haar)==144
    assert len(disorder['rows'])==63 and len({r['id'] for r in disorder['rows']})==63
    assert 'mesh_check' in disorder and 'last_invocation_cpu_s' in disorder
    out=dict(slope_summary=summary(slopes),haar_summary=summary(haar),selection=pairfits(haar))
    # Exact-V regressions use phase-averaged cells, with no imported QC reference.
    fixed=[r for r in controls if r['group'].startswith('fixed') and r['angular']['sub']==2]
    fs=summary(fixed);out['fixed_V_cells']=fs;out['fixed_V_fits']={}
    for V in [1440,1288]:
        rows=[v for v in fs.values() if v['V']==V];x=np.array([r['components']['delta'] for r in rows]);y=np.array([r['c0']['mean'] for r in rows])
        X=np.c_[np.ones(len(x)),x];coef=np.linalg.lstsq(X,y,rcond=None)[0]
        rr=[r for r in fixed if r['V']==V]
        out['fixed_V_fits'][str(V)]=dict(n_cells=len(rows),intercept=coef[0],penalty_coefficient=-coef[1],rms=float(np.sqrt(np.mean((X@coef-y)**2))),spearman=float(spearmanr(x,y).statistic),
                thin_counts=sorted(set(r['geometry']['thin_faces_from_periods'] for r in rr)),
                fat_counts=sorted(set(r['geometry']['fat_faces_from_periods'] for r in rr)),
                area_per_vertex=stats([r['geometry']['area']/r['V'] for r in rr]),
                mean_axial_force=stats([r['angular']['pre']['axial_compression_mean'] for r in rr]))
    # Free-reference exponent is descriptive, not proof of an asymptotic cusp.
    sm=out['slope_summary'];xx=np.array([r['components']['delta'] for r in sm.values()]);yy=np.array([r['c0']['mean'] for r in sm.values()])
    def model(x,p0,c,beta):return p0-c*x**beta
    coef,cov=curve_fit(model,xx,yy,p0=[.811,.15,.9],bounds=([.809,0,.1],[.82,10,3]),maxfev=20000)
    out['isotropic_descriptive_power_fit']=dict(parameters=coef.tolist(),standard_errors=np.sqrt(np.diag(cov)).tolist(),parameter_names=['p0','c','beta'])
    X=np.c_[np.ones(len(xx)),xx];lin=np.linalg.lstsq(X,yy,rcond=None)[0]
    out['isotropic_descriptive_linear_fit']=dict(intercept=lin[0],penalty_coefficient=-lin[1],rms=float(np.sqrt(np.mean((X@lin-yy)**2))))
    out['offset_summary']={}
    for name in ['fib_4','fib_5']:
        rr=[r for r in offsets if r['group']==name and r['angular']['sub']==2]
        base={r['seed']:r for r in rr if r['s']==0}
        out['offset_summary'][name]={}
        for s in sorted(set(r['s'] for r in rr)):
            ss=[r for r in rr if r['s']==s]
            out['offset_summary'][name][str(s)]=dict(c0=stats([r['angular']['c0'] for r in ss]),pc=stats([r['continuous_pc'] for r in ss]),
                     pressure_drop_pct=stats([100*(1-r['continuous_pc']/base[r['seed']]['continuous_pc']) for r in ss]),
                     c0_drop_pct=stats([100*(1-r['angular']['c0']/base[r['seed']]['angular']['c0']) for r in ss]))
    # Pair all mesh-four rows to their identical geometry and phase.
    allrows=slopes+controls+offsets;mesh=[];rich=[]
    for r in allrows:
        if 'richardson' in r:
            dr=np.array(r['richardson']['directions'])-r['angular']['directions']
            rich.append(dict(id=r['id'],max_angular_error=float(max(abs(dr))),c0_error=r['richardson']['c0']-r['angular']['c0'],
                             finite_k_min=min(v['pc'] for v in r['richardson']['finite_k']),pc=r['continuous_pc']))
        if r['angular']['sub']!=4:continue
        candidates=[q for q in allrows if q['angular']['sub']==2 and q['L']==r['L'] and q['seed']==r['seed'] and q['s']==r['s']]
        if candidates:
            q=candidates[0];mesh.append(dict(id=r['id'],c0_sub2=q['angular']['c0'],c0_sub4=r['angular']['c0'],
                pc_sub2=q['continuous_pc'],pc_sub4=r['continuous_pc'],relative_pc_change_pct=100*(r['continuous_pc']/q['continuous_pc']-1),
                h4_re_sub2=q['angular']['h4'][0],h4_re_sub4=r['angular']['h4'][0]))
    out['mesh']=mesh;out['richardson']=rich;out['rotation']=raw[files[1]].get('rotation')
    base=disorder['baseline']['threshold']['pc'];out['disorder_baseline']=base
    rows=[r for r in disorder['rows'] if r['protocol']!='mixing_check']
    primary=[r for r in rows if r['protocol']!='uniform' or r['seed']<=503]
    out['correlation_primary_protocol_matched']=correlation(primary);out['correlation_all_fresh']=correlation(rows)
    out['small_disorder']={}
    for k in [1,3,10,30,100,300,1000,3000]:
        rr=[r for r in rows if r['protocol']=='uniform' and r['flips']==k];delta=[r['threshold']['pc']-base for r in rr]
        out['small_disorder'][str(k)]=dict(change=stats(delta),raises=sum(v>0 for v in delta),drop_pct=stats([-100*v/base for v in delta]),ids=[r['id'] for r in rr])
    out['long_runs']=[dict(id=r['id'],pc=r['threshold']['pc'],loss_pct=100*(1-r['threshold']['pc']/base),window=r['window']) for r in disorder['rows'] if r['protocol'] in ['long_uniform','mixing_check']]
    out['region_runs']=[dict(id=r['id'],pc=r['threshold']['pc'],loss_pct=100*(1-r['threshold']['pc']/base),window=r['window']) for r in rows if r['protocol']=='region']
    if (ROOT/'penrose_fine_raw.json').exists():
        fine=json.loads((ROOT/'penrose_fine_raw.json').read_text());out['fine_reference']=fine
    else:fine={}
    out['recorded_complete_script_invocations_cpu_s']=sum(d.get('last_invocation_cpu_s',0) for d in raw.values())+fine.get('last_invocation_cpu_s',0)
    audit=json.loads((ROOT/'penrose_continuation_audit.json').read_text()) if (ROOT/'penrose_continuation_audit.json').exists() else {}
    out['recorded_preinterruption_disorder_component_cpu_s']=audit.get('preexisting_disorder_component_cpu_s',0.)
    out['accounted_calculation_cpu_s']=out['recorded_complete_script_invocations_cpu_s']+out['recorded_preinterruption_disorder_component_cpu_s']
    auxiliary=['penrose_small_mesh_raw.json','penrose_raise_mesh_raw.json','penrose_localization_raw.json','penrose_membership_raw.json','penrose_offset_plateau_raw.json']
    aux_cpu=0.
    for name in auxiliary:
        if (ROOT/name).exists():
            dd=json.loads((ROOT/name).read_text())
            if 'cpu_s' in dd:aux_cpu+=dd['cpu_s']
            else:aux_cpu+=sum(v.get('cpu_s',0) for v in dd.values() if isinstance(v,dict))
    out['recorded_auxiliary_cpu_s']=aux_cpu
    out['total_CPU_hours_with_500s_overhead_allowance']=(out['accounted_calculation_cpu_s']+aux_cpu+500)/3600
    out['CPU_accounting_note']='Completed-script process times plus saved preinterruption MC/mechanics components and auxiliary times. An explicit 500-second allowance covers imports, geometry, smoke checks, assembly and untimed overhead; not an OS-wide exact total.'
    out['analysis_cpu_s']=time.process_time()-start
    save('penrose_statistics.json',out)
    wrapped=dict(schema='Every numeric leaf has C/P and source_script. C=exact algebraic check, P=numerical evidence without interval certification.',
                 model='Independent Penrose multigrid plus independent AB Euler-Bernoulli rigid-frame solver, r=.03, unit EA and edge, hydrostatic stress; axial-force geometric stiffness.',
                 independence='Only and penrose/penrose_results.json were read from the Penrose solver. Its code was not read, imported or reused.',
                 statistics=numeric_labels(out,Path(__file__).name),datasets={})
    wrapped['verdicts']={
        'representation':'Confirmed for the differentiable linear response. Symmetry alone allows nondifferentiable degree-one b-driven fourfold terms.',
        'matched_pairs':'Substantial a coupling and unresolved small b coefficient reproduced on three pairs with fresh phase averages.',
        'isotropic_penalty':'Reproduced at fixed vertex count, area, tile composition and mean axial force; universal slope coefficient, exact asymptotic exponent and every phason direction are unproved.',
        'offset':'Reported finite-offset penalty reproduced; strict finite-cell maximum over continuous offset is ruled out by identical-geometry plateaus.',
        'small_flips':'Universal lowering is refuted by a mesh-checked raising flip, independently regenerated as another canonical approximant phase.',
        'long_disorder':'Large pressure loss reproduced; run length does not certify equilibration and the observed range depends on sample/time.',
        'window_metric':'Empirical correlations are reported for both the protocol-matched and expanded ensembles; they are not a universal monotonicity law.'}
    reported=dict(kappa_a_range=[-.54,-.35],kappa_b_over_a_joint=.02,isotropic_penalty_coefficient=.2,
                  offset_loss_percent={'s_0.1':1.3,'s_0.5':2.8},long_run_loss_percent_range=[47,52],
                  max_window_excess_spearman=-.96,correlation_n=33)
    wrapped['reported_comparison_claims']=numeric_labels(reported,'../penrose/penrose_results.json (reported input; Penrose code not read or executed)')
    for name,d in raw.items():wrapped['datasets'][name]=numeric_labels(d,d.get('source','penrose_slopes.py'))
    sym=json.loads((ROOT/'penrose_symmetry.json').read_text());wrapped['symbolic']=numeric_labels(sym,'penrose_symmetry.py','C')
    wrapped['symbolic']['cpu_s']=numeric_labels(sym['cpu_s'],'penrose_symmetry.py','P')
    for name in ['penrose_fine_raw.json']+auxiliary:
        if (ROOT/name).exists():
            d=json.loads((ROOT/name).read_text());wrapped['datasets'][name]=numeric_labels(d,d['source'])
    if audit:wrapped['continuation_audit']=numeric_labels(audit,'penrose_analyze.py; saved checkpoint timings')
    scripts=sorted(ROOT.glob('penrose_*.py'))
    wrapped['script_SHA256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    save('penrose_check_results.json',wrapped)
    print(json.dumps({k:v for k,v in out.items() if k not in ['slope_summary','haar_summary','fixed_V_cells','richardson','rotation']},indent=2))

if __name__=='__main__':main()
