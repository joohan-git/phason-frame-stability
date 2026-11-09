"""Round-3 fits and audit. CIs are conditional/descriptive, not certification."""
from ab_common import *
from scipy.stats import linregress,t as student,spearmanr,rankdata

def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def logfit(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if np.any(y<=0):return dict(status='Nonpositive mean/drop; log fit undefined',x=x.tolist(),y=y.tolist())
    fit=linregress(np.log(x),np.log(y));n=len(x)
    width=student.ppf(.975,n-2)*fit.stderr if n>2 else np.nan
    return dict(n=n,alpha=float(fit.slope),alpha_CI95=[float(fit.slope-width),float(fit.slope+width)],
                prefactor=float(np.exp(fit.intercept)),R2=float(fit.rvalue**2),x=x.tolist(),y=y.tolist(),
                interpretation='OLS log-residual t interval conditional on reference, points and iid residual model; not an asymptotic or numerical-error bound.')

def pell_analysis(rows):
    out=[];points=[(1,2),(4,5),(2,3),(7,10),(12,17),(29,41)]
    for sub,matched,key in [(2,True,'rich'),(2,True,'fine_rich'),(2,True,'zero'),(4,True,'zero'),(8,True,'zero')]:
        rr=[next((r for r in rows if (r['p'],r['q'])==pq and r['sub']==sub and r['matched_offset']==matched),None) for pq in points]
        if any(r is None or key+'_min' not in r for r in rr):continue
        anchor=rr[-1][key+'_c0'];x=np.array([abs(r['b']) for r in rr]);pc=np.array([r[key+'_min'] for r in rr])
        fits=[dict(retained_points=[f"{r['p']}/{r['q']}" for r in rr[i:]],**logfit(x[i:],anchor-pc[i:])) for i in range(4)]
        pure=[i for i,r in enumerate(rr) if (r['p'],r['q'])!=(4,5)]
        # Isotropic-centre drift is a systematic sensitivity scale, not a CI.
        scale=abs(rr[-1][key+'_c0']-rr[-2][key+'_c0'])
        sensitivity=[]
        for shift in [-scale,0.,scale]:
            sensitivity.append(dict(reference=anchor+shift,shift=shift,all_points=logfit(x,anchor+shift-pc),
                                    last_three=logfit(x[-3:],anchor+shift-pc[-3:])))
        tail=rr[-3:];xx=x[-3:];yy=pc[-3:];ratio=(yy[0]-yy[1])/(yy[1]-yy[2])
        free_alpha=np.log(abs(ratio))/np.log(xx[0]/xx[1]);free_limit=yy[2]-(yy[1]-yy[2])/(ratio-1)
        entry=dict(sub=sub,method=key,reference=anchor,reference_sensitivity_scale=scale,nested=fits,
                   pell_only=logfit(x[pure],anchor-pc[pure]),reference_sensitivity=sensitivity,
                   free_three_point_fit=dict(alpha=float(free_alpha),limit=float(free_limit),degrees_of_freedom=0),
                   centers=[dict(p=r['p'],q=r['q'],b=r['b'],c0=r[key+'_c0'],h4=r[key+'_h4'],
                                 kappa=r[key+'_h4']/r['b'],pc=r[key+'_min']) for r in rr])
        refpath=ROOT/'ab_reference_raw.json'
        if refpath.exists() and sub==2:
            ref=read('ab_reference_raw.json')['row']['zero_c0']
            entry['larger_cell_reference']=dict(reference=ref,nested=[logfit(x[i:],ref-pc[i:]) for i in range(4)])
        out.append(entry)
    main=read('../phason_paths/paths_results.json')
    reported=dict(next(v['value'] for v in main['items'] if v['key']=='pell_cusp_fit'))
    reported['source_script']='../phason_paths/assemble_paths.py'
    return dict(fits=out,main_reported=reported)

def disorder_analysis(data):
    rows=data['rows'];ks=sorted({r['k'] for r in rows});seeds=sorted({r['seed'] for r in rows})
    # Only full seed trajectories enter joint confidence bands and power fits.
    seeds=[s for s in seeds if all(any(r['seed']==s and r['k']==k for r in rows) for k in ks)]
    rr={(r['seed'],r['k']):r for r in rows}
    drop=np.array([[rr[s,k]['relative_drop'] for k in ks] for s in seeds]);moved=np.array([[rr[s,k]['moved_fraction'] for k in ks] for s in seeds])
    rng=np.random.default_rng(92471);idx=rng.integers(0,len(seeds),(10000,len(seeds)))
    boot=drop[idx].mean(axis=1);bm=moved[idx].mean(axis=1)
    groups=[]
    for j,k in enumerate(ks):
        mean=drop[:,j].mean();sd=drop[:,j].std(ddof=1);ci=np.quantile(boot[:,j],[.025,.975])
        groups.append(dict(k=k,n=len(seeds),moved_mean=float(moved[:,j].mean()),mean_relative_drop=float(mean),
                           std_relative_drop=float(sd),bootstrap_CI95=ci.tolist(),
                           mean_t_CI95=[float(mean-student.ppf(.975,len(seeds)-1)*sd/np.sqrt(len(seeds))),
                                        float(mean+student.ppf(.975,len(seeds)-1)*sd/np.sqrt(len(seeds)))],
                           raising_count=int(np.sum(drop[:,j]<-1e-7)),lowering_count=int(np.sum(drop[:,j]>1e-7)),
                           minimum=float(drop[:,j].min()),maximum=float(drop[:,j].max())))
    fits=[]
    for low,high in [(1,30),(1,100),(10,300),(30,1000)]:
        jj=[j for j,k in enumerate(ks) if low<=k<=high]
        fit=logfit(moved.mean(0)[jj],drop.mean(0)[jj]);powers=[]
        for b in range(len(boot)):
            if np.all(boot[b,jj]>0):powers.append(np.polyfit(np.log(bm[b,jj]),np.log(boot[b,jj]),1)[0])
        fit.update(k_range=[low,high],seed_bootstrap_alpha_CI95=np.quantile(powers,[.025,.975]).tolist() if powers else None,
                   valid_bootstrap_fraction=len(powers)/len(boot))
        fits.append(fit)
    small=[j for j,k in enumerate(ks) if k<=30]
    xm=moved.mean(0)[small];ym=drop.mean(0)[small]
    coef=np.linalg.lstsq(np.c_[xm,xm*xm],ym,rcond=None)[0]
    cb=[];linear=[]
    for j in range(len(boot)):
        xb=bm[j,small];yb=boot[j,small]
        cb.append(np.linalg.lstsq(np.c_[xb,xb*xb],yb,rcond=None)[0]);linear.append(float(xb@yb/(xb@xb)))
    poly=dict(k_max=30,linear_plus_quadratic_coefficients=coef.tolist(),
              coefficient_CI95=np.quantile(cb,[.025,.975],axis=0).tolist(),
              linear_only_coefficient=float(xm@ym/(xm@xm)),linear_only_CI95=np.quantile(linear,[.025,.975]).tolist(),
              definition='Mean relative pressure loss = a*moved + b*moved^2, zero intercept; descriptive local fit.')
    return dict(complete_seeds=seeds,groups=groups,power_fits=fits,small_density_polynomial=poly,
                note='Pointwise mean CIs and exponent CIs resample entire seed trajectories; accepted-flip trajectories are not equilibrium draws.')

def rho_stats(x,y,rng,groups=None,B=5000):
    x=np.asarray(x);y=np.asarray(y);n=len(x);r=spearmanr(x,y);bs=[]
    if groups is None:blocks=[np.arange(n)]
    else:blocks=[np.flatnonzero(np.array(groups)==g) for g in sorted(set(groups))]
    for _ in range(B):
        ix=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in blocks])
        if len(set(x[ix]))>1 and len(set(y[ix]))>1:bs.append(spearmanr(x[ix],y[ix]).statistic)
    return dict(n=n,rho=float(r.statistic),asymptotic_p=float(r.pvalue),bootstrap_CI95=np.quantile(bs,[.025,.975]).tolist())

def partial_rho(x,y,moved,protocol):
    labs=sorted(set(protocol));D=np.c_[np.ones(len(x)),rankdata(moved),*[np.array(protocol)==g for g in labs[1:]]]
    rx=rankdata(x);ry=rankdata(y);rx-=D@np.linalg.lstsq(D,rx,rcond=None)[0];ry-=D@np.linalg.lstsq(D,ry,rcond=None)[0]
    return float(np.corrcoef(rx,ry)[0,1])

def window_analysis(data):
    rows=data['rows'];rng=np.random.default_rng(95713);groups=[r['protocol'] for r in rows]
    pc=np.array([r['mechanics']['Gamma'] for r in rows]);allpc=np.array([r['mechanics']['minimum_sampled'] for r in rows])
    out={}
    for metric in ['mean_excess','euclidean_mean_excess','recentered_mean_excess','moved_fraction','internal_variance']:
        x=[r[metric] for r in rows]
        out[metric]=dict(Gamma=rho_stats(x,pc,rng,groups),minimum_sampled=rho_stats(x,allpc,rng,groups))
    out['within_protocol']={g:rho_stats([r['mean_excess'] for r in rows if r['protocol']==g],
                                      [r['mechanics']['Gamma'] for r in rows if r['protocol']==g],rng,B=3000) for g in sorted(set(groups))}
    x=np.array([r['mean_excess'] for r in rows]);mv=np.array([r['moved_fraction'] for r in rows]);proto=np.array(groups)
    point=partial_rho(x,pc,mv,proto);boot=[];advantage=[];blocks=[np.flatnonzero(proto==g) for g in sorted(set(groups))]
    for _ in range(5000):
        ix=np.concatenate([rng.choice(a,len(a),replace=True) for a in blocks])
        boot.append(partial_rho(x[ix],pc[ix],mv[ix],proto[ix]))
        advantage.append(spearmanr(mv[ix],pc[ix]).statistic-spearmanr(x[ix],pc[ix]).statistic)
    out['partial_rank_controlling_moved_and_protocol']=dict(rho=point,bootstrap_CI95=np.quantile(boot,[.025,.975]).tolist())
    out['window_vs_moved_rank_advantage']=dict(
        value=float(spearmanr(mv,pc).statistic-spearmanr(x,pc).statistic),
        bootstrap_CI95=np.quantile(advantage,[.025,.975]).tolist(),
        definition='rho(moved,pressure) minus rho(excess,pressure); positive means stronger negative association for window excess.')
    # Preserve severity and protocol while removing within-bin association.
    strata={}
    for i,r in enumerate(rows):strata.setdefault((r['protocol'],r['severity']),[]).append(i)
    observed=spearmanr(x,pc).statistic;permuted=[]
    for _ in range(10000):
        yp=pc.copy()
        for ids in strata.values():yp[ids]=rng.permutation(yp[ids])
        permuted.append(spearmanr(x,yp).statistic)
    out['severity_stratified_permutation']=dict(observed=float(observed),
        lower_tail_p=float((1+np.sum(np.array(permuted)<=observed))/(len(permuted)+1)),
        null_quantiles=np.quantile(permuted,[.025,.5,.975]).tolist(),
        interpretation='Tests additional within-severity association; pooled trend survives even under this null.')
    return out

def protocol_conversion():
    rows=read('cusp_base.json')['rows']+read('cusp_large.json')['rows'];out=[]
    for r in rows:
        if r['p'] not in [5,29]:continue
        s=r['protocols']['strain'];p=r['protocols']['stress']
        out.append(dict(family=r['family'],p=r['p'],q=r['q'],amplitude=r['amplitude'],
             imposed_strain_threshold=s['critical']['normalized'],pressure_per_unit_strain=s['preload']['pressure_per_unit'],
             mean_pressure_at_strain_failure=s['critical']['normalized']*s['preload']['pressure_per_unit'],
             hydrostatic_pressure_threshold=p['critical']['normalized'],stress_under_imposed_strain=s['preload']['stress']))
    return out

def label(obj,status,source):
    if isinstance(obj,bool) or obj is None or isinstance(obj,str):return obj
    if isinstance(obj,(int,float)):
        if not np.isfinite(obj):return dict(value=None,classification=status,source_script=source,note='Undefined/nonfinite statistic')
        return dict(value=obj,classification=status,source_script=source)
    if isinstance(obj,list):return [label(v,status,source) for v in obj]
    source=obj.get('source_script',source);status=obj.get('classification',status)
    return {k:label(v,status,source) for k,v in obj.items()}

if __name__=='__main__':
    start=time.process_time();summary=dict(classification='P',source_script='ab_analyze.py')
    summary['pell']=pell_analysis(read('ab_pell_raw.json')['rows'])
    summary['disorder']=disorder_analysis(read('ab_disorder_raw.json'))
    summary['window']=window_analysis(read('ab_window_raw.json'))
    summary['protocol_conversion']=protocol_conversion()
    if (ROOT/'ab_small_mesh_raw.json').exists():
        mesh=read('ab_small_mesh_raw.json');ref={(r['seed'],r['k']):r for r in read('ab_disorder_raw.json')['rows']}
        out=[]
        for k in [1,10]:
            rr=[r for r in mesh['rows'] if r['k']==k]
            change=[r['relative_drop']-ref[r['seed'],k]['relative_drop'] for r in rr]
            out.append(dict(k=k,n=len(rr),mean_drop_sub8=float(np.mean([r['relative_drop'] for r in rr])),
                mean_paired_sub8_minus_sub4=float(np.mean(change)),max_abs_paired_change=float(np.max(np.abs(change)))))
        summary['small_disorder_mesh']=out
    one=read('ab_single_flip_raw.json');arr=np.array([r['delta'] for r in one['all_legal']])
    summary['single_flip']=dict(mesh=one['mesh'],all_legal_count=len(arr),raising_count=int(np.sum(arr>1e-7)),
        lowering_count=int(np.sum(arr<-1e-7)),mean_change=float(arr.mean()),range=[float(arr.min()),float(arr.max())],
        zone=[dict(state=r['state'],min_grid=float(min(a['normalized'] for a in r['grid'])),acoustic=r['acoustic'],DQ=r['DQ']) for r in one['zone']])
    summary['cpu_s']=time.process_time()-start;save('ab_analysis.json',summary)
    files=['ab_pell_raw.json','ab_disorder_raw.json','ab_window_raw.json','ab_single_flip_raw.json','ab_analysis.json']
    if (ROOT/'ab_reference_raw.json').exists():files.append('ab_reference_raw.json')
    if (ROOT/'ab_small_mesh_raw.json').exists():files.append('ab_small_mesh_raw.json')
    if (ROOT/'ab_run_audit.json').exists():files.append('ab_run_audit.json')
    datasets={};cpu=0
    for name in files:
        r=read(name);datasets[name]=label(r,'P',r.get('source_script','ab_analyze.py'));cpu+=r.get('cpu_s',0.)
    save('ab_results.json',dict(schema='Every numeric leaf has C/P classification and source_script; all empirical results here are P.',
        recorded_CPU_s=label(cpu,'P','ab_analyze.py'),
        total_with_failed_run_and_auxiliary_budget_allowances_s=label(cpu+1500.,'P','ab_analyze.py'),datasets=datasets))
    print(json.dumps(summary,indent=2))
