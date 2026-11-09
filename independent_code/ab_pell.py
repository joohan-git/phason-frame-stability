"""Match main B1 data and separate mesh/DQ/fitting definitions. All values P."""
from ab_common import *
import argparse

def run(p,q,sub=2,matched=True,rich=True):
    start=time.process_time();t=canonical(p,q,matched);f=Frame(t,sub=sub);f.set_protocol('stress')
    angles=[0.,np.pi/4];zero=[directional(f,a) for a in angles]
    row=dict(p=p,q=q,N=len(t.n),sub=sub,matched_offset=matched,b=float(t.S[0,0]),
             zero_phi=zero,zero_c0=float(np.mean(zero)),zero_h4=float((zero[0]-zero[1])/2),
             zero_min=float(min(zero)),critical=f.critical()['normalized'])
    if rich:
        tab={}
        full=None;residuals={}
        if len(t.n)>2000:
            # The no-pivot complex inertia factorization can become unstable
            # at the old far-above-onset bracket in a large cell. Use the
            # independently validated r1 full FE pencil with positive K0.
            from beam_check import Frame as FullFrame
            full=FullFrame(t.x,t.edges(),t.cell,div=sub,EI=f.EI)
        for dq in [.002,.001,.0005]:
            vals=[];res=[]
            for a in angles:
                k=dq*(t.cell@np.array([np.cos(a),np.sin(a)]))
                if full is not None:
                    pressure,eigen_residual=full.eigen(k);vals.append(pressure/f.lambdaE);res.append(eigen_residual)
                else:
                    vals.append(f.bloch_threshold(k,upper=1.04*row['critical']*f.lambdaE)['normalized'])
            tab[str(dq)]=vals
            if res:residuals[str(dq)]=res
            print('DQ',p,q,dq,vals,flush=True)
        for label,big,small in [('rich',.002,.001),('fine_rich',.001,.0005)]:
            vals=(4*np.array(tab[str(small)])-np.array(tab[str(big)]))/3
            row[label+'_phi']=vals.tolist();row[label+'_min']=float(vals.min())
            row[label+'_c0']=float(vals.mean());row[label+'_h4']=float((vals[0]-vals[1])/2)
        row['DQ_table']=tab
        row['DQ_solver']='full uncondensed generalized eigenproblem' if full is not None else 'condensed inertia'
        row['eigen_residuals']=residuals
    row['cpu_s']=time.process_time()-start
    print(json.dumps({k:row[k] for k in ['p','q','N','sub','matched_offset','zero_min','cpu_s']}),flush=True)
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-budget',type=float,default=2400);args=ap.parse_args()
    start=time.process_time();path=ROOT/'ab_pell_raw.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    rows=previous.get('rows',[]);old_cpu=previous.get('cpu_s',0.)
    # The published fit is Pell-plus: it also includes non-Pell 4/5 and the
    # rotated 7/10 representative of 5/7.
    cases=[(p,q,2,True,True) for p,q in [(1,2),(2,3),(4,5),(7,10),(5,7),(12,17),(29,41)]]
    cases += [(p,q,sub,True,False) for sub in [4,8] for p,q in [(1,2),(2,3),(4,5),(7,10),(12,17),(29,41)]]
    cases += [(p,q,4,False,False) for p,q in [(2,3),(5,7),(12,17),(29,41)]]
    for case in cases:
        if any((r['p'],r['q'],r['sub'],r['matched_offset'])==case[:4] for r in rows):continue
        if time.process_time()-start>args.cpu_budget:break
        rows.append(run(*case));save('ab_pell_raw.json',dict(classification='P',source_script='ab_pell.py',
            rows=rows,cpu_s=old_cpu+time.process_time()-start,requested=len(cases),completed=len(rows),
            failed_attempt_note='First large-cell complex inertia bracket failed its imaginary-pivot check. No value retained. Replaced by own full-FE positive-K generalized eigensolve. Failed attempt CPU is accounted separately.'))
