"""Mesh and Bloch checks of the raising flip, plus all one-flip outcomes."""
from ab_common import *

if __name__=='__main__':
    start=time.process_time();base=canonical(5,7,matched=False)
    changed=base.copy();changed.mc(np.random.default_rng(11),accepted=1)
    result=dict(classification='P',source_script='ab_single_flip.py',N=len(base.n),mesh=[],zone=[],all_legal=[])
    for sub in [4,8,16,32]:
        vals=[]
        for name,t in [('base',base),('one_flip',changed)]:
            f=Frame(t,sub=sub);f.set_protocol('stress');crit=f.critical()
            exact=directional(f,crit['theta'])
            vals.append(dict(state=name,critical=crit['normalized'],directional_tight=exact,
                             theta=crit['theta'],Gamma=f.bloch_threshold([0.,0.])['normalized']))
        result['mesh'].append(dict(sub=sub,rows=vals,delta=vals[1]['directional_tight']-vals[0]['directional_tight']))
        result['cpu_s']=time.process_time()-start;save('ab_single_flip_raw.json',result)
        print('mesh',sub,result['mesh'][-1]['delta'],flush=True)
    # A symmetric whole-zone mesh: keep both signs of each coordinate, since
    # an individual flipped configuration need not have mirror symmetry.
    for name,t in [('base',base),('one_flip',changed)]:
        f=Frame(t,sub=4);f.set_protocol('stress');values=[]
        for i,j in itertools.product(range(-4,5),repeat=2):
            k=np.array([i,j])/9
            values.append(f.bloch_threshold(k))
        crit=f.critical();theta=crit['theta'];dq=[]
        for q in [.002,.001,.0005,.00025]:
            v=f.bloch_threshold(q*(t.cell@np.array([np.cos(theta),np.sin(theta)])))
            v['DQ']=q;dq.append(v)
        result['zone'].append(dict(state=name,sub=4,grid=values,DQ=dq,acoustic=crit['normalized']))
        result['cpu_s']=time.process_time()-start;save('ab_single_flip_raw.json',result)
        print('zone',name,min(v['normalized'] for v in values),flush=True)
    class Fixed:
        def __init__(self,i):self.i=i
        def integers(self,n):return self.i
    f0=Frame(base,sub=4);ref=f0.critical()['normalized']
    for i in range(len(base.n)):
        t=base.copy()
        if not t.attempt(Fixed(i)):continue
        f=Frame(t,sub=4);pc=f.critical()['normalized']
        result['all_legal'].append(dict(vertex=i,pc=pc,delta=pc-ref))
    result['all_legal_baseline']=ref
    result['cpu_s']=time.process_time()-start;save('ab_single_flip_raw.json',result)
    print('DONE',result['cpu_s'],len(result['all_legal']),flush=True)
