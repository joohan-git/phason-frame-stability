"""Independent element/Euler-column checks for the minimal beam pilot."""
import json
from pathlib import Path
from beam_check import np,la,element,Frame,approximant

d=np.array([np.cos(.37),np.sin(.37)])
K,G=element(d)
translation=np.array([.2,-.3,0,.2,-.3,0])
rotation=np.array([0,0,1,-d[1],d[0],1])
checks=dict(element_translation_residual=float(la.norm(K@translation)),
            element_rotation_residual=float(la.norm(K@rotation)),columns=[])
for div in [1,2,4,8]:
    kk=np.zeros((2*(div+1),2*(div+1))); gg=kk.copy()
    ke,ge=element(np.array([1/div,0.])); ids=[1,2,4,5]
    ke=ke[np.ix_(ids,ids)]; ge=ge[np.ix_(ids,ids)]
    for i in range(div):
        ix=np.r_[2*i+np.arange(2),2*(i+1)+np.arange(2)]
        kk[np.ix_(ix,ix)]+=ke; gg[np.ix_(ix,ix)]+=ge
    keep=np.setdiff1d(np.arange(len(kk)),[0,2*div])
    mu=la.eigvalsh(gg[np.ix_(keep,keep)],kk[np.ix_(keep,keep)])[-1]
    p=1/mu
    checks['columns'].append(dict(elements=div,pressure_over_EI=float(p/1e-4),
                                  relative_error=float(p/(np.pi**2*1e-4)-1)))
assert checks['columns'][-1]['relative_error']<4e-5
assert max(checks['element_translation_residual'],checks['element_rotation_residual'])<1e-12
n,x,e,c=approximant(2,3); f=Frame(x,e,c,div=2)
k=np.array([.173,.319]); M=f.assemble(f.Ke,k); H=f.assemble(f.He,k)
checks['bloch_K_hermiticity']=float(la.norm((M-M.conj().T).toarray()))
checks['bloch_H_hermiticity']=float(la.norm((H-H.conj().T).toarray()))
assert checks['bloch_K_hermiticity']<1e-12 and checks['bloch_H_hermiticity']<1e-12
Path(__file__).with_name('beam_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
