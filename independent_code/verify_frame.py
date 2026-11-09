"""Meaningful numerical checks, independent full vs condensed DOF paths."""
import json,time
from pathlib import Path
import numpy as np
from geometry import Tiling,family_lifts,ROT
from frame import Frame
from beam_check import Frame as Uncondensed,geometry_check

start=time.process_time(); rows=[]
for flips in [0,200]:
    t=Tiling(family_lifts(2,3)); t.mc(np.random.default_rng(43),accepted=flips) if flips else None
    for sub in [2,4]:
        f=Frame(t,sub=sub); full=Uncondensed(t.x,t.edges(),t.cell,div=sub,EI=.03**2/4)
        for k in [[0.,0.],[.173,.319]]:
            a=f.bloch_threshold(k)['load']; b,res=full.eigen(k)
            error=abs(a-b)/b
            assert error<2e-6,(flips,sub,k,a,b,error)
            rows.append(dict(flips=flips,sub=sub,k=k,condensed=a,uncondensed=b,relative_error=error,
                             full_eigen_residual=res))
# Unbiased MC preserves exact tiling validity and is not the identity.
t=Tiling(family_lifts(5,7)); before=t.n.copy(); moves=t.mc(np.random.default_rng(81),attempts=10000)
geo=geometry_check(t.x,t.edges(),t.cell)
assert not np.array_equal(t.n,before)
# A parent 45-degree rotation must rotate the E2 phason component by 90 degrees.
L=family_lifts(2,3,'E2a'); gamma=np.array([.137,.271,.319,.413])
a=Tiling(L,offset=gamma); b=Tiling(L@ROT.T,offset=ROT@(gamma+.5)-.5)
fa=Frame(a,sub=4);fb=Frame(b,sub=4)
va=fa.critical()['normalized'];vb=fb.critical()['normalized']
assert abs(va-vb)<1e-8,(va,vb)
result=dict(classification='P',source_script='verify_frame.py',full_vs_condensed=rows,
            mc_geometry=geo,mc_moves=moves,rotation_equivalence=[va,vb],cpu_s=time.process_time()-start)
Path(__file__).with_name('frame_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
