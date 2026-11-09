"""Exact symbolic AB representation and transverse-harmonic identities (C)."""
import json
from pathlib import Path
import sympy as s

a,b,c,d=s.symbols('a b c d',real=True)
J=s.Matrix([[0,-1],[1,0]]); Z=s.diag(1,-1); X=s.Matrix([[0,1],[1,0]])
S=a*s.eye(2)+b*J+c*Z+d*X
R=s.Matrix([[1,-1],[1,1]])/s.sqrt(2)
Rp=s.Matrix([[-1,-1],[1,-1]])/s.sqrt(2)
trans=s.simplify(Rp*S*R.T)
expected=-b*s.eye(2)+a*J-c*Z-d*X
assert s.simplify(trans-expected)==s.zeros(2)
reflection=s.diag(1,-1)
assert reflection*S*reflection==a*s.eye(2)-b*J+c*Z-d*X
theta=s.symbols('theta',real=True)
n=s.Matrix([s.cos(theta),s.sin(theta)]); t=s.Matrix([-s.sin(theta),s.cos(theta)])
F=t*n.T; eps=(F+F.T)/2
e0=s.simplify(s.trace(eps)/s.sqrt(2))
e1=s.trigsimp((eps[0,0]-eps[1,1])/s.sqrt(2))
e2=s.trigsimp(s.sqrt(2)*eps[0,1])
assert e0==0
assert s.trigsimp(e1+s.sin(2*theta)/s.sqrt(2))==0
assert s.trigsimp(e2-s.cos(2*theta)/s.sqrt(2))==0
c11,c12,c22=s.symbols('C11 C12 C22',real=True)
expr=s.expand(c11*e1**2+2*c12*e1*e2+c22*e2**2)
formula=(c11+c22)/4+(c22-c11)*s.cos(4*theta)/4-c12*s.sin(4*theta)/2
assert s.trigsimp(s.expand_trig(expr-formula))==0
result=dict(classification='C',source_script='representation.py',
    physical_rotation_degrees=45,internal_rotation_degrees=135,
    phason_action=[[0,-1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,-1]],
    reflection_action=[1,-1,1,-1],phason_dimension=4,elastic_dimension=6,general_major_tangent_dimension=10,
    D8_phason='E2 + B1 + B2',D8_elastic='2 A1 + E2 + B1 + B2',
    D8_major_F_tangent='3 A1 + A2 + 2 E2 + B1 + B2',
    transverse_strain_trace=0,transverse_formula=str(formula),
    caveat='C certifies symbolic representation identities, not numerical stability thresholds.')
Path(__file__).with_name('representation_exact.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
