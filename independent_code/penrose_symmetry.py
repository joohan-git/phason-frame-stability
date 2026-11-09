"""Exact cyclotomic C10 checks, independently of any tiling implementation."""
import json,time
from pathlib import Path
import sympy as s

def main():
    start=time.process_time();z=s.symbols('z');poly=s.cyclotomic_poly(10,z)
    def red(x):return s.rem(s.expand(x),poly,z)
    G=s.zeros(5)
    for k in range(4):G[k+1,k]=1
    G[0,4]=-1
    def row(m):return s.Matrix([[z**(m*k) for k in range(5)]])
    checks={}
    for m in [1,3,5,7,9]:
        checks[f'left_eigenvector_weight_{m}']=all(red(v)==0 for v in row(m)*G-z**m*row(m))
    checks['G_power5_minus_I']=G**5==-s.eye(5)
    checks['G_power10_I']=G**10==s.eye(5)
    checks['a_to_h4_via_conjugation']=(-6)%10==4
    checks['v_to_h4_via_conjugation']=(-6)%10==4
    checks['b_neither_chirality_weight4']=all(w!=4 for w in [8,(-8)%10])
    checks['quadratic_b_allowed']=(2*8)%10==6
    checks['nonanalytic_first_order_b_covariant_allowed']=(-2*8)%10==4
    # Transverse symmetric strain is traceless; its quadratic energy contains
    # only spin 0 and spin 4, irrespective of the spin-2 bulk/shear couplings.
    th=s.symbols('theta',real=True);n=s.Matrix([s.cos(th),s.sin(th)]);t=s.Matrix([-s.sin(th),s.cos(th)])
    E=(t*n.T+n*t.T)/2
    checks['transverse_strain_traceless']=s.simplify(s.trace(E))==0
    p,q=s.symbols('p q',real=True)
    checks['deviator_basis_spin2']=s.simplify(E[0,0]+s.sin(2*th)/2)==0 and s.simplify(E[0,1]-s.cos(2*th)/2)==0
    weights=sorted([(w-k)%10 for w in [3,5,7] for k in [1,9]])
    assert all(checks.values()) and weights==[2,4,4,6,6,8]
    out=dict(label='C',source=Path(__file__).name,method='exact polynomial reduction modulo Phi_10(z)',
             cyclotomic=str(poly),checks=checks,hom_weights=weights,
             decomposition='E2 + E4 + E4',a_weight=6,b_weight=8,v_weight=6,h4_weight=4,
             covariant='h4 = kappa_a conjugate(a) + kappa_v conjugate(v) + higher order',
             caveats=['Complex kappas under C10 alone; reflection symmetry makes them real in compatible bases.',
                      'No invariant real linear functional on this phason representation.',
                      'Symmetry does not prohibit nonanalytic invariant norms or ensure differentiability.',
                      'h4 proportional to conjugate(b)^2/abs(b) is C10-covariant and degree one but not differentiable at b=0. Excluding it needs regularity or empirical bounds.',
                      'Allowed does not imply nonzero coupling; generic E4+E4 to E4 map has a two-real-dimensional kernel.'],
             cpu_s=time.process_time()-start)
    Path(__file__).with_name('penrose_symmetry.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
