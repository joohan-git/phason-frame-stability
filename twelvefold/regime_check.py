"""stretching- vs bending-dominated regime. Mean coordination z = 2E/V, homogenised in-plane
moduli from the stress-protocol C3 (Voigt-like basis e11, e22, sqrt2 e12), and the macroscopic shear scale
mu/(EA lambda_E) to compare with the computed threshold.  D12 (twelvefold_core) vs AB (main/ab_frame, Pell 5/7 B1 family).
-> regime_check.json"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths')]  # shared modules
import json, math, sys
from pathlib import Path
import numpy as np
import twelvefold_core as D, run_twelvefold as R
import ab_frame as C
from ab_families import family_lifts
out = dict(status='P')
L1, L2 = R.cell('A', 5, 'C4'); tl = D.tiling(L1, L2, offset=R.offset_of(1)); f = D.Frame(tl, sub=2); f.set_load()
mu = f.C3[2, 2] / 2; K2 = (f.C3[0, 0] + f.C3[0, 1]) / 2
out['D12_A5_C4'] = dict(z=2 * len(tl['edges']) / len(tl['lifts']), C3=f.C3.tolist(), mu=mu, bulk2D=K2, mu_over_lamE=mu / D.LAM_E, pc_computed=1.4645)
L = family_lifts(5, 7, 'B1'); ta = C.canonical(L[0], L[1]); fa = C.Frame(ta['lifts'], ta['L1'], ta['L2'], sub=2); fa.set_load('stress')
mua = fa.C3[2, 2] / 2; Ka = (fa.C3[0, 0] + fa.C3[0, 1]) / 2
ph, ext, _ = fa.longwave(nphi=8)
out['AB_B1_5_7'] = dict(z=4.0, C3=fa.C3.tolist(), mu=mua, bulk2D=Ka, mu_over_lamE=mua / C.LAM_E, pc_computed=float(ext.min()))
Path('regime_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
