"""Quick test: recompute a few numbers of the article on one CPU and compare them with the stored results.

    python smoke_test.py

Writes nothing into the repository. Exits with a nonzero code if a check fails. Takes a few minutes.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tempfile
os.environ['NUMBA_CACHE_DIR'] = tempfile.mkdtemp()   # keep compiled caches out of the repository
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
for d in ('frames', 'phason_paths', 'penrose', 'heldout_test', 'objectivity', 'disorder', 'nonlinear'):
    sys.path.insert(0, str(ROOT / d))
FAILED = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got:.10g} (stored {want:.10g})", flush=True)
    if not ok:
        FAILED.append(label)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def stored():
    from scipy.stats import spearmanr
    st = load('heldout_test/heldout_stats.json')
    for cls in ('ab', 'pen'):
        rows = [r for r in st['rows'] if r['cls'] == cls]
        d = [r['deficit'] for r in rows]
        check(f'held-out {cls}: rank correlation of loss with patch fraction', spearmanr(d, [r['new2'] for r in rows])[0],
              st['stats'][cls]['rho_new2'], 1e-12)
        check(f'held-out {cls}: rank correlation of loss with amplitude', spearmanr(d, [r['delta'] for r in rows])[0],
              st['stats'][cls]['rho_delta'], 1e-12)


def selection_rule_ab():
    import ab_families as F
    rows = {(r['family'], r['p'], r['q']): r for r in load('phason_paths/ab_families.json')['rows']}
    h4 = {}
    for fam in ('B1', 'E2a'):
        r = F.run(fam, 5, 7)
        ref = rows[(fam, 5, 7)]
        check(f'AB 5/7 {fam}: minimum directional pressure', r['pc_min_sampled'], ref['pc_min_sampled'], 1e-9)
        check(f'AB 5/7 {fam}: fourfold splitting |h4|', r['h4_abs'], ref['h4_abs'], 1e-7)
        h4[fam] = r['h4_abs']
    print(f'     B1/E2a fourfold splitting ratio at 5/7: {h4["B1"] / h4["E2a"]:.1f}', flush=True)


def selection_rule_penrose():
    import penrose_cusp as Pc
    ref = next(r for r in load('penrose/penrose_cusp_aline.json')['rows'] if r['name'] == 'a_3')
    r = Pc.evaluate('a_3', *Pc.A[3])
    check('Penrose a-line cell (V = %d): minimum directional pressure' % r['V'], r['pc_lw_min_sampled'], ref['pc_lw_min_sampled'], 1e-9)
    check('Penrose a-line cell: fourfold splitting |h4|', r['abs_h4'], ref['abs_h4'], 1e-7)


def census():
    import run_cell
    cells = [json.loads(l) for l in (ROOT / 'heldout_test/heldout_ab_circ.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    c = next(x for x in cells if x['name'] == 'ab_03_bin1_min')
    r = run_cell.evaluate('ab', c['L1'], c['L2'], c['name'], c['cell_index'], offsets=(0,), zone=False, section='circ')
    o, s = r['offsets'][0], c['offsets'][0]
    check('held-out cell ab_03: forbidden radius-two patch fraction', o['new2'], s['new2'], 1e-12)
    check('held-out cell ab_03: angular mean threshold (independent frame)', o['c0'], s['c0'], 1e-9)


def threshold():
    import ab_frame as C
    ref = load('disorder/ab_ensemble/drift_12-17_sc1_canonical_s0.json')
    L1, L2 = C.ab_approx(12, 17, 1)
    tl = C.canonical(L1, L2)
    f = C.Frame(tl['lifts'], tl['L1'], tl['L2'], sub=2)
    f.set_load('stress')
    phis, ext, _ = f.longwave(nphi=8)
    check('ordered 12/17 (V = %d): long-wave threshold' % len(tl['lifts']), float(ext.min()), ref['pc_lw_min'], 1e-8)


def nonlinear():
    import run_perfect as R
    ref = load('nonlinear/out/perfect_square_s2.json')['modes'][0]
    mesh = R.get_mesh('square', 2)
    model = R.Model(mesh)
    lam, vec, _ = model.linear_critical(macro=True, nev=6)
    cr = R.channels_extrapolated(model, lam[:1], vec[:, :1])[0]
    check('square lattice: nonlinear bifurcation pressure', cr['p'], ref['pc'], 1e-8)
    exact = (2 * 1.1655611852072114) ** 2 / math.pi ** 2   # tan x = 2x
    print(f'     exact macroscopic threshold {exact:.5f}; linear {lam[0]:.5f}', flush=True)
    br = R.branches_for(model, cr)
    for sign in ('1', '-1'):
        got, want = br[sign]['rows'], ref['branches'][sign]['rows']
        n = min(len(got), len(want))
        dp = max(abs(a['p'] - b['p']) for a, b in zip(got[:n], want[:n]))
        dr = max(abs(a['max_chord'] - b['max_chord']) for a, b in zip(got[:n], want[:n]))
        ok = n == len(want) and dp < 1e-9 and dr < 1e-9
        print(f"{'ok  ' if ok else 'FAIL'} square lattice post-buckling path, sign {sign}: {n} of {len(want)} steps, "
              f"max |dp| {dp:.1e}, max |d rotation| {dr:.1e}", flush=True)
        if not ok:
            FAILED.append(f'nonlinear path {sign}')


def intervention():
    import shutil, tempfile
    sys.path.insert(0, str(ROOT / 'design_intervention'))
    import intervention_core as I
    import ab_frame as C
    tl = I.cell('ab', (5, 7))
    s = I.Sampler('ab', tl['L1'], tl['L2'], tl['lifts'], tl['perp'], tl['lifts'])
    lp, a = C.flip_mc(tl['lifts'].copy(), tl['L1'], tl['L2'], 50 * s.V, 100003)
    a2, _, _ = s.run(50 * s.V, 100003, beta=0.0)
    ok = a == a2 and np.array_equal(lp, s.lifts)
    print(f"{'ok  ' if ok else 'FAIL'} penalized sampler at zero penalty equals the unbiased sampler ({a} accepted flips)", flush=True)
    if not ok:
        FAILED.append('sampler at zero penalty')
    s.run(50 * s.V, 7, beta=1.0)
    s.refresh()
    ok = C.check_tiling(s.lifts, tl['L1'], tl['L2']) == 0
    print(f"{'ok  ' if ok else 'FAIL'} penalized sampler (beta = 1) keeps a legal tiling", flush=True)
    if not ok:
        FAILED.append('penalized sampler legality')
    import analyse_intervention as A
    ref = load('design_intervention/intervention_results.json')
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(ROOT / 'design_intervention' / 'pairs.json', tmp)
        A.HERE, A.OUT = Path(tmp), ROOT / 'design_intervention' / 'out'
        A.main()
        new = json.loads((Path(tmp) / 'intervention_results.json').read_text(encoding='utf-8'))
    for key, cell in (('arm_A', 'ab1217'), ('arm_A', 'fib5')):
        check(f"intervention {cell}: penalized minus unbiased threshold", new[key][cell]['primary']['delta'] if 'delta' in new[key][cell]['primary'] else new[key][cell]['primary']['lam'][0],
              ref[key][cell]['primary']['delta'] if 'delta' in ref[key][cell]['primary'] else ref[key][cell]['primary']['lam'][0], 1e-12)
    same = json.dumps(new['arm_A'], sort_keys=True) == json.dumps(ref['arm_A'], sort_keys=True) and         json.dumps(new['arm_B'], sort_keys=True) == json.dumps(ref['arm_B'], sort_keys=True) and new['verdict_A'] == ref['verdict_A']
    print(f"{'ok  ' if same else 'FAIL'} frozen intervention analysis reproduces both arms and the verdicts", flush=True)
    if not same:
        FAILED.append('intervention analysis')


def main():
    t0 = time.time()
    for step in (stored, selection_rule_ab, selection_rule_penrose, census, threshold, nonlinear, intervention):
        t = time.time()
        step()
        print(f'     ({step.__name__}: {time.time() - t:.1f} s)', flush=True)
    if FAILED:
        print('FAILED:', ', '.join(FAILED))
        sys.exit(1)
    print(f'ALL CHECKS PASSED ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
