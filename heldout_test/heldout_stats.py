"""Held-out test of the isotropic loss: join the main-code thresholds (main_code_out/, three window offsets per cell)
with the cell covariates of the independent implementation (heldout_*_circ.jsonl) and rank-correlate the loss
with the forbidden-patch fraction and with the phason amplitude. Writes heldout_stats.json.

    python heldout_stats.py
"""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import collections, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
LIMIT = {'ab': 0.79984, 'pen': 0.8106}   # ordered references at two elements per edge (Fig. 3)


def main():
    runs = collections.defaultdict(list)
    for f in sorted((HERE / 'main_code_out').glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        runs[(r['cls'], r['name'])].append(r)
    rows, stats = [], {}
    for cls in ('ab', 'pen'):
        cells = [json.loads(l) for l in (HERE / f'heldout_{cls}_circ.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        for c in cells:
            o, v = c['offsets'], runs[(cls, c['name'])]
            c0 = np.array([x['c0'] for x in v])
            rows.append(dict(cls=cls, name=c['name'], V=v[0]['V'], delta=float(np.mean([x['delta'] for x in o])),
                             new2=float(np.mean([x['new2'] for x in o])), K=float(np.mean([x['C']['K'] for x in o])),
                             c0=float(c0.mean()), c0_sd=float(c0.std(ddof=1)), deficit=float(LIMIT[cls] - c0.mean()),
                             pure_B1=c.get('role') == 'pureB1'))
        rr = [r for r in rows if r['cls'] == cls]
        d = [r['deficit'] for r in rr]
        rd, pd = spearmanr(d, [r['delta'] for r in rr])
        rn, pn = spearmanr(d, [r['new2'] for r in rr])
        stats[cls] = dict(n=len(rr), rho_delta=float(rd), rho_new2=float(rn), p_delta=float(pd), p_new2=float(pn))
    out = dict(source='heldout_test (main codes), covariates heldout_test circular section', stats=stats, rows=rows)
    (HERE / 'heldout_stats.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
