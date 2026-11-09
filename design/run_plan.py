"""Driver: at most 6 single-thread workers, one job. python run_plan.py PLAN KINDS(comma) [NAME-substr] [--max N]."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'penrose', 'phason_paths', 'robustness', 'heldout_test', 'isotropic_loss')]  # shared modules
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out_part1'; OUT.mkdir(parents=True, exist_ok=True)
plan = HERE / sys.argv[1]; kinds = set(sys.argv[2].split(',')); T = json.loads(plan.read_text())
sel = [i for i, t in enumerate(T) if t['kind'] in kinds and not (OUT / f'{plan.stem}_{i:05d}.json').exists()]
sel.sort(key=lambda i: (T[i].get('sigma', 0), T[i].get('i', 0)))
workers = 6
print(len(sel), 'tasks', flush=True)


def go(i):
    p = subprocess.run([sys.executable, '-X', 'utf8', str(HERE / 'design_run.py'), str(plan), str(i), str(OUT)], capture_output=True, text=True)
    if p.returncode:
        print('FAIL', i, T[i].get('name'), p.stderr[-600:], flush=True)
    else:
        print(p.stdout.strip(), flush=True)


t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(go, sel))
print('done', time.time() - t0, flush=True)
