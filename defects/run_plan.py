"""Driver: at most 6 single-thread workers, one job. Resumable. python run_plan.py lin|nl [--max N] [group,group]."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'design', 'nonlinear', 'phason_paths')]  # shared modules
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'; OUT.mkdir(parents=True, exist_ok=True)
which = sys.argv[1]
plan = HERE / f'plan_{which}.json'; script = HERE / ('defects_run.py' if which == 'lin' else 'defects_run_nonlinear.py')
T = json.loads(plan.read_text())
sel = [i for i, t in enumerate(T) if not (OUT / f'{plan.stem}_{i:05d}.json').exists()]
if '--max' in sys.argv:
    sel = sel[:int(sys.argv[sys.argv.index('--max') + 1])]
# cheap tasks first; heavy size tasks last
sel.sort(key=lambda i: (T[i]['kind'] != 'base', T[i].get('group') == 'size', T[i].get('sub', 2), T[i].get('i', 0)))
workers = 6
print(len(sel), 'tasks', flush=True)


def go(i):
    p = subprocess.run([sys.executable, '-X', 'utf8', str(script), str(plan), str(i), str(OUT)], capture_output=True, text=True, cwd=str(HERE))
    if p.returncode:
        print('FAIL', i, T[i].get('name'), T[i].get('level'), p.stderr[-600:], flush=True)
    else:
        print(p.stdout.strip()[-200:], flush=True)


t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(go, sel))
print('done', time.time() - t0, flush=True)
