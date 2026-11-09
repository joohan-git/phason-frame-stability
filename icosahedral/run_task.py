"""Run one plan task: python run_task.py PLAN.json INDEX (set ICOSA_OUT for the output folder)."""
import json, sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_icosa import run
t = json.load(open(sys.argv[1]))[int(sys.argv[2])]
print('task', sys.argv[2], t, flush=True)
run(t['kind'], tuple(t['pars']), t['seed'], flips=t['flips'], sub=t['sub'])
print('TASKDONE', flush=True)
