"""Covariates for an existing raw pool (after the LLL fix)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enumerate_cells import covariates
cls = sys.argv[1]
raw = json.load(open(f"pool_{cls}_raw.json"))
pool = {i: c for i, c in enumerate(raw["sample"] + raw["pure"])}
Vmin, Vmax = raw["V_window"]
cov = covariates(cls, pool, log=True)
cov = [c for c in cov if Vmin <= c["V"] <= Vmax]
json.dump(cov, open(f"pool_{cls}.json", "w"), indent=1)
print("pool with covariates (V in window)", len(cov))
