"""Run all selected held-out cells of a class sequentially (resumable)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cell import evaluate  # noqa

if __name__ == "__main__":
    cls = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else "rect"
    sel = json.load(open(f"selection_{cls}.json"))["selected"]
    out = f"heldout_{cls}.jsonl" if section == "rect" else f"heldout_{cls}_{section}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf8"):
            done.add(json.loads(line)["name"])
    for c in sel:
        name = f"{cls}_{c['sel_index']:02d}_{c['role']}"
        if name in done:
            continue
        rec = evaluate(cls, c["L1"], c["L2"], name, 100 + c["sel_index"], offsets=(0, 1, 2),
                       bloch_check=(c["sel_index"] < 3 and section == "rect"), zone=(section == "rect"), out=None,
                       section=section)
        rec["role"] = c["role"]
        rec["covariates"] = {k: c[k] for k in ("V", "delta", "new1", "new2", "sectors", "ratio") if k in c}
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
