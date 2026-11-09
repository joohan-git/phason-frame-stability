"""Find the minimal-delta compact Penrose cells near V = 1042, 2728, 7142 (the
Fibonacci-type sequence) with my own enumeration, then evaluate them."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa
from enumerate_cells import enumerate_pool  # noqa
from cell_tiling import Star, Cell, make_gamma  # noqa
from run_cell import evaluate  # noqa

if __name__ == "__main__":
    windows = [(1020, 1065), (2690, 2770), (7080, 7220)]
    star = Star("pen")
    found = []
    for (a, b) in windows:
        t = time.time()
        sample, pure, counts, _ = enumerate_pool("pen", a, b, 1e-5, 0.01, per_bin=400, nbins=1)
        cells = sorted(sample.values(), key=lambda c: c["delta"])
        print(f"window {a}-{b}: {sum(counts)} cells with delta<=0.01; best:", flush=True)
        for c in cells[:3]:
            cell = Cell(star, c["L1"], c["L2"], make_gamma(star, 5))
            print("   V", cell.V, "delta", round(c["delta"], 6), "sectors", {k: round(v, 3) for k, v in c["sectors"].items()}, c["L1"], c["L2"], flush=True)
        best = cells[0]
        cell = Cell(star, best["L1"], best["L2"], make_gamma(star, 5))
        found.append({"L1": best["L1"], "L2": best["L2"], "V": cell.V, "delta": best["delta"]})
        print("  time", round(time.time() - t, 1), flush=True)
    json.dump(found, open("penrose_references_found.json", "w"), indent=1)
    if os.path.exists("refs_pen.jsonl"):
        os.remove("refs_pen.jsonl")
    for i, f in enumerate(found):
        evaluate("pen", f["L1"], f["L2"], f"ref_pen_V{f['V']}", 9101 + i, offsets=(0, 1, 2), bloch_check=(i == 0),
                 zone=(i == 0), out="refs_pen.jsonl")
