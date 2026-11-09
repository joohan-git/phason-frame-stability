"""Assemble heldout_results.json from the analysis, reference, selection and cross-check files."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import os

import numpy as np

from analyse_heldout import collect

out = {"status": "C: derivation of the first-order isotropic shift; P: all numerics",
       "prereg": "registered plan", "code": ["cell_tiling.py", "cell_frame.py", "enumerate_cells.py", "select_heldout.py",
                                            "run_cell.py", "run_heldout.py", "analyse_heldout.py"]}
t1 = json.load(open("../isotropic_loss/isotropic_results.json"))
for cls in ("ab", "pen"):
    a = json.load(open(f"analysis_{cls}.json"))
    r = json.load(open(f"refs_{cls}_summary.json"))
    s = json.load(open(f"selection_{cls}.json"))
    out[cls] = {"analysis": a, "references": r,
                "selection": {"bin_counts": s["bin_counts"], "n_pure": s.get("n_pure"), "n_near": s.get("n_near"),
                              "spearman_delta_new2_all": s["spearman_delta_new2_all"],
                              "spearman_delta_new2_general": s["spearman_delta_new2_general"],
                              "cells": [{k: c[k] for k in ("sel_index", "role", "V", "delta", "new1", "new2", "sectors", "L1", "L2")} for c in s["selected"]]}}
    raw = json.load(open(f"pool_{cls}_raw.json"))
    out[cls]["enumeration"] = {"complete_count_per_bin": raw["counts"], "V_window": raw["V_window"],
                               "fixedV_hits": raw.get("fixedV_hits"), "sample_size": len(raw["sample"]), "n_pure_exact": len(raw["pure"])}

for cls in ("ab", "pen"):
    out[cls]["robustness_circular_section"] = {"analysis": json.load(open(f"analysis_{cls}_circ.json")),
                                               "references": json.load(open(f"refs_{cls}_circ_summary.json"))}
# cross-checks
xc = {}
stored = {"ab": {c["V"]: c for c in t1["fixedV"]["ab"]["cells"]}, "pen": {c["V"]: c for c in t1["fixedV"]["pen"]["cells"]}}
stored_pell = {r["pq"]: r for r in t1["ab_sequences_c0"] if r["family"] == "B1"}
for fn, cls in (("xcheck_ab.jsonl", "ab"), ("xcheck_ab_circ.jsonl", "ab"), ("xcheck_pen.jsonl", "pen"), ("xcheck_pen_circ.jsonl", "pen")):
    if not os.path.exists(fn):
        continue
    rows = []
    for c in collect(fn):
        V = c["V"]
        row = {"name": c["name"], "V": V, "delta_mine": c["delta"], "c0_mine": c["c0"], "c0_sd_mine": c["c0_sd"],
               "new2_mine": c["new2"], "K_mine": c["K"], "h4_mine": c["h4"]}
        if V in stored[cls]:
            sc = stored[cls][V]
            row.update({"delta_stored": sc["delta"], "c0_stored": sc["c0_mean"], "c0_sd_stored": sc["c0_sd"],
                        "new2_stored": sc["new2"], "deficit_stored": sc["deficit"],
                        "c0_diff_mine_minus_stored": c["c0"] - sc["c0_mean"]})
        else:
            for pq, sp in stored_pell.items():
                if sp["V"] == V:
                    row.update({"delta_stored": sp["delta"], "c0_stored": sp["c0"], "c0_diff_mine_minus_stored": c["c0"] - sp["c0"]})
        rows.append(row)
    xc[fn] = rows
# add my own references to the Pell comparison
rows = []
for r in json.load(open("refs_ab_summary.json"))["rows"]:
    for pq, sp in stored_pell.items():
        if sp["V"] == r["V"]:
            rows.append({"name": r["name"], "V": r["V"], "c0_mine_rect": r["c0"], "c0_stored": sp["c0"], "diff": r["c0"] - sp["c0"],
                         "K_mine": r["K"]})
xc["pell_rect"] = rows
out["cross_check"] = xc
out["S8c_proxy_test"] = json.load(open("proxy_test.json"))
json.dump(out, open("heldout_results.json", "w"), indent=1)
print("written heldout_results.json")
for k, v in xc.items():
    print("==", k)
    for row in v:
        print("  ", json.dumps({kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in row.items()}))
