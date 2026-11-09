"""Build refs_<cls>_summary.json: lambda_QC (linear extrapolation in delta from
the two smallest-delta reference cells), K_ref, cT_ref, g_ref (largest cell)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import sys

import numpy as np
from analyse_heldout import collect

cls = sys.argv[1]
suf = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
rows = []
for line in open(f"refs_{cls}{suf}.jsonl", encoding="utf8"):
    r = json.loads(line)
    o = r["offsets"]
    rows.append({"name": r["name"], "V": o[0]["V"], "delta": float(np.mean([x["delta"] for x in o])),
                 "c0": float(np.mean([x["c0"] for x in o])), "c0_sd": float(np.std([x["c0"] for x in o])),
                 "K": float(np.mean([x["C"]["K"] for x in o])), "cT": float(np.mean([x["cT_mean"] for x in o])),
                 "g": float(np.mean([x["g_mean"] for x in o])), "mu_mean": float(np.mean([x["C"]["mu_mean"] for x in o])),
                 "new2": float(np.mean([x.get("new2", np.nan) for x in o])), "h4": float(np.mean([np.hypot(*x["h4"]) for x in o])),
                 "pc_min": float(np.min([x.get("pc_min", np.nan) for x in o])), "gamma": o[0].get("gamma"),
                 "zone_boundary": o[0].get("zone_boundary"), "bloch_check": o[0].get("bloch_check"),
                 "lw": o[0]["lw_exact"], "n_offsets": len(o)})
rows.sort(key=lambda r: r["delta"])
a, b = rows[0], rows[1]  # two smallest delta
lam = (a["c0"] * b["delta"] - b["c0"] * a["delta"]) / (b["delta"] - a["delta"])
stored = {"ab": 0.79984, "pen": 0.8106}[cls]
eig = {c["name"]: c for c in collect(f"refs_{cls}{suf}.jsonl")}
summary = {"cls": cls, "rows": rows, "lambda_QC_mine": float(lam), "lambda_QC_stored": stored,
           "ref": {"name": a["name"], "K": a["K"], "cT": a["cT"], "g": a["g"], "mu_mean": a["mu_mean"],
                   "cT_eig": eig[a["name"]]["cT_eig"], "g_eig": eig[a["name"]]["g_eig"]},
           "deficit_over_delta_path": [(lam - r["c0"]) / r["delta"] for r in rows],
           "Kdef_over_delta_path": [(a["K"] - r["K"]) / r["delta"] for r in rows]}
json.dump(summary, open(f"refs_{cls}{suf}_summary.json", "w"), indent=1)
for r in rows:
    print(f"{r['name']:16s} V={r['V']:5d} delta={r['delta']:.6f} c0={r['c0']:.6f}+-{r['c0_sd']:.1e} K={r['K']:.6f} cT={r['cT']:.3e} g={r['g']:.3e} new2={r['new2']:.4f} h4={r['h4']:.5f} pcmin={r['pc_min']:.5f}")
    if r["bloch_check"]:
        print("    bloch Richardson c0:", round(r["bloch_check"]["c0_richardson"], 6), "k:", [round(k, 4) for k in r["bloch_check"]["kmags"]])
    if r["gamma"]:
        print("    gamma:", [round(x, 4) for x in r["gamma"]], "ZB:", [round(x, 4) for x in r["zone_boundary"]])
print("lambda_QC mine", lam, "stored", stored)
print("deficit/delta along path", summary["deficit_over_delta_path"])
print("Kdef/delta along path", summary["Kdef_over_delta_path"])
