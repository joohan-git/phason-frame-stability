"""Apply the registered selection rule to a covariate pool (registered plan, section 2)."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import sys

import numpy as np

DMIN, DMAX, NBINS = 0.004, 0.065, 6
EDGES = np.geomspace(DMIN, DMAX, NBINS + 1)


def select(cls, pool, excluded_V=(), stored=()):
    # exclusion by (V, delta) for Penrose
    def excluded(c):
        if c["V"] in excluded_V:
            return True
        for s in stored:
            if s["V"] == c["V"] and abs(s["delta"] - c["delta"]) < 1e-6:
                return True
        return False

    pool = [c for c in pool if not excluded(c)]
    pure = [c for c in pool if cls == "ab" and c["sectors"].get("B1", 0) > 1 - 1e-6]
    near = [c for c in pool if cls == "ab" and 0.99 < c["sectors"].get("B1", 0) <= 1 - 1e-6]
    general = [c for c in pool if c not in pure and c not in near]
    chosen = []
    bins = rule_ii(general, chosen, "bin")
    if near:
        rule_ii(near, chosen, "nearB1", fill=False)
    for c in pure:
        c = dict(c)
        c["role"] = "pureB1"
        chosen.append(c)
    return chosen, [len(b) for b in bins], len(pure), len(near)


def rule_ii(general, chosen, prefix, fill=True):
    bins = [[] for _ in range(NBINS)]
    for c in general:
        b = min(int(np.searchsorted(EDGES, c["delta"], side="right") - 1), NBINS - 1)
        c["bin"] = b
        c["ratio"] = c["new2"] / c["delta"]
        bins[b].append(c)
    n_before = len(chosen)
    for b, cells in enumerate(bins):
        if not cells:
            continue
        cells = sorted(cells, key=lambda c: (c["ratio"], abs(c["V"] - 1050)))
        picks = []
        if len(cells) <= 3:
            picks = cells
        else:
            lo = cells[0]
            hi = sorted(cells, key=lambda c: (-c["ratio"], abs(c["V"] - 1050)))[0]
            med_r = np.median([c["ratio"] for c in cells])
            med = sorted(cells, key=lambda c: (abs(c["ratio"] - med_r), abs(c["V"] - 1050)))
            med = [c for c in med if c is not lo and c is not hi][0]
            picks = [lo, hi, med]
        for role, c in zip(["min", "max", "med"], picks):
            c = dict(c)
            c["role"] = f"{prefix}{b}_{role}"
            chosen.append(c)
    # fill to 18 from the most populated bins by next most extreme ratio
    target = 18
    if fill and len(chosen) - n_before < target:
        chosen_keys = {(tuple(c["L1"]), tuple(c["L2"])) for c in chosen}
        order = sorted(range(NBINS), key=lambda b: -len(bins[b]))
        extra = []
        for b in order:
            rest = [c for c in bins[b] if (tuple(c["L1"]), tuple(c["L2"])) not in chosen_keys]
            rest = sorted(rest, key=lambda c: -abs(c["ratio"] - np.median([x["ratio"] for x in bins[b]])))
            for c in rest:
                c = dict(c)
                c["role"] = f"{prefix}{b}_fill"
                extra.append(c)
        chosen += extra[: target - (len(chosen) - n_before)]
    return bins


if __name__ == "__main__":
    cls = sys.argv[1]
    pool = json.load(open(f"pool_{cls}.json"))
    t1 = json.load(open("../isotropic_loss/isotropic_results.json"))
    stored = t1["fixedV"][cls]["cells"]
    exclV = {c["V"] for c in stored} if cls == "pen" else set()
    chosen, nb, npure, nnear = select(cls, pool, exclV, stored)
    d = np.array([c["delta"] for c in chosen])
    n2 = np.array([c["new2"] for c in chosen])
    from scipy.stats import spearmanr
    rho_all = spearmanr(d, n2).correlation
    gen = [c for c in chosen if c["role"].startswith("bin")]
    rho_gen = spearmanr([c["delta"] for c in gen], [c["new2"] for c in gen]).correlation
    print(f"{cls}: pool {len(pool)} bins {nb} exact-pure {npure} near-pure {nnear} -> selected {len(chosen)}; Spearman(delta,new2) all={rho_all:.3f} general={rho_gen:.3f}")
    for i, c in enumerate(chosen):
        c["sel_index"] = i
        print(f"  {i:2d} {c['role']:12s} V={c['V']:5d} delta={c['delta']:.5f} new2={c['new2']:.4f} new1={c['new1']:.4f} sect={ {k: round(v, 2) for k, v in c['sectors'].items() if k != 'delta2'} } L1={c['L1']} L2={c['L2']}")
    json.dump({"selected": chosen, "bin_counts": nb, "n_pure": npure, "n_near": nnear, "spearman_delta_new2_all": rho_all,
               "spearman_delta_new2_general": rho_gen}, open(f"selection_{cls}.json", "w"), indent=1)
