"""Registered statistics S1-S8 of the registered plan on the held-out results."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json
import sys

import numpy as np
from scipy import stats

RNG = np.random.default_rng(13)
NPERM = 10000


def ranks(x):
    return stats.rankdata(x)


def spearman_perm(x, y, nperm=NPERM):
    rx, ry = ranks(x), ranks(y)
    r0 = np.corrcoef(rx, ry)[0, 1]
    cnt = 0
    for _ in range(nperm):
        r = np.corrcoef(rx, RNG.permutation(ry))[0, 1]
        if abs(r) >= abs(r0) - 1e-12:
            cnt += 1
    return float(r0), (cnt + 1) / (nperm + 1)


def partial_spearman_perm(x, y, controls, nperm=NPERM):
    """Partial Spearman of x and y given controls (list of arrays), with a
    permutation p-value obtained by permuting the ranks of x against fixed
    (y, controls)."""
    rx, ry = ranks(x), ranks(y)
    Z = np.column_stack([np.ones(len(x))] + [ranks(c) for c in controls])
    P = np.eye(len(x)) - Z @ np.linalg.pinv(Z)

    def pc(a, b):
        ra, rb = P @ a, P @ b
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    r0 = pc(rx, ry)
    cnt = 0
    for _ in range(nperm):
        if abs(pc(RNG.permutation(rx), ry)) >= abs(r0) - 1e-12:
            cnt += 1
    return r0, (cnt + 1) / (nperm + 1)


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    ss_res = res @ res
    ss_tot = ((y - y.mean()) ** 2).sum()
    return beta, 1 - ss_res / ss_tot, ss_res, X.shape[1]


def f_test_nested(y, X_small, X_big):
    _, r2s, sss, ps = ols(y, X_small)
    _, r2b, ssb, pb = ols(y, X_big)
    n = len(y)
    df1, df2 = pb - ps, n - pb
    F = ((sss - ssb) / df1) / (ssb / df2)
    p = 1 - stats.f.cdf(F, df1, df2)
    return float(r2b - r2s), float(F), float(p), float(r2s), float(r2b)


def verdict_S3(rho, p):
    if rho <= -0.5 and p < 0.01:
        return "K carries information beyond amplitude"
    if abs(rho) < 0.3 or p > 0.05:
        return "no information beyond amplitude"
    return "inconclusive"


def analyse(cls, cells, lam_qc, refs, lam_qc_stored):
    """cells: list of dicts with per-cell means: delta, delta2, new2, K, c0, cT, g, mu_mean, role."""
    out = {"cls": cls, "n": len(cells), "lambda_QC_mine": lam_qc, "lambda_QC_stored": lam_qc_stored}
    d = np.array([c["delta"] for c in cells])
    d2 = np.array([c["delta2"] for c in cells])
    n2 = np.array([c["new2"] for c in cells])
    K = np.array([c["K"] for c in cells])
    c0 = np.array([c["c0"] for c in cells])
    cT = np.array([c["cT"] for c in cells])
    g = np.array([c["g"] for c in cells])
    mu = np.array([c["mu_mean"] for c in cells])
    deficit = lam_qc - c0
    Kdef = refs["K"] - K
    out["deficit_range"] = [float(deficit.min()), float(deficit.max())]
    out["K_range"] = [float(K.min()), float(K.max())]
    out["spearman_delta_new2"] = spearman_perm(d, n2)
    # S1, S2
    out["S1_rho_K_new2"] = spearman_perm(K, n2)
    out["S2_rho_deficit_K"] = spearman_perm(deficit, K)
    out["rho_deficit_delta"] = spearman_perm(deficit, d)
    out["rho_deficit_new2"] = spearman_perm(deficit, n2)
    out["rho_K_delta"] = spearman_perm(K, d)
    # S3-S5
    r, p = partial_spearman_perm(deficit, K, [d])
    out["S3_partial_deficit_K_given_delta"] = [r, p, verdict_S3(r, p)]
    r, p = partial_spearman_perm(deficit, K, [d, n2])
    out["S4_partial_deficit_K_given_delta_new2"] = [r, p]
    out["S5_partial_deficit_new2_given_delta"] = list(partial_spearman_perm(deficit, n2, [d]))
    out["S5_partial_deficit_delta_given_K"] = list(partial_spearman_perm(deficit, d, [K]))
    out["S5_partial_deficit_delta_given_new2"] = list(partial_spearman_perm(deficit, d, [n2]))
    out["S5_partial_K_new2_given_delta"] = list(partial_spearman_perm(K, n2, [d]))
    if cls == "pen":
        r, p = partial_spearman_perm(deficit, K, [d2])
        out["S3alt_partial_deficit_K_given_delta2"] = [r, p, verdict_S3(r, p)]
    # S7 nested models
    out["S7"] = {}
    out["S7"]["dR2_K_given_delta"] = f_test_nested(deficit, [d], [d, Kdef])
    out["S7"]["dR2_new2_given_delta"] = f_test_nested(deficit, [d], [d, n2])
    out["S7"]["dR2_K_given_delta_new2"] = f_test_nested(deficit, [d, n2], [d, n2, Kdef])
    out["S7"]["dR2_new2_given_delta_K"] = f_test_nested(deficit, [d, Kdef], [d, n2, Kdef])
    out["S7"]["dR2_delta_given_K"] = f_test_nested(deficit, [Kdef], [d, Kdef])
    dr2, F, p, *_ = out["S7"]["dR2_K_given_delta"]
    out["S7"]["verdict"] = "K adds information (dR2>=0.15, p<0.01)" if (dr2 >= 0.15 and p < 0.01) else "K does not add information by the registered criterion"
    if cls == "pen":
        out["S7"]["dR2_K_given_delta2"] = f_test_nested(deficit, [d2], [d2, Kdef])
    # S6 pure-B1 (AB)
    roles = [c["role"] for c in cells]
    if cls == "ab":
        pure = np.array([r == "pureB1" for r in roles])
        if pure.sum() >= 3:
            sp = np.polyfit(d[pure], Kdef[pure], 1)[0]
            sg = np.polyfit(d[~pure], Kdef[~pure], 1)[0]
            out["S6"] = {"n_pure": int(pure.sum()), "new2_pure": n2[pure].tolist(),
                         "rho_Kdef_delta_pure": spearman_perm(Kdef[pure], d[pure], 2000),
                         "slope_Kdef_delta_pure": float(sp), "slope_Kdef_delta_general": float(sg),
                         "slope_ratio": float(sp / sg) if sg != 0 else None,
                         "Kdef_over_delta_pure": (Kdef[pure] / d[pure]).tolist(),
                         "Kdef_over_delta_general_median": float(np.median(Kdef[~pure] / d[~pure])),
                         "deficit_over_delta_pure": (deficit[pure] / d[pure]).tolist(),
                         "deficit_over_delta_general_median": float(np.median(deficit[~pure] / d[~pure]))}
            # S1-S3 on the general (rule ii) set alone
            gsel = ~pure
            out["general_only"] = {
                "n": int(gsel.sum()),
                "S1_rho_K_new2": spearman_perm(K[gsel], n2[gsel]),
                "S2_rho_deficit_K": spearman_perm(deficit[gsel], K[gsel]),
                "S3_partial_deficit_K_given_delta": list(partial_spearman_perm(deficit[gsel], K[gsel], [d[gsel]])),
                "spearman_delta_new2": spearman_perm(d[gsel], n2[gsel]),
            }
    out["table"] = [{"name": c["name"], "role": c["role"], "V": c["V"], "delta": c["delta"], "new2": c["new2"],
                     "K": c["K"], "Kdef": float(refs["K"] - c["K"]), "c0": c["c0"], "deficit": float(lam_qc - c["c0"]),
                     "cT": c["cT"], "g": c["g"], "pc_min": c.get("pc_min"), "c0_sd": c.get("c0_sd"),
                     "gamma_min": c.get("gamma_min"), "h4": c.get("h4"), "sectors": c.get("sectors")} for c in cells]
    # S8 mechanism numerics
    g_ref, cT_ref = refs["g"], refs["cT"]
    pred = (cT - cT_ref) / g_ref - cT_ref * (g - g_ref) / g_ref**2  # first-order shift of lambda (negative = deficit)
    pred_def = -pred
    lam_lin_def = lam_qc - cT / g
    r_pred = stats.pearsonr(deficit, pred_def)
    slope = np.polyfit(pred_def, deficit, 1)[0]
    term_mat = -(cT - cT_ref) / g_ref  # material (shear-stiffness) part of the deficit
    term_geo = cT_ref * (g - g_ref) / g_ref**2  # geometric part
    out["S8"] = {
        "pearson_deficit_vs_firstorder": [float(r_pred[0]), float(r_pred[1])],
        "slope_deficit_vs_firstorder": float(slope),
        "pearson_deficit_vs_linear_cT_over_g": [float(x) for x in stats.pearsonr(deficit, lam_lin_def)],
        "material_term_share_median": float(np.median(term_mat / (term_mat + term_geo))),
        "rho_deficit_cT": spearman_perm(deficit, cT),
        "rho_deficit_g": spearman_perm(deficit, g),
        "rho_K_cT": spearman_perm(K, cT),
        "rho_K_mu_static": spearman_perm(K, mu),
        "partial_deficit_K_given_cT_g": list(partial_spearman_perm(deficit, K, [cT, g])),
        "partial_deficit_cT_given_K": list(partial_spearman_perm(deficit, cT, [K])),
        "partial_deficit_K_given_cT": list(partial_spearman_perm(deficit, K, [cT])),
    }
    if "cT_eig" in cells[0] and "cT_eig" in refs:
        cTe = np.array([c["cT_eig"] for c in cells])
        ge = np.array([c["g_eig"] for c in cells])
        g_ref_e, cT_ref_e = refs["g_eig"], refs["cT_eig"]
        pred_e = -((cTe - cT_ref_e) / g_ref_e - cT_ref_e * (ge - g_ref_e) / g_ref_e**2)
        mat_e = -(cTe - cT_ref_e) / g_ref_e
        geo_e = cT_ref_e * (ge - g_ref_e) / g_ref_e**2
        sl = np.polyfit(pred_e, deficit, 1)[0]
        out["S8b_eigen_based"] = {
            "note": "post hoc variant: c_T and g from the lowest eigenvalue of Q instead of m.Q.m (E2 trace-traceless coupling makes m.Q.m overshoot at large E2 amplitude)",
            "pearson_deficit_vs_firstorder": [float(x) for x in stats.pearsonr(deficit, pred_e)],
            "slope_deficit_vs_firstorder": float(sl),
            "max_abs_rel_error_firstorder": float(np.max(np.abs(pred_e - deficit) / np.abs(deficit))),
            "material_term_share_median": float(np.median(mat_e / (mat_e + geo_e))),
            "material_term_median": float(np.median(mat_e)), "geometric_term_median": float(np.median(geo_e)),
            "rho_deficit_cT": spearman_perm(deficit, cTe), "rho_deficit_g": spearman_perm(deficit, ge),
            "rho_K_g": spearman_perm(K, ge), "rho_K_cT": spearman_perm(K, cTe), "rho_new2_g": spearman_perm(n2, ge),
            "partial_deficit_K_given_cT_g": list(partial_spearman_perm(deficit, K, [cTe, ge])),
            "partial_deficit_g_given_K": list(partial_spearman_perm(deficit, ge, [K])),
            "partial_deficit_K_given_g": list(partial_spearman_perm(deficit, K, [ge])),
            "rel_change_ranges": {"cT": [float(((cTe - cT_ref_e) / cT_ref_e).min()), float(((cTe - cT_ref_e) / cT_ref_e).max())],
                                   "g": [float(((ge - g_ref_e) / g_ref_e).min()), float(((ge - g_ref_e) / g_ref_e).max())],
                                   "K": [float(((K - refs["K"]) / refs["K"]).min()), float(((K - refs["K"]) / refs["K"]).max())],
                                   "lambda": [float((-deficit / lam_qc).min()), float((-deficit / lam_qc).max())]},
        }
        for c, a_, b_ in zip(out["table"], cTe, ge):
            c["cT_eig"], c["g_eig"] = float(a_), float(b_)
    r8 = out["S8"]["pearson_deficit_vs_firstorder"][0]
    out["S8"]["firstorder_adequate"] = bool(r8 >= 0.9 and 0.7 <= slope <= 1.3)
    rp = out["S8"]["partial_deficit_K_given_cT_g"][0]
    out["S8"]["K_is_proxy"] = bool(abs(rp) < 0.3)
    if cls == "pen":
        sub = np.array([c["sectors"].get("E4diag", 0) < 0.01 for c in cells])
        if sub.sum() >= 5:
            ds, n2s, Ks, defs = d[sub], n2[sub], K[sub], deficit[sub]
            out["secondary_no_diagonal_subset"] = {
                "n": int(sub.sum()), "names": [c["name"] for c, m in zip(cells, sub) if m],
                "spearman_delta_new2": spearman_perm(ds, n2s),
                "S1_rho_K_new2": spearman_perm(Ks, n2s), "S2_rho_deficit_K": spearman_perm(defs, Ks),
                "rho_deficit_delta": spearman_perm(defs, ds), "rho_deficit_new2": spearman_perm(defs, n2s),
                "S3_partial_deficit_K_given_delta": list(partial_spearman_perm(defs, Ks, [ds])),
                "S4_partial_deficit_K_given_delta_new2": list(partial_spearman_perm(defs, Ks, [ds, n2s])),
                "S7_dR2_K_given_delta": f_test_nested(defs, [ds], [ds, refs["K"] - Ks]),
                "S7_dR2_new2_given_delta": f_test_nested(defs, [ds], [ds, n2s]),
            }
    # falsifier / replication
    s1, s2 = out["S1_rho_K_new2"][0], out["S2_rho_deficit_K"][0]
    out["claim_replicated"] = bool(s1 <= -0.8 and s2 <= -0.8)
    out["claim_falsified"] = bool(s1 > -0.5 or s2 > -0.5)
    # sensitivity with stored lambda_QC
    deficit_s = lam_qc_stored - c0
    out["stored_lambda_sensitivity"] = {
        "S2_rho_deficit_K": spearman_perm(deficit_s, K),
        "S3_partial": list(partial_spearman_perm(deficit_s, K, [d])),
    }
    return out


def collect(jsonl):
    cells = []
    for line in open(jsonl, encoding="utf8"):
        r = json.loads(line)
        offs = r["offsets"]
        m = lambda key: float(np.mean([o[key] for o in offs]))
        c = {"name": r["name"], "role": r.get("role", ""), "V": offs[0]["V"], "delta": m("delta"),
             "delta2": float(np.mean([o["sectors"]["delta2"] for o in offs])),
             "new2": m("new2"), "new1": m("new1"), "K": float(np.mean([o["C"]["K"] for o in offs])),
             "mu_mean": float(np.mean([o["C"]["mu_mean"] for o in offs])),
             "c0": m("c0"), "c0_sd": float(np.std([o["c0"] for o in offs])), "cT": m("cT_mean"), "g": m("g_mean"),
             "pc_min": float(np.min([o.get("pc_min", np.inf) for o in offs])),
             "lw_min": m("lw_min"), "gamma_min": float(np.min([o["gamma"][0] for o in offs if "gamma" in o] or [np.nan])),
             "h4": float(np.mean([np.hypot(*o["h4"]) for o in offs])),
             "sectors": offs[0]["sectors"], "n_offsets": len(offs)}
        ce, ge = [], []
        for o in offs:
            A0, Ap = np.array(o["A0"]), np.array(o["Ap"])
            ph = o["p_hint"]
            dirs = np.radians(o["directions_deg"])
            x0, xp = [], []
            for th in dirs:
                n = np.array([np.cos(th), np.sin(th)])
                x0.append(np.linalg.eigvalsh(acoustic(A0, n))[0])
                xp.append(np.linalg.eigvalsh(acoustic(Ap, n))[0])
            x0, xp = np.array(x0), np.array(xp)
            ce.append(x0.mean())
            ge.append((-(xp - x0) / ph).mean())
        c["cT_eig"] = float(np.mean(ce))
        c["g_eig"] = float(np.mean(ge))
        cells.append(c)
    return cells


def acoustic(A, n):
    idx = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 3}
    Q = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            Q[i, j] = sum(n[k] * A[idx[(i, k)], idx[(j, l)]] * n[l] for k in range(2) for l in range(2))
    return Q


if __name__ == "__main__":
    cls = sys.argv[1]
    suf = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
    cells = collect(f"heldout_{cls}{suf}.jsonl")
    if suf:  # covariates (new1/new2) were not recomputed in the robustness run: take them from the registered run
        base = {c["name"]: c for c in collect(f"heldout_{cls}.jsonl")}
        for c in cells:
            c["new2"], c["new1"] = base[c["name"]]["new2"], base[c["name"]]["new1"]
    cfg = json.load(open(f"refs_{cls}{suf}_summary.json"))
    res = analyse(cls, cells, cfg["lambda_QC_mine"], cfg["ref"], cfg["lambda_QC_stored"])
    json.dump(res, open(f"analysis_{cls}{suf}.json", "w"), indent=1)
    for k, v in res.items():
        if k != "table":
            print(k, json.dumps(v)[:300])
