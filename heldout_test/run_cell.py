"""Evaluate one cell (L1, L2, class) for given offsets: K, C, c0, h4, zone checks,
optionally the long-wave moduli at p = 0 and p_ref. Appends a JSON line."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames',)]  # shared modules
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_tiling import Star, Cell, make_gamma, build_atlas, forbidden_fraction  # noqa
from cell_frame import Frame  # noqa

_ATLAS = {}


def atlas(star):
    if star.cls not in _ATLAS:
        _ATLAS[star.cls], _ = build_atlas(star, 110, make_gamma(star, 999), reduce=False)
    return _ATLAS[star.cls]


def evaluate(cls, L1, L2, name, cell_index, offsets=(0, 1, 2), bloch_check=False,
             do_patch=True, zone=True, out=None, nelem=2, section="rect"):
    import cell_frame
    cell_frame.set_section(section)
    star = Star(cls)
    dirs = np.arange(8) * np.pi / 8 if cls == "ab" else np.arange(12) * np.pi / 12
    rec = {"name": name, "cls": cls, "L1": list(map(int, L1)), "L2": list(map(int, L2)),
           "cell_index": cell_index, "section": section, "offsets": []}
    for oi in offsets:
        t0 = time.time()
        seed = 1300 + 7 * cell_index + oi
        g = make_gamma(star, seed)
        cell = Cell(star, L1, L2, g)
        o = {"offset": oi, "seed": seed, "V": cell.V, "E": cell.E, "delta": float(cell.delta),
             "sectors": {k: float(v) for k, v in cell.sectors.items()}, "area": float(cell.area)}
        if do_patch:
            ff = forbidden_fraction(cell, atlas(star))
            o["new1"], o["new2"] = ff[1], ff[2]
        fr = Frame(cell, nelem=nelem)
        o["C"] = {k: float(v) for k, v in fr.homogenize().items()}
        ex = fr.longwave_exact(dirs, p_hint=0.8)
        o.update(ex)
        o["directions_deg"] = [float(np.degrees(t)) for t in dirs]
        if zone:
            G = fr.reciprocal()
            lamG, _ = fr.critical_factor(np.zeros(2), gamma_penalty=True, lam_guess=0.4)
            o["gamma"] = lamG[:3].tolist()
            zb = []
            for kv in (G[:, 0] / 2, G[:, 1] / 2, (G[:, 0] + G[:, 1]) / 2):
                lamz, _ = fr.critical_factor(kv, lam_guess=0.4)
                zb.append(float(lamz[0]))
            o["zone_boundary"] = zb
            o["pc_min"] = float(min(o["lw_min"], lamG[0], min(zb)))
        if bloch_check and oi == offsets[0]:
            kL = 1.0 / np.sqrt(cell.area)
            rb = fr.analyse(dirs, kmags=(0.15 * kL, 0.30 * kL), zone=False)
            o["bloch_check"] = {"kmags": [0.15 * kL, 0.30 * kL], "c0_richardson": rb["c0"],
                                "lw_extrap": rb["lw_extrap"], "lw_k1": rb["lw_k1"]}
        o["seconds"] = time.time() - t0
        rec["offsets"].append(o)
        print(f"  {name} off{oi} V={cell.V} delta={cell.delta:.5f} new2={o.get('new2', -1):.4f} "
              f"K={o['C']['K']:.6f} c0={o['c0']:.6f} pcmin={o.get('pc_min', 0):.5f} t={o['seconds']:.0f}s",
              flush=True)
    if out:
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    evaluate(**spec)
