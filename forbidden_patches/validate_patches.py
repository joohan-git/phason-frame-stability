"""Final integrity check binding frozen design, snapshots, census, and scores."""
import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('independent_code',)]  # shared modules
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from run_cells import ROOT,sha,save
from patch_atlas import census
from check_cells import load_t

def main():
    start=time.process_time();manifest=json.loads((ROOT/'cell_manifest.json').read_text())
    atlas=json.loads((ROOT/'patch_atlas.json').read_text());result=json.loads((ROOT/'forbidden_results.json').read_text())
    if (ROOT/'registered_plan.md').exists():assert sha(ROOT/'registered_plan.md')==manifest['prereg_sha256']
    for item in manifest['copied_sources']:assert sha(ROOT.parent/'independent_code'/item['file'])==item['sha256']
    records=[];ids=set()
    for spec in manifest['rows']+manifest['F4']:
        path=ROOT/'raw'/(spec['id']+'.json');r=json.loads(path.read_text())
        assert r['id'] not in ids;ids.add(r['id'])
        assert r['L']==spec['L'] and r['phase_seed']==spec['phase_seed'] and r['mc_seed']==spec['mc_seed']
        assert r['manifest_sha256']==sha(ROOT/'cell_manifest.json') and r['atlas_sha256']==sha(ROOT/'patch_atlas.json')
        t=load_t(r);assert len(t.n)==r['N']
        counted=census(t,set(atlas['symmetries'][r['symmetry']]['patches']))
        assert counted==r['census'],r['id']
        geo=r['geometry'];assert geo['vertices']-geo['edges']+geo['faces']==0 and geo['edges']==2*r['N']
        p0=.7998 if r['symmetry']=='AB' else .8106
        assert r['Delta']==1-r['mechanics']['pc']/p0 and r['Delta_iso']==1-r['mechanics']['c0']/p0
        if r['family']!='F4_Pell':assert len(r['mechanics']['finite_k'])==10
        rec=next(s for s in result['configurations'] if s['id']==r['id'])
        assert rec['phi2']==counted['phi2'] and rec['pc']==r['mechanics']['pc'] and rec['Delta']==r['Delta']
        records.append(dict(id=r['id'],label='C',raw_sha256=sha(path),snapshot_sha256=sha(ROOT/rec['snapshot']),
                            finite_census_reproduced=True))
    rr=[r for r in result['configurations'] if r['family']!='F4_Pell']
    for sym,key,minimum in [('AB','F1',40),('Penrose','F3',30)]:
        sub=[r for r in rr if r['symmetry']==sym];assert len(sub)>=minimum
        value=float(spearmanr([r['phi2'] for r in sub],[r['Delta'] for r in sub]).statistic)
        assert abs(value-result['verdicts'][key]['spearman'])<1e-14
    out=dict(label='C for finite data/hash checks; P for numerical pressure estimates',status='PASS',
             primary_configurations=len(manifest['rows']),F4_controls=len(manifest['F4']),
             preregistration_unchanged=True,copied_modules_hash_verified=True,
             sources=['validate_patches.py','cell_manifest.json','snapshots/*.npz','raw/*.json','forbidden_results.json'],
             records=records,cpu_s=time.process_time()-start)
    save(ROOT/'validation.json',out);print('PASS:',len(records),'snapshot censuses and stored values reproduced; CPU',out['cpu_s'])
if __name__=='__main__':main()
