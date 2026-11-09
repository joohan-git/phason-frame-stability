import sys as _sys, pathlib as _pl; _sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[1] / _d) for _d in ('frames', 'phason_paths', 'penrose', 'isotropic_loss')]  # shared modules
if __name__ == '__main__': __import__('os').chdir(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # file names below are relative to this folder
import json, glob, collections
R=[json.load(open(f)) for f in glob.glob('out/*.json')]
print(len(R))
g=collections.defaultdict(dict)
for r in R:
    if r['group']=='mesh': g[(r['cls'],r['name'])][r['sub']]=r
print('MESH c0 and deficit')
ref={s:[r for r in R if r['group']=='mesh' and r['name']=='pell29/41' and r['sub']==s][0]['c0'] for s in (2,4)}
print('pell29/41 c0',ref)
for k,v in sorted(g.items()):
    print(k,{s:round(v[s]['c0'],6) for s in sorted(v)},'shift2-4 %.3f%%'%(100*(v[4]['c0']/v[2]['c0']-1)) if 4 in v else '', 'hh2 %.4f'%v[2]['pc_hh'], {s:round(abs(complex(*v[s]['h4'])),6) for s in sorted(v)})
print('AB deficit/delta at sub 2,4:')
for k,v in sorted(g.items()):
    if k[0]=='ab' and k[1].startswith('cell') and 2 in v and 4 in v:
        d=v[2].get('delta'); print(k[1],d,'%.3f %.3f'%((ref[2]-v[2]['c0'])/d,(ref[4]-v[4]['c0'])/d))
print('SLENDER')
s=collections.defaultdict(dict)
for r in R:
    if r['group']=='slender': s[r['r']].setdefault(r['name'],[]).append(r)
for rr in sorted(s):
    print('--r',rr,'pell',round(s[rr]['pell29/41'][0]['c0'],5))
    for n,v in sorted(s[rr].items()):
        h=[complex(*x['h4']) for x in v]; hm=abs(sum(h)/len(h)); c=sum(x['c0'] for x in v)/len(v)
        d=v[0].get('b',0) or 0; a=v[0].get('a',0) or 0
        pm=min(x['pc_lw_min'] for x in v); g_=min(x['pc_gamma'] for x in v); hh=min(x['pc_hh'] for x in v)
        print(f"{n:14s} n{len(v)} c0 {c:.5f} |h4avg| {hm:.3e} |h4|/d {hm/max(a,d,1e-9):.3f} lw {pm:.4f} G {g_:.4f} hh {hh:.4f}")
