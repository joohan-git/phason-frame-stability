"""Merge runs/ and runs_axial_large/ (the second folder wins; duplicates verified identical)."""
import json,glob,os,numpy as np,sys
here=os.path.dirname(os.path.abspath(__file__))
def load():
    R={}
    for f in glob.glob(os.path.join(here,'runs','*.json')): R[os.path.basename(f)]=json.load(open(f))
    dup=[]
    for f in glob.glob(os.path.join(here,'runs_axial_large','*.json')):
        b=os.path.basename(f); r=json.load(open(f))
        if b in R: dup.append((b,abs(R[b]['p_c']-r['p_c']),abs(R[b]['gamma'][0]-r['gamma'][0])))
        R[b]=r; r['src']='large'
    return R,dup
