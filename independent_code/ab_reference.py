"""A larger independent zero-DQ angular-midpoint reference, not fitted data."""
from ab_pell import run
from ab_common import save,ROOT,canonical,Frame,directional,np,json
import time

if __name__=='__main__':
    start=time.process_time();path=ROOT/'ab_reference_raw.json'
    if path.exists():data=json.loads(path.read_text());row=data['row']
    else:
        row=run(70,99,2,True,False)
        data=dict(classification='P',source_script='ab_reference.py',row=row,cpu_s=0.,
            note='47,321-vertex angular midpoint. A finite approximant reference, not an exact irrational QC value.')
    if 'zero_phi22_5' not in row:
        f=Frame(canonical(70,99),sub=2);f.set_protocol('stress')
        row['zero_phi22_5']=directional(f,np.pi/8)
        row['midpoint_minus_phi22_5']=row['zero_c0']-row['zero_phi22_5']
    data['cpu_s']+=time.process_time()-start
    save('ab_reference_raw.json',data)
    print(json.dumps(data),flush=True)
