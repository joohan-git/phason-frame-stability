"""Independent full-pencil eigenvector check of a fresh long-MC sample."""
from penrose_core import *
import scipy.sparse.linalg as sla
import scipy.linalg as la

def mode(t,k,gamma=False):
    start=time.process_time();f=FullFrame(t.x,t.edges(),t.cell,div=2,EI=.03**2/4)
    free=f.free if gamma else np.arange(f.ndof)
    K=f.assemble(f.Ke,k)[free][:,free].tocsc();H=f.assemble(f.He,k)[free][:,free].tocsc()
    if gamma:K=K.real.copy();H=H.real.copy()
    fac=sla.splu(K);inv=sla.LinearOperator(K.shape,matvec=fac.solve,dtype=K.dtype)
    val,vec=sla.eigsh(H,k=1,M=K,Minv=inv,which='LA',tol=1e-9,v0=np.random.default_rng(44501).normal(size=len(free)))
    p=1/val[0];v=vec[:,0];res=la.norm(K@v-p*(H@v))/(la.norm(K@v)+p*la.norm(H@v))
    full=np.zeros(f.ndof,dtype=v.dtype);full[free]=v;u=full[:3*len(t.n)].reshape(-1,3)[:,:2]
    if gamma:u=u-u.mean(0)
    weight=np.sum(abs(u)**2,axis=1);weight/=weight.sum()
    base=Penrose(t.L,t.gamma);w=Window(base);y=t.n@w.perp.T-w.center
    outside=np.max(y@w.normals.T+w.constants,axis=1)>1e-9
    peak=int(np.argmax(weight));d=(t.x-t.x[peak])@t.invcell;d=(d-np.round(d))@t.cell
    rr=np.linalg.norm(d,axis=1);order=np.argsort(rr);r90=rr[order[np.searchsorted(np.cumsum(weight[order]),.9)]]
    de=full[f.dof].copy();de[:,3:]*=np.exp(2j*np.pi*(f.shifts@np.asarray(k)))[:,None] if np.iscomplexobj(de) else 1.
    element_energy=np.real(np.einsum('ei,eij,ej->e',de.conj(),f.Ke,de));element_energy=np.maximum(element_energy,0.)
    beam_energy=element_energy.reshape(-1,2).sum(1);beam_energy/=beam_energy.sum()
    edge=np.array(t.edges(),dtype=int);joint_energy=np.zeros(len(t.n))
    np.add.at(joint_energy,edge[:,0],beam_energy/2);np.add.at(joint_energy,edge[:,1],beam_energy/2)
    fullu=full.reshape(-1,3)[:,:2]
    if gamma:fullu=fullu-fullu.mean(0)
    fullw=np.sum(abs(fullu)**2,axis=1);fullw/=fullw.sum()
    midpoint=(t.x[edge[:,0]]+t.x[edge[:,1]]+edge[:,2:]@t.cell)/2
    peakedge=int(np.argmax(beam_energy));dr=(midpoint-midpoint[peakedge])@t.invcell;dr=(dr-np.round(dr))@t.cell
    radii=np.linalg.norm(dr,axis=1);order2=np.argsort(radii);er90=radii[order2[np.searchsorted(np.cumsum(beam_energy[order2]),.9)]]
    return dict(k=list(k),pc=float(p/LAM),residual=float(res),participation_ratio=float(1/(len(t.n)*np.sum(weight**2))),
                participation_definition='original-joint translation amplitude, mean translation removed at Gamma',
                full_node_translation_PR=float(1/(len(fullu)*np.sum(fullw**2))),
                beam_strain_energy_PR=float(1/(len(beam_energy)*np.sum(beam_energy**2))),
                joint_assigned_strain_energy_PR=float(1/(len(t.n)*np.sum(joint_energy**2))),
                largest_beam_energy_fraction=float(beam_energy.max()),beam_energy_radius90=float(er90),
                energy_weight_outside_window=float(joint_energy[outside].sum()),
                weight_outside_window=float(weight[outside].sum()),radius_90pct=float(r90),peak_vertex=peak,cpu_s=time.process_time()-start)

def main():
    path=ROOT/'penrose_long_501.npz'
    while not path.exists():time.sleep(5)
    z=np.load(path);t=Penrose(z['L'],z['gamma'],z['n']);base=Penrose(t.L,t.gamma)
    out=dict(source=Path(__file__).name,long_run_Gamma=mode(t,[0,0],True),canonical_small_q=mode(base,[.001,0]))
    save('penrose_localization_raw.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
