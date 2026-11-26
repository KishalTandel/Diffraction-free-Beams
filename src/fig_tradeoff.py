import numpy as np, json
from scipy import special as sp
from style import *
j01=sp.jn_zeros(0,1)[0]
def core_frac_numeric(kt_w0):
    rr=np.linspace(0,8*kt_w0,400000)
    I=sp.j0(rr)**2*np.exp(-2*rr**2/kt_w0**2)
    tot=np.trapezoid(I*rr,rr); cr=rr<j01
    return np.trapezoid((I*rr)[cr],rr[cr])/tot
ktw=np.array([5,8,12,20,40,80,160,320])
fnum=np.array([core_frac_numeric(v) for v in ktw])
G=ktw/1.83041
fig,a=plt.subplots(figsize=(COL1,2.4))
Gc=np.logspace(0.3,2.8,100)
a.loglog(Gc,2.134/Gc,color="k",lw=0.8,label=r"$f\,G\simeq2.13$ (asymptotic)")
a.loglog(G,fnum,"o",color=BLUE,ms=3.2,label="numerical integration")
a.set_xlabel(r"range gain $G=z_{\max}/z_R^{\rm (Gauss)}$"); a.set_ylabel(r"central-lobe power fraction $f$")
a.legend(fontsize=6.5,loc="upper right")
plt.savefig(os.path.join(FIGDIR,"fig_tradeoff_a.pdf")); plt.close()

# light sheet (water, 488 nm)
lam_ls=488e-9; n_w=1.33; k_m=2*np.pi*n_w/lam_ls
NAr=0.40; sin_t=NAr/n_w; kt=k_m*sin_t; tan_t=sin_t/np.sqrt(1-sin_t**2)
zmax=50e-6; w0=zmax*tan_t
wG=1.91331/kt; wL=np.sqrt(lam_ls*zmax/(np.pi*n_w))
xq=np.linspace(-6e-6,6e-6,1201); yq=np.linspace(-3*w0,3*w0,6001)
Xq,Yq=np.meshgrid(xq,yq,indexing="ij")
Ib=sp.j0(kt*np.hypot(Xq,Yq))**2*np.exp(-2*(Xq**2+Yq**2)/w0**2)
S=np.trapezoid(Ib,yq,axis=1); Sn=S/S.max()
fig,a=plt.subplots(figsize=(COL1,2.6))
a.semilogy(xq*1e6,np.exp(-2*xq**2/wL**2),color=ORANGE,ls="--",label=r"Gaussian sheet, same length ($z_R=z_{\max}$)")
a.semilogy(xq*1e6,np.exp(-2*xq**2/wG**2),color=GREEN,ls=":",label=r"Gaussian sheet, same core ($z_R\approx1\,\mu$m)")
a.semilogy(xq*1e6,Sn,color=BLUE,label="scanned Bessel beam")
a.set_xlim(-6,6); a.set_ylim(1e-3,1.5); a.set_xlabel(r"$x$ ($\mu$m)"); a.set_ylabel("light-sheet profile (norm.)")
a.legend(fontsize=6,loc="upper center",bbox_to_anchor=(0.5,-0.2))
plt.savefig(os.path.join(FIGDIR,"fig_tradeoff_b.pdf")); plt.close()
print("ok", 1/1.83041)
