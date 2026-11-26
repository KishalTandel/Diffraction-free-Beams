import numpy as np, json
from scipy import special as sp
from scipy import integrate
from style import *
from sim import *

out = {}
lam=632.8e-9; k=2*np.pi/lam
w0=1.0e-3; theta=8e-3; kt=k*np.sin(theta); tth=np.tan(theta)
zmax=w0/tth; r0=sp.jn_zeros(0,1)[0]/kt
out.update(lam=lam,w0=w0,theta=theta,kt=kt,zmax=zmax,r0=r0)

# ================= Fig. 2: Bessel profile & ring powers =================
zs=np.concatenate([[0],sp.jn_zeros(0,12)])
rp=[integrate.quad(lambda t: sp.j0(t)**2*t,zs[i],zs[i+1])[0] for i in range(12)]
fig,ax=plt.subplots(1,2,figsize=(COL2,2.3))
a=ax[0]
xx=np.linspace(-12,12,3000)
a.plot(xx,sp.j0(xx)**2,color=BLUE,label=r"$J_0^2(k_t r)$")
xg=1.9133
a.plot(xx,np.exp(-2*xx**2/xg**2),color=ORANGE,ls="--",label=r"Gaussian, equal FWHM")
a.axvline(2.405,color=GREY,lw=0.6,ls=":"); a.axvline(-2.405,color=GREY,lw=0.6,ls=":")
a.annotate("",xy=(-1.1264,0.5),xytext=(1.1264,0.5),arrowprops=dict(arrowstyle="<->",lw=0.6,color="k"))
a.text(0,0.53,r"FWHM $=2.25/k_t$",ha="center",fontsize=7,bbox=dict(fc="w",ec="none",pad=0.5))
a.text(2.7,0.45,r"$r_0=2.405/k_t$",fontsize=7,color=GREY)
a.set_xlabel(r"$k_t x$"); a.set_ylabel(r"$I/I_0$"); a.set_xlim(-12,12); a.set_ylim(0,1.05)
a.legend(loc="upper left",bbox_to_anchor=(0.0,1.0),fontsize=6.5); panel(a,"(a)")
a=ax[1]
idx=np.arange(0,12)
a.bar(idx,rp,color=[ORANGE]+[BLUE]*11,width=0.7)
a.axhline(1.0,color="k",lw=0.6,ls="--")
a.set_xticks(idx); a.set_xticklabels(["core"]+[str(i) for i in range(1,12)])
a.set_xlabel("lobe index"); a.set_ylabel(r"power in lobe $/\,(2\pi I_0/k_t^2)$")
a.set_ylim(0,1.25); panel(a,"(b)")
a.text(11.4,1.05,r"asymptotic value $=1$",ha="right",fontsize=7)
plt.savefig(os.path.join(FIGDIR,"fig_bessel_profile.pdf")); plt.close()
out["ring_powers"]=[float(v) for v in rp[:6]]

# ================= 2D simulation setup =================
N=2048; dx=4e-6
xs,X,Y=grid2d(N,dx); R=np.hypot(X,Y)
kx=2*np.pi*np.fft.fftfreq(N,d=dx); KX,KY=np.meshgrid(kx,kx,indexing='xy')
kz=np.sqrt(np.maximum(k**2-KX**2-KY**2,0)); ev=(k**2-KX**2-KY**2)>0
nz=100; zend=0.2; dz=zend/nz; zgrid=np.arange(nz+1)*dz
Hdz=np.exp(1j*kz*dz)*ev
half=N//2
def run(E0,mask_at=None,mask=None):
    spec=np.fft.fft2(E0); sl=[]; E=E0.copy()
    for i in range(nz+1):
        if i>0:
            spec=spec*Hdz; E=np.fft.ifft2(spec)
        if mask is not None and i==mask_at:
            E=E*mask; spec=np.fft.fft2(E)
        sl.append(np.abs(E[half,:])**2)
    sl=np.array(sl); return sl, sl[:,half].copy()
E0=np.exp(-R**2/w0**2)*np.exp(-1j*kt*R)     # axicon-illuminated Gaussian
sl,axint=run(E0)
Imax=sl.max()

# ---- Fig. 3 ----
wG=1.9133/kt; zR=np.pi*wG**2/lam
sel=np.abs(xs)<0.35e-3
zz=np.linspace(0,zend,400)
wz=wG*np.sqrt(1+(zz/zR)**2)
IG=(wG/wz)[:,None]**2*np.exp(-2*xs[sel][None,:]**2/wz[:,None]**2)
fig=plt.figure(figsize=(COL2,4.6))
gs=fig.add_gridspec(2,2,hspace=0.4,wspace=0.3)
for j,(arr,zv,ttl) in enumerate([(IG,zz,"Gaussian beam (same core size)"),(sl[:,sel]/Imax,zgrid,"axicon + Gaussian input")]):
    a=fig.add_subplot(gs[0,j])
    im=a.imshow(arr.T**0.5,origin="lower",aspect="auto",cmap="inferno",
        extent=[zv[0]*1e3,zv[-1]*1e3,xs[sel][0]*1e6,xs[sel][-1]*1e6],vmin=0,vmax=1)
    a.set_xlabel(r"$z$ (mm)"); a.set_ylabel(r"$x$ ($\mu$m)"); panel(a,["(a)","(b)"][j],color="w")
    a.set_title(ttl,fontsize=8)
    if j==0:
        a.axvline(zR*1e3,color="c",lw=0.6,ls="--"); a.text(zR*1e3+3,330,r"$z_R$",color="c",fontsize=7,va="top")
    else:
        a.axvline(zmax*1e3,color="c",lw=0.6,ls="--"); a.text(zmax*1e3+3,330,r"$z_{\max}$",color="c",fontsize=7,va="top")
cb=fig.colorbar(im,ax=fig.axes[:2],fraction=0.025,pad=0.02); cb.set_label(r"$\sqrt{I/I_{\max}}$")
a=fig.add_subplot(gs[1,0])
a.semilogy(zz*1e3,1/(1+(zz/zR)**2),color=ORANGE,ls="--",label=r"Gaussian, $1/(1+z^2/z_R^2)$")
a.semilogy(zgrid*1e3,axint/axint.max(),color=BLUE,label="axicon (FFT sim.)")
zf=np.linspace(0.0005,zend,400); th=zf*np.exp(-2*zf**2/zmax**2); th/=th.max()
a.semilogy(zf*1e3,th,color="k",lw=0.7,ls=":",label=r"$\propto z\,e^{-2z^2/z_{\max}^2}$")
a.set_ylim(1e-4,2e3); a.set_xlabel(r"$z$ (mm)"); a.set_ylabel(r"on-axis $I/I_{\max}$"); a.legend(loc="upper right",fontsize=6.3); panel(a,"(c)",x=0.02,y=0.2)
a=fig.add_subplot(gs[1,1])
sel2=np.abs(xs)<250e-6
for c,i in zip([BLUE,GREEN,ORANGE],[15,30,50]):
    prof=sl[i]/sl[i][half]
    a.plot(xs[sel2]*1e6,prof[sel2],color=c,lw=0.9,label=r"$z=%d$ mm"%round(zgrid[i]*1e3))
a.plot(xs[sel2]*1e6,sp.j0(kt*xs[sel2])**2,"k",lw=0.6,ls=":",label=r"$J_0^2(k_tx)$")
a.set_xlabel(r"$x$ ($\mu$m)"); a.set_ylabel(r"$I(x,0,z)/I(0,0,z)$"); a.legend(fontsize=6.5,loc="upper right"); panel(a,"(d)")
plt.savefig(os.path.join(FIGDIR,"fig_propagation.pdf")); plt.close()
out.update(wG=wG,zR=zR,gain=zmax/zR)

# ---- Fig. 4: Bessel-Gauss closed form check ----
E0bg=sp.j0(kt*R)*np.exp(-R**2/w0**2)
def BG(r,z):
    q=1+2j*z/(k*w0**2)
    return np.exp(1j*k*z)*(1/q)*sp.jv(0,kt*r/q)*np.exp(-r**2/(w0**2*q))*np.exp(-1j*kt**2*z/(2*k*q))
fig,ax=plt.subplots(figsize=(COL1,2.3))
sel3=np.abs(xs)<220e-6
errs=[]
for c,z in zip([BLUE,GREEN,ORANGE],[0.0,0.06,0.12]):
    Ez=propagate_as(E0bg,dx,k,z) if z>0 else E0bg
    row=np.abs(Ez[half,:])**2
    ref=np.abs(BG(np.abs(xs),z))**2
    ax.plot(xs[sel3]*1e6,ref[sel3],color=c,lw=1.0,label=r"$z=%d$ mm"%round(z*1e3))
    ax.plot(xs[sel3][::6]*1e6,row[sel3][::6],"o",ms=1.8,color=c,mfc="none",mew=0.5)
    m=R<1.5e-3
    errs.append(float(np.max(np.abs(Ez[m]-BG(R,z)[m]))/np.max(np.abs(BG(R,z)[m]))))
ax.set_xlabel(r"$x$ ($\mu$m)"); ax.set_ylabel(r"$|E|^2$ (arb.)"); ax.legend(fontsize=6.5,loc="upper right")
ax.text(0.03,0.95,"lines: closed form\ncircles: FFT",transform=ax.transAxes,ha="left",va="top",fontsize=7)
plt.savefig(os.path.join(FIGDIR,"fig_bg_check.pdf")); plt.close()
out["bg_errs"]=errs

# ---- Fig. 5: self-healing ----
a_obs=150e-6; zo_i=15; zo=zgrid[zo_i]
mask=(R>a_obs).astype(float)
sl2,ax2=run(E0,mask_at=zo_i,mask=mask)
zheal=a_obs/tth
fig=plt.figure(figsize=(COL2,4.4))
gs=fig.add_gridspec(2,2,hspace=0.45,wspace=0.3,height_ratios=[1.1,1])
a=fig.add_subplot(gs[0,:])
sel4=np.abs(xs)<0.3e-3
im=a.imshow(sl2[:,sel4].T**0.5/np.sqrt(Imax),origin="lower",aspect="auto",cmap="inferno",
   extent=[0,zend*1e3,xs[sel4][0]*1e6,xs[sel4][-1]*1e6],vmin=0,vmax=1)
a.add_patch(plt.Rectangle((zo*1e3-0.6,-a_obs*1e6),1.2,2*a_obs*1e6,color="w",ec="c",lw=0.5))
zz2=np.linspace(zo,zend,50)
a.plot(zz2*1e3,a_obs*1e6-(zz2-zo)*tth*1e6,"c--",lw=0.7); a.plot(zz2*1e3,-a_obs*1e6+(zz2-zo)*tth*1e6,"c--",lw=0.7)
a.set_ylim(-300,300)
a.annotate("",xy=((zo+zheal)*1e3,-250),xytext=(zo*1e3,-250),arrowprops=dict(arrowstyle="<->",color="c",lw=0.7))
a.text((zo+zheal/2)*1e3,-230,r"$z_{\rm heal}=a/\tan\theta$",color="c",ha="center",va="bottom",fontsize=7)
a.set_xlabel(r"$z$ (mm)"); a.set_ylabel(r"$x$ ($\mu$m)"); panel(a,"(a)",color="w")
cb=fig.colorbar(im,ax=a,fraction=0.02,pad=0.01); cb.set_label(r"$\sqrt{I/I_{\max}}$")
a=fig.add_subplot(gs[1,0])
a.plot(zgrid*1e3,axint/Imax,color=GREY,ls="--",label="no obstacle")
a.plot(zgrid*1e3,ax2/Imax,color=BLUE,label="with obstacle")
a.axvline((zo+zheal)*1e3,color="c",lw=0.7,ls=":"); a.axvline(zo*1e3,color="k",lw=0.5,ls=":")
a.set_xlabel(r"$z$ (mm)"); a.set_ylabel(r"on-axis $I/I_{\max}$"); a.legend(fontsize=6.5,loc="upper right"); panel(a,"(b)")
a=fig.add_subplot(gs[1,1])
i_far=70
sel5=np.abs(xs)<250e-6
a.plot(xs[sel5]*1e6,sl[i_far][sel5]/sl[i_far][half],color=GREY,ls="--",label="no obstacle")
a.plot(xs[sel5]*1e6,sl2[i_far][sel5]/sl[i_far][half],color=BLUE,label="with obstacle")
a.set_xlabel(r"$x$ ($\mu$m)"); a.set_ylabel(r"$I(x,0,z)/I(0,0,z)_{\rm free}$")
a.set_title(r"$z=%d$ mm"%round(zgrid[i_far]*1e3),fontsize=7); a.legend(fontsize=6.5,loc="upper right"); panel(a,"(c)")
plt.savefig(os.path.join(FIGDIR,"fig_selfheal_bessel.pdf")); plt.close()
# quantify healing: axis intensity recovery
ihe=int(round((zo+zheal)/dz))
out.update(zo=zo,zheal=zheal,a_obs=a_obs,
    axis_ratio_at_heal=float(ax2[ihe]/axint[ihe]), axis_ratio_far=float(ax2[i_far]/axint[i_far]),
    axis_ratio_early=float(ax2[zo_i+3]/axint[zo_i+3]))
# find first z beyond obstacle where axis intensity exceeds 50% of free
rec=None
for i in range(zo_i+1,nz+1):
    if ax2[i]>0.5*axint[i]: rec=zgrid[i]; break
out["z_half_recovery"]=rec

# ================= Airy (1D) =================
a_air=0.02
Ns=2**15; ds=0.02; s=(np.arange(Ns)-Ns//2)*ds
psi0=sp.airy(s)[0]*np.exp(a_air*s)
nxi=240; xi_end=12.0; dxi=xi_end/nxi; xis=np.arange(nxi+1)*dxi
def airy_run(psi0,mask_xi=None,mask_fn=None):
    ps=psi0.astype(complex); rows=[np.abs(ps)**2]
    for i in range(1,nxi+1):
        ps=propagate_1d_paraxial(ps,ds,dxi)
        if mask_xi is not None and i==mask_xi:
            ps=ps*mask_fn(xis[i])
        rows.append(np.abs(ps)**2)
    return np.array(rows),ps
rowsA,psf=airy_run(psi0)
# peak tracking of main lobe
pk=[]
for i,xi in enumerate(xis):
    c=xi**2/4-1.0188
    m=(s>c-1.5)&(s<c+1.5)
    pk.append(s[m][np.argmax(rowsA[i][m])])
pk=np.array(pk); par=xis**2/4-1.0188
out["airy_peak_dev_max"]=float(np.max(np.abs(pk-par)))
cent=(rowsA*s[None,:]).sum(1)/rowsA.sum(1); out["airy_centroid_spread"]=float(cent.max()-cent.min())
out["airy_power_norm_check"]=float(np.sum(rowsA[0])*ds), float(np.sum(rowsA[-1])*ds)
out["airy_mainlobe_peak_ratio_xi12"]=float(rowsA[-1][(s>par[-1]-1.5)&(s<par[-1]+1.5)].max()/rowsA[0][(s>-2.5)&(s<0.5)].max())

fig=plt.figure(figsize=(COL2,4.9))
gs=fig.add_gridspec(2,3,hspace=0.42,wspace=0.42,height_ratios=[1,1],width_ratios=[1,1,1])
a=fig.add_subplot(gs[0,0])
sx=np.linspace(-14,4,2000)
a.plot(sx,sp.airy(sx)[0]**2,color=ORANGE,ls="--",lw=0.8,label=r"$\mathrm{Ai}^2(s)$")
a.plot(sx,(sp.airy(sx)[0]*np.exp(a_air*sx))**2,color=BLUE,label=r"$\mathrm{Ai}^2e^{2as}$, $a=0.02$")
a.set_xlabel(r"$s=x/x_0$"); a.set_ylabel(r"$I$ (arb.)"); a.legend(fontsize=6.3,loc="upper left"); panel(a,"(a)",x=0.02,y=0.55)
a=fig.add_subplot(gs[0,1:])
sw=(s>-12)&(s<42)
im=a.imshow(rowsA[:,sw].T**0.35,origin="lower",aspect="auto",cmap="inferno",
    extent=[0,xi_end,s[sw][0],s[sw][-1]],vmin=0,vmax=(rowsA.max())**0.35)
a.plot(xis,par,"c--",lw=0.8,label=r"$s=\xi^2/4-1.02$")
a.plot(xis,cent,"w:",lw=1.0,label=r"centroid $\langle s\rangle$")
a.set_ylim(-12,42); a.set_xlabel(r"$\xi=z/(kx_0^2)$"); a.set_ylabel(r"$s$"); a.legend(fontsize=6.5,loc="upper left",labelcolor="w",facecolor="none"); panel(a,"(b)",x=0.35,y=0.18,color="w")
# 2D beam
def i2d(row): 
    r=row[(s>-8)&(s<24)]; return np.outer(r,r)
ss=s[(s>-8)&(s<24)]
for j,(i,lab) in enumerate([(0,r"$\xi=0$"),(int(round(8/dxi)),r"$\xi=8$")]):
    a=fig.add_subplot(gs[1,j])
    arr=i2d(rowsA[i]); arr/=arr.max()
    a.imshow(arr**0.35,origin="lower",extent=[ss[0],ss[-1],ss[0],ss[-1]],cmap="inferno",vmin=0,vmax=1)
    a.set_xlabel(r"$s_x$"); a.set_ylabel(r"$s_y$"); a.set_title(lab,fontsize=8); panel(a,["(c)","(d)"][j],color="w")
a=fig.add_subplot(gs[1,2])
for c,i in zip([BLUE,GREEN,ORANGE],[0,int(round(5/dxi)),int(round(10/dxi))]):
    shift=xis[i]**2/4
    m=(s-shift>-9)&(s-shift<4)
    a.plot((s-shift)[m],rowsA[i][m]/rowsA[i][m].max(),color=c,lw=0.9,label=r"$\xi=%g$"%xis[i])
a.set_xlabel(r"$s-\xi^2/4$"); a.set_ylabel(r"$I/I_{\max}$"); a.legend(fontsize=6.5); panel(a,"(e)",x=0.02)
plt.savefig(os.path.join(FIGDIR,"fig_airy.pdf")); plt.close()

# ---- Fig 7: Airy self-healing ----
xi_b=3.0; ib=int(round(xi_b/dxi))
def blockfn(xi):
    c=xi**2/4-1.0188
    return (np.abs(s-c)>1.4).astype(float)
rowsB,psB=airy_run(psi0,mask_xi=ib,mask_fn=blockfn)
fig,ax=plt.subplots(1,2,figsize=(COL2,2.5),gridspec_kw=dict(width_ratios=[1.3,1],wspace=0.3))
a=ax[0]
im=a.imshow(rowsB[:,sw].T**0.35,origin="lower",aspect="auto",cmap="inferno",extent=[0,xi_end,s[sw][0],s[sw][-1]],vmin=0,vmax=(rowsA.max())**0.35)
a.axvline(xi_b,color="c",lw=0.6,ls=":"); a.text(xi_b+0.15,38,"main lobe\nblocked",color="c",fontsize=6.5,va="top")
a.set_ylim(-12,42); a.set_xlabel(r"$\xi$"); a.set_ylabel(r"$s$"); panel(a,"(a)",color="w",y=0.12)
a=ax[1]
ie=int(round(9/dxi)); c=xis[ie]**2/4-1.0188
m=(s-c>-9)&(s-c<4)
a.plot((s-c)[m],rowsA[ie][m]/rowsA[ie].max(),color=GREY,ls="--",label="unblocked")
a.plot((s-c)[m],rowsB[ie][m]/rowsA[ie].max(),color=BLUE,label="blocked at $\\xi=3$")
a.set_xlabel(r"$s-\xi^2/4+1.02$"); a.set_ylabel(r"$I/I_{\max}$ (unblocked)"); a.set_title(r"$\xi=9$",fontsize=7); a.legend(fontsize=6.5); panel(a,"(b)")
plt.savefig(os.path.join(FIGDIR,"fig_airy_heal.pdf")); plt.close()
def lobe_peak(rows,i):
    cc=xis[i]**2/4-1.0188; m=(s>cc-1.5)&(s<cc+1.5); return rows[i][m].max()
for xv in [4,6,9,12]:
    i=int(round(xv/dxi)); out["airy_heal_ratio_xi%d"%xv]=float(lobe_peak(rowsB,i)/lobe_peak(rowsA,i))
out["airy_lobe_decay_theory_xi12"]=float(np.exp(-a_air*12**2/2))
out["airy_lobe_decay_numeric_xi12"]=float(lobe_peak(rowsA,int(round(12/dxi)))/lobe_peak(rowsA,0))

# ---- Fig 8: trade-off & light sheet ----
# (a) f vs G
def core_frac_numeric(kt_w0):
    # radial integration with kt=1 units: w0=kt_w0
    rr=np.linspace(0,8*kt_w0,400000)
    I=sp.j0(rr)**2*np.exp(-2*rr**2/kt_w0**2)
    tot=np.trapezoid(I*rr,rr); cr=rr<sp.jn_zeros(0,1)[0]
    return np.trapezoid((I*rr)[cr],rr[cr])/tot
ktw=np.array([5,8,12,20,40,80,160,320])
fnum=np.array([core_frac_numeric(v) for v in ktw])
fform=3.9068/ktw
Gform=0.5457*ktw
fig,ax=plt.subplots(1,2,figsize=(COL2,2.5),gridspec_kw=dict(wspace=0.32))
a=ax[0]
Gc=np.logspace(0.5,2.8,100)
a.loglog(Gc,2.132/Gc,color="k",lw=0.8,label=r"$fG\simeq2.13$")
a.loglog(Gform,fnum,"o",color=BLUE,ms=3.2,label="numerical")
a.set_xlabel(r"range gain $G=z_{\max}/z_R^{\rm (Gauss)}$"); a.set_ylabel(r"central-lobe power fraction $f$")
a.legend(fontsize=6.5); panel(a,"(a)",x=0.45,y=0.97)
out["f_check"]=[(float(a_),float(b_)) for a_,b_ in zip(fnum,fform)]
# (b) light sheet: scanned Gaussian vs scanned Bessel (water, 488 nm)
lam_ls=488e-9; n_w=1.33; k_m=2*np.pi*n_w/lam_ls
NAr=0.40; sin_t=NAr/n_w; kt_ls=k_m*sin_t; tan_t=sin_t/np.sqrt(1-sin_t**2)
zmax_ls=50e-6; w0_ls=zmax_ls*tan_t
core_fwhm=2.2527/kt_ls; wG_ls=1.9133/kt_ls; zR_ls=np.pi*n_w*wG_ls**2/lam_ls
f_ls=3.9068/(kt_ls*w0_ls); G_ls=zmax_ls/zR_ls
out.update(ls=dict(kt=kt_ls,sin_t=sin_t,core_fwhm=core_fwhm,r0=2.405/kt_ls,w0=w0_ls,zmax=zmax_ls,zR=zR_ls,G=G_ls,f=f_ls,N=w0_ls*kt_ls/np.pi))
xq=np.linspace(-6e-6,6e-6,801); yq=np.linspace(-3*w0_ls,3*w0_ls,6001)
Xq,Yq=np.meshgrid(xq,yq,indexing="ij")
Rq=np.hypot(Xq,Yq)
Ib=sp.j0(kt_ls*Rq)**2*np.exp(-2*Rq**2/w0_ls**2)
S=np.trapezoid(Ib,yq,axis=1)   # scanned sheet profile
# full sheet energy (wide x)
xw=np.linspace(-3*w0_ls,3*w0_ls,2401); 
def sheet_full():
    tot=0
    Xw,Yw=np.meshgrid(xw,yq,indexing="ij")
    Iw=sp.j0(kt_ls*np.hypot(Xw,Yw))**2*np.exp(-2*(Xw**2+Yw**2)/w0_ls**2)
    Sw=np.trapezoid(Iw,yq,axis=1); return xw,Sw
xw_,Sw_=sheet_full()
Stot=np.trapezoid(Sw_,xw_)
# FWHM of scanned sheet
Sn=S/S.max(); above=xq[Sn>=0.5]; sheet_fwhm=above.max()-above.min()
frac_in=np.trapezoid(S[np.abs(xq)<=core_fwhm/2],xq[np.abs(xq)<=core_fwhm/2])/Stot
frac_in_r0=np.trapezoid(S[np.abs(xq)<=2.405/kt_ls],xq[np.abs(xq)<=2.405/kt_ls])/Stot
out["ls"].update(sheet_fwhm=float(sheet_fwhm),frac_in_fwhm=float(frac_in),frac_in_r0=float(frac_in_r0))
wL=np.sqrt(lam_ls*zmax_ls/(np.pi*n_w)); gfwhm=1.1774*wL
out["ls"].update(wG_equal_length=float(wL),gauss_equal_length_fwhm=float(gfwhm))
Sg=np.exp(-2*xq**2/wL**2)
Sg0=np.exp(-2*xq**2/wG_ls**2)
a=ax[1]
a.semilogy(xq*1e6,Sg,color=ORANGE,ls="--",label=r"Gaussian sheet, same length ($z_R=z_{\max}$)")
a.semilogy(xq*1e6,Sg0,color=GREEN,ls=":",label=r"Gaussian sheet, same core ($z_R\approx1\,\mu$m)")
a.semilogy(xq*1e6,Sn,color=BLUE,label="scanned Bessel beam")
a.set_xlim(-6,6); a.set_ylim(1e-3,1.5); a.set_xlabel(r"$x$ ($\mu$m)"); a.set_ylabel(r"sheet profile (norm.)"); a.legend(fontsize=6,loc="upper center",bbox_to_anchor=(0.5,-0.22)); panel(a,"(b)")
plt.savefig(os.path.join(FIGDIR,"fig_tradeoff.pdf")); plt.close()

json.dump(out,open(os.path.join(FIGDIR,"results.json"),"w"),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
