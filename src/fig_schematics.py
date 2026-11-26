import numpy as np
from scipy import special as sp
from matplotlib.patches import Polygon, Rectangle, Ellipse, FancyArrowPatch, Circle, Arc
from style import *

def arrow(a,p,q,color="k",lw=0.8,style="-|>",ms=6,ls="-"):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=ms,color=color,lw=lw,ls=ls,shrinkA=0,shrinkB=0))

# ================= Fig 1 =================
fig,ax=plt.subplots(1,3,figsize=(COL2,2.45),gridspec_kw=dict(width_ratios=[1.35,1,1.25],wspace=0.12))
# ---- (a) axicon geometry
a=ax[0]; a.set_xlim(0,10.2); a.set_ylim(-2.1,2.1); a.axis("off"); panel(a,"(a)")
y0=1.25; z0=2.9; zm=8.4; tanth=y0/(zm-z0)
a.add_patch(Rectangle((0.0,-y0*1.0),2.0,2*y0,fc="#cfe0f5",ec="none",alpha=0.8))     # incident beam
a.text(1.0,y0+0.3,"Gaussian\ninput",ha="center",fontsize=6.5)
a.add_patch(Polygon([[2,-1.5],[2,1.5],[2.6,1.5],[z0,0],[2.6,-1.5]],fc="#b8b8b8",ec="k",lw=0.6))   # axicon
a.text(2.45,-1.95,"axicon",ha="center",fontsize=6.5)
a.add_patch(Polygon([[z0,0],[2.65,y0],[2.65,-y0]],closed=True,fc="none",ec="none"))
a.add_patch(Polygon([[2.65,y0],[2.65,-y0],[zm,0]],fc="#ffe3a8",ec="none",alpha=0.9))       # overlap region
for yy in [y0,0.6,-0.6,-y0]:
    zz=np.array([2.65,2.65+abs(yy)/tanth]); a.plot(zz,[yy,0],color=ORANGE,lw=0.7)
    # continue past axis
    zc=2.65+abs(yy)/tanth; zend=min(zc+2.2,10.0); a.plot([zc,zend],[0,-np.sign(yy)*(zend-zc)*tanth],color=ORANGE,lw=0.7)
a.plot([0,10.2],[0,0],color="k",lw=0.5,ls="-.")
a.plot([3.2,zm],[0,0],color=RED,lw=2.0,solid_capstyle="round")
a.text(5.6,0.32,"Bessel core",color=RED,fontsize=6.5,ha="center",va="bottom",bbox=dict(fc="w",ec="none",pad=0.8,alpha=0.85))
a.annotate("",xy=(2.65,-1.75),xytext=(zm,-1.75),arrowprops=dict(arrowstyle="<->",lw=0.6))
a.text((2.65+zm)/2,-2.07,r"$z_{\max}=R/\tan\theta$",ha="center",fontsize=7)
a.annotate("",xy=(0.9,0),xytext=(0.9,y0),arrowprops=dict(arrowstyle="<->",lw=0.6)); a.text(0.8,y0/2,r"$R$",ha="right",va="center",fontsize=7)
a.add_patch(Arc((zm,0),1.6,1.6,theta1=180,theta2=180+np.degrees(np.arctan(tanth)),lw=0.6))
a.text(zm-1.3,0.12,r"$\theta$",fontsize=7)
# ---- (b) k-space
a=ax[1]; a.set_xlim(-0.2,5.2); a.set_ylim(-2.9,2.4); a.axis("off"); panel(a,"(b)")
th=0.55; K=4.2
arrow(a,(0,0),(5.0,0),lw=0.5); a.text(5.05,-0.35,r"$k_z$",fontsize=7)
for sgn in [1,-1]:
    arrow(a,(0,0),(K*np.cos(th),sgn*K*np.sin(th)),color=ORANGE,lw=1.0)
a.plot([K*np.cos(th)]*2,[-K*np.sin(th),K*np.sin(th)],"k:",lw=0.6)
a.text(K*np.cos(th)+0.1,0.15,r"$\beta=k\cos\theta$",fontsize=6.5,ha="left")
a.text(K*np.cos(th)-0.1,K*np.sin(th)+0.25,r"$\mathbf{k}$",fontsize=7,color=ORANGE,ha="center")
a.text(K*np.cos(th)+0.1,K*np.sin(th)/2,r"$k_t=k\sin\theta$",fontsize=6.5,rotation=90,va="center")
a.add_patch(Arc((0,0),2.0,2.0,theta1=0,theta2=np.degrees(th),lw=0.6)); a.text(1.15,0.2,r"$\theta$",fontsize=7)
a.text(1.8,-1.9,"cone of plane waves\n(half-angle $\\theta$)",ha="center",fontsize=6.5)
# ring inset (k_x,k_y)
ins=a.inset_axes([0.0,0.0,0.01,0.01]); ins.axis("off")
cx,cy=1.1,-1.35
a.add_patch(Ellipse((4.4,-1.75),0.55,1.4,fc="none",ec=BLUE,lw=1.2))
a.text(4.2,-2.8,r"ring $|\mathbf{k}_\perp|=k_t$"+"\nin the $(k_x,k_y)$ plane",ha="center",va="center",fontsize=6.0,color=BLUE)
# ---- (c) Airy generation
a=ax[2]; a.set_xlim(0,10.4); a.set_ylim(-1.2,3.3); a.axis("off"); panel(a,"(c)")
a.add_patch(Rectangle((0.0,0.55),1.6,0.7,fc="#cfe0f5",ec="none")); a.text(0.8,1.55,"laser",ha="center",fontsize=6.5)
a.add_patch(Rectangle((1.6,0.0),0.25,1.8,fc="#b8b8b8",ec="k",lw=0.6)); a.text(1.72,-0.45,"SLM",ha="center",fontsize=6.5)
a.text(1.72,-0.85,r"$e^{icx^3}$",ha="center",fontsize=6.5)
a.add_patch(Rectangle((1.85,0.55),2.2,0.7,fc="#cfe0f5",ec="none"))
a.add_patch(Ellipse((4.2,0.9),0.35,1.9,fc="#e6f0fa",ec="k",lw=0.7)); a.text(4.2,-0.45,"lens",ha="center",fontsize=6.5)
fx=4.2+3.0
a.fill([4.2,fx,fx],[1.25,0.9,0.9],color="#cfe0f5",lw=0); a.fill([4.2,fx,fx],[0.55,0.9,0.9],color="#cfe0f5",lw=0)
a.plot([fx,fx],[-0.2,2.0],"k:",lw=0.6); a.text(fx,-0.45,"Fourier plane\n$(f)$",ha="center",fontsize=6.3)
arrow(a,(4.25,2.45),(fx,2.45),lw=0.5,style="<|-|>",ms=5); a.text((4.2+fx)/2,2.6,r"$f$",ha="center",fontsize=6.5)
# Airy profile at focal plane
yy=np.linspace(-0.05,1.9,300); ss=-(yy-0.9)/0.35*1.0 # map
I=sp.airy(-(yy-0.35)/0.28)[0]**2; I=I/I.max()
a.plot(fx-0.05+0.0*I-I*0.7*(-1),yy,color=BLUE,lw=0.9) if False else a.plot(fx+I*0.9,yy,color=BLUE,lw=0.9)
# parabolic trajectory after focal plane
zp=np.linspace(0,3.0,60); a.plot(fx+zp,0.95+0.16*zp**2,color=RED,lw=1.2)
a.plot(fx+zp,0.95+zp*0*0+0.0,color=GREY,lw=0.6,ls="--")
a.text(fx+2.0,0.25,"straight\n(Gaussian)",fontsize=6,color=GREY,ha="center")
a.text(fx+0.5,2.9,"Airy main lobe,  $x_c\\propto z^2$",fontsize=6.3,color=RED,ha="center")
plt.savefig(os.path.join(FIGDIR,"fig_schematic_generation.pdf")); plt.close()

# ================= Fig 9: applications =================
fig,ax=plt.subplots(2,2,figsize=(COL2,5.0),gridspec_kw=dict(hspace=0.25,wspace=0.12))
# ---- (a) multi-plane tweezers
a=ax[0,0]; a.set_xlim(0,10); a.set_ylim(-2.3,2.3); a.axis("off"); panel(a,"(a)")
a.plot([0,10],[0,0],color="k",lw=0.5,ls="-.")
a.add_patch(Polygon([[0.3,1.7],[0.3,-1.7],[8.8,0]],fc="#ffe3a8",ec="none",alpha=0.7)); a.add_patch(Polygon([[0.3,1.7],[0.3,-1.7],[9.5,0]],fc="none",ec=ORANGE,lw=0.6))
for zc,w in [(3.0,0.9),(7.0,0.9)]:
    a.add_patch(Rectangle((zc-0.5,-1.55),1.0,3.1,fc="#e6f0fa",ec=BLUE,lw=0.7,alpha=0.8))
a.add_patch(Circle((3.0,0),0.22,fc=RED,ec="k",lw=0.5,zorder=5)); a.add_patch(Circle((7.0,0),0.22,fc=RED,ec="k",lw=0.5,zorder=5))
# shadow behind first particle
a.add_patch(Polygon([[3.2,0.2],[3.2,-0.2],[4.4,0]],fc="#555555",ec="none",alpha=0.55)); 
a.text(3.0,-1.95,"cell 1",ha="center",fontsize=6.5); a.text(7.0,-1.95,"cell 2",ha="center",fontsize=6.5)
a.text(4.9,0.5,"shadow\nheals",fontsize=6.3,ha="center",color="#444444")
a.text(5.0,1.95,"Bessel beam traps particles in both cells",ha="center",fontsize=6.5)
# ---- (b) light-sheet
a=ax[0,1]; a.set_xlim(0,10); a.set_ylim(-2.3,2.5); a.axis("off"); panel(a,"(b)")
a.add_patch(Polygon([[0.2,-0.9],[0.2,0.9],[1.4,0.45],[1.4,-0.45]],fc="#b8b8b8",ec="k",lw=0.6)); a.text(0.8,-1.35,"illumination\nobjective",ha="center",fontsize=6.3)
a.add_patch(Ellipse((6.0,0),3.2,2.6,fc="#e8f5e8",ec=GREEN,lw=0.8))
a.plot([1.4,9.3],[0,0],color=RED,lw=1.5)
for off in [0.18,-0.18,0.35,-0.35]: a.plot([1.4,9.3],[off,off],color=ORANGE,lw=0.4,alpha=0.7)
a.text(8.3,0.55,"Bessel core\n+ side lobes",fontsize=6.3,color=RED,ha="center")
arrow(a,(6.0,-1.9),(6.0,-1.1),color="k",lw=0.7); a.text(6.15,-1.95,"scan",fontsize=6.3)
a.add_patch(Polygon([[5.4,2.4],[6.6,2.4],[6.25,1.55],[5.75,1.55]],fc="#b8b8b8",ec="k",lw=0.6)); a.text(7.7,2.0,"detection\nobjective",ha="center",fontsize=6.3)
arrow(a,(6.0,1.5),(6.0,0.8),color=GREEN,lw=0.8); a.text(5.3,1.15,"fluor.",fontsize=6.3,color=GREEN,ha="right")
# ---- (c) machining
a=ax[1,0]; a.set_xlim(0,10); a.set_ylim(-2.3,2.6); a.axis("off"); panel(a,"(c)")
a.add_patch(Rectangle((0.8,-1.0),8.4,1.8,fc="#d9e8f7",ec="k",lw=0.7)); a.text(5.0,-1.55,"transparent slab (thickness $L$)",ha="center",fontsize=6.5)
a.add_patch(Polygon([[2.0,2.3],[3.0,0.0],[4.0,2.3]],fc="#ffe3a8",ec=ORANGE,lw=0.6,alpha=0.8))
a.add_patch(Polygon([[2.75,-0.95],[3.0,0.0],[3.25,-0.95]],fc="#ffe3a8",ec=ORANGE,lw=0.6,alpha=0.8))
a.plot([3.0,3.0],[-0.2,0.25],color=RED,lw=2.6); a.text(3.0,2.45,"Gaussian focus",ha="center",fontsize=6.5)
a.text(3.0,-2.05,"short modification",ha="center",fontsize=6.2,color=RED)
a.add_patch(Polygon([[6.0,2.3],[7.5,-1.0],[9.0,2.3]],fc="#ffe3a8",ec=ORANGE,lw=0.6,alpha=0.8)); a.plot([7.5,7.5],[-1.0,0.8],color=RED,lw=2.6)
a.text(7.5,2.45,"axicon / Bessel line",ha="center",fontsize=6.5)
a.text(7.5,-2.05,"channel through slab",ha="center",fontsize=6.2,color=RED)
# ---- (d) Airy
a=ax[1,1]; a.set_xlim(0,10); a.set_ylim(-2.3,2.5); a.axis("off"); panel(a,"(d)")
a.add_patch(Rectangle((4.2,-0.55),1.1,1.1,fc="#9a9a9a",ec="k",lw=0.7)); a.text(4.75,-1.1,"obstacle",ha="center",fontsize=6.5)
a.plot([0.6,9.4],[0,0],color=GREY,lw=0.7,ls="--"); a.text(1.0,-0.4,"Gaussian: blocked",fontsize=6.3,color=GREY)
zp=np.linspace(0,8.8,80); a.plot(0.6+zp,-0.0+0.0+ (0.045*(zp-0.0)**2 -0.0)*0.9 -0.0 - 0.0 ,color=RED,lw=1.3) if False else None
zp=np.linspace(0,8.8,100); yc=0.05*(zp)**2-0.0
a.plot(0.6+zp,yc*1.0,color=RED,lw=1.3)
a.add_patch(Circle((9.3,yc[-1]),0.17,fc=RED,ec="k",lw=0.5,zorder=5)); a.text(9.2,yc[-1]+0.5,"target\n(plasma / particles)",fontsize=6.2,ha="right",color=RED)
a.text(1.0,1.35,"Airy: curved path",fontsize=6.5,color=RED)
a.set_ylim(-2.3,4.6)
plt.savefig(os.path.join(FIGDIR,"fig_schematic_apps.pdf")); plt.close()
print("ok")
