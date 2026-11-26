"""Numerical checks of the analytic results quoted in the paper (Appendix B table and Sec. VII numbers).
Run:  python checks.py        (about 1-2 minutes)"""
import numpy as np
from scipy import special as sp, integrate
from scipy.optimize import brentq
from sim import grid2d, propagate_as, propagate_1d_paraxial

def row(name, analytic, numeric):
    print(f"{name:<62s} analytic = {analytic:<14} numeric = {numeric}")

j01 = sp.jn_zeros(0, 1)[0]
print("== Bessel beam facts ==")
xh = brentq(lambda x: sp.j0(x)**2 - 0.5, 0.5, 1.5)
print("first zero of J0:", j01, "| J0^2=1/2 at x =", xh, "-> FWHM*kt =", 2*xh)
print("core r0*kt / FWHM*kt / r0 in lambda/sin(theta):", j01, 2*xh, j01/2/np.pi, 2*xh/2/np.pi)
zs = sp.jn_zeros(0, 4); prev = 0; rp = []
for z in zs:
    rp.append(integrate.quad(lambda x: sp.j0(x)**2*x, prev, z)[0]); prev = z
print("power in core + first 3 rings (units 2 pi I0/kt^2):", np.round(rp, 3))

# ---------------- Gaussian / Bessel-Gauss, running example -----------------
lam = 632.8e-9; k = 2*np.pi/lam
w0 = 1.0e-3; sth = 8e-3; theta = np.arcsin(sth); kt = k*sth
zmax = w0/np.tan(theta); N = 2048; dx = 4e-6
xs, X, Y = grid2d(N, dx); R = np.hypot(X, Y); h = N//2
wG = (2.253/kt)/1.1774
zRG = k*wG**2/2
print("\n== Running example ==  kt =", kt, " zmax =", zmax, " Gaussian zR =", zRG, " G =", zmax/zRG, " kt*w0 =", kt*w0)
print("Gaussian of equal FWHM (%.1f um): zR (m) ="%(2.253/kt*1e6), np.pi*((2.253/kt)/1.1774)**2/lam)
print("N rings ~ kt w0/pi =", kt*w0/np.pi, " G/N =", (zmax/zRG)/(kt*w0/np.pi))

def BG(r, z):
    q = 1 + 2j*z/(k*w0**2)
    return np.exp(1j*k*z)/q*sp.jv(0, kt*r/q)*np.exp(-r**2/(w0**2*q))*np.exp(-1j*kt**2*z/(2*k*q))

E0 = sp.j0(kt*R)*np.exp(-R**2/w0**2)
print("\n== Table: analytic vs numerics ==")
for z in (0.06, 0.12):
    Ez = propagate_as(E0, dx, k, z); ref = BG(R, z); m = R < 1.5e-3
    err = np.max(np.abs(Ez[m]-ref[m]))/np.max(np.abs(ref[m]))
    row(f"Bessel-Gauss field vs FFT, z={z*1e3:.0f} mm (rel. error)", f"{kt**4*z/(8*k**3):.2e}", f"{err:.2e}")
z = 0.06; q = 1 + 2j*z/(k*w0**2); eps = z/(k*w0**2/2)
Ia = np.exp(-(2*z**2/zmax**2)/(1+eps**2))/(1+eps**2)
In = abs(propagate_as(E0, dx, k, z)[h, h])**2
row("Bessel-Gauss I(0,z)/I0 at z=60 mm", f"{Ia:.4f}", f"{In:.4f}")

# axicon axial intensity
Ea = np.exp(-R**2/w0**2)*np.exp(-1j*kt*R)
kx = 2*np.pi*np.fft.fftfreq(N, d=dx); KX, KY = np.meshgrid(kx, kx, indexing='xy')
kz = np.sqrt(np.maximum(k**2-KX**2-KY**2, 0)); H = np.exp(1j*kz*0.002)*((k**2-KX**2-KY**2) > 0)
spec = np.fft.fft2(Ea); zz = []; I = []
for i in range(1, 101):
    spec = spec*H; zz.append(i*0.002); I.append(abs(np.fft.ifft2(spec)[h, h])**2)
zz = np.array(zz); I = np.array(I)
th = zz*np.exp(-2*zz**2/zmax**2); sel = (zz > 0.02) & (zz < 0.18)
c = np.sum(I[sel]*th[sel])/np.sum(th[sel]**2); dev = np.abs(I[sel]-c*th[sel])/(c*th[sel])
Iabs = 2*np.pi*kt**2/k*zz*np.exp(-2*(zz*np.tan(theta))**2/w0**2)
row("Axicon I(0,z) sim/analytic at z=60, 100 mm", "1", f"{I[29]/Iabs[29]:.3f}, {I[49]/Iabs[49]:.3f}")
row("Axicon axial maximum (mm)", f"{zmax/2*1e3:.1f}", f"{zz[np.argmax(I)]*1e3:.0f}")
print("   axicon max shape deviation over 20-180 mm:", f"{dev.max():.3f}")

# core power fraction
rr = np.linspace(0, 8*kt*w0, 400000)
Ir = sp.j0(rr)**2*np.exp(-2*rr**2/(kt*w0)**2)
fnum = np.trapezoid((Ir*rr)[rr < j01], rr[rr < j01])/np.trapezoid(Ir*rr, rr)
fa = 0.5*j01**2*sp.j1(j01)**2*2*np.pi/kt**2/((w0/kt)*np.sqrt(np.pi/2))  # exact core / asymptotic total
row("Core power fraction f, kt w0 = 79.4", f"{fa:.5f}", f"{fnum:.5f}")
print("   f (exact core / asymptotic total) =", fa, "; f*G =", fnum*zmax/zRG, "; peak ratio 4.59/(kt w0) =", 4.59/(kt*w0), "=", 4.59/(kt*w0)/fnum, "f")

# ---------------- Airy ------------------
print("\n== Airy ==")
def airy_c(s, xi, a):
    arg = s - xi**2/4 + 1j*a*xi
    return sp.airy(arg)[0]*np.exp(a*s - a*xi**2/2 - 1j*xi**3/12 + 1j*a**2*xi/2 + 1j*s*xi/2)
a = 0.1; Ns = 2**14; ds = 0.02; s = (np.arange(Ns)-Ns//2)*ds
psi0 = sp.airy(s)[0]*np.exp(a*s); worst = 0
for xi in (2., 5., 8.):
    ps = propagate_1d_paraxial(psi0, ds, xi); ref = airy_c(s, xi, a); m = (s > -30) & (s < 10)
    worst = max(worst, np.max(np.abs(ps[m]-ref[m]))/np.max(np.abs(ref[m])))
row("Airy closed form vs split-step, a=0.1, xi<=8 (max rel. error)", "---", f"{worst:.1e}")
row("Airy power, a=0.1", f"{np.exp(2*a**3/3)/(2*np.sqrt(2*np.pi*a)):.6f}", f"{np.sum(np.abs(psi0)**2)*ds/(2*np.pi)*2*np.pi:.6f}")
# lobe decay, a = 0.02 (larger window)
a = 0.02; Ns = 2**15; s = (np.arange(Ns)-Ns//2)*ds; psi0 = sp.airy(s)[0]*np.exp(a*s)
def mainpeak(xi):
    I = np.abs(propagate_1d_paraxial(psi0, ds, xi))**2; m = (s > xi**2/4-4) & (s < xi**2/4+4); return I[m].max()
row("Main-lobe decay I(xi=12)/I(0), a=0.02", f"{np.exp(-a*144/2):.3f}", f"{mainpeak(12.)/mainpeak(0.):.3f}")
for xi in (0, 4, 8):
    I = np.abs(propagate_1d_paraxial(psi0, ds, float(xi)))**2
    print("   centroid at xi =", xi, ":", np.sum(s*I)/np.sum(I))

# ---------------- Applications: numbers quoted in Sec. VII --------------
print("\n== Applications ==")
print("Airy: kx0^2 (cm), displacement after 10 cm (mm):", end=" ")
x0 = 50e-6; print(k*x0**2*100, (0.10)**2/(4*k**2*x0**3)*1e3)
# tweezers
th_ = 0.1; kt_ = k*th_; r0_ = j01/kt_; zmax_ = 3e-3; w0_ = zmax_*np.tan(th_)
print(f"Tweezers: r0 = {r0_*1e6:.2f} um, z_heal(a=1um) = {1e-6/np.tan(th_)*1e6:.1f} um, w0 = {w0_*1e3:.2f} mm, kt w0 = {kt_*w0_:.0f}, N = {kt_*w0_/np.pi:.0f}, f = {3.91/(kt_*w0_)*100:.2f}%, peak ratio = {4.59/(kt_*w0_)*100:.2f}% -> {1/(4.59/(kt_*w0_)):.0f}x power")
# light sheet
lam0 = 488e-9; nw = 1.33
print(f"Gaussian sheet: t=1um -> 2zR = {4.5*nw*(1e-6)**2/lam0*1e6:.1f} um; for 100 um FOV, t_min = {np.sqrt(100e-6*lam0/(4.5*nw))*1e6:.2f} um")
km = 2*np.pi*nw/lam0; sin_t = 0.40/nw; ktl = km*sin_t; tan_t = sin_t/np.sqrt(1-sin_t**2)
zml = 50e-6; w0l = zml*tan_t; Gl = zml/(km*((2.253/ktl)/1.1774)**2/2); fl = 3.91/(ktl*w0l)
print(f"Light sheet: sin(theta) = {sin_t:.3f}, kt = {ktl:.3e}, FWHM = {2.253/ktl*1e6:.2f} um, Gaussian zR = {km*((2.253/ktl)/1.1774)**2/2*1e6:.2f} um, w0 = {w0l*1e6:.1f} um, N = {ktl*w0l/np.pi:.0f}, G = {Gl:.0f}, f = {fl*100:.1f}%")
import json, os
res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "results.json")))["ls"]  # from fig_numeric.py
print(f"   sheet FWHM = {res['sheet_fwhm']*1e6:.2f} um; energy in core-FWHM stripe {res['frac_in_fwhm']*100:.1f}%, in |x|<r0 {res['frac_in_r0']*100:.1f}%  (from fig_numeric.py; run it first)")
# machining
Rm = 1e-3; lm = 1.03e-6
print(f"Machining: 1.3 R/lambda = {1.3*Rm/lm:.0f}; exact zmax/2r0 at sin(theta)=0.05 = {(Rm/np.tan(np.arcsin(0.05)))/(2*j01/(2*np.pi/lm*0.05)):.0f}; J0^2(3.83) = {sp.j0(3.8317)**2:.3f}; 1/0.162 = {1/sp.j0(3.8317)**2:.1f}")
