"""How much does a small obstacle (the first trapped particle) perturb the beam at a second plane?
Opaque disc of radius a on the axis at z1 in the Bessel-Gauss/axicon beam of the paper (lambda = 632.8 nm,
w0 = 1 mm, sin(theta) = 8e-3, core radius r0 = 30 um).  The disc is scaled to a/r0 ~ 0.5, like a 1 um particle in a
theta = 0.1 trap (r0 = 2.4 um).  Prints the on-axis intensity with/without the disc, and the ray-optics
estimate of the intercepted power fraction 8 a r_s exp(-2 r_s^2/w0^2)/w0^2."""
import numpy as np
from scipy import special as sp
from sim import grid2d, propagate_as

lam = 632.8e-9; k = 2*np.pi/lam; w0 = 1e-3; sth = 8e-3; theta = np.arcsin(sth); kt = k*sth
tth = np.tan(theta); zmax = w0/tth; N = 2048; dx = 4e-6
x, X, Y = grid2d(N, dx); R = np.hypot(X, Y); h = N//2
r0 = sp.jn_zeros(0, 1)[0]/kt
z1 = 0.030; dz = 2e-3
E0 = np.exp(-R**2/w0**2)*np.exp(-1j*kt*R)           # axicon
for a in (16e-6, 40e-6):
    zheal = a/tth
    Eu = E0.copy(); Eo = E0.copy(); zs = []; Iu = []; Io = []
    for i in range(1, 91):
        Eu = propagate_as(Eu, dx, k, dz); Eo = propagate_as(Eo, dx, k, dz)
        if abs(i*dz - z1) < 1e-9:
            Eo = Eo*(R >= a)
        zs.append(i*dz); Iu.append(abs(Eu[h, h])**2); Io.append(abs(Eo[h, h])**2)
    zs = np.array(zs); ratio = np.array(Io)/np.array(Iu)
    rs = z1*tth; frac = 8*a*rs*np.exp(-2*rs**2/w0**2)/w0**2
    print(f"a = {a*1e6:.0f} um (a/r0 = {a/r0:.2f}), z_heal = a/tan(theta) = {zheal*1e3:.1f} mm; ray estimate of intercepted power = {frac*100:.2f}% (max {2.43*a/w0*100:.2f}%)")
    for dzz in (1, 2, 3, 4, 6):
        j = np.argmin(abs(zs - (z1 + dzz*zheal)))
        print(f"   z - z1 = {dzz} z_heal ({(zs[j]-z1)*1e3:.1f} mm): I_obstructed / I_free on axis = {ratio[j]:.3f}")
    far = (zs > z1 + 6*zheal) & (zs < 0.12)
    print(f"   for z > z1 + 6 z_heal (up to 120 mm): ratio between {ratio[far].min():.3f} and {ratio[far].max():.3f}")
