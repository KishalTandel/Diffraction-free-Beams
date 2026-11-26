"""Shared numerical tools: scalar angular-spectrum propagation (2D FFT) and 1D paraxial propagation."""
import numpy as np

def grid2d(N, dx):
    x = (np.arange(N) - N//2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return x, X, Y

def propagate_as(E, dx, k, dz):
    """Exact scalar angular-spectrum propagation by dz (evanescent waves dropped)."""
    N = E.shape[0]
    kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, kx, indexing='xy')
    kz2 = k**2 - KX**2 - KY**2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    H = np.exp(1j*kz*dz) * (kz2 > 0)
    return np.fft.ifft2(np.fft.fft2(E) * H)

def propagate_1d_paraxial(psi, ds, dxi):
    """i psi_xi + 1/2 psi_ss = 0  ->  psi(xi+dxi)."""
    N = psi.size
    kap = 2*np.pi*np.fft.fftfreq(N, d=ds)
    return np.fft.ifft(np.fft.fft(psi) * np.exp(-0.5j*kap**2*dxi))
