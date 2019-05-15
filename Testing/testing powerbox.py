from pyfftw.interfaces.numpy_fft import fftn as _fftn, ifftn as _ifftn, ifftshift as _ifftshift, fftshift as _fftshift, \
    fftfreq as _fftfreq
from pyfftw.interfaces.cache import enable, set_keepalive_time
import numpy as np

THREADS = 1
def fftn(*args, **kwargs):
    return _fftn(threads=THREADS, *args, **kwargs)


def ifftn(*args, **kwargs):
    return _ifftn(threads=THREADS, *args, **kwargs)

def fftshift(x,*args,**kwargs):
    out = _fftshift(x,*args,**kwargs)

    if hasattr(x,"unit"):
        return out*x.unit
    else:
        return out



def ifftshift(x, *args, **kwargs):
    out = _ifftshift(x, *args, **kwargs)

    if hasattr(x, "unit"):
        return out * x.unit
    else:
        return out


def fftfreq(N,d=1.0,b=2*np.pi):
    """
    Return the fourier frequencies for a box with N cells, using general Fourier convention.

    Parameters
    ----------
    N : int
        The number of grid cells

    d : float, optional
        The interval between cells

    b : float, optional
        The fourier-convention of the frequency component (see :mod:`powerbox.dft` for details).

    Returns
    -------
    freq : array
        The N symmetric frequency components of the Fourier transform. Always centred at 0.

    """
    return fftshift(_fftfreq(N, d=d))*(2*np.pi/b)



image = np.zeros((501,501))
image[250,250] = 200

X = image
Lk = None

axes = (0,1)
L = 2

a=0
b=2*np.pi
if axes is None:
    axes = list(range(len(X.shape)))

N = np.array([X.shape[axis] for axis in axes])

# Get the box volume if given the fourier-space box volume
if L is None and Lk is None:
    L = N
elif L is not None:  # give precedence to L
    if np.isscalar(L):
        L = L * np.ones(len(axes))
elif Lk is not None:
    if np.isscalar(Lk):
        Lk = Lk * np.ones(len(axes))
    L = N * 2 * np.pi / (Lk * b)  # Take account of the fourier convention.

V = float(np.product(L))  # Volume of box
Vx = V / np.product(N)  # Volume of cell

ft = Vx * fftshift(fftn(X, axes=axes), axes=axes) * np.sqrt(np.abs(b) / (2 * np.pi) ** (1 - a)) ** len(axes)

dx = np.array([float(l) / float(n) for l, n in zip(L, N)])

freq = np.array([fftfreq(n, d=d, b=b) for n, d in zip(N, dx)])

print(np.abs(ft))