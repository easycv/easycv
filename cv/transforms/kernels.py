import numpy as np
import scipy.stats as st


def uniform_kernel(size=21):
    """Returns a 2D uniform kernel."""
    return np.ones((size, size)) / size**2


def gaussian_kernel(size=21, sig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sig, sig, size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/np.sum(kern2d)

