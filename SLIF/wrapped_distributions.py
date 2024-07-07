import numpy as np
from numba import njit

@njit(cache = True, fastmath = True)
def wrapped_cauchy_pdf(x, mu, scale):
    """
    Calculate the PDF of a wrapped Cauchy distribution.

    Parameters:
    - x: np.array, the angles (in radians) at which to evaluate the PDF.
    - mu: float, the location parameter of the distribution, corresponding to the peak (peak position).
    - scale: float, the scale parameter of the distribution (peak width).
    
    Returns:
    - pdf: np.array, the PDF values corresponding to x.
    """

    # wrap x and mu angles
    x = x % (2 * np.pi)
    mu = mu % (2 * np.pi)

    pdf = 1 / (2*np.pi) * np.sinh(scale) / (np.cosh(scale) - np.cos(x - mu))

    """
    with warnings.catch_warnings(record=True) as w:
        pdf = 1 / (2*np.pi) * np.sinh(scale) / (np.cosh(scale) - np.cos(x - mu))
        if len(w) > 0:
            print("Warning on value:")
            print(np.cosh(scale), scale)
    """
    
    return pdf

@njit(cache = True, fastmath = True)
def bessel_i0(scale):
    result = 1.0
    term = 1.0
    k = 1
    while term > 1e-10 * result:
        term *= (scale / (2 * k)) ** 2
        result += term
        k += 1
    return result

@njit(cache = True, fastmath = True)
def von_mises_pdf(x, mu, scale):
  
    # wrap x and mu angles
    x = x % (2 * np.pi)
    mu = mu % (2 * np.pi)

    I0_scale = bessel_i0(scale)
    return np.exp(scale * np.cos(x - mu)) / (2 * np.pi * I0_scale)

@njit(cache = True, fastmath = True)
def wrapped_laplace_pdf(x, mu, scale):
    """
    Compute the probability density function (PDF) of the wrapped Laplace distribution.
    
    Parameters:
        x (float or numpy array): Value(s) at which to evaluate the PDF.
        mu (float): Mean of the wrapped Laplace distribution.
        scale (float): Scale parameter of the wrapped Laplace distribution.
        
    Returns:
        float or numpy array: PDF value(s) corresponding to the input value(s) x.
    """

    # wrap x and mu angles
    x = x % (2 * np.pi)
    mu = mu % (2 * np.pi)

    tau = 2 * np.pi
    pdf = np.exp(-np.abs((x - mu) / scale)) / (2 * scale)
    for k in range(1, 10):
        pdf = pdf + np.exp(-np.abs((x - mu + 2*k*np.pi) / scale)) / (2 * scale)
    return pdf / (2 * scale)


@njit(cache = True, fastmath = True)
def distribution_pdf(x, mu, scale, distribution):
    if distribution == "wrapped_cauchy":
        return wrapped_cauchy_pdf(x, mu, scale)
    elif distribution == "von_mises":
        return von_mises_pdf(x, mu, scale)
    elif distribution == "wrapped_laplace":
        return wrapped_laplace_pdf(x, mu, scale)
    
    return wrapped_cauchy_pdf(x, mu, scale)