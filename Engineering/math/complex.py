"""
The original code is from: https://raw.githubusercontent.com/facebookresearch/fastMRI/main/fastmri/math.py meant for old Pytorch complex (last dim, real image seperately)
which has been modified for new PyTorch complex, and works for both PyTorch and Numpy
"""


def complex_mul(x, y):
    """
    Complex multiplication.
    """
    re = x.real * y.real - x.imag * y.imag
    im = x.real * y.imag + x.imag * y.real
    return re + 1j * im


def complex_conj(x):
    """
    Complex conjugate.
    """
    return x.real - 1j * x.imag


def complex_abs(x):
    """
    Compute the absolute value of a complex valued input tensor.
    """
    return complex_abs_sq(x).sqrt()


def complex_abs_sq(x):
    """
    Compute the squared absolute value of a complex tensor.
    """
    re = x.real ** 2
    im = x.imag ** 2
    return (re+im)
