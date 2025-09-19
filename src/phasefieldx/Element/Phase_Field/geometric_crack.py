"""
Geometric Crack functions
=========================

"""

def geometric_crack_function(phi, case='AT2'):
    """
    Computes the geometric function alpha(phi) for various phase-field models.

    Parameters
    ----------
    phi : float or array_like
        Phase-field variable.
    case : str, optional
        The phase-field model. Supported values are 'AT2', 'AT1', 'WU', 'DOUBLE'.
        Default is 'AT2'.

    Returns
    -------
    float or array_like
        The value of the geometric function.
    """
    if case == 'AT2':
        return phi**2
    elif case == 'AT1':
        return phi
    elif case == 'WU':
        return 2.0 * phi - phi**2
    elif case == 'DOUBLE':
        return 16.0*phi**2 * (1 - phi)**2
    else:
        raise ValueError(f"Unknown case: {case}")

def geometric_crack_function_derivative(phi, case='AT2'):
    """
    Computes the derivative of the geometric function, alpha'(phi).

    Parameters
    ----------
    phi : float or array_like
        Phase-field variable.
    case : str, optional
        The phase-field model. Supported values are 'AT2', 'AT1', 'WU', 'DOUBLE'.
        Default is 'AT2'.

    Returns
    -------
    float or array_like
        The derivative of the geometric function.
    """
    if case == 'AT2':
        return 2 * phi
    elif case == 'AT1':
        return 1.0
    elif case == 'WU':
        return 2.0 - 2 * phi
    elif case == 'DOUBLE':
        return 32.0 * phi * (1 - phi) * (1 - 2 * phi)
    else:
        raise ValueError(f"Unknown case: {case}")

def geometric_crack_coefficient(case='AT2'):
    """
    Returns the geometric coefficient c_w for the crack surface density function.

    This coefficient ensures that the integral of the dissipation function w(phi)
    over the crack length equals 1.

    Parameters
    ----------
    case : str, optional
        The phase-field model. Supported values are 'AT2', 'AT1', 'Wu', 'DOUBLE'.
        Default is 'AT2'.

    Returns
    -------
    float
        The geometric coefficient.
    """
    if case == 'AT2':
        return 2.0
    elif case == 'AT1':
        return 8.0/3.0
    elif case == 'WU':
        return 3.141592653589793116
    elif case == 'DOUBLE':
        return 8.0/3.0
    else:
        raise ValueError(f"Unknown case: {case}")
