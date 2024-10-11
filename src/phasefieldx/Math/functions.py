"""
Functions
=========

This module provides utility functions for computing the Macaulay brackets, which are commonly used in various mathematical and engineering applications. The Macaulay bracket is a piecewise function that extracts the positive or negative part of a real number. These functions are useful for handling operations that involve conditional expressions in a continuous and differentiable manner.

"""


def macaulay_bracket_positive(x):
    """
    Compute the Macaulay bracket (positive part) of a real number x.

    Parameters
    ----------
    x : float or numpy.ndarray
        Real number or array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        Macaulay bracket of x, which is defined as 0.5 * (x + abs(x)).
    """
    return 0.5 * (x + abs(x))


def macaulay_bracket_negative(x):
    """
    Compute the Macaulay bracket (negative part) of a real number x.

    Parameters
    ----------
    x : float or numpy.ndarray
        Real number or array of real numbers.

    Returns
    -------
    float or numpy.ndarray
        Macaulay bracket of x, which is defined as 0.5 * (x - abs(x)).
    """
    return 0.5 * (x - abs(x))
