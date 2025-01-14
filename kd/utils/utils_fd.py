"""Finite difference utilities for numerical differentiation.

This module provides utilities for computing numerical derivatives using finite difference methods.
It supports derivatives up to 4th order with appropriate boundary handling.

Example:
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x)
    >>> dx = x[1] - x[0]
    >>> dy_dx = FiniteDiff(y, dx, order=1)  # First derivative
    >>> d2y_dx2 = FiniteDiff(y, dx, order=2)  # Second derivative
"""

import numpy as np

def FiniteDiff(u, dx, order=1):
    """Calculate finite difference approximation of derivatives.
    
    This function computes numerical derivatives using central difference formulas
    for interior points and one-sided differences for boundary points.
    
    Supported orders:
        1: First derivative (central difference)
        2: Second derivative
        3: Third derivative
        4: Fourth derivative
    
    Args:
        u (ndarray): Input array
        dx (float): Grid spacing
        order (int): Order of derivative (default: 1)
        
    Returns:
        ndarray: Finite difference approximation
        
    Notes:
        - For order=1: Uses central difference for interior points and
          one-sided 2nd order formulas for boundaries
        - For order=2: Uses central difference for interior points and
          one-sided 3rd order formulas for boundaries
        - For orders 3 and 4: Sets boundary points to zero
        
    Example:
        >>> x = np.linspace(0, 1, 100)
        >>> u = x**2  # Function to differentiate
        >>> dx = x[1] - x[0]
        >>> dudx = FiniteDiff(u, dx)  # Should approximate 2*x
    """
    n = u.size
    ux = np.zeros(n)
    
    if order == 1:
        for i in range(1, n-1):
            ux[i] = (u[i+1] - u[i-1]) / (2*dx)
        ux[0] = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)
        ux[n-1] = (3*u[n-1] - 4*u[n-2] + u[n-3]) / (2*dx)
    elif order == 2:
        for i in range(1, n-1):
            ux[i] = (u[i+1] - 2*u[i] + u[i-1]) / (dx**2)
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / (dx**2)
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / (dx**2)
    elif order == 3:
        for i in range(2, n-2):
            ux[i] = (-u[i+2] + 2*u[i+1] - 2*u[i-1] + u[i-2]) / (2*dx**3)
        ux[0:2] = 0
        ux[n-2:] = 0
    elif order == 4:
        for i in range(2, n-2):
            ux[i] = (u[i+2] - 4*u[i+1] + 6*u[i] - 4*u[i-1] + u[i-2]) / (dx**4)
        ux[0:2] = 0
        ux[n-2:] = 0
        
    return ux 