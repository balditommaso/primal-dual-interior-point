import numpy as np
from scipy.linalg import cho_factor, cho_solve


def alpha_max(x, dx, hi):
    """
    compute 
        arg max{ alpha in [0,hi] | x + alpha*dx >=0 }

    """
    n = x.shape[0]
    
    alpha = -1.0
    for i in range(n):
        if dx[i] < 0:
            a = -x[i] / dx[i]
            if alpha < 0:
                alpha = a
            else:
                alpha = min(alpha, a)
                
    if alpha < 0:
        alpha = np.inf
        
    alpha = min(alpha, hi)
    
    return alpha


def starting_point_lp(A, b, c):
        
    AA = A @ A.T

    # Cholesky factorization
    f, lower = cho_factor(AA)
    
    # tilde
    x = cho_solve((f, lower), b)
    x = A.T @ x

    lam = A @ c
    lam = cho_solve((f, lower), lam)

    s = A.T @ lam
    s = c - s

    # hat
    dx = max(-1.5 * np.min(x), 0.0)
    ds = max(-1.5 * np.min(s), 0.0)

    x = x + dx
    s = s + ds

    xs = np.dot(x, s) / 2.0

    dx = xs / np.sum(s)
    ds = xs / np.sum(x)

    x = x + dx
    s = s + ds

    return x.astype(np.float64), lam.astype(np.float64), s.astype(np.float64)


def starting_point_qp(A, b, c, Q):
    
    AA = A @ A.T

    # Cholesky factorization
    f, lower = cho_factor(AA)
    
    # tilde
    x = cho_solve((f, lower), b)
    x = A.T @ x
        
    lam = A @ (c + Q @ x)
    lam = cho_solve((f, lower), lam)
    
    s = A.T @ lam
    s = c + Q @ x - s
    
    # hat
    dx = max(-1.5 * np.min(x), 0.0)
    ds = max(-1.5 * np.min(s), 0.0)

    x = x + dx
    s = s + ds

    xs = np.dot(x, s) / 2.0

    dx = xs / np.sum(s)
    ds = xs / np.sum(x)

    x = x + dx
    s = s + ds

    return x.astype(np.float64), lam.astype(np.float64), s.astype(np.float64)
