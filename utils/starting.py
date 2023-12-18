import numpy as np
from scipy.linalg import cholesky

def starting_point(A, b, c):
    AA = A @ A.T

    # Cholesky factorization
    f = cholesky(AA)

    # Tilde
    x = np.linalg.solve(f, b)
    x = A.T @ x

    lam = A @ c
    lam = np.linalg.solve(f, lam)

    s = A.T @ lam
    s = c - s

    # Hat
    dx = np.maximum(-1.5 * np.min(x), 0.0)
    ds = np.maximum(-1.5 * np.min(s), 0.0)

    x = x + dx
    s = s + ds

    # ^0
    xs = np.dot(x, s) / 2.0

    dx = xs / np.sum(s)
    ds = xs / np.sum(x)

    x = x + dx
    s = s + ds

    return x.astype(np.float64), lam.astype(np.float64), s.astype(np.float64)