import numpy as np
from solver import standard_lp

def is_not_feasible(A, b):
    m, n = A.shape
    A = np.hstack((A, np.eye(m)), dtype=np.float64)
    c = np.hstack((np.zeros(n), np.ones(m)), dtype=np.float64)
    x1, lam1, s1, flag, iter = standard_lp.solve_standard_lp(A, b, c)
    if np.abs(np.dot(c, x1)) > 1e-9:
        return True
    
    return False
    