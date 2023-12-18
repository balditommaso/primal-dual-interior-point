import numpy as np

def starting_point(A, b, c):
    AA = A @ A.T
        
    x = A.T @ np.linalg.inv(AA)
    x = x @ b

    lam = A @ c
    lam = np.linalg.inv(AA) @ lam
        
    s = A.T @ lam
    s = c - s
    
    # hat
    dx = np.maximum(-1.5 * np.min(x), 0.0)
    ds = np.maximum(-1.5 * np.min(s), 0.0)
    
    
    x = x + dx
    s = s + ds
        
    xs = x @ s 
    
    dx = xs / np.sum(s)
    ds = xs / np.sum(x)
    
    x = x + dx
    s = s + ds
    
    return x, lam, s