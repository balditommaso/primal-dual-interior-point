import numpy as np

def alpha_max(x, dx, hi):
    """
    compute 
        arg max{ alpha in [0,hi] | x + alpha*dx >=0 }

    """
    n = x.shape[0]
    
    alpha = -1.0
    for i in range(n):
        if dx[i] < 0:
            a = x[i] / -dx[i]
            
            if alpha < 0:
                alpha = a
            else:
                alpha = min(alpha, a)
                
    if alpha < 0:
        alpha = np.inf
        
    alpha = min(alpha, hi)
    
    return alpha