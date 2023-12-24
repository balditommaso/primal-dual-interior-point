from utils.utils import alpha_max, starting_point
import numpy as np
import scipy

def solve_standard_qp(A, b, c, Q, tolerance=1e-9, max_it=100):
    m, n = A.shape
    points = []
    
    x0, lam0, s0 = starting_point(A, b, c)
    
    for iter in range(max_it+1):
        print('-'*80)
        print(f'iter [{iter}]:\nx:\t{x0},\nlam:\t{lam0},\ns:\t{s0}')
        points.append(x0)
        
        f3, pivots = fact3(A, x0, s0, Q)
        
        rb = A @ x0 - b
        rc = A.T @ lam0 + s0 - Q @ x0 - c
        rxs = x0 * s0
        lam_aff, x_aff, s_aff = solve3(f3, pivots, rb, rc, rxs)
        
        # compute alpha_aff^pr, alpha_aff^dual, mu_aff
        alpha_aff_pri = alpha_max(x0, x_aff, 1.0)
        alpha_aff_dual = alpha_max(s0, s_aff, 1.0)
        
        mu = np.mean(rxs, dtype=np.float64)
        # calculate mu_aff
        mu_aff = np.dot(x0 + alpha_aff_pri * x_aff, s0 + alpha_aff_dual * s_aff) / n
        
        # centering parameter sigma
        sigma = (mu_aff/mu) ** 3
        
        rb = np.zeros((m,))
        rc = np.zeros((n,))
        rxs = x_aff * s_aff - sigma * mu
        
        lam_cc, x_cc, s_cc = solve3(f3, pivots, rb, rc, rxs)
        
        # compute the search direction step boundaries
        dx = x_aff + x_cc
        dlam = lam_aff + lam_cc
        ds = s_aff + s_cc
        
        alpha_max_pri = alpha_max(x0, dx, np.inf)
        alpha_max_dual = alpha_max(s0, ds, np.inf)
        
        alpha_pri = min(0.99 * alpha_max_pri, 1)
        alpha_dual = min(0.99 * alpha_max_dual, 1)
        
        if alpha_pri > 1e308 or alpha_dual > 1e308:
            print("this problem is unbounded")
            return x0, lam0, s0, False, iter, points

        x1 = x0 + alpha_pri * dx
        lam1 = lam0 + alpha_dual * dlam
        s1 = s0 + alpha_dual * ds
        
        # termination
        r1 = np.linalg.norm(A @ x1 - b) / (1 + np.linalg.norm(b))
        if r1 < tolerance:
            r2 = np.linalg.norm(A.T @ lam1 + s1 - Q @ x1 - c) / (1 + np.linalg.norm(c))
            
            if r2 < tolerance:
                r3 = mu / (1 + np.abs(0.5 * x1.T @ Q @ x1 + np.dot(c, x1)))
                
                if r3 < tolerance:
                    return x1, lam1, s1, True, iter, points
                
        if iter == max_it:
            return x1, lam1, s1, False, max_it, points
        
        x0 = x1
        lam0 = lam1
        s0 = s1
        
        
        
def fact3(A, x, s, Q):
    m, n = A.shape
    
    S = np.zeros((s.shape[0], s.shape[0]))
    np.fill_diagonal(S, s)
    X = np.zeros((x.shape[0], x.shape[0]))
    np.fill_diagonal(X, x)
    
    M1 = np.hstack((np.zeros((m, m)), A, np.zeros((m, n))))
    M2 = np.hstack((A.T, -Q, np.eye(n,n)))
    M3 = np.hstack((np.zeros((n, m)), S, X))
    M = np.vstack((M1, M2, M3))
    
    f, pivots = scipy.linalg.lu_factor(M)

    return f.astype(np.float64), pivots


def solve3(f, pivots, rb, rc, rxs):
    m = rb.shape[0]
    n = rc.shape[0]
    
    b = np.hstack((-rb, -rc, -rxs), dtype=np.float64)
    # Solve the linear system of equations A * x = b
    b = scipy.linalg.lu_solve((f, pivots), b.T)

    # Extract the solution into separate arrays for dlam, dx, and ds
    dlam = b[:m]
    dx = b[m:m+n]
    ds = b[m+n:]
    # Return the solutions
    return dlam.astype(np.float64), dx.astype(np.float64), ds.astype(np.float64)