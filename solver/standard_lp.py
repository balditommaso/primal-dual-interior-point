from utils import starting, utils
import numpy as np
import scipy

def solve_standard_lp(A, b, c, max_it=100, tolerance=1e-8, verbose=False):
    gamma_f = 0.01
    
    scaling = 1
    
    m, n = A.shape
    
    # compute initial value
    
    x0, lam0, s0 = starting.starting_point(A,b,c)
    
    for iter in range(max_it):
        f3 = fact3(A, x0, s0)
        rb = A @ x0 - b
        rc = A.T @ lam0 + s0 - c
        rxs = x0 * s0
        lam_aff, x_aff, s_aff = solve3(f3, rb, rc, rxs)
        
        # compute alpha_aff^pr, alpha_aff^dual, mu_aff
        alpha_aff_pri = utils.alpha_max(x0, x_aff, 1.0)
        alpha_aff_dual = utils.alpha_max(s0, s_aff, 1.0)
        
        mu = np.mean(rxs)
        
        # Calculate mu_aff
        mu_aff = np.dot(x0 + alpha_aff_pri * x_aff, s0 + alpha_aff_dual * s_aff) / n

        
        # centering parameter sigma
        
        sigma = (mu_aff/mu) ** 3
        
        rb = np.zeros((m,))
        rc = np.zeros((n,))
        rxs = x_aff * s_aff - sigma * mu
        
        lam_cc, x_cc, s_cc = solve3(f3, rb, rc, rxs)
        
        # compute the search direction step bounderies
        dx = x_aff + x_cc
        dlam = lam_aff + lam_cc
        ds = s_aff = s_cc
        
        alpha_max_pri = utils.alpha_max(x0, dx, np.inf)
        alpha_max_dual = utils.alpha_max(s0, ds, np.inf)
        
        if scaling == 0:
            alpha_pri = min(0.99 * alpha_max_pri, 1)
            alpha_dual = min(0.99 * alpha_max_dual, 1)
        else:
            x1_pri = x0 + alpha_max_pri * dx
            s1_dual = s0 + alpha_max_dual * ds
            mu_p = (x1_pri @ s1_dual) / n
            
            xind = np.argmin(x1_pri)
            
            f_pri = (gamma_f * mu_p / s1_dual[xind] - x0[xind]) / alpha_max_pri / dx[xind]
            sind = np.argmin(s1_dual)
            
            f_dual = (gamma_f * mu_p / x1_pri[sind] - s0[sind]) / alpha_max_dual / ds[sind]
            
            alpha_pri = max(1 - gamma_f, f_pri) * alpha_max_pri
            alpha_dual = max(1 - gamma_f, f_dual) * alpha_max_dual
            
        
        if alpha_pri > 1e308 or alpha_dual > 1e308:
            # TODO: check this part
            print("this problem is unbounded")
            return False

        x1 = x0 + alpha_pri * dx
        lam1 = lam0 + alpha_dual * dlam
        s1 = s0 + alpha_dual * ds
        
        # termination
        r1 = np.linalg.norm(A @ x1 - b) / (1 + np.linalg.norm(b))
        
        if r1 < tolerance:
            r2 = np.linalg.norm(A.T @ lam1 + s1 - c) / (1 + np.linalg.norm(c))
            
            if r2 < tolerance:
                cx = c @ x1
                r3 = np.abs(cx - b @ lam1) / (1 + np.abs(cx))
                
                if r3 < tolerance:
                    return x1, lam1, s1, True, iter
                
        if iter == max_it:
            return x1, lam1, s1, False, max_it

        x0 = x1
        lam0 = lam1
        s0 = s1
        
        break # DEBUG
    

def fact3(A, x, s):
    m, n = A.shape
    
    S = np.zeros((s.shape[0], s.shape[0]))
    np.fill_diagonal(S, s)
    X = np.zeros((x.shape[0], x.shape[0]))
    np.fill_diagonal(X, x)
    
    # permutation of rows and columns
    M1 = np.hstack((np.zeros((m, m)), A, np.zeros((m, n))))
    M2 = np.hstack((A.T, np.zeros((n, n)), np.eye(n,n)))
    M3 = np.hstack((np.zeros((n, m)), S, X))
    M = np.vstack((M1, M2, M3))
    
    f, _ = scipy.linalg.lu_factor(M)

    return f

def solve3(f, rb, rc, rxs):
    m = rb.shape[0]
    n = rc.shape[0]
    
    b = np.hstack((-rb, -rc, -rxs))
    # Solve the linear system of equations A * x = b
    b = np.linalg.solve(f, b.T)

    # Extract the solution into separate arrays for dlam, dx, and ds
    dlam = b[:m]
    dx = b[m:m+n]
    ds = b[m+n:]

    # Return the solutions
    return dlam, dx, ds