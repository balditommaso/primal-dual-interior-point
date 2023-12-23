import numpy as np

import matplotlib.pyplot as plt
from utils import feasibility
from solver import standard_lp

class LPProblem:
    
    def __init__(self, A, b, c, lo=None, hi=None):
        self.A = A
        self.b = b
        self.c = c
        self.lo = lo
        self.hi = hi
        if lo is None:
            self.lo = np.full_like(c, -np.inf, dtype=np.float64)
        if hi is None:
            self.hi = np.full_like(c, np.inf, dtype=np.float64)
        
        
    def __str__(self) -> str:
        return f"A {self.A.shape}:\n{self.A}\n" \
               f"b {self.b.shape}:\n{self.b}\n" \
               f"c {self.c.shape}:\n{self.c}\n" \
               f"lo {self.lo.shape}:\n{self.lo}\n" \
               f"hi {self.hi.shape}:\n{self.hi}"
    
    
    def presolve(self):
        '''
        Presolve the function
        '''
        m, n = A.shape
        
        # zero rows and columns in A
        for i in range(m):
            # check empty row
            if not np.any(self.A[i, :]):
                if self.b[i] == 0.0:
                    # delete empty row
                    self.A = np.delete(self.A, i, axis=0)
                    self.b = np.delete(self.b, i)
                else:
                    print("This problem is infeasible")
                    return False
                
        for j in range(n):
            # check empty column
            if not np.any(self.A[:, j]):
                self.A = np.delete(self.A, i, 1)
                
                
        # TODO: check uniquness of col and row
        return True
        
        
    def convert_to_standard(self):
        """
        conver the lp problem
        min c^T*x
        s.t. Ax = b
        lo <= x <= hi

        to an lp in standard form
        min c^T*x
        s.t. Ax = b
        x >= 0
        """

        inf = 1e300
        
        m, n0 = self.A.shape
        print(np.inf > 1e300)
        index1 = np.where((self.lo < -inf) & (self.hi > inf))[0] + 1
        index2 = np.where((self.lo >= -inf) & (self.hi > inf))[0] + 1
        index3 = np.where((self.lo < -inf) & (self.hi <= inf))[0] + 1
        index4 = np.where((self.lo >= -inf) & (self.hi <= inf))[0] + 1

        n = np.array([len(index1), len(index2), len(index3), len(index4)])

        cs = np.concatenate([np.take(self.c, index1-1), -np.take(self.c, index1-1),
                             np.take(self.c, index2-1), -np.take(self.c, index3-1),
                             np.take(self.c, index4-1), np.zeros(n[3])], dtype=np.float64)

        As = np.vstack([
                    np.hstack([np.take(self.A, index1-1, axis=1), -np.take(self.A, index1-1, axis=1),
                            np.take(self.A, index2-1, axis=1), -np.take(self.A, index3-1, axis=1),
                            np.take(self.A, index4-1, axis=1), np.zeros((m, n[3]))]),
                    np.hstack([np.zeros((n[3], 2*sum(n[:3]))), np.eye(n[3]), np.eye(n[3])])
                ], dtype=np.float64)
        
        bs = np.concatenate([self.b - np.dot(self.A[:, index2-1], self.lo[index2-1]) -
                             np.dot(self.A[:, index3-1], self.hi[index3-1]) -
                             np.dot(self.A[:, index4-1], self.lo[index4-1]),
                             self.hi[index4-1] - self.lo[index4-1]], dtype=np.float64)

        self.A = As
        self.b = bs
        self.c = cs
        
        return index1, index2, index3, index4
        
        
    def internal_point(self, tolerance, max_it=100):
        
        m, n = self.A.shape
        assert self.b.shape[0] == m
        assert self.c.shape[0] == n 
        assert self.lo.shape[0] == n 
        assert self.hi.shape[0] == n
        
        print(f"Problem size: {m}, {n}")
                
        # presolve stage
        self.presolve()
        
        print(self)
        
        # solve the original problem
        return standard_lp.solve_standard_lp(self.A, self.b, self.c, tolerance=tolerance, max_it=max_it)
        
        
if __name__ == "__main__":
    
    # ---------------------------------------------------------------------------- #
    #                                    PROBLEM                                   #
    # ---------------------------------------------------------------------------- #
    A = np.array(
        [[ 2,  1],
         [ 2,  3],
         [ 4,  3],
         [-1, -2],
        ], dtype=np.float64)
    b = np.array([120, 210, 270, -60], dtype=np.float64)
    c = np.array([-10, -14], dtype=np.float64)
    
    # standard form
    A_std = np.hstack([A, np.eye(A.shape[0])])
    c_std = np.concatenate((c, np.zeros((A.shape[0],))))

    # ---------------------------------------------------------------------------- #
    #                                      IPM                                     #
    # ---------------------------------------------------------------------------- #
    P = LPProblem(A_std, b, c_std)
    x, lam, s, converge, n_iter, steps = P.internal_point(tolerance=1e-8)
    
    if converge:
        solution = 0.0
        for i in range(A.shape[1]):
            solution += c[i] * x[i]
        
        print(f'The Algorithm is converged after {n_iter} iterations.\n' \
              f'The Optimal solution is {solution}')
    else:
        print("The Algorithm is not converged to a solution!")
        
    
    # ---------------------------------------------------------------------------- #
    #                                     PLOT                                     #
    # ---------------------------------------------------------------------------- #
    plt.figure(figsize=(8, 6))
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            print(f"ax = {b[i]}")
        elif A[i][1] == 0:
            print(f"ay = {b[i]}")
        else:
            x = np.linspace(0, 100, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y)
        
    
    x_iter, y_iter, _, _, _, _ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    
    plt.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=2, color='green', label='Objective Function Direction')

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('LP Problem Constraints')
    plt.legend()
    plt.grid(True)
    plt.show()