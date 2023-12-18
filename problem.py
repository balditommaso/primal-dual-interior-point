import numpy as np
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
                             np.take(self.c, index4-1), np.zeros(n[3])])

        As = np.vstack([
            np.hstack([np.take(self.A, index1-1, axis=1), -np.take(self.A, index1-1, axis=1),
                       np.take(self.A, index2-1, axis=1), -np.take(self.A, index3-1, axis=1),
                       np.take(self.A, index4-1, axis=1), np.zeros((m, n[3]))]),
            np.hstack([np.zeros((n[3], 2*sum(n[:3]))), np.eye(n[3]), np.eye(n[3])])
        ])
        
        bs = np.concatenate([self.b - np.dot(self.A[:, index2-1], self.lo[index2-1]) -
                             np.dot(self.A[:, index3-1], self.hi[index3-1]) -
                             np.dot(self.A[:, index4-1], self.lo[index4-1]),
                             self.hi[index4-1] - self.lo[index4-1]])

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
        
        # TODO: convert to standard form
        ind1, ind2, ind3, ind4 = self.convert_to_standard()
        print(self)
        # print(result)
        # detect infeasibility
        if feasibility.is_not_feasible(self.A, self.b):
            print("This problem is not feasible.")
            return
        
        # solve the original problem
        x1, lam1, s1, flag, iter = standard_lp.solve_standard_lp(A, b, c, max_it, tolerance)
        print(iter, flag)
        
        
if __name__ == "__main__":
    
    A = np.array(
        [[-2, 1],
         [-1, 2],
         [1, 0],
        ])
    b = np.array([2, 7, 3])
    c = np.array([-1, -2])
    
    lo = np.array([0, 0])
    hi = np.array([5, 5])
                   
    P = LPProblem(A, b, c)
    P.internal_point(tolerance=1e-8)