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
        m, n0 = self.A.shape

        index1 = np.zeros(0, dtype=int)
        index2 = np.zeros(0, dtype=int)
        index3 = np.zeros(0, dtype=int)
        index4 = np.zeros(0, dtype=int)
        
        n = np.zeros(4, dtype=int)
        
        for i in range(1, n0 + 1):
            if self.lo[i - 1] <= -np.inf:
                if self.hi[i - 1] >= np.inf:
                    n[0] += 1
                    index1 = np.append(index1, i)
                else:
                    n[2] += 1
                    index3 = np.append(index3, i)
            else:
                print(self.hi[i - 1], np.inf)
                if self.hi[i - 1] >= np.inf:
                    n[1] += 1
                    index2 = np.append(index2, i)
                else:
                    n[3] += 1
                    index4 = np.append(index4, i)
                    
        cs = np.concatenate((self.c[index1 - 1], -self.c[index1 - 1], self.c[index2 - 1],
                            -self.c[index3 - 1], self.c[index4 - 1], np.zeros(n[3])))

        print(index1, self.A[:, index2 - 1])
        As = np.concatenate((self.A[:, index1 - 1], -self.A[:, index1 - 1], self.A[:, index2 - 1], 
                            -self.A[:, index3 - 1], self.A[:, index4 - 1]), axis=0)
        As = np.concatenate((As, np.zeros((n[3], 2*n[0] + n[1] + n[2])), np.eye(n[3]), np.eye(n[3])), axis=1)

        bs = np.concatenate((self.b - self.A[:, index2 - 1] @ self.lo[index2 - 1] - self.A[:, index3 - 1] @ self.hi[index3 - 1] -
                            self.A[:, index4 - 1] @ self.lo[index4 - 1],
                            self.hi[index4 - 1] - self.lo[index4 - 1]))

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
        # result = self.convert_to_standard()
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
         [0, 0]
        ])
    b = np.array([2, 7, 3, 0])
    c = np.array([-1, -2])
    
    lo = np.array([10, 0])
    hi = np.array([0, 200])
                   
    P = LPProblem(A, b, c, lo, hi)
    P.internal_point(tolerance=1)