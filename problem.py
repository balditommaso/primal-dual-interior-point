import numpy as np
from standard_lp import solve_standard_lp
from standard_qp import solve_standard_qp
from variant_qp import solve_variant_qp
from plot import plot_LMOLP_3d

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
        m, n = self.A.shape
        
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
                
        # presolve stage
        self.presolve()
        
        print(self)
        
        # solve the original problem
        return solve_standard_lp(self.A, 
                                 self.b, 
                                 self.c, 
                                 tolerance=tolerance, 
                                 max_it=max_it)
        
        
class QPProblem:
    def __init__(self, A, b, c, Q, lo=None, hi=None):
        self.A = A
        self.b = b
        self.c = c
        self.Q = Q
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
               f"Q {self.Q.shape}:\n{self.Q}\n" \
               f"lo {self.lo.shape}:\n{self.lo}\n" \
               f"hi {self.hi.shape}:\n{self.hi}"

            
    def internal_point(self, tolerance, max_it=100):
        
        m, n = self.A.shape
        assert self.b.shape[0] == m
        assert self.c.shape[0] == n 
        assert self.lo.shape[0] == n 
        assert self.hi.shape[0] == n
        assert self.Q.shape[0] == self.Q.shape[1]
        
        print(f"Problem size: {m}, {n}")
        print(self)
        
        # solve the original problem
        return solve_standard_qp(self.A, 
                                 self.b, 
                                 self.c, 
                                 self.Q, 
                                 tolerance=tolerance, 
                                 max_it=max_it)
    
    
class LMOPProblems:
    
    def __init__(self, A, b, objectives, lo=None, hi=None):
        self.A = A
        self.b = b
        self.objectives = objectives
        self.lo = lo
        self.hi = hi
        if lo is None:
            self.lo = np.full_like(A.shape[1], -np.inf, dtype=np.float64)
        if hi is None:
            self.hi = np.full_like(A.shape[1], np.inf, dtype=np.float64)

            
    def __str__(self) -> str:
        text = 'Objective function sorted:\n'
        for index, obj in enumerate(self.objectives, 1):
            text += f"[#{index}]\nlinear\n{obj['c']}\nquadratic\n{obj['Q']}\n"
        
        return text + f"A {self.A.shape}:\n{self.A}\n" \
               f"b {self.b.shape}:\n{self.b}\n" \
               f"lo {self.lo.shape}:\n{self.lo}\n" \
               f"hi {self.hi.shape}:\n{self.hi}"


    @staticmethod
    def update_constraints(A, b, new_row, value, n_col, tolerance=1e-6):
        '''
        This method is used to update the constraints of the preemptive approach 
        by checking if the new constraints it's not linearly dependant respect to 
        the existing ones, if the new constrain is linear dependent respect an 
        already existing one, I can just update it. 
        '''
        norm_new_row = np.linalg.norm(new_row[:n_col])
        for i in range(A.shape[0]):
            print(A[i, :n_col], new_row[:n_col])
            inner_prod = np.abs(np.inner(A[i, :n_col], new_row[:n_col]))
            norm_row_i = np.linalg.norm(A[i, :n_col])
            diff = float(abs(inner_prod - norm_row_i * norm_new_row))
            print(inner_prod, norm_new_row * norm_row_i)
            if diff < tolerance:
                print(f"The row {i} is liner dependent respect to the new row")
                # update the already existing constraint
                A[i] = np.hstack([A[i, :n_col], np.zeros(A.shape[1]-n_col)])
                # b[i] = value
                return A, b, True
                
                
        print("There are no constraints liner dependent respect to the new one")
        A = np.vstack([A, new_row])
        b = np.hstack([b, value])
        return A, b, False


    def internal_point_preemptive(self, tolerance, max_it=100):
        '''
        The idea is to apply a preemptive approach, which usually has problem,
        with quadratic objectives because if they are added as constrains we 
        cannot use anymore IPM, the idea is to add the as constrain the tangent
        plane in the best solution of that iteration  
        '''
        # convert to standard form 
        m, n = self.A.shape
        self.A = np.hstack([self.A, np.eye(m)])
        
        assert self.b.shape[0] == m
        assert len(self.objectives) > 0
        
        print(self)
        
        # init the variables
        iter = 0
        steps = []
        flag = True
        x = lam = s = None
        start = None
        
        # preemptive iteration
        for index, obj in enumerate(self.objectives, 1):
            print('-'*80)
            print(f"Solving #{index} objective")
            
            # convert to standard form
            obj['c'] = np.concatenate((obj['c'], np.zeros((m,))))
            if obj['Q'] is not None:
                assert obj['Q'].shape[0] == obj['Q'].shape[1]
                obj['Q'] = np.pad(obj['Q'], (0, m), 'constant', constant_values=0)
                
            # solve the problem
            if obj['Q'] is not None:
                x, lam, s, flag, n_iter, points = solve_standard_qp(self.A, 
                                                                    self.b,
                                                                    obj['c'], 
                                                                    obj['Q'], 
                                                                    tolerance=tolerance, 
                                                                    max_it=max_it,
                                                                    start=start)
                # grad = Qx + c
                grad = obj['Q'] @ x + obj['c']
                # solution
                solution = 0
                for i in range(x.shape[0]):
                    solution += grad[i] * x[i]
            else:
                x, lam, s, flag, n_iter, points = solve_standard_lp(self.A,
                                                                    self.b,
                                                                    obj['c'],
                                                                    tolerance=tolerance,
                                                                    max_it=max_it,
                                                                    start=start)
                solution = np.sum(obj['c'] * x) 
            iter += n_iter
            steps.extend(points)
            
            # not converged
            if not flag or iter >= max_it:
                print(f"Algorithm not converged at the {index} objective function")
                return x, lam, s, False, iter, steps
                        
            # update constraints
            if obj['Q'] is not None:
                # self.A, self.b, restart = self.update_constraints(self.A, self.b, grad, solution, n)
                self.A = np.vstack([self.A, grad])
            else:
                # self.A, self.b, restart = self.update_constraints(self.A, self.b, obj['c'], solution, n)
                self.A = np.vstack([self.A, obj['c']])
                
            self.b = np.hstack([self.b, solution])
            # if restart:
            #     start = (x, lam, s)
            # else:
            #     start = None
            # print(self)
        
        return x, lam, s, True, iter, steps
    
    
    def internal_point_penality(self, tolerance, max_it=100):
        '''
        The idea of this approach is to rewrite the objective functions 
        in a single objective function which incrementally activate the 
        directions and the penalities
        '''
        m, n = self.A.shape
        # standard form
        self.A = np.hstack([self.A, np.eye(m), np.zeros((m, len(self.objectives) - 1))])
        for obj in self.objectives:
            obj['c'] = np.concatenate((obj['c'], np.zeros((m + len(self.objectives) - 1,))))
            if obj['Q'] is not None:
                assert obj['Q'].shape[0] == obj['Q'].shape[1]
                obj['Q'] = np.pad(obj['Q'], (0, m + len(self.objectives) - 1), 
                                  'constant', 
                                  constant_values=0)
            else:
                obj['Q'] = np.zeros((obj['c'].shape[0], obj['c'].shape[0]))
        
        # check the dimensions
        assert self.b.shape[0] == m
        assert len(self.objectives) > 0
        
        print(self)
        
        # solve the problem by updating the objective functions
        x, lam, s, flag, n_iter, points = solve_variant_qp(self.A, 
                                                            self.b,
                                                            self.objectives, 
                                                            tolerance=tolerance, 
                                                            max_it=max_it)

        return x, lam, s, flag, n_iter, points
        
