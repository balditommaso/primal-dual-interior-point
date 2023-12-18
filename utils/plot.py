import numpy as np
import matplotlib.pyplot as plt

# # Define the constraints (replace these with your actual LP problem constraints)
# # For example, let's consider two constraints in 2D: 2x + y <= 10 and x + 3y <= 12

# def plot_constraints_2d(A, b):
#     assert A.shape[0] == b.shape[0] and \
#            A.shape[1] == 2
    
#     m, n = A.shape

# # Constraint 1: 2x + y <= 10
# x_values = np.linspace(0, 10, 100)
# y_constraint1 = 10 - 2 * x_values

# # Constraint 2: x + 3y <= 12
# y_values = np.linspace(0, 10, 100)
# x_constraint2 = 12 - 3 * y_values

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, y_constraint1, label=r'$2x + y \leq 10$')
# plt.plot(x_constraint2, y_values, label=r'$x + 3y \leq 12$')

# # Additional plot settings
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('LP Problem Constraints')
# plt.legend()
# plt.grid(True)
# plt.show()
