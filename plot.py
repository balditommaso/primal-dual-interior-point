import matplotlib.pyplot as plt
import numpy as np

def plot_LP(A, b, c, steps, max_scale=100):
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    plt.figure(figsize=(8, 6))
    
    # plot constraints
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            print(f"ax = {b[i]}")
        elif A[i][1] == 0:
            print(f"ay = {b[i]}")
        else:
            x = np.linspace(0, max_scale, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y)
        
    # plot steps of the algorithm
    x_iter, y_iter, *_ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    
    # plot objective function
    plt.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=2, color='green', label='Objective Function Direction')

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('LP Problem Constraints')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_QP(A, b, c, Q, steps, max_scale=100):
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    plt.figure(figsize=(8, 6))
    
    # quadratic objective function
    quadratic_part = lambda x, y: 0.5 * (Q[0, 0] * x**2 + (Q[0, 1] + Q[1, 0]) * x * y + Q[1, 1] * y**2) + c[0] * x + c[1] * y
    x_range = np.linspace(0, max_scale, 1000)
    y_range = np.linspace(0, max_scale, 1000)
    X, Y = np.meshgrid(x_range, y_range)
    Z = quadratic_part(X, Y)
    plt.contour(X, Y, Z, levels=40, cmap='viridis') 
    
    # plot constraints
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            print(f"ax = {b[i]}")
        elif A[i][1] == 0:
            print(f"ay = {b[i]}")
        else:
            x = np.linspace(0, max_scale, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y)
        
    # plot steps of the algorithm
    x_iter, y_iter, *_ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('LP Problem Constraints')
    plt.legend()
    plt.grid(True)
    plt.show()